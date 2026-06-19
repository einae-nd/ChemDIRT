import os
import pandas as pd
import requests
import json
import re
import base64

from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

DATASETS = {
    "bace":    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
    "bbbp":    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "clintox": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
    "hiv":     "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "lipo":     "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/lipo.csv",
    "esol":     "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "freesolv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/freesolv.csv.gz",
}

LABEL_COLUMNS = {
    "bace":    "Class",
    "bbbp":    "p_np",
    "clintox": "CT_TOX",
    "hiv":     "HIV_active",
    "esol":     "measured log solubility in mols per litre",
    "freesolv": "y",
    "lipo":     "target",
    "o2":       "o2"
}

def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

def scaffold_split(df, smiles_col="smiles", test_size=0.1, valid_size=0.1, seed=42):
    df = df.copy()
    df["scaffold"] = df[smiles_col].apply(generate_scaffold)

    # group molecules by scaffold
    scaffold_groups = {}
    for i, row in df.iterrows():
        scaffold = row["scaffold"]
        scaffold_groups.setdefault(scaffold, []).append(i)

    # sort scaffolds by size (largest first)
    sorted_scaffolds = sorted(scaffold_groups.items(), key=lambda x: len(x[1]), reverse=True)

    train_idx, valid_idx, test_idx = [], [], []
    n_total = len(df)
    n_test = int(n_total * test_size)
    n_valid = int(n_total * valid_size)
    n_train = n_total - n_test - n_valid

    for scaffold, indices in sorted_scaffolds:
        if len(train_idx) < n_train:
            train_idx.extend(indices)
        elif len(valid_idx) < n_valid:
            valid_idx.extend(indices)
        else:
            test_idx.extend(indices)

    df_train = df.loc[train_idx].drop(columns=["scaffold"])
    df_valid = df.loc[valid_idx].drop(columns=["scaffold"])
    df_test  = df.loc[test_idx].drop(columns=["scaffold"])

    return df_train, df_valid, df_test

def random_split(df, test_size=0.1, valid_size=0.1, seed=42):
    """
    Random split into train/valid/test with no scaffold grouping.
    Splits proportionally: first test, then valid from remaining.
    """
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle

    # First extract test
    while (test_size * len(df) > 200):
        test_size /= 2
    df_train_valid, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    # Now split train vs valid
    # valid size relative to train_valid: adjust proportion
    valid_adjusted = valid_size / (1 - test_size)

    df_train, df_valid = train_test_split(
        df_train_valid,
        test_size=valid_adjusted,
        random_state=seed,
        shuffle=True
    )

    return df_train.reset_index(drop=True), df_valid.reset_index(drop=True), df_test.reset_index(drop=True)

def download_and_split_all(output_dir="datasets"):
    os.makedirs(output_dir, exist_ok=True)

    for name, url in DATASETS.items():
        print(f"\n=== Processing {name} ===")

        dataset_dir = os.path.join(output_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)

        ## download
        csv_path = os.path.join(dataset_dir, f"{name}.csv")

        print(f"Downloading from {url} ...")
        r = requests.get(url)
        if r.status_code != 200:
            print(f"Failed to download {name}")
            continue

        with open(csv_path, "wb") as f:
            f.write(r.content)

        if url.endswith(".gz"):
            df = pd.read_csv(csv_path, compression="gzip")
        else:
            df = pd.read_csv(csv_path)

        ## get smiles
        smiles_col = "smiles"
        if smiles_col not in df.columns:
            # MoleculeNet sometimes uses "mol" or "SMILES"
            for col in df.columns:
                if col.lower() == "smiles" or col.lower() == "mol" or col.lower() == "smile":
                    smiles_col = col
                    break
        df = df[df[smiles_col].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)]

        ## split
        train_df, valid_df, test_df = random_split(df)

        ## save
        train_df.to_csv(os.path.join(dataset_dir, "train.csv"), index=False)
        valid_df.to_csv(os.path.join(dataset_dir, "valid.csv"), index=False)
        test_df.to_csv(os.path.join(dataset_dir, "test.csv"), index=False)

        print(f"Saved to {dataset_dir}/")

    print("\nAll datasets downloaded + split!")

def load_test_smiles_and_y(dataset, root="datasets"):
    """Returns test_smiles_list, test_y_list for any MoleculeNet dataset."""

    test_path = os.path.join(root, dataset, "test.csv")
    df = pd.read_csv(test_path)

    # detect smiles column
    smiles_col = None
    for col in df.columns:
        if col.lower() in ["smiles", "smile", "mol"]:
            smiles_col = col
            break
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {test_path}")

    test_smiles_list = df[smiles_col].astype(str).tolist()

    # label columns
    label_cols = LABEL_COLUMNS.get(dataset)

    if label_cols is None:  # for Sider and ToxCast
        # autodetect numeric tasks (drop SMILES + non-numeric)
        label_cols = [
            c for c in df.columns
            if c != smiles_col and df[c].dtype in ["float64", "int64"]
        ]

        LABEL_COLUMNS[dataset] = label_cols

    test_y_list = df[label_cols].values.tolist()
    
    if dataset == "hiv":
        # convert to int for classification
        subset = pd.read_csv("data/hiv_subset.csv")
        test_smiles_list = [test_smiles_list[i] for i in subset["index"]]
        test_y_list = [test_y_list[i] for i in subset["index"]]

    return test_smiles_list, test_y_list

def has_answer_type(example, target_type="physical unit"):
    parsed = parse_answer_json(example["answer_json"])
    return any(item.get("type") == target_type for item in parsed)

def parse_answer_json(x):
    """
    Safely parse answer_json.
    Returns an empty list if parsing fails.
    """
    if x is None:
        return []
    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return []
    
def extract_numbers(text):
    sci_notation_pattern = re.compile(r'([-\d\.]+)\s*[×xX]\s*10\^\(?(-?\d+)\)?')
    text = sci_notation_pattern.sub(lambda m: f"{float(m.group(1)) * (10 ** int(m.group(2)))}", text)

    number_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    matches = re.findall(number_pattern, text)
    
    numbers = [float(m) for m in matches]
    if not numbers:
        return [float(0)]
    return numbers

def clean_equation(equation):
    # Remove ->[...] notation like '->[Delta]'
    return re.sub(r'\->\[[^\]]*\]', '->', equation)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")