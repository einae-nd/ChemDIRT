import numpy as np
import pandas as pd
import random
import argparse
import ast
import itertools
from rdkit import Chem
from rdkit.Chem import Draw

from datasets import load_dataset, Dataset, Features, Value, Image

from utils.prompts import *
from utils.chem_utils import *
from utils.data_utils import *

def get_smiles_iupac_pairs():
    
    num_pairs = 2000 # description x 4, translate x 2, mol_weight, relational x 3 x 2
    
    pubchem_df = pd.read_csv("../../../afs/crc/group/dmsquare/vol5/einae/datasets/pubchem.csv").sample(1000)
    chembl_df = pd.read_csv("../../../afs/crc/group/dmsquare/vol5/einae/Motif-code/dataset/chembl/raw/all.txt", header=None).sample(1000)
    zinc_df = pd.read_csv("../../../afs/crc/group/dmsquare/vol5/einae/Motif-code/dataset/zinc_standard_agent/raw/smiles.csv").sample(1000)
    qm9_df = pd.read_csv("../../../afs/crc/group/dmsquare/vol5/einae/Motif-code/dataset/qm9/raw/all.txt").sample(1000)

    pubchem_smiles_list = pubchem_df.smiles.values
    chembl_smiles_list = chembl_df[0].values
    zinc_smiles_list = zinc_df.smiles.values
    qm9_smiles_list = qm9_df["SMILES1"].values

    pubchem_smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in pubchem_smiles_list if Chem.MolFromSmiles(smiles) is not None]
    chembl_smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in chembl_smiles_list if Chem.MolFromSmiles(smiles) is not None]
    zinc_smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in zinc_smiles_list if Chem.MolFromSmiles(smiles) is not None]
    qm9_smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in qm9_smiles_list if Chem.MolFromSmiles(smiles) is not None]

    smiles_list = np.concatenate((pubchem_smiles_list, chembl_smiles_list, zinc_smiles_list, qm9_smiles_list), axis=0)
    random.shuffle(smiles_list)
    
    rows = []
    
    for smiles in smiles_list:
        if Chem.MolFromSmiles(smiles) is not None:
            iupac = smiles2name(smiles)
            if iupac is not None:
                print(f"SMILES: {smiles} | IUPAC: {iupac}")

                row = {
                    "smiles": smiles,
                    "iupac": iupac
                }

                rows.append(row)
                
        if len(rows) == num_pairs:
            break

    df = pd.DataFrame(rows)
    df.to_csv(f"data/smiles_iupac_pairs.csv", index=False)

    return smiles_list

def create_questions_all(args):

    df = pd.read_csv("data/smiles_iupac_pairs.csv")

    if args.format == "smiles":
        print("smiles")
        df["mol"] = df["smiles"]
    elif args.format == "iupac":
        print("iupac")
        df["mol"] = df["iupac"]
    elif args.format == "both":
        print("both")
        mask = np.random.rand(len(df)) < 0.5
        df["mol"] = np.where(
            mask,
            df["smiles"],
            df["iupac"]
        )
    else:
        raise ValueError("Must specify either 'smiles', 'iupac', or 'both' as the representation format")

    N = 100

    if len(df) < 4 * N:
        raise ValueError(
            f"Need at least {4*N} molecules, found {len(df)}"
        )

    rows = []
    
    #### Description ####

    ## Carbon Count
    for _, row in df.iloc[0:N].iterrows():

        mol = row["mol"]
        smiles = row["smiles"]
        rows.append({
            "task": "carbon_count",
            "question": get_prompt("carbon_count", args.rand, mol=mol),
            "answer": str(int(get_carbon_count(smiles))),
            "image": None,
        })

    ## Ring Count
    for _, row in df.iloc[N:2*N].iterrows():

        mol = row["mol"]
        smiles = row["smiles"]
        rows.append({
            "task": "ring_count",
            "question": get_prompt("ring_count", args.rand, mol=mol),
            "answer": str(int(get_ring_count(smiles))),
            "image": None,
        })

    ## Functional Group Presence
    fg_tracker = BalanceTracker()
    for _, row in df.iloc[2*N:3*N].iterrows():

        mol = row["mol"]
        smiles = row["smiles"]
        
        fg_name, fg_presence = get_functional_group(
            smiles,
            fg_tracker
        )

        rows.append({
            "task": "functional_group",
            "question": get_prompt(
                "functional_group",
                args.rand,
                mol=mol,
                fg_name=fg_name
            ),
            "answer": str(int(fg_presence)),
            "image": None,
        })
    
    ## Bond Presence
    bond_tracker = BalanceTracker()
    for _, row in df.iloc[3*N:4*N].iterrows():

        mol = row["mol"]
        smiles = row["smiles"]

        bond_type, bond_presence = get_bond_type(
            smiles,
            bond_tracker
        )

        rows.append({
            "task": "bond_type",
            "question": get_prompt(
                "bond_type",
                args.rand,
                mol=mol,
                bond_type=bond_type
            ),
            "answer": str(int(bond_presence)),
            "image": None,
        })
        
    
    #### Classification ####
    
    ## BACE, BBBP, ClinTox, HIV
    classification_list = ["bace", "bbbp", "clintox", "hiv"]
    for task in classification_list:
        test_smiles_list, test_y_list = load_test_smiles_and_y(task)
        for i in range(0, len(test_smiles_list)):
            smiles = test_smiles_list[i]
            y = int(test_y_list[i])
            
            rows.append({
                "task": task,
                "question": get_prompt(
                    task,
                    args.rand,
                    smiles=smiles
                ),
                "answer": str(y),
                "image": None,
            })


    #### Design ####
    
    ## IUPAC2SMILES
    for _, row in df.iloc[4*N:5*N].iterrows():
        rows.append({
            "task": "i2s",
            "question": get_prompt("i2s", args.rand, iupac=row["iupac"]),
            "answer": row["smiles"],
            "image": None,
        })
    
    ## SMILES2IUPAC
    for _, row in df.iloc[4*N:5*N].iterrows():
        rows.append({
            "task": "s2i",
            "question": get_prompt("s2i", args.rand, smiles=row["smiles"]),
            "answer": row["iupac"],
            "image": None,
        })
        
    ## Photo2SMILES
    image_dir = "data/images"
    os.makedirs(image_dir, exist_ok=True)
    for i, (_, row) in enumerate(df.iloc[5*N:6*N].iterrows()):
        smiles = row["smiles"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        image_path = os.path.join(image_dir, f"design_mol_{i}.png")

        img = Draw.MolToImage(mol, size=(600, 600))
        img.save(image_path)

        rows.append({
            "task": "p2s",
            "question": get_prompt("p2s", args.rand),
            "answer": row["smiles"],
            "image": image_path,
            #"image_path": image_path,
        })
        
    ## Photo2IUPAC
    for i, (_, row) in enumerate(df.iloc[5*N:6*N].iterrows()):
        image_path = os.path.join(image_dir, f"design_mol_{i}.png")
        rows.append({
            "task": "p2i",
            "question": get_prompt("p2i", args.rand),
            "answer": row["iupac"],
            "image": image_path,
            #"image_path": image_path,
        })
    
    ## Description-based Design
    design_df = pd.read_csv("data/molecule_design_test.csv")
    for _, row in design_df.iterrows():
        rows.append({
            "task": "design",
            "question": row["description"],
            "answer": row["SMILES"],
            "image": None,
        })
    
    
    #### Analytical ####
    
    ## RAG-based QA
    # TODO Rewrite questions as necessary
    rag_qa_ds = pd.read_csv('data/chemlit_qa_test_split_211_entries.csv')
    for _, row in rag_qa_ds.iterrows():
        question = row['Context'] + " " +row['Question']
        rows.append({
            "task": "rag_qa",
            "question": question,
            "answer": row['Answer'],
        })

    mol_instruct_df = load_dataset("zjunlp/Mol-Instructions", "Biomolecular Text Instructions", trust_remote_code=True)

    ## True/False
    # TODO Check questions
    tfq_df = mol_instruct_df["true_or_false_question"].filter(lambda x: ast.literal_eval(x["metadata"]).get("split") == "test")
    for row in tfq_df:
        target = re.match(r"^\s*(yes|no|maybe)\b", row['output'].strip(), re.IGNORECASE).group(1)
        rows.append({
            "task": "tfq",
            "question": row['instruction'].strip(),
            "answer": target,
            "image": None,
        })

    ## Open Response
    # TODO Rewrite questions as necessary
    op_df = pd.read_csv("data/molinstructions_op_subset.csv")
    for _, row in op_df.iterrows():
        rows.append({
            "task": "open_response",
            "question": row['instruction'].strip(),
            "answer": row['output'].strip(),
            "image": None,
        })
            
    
    #### Regression ####
    
    ## ESOL, Lipophilicity, FreeSolv
    regression_list = ["esol", "lipo", "freesolv", "o2"]
    for task in regression_list:
        test_smiles_list, test_y_list = load_test_smiles_and_y(task)
        for i in range(0, len(test_smiles_list)):
            smiles = test_smiles_list[i]
            y = test_y_list[i]
            
            rows.append({
                "task": task,
                "question": get_prompt(
                    task,
                    args.rand,
                    smiles=smiles
                ),
                "answer": str(y),
                "image": None,
            })
    
    #### Computation ####
    
    ## Physical Unit Calculation
    # TODO Rewrite questions as necessary
    pu_ds = load_dataset("avaliev/ChemistryQA", split="test", trust_remote_code=True)
    pu_ds_filtered = pu_ds.filter(lambda x: has_answer_type(x, "physical unit")) 
    for row in pu_ds_filtered:
        answer = row["answer"]
        numeric = ast.literal_eval(row['target_var_json'])[0]['type'] == 'physical unit'
        if numeric: answer = extract_numbers(answer)[0]
        else:
            #if ast.literal_eval(row['target_var_json'])[0]['value'][:17] in ["Chemical Equation", "other"]:
                #answer = clean_equation(answer)
            continue
            
        question = row['question']
        if row['question_description']: question += row['question_description']
        
        rows.append({
            "task": "unit_calc",
            "question": question,
            "answer": str(answer),
            "image": None,
        })

    ## Coefficient Calculation
    src = pd.read_csv("data/src-test_in.txt", header=None, names=["reactants"])[:N]
    tgt = pd.read_csv("data/tgt-test_in.txt", header=None, names=["products"])[:N]
    for reactant_str, product_str in zip(src["reactants"], tgt["products"]):
        
        reactants = parse_species(reactant_str)
        products = parse_species(product_str)

        # Format the prompt
        reactant_str_formatted = ', '.join([f"{qty} {mol}" for qty, mol in reactants])
        product_names_only = ', '.join([mol for qty, mol in products])
        
        reactants_set = set(reactants)
        main_products = []

        main_products = [x for x in products if x not in reactants_set]
                
        target_quantity, target_product = random.choice(main_products)

        prompt = get_prompt(
            "coeff_calc",
            args.rand,
            reactants=reactant_str_formatted,
            products=product_names_only,
            target_product=target_product
        )

        rows.append({
            "task": "coeff_calc",
            "question": prompt,
            "answer": str(target_quantity),
            "image": None,
        })

    ## Yield Prediction
    yield_df = pd.read_csv("data/yield_prompts.csv")
    for _, row in yield_df.iterrows():
        rows.append({
            "task": "yield",
            "question": row["prompt"],
            "answer": str(row["answer"]),
            "image": None,
        })

    ## Molecular Weight
    for _, row in df.iloc[6*N:7*N].iterrows():

        mol = row["mol"]
        smiles = row["smiles"]
        

        mol_weight = round(get_mol_weight(smiles), 3)
        
        rows.append({
            "task": "mol_weight",
            "question": get_prompt(
                "mol_weight",
                args.rand,
                mol=mol
            ),
            "answer": str(mol_weight),
            "image": None,
        })

    #### Mechanistic ####
    
    ## Forward Reaction Prediction (SMILES)
    forward_df = pd.read_csv("data/uspto_test.csv")
    for _, row in forward_df.iterrows():
        reactants = row['reactant']
        products = row['product']
        rows.append({
            "task": "forward_smiles",
            "question": get_prompt(
                "forward_smiles",
                args.rand,
                reactants=reactants,
            ),
            "answer": products,
            "image": None,
        })
        
    # Forward Reaction Prediction (Image)
    for i, row in forward_df.iterrows():
        reactants = row['reactant']
        products = row['product']
        
        mol = Chem.MolFromSmiles(reactants)
        image_path = os.path.join(image_dir, f"forward_mol_{i}.png")
        img = Draw.MolToImage(mol, size=(600, 600))
        img.save(image_path)
        
        rows.append({
            "task": "forward_image",
            "question": get_prompt(
                "forward_image",
                args.rand,
            ),
            "answer": products,
            "image": image_path,
        })

    ## Forward Reaction Prediction (Formula)
    src = pd.read_csv("data/src-test_in.txt", header=None, names=["reactants"])[1*N:2*N]
    tgt = pd.read_csv("data/tgt-test_in.txt", header=None, names=["products"])[1*N:2*N]
    for reactant_str, product_str in zip(src["reactants"], tgt["products"]):
        
        reactants = parse_species(reactant_str)
        products = parse_species(product_str)

        # Format the prompt
        reactants_formatted = '.'.join([f"{mol}" for _, mol in reactants])
        products_formatted = '.'.join([mol for _, mol in products])

        prompt = get_prompt(
            "forward_formula",
            args.rand,
            reactants=reactants_formatted
        )

        rows.append({
            "task": "forward_formula",
            "question": prompt,
            "answer": products_formatted,
            "image": None,
        })

    ## Retrosynthesis Planning (SMILES)
    retro_df = pd.read_csv("data/uspto50k_retro_test.csv")
    for _, row in retro_df.iterrows():
        reactants = row['reactants_smiles']
        products = row['products_smiles']
        rows.append({
            "task": "retro_smiles",
            "question": get_prompt(
                "retro_smiles",
                args.rand,
                products=products
            ),
            "answer": reactants,
            "image": None,
        })
        
    # Retrosynthesis Planning (Image)
    for i, row in retro_df.iterrows():
        reactants = row['reactants_smiles']
        products = row['products_smiles']
        
        mol = Chem.MolFromSmiles(products)
        image_path = os.path.join(image_dir, f"retro_mol_{i}.png")
        img = Draw.MolToImage(mol, size=(600, 600))
        img.save(image_path)
        
        rows.append({
            "task": "retro_image",
            "question": get_prompt(
                "retro_image",
                args.rand,
            ),
            "answer": reactants,
            "image": image_path,
        })
    
    ## Retrosynthesis Planning (Formula)
    src = pd.read_csv("data/src-test_in.txt", header=None, names=["reactants"])[2*N:3*N]
    tgt = pd.read_csv("data/tgt-test_in.txt", header=None, names=["products"])[2*N:3*N]
    for reactant_str, product_str in zip(src["reactants"], tgt["products"]):
        
        reactants = parse_species(reactant_str)
        products = parse_species(product_str)

        # Format the prompt
        reactants_formatted = '.'.join([f"{mol}" for _, mol in reactants])
        products_formatted = '.'.join([mol for _, mol in products])

        prompt = get_prompt(
            "retro_formula",
            args.rand,
            products=products_formatted
        )

        rows.append({
            "task": "retro_formula",
            "question": prompt,
            "answer": reactants_formatted,
            "image": None,
        })
        
    
    #### Relational ####
    
    ## Name Equivalence
    for batch in itertools.batched(df.iloc[7*N:9*N].iterrows(), 2):
        rand = random.randint(0, 1)
        if rand:
            mol_a = batch[0][1].iloc[0]
            mol_b = batch[0][1].iloc[1]
            gt = 1
        else:
            mol_a = batch[0][1].iloc[0]
            mol_b = batch[1][1].iloc[1]
            gt = 0
            
        rows.append({
            "task": "name_equiv",
            "question": get_prompt(
                "name_equiv",
                args.rand,
                mol_a=mol_a,
                mol_b=mol_b
            ),
            "answer": str(gt),
            "image": None,
        })
        
    ## Chemical Similarity
    for batch in itertools.batched(df.iloc[9*N:12*N].iterrows(), 3):
        query = batch[0][1].iloc[2]
        query_smiles = batch[0][1].iloc[0]
        mol_a = batch[1][1].iloc[2]
        smiles_a = batch[1][1].iloc[0]
        mol_b = batch[2][1].iloc[2]
        smiles_b = batch[2][1].iloc[0]
        gt = eval_similarity(query_smiles, smiles_a, smiles_b)
        rows.append({
            "task": "chemical_sim",
            "question": get_prompt(
                "chemical_sim",
                args.rand,
                query=query,
                mol_a=mol_a,
                mol_b=mol_b
            ),
            "answer": str(gt),
            "image": None,
        })

    ## Property Comparison
    for batch in itertools.batched(df.iloc[12*N:14*N].iterrows(), 2):
        mol_a = batch[0][1].iloc[2]
        smiles_a = batch[0][1].iloc[0]
        mol_b = batch[1][1].iloc[2]
        smiles_b = batch[1][1].iloc[0]
        direction = random.choice(["higher", "lower"])
        gt = eval_property_comparison(smiles_a, smiles_b, direction)
        rows.append({
            "task": "property_comp",
            "question": get_prompt(
                "property_comp",
                args.rand,
                mol_a=mol_a,
                mol_b=mol_b,
                property_name="molecular weight",
                direction=direction
            ),
            "answer": str(gt),
            "image": None,
        })

    
    
    # Save all questions
    out_df = pd.DataFrame(rows)
    out_df.to_csv(f"data/all_questions_{args.format}_{args.rand}.csv", index=False)
    
    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("image", Image())
    
    dataset.push_to_hub(
        "Einae/ChemDIRT",
        private=False
    )
    
    
if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--rand', type=int, default=-1, help='Prompt selector')
    argparser.add_argument('--format', type=str, default='both', choices=["smiles", "iupac", "both"], help='Representation format to use')
    argparser.add_argument('--create_questions', type=bool, default=False, help='Create questions from the dataset')
    args = argparser.parse_args()
    
    create_questions_all(args)
    #get_smiles_iupac_pairs()