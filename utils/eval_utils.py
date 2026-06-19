import re
import numpy as np
import math

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem

import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from transformers import BertTokenizerFast

from utils.chem_utils import *

def get_eval_type(task):
    """
    Get the evaluation function for a given task.
    """
    return {
        "carbon_count": exact_match_num,
        "ring_count": exact_match_num,
        "functional_group": binary_yes_no,
        "bond_type": binary_yes_no,
        "bace": binary_yes_no,
        "bbbp": binary_yes_no,
        "clintox": binary_yes_no,
        "hiv": binary_yes_no,
        "i2s": eval_smiles,
        "s2i": eval_iupac,
        "p2s": eval_smiles,
        "p2i": eval_iupac,
        "design": eval_smiles,
        "rag_qa": text,
        "tfq": text,
        "open_response": text,
        "molecule_caption": long_answer_eval,
        "esol": regression,
        "lipo": regression,
        "freesolv": regression,
        "o2": regression,
        "coeff": exact_match_num,
        "unit_calc": regression,
        "yield": regression,
        "mol_weight": regression,
        "forward_smiles": eval_smiles,
        "retro_smiles": eval_smiles,
        "forward_formula": eval_formula,
        "retro_formula": eval_formula,
        "forward_image": eval_smiles,
        "retro_image": eval_smiles,
        "name_equiv": binary_yes_no,
        "chemical_sim": binary_yes_no,
        "property_comp": binary_yes_no,
    }.get(task, exact_match_text)


def exact_match_text(pred, true):
    """
    Check if the prediction exactly matches the ground truth.
    """
    
    return {
        "accuracy": int(
            str(pred).strip().lower() ==
            str(true).strip().lower()
        )
    }

def exact_match_num(pred, true):
    """
    Check if the prediction exactly matches the ground truth.
    """
    try:
        return {
            "accuracy": int(
                int(float(pred)) == int(float(true))
            )
        }
    except ValueError:
        return {"accuracy": 0}

def binary_yes_no(pred, true):
    """
    Check if the prediction is a binary yes/no answer and matches the ground truth.
    """
    pred = pred.strip().lower()
    #true = true.strip().lower()

    if pred in ["yes", "true", "1"]:
        pred = 1
    elif pred in ["no", "false", "0", "2"]:
        pred = 0
    else:
        pred = -1

    return {"accuracy": int(int(pred) == int(true))}

def regression(pred, true):
    """
    Evaluate regression predictions.

    Returns per-example squared errors so they can be averaged
    and square-rooted later to obtain:
        - RMSE
        - Relative RMSE
        - Log RMSE
    """

    true = float(true)

    try:
        pred = float(pred)
    except:
        pred = 0.0

    # Standard MSE
    mse = (pred - true) ** 2

    # Relative MSE
    if true != 0:
        rel_mse = ((pred - true) / true) ** 2
    else:
        rel_mse = float("nan")

    # Log-space MSE
    if pred > 0 and true > 0:
        log_mse = (math.log10(pred) - math.log10(true)) ** 2
    else:
        log_mse = float("nan")

    return {
        "mse": mse,
        "rel_mse": rel_mse,
        "log_mse": log_mse,
    }
    
def eval_smiles(pred, true, morgan_r=2):
    
    try:
        gt_m = Chem.MolFromSmiles(true)
        ot_m = Chem.MolFromSmiles(pred)

        if ot_m == None: raise ValueError('Bad SMILES')
        validity_score = 1
        
        MACCS_sim = DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity)
        RDK_sim = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity)
        morgan_sim = DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r))

    except:
        validity_score = 0

        MACCS_sim = None
        RDK_sim = None
        morgan_sim = None

    return {
        "validity": validity_score,
        "maccs": MACCS_sim,
        "rdk": RDK_sim,
        "morgan": morgan_sim,
    }

def eval_iupac(pred, true, morgan_r=2):
    
    try:
        pred = pyopsin(pred)
        true = pyopsin(true)
        
        gt_m = Chem.MolFromSmiles(true)
        ot_m = Chem.MolFromSmiles(pred)

        if ot_m == None: raise ValueError('Bad SMILES')
        validity_score = 1
        
        MACCS_sim = DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity)
        RDK_sim = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity)
        morgan_sim = DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r))

    except:
        validity_score = 0

        MACCS_sim = None
        RDK_sim = None
        morgan_sim = None

    return {
        "validity": validity_score,
        "maccs": MACCS_sim,
        "rdk": RDK_sim,
        "morgan": morgan_sim,
    }

def long_answer_eval(pred, true, text_model='allenai/scibert_scivocab_uncased', text_trunc_length=512):
    
    pred = pred[6:] if pred.startswith('[CLS] ') else pred
    
    """for smiles, gt, output in raw_outputs:
        out_tmp = output[6:] if output.startswith('[CLS] ') else output
        outputs.append((smiles, gt, out_tmp))"""

    text_tokenizer = BertTokenizerFast.from_pretrained(text_model)

    true = str(true)
    pred = str(pred)

    gt_tokens = text_tokenizer.tokenize(true, truncation=True, max_length=text_trunc_length,
                                        padding='max_length')
    gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
    gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
    gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

    out_tokens = text_tokenizer.tokenize(pred, truncation=True, max_length=text_trunc_length,
                                        padding='max_length')
    out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
    out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
    out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

    mscore = meteor_score([gt_tokens], out_tokens)

    bleu2 = corpus_bleu([gt_tokens], [ out_tokens], weights=(.5,.5))
    bleu4 = corpus_bleu([gt_tokens], [out_tokens], weights=(.25,.25,.25,.25))

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    rs = scorer.score(pred, true)
    rouge_scores.append(rs)

    rouge_1 = rs['rouge1'].fmeasure
    rouge_2 = rs['rouge2'].fmeasure
    rouge_l = rs['rougeL'].fmeasure

    return {
        "bleu2": bleu2,
        "bleu4": bleu4,
        "rouge1": rouge_1,
        "rouge2": rouge_2,
        "rougeL": rouge_l,
        "meteor": mscore,
    }
    
FORMULA_ELEMENTS = [
    "H", "C", "N", "O", "F",
    "P", "S", "Cl", "Br", "I",
    "B", "Si", "Se", "Na", "K",
    "Li", "Mg", "Ca", "Al", "Fe",
    "Cu", "Zn", "Mn", "Co", "Ni"
]


def parse_formula(formula):
    """
    Parse molecular formulas, including:

        C6H6O
        C6H14N2.C7H8O3S
        CuSO4.5H2O

    Returns:
        component_list
        aggregate_counts
    """

    formula = str(formula).strip()

    if len(formula) == 0:
        raise ValueError("Empty formula")

    components = formula.split(".")

    aggregate = {}
    normalized_components = []

    for component in components:

        component = component.strip()

        if len(component) == 0:
            continue

        # Handle hydrate notation:
        # 5H2O -> multiplier=5, formula=H2O
        match = re.match(r"^(\d+)([A-Z].*)$", component)

        if match:
            multiplier = int(match.group(1))
            component = match.group(2)
        else:
            multiplier = 1

        counts = {}

        tokens = re.findall(r"([A-Z][a-z]?)(\d*)", component)

        if len(tokens) == 0:
            raise ValueError(f"Invalid component: {component}")

        for element, count in tokens:

            count = int(count) if count else 1
            count *= multiplier

            counts[element] = counts.get(element, 0) + count
            aggregate[element] = aggregate.get(element, 0) + count

        normalized_components.append(counts)

    return normalized_components, aggregate


def formula_vector(counts):
    """
    Convert atom counts to fingerprint vector.
    """

    return np.array(
        [counts.get(elem, 0) for elem in FORMULA_ELEMENTS],
        dtype=float
    )


def normalize_component(counts):
    """
    Convert component dict into canonical tuple.

    Example:
        {"H":2,"O":1}
        ->
        (("H",2),("O",1))
    """

    return tuple(sorted(counts.items()))


def eval_formula(pred, true):
    """
    Evaluate molecular formulas.

    Metrics:
        validity
        exact_match
        cosine
        atom_f1
    """

    try:

        pred_components, pred_counts = parse_formula(pred)
        true_components, true_counts = parse_formula(true)

        validity = 1

        #
        # Component-level exact match
        #
        pred_norm = sorted(
            normalize_component(x)
            for x in pred_components
        )

        true_norm = sorted(
            normalize_component(x)
            for x in true_components
        )

        exact_match = int(pred_norm == true_norm)

        #
        # Aggregate cosine similarity
        #
        pred_vec = formula_vector(pred_counts)
        true_vec = formula_vector(true_counts)

        pred_norm_val = np.linalg.norm(pred_vec)
        true_norm_val = np.linalg.norm(true_vec)

        if pred_norm_val == 0 or true_norm_val == 0:
            cosine = 0.0
        else:
            cosine = float(
                np.dot(pred_vec, true_vec)
                /
                (pred_norm_val * true_norm_val)
            )

        #
        # Atom-count F1
        #
        overlap = 0

        all_elements = (
            set(pred_counts.keys())
            |
            set(true_counts.keys())
        )

        for elem in all_elements:
            overlap += min(
                pred_counts.get(elem, 0),
                true_counts.get(elem, 0)
            )

        pred_total = sum(pred_counts.values())
        true_total = sum(true_counts.values())

        precision = (
            overlap / pred_total
            if pred_total > 0 else 0
        )

        recall = (
            overlap / true_total
            if true_total > 0 else 0
        )

        if precision + recall > 0:
            atom_f1 = (
                2 * precision * recall
                /
                (precision + recall)
            )
        else:
            atom_f1 = 0.0

    except Exception:

        validity = 0
        exact_match = 0

        cosine = None
        atom_f1 = None

    return {
        "validity": validity,
        "exact_match": exact_match,
        "cosine": cosine,
        "atom_f1": atom_f1,
    }
    
def text(pred, true):
    return exact_match_text(pred, true) | long_answer_eval(pred, true)