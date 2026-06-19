from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from chempy.chemistry import Substance
import requests
import random
import re
from py2opsin import py2opsin

def get_carbon_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)

def get_ring_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return len(Chem.GetSSSR(mol))

def molar_mass(formula):
    """Rough molar mass using chempy Substance."""
    return Substance.from_formula(formula).mass

def get_mol_weight(smiles):
    """Get the molecular weight of a molecule given its SMILES representation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return Descriptors.MolWt(mol)

def smiles2name(smi, single_name=True):
    """This function queries the given molecule smiles and returns a name record or iupac"""

    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
    except Exception:
        raise ValueError("Invalid SMILES string")
    # query the PubChem database
    try:
        r = requests.get(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
            + smi
            + "/property/IUPACName/JSON"
        )
        # convert the response to a json object
        data = r.json()
    except Exception:
        #print(f"Failed to retrieve data for SMILES: {smi}")
        return
    
    try:
        iupac = data['PropertyTable']['Properties'][0]['IUPACName']
        return iupac
    except KeyError:
        #raise ValueError(f"Unknown molecule or no synonyms for SMILES: {smi}")
        #print(f"Unknown molecule or no synonyms for SMILES: {smi}")
        return
    
def smiles_to_iupac(smiles):
    url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles}/iupac_name"
    res = requests.get(url)
    
    content_type = res.headers.get("Content-Type", "")
    if "html" in content_type.lower():
        return None
    # Fallback: check actual content
    if res.text.lstrip().lower().startswith(("<!doctype html", "<html")):
        return None
    
    return res.text if res.ok else None
    
class BalanceTracker:
    def __init__(self):
        self.positive = 0
        self.negative = 0

    def want_positive(self):
        return self.positive <= self.negative

    def update(self, is_positive):
        if is_positive:
            self.positive += 1
        else:
            self.negative += 1   
    
FUNCTIONAL_GROUPS = {
    "alcohol": "[OX2H]",
    "phenol": "c[OX2H]",
    "amine": "[NX3;H2,H1,H0]",
    "amide": "C(=O)N",
    "carboxylic_acid": "C(=O)[OX2H1]",
    "ester": "C(=O)O",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[CX3](=O)[#6]",
    "nitro": "[NX3](=O)=O",
    "halide": "[F,Cl,Br,I]"
}

def get_functional_group(smiles, tracker: BalanceTracker):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, 0

    present = []
    absent = []

    for name, smarts in FUNCTIONAL_GROUPS.items():
        fg = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(fg):
            present.append(name)
        else:
            absent.append(name)

    # Edge cases
    if not present:
        chosen = random.choice(absent)
        tracker.update(False)
        return chosen, 0

    if not absent:
        chosen = random.choice(present)
        tracker.update(True)
        return chosen, 1

    # Balanced selection
    if tracker.want_positive():
        chosen = random.choice(present)
        tracker.update(True)
        return chosen, 1
    else:
        chosen = random.choice(absent)
        tracker.update(False)
        return chosen, 0

BOND_TYPES = {
    #"single": Chem.rdchem.BondType.SINGLE,
    "double": Chem.rdchem.BondType.DOUBLE,
    "triple": Chem.rdchem.BondType.TRIPLE,
    "aromatic": Chem.rdchem.BondType.AROMATIC,
}

def get_bond_type(smiles, tracker: BalanceTracker):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, 0

    present = set(bond.GetBondType() for bond in mol.GetBonds())

    present_types = []
    absent_types = []

    for name, bond_type in BOND_TYPES.items():
        if bond_type in present:
            present_types.append(name)
        else:
            absent_types.append(name)

    # Edge cases
    if not present_types:
        chosen = random.choice(absent_types)
        tracker.update(False)
        return chosen, 0

    if not absent_types:
        chosen = random.choice(present_types)
        tracker.update(True)
        return chosen, 1

    # Balanced selection
    if tracker.want_positive():
        chosen = random.choice(present_types)
        tracker.update(True)
        return chosen, 1
    else:
        chosen = random.choice(absent_types)
        tracker.update(False)
        return chosen, 0
    
app_id = 'U7JW62-U9XE37AXYU'
BASE_URL = "http://api.wolframalpha.com/v2/query"    

def ask_wolfram(query: str, app_id: str = app_id):
    """
    Query Wolfram Alpha Full Results API and return parsed pods.

    Args:
        query (str): Natural language or math query
        app_id (str): Your Wolfram Alpha AppID

    Returns:
        dict: Full parsed JSON response
    """
    params = {
        "input": query,
        "appid": app_id,
        "output": "JSON"  # request structured JSON
    }
    resp = requests.get(BASE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    if not data["queryresult"]["success"]:
        print("Query failed:", data["queryresult"].get("error", {}))
        return data

    # Extract pods
    pods = data["queryresult"].get("pods", [])
    results = {}
    for pod in pods:
        title = pod.get("title", "Unknown")
        subpods = pod.get("subpods", [])
        results[title] = [sp.get("plaintext", "") for sp in subpods if "plaintext" in sp]

    return {
        "raw": data,      # full JSON response
        "results": results  # simplified pod -> plaintext list
    }

def is_cas(text):
    pattern = r"^\d{2,7}-\d{2}-\d$"
    return re.match(pattern, text) is not None

def is_chembl_id(text):
    pattern = r"^CHEMBL\d+$"
    return re.match(pattern, text) is not None

def is_surechembl_id(text):
    pattern = r"^SCHEMBL\d+$"
    return re.match(pattern, text) is not None

def is_research_code_name(text):
    """
    Matches things like PD0332991, LY294002, etc.
    Pattern: 2–5 capital letters followed by 3–6 digits.
    """
    pattern = r"^[A-Z]{2,5}\d{3,6}$"
    return re.match(pattern, text) is not None 

def pyopsin(iupac):
    return py2opsin(iupac)

def parse_species(mixture_str):
    """Extracts tuples of (quantity, molecule) from a string like '{1}Na+.{2}H2O'."""
    matches = re.findall(r"\{(\d+)\}([^\.]+)", mixture_str)
    return [(int(qty), mol) for qty, mol in matches]

def eval_similarity(query, a, b):
    sim_a = tanimoto(query, a)
    sim_b = tanimoto(query, b)
    return "1" if sim_a >= sim_b else "0"

morgan_gen = GetMorganGenerator(
    radius=2,
    fpSize=2048
)

def tanimoto(smiles_a, smiles_b):
    m1 = Chem.MolFromSmiles(smiles_a)
    m2 = Chem.MolFromSmiles(smiles_b)
    if m1 is None or m2 is None:
        return 0.0
    #f1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    scaffold = MurckoScaffold.GetScaffoldForMol(m1)
    f1 = morgan_gen.GetFingerprint(scaffold)
    #f2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
    scaffold = MurckoScaffold.GetScaffoldForMol(m2)
    f2 = morgan_gen.GetFingerprint(scaffold)
    return DataStructs.TanimotoSimilarity(f1, f2)

def eval_property_comparison(a, b, direction="higher"):
    va = get_mol_weight(a)
    vb = get_mol_weight(b)
    if direction == "higher":
        return "1" if va >= vb else "0"
    else:
        return "1" if va <= vb else "0"