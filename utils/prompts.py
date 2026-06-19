import random

def get_i2s_prompt(iupac, num):
    templates = [
        f"Convert the following IUPAC name to its SMILES representation: {iupac}",
        f"What is the SMILES string for the compound named {iupac}?",
        f"Generate the canonical SMILES corresponding to the IUPAC name: {iupac}",
        f"Given the IUPAC name {iupac}, return its SMILES notation.",
        f"Translate this chemical IUPAC name into SMILES format: {iupac}",
        f"Can you provide the SMILES representation of {iupac}?",
        f"Find the SMILES encoding for the molecule with IUPAC name: {iupac}",
        f"I have the IUPAC name {iupac}. What's the equivalent SMILES string?",
        f"Please output the SMILES structure that matches this IUPAC name: {iupac}",
        f"From the IUPAC name {iupac}, generate the corresponding SMILES string."
    ]
    if num < 0: return random.choice(templates) #.format(iupac_name=iupac_name)
    else: return templates[num].format(iupac=iupac)

def get_s2i_prompt(smiles, num):
    templates = [
        f"Convert the following SMILES string into its IUPAC name: {smiles}",
        f"What is the IUPAC name for the molecule with SMILES: {smiles}?",
        f"Generate the IUPAC name corresponding to this SMILES: {smiles}",
        f"Given the SMILES {smiles}, return its official IUPAC name.",
        f"Translate this SMILES notation into an IUPAC chemical name: {smiles}",
        f"Can you provide the IUPAC nomenclature for the compound {smiles}?",
        f"Find the systematic IUPAC name of the molecule represented by {smiles}.",
        f"I have the SMILES {smiles}. What's the equivalent IUPAC name?",
        f"Please output the IUPAC chemical name that matches this SMILES: {smiles}",
        f"From the SMILES {smiles}, generate the corresponding IUPAC name."
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)
    
def get_p2s_prompt(num):
    templates = [
        f"Convert the molecular structure shown in this image into a SMILES string.",
        f"What is the SMILES representation of the compound depicted in this picture?",
        f"Analyze the chemical structure in the image and provide its canonical SMILES.",
        f"Given this molecular diagram, return the corresponding SMILES notation.",
        f"Translate the structure shown in the attached image into a SMILES string.",
        f"Identify the molecule in this image and output its SMILES representation.",
        f"Please extract the chemical structure from this figure and express it as SMILES.",
        f"Looking at this molecular drawing, what is the equivalent SMILES encoding?",
        f"Determine the SMILES string corresponding to the compound shown in the image.",
        f"Interpret the chemical structure displayed here and generate its SMILES notation."
    ]
    if num < 0: return random.choice(templates)
    else: return templates[num]

def get_p2i_prompt(num):
    templates = [
        f"Convert the molecular structure shown in this image into its IUPAC name.",
        f"What is the IUPAC name of the compound depicted in this picture?",
        f"Analyze the chemical structure in the image and provide its systematic IUPAC name.",
        f"Given this molecular diagram, return the corresponding IUPAC nomenclature.",
        f"Translate the structure shown in the attached image into an IUPAC chemical name.",
        f"Identify the molecule in this image and output its IUPAC name.",
        f"Please extract the chemical structure from this figure and provide its official IUPAC designation.",
        f"Looking at this molecular drawing, what is the equivalent IUPAC name?",
        f"Determine the systematic chemical name corresponding to the compound shown in the image.",
        f"Interpret the chemical structure displayed here and generate its IUPAC name."
    ]
    if num < 0: return random.choice(templates)
    else: return templates[num]

def bace_pred_prompt(smiles, num):
    templates = [
        f"Predict the BACE inhibitor property for the molecule with SMILES: {smiles}",
        f"Is the compound represented by {smiles} a BACE inhibitor?",
        f"Classify the molecule {smiles} with respect to its BACE inhibitory activity.",
        f"Given the SMILES {smiles}, determine if it has BACE inhibition properties.",
        f"Does the structure {smiles} act as a BACE inhibitor?",
        f"Evaluate the BACE activity of the molecule encoded as {smiles}.",
        f"For the molecule {smiles}, return whether it is a BACE inhibitor or not.",
        f"From the SMILES {smiles}, predict the binary BACE property.",
        f"Can you assess whether {smiles} corresponds to a BACE inhibitor?",
        f"What is the BACE property classification for the compound {smiles}?"
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def bbbp_pred_prompt(smiles, num):
    templates = [
        f"Predict the blood-brain barrier penetration (BBBP) property for the molecule with SMILES: {smiles}",
        f"Does the compound {smiles} cross the blood-brain barrier?",
        f"Classify the molecule {smiles} with respect to blood-brain barrier permeability.",
        f"Given the SMILES {smiles}, determine if it penetrates the BBB.",
        f"Will the molecule {smiles} be brain-penetrant?",
        f"Evaluate whether {smiles} crosses the blood-brain barrier.",
        f"For the molecule {smiles}, return if it has BBB permeability or not.",
        f"From the SMILES {smiles}, predict the binary BBBP property.",
        f"Can you assess if {smiles} penetrates the blood-brain barrier?",
        f"What is the BBBP classification for the compound {smiles}?"
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)
    
def clintox_pred_prompt(smiles, num):
    templates = [
        f"Predict the clinical trial toxicity outcome for the molecule with SMILES: {smiles}.",
        f"Did the compound represented by {smiles} fail clinical trials due to toxicity?",
        f"Classify the molecule {smiles} with respect to clinical trial toxicity.",
        f"Given the SMILES {smiles}, determine whether the compound exhibits clinical trial toxicity.",
        f"Does the structure {smiles} indicate failure in clinical trials due to toxicity?",
        f"Evaluate the clinical trial toxicity of the molecule encoded as {smiles}.",
        f"For the molecule {smiles}, return whether it failed clinical trials because of toxicity.",
        f"From the SMILES {smiles}, predict the binary clinical trial toxicity label.",
        f"Can you assess whether {smiles} corresponds to a clinically toxic compound?",
        f"What is the clinical trial toxicity classification for the compound {smiles}?"
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def hiv_pred_prompt(smiles, num):
    templates = [
        f"Predict the HIV inhibitor property for the molecule with SMILES: {smiles}",
        f"Is the compound {smiles} active against HIV?",
        f"Classify the molecule {smiles} with respect to its HIV inhibitory activity.",
        f"Given the SMILES {smiles}, determine if it has HIV inhibition properties.",
        f"Does the structure {smiles} inhibit HIV?",
        f"Evaluate the HIV activity of the molecule encoded as {smiles}.",
        f"For the molecule {smiles}, return whether it is an HIV inhibitor or not.",
        f"From the SMILES {smiles}, predict the binary HIV property.",
        f"Can you assess whether {smiles} corresponds to an HIV inhibitor?",
        f"What is the HIV property classification for the compound {smiles}?"
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def esol_pred_prompt(smiles, num):
    templates = [
        f"Predict the aqueous solubility (ESOL) value for the molecule with SMILES: {smiles}",
        f"What is the solubility of {smiles} in water?",
        f"Estimate the ESOL property of {smiles}.",
        f"Given the SMILES {smiles}, compute its solubility value.",
        f"How soluble is {smiles} in water?",
        f"Report the predicted ESOL value for {smiles}.",
        f"For the molecule {smiles}, return its solubility estimate.",
        f"From the SMILES {smiles}, predict the ESOL property.",
        f"Provide the aqueous solubility for the structure {smiles}.",
        f"What is the predicted ESOL value for {smiles}?"
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def freesolv_pred_prompt(smiles, num):
    templates = [
        f"Predict the hydration free energy (FreeSolv) for the molecule with SMILES: {smiles}",
        f"What is the hydration free energy of {smiles}?",
        f"Estimate the FreeSolv value for {smiles}.",
        f"Given the SMILES {smiles}, compute its solvation free energy.",
        f"How favorable is hydration for {smiles}?",
        f"Report the predicted FreeSolv property for {smiles}.",
        f"For the molecule {smiles}, return its hydration free energy.",
        f"From the SMILES {smiles}, predict the FreeSolv value.",
        f"Provide the solvation free energy for the compound {smiles}.",
        f"What is the hydration free energy of {smiles} in water?"
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def lipo_pred_prompt(smiles, num):
    templates = [
        f"Predict the lipophilicity (Lipo) value for the molecule with SMILES: {smiles}",
        f"What is the logD/lipophilicity of {smiles}?",
        f"Estimate the lipophilicity property of {smiles}.",
        f"Given the SMILES {smiles}, compute its lipophilicity.",
        f"How lipophilic is {smiles}?",
        f"Report the predicted Lipo value for {smiles}.",
        f"For the molecule {smiles}, return its lipophilicity score.",
        f"From the SMILES {smiles}, predict the Lipo property.",
        f"Provide the lipophilicity index for the compound {smiles}.",
        f"What is the predicted lipophilicity (Lipo) of {smiles}?"
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def polymer_density_pred_prompt(smiles, num):
    templates = [
        f"Predict the polymer density for the structure with SMILES: {smiles}",
        f"What is the density of the polymer unit {smiles}?",
        f"Estimate the polymer density corresponding to {smiles}.",
        f"Given the SMILES {smiles}, compute its density.",
        f"Report the polymer density value for {smiles}.",
        f"For the polymer with unit {smiles}, return its density.",
        f"From the SMILES {smiles}, predict the density property.",
        f"Provide the density (g/cm³) estimate for the polymer {smiles}.",
        f"What is the predicted density of the polymer unit {smiles}?",
        f"Calculate the polymer density for {smiles}."
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)
    
def polymer_mt_pred_prompt(smiles, num):
    templates = [
        f"Predict the mechanical strength (MT) of the polymer with SMILES: {smiles}",
        f"What is the modulus/strength property for the polymer {smiles}?",
        f"Estimate the mechanical toughness of {smiles}.",
        f"Given the SMILES {smiles}, compute its MT property.",
        f"Report the polymer mechanical property (MT) of {smiles}.",
        f"For the polymer with unit {smiles}, return its MT estimate.",
        f"From the SMILES {smiles}, predict the polymer MT value.",
        f"Provide the mechanical strength property for {smiles}.",
        f"What is the MT prediction for the polymer {smiles}?",
        f"Calculate the polymer mechanical toughness for {smiles}."
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def polymer_o2_pred_prompt(smiles, num):
    templates = [
        f"Predict the oxygen permeability (in Barrer) of the polymer with SMILES: {smiles}.",
        f"What is the O₂ permeability (Barrer) of the polymer {smiles}?",
        f"Given the polymer SMILES {smiles}, predict its oxygen permeability expressed in Barrer.",
        f"Estimate the oxygen permeability of {smiles} in Barrer units.",
        f"Based on solution–diffusion transport behavior, predict the O₂ permeability (Barrer) of the polymer {smiles}.",
        f"For gas separation membrane applications, predict the oxygen permeability (Barrer) of the polymer {smiles}.",
        f"Predict the oxygen permeability value (Barrer) for the polymer with SMILES {smiles}.",
        f"Determine the oxygen permeability coefficient, reported in Barrer, for the polymer {smiles}.",
        f"Predict the oxygen permeability of the polymer {smiles}. Output a single numeric value in Barrer.",
        f"Return the oxygen permeability (Barrer) for the polymer represented by {smiles}."
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def polymer_tg_pred_prompt(smiles, num):
    templates = [
        f"Predict the glass transition temperature (Tg) of the polymer with SMILES: {smiles}",
        f"What is the Tg of the polymer {smiles}?",
        f"Estimate the glass transition temperature for {smiles}.",
        f"Given the SMILES {smiles}, compute its Tg.",
        f"Report the predicted Tg value for {smiles}.",
        f"For the polymer {smiles}, return its glass transition temperature.",
        f"From the SMILES {smiles}, predict the polymer Tg property.",
        f"Provide the glass transition temperature (°C) of {smiles}.",
        f"What is the Tg prediction for the polymer {smiles}?",
        f"Calculate the glass transition temperature of {smiles}."
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def mol_weight_prompt(mol, num):
    templates = [
        f"Calculate exactly the molecular weight for the compound {mol}.",
        f"What is the exact molar mass of {mol}?",
        f"Determine the molecular weight corresponding to {mol}.",
        f"Find the molecular weight of the molecule with SMILES {mol}.",
        f"Compute the molecular weight of {mol} in Daltons.",
        f"How heavy is the molecule {mol} in Daltons?",
        f"What is the MW (g/mol) for {mol}?",
        f"Report the molecular weight of the structure {mol}.",
        f"Provide the molar mass of the compound encoded as {mol}.",
        f"For the SMILES {mol}, return its molecular weight."
    ]
    if num < 0: return random.choice(templates).format(mol=mol)
    else: return templates[num].format(mol=mol)

def logp_prompts(smiles, num):
    templates = [
        f"Calculate the LogP value for the compound {smiles}.",
        f"What is the octanol–water partition coefficient (LogP) of {smiles}?",
        f"Determine the lipophilicity (LogP) for the molecule {smiles}.",
        f"Estimate the LogP for {smiles}.",
        f"Find the partition coefficient (LogP) of the structure {smiles}.",
        f"What is the LogP value of {smiles}?",
        f"Compute the lipophilicity index LogP for {smiles}.",
        f"Return the LogP property of the compound {smiles}.",
        f"How hydrophobic is {smiles}? Report its LogP.",
        f"For the SMILES {smiles}, provide its LogP value."
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def hdonor_prompts(smiles, num):
    templates = [
        f"Count the number of hydrogen bond donors in {smiles}.",
        f"How many H-bond donors does {smiles} have?",
        f"Determine the hydrogen donor count for {smiles}.",
        f"Report the number of hydrogen bond donors in the molecule {smiles}.",
        f"For {smiles}, how many donor groups are present?",
        f"What is the H-bond donor count for {smiles}?",
        f"Compute the number of hydrogen donors in {smiles}.",
        f"How many groups in {smiles} can donate hydrogen bonds?",
        f"Find the hydrogen bond donor count of {smiles}.",
        f"Given the SMILES {smiles}, return its donor count."
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def hacceptor_prompts(smiles, num):
    templates = [
        f"Count the number of hydrogen bond acceptors in {smiles}.",
        f"How many H-bond acceptors does {smiles} have?",
        f"Determine the hydrogen acceptor count for {smiles}.",
        f"Report the number of hydrogen bond acceptors in the molecule {smiles}.",
        f"For {smiles}, how many acceptor groups are present?",
        f"What is the H-bond acceptor count for {smiles}?",
        f"Compute the number of hydrogen acceptors in {smiles}.",
        f"How many groups in {smiles} can accept hydrogen bonds?",
        f"Find the hydrogen bond acceptor count of {smiles}.",
        f"Given the SMILES {smiles}, return its acceptor count."
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def tpsa_prompts(smiles, num):
    templates = [
        f"Calculate the topological polar surface area (TPSA) of {smiles}.",
        f"What is the TPSA value for the molecule {smiles}?",
        f"Determine the polar surface area of {smiles}.",
        f"Compute the TPSA for {smiles}.",
        f"Report the topological polar surface area (in Å²) of {smiles}.",
        f"Find the TPSA for the compound represented by {smiles}.",
        f"How polar is {smiles}? Give its TPSA value.",
        f"Provide the TPSA of {smiles}.",
        f"Estimate the topological polar surface area of {smiles}.",
        f"For the SMILES {smiles}, return its TPSA."
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def rotatable_bond_prompts(smiles, num):
    templates = [
        f"Count the number of rotatable bonds in {smiles}.",
        f"How many rotatable bonds does {smiles} have?",
        f"Determine the rotatable bond count for {smiles}.",
        f"Report the number of rotatable bonds in the molecule {smiles}.",
        f"For {smiles}, how many freely rotating bonds are present?",
        f"What is the rotatable bond count for {smiles}?",
        f"Compute the number of rotatable bonds in {smiles}.",
        f"How flexible is {smiles}? Give the rotatable bond count.",
        f"Find the rotatable bond number of {smiles}.",
        f"Given the SMILES {smiles}, return its rotatable bonds."
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)

def ring_count_prompts(mol, num):
    templates = [
        f"Count the number of rings in {mol}.",
        f"How many ring systems does {mol} contain?",
        f"Determine the ring count for {mol}.",
        f"Report the number of rings in the molecule {mol}.",
        f"For {mol}, how many distinct rings are present?",
        f"What is the total number of rings in {mol}?",
        f"Compute the ring count of {mol}.",
        f"How many cyclic structures are in {mol}?",
        f"Find the number of rings contained in {mol}.",
        f"Given the SMILES {mol}, return its ring count."
    ]
    if num < 0: return random.choice(templates).format(mol=mol)
    else: return templates[num].format(mol=mol)

def carbon_count_prompts(mol, num):
    templates = [
        f"Count the number of carbon atoms in {mol}.",
        f"How many carbons are present in the molecule {mol}?",
        f"Determine the total carbon count for {mol}.",
        f"Report the number of carbon atoms in {mol}.",
        f"For the molecule {mol}, how many carbon atoms does it contain?",
        f"Compute the total number of carbons in {mol}.",
        f"Identify the carbon atom count in the molecule {mol}.",
        f"What is the total count of carbon atoms in {mol}?",
        f"Given {mol}, return the number of carbon atoms.",
        f"How many carbon atoms make up the structure {mol}?"
    ]
    if num < 0: return random.choice(templates).format(mol=mol)
    else: return templates[num].format(mol=mol)

def functional_group_prompts(mol, num, fg_name):
    templates = [
        f"Does the molecule {mol} contain a {fg_name} functional group?",
        f"Is a {fg_name} group present in {mol}?",
        f"Determine whether {mol} includes a {fg_name} functional group.",
        f"Does {mol} have at least one {fg_name} group?",
        f"Check if the structure {mol} contains a {fg_name}.",
        f"For the molecule {mol}, is a {fg_name} functional group present?",
        f"Assess whether {mol} exhibits a {fg_name} group.",
        f"Does the SMILES {mol} encode a {fg_name} functional group?",
        f"Identify if {mol} contains a {fg_name}.",
        f"Is the {fg_name} functional group found in {mol}?"
    ]
    if num < 0: return random.choice(templates).format(mol=mol, fg_name=fg_name)
    else: return templates[num].format(mol=mol, fg_name=fg_name)
    
def bond_type_prompts(mol, num, bond_type):
    templates = [
        f"Does the molecule {mol} contain any {bond_type} bonds?",
        f"Are {bond_type} bonds present in {mol}?",
        f"Determine whether {mol} includes at least one {bond_type} bond.",
        f"Check if the structure {mol} has a {bond_type} bond.",
        f"For the molecule {mol}, does a {bond_type} bond exist?",
        f"Does {mol} feature any {bond_type} bonds?",
        f"Assess whether {mol} contains a {bond_type} bond type.",
        f"Is a {bond_type} bond found in the SMILES {mol}?",
        f"Identify whether {mol} has any {bond_type} bonds.",
        f"Does the chemical structure {mol} include {bond_type} bonding?"
    ]
    if num < 0: return random.choice(templates).format(mol=mol, bond_type=bond_type)
    else: return templates[num].format(mol=mol, bond_type=bond_type)

def formal_charge_prompts(smiles, num):
    templates = [
        f"Calculate the formal charge of {smiles}.",
        f"What is the total formal charge of the molecule {smiles}?",
        f"Determine the net formal charge for {smiles}.",
        f"Compute the overall charge (formal) of {smiles}.",
        f"Report the formal charge on the compound {smiles}.",
        f"For {smiles}, what is its net formal charge?",
        f"Find the formal charge of the SMILES string {smiles}.",
        f"How much formal charge does {smiles} carry?",
        f"Provide the formal charge value of {smiles}.",
        f"Given the SMILES {smiles}, return its formal charge."
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles)
    else: return templates[num].format(smiles=smiles)
    
def mole_pred_prompt(smiles, num, quantity):
    templates = [
        f"Given {quantity} of {smiles}, how many moles does this represent?",
        f"Convert {quantity} of the compound {smiles} into moles.",
        f"What is the molar amount of {smiles} in {quantity}?",
        f"For {quantity} of {smiles}, calculate the number of moles.",
        f"How many moles are contained in {quantity} of {smiles}?",
        f"From {quantity} of the molecule {smiles}, determine the moles.",
        f"Express {quantity} of {smiles} in terms of moles.",
        f"What is the mole equivalent of {quantity} of the compound {smiles}?",
        f"If I have {quantity} of {smiles}, how many moles is that?",
        f"Calculate the number of moles present in {quantity} of {smiles}."
    ]
    if num < 0: return random.choice(templates).format(smiles=smiles, quantity=quantity)
    else: return templates[num].format(smiles=smiles, quantity=quantity)

def coeff_pred_prompt(smiles, num, equation, species):
    templates = [
        f"Balance the following equation by finding the missing coefficient: {equation}",
        f"What coefficient should go in front of {species} to balance {equation}?",
        f"Complete the stoichiometry: fill in the missing coefficient in {equation}.",
        f"For the reaction {equation}, what is the correct stoichiometric coefficient of {species}?",
        f"Find the integer coefficient that balances the molecule {species} in {equation}.",
        f"In the equation {equation}, determine the missing multiplier for {species}.",
        f"What value should replace the blank to balance the equation: {equation}?",
        f"Balance this reaction by supplying the missing number in front of {species}: {equation}",
        f"For {equation}, calculate the correct coefficient of {species}.",
        f"Fill in the missing stoichiometric number in {equation}."
    ]
    if num < 0: return random.choice(templates).format(equation=equation, species=species)
    else: return templates[num].format(equation=equation, species=species)

def yield_pred_prompt(smiles, num, equation, given_species, given_mass, asked_species):
    templates = [
        f"Given the balanced equation {equation}, if {given_mass} of {given_species} is present, how much {asked_species} will form?",
        f"For the reaction {equation}, calculate the required amount of {asked_species} if {given_mass} of {given_species} is used.",
        f"In the balanced chemical equation {equation}, determine the quantity of {asked_species} corresponding to {given_mass} of {given_species}.",
        f"If {given_mass} of {given_species} reacts in {equation}, what is the yield of {asked_species}?",
        f"Using the balanced equation {equation}, find the number of moles of {asked_species} that react with {given_mass} of {given_species}.",
        f"For {equation}, compute the mass of {asked_species} produced when {given_mass} of {given_species} is available.",
        f"According to {equation}, how many moles of {asked_species} correspond to {given_mass} of {given_species}?",
        f"Using stoichiometry from {equation}, calculate the amount of {asked_species} from {given_mass} of {given_species}.",
        f"Based on the balanced equation {equation}, if {given_mass} of {given_species} is consumed, what is the resulting amount of {asked_species}?",
        f"In {equation}, find the stoichiometric equivalent of {asked_species} when {given_mass} of {given_species} is given."
    ]
    if num < 0: return random.choice(templates).format(equation=equation, given_mass=given_mass, given_species=given_species, asked_species=asked_species)
    else: return templates[num].format(equation=equation, given_mass=given_mass, given_species=given_species, asked_species=asked_species)


def reaction_pred_prompt(reactants, quantities, num):
    templates = [
        f"Given the reactants {reactants} with quantities {quantities}, predict the reaction outcome.",
        f"What is the expected product of the reaction involving {reactants} in amounts {quantities}?",
        f"Using the reactants {reactants} and their quantities {quantities}, determine the reaction result.",
        f"Predict the chemical reaction outcome for {reactants} with specified quantities {quantities}.",
        f"From the reactants {reactants} and their respective quantities {quantities}, what is the product?",
        f"Can you forecast the reaction product when using {reactants} in amounts {quantities}?",
        f"Identify the likely outcome of a reaction with reactants {reactants} and quantities {quantities}.",
        f"What product forms from the reaction of {reactants} given their quantities {quantities}?",
        f"Determine the result of a chemical reaction using {reactants} with quantities {quantities}.",
        f"Using the provided reactants {reactants} and their quantities {quantities}, predict the reaction product."
    ]
    if num < 0: return random.choice(templates).format(reactants=reactants, quantities=quantities)
    else: return templates[num].format(reactants=reactants, quantities=quantities)

import random

def name_equiv_prompt(mol_a, mol_b, num):
    templates = [
        f"Do the two chemical representations {mol_a} and {mol_b} describe the same molecule?",
        f"Determine whether {mol_a} and {mol_b} refer to the same chemical compound.",
        f"Are {mol_a} and {mol_b} equivalent representations of the same molecule?",
        f"Do {mol_a} and {mol_b} correspond to the same molecular structure?",
        f"Is the molecule described by {mol_a} identical to the one described by {mol_b}?",
        f"Assess whether {mol_a} and {mol_b} denote the same chemical substance.",
        f"Do these two representations, {mol_a} and {mol_b}, describe the same molecule?",
        f"Are the chemical entities represented by {mol_a} and {mol_b} the same?",
        f"Verify whether {mol_a} and {mol_b} correspond to an identical molecule.",
        f"Given {mol_a} and {mol_b}, determine if they describe the same molecular structure."
    ]
    if num < 0: return random.choice(templates).format(mol_a=mol_a, mol_b=mol_b)
    else: return templates[num].format(mol_a=mol_a, mol_b=mol_b)

def chemical_sim_prompt(query, mol_a, mol_b, num):
    templates = [
        f"Which molecule is more chemically similar to {query}: {mol_a} or {mol_b}?",
        f"Between {mol_a} and {mol_b}, which is more similar in structure to {query}?",
        f"Determine whether {mol_a} or {mol_b} is more chemically similar to {query}.",
        f"Which compound more closely resembles {query}: {mol_a} or {mol_b}?",
        f"Assess the chemical similarity of {mol_a} and {mol_b} to {query}. Which is more similar?",
        f"Which molecule shares greater structural similarity with {query}, {mol_a} or {mol_b}?",
        f"Compare {mol_a} and {mol_b} to {query}. Which is more chemically similar?",
        f"Identify which molecule is more similar to {query}: {mol_a} or {mol_b}.",
        f"Between the two options, {mol_a} and {mol_b}, which better matches {query} chemically?",
        f"Given {query}, select the molecule that is more chemically similar: {mol_a} or {mol_b}."
    ]
    if num < 0:
        return random.choice(templates).format(query=query, mol_a=mol_a, mol_b=mol_b)
    else:
        return templates[num].format(query=query, mol_a=mol_a, mol_b=mol_b)

def property_comp_prompt(mol_a, mol_b, property_name, direction, num):
    """
    direction: 'higher' or 'lower'
    """

    templates = [
        f"Which molecule has a {direction} {property_name}, {mol_a} or {mol_b}?",
        f"Determine whether {mol_a} or {mol_b} has the {direction} value of {property_name}.",
        f"Between {mol_a} and {mol_b}, which exhibits a {direction} {property_name}?",
        f"Which compound shows a {direction} {property_name}: {mol_a} or {mol_b}?",
        f"Assess the {property_name} of {mol_a} and {mol_b}. Which one is {direction}?",
        f"Identify whether {mol_a} or {mol_b} has the {direction} {property_name}.",
        f"Compare {mol_a} and {mol_b} in terms of {property_name}. Which is {direction}?",
        f"Which molecule demonstrates a {direction} value for {property_name}, {mol_a} or {mol_b}?",
        f"For the property {property_name}, which molecule is {direction}: {mol_a} or {mol_b}?",
        f"Given {mol_a} and {mol_b}, determine which has the {direction} {property_name}."
    ]
    if num < 0:
        return random.choice(templates).format(mol_a=mol_a, mol_b=mol_b)
    else:
        return templates[num].format(mol_a=mol_a, mol_b=mol_b)

def coeff_calc_prompt(reactants, products, target_product, num):
    templates = [
        f"Reactants: {reactants}\nProducts: {products}\nWhat is the stoichiometric coefficient of {target_product}?",
        f"Given the following reaction:\nReactants: {reactants}\nProducts: {products}\nDetermine the stoichiometric amount of {target_product}.",
        f"Consider the reaction with reactants {reactants} forming products {products}. What stoichiometric quantity corresponds to {target_product}?",
        f"For the chemical transformation involving {reactants} → {products}, what is the stoichiometric coefficient associated with {target_product}?",
        f"Based on the listed reactants ({reactants}) and products ({products}), compute the stoichiometric quantity of {target_product}.",
        f"In the reaction where {reactants} yield {products}, what is the required stoichiometric coefficient for {target_product}?",
        f"Using the given reaction specification:\nReactants: {reactants}\nProducts: {products}\nIdentify the stoichiometric amount of {target_product}.",
        f"Determine how many stoichiometric units of {target_product} are produced from the reaction involving {reactants}.",
        f"What is the balanced stoichiometric coefficient for {target_product} in the reaction with reactants {reactants} and products {products}?",
        f"From the reaction defined by reactants {reactants} and products {products}, calculate the stoichiometric quantity of {target_product}."
    ]

    if num < 0:
        return random.choice(templates).format(reactants=reactants, products=products, target_product=target_product)
    else:
        return templates[num].format(reactants=reactants, products=products, target_product=target_product)

def forward_prompt(reactants, num):
    templates = [
        f"Reactants: {reactants}\nProducts:",
        f"Given the following reactants:\n{reactants}\nPredict the reaction products.",
        f"Consider the chemical reaction involving the reactants {reactants}. What products are formed?",
        f"For the reaction starting from reactants {reactants}, determine the resulting products.",
        f"Based on the listed reactants ({reactants}), predict the products of the reaction.",
        f"In the chemical transformation where the reactants are {reactants}, what compounds are produced?",
        f"Using the given reaction specification:\nReactants: {reactants}\nIdentify the products.",
        f"Determine the products formed when the reactants {reactants} undergo reaction.",
        f"What are the expected reaction products from the reactants {reactants}?",
        f"From the reaction defined solely by the reactants {reactants}, predict the products."
    ]

    if num < 0:
        return random.choice(templates).format(reactants=reactants)
    else:
        return templates[num].format(reactants=reactants)

def retro_prompt(products, num):
    templates = [
        f"Products: {products}\nReactants:",
        f"Given the following products:\n{products}\nPredict the reactants required for their synthesis.",
        f"Consider the target products {products}. What reactants would be used to synthesize them?",
        f"For the chemical transformation leading to products {products}, determine the necessary reactants.",
        f"Based on the listed products ({products}), predict the reactants involved in their formation.",
        f"In the retrosynthetic analysis of products {products}, what reactants are required?",
        f"Using the given reaction specification:\nProducts: {products}\nIdentify the reactants.",
        f"Determine the reactants that could give rise to the products {products}.",
        f"What reactants would produce the products {products} in a chemical reaction?",
        f"From the reaction defined by the products {products}, predict the corresponding reactants."
    ]

    if num < 0:
        return random.choice(templates).format(products=products)
    else:
        return templates[num].format(products=products)
    
def forward_image_prompt(num):
    templates = [
        "The image shows the reactants of a chemical reaction. Predict the product SMILES.",
        "Given the reactant structures shown in the image, predict the major product. Return only the product SMILES.",
        "Consider the reactants depicted in the image. What product is formed? Respond with the product SMILES only.",
        "The image contains the starting materials for a chemical reaction. Determine the resulting product and return its SMILES representation.",
        "Based on the reactants shown in the image, predict the reaction product. Output only the product SMILES.",
        "Identify the product formed from the reactants depicted in the image. Return only the SMILES string.",
        "The image illustrates the reactants participating in a chemical transformation. Predict the product SMILES.",
        "Determine the product generated from the reactants shown in the image. Respond only with the product SMILES.",
        "What is the expected product of the reaction represented by the reactants in the image? Return only the product SMILES.",
        "Using the reactant structures provided in the image, predict the reaction outcome and return only the product SMILES."
    ]

    if num < 0:
        return random.choice(templates)
    else:
        return templates[num]

def retro_image_prompt(num):
    templates = [
        "The image shows the product of a chemical reaction. Predict the reactant SMILES.",
        "Given the product structure shown in the image, determine the reactants required for its synthesis. Return only the reactant SMILES.",
        "Consider the product depicted in the image. What reactants could produce it? Respond only with reactant SMILES.",
        "The image contains the target product of a chemical transformation. Predict the reactants and return their SMILES representation.",
        "Based on the product shown in the image, perform retrosynthetic analysis and predict the reactants. Output only reactant SMILES.",
        "Identify the reactants that could give rise to the product depicted in the image. Return only the SMILES strings.",
        "The image illustrates the product of a reaction. Determine the corresponding reactants and return only reactant SMILES.",
        "Predict the reactants that would produce the structure shown in the image. Respond only with reactant SMILES.",
        "What reactants are required to synthesize the product shown in the image? Return only the reactant SMILES.",
        "Using the product structure provided in the image, infer the reactants and return only their SMILES representations."
    ]

    if num < 0:
        return random.choice(templates)
    else:
        return templates[num]
    
def molecule_caption_prompt(smiles, num):
    templates = [
        f"Given the SMILES {smiles}, provide a detailed description of the molecule, including its key structural features.",
        f"Describe the molecular structure represented by the SMILES {smiles} using clear chemical terminology.",
        f"Write a comprehensive chemical description of the molecule encoded as {smiles}.",
        f"Based on the SMILES {smiles}, explain the molecule’s functional groups, ring systems, and overall structure.",
        f"You are an expert chemist. Describe the molecule represented by {smiles} in detail.",
        f"Provide a detailed molecular caption for the compound specified by the SMILES {smiles}.",
        f"From the SMILES notation {smiles}, generate a thorough description of the molecule’s chemical characteristics.",
        f"Explain the structure and notable chemical features of the molecule defined by {smiles}.",
        f"Using chemical knowledge, describe the molecule represented by the SMILES {smiles}, highlighting important substructures.",
        f"Produce a detailed, text-based molecular description for the compound encoded as {smiles}."
    ]
    if num < 0:
        return random.choice(templates).format(smiles=smiles)
    else:
        return templates[num].format(smiles=smiles)


prompt_dispatcher = {
    "bbbp": bbbp_pred_prompt,
    "hiv": hiv_pred_prompt,
    "bace": bace_pred_prompt,
    "clintox": clintox_pred_prompt,
    "esol": esol_pred_prompt,
    "freesolv": freesolv_pred_prompt,
    "lipo": lipo_pred_prompt,
    "density": polymer_density_pred_prompt,
    "mt": polymer_mt_pred_prompt,
    "o2": polymer_o2_pred_prompt,
    "tg": polymer_tg_pred_prompt,
    "i2s": get_i2s_prompt,
    "s2i": get_s2i_prompt,
    "p2s": get_p2s_prompt,
    "p2i": get_p2i_prompt,
    #"reaction": reaction_pred_prompt,
    #"mol_weight": mol_weight_prompt,
    "logp": logp_prompts,
    "hdonor": hdonor_prompts,
    "hacceptor": hacceptor_prompts,
    "tpsa": tpsa_prompts,
    "rotatable_bond": rotatable_bond_prompts,
    "ring_count": ring_count_prompts,
    "carbon_count": carbon_count_prompts,
    "functional_group": functional_group_prompts,
    "bond_type": bond_type_prompts,
    "formal_charge": formal_charge_prompts,
    #"mole": mole_pred_prompt,
    "mol_weight": mol_weight_prompt,
    "yield": yield_pred_prompt,
    "coeff": coeff_pred_prompt,
    "name_equiv": name_equiv_prompt,
    "chemical_sim": chemical_sim_prompt,
    "property_comp": property_comp_prompt,
    "coeff_calc": coeff_calc_prompt,
    "forward_smiles": forward_prompt,
    "retro_smiles": retro_prompt,
    "forward_formula": forward_prompt,
    "retro_formula": retro_prompt,
    "forward_image": forward_image_prompt,
    "retro_image": retro_image_prompt,
    "molecule_caption": molecule_caption_prompt,
}

def get_prompt(task, num=-1, **kwargs):
    if task not in prompt_dispatcher:
        raise ValueError(f"Task '{task}' not recognized. Available tasks: {list(prompt_dispatcher.keys())}")
    return prompt_dispatcher[task](num=num, **kwargs)