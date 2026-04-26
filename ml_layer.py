from rdkit import Chem
from rdkit.Chem import Descriptors

def predict_drug_properties(smiles, name="Unknown Drug"):
    print(f"\n🔬 --- Analyzing: {name} ---")
    print(f"SMILES: {smiles}")
    
    # 1. Parse the text string into a 3D molecular graph
    molecule = Chem.MolFromSmiles(smiles)
    
    if molecule is None:
        print("❌ Error: Invalid SMILES string!")
        return None
        
    # 2. Calculate the core pharmacokinetic variables
    properties = {
        "Molecular Weight (g/mol)": Descriptors.MolWt(molecule),
        "LogP (Fat Solubility)": Descriptors.MolLogP(molecule),
        "H-Bond Donors": Descriptors.NumHDonors(molecule),
        "H-Bond Acceptors": Descriptors.NumHAcceptors(molecule),
        "Polar Surface Area (Membrane Permeability)": Descriptors.TPSA(molecule)
    }
    
    # 3. Print the predicted results
    for prop, value in properties.items():
        print(f"  {prop}: {value:.2f}")
        
    return properties

# Let's test it on three very different drugs
if __name__ == "__main__":
    # Caffeine (Fast absorbing, crosses blood-brain barrier)
    predict_drug_properties("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine")
    
    # Metformin (Our core test drug - highly water soluble)
    predict_drug_properties("CN(C)C(=N)N=C(N)N", "Metformin")
    
    # Aspirin
    predict_drug_properties("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin")