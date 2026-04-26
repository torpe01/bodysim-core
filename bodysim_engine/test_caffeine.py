import sys
import os

# Fix path visibility for Codespaces
sys.path.append(os.getcwd())

# Import the master controller
from engine.simulator import Simulator

def run_trial(smiles, drug_name, dose):
    print(f"\n==================================================")
    print(f">>> Starting Trial: {drug_name} ({dose}mg)")
    print(f"==================================================")
    
    # Initialize the master Simulator
    sim = Simulator(verbose=False)
    
    # Run the simulation using JUST the SMILES and Dose (No num_subjects flag!)
    pop_result = sim.run_population(
        smiles=smiles, 
        name=drug_name,
        dose_mg=dose
    )
    
    # Print the Risk Summary
    summary = pop_result["population_risk"]["population_summary"]
    print(f"\n--- {drug_name.upper()} TRIAL COMPLETE ---")
    print(f"Population Risk Summary: {summary}")

if __name__ == "__main__":
    # TRIAL 1: Caffeine (Typical Dose: 100mg)
    caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    run_trial(caffeine_smiles, "Caffeine", dose=100)
    
    # TRIAL 2: Aspirin (Typical Dose: 325mg)
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    run_trial(aspirin_smiles, "Aspirin", dose=325)