from engine.simulator import Simulator

sim = Simulator(verbose=True)

# 1. Input: Just the SMILES for Aspirin
aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"

# 2. Action: Run a 300-person virtual trial
print(f"--- STARTING 300-SUBJECT VIRTUAL TRIAL FOR ASPIRIN ---")
pop_result = sim.run_population(
    smiles="CC(=O)Oc1ccccc1C(=O)O", 
    name="Aspirin (Validated)",
    dose_mg=325,
    # These overrides act as "Perfect AI" predictions
    logp=1.19, 
    fup=0.15,  # 85% protein bound
    clint_override=10.0, 
    clrenal_override=0.5
)

# 3. Output: Check the performance
summary = pop_result["population_risk"]["population_summary"]
print(f"\n--- TRIAL COMPLETE ---")
print(f"Total Subjects: {pop_result['n_subjects']}")
print(f"Population Risk Summary: {summary}")