import sys
import os
sys.path.append(os.getcwd())

from engine.simulator import Simulator
from engine.admet import REFERENCE_DRUGS

sim = Simulator(verbose=True)
caffeine = REFERENCE_DRUGS["caffeine"]

# Run the NEW Uncertainty Analysis (200 simulations in one)
uncertainty_results = sim.run_uncertainty(
    drug=caffeine, 
    dose_mg=100, 
    n_samples=200
)

# Print the NEW Research-Grade report
sim.print_uncertainty_report(uncertainty_results)