# sanity_check_tmdd.py — run standalone, not part of validate_drugs.py
import sys
import os

# Adjust path to import engine components from the root directory
sys.path.append(os.getcwd())
import numpy as np
from engine.hepatic_module import HepaticClearanceModule

mod = HepaticClearanceModule()
Bmax, Kd = 100.0, 0.1   # Warfarin's literature values, unchanged

# 1. Monotonicity check: as C_tissue_free (naive, pre-target-binding)
#    increases from 0 to a value well above Bmax, f_tmdd_free must increase
#    monotonically toward 1.0 (at high concentration, the binding sites
#    saturate and most NEW drug is free).
prev_f = -1.0
for C_naive in np.linspace(0.0, 500.0, 50):
    # Inline the same algebra as Step 3a for a quick check:
    a, b, c = 1.0, (Kd + Bmax - C_naive), -Kd * C_naive
    disc = max(b*b - 4*a*c, 0.0)
    C_free = max((-b + np.sqrt(disc)) / (2*a), 0.0)
    f = 0.0 if C_naive <= 1e-15 else np.clip(C_free / C_naive, 0.0, 1.0)
    assert f >= prev_f - 1e-9, f"f_tmdd_free not monotonic at C_naive={C_naive}"
    assert 0.0 <= f <= 1.0, f"f_tmdd_free out of bounds at C_naive={C_naive}"
    prev_f = f

# 2. Conservation check: C_free + C_bound must equal C_naive
#    (within float tolerance) across the same sweep — confirms the
#    quadratic root is being solved correctly and no mass is invented
#    or destroyed by the algebra itself.
for C_naive in np.linspace(0.0, 500.0, 50):
    a, b, c = 1.0, (Kd + Bmax - C_naive), -Kd * C_naive
    disc = max(b*b - 4*a*c, 0.0)
    C_free = max((-b + np.sqrt(disc)) / (2*a), 0.0)
    C_bound = Bmax * C_free / (Kd + C_free) if (Kd + C_free) > 1e-15 else 0.0
    assert abs((C_free + C_bound) - C_naive) < 1e-6 * max(C_naive, 1.0), \
        f"Mass not conserved at C_naive={C_naive}: free={C_free}, bound={C_bound}, naive={C_naive}"

print("TMDD algebra: monotonicity and mass-conservation checks passed.")