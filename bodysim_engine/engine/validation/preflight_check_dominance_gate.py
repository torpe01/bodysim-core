# preflight_check_dominance_gate.py — run standalone before Step 1's
# code is wired into build_drug_profile(), and again any time
# reference_pk.py is edited.
#
# IMPORTANT: these three constants must stay numerically identical to
# admet.py's _LIVER_CL_PASSIVE_TYPICAL_LH / _LIVER_DOMINANCE_RATIO /
# _LIVER_BORDERLINE_RATIO. Recommended: factor them into one shared
# module (e.g. engine/constants.py) and import in both places, rather
# than maintaining two copies that can silently drift apart.
import sys
import os

# Adjust path to import engine components from the root directory
sys.path.append(os.getcwd())
from engine.validation.reference_pk import REFERENCE_PK

CL_PASSIVE_TYPICAL_LH = 10.0   # = admet.py's _LIVER_CL_PASSIVE_TYPICAL_LH
DOMINANCE_RATIO       = 3.0    # = admet.py's _LIVER_DOMINANCE_RATIO
BORDERLINE_RATIO      = 1.0    # = admet.py's _LIVER_BORDERLINE_RATIO

indeterminate, dominant, minor, borderline = [], [], [], []

for drug_name, data in REFERENCE_PK.items():
    if not data.get("is_uptake_substrate"):
        continue
    vmax = data.get("vmax_uptake")
    km   = data.get("km_uptake")
    if not vmax or not km or float(km) <= 0.0:
        indeterminate.append(drug_name)
        continue
    cl_active = float(vmax) / float(km)
    ratio = cl_active / CL_PASSIVE_TYPICAL_LH
    if ratio > DOMINANCE_RATIO:
        dominant.append((drug_name, ratio))
    elif ratio > BORDERLINE_RATIO:
        borderline.append((drug_name, ratio))
    else:
        minor.append((drug_name, ratio))

print(f"Dominant (liver exempted):      {dominant}")
print(f"Borderline (review suggested):  {borderline}")
print(f"Minor (no exemption):           {minor}")
print(f"Indeterminate (missing data):   {indeterminate}")

assert not indeterminate, (
    f"{len(indeterminate)} drug(s) have is_uptake_substrate=True but "
    f"missing/invalid vmax_uptake or km_uptake: {indeterminate}. "
    f"Fix reference_pk.py before proceeding, or accept that these drugs "
    f"will silently use the conservative (non-exempt) path with a runtime "
    f"warning each time validate_drugs.py runs."
)
print("Pre-flight check passed: no drug hits the indeterminate branch.")