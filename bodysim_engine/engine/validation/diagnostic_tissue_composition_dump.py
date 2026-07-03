"""
diagnostic_tissue_composition_dump.py — Script 10 (v5.3 investigation log)

PURPOSE
-------
Script 9 found that 8 of 11 tissues (muscle, bone, skin, rest, liver, gut,
heart, kidney) produce near-identical Kp values for every drug tested —
in Furosemide's case, EXACTLY identical (0.050) across all 10 non-lung
tissues before the floor clamp.

estimate_kp_values() in admet.py computes:
    kp_passive = (fw + fn*Kn*P + fp*Kph*P) / (fw*P + fn*Kn + fp*Kph)
where Kn, Kph, P are DRUG-level globals (same for every tissue), and the
only per-tissue inputs are (fw, fn, fp) from TISSUE_COMPOSITION and the
organ pH (via ion_correction). This script settles, with zero computation
of its own, whether the clustering originates in TISSUE_COMPOSITION having
near-duplicate (fw, fn, fp) tuples for those 8 tissues, or whether the
composition table is fine and the clustering comes from somewhere else
(e.g. ion_correction dominating because ORGAN_PH is narrow-banded).

This is a raw value dump only — no drug profile is built, no Kp is computed.
"""

import sys
import os

sys.path.append(os.getcwd())

from engine.physiology import TISSUE_COMPOSITION

print(f"\n{'='*72}")
print(f"  SCRIPT 10 — TISSUE_COMPOSITION RAW VALUE DUMP")
print(f"  Testing: does the composition table itself differentiate tissues?")
print(f"{'='*72}\n")

print(f"  {'Organ':<16} {'fw (water)':>12} {'fn (neut.lipid)':>16} {'fp (phospho.)':>14}")
print(f"  {'-'*60}")

rows = []
for organ, vals in TISSUE_COMPOSITION.items():
    fw, fn, fp = vals
    rows.append((organ, fw, fn, fp))
    print(f"  {organ:<16} {fw:>12.4f} {fn:>16.4f} {fp:>14.4f}")

print(f"\n{'='*72}")
print(f"  DUPLICATE / NEAR-DUPLICATE CHECK")
print(f"{'='*72}")

# Group organs whose (fw, fn, fp) tuples are within 1% of each other on
# every dimension — this is the direct test of the Script 9 hypothesis.
TOL = 0.01
clusters = []
used = set()
for i, (o1, fw1, fn1, fp1) in enumerate(rows):
    if o1 in used:
        continue
    cluster = [o1]
    used.add(o1)
    for o2, fw2, fn2, fp2 in rows[i+1:]:
        if o2 in used:
            continue
        if (abs(fw1 - fw2) <= TOL and abs(fn1 - fn2) <= TOL and abs(fp1 - fp2) <= TOL):
            cluster.append(o2)
            used.add(o2)
    if len(cluster) > 1:
        clusters.append(cluster)

if clusters:
    print(f"\n  [FOUND] {len(clusters)} cluster(s) of tissues with (fw,fn,fp) "
          f"matching within {TOL} on every dimension:")
    for c in clusters:
        print(f"    {c}")
    total_clustered = sum(len(c) for c in clusters)
    print(
        f"\n  {total_clustered}/{len(rows)} tissues fall into duplicate/near-duplicate "
        f"composition groups. This directly explains the Script 9 finding: "
        f"any two tissues with matching (fw,fn,fp) will compute IDENTICAL "
        f"kp_passive for a given drug (same Kn/Kph/P, same formula) UNLESS "
        f"ORGAN_PH differs enough between them to produce a different "
        f"ion_correction. Check ORGAN_PH for these clustered organs next — "
        f"if pH is also similar across the cluster, the model is structurally "
        f"unable to differentiate these tissues for ANY drug, regardless of "
        f"chemistry. This is a tissue-resolution defect, not a calibration issue."
    )
else:
    print(
        f"\n  [NOT FOUND] No tissues have matching (fw,fn,fp) within {TOL}. "
        f"TISSUE_COMPOSITION is NOT the source of the Script 9 clustering. "
        f"The near-identical Kp values must originate from ion_correction "
        f"dominating the formula (ORGAN_PH is narrow-banded: most non-gut "
        f"tissues sit at pH 7.0-7.4). Next step: recompute kp_passive for "
        f"two clustered tissues with ion_correction forced to 1.0 for both, "
        f"to confirm whether removing the ionization term restores "
        f"tissue-to-tissue differentiation."
    )

print(f"\n{'='*72}\n")