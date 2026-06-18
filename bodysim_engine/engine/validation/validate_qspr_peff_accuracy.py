"""
validate_qspr_peff_accuracy.py — v5.3 diagnostic #7 (universal).

Purpose
-------
diagnostic_universal_peff_coverage.py (diagnostic #5) found that 12 of 23
drugs in reference_pk.py have no explicit p_eff and silently fall through
to acat_module.py's generic QSPR estimate (Egan-style transcellular +
paracellular permeability from logp/mw/hbd). That diagnostic flagged two
fallback drugs (Digoxin, Rosuvastatin) as mechanistically suspect by
COMPARING fallback estimates against the CLOSEST-logP literature drug —
a rough plausibility check, not a real accuracy measurement.

This script does the real measurement the prior diagnostic could not:
it runs the QSPR formula AS-IS against the 11 drugs that already HAVE an
explicit, literature-sourced p_eff, computes the fold-error between the
QSPR estimate and the real value for each one, and reports an honest,
measured accuracy bound for the fallback formula — e.g. "QSPR is within
3x of literature for 8/11 reference compounds" — rather than an assumed
or hoped-for accuracy.

This number is directly useful for two purposes:
  1. Deciding whether the QSPR fallback is "good enough" to leave running
     silently for the 12 drugs without literature data, as a documented,
     measured policy rather than an assumption.
  2. Providing a literature-grounded uncertainty bound to attach to every
     Tier-4 (QSPR_ESTIMATE) drug's results in any production-facing
     report, per the p_eff_provenance_schema.md design (Section 1).

This script does NOT call build_drug_profile() or run any simulation -- it
only re-implements the QSPR formula verbatim (matching acat_module.py
exactly) and compares it against reference_pk.py's existing explicit
values, for the 11 drugs where ground truth already exists in the dataset.

Run from the repository root:
    python engine/validation/validate_qspr_peff_accuracy.py
"""

import sys
import os
import numpy as np

sys.path.append(os.getcwd())

from engine.validation.reference_pk import REFERENCE_PK

# Fold-error thresholds used to bucket each reference compound's QSPR
# accuracy. These are reporting buckets only, not pass/fail gates --
# unlike validate_drugs.py's drug-level Cmax/AUC tolerance, there is no
# established "acceptable" QSPR permeability fold-error in the literature,
# so this script reports the measured distribution rather than asserting
# a threshold.
FOLD_ERROR_BUCKETS = [
    (2.0,  "within 2x"),
    (3.0,  "within 3x"),
    (5.0,  "within 5x"),
    (10.0, "within 10x"),
    (float("inf"), "beyond 10x"),
]


def _qspr_p_eff_fallback(logp, mw, hbd=0):
    """
    Verbatim reimplementation of acat_module.py's QSPR p_eff fallback
    (lines ~119-124). Must be kept in sync with that function -- if
    acat_module.py's formula changes, update this copy too, or this
    script will silently validate the WRONG formula.
    """
    p_trans = float(np.clip(10.0 ** (0.4 * logp - 5.5), 1e-8, 1e-3))
    p_para = float(np.clip(
        1.5e-5 * np.exp(-0.010 * max(0.0, mw - 100.0)) * (0.85 ** hbd),
        1e-9, 1e-4,
    ))
    return float(np.clip(np.sqrt(p_trans**2 + p_para**2), 1e-8, 5e-4))


def _fold_error_bucket(fold_error):
    for threshold, label in FOLD_ERROR_BUCKETS:
        if fold_error <= threshold:
            return label
    return FOLD_ERROR_BUCKETS[-1][1]


def run_qspr_accuracy_validation():
    print(f"\n{'='*96}")
    print(" v5.3 DIAGNOSTIC #7: QSPR p_eff fallback accuracy against literature reference drugs")
    print(f"{'='*96}\n")

    reference_drugs = []
    for drug_name, data in REFERENCE_PK.items():
        explicit_p_eff = data.get("p_eff", None)
        if explicit_p_eff is None:
            continue  # this drug IS a fallback drug -- not ground truth
        logp = data.get("logp", 0.0)
        mw = data.get("mw", 300.0)
        hbd = data.get("hbd", 0)  # always 0 in current dataset -- see below
        qspr_estimate = _qspr_p_eff_fallback(logp, mw, hbd)
        fold_error = max(qspr_estimate, explicit_p_eff) / min(qspr_estimate, explicit_p_eff)
        direction = "OVER" if qspr_estimate > explicit_p_eff else "UNDER"
        reference_drugs.append({
            "drug": drug_name,
            "logp": logp,
            "mw": mw,
            "literature_p_eff": explicit_p_eff,
            "qspr_p_eff": qspr_estimate,
            "fold_error": fold_error,
            "direction": direction,
        })

    if not reference_drugs:
        print(" [!] No drugs with explicit p_eff found in REFERENCE_PK -- cannot validate.")
        return

    reference_drugs.sort(key=lambda r: r["fold_error"])

    # ── Table: every reference drug, sorted best-to-worst QSPR agreement ──
    header = f" {'Drug':<16} {'logP':>7} {'Lit. p_eff':>13} {'QSPR p_eff':>13} {'Fold-err':>9} {'Dir':>6} {'Bucket':<12}"
    print(header)
    print("-" * len(header))
    for r in reference_drugs:
        bucket = _fold_error_bucket(r["fold_error"])
        print(f" {r['drug']:<16} {r['logp']:>7.2f} {r['literature_p_eff']:>13.3e} "
              f"{r['qspr_p_eff']:>13.3e} {r['fold_error']:>9.2f} {r['direction']:>6} {bucket:<12}")

    # ── Summary statistics ─────────────────────────────────────────────
    n = len(reference_drugs)
    fold_errors = [r["fold_error"] for r in reference_drugs]
    print(f"\n{'='*96}")
    print(f" Measured QSPR fallback accuracy across {n} literature-sourced reference compounds:")
    print(f"   Median fold-error: {np.median(fold_errors):.2f}x")
    print(f"   Mean fold-error:   {np.mean(fold_errors):.2f}x")
    print(f"   Best case:         {min(fold_errors):.2f}x ({reference_drugs[0]['drug']})")
    print(f"   Worst case:        {max(fold_errors):.2f}x ({reference_drugs[-1]['drug']})")

    bucket_counts = {}
    for r in reference_drugs:
        b = _fold_error_bucket(r["fold_error"])
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
    print(f"\n   Distribution:")
    for _, label in FOLD_ERROR_BUCKETS:
        count = bucket_counts.get(label, 0)
        if count:
            print(f"     {label:<14} {count}/{n} drugs ({100.0*count/n:.0f}%)")

    n_over = sum(1 for r in reference_drugs if r["direction"] == "OVER")
    n_under = n - n_over
    print(f"\n   Direction bias: QSPR over-predicts for {n_over}/{n} drugs, "
          f"under-predicts for {n_under}/{n} drugs.")
    print(f"   (A roughly even split suggests no systematic directional bias; a lopsided")
    print(f"   split suggests the formula's coefficients may need recalibration in one")
    print(f"   direction, not just tighter variance.)")

    print(f"\n{'='*96}\n")

    print(" Interpretation guide:")
    print(" - This is a MEASURED accuracy bound, not an assumed one. Use the median/mean")
    print("   fold-error above as the documented uncertainty caveat for every Tier-4")
    print("   (QSPR_ESTIMATE) drug's results in any production-facing report, per")
    print("   p_eff_provenance_schema.md.")
    print(" - hbd is 0 for every drug tested here (no 'hbd' key exists anywhere in the")
    print("   current reference_pk.py dataset) -- meaning this accuracy measurement does")
    print("   NOT test the hbd-dependent paracellular term at all. If hbd is later added")
    print("   for any drug, re-run this validation, since the formula's accuracy on")
    print("   high-HBD compounds is currently completely unverified.")
    print(" - A drug landing in 'beyond 10x' here is a strong, MEASURED signal (not just")
    print("   a logP-neighbor heuristic) that the QSPR formula fails for that drug's")
    print("   specific physicochemical profile -- worth checking whether similar")
    print("   fallback-tier drugs share that same profile (e.g. known efflux/uptake")
    print("   transporter substrates the formula cannot represent, since it is purely")
    print("   passive-permeability based).")
    print()


if __name__ == "__main__":
    run_qspr_accuracy_validation()