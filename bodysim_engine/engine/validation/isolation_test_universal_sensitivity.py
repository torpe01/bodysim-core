"""
isolation_test_universal_sensitivity.py — v5.3 diagnostic isolation test #4.

Purpose
-------
Isolation tests #2 and #3 found that for WARFARIN specifically, F, ka, and
CLint are all completely inert (identical Cmax/AUC to 4 decimals whether
halved, doubled, or forced to F=1.0), while fup and kp_scalar DO move the
result (in the wrong direction, but they move it). Three independently-dead
parameters for one drug is suspicious enough that it could be either:

  (a) something specific to Warfarin's parameter combination (e.g. its
      explicit Vmax_hepatic/Km_hepatic MM kinetics, or its very low fup,
      causing some OTHER term to dominate the ODE so completely that F/ka/
      CLint's contributions become numerically negligible by comparison
      for THIS drug only), or

  (b) a genuine architectural/class-level bug in how F, ka, or CLint are
      wired from the drug profile into the ODE/solver -- e.g. a stale
      cached value, a parameter that is read but never actually applied,
      or a code path that recomputes/overrides these values internally
      regardless of what the profile specifies -- which would affect EVERY
      drug, not just Warfarin.

This script distinguishes (a) from (b) by repeating the same three
perturbations (F -> 1.0, ka x0.5, ka x2.0, clint x0.5, clint x2.0) across
ALL 23 drugs in REFERENCE_PK, not just Warfarin, and flagging any drug/
parameter combination where the perturbed result is numerically identical
(within a tight relative tolerance) to that drug's own unperturbed baseline.

Interpretation:
  - If ONLY Warfarin (or a small handful of drugs sharing some specific
    feature, e.g. explicit Vmax_hepatic/Km_hepatic) show dead parameters,
    this points to (a) -- a drug-specific data/mechanism interaction, not
    a universal engine bug.
  - If MANY or ALL drugs show the same dead-parameter pattern, this points
    to (b) -- a shared, class-level wiring fault in the simulator/ADMET
    pipeline that happens to be most visible for Warfarin because its
    other parameters (fup, kp_scalar) are unusually extreme.

Run from the repository root:
    python engine/validation/isolation_test_universal_sensitivity.py

Does not modify reference_pk.py or any engine file on disk -- every variant
is an in-memory copy of each drug's reference_pk.py dict. tmdd_params is
intentionally NOT removed here (unlike the Warfarin-specific tests) because
this script's purpose is to check F/ka/CLint sensitivity as each drug is
ACTUALLY defined in production, not under a modified baseline.
"""

import copy
import sys
import os

sys.path.append(os.getcwd())

from engine.simulator import Simulator
from engine.admet import build_drug_profile
from engine.validation.reference_pk import REFERENCE_PK

# Relative tolerance below which two results are considered "identical" /
# the parameter is "dead" for that drug. 0.1% — tight enough to catch real
# inertness, loose enough to ignore solver/floating-point noise.
DEAD_PARAM_RTOL = 1e-3


def _build_profile_and_run(name, data, sim):
    """Build a drug profile from a (possibly modified) reference_pk-style
    dict and run a single simulation, exactly mirroring validate_drugs.py's
    call pattern. Returns (cmax, auc) or raises on failure."""
    advanced_keys = [
        "gut_transporter", "phaseII_kinetics", "fu_gut", "CLint_gut_cyp3a4",
        "tmdd_params", "kp_scalar", "cl_bile_lh", "f_reabs_bile", "p_eff",
        "is_uptake_substrate", "vmax_uptake", "km_uptake",
        "Vmax_hepatic", "Km_hepatic", "absorption_segments",
        "enteric_coated", "peff_is_measured_net",
        "albumin_facilitation_threshold", "albumin_facilitation_eff",
    ]
    advanced_kwargs = {k: data[k] for k in advanced_keys if k in data}

    profile = build_drug_profile(
        name=name,
        logp=data["logp"],
        fup=data["fup"],
        mw=data["mw"],
        pka=data.get("pka"),
        drug_type=data.get("drug_type", "neutral"),
        smiles=data["smiles"],
        ka_override=data.get("ka"),
        F_override=data.get("F"),
        clint_override=data.get("clint"),
        clrenal_override=data.get("clrenal"),
        **advanced_kwargs,
    )

    res = sim.run_single(
        drug=profile,
        dose_mg=data["dose"],
        route=data["route"],
        t_end_h=48.0,
    )
    return res["cmax_plasma"], res["auc_plasma"]


def _make_variant(base_data, **overrides):
    data = copy.deepcopy(base_data)
    for key, value in overrides.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def _is_dead(baseline_val, perturbed_val):
    """True if perturbed_val is numerically indistinguishable from
    baseline_val within DEAD_PARAM_RTOL (handles baseline_val == 0)."""
    if baseline_val is None or perturbed_val is None:
        return False  # a failure isn't "dead", it's a different problem
    denom = max(abs(baseline_val), 1e-12)
    return abs(perturbed_val - baseline_val) / denom < DEAD_PARAM_RTOL


def run_universal_sensitivity_test():
    print(f"\n{'='*90}")
    print(" v5.3 ISOLATION TEST #4: Universal F / ka / CLint sensitivity across all drugs")
    print(f"{'='*90}\n")

    sim = Simulator(verbose=False)

    # perturbation_name -> function(base_data) -> overrides dict
    perturbations = {
        "F->1.0":      lambda d: {"F": 1.0},
        "ka x0.5":     lambda d: {"ka": (d.get("ka") / 2.0) if d.get("ka") else None},
        "ka x2.0":     lambda d: {"ka": (d.get("ka") * 2.0) if d.get("ka") else None},
        "clint x0.5":  lambda d: {"clint": (d.get("clint") / 2.0) if d.get("clint") else None},
        "clint x2.0":  lambda d: {"clint": (d.get("clint") * 2.0) if d.get("clint") else None},
    }

    # drug_name -> {"baseline": (cmax, auc), perturbation_name: (cmax, auc), ...}
    all_results = {}
    # drug_name -> list of perturbation_names that were "dead" for this drug
    dead_map = {}

    for drug_name, data in REFERENCE_PK.items():
        print(f" Testing {drug_name} ...")
        drug_results = {}

        try:
            base_cmax, base_auc = _build_profile_and_run(drug_name, data, sim)
        except Exception as e:
            print(f"   BASELINE ERROR: {e}")
            all_results[drug_name] = {"baseline": (None, None)}
            dead_map[drug_name] = []
            continue

        drug_results["baseline"] = (base_cmax, base_auc)
        dead_params_this_drug = []

        for pert_name, pert_fn in perturbations.items():
            overrides = pert_fn(data)
            variant_data = _make_variant(data, **overrides)
            try:
                cmax, auc = _build_profile_and_run(drug_name, variant_data, sim)
            except Exception as e:
                drug_results[pert_name] = (None, None)
                continue
            drug_results[pert_name] = (cmax, auc)
            if _is_dead(base_cmax, cmax) and _is_dead(base_auc, auc):
                dead_params_this_drug.append(pert_name)

        all_results[drug_name] = drug_results
        dead_map[drug_name] = dead_params_this_drug

    # ── Summary table: which drugs have ANY dead parameter ────────────────
    print(f"\n{'='*90}")
    print(" SUMMARY: drugs with at least one numerically dead F/ka/CLint perturbation")
    print(f"{'='*90}")
    header = f" {'Drug':<16} {'Baseline Cmax':>13} {'Baseline AUC':>13}   Dead perturbations"
    print(header)
    print("-" * len(header))

    n_drugs_with_dead = 0
    n_drugs_total = 0
    for drug_name, dead_list in dead_map.items():
        n_drugs_total += 1
        base_cmax, base_auc = all_results[drug_name].get("baseline", (None, None))
        if base_cmax is None:
            print(f" {drug_name:<16} {'FAILED':>13} {'FAILED':>13}   (baseline error)")
            continue
        if dead_list:
            n_drugs_with_dead += 1
            print(f" {drug_name:<16} {base_cmax:>13.4f} {base_auc:>13.4f}   {', '.join(dead_list)}")

    if n_drugs_with_dead == 0:
        print(" (none — every drug responded to every perturbation; no dead parameters found)")

    print(f"\n {n_drugs_with_dead} / {n_drugs_total} drugs have at least one dead F/ka/CLint perturbation.")
    print(f"{'='*90}\n")

    # ── Full detail table (all drugs, all perturbations, raw values) ──────
    print(f"{'='*90}")
    print(" FULL DETAIL: Cmax for every drug x every perturbation (AUC omitted for width;")
    print(" re-run with PRINT_AUC=1 env var or inspect all_results in an interactive session")
    print(" if you need AUC too — both are checked for 'dead' status above)")
    print(f"{'='*90}")
    pert_names = list(perturbations.keys())
    header2 = f" {'Drug':<16} {'baseline':>10} " + " ".join(f"{p:>10}" for p in pert_names)
    print(header2)
    print("-" * len(header2))
    for drug_name, drug_results in all_results.items():
        base_cmax = drug_results.get("baseline", (None, None))[0]
        if base_cmax is None:
            print(f" {drug_name:<16} {'FAILED':>10}")
            continue
        row = f" {drug_name:<16} {base_cmax:>10.4f} "
        for p in pert_names:
            val = drug_results.get(p, (None, None))[0]
            row += f"{val:>10.4f} " if val is not None else f"{'ERR':>10} "
        print(row)
    print(f"{'='*90}\n")

    # ── Interpretation ──────────────────────────────────────────────────
    print(" Interpretation guide:")
    print(" - If dead parameters cluster on ONLY Warfarin (or a small group")
    print("   sharing a specific feature, e.g. explicit Vmax_hepatic/")
    print("   Km_hepatic, or very low fup/extreme kp_scalar), this points to")
    print("   a drug-specific data/mechanism interaction — investigate")
    print("   Warfarin's particular parameter combination further, not the")
    print("   shared simulator/ADMET code.")
    print(" - If dead parameters appear across MANY or MOST drugs regardless")
    print("   of their individual mechanism flags, this points to a genuine")
    print("   class-level/architectural bug: F, ka, or CLint may be read")
    print("   from the drug profile but never actually consumed by the ODE,")
    print("   silently overridden by an internally-recomputed value, or")
    print("   shadowed by a stale cached attribute somewhere in Simulator")
    print("   or the PBPK ODE assembly — in which case the fix belongs in")
    print("   the shared pipeline, not in any one drug's reference_pk.py")
    print("   entry.")
    print(" - Pay particular attention to whether the SAME perturbation")
    print("   (e.g. 'clint x0.5') is dead for many drugs while a DIFFERENT")
    print("   perturbation (e.g. 'ka x0.5') is live for those same drugs —")
    print("   that pattern would localize the bug to one specific parameter")
    print("   pipeline (e.g. clint_override handling) rather than implicate")
    print("   the whole engine.")
    print()


if __name__ == "__main__":
    run_universal_sensitivity_test()