"""
isolation_test_universal_sensitivity_2.py — v5.3 diagnostic isolation test #6
(universal; direct extension of isolation_test_universal_sensitivity.py).

Purpose
-------
isolation_test_universal_sensitivity.py (script #4) confirmed F and ka are
universally dead and CLint is correctly bypassed for the 3 explicit-MM
drugs only. That investigation left two parameters unchecked that sit in
the exact same risk category -- "looks correctly wired in admet.py, never
verified at the actual consumption end in the ODE/acat code":

  1. CLrenal (clrenal_override) -- consumed by renal_module.py, in
     principle, but never isolation-tested the way clint was.
  2. absorption_segments / enteric_coated -- confirmed read by
     acat_module.py during the v5.2 work, but never swept across all 23
     drugs to confirm they actually change simulation output for every
     drug they're applied to, not just the one or two drugs that already
     define them in reference_pk.py.

This script runs the same universal dead-parameter methodology as script
#4, extended to these three parameters, across all 23 drugs.

Variants per drug:
  - clrenal x0.5 / x2.0          (only if the drug has a clrenal value)
  - absorption_segments-RESTRICT  (existing value ignored; a synthetic
                                   restriction to the first two ACAT
                                   segments is applied to EVERY drug,
                                   regardless of whether it already defines
                                   absorption_segments, so the test is
                                   uniform across all 23 -- a drug that
                                   already restricts to [1,2] gets the same
                                   restriction re-applied as a no-op sanity
                                   check; one that has no restriction at
                                   all should show a LARGE change if the
                                   parameter is actually wired correctly)
  - enteric_coated-TOGGLE          (forces enteric_coated=True for every
                                   drug, regardless of its current value,
                                   for the same uniformity reason)

Run from the repository root:
    python engine/validation/isolation_test_universal_sensitivity_2.py
"""

import copy
import sys
import os

sys.path.append(os.getcwd())

from engine.simulator import Simulator
from engine.admet import build_drug_profile
from engine.validation.reference_pk import REFERENCE_PK

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
    if baseline_val is None or perturbed_val is None:
        return False
    denom = max(abs(baseline_val), 1e-12)
    return abs(perturbed_val - baseline_val) / denom < DEAD_PARAM_RTOL


def _clrenal_variants(data):
    """Return {perturbation_name: overrides_dict} for clrenal, or {} if
    this drug has no clrenal value to perturb."""
    base_clrenal = data.get("clrenal")
    if not base_clrenal:
        return {}
    return {
        "clrenal x0.5": {"clrenal": base_clrenal / 2.0},
        "clrenal x2.0": {"clrenal": base_clrenal * 2.0},
    }


def run_universal_sensitivity_test_2():
    print(f"\n{'='*96}")
    print(" v5.3 ISOLATION TEST #6: Universal CLrenal / absorption_segments / enteric_coated")
    print(f"{'='*96}\n")

    sim = Simulator(verbose=False)

    all_results = {}
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

        # Build this drug's perturbation set. clrenal is conditional
        # (skipped if absent); the other two are applied uniformly to
        # every drug regardless of their current reference_pk.py value.
        perturbations = {}
        perturbations.update(_clrenal_variants(data))
        perturbations["absorption_segments-RESTRICT[1,2]"] = {
            "absorption_segments": [1, 2]
        }
        perturbations["enteric_coated-TOGGLE(True)"] = {
            "enteric_coated": True
        }

        for pert_name, overrides in perturbations.items():
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

    # ── Summary: which drugs show a dead perturbation for each parameter ──
    print(f"\n{'='*96}")
    print(" SUMMARY: drugs with at least one numerically dead perturbation")
    print(f"{'='*96}")
    header = f" {'Drug':<16} {'Baseline Cmax':>13} {'Baseline AUC':>13}   Dead perturbations"
    print(header)
    print("-" * len(header))

    n_with_dead = 0
    n_total = 0
    for drug_name, dead_list in dead_map.items():
        n_total += 1
        base_cmax, base_auc = all_results[drug_name].get("baseline", (None, None))
        if base_cmax is None:
            print(f" {drug_name:<16} {'FAILED':>13} {'FAILED':>13}   (baseline error)")
            continue
        if dead_list:
            n_with_dead += 1
            print(f" {drug_name:<16} {base_cmax:>13.4f} {base_auc:>13.4f}   {', '.join(dead_list)}")

    if n_with_dead == 0:
        print(" (none — every drug responded to every perturbation; no dead parameters found)")
    print(f"\n {n_with_dead} / {n_total} drugs have at least one dead perturbation among "
          f"clrenal / absorption_segments / enteric_coated.")
    print(f"{'='*96}\n")

    # ── Per-parameter breakdown (counts dead occurrences for each named
    #    perturbation across drugs that were actually tested for it) ──────
    param_groups = {
        "clrenal (x0.5 or x2.0)": ["clrenal x0.5", "clrenal x2.0"],
        "absorption_segments-RESTRICT[1,2]": ["absorption_segments-RESTRICT[1,2]"],
        "enteric_coated-TOGGLE(True)": ["enteric_coated-TOGGLE(True)"],
    }
    print(f"{'='*96}")
    print(" PER-PARAMETER BREAKDOWN")
    print(f"{'='*96}")
    for group_label, pert_names in param_groups.items():
        tested = 0
        dead = 0
        dead_drugs = []
        for drug_name, drug_results in all_results.items():
            relevant = [p for p in pert_names if p in drug_results]
            if not relevant:
                continue
            tested += 1
            base_cmax, base_auc = drug_results.get("baseline", (None, None))
            if base_cmax is None:
                continue
            is_dead_for_drug = all(
                _is_dead(base_cmax, drug_results[p][0]) and _is_dead(base_auc, drug_results[p][1])
                for p in relevant if drug_results.get(p, (None, None))[0] is not None
            )
            if is_dead_for_drug:
                dead += 1
                dead_drugs.append(drug_name)
        print(f" {group_label}: dead for {dead}/{tested} drugs tested.")
        if dead_drugs:
            print(f"   Dead in: {', '.join(dead_drugs)}")
    print(f"{'='*96}\n")

    # ── Full detail table ──────────────────────────────────────────────
    print(f"{'='*96}")
    print(" FULL DETAIL: Cmax for every drug x every perturbation it was tested with")
    print(f"{'='*96}")
    for drug_name, drug_results in all_results.items():
        base_cmax = drug_results.get("baseline", (None, None))[0]
        if base_cmax is None:
            print(f" {drug_name:<16} FAILED")
            continue
        line = f" {drug_name:<16} baseline={base_cmax:.4f}  "
        for pert_name, (cmax, auc) in drug_results.items():
            if pert_name == "baseline":
                continue
            val_str = f"{cmax:.4f}" if cmax is not None else "ERR"
            line += f"| {pert_name}={val_str}  "
        print(line)
    print(f"{'='*96}\n")

    print(" Interpretation guide:")
    print(" - clrenal dead for many drugs -> renal_module.py is not actually")
    print("   consuming CLrenal from the drug profile correctly, OR renal")
    print("   clearance is structurally negligible relative to other")
    print("   elimination pathways for those drugs (check which before")
    print("   concluding it's a bug — some drugs are legitimately hepatic-")
    print("   dominant with near-zero renal contribution by design).")
    print(" - absorption_segments-RESTRICT[1,2] dead for a drug with NO prior")
    print("   absorption_segments key -> the parameter is not being read at")
    print("   all for that drug's code path; if dead for the ONE drug that")
    print("   already defines it (re-applying the same restriction), that is")
    print("   an expected no-op, not a bug — check the per-parameter")
    print("   breakdown's drug list against reference_pk.py's existing")
    print("   absorption_segments entries before concluding anything.")
    print(" - enteric_coated-TOGGLE(True) dead for any drug -> the same")
    print("   concern: confirm whether that drug's absorption is dominated")
    print("   by segments other than the stomach (segment 0) such that")
    print("   blocking segment-0 absorption genuinely has negligible effect")
    print("   (plausible for some drugs), versus the flag not being read at")
    print("   all (a real bug, expected to show up across MANY drugs if so,")
    print("   per the F/ka precedent from isolation test #4).")
    print()


if __name__ == "__main__":
    run_universal_sensitivity_test_2()