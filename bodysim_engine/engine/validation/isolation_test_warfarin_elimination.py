"""
isolation_test_warfarin_elimination.py — v5.3 diagnostic isolation test #2.

Purpose
-------
isolation_test_warfarin_tmdd.py established that removing tmdd_params
entirely (NO_TMDD) gets CLOSER to Warfarin's clinical target than either
binding architecture (the new v5.3 parallel quadratic OR the old v5.2
sequential filter) — meaning TMDD is not the dominant lever holding Warfarin
back, and may even be actively hurting the result for this drug's specific
parameters. This script follows up by isolating four OTHER candidate causes,
one at a time, with tmdd_params removed in every variant (since variant 1
above showed it's a net negative) so each test cleanly measures its own
lever rather than a residual interaction with TMDD.

Variants (each holds every reference_pk.py value fixed except the one
change named):

  1. NO_TMDD_BASELINE   — tmdd_params removed, everything else stock.
                            (Reference point for all other variants below.)
  2. NO_TMDD_KP1         — tmdd_params removed AND kp_scalar forced to 1.0
                            (no peripheral Vd correction at all). Tests
                            whether the passive Kp/Vd side is a bottleneck
                            independent of binding.
  3. NO_TMDD_LINEAR_CL   — tmdd_params removed AND Vmax_hepatic/Km_hepatic
                            removed (forces Path B: pure linear
                            CLh * C_tissue_free). Tests whether explicit MM
                            hepatic kinetics are suppressing or inflating
                            elimination relative to simple linear clearance.
  4. NO_TMDD_CLINT_HALF  — tmdd_params removed AND clint halved. Tests basic
                            sensitivity: is Cmax/AUC even in a regime where
                            CLint changes move the result meaningfully, or
                            is something else structurally dominant?
  5. NO_TMDD_CLINT_DOUBLE— tmdd_params removed AND clint doubled. Same
                            sensitivity check in the other direction.
  6. NO_TMDD_NO_BILE     — tmdd_params removed AND cl_bile_lh/f_reabs_bile
                            removed. Tests how much of total elimination is
                            currently attributed to the biliary pathway vs.
                            hepatic CYP/renal — if removing bile barely
                            changes the result, biliary secretion isn't a
                            meaningful contributor for Warfarin as currently
                            parameterized (or vice versa).

Run from the repository root:
    python engine/validation/isolation_test_warfarin_elimination.py

Does not modify reference_pk.py or any engine file on disk — every variant
is an in-memory copy of the Warfarin dict.
"""

import copy
import sys
import os

sys.path.append(os.getcwd())

from engine.simulator import Simulator
from engine.admet import build_drug_profile
from engine.validation.reference_pk import REFERENCE_PK


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


def _make_variant(base_data, **overrides_and_removals):
    """
    Build an in-memory copy of base_data with keys overridden or removed.
    Pass a value of None for a key to REMOVE it (pop), any other value to
    SET it. Keys not mentioned are left untouched.
    """
    data = copy.deepcopy(base_data)
    for key, value in overrides_and_removals.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def run_isolation_test():
    print(f"\n{'='*78}")
    print(" v5.3 ISOLATION TEST #2: Warfarin elimination/distribution levers")
    print(f"{'='*78}\n")

    if "Warfarin" not in REFERENCE_PK:
        print("[!] 'Warfarin' not found in REFERENCE_PK — cannot run test.")
        return

    base = REFERENCE_PK["Warfarin"]
    target_cmax = base["cmax"]
    target_auc = base["auc"]
    base_clint = base.get("clint")

    sim = Simulator(verbose=False)

    variants = {
        "NO_TMDD_BASELINE (tmdd removed, else stock)":
            _make_variant(base, tmdd_params=None),

        "NO_TMDD_KP1 (+ kp_scalar forced to 1.0)":
            _make_variant(base, tmdd_params=None, kp_scalar=1.0),

        "NO_TMDD_LINEAR_CL (+ MM kinetics removed, Path B linear)":
            _make_variant(base, tmdd_params=None, Vmax_hepatic=None, Km_hepatic=None),

        "NO_TMDD_CLINT_HALF (+ clint halved)":
            _make_variant(base, tmdd_params=None,
                           clint=(base_clint / 2.0) if base_clint else None),

        "NO_TMDD_CLINT_DOUBLE (+ clint doubled)":
            _make_variant(base, tmdd_params=None,
                           clint=(base_clint * 2.0) if base_clint else None),

        "NO_TMDD_NO_BILE (+ biliary EHC removed)":
            _make_variant(base, tmdd_params=None, cl_bile_lh=None, f_reabs_bile=None),
    }

    results = {}
    for label, data in variants.items():
        print(f" Running {label} ...")
        try:
            cmax, auc = _build_profile_and_run("Warfarin", data, sim)
            results[label] = (cmax, auc)
        except Exception as e:
            print(f"   ERROR: {e}")
            results[label] = (None, None)

    print(f"\n{'='*78}")
    print(f" Target (clinical):  Cmax = {target_cmax:.4f}   AUC = {target_auc:.4f}")
    print(f" Baseline clint (reference_pk.py): {base_clint}")
    print(f"{'='*78}")
    header = f" {'Variant':<52} {'Cmax':>9} {'CmaxFold':>9} {'AUC':>9} {'AUCFold':>9}"
    print(header)
    print("-" * len(header))
    for label, (cmax, auc) in results.items():
        if cmax is None:
            print(f" {label:<52} {'FAILED':>9}")
            continue
        cmax_fold = cmax / target_cmax if target_cmax else float("nan")
        auc_fold = auc / target_auc if target_auc else float("nan")
        print(f" {label:<52} {cmax:>9.4f} {cmax_fold:>9.4f} {auc:>9.4f} {auc_fold:>9.4f}")
    print(f"{'='*78}\n")

    print(" Interpretation guide:")
    print(" - Compare every row against NO_TMDD_BASELINE, not against the")
    print("   clinical target directly — the question each row answers is")
    print("   'does THIS lever move the result toward or away from target,")
    print("   and by how much, relative to the no-TMDD reference point.'")
    print(" - NO_TMDD_KP1 far from NO_TMDD_BASELINE -> kp_scalar/passive Kp")
    print("   is a meaningful lever; if nearly identical, Kp is not the")
    print("   bottleneck and the Rodgers-Rowland/kp_scalar side can likely")
    print("   be ruled out for Warfarin specifically.")
    print(" - NO_TMDD_LINEAR_CL far from NO_TMDD_BASELINE -> the explicit")
    print("   Vmax_hepatic/Km_hepatic MM kinetics matter a lot; if nearly")
    print("   identical, Warfarin's CLint=... is operating in the linear")
    print("   range regardless of the MM parameters and they aren't the")
    print("   issue.")
    print(" - Compare NO_TMDD_CLINT_HALF and NO_TMDD_CLINT_DOUBLE to see how")
    print("   sensitive Cmax/AUC are to CLint magnitude at all. If doubling")
    print("   or halving CLint barely moves the fold error, NO realistic")
    print("   CLint value can close a gap this large alone -- the dominant")
    print("   defect is structural (absorption F/ka, renal contribution,")
    print("   or volume of distribution), not hepatic clearance magnitude.")
    print(" - NO_TMDD_NO_BILE far from NO_TMDD_BASELINE -> biliary secretion")
    print("   is a material fraction of total elimination for Warfarin as")
    print("   currently parameterized; if nearly identical, it is")
    print("   negligible and not worth further attention here.")
    print()


if __name__ == "__main__":
    run_isolation_test()