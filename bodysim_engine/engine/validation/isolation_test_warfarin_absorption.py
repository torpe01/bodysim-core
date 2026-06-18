"""
isolation_test_warfarin_absorption.py — v5.3 diagnostic isolation test #3.

Purpose
-------
Isolation test #1 (TMDD) showed removing tmdd_params gets closer to target
than either binding architecture. Isolation test #2 (elimination) showed
CLint is completely inert across a 4x sweep (half/double identical to 4
decimals), MM-vs-linear hepatic kinetics barely move the result, and
biliary secretion is negligible -- only kp_scalar produced a meaningful
swing, and removing it entirely made things WORSE, not better. Together
these results rule out the elimination side as the dominant bottleneck:
something is preventing elimination-rate changes from mattering at all,
which is the signature of an absorption or distribution-volume ceiling
upstream of clearance -- if too little drug ever reaches/stays in the
central compartment, no downstream clearance parameter can raise Cmax/AUC.

reference_pk.py's own v5.0 comment for Warfarin states this directly:
"Without the TMDD depot, fup=0.007 produces near-zero free-drug driving
force and an effective Vd of ~7 L vs the observed ~70 L -- 10x AUC
underprediction." This script tests that hypothesis directly, plus the
absorption-side parameters (F, ka) as a second candidate ceiling.

Variants (tmdd_params removed in every variant, consistent with isolation
test #2's finding that TMDD is a net negative for this drug's parameters,
so each row isolates ONLY the named change against that same reference
point):

  1. NO_TMDD_BASELINE     -- tmdd removed, everything else stock. (Same
                              reference point as isolation test #2, repeated
                              here for side-by-side comparison.)
  2. NO_TMDD_F1            -- + F (bioavailability) forced to 1.0. Tests
                              whether F=0.93 (already near-complete) is
                              meaningfully limiting, or whether absorption
                              fraction was never the bottleneck.
  3. NO_TMDD_KA_HALF       -- + ka halved. Tests sensitivity of the result
                              to absorption RATE (as opposed to fraction).
  4. NO_TMDD_KA_DOUBLE     -- + ka doubled. Same sensitivity check, other
                              direction.
  5. NO_TMDD_FUP_X10       -- + fup multiplied by 10 (0.007 -> 0.07), with
                              kp_scalar left at its current value. Directly
                              tests reference_pk.py's own stated hypothesis:
                              does a larger fup (closing some of the gap to
                              the literature ~0.01-0.03 range reported for
                              warfarin in some sources) restore a larger
                              free-drug driving force and push Cmax/AUC up,
                              consistent with the "near-zero free-drug
                              driving force" comment?
  6. NO_TMDD_KP_SCALAR_X5  -- + kp_scalar multiplied by 5x relative to
                              its current stock value (NOT forced to a new
                              absolute number -- isolation test #2 already
                              showed kp_scalar=1.0 makes things worse, so
                              this tests the OTHER direction: does more
                              aggressive Vd inflation, independent of
                              fup, move toward target?)

Run from the repository root:
    python engine/validation/isolation_test_warfarin_absorption.py

Does not modify reference_pk.py or any engine file on disk -- every variant
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
    print(" v5.3 ISOLATION TEST #3: Warfarin absorption / Vd-collapse levers")
    print(f"{'='*78}\n")

    if "Warfarin" not in REFERENCE_PK:
        print("[!] 'Warfarin' not found in REFERENCE_PK — cannot run test.")
        return

    base = REFERENCE_PK["Warfarin"]
    target_cmax = base["cmax"]
    target_auc = base["auc"]
    base_ka = base.get("ka")
    base_F = base.get("F")
    base_fup = base.get("fup")
    base_kp_scalar = base.get("kp_scalar", 1.0)

    sim = Simulator(verbose=False)

    variants = {
        "NO_TMDD_BASELINE (tmdd removed, else stock)":
            _make_variant(base, tmdd_params=None),

        "NO_TMDD_F1 (+ F forced to 1.0)":
            _make_variant(base, tmdd_params=None, F=1.0),

        "NO_TMDD_KA_HALF (+ ka halved)":
            _make_variant(base, tmdd_params=None,
                           ka=(base_ka / 2.0) if base_ka else None),

        "NO_TMDD_KA_DOUBLE (+ ka doubled)":
            _make_variant(base, tmdd_params=None,
                           ka=(base_ka * 2.0) if base_ka else None),

        "NO_TMDD_FUP_X10 (+ fup x10, kp_scalar unchanged)":
            _make_variant(base, tmdd_params=None,
                           fup=(base_fup * 10.0) if base_fup else None),

        "NO_TMDD_KP_SCALAR_X5 (+ kp_scalar x5 vs. stock)":
            _make_variant(base, tmdd_params=None,
                           kp_scalar=base_kp_scalar * 5.0),
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
    print(f" Baseline ka={base_ka}, F={base_F}, fup={base_fup}, kp_scalar={base_kp_scalar}")
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
    print(" - Compare every row against NO_TMDD_BASELINE (same reference")
    print("   point as isolation test #2's CLint sweep — CLint half/double")
    print("   there were IDENTICAL to this baseline to 4 decimals).")
    print(" - NO_TMDD_F1 close to baseline -> bioavailability fraction was")
    print("   never the bottleneck (F=0.93 is already near-complete, so")
    print("   this is mostly a sanity check, not expected to move much).")
    print(" - NO_TMDD_KA_HALF / NO_TMDD_KA_DOUBLE far apart from each other")
    print("   -> absorption RATE matters for this drug's Cmax in particular")
    print("   (rate changes reshape the curve even if total absorbed mass")
    print("   is similar); if both close to baseline, ka is not the lever.")
    print(" - NO_TMDD_FUP_X10 is the key test of reference_pk.py's own")
    print("   stated hypothesis ('near-zero free-drug driving force from")
    print("   fup=0.007'). If this variant moves SUBSTANTIALLY toward")
    print("   target (much more than kp_scalar alone did in isolation test")
    print("   #2), the dominant defect is that fup=0.007 itself collapses")
    print("   the free-drug driving force everywhere in the model")
    print("   (distribution, passive diffusion, AND clearance simultaneously)")
    print("   in a way no single downstream parameter (CLint, kp_scalar,")
    print("   TMDD) can fully compensate for in isolation.")
    print(" - NO_TMDD_KP_SCALAR_X5 vs. NO_TMDD_BASELINE: isolation test #2")
    print("   showed kp_scalar=1.0 (removed) made things worse than stock.")
    print("   If x5 the stock value moves further TOWARD target, kp_scalar")
    print("   is monotonically helpful in this direction and may simply be")
    print("   under-tuned (a real Vd lever); if x5 overshoots PAST target")
    print("   or makes Cmax worse while AUC improves (or vice versa), the")
    print("   single scalar cannot independently fit both metrics --")
    print("   evidence of a genuine structural defect, not a magnitude")
    print("   tuning problem.")
    print()


if __name__ == "__main__":
    run_isolation_test()