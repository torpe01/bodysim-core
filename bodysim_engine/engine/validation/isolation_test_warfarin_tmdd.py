"""
isolation_test_warfarin_tmdd.py — v5.3 diagnostic isolation test.

Purpose
-------
An external review correctly pointed out that the validation table alone
cannot distinguish "the TMDD parallel-binding architecture fix is what
improved Warfarin" from "something else moved Warfarin" (e.g. an unrelated
elimination-pathway mismatch, or the mere presence of any TMDD term —
correct or not — changing the result). This script isolates the variable
by running Warfarin through the REAL engine three times, changing only
whether/how tmdd_params is applied, with every other reference_pk.py value
(kp_scalar, CLint, dose, route, etc.) held identical:

  1. BASELINE   — Warfarin exactly as defined in reference_pk.py (current
                  v5.3 parallel-binding quadratic active, tmdd_params present).
  2. NO_TMDD    — tmdd_params removed entirely (Bmax=0 path). Isolates how
                  much of the Cmax/AUC result depends on TMDD existing at all.
  3. OLD_SEQUENTIAL — the prior v5.2 Step 3 sequential-filter algebra,
                  reimplemented standalone here for comparison only (NOT
                  patched into hepatic_module.py — this is a side-by-side
                  reference calculation, run by temporarily monkey-patching
                  the module for this script's duration only).

Run from the repository root:
    python engine/validation/isolation_test_warfarin_tmdd.py

Does not modify reference_pk.py, hepatic_module.py, or any other engine
file on disk — all variants are constructed as in-memory copies of the
Warfarin dict, and the OLD_SEQUENTIAL comparison uses a monkey-patch that
is restored before the script exits.
"""

import copy
import sys
import os
import types

sys.path.append(os.getcwd())

import numpy as np

from engine.simulator import Simulator
from engine.admet import build_drug_profile
from engine.validation.reference_pk import REFERENCE_PK
import engine.hepatic_module as hepatic_module_mod


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


def _old_sequential_calculate_liver_flux(self, y, drug, liver_volume, C_art,
                                          fup, flow_rate, tp, params, extra):
    """
    Reimplementation of the PRE-v5.3 (v5.2 Step 3) sequential f_tmdd_free
    algebra, for side-by-side comparison only. This treats C_tissue_free
    (already Kp-attenuated) as the naive-free input to a Langmuir isotherm,
    applied AFTER the passive Kp division — the architecture the v5.3
    blueprint replaced. Everything else (uptake, passive diffusion, CYP,
    Phase II, biliary, TMDD-free ODE assembly) is copied verbatim from the
    current module so only the binding architecture differs.
    """
    Rb = drug.get("Rb", 1.0)
    kp = drug["kp"]

    v_liv_vasc = liver_volume * 0.15
    v_liv_tiss = liver_volume * 0.85

    C_liv_vasc = extra["C_liv_vasc"]
    C_liv_tiss = extra["C_liv_tiss"]

    C_vascular_free = fup * C_liv_vasc
    C_tissue_free = fup * C_liv_tiss / kp["liver"]  # passive-only, pre-TMDD

    C_art_blood = extra["C_art_blood"]
    C_gut_blood_out_blood = extra["C_gut_blood_out_blood"]
    C_liv_vasc_blood = extra["C_liv_vasc_blood"]

    Q_ha = flow_rate["liver_hepatic"]
    Q_pv = flow_rate["liver_portal"]
    Q_liv = Q_ha + Q_pv

    threshold = drug.get("albumin_facilitation_threshold", 0.05)
    eff_floor = drug.get("albumin_facilitation_eff", 0.15)
    is_uptake_substrate = drug.get("is_uptake_substrate", False)
    if fup < threshold and is_uptake_substrate:
        _fu_eff = max(fup, eff_floor)
        C_enhanced = C_liv_vasc * _fu_eff
    else:
        C_enhanced = C_vascular_free

    active_uptake_liv = 0.0
    for _name, trans in tp["hepatic_uptake"].items():
        km = trans["Km_mgl"]
        cl = trans["cl_linear"]
        if km > 0 and cl > 0:
            sat = km / (km + C_enhanced) if (km + C_enhanced) > 0 else 1.0
            active_uptake_liv += float(cl * sat)

    vmax_up = drug.get("vmax_uptake", 0.0)
    km_up = drug.get("km_uptake", 1.0)
    j_uptake = 0.0
    if is_uptake_substrate and vmax_up > 0.0:
        j_uptake = (vmax_up * C_enhanced) / (km_up + C_enhanced)

    CL_pd = params.get("liver_CL_pd", 10.0)
    j_passive = CL_pd * (C_vascular_free - C_tissue_free)

    # ── OLD sequential TMDD: Langmuir partition applied to the ALREADY
    # Kp-attenuated C_tissue_free (this is the architecture being compared
    # against, not the current module's parallel quadratic) ──────────────
    _tmdd = drug.get("tmdd_params", None)
    f_tmdd_free = 1.0
    if _tmdd is not None:
        _Bmax = float(_tmdd.get("Bmax_mg_L", 0.0))
        _Kd = float(_tmdd.get("Kd_mg_L", 1.0))
        if _Bmax > 0.0 and _Kd > 1e-12:
            _C_naive = max(C_tissue_free, 0.0)
            _a = 1.0
            _b = (_Kd + _Bmax - _C_naive)
            _c = -_Kd * _C_naive
            _disc = max(_b * _b - 4.0 * _a * _c, 0.0)
            _C_free_corrected = max((-_b + np.sqrt(_disc)) / (2.0 * _a), 0.0)
            f_tmdd_free = float(np.clip(
                _C_free_corrected / _C_naive, 0.0, 1.0
            )) if _C_naive > 1e-15 else 1.0
    C_tissue_free = C_tissue_free * f_tmdd_free

    CLh = drug["CLint"]
    _vmax_raw = drug.get("Vmax_hepatic")
    _km_raw = drug.get("Km_hepatic")
    if (_vmax_raw is not None and float(_vmax_raw) > 0.0
            and _km_raw is not None and float(_km_raw) > 1e-9):
        vmax_hep = float(_vmax_raw)
        km_hep = float(_km_raw)
        use_mm_hepatic = True
    else:
        vmax_hep = 0.0
        km_hep = 1.0
        use_mm_hepatic = False

    if use_mm_hepatic:
        J_cyp = (vmax_hep * C_tissue_free) / (km_hep + C_tissue_free)
    else:
        J_cyp = CLh * C_tissue_free

    J_phaseII = 0.0
    sult_p = tp.get("phaseII_sult")
    if sult_p is not None:
        _km_s = sult_p["Km_mg_L"]
        _denom_s = _km_s + C_tissue_free
        J_phaseII += (sult_p["Vmax_mg_h"] * C_tissue_free) / _denom_s \
            if _denom_s > 1e-15 else 0.0
    ugt_p = tp.get("phaseII_ugt")
    if ugt_p is not None:
        _km_u = ugt_p["Km_mg_L"]
        _denom_u = _km_u + C_tissue_free
        J_phaseII += (ugt_p["Vmax_mg_h"] * C_tissue_free) / _denom_u \
            if _denom_u > 1e-15 else 0.0

    J_bile_secretion = extra.get("J_bile_secretion", 0.0)
    metabolic_rate = J_cyp + J_phaseII + J_bile_secretion

    dydt_liv_vasc = (
        Q_ha * C_art_blood
        + Q_pv * C_gut_blood_out_blood
        - Q_liv * C_liv_vasc_blood
        - active_uptake_liv * C_vascular_free
        - j_uptake
        - j_passive
    ) / v_liv_vasc

    dydt_liv_tiss = (
        active_uptake_liv * C_enhanced
        + j_uptake
        + j_passive
        - metabolic_rate
    ) / v_liv_tiss

    return {
        "dydt_liv_vasc": dydt_liv_vasc,
        "dydt_liv_tiss": dydt_liv_tiss,
        "metabolic_rate": metabolic_rate,
        "C_tissue_free": C_tissue_free,
    }


def run_isolation_test():
    print(f"\n{'='*70}")
    print(" v5.3 ISOLATION TEST: Warfarin TMDD architecture")
    print(f"{'='*70}\n")

    if "Warfarin" not in REFERENCE_PK:
        print("[!] 'Warfarin' not found in REFERENCE_PK — cannot run test.")
        return

    warfarin_data = REFERENCE_PK["Warfarin"]
    target_cmax = warfarin_data["cmax"]
    target_auc = warfarin_data["auc"]

    sim = Simulator(verbose=False)
    results = {}

    # ── Variant 1: BASELINE (current v5.3 parallel-binding quadratic, as-is) ──
    print(" Running BASELINE (v5.3 parallel-binding quadratic, tmdd_params present)...")
    try:
        cmax, auc = _build_profile_and_run("Warfarin", warfarin_data, sim)
        results["BASELINE (v5.3 parallel quadratic)"] = (cmax, auc)
    except Exception as e:
        print(f"   ERROR: {e}")
        results["BASELINE (v5.3 parallel quadratic)"] = (None, None)

    # ── Variant 2: NO_TMDD (tmdd_params removed entirely) ──────────────────
    print(" Running NO_TMDD (tmdd_params removed)...")
    data_no_tmdd = copy.deepcopy(warfarin_data)
    data_no_tmdd.pop("tmdd_params", None)
    try:
        cmax, auc = _build_profile_and_run("Warfarin", data_no_tmdd, sim)
        results["NO_TMDD (tmdd_params absent)"] = (cmax, auc)
    except Exception as e:
        print(f"   ERROR: {e}")
        results["NO_TMDD (tmdd_params absent)"] = (None, None)

    # ── Variant 3: OLD_SEQUENTIAL (pre-v5.3 sequential filter, monkey-patched) ──
    print(" Running OLD_SEQUENTIAL (v5.2 Step 3 sequential filter, for comparison)...")
    _original_method = hepatic_module_mod.HepaticClearanceModule.calculate_liver_flux
    try:
        hepatic_module_mod.HepaticClearanceModule.calculate_liver_flux = \
            _old_sequential_calculate_liver_flux
        cmax, auc = _build_profile_and_run("Warfarin", warfarin_data, sim)
        results["OLD_SEQUENTIAL (v5.2 Step 3, tmdd_params present)"] = (cmax, auc)
    except Exception as e:
        print(f"   ERROR: {e}")
        results["OLD_SEQUENTIAL (v5.2 Step 3, tmdd_params present)"] = (None, None)
    finally:
        # Always restore the real implementation, even if the run above failed.
        hepatic_module_mod.HepaticClearanceModule.calculate_liver_flux = _original_method

    # ── Report ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" Target (clinical):  Cmax = {target_cmax:.4f}   AUC = {target_auc:.4f}")
    print(f"{'='*70}")
    header = f" {'Variant':<48} {'Cmax':>10} {'Cmax_Fold':>10} {'AUC':>10} {'AUC_Fold':>10}"
    print(header)
    print("-" * len(header))
    for variant, (cmax, auc) in results.items():
        if cmax is None:
            print(f" {variant:<48} {'FAILED':>10}")
            continue
        cmax_fold = cmax / target_cmax if target_cmax else float("nan")
        auc_fold = auc / target_auc if target_auc else float("nan")
        print(f" {variant:<48} {cmax:>10.4f} {cmax_fold:>10.4f} {auc:>10.4f} {auc_fold:>10.4f}")
    print(f"{'='*70}\n")

    print(" Interpretation guide:")
    print(" - BASELINE vs NO_TMDD: the gap shows how much of Warfarin's result")
    print("   depends on tmdd_params existing at all (correctly implemented or not).")
    print(" - BASELINE vs OLD_SEQUENTIAL: the gap shows how much of the v5.3")
    print("   Target 2 architecture change (parallel vs sequential binding)")
    print("   alone explains the improvement reported in validate_drugs.py.")
    print(" - If NO_TMDD lands closer to target than BASELINE, the residual")
    print("   Warfarin error is likely NOT primarily a binding-architecture")
    print("   issue (e.g. could be CLint/elimination-pathway misclassification")
    print("   instead) — investigate elsewhere before further TMDD tuning.")
    print()


if __name__ == "__main__":
    run_isolation_test()