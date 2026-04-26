"""
admet.py — Drug property estimation for the BodySim PBPK engine.

Sources:
  Rodgers & Rowland, J Pharm Sci 2006  — Kp estimation
  Houston JB, Biochem Pharmacol 1994   — hepatocyte scaling
  Lobell & Sivarajah, J Med Chem 2003  — fup model
"""

import numpy as np
from .physiology import TISSUE_COMPOSITION, lung_kp

PLASMA_COMPOSITION = {"water": 0.93, "neutral_lipid": 0.0023, "phospholipid": 0.0023}


def estimate_kp_values(logp, fup, pka=None, drug_type="neutral",
                       cyp3a4_activity=1.0):
    """
    Estimate tissue-to-plasma Kp for all organ compartments.
    Lung Kp is now drug-specific (fixes caffeine false-positive).
    """
    logp_c = np.clip(logp, -3.0, 6.0)
    Kn     = 10 ** (0.7 * logp_c)

    if   drug_type == "basic":   Kph = 10 ** (0.4 * logp_c + 0.5)
    elif drug_type == "acidic":  Kph = 10 ** (0.2 * logp_c - 0.3)
    else:                        Kph = 10 ** (0.3 * logp_c)

    kp = {}
    for organ, (fw, fn, fp) in TISSUE_COMPOSITION.items():
        if organ == "lung":
            continue   # handled separately below
        kp_raw = (fw / PLASMA_COMPOSITION["water"]
                  + fn * Kn  / PLASMA_COMPOSITION["neutral_lipid"]
                  + fp * Kph / PLASMA_COMPOSITION["phospholipid"])
        kp[organ] = max(0.05, kp_raw * fup)

    # Liver — OCT/OATP transporter correction
    kp["liver"] = max(kp.get("liver", 1.0),
                      1.5) * cyp3a4_activity * 0.6 + 0.4

    # Kidney — active transporters for hydrophilic drugs
    if logp < 0:
        kp["kidney"] = kp.get("kidney", 1.0) * (1.0 + 3.0 * abs(logp))
    else:
        kp["kidney"] = max(kp.get("kidney", 1.0), 0.8)

    # Fat — hydrophilic drugs barely enter adipose
    if logp < 1.0:
        fw_fat = TISSUE_COMPOSITION["fat"][0]
        kp["fat"] = max(0.05,
                        (fw_fat / PLASMA_COMPOSITION["water"]) * fup * 0.5)

    # Brain — BBB permeability
    bbb_factor = _bbb_permeability(logp, pka, drug_type)
    fw_brain   = TISSUE_COMPOSITION["brain"][0]
    kp["brain"] = max(0.01,
                      (fw_brain / PLASMA_COMPOSITION["water"]) * fup * bbb_factor)

    # Gut
    kp["gut"] = max(kp.get("gut", 1.0), 0.8)

    # ── LUNG — drug-specific (the key fix) ────────────────────────────────
    kp["lung"] = lung_kp(logp, pka, drug_type)

    # Fill any gaps
    for organ in ["liver","kidney","brain","heart","muscle","fat",
                  "gut","skin","bone","lung","rest"]:
        if organ not in kp:
            kp[organ] = 1.0

    return kp


def _bbb_permeability(logp, pka, drug_type):
    if   logp > 3:  base = 0.8
    elif logp > 1:  base = 0.5
    elif logp > -1: base = 0.2
    else:           base = 0.05
    if drug_type == "basic"  and pka and pka > 7.4: base *= 0.6
    if drug_type == "acidic" and pka and pka < 4:   base *= 0.4
    return base


def estimate_absorption_params(logp, mw, pka=None, drug_type="neutral",
                                formulation="immediate_release"):
    logp_c = np.clip(logp, -3.0, 6.0)
    if   logp_c < -1: fa, ka = 0.80, 0.6
    elif logp_c <  1: fa, ka = 0.75, 0.8
    elif logp_c <  3: fa, ka = 0.90, 1.2
    elif logp_c <  5: fa, ka = 0.70, 0.5
    else:             fa, ka = 0.40, 0.3
    if mw > 500: fa *= 0.7; ka *= 0.7
    if mw > 700: fa *= 0.4

    if   logp_c > 4: eh_base = 0.50
    elif logp_c > 2: eh_base = 0.25
    elif logp_c > 0: eh_base = 0.15
    else:            eh_base = 0.05
    if drug_type == "acidic":               eh_base *= 0.15
    elif drug_type == "neutral" and logp_c > 3: eh_base *= 0.4

    F    = np.clip(fa * (1.0 - eh_base), 0.01, 1.0)
    tlag = 0.25 if formulation == "immediate_release" else 0.5
    return {"ka": ka, "F": F, "tlag": tlag, "fa": fa, "eh": eh_base}


def estimate_clearance(logp, fup, mw, drug_type="neutral",
                       cyp3a4_activity=1.0, egfr_ml_min=100.0):
    if   logp > 3:  cl_base = 80.0
    elif logp > 1:  cl_base = 30.0
    elif logp > -1: cl_base = 10.0
    else:           cl_base = 3.0
    cl_int = cl_base * fup * cyp3a4_activity
    if mw > 600: cl_int *= 0.5

    gfr_lh  = egfr_ml_min * 60.0 / 1000.0
    clr_gfr = gfr_lh * fup
    tubular  = 3.0 if (logp < 0 and drug_type in ("basic","acidic")) else (
                1.5 if logp < 0 else 1.0)
    cl_renal = clr_gfr * tubular

    return {"CLint": cl_int, "CLrenal": cl_renal, "CLr_gfr": clr_gfr}


def blood_plasma_ratio(logp, drug_type="neutral"):
    if drug_type == "basic"  and logp > 1: return 1.5
    if drug_type == "acidic":              return 0.85
    return 1.0


def build_drug_profile(name, logp, fup, mw, pka=None,
                       drug_type="neutral",
                       clint_override=None, clrenal_override=None,
                       ka_override=None,    F_override=None,
                       kp_overrides=None,
                       egfr_ml_min=100.0,  cyp3a4_activity=1.0):
    kp   = estimate_kp_values(logp, fup, pka, drug_type, cyp3a4_activity)
    abs_ = estimate_absorption_params(logp, mw, pka, drug_type)
    cl_  = estimate_clearance(logp, fup, mw, drug_type,
                               cyp3a4_activity, egfr_ml_min)
    Rb   = blood_plasma_ratio(logp, drug_type)

    profile = {
        "name": name, "logp": logp, "fup": fup, "mw": mw,
        "pka": pka, "drug_type": drug_type, "kp": kp,
        "ka":    ka_override    if ka_override    is not None else abs_["ka"],
        "F":     F_override     if F_override     is not None else abs_["F"],
        "tlag":  abs_["tlag"],
        "CLint":   clint_override  if clint_override  is not None else cl_["CLint"],
        "CLrenal": clrenal_override if clrenal_override is not None else cl_["CLrenal"],
        "Rb": Rb,
    }
    if kp_overrides:
        for organ, val in kp_overrides.items():
            profile["kp"][organ] = val
    return profile


REFERENCE_DRUGS = {
    "metformin": build_drug_profile(
        name="Metformin", logp=-1.43, fup=0.97, mw=129.16,
        pka=11.5, drug_type="basic",
        clint_override=20.0, clrenal_override=30.6,
        ka_override=0.5, F_override=0.75,
        kp_overrides={"liver":4.0,"kidney":4.5,"muscle":3.5,"fat":0.3,
                      "heart":3.0,"gut":5.0,"brain":0.05,"skin":2.0,
                      "bone":2.5,"rest":2.5,
                      "lung": lung_kp(-1.43, 11.5, "basic")},
    ),
    "caffeine": build_drug_profile(
        name="Caffeine", logp=-0.07, fup=0.64, mw=194.19,
        pka=0.52, drug_type="neutral",
        clint_override=12.0, clrenal_override=0.3,
        ka_override=1.8, F_override=0.99,
        kp_overrides={"liver":0.8,"kidney":0.6,"muscle":0.35,"fat":0.4,
                      "gut":0.7,"skin":0.6,"bone":0.3,"rest":0.5,
                      "heart":0.7,
                      "lung": lung_kp(-0.07, 0.52, "neutral"),   # ~0.6 not 1.0
                      "brain":0.7},
    ),
    "ibuprofen": build_drug_profile(
        name="Ibuprofen", logp=3.97, fup=0.01, mw=206.29,
        pka=4.91, drug_type="acidic",
        clint_override=180.0, clrenal_override=0.1,
        ka_override=1.5, F_override=0.87,
    ),
    "warfarin": build_drug_profile(
        name="Warfarin", logp=2.70, fup=0.007, mw=308.33,
        pka=5.08, drug_type="acidic",
        clint_override=4.5, clrenal_override=0.0,
        ka_override=0.8, F_override=0.93,
    ),
}
