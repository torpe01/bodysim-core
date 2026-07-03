"""
diagnostic_vd_prediction_accuracy.py — Script 9 (v5.3 investigation log)

PURPOSE
-------
Test the untested axis: does the engine's Rodgers-Rowland + lysosomal-trapping
Kp model correctly predict volume of distribution (Vd) for the 6+2 unexplained
failing drugs?

MOTIVATION
----------
Scripts 1-8 collectively established that:
  - Absorption permeability (p_eff) is well-sourced for all 7 target drugs
    (all have explicit literature values, not QSPR fallback)
  - CLint is confirmed correctly consumed by the ODE for all non-MM drugs
  - CLrenal is confirmed correctly wired by magnitude for all 7 drugs
  - F/ka are universally dead (ACAT migration artefact, not drug-specific)

The ONE axis that has never been measured for these drugs is tissue distribution
(Vd / Kp prediction). Metoprolol's known failure direction — Cmax 3.02× OVER
despite good absorption data and working elimination — is the textbook signature
of Vd under-prediction: drug pools in plasma because the model doesn't distribute
it into tissues. Propranolol and Metoprolol both have large clinical Vd (3.2–3.9
L/kg) that should be driven by lysosomal ion-trapping (Gap 3 in admet.py);
that correction has never been empirically validated. Caffeine and Aciclovir are
neutral (no lysosomal trapping), so if they also show large Vd fold-errors, the
R&R baseline model itself is at fault, not just the trapping correction.

APPROACH
--------
STATIC — no ODE integration. For each drug:
  1. Build the full engine profile via build_drug_profile() with the identical
     parameter forwarding used by validate_drugs.py, so the profile is exactly
     what the live simulation sees.
  2. Extract drug["kp"] (tissue:plasma partition coefficient dict) from the
     returned profile. These values already incorporate kp_scalar if present,
     and the lysosomal trapping amplification for eligible basic drugs.
  3. Compute Vd_predicted analytically using the standard PBPK Vdss formula:

       Vd_predicted [L] = V_plasma
                        + V_rbc × Rb
                        + Σ_tissues(V_tissue[t] × Kp[t])

     where Kp[t] = tissue:plasma concentration ratio at equilibrium, V_tissue
     from human physiology (Rodgers & Rowland 2006; Davies & Morris 1993), and
     Rb = blood:plasma ratio from the drug profile (default 1.0).

  4. Compare to well-established literature Vd values. All 7 drugs have
     consensus Vd references; no specialist sourcing is required.

INTERPRETATION GUIDE
--------------------
  Lysosomal-trapping-eligible drugs (Metoprolol, Propranolol):
    Vd_predicted << Vd_lit  →  trapping correction is too weak / absent
    Vd_predicted >> Vd_lit  →  trapping correction is over-applied
    Vd_predicted ≈ Vd_lit   →  Vd is not the lever; pivot to k_abs investigation

  Non-trapping drugs (Aciclovir, Caffeine, Ciprofloxacin, Furosemide, Omeprazole):
    Vd_predicted << Vd_lit  →  R&R baseline model under-distributes these drugs
    Both groups failing in same direction  →  systemic R&R issue, not trapping
    Pattern splits by drug class  →  class-specific model component is at fault

  For Furosemide (kp_scalar=0.20 already present):
    If fold-error is still large, the scalar value itself needs recalibration.
    If fold-error is small, the kp_scalar for Furosemide is working correctly.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from engine.admet import build_drug_profile
from engine.validation.reference_pk import REFERENCE_PK


# ─────────────────────────────────────────────────────────────────────────────
# INVESTIGATION TARGETS
# 7 unique drugs from the "unexplained failing" thread opened at Script 9.
# Metoprolol appears in both the "6 untouched" and the "good p_eff, still
# fails" sub-groups; it is listed once here.
# ─────────────────────────────────────────────────────────────────────────────

TARGET_DRUGS = [
    "Aciclovir",
    "Caffeine",
    "Ciprofloxacin",
    "Furosemide",
    "Metoprolol",
    "Omeprazole",
    "Propranolol",
]


# ─────────────────────────────────────────────────────────────────────────────
# LITERATURE Vd VALUES — 70 kg reference adult, all in litres.
# Sources selected to match the standard used in the engine's reference_pk.py:
# primary PK literature (direct measurement), FDA label, or DrugBank consensus.
# All values are well-validated; the PBPK community treats these as ground truth.
# ─────────────────────────────────────────────────────────────────────────────

LITERATURE_VD = {
    "Aciclovir":     {
        "vd_L":   49.0,
        "vd_Lkg": 0.70,
        "source": "Laskin et al., Antimicrob Agents Chemother 1982;21:393",
    },
    "Caffeine":      {
        "vd_L":   42.0,
        "vd_Lkg": 0.60,
        "source": "Callahan et al., NEJM 1982;307:397; DrugBank DB00201",
    },
    "Ciprofloxacin": {
        "vd_L":   175.0,
        "vd_Lkg": 2.50,
        "source": "Bergan, Rev Infect Dis 1988;10(Suppl 1):S45; FDA label",
    },
    "Furosemide":    {
        "vd_L":   7.7,
        "vd_Lkg": 0.11,
        "source": "Tilstone & Fine, Clin Pharmacol Ther 1978;23:644",
    },
    "Metoprolol":    {
        "vd_L":   224.0,
        "vd_Lkg": 3.20,
        "source": "Regardh & Johnsson, Clin Pharmacokinet 1980;5:557",
    },
    "Omeprazole":    {
        "vd_L":   25.0,
        "vd_Lkg": 0.35,
        "source": "Regardh et al., Clin Pharmacokinet 1992;23:453",
    },
    "Propranolol":   {
        "vd_L":   273.0,
        "vd_Lkg": 3.90,
        "source": "Nies & Shand, Drugs 1975;10:59; Evans et al. 1973",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# ORGAN VOLUMES [L] — 70 kg reference individual
#
# Primary source: Rodgers & Rowland, J Pharm Sci 2006;95(6):1238, Table 1
# Supplementary:  Davies & Morris, Pharm Res 1993;10(7):1093
#
# Attempt to import from the engine's physiology module (which is what
# build_drug_profile uses internally for the R&R Kp calculation), so the
# Vd formula uses the same volumes as the engine itself. Fall back to the
# R&R 2006 reference values if the module import fails — these differ by at
# most ~10% from any standard physiology table and will not change the
# diagnostic conclusion.
# ─────────────────────────────────────────────────────────────────────────────

# Rodgers & Rowland 2006 reference organ volumes [L] — fallback values,
# only used if the import below fails for some other reason.
_ORGAN_VOL_FALLBACK = {
    "fat":     18.2, "bone":    10.5, "brain":    1.45, "gut":      1.65,
    "heart":   0.33, "kidney":  0.28, "liver":    1.69, "lung":     0.50,
    "muscle": 29.0,  "skin":    7.8,  "rest":     4.0,
}
_V_PLASMA_FALLBACK = 3.0
_V_RBC_FALLBACK    = 2.6

ORGAN_VOL = _ORGAN_VOL_FALLBACK.copy()
V_PLASMA  = _V_PLASMA_FALLBACK
V_RBC     = _V_RBC_FALLBACK
_VOL_SOURCE = "Rodgers & Rowland 2006 (fallback — engine physiology.py unavailable)"

try:
    # v5.4 fix: the real export is ORGAN_VOLUMES (no _L suffix) — the
    # original ORGAN_VOLUMES_L import silently failed every run, so this
    # diagnostic was ALWAYS using the fallback table, never the engine's
    # actual organ volumes. Confirmed by direct inspection of physiology.py.
    from engine.physiology import ORGAN_VOLUMES as _phys_vols
    for _k, _v in _phys_vols.items():
        if _k in ORGAN_VOL:
            ORGAN_VOL[_k] = float(_v)
    # physiology.py has no separate plasma/RBC split — it has
    # arterial_blood + venous_blood as total blood volume. Decompose via
    # hematocrit (0.45, matching the engine's own HCT elsewhere) into
    # plasma and RBC volumes for the Vd formula.
    _v_blood_total = float(_phys_vols.get("arterial_blood", 1.68)) + float(_phys_vols.get("venous_blood", 3.92))
    _HCT = 0.45
    V_PLASMA = _v_blood_total * (1.0 - _HCT)
    V_RBC    = _v_blood_total * _HCT
    _VOL_SOURCE = "engine/physiology.py (ORGAN_VOLUMES, real values)"
except Exception as _e:
    _VOL_SOURCE = f"Rodgers & Rowland 2006 (fallback — import failed: {_e})"


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED KEYS — identical to validate_drugs.py so profiles match exactly.
# ─────────────────────────────────────────────────────────────────────────────

_ADVANCED_KEYS = [
    "gut_transporter",
    "phaseII_kinetics",
    "fu_gut",
    "CLint_gut_cyp3a4",
    "tmdd_params",
    "kp_scalar",
    "cl_bile_lh",
    "f_reabs_bile",
    "p_eff",
    "is_uptake_substrate",
    "vmax_uptake",
    "km_uptake",
    "Vmax_hepatic",
    "Km_hepatic",
    "absorption_segments",
    "enteric_coated",
    "peff_is_measured_net",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _build_profile(name: str, data: dict) -> dict:
    """
    Build a drug profile via build_drug_profile() using the same parameter
    forwarding as validate_drugs.py.  The returned profile is identical to
    what the live simulation receives — kp_scalar, lysosomal trapping, and
    all active module parameters are already baked in.
    """
    advanced_kwargs = {k: data[k] for k in _ADVANCED_KEYS if k in data}
    return build_drug_profile(
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


def _compute_vd(profile: dict) -> dict:
    """
    Compute PBPK apparent Vdss from the Kp tissue partition coefficient dict.

    Formula (plasma concentration as reference):
        Vd = V_plasma + V_rbc × Rb + Σ_tissues(V_tissue[t] × Kp[t])

    Notes
    -----
    - Kp[t] values from the profile already include kp_scalar and lysosomal
      trapping amplification, so this is the Vd the engine actually operates at.
    - Liver: the engine uses a two-sub-compartment split (15% vascular,
      85% tissue within liver_volume), but for Vd estimation purposes the
      full liver volume × Kp["liver"] is used. The resulting error (~0.25 L
      for Kp["liver"] near 1) is negligible relative to the fold-errors
      this script is designed to detect.
    - Rb (blood:plasma ratio) defaults to 1.0 if not set in the profile.

    Returns dict with vd, per-tissue breakdown, and any missing kp keys.
    """
    kp  = profile.get("kp", {})
    Rb  = float(profile.get("Rb", 1.0))
    fup = float(profile.get("fup", 1.0))

    blood_contrib = V_PLASMA + V_RBC * Rb

    tissue_contribs = {}
    kp_missing      = []
    kp_defaults_used = {}

    for tissue, vol in ORGAN_VOL.items():
        if tissue in kp:
            tissue_contribs[tissue] = vol * float(kp[tissue])
        else:
            # Flag missing key; use Kp=1.0 (neutral partition) as a
            # conservative placeholder so the Vd estimate doesn't collapse.
            kp_missing.append(tissue)
            kp_defaults_used[tissue] = 1.0
            tissue_contribs[tissue] = vol * 1.0

    vd_predicted = blood_contrib + sum(tissue_contribs.values())

    # Top tissue contributors (sorted descending by absolute volume-Kp product)
    top_tissues = sorted(tissue_contribs.items(), key=lambda x: -x[1])

    return {
        "vd_predicted":    vd_predicted,
        "blood_contrib":   blood_contrib,
        "tissue_contribs": tissue_contribs,
        "top_tissues":     top_tissues,
        "Rb":              Rb,
        "fup":             fup,
        "kp_dict":         kp,
        "kp_missing":      kp_missing,
        "kp_defaults_used":kp_defaults_used,
    }


def _lysosomal_eligible(data: dict) -> bool:
    """
    Gap 3 (admet.py) eligibility: lipophilic basic amine with logP > 1.5 and
    pKa_base > 7.0.  Replicates the guard condition documented in admet.py.
    For zwitterions with a pKa dict, uses the base pKa component.
    """
    if data.get("drug_type") != "basic":
        return False
    if float(data.get("logp", 0.0)) <= 1.5:
        return False
    pka = data.get("pka")
    if isinstance(pka, dict):
        pka_val = float(pka.get("base", pka.get("acid", 0.0)))
    elif pka is not None:
        pka_val = float(pka)
    else:
        return False
    return pka_val > 7.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic():
    print(f"\n{'='*72}")
    print(f"  SCRIPT 9 — Vd PREDICTION ACCURACY (Static Kp Diagnostic)")
    print(f"  Axis under test: tissue distribution (R&R Kp + lysosomal trapping)")
    print(f"  Target set: 6+2 unexplained failing drugs (7 unique drugs)")
    print(f"  Organ volumes: {_VOL_SOURCE}")
    print(f"  V_plasma = {V_PLASMA:.1f} L   V_rbc = {V_RBC:.1f} L")
    print(f"{'='*72}\n")

    rows        = []
    kp_tables   = {}   # per-drug tissue Kp breakdown, printed after summary

    for name in TARGET_DRUGS:
        data = REFERENCE_PK.get(name)
        if data is None:
            print(f"  [!] {name}: not found in REFERENCE_PK — skipped")
            continue

        # ── Build profile ──────────────────────────────────────────────────
        try:
            profile = _build_profile(name, data)
        except Exception as exc:
            print(f"  [ERROR] {name}: build_drug_profile() raised {type(exc).__name__}: {exc}")
            continue

        kp = profile.get("kp", {})
        if not kp:
            print(f"  [ERROR] {name}: drug['kp'] is empty or absent in returned profile")
            continue

        # ── Compute Vd ────────────────────────────────────────────────────
        vd_info = _compute_vd(profile)
        vd_pred = vd_info["vd_predicted"]

        # ── Compare to literature ─────────────────────────────────────────
        lit      = LITERATURE_VD[name]
        vd_lit   = lit["vd_L"]
        fold_raw = vd_pred / vd_lit          # > 1 = over, < 1 = under
        fold_abs = max(fold_raw, 1.0 / fold_raw)
        direction = "OVER " if fold_raw >= 1.0 else "UNDER"
        within_2x = fold_abs <= 2.0

        # ── Annotations ───────────────────────────────────────────────────
        trap          = _lysosomal_eligible(data)
        kp_scalar_val = data.get("kp_scalar", None)
        kp_scalar_str = f"{kp_scalar_val:.2f}" if kp_scalar_val is not None else "none"

        top3 = ", ".join(
            f"{t}({v:.1f}L)"
            for t, v in vd_info["top_tissues"][:3]
        )

        flag = "✓" if within_2x else ("⚠ " if fold_abs <= 5.0 else "✗ ")
        print(
            f"  {flag} {name:<16}"
            f"  pred={vd_pred:>7.1f}L  lit={vd_lit:>6.1f}L"
            f"  fold={fold_raw:>6.2f}×({direction})"
            f"  lyso={('Y' if trap else 'N')}"
            f"  kp_scalar={kp_scalar_str}"
        )
        if vd_info["kp_missing"]:
            print(f"      [WARN] kp dict missing keys {vd_info['kp_missing']} — "
                  f"Kp=1.0 used as placeholder for each; Vd may be under-estimated")

        # Stash kp table for detailed printout below
        kp_tables[name] = {
            "profile":   profile,
            "vd_info":   vd_info,
            "vd_pred":   vd_pred,
            "vd_lit":    vd_lit,
            "fold_raw":  fold_raw,
            "fold_abs":  fold_abs,
            "direction": direction,
            "trap":      trap,
            "within_2x": within_2x,
            "lit_source":lit["source"],
        }

        rows.append({
            "Drug":        name,
            "Type":        data.get("drug_type", "?"),
            "logP":        data["logp"],
            "pKa":         str(data.get("pka", "?")),
            "fup":         data["fup"],
            "kp_scalar":   kp_scalar_str,
            "Vd_pred_L":   round(vd_pred, 1),
            "Vd_lit_L":    vd_lit,
            "Fold":        round(fold_raw, 3),
            "AbsFold":     round(fold_abs, 3),
            "Dir":         direction.strip(),
            "Pass(2x)":    "✓" if within_2x else "✗",
            "Lyso":        "Y" if trap else "N",
            "Top3":        top3,
        })

    if not rows:
        print("\n[!] No results produced — check engine import paths.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    abs_folds = df["AbsFold"].values

    print(f"\n{'='*72}")
    print(f"  RESULTS SUMMARY TABLE")
    print(f"{'='*72}")
    print(
        df[["Drug","Type","logP","fup","kp_scalar",
            "Vd_pred_L","Vd_lit_L","Fold","Dir","Pass(2x)","Lyso"]]
        .to_string(index=False)
    )
    print(f"\n  Aggregate statistics:")
    print(f"    Mean abs fold-error  : {np.mean(abs_folds):.2f}×")
    print(f"    Median abs fold-error: {np.median(abs_folds):.2f}×")
    print(f"    Within 2× (pass)     : {(abs_folds <= 2.0).sum()}/{len(abs_folds)}")
    print(f"    OVER-predicted       : {(df['Dir'].str.strip()=='OVER').sum()}/{len(df)}")
    print(f"    UNDER-predicted      : {(df['Dir'].str.strip()=='UNDER').sum()}/{len(df)}")

    # ── Pattern split: lysosomal-eligible vs not ──────────────────────────
    trap_rows    = df[df["Lyso"] == "Y"]
    notrap_rows  = df[df["Lyso"] == "N"]
    print(f"\n  Lysosomal-trapping eligible (pKa>7, logP>1.5, drug_type=basic):")
    if len(trap_rows):
        for _, r in trap_rows.iterrows():
            print(f"    {r['Drug']:<16} {r['Fold']:>6.2f}× {r['Dir']}  (Vd_pred={r['Vd_pred_L']}L vs lit={r['Vd_lit_L']}L)")
    else:
        print("    (none in target set)")
    print(f"  Non-eligible (neutral / acidic / zwitterion):")
    for _, r in notrap_rows.iterrows():
        print(f"    {r['Drug']:<16} {r['Fold']:>6.2f}× {r['Dir']}  (Vd_pred={r['Vd_pred_L']}L vs lit={r['Vd_lit_L']}L)")

    # ─────────────────────────────────────────────────────────────────────────
    # PER-DRUG TISSUE Kp BREAKDOWN
    # Printed for every drug to expose which tissues drive the Vd error.
    # For a drug with Vd under-prediction: which tissues have unexpectedly
    # low Kp? For over-prediction: which are inflated?
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  PER-DRUG TISSUE Kp BREAKDOWN")
    print(f"  Format: tissue  Kp  V_tissue(L)  contribution(L)")
    print(f"{'='*72}")

    for name, info in kp_tables.items():
        vd_info   = info["vd_info"]
        fold      = info["fold_raw"]
        direction = info["direction"]
        trap      = info["trap"]
        kp_dict   = vd_info["kp_dict"]
        contribs  = vd_info["tissue_contribs"]

        print(f"\n  ── {name}  "
              f"(Vd {info['vd_pred']:.1f}L pred vs {info['vd_lit']:.1f}L lit, "
              f"{fold:.2f}× {direction.strip()}, "
              f"lyso={'eligible' if trap else 'not eligible'}) ──")
        print(f"    {'Tissue':<10}  {'Kp':>8}  {'Vol(L)':>7}  {'Contrib(L)':>10}  {'%Vd':>6}")
        print(f"    {'-'*52}")

        # Blood first
        bc = vd_info["blood_contrib"]
        Rb = vd_info["Rb"]
        print(f"    {'blood':<10}  {'Rb='+str(round(Rb,2)):>8}  "
              f"{'(plasma+rbc)':>7}  {bc:>10.2f}  {100*bc/info['vd_pred']:>5.1f}%")

        # Tissues sorted by contribution
        for tissue, contrib in vd_info["top_tissues"]:
            kp_val   = float(kp_dict.get(tissue, 1.0))
            vol      = ORGAN_VOL[tissue]
            missing  = tissue in vd_info["kp_missing"]
            flag_str = " [Kp=1 default]" if missing else ""
            print(f"    {tissue:<10}  {kp_val:>8.3f}  {vol:>7.2f}  "
                  f"{contrib:>10.2f}  {100*contrib/info['vd_pred']:>5.1f}%"
                  f"{flag_str}")

        print(f"    {'-'*52}")
        print(f"    {'TOTAL':<10}  {'':>8}  {'':>7}  {info['vd_pred']:>10.2f}L")
        print(f"    Literature: {info['vd_lit']:.1f}L  ({info['lit_source']})")

        if info["vd_info"]["kp_missing"]:
            print(f"    [WARN] Tissues absent from kp dict: {info['vd_info']['kp_missing']}")

    # ─────────────────────────────────────────────────────────────────────────
    # DIAGNOSTIC INTERPRETATION
    # Auto-generates reading notes based on the actual results.
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  AUTO-INTERPRETATION")
    print(f"{'='*72}")

    trap_names   = [r["Drug"] for _, r in df[df["Lyso"]=="Y"].iterrows()]
    notrap_names = [r["Drug"] for _, r in df[df["Lyso"]=="N"].iterrows()]

    # Check lysosomal-trapping drugs for systematic under-prediction
    trap_folds    = df[df["Lyso"]=="Y"]["Fold"].values
    notrap_folds  = df[df["Lyso"]=="N"]["Fold"].values
    trap_dirs     = df[df["Lyso"]=="Y"]["Dir"].str.strip().values
    notrap_dirs   = df[df["Lyso"]=="N"]["Dir"].str.strip().values

    if len(trap_folds) > 0:
        trap_underpred = (trap_dirs == "UNDER").all()
        trap_overpred  = (trap_dirs == "OVER").all()
        trap_mixed     = not (trap_underpred or trap_overpred)
        if trap_underpred and (trap_folds < 0.5).any():
            print(
                f"\n  [FINDING — STRONG] Lysosomal-trapping eligible drugs "
                f"({', '.join(trap_names)}) show systematic Vd UNDER-prediction "
                f"(fold < 0.5).  The de Duve ion-trapping correction in admet.py "
                f"is either too weak, gated incorrectly, or not activating at all "
                f"for these drugs. This directly explains Cmax OVER-prediction: "
                f"drug cannot distribute into tissues and pools in plasma."
            )
        elif trap_underpred:
            print(
                f"\n  [FINDING — MODERATE] Lysosomal-eligible drugs "
                f"({', '.join(trap_names)}) show Vd UNDER-prediction. "
                f"The lysosomal trapping correction may be too conservative. "
                f"Confirm the de Duve amplification factor and pKa eligibility "
                f"check in admet._lysosomal_kp_correction()."
            )
        elif trap_overpred:
            print(
                f"\n  [FINDING — MODERATE] Lysosomal-eligible drugs "
                f"({', '.join(trap_names)}) show Vd OVER-prediction. "
                f"The trapping amplification factor may be too aggressive, "
                f"or a kp_scalar correction is needed to bring Vd back in range."
            )
        elif trap_mixed:
            print(
                f"\n  [INCONCLUSIVE] Lysosomal-eligible drugs show mixed "
                f"direction — not consistent with a single trapping model "
                f"calibration error. Per-drug investigation needed."
            )

    if len(notrap_folds) > 0:
        notrap_underpred = (notrap_dirs == "UNDER").all()
        notrap_overpred  = (notrap_dirs == "OVER").all()
        if notrap_underpred and (notrap_folds < 0.5).any():
            print(
                f"\n  [FINDING — STRONG] Non-trapping drugs "
                f"({', '.join(notrap_names)}) also show systematic Vd "
                f"UNDER-prediction. Because lysosomal trapping does not apply "
                f"to these drugs, the fault is in the R&R Kp model baseline, "
                f"not in the trapping correction — a broader distribution model "
                f"problem affecting these drug classes."
            )
        elif notrap_underpred:
            print(
                f"\n  [FINDING — MODERATE] Non-trapping drugs "
                f"({', '.join(notrap_names)}) also tend to under-predict Vd, "
                f"suggesting the R&R baseline model may be contributing "
                f"independent of the lysosomal trapping correction."
            )
        elif notrap_overpred:
            print(
                f"\n  [FINDING] Non-trapping drugs "
                f"({', '.join(notrap_names)}) show Vd OVER-prediction. "
                f"This is the opposite of the expected direction for absorption "
                f"issues; check whether a kp_scalar should be applied."
            )

    # Check whether both groups fail in the same direction
    if len(trap_folds) > 0 and len(notrap_folds) > 0:
        all_under = (df["Dir"].str.strip() == "UNDER").all()
        all_over  = (df["Dir"].str.strip() == "OVER").all()
        if all_under:
            print(
                f"\n  [FINDING — STRONG PATTERN] ALL 7 target drugs show Vd "
                f"UNDER-prediction.  A systematic one-sided failure across both "
                f"trapping-eligible and non-eligible drugs, across four drug "
                f"types, strongly implicates a shared component in the Kp "
                f"estimation path (e.g. the tissue composition partition formula, "
                f"a missing fup correction, or the organ volume table) rather "
                f"than a drug-class-specific model."
            )
        elif all_over:
            print(
                f"\n  [FINDING — STRONG PATTERN] ALL 7 target drugs show Vd "
                f"OVER-prediction — drug appears more distributed than reality. "
                f"Check whether kp_scalar is missing for all of them and whether "
                f"the R&R model is systematically over-predicting tissue binding."
            )

    # Specific check: Metoprolol direction vs known Cmax failure direction
    if "Metoprolol" in kp_tables:
        metop_fold = kp_tables["Metoprolol"]["fold_raw"]
        if metop_fold < 0.7:
            print(
                f"\n  [KEY LINK] Metoprolol Vd UNDER-predicted ({metop_fold:.2f}× of lit). "
                f"This is mechanistically consistent with the confirmed Cmax 3.02× "
                f"over-prediction: low predicted Vd → drug cannot leave plasma → "
                f"plasma concentration is too high. Vd mismatch is a plausible "
                f"primary root cause for Metoprolol's validation failure."
            )
        elif metop_fold > 1.4:
            print(
                f"\n  [INCONSISTENCY] Metoprolol Vd is OVER-predicted ({metop_fold:.2f}×) "
                f"but Cmax is also over-predicted (3.02×). Vd and Cmax should move "
                f"in opposite directions — this combination suggests the dominant "
                f"lever for Metoprolol is not Vd. Pivot to k_abs / elimination "
                f"investigation for Metoprolol specifically."
            )
        else:
            print(
                f"\n  [INCONCLUSIVE — Metoprolol] Vd within 1.4× of literature "
                f"({metop_fold:.2f}×) but Cmax is confirmed 3.02× over. "
                f"Vd is not the lever for Metoprolol. Pivot to the k_abs "
                f"effective-rate diagnostic (missing peff_is_measured_net for "
                f"Propranolol; ionization double-count hypothesis for others)."
            )

    # Next-step recommendation
    failing   = df[df["Pass(2x)"] == "✗"]["Drug"].tolist()
    passing   = df[df["Pass(2x)"] == "✓"]["Drug"].tolist()
    print(f"\n  RECOMMENDED NEXT STEP:")
    if len(failing) >= 4:
        print(
            f"  Majority of target drugs have Vd fold-error > 2× "
            f"({', '.join(failing)}). Vd mismatch is a confirmed, measurable "
            f"defect for these drugs. Script 10 should be a kp_scalar sensitivity "
            f"sweep for the failing drugs to quantify how much kp_scalar correction "
            f"is needed to bring Vd into range, and to confirm that correcting Vd "
            f"also closes Cmax/AUC fold-errors in the full simulation."
        )
    elif len(failing) >= 2:
        print(
            f"  Partial Vd mismatch: {', '.join(failing)} failing, "
            f"{', '.join(passing)} passing within 2×. "
            f"Script 10 should run a kp_scalar sweep for the failing subset "
            f"while simultaneously running a k_abs effective-rate diagnostic "
            f"for the passing subset (Vd ruled out → absorption or elimination "
            f"is the residual lever)."
        )
    else:
        print(
            f"  Vd is correct for most/all target drugs. Distribution is NOT the "
            f"dominant lever. Pivot to Script 10: k_abs effective-rate diagnostic "
            f"— compute per-segment k_abs for all 7 drugs and check whether "
            f"ionization corrections are consistent with each drug's p_eff source "
            f"(particular focus: missing peff_is_measured_net for Propranolol, "
            f"zwitterion f_u floor vs MATE1 Vmax balance for Ciprofloxacin)."
        )

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    run_diagnostic()