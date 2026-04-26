"""
risk_scorer.py — Organ risk assessment for BodySim.

v1.1 — Molecule-specific thresholds from ADMET-AI safety signals.
       Generic thresholds used as fallback when drug profile
       does not include ADMET safety data.

Risk score: 0.0 (safe) → 1.0 (high risk)

Colour mapping:
  0.00–0.20  green   Safe
  0.20–0.45  yellow  Monitor
  0.45–0.70  orange  Elevated Risk
  0.70–1.00  red     High Risk
"""

import numpy as np

# ── Generic fallback thresholds (mg·h/L AUC, mg/L Cmax) ──────────────────
# Used when no drug-specific data available
DEFAULT_THRESHOLDS = {
    "liver":   (50.0,  8.0),
    "kidney":  (60.0,  8.0),
    "brain":   (10.0,  2.0),
    "heart":   (20.0,  4.0),
    "muscle":  (80.0, 15.0),
    "fat":     (200.0,30.0),
    "gut":     (40.0,  6.0),
    "skin":    (60.0, 10.0),
    "bone":    (80.0, 12.0),
    "lung":    (25.0,  5.0),
    "rest":    (80.0, 12.0),
}

ORGAN_DISPLAY = {
    "liver":"Liver","kidney":"Kidneys","brain":"Brain",
    "heart":"Heart","muscle":"Muscle","fat":"Adipose",
    "gut":"GI Tract","skin":"Skin","bone":"Bone",
    "lung":"Lungs","rest":"Other Tissues",
}

RISK_BANDS = [
    (0.00, 0.20, "green",  "Safe"),
    (0.20, 0.45, "yellow", "Monitor"),
    (0.45, 0.70, "orange", "Elevated Risk"),
    (0.70, 1.00, "red",    "High Risk"),
]


def build_molecule_thresholds(drug_profile: dict) -> dict:
    """
    Build molecule-specific organ thresholds from ADMET-AI safety signals.

    Uses three signals from the drug profile:
      dili_prob  — drug-induced liver injury probability  [0–1]
      herg_prob  — hERG cardiac channel inhibition        [0–1]
      bbb_prob   — blood-brain barrier penetration        [0–1]

    A high dili_prob tightens the liver threshold.
    A high herg_prob tightens the heart threshold.
    A low bbb_prob loosens the brain threshold (drug can't get in).

    Returns thresholds dict compatible with score_single_simulation().
    """
    thresholds = {k: list(v) for k, v in DEFAULT_THRESHOLDS.items()}

    dili = float(drug_profile.get("dili_prob",  0.0))
    herg = float(drug_profile.get("herg_prob",  0.0))
    bbb  = float(drug_profile.get("bbb_prob",   0.5))

    # ── Liver threshold — tighter when DILI risk is high ──────────────────
    # dili_prob = 0.0 → threshold stays at default (50 mg·h/L)
    # dili_prob = 0.5 → threshold halved (25 mg·h/L)
    # dili_prob = 1.0 → threshold quartered (12.5 mg·h/L)
    liver_scale = 1.0 / (1.0 + 3.0 * dili)
    thresholds["liver"][0] = DEFAULT_THRESHOLDS["liver"][0] * liver_scale
    thresholds["liver"][1] = DEFAULT_THRESHOLDS["liver"][1] * liver_scale

    # ── Heart threshold — tighter when hERG risk is high ──────────────────
    heart_scale = 1.0 / (1.0 + 4.0 * herg)
    thresholds["heart"][0] = DEFAULT_THRESHOLDS["heart"][0] * heart_scale
    thresholds["heart"][1] = DEFAULT_THRESHOLDS["heart"][1] * heart_scale

    # ── Brain threshold — looser when BBB penetration is LOW ──────────────
    # If drug can't cross BBB (bbb_prob≈0), brain exposure is irrelevant
    # Raise threshold → lower risk score for same AUC
    if bbb < 0.3:
        brain_scale = 3.0   # drug barely enters brain → high threshold
    elif bbb < 0.6:
        brain_scale = 1.5
    else:
        brain_scale = 1.0   # drug freely enters brain → keep tight
    thresholds["brain"][0] = DEFAULT_THRESHOLDS["brain"][0] * brain_scale
    thresholds["brain"][1] = DEFAULT_THRESHOLDS["brain"][1] * brain_scale

    return {k: tuple(v) for k, v in thresholds.items()}


def _risk_from_ratio(ratio: float) -> float:
    """Sigmoid: ratio=1 → score=0.5; ratio=2 → score=0.8."""
    if ratio <= 0: return 0.0
    score = ratio ** 2 / (ratio ** 2 + 1.0)
    return float(np.clip(score, 0.0, 1.0))


def score_single_simulation(result: dict,
                             thresholds: dict = None) -> dict:
    """
    Calculate organ risk scores for a single simulation result.

    If the result contains a drug profile with ADMET safety signals
    (dili_prob, herg_prob, bbb_prob), molecule-specific thresholds
    are used automatically.

    Parameters
    ----------
    result     : dict  output from PBPKModel.solve()
    thresholds : dict  optional manual override

    Returns
    -------
    dict with organ_scores, organ_labels, dominant_organ, summary
    """
    # Prefer molecule-specific thresholds if drug profile has safety signals
    if thresholds is None:
        drug_profile = result.get("drug", {})
        has_admet    = any(k in drug_profile
                          for k in ("dili_prob","herg_prob","bbb_prob"))
        thresholds = (build_molecule_thresholds(drug_profile)
                      if has_admet else DEFAULT_THRESHOLDS)

    auc_organs = result["auc_organs"]
    organs_t   = result["organs"]

    organ_scores = {}
    auc_ratios   = {}

    for organ, (auc_lim, cmax_lim) in thresholds.items():
        if organ not in auc_organs: continue

        auc  = auc_organs[organ]
        cmax = float(np.max(organs_t[organ])) if organ in organs_t else 0.0

        auc_ratio  = auc  / auc_lim  if auc_lim  > 0 else 0.0
        cmax_ratio = cmax / cmax_lim if cmax_lim > 0 else 0.0

        score = _risk_from_ratio(max(auc_ratio, cmax_ratio))
        organ_scores[organ] = score
        auc_ratios[organ]   = auc_ratio

    organ_labels = {}
    for organ, score in organ_scores.items():
        for lo, hi, colour, label in RISK_BANDS:
            if score < hi:
                organ_labels[organ] = colour
                break
        else:
            organ_labels[organ] = "red"

    dominant       = max(organ_scores, key=organ_scores.get) if organ_scores else "none"
    dominant_score = organ_scores.get(dominant, 0.0)

    high_risk = [o for o, s in organ_scores.items() if s >= 0.70]
    mod_risk  = [o for o, s in organ_scores.items() if 0.45 <= s < 0.70]
    parts = []
    if high_risk:
        parts.append("High risk: " + ", ".join(ORGAN_DISPLAY.get(o,o) for o in high_risk))
    if mod_risk:
        parts.append("Elevated: " + ", ".join(ORGAN_DISPLAY.get(o,o) for o in mod_risk))
    if not parts:
        parts.append("No organs exceed risk thresholds at this dose.")

    return {
        "organ_scores":   organ_scores,
        "organ_labels":   organ_labels,
        "dominant_organ": dominant,
        "dominant_score": dominant_score,
        "auc_ratios":     auc_ratios,
        "summary":        " | ".join(parts),
        "thresholds_used": "molecule_specific" if has_admet else "generic",
    } if "has_admet" in dir() else {
        "organ_scores":   organ_scores,
        "organ_labels":   organ_labels,
        "dominant_organ": dominant,
        "dominant_score": dominant_score,
        "auc_ratios":     auc_ratios,
        "summary":        " | ".join(parts),
        "thresholds_used": "generic",
    }


def score_population(pop_results: list,
                     thresholds: dict = None) -> dict:
    if not pop_results: return {}

    organ_score_arrays = {}
    auc_arrays         = {}
    plasma_aucs        = []
    plasma_cmaxs       = []

    for res in pop_results:
        scores = score_single_simulation(res, thresholds)
        for organ, s in scores["organ_scores"].items():
            organ_score_arrays.setdefault(organ, []).append(s)
        for organ, auc in res["auc_organs"].items():
            auc_arrays.setdefault(organ, []).append(auc)
        plasma_aucs.append(res["auc_plasma"])
        plasma_cmaxs.append(res["cmax_plasma"])

    organ_pop_risk = {o: float(np.mean(v))           for o,v in organ_score_arrays.items()}
    organ_p95_risk = {o: float(np.percentile(v, 95)) for o,v in organ_score_arrays.items()}
    pct_high       = {o: float(100*np.mean(np.array(v)>=0.70))
                      for o,v in organ_score_arrays.items()}

    auc_stats = {}
    for organ, aucs in auc_arrays.items():
        a = np.array(aucs)
        auc_stats[organ] = {
            "mean":float(np.mean(a)), "std":float(np.std(a)),
            "p5": float(np.percentile(a,5)),
            "p50":float(np.percentile(a,50)),
            "p95":float(np.percentile(a,95)),
        }

    plasma_stats = {
        "auc_mean": float(np.mean(plasma_aucs)),
        "auc_p5":   float(np.percentile(plasma_aucs,5)),
        "auc_p50":  float(np.percentile(plasma_aucs,50)),
        "auc_p95":  float(np.percentile(plasma_aucs,95)),
        "cmax_mean":float(np.mean(plasma_cmaxs)),
        "cmax_p5":  float(np.percentile(plasma_cmaxs,5)),
        "cmax_p50": float(np.percentile(plasma_cmaxs,50)),
        "cmax_p95": float(np.percentile(plasma_cmaxs,95)),
    }

    elevated_pct  = {o: float(100*np.mean(np.array(v)>=0.45))
                     for o,v in organ_score_arrays.items()}
    most_common   = max(elevated_pct, key=elevated_pct.get) if elevated_pct else "none"

    top = sorted(pct_high.items(), key=lambda x:-x[1])[:3]
    parts = [f"{ORGAN_DISPLAY.get(o,o)}: {p:.0f}% at high risk"
             for o,p in top if p > 0]
    pop_summary = " | ".join(parts) if parts else "No organs at high risk."

    return {
        "organ_population_risk": organ_pop_risk,
        "organ_p95_risk":        organ_p95_risk,
        "auc_statistics":        auc_stats,
        "plasma_statistics":     plasma_stats,
        "most_common_organ":     most_common,
        "pct_high_risk":         pct_high,
        "population_summary":    pop_summary,
        "n_subjects":            len(pop_results),
    }


def print_risk_report(single_result, pop_result=None):
    scores    = score_single_simulation(single_result)
    drug_name = single_result["drug"].get("name","Unknown Drug")
    dose      = single_result["dose_mg"]
    route     = single_result["route"]

    print("\n" + "═"*60)
    print(f"  BODYSIM RISK REPORT — {drug_name}")
    print(f"  Dose: {dose} mg  |  Route: {route.upper()}")
    print(f"  Thresholds: {scores.get('thresholds_used','generic')}")
    print("═"*60)
    print(f"\n  Plasma AUC  : {single_result['auc_plasma']:.2f} mg·h/L")
    print(f"  Plasma Cmax : {single_result['cmax_plasma']:.3f} mg/L")
    print(f"  Plasma Tmax : {single_result['tmax_plasma']:.2f} h")
    print(f"\n  {'Organ':<18} {'Score':>6}  {'Status':<14} {'AUC-ratio':>10}")
    print("  " + "─"*52)

    for organ, score in sorted(scores["organ_scores"].items(),
                                key=lambda x:-x[1]):
        colour  = scores["organ_labels"].get(organ,"green")
        label   = {"green":"Safe","yellow":"Monitor",
                   "orange":"Elevated","red":"HIGH RISK"}.get(colour,colour)
        auc_r   = scores["auc_ratios"].get(organ, 0.0)
        flag    = " ◄◄" if score >= 0.70 else (" ◄" if score >= 0.45 else "")
        print(f"  {ORGAN_DISPLAY.get(organ,organ):<18} {score:>6.3f}"
              f"  {label:<14} {auc_r:>10.2f}{flag}")

    print(f"\n  Dominant : {ORGAN_DISPLAY.get(scores['dominant_organ'],'?')}")
    print(f"  Summary  : {scores['summary']}")

    if pop_result:
        ps = pop_result["plasma_statistics"]
        print(f"\n  ── Population ({pop_result['n_subjects']} subjects) ──")
        print(f"  AUC median : {ps['auc_p50']:.2f}"
              f"  [P5={ps['auc_p5']:.2f}, P95={ps['auc_p95']:.2f}]")
        print(f"  Most at risk: "
              f"{ORGAN_DISPLAY.get(pop_result['most_common_organ'],'?')}")
        print(f"  {pop_result['population_summary']}")
    print("═"*60 + "\n")
