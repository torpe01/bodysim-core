"""
risk_scorer.py — Organ risk assessment for BodySim.

v1.2 — Confidence intervals on all risk scores from Monte Carlo output.
        Molecule-specific thresholds from ADMET-AI safety signals.

Risk score: 0.0 (safe) → 1.0 (high risk)

Colour bands:
  0.00 – 0.20  green    Safe
  0.20 – 0.45  yellow   Monitor
  0.45 – 0.70  orange   Elevated Risk
  0.70 – 1.00  red      High Risk

NEW in v1.2
───────────
  score_with_uncertainty()  — attach CI bounds to a single risk score
  report_with_ci()          — formatted risk report including CI columns
  RiskResult dataclass      — structured output with point + CI + verdict
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Generic fallback thresholds (mg·h/L AUC, mg/L Cmax) ──────────────────
# Used when no drug-specific ADMET safety data is available.
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
    "liver":  "Liver",   "kidney": "Kidneys", "brain":  "Brain",
    "heart":  "Heart",   "muscle": "Muscle",  "fat":    "Adipose",
    "gut":    "GI Tract","skin":   "Skin",    "bone":   "Bone",
    "lung":   "Lungs",   "rest":   "Other Tissues",
}

RISK_BANDS = [
    (0.00, 0.20, "green",  "Safe"),
    (0.20, 0.45, "yellow", "Monitor"),
    (0.45, 0.70, "orange", "Elevated Risk"),
    (0.70, 1.00, "red",    "High Risk"),
]


# ── Structured risk result with uncertainty ────────────────────────────────
@dataclass
class OrganRisk:
    """
    Risk assessment for a single organ.

    Attributes
    ----------
    organ        : str    organ key (e.g. 'kidney')
    score        : float  point-estimate risk score [0–1]
    colour       : str    'green'|'yellow'|'orange'|'red'
    label        : str    'Safe'|'Monitor'|'Elevated Risk'|'High Risk'
    auc_ratio    : float  organ AUC / threshold AUC
    ci_lo        : float  lower CI bound on risk score (from Monte Carlo)
    ci_hi        : float  upper CI bound on risk score (from Monte Carlo)
    has_ci       : bool   True if CI bounds are available
    tail_risk    : bool   True if ci_hi ≥ 0.70 but score < 0.45
                          (safe on average, dangerous at extremes)
    """
    organ:     str
    score:     float
    colour:    str
    label:     str
    auc_ratio: float
    ci_lo:     float  = 0.0
    ci_hi:     float  = 0.0
    has_ci:    bool   = False
    tail_risk: bool   = False

    def to_dict(self) -> dict:
        return {
            "organ":     self.organ,
            "score":     self.score,
            "colour":    self.colour,
            "label":     self.label,
            "auc_ratio": self.auc_ratio,
            "ci_lo":     self.ci_lo,
            "ci_hi":     self.ci_hi,
            "has_ci":    self.has_ci,
            "tail_risk": self.tail_risk,
        }


# ── Molecule-specific thresholds ──────────────────────────────────────────

def build_molecule_thresholds(drug_profile: dict) -> dict:
    """
    Build molecule-specific organ thresholds from ADMET-AI safety signals.

    Uses:
      dili_prob  Drug-induced liver injury probability [0–1]
                 → tighter liver threshold when high
      herg_prob  hERG cardiac channel inhibition [0–1]
                 → tighter heart threshold when high
      bbb_prob   BBB penetration probability [0–1]
                 → looser brain threshold when low (drug can't get in)

    Threshold scaling:
      liver  : 1 / (1 + 3 × dili_prob)   → 25% of default at dili=1.0
      heart  : 1 / (1 + 4 × herg_prob)   → 20% of default at herg=1.0
      brain  : ×3 if bbb<0.3, ×1.5 if bbb<0.6, ×1 otherwise

    Parameters
    ----------
    drug_profile : dict  drug profile; may include dili_prob, herg_prob, bbb_prob

    Returns
    -------
    dict {organ: (auc_threshold, cmax_threshold)}
    """
    thresholds = {k: list(v) for k, v in DEFAULT_THRESHOLDS.items()}

    dili = float(drug_profile.get("dili_prob",  0.0))
    herg = float(drug_profile.get("herg_prob",  0.0))
    bbb  = float(drug_profile.get("bbb_prob",   0.5))

    # Liver — DILI signal tightens threshold
    liver_scale = 1.0 / (1.0 + 3.0 * dili)
    thresholds["liver"][0] *= liver_scale
    thresholds["liver"][1] *= liver_scale

    # Heart — hERG signal tightens threshold
    heart_scale = 1.0 / (1.0 + 4.0 * herg)
    thresholds["heart"][0] *= heart_scale
    thresholds["heart"][1] *= heart_scale

    # Brain — low BBB penetration loosens threshold
    # (high brain AUC is only worrying if the drug can actually get in)
    if bbb < 0.3:
        brain_scale = 3.0
    elif bbb < 0.6:
        brain_scale = 1.5
    else:
        brain_scale = 1.0
    thresholds["brain"][0] *= brain_scale
    thresholds["brain"][1] *= brain_scale

    return {k: tuple(v) for k, v in thresholds.items()}


# ── Core scoring ──────────────────────────────────────────────────────────

def _risk_from_ratio(ratio: float) -> float:
    """
    Sigmoid conversion from AUC-ratio to risk score.
    ratio = 1.0 → score = 0.5 (at threshold)
    ratio = 2.0 → score ≈ 0.80 (2× threshold is high risk)
    ratio = 0.5 → score ≈ 0.20 (half threshold is low risk)
    """
    if ratio <= 0: return 0.0
    return float(np.clip(ratio**2 / (ratio**2 + 1.0), 0.0, 1.0))


def _colour_for_score(score: float):
    """Return (colour, label) for a risk score."""
    for lo, hi, colour, label in RISK_BANDS:
        if score < hi:
            return colour, label
    return "red", "High Risk"


def score_single_simulation(result: dict,
                             thresholds: dict = None) -> dict:
    """
    Calculate organ risk scores for a single simulation result.

    Automatically uses molecule-specific thresholds when the drug
    profile contains ADMET-AI safety signals (dili_prob, herg_prob,
    bbb_prob). Falls back to DEFAULT_THRESHOLDS otherwise.

    Parameters
    ----------
    result     : dict  output from PBPKModel.solve()
    thresholds : dict  optional manual threshold override

    Returns
    -------
    dict with:
      organ_scores     {organ: float}
      organ_labels     {organ: colour str}
      dominant_organ   str
      dominant_score   float
      auc_ratios       {organ: float}
      summary          str
      thresholds_used  str
    """
    drug_profile = result.get("drug", {})
    has_admet    = any(k in drug_profile
                       for k in ("dili_prob","herg_prob","bbb_prob"))

    if thresholds is None:
        thresholds     = (build_molecule_thresholds(drug_profile)
                          if has_admet else DEFAULT_THRESHOLDS)
        thresh_label   = "molecule_specific" if has_admet else "generic"
    else:
        thresh_label   = "custom"

    auc_organs = result["auc_organs"]
    organs_t   = result["organs"]

    organ_scores = {}
    auc_ratios   = {}

    for organ, (auc_lim, cmax_lim) in thresholds.items():
        if organ not in auc_organs:
            continue
        auc      = auc_organs[organ]
        cmax     = float(np.max(organs_t[organ])) if organ in organs_t else 0.0
        auc_r    = auc  / auc_lim  if auc_lim  > 0 else 0.0
        cmax_r   = cmax / cmax_lim if cmax_lim > 0 else 0.0
        score    = _risk_from_ratio(max(auc_r, cmax_r))
        organ_scores[organ] = score
        auc_ratios[organ]   = auc_r

    organ_labels = {o: _colour_for_score(s)[0] for o, s in organ_scores.items()}

    dominant       = (max(organ_scores, key=organ_scores.get)
                      if organ_scores else "none")
    dominant_score = organ_scores.get(dominant, 0.0)

    high  = [o for o,s in organ_scores.items() if s >= 0.70]
    mod   = [o for o,s in organ_scores.items() if 0.45 <= s < 0.70]
    parts = []
    if high: parts.append("High risk: "   + ", ".join(ORGAN_DISPLAY.get(o,o) for o in high))
    if mod:  parts.append("Elevated: " + ", ".join(ORGAN_DISPLAY.get(o,o) for o in mod))
    if not parts: parts.append("No organs exceed risk thresholds at this dose.")

    return {
        "organ_scores":   organ_scores,
        "organ_labels":   organ_labels,
        "dominant_organ": dominant,
        "dominant_score": dominant_score,
        "auc_ratios":     auc_ratios,
        "summary":        " | ".join(parts),
        "thresholds_used":thresh_label,
    }


# ── NEW: Attach CI bounds to risk scores from Monte Carlo ─────────────────

def score_with_uncertainty(result: dict,
                            risk_ci: dict,
                            thresholds: dict = None) -> list:
    """
    Attach Monte Carlo confidence intervals to organ risk scores.

    Combines:
      - Point-estimate risk scores (from score_single_simulation)
      - CI bounds from Monte Carlo (from simulator.run_uncertainty)

    Parameters
    ----------
    result    : dict  single simulation result from PBPKModel.solve()
    risk_ci   : dict  {organ: {"p_lo": float, "p50": float, "p_hi": float}}
                      from run_uncertainty()["risk_ci"]
    thresholds: dict  optional override

    Returns
    -------
    list of OrganRisk objects, sorted by risk score descending
    """
    base = score_single_simulation(result, thresholds)
    organ_risks = []

    for organ, score in base["organ_scores"].items():
        colour, label = _colour_for_score(score)
        ci_data       = risk_ci.get(organ, {})
        ci_lo         = float(ci_data.get("p_lo", 0.0))
        ci_hi         = float(ci_data.get("p_hi", 0.0))
        has_ci        = bool(ci_data)
        tail_risk     = has_ci and ci_hi >= 0.70 and score < 0.45

        organ_risks.append(OrganRisk(
            organ     = organ,
            score     = score,
            colour    = colour,
            label     = label,
            auc_ratio = base["auc_ratios"].get(organ, 0.0),
            ci_lo     = ci_lo,
            ci_hi     = ci_hi,
            has_ci    = has_ci,
            tail_risk = tail_risk,
        ))

    return sorted(organ_risks, key=lambda r: -r.score)


def report_with_ci(result: dict, risk_ci: dict,
                   thresholds: dict = None,
                   confidence_level: float = 0.90):
    """
    Print a risk report with confidence interval columns.

    Parameters
    ----------
    result           : dict  simulation result
    risk_ci          : dict  from run_uncertainty()["risk_ci"]
    thresholds       : dict  optional override
    confidence_level : float e.g. 0.90 for 90% CI
    """
    organ_risks = score_with_uncertainty(result, risk_ci, thresholds)
    drug_name   = result["drug"].get("name", "Unknown")
    dose        = result["dose_mg"]
    route       = result["route"]
    ci_pct      = int(confidence_level * 100)
    lop         = int((1 - confidence_level) / 2 * 100)
    hip         = int((1 + confidence_level) / 2 * 100)

    print("\n" + "═" * 72)
    print(f"  RISK REPORT WITH UNCERTAINTY — {drug_name}")
    print(f"  Dose: {dose} mg  |  Route: {route.upper()}  |  {ci_pct}% CI")
    print("═" * 72)
    print(f"\n  Plasma AUC  : {result['auc_plasma']:.2f} mg·h/L")
    print(f"  Plasma Cmax : {result['cmax_plasma']:.3f} mg/L")
    print(f"  Plasma Tmax : {result['tmax_plasma']:.2f} h")

    print(f"\n  {'Organ':<18} {'Score':>6}  "
          f"{'P'+str(lop):>6}  {'P50':>6}  {'P'+str(hip):>6}  "
          f"{'Status':<14} Flag")
    print("  " + "─" * 68)

    for r in organ_risks:
        name   = ORGAN_DISPLAY.get(r.organ, r.organ)
        status = {"green":"Safe","yellow":"Monitor",
                  "orange":"Elevated","red":"HIGH RISK"}.get(r.colour, r.colour)
        flag   = "⚠ TAIL RISK" if r.tail_risk else (
                 "◄◄" if r.score >= 0.70 else (
                 "◄"  if r.score >= 0.45 else ""))

        if r.has_ci:
            ci_str = (f"{r.ci_lo:>6.3f}  "
                      f"{(r.ci_lo+r.ci_hi)/2:>6.3f}  "
                      f"{r.ci_hi:>6.3f}")
        else:
            ci_str = f"{'—':>6}  {'—':>6}  {'—':>6}"

        print(f"  {name:<18} {r.score:>6.3f}  {ci_str}  "
              f"{status:<14} {flag}")

    # Tail risk explanation
    tail_organs = [r for r in organ_risks if r.tail_risk]
    if tail_organs:
        print(f"\n  ⚠ Tail risk organs (safe on average, high risk at P{hip}):")
        for r in tail_organs:
            print(f"    {ORGAN_DISPLAY.get(r.organ, r.organ)}: "
                  f"score={r.score:.3f} but P{hip}={r.ci_hi:.3f} — "
                  f"monitor in sensitive/elderly populations")

    print("═" * 72 + "\n")


# ── Population scoring ────────────────────────────────────────────────────

def score_population(pop_results: list, thresholds: dict = None) -> dict:
    """
    Aggregate risk scores across a virtual population.

    Parameters
    ----------
    pop_results : list of result dicts from PBPKModel.solve()
    thresholds  : dict  optional override

    Returns
    -------
    dict with organ_population_risk, organ_p95_risk, auc_statistics,
         plasma_statistics, most_common_organ, pct_high_risk,
         population_summary, n_subjects
    """
    if not pop_results:
        return {}

    organ_score_arrs = {}
    auc_arrs         = {}
    plasma_aucs      = []
    plasma_cmaxs     = []

    for res in pop_results:
        sc = score_single_simulation(res, thresholds)
        for organ, s in sc["organ_scores"].items():
            organ_score_arrs.setdefault(organ, []).append(s)
        for organ, auc in res["auc_organs"].items():
            auc_arrs.setdefault(organ, []).append(auc)
        plasma_aucs.append(res["auc_plasma"])
        plasma_cmaxs.append(res["cmax_plasma"])

    def stats(arr):
        a = np.array(arr)
        return {"mean":float(np.mean(a)),  "std":float(np.std(a)),
                "p5": float(np.percentile(a,5)),
                "p50":float(np.percentile(a,50)),
                "p95":float(np.percentile(a,95))}

    organ_pop_risk = {o: float(np.mean(v))            for o,v in organ_score_arrs.items()}
    organ_p95_risk = {o: float(np.percentile(v,95))   for o,v in organ_score_arrs.items()}
    pct_high       = {o: float(100*np.mean(np.array(v)>=0.70))
                      for o,v in organ_score_arrs.items()}
    elevated_pct   = {o: float(100*np.mean(np.array(v)>=0.45))
                      for o,v in organ_score_arrs.items()}

    auc_stats    = {o: stats(v) for o,v in auc_arrs.items()}
    pa           = np.array(plasma_aucs)
    pc           = np.array(plasma_cmaxs)
    plasma_stats = {
        "auc_mean":  float(np.mean(pa)),
        "auc_p5":    float(np.percentile(pa,5)),
        "auc_p50":   float(np.percentile(pa,50)),
        "auc_p95":   float(np.percentile(pa,95)),
        "cmax_mean": float(np.mean(pc)),
        "cmax_p5":   float(np.percentile(pc,5)),
        "cmax_p50":  float(np.percentile(pc,50)),
        "cmax_p95":  float(np.percentile(pc,95)),
    }

    most_common = (max(elevated_pct, key=elevated_pct.get)
                   if elevated_pct else "none")

    top_3  = sorted(pct_high.items(), key=lambda x:-x[1])[:3]
    parts  = [f"{ORGAN_DISPLAY.get(o,o)}: {p:.0f}% at high risk"
               for o,p in top_3 if p > 0]
    pop_sum= " | ".join(parts) if parts else "No organs at high risk."

    return {
        "organ_population_risk": organ_pop_risk,
        "organ_p95_risk":        organ_p95_risk,
        "auc_statistics":        auc_stats,
        "plasma_statistics":     plasma_stats,
        "most_common_organ":     most_common,
        "pct_high_risk":         pct_high,
        "population_summary":    pop_sum,
        "n_subjects":            len(pop_results),
    }


# ── Standard report printer ───────────────────────────────────────────────

def print_risk_report(single_result, pop_result=None):
    """Print formatted risk report to stdout."""
    scores    = score_single_simulation(single_result)
    drug_name = single_result["drug"].get("name", "Unknown")
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
                                key=lambda x: -x[1]):
        colour = scores["organ_labels"].get(organ, "green")
        label  = {"green":"Safe","yellow":"Monitor",
                  "orange":"Elevated","red":"HIGH RISK"}.get(colour, colour)
        auc_r  = scores["auc_ratios"].get(organ, 0.0)
        flag   = " ◄◄" if score>=0.70 else (" ◄" if score>=0.45 else "")
        print(f"  {ORGAN_DISPLAY.get(organ,organ):<18} {score:>6.3f}"
              f"  {label:<14} {auc_r:>10.2f}{flag}")

    print(f"\n  Dominant : {ORGAN_DISPLAY.get(scores['dominant_organ'],'?')}")
    print(f"  Summary  : {scores['summary']}")

    if pop_result:
        ps = pop_result["plasma_statistics"]
        print(f"\n  ── Population ({pop_result['n_subjects']} subjects) ──")
        print(f"  AUC median : {ps['auc_p50']:.2f}  "
              f"[P5={ps['auc_p5']:.2f}, P95={ps['auc_p95']:.2f}]")
        print(f"  Most at risk: "
              f"{ORGAN_DISPLAY.get(pop_result['most_common_organ'],'?')}")
        print(f"  {pop_result['population_summary']}")
    print("═"*60 + "\n")