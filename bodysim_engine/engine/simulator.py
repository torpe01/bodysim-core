"""
simulator.py — Main BodySim simulation orchestrator.

v1.2 — Added Monte Carlo uncertainty quantification.

─────────────────────────────────────────────────────────────────
WHY UNCERTAINTY BOUNDS MATTER
─────────────────────────────────────────────────────────────────
Every ADMET property predicted by ADMET-AI has inherent uncertainty.
CLint predictions from in-vitro assays correlate with human in-vivo
clearance at r²≈0.5–0.7. Without propagating this uncertainty,
"Cmax = 1.2 mg/L" is misleading. The honest answer is:
"Cmax = 1.2 mg/L (90% CI: 0.5–3.1 mg/L)."

Monte Carlo approach:
  1. For each uncertain parameter (CLint, fup, CLrenal, ka, Kp),
     sample N values from a log-normal distribution centred on
     the point estimate with published CVs.
  2. Run a full PBPK simulation for each sample.
  3. Report P5, P50, P95 of all PK and risk outputs.

Parameter uncertainty (CV) sources:
  CLint   50%  Shibata et al., Drug Metab Dispos 2002
  fup     25%  Ye et al., J Pharm Sci 2016
  CLrenal 30%  Nair & Jacob, J Basic Clin Pharm 2016
  ka      40%  Thelen et al., Eur J Pharm Biopharm 2011
  Kp      35%  Rodgers & Rowland, J Pharm Sci 2006
─────────────────────────────────────────────────────────────────
"""

import numpy as np
import time

from .admet       import build_drug_profile, REFERENCE_DRUGS
from .physiology  import scale_physiology
from .pbpk_model  import PBPKModel
from .population  import generate_population
from .risk_scorer import (score_single_simulation, score_population,
                           print_risk_report, DEFAULT_THRESHOLDS,
                           ORGAN_DISPLAY)


# ── Parameter uncertainty (coefficient of variation) ──────────────────────
PARAM_CV = {
    "CLint":   0.50,   # Shibata et al. 2002
    "fup":     0.25,   # Ye et al. 2016
    "CLrenal": 0.30,   # Nair & Jacob 2016
    "ka":      0.40,   # Thelen et al. 2011
    "kp_all":  0.35,   # Rodgers & Rowland 2006
}


def _lognormal_sample(mean: float, cv: float, rng,
                      lo: float = 0.0, hi: float = 1e6) -> float:
    """
    Sample from log-normal with given mean and CV.
    Log-normal is correct for positive pharmacokinetic parameters —
    prevents negative values and matches observed right-skewed variability.
    """
    if mean <= 0:
        return max(lo, 0.0)
    sigma_ln = np.sqrt(np.log(1.0 + cv ** 2))
    mu_ln    = np.log(mean) - 0.5 * sigma_ln ** 2
    return float(np.clip(rng.lognormal(mu_ln, sigma_ln), lo, hi))


def _perturb_drug(drug: dict, rng, cv_overrides: dict = None) -> dict:
    """
    Return a copy of the drug profile with parameters perturbed by
    log-normal sampling around the point estimates.
    """
    cv = {**PARAM_CV, **(cv_overrides or {})}
    d      = dict(drug)
    d["kp"]= dict(drug["kp"])

    d["CLint"]   = _lognormal_sample(drug["CLint"],   cv["CLint"],   rng, 0.01)
    d["CLrenal"] = _lognormal_sample(drug["CLrenal"], cv["CLrenal"], rng, 0.0)
    d["ka"]      = _lognormal_sample(drug["ka"],      cv["ka"],      rng, 0.01, 10.0)
    d["fup"]     = _lognormal_sample(drug["fup"],     cv["fup"],     rng, 0.001, 1.0)

    for organ in d["kp"]:
        d["kp"][organ] = _lognormal_sample(
            drug["kp"][organ], cv["kp_all"], rng, 0.01, 100.0
        )
    return d


class Simulator:
    """
    High-level interface for BodySim simulations.

    Methods
    -------
    run_single()            single subject simulation
    run_uncertainty()       Monte Carlo uncertainty quantification  [NEW]
    print_uncertainty_report() formatted CI report                 [NEW]
    run_population()        virtual population simulation
    validate()              compare against literature PK targets
    print_report()          formatted risk report
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

    # ──────────────────────────────────────────────────────────────────────
    # Single simulation
    # ──────────────────────────────────────────────────────────────────────
    def run_single(self, drug, dose_mg, route="oral",
                   subject=None, t_end_h=48.0, n_points=500):
        """
        Run a single-subject PBPK simulation.

        Parameters
        ----------
        drug     : dict   drug profile from admet.build_drug_profile
        dose_mg  : float  administered dose (mg)
        route    : str    'oral' or 'iv'
        subject  : dict   patient dict from population.generate_patient;
                          None → reference 70 kg adult male
        t_end_h  : float  simulation duration (h)
        n_points : int    number of time output points

        Returns
        -------
        dict — t, plasma, organs, auc_plasma, cmax_plasma, tmax_plasma,
               auc_organs, risk, drug, dose_mg, route
        """
        if subject is None:
            volumes, flows, params = scale_physiology()
        else:
            volumes = subject["volumes"]
            flows   = subject["flows"]
            params  = subject["phys_params"]

        drug_subj      = self._adjust_drug_for_subject(drug, params)
        model          = PBPKModel(drug_subj, volumes, flows, params)
        result         = model.solve(dose_mg=dose_mg, route=route,
                                     t_end_h=t_end_h, n_points=n_points)
        result["risk"] = score_single_simulation(result)
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Monte Carlo uncertainty quantification  ← NEW in v1.2
    # ──────────────────────────────────────────────────────────────────────
    def run_uncertainty(self, drug, dose_mg, route="oral",
                        n_samples=200, seed=42,
                        t_end_h=48.0, n_points=200,
                        confidence_level=0.90,
                        cv_overrides=None):
        """
        Quantify prediction uncertainty via Monte Carlo parameter sampling.

        For each sample:
          1. Perturb CLint, fup, CLrenal, ka, and all Kp values by drawing
             from log-normal distributions with published uncertainty CVs.
          2. Run a full PBPK simulation.
          3. Record all PK outputs and risk scores.

        Then compute percentile confidence intervals across all samples.

        Parameters
        ----------
        drug             : dict   drug profile
        dose_mg          : float  dose (mg)
        route            : str    'oral' or 'iv'
        n_samples        : int    Monte Carlo samples (≥100 recommended)
        seed             : int    reproducibility
        t_end_h          : float  simulation duration (h)
        n_points         : int    time resolution per sample
        confidence_level : float  0.90 → 90% CI (P5–P95)
        cv_overrides     : dict   {param: cv} to override defaults
                                  e.g. {"CLint": 0.20} if you have in-vivo data

        Returns
        -------
        dict with keys:
          point_estimate    : simulation at unperturbed parameters
          plasma_ci         : AUC, Cmax, Tmax with P_lo / P50 / P_hi
          organ_ci          : per-organ AUC with confidence intervals
          risk_ci           : per-organ risk score with confidence intervals
          dominant_organ_ci : most at-risk organ + probability across samples
          interpretation    : plain-English summary for researchers
          n_samples, n_ok, n_failed, confidence_level, cv_used
        """
        lo_pct   = (1.0 - confidence_level) / 2.0 * 100
        hi_pct   = (1.0 + confidence_level) / 2.0 * 100
        rng      = np.random.default_rng(seed)
        cv_actual= {**PARAM_CV, **(cv_overrides or {})}

        # Reference physiology (all samples use same subject for parameter CI)
        vol, flow, params = scale_physiology()

        # Point estimate (unperturbed)
        point_result = self.run_single(drug, dose_mg, route,
                                       t_end_h=t_end_h, n_points=n_points)

        if self.verbose:
            print(f"\n[BodySim] Monte Carlo uncertainty analysis")
            print(f"  Drug    : {drug['name']}  {dose_mg}mg  {route}")
            print(f"  Samples : {n_samples}  |  CI: {confidence_level*100:.0f}%")
            print(f"  CVs     : CLint={cv_actual['CLint']*100:.0f}%  "
                  f"fup={cv_actual['fup']*100:.0f}%  "
                  f"Kp={cv_actual['kp_all']*100:.0f}%")

        t0              = time.time()
        plasma_aucs     = []
        plasma_cmaxs    = []
        plasma_tmaxs    = []
        organ_aucs_mc   = {k: [] for k in point_result["auc_organs"]}
        risk_scores_mc  = {k: [] for k in point_result["risk"]["organ_scores"]}
        dominant_counts = {}
        n_failed        = 0

        for i in range(n_samples):
            try:
                drug_p = _perturb_drug(drug, rng, cv_overrides)
                model  = PBPKModel(drug_p, vol, flow, params)
                res    = model.solve(dose_mg, route, t_end_h, n_points)

                plasma_aucs.append(res["auc_plasma"])
                plasma_cmaxs.append(res["cmax_plasma"])
                plasma_tmaxs.append(res["tmax_plasma"])

                for organ, auc in res["auc_organs"].items():
                    if organ in organ_aucs_mc:
                        organ_aucs_mc[organ].append(auc)

                risk = score_single_simulation(res)
                for organ, sc in risk["organ_scores"].items():
                    if organ in risk_scores_mc:
                        risk_scores_mc[organ].append(sc)

                dom = risk["dominant_organ"]
                dominant_counts[dom] = dominant_counts.get(dom, 0) + 1

            except Exception:
                n_failed += 1

        n_ok = n_samples - n_failed

        if self.verbose:
            print(f"  Done: {n_ok}/{n_samples} in {time.time()-t0:.1f}s")
            if n_failed:
                print(f"  ⚠ {n_failed} samples failed ODE solver")

        # ── Compute percentile CIs ─────────────────────────────────────
        def ci(arr):
            if not arr:
                return {"p_lo":0.0,"p50":0.0,"p_hi":0.0,"mean":0.0,"cv":0.0}
            a = np.array(arr)
            m = float(np.mean(a))
            return {
                "p_lo":  float(np.percentile(a, lo_pct)),
                "p50":   float(np.percentile(a, 50.0)),
                "p_hi":  float(np.percentile(a, hi_pct)),
                "mean":  m,
                "cv":    float(np.std(a) / m) if m > 0 else 0.0,
            }

        plasma_ci = {
            "auc":  ci(plasma_aucs),
            "cmax": ci(plasma_cmaxs),
            "tmax": ci(plasma_tmaxs),
        }
        organ_ci = {o: ci(v) for o, v in organ_aucs_mc.items()  if v}
        risk_ci  = {o: ci(v) for o, v in risk_scores_mc.items() if v}

        if dominant_counts and n_ok > 0:
            dom_organ = max(dominant_counts, key=dominant_counts.get)
            dom_prob  = dominant_counts[dom_organ] / n_ok
        else:
            dom_organ, dom_prob = "none", 0.0

        interpretation = self._interpret_uncertainty(
            drug, plasma_ci, risk_ci, dom_organ, dom_prob,
            confidence_level, n_ok, n_samples
        )

        return {
            "point_estimate":    point_result,
            "plasma_ci":         plasma_ci,
            "organ_ci":          organ_ci,
            "risk_ci":           risk_ci,
            "dominant_organ_ci": {
                "organ":    dom_organ,
                "probability": dom_prob,
                "counts":   dominant_counts,
            },
            "n_samples":         n_samples,
            "n_ok":              n_ok,
            "n_failed":          n_failed,
            "confidence_level":  confidence_level,
            "cv_used":           cv_actual,
            "interpretation":    interpretation,
            "drug":              drug,
            "dose_mg":           dose_mg,
            "route":             route,
        }

    def _interpret_uncertainty(self, drug, plasma_ci, risk_ci,
                                dom_organ, dom_prob,
                                ci_level, n_ok, n_samples):
        """Plain-English researcher summary of uncertainty results."""
        ci_pct   = int(ci_level * 100)
        auc      = plasma_ci["auc"]
        cmax     = plasma_ci["cmax"]
        auc_fold = auc["p_hi"]  / auc["p_lo"]  if auc["p_lo"]  > 0 else 999
        cmax_fold= cmax["p_hi"] / cmax["p_lo"] if cmax["p_lo"] > 0 else 999

        if   auc_fold < 3:  conf = "HIGH — suitable for lead optimisation decisions"
        elif auc_fold < 6:  conf = "MODERATE — experimental PK recommended before advancement"
        else:               conf = "LOW — in-vitro or in-vivo PK data required"

        dom_name = ORGAN_DISPLAY.get(dom_organ, dom_organ)
        lines = [
            f"Monte Carlo: {n_ok}/{n_samples} samples  |  {ci_pct}% CI",
            "",
            f"Plasma AUC  : {auc['p50']:.2f} mg·h/L  "
            f"[{ci_pct}% CI: {auc['p_lo']:.2f} – {auc['p_hi']:.2f}]  "
            f"({auc_fold:.1f}× fold range)",
            f"Plasma Cmax : {cmax['p50']:.3f} mg/L  "
            f"[{ci_pct}% CI: {cmax['p_lo']:.3f} – {cmax['p_hi']:.3f}]  "
            f"({cmax_fold:.1f}× fold range)",
            "",
            f"Prediction confidence: {conf}",
            f"Most likely at-risk organ: {dom_name} "
            f"({dom_prob*100:.0f}% of simulations)",
            "",
        ]

        # Hidden tail risks — safe on median, dangerous at P95
        flagged = []
        for organ, ci_v in risk_ci.items():
            if ci_v["p_hi"] >= 0.70 and ci_v["p50"] < 0.45:
                flagged.append(
                    f"  {ORGAN_DISPLAY.get(organ, organ)}: median={ci_v['p50']:.2f} "
                    f"but P95={ci_v['p_hi']:.2f} — monitor in sensitive populations"
                )
        if flagged:
            lines.append("Hidden tail risks (safe on average, dangerous at extremes):")
            lines.extend(flagged)
            lines.append("")

        lines.append(
            "Uncertainty reflects in-vitro → in-vivo extrapolation error. "
            "Experimental PK data will narrow these bounds significantly."
        )
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # Uncertainty report printer  ← NEW in v1.2
    # ──────────────────────────────────────────────────────────────────────
    def print_uncertainty_report(self, unc):
        """Print formatted uncertainty report to stdout."""
        drug   = unc["drug"]
        ci_pct = int(unc["confidence_level"] * 100)
        lop    = int((1 - unc["confidence_level"]) / 2 * 100)
        hip    = int((1 + unc["confidence_level"]) / 2 * 100)
        pe     = unc["point_estimate"]

        print("\n" + "═" * 65)
        print(f"  UNCERTAINTY REPORT — {drug['name']}  "
              f"{unc['dose_mg']}mg {unc['route'].upper()}")
        print(f"  {ci_pct}% CI  |  N={unc['n_ok']} samples")
        print("═" * 65)

        # Plasma PK table
        print(f"\n  {'Metric':<16} {'Point':>8}  "
              f"{'P'+str(lop):>8}  {'P50':>8}  {'P'+str(hip):>8}  Fold")
        print("  " + "─" * 58)
        rows = [
            ("AUC (mg·h/L)",  pe["auc_plasma"],  unc["plasma_ci"]["auc"]),
            ("Cmax (mg/L)",   pe["cmax_plasma"],  unc["plasma_ci"]["cmax"]),
            ("Tmax (h)",      pe["tmax_plasma"],  unc["plasma_ci"]["tmax"]),
        ]
        for label, pt, ci in rows:
            fold = (ci["p_hi"]/ci["p_lo"] if ci["p_lo"] > 0 else 999)
            print(f"  {label:<16} {pt:>8.3f}  "
                  f"{ci['p_lo']:>8.3f}  {ci['p50']:>8.3f}  "
                  f"{ci['p_hi']:>8.3f}  {fold:>4.1f}×")

        # Organ risk CI table
        print(f"\n  {'Organ':<18} {'Point':>6}  "
              f"{'P'+str(lop):>6}  {'P50':>6}  {'P'+str(hip):>6}  Status")
        print("  " + "─" * 60)
        rc    = unc["risk_ci"]
        pt_r  = pe["risk"]["organ_scores"]
        pt_lb = pe["risk"]["organ_labels"]
        status_map = {"green":"Safe","yellow":"Monitor",
                      "orange":"Elevated","red":"HIGH RISK"}

        for organ in sorted(rc, key=lambda o: -rc[o]["p50"]):
            ci   = rc[organ]
            pt   = pt_r.get(organ, 0.0)
            lbl  = pt_lb.get(organ, "green")
            tail = " ⚠" if ci["p_hi"] >= 0.70 and ci["p50"] < 0.45 else ""
            print(f"  {ORGAN_DISPLAY.get(organ,organ):<18} {pt:>6.3f}  "
                  f"{ci['p_lo']:>6.3f}  {ci['p50']:>6.3f}  "
                  f"{ci['p_hi']:>6.3f}  "
                  f"{status_map.get(lbl, lbl)}{tail}")

        # Dominant organ
        dom = unc["dominant_organ_ci"]
        print(f"\n  Most at-risk: "
              f"{ORGAN_DISPLAY.get(dom['organ'], dom['organ'])} "
              f"({dom['probability']*100:.0f}% of simulations)")

        # CVs used
        print(f"\n  CVs used: " +
              "  ".join(f"{k}={v*100:.0f}%"
                         for k,v in unc["cv_used"].items()))

        # Interpretation
        print(f"\n  Interpretation:")
        for line in unc["interpretation"].split("\n"):
            print(f"    {line}")
        print("═" * 65 + "\n")

    # ──────────────────────────────────────────────────────────────────────
    # Population simulation
    # ──────────────────────────────────────────────────────────────────────
    def run_population(self, drug, dose_mg, route="oral",
                       n_subjects=100, seed=42,
                       t_end_h=48.0, n_points=200):
        """
        Run PBPK simulation across a NHANES-calibrated virtual population.

        Parameters
        ----------
        drug       : dict   drug profile
        dose_mg    : float  dose in mg
        route      : str    'oral' or 'iv'
        n_subjects : int    number of virtual patients
        seed       : int    random seed
        t_end_h    : float  simulation duration (h)
        n_points   : int    time resolution

        Returns
        -------
        dict — individual_results, population_risk, population_stats,
               reference_result, drug, dose_mg, route, n_subjects
        """
        if self.verbose:
            print(f"\n[BodySim] Population simulation")
            print(f"  Drug: {drug['name']}  {dose_mg}mg {route}  "
                  f"N={n_subjects}")

        patients   = generate_population(n=n_subjects, seed=seed)
        ref_result = self.run_single(drug, dose_mg, route,
                                     t_end_h=t_end_h, n_points=n_points)
        t0         = time.time()
        pop_results= []

        for i, patient in enumerate(patients):
            try:
                res = self.run_single(
                    drug, dose_mg, route,
                    subject=patient, t_end_h=t_end_h, n_points=n_points,
                )
                res["subject"] = {k: patient[k]
                                  for k in ["age","sex","weight_kg","egfr",
                                             "cyp3a4_activity",
                                             "cyp2d6_phenotype",
                                             "disease_state"]}
                pop_results.append(res)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: subject {i} — {e}")

            if self.verbose and (i+1) % 20 == 0:
                print(f"  {i+1}/{n_subjects}  ({time.time()-t0:.1f}s)")

        if self.verbose:
            print(f"  Done: {len(pop_results)} subjects  "
                  f"({time.time()-t0:.1f}s)")

        from .population import population_summary
        return {
            "individual_results": pop_results,
            "population_risk":    score_population(pop_results),
            "population_stats":   population_summary(patients),
            "reference_result":   ref_result,
            "drug":               drug,
            "dose_mg":            dose_mg,
            "route":              route,
            "n_subjects":         len(pop_results),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────
    def _adjust_drug_for_subject(self, drug, params):
        """Scale clearances to subject's CYP activity and eGFR."""
        d = dict(drug)
        d["kp"]      = dict(drug["kp"])
        d["CLint"]   = drug["CLint"]   * params.get("cyp3a4_activity", 1.0)
        d["CLrenal"] = drug["CLrenal"] * (params.get("egfr", 100.0) / 100.0)
        return d

    def print_report(self, result, pop_result=None):
        """Print formatted single-subject risk report."""
        pop_risk = pop_result["population_risk"] if pop_result else None
        print_risk_report(result, pop_risk)

    def validate(self, result, cmax_target=None, auc_target=None,
                 fold_tolerance=2.0):
        """
        Compare simulated PK against literature targets.

        Parameters
        ----------
        result         : dict   simulation result from run_single()
        cmax_target    : float  observed Cmax (mg/L)
        auc_target     : float  observed AUC (mg·h/L)
        fold_tolerance : float  acceptable fold error (default 2.0 = within 2×)

        Returns
        -------
        dict — pass/fail flags and fold errors for each metric
        """
        checks = {}
        if cmax_target:
            fold = result["cmax_plasma"] / cmax_target
            checks["cmax"] = {
                "predicted": result["cmax_plasma"], "observed": cmax_target,
                "fold_error": fold,
                "pass": 1/fold_tolerance <= fold <= fold_tolerance,
            }
        if auc_target:
            fold = result["auc_plasma"] / auc_target
            checks["auc"] = {
                "predicted": result["auc_plasma"], "observed": auc_target,
                "fold_error": fold,
                "pass": 1/fold_tolerance <= fold <= fold_tolerance,
            }
        overall          = all(c["pass"] for c in checks.values())
        checks["overall_pass"] = overall

        if self.verbose:
            print(f"\n  Validation — {result['drug'].get('name','Drug')}")
            for m, c in checks.items():
                if m == "overall_pass": continue
                flag = "✓ PASS" if c["pass"] else "✗ FAIL"
                print(f"    {m.upper():>6}: pred={c['predicted']:.3f}  "
                      f"obs={c['observed']:.3f}  "
                      f"fold={c['fold_error']:.2f}  {flag}")
            print(f"    Overall: {'PASS ✓' if overall else 'FAIL ✗'}")
        return checks