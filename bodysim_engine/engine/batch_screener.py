"""
batch_screener.py — High-throughput SMILES batch screening for BodySim.

Accepts a CSV of drug candidates (SMILES + optional metadata),
runs ADMET-AI prediction + PBPK simulation + uncertainty analysis
for each compound, and produces a ranked results CSV and summary report.

─────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS
─────────────────────────────────────────────────────────────────────────────
A researcher with 1,000 candidate molecules cannot run full in-vitro
assays on all of them. This screener:

  1. Predicts ADMET properties (ADMET-AI / mock) for each SMILES
  2. Runs PBPK simulation at a standard dose
  3. Scores organ risk across all compartments
  4. Optionally runs Monte Carlo uncertainty (n_mc_samples > 0)
  5. Ranks all compounds by overall safety score
  6. Flags compounds that fail hard thresholds (instant "kill" criteria)
  7. Writes ranked CSV + plain-English summary report

─────────────────────────────────────────────────────────────────────────────
INPUT CSV FORMAT
─────────────────────────────────────────────────────────────────────────────
Required column:
  smiles          SMILES string of the molecule

Optional columns:
  name            display name (default: "Compound_N")
  dose_mg         dose to simulate in mg (default: uses screener default)
  route           'oral' or 'iv' (default: 'oral')
  mw              molecular weight override (g/mol)
  clint           known in-vitro CLint override (L/h per 70kg)
  clrenal         known renal CL override (L/h)

Example input CSV:
  smiles,name,dose_mg
  CN(C)C(=N)NC(=N)N,Metformin,500
  Cn1c(=O)c2c(ncn2C)n(c1=O)C,Caffeine,200
  CC(C)Cc1ccc(cc1)C(C)C(=O)O,Ibuprofen,400

─────────────────────────────────────────────────────────────────────────────
OUTPUT CSV COLUMNS
─────────────────────────────────────────────────────────────────────────────
  rank                overall rank (1 = safest)
  name                compound name
  smiles              input SMILES
  overall_score       0.0–1.0 (0=very safe, 1=very dangerous)
  verdict             PASS / WARN / FAIL / ERROR
  dominant_organ      organ with highest risk score
  plasma_auc          AUC(0-t) mg·h/L
  plasma_cmax         Cmax mg/L
  plasma_tmax         Tmax h
  logp                predicted logP
  fup                 predicted fraction unbound
  clint               predicted/used CLint L/h
  mw                  molecular weight g/mol
  herg_prob           hERG inhibition probability
  dili_prob           DILI risk probability
  bbb_prob            BBB penetration probability
  {organ}_risk        risk score for each organ (0–1)
  {organ}_auc         AUC for each organ
  auc_ci_lo           90% CI lower bound on plasma AUC (if MC enabled)
  auc_ci_hi           90% CI upper bound on plasma AUC (if MC enabled)
  confidence          prediction confidence string
  predictor           admet_ai or mock_chemistry_rules
  error               error message if simulation failed

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
Python API:
  from engine.batch_screener import BatchScreener

  screener = BatchScreener(default_dose_mg=500, default_route='oral')
  results  = screener.run_csv("candidates.csv")
  screener.save_results(results, "output/")

Command line:
  python -m engine.batch_screener \\
      --input candidates.csv \\
      --output results/ \\
      --dose 500 \\
      --route oral \\
      --mc-samples 100

─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import csv
import time
import warnings
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

# ── BodySim imports ───────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from engine.simulator            import Simulator
from engine.admet                import build_drug_profile
from engine.physiology           import scale_physiology
from engine.pbpk_model           import PBPKModel
from engine.risk_scorer          import (score_single_simulation,
                                         ORGAN_DISPLAY, DEFAULT_THRESHOLDS)
from engine.ml.chemprop_predictor import (ChemPropPredictor,
                                           smiles_to_drug_profile,
                                           get_predictor)

# ── Constants ─────────────────────────────────────────────────────────────
ORGANS = ["liver", "kidney", "brain", "heart", "muscle",
          "fat", "gut", "skin", "bone", "lung", "rest"]

# Verdict thresholds (based on overall_score)
VERDICT_THRESHOLDS = {
    "PASS": 0.35,    # overall_score < 0.35 → safe to advance
    "WARN": 0.60,    # 0.35–0.60 → needs investigation
    "FAIL": 1.01,    # > 0.60 → do not advance
}

# Hard-kill criteria — instant FAIL regardless of overall score
HARD_KILL = {
    "herg_prob":   0.80,   # >80% hERG inhibition → cardiac risk unacceptable
    "dili_prob":   0.85,   # >85% DILI probability → liver toxicity unacceptable
    "kidney_risk": 0.90,   # kidney score >0.90
    "liver_risk":  0.90,   # liver score >0.90
}


# ═══════════════════════════════════════════════════════════════════════════
# Single compound result
# ═══════════════════════════════════════════════════════════════════════════

class CompoundResult:
    """
    Holds all screening results for a single compound.
    Designed to be easily serialised to CSV.
    """

    def __init__(self, name: str, smiles: str, dose_mg: float, route: str):
        self.name      = name
        self.smiles    = smiles
        self.dose_mg   = dose_mg
        self.route     = route
        self.rank      = None

        # Filled by screener
        self.overall_score    = None
        self.verdict          = "ERROR"
        self.dominant_organ   = None
        self.hard_kill_reason = None

        # PK
        self.plasma_auc   = None
        self.plasma_cmax  = None
        self.plasma_tmax  = None

        # ADMET predictions
        self.logp       = None
        self.fup        = None
        self.clint      = None
        self.mw         = None
        self.herg_prob  = None
        self.dili_prob  = None
        self.bbb_prob   = None
        self.confidence = None
        self.predictor  = None

        # Per-organ
        self.organ_risk = {}   # {organ: score}
        self.organ_auc  = {}   # {organ: auc}

        # Uncertainty (optional)
        self.auc_ci_lo  = None
        self.auc_ci_hi  = None
        self.cmax_ci_lo = None
        self.cmax_ci_hi = None

        self.error      = None
        self.runtime_s  = None

    def to_dict(self) -> dict:
        """Serialise to flat dict for CSV writing."""
        d = {
            "rank":           self.rank,
            "name":           self.name,
            "smiles":         self.smiles,
            "dose_mg":        self.dose_mg,
            "route":          self.route,
            "overall_score":  _fmt(self.overall_score),
            "verdict":        self.verdict,
            "hard_kill":      self.hard_kill_reason or "",
            "dominant_organ": ORGAN_DISPLAY.get(self.dominant_organ or "",
                                                 self.dominant_organ or ""),
            "plasma_auc":     _fmt(self.plasma_auc),
            "plasma_cmax":    _fmt(self.plasma_cmax),
            "plasma_tmax":    _fmt(self.plasma_tmax),
            "logp":           _fmt(self.logp),
            "fup":            _fmt(self.fup),
            "clint":          _fmt(self.clint),
            "mw":             _fmt(self.mw),
            "herg_prob":      _fmt(self.herg_prob),
            "dili_prob":      _fmt(self.dili_prob),
            "bbb_prob":       _fmt(self.bbb_prob),
        }
        # Organ risk scores
        for organ in ORGANS:
            d[f"{organ}_risk"] = _fmt(self.organ_risk.get(organ))
            d[f"{organ}_auc"]  = _fmt(self.organ_auc.get(organ))

        # Uncertainty CI
        d["auc_ci_lo"]  = _fmt(self.auc_ci_lo)
        d["auc_ci_hi"]  = _fmt(self.auc_ci_hi)
        d["cmax_ci_lo"] = _fmt(self.cmax_ci_lo)
        d["cmax_ci_hi"] = _fmt(self.cmax_ci_hi)

        d["confidence"] = self.confidence or ""
        d["predictor"]  = self.predictor  or ""
        d["runtime_s"]  = _fmt(self.runtime_s)
        d["error"]      = self.error or ""
        return d


def _fmt(v, decimals: int = 4) -> str:
    """Format a value for CSV output."""
    if v is None: return ""
    if isinstance(v, float): return f"{v:.{decimals}f}"
    return str(v)


# ═══════════════════════════════════════════════════════════════════════════
# BatchScreener
# ═══════════════════════════════════════════════════════════════════════════

class BatchScreener:
    """
    High-throughput SMILES batch screener.

    Parameters
    ----------
    default_dose_mg  : float  dose used when not specified per-compound (mg)
    default_route    : str    'oral' or 'iv'
    n_mc_samples     : int    Monte Carlo samples per compound (0 = skip)
    mc_confidence    : float  CI level for Monte Carlo (0.90 = 90%)
    verbose          : bool   print progress per compound
    fail_fast        : bool   skip MC if hard-kill triggered on point estimate
    subject          : dict   optional fixed subject (None = reference 70kg male)
    """

    def __init__(self,
                 default_dose_mg: float = 500.0,
                 default_route:   str   = "oral",
                 n_mc_samples:    int   = 0,
                 mc_confidence:   float = 0.90,
                 verbose:         bool  = True,
                 fail_fast:       bool  = True,
                 subject:         Optional[dict] = None):

        self.default_dose_mg = default_dose_mg
        self.default_route   = default_route
        self.n_mc_samples    = n_mc_samples
        self.mc_confidence   = mc_confidence
        self.verbose         = verbose
        self.fail_fast       = fail_fast
        self.subject         = subject

        # Shared physiology (reference subject unless overridden)
        if subject:
            self._vol    = subject["volumes"]
            self._flow   = subject["flows"]
            self._params = subject["phys_params"]
        else:
            self._vol, self._flow, self._params = scale_physiology()

        # Shared ADMET-AI predictor (loaded once)
        if self.verbose:
            print("[BatchScreener] Loading ADMET predictor …")
        self._predictor = get_predictor(verbose=self.verbose)

        # Simulator (for Monte Carlo)
        self._sim = Simulator(verbose=False)

        if self.verbose:
            mc_str = (f"MC={n_mc_samples} samples" if n_mc_samples > 0
                      else "MC disabled")
            print(f"[BatchScreener] Ready — dose={default_dose_mg}mg "
                  f"{default_route}  {mc_str}")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def screen_smiles(self, smiles: str, name: str = "Compound",
                      dose_mg: Optional[float] = None,
                      route:   Optional[str]   = None,
                      clint_override:   Optional[float] = None,
                      clrenal_override: Optional[float] = None) -> CompoundResult:
        """
        Screen a single SMILES string.

        Parameters
        ----------
        smiles           : str   SMILES string
        name             : str   display name
        dose_mg          : float dose (mg) — None → use screener default
        route            : str   'oral'|'iv' — None → use screener default
        clint_override   : float known in-vitro CLint (L/h) — overrides ADMET-AI
        clrenal_override : float known renal CL (L/h) — overrides ADMET-AI

        Returns
        -------
        CompoundResult
        """
        dose  = dose_mg if dose_mg  is not None else self.default_dose_mg
        route = route   if route    is not None else self.default_route
        res   = CompoundResult(name, smiles, dose, route)
        t0    = time.time()

        try:
            # ── Step 1: ADMET prediction ───────────────────────────────────
            drug = smiles_to_drug_profile(
                smiles          = smiles,
                name            = name,
                predictor       = self._predictor,
                clrenal_override= clrenal_override,
            )
            if clint_override is not None:
                drug["CLint"] = clint_override

            # Attach ADMET metadata to result
            res.logp       = drug.get("logp")
            res.fup        = drug.get("fup")
            res.clint      = drug.get("CLint")
            res.mw         = drug.get("mw")
            res.herg_prob  = drug.get("herg_prob",  0.0)
            res.dili_prob  = drug.get("dili_prob",  0.0)
            res.bbb_prob   = drug.get("bbb_prob",   0.5)
            res.confidence = drug.get("confidence", "")
            res.predictor  = drug.get("predictor",  "")

            # ── Step 2: Point-estimate PBPK simulation ─────────────────────
            sim_result = self._run_pbpk(drug, dose, route)
            risk       = score_single_simulation(sim_result)

            res.plasma_auc   = sim_result["auc_plasma"]
            res.plasma_cmax  = sim_result["cmax_plasma"]
            res.plasma_tmax  = sim_result["tmax_plasma"]
            res.organ_risk   = dict(risk["organ_scores"])
            res.organ_auc    = dict(sim_result["auc_organs"])
            res.dominant_organ = risk["dominant_organ"]

            # ── Step 3: Hard-kill check ────────────────────────────────────
            hk_reason = self._check_hard_kill(res)
            if hk_reason:
                res.hard_kill_reason = hk_reason
                res.overall_score    = 1.0
                res.verdict          = "FAIL"
                # Skip MC if fail_fast
                if self.fail_fast and self.n_mc_samples > 0:
                    if self.verbose:
                        print(f"    ✗ Hard kill: {hk_reason} — skipping MC")
                    res.runtime_s = time.time() - t0
                    return res

            # ── Step 4: Overall score ──────────────────────────────────────
            res.overall_score = self._compute_overall_score(res)
            res.verdict       = self._compute_verdict(res)

            # ── Step 5: Monte Carlo uncertainty (optional) ─────────────────
            if self.n_mc_samples > 0:
                mc = self._sim.run_uncertainty(
                    drug             = drug,
                    dose_mg          = dose,
                    route            = route,
                    n_samples        = self.n_mc_samples,
                    confidence_level = self.mc_confidence,
                    seed             = abs(hash(smiles)) % (2**31),
                )
                auc_ci  = mc["plasma_ci"]["auc"]
                cmax_ci = mc["plasma_ci"]["cmax"]
                res.auc_ci_lo  = auc_ci["p_lo"]
                res.auc_ci_hi  = auc_ci["p_hi"]
                res.cmax_ci_lo = cmax_ci["p_lo"]
                res.cmax_ci_hi = cmax_ci["p_hi"]

                # Upgrade verdict if P95 risk is high even when median is OK
                res.verdict = self._upgrade_verdict_from_ci(res, mc)

        except Exception as exc:
            res.error        = str(exc)
            res.verdict      = "ERROR"
            res.overall_score = 1.0

        res.runtime_s = round(time.time() - t0, 2)
        return res

    def run_list(self, compounds: list[dict]) -> list[CompoundResult]:
        """
        Screen a list of compound dicts.

        Each dict must have 'smiles' and optionally:
        'name', 'dose_mg', 'route', 'clint', 'clrenal'

        Parameters
        ----------
        compounds : list of dicts

        Returns
        -------
        list of CompoundResult, ranked by overall_score ascending
        """
        n      = len(compounds)
        t0     = time.time()
        results = []

        if self.verbose:
            print(f"\n[BatchScreener] Screening {n} compounds …")
            print(f"  {'#':>4}  {'Name':<20}  {'Verdict':<6}  "
                  f"{'Score':>6}  {'Dominant Organ':<16}  Time")
            print("  " + "─" * 72)

        for i, comp in enumerate(compounds):
            smiles = comp.get("smiles", "").strip()
            if not smiles:
                continue

            name     = comp.get("name",     f"Compound_{i+1:03d}")
            dose_mg  = _safe_float(comp.get("dose_mg"),  None)
            route    = comp.get("route",    None)
            clint    = _safe_float(comp.get("clint"),    None)
            clrenal  = _safe_float(comp.get("clrenal"),  None)

            res = self.screen_smiles(
                smiles           = smiles,
                name             = name,
                dose_mg          = dose_mg,
                route            = route,
                clint_override   = clint,
                clrenal_override = clrenal,
            )
            results.append(res)

            if self.verbose:
                verdict_icon = {"PASS":"✓","WARN":"⚠","FAIL":"✗","ERROR":"!"
                                }.get(res.verdict, "?")
                dom = ORGAN_DISPLAY.get(res.dominant_organ or "", "—")
                score_str = f"{res.overall_score:.3f}" if res.overall_score is not None else "—"
                print(f"  {i+1:>4}  {name:<20}  "
                      f"{verdict_icon} {res.verdict:<4}  "
                      f"{score_str:>6}  {dom:<16}  {res.runtime_s:.1f}s")

        # Rank by overall_score (None/ERROR → worst)
        results.sort(key=lambda r: r.overall_score if r.overall_score is not None else 2.0)
        for rank, r in enumerate(results, 1):
            r.rank = rank

        elapsed = time.time() - t0
        if self.verbose:
            self._print_batch_summary(results, elapsed)

        return results

    def run_csv(self, csv_path: str) -> list[CompoundResult]:
        """
        Screen all compounds from a CSV file.

        Parameters
        ----------
        csv_path : str   path to input CSV

        Returns
        -------
        list of CompoundResult, ranked by overall_score
        """
        compounds = self._load_csv(csv_path)
        if self.verbose:
            print(f"[BatchScreener] Loaded {len(compounds)} compounds "
                  f"from {csv_path}")
        return self.run_list(compounds)

    def save_results(self, results: list[CompoundResult],
                     output_dir: str = ".",
                     prefix: str     = "bodysim_screen") -> dict:
        """
        Save results to CSV and plain-text summary report.

        Parameters
        ----------
        results    : list of CompoundResult
        output_dir : str   directory for output files
        prefix     : str   filename prefix

        Returns
        -------
        dict with paths: {'csv': ..., 'report': ...}
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        csv_path    = out / f"{prefix}_ranked.csv"
        report_path = out / f"{prefix}_report.txt"

        # ── Ranked CSV ────────────────────────────────────────────────────
        if results:
            fieldnames = list(results[0].to_dict().keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(r.to_dict())

        # ── Plain-text summary report ─────────────────────────────────────
        with open(report_path, "w") as f:
            f.write(self._build_report(results))

        if self.verbose:
            print(f"\n[BatchScreener] Saved:")
            print(f"  Ranked CSV : {csv_path}")
            print(f"  Report     : {report_path}")

        return {"csv": str(csv_path), "report": str(report_path)}

    # ──────────────────────────────────────────────────────────────────────
    # Internal methods
    # ──────────────────────────────────────────────────────────────────────

    def _run_pbpk(self, drug: dict, dose_mg: float, route: str) -> dict:
        """Run PBPK ODE with fixed reference physiology."""
        # Scale clearances to reference subject
        drug_adj            = dict(drug)
        drug_adj["kp"]      = dict(drug["kp"])
        drug_adj["CLint"]   = drug["CLint"]   * self._params.get("cyp3a4_activity", 1.0)
        drug_adj["CLrenal"] = drug["CLrenal"] * (self._params.get("egfr", 100.0) / 100.0)

        model = PBPKModel(drug_adj, self._vol, self._flow, self._params)
        return model.solve(dose_mg=dose_mg, route=route,
                           t_end_h=48.0, n_points=200)

    def _check_hard_kill(self, res: CompoundResult) -> Optional[str]:
        """
        Check hard-kill criteria. Returns reason string or None.
        Instant FAIL — no further processing needed.
        """
        if res.herg_prob and res.herg_prob >= HARD_KILL["herg_prob"]:
            return f"hERG inhibition {res.herg_prob:.0%} ≥ {HARD_KILL['herg_prob']:.0%}"

        if res.dili_prob and res.dili_prob >= HARD_KILL["dili_prob"]:
            return f"DILI probability {res.dili_prob:.0%} ≥ {HARD_KILL['dili_prob']:.0%}"

        kidney_risk = res.organ_risk.get("kidney", 0.0)
        if kidney_risk >= HARD_KILL["kidney_risk"]:
            return f"Kidney risk {kidney_risk:.2f} ≥ {HARD_KILL['kidney_risk']:.2f}"

        liver_risk = res.organ_risk.get("liver", 0.0)
        if liver_risk >= HARD_KILL["liver_risk"]:
            return f"Liver risk {liver_risk:.2f} ≥ {HARD_KILL['liver_risk']:.2f}"

        return None

    def _compute_overall_score(self, res: CompoundResult) -> float:
        """
        Compute an overall safety score [0–1].

        Weighted average of organ risk scores with safety-signal boosts:
          - Base: weighted organ risks (liver and kidney weighted higher)
          - Boost: hERG and DILI signals add directly to score
          - Result is clipped to [0, 1]

        0.0 = perfectly safe across all organs
        1.0 = maximum danger
        """
        weights = {
            "liver":   2.0,   # major metabolic organ — weight double
            "kidney":  2.0,   # major excretory organ
            "heart":   1.8,   # hERG cardiac risk
            "brain":   1.5,   # CNS toxicity serious
            "lung":    1.2,
            "gut":     1.0,
            "muscle":  0.6,
            "fat":     0.4,
            "skin":    0.4,
            "bone":    0.3,
            "rest":    0.3,
        }
        total_w = sum(weights.values())
        score   = sum(
            weights.get(organ, 0.5) * res.organ_risk.get(organ, 0.0)
            for organ in weights
        ) / total_w

        # ADMET-AI safety signal boosts
        if res.herg_prob:  score += 0.15 * res.herg_prob
        if res.dili_prob:  score += 0.10 * res.dili_prob

        return float(np.clip(score, 0.0, 1.0))

    def _compute_verdict(self, res: CompoundResult) -> str:
        if res.overall_score is None:
            return "ERROR"
        if res.overall_score < VERDICT_THRESHOLDS["PASS"]:
            return "PASS"
        if res.overall_score < VERDICT_THRESHOLDS["WARN"]:
            return "WARN"
        return "FAIL"

    def _upgrade_verdict_from_ci(self, res: CompoundResult, mc: dict) -> str:
        """
        If compound PASses on point estimate but P95 risk is dangerous,
        upgrade to WARN. This catches hidden tail risks.
        """
        if res.verdict != "PASS":
            return res.verdict

        risk_ci = mc.get("risk_ci", {})
        for organ, ci in risk_ci.items():
            if ci.get("p_hi", 0) >= 0.70 and ci.get("p50", 0) < 0.45:
                # Safe on median but dangerous at P95 — flag it
                return "WARN"
        return res.verdict

    @staticmethod
    def _load_csv(csv_path: str) -> list[dict]:
        """Load compound list from CSV file."""
        compounds = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # Normalise column names to lowercase stripped
            for row in reader:
                clean = {k.strip().lower(): v.strip() for k, v in row.items()}
                if clean.get("smiles"):
                    compounds.append(clean)
        return compounds

    @staticmethod
    def _print_batch_summary(results: list[CompoundResult], elapsed: float):
        """Print a summary table after all compounds are screened."""
        n_total = len(results)
        n_pass  = sum(1 for r in results if r.verdict == "PASS")
        n_warn  = sum(1 for r in results if r.verdict == "WARN")
        n_fail  = sum(1 for r in results if r.verdict == "FAIL")
        n_err   = sum(1 for r in results if r.verdict == "ERROR")

        print(f"\n{'─'*72}")
        print(f"  BATCH SUMMARY — {n_total} compounds  ({elapsed:.1f}s total)")
        print(f"{'─'*72}")
        print(f"  ✓ PASS  : {n_pass:>4}  ({100*n_pass/max(n_total,1):.0f}%)")
        print(f"  ⚠ WARN  : {n_warn:>4}  ({100*n_warn/max(n_total,1):.0f}%)")
        print(f"  ✗ FAIL  : {n_fail:>4}  ({100*n_fail/max(n_total,1):.0f}%)")
        if n_err:
            print(f"  ! ERROR : {n_err:>4}")
        print(f"{'─'*72}")

        # Top 5 safest
        passed = [r for r in results if r.verdict in ("PASS","WARN")]
        if passed:
            print(f"\n  Top {min(5,len(passed))} Safest Compounds:")
            for r in passed[:5]:
                dom = ORGAN_DISPLAY.get(r.dominant_organ or "", "—")
                print(f"    #{r.rank:<3} {r.name:<22} score={r.overall_score:.3f}"
                      f"  dominant={dom}")

        # Hard-kill failures
        killed = [r for r in results if r.hard_kill_reason]
        if killed:
            print(f"\n  Hard-Kill Failures ({len(killed)}):")
            for r in killed[:5]:
                print(f"    {r.name:<22} {r.hard_kill_reason}")

    def _build_report(self, results: list[CompoundResult]) -> str:
        """Build plain-text summary report for researchers."""
        lines = []
        lines.append("=" * 70)
        lines.append("  BODYSIM BATCH SCREENING REPORT")
        lines.append(f"  {len(results)} compounds screened")
        lines.append(f"  Default dose: {self.default_dose_mg} mg  ({self.default_route})")
        lines.append("=" * 70)

        n_pass = sum(1 for r in results if r.verdict == "PASS")
        n_warn = sum(1 for r in results if r.verdict == "WARN")
        n_fail = sum(1 for r in results if r.verdict == "FAIL")
        n_err  = sum(1 for r in results if r.verdict == "ERROR")

        lines.append(f"\nSUMMARY")
        lines.append(f"  PASS  : {n_pass}")
        lines.append(f"  WARN  : {n_warn}")
        lines.append(f"  FAIL  : {n_fail}")
        if n_err:
            lines.append(f"  ERROR : {n_err}")

        lines.append(f"\n{'─'*70}")
        lines.append("RANKED RESULTS (best → worst)")
        lines.append(f"{'─'*70}")
        lines.append(f"{'Rank':<5} {'Name':<22} {'Verdict':<7} {'Score':>6}"
                     f"  {'AUC':>7}  {'Cmax':>6}  Dominant Organ")
        lines.append("─" * 70)

        for r in results:
            dom   = ORGAN_DISPLAY.get(r.dominant_organ or "", "—")
            score = f"{r.overall_score:.3f}" if r.overall_score is not None else "ERROR"
            auc   = f"{r.plasma_auc:.2f}"    if r.plasma_auc   is not None else "—"
            cmax  = f"{r.plasma_cmax:.3f}"   if r.plasma_cmax  is not None else "—"
            lines.append(f"{r.rank:<5} {r.name:<22} {r.verdict:<7} {score:>6}"
                         f"  {auc:>7}  {cmax:>6}  {dom}")
            if r.hard_kill_reason:
                lines.append(f"       ↳ Hard kill: {r.hard_kill_reason}")
            if r.auc_ci_lo is not None:
                lines.append(f"       ↳ AUC 90% CI: {r.auc_ci_lo:.2f} – {r.auc_ci_hi:.2f} mg·h/L")

        lines.append(f"\n{'─'*70}")
        lines.append("HARD-KILL CRITERIA USED")
        lines.append(f"  hERG inhibition  ≥ {HARD_KILL['herg_prob']:.0%}")
        lines.append(f"  DILI probability ≥ {HARD_KILL['dili_prob']:.0%}")
        lines.append(f"  Kidney risk      ≥ {HARD_KILL['kidney_risk']:.2f}")
        lines.append(f"  Liver risk       ≥ {HARD_KILL['liver_risk']:.2f}")
        lines.append(f"\nVERDICT THRESHOLDS")
        lines.append(f"  PASS  : overall score < {VERDICT_THRESHOLDS['PASS']}")
        lines.append(f"  WARN  : overall score < {VERDICT_THRESHOLDS['WARN']}")
        lines.append(f"  FAIL  : overall score ≥ {VERDICT_THRESHOLDS['WARN']}")
        lines.append(f"\nPREDICTOR: {results[0].predictor if results else 'unknown'}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _safe_float(val, default):
    try:
        return float(val) if val not in (None, "", "nan") else default
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def _cli():
    parser = argparse.ArgumentParser(
        description="BodySim batch SMILES screener"
    )
    parser.add_argument("--input",      required=True,
                        help="Input CSV with 'smiles' column")
    parser.add_argument("--output",     default="bodysim_results",
                        help="Output directory (default: bodysim_results)")
    parser.add_argument("--dose",       type=float, default=500.0,
                        help="Default dose in mg (default: 500)")
    parser.add_argument("--route",      default="oral",
                        choices=["oral","iv"],
                        help="Default route (default: oral)")
    parser.add_argument("--mc-samples", type=int, default=0,
                        help="Monte Carlo samples per compound (0=skip)")
    parser.add_argument("--mc-ci",      type=float, default=0.90,
                        help="CI level for MC (default: 0.90)")
    parser.add_argument("--prefix",     default="bodysim_screen",
                        help="Output filename prefix")
    parser.add_argument("--quiet",      action="store_true",
                        help="Suppress per-compound output")
    args = parser.parse_args()

    screener = BatchScreener(
        default_dose_mg = args.dose,
        default_route   = args.route,
        n_mc_samples    = args.mc_samples,
        mc_confidence   = args.mc_ci,
        verbose         = not args.quiet,
    )
    results = screener.run_csv(args.input)
    screener.save_results(results, args.output, args.prefix)


# ═══════════════════════════════════════════════════════════════════════════
# Self-test (run with: python -m engine.batch_screener)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--input" in sys.argv:
        _cli()
        sys.exit(0)

    print("BodySim BatchScreener — Self Test")
    print("=" * 55)

    # 8 test compounds: 4 known + 4 fictional candidates
    test_compounds = [
        {"smiles": "CN(C)C(=N)NC(=N)N",               "name": "Metformin",   "dose_mg": "500"},
        {"smiles": "Cn1c(=O)c2c(ncn2C)n(c1=O)C",      "name": "Caffeine",    "dose_mg": "200"},
        {"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",      "name": "Ibuprofen",   "dose_mg": "400"},
        {"smiles": "CC(=O)Oc1ccccc1C(=O)O",            "name": "Aspirin",     "dose_mg": "500"},
        {"smiles": "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34", "name": "Pyrene",      "dose_mg": "100"},
        {"smiles": "OC(=O)c1ccccc1",                   "name": "Benzoic_Acid","dose_mg": "200"},
        {"smiles": "CCO",                               "name": "Ethanol",     "dose_mg": "100"},
        {"smiles": "C(=O)(N)c1ccc(cc1)Cl",             "name": "Cand_008",    "dose_mg": "300"},
    ]

    screener = BatchScreener(
        default_dose_mg = 500,
        default_route   = "oral",
        n_mc_samples    = 0,     # skip MC for fast self-test
        verbose         = True,
    )
    results = screener.run_list(test_compounds)

    # Save to outputs directory
    out_dir = "/mnt/user-data/outputs/batch_test"
    paths   = screener.save_results(results, out_dir, "selftest")
    print(f"\nReport saved to: {paths['report']}")
    print(f"CSV saved to   : {paths['csv']}")
