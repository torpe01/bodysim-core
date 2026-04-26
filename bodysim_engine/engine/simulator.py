from .ml import smiles_to_drug_profile

"""
simulator.py — Main BodySim simulation orchestrator.

Ties together:
  - ADMET property estimation
  - PBPK ODE model
  - Virtual population generation
  - Risk scoring

Usage
-----
from engine.simulator import Simulator
from engine.admet import REFERENCE_DRUGS

sim = Simulator()

# Single subject simulation
result = sim.run_single(
    drug=REFERENCE_DRUGS["metformin"],
    dose_mg=500,
    route="oral",
)

# Population simulation (100 virtual patients)
pop = sim.run_population(
    drug=REFERENCE_DRUGS["metformin"],
    dose_mg=500,
    route="oral",
    n_subjects=100,
)
"""

import numpy as np
import time

from .admet       import build_drug_profile, REFERENCE_DRUGS
from .physiology  import scale_physiology, REFERENCE_HUMAN
from .pbpk_model  import PBPKModel
from .population  import generate_population
from .risk_scorer import score_single_simulation, score_population, print_risk_report


class Simulator:
    """
    High-level interface for BodySim simulations.

    Parameters
    ----------
    verbose : bool   print progress messages (default True)
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Single-subject simulation
    # ------------------------------------------------------------------
    def run_single(self, drug=None, dose_mg=500, route="oral",
                   smiles=None, name="Unknown Drug",
                   subject=None, t_end_h=48.0, n_points=500):
        """
        Run a single-subject PBPK simulation.

        Parameters
        ----------
        drug     : dict   drug profile (from admet.build_drug_profile or REFERENCE_DRUGS)
        dose_mg  : float  administered dose in mg
        route    : str    'oral' or 'iv'
        subject  : dict   optional subject dict (from population.generate_patient)
                          If None, uses reference 70 kg adult male.
        t_end_h  : float  simulation duration in hours
        n_points : int    time resolution

        Returns
        -------
        dict with simulation result + risk scores
        """
        if smiles is not None:
            if self.verbose:
                print(f"[BodySim] AI analyzing molecule: {smiles}")
            drug = smiles_to_drug_profile(smiles, name=name)
        
        if drug is None:
            raise ValueError("You must provide either a drug profile or a SMILES string.")
        
        if subject is None:
            volumes, flows, params = scale_physiology()
        else:
            volumes = subject["volumes"]
            flows   = subject["flows"]
            params  = subject["phys_params"]

        # Update drug clearance for this subject's CYP activity
        drug_subj = self._adjust_drug_for_subject(drug, params)

        model = PBPKModel(drug_subj, volumes, flows, params)
        result = model.solve(dose_mg=dose_mg, route=route,
                             t_end_h=t_end_h, n_points=n_points)

        risk = score_single_simulation(result)
        result["risk"] = risk

        return result

    # ------------------------------------------------------------------
    # Population simulation
    # ------------------------------------------------------------------
    def run_population(self, drug=None, dose_mg=500, route="oral",
                       smiles=None, name="Unknown Drug",
                       n_subjects=100, seed=42,
                       t_end_h=48.0, n_points=200,
                       logp=None, fup=None, clint_override=None, clrenal_override=None):
        """
        Run a PBPK simulation across a virtual population.

        Parameters
        ----------
        drug       : dict   drug profile
        dose_mg    : float  dose in mg (same for all subjects — can extend to
                            weight-based dosing)
        route      : str    'oral' or 'iv'
        n_subjects : int    number of virtual patients
        seed       : int    random seed for reproducibility
        t_end_h    : float  simulation duration
        n_points   : int    time resolution (lower = faster)

        Returns
        -------
        dict with:
            'individual_results'  : list of result dicts
            'population_risk'     : aggregated risk statistics
            'population_stats'    : descriptive stats of the cohort
            'reference_result'    : result for reference 70 kg male
        """
        if smiles is not None:
            if self.verbose:
                print(f"[BodySim] AI analyzing molecule: {smiles}")
            from .ml import smiles_to_drug_profile
            # Pass the overrides into the profile builder
            drug = smiles_to_drug_profile(
                smiles, name=name, 
                clrenal_override=clrenal_override
            )
            # Apply manual overrides if provided
            if logp is not None: drug["logp"] = logp
            if fup is not None: drug["fup"] = fup
            if clint_override is not None: drug["CLint"] = clint_override
            
        if drug is None:
            raise ValueError("You must provide either a drug profile or a SMILES string.")

        # Generate virtual population
        patients = generate_population(n=n_subjects, seed=seed)

        # Reference subject (70 kg male)
        ref_result = self.run_single(drug, dose_mg, route, t_end_h=t_end_h,
                                     n_points=n_points)

        t0 = time.time()
        pop_results = []

        for i, patient in enumerate(patients):
            try:
                res = self.run_single(
                    drug, dose_mg, route,
                    subject=patient,
                    t_end_h=t_end_h,
                    n_points=n_points,
                )
                # Attach subject demographics to result
                res["subject"] = {
                    k: patient[k]
                    for k in ["age","sex","weight_kg","egfr",
                               "cyp3a4_activity","cyp2d6_phenotype",
                               "disease_state"]
                }
                pop_results.append(res)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: subject {i} failed — {e}")

            if self.verbose and (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  Completed {i+1}/{n_subjects} subjects "
                      f"({elapsed:.1f}s elapsed)")

        elapsed = time.time() - t0
        if self.verbose:
            print(f"  Done. {len(pop_results)} subjects completed in {elapsed:.1f}s")

        # Aggregate risk
        pop_risk = score_population(pop_results)

        # Descriptive stats of cohort
        from .population import population_summary
        pop_stats = population_summary(patients)

        return {
            "individual_results": pop_results,
            "population_risk":    pop_risk,
            "population_stats":   pop_stats,
            "reference_result":   ref_result,
            "drug":               drug,
            "dose_mg":            dose_mg,
            "route":              route,
            "n_subjects":         len(pop_results),
        }

    # ------------------------------------------------------------------
    # Drug adjustment for subject-specific enzyme activity
    # ------------------------------------------------------------------
    def _adjust_drug_for_subject(self, drug, params):
        """
        Return a copy of the drug profile with CLint adjusted for
        the subject's CYP3A4 activity.
        """
        drug_adj = dict(drug)   # shallow copy
        drug_adj["kp"] = dict(drug["kp"])  # copy kp dict too

        cyp_factor = params.get("cyp3a4_activity", 1.0)
        egfr_factor = params.get("egfr", 100.0) / 100.0   # normalised

        # Scale hepatic CL by CYP activity
        drug_adj["CLint"]   = drug["CLint"]   * cyp_factor

        # Scale renal CL by eGFR (normalised to 100 mL/min reference)
        drug_adj["CLrenal"] = drug["CLrenal"] * egfr_factor

        return drug_adj

    # ------------------------------------------------------------------
    # Print helpers
    # ------------------------------------------------------------------
    def print_report(self, result, pop_result=None):
        """Print formatted risk report to stdout."""
        pop_risk = pop_result["population_risk"] if pop_result else None
        print_risk_report(result, pop_risk)

    # ------------------------------------------------------------------
    # Quick validation: compare plasma Cmax and AUC against targets
    # ------------------------------------------------------------------
    def validate(self, result, cmax_target=None, auc_target=None,
                 fold_tolerance=2.0):
        """
        Simple validation: check if simulated PK is within fold_tolerance
        of observed targets.

        Parameters
        ----------
        result          : dict  simulation result
        cmax_target     : float expected Cmax (mg/L) from literature
        auc_target      : float expected AUC (mg·h/L) from literature
        fold_tolerance  : float acceptable fold error (default 2.0 = within 2x)

        Returns
        -------
        dict with pass/fail and fold errors
        """
        checks = {}

        if cmax_target:
            fold_cmax = result["cmax_plasma"] / cmax_target
            checks["cmax"] = {
                "predicted": result["cmax_plasma"],
                "observed":  cmax_target,
                "fold_error": fold_cmax,
                "pass": 1/fold_tolerance <= fold_cmax <= fold_tolerance,
            }

        if auc_target:
            fold_auc = result["auc_plasma"] / auc_target
            checks["auc"] = {
                "predicted": result["auc_plasma"],
                "observed":  auc_target,
                "fold_error": fold_auc,
                "pass": 1/fold_tolerance <= fold_auc <= fold_tolerance,
            }

        overall = all(c["pass"] for c in checks.values())
        checks["overall_pass"] = overall

        if self.verbose:
            drug_name = result["drug"].get("name", "Drug")
            print(f"\n  Validation — {drug_name}")
            for metric, c in checks.items():
                if metric == "overall_pass":
                    continue
                flag = "✓ PASS" if c["pass"] else "✗ FAIL"
                print(f"    {metric.upper():>6}: predicted={c['predicted']:.3f}  "
                      f"observed={c['observed']:.3f}  "
                      f"fold={c['fold_error']:.2f}  {flag}")
            print(f"    Overall: {'PASS ✓' if overall else 'FAIL ✗'}")

        return checks
