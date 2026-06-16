import sys
import os
import numpy as np
import pandas as pd

# Adjust path to import engine components
sys.path.append(os.getcwd())

from engine.simulator import Simulator
from engine.admet import build_drug_profile
from engine.validation.reference_pk import REFERENCE_PK

# ── v5.1 Data Bridge Fix ───────────────────────────────────────────────────
# Previously, advanced module keys present in reference_pk.py were never
# forwarded to build_drug_profile(), silently disabling Modules P1/P2/P5/P6/P7
# and the EHC circuit for 11 of the 23 validation drugs.
#
# This list contains every optional advanced key that build_drug_profile()
# accepts via **kwargs.  Keys absent from a drug's reference_pk entry are
# simply skipped — no KeyError, no change to existing behaviour for that drug.
_ADVANCED_KEYS = [
    "gut_transporter",      # Module P5 — active SLC influx (PEPT1 / OCT1 / PMAT)
    "phaseII_kinetics",     # Module P2 — SULT/UGT saturable conjugation
    "fu_gut",               # Module P1 — gut-wall CYP3A4 unbound fraction
    "CLint_gut_cyp3a4",     # Module P1 — gut-wall CYP3A4 intrinsic clearance
    "tmdd_params",          # Module P7 — TMDD quasi-steady state
    "kp_scalar",            # Empirical Vd correction scalar
    "cl_bile_lh",           # Gap 2 EHC — biliary secretion clearance [L/h]
    "f_reabs_bile",         # Gap 2 EHC — fraction reabsorbed from bile
    "p_eff",                # Measured intestinal permeability [cm/s] override
    "is_uptake_substrate",  # Module P6 — OATP1B1/1B3 hepatic uptake flag
    "vmax_uptake",          # Module P6 — OATP Vmax [mg/h]
    "km_uptake",            # Module P6 — OATP Km [mg/L]
    "Vmax_hepatic",         # Explicit Michaelis-Menten hepatic Vmax [mg/h]
    "Km_hepatic",           # Explicit Michaelis-Menten hepatic Km [mg/L]
    "absorption_segments",  # Gap 3 — regional absorption window restriction
    "enteric_coated",       # Gap 3 — enteric-coated release flag
]
# ──────────────────────────────────────────────────────────────────────────


def run_qualification_suite():
    print(f"\n{'='*60}")
    print(f" BODYSIM QUALIFICATION SUITE — v5.1 Data Bridge Fix")
    print(f" Running validation on {len(REFERENCE_PK)} drugs...")
    print(f"{'='*60}\n")

    # Use verbose=False to keep terminal clean during batch run
    sim = Simulator(verbose=False)
    results = []

    for name, data in REFERENCE_PK.items():
        print(f" Validating: {name:<15} ", end="", flush=True)

        try:
            # ── v5.1: collect every advanced key present for this drug ──
            # Keys not in this drug's entry are omitted entirely — graceful
            # degradation means the engine falls back to linear physics for
            # any mechanism whose parameters are absent.
            advanced_kwargs = {k: data[k] for k in _ADVANCED_KEYS if k in data}

            # 1. Build Profile using ADMET with REAL measured properties
            # (logp, fup, mw, pka, drug_type from reference_pk.py)
            # Use clinical PK parameters (ka, F, CLint, CLrenal) when available.
            # v5.1 FIX: the explicit transporter/MM kwargs (is_uptake_substrate,
            # vmax_uptake, km_uptake, Vmax_hepatic, Km_hepatic) were previously
            # passed BOTH by name AND inside **advanced_kwargs (since they are
            # also members of _ADVANCED_KEYS), causing
            # "got multiple values for keyword argument" TypeErrors for any
            # drug whose reference_pk entry defines them. They are now forwarded
            # exclusively via **advanced_kwargs.
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
                # v5.1: all advanced module params (including transporter/MM
                # kinetics) forwarded via **advanced_kwargs — see _ADVANCED_KEYS.
                **advanced_kwargs,
            )

            # 2. Run Simulation
            res = sim.run_single(
                drug=profile,
                dose_mg=data["dose"],
                route=data["route"],
                t_end_h=48.0
            )

            # 3. Compare against Ground Truth
            metrics = sim.validate(
                result=res,
                cmax_target=data["cmax"],
                auc_target=data["auc"],
                fold_tolerance=2.0
            )

            results.append({
                "Drug":      name,
                "Pred_Cmax": res["cmax_plasma"],
                "Obs_Cmax":  data["cmax"],
                "Cmax_Fold": metrics["cmax"]["fold_error"],
                "Pred_AUC":  res["auc_plasma"],
                "Obs_AUC":   data["auc"],
                "AUC_Fold":  metrics["auc"]["fold_error"],
                "Pass":      "✓" if metrics["overall_pass"] else "✗"
            })
            print("✓" if metrics["overall_pass"] else "⚠")

        except Exception as e:
            print(f"ERROR: {str(e)}")

    # 4. Final Report
    if not results:
        print("\n[!] No simulations succeeded. Check engine logs.")
        return

    df = pd.DataFrame(results)
    pass_rate = (df["Pass"] == "✓").mean() * 100

    print(f"\n{'='*60}")
    print(df.to_string(index=False))
    print(f"\n Final Pass Rate: {pass_rate:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_qualification_suite()