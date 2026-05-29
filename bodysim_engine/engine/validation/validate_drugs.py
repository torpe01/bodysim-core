import sys
import os
import numpy as np
import pandas as pd

# Adjust path to import engine components
sys.path.append(os.getcwd())

from engine.simulator import Simulator
from engine.admet import build_drug_profile
from engine.validation.reference_pk import REFERENCE_PK

def run_qualification_suite():
    print(f"\n{'='*60}")
    print(f" BODYSIM QUALIFICATION SUITE — v2.2 Mechanistic Model")
    print(f" Running validation on {len(REFERENCE_PK)} drugs...")
    print(f"{'='*60}\n")

    # Use verbose=False to keep terminal clean during batch run
    sim = Simulator(verbose=False)
    results = []

    for name, data in REFERENCE_PK.items():
        print(f" Validating: {name:<15} ", end="", flush=True)
        
        try:
            # 1. Build Profile using ADMET v2.2 with REAL measured properties
            # (logp, fup, mw, pka, drug_type from reference_pk.py)
            # Use clinical PK parameters (ka, F, CLint, CLrenal) when available
            # v2.7: Now also passes transporter parameters (vmax_uptake, km_uptake, is_uptake_substrate)
            # if they are defined in reference_pk.py
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
                # v2.7: Pass transporter parameters from reference_pk validation data
                is_uptake_substrate=data.get("is_uptake_substrate"),
                vmax_uptake=data.get("vmax_uptake"),
                km_uptake=data.get("km_uptake"),
                Vmax_hepatic=data.get("Vmax_hepatic"),
                Km_hepatic=data.get("Km_hepatic"),
            )

            # 2. FIXED: Run Simulation using run_single()
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
                "Drug": name,
                "Pred_Cmax": res["cmax_plasma"],
                "Obs_Cmax": data["cmax"],
                "Cmax_Fold": metrics["cmax"]["fold_error"],
                "Pred_AUC": res["auc_plasma"],
                "Obs_AUC": data["auc"],
                "AUC_Fold": metrics["auc"]["fold_error"],
                "Pass": "✓" if metrics["overall_pass"] else "✗"
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