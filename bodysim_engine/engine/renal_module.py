"""
renal_module.py — Renal elimination module for the BodySim PBPK engine.
Extracted from pbpk_model.py v5.0. Implements anatomically correct glomerular
filtration (arterial driving force, fup pre-multiplied in cl_filt), active
tubular secretion with well-stirred extraction ratio cap, and passive tubular
reabsorption for the lumped kidney compartment.
"""

import numpy as np


class RenalEliminationModule:
    """
    Computes the derivative contribution for the kidney (KID) compartment.
    """

    def calculate_kidney_flux(
        self,
        y:              np.ndarray,
        drug:           dict,
        kidney_volume:  float,
        C_art:          float,
        cl_filt:        float,
        tp:             dict,
        flow_rate:      dict,
        extra:          dict,
    ) -> float:
        """
        Compute dC_kidney/dt.

        Parameters
        ----------
        y               : full PBPK state vector
        drug            : drug profile dict
        kidney_volume   : kidney compartment volume [L]
        C_art           : arterial plasma concentration [mg/L]
        cl_filt         : GFR filtration clearance [L/h]  (fup already pre-multiplied
                          by _build_transporter_params as cl_filt = gfr_lh × fup)
        tp              : pre-computed transporter params dict:
                            renal_secretion, cl_sec_plasma_cap_lh, cl_reabsorption_lh
        flow_rate       : blood-flow dict; needs "kidney"
        extra           : dict with:
                            "C_kid"      [mg/L]  kidney tissue concentration
                            "C_art_blood"[mg/L blood]  = C_art × Rb

        Returns
        -------
        float  dC_kid/dt [mg/(L·h)]

        Architecture (v2.7 + Bug Fix 2 v5.0):
        ─────────────────────────────────────
        dC_kid/dt = perfusion_inflow
                  − active_secretion_rate       (drug lost to urine)
                  + passive_tubular_reabsorption (drug returned from tubule)
                  − glomerular_filtration_rate   (drug lost to urine; arterial basis)

        Bug Fix 2 (v5.0) — GFR anatomical correction:
          The filtration term was previously placed in dydt[VEN] and used
          cl_filt × fup × C_ven, which:
            (a) attributed mass loss to the venous mixing pool (wrong anatomy),
            (b) squared fup because cl_filt already contains fup,
            (c) used C_ven ≠ C_art during absorption, distorting Cmax shape.
          Correct placement: dydt[KID], driving force = C_art, fup NOT re-applied.
          Dimensional: cl_filt [L/h] × C_art [mg/L] / V_kidney [L] = mg/(L·h) ✓
        """
        Rb  = drug.get("Rb", 1.0)
        kp  = drug["kp"]
        fup = drug["fup"]

        C_kid      = extra["C_kid"]
        C_art_blood= extra["C_art_blood"]
        Q_kid      = flow_rate["kidney"]

        C_kid_free = fup * C_kid / kp["kidney"]

        # ── Perfusion term ─────────────────────────────────────────────────
        # Standard Q×(C_in − C_out/Kp) two-compartment exchange.
        # Blood units: C_art_blood = C_art × Rb, outflow = (C_kid/Kp) × Rb
        passive_kid = (Q_kid / kidney_volume) * (
            C_art_blood - (C_kid / kp["kidney"]) * Rb
        )

        # ── Step 1: Gross active secretion (MM-kinetic, sum over transporters) ──
        # _active_cl pattern: saturation-adjusted clearance [L/h] at C_kid_free.
        # Each transporter's contribution is the Km-weighted linear clearance.
        active_sec_gross_cl = 0.0
        for _name, trans in tp["renal_secretion"].items():
            km  = trans["Km_mgl"]
            cl  = trans["cl_linear"]
            if km > 0 and cl > 0:
                sat = km / (km + C_kid_free) if (km + C_kid_free) > 0 else 1.0
                active_sec_gross_cl += float(cl * sat)

        # ── Step 2: Apply renal extraction ratio cap (v2.7, Gap 3.3) ─────
        # Well-stirred kidney model: CL_obs ≤ Q_kidney_plasma.
        # Cap was pre-computed in _build_transporter_params.
        cl_sec_cap = tp["cl_sec_plasma_cap_lh"]
        if active_sec_gross_cl > cl_sec_cap:
            active_sec_gross_cl = cl_sec_cap

        # Active secretion rate [mg/(L·h)]: CL × C_free / V_kidney
        active_sec_rate = (active_sec_gross_cl * C_kid_free) / kidney_volume

        # ── Step 3: Passive tubular reabsorption (v2.7, Gap 3.1) ─────────
        # CL_reab = Q_tubular_water × f_neutral_urine × k_perm  (pre-computed)
        # Rate = CL_reab × C_kid_free / V_kidney  [mg/(L·h)]
        cl_reab     = tp["cl_reabsorption_lh"]
        reabs_rate  = (cl_reab * C_kid_free) / kidney_volume

        # ── Bug Fix 2 (v5.0): GFR — anatomically correct, arterial basis ──
        # cl_filt already contains fup (cl_filt = gfr_lh × fup).
        # Driving concentration is total arterial plasma (C_art), NOT fup × C_art.
        # Dimensional: cl_filt [L/h] × C_art [mg/L] / V_kidney [L] = mg/(L·h) ✓
        gfr_filtration_rate = (cl_filt * C_art) / kidney_volume

        # ── ODE assembly ──────────────────────────────────────────────────
        dydt_kid = passive_kid - active_sec_rate + reabs_rate - gfr_filtration_rate
        return dydt_kid