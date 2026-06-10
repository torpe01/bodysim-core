"""
hepatic_module.py — Hepatic clearance module for the BodySim PBPK engine.
Extracted from pbpk_model.py v5.0. Handles OATP-facilitated uptake, TMDD QSS
scaling, biliary EHC mass balance, and dual-path (linear vs MM) hepatic
saturation logic for the 2-compartment permeability-limited liver.
"""

import numpy as np


class HepaticClearanceModule:
    """
    Computes derivative contributions for the liver vascular (LIV_VASC) and
    liver tissue (LIV_TISS) compartments.
    """

    def calculate_liver_flux(
        self,
        y:            np.ndarray,
        drug:         dict,
        liver_volume: float,
        C_art:        float,
        fup:          float,
        flow_rate:    dict,
        tp:           dict,
        params:       dict,
        extra:        dict,
    ) -> dict:
        """
        Compute all hepatic derivative contributions.

        Parameters
        ----------
        y            : full PBPK state vector (mg/L or mg)
        drug         : drug profile dict
        liver_volume : total liver volume [L] (V_liv_vasc = 15%, V_liv_tiss = 85%)
        C_art        : arterial plasma concentration [mg/L]
        fup          : unbound plasma fraction [–]
        flow_rate    : blood-flow dict; needs "liver_hepatic", "liver_portal", "gut"
        tp           : pre-computed transporter params dict from _build_transporter_params()
        params       : solver params dict (needs "liver_CL_pd")
        extra        : additional pre-computed values:
                         "C_liv_vasc"           [mg/L]
                         "C_liv_tiss"           [mg/L]
                         "C_gut_enter"          [mg/L]
                         "C_art_blood"          [mg/L blood]  = C_art × Rb
                         "C_gut_blood_out_blood"[mg/L blood]
                         "C_liv_vasc_blood"     [mg/L blood]
                         "J_bile_secretion"     [mg/h]  from acat_module

        Returns
        -------
        dict with:
            dydt_liv_vasc   : float  dC_liv_vasc/dt [mg/(L·h)]
            dydt_liv_tiss   : float  dC_liv_tiss/dt [mg/(L·h)]
            metabolic_rate  : float  total hepatic elimination [mg/h]
            C_tissue_free   : float  free hepatocyte concentration [mg/L]
        """
        Rb    = drug.get("Rb", 1.0)
        kp    = drug["kp"]

        # ── Volume split ───────────────────────────────────────────────────
        v_liv_vasc = liver_volume * 0.15
        v_liv_tiss = liver_volume * 0.85

        # ── Concentrations ─────────────────────────────────────────────────
        C_liv_vasc = extra["C_liv_vasc"]
        C_liv_tiss = extra["C_liv_tiss"]

        C_vascular_free = fup * C_liv_vasc
        C_tissue_free   = fup * C_liv_tiss / kp["liver"]

        # ── Blood-unit concentrations (for convective flows) ───────────────
        C_art_blood           = extra["C_art_blood"]
        C_gut_blood_out_blood = extra["C_gut_blood_out_blood"]
        C_liv_vasc_blood      = extra["C_liv_vasc_blood"]

        # ── Flows ──────────────────────────────────────────────────────────
        Q_ha  = flow_rate["liver_hepatic"]
        Q_pv  = flow_rate["liver_portal"]
        Q_liv = Q_ha + Q_pv

        # ── Module P6: Protein-Facilitated Hepatic Uptake ─────────────────
        # For OATP substrates with fup < 0.05, albumin presents drug at the
        # sinusoidal membrane at an effective fu ≈ 0.15 (Tsao et al. 2020;
        # Poulin & Theil 2009; Noe et al. 2007).
        # C_enhanced is used ONLY for active uptake loops; all other free-drug
        # terms retain fup unchanged.
        is_uptake_substrate = drug.get("is_uptake_substrate", False)

        if fup < 0.05 and is_uptake_substrate:
            _fu_eff    = max(fup, 0.15)
            C_enhanced = C_liv_vasc * _fu_eff
        else:
            C_enhanced = C_vascular_free

        # ── Active Uptake Clearance (Transporter Database, P6-enhanced) ───
        active_uptake_liv = 0.0
        for _name, trans in tp["hepatic_uptake"].items():
            km  = trans["Km_mgl"]
            cl  = trans["cl_linear"]
            if km > 0 and cl > 0:
                sat = km / (km + C_enhanced) if (km + C_enhanced) > 0 else 1.0
                active_uptake_liv += float(cl * sat)

        # ── Generic Active Sinusoidal Influx (P6-enhanced) ────────────────
        vmax_up = drug.get("vmax_uptake", 0.0)
        km_up   = drug.get("km_uptake",   1.0)
        j_uptake = 0.0
        if is_uptake_substrate and vmax_up > 0.0:
            j_uptake = (vmax_up * C_enhanced) / (km_up + C_enhanced)

        # ── Passive Diffusion across Sinusoidal Membrane ──────────────────
        CL_pd     = params.get("liver_CL_pd", 10.0)
        j_passive = CL_pd * (C_vascular_free - C_tissue_free)

        # ──────────────────────────────────────────────────────────────────
        # Bug Fix 1 (v5.0): Strict None-guarded dual-path hepatic saturation.
        #
        # Path A — Explicit MM kinetics (BOTH Vmax_hepatic AND Km_hepatic
        #           must be present, non-None, and positive in drug dict).
        #   J_cyp = (Vmax [mg/h] × C_free [mg/L]) / (Km [mg/L] + C_free)
        #
        # Path B — True linear clearance (all other drugs; THE CORRECT FALLBACK).
        #   J_cyp = CLh [L/h] × C_tissue_free [mg/L]
        #   No default Km.  The prior `get("Km_hepatic", 1.0)` fallback that
        #   spuriously activated MM for every drug is completely removed.
        # ──────────────────────────────────────────────────────────────────
        CLh      = drug["CLint"]
        _vmax_raw = drug.get("Vmax_hepatic")
        _km_raw   = drug.get("Km_hepatic")

        if (_vmax_raw is not None and float(_vmax_raw) > 0.0
                and _km_raw is not None and float(_km_raw) > 1e-9):
            # Path A: validated MM kinetics
            vmax_hep      = float(_vmax_raw)
            km_hep        = float(_km_raw)
            use_mm_hepatic = True
        else:
            # Path B: strictly linear — no default Km, no saturation artifact
            vmax_hep       = 0.0
            km_hep         = 1.0
            use_mm_hepatic = False

        if use_mm_hepatic:
            J_cyp = (vmax_hep * C_tissue_free) / (km_hep + C_tissue_free)
        else:
            J_cyp = CLh * C_tissue_free

        # ── Module P2: Phase II Conjugation Saturation (SULT / UGT) ──────
        # J_sult = (Vmax_sult × C_tissue_free) / (Km_sult + C_tissue_free)
        # J_ugt  = (Vmax_ugt  × C_tissue_free) / (Km_ugt  + C_tissue_free)
        # Absent params → J_phaseII = 0.0  (no regression for existing drugs).
        J_phaseII = 0.0

        sult_p = tp.get("phaseII_sult")
        if sult_p is not None:
            _km_s   = sult_p["Km_mg_L"]
            _denom_s = _km_s + C_tissue_free
            J_phaseII += (sult_p["Vmax_mg_h"] * C_tissue_free) / _denom_s \
                         if _denom_s > 1e-15 else 0.0

        ugt_p = tp.get("phaseII_ugt")
        if ugt_p is not None:
            _km_u   = ugt_p["Km_mg_L"]
            _denom_u = _km_u + C_tissue_free
            J_phaseII += (ugt_p["Vmax_mg_h"] * C_tissue_free) / _denom_u \
                         if _denom_u > 1e-15 else 0.0

        # ── Gap 2 (v5.0): Biliary Secretion ───────────────────────────────
        # J_bile_secretion is passed in from acat_module (pre-computed for
        # consistency; same cl_bile × C_tissue_free calculation).
        J_bile_secretion = extra.get("J_bile_secretion", 0.0)

        # Total metabolic rate: CYP + Phase II + biliary secretion [mg/h]
        metabolic_rate = J_cyp + J_phaseII + J_bile_secretion

        # ── Module P7: TMDD Quasi-Steady State ───────────────────────────
        # f_tmdd_scale = 1 / (1 + Bmax × Kd / (Kd + C_tissue_free)²)
        # Slows dC_liv_tiss/dt to reflect large target-bound pool at QSS.
        # tmdd_params absent → f_tmdd_scale = 1.0 (zero change for all drugs).
        _tmdd        = drug.get("tmdd_params", None)
        f_tmdd_scale = 1.0

        if _tmdd is not None:
            _Bmax = float(_tmdd.get("Bmax_mg_L", 0.0))
            _Kd   = float(_tmdd.get("Kd_mg_L",   1.0))
            if _Bmax > 0.0 and _Kd > 1e-12:
                _kd_plus_c   = max(_Kd + C_tissue_free, 1e-12)
                _sensitivity = _Bmax * _Kd / (_kd_plus_c ** 2)
                f_tmdd_scale = 1.0 / (1.0 + _sensitivity)

        # ── LIV_VASC ODE ──────────────────────────────────────────────────
        # Three flux categories (see inline doc in original pbpk_model.py):
        #   A. Convective (blood units: Q × C_plasma × Rb)
        #   B. Transporter-mediated (plasma basis: CL_trans × C_free, no Rb)
        #   C. Passive diffusion (plasma basis: CL_pd × ΔC_free, no Rb)
        dydt_liv_vasc = (
            Q_ha  * C_art_blood                    # HA inflow  [mg/h]
            + Q_pv * C_gut_blood_out_blood          # PV inflow  [mg/h]
            - Q_liv * C_liv_vasc_blood              # hepatic vein outflow [mg/h]
            - active_uptake_liv * C_vascular_free   # transporter uptake [mg/h]
            - j_uptake                              # generic active influx [mg/h]
            - j_passive                             # passive diffusion [mg/h]
        ) / v_liv_vasc

        # ── LIV_TISS ODE ──────────────────────────────────────────────────
        # All inbound mass (active + passive) minus metabolic sink.
        # TMDD scaling applied to the complete expression (P7).
        # Dimensional: ([mg/h] / L) × [–] = mg/(L·h) ✓
        dydt_liv_tiss = f_tmdd_scale * (
            active_uptake_liv * C_enhanced    # P6: protein-facilitated [mg/h]
            + j_uptake                        # generic active influx [mg/h]
            + j_passive                       # passive diffusion (bidirectional) [mg/h]
            - metabolic_rate                  # CYP + Phase II + biliary sink [mg/h]
        ) / v_liv_tiss

        return {
            "dydt_liv_vasc":  dydt_liv_vasc,
            "dydt_liv_tiss":  dydt_liv_tiss,
            "metabolic_rate": metabolic_rate,
            "C_tissue_free":  C_tissue_free,
        }