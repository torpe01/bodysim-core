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
        # NOTE: C_tissue_free is now computed below by the unified parallel-
        # binding conservation quadratic (v5.3 Target 2), which solves for it
        # directly from C_liv_tiss, kp["liver"], fup, and tmdd_params in one
        # step. It is no longer pre-computed here as a separate passive-only
        # quantity that a subsequent TMDD step would rescale.

        # ── Module P7: TMDD Parallel Binding Sinks (Unified Conservation Quadratic) ──
        # v5.3 Target 2 — REPLACES the v5.2 Step 3 f_tmdd_free partition, which
        # treated C_tissue_free (already Kp-attenuated) as the naive-free input
        # to a Langmuir isotherm. That approach sequentially filtered the same
        # free-drug pool through two binding mechanisms that both implicitly
        # account for albumin affinity — Rodgers-Rowland's kp["liver"] (derived
        # from fup, so it already encodes passive protein-binding-driven tissue
        # retention) and the explicit Bmax/Kd TMDD term (e.g. Warfarin's
        # "VKORC1 + deep albumin depot", per reference_pk.py) — multiplicatively
        # compounding the same physical attenuation instead of summing two
        # genuinely distinct binding capacities.
        #
        # Unified conservation equation (v5.3_physiological_mechanics_blueprint.md
        # Section 3): passive (Kp-implied, linear) and specific (Bmax/Kd,
        # saturable) binding are modeled as PARALLEL sinks competing for one
        # shared free-drug pool, not sequential filters:
        #
        #   C_liv_tiss_total = C_free + C_bound_passive + C_bound_specific
        #
        #   C_bound_passive  = (kp["liver"]/fup - 1) * C_free      (linear)
        #   C_bound_specific = Bmax * C_free / (Kd + C_free)       (saturable)
        #
        # Substituting and collecting terms in C_free yields the single
        # quadratic:
        #   a*C_free^2 + b*C_free + c = 0
        #     a = passive_ratio + 1.0                  (= kp["liver"]/fup)
        #     b = (passive_ratio + 1.0)*Kd + Bmax - C_liv_tiss_total
        #     c = -Kd * C_liv_tiss_total
        #
        # tmdd_params absent (Bmax=0) → quadratic reduces exactly to the
        # original passive-only relation: a=kp/fup, b=a*Kd, c=-Kd*C_total
        # → C_free = C_total / a = fup*C_total/kp["liver"], i.e. IDENTICAL to
        # the pre-v5.3 formula. Zero regression for all non-TMDD drugs.
        C_liv_tiss_total = max(C_liv_tiss, 0.0)
        passive_ratio     = (kp["liver"] / fup) - 1.0

        _tmdd = drug.get("tmdd_params", None)
        _Bmax = 0.0
        _Kd   = 1.0
        if _tmdd is not None:
            _Bmax = float(_tmdd.get("Bmax_mg_L", 0.0))
            _Kd   = float(_tmdd.get("Kd_mg_L",   1.0))
            if _Bmax <= 0.0 or _Kd <= 1e-12:
                _Bmax = 0.0  # invalid/disabled tmdd_params → behaves as absent

        _a = passive_ratio + 1.0
        _b = _a * _Kd + _Bmax - C_liv_tiss_total
        _c = -_Kd * C_liv_tiss_total

        if _a > 1e-15:
            _disc = max(_b * _b - 4.0 * _a * _c, 0.0)
            C_tissue_free = (-_b + np.sqrt(_disc)) / (2.0 * _a)
            C_tissue_free = max(C_tissue_free, 0.0)
        else:
            # Degenerate a (kp["liver"]/fup ≈ 0) — fall back to the
            # pre-existing mechanistic relation rather than divide by zero.
            C_tissue_free = fup * C_liv_tiss_total / max(kp["liver"], 1e-9)

        # ── Blood-unit concentrations (for convective flows) ───────────────
        C_art_blood           = extra["C_art_blood"]
        C_gut_blood_out_blood = extra["C_gut_blood_out_blood"]
        C_liv_vasc_blood      = extra["C_liv_vasc_blood"]

        # ── Flows ──────────────────────────────────────────────────────────
        Q_ha  = flow_rate["liver_hepatic"]
        Q_pv  = flow_rate["liver_portal"]
        Q_liv = Q_ha + Q_pv

        # ── Module P6: Protein-Facilitated Hepatic Uptake ─────────────────
        # For OATP substrates with fup below albumin_facilitation_threshold,
        # albumin presents drug at the sinusoidal membrane at an effective
        # fu floor of albumin_facilitation_eff (Tsao et al. 2020; Poulin &
        # Theil 2009; Noe et al. 2007). Defaults (0.05, 0.15) match the prior
        # hardcoded literature-derived values exactly — zero behavior change
        # for any drug entry that does not override them in reference_pk.py.
        # C_enhanced is used ONLY for active uptake loops; all other free-drug
        # terms retain fup unchanged.
        is_uptake_substrate = drug.get("is_uptake_substrate", False)
        albumin_facilitation_threshold = drug.get("albumin_facilitation_threshold", 0.05)
        albumin_facilitation_eff       = drug.get("albumin_facilitation_eff", 0.15)

        if fup < albumin_facilitation_threshold and is_uptake_substrate:
            _fu_eff    = max(fup, albumin_facilitation_eff)
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
        #
        # v5.3 Target 1 REVERTED: an earlier revision drove J_cyp/J_phaseII
        # from a flux-weighted blend of C_enhanced (vascular-coupled) and
        # C_tissue_free (tissue-coupled), intended to unify the basis between
        # active uptake and metabolism. That blend introduced a derivative
        # discontinuity: f_active_inflow's denominator included the
        # instantaneous, bidirectional j_passive term, which can approach
        # zero or change sign as C_vascular_free and C_tissue_free cross,
        # producing a non-smooth, potentially cliff-like f_active_inflow
        # response that is hostile to adaptive-step ODE solvers (e.g. CVODE
        # step-rejection/order-reduction). It also blended a vascular-
        # compartment concentration into what should be a strictly
        # tissue-compartment metabolic substrate, breaking spatial
        # compartmentalization. Since Target 2's unified conservation
        # quadratic now correctly computes C_tissue_free as the true,
        # non-double-discounted intracellular free concentration (passive
        # AND specific binding both already accounted for there), the
        # metabolic enzymes are restored to act strictly on C_tissue_free —
        # continuous, differentiable, and spatially consistent.
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
        # v5.2 Step 3 / v5.3 Target 2: TMDD's effect is captured entirely
        # upstream via the unified parallel-binding conservation quadratic
        # that produces C_tissue_free, which directly drives J_cyp and
        # J_phaseII above (Target 1's flux-weighted blend was reverted for
        # numerical-stability and spatial-compartmentalization reasons —
        # see the note above J_cyp). C_liv_tiss (this state's own
        # derivative) is NOT touched by TMDD at all — its full,
        # un-suppressed mass balance is preserved.
        # Dimensional: ([mg/h] / L) = mg/(L·h) ✓
        dydt_liv_tiss = (
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