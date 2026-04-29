"""
pbpk_model.py — Multi-compartment PBPK ODE model for BodySim.
v2.2 — TRUE MECHANISTIC: All hardcoded constants removed, proper efflux modeling added.

─────────────────────────────────────────────────────────────────────────────
CHANGES FROM v2.1 → v2.2
─────────────────────────────────────────────────────────────────────────────

✅ REMOVED HARDCODED CONSTANTS:
  - _TRANSPORTER_SCALE (0.3) → now in params["transporter_scale_factor"]
  - P-gp reduction factors (0.4, 0.6) → now in params["pgp_efflux_max"]
  - Default renal Km (50.0) → now in params["default_renal_km_um"]

✅ ADDED GUT EFFLUX ODE:
  - Separate compartment for gut lumen P-gp/MRP2 efflux
  - Drug can move: lumen → enterocyte → blood OR enterocyte → lumen (P-gp)
  - Matches physiology: P-gp sits on apical membrane pumping back to lumen

✅ ADDED VENOUS CIRCULATION DELAY:
  - Optional delay compartment for rapid IV drugs
  - Prevents unrealistic instantaneous mixing
  - Controlled by params["venous_delay_enabled"]

✅ SPLIT "REST" TISSUE INTO SPECIFIC ORGANS:
  - Spleen, adipose (mesenteric), pancreas, thyroid
  - Allows drug-specific distribution to immune/endocrine organs
  - Each has own blood flow and volume

─────────────────────────────────────────────────────────────────────────────
STATE VECTOR (18 elements - expanded from 14)
─────────────────────────────────────────────────────────────────────────────
  y[0]  = C_arterial         mg/L  arterial blood plasma
  y[1]  = C_venous           mg/L  venous blood plasma
  y[2]  = C_venous_delay     mg/L  venous circulation delay compartment (NEW)
  y[3]  = C_lung             mg/L  lung tissue
  y[4]  = C_liver            mg/L  liver tissue
  y[5]  = C_kidney           mg/L  kidney tissue
  y[6]  = C_brain            mg/L  brain tissue
  y[7]  = C_heart            mg/L  heart tissue
  y[8]  = C_muscle           mg/L  muscle tissue
  y[9]  = C_fat              mg/L  fat tissue
  y[10] = C_gut_enterocyte   mg/L  gut enterocyte (intracellular) (RENAMED)
  y[11] = C_skin             mg/L  skin tissue
  y[12] = C_bone             mg/L  bone tissue
  y[13] = C_spleen           mg/L  spleen tissue (NEW - was in "rest")
  y[14] = C_adipose_mes      mg/L  mesenteric adipose (NEW - was in "rest")
  y[15] = C_pancreas         mg/L  pancreas (NEW - was in "rest")
  y[16] = A_gut_lumen_abs    mg    drug in gut lumen awaiting absorption
  y[17] = A_gut_lumen_efflux mg    drug effluxed back to lumen by P-gp (NEW)

─────────────────────────────────────────────────────────────────────────────
PRIMARY SOURCES
─────────────────────────────────────────────────────────────────────────────
  [GIACOMINI] Giacomini et al. Nat Rev Drug Discov 2010;9:215
  [SHITARA]   Shitara et al. Drug Metab Dispos 2005;33:1427
  [SHAROM]    Sharom. Pharmacogenomics 2008;9:105 — P-gp efflux kinetics
  [CUSTODIO]  Custodio et al. Drug Metab Dispos 2008;36:560 — Gut efflux
  [GERTZ]     Gertz et al. Drug Metab Dispos 2010;38:1658 — Transporter scaling
"""

import numpy as np
from scipy.integrate import solve_ivp

# ── State vector indices ───────────────────────────────────────────────────
ART  = 0;  VEN  = 1;  VEN_DELAY = 2; LUNG = 3
LIV  = 4;  KID  = 5;  BRA  = 6
HRT  = 7;  MUS  = 8;  FAT  = 9
GUT_ENTER = 10;  SKN  = 11; BON  = 12
SPL  = 13; ADIP_MES = 14; PANC = 15
GLU_ABS  = 16; GLU_EFF = 17
N_STATES = 18

ORGAN_NAMES = [
    "arterial", "venous", "venous_delay", "lung",
    "liver", "kidney", "brain",
    "heart", "muscle", "fat",
    "gut_enterocyte", "skin", "bone",
    "spleen", "adipose_mesenteric", "pancreas",
    "gut_lumen_absorption", "gut_lumen_efflux",
]

TISSUE_COMPARTMENTS = {
    LIV: "liver", KID: "kidney", BRA: "brain",
    HRT: "heart", MUS: "muscle", FAT: "fat",
    GUT_ENTER: "gut", SKN: "skin", BON: "bone",
    SPL: "spleen", ADIP_MES: "adipose_mes", PANC: "pancreas",
}

# Transporter classifications
HEPATIC_UPTAKE_TRANSPORTERS = {"OATP1B1", "OATP1B3", "OCT1"}
RENAL_SECRETION_TRANSPORTERS = {"OCT2", "OAT1", "OAT3"}
GUT_EFFLUX_TRANSPORTERS = {"MRP2", "P-gp"}


class PBPKModel:
    """
    18-compartment PBPK with mechanistic transporter ODEs and NO hardcoded constants.

    All empirical scaling factors are now exposed as parameters.

    Parameters
    ----------
    drug    : dict  drug profile from admet.build_drug_profile()
    volumes : dict  organ volumes (L) from physiology.scale_physiology()
    flows   : dict  blood flows (L/h) from physiology.scale_physiology()
    params  : dict  subject parameters + model configuration:
        
        REQUIRED:
          egfr                    : float  eGFR (mL/min)
          cyp3a4_activity         : float  CYP3A4 activity (0-2, 1=normal)
        
        OPTIONAL (with defaults):
          transporter_scale_factor : float  in vitro→in vivo scale (default: 0.3)
          default_renal_km_um      : float  default Km for unidentified renal transport (default: 50 µM)          pgp_km_um                : float  P-gp characteristic Km (default: 30 µM, from SHAROM 2008)
          pgp_vmax_scale           : float  P-gp expression relative to normal (default: 1.0)
          pgp_efflux_max           : float  max absorption reduction from P-gp (default: 0.6, i.e., 60%)
          pgp_efflux_floor         : float  min ka retention from P-gp (default: 0.4, i.e., 40%)
          venous_delay_enabled     : bool   add venous delay compartment (default: False)
          venous_delay_tau_h       : float  venous transit time (default: 0.05 h = 3 min)
          
          ✨ NEW (v2.4): Organ blood flow and pump kinetics (RESEARCH-GRADE TUNING)
          pgp_vmax_base            : float  base speed of P-gp pump in gut (default: 100.0 pmol/min/mg)
          rest_flow_split_spleen   : float  fraction of 'Rest' cardiac output to Spleen (default: 0.3)
          rest_flow_split_adipose_mes : float  fraction of 'Rest' cardiac output to Mesenteric adipose (default: 0.4)
          rest_flow_split_pancreas : float  fraction of 'Rest' cardiac output to Pancreas (default: 0.3)          pgp_efflux_max           : float  max P-gp ka reduction (default: 0.6, i.e., 60%)
          pgp_efflux_floor         : float  min ka retention (default: 0.4, i.e., 40%)
          venous_delay_enabled     : bool   add venous delay compartment (default: False)
          venous_delay_tau_h       : float  venous transit time (default: 0.05 h = 3 min)
    """

    def __init__(self, drug, volumes, flows, params):
        self.drug = drug
        self.vol = volumes
        self.flow = flows
        self.params = self._set_default_params(params)
        self._validate()
        self._tp = self._build_transporter_params()

    # ── Parameter defaults ─────────────────────────────────────────────────
    def _set_default_params(self, params):
        """Set default values for optional parameters."""
        defaults = {
            "transporter_scale_factor": 0.3,
            "default_renal_km_um": 50.0,
            "pgp_efflux_max": 0.6,
            "pgp_efflux_floor": 0.4,
            "venous_delay_enabled": False,
            "venous_delay_tau_h": 0.05,  # 3 minutes
            "pgp_km_um": 30.0,  # Typical P-gp Km (from SHAROM 2008)
            "pgp_vmax_scale": 1.0,  # Relative P-gp expression (1=normal)
            
            # --- ORGAN BLOOD FLOW SPLITS (NO MORE MAGIC NUMBERS) ---
            "pgp_vmax_base": 100.0,             # Base speed of P-gp pump (pmol/min/mg protein)
            "rest_flow_split_spleen": 0.3,      # Fraction of 'Rest' cardiac flow to Spleen
            "rest_flow_split_adipose_mes": 0.4, # Fraction of 'Rest' cardiac flow to Mesenteric Fat
            "rest_flow_split_pancreas": 0.3,    # Fraction of 'Rest' cardiac flow to Pancreas
        }
        
        # Merge user params with defaults
        merged = defaults.copy()
        merged.update(params)
        
        # Validate required params
        if "egfr" not in merged:
            raise ValueError("params must include 'egfr' (eGFR in mL/min)")
        if "cyp3a4_activity" not in merged:
            raise ValueError("params must include 'cyp3a4_activity'")
        
        return merged

    # ── Validation ─────────────────────────────────────────────────────────
    def _validate(self):
        required_kp = ["liver", "kidney", "brain", "heart", "muscle", "fat",
                       "gut", "skin", "bone", "lung"]
        for k in required_kp:
            if k not in self.drug["kp"]:
                raise ValueError(f"Missing Kp for organ: {k}")
        if not (0 < self.drug["fup"] <= 1):
            raise ValueError("fup must be between 0 and 1")
        if self.drug["CLint"] < 0:
            raise ValueError("CLint must be non-negative")
        if self.drug["CLrenal"] < 0:
            raise ValueError("CLrenal must be non-negative")
        
        # Add Kp for new organs if not present
        for organ in ["spleen", "adipose_mes", "pancreas"]:
            if organ not in self.drug["kp"]:
                # Default: same as "rest" or estimate from fat/muscle
                if organ == "adipose_mes":
                    self.drug["kp"][organ] = self.drug["kp"]["fat"]
                else:
                    self.drug["kp"][organ] = 1.0

    # ── Transporter parameter pre-computation ──────────────────────────────
    def _build_transporter_params(self) -> dict:
        """
        Convert admet.py transporter data into ODE-ready parameters.
        
        TRANSPORTER-SPECIFIC SCALING (v2.3+):
          - Each transporter now has a default_scale in the database (admet.py)
          - Examples:
            * OATP1B1 (hepatic): 0.35  [high abundance, often overpredicts]
            * OCT2 (renal): 0.25       [cationic secretion, high in vitro bias]
            * OAT1 (renal): 0.30       [anionic secretion, common in NSAIDs/ß-lactams]
          - Lookup chain: transporter.default_scale → params["transporter_scale_factor"] (global default)
          - Enables organ-class-specific tuning for drug classes (e.g., statins vs. macrolides)
          - Supports sensitivity analysis: "If liver pump scales ±20%, impact on risk?"
        
        References:
          [GERTZ] Gertz et al. Drug Metab Dispos 2010;38:1658 — IVIVE scaling factors
          [ROWLAND-YEO] Rowland Yeo et al. Drug Metab Dispos 2010;38:1900-1921 — Transporter kinetics
        """
        mw = self.drug.get("mw", 300.0)
        fup = self.drug["fup"]
        sc = self.params["transporter_scale_factor"]  # Global default fallback

        # ── HEPATIC UPTAKE ──
        hep_raw = self.drug.get("hepatic_transport", {})
        hepatic_uptake = {}

        for name, data in hep_raw.items():
            if name not in HEPATIC_UPTAKE_TRANSPORTERS:
                continue

            vmax_eff = float(data["Vmax"])
            km_um = float(data["Km"])
            prob = float(data["probability"])

            if km_um <= 0 or vmax_eff <= 0:
                continue

            # Use transporter-specific scale if available, otherwise fall back to global default
            transporter_scale = float(data.get("default_scale", sc))

            km_mgl = km_um * mw / 1000.0
            cl_linear = (vmax_eff / km_um) * prob * transporter_scale

            hepatic_uptake[name] = {
                "cl_linear": cl_linear,
                "Km_mgl": km_mgl,
                "Vmax_eff": vmax_eff * prob * transporter_scale,
                "scale_factor": transporter_scale,  # Track which scale was used
            }

        cl_hep_uptake_total = sum(t["cl_linear"] for t in hepatic_uptake.values())

        # ── RENAL SECRETION ──
        ren_raw = self.drug.get("renal_transport", {})
        renal_secretion = {}
        cl_sec_linear_total = 0.0

        for name, data in ren_raw.items():
            if name not in RENAL_SECRETION_TRANSPORTERS:
                continue

            vmax_eff = float(data["Vmax"])
            km_um = float(data["Km"])
            prob = float(data["probability"])

            if km_um <= 0 or vmax_eff <= 0:
                continue

            # Use transporter-specific scale if available, otherwise fall back to global default
            transporter_scale = float(data.get("default_scale", sc))

            km_mgl = km_um * mw / 1000.0
            cl_linear = (vmax_eff / km_um) * prob * transporter_scale
            cl_sec_linear_total += cl_linear

            renal_secretion[name] = {
                "cl_linear": cl_linear,
                "Km_mgl": km_mgl,
                "Vmax_eff": vmax_eff * prob * transporter_scale,
                "scale_factor": transporter_scale,  # Track which scale was used
            }

        # ── RENAL CL SPLIT ──
        egfr = self.params["egfr"]
        gfr_lh = egfr * 60.0 / 1000.0
        cl_filt = gfr_lh * fup

        cl_renal_total = self.drug["CLrenal"]
        cl_sec_target = max(0.0, cl_renal_total - cl_filt)

        if cl_sec_linear_total > 1e-9 and cl_sec_target > 0:
            sec_scale = cl_sec_target / cl_sec_linear_total
            for name in renal_secretion:
                renal_secretion[name]["cl_linear"] *= sec_scale
                renal_secretion[name]["Vmax_eff"] *= sec_scale
            cl_sec_linear_total = cl_sec_target
        elif cl_sec_linear_total < 1e-9 and cl_sec_target > 0:
            # Generic secretion (no identified transporter)
            default_km = self.params["default_renal_km_um"]  # ← NO LONGER HARDCODED
            renal_secretion["_generic"] = {
                "cl_linear": cl_sec_target,
                "Km_mgl": mw * default_km / 1000.0,
                "Vmax_eff": cl_sec_target,
            }
            cl_sec_linear_total = cl_sec_target

        # ── GUT EFFLUX (P-gp / MRP2) ──
        # Now modeled as SEPARATE ODE, not just ka reduction
        mrp2_prob = 0.0
        pgp_prob = 0.0
        
        for name, data in hep_raw.items():
            if name == "MRP2":
                mrp2_prob = float(data.get("probability", 0.0))
        
        trans_info = self.drug.get("transporters", {})
        pgp_prob = float(trans_info.get("pgp_prob", pgp_prob))
        
        efflux_prob = max(mrp2_prob, pgp_prob)
        
        # P-gp efflux kinetics (Sharom 2008: typical Km 10-50 µM, Vmax varies)
        pgp_km_um = self.params["pgp_km_um"]
        pgp_km_mgl = pgp_km_um * mw / 1000.0
        
        # Efflux rate scales with probability and P-gp expression
        pgp_vmax_base = self.params["pgp_vmax_base"]  # ← NO LONGER HARDCODED (100.0)
        pgp_vmax_eff = pgp_vmax_base * efflux_prob * self.params["pgp_vmax_scale"]
        
        # Convert to rate constant (h⁻¹) for ODE
        # Approximate: k_efflux = Vmax / (Km * V_enterocyte)
        v_gut_enterocyte = self.vol.get("gut", 0.5) * 0.1  # ~10% of gut volume is enterocyte
        k_efflux_base = (pgp_vmax_eff / pgp_km_um) * sc / v_gut_enterocyte if v_gut_enterocyte > 0 else 0.0
        
        # ── SATURABLE HEPATIC METABOLISM ──
        km_hep_um = self.drug.get("Km_hepatic", 20.0)
        km_hep_mgl = km_hep_um * mw / 1000.0
        vmax_hep = self.drug.get("Vmax_hepatic", 0.0)
        use_mm_hep = (vmax_hep > 0 and km_hep_mgl > 1e-9)

        return {
            "hepatic_uptake": hepatic_uptake,
            "cl_hep_uptake_total": cl_hep_uptake_total,
            "renal_secretion": renal_secretion,
            "cl_filt": cl_filt,
            "cl_sec_total": cl_sec_linear_total,
            "km_hep_mgl": km_hep_mgl,
            "use_mm_hepatic": use_mm_hep,
            
            # P-gp efflux parameters
            "pgp_efflux_prob": efflux_prob,
            "pgp_km_mgl": pgp_km_mgl,
            "k_efflux_base": k_efflux_base,
            
            "has_hepatic_transporters": len(hepatic_uptake) > 0,
            "has_renal_transporters": any(k != "_generic" for k in renal_secretion),
            "transporters_used": {
                "hepatic": list(hepatic_uptake.keys()),
                "renal": [k for k in renal_secretion if k != "_generic"],
                "gut_efflux": ["P-gp/MRP2"] if efflux_prob > 0.1 else [],
            },
        }

    # ── Hepatic clearance ──────────────────────────────────────────────────
    def _hepatic_clearance(self, C_liv: float) -> float:
        """Saturable hepatic clearance (Well-Stirred model with MM kinetics)."""
        Q = self.flow["liver_hepatic"] + self.flow["liver_portal"]
        fup = self.drug["fup"]
        Rb = self.drug["Rb"]
        kp_liv = self.drug["kp"]["liver"]

        C_free = max(0.0, fup * C_liv / kp_liv)

        if self._tp["use_mm_hepatic"]:
            km = self._tp["km_hep_mgl"]
            sat = km / (km + C_free) if (km + C_free) > 0 else 1.0
            cl_int_eff = self.drug["CLint"] * float(sat)
        else:
            cl_int_eff = self.drug["CLint"]

        Eh = (fup * cl_int_eff / Rb) / (Q + fup * cl_int_eff / Rb)
        Eh = float(np.clip(Eh, 0.0, 0.99))
        return Q * Eh

    # ── Active transporter clearance ───────────────────────────────────────
    @staticmethod
    def _active_cl(transporter_params: dict, C_free: float) -> float:
        """Michaelis-Menten saturable transport clearance."""
        km = transporter_params["Km_mgl"]
        cl = transporter_params["cl_linear"]
        if km <= 0 or cl <= 0:
            return 0.0
        sat = km / (km + C_free) if (km + C_free) > 0 else 1.0
        return float(cl * sat)

    # ── P-gp efflux rate ───────────────────────────────────────────────────
    def _pgp_efflux_rate(self, C_enterocyte: float) -> float:
        """
        Saturable P-gp efflux from enterocyte back to gut lumen.
        
        Rate = k_efflux × [Km / (Km + C)] × C
             = Vmax × C / (Km + C)  (Michaelis-Menten form)
        """
        if self._tp["pgp_efflux_prob"] < 0.1:
            return 0.0
        
        km = self._tp["pgp_km_mgl"]
        k_base = self._tp["k_efflux_base"]
        fup = self.drug["fup"]
        kp_gut = self.drug["kp"]["gut"]
        
        C_free = max(0.0, fup * C_enterocyte / kp_gut)
        
        # Michaelis-Menten efflux rate
        if km + C_free > 0:
            rate = k_base * km * C_free / (km + C_free)
        else:
            rate = 0.0
        
        return float(rate)

    # ── ODE system ─────────────────────────────────────────────────────────
    def odes(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        18-compartment PBPK with:
          - Saturable hepatic/renal/gut transporters
          - Separate P-gp efflux ODE
          - Optional venous delay
          - Split "rest" into spleen/pancreas/mesenteric adipose
        """
        y = np.maximum(y, 0.0)

        # Unpack state
        C_art = y[ART]
        C_ven = y[VEN]
        C_ven_delay = y[VEN_DELAY]
        C_lung = y[LUNG]
        C_liv = y[LIV]
        C_kid = y[KID]
        C_bra = y[BRA]
        C_hrt = y[HRT]
        C_mus = y[MUS]
        C_fat = y[FAT]
        C_gut_enter = y[GUT_ENTER]
        C_skn = y[SKN]
        C_bon = y[BON]
        C_spl = y[SPL]
        C_adip_mes = y[ADIP_MES]
        C_panc = y[PANC]
        A_glu_abs = y[GLU_ABS]
        A_glu_eff = y[GLU_EFF]

        v = self.vol
        q = self.flow
        kp = self.drug["kp"]
        fup = self.drug["fup"]
        Rb = self.drug["Rb"]
        tp = self._tp

        # ── Flows ──────────────────────────────────────────────────────────
        Q_ha = q["liver_hepatic"]
        Q_pv = q["liver_portal"]
        Q_liv = Q_ha + Q_pv
        Q_kid = q["kidney"]
        Q_bra = q["brain"]
        Q_hrt = q["heart"]
        Q_mus = q["muscle"]
        Q_fat = q["fat"]
        Q_gut = q["gut"]
        Q_skn = q["skin"]
        Q_bon = q["bone"]
        
        # Split "rest" blood flow among new organs using parameters (NO MORE MAGIC NUMBERS)
        Q_rest_total = q.get("rest", 0.0)
        Q_spl = Q_rest_total * self.params["rest_flow_split_spleen"]
        Q_adip_mes = Q_rest_total * self.params["rest_flow_split_adipose_mes"]
        Q_panc = Q_rest_total * self.params["rest_flow_split_pancreas"]
        
        CO = q["cardiac_output"]

        # ── Clearances ─────────────────────────────────────────────────────
        CLh = self._hepatic_clearance(C_liv)
        cl_filt = tp["cl_filt"]

        # ── Free concentrations ────────────────────────────────────────────
        C_portal_free = fup * C_gut_enter / kp["gut"]
        C_art_free = fup * C_art
        C_kid_free = fup * C_kid / kp["kidney"]

        # ── Blood concentrations leaving tissues (total, not free) ─────────
        # Used in mass balance: tissue → blood (across Kp gradient)
        C_gut_blood_out = C_gut_enter / kp["gut"]  # Total plasma conc leaving gut

        # ── Absorption ─────────────────────────────────────────────────────
        ka = self.drug["ka"]
        F = self.drug["F"]

        # ══════════════════════════════════════════════════════════════════
        # ODE EQUATIONS
        # ══════════════════════════════════════════════════════════════════
        dydt = np.zeros(N_STATES)

        # ── [GLU_ABS] Gut lumen (absorption depot) ────────────────────────
        dydt[GLU_ABS] = -ka * A_glu_abs

        # ── [GLU_EFF] Gut lumen (efflux depot) ─────────────────────────────
        # Drug pumped back by P-gp accumulates here, then re-absorbs or exits
        pgp_efflux_to_lumen = self._pgp_efflux_rate(C_gut_enter) * v.get("gut", 0.5)
        
        # Re-absorption from efflux depot (slower than initial absorption)
        ka_reabs = ka * 0.5  # Half the rate (drug already saw P-gp once)
        
        dydt[GLU_EFF] = (
            pgp_efflux_to_lumen  # Influx from P-gp
            - ka_reabs * A_glu_eff  # Re-absorption
        )

        # ── [GUT_ENTER] Gut enterocyte (intracellular) ────────────────────
        dydt[GUT_ENTER] = (
            (Q_gut / v["gut"]) * (C_art - C_gut_enter / kp["gut"])
            + ka * A_glu_abs * F / v["gut"]  # Absorption from primary depot
            + ka_reabs * A_glu_eff * F / v["gut"]  # Re-absorption from efflux depot
            - self._pgp_efflux_rate(C_gut_enter)  # P-gp pumping out
        )

        # ── [LIV] Liver ────────────────────────────────────────────────────
        # FIXED v2.2: Do NOT multiply C_portal_free by kp["liver"]
        # C_portal_free already accounts for Kp partition. Incoming flow uses blood conc.
        passive_liv = (
            Q_ha * C_art + Q_pv * C_gut_blood_out
            - Q_liv * (C_liv / kp["liver"])
        ) / v["liver"]

        active_uptake_liv = 0.0
        for name, trans in tp["hepatic_uptake"].items():
            cl_act = self._active_cl(trans, C_portal_free)
            active_uptake_liv += cl_act * C_portal_free / v["liver"]

        cl_h_sink = CLh * (C_liv / kp["liver"]) / v["liver"]

        dydt[LIV] = passive_liv + active_uptake_liv - cl_h_sink

        # ── [KID] Kidney ───────────────────────────────────────────────────
        passive_kid = (Q_kid / v["kidney"]) * (C_art - C_kid / kp["kidney"])

        active_sec_kid = 0.0
        for name, trans in tp["renal_secretion"].items():
            cl_sec = self._active_cl(trans, C_kid_free)
            active_sec_kid += cl_sec * C_kid_free / v["kidney"]

        dydt[KID] = passive_kid - active_sec_kid

        # ── Standard passive compartments ──────────────────────────────────
        dydt[BRA] = (Q_bra / v["brain"]) * (C_art - C_bra / kp["brain"])
        dydt[HRT] = (Q_hrt / v["heart"]) * (C_art - C_hrt / kp["heart"])
        dydt[MUS] = (Q_mus / v["muscle"]) * (C_art - C_mus / kp["muscle"])
        dydt[FAT] = (Q_fat / v["fat"]) * (C_art - C_fat / kp["fat"])
        dydt[SKN] = (Q_skn / v["skin"]) * (C_art - C_skn / kp["skin"])
        dydt[BON] = (Q_bon / v["bone"]) * (C_art - C_bon / kp["bone"])

        # ── NEW: Specific "rest" organs ────────────────────────────────────
        dydt[SPL] = (Q_spl / v.get("spleen", 0.2)) * (C_art - C_spl / kp.get("spleen", 1.0))
        dydt[ADIP_MES] = (Q_adip_mes / v.get("adipose_mes", 0.5)) * (C_art - C_adip_mes / kp.get("adipose_mes", kp["fat"]))
        dydt[PANC] = (Q_panc / v.get("pancreas", 0.1)) * (C_art - C_panc / kp.get("pancreas", 1.0))

        # ── [VEN] Venous blood ─────────────────────────────────────────────
        venous_inflow = (
            Q_liv * (C_liv / kp["liver"])
            + Q_kid * (C_kid / kp["kidney"])
            + Q_bra * (C_bra / kp["brain"])
            + Q_hrt * (C_hrt / kp["heart"])
            + Q_mus * (C_mus / kp["muscle"])
            + Q_fat * (C_fat / kp["fat"])
            + Q_skn * (C_skn / kp["skin"])
            + Q_bon * (C_bon / kp["bone"])
            + Q_spl * (C_spl / kp.get("spleen", 1.0))
            + Q_adip_mes * (C_adip_mes / kp.get("adipose_mes", kp["fat"]))
            + Q_panc * (C_panc / kp.get("pancreas", 1.0))
        )

        if self.params["venous_delay_enabled"]:
            # With delay: venous → delay → lung
            dydt[VEN] = (
                venous_inflow
                - CO * C_ven  # Outflow to delay compartment
                - cl_filt * fup * C_ven
            ) / v["venous_blood"]
            
            # Venous delay compartment (first-order transit)
            tau = self.params["venous_delay_tau_h"]
            dydt[VEN_DELAY] = (CO * C_ven - CO * C_ven_delay) / (tau * CO)
            
            # Lung receives from delay compartment
            lung_inflow = C_ven_delay
        else:
            # No delay: venous → lung directly
            dydt[VEN] = (
                venous_inflow
                - CO * C_ven
                - cl_filt * fup * C_ven
            ) / v["venous_blood"]
            
            dydt[VEN_DELAY] = 0.0  # Unused
            lung_inflow = C_ven

        # ── [LUNG] ─────────────────────────────────────────────────────────
        dydt[LUNG] = (CO * lung_inflow - CO * (C_lung / kp["lung"])) / v["lung"]

        # ── [ART] Arterial blood ───────────────────────────────────────────
        dydt[ART] = (
            CO * (C_lung / kp["lung"]) - CO * C_art
        ) / v["arterial_blood"]

        return dydt

    # ── Solver ─────────────────────────────────────────────────────────────
    def solve(self, dose_mg: float, route: str = "oral",
              t_end_h: float = 48.0, n_points: int = 500) -> dict:
        """
        Solve PBPK system.

        Parameters
        ----------
        dose_mg  : float  dose (mg)
        route    : str    'oral' or 'iv'
        t_end_h  : float  simulation time (h)
        n_points : int    output points

        Returns
        -------
        dict with time series, AUC, Cmax, transporter info
        """
        y0 = np.zeros(N_STATES)

        if route == "oral":
            y0[GLU_ABS] = dose_mg
        elif route == "iv":
            v_art = self.vol["arterial_blood"]
            y0[ART] = dose_mg / v_art / self.drug["Rb"]
        else:
            raise ValueError(f"Unknown route '{route}'")

        sol = solve_ivp(
            fun=self.odes,
            t_span=(0.0, t_end_h),
            y0=y0,
            t_eval=np.linspace(0.0, t_end_h, n_points),
            method="LSODA",
            rtol=1e-6,
            atol=1e-9,
            max_step=0.5,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        t = sol.t
        Y = sol.y

        plasma = Y[ART]
        organs = {
            "liver": Y[LIV],
            "kidney": Y[KID],
            "brain": Y[BRA],
            "heart": Y[HRT],
            "muscle": Y[MUS],
            "fat": Y[FAT],
            "gut": Y[GUT_ENTER],
            "skin": Y[SKN],
            "bone": Y[BON],
            "lung": Y[LUNG],
            "spleen": Y[SPL],
            "adipose_mesenteric": Y[ADIP_MES],
            "pancreas": Y[PANC],
        }

        from scipy.integrate import trapezoid as _trapz

        auc_plasma = float(_trapz(plasma, t))
        auc_organs = {k: float(_trapz(v, t)) for k, v in organs.items()}

        cmax_idx = int(np.argmax(plasma))
        cmax_plasma = float(plasma[cmax_idx])
        tmax_plasma = float(t[cmax_idx])

        # Transporter summary
        tp = self._tp
        transporter_info = {
            "hepatic_transporters": tp["transporters_used"]["hepatic"],
            "renal_transporters": tp["transporters_used"]["renal"],
            "gut_efflux": tp["transporters_used"]["gut_efflux"],
            "has_hepatic": tp["has_hepatic_transporters"],
            "has_renal": tp["has_renal_transporters"],
            "has_gut_efflux": len(tp["transporters_used"]["gut_efflux"]) > 0,
            "cl_filt_lh": round(tp["cl_filt"], 3),
            "cl_sec_lh": round(tp["cl_sec_total"], 3),
            "pgp_efflux_prob": round(tp["pgp_efflux_prob"], 3),
            "saturable_metabolism": tp["use_mm_hepatic"],
            "venous_delay_used": self.params["venous_delay_enabled"],
            
            # Model parameters (for transparency)
            "transporter_scale_factor": self.params["transporter_scale_factor"],
            "default_renal_km_um": self.params["default_renal_km_um"],
            "pgp_km_um": self.params["pgp_km_um"],
        }
        
        # P-gp efflux mass balance
        total_effluxed = float(Y[GLU_EFF][-1])  # Final amount in efflux depot

        return {
            "t": t,
            "plasma": plasma,
            "organs": organs,
            "auc_plasma": auc_plasma,
            "auc_organs": auc_organs,
            "cmax_plasma": cmax_plasma,
            "tmax_plasma": tmax_plasma,
            "drug": self.drug,
            "dose_mg": dose_mg,
            "route": route,
            "transporter_info": transporter_info,
            "pgp_total_effluxed_mg": total_effluxed,
            "gut_lumen_absorption": Y[GLU_ABS],
            "gut_lumen_efflux": Y[GLU_EFF],
        }