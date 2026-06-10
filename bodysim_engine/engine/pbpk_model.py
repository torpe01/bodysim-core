"""
pbpk_model.py — Multi-compartment PBPK ODE model for BodySim.
v5.0 — BUG FIXES (Bug 1: hepatic MM fallback; Bug 2: GFR anatomy) +
       GAP IMPLEMENTATIONS (Gap 1: zwitterion two-pKa; Gap 2: biliary EHC)

Refactored (modular split):
  acat_module.py    — ACATAbsorptionModule   (luminal transit, ionization, gut-wall met.)
  hepatic_module.py — HepaticClearanceModule (OATP-PFT, TMDD-QSS, biliary EHC, dual-path CYP)
  renal_module.py   — RenalEliminationModule (GFR anatomy fix, tubular reabs, ER cap)

The odes() method is now a clean orchestrator: it calls each module's flux
method and assembles dydt from the returned values.  All arithmetic, parameter
values, variable names, and state-vector indices are numerically identical to
the original monolithic implementation — simulation outputs are unchanged.

v5.0 Additional Fix — _build_transporter_params() Km_hepatic sentinel:
  The lingering `self.drug.get("Km_hepatic", 20.0)` default has been replaced
  with a None-guarded dual-check matching the logic in hepatic_module.py Bug Fix 1.
  Both Vmax_hepatic AND Km_hepatic must be explicitly present and positive for
  use_mm_hepatic (stored in the tp dict) to be set True.  This ensures the
  transporter_info reporting layer (use_mm_hepatic → "MM" vs "linear") is
  consistent with the actual ODE branch selected by HepaticClearanceModule.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .acat_module    import ACATAbsorptionModule,    ACAT_SA_FOLDING, _ACAT_SA_NORM, \
                            ACAT_SA_FACTORS, ACAT_SEGMENT_NAMES, ACAT_TRANSIT_TIMES, \
                            ACAT_PH_DEFAULT, N_ACAT_SEGMENTS
from .hepatic_module import HepaticClearanceModule
from .renal_module   import RenalEliminationModule

# ── State vector indices ───────────────────────────────────────────────────────
ART  = 0;  VEN  = 1;  VEN_DELAY = 2; LUNG = 3
LIV_VASC = 4;  LIV_TISS = 5;  KID  = 6;  BRA  = 7
HRT  = 8;  MUS  = 9;  FAT  = 10
GUT_ENTER = 11;  SKN  = 12; BON  = 13
SPL  = 14; ADIP_MES = 15; PANC = 16
GLU_ABS  = 17; GLU_EFF = 18
DOSE_DEPOT = 19
LUMEN_BASE = 20
BILE       = 27
N_STATES   = 19 + 1 + N_ACAT_SEGMENTS + 1   # 28 total

ORGAN_NAMES = [
    "arterial", "venous", "venous_delay", "lung",
    "liver_vascular", "liver_tissue", "kidney", "brain",
    "heart", "muscle", "fat",
    "gut_enterocyte", "skin", "bone",
    "spleen", "adipose_mesenteric", "pancreas",
    "gut_lumen_absorption", "gut_lumen_efflux",
    "dose_depot", "stomach", "duodenum", "jejunum_1", "jejunum_2",
    "ileum_1", "ileum_2", "cecum_colon",
    "bile_pool",
]

TISSUE_COMPARTMENTS = {
    LIV_VASC: "liver_vasc", LIV_TISS: "liver_tiss", KID: "kidney", BRA: "brain",
    HRT: "heart", MUS: "muscle", FAT: "fat",
    GUT_ENTER: "gut", SKN: "skin", BON: "bone",
    SPL: "spleen", ADIP_MES: "adipose_mes", PANC: "pancreas",
}

HEPATIC_UPTAKE_TRANSPORTERS  = {"OATP1B1", "OATP1B3", "OCT1"}
RENAL_SECRETION_TRANSPORTERS = {"OCT2", "OAT1", "OAT3"}
GUT_EFFLUX_TRANSPORTERS      = {"MRP2", "P-gp"}


class PBPKModel(ACATAbsorptionModule, HepaticClearanceModule, RenalEliminationModule):
    """
    28-compartment PBPK model (v5.0) with modular organ sub-systems.

    Inherits:
      ACATAbsorptionModule   — _build_acat_params(), calculate_gut_flux()
      HepaticClearanceModule — calculate_liver_flux()
      RenalEliminationModule — calculate_kidney_flux()

    Parameters
    ----------
    drug    : dict  from admet.build_drug_profile()
    volumes : dict  from physiology.scale_physiology()
    flows   : dict  from physiology.scale_physiology()
    params  : dict  subject + model configuration (see _set_default_params)
    """

    def __init__(self, drug, volumes, flows, params):
        self.drug   = drug
        self.vol    = volumes
        self.flow   = flows
        self.params = self._set_default_params(params)
        self._validate()
        self._acat = self._build_acat_params(self.drug, self.flow)
        self._tp   = self._build_transporter_params()

    # ── Parameter defaults ─────────────────────────────────────────────────────
    def _set_default_params(self, params):
        defaults = {
            "transporter_scale_factor":    0.3,
            "default_renal_km_um":         50.0,
            "pgp_efflux_max":              0.6,
            "pgp_efflux_floor":            0.4,
            "venous_delay_enabled":        False,
            "venous_delay_tau_h":          0.05,
            "pgp_km_um":                   30.0,
            "pgp_vmax_scale":              1.0,
            "pgp_vmax_base":               100.0,
            "rest_flow_split_spleen":      0.3,
            "rest_flow_split_adipose_mes": 0.4,
            "rest_flow_split_pancreas":    0.3,
            "liver_CL_pd":                 None,
            "urine_ph":                    6.0,
        }
        merged = defaults.copy()
        merged.update(params)

        if "egfr" not in merged:
            raise ValueError("params must include 'egfr' (eGFR in mL/min)")
        if "cyp3a4_activity" not in merged:
            raise ValueError("params must include 'cyp3a4_activity'")

        if merged["liver_CL_pd"] is None:
            logp               = self.drug.get("logp",               0.0)
            clint              = self.drug.get("CLint",               0.0)
            is_uptake_substrate = self.drug.get("is_uptake_substrate", False)

            if logp < 0:
                cl_pd = float(np.clip(5.0 * np.exp(logp / 2.0), 1.0, 10.0))
            else:
                cl_pd = float(np.clip(10.0 * (10 ** logp), 10.0, 1500.0))

            if is_uptake_substrate:
                cl_pd *= 3.0
            if clint > 50.0:
                cl_pd = max(cl_pd, 10000.0)

            merged["liver_CL_pd"] = cl_pd

        return merged

    # ── Validation ─────────────────────────────────────────────────────────────
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

        for organ in ["spleen", "adipose_mes", "pancreas"]:
            if organ not in self.drug["kp"]:
                if organ == "adipose_mes":
                    self.drug["kp"][organ] = self.drug["kp"]["fat"]
                else:
                    self.drug["kp"][organ] = 1.0

    # ── Transporter parameter pre-computation ──────────────────────────────────
    def _build_transporter_params(self) -> dict:
        """
        Convert admet.py transporter data into ODE-ready parameters.
        Unchanged from the original monolithic implementation.
        """
        mw  = self.drug.get("mw",  300.0)
        fup = self.drug["fup"]
        sc  = self.params["transporter_scale_factor"]

        hep_raw = self.drug.get("hepatic_transport", {})
        hepatic_uptake = {}
        for name, data in hep_raw.items():
            if name not in HEPATIC_UPTAKE_TRANSPORTERS:
                continue
            vmax_eff = float(data["Vmax"])
            km_um    = float(data["Km"])
            prob     = float(data["probability"])
            if km_um <= 0 or vmax_eff <= 0:
                continue
            transporter_scale = float(data.get("default_scale", sc))
            km_mgl = km_um * mw / 1000.0
            cl_linear = (vmax_eff / km_um) * prob * transporter_scale
            hepatic_uptake[name] = {
                "cl_linear":    cl_linear,
                "Km_mgl":       km_mgl,
                "Vmax_eff":     vmax_eff * prob * transporter_scale,
                "scale_factor": transporter_scale,
            }
        cl_hep_uptake_total = sum(t["cl_linear"] for t in hepatic_uptake.values())

        ren_raw = self.drug.get("renal_transport", {})
        renal_secretion      = {}
        cl_sec_linear_total  = 0.0
        for name, data in ren_raw.items():
            if name not in RENAL_SECRETION_TRANSPORTERS:
                continue
            vmax_eff = float(data["Vmax"])
            km_um    = float(data["Km"])
            prob     = float(data["probability"])
            if km_um <= 0 or vmax_eff <= 0:
                continue
            transporter_scale = float(data.get("default_scale", sc))
            km_mgl    = km_um * mw / 1000.0
            cl_linear = (vmax_eff / km_um) * prob * transporter_scale
            cl_sec_linear_total += cl_linear
            renal_secretion[name] = {
                "cl_linear":    cl_linear,
                "Km_mgl":       km_mgl,
                "Vmax_eff":     vmax_eff * prob * transporter_scale,
                "scale_factor": transporter_scale,
            }

        egfr        = self.params["egfr"]
        gfr_lh      = egfr * 60.0 / 1000.0
        cl_filt     = gfr_lh * fup
        cl_renal_total = self.drug["CLrenal"]
        cl_sec_target  = max(0.0, cl_renal_total - cl_filt)

        if cl_sec_linear_total > 1e-9 and cl_sec_target > 0:
            sec_scale = cl_sec_target / cl_sec_linear_total
            for name in renal_secretion:
                renal_secretion[name]["cl_linear"] *= sec_scale
                renal_secretion[name]["Vmax_eff"]  *= sec_scale
            cl_sec_linear_total = cl_sec_target
        elif cl_sec_linear_total < 1e-9 and cl_sec_target > 0:
            default_km = self.params["default_renal_km_um"]
            renal_secretion["_generic"] = {
                "cl_linear": cl_sec_target,
                "Km_mgl":    mw * default_km / 1000.0,
                "Vmax_eff":  cl_sec_target,
            }
            cl_sec_linear_total = cl_sec_target

        Rb              = self.drug.get("Rb", 1.0)
        Q_kidney_blood  = self.flow["kidney"]
        Q_kidney_plasma = Q_kidney_blood / max(Rb, 0.01)

        if cl_sec_linear_total > 1e-9:
            _er_sec = (
                (fup * cl_sec_linear_total / Rb)
                / (Q_kidney_plasma + fup * cl_sec_linear_total / Rb)
            )
            cl_sec_plasma_cap_lh = float(Q_kidney_plasma * _er_sec)
        else:
            cl_sec_plasma_cap_lh = 0.0

        pka       = self.drug.get("pka",       None)
        drug_type = self.drug.get("drug_type", "neutral")
        logp      = self.drug.get("logp",      0.0)
        urine_ph  = float(self.params.get("urine_ph", 6.0))

        Q_tubular_water_lh = 0.01 * gfr_lh

        if pka is None or drug_type == "neutral":
            f_neutral_urine = 1.0
        elif drug_type == "acidic":
            f_neutral_urine = float(1.0 / (1.0 + 10.0 ** (urine_ph - pka)))
        elif drug_type == "basic":
            f_neutral_urine = float(1.0 / (1.0 + 10.0 ** (pka - urine_ph)))
        elif drug_type == "zwitterion":
            if isinstance(pka, dict) and "acid" in pka and "base" in pka:
                _f_acid_u = 1.0 / (1.0 + 10.0 ** (urine_ph - float(pka["acid"])))
                _f_base_u = 1.0 / (1.0 + 10.0 ** (float(pka["base"]) - urine_ph))
                f_neutral_urine = float(_f_acid_u * _f_base_u)
            else:
                f_neutral_urine = 0.02
        else:
            f_neutral_urine = 1.0
        f_neutral_urine = float(np.clip(f_neutral_urine, 0.0, 1.0))

        k_perm = float(np.clip((logp - (-1.0)) / (1.0 - (-1.0)), 0.0, 1.0))
        cl_reabsorption_lh = float(Q_tubular_water_lh * f_neutral_urine * k_perm)

        mrp2_prob = 0.0
        pgp_prob  = 0.0
        for name, data in hep_raw.items():
            if name == "MRP2":
                mrp2_prob = float(data.get("probability", 0.0))
        trans_info = self.drug.get("transporters", {})
        pgp_prob   = float(trans_info.get("pgp_prob", pgp_prob))
        efflux_prob = max(mrp2_prob, pgp_prob)

        pgp_km_um   = self.params["pgp_km_um"]
        pgp_km_mgl  = pgp_km_um * mw / 1000.0
        pgp_vmax_base = self.params["pgp_vmax_base"]
        pgp_vmax_eff  = pgp_vmax_base * efflux_prob * self.params["pgp_vmax_scale"]
        v_gut_enterocyte = self.vol.get("gut", 0.5) * 0.1
        k_efflux_base = (
            (pgp_vmax_eff / pgp_km_um) * sc / v_gut_enterocyte
            if v_gut_enterocyte > 0 else 0.0
        )

        # ── Hepatic MM kinetics sentinel (mirrors Bug Fix 1 in hepatic_module.py) ──
        # Both Vmax_hepatic AND Km_hepatic must be explicitly present and positive
        # for MM kinetics to activate.  The former default of Km=20.0 µM caused
        # spurious saturation for every drug with CLint > 0; the correct fallback
        # is linear clearance (use_mm_hep = False).
        # The stored km_hep_mgl is used only by the legacy _hepatic_clearance()
        # helper and by the transporter_info dict (reporting only — not the ODE).
        _km_hep_raw  = self.drug.get("Km_hepatic")    # None when not calibrated
        _vmax_raw_tp = self.drug.get("Vmax_hepatic")  # None when not calibrated

        _km_hep_valid   = (_km_hep_raw  is not None and float(_km_hep_raw)  > 1e-9)
        _vmax_hep_valid = (_vmax_raw_tp is not None and float(_vmax_raw_tp) > 0.0)

        if _vmax_hep_valid and _km_hep_valid:
            vmax_hep_tp = float(_vmax_raw_tp)
            km_hep_mgl  = float(_km_hep_raw) * mw / 1000.0
            use_mm_hep  = True
        else:
            # Strictly linear fallback: no default Km, no saturation artefact.
            vmax_hep_tp = 0.0
            km_hep_mgl  = 0.0   # sentinel — unused in linear branch
            use_mm_hep  = False

        phaseII_raw = self.drug.get("phaseII_kinetics", None)
        sult_params = None
        if phaseII_raw is not None and "sult" in phaseII_raw:
            _s = phaseII_raw["sult"]
            vmax_s = float(_s.get("Vmax_mg_h", 0.0))
            km_s   = float(_s.get("Km_mg_L",   1.0))
            if vmax_s > 0.0 and km_s > 1e-12:
                sult_params = {"Vmax_mg_h": vmax_s, "Km_mg_L": km_s}
        ugt_params = None
        if phaseII_raw is not None and "ugt" in phaseII_raw:
            _u = phaseII_raw["ugt"]
            vmax_u = float(_u.get("Vmax_mg_h", 0.0))
            km_u   = float(_u.get("Km_mg_L",   1.0))
            if vmax_u > 0.0 and km_u > 1e-12:
                ugt_params = {"Vmax_mg_h": vmax_u, "Km_mg_L": km_u}

        return {
            "hepatic_uptake":         hepatic_uptake,
            "cl_hep_uptake_total":    cl_hep_uptake_total,
            "renal_secretion":        renal_secretion,
            "cl_filt":                cl_filt,
            "cl_sec_total":           cl_sec_linear_total,
            "cl_sec_plasma_cap_lh":   cl_sec_plasma_cap_lh,
            "cl_reabsorption_lh":     cl_reabsorption_lh,
            "urine_ph":               urine_ph,
            "f_neutral_urine":        f_neutral_urine,
            "k_perm":                 k_perm,
            "km_hep_mgl":             km_hep_mgl,
            "use_mm_hepatic":         use_mm_hep,
            "pgp_efflux_prob":        efflux_prob,
            "pgp_km_mgl":             pgp_km_mgl,
            "k_efflux_base":          k_efflux_base,
            "has_hepatic_transporters": len(hepatic_uptake) > 0,
            "has_renal_transporters":   any(k != "_generic" for k in renal_secretion),
            "transporters_used": {
                "hepatic":    list(hepatic_uptake.keys()),
                "renal":      [k for k in renal_secretion if k != "_generic"],
                "gut_efflux": ["P-gp/MRP2"] if efflux_prob > 0.1 else [],
            },
            "phaseII_sult": sult_params,
            "phaseII_ugt":  ugt_params,
        }

    # ── Legacy helpers kept for API compatibility ──────────────────────────────
    def _hepatic_clearance(self, C_liv: float) -> float:
        Q   = self.flow["liver_hepatic"] + self.flow["liver_portal"]
        fup = self.drug["fup"]
        Rb  = self.drug["Rb"]
        kp_liv = self.drug["kp"]["liver"]
        C_free = max(0.0, fup * C_liv / kp_liv)
        if self._tp["use_mm_hepatic"]:
            km  = self._tp["km_hep_mgl"]
            sat = km / (km + C_free) if (km + C_free) > 0 else 1.0
            cl_int_eff = self.drug["CLint"] * float(sat)
        else:
            cl_int_eff = self.drug["CLint"]
        Eh = (fup * cl_int_eff / Rb) / (Q + fup * cl_int_eff / Rb)
        return Q * float(np.clip(Eh, 0.0, 0.99))

    @staticmethod
    def _active_cl(transporter_params: dict, C_free: float) -> float:
        km = transporter_params["Km_mgl"]
        cl = transporter_params["cl_linear"]
        if km <= 0 or cl <= 0:
            return 0.0
        sat = km / (km + C_free) if (km + C_free) > 0 else 1.0
        return float(cl * sat)

    def _pgp_efflux_rate(self, C_enterocyte: float) -> float:
        if self._tp["pgp_efflux_prob"] < 0.1:
            return 0.0
        km    = self._tp["pgp_km_mgl"]
        k_base= self._tp["k_efflux_base"]
        fup   = self.drug["fup"]
        kp_gut= self.drug["kp"]["gut"]
        C_free = max(0.0, fup * C_enterocyte / kp_gut)
        rate   = k_base * km * C_free / (km + C_free) if (km + C_free) > 0 else 0.0
        return float(rate)

    # ── ODE orchestrator ───────────────────────────────────────────────────────
    def odes(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        28-compartment PBPK ODE system — clean orchestrator.

        Delegates:
          ACAT luminal loop     → ACATAbsorptionModule.calculate_gut_flux()
          Liver ODE             → HepaticClearanceModule.calculate_liver_flux()
          Kidney ODE            → RenalEliminationModule.calculate_kidney_flux()
          All other compartments assembled here from their standard perfusion equations.
        """
        y = np.maximum(y, 0.0)

        # ── Unpack state vector ────────────────────────────────────────────
        C_art        = y[ART]
        C_ven        = y[VEN]
        C_ven_delay  = y[VEN_DELAY]
        C_lung       = y[LUNG]
        C_liv_vasc   = y[LIV_VASC]
        C_liv_tiss   = y[LIV_TISS]
        C_kid        = y[KID]
        C_bra        = y[BRA]
        C_hrt        = y[HRT]
        C_mus        = y[MUS]
        C_fat        = y[FAT]
        C_gut_enter  = y[GUT_ENTER]
        C_skn        = y[SKN]
        C_bon        = y[BON]
        C_spl        = y[SPL]
        C_adip_mes   = y[ADIP_MES]
        C_panc       = y[PANC]
        A_glu_abs    = y[GLU_ABS]
        A_glu_eff    = y[GLU_EFF]
        A_dose_depot = y[DOSE_DEPOT]
        M_lumen      = y[LUMEN_BASE: LUMEN_BASE + N_ACAT_SEGMENTS]
        A_bile       = y[BILE]

        v   = self.vol
        q   = self.flow
        kp  = self.drug["kp"]
        fup = self.drug["fup"]
        Rb  = self.drug["Rb"]
        tp  = self._tp
        ac  = self._acat
        ka  = self.drug["ka"]

        # ── Flows ──────────────────────────────────────────────────────────
        Q_ha  = q["liver_hepatic"]
        Q_pv  = q["liver_portal"]
        Q_liv = Q_ha + Q_pv
        Q_kid = q["kidney"]
        Q_bra = q["brain"]
        Q_hrt = q["heart"]
        Q_mus = q["muscle"]
        Q_fat = q["fat"]
        Q_gut = q["gut"]
        Q_skn = q["skin"]
        Q_bon = q["bone"]
        Q_rest_total = q.get("rest", 0.0)
        Q_spl     = Q_rest_total * self.params["rest_flow_split_spleen"]
        Q_adip_mes= Q_rest_total * self.params["rest_flow_split_adipose_mes"]
        Q_panc    = Q_rest_total * self.params["rest_flow_split_pancreas"]
        CO = q["cardiac_output"]

        # ── Blood-unit concentrations ──────────────────────────────────────
        C_art_blood            = C_art * Rb
        C_ven_blood            = C_ven * Rb
        C_ven_delay_blood      = C_ven_delay * Rb
        C_gut_blood_out        = C_gut_enter / kp["gut"]
        C_gut_blood_out_blood  = C_gut_blood_out * Rb
        C_liv_vasc_blood       = C_liv_vasc * Rb

        # Needed for ACAT bile-secretion: pre-compute C_tissue_free
        C_tissue_free_prelim = fup * C_liv_tiss / kp["liver"]

        # ── [GLU_ABS] Legacy (backward compat) ────────────────────────────
        dydt = np.zeros(N_STATES)
        dydt[GLU_ABS] = 0.0

        # ── [GLU_EFF] Gut efflux depot ─────────────────────────────────────
        pgp_efflux_to_lumen = self._pgp_efflux_rate(C_gut_enter) * v.get("gut", 0.5)
        ka_reabs = ka * 0.5
        dydt[GLU_EFF] = pgp_efflux_to_lumen - ka_reabs * A_glu_eff

        # ── ACAT Module: luminal transit + absorption ──────────────────────
        ph_profiles = {
            "A_dose_depot":   A_dose_depot,
            "A_glu_eff":      A_glu_eff,
            "M_lumen":        M_lumen,
            "A_bile":         A_bile,
            "ka_reabs":       ka_reabs,
            "C_tissue_free":  C_tissue_free_prelim,
        }

        (total_abs_flux,
         dydt_lumen,
         dydt_dose_depot,
         dydt_bile_prelim,
         J_bile_secretion) = self.calculate_gut_flux(
            y            = y,
            drug         = self.drug,
            acat_params  = ac,
            ph_profiles  = ph_profiles,
        )

        dydt[LUMEN_BASE: LUMEN_BASE + N_ACAT_SEGMENTS] = dydt_lumen
        dydt[DOSE_DEPOT] = dydt_dose_depot

        # ── [GUT_ENTER] ────────────────────────────────────────────────────
        dydt[GUT_ENTER] = (
            (Q_gut / v["gut"]) * (C_art_blood - C_gut_blood_out * Rb)
            + total_abs_flux / v["gut"]
            - self._pgp_efflux_rate(C_gut_enter)
        )

        # ── Hepatic Module ─────────────────────────────────────────────────
        hep_extra = {
            "C_liv_vasc":            C_liv_vasc,
            "C_liv_tiss":            C_liv_tiss,
            "C_art_blood":           C_art_blood,
            "C_gut_blood_out_blood": C_gut_blood_out_blood,
            "C_liv_vasc_blood":      C_liv_vasc_blood,
            "J_bile_secretion":      J_bile_secretion,
        }

        hep = self.calculate_liver_flux(
            y            = y,
            drug         = self.drug,
            liver_volume = v["liver"],
            C_art        = C_art,
            fup          = fup,
            flow_rate    = q,
            tp           = tp,
            params       = self.params,
            extra        = hep_extra,
        )

        dydt[LIV_VASC] = hep["dydt_liv_vasc"]
        dydt[LIV_TISS] = hep["dydt_liv_tiss"]

        # ── [BILE] Gallbladder / Bile Duct Pool (Gap 2, v5.0) ─────────────
        # The ACAT module already computed J_bile_secretion and J_bile_emptying.
        # dydt_bile_prelim from calculate_gut_flux = J_bile_secretion - k_bile_empty × A_bile
        dydt[BILE] = dydt_bile_prelim

        # ── Renal Module ───────────────────────────────────────────────────
        dydt[KID] = self.calculate_kidney_flux(
            y             = y,
            drug          = self.drug,
            kidney_volume = v["kidney"],
            C_art         = C_art,
            cl_filt       = tp["cl_filt"],
            tp            = tp,
            flow_rate     = q,
            extra         = {"C_kid": C_kid, "C_art_blood": C_art_blood},
        )

        # ── Standard passive tissue compartments ──────────────────────────
        dydt[BRA] = (Q_bra / v["brain"])  * (C_art_blood - (C_bra  / kp["brain"])  * Rb)
        dydt[HRT] = (Q_hrt / v["heart"])  * (C_art_blood - (C_hrt  / kp["heart"])  * Rb)
        dydt[MUS] = (Q_mus / v["muscle"]) * (C_art_blood - (C_mus  / kp["muscle"]) * Rb)
        dydt[FAT] = (Q_fat / v["fat"])    * (C_art_blood - (C_fat  / kp["fat"])    * Rb)
        dydt[SKN] = (Q_skn / v["skin"])   * (C_art_blood - (C_skn  / kp["skin"])   * Rb)
        dydt[BON] = (Q_bon / v["bone"])   * (C_art_blood - (C_bon  / kp["bone"])   * Rb)

        dydt[SPL]      = (Q_spl      / v.get("spleen",      0.2)) * (C_art_blood - (C_spl      / kp.get("spleen",     1.0))           * Rb)
        dydt[ADIP_MES] = (Q_adip_mes / v.get("adipose_mes", 0.5)) * (C_art_blood - (C_adip_mes / kp.get("adipose_mes", kp["fat"]))     * Rb)
        dydt[PANC]     = (Q_panc     / v.get("pancreas",    0.1)) * (C_art_blood - (C_panc     / kp.get("pancreas",   1.0))           * Rb)

        # ── [VEN] Venous blood ─────────────────────────────────────────────
        venous_inflow = (
            Q_liv       * C_liv_vasc_blood
            + Q_kid     * (C_kid      / kp["kidney"])          * Rb
            + Q_bra     * (C_bra      / kp["brain"])           * Rb
            + Q_hrt     * (C_hrt      / kp["heart"])           * Rb
            + Q_mus     * (C_mus      / kp["muscle"])          * Rb
            + Q_fat     * (C_fat      / kp["fat"])             * Rb
            + Q_skn     * (C_skn      / kp["skin"])            * Rb
            + Q_bon     * (C_bon      / kp["bone"])            * Rb
            + Q_spl     * (C_spl      / kp.get("spleen",     1.0))          * Rb
            + Q_adip_mes* (C_adip_mes / kp.get("adipose_mes", kp["fat"]))   * Rb
            + Q_panc    * (C_panc     / kp.get("pancreas",   1.0))          * Rb
        )

        if self.params["venous_delay_enabled"]:
            dydt[VEN] = (venous_inflow - CO * C_ven_blood) / v["venous_blood"]
            tau = self.params["venous_delay_tau_h"]
            dydt[VEN_DELAY] = (CO * C_ven_blood - CO * C_ven_delay_blood) / (tau * CO)
            lung_inflow = C_ven_delay_blood
        else:
            dydt[VEN]       = (venous_inflow - CO * C_ven_blood) / v["venous_blood"]
            dydt[VEN_DELAY] = 0.0
            lung_inflow     = C_ven_blood

        # ── [LUNG] ─────────────────────────────────────────────────────────
        dydt[LUNG] = (CO * lung_inflow - CO * (C_lung / kp["lung"]) * Rb) / v["lung"]

        # ── [ART] Arterial blood ───────────────────────────────────────────
        dydt[ART] = (
            CO * (C_lung / kp["lung"]) * Rb - CO * C_art_blood
        ) / v["arterial_blood"]

        return dydt

    # ── Solver ─────────────────────────────────────────────────────────────────
    def solve(self, dose_mg: float, route: str = "oral",
              t_end_h: float = 48.0, n_points: int = 500) -> dict:
        """
        Solve the PBPK ODE system and return time-series + NCA summary.

        Parameters
        ----------
        dose_mg  : float  dose (mg)
        route    : str    'oral' or 'iv'
        t_end_h  : float  simulation end time (h)
        n_points : int    number of output points
        """
        y0 = np.zeros(N_STATES)
        if route == "oral":
            y0[DOSE_DEPOT] = dose_mg
            y0[GLU_ABS]    = 0.0
        elif route == "iv":
            y0[ART] = dose_mg / self.vol["arterial_blood"] / self.drug["Rb"]
        else:
            raise ValueError(f"Unknown route '{route}'")

        sol = solve_ivp(
            fun     = self.odes,
            t_span  = (0.0, t_end_h),
            y0      = y0,
            t_eval  = np.linspace(0.0, t_end_h, n_points),
            method  = "LSODA",
            rtol    = 1e-6,
            atol    = 1e-9,
            max_step= 0.5,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        t  = sol.t
        Y  = sol.y

        plasma = Y[ART]
        organs = {
            "liver_vascular":    Y[LIV_VASC],
            "liver_tissue":      Y[LIV_TISS],
            "kidney":            Y[KID],
            "brain":             Y[BRA],
            "heart":             Y[HRT],
            "muscle":            Y[MUS],
            "fat":               Y[FAT],
            "gut":               Y[GUT_ENTER],
            "skin":              Y[SKN],
            "bone":              Y[BON],
            "lung":              Y[LUNG],
            "spleen":            Y[SPL],
            "adipose_mesenteric":Y[ADIP_MES],
            "pancreas":          Y[PANC],
        }

        from scipy.integrate import trapezoid as _trapz
        auc_plasma = float(_trapz(plasma, t))
        auc_organs = {k: float(_trapz(v, t)) for k, v in organs.items()}
        cmax_idx   = int(np.argmax(plasma))
        cmax_plasma= float(plasma[cmax_idx])
        tmax_plasma= float(t[cmax_idx])

        tp = self._tp
        transporter_info = {
            "hepatic_transporters":  tp["transporters_used"]["hepatic"],
            "renal_transporters":    tp["transporters_used"]["renal"],
            "gut_efflux":            tp["transporters_used"]["gut_efflux"],
            "has_hepatic":           tp["has_hepatic_transporters"],
            "has_renal":             tp["has_renal_transporters"],
            "has_gut_efflux":        len(tp["transporters_used"]["gut_efflux"]) > 0,
            "cl_filt_lh":            round(tp["cl_filt"], 3),
            "cl_sec_lh":             round(tp["cl_sec_total"], 3),
            "pgp_efflux_prob":       round(tp["pgp_efflux_prob"], 3),
            "saturable_metabolism":  tp["use_mm_hepatic"],
            "venous_delay_used":     self.params["venous_delay_enabled"],
            "transporter_scale_factor": self.params["transporter_scale_factor"],
            "default_renal_km_um":   self.params["default_renal_km_um"],
            "pgp_km_um":             self.params["pgp_km_um"],
            "liver_cl_pd":           self.params.get("liver_CL_pd", 10.0),
            "liver_model":           "permeability-limited 2-compartment (v2.3+)",
            "cl_reab_lh":            round(tp.get("cl_reabsorption_lh", 0.0), 4),
            "cl_sec_plasma_cap_lh":  round(tp.get("cl_sec_plasma_cap_lh", 0.0), 4),
            "urine_ph":              tp.get("urine_ph", self.params.get("urine_ph", 6.0)),
            "f_neutral_urine":       round(tp.get("f_neutral_urine", 1.0), 4),
            "k_perm_reab":           round(tp.get("k_perm", 0.0), 4),
            "F_gut_scalar":          round(self._acat.get("F_gut_scalar", 1.0), 6),
            "fu_gut":                float(self.drug.get("fu_gut", 1.0)),
            "CLint_gut_cyp3a4":      float(self.drug.get("CLint_gut_cyp3a4", 0.0)),
            "has_phaseII_sult":      tp.get("phaseII_sult") is not None,
            "has_phaseII_ugt":       tp.get("phaseII_ugt")  is not None,
            "phaseII_sult_params":   tp.get("phaseII_sult"),
            "phaseII_ugt_params":    tp.get("phaseII_ugt"),
            "hepatic_kinetics_mode": "MM" if tp["use_mm_hepatic"] else "linear",
            "cl_bile_lh":            float(self.drug.get("cl_bile_lh",     0.0)),
            "k_bile_empty_h":        float(self.drug.get("k_bile_empty_h", 0.05)),
            "f_reabs_bile":          float(self.drug.get("f_reabs_bile",   0.0)),
            "ehc_active":            self.drug.get("cl_bile_lh", 0.0) > 0.0,
        }

        total_effluxed = float(Y[GLU_EFF][-1])
        nca = self._calculate_nca(t, plasma, dose_mg, route)

        return {
            "t":             t,
            "plasma":        plasma,
            "organs":        organs,
            "auc_plasma":    auc_plasma,
            "auc_organs":    auc_organs,
            "cmax_plasma":   cmax_plasma,
            "tmax_plasma":   tmax_plasma,
            "drug":          self.drug,
            "dose_mg":       dose_mg,
            "route":         route,
            "transporter_info":        transporter_info,
            "pgp_total_effluxed_mg":   total_effluxed,
            "gut_lumen_absorption":    Y[GLU_ABS],
            "gut_lumen_efflux":        Y[GLU_EFF],
            "bile_pool_mg":            Y[BILE],
            "t_half_h":         nca["t_half_h"],
            "lambda_z_per_h":   nca["lambda_z_per_h"],
            "cl_total_lh":      nca["cl_total_lh"],
            "vss_l":            nca["vss_l"],
            "mrt_h":            nca["mrt_h"],
            "auc0inf_mg_l_h":   nca["auc0inf_mg_l_h"],
        }

    # ── NCA helper ─────────────────────────────────────────────────────────────
    def _calculate_nca(self, t: np.ndarray, plasma: np.ndarray,
                       dose_mg: float, route: str) -> dict:
        """
        Non-compartmental analysis from a simulated plasma profile.
        Numerically identical to the original implementation.
        """
        from scipy.integrate import trapezoid as _trapz

        _SENTINEL = None
        auc_0_t   = float(_trapz(plasma, t))
        if auc_0_t < 1e-15:
            return {
                "t_half_h": _SENTINEL, "lambda_z_per_h": _SENTINEL,
                "cl_total_lh": _SENTINEL, "vss_l": _SENTINEL,
                "mrt_h": _SENTINEL, "auc0inf_mg_l_h": 0.0,
            }

        n        = len(t)
        c_floor  = 1e-3 * float(np.max(plasma))
        idx_start= max(int(0.80 * n), n - 50)
        t_win    = t[idx_start:]
        c_win    = plasma[idx_start:]
        mask     = c_win > c_floor

        lambda_z = _SENTINEL
        t_half   = _SENTINEL

        if int(np.sum(mask)) >= 3:
            try:
                log_c = np.log(c_win[mask])
                t_fit = t_win[mask]
                slope, _intercept = np.polyfit(t_fit, log_c, 1)
                if slope < -1e-6:
                    lambda_z = float(-slope)
                    t_half   = float(np.log(2.0) / lambda_z)
            except (np.linalg.LinAlgError, ValueError):
                pass

        c_last = float(c_win[mask][-1]) if (mask.any() and lambda_z is not None) \
                 else float(plasma[-1])

        auc_0_inf = float(auc_0_t + c_last / lambda_z) if lambda_z is not None \
                    else float(auc_0_t)

        cl_total  = float(dose_mg / auc_0_inf) if auc_0_inf > 1e-15 else _SENTINEL

        aumc_0_t  = float(_trapz(t * plasma, t))
        t_last    = float(t[-1])
        if lambda_z is not None:
            aumc_extra = (c_last / lambda_z ** 2) + (t_last * c_last / lambda_z)
            aumc_0_inf = aumc_0_t + aumc_extra
        else:
            aumc_0_inf = aumc_0_t

        mrt = float(aumc_0_inf / auc_0_inf) if auc_0_inf > 1e-15 else _SENTINEL

        vss = _SENTINEL
        if cl_total is not None and mrt is not None:
            if route == "iv":
                vss = float(cl_total * mrt)
            else:
                ka  = float(self.drug.get("ka", 1.5))
                mat = 1.0 / ka if ka > 1e-6 else 0.0
                mrt_iv_equiv = mrt - mat
                if mrt_iv_equiv > 0.0:
                    vss = float(cl_total * mrt_iv_equiv)

        return {
            "t_half_h":       round(t_half,   3) if t_half   is not None else _SENTINEL,
            "lambda_z_per_h": round(lambda_z, 6) if lambda_z is not None else _SENTINEL,
            "cl_total_lh":    round(cl_total, 4) if cl_total is not None else _SENTINEL,
            "vss_l":          round(vss,      3) if vss      is not None else _SENTINEL,
            "mrt_h":          round(mrt,      3) if mrt      is not None else _SENTINEL,
            "auc0inf_mg_l_h": round(auc_0_inf, 4),
        }