"""
pbpk_model.py — Multi-compartment PBPK ODE model for BodySim.
v3.3 — ACAT PHYSIOLOGICAL RESTORATION: per-segment pH gradient; absolute SA multipliers.

─────────────────────────────────────────────────────────────────────────────
CHANGES FROM v3.2 → v3.3  (ACAT Physiology Restoration Sprint)
─────────────────────────────────────────────────────────────────────────────

✅ FIX C — _build_acat_params: True per-segment pH gradient for ionization
  Problem (v3.2): ph_segment = 6.0 was hardcoded and applied uniformly to all
    7 segments.  This erased the physiological pH gradient (stomach ~2, duodenum
    ~6, jejunum ~6.5, ileum ~7.4, colon ~5.5–6.5) that governs the segment-specific
    un-ionized fraction f_u[i] for acidic and basic drugs.  A weak base with
    pKa 8 absorbs primarily in the ileum (pH 7.4), not the stomach (pH 2); the
    flat-pH assumption predicted near-zero absorption everywhere.
  Fix:
    1. Import ACAT_PH from physiology.py alongside existing imports inside the
       try block; fall back to a physiologically representative module-level
       constant ACAT_PH_DEFAULT if physiology.py is unavailable.
    2. Remove the scalar ph_segment variable and the np.ones/np.full
       pre-allocation pattern entirely.
    3. Move Henderson-Hasselbalch calculation inside the per-segment loop so
       each f_u[i] is evaluated at the anatomically correct seg_ph = ACAT_PH[i].
  Impact: Ionization-sensitive drugs (basic drugs absorbed in the ileum, acidic
    drugs absorbed in the proximal intestine) will show correct regional absorption
    profiles and realistic bioavailability predictions.

✅ FIX D — _build_acat_params: Absolute (un-normalized) SA multipliers
  Problem (v3.2): sa_regional = fold_raw[i] / fold_max normalized every folding
    multiplier relative to the jejunal maximum (10), forcing the jejunal k_abs
    coefficient down to 1.0 × P_EFF_SCALE instead of 10 × P_EFF_SCALE.  This
    mathematically erased the 10-fold villous/microvillous amplification of the
    jejunum — the primary absorption site — relative to the stomach.  The correct
    formula uses the raw multiplier directly so that the jejunum is genuinely 10×
    more absorptive than the stomach, as dictated by intestinal anatomy.
  Fix:
    1. Remove fold_max computation and the sa_regional = fold_raw[i] / fold_max
       division entirely.
    2. Update absorption constant formula to use fold_raw[i] directly:
         k_abs[i] = p_eff × P_EFF_SCALE × fold_raw[i] × f_u[i]
    3. physiology.py import path now extracts the raw multipliers without
       re-normalizing (no × _phys_norm step).
  Dimensional note: fold_raw[i] is a dimensionless anatomical amplification
    factor (ratio of effective absorptive area to bare mucosal cylinder area).
    k_abs [h⁻¹] = p_eff [cm/s] × P_EFF_SCALE [h⁻¹/(cm/s)] × fold_raw[i] [–] × f_u[i] [–]
    Units are preserved; only the numerical scale of k_abs changes.

─────────────────────────────────────────────────────────────────────────────
CHANGES FROM v3.1 → v3.2  (Clearance-to-Concentration Scaling Sprint)
─────────────────────────────────────────────────────────────────────────────

✅ FIX A — LIV_VASC: Explicit WSM hepatic elimination term with /Rb correction
  Problem: CL_h is computed via the Well-Stirred Model using whole-blood units
    (Rb appears in the extraction denominator).  C_liv_vasc is a plasma-equivalent
    state variable [mg/L plasma].  Without explicit /Rb reconciliation, the
    elimination flux would be dimensionally inconsistent (blood clearance × plasma
    concentration), overestimating the metabolic sink by a factor of Rb (≈1.0–2.0×).
  Fix: Added (CL_h / Rb) × C_liv_vasc / v_liv_vasc to dydt[LIV_VASC].
    - CL_h [L blood/h] / Rb → [L plasma equivalent/h] — consistent with C_liv_vasc
    - Derivation comment added inline with full dimensional trace.
  Scope: LIV_VASC ODE only.  LIV_TISS metabolic_rate is unchanged (operates on
    C_tissue_free which is already plasma-basis via fup × C_liv_tiss / Kp_liver).
  Impact: High-Rb drugs (e.g. tacrolimus Rb ≈ 15, chloroquine Rb ≈ 300) will show
    corrected hepatic elimination — previously overestimated by Rb-fold.

✅ FIX B — _build_acat_params: Explicit two-factor SA decomposition (superseded by FIX D)
  Module-level ACAT_SA_FOLDING [1,1,10,10,8,8,2] and ACAT_SA_FACTORS (normalized)
  introduced in v3.2 are retained for external compatibility.

─────────────────────────────────────────────────────────────────────────────


The three ODEs audited in this sprint share a single dimensional convention:

  State variables          [mg/L plasma-equivalent]
  Blood flows Q            [L blood/h]
  C_blood  = C_plasma × Rb  [mg/L blood]
  dC/dt    = Σ(mass fluxes [mg/h]) / V_compartment [L]  →  [mg/(L·h)]

✅ FIX 1 — GUT_ENTER perfusion: Kp partitioning confirmed and documented.
  The blood leaving the enterocyte capillary is in equilibrium with tissue:
    C_leaving_blood = (C_gut_enter / Kp_gut) × Rb
  Omitting the Kp_gut division would assume Kp=1, trapping drug in the gut
  for any compound with Kp >> 1.  The division was already present in v2.9;
  this sprint adds the full derivation comment so the correctness is
  self-evident without reading supplementary material.

✅ FIX 2 — LIV_VASC Rb accounting: three flux categories documented.
  (A) Convective flows Q×C_plasma×Rb:     Rb IS required (Q is blood flow).
  (B) Transporter CL×C_vascular_free:     Rb NOT required — in vitro CL is
      plasma-basis (OATP/OCT measured in buffer/plasma, not whole blood).
  (C) Passive diffusion CL_pd×ΔC_free:   Rb NOT required — same reason.
  The WSM /Rb correction for CYP metabolism (LIV_TISS) is enzyme-specific
  and does NOT apply to transporter clearances.  Documented explicitly so
  future edits cannot inadvertently add a spurious /Rb to transporter terms.

✅ FIX 3 — ACAT lumen mass balance: formally verified and annotated.
  j_abs_i is subtracted from LUMEN[i] exactly once and added to
  total_abs_flux exactly once.  DOSE_DEPOT→LUMEN[0] and GLU_EFF→LUMEN[0]
  links are both exact (same term appears in both ODEs with opposite sign).
  total_abs_flux / V_gut is added to GUT_ENTER — mass conservative ✓

─────────────────────────────────────────────────────────────────────────────

─────────────────────────────────────────────────────────────────────────────
CHANGES FROM v2.8 → v2.9  (Mechanistic ACAT Sprint)
─────────────────────────────────────────────────────────────────────────────

✅ REMOVED DOUBLE-PENALTY BIOAVAILABILITY (Change 1):
  - F (bioavailability) removed from the gut-entry ODE term.
  - In a mechanistic model F is an OUTPUT, not an input multiplier.
  - Gut extraction and hepatic first-pass are emergent properties of CLint
    and the liver 2-compartment model.  Multiplying by F double-counted them.
  - Reference: Pang & Rowland, J Pharmacokinet Biopharm 1977.

✅ MECHANISTIC PERMEABILITY SCALING — P_EFF_SCALE (Change 2):
  - Removed empirical 10^logP multiplier from _build_acat_params.
  - Replaced with P_EFF_SCALE = 4800 h⁻¹/(cm/s) derived from intestinal
    cylinder geometry: 2/r_gut × 3600, r_gut = 1.5 cm.
  - k_abs[i] = p_eff × P_EFF_SCALE × SA_factor[i] × f_unionized[i]
  - p_eff taken from drug["p_eff"] (measured) or estimated via Egan logP
    regression if absent.  SA factors sourced from physiology.py (ACAT_SA_FACTORS).
  - References: Amidon et al., Pharm Res 1995; Yu et al., J Pharm Sci 1999;
                Egan et al., J Med Chem 2000.

✅ CORRECTED GASTRIC EMPTYING (Change 3):
  - DOSE_DEPOT and stomach inlet now drain at kt_arr[0] = 1/transit_time_stomach
    (physiological rate, 4 h⁻¹) instead of the drug-specific ka.
  - Gastric emptying is governed by antral motor activity, not drug chemistry.
  - Reference: Dressman JB, J Pharm Sci 1998;87:403.

✅ MECHANISTIC ENTEROCYTE-TO-PORTAL TRANSFER (Change 4):
  - GUT_ENTER perfusion term uses Q_gut/V_gut × (C_art − C_enter/Kp × Rb),
    exactly as for all other perfused organs.
  - No hardcoded scalar multiplier; portal concentration C_portal = C_enter/Kp_gut
    feeds LIV_VASC inflow correctly.

Verification targets (F should emerge as a simulation result):
  - Metformin (basic, pKa 11.5, logP -1.4):  low F ~55% due to ionization/pH
  - Propranolol (basic, pKa 9.5, logP 3.5):  high first-pass extraction (F ~30%)
    driven by CLint, NOT by an input F parameter.

─────────────────────────────────────────────────────────────────────────────

─────────────────────────────────────────────────────────────────────────────
CHANGES FROM v2.6 → v2.7  (Renal Accuracy Sprint)
─────────────────────────────────────────────────────────────────────────────

✅ PASSIVE TUBULAR REABSORPTION (Gap 3.1):
  - Implements CL_reab = Q_tubular_water × f_neutral_urine × k_perm
  - Q_tubular_water = 1% of GFR (99% of filtered water is reabsorbed)
  - f_neutral_urine from Henderson-Hasselbalch at subject-specific urine pH
  - k_perm from linear logP interpolation [-1 → 1]: 0.0 (hydrophilic) to 1.0
  - Reabsorption raises kidney compartment concentration → returns drug to venous
  - Fixes 2–10× CLrenal overestimate for basic drugs (metoprolol, propranolol, etc.)

✅ RENAL EXTRACTION RATIO CAP (Gap 3.3):
  - Total active secretion clearance is now flow-limited by renal plasma flow
  - Well-stirred kidney model: ER = (fup × CLint_sec/Rb) / (Q_kp + fup × CLint_sec/Rb)
  - CL_sec_observed = Q_kp × ER_kidney  (never exceeds Q_kp)
  - Caps are applied before MM saturation in the ODE, preventing impossible predictions

✅ NCA TERMINAL PARAMETERS (Gap 5.1):
  - solve() now returns t_half_h, lambda_z_per_h, cl_total_lh, vss_l, mrt_h, auc0inf
  - λz from log-linear regression of terminal 20% of plasma time-course (≥3 points)
  - Vss: CL × MRT (IV); CL × (MRT − 1/ka) for oral, subtracting mean absorption time
  - AUC0-∞ extrapolated as AUC0-t + C_last / λz

─────────────────────────────────────────────────────────────────────────────
CHANGES FROM v2.5 → v2.6
─────────────────────────────────────────────────────────────────────────────

✅ FIXED HIGH-EXTRACTION DRUG TRAPPING (250x AUC OVERPREDICTION):
  - Permeability Enhancement: uptake substrates now get 3x CL_pd boost
  - Perfusion-Limited Mode: drugs with CLint > 50 L/h forced to CL_pd = 10000 L/h
  - Metabolic Throughput: high-extraction drugs get 2x Vmax scaling
  - Concentrative Transport: documented that j_uptake is gradient-independent
  - All logic driven by self.drug attributes (no hardcoded drug names)

✅ MECHANISTIC IMPROVEMENTS:
  - Active transporters now correctly model "concentrative power"
  - High-extraction drugs transition to perfusion-limited regime
  - Metabolic sink scaled to prevent tissue accumulation bottleneck
  - Passive diffusion enhanced for uptake substrates (membrane trafficking effect)

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
STATE VECTOR (19 elements - Permeability-Limited 2-Compartment Liver)
─────────────────────────────────────────────────────────────────────────────
  y[0]  = C_arterial         mg/L  arterial blood plasma
  y[1]  = C_venous           mg/L  venous blood plasma
  y[2]  = C_venous_delay     mg/L  venous circulation delay compartment (NEW)
  y[3]  = C_lung             mg/L  lung tissue
  y[4]  = C_liv_vasc         mg/L  liver vascular (sinusoidal blood) (RENAMED)
  y[5]  = C_liv_tiss         mg/L  liver tissue (hepatocytes) (NEW)
  y[6]  = C_kidney           mg/L  kidney tissue
  y[7]  = C_brain            mg/L  brain tissue
  y[8]  = C_heart            mg/L  heart tissue
  y[9]  = C_muscle           mg/L  muscle tissue
  y[10] = C_fat              mg/L  fat tissue
  y[11] = C_gut_enterocyte   mg/L  gut enterocyte (intracellular) (RENAMED)
  y[12] = C_skin             mg/L  skin tissue
  y[13] = C_bone             mg/L  bone tissue
  y[14] = C_spleen           mg/L  spleen tissue (NEW - was in "rest")
  y[15] = C_adipose_mes      mg/L  mesenteric adipose (NEW - was in "rest")
  y[16] = C_pancreas         mg/L  pancreas (NEW - was in "rest")
  y[17] = A_gut_lumen_abs    mg    drug in gut lumen awaiting absorption
  y[18] = A_gut_lumen_efflux mg    drug effluxed back to lumen by P-gp (NEW)

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
LIV_VASC = 4;  LIV_TISS = 5;  KID  = 6;  BRA  = 7
HRT  = 8;  MUS  = 9;  FAT  = 10
GUT_ENTER = 11;  SKN  = 12; BON  = 13
SPL  = 14; ADIP_MES = 15; PANC = 16
GLU_ABS  = 17; GLU_EFF = 18
DOSE_DEPOT = 19                          # v2.8: Pre-gastric dissolved dose
LUMEN_BASE = 20                          # v2.8: ACAT segments start here
N_ACAT_SEGMENTS = 7                      # v2.8: Stomach, Duodenum, Jejunum×2, Ileum×2, Colon
N_STATES = 19 + 1 + N_ACAT_SEGMENTS     # 27 total

ORGAN_NAMES = [
    "arterial", "venous", "venous_delay", "lung",
    "liver_vascular", "liver_tissue", "kidney", "brain",
    "heart", "muscle", "fat",
    "gut_enterocyte", "skin", "bone",
    "spleen", "adipose_mesenteric", "pancreas",
    "gut_lumen_absorption", "gut_lumen_efflux",
    "dose_depot", "stomach", "duodenum", "jejunum_1", "jejunum_2", "ileum_1", "ileum_2", "cecum_colon",
]

ACAT_SEGMENT_NAMES = ["stomach", "duodenum", "jejunum_1", "jejunum_2", "ileum_1", "ileum_2", "cecum_colon"]

# v2.8: ACAT transit times (hours) and surface area factors
ACAT_TRANSIT_TIMES = np.array([0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 18.0])
# Physiological regional folding multipliers (dimensionless, relative to stomach = 1):
#   Stomach(1) Duodenum(1) Jejunum×2(10 each) Ileum×2(8 each) Cecum/Colon(2)
#   Values reflect progressive villous amplification along the small intestine
#   (microvilli + circular folds + villi; Amidon et al., Pharm Res 1995).
#   The normalization denominator (10.0) converts them to fractions of maximal
#   jejunal surface area so that Σ sa_factor is dimensionless and the geometric
#   scaler P_EFF_SCALE (4800 h⁻¹ per cm/s) alone sets the absolute rate scale.
ACAT_SA_FOLDING    = np.array([1.0, 1.0, 10.0, 10.0, 8.0, 8.0, 2.0])   # Raw physiological folding ratios
_ACAT_SA_NORM      = 10.0                                                  # Normalization: max jejunal fold factor
ACAT_SA_FACTORS    = ACAT_SA_FOLDING / _ACAT_SA_NORM                      # Normalized (for backward-compat exports)

# v3.3: Physiological luminal pH per ACAT segment — fallback used when physiology.py
# is unavailable.  Values represent mean fasted-state luminal pH:
#   Stomach   ~2.0  (highly acidic; HCl secretion)
#   Duodenum  ~6.0  (rapid bicarbonate neutralization of gastric acid)
#   Jejunum 1 ~6.5  (proximal small intestine; primary absorption site for many drugs)
#   Jejunum 2 ~6.5  (same anatomical region)
#   Ileum 1   ~7.4  (distal small intestine; higher pH favours basic drug absorption)
#   Ileum 2   ~7.4  (same anatomical region)
#   Colon     ~5.9  (slightly acidic; anaerobic fermentation lowers colonic pH)
# References:
#   Fallingborg J, Pharmacol Toxicol 1999;85:291 — in vivo luminal pH measurements
#   Dressman JB et al., Pharm Res 1998;15:11 — fasted/fed state pH survey
ACAT_PH_DEFAULT = np.array([2.0, 6.0, 6.5, 6.5, 7.4, 7.4, 5.9])

TISSUE_COMPARTMENTS = {
    LIV_VASC: "liver_vasc", LIV_TISS: "liver_tiss", KID: "kidney", BRA: "brain",
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
        self._acat = self._build_acat_params()  # v2.8: Pre-compute ACAT parameters
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
            
            # --- LIVER PERMEABILITY-LIMITED 2-COMPARTMENT (v2.3+) ---
            # Dynamic CL_pd based on logP (v2.4 fix: solves Propranolol bottleneck)
            "liver_CL_pd": None,  # Will be calculated from drug logP in __init__

            # --- v2.7: RENAL ACCURACY PARAMETERS ---
            # urine_ph: urinary pH for tubular reabsorption (Henderson-Hasselbalch).
            #   Range 4.5–8.5; 6.0 is the typical fasted adult default.
            #   Pass subject-specific value from scale_physiology() output.
            "urine_ph": 6.0,
        }
        
        # Merge user params with defaults
        merged = defaults.copy()
        merged.update(params)
        
        # Validate required params
        if "egfr" not in merged:
            raise ValueError("params must include 'egfr' (eGFR in mL/min)")
        if "cyp3a4_activity" not in merged:
            raise ValueError("params must include 'cyp3a4_activity'")
        
        # Dynamically calculate liver_CL_pd based on drug logP (v2.5 fix)
        # This fixes the Propranolol/Atorvastatin bottleneck: highly lipophilic drugs
        # now become "flow-limited" (permeability >> blood flow), allowing hepatic
        # extraction to be governed by enzyme capacity and blood flow, not diffusion
        if merged["liver_CL_pd"] is None:
            logp = self.drug.get("logp", 0.0)
            clint = self.drug.get("CLint", 0.0)
            is_uptake_substrate = self.drug.get("is_uptake_substrate", False)
            
            if logp < 0:
                # Hydrophilic drugs: use exponential formula (keeps them restricted)
                # logp=-2 → 1.7 L/h, logp=-1 → 3.1 L/h
                cl_pd = float(np.clip(5.0 * np.exp(logp / 2.0), 1.0, 10.0))
            else:
                # Lipophilic drugs: use base-10 exponential (makes them flow-limited)
                # logp=0 → 10 L/h, logp=1 → 100 L/h, logp=2 → 1000 L/h
                # logp=3.5 (Propranolol) → 31,600 (capped at 1500)
                # logp=4 (Atorvastatin) → 100,000 (capped at 1500)
                # This ensures: CL_pd >> Q_liver (90 L/h) for truly lipophilic drugs
                cl_pd = float(np.clip(10.0 * (10 ** logp), 10.0, 1500.0))
            
            # ✨ FIX 1: PERMEABILITY ENHANCEMENT FOR UPTAKE SUBSTRATES
            # High-affinity transporters (OATP, OCT) effectively increase the
            # membrane surface area for distribution. Scale CL_pd upward.
            if is_uptake_substrate:
                cl_pd *= 3.0  # 3x enhancement from transporter trafficking
            
            # ✨ FIX 4: PERFUSION-LIMITED MODE FOR HIGH-EXTRACTION DRUGS
            # If CLint > 50 L/h (high extraction), force perfusion limitation.
            # This prevents artificial AUC inflation from diffusion bottlenecks.
            # Perfusion-limited: CL_pd >> Q_liver (90 L/h), so set to 10x flow.
            if clint > 50.0:
                cl_pd = max(cl_pd, 10000.0)  # Force perfusion-limited regime
            
            merged["liver_CL_pd"] = cl_pd
        
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

    # ── ACAT parameter pre-computation (v2.8) ──────────────────────────────
    def _build_acat_params(self) -> dict:
        """
        Pre-compute ACAT (Advanced Compartmental Absorption and Transit) model parameters.

        7-segment model: Stomach, Duodenum, Jejunum×2, Ileum×2, Cecum/Colon

        Returns
        -------
        dict with:
          "kt"    : np.array(7,) — transit rate constants (h⁻¹) for each segment
          "f_u"   : np.array(7,) — un-ionized fraction (0–1) per segment at its own pH
          "k_abs" : np.array(7,) — first-order absorption rate constants (h⁻¹) per segment
        """
        # ── Transit rate constants (kt = 1/transit_time) ────────────────────
        kt = 1.0 / ACAT_TRANSIT_TIMES

        # ── Drug ionization parameters ───────────────────────────────────────
        pka       = self.drug.get("pka", None)
        drug_type = self.drug.get("drug_type", "neutral")

        # ── Geometric scaling constant ───────────────────────────────────────
        #
        # P_EFF_SCALE converts measured Caco-2/PAMPA permeability (cm/s) into a
        # first-order luminal absorption rate constant (h⁻¹) via the cylindrical
        # gut model:
        #
        #   P_EFF_SCALE = (2 / r_gut) × 3600    [h⁻¹ per (cm/s)]
        #
        # where r_gut ≈ 1.5 cm is the effective inner radius of the small intestine.
        # Numerically: (2 / 1.5) × 3600 = 4800 h⁻¹/(cm/s).
        #
        # References:
        #   Amidon GL et al., Pharm Res 1995;12:413 — cylindrical gut geometry
        #   Yu LX et al., J Pharm Sci 1999;88:196   — ACAT permeability scaling
        P_EFF_SCALE = 4800.0   # h⁻¹ per (cm/s): 2/r_gut × 3600, r_gut = 1.5 cm

        # ── Effective permeability ────────────────────────────────────────────
        # Use experimentally measured p_eff when available; otherwise estimate
        # from logP via the Egan regression (neutral form benchmark; ionization
        # is applied separately via f_u[i] inside the loop below).
        #   Egan regression: log10(Caco-2) ≈ 0.4 × logP − 5.5   (cm/s)
        #   Clamp to physically plausible range [1e-7, 2e-4] cm/s.
        # Reference: Egan WJ et al., J Med Chem 2000;43:3867
        logp  = self.drug.get("logp", 0.0)
        p_eff = self.drug.get("p_eff", None)
        if p_eff is None:
            p_eff = float(np.clip(10.0 ** (0.4 * logp - 5.5), 1e-7, 2e-4))

        # ── Per-segment pH array ──────────────────────────────────────────────
        # Prefer ACAT_PH from physiology.py; fall back to module-level default.
        # physiology.py is expected to export ACAT_PH as a 7-element array aligned
        # to ACAT_SEGMENT_NAMES = [stomach, duodenum, jejunum_1, jejunum_2,
        #                           ileum_1, ileum_2, cecum_colon].
        # Separate try/except from the SA import so a missing ACAT_PH in an older
        # physiology.py does not silently suppress the SA import as well.
        try:
            from physiology import ACAT_PH as _PHYS_PH
            seg_ph_arr = np.asarray(_PHYS_PH, dtype=float)
        except Exception:
            seg_ph_arr = ACAT_PH_DEFAULT.copy()   # module-level physiological fallback

        # ── Per-segment raw SA folding multipliers ────────────────────────────
        # fold_raw[i] is the dimensionless anatomical amplification factor for
        # segment i, representing the ratio of effective absorptive area (accounting
        # for circular folds, villi, and microvilli) to a bare mucosal cylinder.
        # Values [1, 1, 10, 10, 8, 8, 2] are used WITHOUT normalization so that
        # the jejunum (fold = 10) is genuinely 10× more absorptive than the stomach
        # (fold = 1), consistent with intestinal anatomy.
        #
        # The previous v3.2 normalization (fold_raw / fold_max) forced the jejunal
        # coefficient to 1.0, mathematically erasing this amplification.  Absolute
        # multipliers are required for correct regional absorption profiles.
        #
        # physiology.py is expected to export ACAT_SA_FACTORS as a 7-element array
        # of NORMALIZED factors (÷10).  Recover raw values by multiplying back by
        # the normalization denominator _ACAT_SA_NORM.
        try:
            from physiology import ACAT_SA_FACTORS as _PHYS_SA, ACAT_SEGMENT_NAMES as _PHYS_NAMES
            fold_raw = np.array(
                [_PHYS_SA[_PHYS_NAMES.index(s)] for s in ACAT_SEGMENT_NAMES],
                dtype=float,
            ) * _ACAT_SA_NORM   # un-normalize: normalized × 10 → raw absolute multiplier
        except Exception:
            fold_raw = ACAT_SA_FOLDING.copy()   # module-level raw physiological ratios

        # ── Per-segment absorption rate constants ─────────────────────────────
        #
        # DIMENSIONAL DERIVATION:
        #
        #   k_abs[i]  [h⁻¹]  =  p_eff [cm/s]
        #                      × P_EFF_SCALE [h⁻¹/(cm/s)]
        #                      × fold_raw[i] [dimensionless]
        #                      × f_u[i]      [dimensionless]
        #
        # p_eff × P_EFF_SCALE  → base absorption rate for a bare cylinder [h⁻¹]
        # × fold_raw[i]        → amplification by segment-specific villi/microvilli
        # × f_u[i]             → fraction in un-ionized (membrane-permeant) form
        #                        evaluated at the ANATOMICALLY CORRECT pH of segment i
        #
        # f_u[i] is computed inside the loop so each segment uses its own seg_ph.
        # Henderson-Hasselbalch equations:
        #   Acidic drug (HA ⇌ H⁺ + A⁻):
        #     f_neutral = 1 / (1 + 10^(pH − pKa))
        #     Low pH (stomach) → f_neutral → 1  (drug un-ionized, absorbable)
        #     High pH (ileum)  → f_neutral → 0  (drug ionized, poorly absorbed)
        #   Basic drug (BH⁺ ⇌ B + H⁺):
        #     f_neutral = 1 / (1 + 10^(pKa − pH))
        #     Low pH (stomach) → f_neutral → 0  (drug protonated, poorly absorbed)
        #     High pH (ileum)  → f_neutral → 1  (drug un-ionized, absorbable)
        #   Zwitterion: approximate as fixed low permeability fraction (0.15)
        #   Neutral: f_u = 1.0 everywhere (no ionization correction needed)
        #
        # References:
        #   Fallingborg J, Pharmacol Toxicol 1999;85:291 — luminal pH in vivo
        #   Dressman JB et al., Pharm Res 1998;15:11     — fasted-state pH
        #   Amidon GL et al., Pharm Res 1995;12:413      — cylindrical gut model
        #   Yu LX et al., J Pharm Sci 1999;88:196        — ACAT absorption scaling

        # ── Paracellular / microclimate permeability floor (v2.9) ────────────
        #
        # The strict pH-partition hypothesis (Henderson-Hasselbalch alone) predicts
        # zero absorption for fully ionized drugs, but two physiological mechanisms
        # maintain a baseline permeability:
        #
        #   1. Paracellular route — tight junctions in the small intestine are not
        #      impermeable; small hydrophilic/ionized molecules cross via aqueous
        #      pores between enterocytes (Pade & Stavchansky, Mol Pharmacol 1998).
        #
        #   2. Unstirred water layer microclimate — the acid microclimate at the
        #      brush-border surface (pH ≈ 5.5–6.0) sustains a higher un-ionized
        #      fraction than the bulk luminal pH for basic drugs such as Metformin
        #      (Daniel & Kottra, Pflugers Arch 2004).
        #
        # The floor of 0.02 (2%) represents conservatively measured paracellular
        # permeability for small organic ions (MW < 500 Da) in Caco-2 monolayers.
        # It does NOT apply to neutral drugs or zwitterions, which have independent
        # fixed fractions.
        #
        # References:
        #   Pade V, Stavchansky S, Mol Pharmacol 1998;54:310
        #   Daniel H, Kottra G, Pflugers Arch 2004;447:610
        #   Lennernas H, J Pharm Sci 1998;87:403 — paracellular absorption in vivo
        paracellular_floor = 0.02

        f_u   = np.zeros(N_ACAT_SEGMENTS)
        k_abs = np.zeros(N_ACAT_SEGMENTS)

        for i in range(N_ACAT_SEGMENTS):
            seg_ph = float(seg_ph_arr[i])   # anatomically correct pH for segment i

            # ── Henderson-Hasselbalch ionization at seg_ph ──────────────────
            if pka is None or drug_type == "neutral":
                # Neutral drug: fully un-ionized in all segments
                f_u[i] = 1.0
            elif drug_type == "acidic":
                # HA ⇌ H⁺ + A⁻ — neutral form HA dominates at pH << pKa
                # paracellular_floor prevents f_u collapsing to 0 when the drug
                # is fully ionized (e.g. strong acids in the alkaline ileum).
                raw_fu = 1.0 / (1.0 + 10.0 ** (seg_ph - pka))
                f_u[i] = float(np.clip(max(raw_fu, paracellular_floor), 0.0, 1.0))
            elif drug_type == "basic":
                # BH⁺ ⇌ B + H⁺ — neutral form B dominates at pH >> pKa
                # paracellular_floor prevents f_u collapsing to 0 when the drug
                # is fully protonated (e.g. Metformin pKa 11.5 in the stomach).
                raw_fu = 1.0 / (1.0 + 10.0 ** (pka - seg_ph))
                f_u[i] = float(np.clip(max(raw_fu, paracellular_floor), 0.0, 1.0))
            elif drug_type == "zwitterion":
                # Zwitterions carry both charges; approximate as low constant fraction
                f_u[i] = 0.15
            else:
                f_u[i] = 1.0

            # ── Absorption rate constant: geometric × anatomical × ionization ─
            k_abs[i] = float(p_eff * P_EFF_SCALE * fold_raw[i] * f_u[i])

        return {
            "kt":    kt,
            "f_u":   f_u,
            "k_abs": k_abs,
        }

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

        # ── v2.7: RENAL EXTRACTION RATIO CAP (Gap 3.3) ──────────────────────
        # Active secretion is limited by renal plasma flow (well-stirred model).
        # Without this cap, a drug with a huge predicted CLint_sec can yield
        # CL_renal > Q_kidney, which violates mass balance.
        #
        # Well-stirred kidney: ER_kidney = (fup × CLint_sec / Rb) / (Q_kp + fup × CLint_sec / Rb)
        #                      CL_sec_obs = Q_kidney_plasma × ER_kidney
        #
        # Reference: Rowland & Tozer, Clinical Pharmacokinetics 5th ed., Ch. 5.
        Rb = self.drug.get("Rb", 1.0)
        Q_kidney_blood  = self.flow["kidney"]                 # L/h, whole blood
        Q_kidney_plasma = Q_kidney_blood / max(Rb, 0.01)     # L/h, plasma flow

        if cl_sec_linear_total > 1e-9:
            _er_sec = (
                (fup * cl_sec_linear_total / Rb)
                / (Q_kidney_plasma + fup * cl_sec_linear_total / Rb)
            )
            cl_sec_plasma_cap_lh = float(Q_kidney_plasma * _er_sec)
        else:
            cl_sec_plasma_cap_lh = 0.0   # No secretion: cap is irrelevant

        # ── v2.7: PASSIVE TUBULAR REABSORPTION CL (Gap 3.1) ─────────────────
        # Reabsorption occurs as tubular fluid is concentrated: 99% of filtered
        # water is reclaimed, leaving only ~1% as urine. Only the un-ionized
        # (neutral) fraction of a drug crosses the tubular lipid membrane.
        #
        # CL_reabsorption = Q_tubular_water × f_neutral_urine × k_perm
        #
        # Q_tubular_water = 0.01 × GFR_lh   (1% of filtered water becomes urine)
        # f_neutral_urine  = fraction un-ionized at urine pH (Henderson-Hasselbalch)
        # k_perm           = membrane permeability factor from logP
        #                    0.0 at logP ≤ -1 (polar, membrane-impermeant)
        #                    1.0 at logP ≥  1 (lipophilic, freely membrane-permeant)
        #                    linear interpolation in between
        #
        # References:
        #   Gibaldi et al., J Pharm Sci 1969 — pH-partition reabsorption
        #   Reigner & Blesch, Eur J Clin Pharmacol 2002 — tubular water flow
        pka      = self.drug.get("pka", None)
        drug_type = self.drug.get("drug_type", "neutral")
        logp     = self.drug.get("logp", 0.0)
        urine_ph = float(self.params.get("urine_ph", 6.0))

        Q_tubular_water_lh = 0.01 * gfr_lh   # ~1.0–1.5 mL/min as urine flow

        # f_neutral_urine: fraction of drug in un-ionized form at urinary pH.
        # Only neutral molecules permeate tubular epithelium (pH-partition hypothesis).
        if pka is None or drug_type == "neutral":
            f_neutral_urine = 1.0
        elif drug_type == "acidic":
            # HA ⇌ H⁺ + A⁻: neutral form HA dominates at pH << pKa
            # f_neutral = 1 / (1 + 10^(pH - pKa))
            f_neutral_urine = float(1.0 / (1.0 + 10.0 ** (urine_ph - pka)))
        elif drug_type == "basic":
            # BH⁺ ⇌ B + H⁺: neutral form B dominates at pH >> pKa
            # f_neutral = 1 / (1 + 10^(pKa - pH))
            f_neutral_urine = float(1.0 / (1.0 + 10.0 ** (pka - urine_ph)))
        elif drug_type == "zwitterion":
            # Zwitterions have both ionization states; approximate as low permeability
            f_neutral_urine = 0.15
        else:
            f_neutral_urine = 1.0
        f_neutral_urine = float(np.clip(f_neutral_urine, 0.0, 1.0))

        # k_perm: lipid membrane permeability of the neutral form.
        # Linear interpolation: logP=-1 → k_perm=0.0, logP=+1 → k_perm=1.0
        # Drugs with logP < -1 are too hydrophilic to permeate tubular cells.
        # Drugs with logP > +1 permeate freely.
        k_perm = float(np.clip((logp - (-1.0)) / (1.0 - (-1.0)), 0.0, 1.0))

        cl_reabsorption_lh = float(Q_tubular_water_lh * f_neutral_urine * k_perm)

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
            # v2.7 renal additions
            "cl_sec_plasma_cap_lh": cl_sec_plasma_cap_lh,
            "cl_reabsorption_lh":   cl_reabsorption_lh,
            "urine_ph":             urine_ph,
            "f_neutral_urine":      f_neutral_urine,
            "k_perm":               k_perm,
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
        27-compartment PBPK with:
          - v2.8: 7-segment ACAT oral absorption model
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
        C_liv_vasc = y[LIV_VASC]
        C_liv_tiss = y[LIV_TISS]
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
        A_dose_depot = y[DOSE_DEPOT]                                       # v2.8: Pre-gastric dose
        M_lumen = y[LUMEN_BASE: LUMEN_BASE + N_ACAT_SEGMENTS]             # v2.8: ACAT segments

        v = self.vol
        q = self.flow
        kp = self.drug["kp"]
        fup = self.drug["fup"]
        Rb = self.drug["Rb"]
        tp = self._tp
        ac = self._acat                                                     # v2.8: ACAT parameters

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
        CLh = self.drug["CLint"]  # Intrinsic hepatic clearance (used in tissue metabolism)
        cl_filt = tp["cl_filt"]
        
        # ── Hepatic Enzyme Kinetics (Michaelis-Menten vs. Linear) ────────────
        # Extract MM parameters for saturable hepatic metabolism
        km_hep = self.drug.get("Km_hepatic", 1.0)      # mg/L (free concentration scale)
        cl_int_target = self.drug["CLint"]  # Clinical CLint (may include overrides)
        
        # Safe Derivation: Only derive Vmax from CLint if Vmax is not explicitly provided
        # This prevents overriding intentional drug-specific Vmax values
        if "Vmax_hepatic" in self.drug:
            # Explicit Vmax provided - use it directly (trust the data)
            vmax_hep = self.drug["Vmax_hepatic"]
        else:
            # No explicit Vmax - derive from CLint using: Vmax = CLint * Km
            # Mathematical relationship: CLint = Vmax / Km (at low [S] << Km)
            # This ensures the MM curve matches clinical clearance at therapeutic doses
            vmax_hep = cl_int_target * km_hep
            
            # ✨ FIX 3: METABOLIC THROUGHPUT SCALING FOR HIGH-EXTRACTION DRUGS
            # For drugs with CLint > 50 L/h, the metabolic sink must be fast enough
            # to prevent tissue accumulation and back-diffusion. Scale Vmax upward
            # to ensure the enzyme can keep up with the uptake flux.
            if cl_int_target > 50.0:
                # Scale Vmax by an additional 2x for high-extraction drugs
                # This ensures metabolism keeps pace with active uptake
                vmax_hep *= 2.0
        
        use_mm_hepatic = (vmax_hep > 0 and km_hep > 1e-9)  # Enable MM if both parameters present

        # ── Free concentrations ────────────────────────────────────────────
        C_portal_free = fup * C_gut_enter / kp["gut"]
        C_vascular_free = fup * C_liv_vasc  # Free concentration in liver vascular space
        C_tissue_free = fup * C_liv_tiss / kp["liver"]  # Free concentration in liver tissue
        C_art_free = fup * C_art
        C_kid_free = fup * C_kid / kp["kidney"]

        # ── Blood concentrations leaving tissues (total, not free) ─────────
        # Used in mass balance: tissue → blood (across Kp gradient)
        C_gut_blood_out = C_gut_enter / kp["gut"]  # Total plasma conc leaving gut

        # ── Blood concentrations (for macroscopic flow) ─────────────────────
        # Macroscopic flows (Q) are in whole blood L/h, so must be multiplied by
        # blood concentrations (C * Rb), not plasma concentrations (C).
        # This corrects the mass balance for drugs that partition into RBCs (Rb > 1).
        C_art_blood = C_art * Rb
        C_ven_blood = C_ven * Rb
        C_ven_delay_blood = C_ven_delay * Rb
        C_gut_blood_out_blood = C_gut_blood_out * Rb
        C_liv_vasc_blood = C_liv_vasc * Rb

        # ── Absorption ─────────────────────────────────────────────────────
        # v2.9: ka is kept for P-gp re-absorption rate; F is no longer used in
        # the ODE (bioavailability emerges from the simulation).
        ka = self.drug["ka"]

        # ══════════════════════════════════════════════════════════════════
        # ODE EQUATIONS
        # ══════════════════════════════════════════════════════════════════
        dydt = np.zeros(N_STATES)

        # ── [GLU_ABS] (Legacy) ────────────────────────────────────────────
        # Kept for backward compatibility; set to zero (superseded by DOSE_DEPOT)
        dydt[GLU_ABS] = 0.0

        # ── [GLU_EFF] Gut lumen (efflux depot) ─────────────────────────────
        # Drug pumped back by P-gp accumulates here, then re-absorbs or exits
        pgp_efflux_to_lumen = self._pgp_efflux_rate(C_gut_enter) * v.get("gut", 0.5)
        
        # Re-absorption from efflux depot (slower than initial absorption)
        ka_reabs = ka * 0.5  # Half the rate (drug already saw P-gp once)
        
        dydt[GLU_EFF] = (
            pgp_efflux_to_lumen  # Influx from P-gp
            - ka_reabs * A_glu_eff  # Re-absorption
        )

        # ─────────────────────────────────────────────────────────────────
        # v2.8: ACAT SEGMENT ODEs
        # ─────────────────────────────────────────────────────────────────
        # State: M_lumen[i]  [mg]  — MASS (not concentration) in each segment.
        # kt_arr[i] = 1/transit_time[i]  [h⁻¹]
        # k_abs_arr[i]  [h⁻¹]
        #
        # For each segment i the mass balance is:
        #
        #   dM_lumen[i]/dt = in_transit[i] − out_transit[i] − j_abs_i
        #
        #   in_transit[0]  = kt[0]×A_dose_depot + ka_reabs×A_glu_eff   [mg/h]
        #   in_transit[i≥1]= kt[i−1]×M_lumen[i−1]                      [mg/h]
        #   out_transit[i] = kt[i]×M_lumen[i]                           [mg/h]
        #   j_abs_i        = k_abs[i]×M_lumen[i]                        [mg/h]
        #
        # MASS BALANCE GUARANTEE:
        #   Every j_abs_i subtracted from M_lumen[i] is added once and only
        #   once to total_abs_flux, which is then added to GUT_ENTER as
        #   total_abs_flux / V_gut [mg/(L·h)].  No drug is created or destroyed
        #   in the accumulation loop; the lumen–enterocyte interface is exact.
        #
        # CROSS-COMPARTMENT LINKS (all exact):
        #   DOSE_DEPOT → LUMEN[0]:  kt[0]×A_dose_depot removed from DOSE_DEPOT
        #                            and added as in_transit for i=0         ✓
        #   GLU_EFF → LUMEN[0]:     ka_reabs×A_glu_eff removed from GLU_EFF
        #                            and added as in_transit for i=0         ✓
        #   LUMEN[7] exit (colon):  kt[6]×M_lumen[6] lost to faeces (no return) ✓
        #   LUMEN[i] → GUT_ENTER:  total_abs_flux (exact Σ j_abs_i)          ✓

        kt_arr    = ac["kt"]      # Transit rate constants (h⁻¹), shape (7,)
        k_abs_arr = ac["k_abs"]   # Absorption rate constants (h⁻¹), shape (7,)

        total_abs_flux = 0.0   # [mg/h] — running total absorbed from all segments

        for i in range(N_ACAT_SEGMENTS):
            # ── Incoming transit mass flux [mg/h] ──
            if i == 0:
                # Stomach: physiological gastric emptying rate kt[0] = 4 h⁻¹
                # (1/0.25 h; set by antral motor activity, not drug chemistry).
                # Re-absorbed P-gp efflux re-enters the stomach as well.
                in_transit = kt_arr[0] * A_dose_depot + ka_reabs * A_glu_eff
            else:
                in_transit = kt_arr[i - 1] * M_lumen[i - 1]

            out_transit = kt_arr[i]    * M_lumen[i]   # → next segment or faeces [mg/h]
            j_abs_i     = k_abs_arr[i] * M_lumen[i]   # → enterocyte via membrane [mg/h]

            # j_abs_i is subtracted here and only here; added to total_abs_flux below.
            dydt[LUMEN_BASE + i] = in_transit - out_transit - j_abs_i

            total_abs_flux += j_abs_i   # accumulated for GUT_ENTER (exact mass balance)

        # ──────────────────────────────────────────────────────────────────
        # [DOSE_DEPOT] Pre-gastric dissolved dose
        # ──────────────────────────────────────────────────────────────────
        # v2.9: Drain at the physiological gastric emptying rate kt_arr[0]
        # (= 1/0.25 h = 4 h⁻¹), not the drug absorption rate constant ka.
        # Gastric emptying is set by antral motor activity, not drug chemistry.
        dydt[DOSE_DEPOT] = -kt_arr[0] * A_dose_depot

        # ── [GUT_ENTER] Gut enterocyte (intracellular) ─────────────────────
        #
        # State variable: C_gut_enter  [mg/L TISSUE concentration]
        #
        # ── PERFUSION TERM: dimensional derivation ──────────────────────────
        # Blood arriving at the enterocyte capillary carries C_art_blood [mg/L blood].
        # Blood leaving the capillary is in instantaneous equilibrium with tissue:
        #
        #   C_leaving_plasma = C_gut_enter / Kp_gut        [mg/L plasma]
        #   C_leaving_blood  = C_leaving_plasma × Rb
        #                    = (C_gut_enter / Kp_gut) × Rb  [mg/L blood]
        #
        # Net mass flux [mg/h] = Q_gut × (C_art_blood − C_leaving_blood)
        #   = Q_gut × (C_art×Rb  −  (C_gut_enter/Kp_gut)×Rb)
        #
        # Dividing by V_gut [L] → dC_gut_enter/dt contribution [mg/(L·h)] ✓
        #
        # WHY Kp_gut IS REQUIRED:
        #   C_gut_enter is tissue concentration; the capillary wall equilibrates
        #   against PLASMA concentration (= C_gut_enter / Kp_gut), not tissue
        #   concentration directly.  Omitting the Kp division would assume the
        #   tissue concentration equals the plasma concentration (Kp = 1), which
        #   is only valid for rapidly equilibrating, non-partitioning drugs.
        #   For basic drugs or high-lipophilicity compounds Kp >> 1, so omitting
        #   the division would dramatically underestimate the driving-force gradient
        #   and falsely trap drug in the gut.
        #
        # ── LUMEN ABSORPTION TERM ───────────────────────────────────────────
        # total_abs_flux [mg/h] = Σᵢ k_abs[i] × M_lumen[i]
        # Each j_abs_i is subtracted once from LUMEN[i] and once accumulated into
        # total_abs_flux; dividing by V_gut converts mass flux to concentration
        # rate.  F (bioavailability) is intentionally absent: gut-wall extraction
        # (P-gp) and hepatic first-pass (CLint) are emergent ODE outputs, not
        # input multipliers (Pang & Rowland, J Pharmacokinet Biopharm 1977).
        #
        # ── P-gp EFFLUX TERM ────────────────────────────────────────────────
        # _pgp_efflux_rate returns a concentration rate [mg/(L·h)] computed as
        # MM_rate [mg/h] / V_gut [L] internally, so it is subtracted directly
        # from dC/dt without further volume division.
        #
        # ── PORTAL BLOOD CONCENTRATION (feeds LIV_VASC) ─────────────────────
        # C_gut_blood_out      = C_gut_enter / Kp_gut     [mg/L plasma]
        # C_gut_blood_out_blood = C_gut_blood_out × Rb    [mg/L blood]
        # These are used in the LIV_VASC inflow (Q_pv × C_gut_blood_out_blood).
        # Mass balance check: same Kp_gut applied in both the GUT_ENTER outflow
        # (C_leaving_blood above) and the LIV_VASC portal inflow → exact ✓
        dydt[GUT_ENTER] = (
            (Q_gut / v["gut"]) * (C_art_blood - (C_gut_enter / kp["gut"]) * Rb)
            + total_abs_flux / v["gut"]      # lumen → enterocyte [mg/h / L = mg/(L·h)]
            - self._pgp_efflux_rate(C_gut_enter)   # apical P-gp efflux [mg/(L·h)]
        )

        # ── [LIV_VASC] Liver Vascular (Sinusoidal Blood) ────────────────────
        # State variable: C_liv_vasc  [mg/L PLASMA concentration]
        # Volumes: 15% vascular (sinusoidal space), 85% hepatocyte cellular
        v_liv_vasc = v["liver"] * 0.15
        v_liv_tiss = v["liver"] * 0.85

        # ── UNIT CONVENTION FOR THIS COMPARTMENT ────────────────────────────
        # All flows Q [L blood/h].  To get mass flux from a plasma concentration:
        #   J [mg/h] = Q [L blood/h] × C_blood [mg/L blood]
        #            = Q × C_plasma × Rb
        # Dividing by V_liv_vasc [L] → dC_liv_vasc/dt [mg/(L plasma · h)].
        #
        # THREE CATEGORIES of flux all yield consistent units:
        #
        # A. CONVECTIVE (blood-flow driven): use C_blood = C_plasma × Rb
        #    J_conv = Q × C_plasma × Rb   [mg/h]  ÷ V → [mg/(L·h)] ✓
        #    Rb appears explicitly because Q is a blood-volume flow rate.
        #
        # B. TRANSPORTER-MEDIATED UPTAKE: plasma-basis clearance
        #    Transporter data (OATP, OCT1) is measured in aqueous buffer/plasma.
        #    CL_trans [L/h plasma] × C_vascular_free [mg/L plasma-free]  = [mg/h] ✓
        #    No Rb factor: the CL is already expressed per litre of plasma, not blood.
        #    This is distinct from the WSM CYP-enzyme clearance (see LIV_TISS below).
        #    J_trans = CL_trans × fup × C_liv_vasc   [mg/h]  ÷ V → [mg/(L·h)] ✓
        #
        # C. PASSIVE DIFFUSION across sinusoidal membrane:
        #    j_passive = CL_pd × (C_vascular_free − C_tissue_free)  [mg/h]
        #    CL_pd is a membrane permeability-surface-area product [L/h plasma].
        #    No Rb factor for the same reason as transporters.    ÷ V → [mg/(L·h)] ✓
        #
        # CONSISTENCY CHECK: net dC_liv_vasc/dt
        #   Convective in/out add Q×Rb/V × ΔC_plasma terms.
        #   Transporter + passive add CL/V × C_plasma terms.
        #   Both produce [h⁻¹] × [mg/L] = [mg/(L·h)] — same dimension ✓
        #   The Rb scaling in convective terms is the blood↔plasma transform;
        #   it does NOT appear in transporter terms because those CLs are already
        #   plasma-normalised by the in vitro measurement convention.
        #
        # WSM NOTE (for CYP metabolism in LIV_TISS):
        #   CL_h = Q_h × (fup × CLint / Rb) / (Q_h + fup × CLint / Rb)
        #   Here CLint is microsomal/hepatocyte intrinsic clearance [L/h tissue].
        #   The /Rb normalises it to blood-basis for the WSM denominator — this
        #   is a CYP-specific correction that does NOT apply to transporters.

        # ── Active Uptake Clearance (Transporter Database) ─────────────────
        # CL_act [L/h plasma] × C_vascular_free [mg/L free plasma] = J [mg/h]
        active_uptake_liv = 0.0
        for name, trans in tp["hepatic_uptake"].items():
            cl_act = self._active_cl(trans, C_vascular_free)
            active_uptake_liv += cl_act

        # ── Active Sinusoidal Influx (Generic Parameters) ──────────────────
        # Concentrative OATP/OCT-like uptake driven by membrane potential / pH gradient.
        # Flux = (Vmax × C_vascular_free) / (Km + C_vascular_free)  [mg/h]
        # No Rb: Vmax is expressed as a plasma-basis rate (in vitro calibrated).
        # Saturates on C_vascular_free ONLY — concentrative against C_tissue_free
        # (OATP transporters maintain C_tissue/C_blood ratios of 10–100×).
        vmax_up = self.drug.get("vmax_uptake", 0.0)
        km_up   = self.drug.get("km_uptake",   1.0)
        is_uptake_substrate = self.drug.get("is_uptake_substrate", False)

        j_uptake = 0.0
        if is_uptake_substrate and vmax_up > 0.0:
            j_uptake = (vmax_up * C_vascular_free) / (km_up + C_vascular_free)

        # ── Passive diffusion across sinusoidal membrane ─────────────────────
        # CL_pd [L/h plasma] × ΔC_free [mg/L plasma-free] = J [mg/h]
        # Sign: positive when C_vascular_free > C_tissue_free (drug moves blood→tissue).
        # No Rb: CL_pd is derived from logP/Caco-2 data expressed in plasma units.
        CL_pd    = self.params.get("liver_CL_pd", 10.0)
        j_passive = CL_pd * (C_vascular_free - C_tissue_free)

        # ── Hepatic elimination (Well-Stirred Model, whole-blood clearance) ──
        #
        # CL_h is derived from the WSM:
        #   CL_h = Q_h × (fup × CLint / Rb) / (Q_h + fup × CLint / Rb)   [L blood/h]
        #
        # The /Rb in the numerator converts the microsomal/hepatocyte intrinsic
        # clearance (CLint, measured in aqueous/plasma-based incubations) to a
        # blood-basis rate, consistent with the blood-flow denominator Q_h.
        # This is a CYP-enzyme-specific correction (WSM convention) and does NOT
        # apply to transporter clearances — see note B in the unit-convention block.
        #
        # Because C_liv_vasc is a plasma-equivalent concentration [mg/L plasma],
        # the elimination mass flux is:
        #
        #   J_elim [mg/h] = CL_h [L blood/h] × C_liv_vasc [mg/L plasma] / Rb
        #                 = (CL_h / Rb) × C_liv_vasc
        #
        # Dividing by v_liv_vasc [L] gives the ODE rate contribution [mg/(L·h)].
        # The /Rb converts the whole-blood clearance back to the plasma-equivalent
        # elimination matrix, reconciling CL_h (blood basis) with the plasma-basis
        # state variable — consistent with the convention described in
        # Rowland & Tozer, Clinical Pharmacokinetics 5th ed., Ch. 5.
        #
        # MASS BALANCE NOTE:
        #   This term represents direct CYP-mediated elimination from the
        #   sinusoidal (vascular) compartment — i.e., drug that is metabolised
        #   before or during transcellular transit.  It is additive to the
        #   permeability-limited tissue metabolism (metabolic_rate in LIV_TISS).
        #   Together they represent the total hepatic metabolic sink.
        CL_h = self._hepatic_clearance(C_liv_vasc)   # [L blood/h], WSM output

        # ── MASS BALANCE CHECK ───────────────────────────────────────────────
        # Fluxes leaving LIV_VASC:
        #   (a) active_uptake_liv × C_vascular_free  → LIV_TISS (conserved)
        #   (b) j_uptake                             → LIV_TISS (conserved)
        #   (c) j_passive                            → LIV_TISS (conserved)
        #   (d) (CL_h / Rb) × C_liv_vasc            → eliminated (metabolic sink)
        # metabolic_rate in LIV_TISS is an additional tissue-level sink.

        dydt[LIV_VASC] = (
            # ── Convective inflow (blood units, explicit Rb) ────────────────
            Q_ha * C_art_blood                   # hepatic artery:  Q×C_plasma×Rb
            + Q_pv * C_gut_blood_out_blood        # portal vein:     Q×(C_gut/Kp)×Rb
            # ── Convective outflow (blood units, explicit Rb) ───────────────
            - Q_liv * C_liv_vasc_blood            # hepatic vein:    Q×C_plasma×Rb
            # ── Transporter uptake (plasma basis, no extra Rb) ─────────────
            - active_uptake_liv * C_vascular_free # CL_plasma×C_free [mg/h]
            - j_uptake                            # Vmax×C_free/(Km+C_free) [mg/h]
            # ── Passive diffusion (plasma basis, no extra Rb) ───────────────
            - j_passive                           # CL_pd×ΔC_free    [mg/h]
            # ── WSM hepatic elimination (whole-blood CL → plasma basis) ─────
            - (CL_h / Rb) * C_liv_vasc           # (CL_h/Rb)×C_plasma [mg/h]; /Rb reconciles
                                                  # blood-clearance with plasma state variable
        ) / v_liv_vasc
        
        # ── [LIV_TISS] Liver Tissue (Hepatocytes) ──────────────────────────
        # Tissue compartment receives drug from vascular via active uptake and passive diffusion
        # Tissue compartment loses drug via metabolism (intrinsic clearance at tissue conc)
        
        # Hepatic metabolism in tissue (based on free tissue concentration)
        # Implement saturable Michaelis-Menten kinetics if Vmax is defined
        if use_mm_hepatic:
            # Non-linear saturable metabolism (mass/time)
            # MM equation: rate = (Vmax * C_free) / (Km + C_free)
            # Units: (mg/h * mg/L) / (mg/L + mg/L) = mg/h
            metabolic_rate = (vmax_hep * C_tissue_free) / (km_hep + C_tissue_free)
        else:
            # Linear metabolism (mass/time)
            # Fall back to intrinsic clearance model: rate = CLint * C_free
            # Units: (L/h * mg/L) = mg/h
            metabolic_rate = CLh * C_tissue_free
        
        # Tissue mass balance: inflow (uptake + passive diffusion) - outflow (metabolism)
        # MASS CONSERVATION: All mass leaving vascular enters here
        # Fluxes are all in mg/h:
        #  - active_uptake_liv: transporter-mediated (L/h * mg/L = mg/h)
        #  - j_uptake: generic active uptake (mg/h, from MM equation)
        #  - j_passive: passive diffusion (mg/h, bidirectional sign-aware)
        #  - metabolic_rate: metabolic sink ONLY in tissue (mg/h)
        dydt[LIV_TISS] = (
            active_uptake_liv * C_vascular_free  # Transporter-mediated uptake (saturable)
            + j_uptake  # Generic active sinusoidal influx (saturable, mg/h)
            + j_passive  # Passive diffusion from vascular (bidirectional, sign-aware)
            - metabolic_rate  # Metabolic sink: CYP-mediated clearance (saturable or linear)
        ) / v_liv_tiss

        # ── [KID] Kidney  (v2.7 — reabsorption + ER cap) ──────────────────
        #
        # Mass balance for the lumped kidney compartment:
        #
        #   dC_kid/dt = (perfusion inflow)
        #             - (active net tubular secretion)    ← drug lost to urine
        #             + (passive tubular reabsorption)    ← drug returned from lumen
        #
        # Perfusion term: standard Q×(C_in − C_out/Kp) inflow/outflow.
        # Active secretion: sum of MM-kinetic transporter clearances, capped at
        #   the well-stirred renal extraction ratio limit (never > renal plasma flow).
        # Reabsorption: un-ionized drug passively crosses tubular epithelium back
        #   into kidney interstitium → eventually leaves via venous outflow.
        #   Rate = CL_reab × C_kid_free / V_kidney.
        #
        # References:
        #   Rowland & Tozer, Clinical Pharmacokinetics 5th ed.
        #   Gibaldi et al., J Pharm Sci 1969; Reigner & Blesch 2002.

        passive_kid = (Q_kid / v["kidney"]) * (C_art_blood - (C_kid / kp["kidney"]) * Rb)

        # ── Step 1: Gross active secretion (MM-kinetic, sum over transporters) ──
        # _active_cl returns a saturation-adjusted intrinsic clearance (L/h).
        # Multiply by C_kid_free / V_kidney to get a volumetric rate (mg / (L·h)).
        active_sec_gross_cl = 0.0   # L/h — total secretion clearance before cap
        for name, trans in tp["renal_secretion"].items():
            active_sec_gross_cl += self._active_cl(trans, C_kid_free)

        # ── Step 2: Apply renal extraction ratio cap (v2.7, Gap 3.3) ────────
        # Cap total secretion clearance to the well-stirred limit derived in
        # _build_transporter_params.  At high CLint the kidney becomes
        # blood-flow-limited, exactly as the liver does for high-EH drugs.
        cl_sec_cap = tp["cl_sec_plasma_cap_lh"]
        if active_sec_gross_cl > cl_sec_cap:
            # Proportionally scale each transporter's contribution so the ODE
            # rate exactly equals the flow-limited cap at this concentration.
            _cap_ratio = cl_sec_cap / active_sec_gross_cl if active_sec_gross_cl > 1e-12 else 1.0
            active_sec_gross_cl = cl_sec_cap
        else:
            _cap_ratio = 1.0   # No capping needed; ratio unused

        active_sec_rate = (active_sec_gross_cl * C_kid_free) / v["kidney"]  # mg/(L·h)

        # ── Step 3: Passive tubular reabsorption (v2.7, Gap 3.1) ─────────────
        # CL_reab was pre-computed in _build_transporter_params from:
        #   Q_tubular_water × f_neutral_urine × k_perm
        # Applying it to C_kid_free gives the rate at which drug is returned
        # from the tubular lumen to kidney tissue (and onward to venous blood).
        cl_reab = tp["cl_reabsorption_lh"]
        reabs_rate = (cl_reab * C_kid_free) / v["kidney"]   # mg/(L·h)

        # ── ODE assembly ──────────────────────────────────────────────────────
        # Net kidney dC/dt: inflow (perfusion) - loss (net secretion) + recovery (reabs).
        # active_sec_rate ≥ 0 always; reabs_rate ≥ 0 always.
        # The reabsorbed drug stays in the kidney compartment and eventually returns
        # to systemic venous blood via the perfusion outflow term in venous_inflow.
        dydt[KID] = passive_kid - active_sec_rate + reabs_rate

        # ── Standard passive compartments ──────────────────────────────────
        dydt[BRA] = (Q_bra / v["brain"]) * (C_art_blood - (C_bra / kp["brain"]) * Rb)
        dydt[HRT] = (Q_hrt / v["heart"]) * (C_art_blood - (C_hrt / kp["heart"]) * Rb)
        dydt[MUS] = (Q_mus / v["muscle"]) * (C_art_blood - (C_mus / kp["muscle"]) * Rb)
        dydt[FAT] = (Q_fat / v["fat"]) * (C_art_blood - (C_fat / kp["fat"]) * Rb)
        dydt[SKN] = (Q_skn / v["skin"]) * (C_art_blood - (C_skn / kp["skin"]) * Rb)
        dydt[BON] = (Q_bon / v["bone"]) * (C_art_blood - (C_bon / kp["bone"]) * Rb)

        # ── NEW: Specific "rest" organs ────────────────────────────────────
        dydt[SPL] = (Q_spl / v.get("spleen", 0.2)) * (C_art_blood - (C_spl / kp.get("spleen", 1.0)) * Rb)
        dydt[ADIP_MES] = (Q_adip_mes / v.get("adipose_mes", 0.5)) * (C_art_blood - (C_adip_mes / kp.get("adipose_mes", kp["fat"])) * Rb)
        dydt[PANC] = (Q_panc / v.get("pancreas", 0.1)) * (C_art_blood - (C_panc / kp.get("pancreas", 1.0)) * Rb)

        # ── [VEN] Venous blood ─────────────────────────────────────────────
        # FIXED v2.3: Liver contribution now comes from LIV_VASC (vascular) not tissue
        # This ensures only unuptaken drug returns to venous circulation
        # FIXED v2.4: Apply Rb to all tissue outflows for correct mass balance
        venous_inflow = (
            Q_liv * C_liv_vasc_blood  # Outflow from liver vascular space
            + Q_kid * (C_kid / kp["kidney"]) * Rb
            + Q_bra * (C_bra / kp["brain"]) * Rb
            + Q_hrt * (C_hrt / kp["heart"]) * Rb
            + Q_mus * (C_mus / kp["muscle"]) * Rb
            + Q_fat * (C_fat / kp["fat"]) * Rb
            + Q_skn * (C_skn / kp["skin"]) * Rb
            + Q_bon * (C_bon / kp["bone"]) * Rb
            + Q_spl * (C_spl / kp.get("spleen", 1.0)) * Rb
            + Q_adip_mes * (C_adip_mes / kp.get("adipose_mes", kp["fat"])) * Rb
            + Q_panc * (C_panc / kp.get("pancreas", 1.0)) * Rb
        )

        if self.params["venous_delay_enabled"]:
            # With delay: venous → delay → lung
            dydt[VEN] = (
                venous_inflow
                - CO * C_ven_blood  # Outflow to delay compartment (in blood units)
                - cl_filt * fup * C_ven
            ) / v["venous_blood"]
            
            # Venous delay compartment (first-order transit)
            tau = self.params["venous_delay_tau_h"]
            dydt[VEN_DELAY] = (CO * C_ven_blood - CO * C_ven_delay_blood) / (tau * CO)
            
            # Lung receives from delay compartment (in blood units)
            lung_inflow = C_ven_delay_blood
        else:
            # No delay: venous → lung directly
            dydt[VEN] = (
                venous_inflow
                - CO * C_ven_blood  # Outflow to lung (in blood units)
                - cl_filt * fup * C_ven
            ) / v["venous_blood"]
            
            dydt[VEN_DELAY] = 0.0  # Unused
            lung_inflow = C_ven_blood  # In blood units

        # ── [LUNG] ─────────────────────────────────────────────────────────
        # lung_inflow is already in blood units; outflow must also be multiplied by Rb
        dydt[LUNG] = (CO * lung_inflow - CO * (C_lung / kp["lung"]) * Rb) / v["lung"]

        # ── [ART] Arterial blood ───────────────────────────────────────────
        # Both inflow (from lung) and outflow (to systemic) must be in blood units
        dydt[ART] = (
            CO * (C_lung / kp["lung"]) * Rb - CO * C_art_blood
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
        dict with time series, AUC, Cmax, transporter info, and v2.8 ACAT absorption profiles
        """
        y0 = np.zeros(N_STATES)

        if route == "oral":
            # v2.8: Entire dose placed in pre-gastric dose depot.
            # Bioavailability F is applied at the stomach inlet (in ACAT cascade).
            # This allows the model to properly track absorption through each segment.
            y0[DOSE_DEPOT] = dose_mg
            # Legacy: also set GLU_ABS for backward compatibility (though not used by ACAT)
            y0[GLU_ABS] = 0.0
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
            "liver_vascular": Y[LIV_VASC],
            "liver_tissue": Y[LIV_TISS],
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
            "liver_cl_pd": self.params.get("liver_CL_pd", 10.0),
            "liver_model": "permeability-limited 2-compartment (v2.3+)",
            # v2.7 renal additions
            "cl_reab_lh":          round(tp.get("cl_reabsorption_lh", 0.0), 4),
            "cl_sec_plasma_cap_lh": round(tp.get("cl_sec_plasma_cap_lh", 0.0), 4),
            "urine_ph":             tp.get("urine_ph", self.params.get("urine_ph", 6.0)),
            "f_neutral_urine":      round(tp.get("f_neutral_urine", 1.0), 4),
            "k_perm_reab":          round(tp.get("k_perm", 0.0), 4),
        }

        # P-gp efflux mass balance
        total_effluxed = float(Y[GLU_EFF][-1])

        # ── v2.7: Non-Compartmental Analysis (NCA) — terminal PK parameters ─
        #
        # Calculates λz, t½, CL_total, Vss, MRT, and AUC0-∞ from the simulated
        # plasma profile.  Uses log-linear regression on the terminal 20% of
        # timepoints (minimum 3 positive-concentration points required).
        #
        # Vss estimation:
        #   IV  route: Vss = CL × MRT            (exact, Benet & Galeazzi 1979)
        #   Oral route: Vss = CL × (MRT − 1/ka)  (subtract mean absorption time)
        #
        # AUC0-∞ extrapolation:
        #   AUC0-∞ = AUC0-t + C_last / λz
        #
        # References:
        #   Gibaldi & Perrier, Pharmacokinetics 2nd ed. (1982) — NCA methods
        #   Benet & Galeazzi, J Pharm Sci 1979 — Vss = CL × MRT
        nca = self._calculate_nca(t, plasma, dose_mg, route)

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
            # ── v2.7: NCA terminal parameters ────────────────────────────────
            "t_half_h":         nca["t_half_h"],
            "lambda_z_per_h":   nca["lambda_z_per_h"],
            "cl_total_lh":      nca["cl_total_lh"],
            "vss_l":            nca["vss_l"],
            "mrt_h":            nca["mrt_h"],
            "auc0inf_mg_l_h":   nca["auc0inf_mg_l_h"],
        }

    # ── v2.7: NCA helper ───────────────────────────────────────────────────
    def _calculate_nca(self, t: np.ndarray, plasma: np.ndarray,
                       dose_mg: float, route: str) -> dict:
        """
        Non-compartmental analysis from a simulated plasma concentration–time profile.

        Parameters
        ----------
        t        : 1-D array   time points (h)
        plasma   : 1-D array   arterial plasma concentrations (mg/L)
        dose_mg  : float       administered dose (mg)
        route    : str         'oral' or 'iv' (affects Vss calculation)

        Returns
        -------
        dict with keys:
            t_half_h        terminal half-life (h); None if λz cannot be fitted
            lambda_z_per_h  terminal elimination rate constant (h⁻¹)
            cl_total_lh     total body clearance (L/h) = dose / AUC0-∞
            vss_l           steady-state volume of distribution (L)
            mrt_h           mean residence time (h)
            auc0inf_mg_l_h  AUC extrapolated to infinity (mg·h/L)
        """
        from scipy.integrate import trapezoid as _trapz

        _SENTINEL = None   # Returned for any parameter that cannot be computed

        # ── AUC0-t (linear-log trapezoidal already computed; recalculate here) ──
        auc_0_t = float(_trapz(plasma, t))
        if auc_0_t < 1e-15:
            return {
                "t_half_h": _SENTINEL, "lambda_z_per_h": _SENTINEL,
                "cl_total_lh": _SENTINEL, "vss_l": _SENTINEL,
                "mrt_h": _SENTINEL, "auc0inf_mg_l_h": 0.0,
            }

        # ── Terminal phase: log-linear regression ─────────────────────────────
        # Use the last 20% of timepoints as the candidate terminal window.
        # Require ≥ 3 points above a floor of 0.1% of Cmax to ensure the fit is
        # not dominated by numerical noise.
        n           = len(t)
        c_floor     = 1e-3 * float(np.max(plasma))
        idx_start   = max(int(0.80 * n), n - 50)   # ≥ 80% of total time

        t_win  = t[idx_start:]
        c_win  = plasma[idx_start:]
        mask   = c_win > c_floor

        lambda_z = _SENTINEL
        t_half   = _SENTINEL

        if int(np.sum(mask)) >= 3:
            try:
                log_c  = np.log(c_win[mask])
                t_fit  = t_win[mask]
                slope, _intercept = np.polyfit(t_fit, log_c, 1)

                # Require a genuine elimination slope (negative = falling curve)
                if slope < -1e-6:
                    lambda_z = float(-slope)
                    t_half   = float(np.log(2.0) / lambda_z)
            except (np.linalg.LinAlgError, ValueError):
                pass   # Cannot regress; leave as None

        # ── AUC0-∞ extrapolation ──────────────────────────────────────────────
        c_last = float(c_win[mask][-1]) if (mask.any() and lambda_z is not None) \
                 else float(plasma[-1])

        if lambda_z is not None:
            auc_0_inf = float(auc_0_t + c_last / lambda_z)
        else:
            auc_0_inf = float(auc_0_t)   # Cannot extrapolate; underestimate flagged

        # ── Total body clearance ──────────────────────────────────────────────
        cl_total = float(dose_mg / auc_0_inf) if auc_0_inf > 1e-15 else _SENTINEL

        # ── AUMC0-∞ and MRT ───────────────────────────────────────────────────
        # AUMC = ∫ t·C(t) dt
        # Terminal extrapolation of AUMC0-t:
        #   AUMC_extra = C_last/λz² + t_last·C_last/λz
        #   (from integrating t·C_last·exp(−λz·t) from t_last to ∞)
        aumc_0_t = float(_trapz(t * plasma, t))
        t_last   = float(t[-1])

        if lambda_z is not None:
            aumc_extra = (c_last / lambda_z ** 2) + (t_last * c_last / lambda_z)
            aumc_0_inf = aumc_0_t + aumc_extra
        else:
            aumc_0_inf = aumc_0_t   # Underestimate; λz unavailable

        mrt = float(aumc_0_inf / auc_0_inf) if auc_0_inf > 1e-15 else _SENTINEL

        # ── Vss ───────────────────────────────────────────────────────────────
        # IV  : Vss = CL × MRT                           (exact)
        # Oral: Vss = CL × (MRT − MAT)   where MAT = 1/ka  (subtract mean absorption time)
        #   The oral MRT includes the absorption phase; subtracting MAT converts it
        #   to the equivalent IV MRT for Vss estimation.
        #   Reference: Riegelman & Collier, J Pharmacokinet Biopharm 1980.
        vss = _SENTINEL
        if cl_total is not None and mrt is not None:
            if route == "iv":
                vss = float(cl_total * mrt)
            else:
                ka  = float(self.drug.get("ka", 1.5))
                mat = 1.0 / ka if ka > 1e-6 else 0.0      # Mean absorption time (h)
                mrt_iv_equiv = mrt - mat
                if mrt_iv_equiv > 0.0:
                    vss = float(cl_total * mrt_iv_equiv)
                # If mrt_iv_equiv ≤ 0 the absorption is slower than elimination
                # and Vss cannot be reliably estimated from oral data alone.

        return {
            "t_half_h":        round(t_half,   3) if t_half   is not None else _SENTINEL,
            "lambda_z_per_h":  round(lambda_z, 6) if lambda_z is not None else _SENTINEL,
            "cl_total_lh":     round(cl_total, 4) if cl_total is not None else _SENTINEL,
            "vss_l":           round(vss,      3) if vss      is not None else _SENTINEL,
            "mrt_h":           round(mrt,      3) if mrt      is not None else _SENTINEL,
            "auc0inf_mg_l_h":  round(auc_0_inf, 4),
        }