"""
physiology.py — Reference physiological parameters for the BodySim PBPK engine.

Sources:
  [ICRP89]  ICRP Publication 89 (2002) — organ volumes and blood flows
  [RODGERS] Rodgers & Rowland, J Pharm Sci 2006 — tissue composition
  [BROWN97] Brown et al., Toxicol Sci 1997 — allometric scaling
  [YU1996]  Yu et al., Pharm Res 1996 — ACAT GI transit / surface area
  [DRESSMAN] Dressman & Reppas, Eur J Pharm Biopharm 2000 — intestinal pH
  [SIMCYP]  Jamei et al., Expert Opin Drug Metab Toxicol 2009 — ACAT parameterisation
"""

import numpy as np

REFERENCE_HUMAN = {
    "weight_kg":  70.0,
    "height_cm": 170.0,
    "age_yr":     35.0,
    "sex":        "male",
    "bmi":        24.2,
    "egfr":      100.0,
    "cyp3a4_activity": 1.0,
    "cyp2d6_activity": 1.0,
    "disease_state": "healthy",
    # ── v2.7 ─────────────────────────────────────────────────────────────────
    # Urine pH governs passive tubular reabsorption for ionizable drugs.
    # Physiological range: 4.5 (acid load) → 8.5 (alkaline load); default 6.0.
    # Sources: Remer & Manz, J Am Diet Assoc 1995; Toto, Am J Kidney Dis 1992.
    "urine_ph": 6.0,
}

# Organ volumes (L) — ICRP-89 Table 2.8
ORGAN_VOLUMES = {
    "arterial_blood": 1.68,
    "venous_blood":   3.92,
    "lung":           1.17,
    "liver":          1.69,
    "kidney":         0.31,
    "brain":          1.45,
    "heart":          0.33,
    "muscle":        29.0,
    "fat":           14.5,
    "gut":            1.44,
    "skin":           7.8,
    "bone":          10.5,
    "rest":           5.5,
}

# Organ blood flows (L/h) — ICRP-89 Table 2.8
# hepatic artery = 17.4 L/h; portal vein = 69.6 L/h
ORGAN_FLOWS = {
    "cardiac_output":    374.0,
    "liver_hepatic":      17.4,
    "liver_portal":       69.6,
    "kidney":             74.4,
    "brain":              42.0,
    "heart":              13.5,
    "muscle":             66.0,
    "fat":                20.0,
    "gut":                69.6,
    "skin":               18.0,
    "bone":                5.0,
    "rest":               48.0,
}

# Tissue composition (water, neutral lipid, phospholipid) — [RODGERS]
TISSUE_COMPOSITION = {
    "liver":  (0.751, 0.0348, 0.0252),
    "kidney": (0.783, 0.0128, 0.0242),
    "brain":  (0.774, 0.0510, 0.0565),
    "heart":  (0.758, 0.0139, 0.0111),
    "muscle": (0.760, 0.0238, 0.0072),
    "fat":    (0.135, 0.8530, 0.0021),
    "gut":    (0.718, 0.0403, 0.0123),
    "skin":   (0.718, 0.0603, 0.0044),
    "bone":   (0.439, 0.0740, 0.0011),
    "lung":   (0.811, 0.0220, 0.0128),
    "rest":   (0.700, 0.0300, 0.0100),
}

ALLOMETRIC_EXPONENTS = {"volume": 0.75, "flow": 0.75, "clearance": 0.75}

# ═══════════════════════════════════════════════════════════════════════════
# v2.8 — ACAT (Advanced Compartmental Absorption and Transit) GI Parameters
# ═══════════════════════════════════════════════════════════════════════════
#
# The GI tract is divided into 7 luminal segments.  Each segment is
# characterised by:
#
#   transit_time_h      : mean residence time of luminal contents in that
#                         segment (h).  Transit rate constant k_t = 1/transit_time_h.
#   ph                  : local luminal pH used in the Henderson–Hasselbalch
#                         equation to compute the fraction unionized (f_u,i).
#   surface_area_factor : dimensionless scaling factor (relative to a reference
#                         jejunum surface area) that accounts for villi and
#                         microvilli amplification along the GI tract.
#
# Ordering: index 0 = Stomach (i=1 in the ODE), …, index 6 = Cecum/Colon (i=7).
#
# Primary sources:
#   [YU1996]    Yu LX et al., Pharm Res 1996;13:1673 — GI transit times
#   [DRESSMAN] Dressman JB, Reppas C, Eur J Pharm Biopharm 2000;49:241
#              — luminal pH values
#   [SIMCYP]   Jamei M et al., Expert Opin Drug Metab Toxicol 2009;5:211
#              — surface-area scaling factors
# ═══════════════════════════════════════════════════════════════════════════

GI_TRACT_ACAT = {
    # Segment name  : (transit_time_h, luminal_pH, surface_area_factor)
    # Keys must remain stable — the ODE uses integer indexing over the ordered
    # list, so do NOT reorder or rename entries between v2.8 releases.
    "stomach": {
        "transit_time_h":      0.25,
        "ph":                  1.5,
        "surface_area_factor": 0.01,
    },
    "duodenum": {
        "transit_time_h":      0.25,
        "ph":                  6.0,
        "surface_area_factor": 0.10,
    },
    "jejunum_1": {
        "transit_time_h":      0.50,
        "ph":                  6.2,
        "surface_area_factor": 1.00,
    },
    "jejunum_2": {
        "transit_time_h":      0.75,
        "ph":                  6.6,
        "surface_area_factor": 0.80,
    },
    "ileum_1": {
        "transit_time_h":      1.00,
        "ph":                  7.4,
        "surface_area_factor": 0.60,
    },
    "ileum_2": {
        "transit_time_h":      1.00,
        "ph":                  7.8,
        "surface_area_factor": 0.40,
    },
    "cecum_colon": {
        "transit_time_h":     18.00,
        "ph":                  6.5,
        "surface_area_factor": 0.10,
    },
}

# Convenience: ordered tuple of segment names (preserves index ↔ name mapping)
ACAT_SEGMENT_NAMES = tuple(GI_TRACT_ACAT.keys())   # length == 7

# Convenience: flat arrays for use in ODE hot-path (avoid dict lookup per step)
ACAT_TRANSIT_TIMES = tuple(GI_TRACT_ACAT[s]["transit_time_h"]      for s in ACAT_SEGMENT_NAMES)
ACAT_PH            = tuple(GI_TRACT_ACAT[s]["ph"]                   for s in ACAT_SEGMENT_NAMES)
ACAT_SA_FACTORS    = tuple(GI_TRACT_ACAT[s]["surface_area_factor"]  for s in ACAT_SEGMENT_NAMES)
N_ACAT_SEGMENTS    = len(ACAT_SEGMENT_NAMES)   # == 7


def gfr_from_age(age_yr, sex="male"):
    if age_yr <= 30:   base = 125.0 if sex == "male" else 118.0
    elif age_yr <= 50: base = 115.0 if sex == "male" else 108.0
    elif age_yr <= 65: base =  95.0 if sex == "male" else  88.0
    else:              base =  75.0 if sex == "male" else  68.0
    return base


# ── Lung Kp calculation ────────────────────────────────────────────────────
def lung_kp(logp: float, pka: float = None, drug_type: str = "neutral") -> float:
    """
    Calculate lung tissue-to-plasma partition coefficient (Kp_lung).

    The lung is NOT a simple mixing chamber. Drug accumulates differently
    based on physicochemical properties:

    Basic drugs (pKa > 7.4):  Ion trapping in acidic lung lysosomes
                               → HIGH Kp (2–20× plasma)
    Acidic drugs (pKa < 6):   Repelled by negative charge on membranes
                               → LOW Kp (0.2–0.6× plasma)
    Neutral drugs:             Driven by lipophilicity only
                               logP < 0  → Kp ≈ 0.5–0.8 (stays in water phase)
                               logP > 2  → Kp ≈ 1.5–4.0 (lipid partition)

    Source: Yeh & Bhatt, Drug Metab Dispos 2011;
            Rodgers & Rowland, J Pharm Sci 2006
    """
    fw_lung  = TISSUE_COMPOSITION["lung"][0]   # 0.811 water fraction
    fn_lung  = TISSUE_COMPOSITION["lung"][1]   # 0.022 neutral lipid
    fp_lung  = TISSUE_COMPOSITION["lung"][2]   # 0.013 phospholipid

    fw_plasma = 0.93
    fn_plasma = 0.0023
    fp_plasma = 0.0023

    logp_c = np.clip(logp, -4, 6)

    # Neutral lipid partition
    Kn  = 10 ** (0.7 * logp_c)
    # Phospholipid partition
    Kph = 10 ** (0.3 * logp_c)

    # Base Kp from tissue composition
    kp_base = (fw_lung / fw_plasma
               + fn_lung * Kn  / fn_plasma
               + fp_lung * Kph / fp_plasma) * 0.05  # scale to physiological range

    # Drug-type correction
    if drug_type == "basic" and pka and pka > 7.4:
        # Ion trapping in acidic lysosomes (pH 4.7 vs plasma pH 7.4)
        # Kp_lung_basic = Kp_neutral × 10^(pKa - 7.4) [capped]
        trap_factor = min(10 ** (pka - 7.4), 50.0)
        kp_base *= (1.0 + 0.3 * trap_factor)

    elif drug_type == "acidic" and pka and pka < 6.0:
        # Acidic drugs largely excluded from lung tissue
        kp_base *= 0.3

    # Neutral hydrophilic (caffeine-like): Kp should be ~0.6
    # Neutral lipophilic: Kp should be ~2–4
    if drug_type == "neutral":
        if logp_c < 0:
            kp_base = np.clip(kp_base, 0.4, 0.9)
        else:
            kp_base = np.clip(kp_base, 0.8, 5.0)

    return float(np.clip(kp_base, 0.1, 30.0))


def scale_physiology(weight_kg=70.0, age_yr=35.0, sex="male",
                     height_cm=170.0, egfr=None, disease_state="healthy",
                     urine_ph=6.0):
    """
    Scale physiological parameters to a specific subject.

    Parameters
    ----------
    urine_ph : float, optional
        Urinary pH for tubular reabsorption calculations (v2.7).
        Physiological range 4.5–8.5; default 6.0 (typical fasted adult).
        Acidic urine (pH < 6): promotes reabsorption of bases, excretion of acids.
        Alkaline urine (pH > 7): promotes reabsorption of acids, excretion of bases.

    Returns
    -------
    volumes : dict  organ volumes (L) scaled to subject
    flows   : dict  organ blood flows (L/h) scaled to subject
    params  : dict  subject physiological parameters (egfr, CYP activities, urine_ph)
    """
    bw_ratio   = weight_kg / 70.0
    scale_v    = bw_ratio ** 0.75
    scale_q    = bw_ratio ** 0.75
    sex_factor = 0.88 if sex == "female" else 1.0

    volumes = {k: v * scale_v * sex_factor for k, v in ORGAN_VOLUMES.items()}
    flows   = {k: v * scale_q              for k, v in ORGAN_FLOWS.items()}

    if egfr is None:
        egfr = gfr_from_age(age_yr, sex)

    disease_modifiers = _get_disease_modifiers(disease_state, egfr)
    egfr_effective    = egfr * disease_modifiers["egfr_factor"]
    volumes["kidney"] *= disease_modifiers["kidney_vol_factor"]
    volumes["liver"]  *= disease_modifiers["liver_vol_factor"]

    cyp3a4 = (_cyp3a4_activity(age_yr, sex)
               * disease_modifiers["cyp3a4_factor"])
    cyp2d6 = _cyp2d6_activity(age_yr, sex)

    # v2.7: clamp urine_ph to physiological range; disease modifiers can override
    urine_ph_effective = float(np.clip(urine_ph, 4.5, 8.5))
    if disease_state == "severe_ckd":
        # Metabolic acidosis in advanced CKD acidifies urine slightly
        urine_ph_effective = min(urine_ph_effective, 5.5)

    params = {
        "weight_kg":       weight_kg,
        "age_yr":          age_yr,
        "sex":             sex,
        "egfr":            egfr_effective,
        "cyp3a4_activity": cyp3a4,
        "cyp2d6_activity": cyp2d6,
        "disease_state":   disease_state,
        # ── v2.7 ─────────────────────────────────────────────────────────────
        "urine_ph":        urine_ph_effective,
    }
    return volumes, flows, params


def _get_disease_modifiers(disease_state, egfr):
    d = {"egfr_factor": 1.0, "kidney_vol_factor": 1.0,
         "liver_vol_factor": 1.0, "cyp3a4_factor": 1.0}
    if   disease_state == "mild_ckd":
        d.update({"egfr_factor": 0.6,  "kidney_vol_factor": 0.85})
    elif disease_state == "moderate_ckd":
        d.update({"egfr_factor": 0.35, "kidney_vol_factor": 0.70})
    elif disease_state == "severe_ckd":
        d.update({"egfr_factor": 0.1,  "kidney_vol_factor": 0.55})
    elif disease_state == "liver_disease":
        d.update({"liver_vol_factor": 1.3, "cyp3a4_factor": 0.4})
    return d

def _cyp3a4_activity(age_yr, sex):
    if age_yr < 18:  return 0.6
    if age_yr > 70:  return 0.75
    return 1.1 if sex == "female" else 1.0

def _cyp2d6_activity(age_yr, sex):
    if age_yr < 18:  return 0.7
    if age_yr > 70:  return 0.80
    return 1.0