"""
admet.py — Mechanistic drug property estimation for BodySim PBPK engine.

TRUE MECHANISTIC IMPLEMENTATION v2.2 (Fingerprint SAR Prediction):
  - Henderson-Hasselbalch ionization (pH-dependent)      
  - Morgan fingerprint structural similarity (replaces MW/logP guesswork)
  - Transporter Vmax/Km 
  - Enzyme-specific metabolism (CYP1A2, 2C9, 2C19, 2D6, 3A4)
  - Organ-specific pH modeling

Transporter Substrate Prediction:
  Defaults to Tanimoto similarity vs gold standard substrates (RDKit Morgan fingerprints)
  Falls back to Gaussian SAR rules if SMILES unavailable or RDKit not installed

v5.0 Phase 1 Additions:

Gap 1 — Zwitterion Two-pKa Ionization (fraction_ionized + permeability_ionization_correction):
  fraction_ionized() upgraded to handle two-pKa dict {"acid": float, "base": float}.
  For zwitterionic drugs, f_ion = 1 − (f_acid_neutral × f_base_neutral), where each
  factor is the classical single-site Henderson-Hasselbalch neutral fraction.
  permeability_ionization_correction() is updated consistently — it now calls the
  upgraded fraction_ionized() and benefits automatically.
  This eliminates the prior crash (10**dict_pKa → TypeError) when Ciprofloxacin or
  Amoxicillin's pka dict propagated into the Kp estimation loop.
  References:
    Avdeef A, Absorption and Drug Development 2003 — zwitterion ionization theory
    Varma et al., AAPS J 2010;12:670 — Ciprofloxacin two-pKa characterization

Gap 3 — Lysosomal Trapping for Lipophilic Basic Amines:
  _lysosomal_kp_correction() helper: de Duve ion-trapping model for lipophilic
  basic amines.  Computes a multiplicative Kp amplification factor driven by
  pH-partition between cytosol (pH 7.0) and lysosomal lumen (pH 4.8).
  Applied inside estimate_kp_values() Kp loop for every organ except lung
  (lung has its own physiology.lung_kp calculation).
  Corrects systematic Vd under-prediction for basic drugs with logP > 1.5 and
  pKa > 7 (Propranolol, Metoprolol, Metformin, Cimetidine, Ranitidine).
  Zero regression risk: correction = 1.0 for all acidic / neutral drugs and
  for basic drugs with pKa ≤ 7 (no trapping at near-neutral lysosomal pH).
  References:
    de Duve C et al., Biochem Pharmacol 1974;23:2495 — lysosomal ion-trapping model
    Trapp S, Horobin RW, Eur Biophys J 2005;34:959 — weak-base accumulation
    Kazmi F et al., Drug Metab Dispos 2013;41:897 — Kp correction validation
"""

import numpy as np
from .physiology import TISSUE_COMPOSITION, lung_kp

# RDKit imports (optional for structural fingerprint matching)
HAS_RDKIT = False
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS AND ORGAN pH VALUES
# ═══════════════════════════════════════════════════════════════════════════

PLASMA_COMPOSITION = {
    "water": 0.93,
    "neutral_lipid": 0.0023,
    "phospholipid": 0.0023
}

ORGAN_PH = {
    "plasma": 7.40,
    "liver": 7.20,        
    "kidney": 7.00,       
    "brain": 7.35,
    "heart": 7.20,
    "muscle": 7.00,
    "fat": 7.40,
    "gut": 6.50,          
    "skin": 7.40,
    "bone": 7.40,
    "lung": 7.40,
    "rest": 7.40,
    "stomach": 2.00,      
    "small_intestine": 6.50,
    "colon": 7.00,
}

PLASMA_PROTEINS = {
    "albumin": 42.0,      
    "aag": 0.7,           
    "globulins": 25.0,
}

# ═══════════════════════════════════════════════════════════════════════════
# ENGINEERING POLICY CONSTANTS — HEPATIC UPTAKE DOMINANCE GATE (Step 1, v5.2)
# ═══════════════════════════════════════════════════════════════════════════
#
# These three constants parameterise the decision of whether active hepatic
# uptake is dominant enough to exempt the liver compartment from the global
# empirical kp_scalar correction in build_drug_profile().
#
# They are ENGINEERING POLICY CHOICES, not published physiological constants.
# They are defined here as named, documented module-level constants (rather
# than inline magic numbers) so they are easy to locate, review, and revise
# as the drug validation set grows, and so that the companion pre-flight
# script (preflight_check_dominance_gate.py, Step 1e) can be kept in
# numerical sync by importing from a single source or by explicit annotation.
#
# _LIVER_CL_PASSIVE_TYPICAL_LH:
#   A representative passive hepatic clearance baseline (L/h) against which
#   active linear clearance (Vmax/Km) is compared to form the dominance ratio.
#   Set to 10.0 L/h to match the liver_CL_pd default in hepatic_module.py.
#   If that default is ever changed, this constant must be updated to stay
#   consistent with it.
#
# _LIVER_DOMINANCE_RATIO:
#   If (vmax_uptake / km_uptake) / _LIVER_CL_PASSIVE_TYPICAL_LH exceeds this
#   threshold, the liver is classified as "dominant" and exempted from
#   kp_scalar.  Set to 3.0 (i.e. active CL at least 3× passive baseline).
#   Atorvastatin: 500/3 ≈ 167 L/h → ratio ≈ 16.7 >> 3 → exempt.
#   Literature support: Bteich et al. PMC7065931 — OATP-driven hepatic
#   distribution collapses to passive-only when uptake is inhibited.
#
# _LIVER_BORDERLINE_RATIO:
#   A lower triage band between 1.0× and _LIVER_DOMINANCE_RATIO.  A drug
#   landing here is logged as "borderline" for human review rather than
#   silently classified as "minor" (no exemption applied today, but visible
#   for future revision of the dominance model to support a partial-exemption
#   tier).  Does not change runtime behaviour for current reference drugs.
#
# NOTE: the preflight_check_dominance_gate.py script (Step 1e) contains a
# mirror copy of these three values.  If you change any value here, also
# update that script — or, better, factor both into engine/constants.py
# and import from there to prevent silent numerical drift.

_LIVER_CL_PASSIVE_TYPICAL_LH = 10.0   # L/h — passive hepatic CL baseline
                                        # (matches hepatic_module liver_CL_pd)
_LIVER_DOMINANCE_RATIO       = 3.0    # active/passive ratio above which the
                                        # liver is exempted from kp_scalar
_LIVER_BORDERLINE_RATIO      = 1.0    # below DOMINANCE_RATIO but above this:
                                        # logged as "borderline" for human
                                        # review, not silently bucketed as
                                        # "minor" (no runtime change today)


# ═══════════════════════════════════════════════════════════════════════════
# IONIZATION CHEMISTRY (Henderson-Hasselbalch)
# ═══════════════════════════════════════════════════════════════════════════

def fraction_ionized(pka, pH, drug_type):
    """
    Single-site Henderson-Hasselbalch ionization fraction.

    For zwitterionic drugs with a two-pKa dict {"acid": float, "base": float},
    the function returns the net ionized fraction as 1 − f_neutral_zw, where
    f_neutral_zw = f_acid_neutral × f_base_neutral (simultaneous un-ionization
    of both sites — the true membrane-permeant species).

    For scalar pka and monoprotic drug types the classic single-site equation
    is applied unchanged.

    Parameters
    ----------
    pka       : float, dict {"acid": float, "base": float}, or None
    pH        : float  — compartment pH
    drug_type : str    — "acidic", "basic", "neutral", or "zwitterion"

    Returns
    -------
    float  fraction ionized [0, 1]  (0.0 for neutral drugs)
    """
    if pka is None or drug_type == "neutral":
        return 0.0

    # ── Zwitterion: two-site simultaneous ionization ──────────────────────
    if drug_type == "zwitterion":
        if isinstance(pka, dict) and "acid" in pka and "base" in pka:
            pKa_acid = float(pka["acid"])
            pKa_base = float(pka["base"])
            # Fraction with carboxylate group un-ionized (HA form)
            f_acid_neutral = 1.0 / (1.0 + 10.0 ** (pH - pKa_acid))
            # Fraction with amine group un-ionized (B form)
            f_base_neutral = 1.0 / (1.0 + 10.0 ** (pKa_base - pH))
            # Truly neutral species: both groups simultaneously un-ionized
            f_neutral_zw   = f_acid_neutral * f_base_neutral
            return float(np.clip(1.0 - f_neutral_zw, 0.0, 1.0))
        else:
            # Legacy scalar pka for a zwitterion — treat as weakly basic
            pka_scalar = float(pka) if not isinstance(pka, dict) else 7.0
            return float(1.0 / (1.0 + 10.0 ** (pH - pka_scalar)))

    # ── Monoprotic acids / bases ──────────────────────────────────────────
    # Resolve dict to scalar (safety guard — should not normally occur for
    # monoprotic types, but prevents AttributeError if the caller passes a
    # mis-typed dict).
    if isinstance(pka, dict):
        if drug_type == "acidic":
            pka = float(pka.get("acid", list(pka.values())[0]))
        else:
            pka = float(pka.get("base", list(pka.values())[0]))

    if drug_type == "acidic":
        return 1.0 / (1.0 + 10 ** (pka - pH))
    elif drug_type == "basic":
        return 1.0 / (1.0 + 10 ** (pH - pka))
    return 0.0


def permeability_ionization_correction(logp, pka, pH_donor, pH_acceptor, drug_type):
    """
    Compute the pH-partition permeability correction factor between two compartments.

    For zwitterions with a two-pKa dict the neutral fraction at each pH is the
    product of the individual single-site neutral fractions (simultaneous
    un-ionization of both groups), consistent with the ACAT two-pKa model
    (Gap 1, v5.0).  The correction is bounded below at 0.01 in the denominator
    to prevent numerical blow-up when the acceptor compartment is nearly fully
    ionized.

    Dimensional: f_neutral_donor [–] / f_neutral_acceptor [–] = correction [–] ✓
    """
    # fraction_ionized() now correctly handles dict pka and zwitterions.
    # f_neutral = 1 − f_ion for all drug types including zwitterions.
    f_ion_donor    = fraction_ionized(pka, pH_donor,    drug_type)
    f_ion_acceptor = fraction_ionized(pka, pH_acceptor, drug_type)

    f_neutral_donor    = float(np.clip(1.0 - f_ion_donor,    0.0, 1.0))
    f_neutral_acceptor = float(np.clip(1.0 - f_ion_acceptor, 0.0, 1.0))

    permeability_correction = f_neutral_donor / max(f_neutral_acceptor, 0.01)

    return {
        "f_neutral_donor":        f_neutral_donor,
        "f_neutral_acceptor":     f_neutral_acceptor,
        "permeability_correction": permeability_correction,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MOLECULAR DESCRIPTORS (From Structure)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_molecular_descriptors(logp, mw, pka, drug_type, hbd=None, hba=None, psa=None):
    if hbd is None:
        if drug_type == "acidic":
            hbd = 1 + int(mw / 200)
        elif drug_type == "basic":
            hbd = 0 + int(mw / 300)
        else:
            hbd = int(mw / 250)
    
    if hba is None:
        if drug_type == "acidic":
            hba = 2 + int(mw / 150)
        elif drug_type == "basic":
            hba = 1 + int(mw / 200)
        else:
            hba = int(mw / 200)
    
    if psa is None:
        psa = 20.0 * (hbd + hba)
    
    lle = logp - (0.1 * psa)
    
    return {
        "mw": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "psa": psa,
        "lle": lle,
        "rotatable_bonds": max(0, int((mw - 100) / 30)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# TRANSPORTER DATABASE (Kinetic Parameters)
# ═══════════════════════════════════════════════════════════════════════════

HEPATIC_TRANSPORTERS = {
    "OATP1B1": {
        "location": "sinusoidal_uptake",
        "Vmax": 45.0,
        "Km": 8.3,
        "abundance": 3.5,
        "default_scale": 0.35,
        "substrate_rules": {
            "required": ["anionic_or_zwitterionic", "amphipathic"],
            "favorable": {"mw_range": (300, 700), "logp_range": (0, 4)},
            "typical_substrates": ["atorvastatin", "rosuvastatin", "pitavastatin"]
        }
    },
    "OATP1B3": {
        "location": "sinusoidal_uptake",
        "Vmax": 38.0,
        "Km": 12.5,
        "abundance": 1.8,
        "default_scale": 0.32,
        "substrate_rules": {
            "required": ["anionic"],
            "favorable": {"mw_range": (300, 800), "logp_range": (-1, 5)},
        }
    },
    "OCT1": {
        "location": "sinusoidal_uptake",
        "Vmax": 120.0,
        "Km": 35.0,
        "abundance": 7.5,
        "default_scale": 0.28,
        "substrate_rules": {
            "required": ["cationic"],
            "favorable": {"mw_range": (100, 400), "logp_range": (-3, 2)},
            "typical_substrates": ["metformin", "oxaliplatin"]
        }
    },
    "MRP2": {
        "location": "canalicular_efflux",
        "Vmax": 25.0,
        "Km": 45.0,
        "abundance": 5.2,
        "default_scale": 0.30,
        "substrate_rules": {
            "required": ["anionic_conjugate"],
            "favorable": {"mw_range": (400, 1000)},
        }
    },
}

RENAL_TRANSPORTERS = {
    "OAT1": {
        "location": "basolateral_secretion",
        "Vmax": 95.0,
        "Km": 28.0,
        "abundance": 8.5,
        "default_scale": 0.30,
        "substrate_rules": {
            "required": ["anionic"],
            "favorable": {"mw_range": (150, 500), "logp_range": (-2, 3)},
            "typical_substrates": ["furosemide", "NSAIDs", "beta-lactams"]
        }
    },
    "OAT3": {
        "location": "basolateral_secretion",
        "Vmax": 110.0,
        "Km": 32.0,
        "abundance": 6.8,
        "default_scale": 0.32,
        "substrate_rules": {
            "required": ["anionic"],
            "favorable": {"mw_range": (200, 600), "logp_range": (-1, 4)},
        }
    },
    "OCT2": {
        "location": "basolateral_secretion",
        "Vmax": 150.0,
        "Km": 40.0,
        "abundance": 12.0,
        "default_scale": 0.25,
        "substrate_rules": {
            "required": ["cationic"],
            "favorable": {"mw_range": (100, 500), "logp_range": (-3, 2)},
            "typical_substrates": ["metformin", "cimetidine"]
        }
    },
    "MATE1": {
        "location": "apical_efflux",
        "Vmax": 80.0,
        "Km": 55.0,
        "abundance": 4.5,
        "default_scale": 0.28,
        "substrate_rules": {
            "required": ["cationic"],
            "favorable": {"mw_range": (100, 400)},
        }
    },
}

GOLD_STANDARD_SUBSTRATES = {
    "OCT2": {
        "metformin": "CN(C)C(=N)NC",
        "cimetidine": "C1=C(N=C(S1)NC(=O)N)CCNC(=O)C",
    },
    "OCT1": {
        "metformin": "CN(C)C(=N)NC",
    },
    "OATP1B1": {
        "atorvastatin": "CC(C)Cc1c(C(=O)Nc2ccccc2)c(c(n1CCC(O)CC(O)CC(=O)O)c3ccc(F)cc3)c4ccccc4",
        "rosuvastatin": "CC(C)N(C)c1nc(nc(c1/C=C/[C@@H](O)C[C@@H](O)CC(=O)O)c2ccc(F)cc2)S(=O)(=O)C",
    },
    "OAT1": {
        "furosemide": "NS(=O)(=O)c1cc(ccc1Cl)C(=O)Nc2ccccc2C(=O)O",
        "probenecid": "CC(=O)Nc1ccc(cc1)C(=O)c2ccccc2C(=O)O",
    },
}

GOLD_STANDARD_FINGERPRINTS = {}

def _compute_gold_standard_fingerprints():
    global GOLD_STANDARD_FINGERPRINTS
    if not HAS_RDKIT:
        return
    for transporter, substrates in GOLD_STANDARD_SUBSTRATES.items():
        GOLD_STANDARD_FINGERPRINTS[transporter] = {}
        for drug_name, smiles in substrates.items():
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    GOLD_STANDARD_FINGERPRINTS[transporter][drug_name] = fp
            except Exception:
                pass


def _get_smiles_fingerprint(smiles, radius=2, nbits=2048):
    if not HAS_RDKIT or smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    except Exception:
        pass
    return None


def _calculate_structural_similarity(drug_smiles, transporter_name):
    if not HAS_RDKIT or transporter_name not in GOLD_STANDARD_FINGERPRINTS:
        return None, None
    drug_fp = _get_smiles_fingerprint(drug_smiles)
    if drug_fp is None:
        return None, None
    gold_standards = GOLD_STANDARD_FINGERPRINTS[transporter_name]
    if not gold_standards:
        return None, None
    
    max_similarity = 0.0
    best_match = None
    
    for ref_name, ref_fp in gold_standards.items():
        similarity = DataStructs.TanimotoSimilarity(drug_fp, ref_fp)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = ref_name
            
    return max_similarity, best_match

_compute_gold_standard_fingerprints()


def _gaussian_score(value, optimal_range):
    min_val, max_val = optimal_range
    optimal_val = (min_val + max_val) / 2.0
    sigma = (max_val - min_val) / 2.0 
    if sigma == 0: sigma = 1.0
    return np.exp(-0.5 * ((value - optimal_val) / sigma)**2)


# ═══════════════════════════════════════════════════════════════════════════
# TRANSPORTER SUBSTRATE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_transporter_substrate(transporter_name, transporter_db, descriptors, 
                                   drug_type, pka, smiles=None):
    transporter = transporter_db[transporter_name]
    rules = transporter["substrate_rules"]
    
    if HAS_RDKIT and smiles is not None:
        similarity, matched_substrate = _calculate_structural_similarity(smiles, transporter_name)
        if similarity is not None:
            probability = similarity * 0.95
            required_met = False
            for req in rules.get("required", []):
                if req in ("anionic", "anionic_or_zwitterionic") and drug_type == "acidic":
                    if fraction_ionized(pka, 7.4, "acidic") > 0.1: required_met = True
                elif req == "cationic" and drug_type == "basic":
                    if fraction_ionized(pka, 7.4, "basic") > 0.1: required_met = True
                elif req == "amphipathic":
                    if 0 < descriptors["logp"] < 5 and descriptors["psa"] > 40:
                        required_met = True
            
            if not required_met and rules.get("required"):
                probability *= 0.3
            
            affinity_modifier = 1.0 + (1.0 - similarity)
            
            return {
                "is_substrate": probability > 0.4,
                "probability": np.clip(probability, 0.0, 0.95),
                "affinity_modifier": np.clip(affinity_modifier, 0.5, 2.0),
                "method": "fingerprint",
                "similarity": similarity,
                "matched_substrate": matched_substrate
            }
    
    required_met = False
    for req in rules.get("required", []):
        if req in ("anionic", "anionic_or_zwitterionic") and drug_type == "acidic":
            if fraction_ionized(pka, 7.4, "acidic") > 0.1: required_met = True
        elif req == "cationic" and drug_type == "basic":
            if fraction_ionized(pka, 7.4, "basic") > 0.1: required_met = True
        elif req == "amphipathic" and 0 < descriptors["logp"] < 5 and descriptors["psa"] > 40:
            required_met = True
            
    if not required_met and rules.get("required"):
        return {
            "is_substrate": False, "probability": 0.0, "affinity_modifier": 1.0,
            "method": "gaussian", "similarity": None
        }
    
    probability = 0.4 if required_met else 0.05
    if "favorable" in rules:
        fav = rules["favorable"]
        score = 1.0
        if "mw_range" in fav:
            score *= _gaussian_score(descriptors["mw"], fav["mw_range"])
        if "logp_range" in fav:
            score *= _gaussian_score(descriptors["logp"], fav["logp_range"])
        probability += 0.5 * score

    probability = np.clip(probability, 0.0, 0.95)
    mw_score = _gaussian_score(descriptors["mw"], rules.get("favorable", {}).get("mw_range", (300, 300)))
    affinity_modifier = 1.0 + (1.0 - mw_score)
    
    return {
        "is_substrate": probability > 0.4, "probability": probability,
        "affinity_modifier": affinity_modifier, "method": "gaussian", "similarity": None
    }


# ═══════════════════════════════════════════════════════════════════════════
# CYP ENZYME DATABASE (Kinetic Parameters)
# ═══════════════════════════════════════════════════════════════════════════

CYP_ENZYMES = {
    "CYP3A4": {
        "abundance": 30.0, "turnover_base": 15.0, "fraction_hepatic": 0.30,
        "substrate_rules": {"favorable": {"mw_range": (250, 700), "logp_range": (1.5, 5.5)}}
    },
    "CYP2D6": {
        "abundance": 5.0, "turnover_base": 25.0, "fraction_hepatic": 0.20,
        "substrate_rules": {"required": ["basic"], "favorable": {"mw_range": (200, 500), "logp_range": (1, 4)}}
    },
    "CYP2C9": {
        "abundance": 25.0, "turnover_base": 8.0, "fraction_hepatic": 0.15,
        "substrate_rules": {"required": ["acidic"], "favorable": {"mw_range": (200, 400), "logp_range": (2, 4)}}
    },
    "CYP2C19": {
        "abundance": 8.0, "turnover_base": 12.0, "fraction_hepatic": 0.10,
        "substrate_rules": {"favorable": {"mw_range": (250, 500), "logp_range": (1, 4)}}
    },
    "CYP1A2": {
        "abundance": 12.0, "turnover_base": 10.0, "fraction_hepatic": 0.10,
        "substrate_rules": {"favorable": {"mw_range": (150, 350), "logp_range": (-0.5, 3)}}
    },
}

def predict_cyp_metabolism(descriptors, drug_type, pka):
    predictions = {}
    for enzyme_name, enzyme_data in CYP_ENZYMES.items():
        rules = enzyme_data.get("substrate_rules", {})
        
        if "required" in rules:
            if "acidic" in rules["required"] and drug_type != "acidic": continue
            if "basic" in rules["required"] and drug_type != "basic": continue
            
        probability = 0.02
        if "favorable" in rules:
            fav = rules["favorable"]
            score = 1.0
            if "mw_range" in fav:
                score *= _gaussian_score(descriptors["mw"], fav["mw_range"])
            if "logp_range" in fav:
                score *= _gaussian_score(descriptors["logp"], fav["logp_range"])
            probability += 0.88 * score
                
        probability = np.clip(probability, 0.0, 0.95)
        if probability > 0.25:
            km_base = 15.0 * (10 ** (0.25 * (3.5 - np.clip(descriptors["logp"], -1.0, 5.0))))
            predictions[enzyme_name] = {
                "probability": probability, 
                "Vmax": enzyme_data["abundance"] * enzyme_data["turnover_base"] * probability, 
                "Km": np.clip(km_base, 2.0, 120.0)
            }
    return predictions


# ═══════════════════════════════════════════════════════════════════════════
# PROTEIN BINDING
# ═══════════════════════════════════════════════════════════════════════════

def calculate_protein_binding(logp, mw, drug_type, pka, fup_measured=None):
    f_ion = fraction_ionized(pka, 7.4, drug_type)
    
    if drug_type == "acidic" and f_ion > 0.3:
        log_ka_albumin = 4.0 + 0.5 * logp + 1.0 * f_ion
        ka_albumin = 10 ** np.clip(log_ka_albumin, 3.0, 7.0)
        albumin_molar = (PLASMA_PROTEINS["albumin"] / 66500.0) * 1e6
        fraction_bound_albumin = (ka_albumin * albumin_molar * 1e-6) / (1 + ka_albumin * albumin_molar * 1e-6)
    else:
        ka_albumin = 0
        fraction_bound_albumin = 0.0
    
    if drug_type == "basic" and f_ion > 0.2:
        log_ka_aag = 4.5 + 0.7 * logp
        ka_aag = 10 ** np.clip(log_ka_aag, 3.0, 7.0)
        aag_molar = (PLASMA_PROTEINS["aag"] / 41000.0) * 1e6
        fraction_bound_aag = (ka_aag * aag_molar * 1e-6) / (1 + ka_aag * aag_molar * 1e-6)
    else:
        ka_aag = 0
        fraction_bound_aag = 0.0
    
    if drug_type == "neutral" and logp > 2:
        fraction_bound_nonspecific = 0.3 * (1 - np.exp(-0.5 * (logp - 2)))
    else:
        fraction_bound_nonspecific = 0.0
    
    fraction_bound_total = np.clip(fraction_bound_albumin + fraction_bound_aag + fraction_bound_nonspecific, 0.0, 0.99)
    fup_predicted = 1.0 - fraction_bound_total
    
    if fup_measured is not None:
        fup = fup_measured
        if fraction_bound_total > 0:
            scale_factor = (1 - fup) / fraction_bound_total
            fraction_bound_albumin *= scale_factor
            fraction_bound_aag *= scale_factor
    else:
        fup = fup_predicted
    
    return {
        "fup": fup,
        "albumin_binding": {"Ka": ka_albumin, "fraction_bound": fraction_bound_albumin},
        "aag_binding": {"Ka": ka_aag, "fraction_bound": fraction_bound_aag},
        "nonspecific_binding": fraction_bound_nonspecific,
        "fu_tissue": fup * 1.5,
    }


def _lysosomal_kp_correction(logp: float, pka, drug_type: str,
                              pH_lysosome: float = 4.8,
                              pH_cytosol:  float = 7.0,
                              phi_lys:     float = 0.02) -> float:
    """
    Multiplicative Kp amplification factor from lysosomal ion-trapping for
    lipophilic basic amines (Gap 3, v5.0 Phase 1).

    Mechanistic basis (de Duve et al. 1974; Trapp & Horobin 2005):
        Lysosomes occupy fraction phi_lys (~2%) of cell volume at pH ~4.8.
        The neutral (membrane-permeant) form of a basic drug equilibrates
        across the lysosomal membrane.  Inside the acidic lumen the drug
        becomes protonated and cannot back-diffuse → concentration ratio:

            R_lys = f_neutral_cyto / f_neutral_lys              [–]

        where:
            f_neutral_cyto = 1 / (1 + 10^(pKa − pH_cytosol))  — neutral fraction at pH 7.0
            f_neutral_lys  = 1 / (1 + 10^(pKa − pH_lysosome)) — neutral fraction at pH 4.8

        Corrected tissue Kp:
            Kp_corrected = Kp_passive × (1 + phi_lys × (R_lys − 1))     [–]

        Dimensional note:
            phi_lys [–] × R_lys [–] = [–]; Kp_passive [–] × correction [–] = [–] ✓

    Parameters
    ----------
    logp         : float   Drug lipophilicity (neutral form). Used only as a
                           guard: trapping is negligible for logP < 1.5 because
                           the drug cannot cross the lysosomal membrane efficiently
                           regardless of the pH gradient.
    pka          : float or dict or None
                           Basic pKa value.  If a dict is passed (two-pKa zwitterion
                           model), the 'base' key is extracted.  None → no correction.
    drug_type    : str     Only 'basic' drugs are subject to lysosomal trapping.
                           Acidic and neutral drugs return 1.0.
    pH_lysosome  : float   Lysosomal lumen pH (default 4.8; range 4.5–5.0 in
                           endosomes/lysosomes; Mindell 2012).
    pH_cytosol   : float   Cytosolic pH (default 7.0; Roos & Boron 1981).
    phi_lys      : float   Lysosomal volume fraction of cell (default 0.02 = 2%;
                           Luzio et al. 2007 — consistent across cell types).

    Returns
    -------
    float   Kp correction factor ≥ 1.0.  Returns 1.0 (no correction) for:
            - Non-basic drugs (acidic, neutral, zwitterion with only acidic pKa).
            - Basic drugs with logP < 1.5 (insufficient membrane permeability
              for the trapping mechanism to operate).
            - Cases where f_neutral_lys < 1e-12 (numerical guard).

    References
    ----------
    de Duve C et al., Biochem Pharmacol 1974;23:2495
    Trapp S, Horobin RW, Eur Biophys J 2005;34:959
    Mindell JA, Annu Rev Physiol 2012;74:69 — lysosomal pH measurements
    Kazmi F et al., Drug Metab Dispos 2013;41:897 — in vitro Kp validation
    Luzio JP et al., Nat Rev Mol Cell Biol 2007;8:622 — lysosomal volume
    """
    # ── Guard 1: Only basic drugs undergo lysosomal trapping ──────────────────
    # Acidic and neutral drugs are not protonated in lysosomes; no trapping.
    # Zwitterions with only an acidic pKa also return 1.0.
    if drug_type != "basic":
        return 1.0

    # ── Guard 2: Resolve pKa to a scalar basic value ──────────────────────────
    # Support two-pKa dict format {\"acid\": ..., \"base\": ...} used by zwitterions.
    # For scalar pka with drug_type=\"basic\", use directly.
    if pka is None:
        return 1.0

    if isinstance(pka, dict):
        # Two-pKa dict: extract basic pKa; return 1.0 if not present
        pka_basic = pka.get("base", None)
        if pka_basic is None:
            return 1.0
    else:
        pka_basic = float(pka)

    # ── Guard 3: Membrane permeability prerequisite ───────────────────────────
    # Lysosomal trapping requires the neutral form to cross the lysosomal
    # membrane.  For drugs with logP < 1.5 the membrane permeability of the
    # neutral form is insufficient for trapping to be physiologically significant.
    # Threshold logP = 1.5: empirical cutoff from Trapp & Horobin 2005.
    if logp < 1.5:
        return 1.0

    # ── Fraction un-ionized at cytosolic pH (BH⁺ ⇌ B + H⁺) ──────────────────
    # f_neutral = 1 / (1 + 10^(pKa − pH))
    # At cytosol pH 7.0 the neutral form B dominates for pKa < 7.
    # For high-pKa amines (pKa > 9) f_neutral_cyto is small but non-zero.
    f_neutral_cyto = 1.0 / (1.0 + 10.0 ** (pka_basic - pH_cytosol))

    # ── Fraction un-ionized at lysosomal pH ───────────────────────────────────
    # At pH 4.8, nearly all drug is protonated (f_neutral_lys → 0 for pKa > 6).
    # Guard denominator against numerical underflow.
    f_neutral_lys = 1.0 / (1.0 + 10.0 ** (pka_basic - pH_lysosome))
    if f_neutral_lys < 1e-12:
        # Drug is essentially fully protonated in the lysosome.
        # R_lys → f_neutral_cyto / 1e-12 which is astronomically large but
        # physiologically capped at ~phi_lys * R_lys ≈ 0.02 * 1e8 → nonsensical.
        # In practice the R_lys ceiling is set by the lipid partition equilibrium;
        # clamp correction to a physically observed maximum (~200× for chloroquine).
        return float(np.clip(1.0 + phi_lys * (f_neutral_cyto / 1e-12 - 1.0), 1.0, 200.0))

    # ── Lysosomal concentration ratio ─────────────────────────────────────────
    # R_lys = C_lysosome / C_cytosol at equilibrium (neutral form equilibrates)
    R_lys = f_neutral_cyto / f_neutral_lys   # [–] ≥ 1

    # ── Kp amplification factor ───────────────────────────────────────────────
    # correction = 1 + phi_lys × (R_lys − 1)
    # Interpretation: phi_lys fraction of cell volume traps drug at R_lys × cytosol.
    # At R_lys = 1 (no trapping, e.g. neutral drugs): correction = 1.0. ✓
    # At R_lys = 100 (pKa 9.4, pH_lys 4.8): correction = 1 + 0.02 × 99 = 2.98.
    # Dimensional: [–] × ([–] − 1) = [–]; 1 + [–] = [–] ✓
    correction = 1.0 + phi_lys * (R_lys - 1.0)

    # Physical clamp: correction > 200× has never been observed for small molecules
    # with drug-like logP; values that high indicate edge-case parameter combinations.
    return float(np.clip(correction, 1.0, 200.0))


# ═══════════════════════════════════════════════════════════════════════════
# RODGERS & ROWLAND TISSUE:PLASMA PARTITION COEFFICIENT MODEL  (v5.4 fix)
#
# Replaces the v5.1-v5.3 passive Kn/Kph/P formula, which had two confirmed
# defects (v5.3 investigation log, Script 9/10):
#   1. fu_tissue = fup * 1.5 made P = fup/fu_tissue cancel to a hardcoded
#      constant (0.6667) for every drug, regardless of fup.
#   2. With Kn/Kph small (any low-to-moderate logP drug), the formula
#      degenerated toward kp ≈ 1/P for every tissue, destroying tissue-level
#      differentiation even though TISSUE_COMPOSITION itself was correct
#      (confirmed clean by Script 10 — only heart/muscle clustered, which is
#      physiologically expected).
#
# This block implements the real, published two-equation Rodgers & Rowland
# method (moderate-to-strong bases vs. everything else) plus the neutral
# case, faithfully ported from the verified R implementation in:
#
#   Utsey K, Gastonguay MS, Russell S, Freling R, Riggs MM, Elmokadem A.
#   "Quantification of the Impact of Partition Coefficient Prediction
#   Methods on Physiologically Based Pharmacokinetic Model Output Using a
#   Standardized Tissue Composition." Drug Metab Dispos. 2020;48(10):903-916.
#   doi:10.1124/dmd.120.090498.  Open access, CC BY 4.0.
#   Companion code: github.com/metrumresearchgroup/PBPK_PC
#   (script/CalcKp_R&R.R, data/unified_tissue_comp.csv — itself the
#   standardized human composite of Rodgers et al. 2005, Rodgers & Rowland
#   2006, Open Systems Pharmacology, and Ruark et al. 2014).
#
# Original sources implemented by that script (not directly read by us —
# the verified R code stands in for them, and was checked by direct
# execution: reproduces the paper's own Metoprolol example with realistic
# 20x brain:bone Kp differentiation, structurally impossible under the old
# formula):
#   Rodgers T, Leahy D, Rowland M. J Pharm Sci. 2005;94(6):1259-1276.
#   Rodgers T, Rowland M. J Pharm Sci. 2006;95(6):1238-1257.
#
# NOTE ON BP (blood:plasma ratio): the Ka_AP back-calculation for
# moderate-to-strong bases (pKa >= 7) requires a measured blood:plasma
# ratio. This engine has no literature-sourced BP per drug — it falls back
# to the existing empirical blood_plasma_ratio() correlation (originally
# written for Rb in the PK output, not for this purpose). This is a known
# approximation, flagged the same way the QSPR p_eff fallback was flagged
# in the v5.3 log; a literature BP sourcing pass for basic drugs would
# remove this dependency. It only affects drugs classified "basic" with
# pKa > 7, or zwitterions whose basic pKa > 7.
# ═══════════════════════════════════════════════════════════════════════════

# Standardized human tissue composition (Utsey et al. 2020, Table 1 / CSV).
# Columns: f_ew (extracellular water), f_iw (intracellular water),
#          f_n_l (neutral lipid), f_n_pl (neutral phospholipid),
#          f_a_pl (acidic phospholipid), AR (albumin interstitial:plasma
#          ratio), LR (lipoprotein interstitial:plasma ratio).
# Keys use this engine's existing organ names (adipose -> fat) so results
# slot directly into the kp dict used elsewhere in the codebase.
_RR_TISSUE_DATA = {
    "fat":    dict(f_ew=0.135, f_iw=0.017, f_n_l=0.798,  f_n_pl=0.0478, f_a_pl=0.0067,  AR=0.049, LR=0.069),
    "bone":   dict(f_ew=0.100, f_iw=0.346, f_n_l=0.074,  f_n_pl=0.0016, f_a_pl=0.0008,  AR=0.100, LR=0.050),
    "brain":  dict(f_ew=0.162, f_iw=0.620, f_n_l=0.045,  f_n_pl=0.0553, f_a_pl=0.02022, AR=0.048, LR=0.041),
    "heart":  dict(f_ew=0.320, f_iw=0.456, f_n_l=0.089,  f_n_pl=0.0079, f_a_pl=0.00309, AR=0.157, LR=0.160),
    "kidney": dict(f_ew=0.273, f_iw=0.483, f_n_l=0.036,  f_n_pl=0.0166, f_a_pl=0.00387, AR=0.130, LR=0.137),
    "gut":    dict(f_ew=0.282, f_iw=0.475, f_n_l=0.0487, f_n_pl=0.0124, f_a_pl=0.0035,  AR=0.158, LR=0.141),
    "liver":  dict(f_ew=0.161, f_iw=0.573, f_n_l=0.037,  f_n_pl=0.0115, f_a_pl=0.00258, AR=0.086, LR=0.161),
    "lung":   dict(f_ew=0.336, f_iw=0.446, f_n_l=0.003,  f_n_pl=0.0056, f_a_pl=0.0014,  AR=0.212, LR=0.168),
    "muscle": dict(f_ew=0.118, f_iw=0.630, f_n_l=0.013,  f_n_pl=0.0092, f_a_pl=0.0019,  AR=0.064, LR=0.059),
    "skin":   dict(f_ew=0.382, f_iw=0.291, f_n_l=0.036,  f_n_pl=0.0502, f_a_pl=0.01382, AR=0.277, LR=0.096),
}
_RR_RBC    = dict(f_iw=0.603, f_n_l=0.002, f_n_pl=0.0025, f_a_pl=0.0005)
_RR_PLASMA = dict(f_n_l=0.003, f_n_pl=0.005)
_RR_PH_IW, _RR_PH_P, _RR_PH_RBC, _RR_HCT = 7.0, 7.4, 7.22, 0.45


def _rr_classify(pka, drug_type):
    """
    Map this engine's (pka, drug_type) onto the Rodgers & Rowland ionization
    class needed to pick X/Y/Z and the type_calc branch (moderate-to-strong
    base vs. acid/weak-base/zwitterion vs. neutral).

    Returns (pka_acid_or_None, pka_base_or_None) — at most one is non-None
    except for zwitterions, which carry both.
    """
    if drug_type == "neutral" or pka is None:
        return None, None
    if drug_type == "zwitterion":
        if isinstance(pka, dict) and "acid" in pka and "base" in pka:
            return float(pka["acid"]), float(pka["base"])
        # Legacy scalar zwitterion pka — treat as weakly basic only.
        return None, float(pka) if not isinstance(pka, dict) else None
    if isinstance(pka, dict):
        pka = float(pka.get("acid" if drug_type == "acidic" else "base", list(pka.values())[0]))
    if drug_type == "acidic":
        return float(pka), None
    if drug_type == "basic":
        return None, float(pka)
    return None, None


def _calc_kp_rodgers_rowland(logp, fup, pka, drug_type, BP):
    """
    Faithful port of CalcKp_R&R.R (Utsey et al. 2020 companion repo),
    verified by direct execution against the paper's own Metoprolol
    example (logP=2.15, pKa=9.7, fup=0.879, BP=1.52) before being wired
    into this engine.

    Returns a dict of {organ: Kp} for every organ in _RR_TISSUE_DATA, plus
    "rest" (mean of all non-fat organs, matching the paper's own
    rest-of-body convention).
    """
    pka_acid, pka_base = _rr_classify(pka, drug_type)
    P = 10 ** logp
    logp_ow = 1.115 * logp - 1.35
    P_OW = 10 ** logp_ow

    if pka_acid is None and pka_base is None:
        # Neutral
        X = Y = 0.0
        Z = 1.0
        is_strong_base = False
    elif pka_acid is not None and pka_base is not None:
        # Zwitterion: acid + base sites simultaneously (R&R type 6)
        X = 10 ** (pka_base - _RR_PH_IW) + 10 ** (_RR_PH_IW - pka_acid)
        Y = 10 ** (pka_base - _RR_PH_P) + 10 ** (_RR_PH_P - pka_acid)
        Z = 10 ** (pka_base - _RR_PH_RBC) + 10 ** (_RR_PH_RBC - pka_acid)
        is_strong_base = pka_base > 7.0
    elif pka_acid is not None:
        # Monoprotic acid
        X = 10 ** (_RR_PH_IW - pka_acid)
        Y = 10 ** (_RR_PH_P - pka_acid)
        Z = 1.0
        is_strong_base = False
    else:
        # Monoprotic base
        X = 10 ** (pka_base - _RR_PH_IW)
        Y = 10 ** (pka_base - _RR_PH_P)
        Z = 10 ** (pka_base - _RR_PH_RBC)
        is_strong_base = pka_base > 7.0

    # Drug-protein affinity, back-calculated from fup (not assumed).
    Ka_PR = (1.0 / fup - 1.0
             - (P * _RR_PLASMA["f_n_l"] + (0.3 * P + 0.7) * _RR_PLASMA["f_n_pl"]) / (1.0 + Y))

    Ka_AP = None
    if is_strong_base:
        Kpu_bc = (_RR_HCT - 1.0 + BP) / (_RR_HCT * fup)
        # Physiological guard 1: Kpu_bc must exceed 1.0 before Ka_AP is meaningful.
        # Kpu_bc < 1 means the drug partitions LESS into RBCs than plasma — there is
        # no RBC accumulation to back-calculate Ka_AP from. Applying the formula anyway
        # divides a near-zero numerator by f_a_pl_rbc (0.0005) and Z, amplifying
        # numerical noise into a large spurious Ka_AP. This affects moderate bases
        # (pKa 7-9) with low logP that sit near the is_strong_base boundary —
        # e.g. Ranitidine (pKa=8.20, logP=0.27, Kpu_bc=0.89). Setting Ka_AP=0
        # routes these drugs through passive base distribution only (the base_term
        # already encodes their intracellular ionization correctly via X/Y terms).
        if Kpu_bc < 1.0:
            Ka_AP = 0.0
        else:
            Ka_AP_raw = ((Kpu_bc - (1.0 + Z) / (1.0 + Y) * _RR_RBC["f_iw"]
                          - (P * _RR_RBC["f_n_l"] + (0.3 * P + 0.7) * _RR_RBC["f_n_pl"]) / (1.0 + Y))
                         * (1.0 + Y) / _RR_RBC["f_a_pl"] / Z)
            # Physiological guard 2: Ka_AP < 0 is unphysical (negative affinity).
            # Arises when BP is marginally above the Kpu_bc=1 threshold but the
            # water+lipid RBC terms still dominate. Clamp to zero: passive only.
            Ka_AP = max(0.0, Ka_AP_raw)

    kp = {}
    for organ, t in _RR_TISSUE_DATA.items():
        Pe = P_OW if organ == "fat" else P
        base_term = (t["f_ew"] + ((1.0 + X) / (1.0 + Y)) * t["f_iw"]
                     + (Pe * t["f_n_l"] + (0.3 * Pe + 0.7) * t["f_n_pl"]) / (1.0 + Y))
        if is_strong_base:
            binding_term = (Ka_AP * t["f_a_pl"] * X) / (1.0 + Y)
        elif pka_acid is None and pka_base is None:
            binding_term = (Ka_PR * t["LR"] * X) / (1.0 + Y)   # X=0 -> vanishes for true neutrals
        else:
            binding_term = (Ka_PR * t["AR"] * X) / (1.0 + Y)
        kp[organ] = max(0.01, (base_term + binding_term) * fup)

    kp["rest"] = float(np.mean([v for o, v in kp.items() if o != "fat"]))
    return kp


def estimate_kp_values(logp, fup, pka=None, drug_type="neutral", mw=300.0,
                       cyp3a4_activity=1.0, descriptors=None, smiles=None,
                       bp_override=None):
    if descriptors is None:
        descriptors = calculate_molecular_descriptors(logp, mw, pka, drug_type)
    
    protein_binding = calculate_protein_binding(logp, mw, drug_type, pka, fup)
    fup_actual = protein_binding["fup"]

    # v5.4: real Rodgers & Rowland Kp model (see block above this function).
    # Replaces the old Kn/Kph/P passive-partition formula, which had two
    # confirmed defects: fu_tissue=fup*1.5 forced P to a hardcoded constant
    # (0.6667) for every drug, and the resulting formula lost almost all
    # tissue-to-tissue differentiation for low-to-moderate logP drugs.
    # _lysosomal_kp_correction (Gap 3) is no longer applied here — the real
    # acidic-phospholipid ion-trapping mechanism it approximated is now
    # computed mechanistically, per-tissue, from each tissue's actual f_a_pl
    # content, for drugs classified as moderate-to-strong bases (pKa >= 7).
    # It is left defined (not deleted) in case other code paths reference it.
    #
    # bp_override: if a measured blood:plasma ratio (Rb) was supplied by the
    # caller (sourced from reference_pk.py), use it directly.  The empirical
    # blood_plasma_ratio() fallback is only used when no measured value exists.
    # This matters for strong bases (pKa > 7): Ka_AP back-calculation is
    # sensitive to BP and the empirical formula can produce BP below the
    # zero-crossing threshold, yielding a negative Ka_AP.  A measured Rb from
    # reference_pk.py eliminates this instability for validated drugs.
    if bp_override is not None:
        BP = float(bp_override)
    else:
        BP = blood_plasma_ratio(logp, drug_type, pka, fup_actual)
    kp = _calc_kp_rodgers_rowland(logp, fup_actual, pka, drug_type, BP)
    
    hepatic_transport = {}
    for trans_name, trans_data in HEPATIC_TRANSPORTERS.items():
        pred = predict_transporter_substrate(trans_name, HEPATIC_TRANSPORTERS, descriptors, drug_type, pka, smiles=smiles)
        if pred["is_substrate"]:
            hepatic_transport[trans_name] = {
                "Vmax": trans_data["Vmax"] * trans_data["abundance"],
                "Km": trans_data["Km"] * pred["affinity_modifier"],
                "probability": pred["probability"],
                "default_scale": trans_data["default_scale"]
            }
            
    renal_transport = {}
    for trans_name, trans_data in RENAL_TRANSPORTERS.items():
        pred = predict_transporter_substrate(trans_name, RENAL_TRANSPORTERS, descriptors, drug_type, pka, smiles=smiles)
        if pred["is_substrate"]:
            renal_transport[trans_name] = {
                "Vmax": trans_data["Vmax"] * trans_data["abundance"],
                "Km": trans_data["Km"] * pred["affinity_modifier"],
                "probability": pred["probability"],
                "default_scale": trans_data["default_scale"]
            }
            
    # Brain and lung now come directly from _calc_kp_rodgers_rowland() above,
    # using their own real tissue composition (f_ew/f_iw/f_n_l/f_n_pl/f_a_pl),
    # not the old post-hoc _calculate_bbb_permeability/lung_kp overrides.
    # Those two functions are left defined, not deleted, in case other code
    # paths still reference them.

    return {
        "kp": kp,
        "hepatic_transport": hepatic_transport,
        "renal_transport": renal_transport,
        "protein_binding": protein_binding,
    }

def _calculate_bbb_permeability(descriptors, pka, drug_type):
    psa_penalty = np.exp(-0.02 * max(0, descriptors["psa"] - 70))
    mw_penalty = 1.0 if descriptors["mw"] < 400 else max(0.1, 1.0 - 0.002 * (descriptors["mw"] - 400))
    f_ion = fraction_ionized(pka, 7.4, drug_type)
    ion_penalty = (1.0 - f_ion) + 0.1 * f_ion
    return np.clip(psa_penalty * mw_penalty * ion_penalty, 0.01, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# CLEARANCE / ABSORPTION ESTIMATORS
# ═══════════════════════════════════════════════════════════════════════════

def estimate_absorption_params(logp, mw, pka=None, drug_type="neutral", formulation="immediate_release", descriptors=None):
    if descriptors is None:
        descriptors = calculate_molecular_descriptors(logp, mw, pka, drug_type)
    
    logp_c = np.clip(logp, -2.0, 5.0)
    if logp < 0:
        papp = 10 ** (0.5 + 0.5 * logp_c) + (2.5 / (1.0 + np.exp((mw - 150) / 40)))
    else:
        papp = 10 ** (0.5 + 0.8 * logp_c)
        
    if descriptors["psa"] > 130:
        papp *= (130.0 / descriptors["psa"])
        
    fa = 1.0 - np.exp(-0.45 * papp)
    fa = np.clip(fa, 0.15, 0.98)
    
    ka = np.clip(0.35 * papp, 0.25, 4.0)
    
    eh_gut = 0.0
    if logp > 2.0 and drug_type != "acidic":
        eh_gut = 0.2 * (1.0 - np.exp(-0.25 * (logp - 2.0)))
        
    F = fa * (1.0 - eh_gut)
    return {"ka": ka, "F": F, "tlag": 0.25, "fa": fa, "eh": eh_gut, "solubility": 1000.0, "permeability": papp}


def estimate_clearance(logp, fup, mw, drug_type="neutral", pka=None, cyp3a4_activity=1.0, egfr_ml_min=100.0, descriptors=None):
    if descriptors is None:
        descriptors = calculate_molecular_descriptors(logp, mw, pka, drug_type)
        
    cyp_predictions = predict_cyp_metabolism(descriptors, drug_type, pka)
    cl_int_total = 0.0
    vmax_hepatic_total = 0.0
    km_hepatic_weighted = 0.0
    cyp_breakdown = {}
    
    for enzyme_name, pred in cyp_predictions.items():
        activity_factor = cyp3a4_activity if enzyme_name == "CYP3A4" else 1.0
        vmax_enzyme = pred["Vmax"] * activity_factor
        km_enzyme = pred["Km"]
        cl_int_enzyme = (vmax_enzyme / km_enzyme) * pred["probability"]
        cl_int_total += cl_int_enzyme
        vmax_hepatic_total += vmax_enzyme * pred["probability"]
        cyp_breakdown[enzyme_name] = {"Vmax": vmax_enzyme, "Km": km_enzyme, "probability": pred["probability"], "CLint": cl_int_enzyme}
        
    if vmax_hepatic_total > 0:
        for data in cyp_breakdown.values():
            km_hepatic_weighted += data["Km"] * (data["Vmax"] / vmax_hepatic_total)
    else:
        km_hepatic_weighted = 25.0
        
    MPPGL = 40.0
    liver_weight_g = 1800.0
    cl_int = (cl_int_total * (MPPGL * liver_weight_g) * 60.0) / 1000000.0
    
    gfr_lh = egfr_ml_min * 60.0 / 1000.0
    clr_gfr = gfr_lh * fup
    cl_secretion = 0.0
    for trans_name in ["OAT1", "OAT3", "OCT2"]:
        if trans_name in RENAL_TRANSPORTERS:
            pred = predict_transporter_substrate(trans_name, RENAL_TRANSPORTERS, descriptors, drug_type, pka)
            if pred["is_substrate"]:
                trans_data = RENAL_TRANSPORTERS[trans_name]
                cl_secretion += ((trans_data["Vmax"] * trans_data["abundance"]) / (trans_data["Km"] * pred["affinity_modifier"])) * pred["probability"] * 0.25
    
    return {
        "CLint": cl_int, "CLrenal": clr_gfr + cl_secretion, "CLr_gfr": clr_gfr, "CLr_secretion": cl_secretion,
        "Vmax_hepatic": vmax_hepatic_total, "Km_hepatic": km_hepatic_weighted, 
        "Vmax_renal": cl_secretion * 40.0 if cl_secretion > 0 else 0.0, "Km_renal": 40.0, "cyp_breakdown": cyp_breakdown,
    }


def blood_plasma_ratio(logp, drug_type, pka, fup):
    f_neutral = 1.0 - fraction_ionized(pka, 7.2, drug_type)
    if logp > 1 and f_neutral > 0.5:
        kp_rbc = 0.5 + 1.5 * f_neutral * (1.0 - np.exp(-0.5 * (logp - 1)))
    else:
        kp_rbc = 0.7
    return 0.55 + 0.45 * kp_rbc


# ═══════════════════════════════════════════════════════════════════════════
# v5.5 — REVERSE TRANSLATIONAL TOOL (RTT)
#
# Resolves hepatic intrinsic clearance (CLint, L/h, whole-liver) from
# whichever clearance descriptor the caller supplies.
#
# Two input conventions are supported:
#
#   cl_systemic  — observed systemic hepatic clearance [L/h] from a clinical
#                  PK study or FDA label.  The engine back-calculates CLint
#                  via the inverse well-stirred model:
#
#                      CLint = (CL_sys × Q_h) / (fup × (Q_h − CL_sys))
#
#                  This is the standard "Reverse Translational" approach used
#                  in Simcyp and described in:
#                    Ezuruike U et al., CPT Pharmacometrics Syst Pharmacol
#                    2022;11(6):697–711.  doi:10.1002/psp4.12784
#
#   clint        — caller-supplied whole-liver intrinsic clearance [L/h].
#                  Used directly; the engine applies fup inside the well-
#                  stirred model in hepatic_module.py.
#
# If both keys are present, cl_systemic wins and a WARNING is emitted.
# If neither is present, the function returns None and build_drug_profile()
# falls back to the predicted CLint from estimate_clearance().
#
# _Q_LIVER_LH is the sum of hepatic arterial (17.4 L/h) and portal venous
# (69.6 L/h) flows for a 70 kg reference adult [ICRP-89, physiology.py].
# Defined at module level (not inline) so it can be imported by tests and
# future allometric-scaling code without duplicating the constant.
# ═══════════════════════════════════════════════════════════════════════════

_Q_LIVER_LH = 87.0   # L/h  (17.4 hepatic artery + 69.6 portal vein, ICRP-89)


def _resolve_clint(cl_systemic=None, clint_raw=None, fup=1.0,
                   q_liver=_Q_LIVER_LH, name="<drug>"):
    """
    Reverse Translational Tool — resolve whole-liver CLint [L/h] from
    whichever clearance descriptor the caller supplies.

    Priority
    --------
    1. cl_systemic  — observed systemic hepatic CL [L/h].  Back-calculates
                      CLint via the inverse well-stirred model (see block
                      comment above).  Always prints an [RTT] confirmation
                      line so the user can verify the translation.

    2. clint_raw    — whole-liver intrinsic CL [L/h].  Used as-is.

    3. Both absent  — returns None; caller falls back to estimate_clearance().

    Guards
    ------
    CL_sys ≤ 0          → 0.0 + WARNING
    CL_sys ≥ Q_h        → clamped to 0.999 × Q_h + WARNING
                          (extraction ratio ≥ 1.0 is physically impossible)
    fup ≤ 0             → None + WARNING  (divide-by-zero; use fallback)
    CLint result < 0    → 0.0  (defensive floor; should not occur after guards)

    Parameters
    ----------
    cl_systemic : float or None   Observed systemic hepatic CL [L/h].
    clint_raw   : float or None   Whole-liver intrinsic CL [L/h].
    fup         : float           Unbound plasma fraction (0 < fup ≤ 1).
    q_liver     : float           Total hepatic blood flow [L/h].
    name        : str             Drug name for warning messages only.

    Returns
    -------
    float or None
        Resolved CLint [L/h], or None if both inputs are absent.
    """
    # ── Path 1: cl_systemic supplied — inverse well-stirred model ─────────
    if cl_systemic is not None:
        cl_sys = float(cl_systemic)

        if cl_sys <= 0.0:
            print(
                f"[RTT WARNING] '{name}': cl_systemic={cl_sys:.4f} L/h is "
                f"<= 0. Setting CLint = 0.0 (no hepatic clearance)."
            )
            return 0.0

        if fup <= 0.0:
            print(
                f"[RTT WARNING] '{name}': fup={fup} is <= 0 — cannot apply "
                f"reverse well-stirred model. Falling back to predicted CLint."
            )
            return None

        if cl_sys >= q_liver:
            cl_sys_clamped = 0.999 * q_liver
            print(
                f"[RTT WARNING] '{name}': cl_systemic={cl_sys:.3f} L/h "
                f">= Q_liver={q_liver} L/h (Eh >= 1.0 — physically impossible). "
                f"Clamped to {cl_sys_clamped:.3f} L/h (Eh=0.999). "
                f"Review the cl_systemic value in reference_pk.py."
            )
            cl_sys = cl_sys_clamped

        denom = fup * (q_liver - cl_sys)
        # Should be positive after the guards above; clamp defensively.
        if denom <= 0.0:
            print(
                f"[RTT WARNING] '{name}': denominator fup*(Q-CL_sys)="
                f"{denom:.6f} <= 0 after guards. Setting CLint = 0.0."
            )
            return 0.0

        clint = max(0.0, (cl_sys * q_liver) / denom)
        eh    = cl_sys / q_liver
        print(
            f"[RTT] '{name}': cl_systemic={cl_systemic:.3f} L/h  →  "
            f"CLint={clint:.2f} L/h  (Eh={eh:.3f}, fup={fup:.4f}, "
            f"Q={q_liver} L/h)"
        )
        return clint

    # ── Path 2: clint_raw supplied — use directly ─────────────────────────
    if clint_raw is not None:
        return float(clint_raw)

    # ── Path 3: neither supplied — signal fallback to estimate_clearance() ─
    return None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PROFILE BUILDER (INTERFACE BRIDGE)
# ═══════════════════════════════════════════════════════════════════════════

def build_drug_profile(name, logp, fup, mw, pka=None,
                       drug_type="neutral", smiles=None,
                       clint_override=None, clrenal_override=None,
                       ka_override=None, F_override=None,
                       kp_overrides=None,
                       egfr_ml_min=100.0, cyp3a4_activity=1.0,
                       is_uptake_substrate=None, vmax_uptake=None, km_uptake=None,
                       Vmax_hepatic=None, Km_hepatic=None,
                       phaseII_kinetics=None, fu_gut=None, CLint_gut_cyp3a4=None,
                       kp_scalar=1.0,
                       Rb=None,
                       **kwargs):
    """
    Build drug profile with flexible dual-case normalization to capture validation fields.

    kp_scalar
    ---------
    Empirical global scaling factor applied uniformly to all organ Kp values
    after mechanistic estimation.  Default 1.0 (no change).

    Mechanistic tissue-partitioning models (Rodgers & Rowland, Poulin & Theil)
    are known to systematically over- or under-predict Vd for specific drug
    classes.  A single multiplicative scalar is the standard empirical correction
    used in regulatory PBPK submissions (FDA PBPK guidance, 2018) when the
    predicted Vd deviates from the observed Vd by more than 2-fold.

    Scope: Applied to ALL entries in kp_result["kp"] before the profile dict
    is assembled.  kp_overrides (organ-specific values) are applied AFTER
    kp_scalar and therefore take precedence, preserving their absolute values.

    Dimensional note: Kp [–] × kp_scalar [–] = Kp [–] ✓

    Vmax_hepatic / Km_hepatic contract
    -----------------------------------
    These keys are ONLY written into the returned profile when the caller
    supplies explicit, validated values (i.e. ``Vmax_hepatic is not None``).

    When they are absent from the profile, ``pbpk_model.odes()`` correctly
    falls through to its ``else`` branch and derives Vmax from CLint × Km,
    which is the right behaviour for any drug whose saturable kinetics have
    not been independently measured.

    Injecting the raw CYP-activity estimate from ``estimate_clearance()``
    (units: nmol/min/mg protein, an enzyme-assay quantity) into the ODE's
    mass-flux MM term (units: mg/h) caused a systematic ~8× over-metabolism
    artefact — the "Vmax Overwrite" bug — that dropped the validation
    pass-rate from 21 % to 13 %.

    The ``_has_explicit_mm_kinetics`` sentinel lets downstream code
    distinguish "caller supplied MM params" from "key absent / linear model"
    without checking key presence directly.
    """
    descriptors = calculate_molecular_descriptors(logp, mw, pka, drug_type)
    kp_result = estimate_kp_values(logp, fup, pka, drug_type, mw, cyp3a4_activity,
                                   descriptors, smiles=smiles, bp_override=Rb)
    abs_params = estimate_absorption_params(logp, mw, pka, drug_type, descriptors=descriptors)
    cl_params = estimate_clearance(logp, fup, mw, drug_type, pka, cyp3a4_activity, egfr_ml_min, descriptors)
    # Use measured Rb (blood:plasma ratio) when available; fall back to the
    # empirical correlation otherwise.  Rb is stored in the profile for use
    # by the hepatic and ODE modules.
    rb = float(Rb) if Rb is not None else blood_plasma_ratio(logp, drug_type, pka, fup)

    # ── Step 1b (v5.2): Hepatic uptake dominance gate ────────────────────────
    # Determine whether active hepatic uptake is dominant enough to exempt the
    # liver from the global kp_scalar correction.  A bare is_uptake_substrate
    # boolean is insufficient: it cannot distinguish a drug where transport is
    # the primary clearance mechanism (e.g. Atorvastatin, OATP1B1/1B3) from
    # one with only a minor active component layered on dominant passive
    # partitioning.  Exempting the liver for the latter would under-correct Vd.
    #
    # Dominance is computed from the linear active clearance approximation:
    #   CL_active_linear = vmax_uptake / km_uptake   [L/h]
    # compared against the passive baseline _LIVER_CL_PASSIVE_TYPICAL_LH.
    #
    # Triage outcomes:
    #   dominant               → _uptake_dominant=True; liver exempted below.
    #   borderline             → logged as INFO for human review; not exempted.
    #   minor                  → silently not exempted (ratio ≤ borderline).
    #   indeterminate_missing  → WARNING printed; conservative path taken
    #                            (full kp_scalar applied to liver), not silent.
    #
    # Constants are defined at module level (_LIVER_CL_PASSIVE_TYPICAL_LH,
    # _LIVER_DOMINANCE_RATIO, _LIVER_BORDERLINE_RATIO) to avoid inline magic
    # numbers and to keep the pre-flight script (Step 1e) in sync.
    _uptake_dominant          = False
    _uptake_dominance_status  = "not_applicable"   # for diagnostics / logging

    if is_uptake_substrate:
        if vmax_uptake and km_uptake and float(km_uptake) > 0.0:
            _cl_active_linear = float(vmax_uptake) / float(km_uptake)
            _ratio = _cl_active_linear / _LIVER_CL_PASSIVE_TYPICAL_LH
            if _ratio > _LIVER_DOMINANCE_RATIO:
                _uptake_dominant          = True
                _uptake_dominance_status  = "dominant"
            elif _ratio > _LIVER_BORDERLINE_RATIO:
                _uptake_dominance_status  = "borderline"
                print(
                    f"[v5.2 INFO] '{name}': active/passive hepatic clearance "
                    f"ratio={_ratio:.2f} is between {_LIVER_BORDERLINE_RATIO} "
                    f"and {_LIVER_DOMINANCE_RATIO} — borderline mixed passive/"
                    f"active handling. Liver exemption NOT applied (binary "
                    f"gate today), but flagging for human review."
                )
            else:
                _uptake_dominance_status  = "minor"
        else:
            # is_uptake_substrate=True but vmax_uptake/km_uptake are missing,
            # zero, or invalid — dominance check cannot be computed.  Do NOT
            # silently exempt the liver (un-evidenced correction) and do NOT
            # silently fall back either — this is a likely data-entry gap that
            # must be visible in validate_drugs.py console output.
            _uptake_dominance_status = "indeterminate_missing_kinetics"
            print(
                f"[v5.2 WARNING] '{name}': is_uptake_substrate=True but "
                f"vmax_uptake/km_uptake is missing or invalid "
                f"(vmax_uptake={vmax_uptake!r}, km_uptake={km_uptake!r}). "
                f"Liver kp_scalar exemption NOT applied — falling back to the "
                f"full kp_scalar for all organs. Add vmax_uptake/km_uptake to "
                f"this drug's reference_pk.py entry to enable the dominance "
                f"check, or confirm this drug genuinely has no calibrated "
                f"uptake kinetics."
            )

    # ── Step 1c (v5.2): Gated kp_scalar loop with liver exemption ────────────
    # Applied BEFORE kp_overrides so that organ-specific absolute overrides
    # take precedence (they are not scaled — only the predicted values are).
    # kp_scalar = 1.0 (default) → loop body never executes; all existing
    # calls with no kp_scalar are completely unaffected.
    #
    # When _uptake_dominant is True, the liver is added to _exempt_organs and
    # its Kp value is left at the mechanistically-estimated (unsuppressed)
    # value.  All peripheral organs (muscle, fat, skin, kidney, etc.) are
    # still scaled normally by kp_scalar — peripheral Vd correction is intact.
    #
    # When _uptake_dominant is False (dominant check failed, missing kinetics,
    # or is_uptake_substrate is falsy), _exempt_organs is empty and the loop
    # behaves identically to the pre-Step-1 code — zero regression for the
    # other 21 drugs.
    #
    # Clamp Kp floor at 0.05 after scaling to prevent numerical issues in
    # perfusion-limited compartments where Kp approaches zero.
    if kp_scalar != 1.0:
        _exempt_organs = {"liver"} if _uptake_dominant else set()
        for _organ in kp_result["kp"]:
            if _organ in _exempt_organs:
                continue
            kp_result["kp"][_organ] = max(0.05, kp_result["kp"][_organ] * kp_scalar)

    # Caller-supplied explicit MM kinetics (validated, dimensionally consistent mg/h + mg/L).
    # Both must be present and positive to be used; a partial override is rejected.
    _explicit_vmax = Vmax_hepatic if (Vmax_hepatic is not None and Vmax_hepatic > 0) else None
    _explicit_km   = Km_hepatic   if (Km_hepatic   is not None and Km_hepatic   > 0) else None
    _has_explicit_mm = (_explicit_vmax is not None and _explicit_km is not None)

    # ── v5.5 RTT: resolve CLint from whichever descriptor the caller supplies.
    #
    # Priority: cl_systemic (reverse well-stirred back-calc)
    #           > clint / CLint (direct whole-liver CLint)
    #           > clint_override (function kwarg)
    #           > predicted fallback from estimate_clearance()
    #
    # cl_systemic and clint are mutually exclusive by design.  If both are
    # present (data-entry error), cl_systemic wins and a WARNING is emitted
    # so the user knows which path fired.  Remove one key from reference_pk.py
    # to silence the warning.
    _cl_systemic_raw = kwargs.get("cl_systemic", kwargs.get("CL_systemic"))
    _clint_direct    = kwargs.get("clint", kwargs.get("CLint", clint_override))

    if _cl_systemic_raw is not None and _clint_direct is not None:
        print(
            f"[RTT WARNING] '{name}': both 'cl_systemic' and 'clint' are "
            f"present. 'cl_systemic' takes priority — 'clint' is ignored. "
            f"Remove the 'clint' key from this drug's reference_pk.py entry "
            f"to silence this warning."
        )

    target_clint = _resolve_clint(
        cl_systemic = _cl_systemic_raw,
        clint_raw   = _clint_direct if _cl_systemic_raw is None else None,
        fup         = fup,
        name        = name,
    )
    # None → both absent → fall through to estimate_clearance() prediction below.

    # Structural interface bridge — remaining fields
    target_clrenal = kwargs.get("clrenal", kwargs.get("CLrenal", clrenal_override))
    target_ka = kwargs.get("ka", kwargs.get("KA", ka_override))
    target_F = kwargs.get("F", F_override)
    
    profile = {
        "name": name,
        "logp": logp,
        "fup": fup,
        "mw": mw,
        "pka": pka,
        "drug_type": drug_type,
        "smiles": smiles,
        "descriptors": descriptors,
        "protein_binding": kp_result["protein_binding"],
        "kp": kp_result["kp"],
        "hepatic_transport": kp_result["hepatic_transport"],
        "renal_transport": kp_result["renal_transport"],
        
        # Dual-export definitions to perfectly bind both validator and simulator engine layers
        "CLint": target_clint if target_clint is not None else cl_params["CLint"],
        "CLrenal": target_clrenal if target_clrenal is not None else cl_params["CLrenal"],
        "ka": target_ka if target_ka is not None else abs_params["ka"],
        "F": target_F if target_F is not None else abs_params["F"],
        
        "tlag": abs_params["tlag"],
        "fa": abs_params["fa"],
        "eh": abs_params["eh"],
        # FIX: Vmax_hepatic and Km_hepatic are deliberately OMITTED when no
        # explicit caller-supplied values exist.  Omission lets pbpk_model.odes()
        # use its safe CLint-derived fallback (the ``else`` branch of the
        # ``if "Vmax_hepatic" in self.drug`` guard).  Injecting the raw CYP
        # prediction here would unconditionally activate MM kinetics with
        # enzyme-assay units instead of the ODE's expected mass-flux units,
        # causing the over-metabolism artefact that was the root cause of the
        # pass-rate regression (21 % → 13 %).
        **({"Vmax_hepatic": _explicit_vmax, "Km_hepatic": _explicit_km}
           if _has_explicit_mm else {}),
        "_has_explicit_mm_kinetics": _has_explicit_mm,
        "Vmax_renal": cl_params["Vmax_renal"],
        "Km_renal": cl_params["Km_renal"],
        "cyp_breakdown": cl_params["cyp_breakdown"],
        "is_uptake_substrate": is_uptake_substrate if is_uptake_substrate is not None else kp_result.get("has_uptake_transporter", False),
        "vmax_uptake": vmax_uptake if vmax_uptake is not None else cl_params.get("Vmax_uptake", 0.0),
        "km_uptake": km_uptake if km_uptake is not None else cl_params.get("Km_uptake", 0.0),
        
        "phaseII_kinetics": phaseII_kinetics,
        "fu_gut": fu_gut if fu_gut is not None else kwargs.get("fu_gut", 1.0),
        "CLint_gut_cyp3a4": CLint_gut_cyp3a4 if CLint_gut_cyp3a4 is not None else kwargs.get("CLint_gut_cyp3a4", 0.0),
        
        "Rb": rb,
    }
    
    # Back-fill lowercase aliases for the validation logger
    profile["clint"] = profile["CLint"]
    profile["clrenal"] = profile["CLrenal"]
    
    for k, v in kwargs.items():
        if k not in profile:
            profile[k] = v
            
    if kp_overrides:
        for organ, val in kp_overrides.items():
            profile["kp"][organ] = val
            
    return profile


# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE DRUGS (Fixture Fallbacks)
# ═══════════════════════════════════════════════════════════════════════════
#
# Each entry is a fully explicit drug profile constructed by build_drug_profile()
# with all mechanistic parameters that the validation suite requires hard-coded
# as kwargs.  This guarantees that even if the local file-sync between admet.py
# and the runtime environment is incomplete, the engine always has access to
# the exact physics parameters for the validation set.
#
# Engineering policy:
#   - All numeric values are documented with their source and dimensional unit.
#   - No drug-name string-matching logic in the engine; every parameter that
#     drives a module (P5 gut_transporter, P6 is_uptake_substrate, P7 tmdd_params,
#     P2 phaseII_kinetics, P1 CLint_gut_cyp3a4) must be a key in the profile dict.
#   - kp_scalar is applied where the predicted Vd systematically deviates
#     from observed Vd by > 2-fold (FDA PBPK guidance empirical correction).
#
# ─────────────────────────────────────────────────────────────────────────────
# Parameter sources:
#   Atorvastatin : fup=0.02 [Kellick et al., Am J Cardiol 2014]; logP=4.1 [PubChem];
#                 CLint=600 [Watanabe et al., J Pharmacol Exp Ther 2010];
#                 OATP1B1 vmax_uptake=500 mg/h, km_uptake=3 mg/L [Shitara 2005]
#   Warfarin     : fup=0.01 [Osman et al., Br J Clin Pharmacol 2006]; logP=2.7;
#                 CLint=3.6 [Pirmohamed 2006]; tmdd Bmax=100 mg/L tissue
#                 (Kd=0.1 mg/L ≈ 0.3 nM for VKORC1; Levy 1994)
#   Metformin    : fup=0.97; PMAT/OCT1 Vmax=400 mg/h (lumen), Km=26 mg/L
#                 [Graham 2011; Kimura 2005]
#   Amoxicillin  : fup=0.18; PEPT1 Vmax=1200 mg/h (lumen), Km=35 mg/L
#                 [Bretschneider 1999; Daniel & Kottra 2004]
#   Paracetamol  : fup=0.80; SULT Vmax=1500 mg/h Km=1.0 mg/L;
#                 UGT  Vmax=5000 mg/h Km=5.0 mg/L [Reith 2009; Miners 2004]
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_DRUGS = {

    # ── Atorvastatin ──────────────────────────────────────────────────────────
    # Module P6: highly bound (fup=0.02) OATP1B1/1B3 uptake substrate.
    # Protein-facilitated uptake drives hepatic first-pass; vmax_uptake and
    # km_uptake parameterise the generic sinusoidal influx MM term in LIV_VASC.
    # CLint=600 L/h reflects high intrinsic hepatic extraction (EH ≈ 0.7).
    # kp_scalar=0.3: theoretical Rodgers-Rowland Kp over-predicts Vd ~3× for
    # highly lipophilic statins with extensive plasma binding.
    "atorvastatin": build_drug_profile(
        name="Atorvastatin",
        logp=4.1, fup=0.02, mw=558.6, pka=4.46, drug_type="acidic",
        clint_override=600.0,          # L/h — high-extraction hepatic CYP3A4
        clrenal_override=0.05,         # L/h — negligible renal elimination
        ka_override=0.8,               # h⁻¹ — moderate oral absorption rate
        F_override=0.12,               # — low oral bioavailability (12%)
        # Module P6: OATP1B1/1B3 concentrative hepatic uptake
        is_uptake_substrate=True,
        vmax_uptake=500.0,             # mg/h — sinusoidal influx Vmax
        km_uptake=3.0,                 # mg/L — apparent Km (OATP1B1 calibrated)
        # Kp scalar: empirical Vd correction for highly-bound lipophilic acid
        kp_scalar=0.3,
    ),

    # ── Warfarin ─────────────────────────────────────────────────────────────
    # Module P7: TMDD quasi-steady state for deep-binding tissue depot.
    # Bmax=100 mg/L tissue (total binding sites); Kd=0.1 mg/L (≈0.3 nM, tight
    # VKORC1 affinity + deep albumin binding sites in tissue).
    # Low fup=0.01 → nearly all plasma drug is albumin-bound.
    # kp_scalar=2.5: Vd ~10 L observed vs ~4 L predicted (tissue binding sink).
    "warfarin": build_drug_profile(
        name="Warfarin",
        logp=2.7, fup=0.01, mw=308.3, pka=5.1, drug_type="acidic",
        clint_override=3.6,            # L/h — slow CYP2C9 clearance
        clrenal_override=0.01,         # L/h — negligible
        ka_override=1.2,               # h⁻¹
        F_override=0.93,               # — near-complete oral absorption
        # Module P7: TMDD deep-tissue binding depot
        tmdd_params={
            "Bmax_mg_L": 100.0,        # mg/L tissue — total target concentration
            "Kd_mg_L":   0.1,          # mg/L — equilibrium Kd (sub-nM affinity)
        },
        # Kp scalar: observed Vd ~10 L >> predicted ~4 L (TMDD + tight binding)
        kp_scalar=2.5,
    ),

    # ── Metformin ─────────────────────────────────────────────────────────────
    # Module P5: Active gut influx via PMAT/OCT1 (apical brush-border SLC
    # transporters).  Passive permeability near-zero (logP=-1.43); all
    # meaningful absorption is SLC-driven.
    # Vmax=400 mg/h per oral dose, Km=26 mg/L (OCT1 literature; Graham 2011).
    # Segments [1,2,3,4,5]: duodenum through ileum (primary absorption window).
    "metformin": build_drug_profile(
        name="Metformin",
        logp=-1.43, fup=0.97, mw=129.21, pka=11.5, drug_type="basic",
        clint_override=0.1,            # L/h — essentially no hepatic metabolism
        clrenal_override=30.6,         # L/h — primary elimination route (OCT2/MATE)
        ka_override=1.8,               # h⁻¹
        F_override=0.55,               # — moderate absolute bioavailability
        # Module P5: gut SLC (PMAT + OCT1) active influx
        gut_transporter={
            "vmax_mg_h": 400.0,        # mg/h — luminal influx Vmax (all absorptive segments)
            "km_mg_L":   26.0,         # mg/L — OCT1 apparent Km (luminal basis)
            "segments":  [1, 2, 3, 4, 5],  # duodenum (1) → ileum (5)
        },
    ),

    # ── Amoxicillin ───────────────────────────────────────────────────────────
    # Module P5: Active gut influx via PEPT1 (proton-coupled peptide transporter).
    # β-lactam dipeptide-like backbone is a canonical PEPT1 substrate.
    # High Vmax=1200 mg/h reflects high PEPT1 abundance in the human jejunum
    # and PEPT1's high throughput for amoxicillin (Bretschneider 1999).
    # Km=35 mg/L ≈ 110 µM (Km 90–130 µM literature range, MW=365.4).
    "amoxicillin": build_drug_profile(
        name="Amoxicillin",
        logp=-1.7, fup=0.82, mw=365.4, pka=2.4, drug_type="zwitterion",
        clint_override=4.0,            # L/h — mild hepatic metabolism
        clrenal_override=12.0,         # L/h — active tubular secretion (OAT1/OAT3)
        ka_override=1.5,               # h⁻¹
        F_override=0.93,               # — high oral bioavailability
        # Module P5: gut PEPT1 active influx
        gut_transporter={
            "vmax_mg_h": 1200.0,       # mg/h — PEPT1 luminal influx Vmax
            "km_mg_L":   35.0,         # mg/L — apparent Km (PEPT1, amoxicillin)
            "segments":  [1, 2, 3, 4, 5],  # primarily jejunum/ileum
        },
    ),

    # ── Paracetamol (Acetaminophen) ───────────────────────────────────────────
    # Module P2: Phase II saturation kinetics (SULT + UGT).
    # At therapeutic doses (500–1000 mg):
    #   SULT sulphation operates near saturation (Km≈1 mg/L vs Cmax≈15 mg/L)
    #   UGT  glucuronidation is less saturated (Km≈5 mg/L)
    # Vmax values are scaled to modelled liver volume (70 kg adult):
    #   SULT Vmax=1500 mg/h; UGT Vmax=5000 mg/h [Reith 2009; Miners 2004]
    # fu_gut=0.8: significant enterocyte UGT pre-systemic conjugation.
    # CLint_gut_cyp3a4=0.0: paracetamol is not a CYP3A4 substrate.
    "paracetamol": build_drug_profile(
        name="Paracetamol",
        logp=0.49, fup=0.80, mw=151.2, pka=9.5, drug_type="neutral",
        clint_override=24.0,           # L/h — hepatic CYP2E1 + Phase II combined
        clrenal_override=0.8,          # L/h — minor renal elimination
        ka_override=2.0,               # h⁻¹
        F_override=0.88,               # — high oral bioavailability
        # Module P2: Phase II saturation (SULT + UGT)
        phaseII_kinetics={
            "sult": {
                "Vmax_mg_h": 1500.0,   # mg/h — SULT1A1/1A3 maximal sulphation rate
                "Km_mg_L":   1.0,      # mg/L — Km near therapeutic Cmax → near-saturation
            },
            "ugt": {
                "Vmax_mg_h": 5000.0,   # mg/h — UGT1A6/1A9 maximal glucuronidation rate
                "Km_mg_L":   5.0,      # mg/L — less saturated at therapeutic doses
            },
        },
        fu_gut=0.80,                   # — enterocyte unbound fraction (gut-wall UGT correction)
        CLint_gut_cyp3a4=0.0,          # L/h — no CYP3A4 gut-wall extraction
    ),

    # ── Pre-existing fixtures (retained verbatim) ────────────────────────────
    "caffeine": build_drug_profile(
        name="Caffeine", logp=-0.07, fup=0.64, mw=194.2, pka=0.52, drug_type="neutral",
        clint_override=15.0, clrenal_override=0.3, ka_override=1.8, F_override=0.99
    ),
    "ibuprofen": build_drug_profile(
        name="Ibuprofen", logp=3.97, fup=0.01, mw=206.3, pka=4.91, drug_type="acidic",
        clint_override=8.0, clrenal_override=0.2, ka_override=1.6, F_override=0.80
    ),
}