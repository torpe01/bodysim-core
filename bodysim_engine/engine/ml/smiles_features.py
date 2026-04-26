"""
smiles_features.py — Lightweight SMILES parser for the mock predictor.

Extracts molecular descriptors from a SMILES string using only the
Python standard library (no RDKit required). Used by the mock predictor
when ChemProp weights are not available.

Descriptors extracted:
  - Atom counts (C, N, O, S, F, Cl, Br, I, P)
  - Heavy atom count
  - Estimated molecular weight
  - Ring count (SSSR approximation)
  - Aromatic atom fraction
  - H-bond donors / acceptors
  - Key functional group flags
  - Charge state (positive/negative/neutral)
"""

import re
import numpy as np

# Atomic weights for MW estimation
ATOMIC_WEIGHTS = {
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.065,
    'F': 18.998, 'Cl': 35.453, 'Br': 79.904, 'I': 126.904,
    'P': 30.974, 'H': 1.008,  'B': 10.811,  'Si': 28.086,
}

# Average H count per heavy atom (rough estimate)
H_PER_ATOM = {'C': 2.0, 'N': 1.0, 'O': 0.5, 'S': 0.5,
               'F': 0.0, 'Cl': 0.0, 'Br': 0.0, 'I': 0.0, 'P': 1.0}


def parse_smiles(smiles: str) -> dict:
    """
    Extract molecular descriptors from a SMILES string.

    Parameters
    ----------
    smiles : str   valid SMILES string

    Returns
    -------
    dict of molecular descriptors
    """
    s = smiles.strip()

    # ── Atom counting ──────────────────────────────────────────────────────
    # Two-letter elements first (order matters)
    count_Cl = len(re.findall(r'Cl', s))
    count_Br = len(re.findall(r'Br', s))
    count_Si = len(re.findall(r'Si', s))

    # Remove two-letter elements to avoid double counting
    s_clean = re.sub(r'Cl|Br|Si', 'X', s)

    count_C  = len(re.findall(r'[Cc]', s_clean))  # upper = non-aromatic
    count_N  = len(re.findall(r'[Nn]', s_clean))
    count_O  = len(re.findall(r'[Oo]', s_clean))
    count_S  = len(re.findall(r'[Ss]', s_clean))
    count_F  = len(re.findall(r'F', s_clean))
    count_I  = len(re.findall(r'I', s_clean))
    count_P  = len(re.findall(r'P', s_clean))

    heavy = count_C + count_N + count_O + count_S + count_F + \
            count_Cl + count_Br + count_I + count_P + count_Si

    # ── Molecular weight estimate ──────────────────────────────────────────
    mw = (count_C  * 12.011 + count_N  * 14.007 + count_O  * 15.999 +
          count_S  * 32.065 + count_F  * 18.998 + count_Cl * 35.453 +
          count_Br * 79.904 + count_I  * 126.904 + count_P  * 30.974 +
          count_Si * 28.086)
    # Add hydrogens (rough estimate: ~1.2H per heavy atom average)
    mw += heavy * 1.2 * 1.008

    # ── Aromaticity ────────────────────────────────────────────────────────
    aromatic_atoms = len(re.findall(r'[cnos]', smiles))   # lowercase = aromatic
    aromatic_frac  = aromatic_atoms / max(heavy, 1)

    # ── Rings ──────────────────────────────────────────────────────────────
    # Count ring closure digits in SMILES (each digit pair = 1 ring)
    ring_digits = re.findall(r'\d', smiles)
    ring_count  = len(ring_digits) // 2
    # % notation for ring closures
    ring_pct = len(re.findall(r'%\d\d', smiles))
    ring_count += ring_pct

    # ── H-bond donors ──────────────────────────────────────────────────────
    # [OH], [NH], [NH2], explicit H on O or N
    hbd = len(re.findall(r'(?:O|N)H|(?:\[O@@H\]|\[OH\]|\[NH\]|\[NH2\]|\[NH3\])', smiles))
    # Also count -OH and -NH in SMARTS-like patterns
    hbd += len(re.findall(r'(?<![A-Z])(?:OH|NH)', smiles))
    hbd = min(hbd, count_N + count_O)  # can't exceed N+O count

    # ── H-bond acceptors ───────────────────────────────────────────────────
    # All O and N atoms (rough: not all are acceptors but close enough)
    hba = count_N + count_O

    # ── Functional group flags ─────────────────────────────────────────────
    has_carboxyl   = bool(re.search(r'C\(=O\)O|C\(=O\)\[O-\]|C\(O\)=O', smiles))
    has_amine      = bool(re.search(r'N(?!H?\])', smiles) or count_N > 0)
    has_amide      = bool(re.search(r'C\(=O\)N|NC\(=O\)', smiles))
    has_ester      = bool(re.search(r'C\(=O\)OC|OC\(=O\)', smiles))
    has_ketone     = bool(re.search(r'C\(=O\)C|CC\(=O\)', smiles))
    has_alcohol    = bool(re.search(r'(?<![A-Za-z])O(?![\(=])', smiles))
    has_halogen    = (count_F + count_Cl + count_Br + count_I) > 0
    has_sulfonamide= bool(re.search(r'S\(=O\)\(=O\)N', smiles))
    has_phosphate  = count_P > 0
    has_guanidine  = bool(re.search(r'N=C\(N\)N|NC\(=N\)N', smiles))
    has_biguanide  = bool(re.search(r'NC\(=N\)NC\(=N\)N', smiles))  # metformin core

    # ── Charge state ──────────────────────────────────────────────────────
    has_pos_charge = bool(re.search(r'\[.*\+', smiles))
    has_neg_charge = bool(re.search(r'\[.*\-', smiles))
    if has_pos_charge and not has_neg_charge:
        charge_state = "cationic"
    elif has_neg_charge and not has_pos_charge:
        charge_state = "anionic"
    elif has_pos_charge and has_neg_charge:
        charge_state = "zwitterionic"
    else:
        charge_state = "neutral"

    # ── Lipophilicity descriptors ──────────────────────────────────────────
    # Polar surface area proxy (N and O contribution)
    psa_proxy = count_N * 26.0 + count_O * 20.0 + (10.0 if has_carboxyl else 0)

    # Lipophilic carbon fraction
    lipophilic_C = count_C - count_O - count_N   # rough polarity correction
    lipo_frac    = lipophilic_C / max(heavy, 1)

    return {
        # Atom counts
        "count_C":   count_C,
        "count_N":   count_N,
        "count_O":   count_O,
        "count_S":   count_S,
        "count_F":   count_F,
        "count_Cl":  count_Cl,
        "count_Br":  count_Br,
        "count_I":   count_I,
        "count_P":   count_P,
        "heavy_atoms": heavy,

        # Structural
        "mw":            mw,
        "ring_count":    ring_count,
        "aromatic_frac": aromatic_frac,
        "hbd":           hbd,
        "hba":           hba,
        "psa_proxy":     psa_proxy,
        "lipo_frac":     lipo_frac,

        # Functional groups
        "has_carboxyl":    has_carboxyl,
        "has_amine":       has_amine,
        "has_amide":       has_amide,
        "has_ester":       has_ester,
        "has_ketone":      has_ketone,
        "has_alcohol":     has_alcohol,
        "has_halogen":     has_halogen,
        "has_sulfonamide": has_sulfonamide,
        "has_phosphate":   has_phosphate,
        "has_guanidine":   has_guanidine,
        "has_biguanide":   has_biguanide,
        "charge_state":    charge_state,
        "has_pos_charge":  has_pos_charge,
        "has_neg_charge":  has_neg_charge,
    }
