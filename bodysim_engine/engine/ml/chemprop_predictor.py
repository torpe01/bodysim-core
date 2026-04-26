"""
chemprop_predictor.py — AI Brain for BodySim.

Uses ADMET-AI (pre-trained ChemProp v2 ensemble) to predict drug
properties from a SMILES string, then converts them into the format
that build_drug_profile() and PBPKModel need.

ADMET-AI property → PBPK parameter mapping
────────────────────────────────────────────
  logP                    → logp
  PPBR_AZ                 → fup   (via inverse-logit)
  Clearance_Hepatocyte_AZ → clint (via hepatocyte scaling)
  Clearance_Microsome_AZ  → clint (backup)
  HIA_Hou                 → F (oral bioavailability fraction)
  Caco2_Wang              → ka  (absorption rate)
  BBB_Martins             → Kp_brain modifier
  hERG                    → cardiac risk flag
  DILI                    → liver risk flag
  VDss_Lombardo           → Vd check
  CYP3A4_Substrate_*      → CLint CYP flag
  CYP2D6_Substrate_*      → CLint CYP flag

All transforms are documented with source references.
"""

from __future__ import annotations
import warnings
import numpy as np
from pathlib import Path
from typing import Optional


# ── Sigmoid / logit helpers ────────────────────────────────────────────────
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))

def _softplus(x: float) -> float:
    """log(1 + exp(x)) — numerically stable."""
    return float(np.log1p(np.exp(np.clip(x, -30, 30))))


# ── Hepatocyte scaling constants ───────────────────────────────────────────
# Source: Houston JB. Utility of in vitro drug metabolism data in predicting
#         in vivo metabolic clearance. Biochem Pharmacol. 1994.
#
# ADMET-AI Clearance_Hepatocyte_AZ units: log10(µL/min/10^6 cells)
# Scaling: 120×10^6 hepatocytes/g liver × 1690 g liver (ICRP-89 ref man)
#          → 2.028×10^11 hepatocytes total
# µL/min → L/h: ×60 / 10^6
#
# CLint (L/h) = 10^(raw) × 2.028e11 × (1e6 cells/1e6) × 60 / 1e6
#             = 10^(raw) × 12.168
_HEPATOCYTE_SCALE = 12.168   # L/h per µL/min/10^6 cells (full liver)


class ChemPropPredictor:
    """
    AI-powered ADMET predictor using ADMET-AI (pre-trained ChemProp v2).

    Automatically uses ADMET-AI when installed.
    Falls back to calibrated mock predictor otherwise.

    Parameters
    ----------
    verbose : bool  print backend info on load (default True)
    """

    def __init__(self, verbose: bool = True):
        self.verbose  = verbose
        self._model   = None
        self._backend = "unloaded"
        self._mock    = None
        self._load()

    # ── Loading ────────────────────────────────────────────────────────────
    def _load(self):
        try:
            from admet_ai import ADMETModel
            self._model   = ADMETModel()
            self._backend = "admet_ai"
            if self.verbose:
                print("[ChemPropPredictor] ✓ ADMET-AI loaded "
                      "(pre-trained ChemProp v2 ensemble, 41 endpoints)")
        except ImportError:
            self._backend = "mock"
            from .mock_predictor import MockPredictor
            self._mock = MockPredictor()
            if self.verbose:
                print("[ChemPropPredictor] ADMET-AI not found — "
                      "using calibrated mock predictor.")
                print("  → To use real ML: pip install admet-ai")
        except Exception as exc:
            self._backend = "mock"
            from .mock_predictor import MockPredictor
            self._mock = MockPredictor()
            if self.verbose:
                print(f"[ChemPropPredictor] ADMET-AI load error ({exc})"
                      f" — falling back to mock.")

    # ── Main predict method ────────────────────────────────────────────────
    def predict(self, smiles: str) -> dict:
        """
        Predict ADMET properties from a SMILES string.

        Parameters
        ----------
        smiles : str   e.g. "CN(C)C(=N)NC(=N)N"  (metformin)

        Returns
        -------
        dict with keys:
            logp      : float  octanol-water logP
            pka       : float  most ionisable pKa (or None)
            fup       : float  fraction unbound in plasma [0–1]
            clint     : float  hepatic intrinsic clearance [L/h per 70 kg]
            drug_type : str    'basic'|'acidic'|'neutral'|'zwitterion'
            mw        : float  molecular weight [g/mol]
            hbd       : int    H-bond donors
            hba       : int    H-bond acceptors
            bbb_prob  : float  BBB penetration probability [0–1]
            herg_prob : float  hERG inhibition probability [0–1]
            dili_prob : float  drug-induced liver injury risk [0–1]
            f_oral    : float  oral bioavailability estimate [0–1]
            vd_lkg    : float  volume of distribution [L/kg]
            confidence: str    description of prediction quality
            predictor : str    backend used
            raw       : dict   all raw ADMET-AI outputs
        """
        if not smiles or not smiles.strip():
            raise ValueError("SMILES string cannot be empty.")

        if self._backend == "admet_ai":
            return self._predict_admet_ai(smiles)
        else:
            return self._mock.predict(smiles)

    def _predict_admet_ai(self, smiles: str) -> dict:
        """
        Run ADMET-AI inference and convert outputs to PBPK parameters.
        """
        # ── Run model ─────────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self._model.predict(smiles=smiles)

        # ── 1. logP ───────────────────────────────────────────────────────
        # Use Lipophilicity_AstraZeneca (AZ dataset, widely validated)
        # Fallback: logP from physicochemical calculator
        logp = float(raw.get("Lipophilicity_AstraZeneca",
                              raw.get("logP", 0.0)))
        logp = float(np.clip(logp, -6.0, 8.0))

        # ── 2. fup (fraction unbound in plasma) ───────────────────────────
        # PPBR_AZ = plasma protein binding RATE, in logit space
        # Model trained on logit(PPBR/100) — output ≈ logit(fraction_bound)
        # Metformin: PPBR_AZ=-3.58 → sigmoid(-3.58)=0.027=2.7% bound → fup=0.973 ✓
        # Ibuprofen: PPBR_AZ≈+4.6  → sigmoid(4.6)=0.99=99% bound → fup=0.01 ✓
        ppbr_raw     = float(raw.get("PPBR_AZ", 0.0))
        frac_bound   = _sigmoid(ppbr_raw)
        fup          = float(np.clip(1.0 - frac_bound, 0.001, 1.0))

        # ── 3. CLint (hepatic intrinsic clearance, L/h per 70 kg) ─────────
        # Primary: Clearance_Hepatocyte_AZ in log10(µL/min/10^6 cells)
        # Scale to whole-liver L/h using Houston (1994) hepatocyte scaling.
        hep_raw  = float(raw.get("Clearance_Hepatocyte_AZ", -8.0))
        clint_hep = (10.0 ** hep_raw) * _HEPATOCYTE_SCALE

        # Backup: Clearance_Microsome_AZ in log10(µL/min/mg protein)
        # Scaling: 52.5 mg microsomal protein/g liver × 1690g liver
        # = 88725 mg protein total × 60/1e6 L/h conversion = 5.3235 L/h
        mic_raw   = float(raw.get("Clearance_Microsome_AZ", -8.0))
        clint_mic = (10.0 ** mic_raw) * 5.3235

        # Use hepatocyte as primary; microsome as sanity check
        # Weight toward hepatocyte (more physiologically relevant for PBPK)
        clint = float(0.7 * clint_hep + 0.3 * clint_mic)

        # CYP substrate flags — if drug is NOT a CYP substrate, scale down
        cyp3a4_prob = float(raw.get("CYP3A4_Substrate_CarbonMangels", 0.5))
        cyp2d6_prob = float(raw.get("CYP2D6_Substrate_CarbonMangels", 0.5))
        cyp1a2_prob = float(raw.get("CYP1A2_Veith", 0.5))
        max_cyp     = max(cyp3a4_prob, cyp2d6_prob, cyp1a2_prob)

        # If no CYP substrate signal, cap CLint at low value
        if max_cyp < 0.2 and clint > 5.0:
            clint = clint * max_cyp / 0.2

        clint = float(np.clip(clint, 0.1, 2000.0))

        # ── 4. pKa estimation ─────────────────────────────────────────────
        # ADMET-AI does not directly predict pKa.
        # Use mock predictor's chemistry-based pKa — it is the best
        # available without a dedicated pKa model.
        from .mock_predictor import MockPredictor
        _mock_tmp = MockPredictor()
        from .smiles_features import parse_smiles
        feat = parse_smiles(smiles)
        pka, drug_type = _mock_tmp._predict_pka_and_type(feat, logp)

        # ── 5. Absorption parameters ──────────────────────────────────────
        # HIA_Hou = human intestinal absorption probability [0–1]
        hia      = float(raw.get("HIA_Hou", 0.5))
        # Caco2_Wang = log10(cm/s) membrane permeability
        # Convert to absorption rate ka (h^-1): higher Caco2 → faster ka
        # Typical: Caco2 = -5.5 log(cm/s) → ka ≈ 1.0 h^-1
        caco2    = float(raw.get("Caco2_Wang", -5.5))
        ka       = float(np.clip(10.0 ** (caco2 + 5.5) * 1.2, 0.05, 5.0))

        # Oral bioavailability: use Bioavailability_Ma as F estimate
        # Note: this is probability of F>20%, not the actual F value
        # Use HIA_Hou × (1 - hepatic first pass) as better F estimate
        # First pass Eh from well-stirred model (will be computed by PBPK)
        # Here use HIA as Fa; hepatic first pass handled by CLint in model
        f_oral = float(np.clip(hia, 0.05, 1.0))

        # ── 6. Safety flags ───────────────────────────────────────────────
        bbb_prob  = float(raw.get("BBB_Martins", 0.5))
        herg_prob = float(raw.get("hERG", 0.0))
        dili_prob = float(raw.get("DILI", 0.0))

        # ── 7. Volume of distribution ─────────────────────────────────────
        # VDss_Lombardo in log10(L/kg)
        vdss_raw = float(raw.get("VDss_Lombardo", 0.0))
        vd_lkg   = float(10.0 ** vdss_raw)

        # ── 8. Molecular properties ───────────────────────────────────────
        mw  = float(raw.get("molecular_weight", feat["mw"]))
        hbd = int(round(raw.get("hydrogen_bond_donors",   feat["hbd"])))
        hba = int(round(raw.get("hydrogen_bond_acceptors", feat["hba"])))

        return {
            # ── Core PBPK inputs
            "logp":      round(logp,  2),
            "pka":       round(pka,   1) if pka is not None else None,
            "fup":       round(fup,   4),
            "clint":     round(clint, 2),
            "drug_type": drug_type,
            "mw":        round(mw,    1),
            "hbd":       hbd,
            "hba":       hba,

            # ── Absorption
            "ka":        round(ka,     2),
            "f_oral":    round(f_oral, 3),

            # ── Safety signals
            "bbb_prob":  round(bbb_prob,  3),
            "herg_prob": round(herg_prob, 3),
            "dili_prob": round(dili_prob, 3),

            # ── Distribution
            "vd_lkg":    round(vd_lkg, 3),

            # ── CYP substrate flags
            "cyp3a4_substrate": round(cyp3a4_prob, 3),
            "cyp2d6_substrate": round(cyp2d6_prob, 3),

            # ── Metadata
            "confidence": "ADMET-AI v2 (ChemProp ensemble, 41 endpoints)",
            "predictor":  "admet_ai",
            "raw":        raw,
        }

    # ── Batch prediction ───────────────────────────────────────────────────
    def predict_batch(self, smiles_list: list) -> list:
        """Predict ADMET properties for a list of SMILES strings."""
        return [self.predict(s) for s in smiles_list]

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_ml(self) -> bool:
        return self._backend == "admet_ai"

    def __repr__(self):
        return f"ChemPropPredictor(backend={self._backend!r})"


# ── Bridge: SMILES → drug profile for PBPKModel ───────────────────────────

def smiles_to_drug_profile(smiles: str,
                            name: str = "Unknown",
                            predictor: Optional[ChemPropPredictor] = None,
                            clrenal_override: Optional[float] = None,
                            ka_override: Optional[float] = None,
                            F_override: Optional[float] = None) -> dict:
    """
    Full pipeline: SMILES → drug profile ready for PBPKModel.

    Parameters
    ----------
    smiles           : str   SMILES string of the new drug
    name             : str   drug name (for display)
    predictor        : ChemPropPredictor (creates one if None)
    clrenal_override : float override renal CL if known (L/h)
    ka_override      : float override absorption rate if known (h^-1)
    F_override       : float override bioavailability if known

    Returns
    -------
    dict — drug profile compatible with PBPKModel
    """
    if predictor is None:
        predictor = ChemPropPredictor(verbose=False)

    admet = predictor.predict(smiles)

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from engine.admet import build_drug_profile

    # Use predicted ka and F unless overridden
    ka_use = ka_override if ka_override is not None else admet.get("ka", 1.0)
    F_use  = F_override  if F_override  is not None else admet.get("f_oral", 0.8)

    profile = build_drug_profile(
        name              = name,
        logp              = admet["logp"],
        fup               = admet["fup"],
        mw                = admet["mw"],
        pka               = admet["pka"],
        drug_type         = admet["drug_type"],
        clint_override    = admet["clint"],
        clrenal_override  = clrenal_override,
        ka_override       = ka_use,
        F_override        = F_use,
    )

    # Attach ML metadata and safety flags
    profile.update({
        "smiles":           smiles,
        "bbb_prob":         admet.get("bbb_prob",  0.0),
        "herg_prob":        admet.get("herg_prob", 0.0),
        "dili_prob":        admet.get("dili_prob", 0.0),
        "vd_lkg":           admet.get("vd_lkg",    1.0),
        "cyp3a4_substrate": admet.get("cyp3a4_substrate", 0.5),
        "cyp2d6_substrate": admet.get("cyp2d6_substrate", 0.5),
        "predictor":        admet.get("predictor", "unknown"),
        "confidence":       admet.get("confidence", "unknown"),
        "admet_raw":        admet.get("raw", {}),
    })

    return profile


# ── Convenience function ───────────────────────────────────────────────────

_shared_predictor: Optional[ChemPropPredictor] = None

def get_predictor(verbose: bool = True) -> ChemPropPredictor:
    """Return shared singleton predictor (loaded once)."""
    global _shared_predictor
    if _shared_predictor is None:
        _shared_predictor = ChemPropPredictor(verbose=verbose)
    return _shared_predictor


# ── CLI self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("BodySim ChemPropPredictor — Self Test")
    print("=" * 60)

    p = ChemPropPredictor(verbose=True)
    print(f"\nBackend : {p.backend}")
    print(f"Is ML   : {p.is_ml}")
    print()

    # Reference compounds with literature values
    cases = [
        ("CN(C)C(=N)NC(=N)N",              "Metformin",   -1.43, 0.97,  20.0),
        ("Cn1c(=O)c2c(ncn2C)n(c1=O)C",     "Caffeine",    -0.07, 0.64,  12.0),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O",     "Ibuprofen",    3.97, 0.01, 180.0),
        ("CC(=O)Oc1ccccc1C(=O)O",           "Aspirin",      1.19, 0.49,  50.0),
    ]

    print(f"{'Drug':<12} {'logP':>6} {'lit':>5}  "
          f"{'fup':>5} {'lit':>5}  {'CLint':>7} {'lit':>7}  "
          f"{'hERG':>5} {'BBB':>5} {'DILI':>5}")
    print("-" * 75)

    for smiles, name, lit_logp, lit_fup, lit_cl in cases:
        pr = p.predict(smiles)
        lp = "✓" if abs(pr["logp"] - lit_logp) < 1.5 else "✗"
        fu = "✓" if abs(pr["fup"]  - lit_fup)  < 0.25 else "✗"
        print(f"{name:<12} {pr['logp']:>6.2f}{lp} {lit_logp:>5.2f}  "
              f"{pr['fup']:>5.3f}{fu} {lit_fup:>5.3f}  "
              f"{pr['clint']:>7.2f} {lit_cl:>7.1f}  "
              f"{pr.get('herg_prob',0):>5.3f} "
              f"{pr.get('bbb_prob',0):>5.3f} "
              f"{pr.get('dili_prob',0):>5.3f}")