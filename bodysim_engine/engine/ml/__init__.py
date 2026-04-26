"""
engine/ml — AI Brain for BodySim.

The ML layer sits between raw SMILES input and the PBPK engine.
It predicts ADMET properties using:
  - ChemProp v2 (when weights are installed) — real pre-trained GNN
  - MockPredictor (fallback) — calibrated QSPR rules

Quick usage:
    from engine.ml import predict_admet, smiles_to_drug_profile

    # Just get ADMET predictions
    props = predict_admet("CN(C)C(=N)NC(=N)N")

    # Full pipeline: SMILES → ready-to-simulate drug profile
    drug = smiles_to_drug_profile("CN(C)C(=N)NC(=N)N", name="Metformin")
"""

from .chemprop_predictor import ChemPropPredictor, smiles_to_drug_profile

# Shared singleton predictor (loaded once, reused across calls)
_predictor: ChemPropPredictor | None = None


def _get_predictor() -> ChemPropPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ChemPropPredictor(verbose=True)
    return _predictor


def predict_admet(smiles: str) -> dict:
    """
    Predict ADMET properties from a SMILES string.

    Returns dict with: logp, pka, fup, clint, drug_type, mw, hbd, hba,
                       confidence, predictor
    """
    return _get_predictor().predict(smiles)


__all__ = [
    "ChemPropPredictor",
    "smiles_to_drug_profile",
    "predict_admet",
]
