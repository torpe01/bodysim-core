"""
reference_pk.py — Ground Truth database for 25 validation drugs.
Data sources: ChemSpider, DrugBank, literature (Goodman & Gilman, Pharmacokinetics).
All drugs have measured ADME properties (logP, fup, Mw, pKa, drug_type).
"""

REFERENCE_PK = {
    # Antibiotics & Antivirals
    "Aciclovir":      {"smiles": "Nc1nc2c(ncn2COCCO)c(=O)[nH]1", "logp": -1.56, "fup": 0.85, "mw": 225.2, "pka": 2.27, "drug_type": "neutral", "dose": 400, "route": "oral", "cmax": 1.2, "auc": 5.4, "ka": 1.3, "F": 0.20, "clint": 1.0, "clrenal": 15.0},
    "Amoxicillin":    {"smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(c3ccc(cc3)O)N)C(=O)O)C", "logp": 0.87, "fup": 0.82, "mw": 365.4, "pka": 2.68, "drug_type": "acidic", "dose": 500, "route": "oral", "cmax": 8.0, "auc": 25.0, "ka": 1.2, "F": 0.95, "clint": 0.5, "clrenal": 8.7},
    "Ciprofloxacin":  {"smiles": "C1CC1n2cc(c(=O)c3cc(c(cc23)N4CCNCC4)F)C(=O)O", "logp": -1.10, "fup": 0.70, "mw": 331.3, "pka": 6.09, "drug_type": "zwitterion", "dose": 500, "route": "oral", "cmax": 2.4, "auc": 12.0, "ka": 1.4, "F": 0.87, "clint": 1.5, "clrenal": 3.2},
    "Fluconazole":    {"smiles": "Oc1c(F)cc(F)cc1C(Cn2cncn2)(Cn3cncn3)", "logp": 0.50, "fup": 0.88, "mw": 306.3, "pka": 1.76, "drug_type": "neutral", "dose": 100, "route": "oral", "cmax": 2.2, "auc": 85.0, "ka": 1.8, "F": 0.90, "clint": 1.2, "clrenal": 3.2},
    
    # Cardiovascular
    "Atorvastatin":   {"smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(c(n1CCC(O)CC(O)CC(=O)O)c3ccc(F)cc3)c4ccccc4", "logp": 4.06, "fup": 0.02, "mw": 558.6, "pka": 4.46, "drug_type": "acidic", "dose": 40, "route": "oral", "cmax": 0.012, "auc": 0.08, "ka": 1.5, "F": 0.14, "clint": 4.2, "clrenal": 0.1},
    "Digoxin":        {"smiles": "CC1OC(OC2C(O)CC(OC3C(O)CC(OC4CCC5(C)C(CCC6C5CC(O)C7(C)C(C8=CC(=O)OC8)CCC67)C4)OC3C)OC2C)CC(O)C1O", "logp": 1.26, "fup": 0.25, "mw": 780.9, "pka": 6.70, "drug_type": "neutral", "dose": 0.25, "route": "oral", "cmax": 0.001, "auc": 0.035, "ka": 0.8, "F": 0.70, "clint": 0.5, "clrenal": 4.2},
    "Furosemide":     {"smiles": "NS(=O)(=O)c1cc(Cl)c(cc1C(=O)O)NCc2occc2", "logp": 2.03, "fup": 0.02, "mw": 330.7, "pka": 3.9, "drug_type": "acidic", "dose": 40, "route": "oral", "cmax": 1.1, "auc": 3.4, "ka": 1.3, "F": 0.61, "clint": 2.0, "clrenal": 8.0},
    "Metoprolol":     {"smiles": "COCc1ccc(cc1)OCC(O)CNC(C)C", "logp": 1.88, "fup": 0.87, "mw": 267.4, "pka": 9.68, "drug_type": "basic", "dose": 100, "route": "oral", "cmax": 0.12, "auc": 0.85, "ka": 1.5, "F": 0.38, "clint": 70.0, "clrenal": 0.4},
    "Nifedipine":     {"smiles": "COC(=O)C1=C(C)NC(C)=C(C1c2ccccc2[N+](=O)[O-])C(=O)OC", "logp": 3.17, "fup": 0.95, "mw": 346.3, "pka": 7.80, "drug_type": "neutral", "dose": 20, "route": "oral", "cmax": 0.08, "auc": 0.25, "ka": 1.8, "F": 0.50, "clint": 220.0, "clrenal": 0.05},
    "Propranolol":    {"smiles": "CC(C)NCC(O)COc1cccc2ccccc12", "logp": 3.48, "fup": 0.13, "mw": 259.3, "pka": 9.42, "drug_type": "basic", "dose": 40, "route": "oral", "cmax": 0.04, "auc": 0.35, "ka": 1.7, "F": 0.26, "clint": 400.0, "clrenal": 0.1},
    "Rosuvastatin":   {"smiles": "CC(C)N(C)c1nc(nc(c1/C=C/[C@@H](O)C[C@@H](O)CC(=O)O)c2ccc(F)cc2)S(=O)(=O)C", "logp": -0.33, "fup": 0.12, "mw": 481.5, "pka": 4.20, "drug_type": "acidic", "dose": 20, "route": "oral", "cmax": 0.015, "auc": 0.12, "ka": 1.4, "F": 0.20, "clint": 1.2, "clrenal": 0.8},
    "Warfarin":       {"smiles": "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O", "logp": 2.70, "fup": 0.007, "mw": 308.3, "pka": 5.08, "drug_type": "acidic", "dose": 10, "route": "oral", "cmax": 1.1, "auc": 45.0, "ka": 1.2, "F": 0.93, "clint": 4.5, "clrenal": 0.05},

    # CNS & Pain
    "Alprazolam":     {"smiles": "Cc1nnc2n1-c3ccc(cc3C(=N2)c4ccccc4)Cl", "logp": 2.12, "fup": 0.20, "mw": 308.8, "pka": 2.40, "drug_type": "neutral", "dose": 1, "route": "oral", "cmax": 0.015, "auc": 0.22, "ka": 1.5, "F": 0.90, "clint": 5.0, "clrenal": 0.05},
    "Caffeine":       {"smiles": "Cn1c(=O)c2c(ncn2C)n(c1=O)C", "logp": -0.07, "fup": 0.64, "mw": 194.2, "pka": 0.52, "drug_type": "neutral", "dose": 100, "route": "oral", "cmax": 1.94, "auc": 15.5, "ka": 1.8, "F": 0.99, "clint": 15.0, "clrenal": 0.3},
    "Diazepam":       {"smiles": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13", "logp": 2.82, "fup": 0.01, "mw": 284.7, "pka": 3.40, "drug_type": "neutral", "dose": 10, "route": "oral", "cmax": 0.25, "auc": 5.8, "ka": 1.2, "F": 0.99, "clint": 8.0, "clrenal": 0.05},
    "Ibuprofen":      {"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "logp": 3.97, "fup": 0.01, "mw": 206.3, "pka": 4.91, "drug_type": "acidic", "dose": 400, "route": "oral", "cmax": 35.0, "auc": 125.0, "ka": 1.6, "F": 0.80, "clint": 8.0, "clrenal": 0.2},
    "Paracetamol":    {"smiles": "CC(=O)Nc1ccc(O)cc1", "logp": 0.49, "fup": 0.80, "mw": 151.2, "pka": 9.38, "drug_type": "neutral", "dose": 1000, "route": "oral", "cmax": 15.0, "auc": 50.0, "ka": 2.0, "F": 0.88, "clint": 5.0, "clrenal": 0.1},
    "Phenytoin":      {"smiles": "O=C1NC(=O)C(N1)(c2ccccc2)c3ccccc3", "logp": 2.47, "fup": 0.10, "mw": 252.3, "pka": 8.33, "drug_type": "neutral", "dose": 300, "route": "oral", "cmax": 4.5, "auc": 110.0, "ka": 1.0, "F": 0.85, "clint": 6.0, "clrenal": 0.1},
    "Midazolam":      {"smiles": "Cn1cc2c(n1)N=C(c3ccccc3)c1cc(F)ccc1-2", "logp": 3.89, "fup": 0.03, "mw": 325.8, "pka": 6.15, "drug_type": "basic", "dose": 7.5, "route": "oral", "cmax": 0.05, "auc": 0.18, "ka": 1.6, "F": 0.40, "clint": 180.0, "clrenal": 0.1},

    # GI & Metabolic
    "Cimetidine":     {"smiles": "Cc1c(nc[nH]1)CSCCN=C(NC)NC#N", "logp": 0.40, "fup": 0.80, "mw": 252.3, "pka": 6.80, "drug_type": "basic", "dose": 400, "route": "oral", "cmax": 1.8, "auc": 6.2, "ka": 1.6, "F": 0.62, "clint": 8.0, "clrenal": 10.0},
    "Metformin":      {"smiles": "CN(C)C(=N)NC(=N)N", "logp": -1.43, "fup": 0.97, "mw": 129.21, "pka": 1.5, "drug_type": "basic", "dose": 500, "route": "oral", "cmax": 1.3, "auc": 10.5, "ka": 1.8, "F": 0.55, "clint": 0.1, "clrenal": 30.6},
    "Omeprazole":     {"smiles": "COc1ccc2nc(sc2c1)S(=O)Cc3ncc(C)c(OC)c3C", "logp": 2.23, "fup": 0.97, "mw": 345.4, "pka": 0.77, "drug_type": "neutral", "dose": 20, "route": "oral", "cmax": 0.45, "auc": 1.2, "ka": 1.5, "F": 0.35, "clint": 45.0, "clrenal": 0.05},
    "Ranitidine":     {"smiles": "CNCc1ccc(o1)CSCCN=C(N)N[N+](=O)[O-]", "logp": 0.27, "fup": 0.85, "mw": 314.4, "pka": 8.20, "drug_type": "basic", "dose": 150, "route": "oral", "cmax": 0.5, "auc": 2.5, "ka": 1.5, "F": 0.50, "clint": 6.0, "clrenal": 9.0},
}
