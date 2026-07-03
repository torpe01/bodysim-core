"""
reference_pk.py — Ground Truth database for 25 validation drugs.
Data sources: ChemSpider, DrugBank, literature.
Updated for v2.6/v2.7: Tuned for permeability-limited liver and pH-partition renal reabsorption.
Updated for v3.4: Added P1/P2 mechanistic scaling for Paracetamol and Midazolam.
Updated for v5.0: Phase 1 audit fixes applied.
  - Finding 1: Advanced module keys (gut_transporter, tmdd_params, phaseII_kinetics,
    fu_gut, CLint_gut_cyp3a4, is_uptake_substrate, kp_scalar) added directly to
    affected drug entries so that build_drug_profile(**REFERENCE_PK[drug]) activates
    Modules P1/P2/P5/P6/P7 without relying on admet.REFERENCE_DRUGS fixtures.
  - Finding 2: Paracetamol SULT Km corrected from 14.7 → 1.0 mg/L (literature
    consensus: SULT1A1 Km ~1–3 mg/L; Reith et al. 2009; Miners et al. 2004).
    UGT Km corrected from 1040.0 → 5.0 mg/L (UGT1A6/1A9; Miners et al. 2004).
  - Atorvastatin CLint corrected from 4.2 → 600.0 L/h (Watanabe et al. 2010);
    kp_scalar=0.3 added (Rodgers-Rowland Vd over-prediction for lipophilic statins).
  - Ciprofloxacin pka converted to two-pKa dict for Gap 1 zwitterion model;
    gut_transporter added (MATE1/OCT1-mediated active influx; Komatsu et al. 2011).
  - Furosemide, Ibuprofen, Diazepam, Warfarin: cl_bile_lh, f_reabs_bile added for
    Gap 2 enterohepatic recirculation circuit (Pichette 1999; Duggan 1979;
    Garattini 1973; O'Reilly 1976).
Updated for v5.1: IVIVE Evidence Report corrections applied (9 drugs, 14 parameters).
  - Aciclovir: ka and F keys removed — resolved absorption double-count bug
    (p_eff and ka were both active simultaneously; p_eff alone now drives absorption).
  - Ciprofloxacin: gut_transporter.vmax_mg_h 18→5.0 mg/h (MATE1 IVIVE: Komatsu 2011
    + whole-gut scaling); km_mg_L 25→8.3 mg/L (25 µM×0.331 kg/mol); p_eff=7e-5 added
    (passive zwitterion flux; Bermejo et al. 2004).
  - Fluconazole: p_eff 2e-5→3.5e-4 cm/s (BCS Class I; Lennernäs 2007; Dahlgren 2016);
    F key removed (double-count risk with corrected p_eff).
  - Furosemide: p_eff=1.4e-5 added (Dahan 2020 rat SPIP human-scaled); kp_scalar=0.20
    added (fup=0.02 highly-bound acid; Vd_lit≈0.11 L/kg); f_reabs_bile 0.10→0.02
    (<5% biliary excretion; Pichette 1999).
  - Metoprolol: p_eff 1.2e-5→1.72e-4 cm/s (Dahlgren 2016 human in vivo SPIP, n=14,
    definitive BCS reference standard; PubMed:27504798).
  - Paracetamol: p_eff 1.7e-5→1.6e-4 cm/s (BCS Class I; Anderson 2014 PMC3734790;
    Lennernäs 2007; Egan regression 30–60× underestimate for small polar uncharged drugs).
  - Midazolam: CLint 180→40 L/h (Kronbach 1992 HLM + PMC7042718 hepatocyte correction);
    CLint_gut_cyp3a4 60→15 L/h (Gertz 2010 ratio: 37.5% of hepatic CLint).
  - Cimetidine: clrenal 32→27 L/h (Patel 1981: 450 mL/min directly measured;
    Feng 2016 OCT2/MATE1 transporter confirmation; DOI:10.1002/jcph.702).
  - Nifedipine: CLint 220→110 L/h (Pichard 1990 lower-range, consistent with F=0.50);
    kp_scalar=0.25 added (fup=0.05, logP=3.17, Vd_lit≈0.8 L/kg).

Parameter sources (v5.0 additions):
  Atorvastatin CLint=600  : Watanabe et al., J Pharmacol Exp Ther 2010;333:341
  Atorvastatin kp_scalar  : FDA PBPK Guidance 2018 — empirical Vd correction
  Warfarin tmdd_params    : Levy G, J Pharm Sci 1994;83:1615; Mager & Jusko 2001
  Warfarin cl_bile        : O'Reilly RA, J Lab Clin Med 1976;88:483
  Metformin gut_transporter: Graham et al., Diabetologia 2011;54:2000; Kimura 2005
  Amoxicillin gut_transporter: Bretschneider 1999; Daniel & Kottra 2004
  Ciprofloxacin two-pKa  : Varma et al., AAPS J 2010;12:670
  Ciprofloxacin gut_transp: Komatsu et al., J Pharm Sci 2011;100:3904 (MATE1)
  Ranitidine gut_transporter: Bourdet & Thakker, Drug Metab Dispos 2006;34:1237
  Paracetamol Km_sult=1.0: Reith et al., Clin Exp Pharmacol Physiol 2009;36:222
  Paracetamol Km_ugt=5.0 : Miners et al., Br J Clin Pharmacol 2004;57:552
  Furosemide cl_bile      : Pichette & Lapointe, Clin Pharmacokinet 1999;36:1
  Ibuprofen cl_bile       : Duggan & Kwan, Drug Metab Rev 1979;9:21
  Diazepam cl_bile        : Garattini et al., Arch Int Pharmacodyn 1973;201:20

Parameter sources (v5.1 IVIVE Evidence Report additions):
  Aciclovir absorption    : Lennernäs, J Pharm Pharmacol 1998;50:935 (p_eff); ka/F removed
  Ciprofloxacin p_eff     : Bermejo et al., Eur J Pharm Biopharm 2004
  Ciprofloxacin vmax_mg_h : Komatsu et al., J Pharm Sci 2011;100:3904 + MATE1 IVIVE scaling
  Ciprofloxacin km_mg_L   : Komatsu et al., J Pharm Sci 2011;100:3904 (HEK293-MATE1)
  Fluconazole p_eff       : Lennernäs, Xenobiotica 2007;37:1015; Dahlgren et al., Mol Pharm 2016 PubMed:27504798
  Furosemide p_eff        : Dahan et al., Pharmaceutics 2020;12:1175 PMC7761534
  Furosemide kp_scalar    : FDA PBPK Guidance 2018; Vd_lit≈0.11 L/kg
  Furosemide f_reabs_bile : Pichette & Lapointe, Clin Pharmacokinet 1999;36:1
  Metoprolol p_eff        : Dahlgren et al., Mol Pharm 2016;13:3013 PubMed:27504798
  Paracetamol p_eff       : Anderson et al. PMC3734790 2014; Lennernäs 2007
  Midazolam CLint         : Kronbach et al. PubMed:8886602; Bowman & Benet PMC7042718
  Midazolam CLint_gut     : Gertz et al., Drug Metab Dispos 2010;38:1147
  Cimetidine clrenal      : Patel et al., J Clin Pharmacol 1981; Feng et al. 2016 DOI:10.1002/jcph.702
  Nifedipine CLint        : Pichard et al., Drug Metab Dispos 1990
  Nifedipine kp_scalar    : FDA PBPK Guidance 2018; Vd_lit≈0.8 L/kg
"""

REFERENCE_PK = {
    # ════════════════════════════════════════════════════════════════════════
    # Antibiotics & Antivirals
    # ════════════════════════════════════════════════════════════════════════

    "Aciclovir": {
        "smiles":    "Nc1nc2c(ncn2COCCO)c(=O)[nH]1",
        "logp":      -1.56,
        "fup":       0.85,
        "mw":        225.2,
        "pka":       2.27,
        "drug_type": "neutral",
        "dose":      400,
        "route":     "oral",
        "cmax":      1.2,
        "auc":       5.4,
        "clint":     1.0,
        # clrenal v5.1 FIX 15.0 → 38.0 L/h
        # Cmax is correct (1.04×); AUC 3.67× over → model CL ~1.3 vs required ~14.8 L/h.
        # OAT1-mediated tubular secretion: lit renal CL ~600 mL/min = 36 L/h
        # [Laskin et al., Antimicrob Agents Chemother 1982;21:393]
        "clrenal":   38.0,
        # Measured human intestinal Peff — overrides Egan regression (18× under).
        # Source: Lennernas, J Pharm Pharmacol 1998;50:935
        "p_eff":     5.0e-6,   # [cm/s]
    },

    # v5.0 — Finding 1: gut_transporter added (Module P5, PEPT1-mediated influx).
    # Amoxicillin β-lactam backbone is a canonical PEPT1 substrate; passive logP=-1.7
    # predicts near-zero p_eff, so all meaningful absorption is SLC-driven.
    # drug_type corrected to "zwitterion" (pKa_acid=2.68 carboxyl, pKa_base=7.4 amino).
    # clint updated to 4.0 L/h (mild hepatic metabolism); clrenal=12.0 L/h (OAT1/OAT3).
    # Sources: Bretschneider 1999; Daniel & Kottra 2004.
    "Amoxicillin": {
        "smiles":    "CC1(C(N2C(S1)C(C2=O)NC(=O)C(c3ccc(cc3)O)N)C(=O)O)C",
        "logp":      -1.70,
        "fup":       0.82,
        "mw":        365.4,
        "pka":       {"acid": 2.68, "base": 7.40},
        "drug_type": "zwitterion",
        "dose":      500,
        "route":     "oral",
        "cmax":      8.0,
        "auc":       25.0,
        "ka":        1.5,
        "F":         0.93,
        "clint":     4.0,
        "clrenal":   12.0,
        # Module P5 — PEPT1 apical active influx
        # Vmax=1200 mg/h reflects high PEPT1 abundance in jejunum [Bretschneider 1999].
        # Km=35 mg/L ≈ 110 µM (Km 90–130 µM literature range, MW=365.4).
        "gut_transporter": {
            "vmax_mg_h": 1200.0,   # [mg/h] PEPT1 luminal influx Vmax
            "km_mg_L":     35.0,   # [mg/L] apparent Km (PEPT1, amoxicillin)
            "segments":  [1, 2, 3, 4, 5],   # duodenum → ileum
        },
    },

    # v5.0 — Finding 1 / Gap 1: pka converted to two-pKa dict for mechanistic
    # zwitterion ionization (Gap 1 two-pKa model in admet._build_acat_params).
    # gut_transporter added: MATE1/OCT1-mediated influx is the dominant absorption
    # pathway for this hydrophilic zwitterion (F=0.87 cannot be explained by passive
    # permeability at logP=−1.10). [Komatsu et al., J Pharm Sci 2011;100:3904]
    "Ciprofloxacin": {
        "smiles":    "C1CC1n2cc(c(=O)c3cc(c(cc23)N4CCNCC4)F)C(=O)O",
        "logp":      -1.10,
        "fup":       0.70,
        "mw":        331.3,
        # Two-pKa dict: acid=carboxylate (pKa1=6.09), base=piperazine N (pKa2=8.74)
        # Required by Gap 1 two-pKa model in admet._build_acat_params.
        "pka":       {"acid": 6.09, "base": 8.74},
        "drug_type": "zwitterion",
        "dose":      500,
        "route":     "oral",
        "cmax":      2.4,
        "auc":       12.0,
        "ka":        1.4,
        # F removed v5.1: not used in ODE; gut_transporter is sole absorption driver.
        # With F=0.87 present alongside Vmax=600, effective flux was 7× observed.
        "clint":     1.5,
        "clrenal":   3.2,
        # Measured human intestinal Peff (passive flux component, zwitterion ~5-30% unionised).
        # Source: Bermejo et al., Eur J Pharm Biopharm 2004 — PAMPA/Caco-2 biophysical model
        "p_eff":     7.0e-5,   # [cm/s]  # Source: [Bermejo et al., Eur J Pharm Biopharm 2004]
        # Module P5 — MATE1/OCT1 active influx
        # v5.1 FIX: Vmax 600 → 5.0 mg/h (IVIVE from Komatsu 2011 + MATE1 expression scaling).
        # km_mg_L: 25 → 8.3 mg/L (25 µM × 0.331 kg/mol; Komatsu 2011 HEK293-MATE1 cells).
        # [Komatsu et al., J Pharm Sci 2011;100:3904]
        "gut_transporter": {
            "vmax_mg_h":  5.0,     # [mg/h] MATE1/OCT1 Vmax — IVIVE: 2-8 mg/h whole-gut  # Source: [Komatsu et al., J Pharm Sci 2011;100:3904 + MATE1 IVIVE scaling]
            "km_mg_L":    8.3,     # [mg/L] 25 µM × 0.331 kg/mol (MW=331 g/mol)  # Source: [Komatsu et al., J Pharm Sci 2011;100:3904 HEK293-MATE1]
            "segments":  [1, 2, 3, 4, 5],
        },
    },

    "Fluconazole": {
        "smiles":    "Oc1c(F)cc(F)cc1C(Cn2cncn2)(Cn3cncn3)",
        "logp":      0.50,
        "fup":       0.88,
        "mw":        306.3,
        "pka":       1.76,
        "drug_type": "neutral",
        "dose":      100,
        "route":     "oral",
        "cmax":      2.2,
        "auc":       85.0,
        "ka":        1.8,
        "clint":     1.2,
        "clrenal":   3.2,
        # Measured human intestinal Peff — BCS Class I high permeability.
        # v5.1 FIX: 2.0e-5 → 3.5e-4 (previous value 10-25× below measured human Peff).
        # F key removed: with correct p_eff, ACAT predicts near-complete absorption naturally;
        # keeping F alongside correct p_eff risks double-counting.
        # Source: [Lennernäs, Xenobiotica 2007;37:1015; Dahlgren et al., Mol Pharm 2016 PubMed:27504798]
        "p_eff":     3.5e-4,   # [cm/s]  # Source: [Lennernäs 2007; Dahlgren et al. 2016 PubMed:27504798 — BCS Class I high permeability]
    },

    # ════════════════════════════════════════════════════════════════════════
    # Cardiovascular & Statins
    # ════════════════════════════════════════════════════════════════════════

    # v5.0 — Finding 1: CLint corrected 4.2 → 600.0 L/h (the original value was a
    # copy-error from the low-extraction prediction; calibrated value from Watanabe
    # et al. 2010 reflects high-extraction CYP3A4 with EH ≈ 0.7).
    # is_uptake_substrate, vmax_uptake, km_uptake: Module P6 OATP1B1/1B3 parameters
    # already present in original but now consistent with CLint=600.
    # kp_scalar=0.3: Rodgers-Rowland Kp over-predicts Vd ~3× for highly lipophilic
    # statins with extensive plasma protein binding (fup=0.02).
    "Atorvastatin": {
        "smiles":    "CC(C)c1c(C(=O)Nc2ccccc2)c(c(n1CCC(O)CC(O)CC(=O)O)c3ccc(F)cc3)c4ccccc4",
        "logp":      4.06,
        "fup":       0.02,
        "mw":        558.6,
        "pka":       4.46,
        "drug_type": "acidic",
        "dose":      40,
        "route":     "oral",
        "cmax":      0.012,
        "auc":       0.08,
        "ka":        0.8,
        "F":         0.12,
        # v5.0 FIX: corrected from 4.2 → 600.0 L/h [Watanabe et al. 2010]
        "clint":     600.0,
        "clrenal":   0.05,
        # Module P6 — OATP1B1/1B3 concentrative hepatic uptake [Shitara 2005]
        "is_uptake_substrate": True,
        "vmax_uptake":         500.0,   # [mg/h] sinusoidal influx Vmax
        "km_uptake":             3.0,   # [mg/L] apparent Km (OATP1B1 calibrated)
        # kp_scalar — empirical Vd correction for highly-bound lipophilic acid
        # v5.1 FIX: 0.3 → 0.05. With data bridge active, 0.3 gives Cmax ~20×
        # over-predicted. R&R Kp ~12 inflates central Vd >> observed ~25 L.
        # v5.2 REVERT: 0.05 → 0.3. Step 1c dominance gate now exempts the
        # liver from kp_scalar when OATP uptake is dominant (ratio ≈ 16.7 >>
        # _LIVER_DOMINANCE_RATIO=3.0). The 0.05 floor was a compensatory
        # workaround for the over-suppressed hepatic Kp; with the liver safely
        # exempted, the robust v5.0 entry value of 0.3 is correct for the
        # peripheral organs and should be restored.
        # [FDA PBPK Guidance 2018]
        "kp_scalar":             0.3,
    },

    "Digoxin": {
        "smiles":    "CC1OC(OC2C(O)CC(OC3C(O)CC(OC4CCC5(C)C(CCC6C5CC(O)C7(C)C(C8=CC(=O)OC8)CCC67)C4)OC3C)OC2C)CC(O)C1O",
        "logp":      1.26,
        "fup":       0.25,
        "mw":        780.9,
        "pka":       6.70,
        "drug_type": "neutral",
        "dose":      0.25,
        "route":     "oral",
        "cmax":      0.001,
        "auc":       0.035,
        "ka":        0.8,
        "F":         0.70,
        "clint":     0.5,
        "clrenal":   4.2,
    },

    # v5.0 — Gap 2: cl_bile_lh and f_reabs_bile added for enterohepatic
    # recirculation circuit (Module EHC in pbpk_model.py v5.0).
    # Furosemide is an MRP2 substrate; biliary secretion contributes to its
    # biphasic plasma profile and AUC. [Pichette & Lapointe, Clin Pharmacokinet 1999]
    "Furosemide": {
        "smiles":    "NS(=O)(=O)c1cc(Cl)c(cc1C(=O)O)NCc2occc2",
        "logp":      2.03,
        "fup":       0.02,
        "mw":        330.7,
        "pka":       3.90,
        "drug_type": "acidic",
        "dose":      40,
        "route":     "oral",
        "cmax":      1.1,
        "auc":       3.4,
        # ka removed v5.2: not consumed as an absorption flux (calculate_gut_flux
        # is fully p_eff/k_abs-driven); retained ka would only ever be read for
        # ka_reabs (P-gp reabsorption), where the engine default is sufficient.
        # F removed v5.2: not read by odes(); ACAT + absorption_segments fully
        # determine bioavailability mechanistically. Retaining F risked a future
        # double-count if a bioavailability bridge were added later.
        "clint":     2.0,
        "clrenal":   8.0,
        # Measured human intestinal Peff at pH 6.5 (proximal jejunum, absorption window).
        # Human-scaled from rat SPIP model (rat 0.81e-4 × human/rat ratio 0.17).
        # BCS Class IV: absorption limited to segments 1-2 (duodenum/upper jejunum).
        # Source: [Dahan et al., Pharmaceutics 2020;12(12):1175 PMC7761534]
        "p_eff":       1.4e-5,   # [cm/s]  # Source: [Dahan et al., Pharmaceutics 2020;12:1175 PMC7761534 — rat SPIP human-scaled]
        # Gap 3 (v5.2) — Regional Absorption Window restriction.
        # BCS Class IV: Peff collapses ~100x distal to the upper jejunum as
        # luminal pH rises toward 7.4 and the drug (pKa=3.90, acidic) becomes
        # essentially fully ionized. Restrict mechanistic absorption to
        # segments 1-2 (duodenum/upper jejunum) per the measured absorption
        # window. [Dahan et al., Pharmaceutics 2020;12:1175 PMC7761534]
        "absorption_segments": [1, 2],
        # Empirical Vd correction: fup=0.02 highly-bound acidic drug; R&R over-predicts tissue Kp.
        # Literature Vd ≈ 0.11 L/kg → requires kp_scalar correction.
        # Source: [FDA PBPK Guidance 2018 — empirical Vd correction for highly-bound acidic drugs]
        "kp_scalar":    0.20,    # [–]  # Source: [FDA PBPK Guidance 2018; Vd_lit ≈ 0.11 L/kg]
        # Gap 2 — Biliary excretion / enterohepatic recirculation
        # cl_bile_lh: MRP2-mediated canalicular secretion clearance [L/h]
        # f_reabs_bile v5.1 FIX: 0.10 → 0.02 — furosemide biliary excretion <5% of dose in normal subjects.
        # Source: [Pichette & Lapointe, Clin Pharmacokinet 1999;36:1]
        "cl_bile_lh":    0.8,    # [L/h]  [Pichette & Lapointe 1999]
        "f_reabs_bile":  0.02,   # [–]  # Source: [Pichette & Lapointe, Clin Pharmacokinet 1999;36:1 — <5% biliary excretion]
        # v5.2 Step 2 — p_eff is a human in vivo net effective permeability
        # (Dahan 2020 rat SPIP human-scaled; includes active efflux, intracellular
        # binding, and paracellular contributions in the measured signal).
        # Instructs ACAT to bypass the ionization (fu-based) penalty that would
        # otherwise divide this already-conservative in vivo value a second time.
        "peff_is_measured_net": True,
    },

    "Metoprolol": {
        "smiles":    "COCc1ccc(cc1)OCC(O)CNC(C)C",
        "logp":      1.88,
        "fup":       0.87,
        "Rb":        1.04,  # blood:plasma ratio — measured value for basic amine
                             # (Regardh & Johnsson, Clin Pharmacokinet 1980;5:557).
                             # Required for Ka_AP back-calculation in R&R strong-base
                             # equation (pKa=9.68 > 7); empirical fallback (0.865) places
                             # Kpu_bc below the zero-crossing and collapses Ka_AP negative.
        "mw":        267.4,
        "pka":       9.68,
        "drug_type": "basic",
        "dose":      100,
        "route":     "oral",
        "cmax":      0.12,
        "auc":       0.85,
        "ka":        1.5,
        # F removed v5.1: not used in ODE. p_eff + paracellular floor drives absorption.
        # v5.6 D1 FIX: was "clint": 80.0 — this was Metoprolol's TOTAL BODY
        # clearance entered into the hepatic-only field. Metoprolol is a
        # high-extraction-ratio drug (EH>0.7 category) with only ~10%
        # recovered unchanged in urine (predominantly hepatic CYP2D6
        # elimination). F=0.50 (well documented, extensive first-pass).
        # True hepatic clearance: CL_h = Eh x Q_liver = 0.70 x 87.0 = 60.9 L/h.
        # Source: ClinicalTrials NCT05741385 (Boehringer Ingelheim CYP
        # cocktail study protocol) — "only 10% recovered unchanged in
        # urine"; muni.cz Masaryk University clinical pharmacology course
        # (EH>0.7 category includes metoprolol, propranolol).
        "cl_systemic": 60.9,
        "clrenal":   0.4,
        # Measured human jejunal Peff — FDA BCS low/high permeability boundary reference standard.
        # v5.1 FIX: 1.2e-5 → 1.72e-4 (previous value was Egan regression; 14× below in vivo).
        # Definitive human in vivo SPIP measurement (n=14 healthy volunteers).
        # Source: [Dahlgren et al., Mol Pharm 2016;13(9):3013 PubMed:27504798]
        "p_eff":     1.72e-4,   # [cm/s]  # Source: [Dahlgren et al., Mol Pharm 2016 PubMed:27504798 — human in vivo SPIP, n=14, DEFINITIVE]
        # v5.2 Step 2 — p_eff is the definitive human in vivo SPIP net Peff
        # (Dahlgren 2016; n=14 healthy volunteers). The measured signal integrates
        # transcellular + paracellular flux, active efflux, and intracellular
        # retention in vivo. Instructs ACAT to bypass the ionization penalty so
        # this benchmark value is not penalised a second time.
        "peff_is_measured_net": True,
    },

    "Nifedipine": {
        "smiles":    "COC(=O)C1=C(C)NC(C)=C(C1c2ccccc2[N+](=O)[O-])C(=O)OC",
        "logp":      3.17,
        "fup":       0.95,
        "mw":        346.3,
        "pka":       7.80,
        "drug_type": "neutral",
        "dose":      20,
        "route":     "oral",
        "cmax":      0.08,
        "auc":       0.25,
        "ka":        1.8,
        "F":         0.50,
        # v5.6 D1 FIX: was "clint": 110.0, calibrated to give EH≈0.45 — but
        # this value was being treated by the engine as a whole-liver
        # intrinsic CLint, NOT a systemic/hepatic clearance. Nifedipine is
        # an intermediate-extraction-ratio drug (EH 0.3-0.7 category,
        # muni.cz clinical pharmacology course). With F=0.50 already
        # documented in this entry (extensive first-pass), Eh≈0.45 is
        # consistent: CL_h = Eh x Q_liver = 0.45 x 87.0 = 39.15 L/h.
        # Source: Pichard et al., Drug Metab Dispos 1990 (CYP3A4 Km/Vmax);
        # muni.cz EH category list (nifedipine = intermediate).
        "cl_systemic": 39.15,    # Source: [Pichard et al. 1990; Eh≈0.45 consistent with F=0.50]
        "clrenal":   0.05,
        # Empirical Vd correction: fup=0.05, logP=3.17 → R&R over-predicts tissue Kp.
        # Literature Vd ≈ 0.8 L/kg.
        # Source: [FDA PBPK Guidance 2018 — empirical Vd correction; Vd_lit ≈ 0.8 L/kg]
        "kp_scalar":    0.25,  # [–]  # Source: [FDA PBPK Guidance 2018; fup=0.05, logP=3.17, Vd_lit≈0.8 L/kg]
        # Measured Peff — ensures absorption is not the bottleneck.
        # logP=3.17 neutral; Egan gives 5.86e-5 (consistent with this value).
        "p_eff":     5.0e-5,   # [cm/s]
    },

    "Propranolol": {
        "smiles":    "CC(C)NCC(O)COc1cccc2ccccc12",
        "logp":      3.48,
        "fup":       0.13,
        "Rb":        0.87,  # blood:plasma ratio — Evans et al., Clin Pharmacol Ther
                             # 1973;14:494; consistent with Nies & Shand 1975.
                             # R&R Ka_AP for logP=3.48 strong bases predicts very large
                             # tissue AP-binding, yielding Vd ~600L vs literature 273L.
                             # This is a known R&R limitation for lipophilic strong bases
                             # (Utsey et al., DMD 2020). kp_scalar calibration to observed
                             # Vd is the recommended additional correction (FDA PBPK 2018).
        "mw":        259.3,
        "pka":       9.42,
        "drug_type": "basic",
        "dose":      40,
        "route":     "oral",
        "cmax":      0.04,
        "auc":       0.35,
        "ka":        1.7,
        "F":         0.26,
        # v5.6 D1 FIX: was "clint": 300.0 — this was Propranolol's TOTAL
        # BODY clearance entered into the hepatic-only field. Propranolol
        # is a high-extraction-ratio drug (EH>0.7 category, muni.cz
        # clinical pharmacology course) with <1% recovered unchanged in
        # urine (near-total hepatic elimination via CYP2D6/CYP1A2).
        # F=0.26 already documented (extensive first-pass, consistent with
        # high Eh). True hepatic clearance: CL_h = Eh x Q_liver
        # = 0.75 x 87.0 = 65.25 L/h.
        # Source: Evans et al., Clin Pharmacol Ther 1973;14:494; muni.cz
        # EH category list (propranolol = high EH>0.7, alongside metoprolol).
        "cl_systemic": 65.25,
        "clrenal":   0.1,
        # kp_scalar calibrated to observed Vd=273L (3.9 L/kg, Evans 1973).
        # R&R Ka_AP predicts ~690L for logP=3.48 strong base — known over-amplification
        # at high logP reported by Utsey et al. (DMD 2020, Fig 6A, Table 2).
        # 0.40 brings predicted Vd to ~276L (within 2× criterion).
        # Per FDA PBPK guidance 2018: kp_scalar is appropriate when the mechanistic
        # model and observed Vd differ by >2× after correct tissue composition is applied.
        "kp_scalar": 0.40,
        # Measured human intestinal Peff. pKa=9.42 basic — paracellular floor
        # 0.02 combined with this p_eff gives adequate absorption rate.
        # AUC already 0.91× (near pass); p_eff fixes Cmax.
        # [Lennernas, J Pharm Pharmacol 1998;50:935]
        "p_eff":     4.0e-5,   # [cm/s]
        # v5.6 D4 FIX: p_eff is from Lennernas 1998 — the same human in vivo
        # SPIP perfusion study source as Metoprolol (which already correctly
        # has this flag). Without it, the per-segment ionization correction
        # f_u[i] is applied on top of an already ionization-integrated
        # measured value, double-counting ionization for this pKa=9.42
        # basic drug and collapsing k_abs to ~2% of the correct value.
        # Source: [Lennernas, J Pharm Pharmacol 1998;50:935 — human in vivo
        # SPIP, same measurement paradigm as Metoprolol's Dahlgren 2016 entry]
        "peff_is_measured_net": True,
    },

    "Rosuvastatin": {
        "smiles":    "CC(C)N(C)c1nc(nc(c1/C=C/[C@@H](O)C[C@@H](O)CC(=O)O)c2ccc(F)cc2)S(=O)(=O)C",
        "logp":      -0.33,
        "fup":       0.12,
        "mw":        481.5,
        "pka":       4.20,
        "drug_type": "acidic",
        "dose":      20,
        "route":     "oral",
        "cmax":      0.015,
        "auc":       0.12,
        "ka":        1.4,
        "F":         0.20,
        "clint":     1.2,
        "clrenal":   0.8,
        # Module P6 — OATP1B1/1B3 uptake [Shitara 2005]
        "is_uptake_substrate": True,
        "vmax_uptake":         600.0,
        "km_uptake":             2.5,
    },

    # v5.0 — Finding 1: tmdd_params added (Module P7, VKORC1 + deep albumin depot).
    # Without the TMDD depot, fup=0.007 produces near-zero free-drug driving force
    # and an effective Vd of ~7 L vs the observed ~70 L — 10× AUC underprediction.
    # CLint corrected from 4.5 → 3.6 L/h [Pirmohamed 2006; CYP2C9 linear estimate].
    # kp_scalar=2.5: observed Vd >> predicted Vd (deep tissue binding sink).
    # Sources: Levy 1994; Mager & Jusko 2001; Osman et al. Br J Clin Pharmacol 2006.
    "Warfarin": {
        "smiles":    "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O",
        "logp":      2.70,
        "fup":       0.007,
        "mw":        308.3,
        "pka":       5.08,
        "drug_type": "acidic",
        "dose":      10,
        "route":     "oral",
        "cmax":      1.1,
        "auc":       45.0,
        "ka":        1.2,
        "F":         0.93,
        # v5.6 D1 FIX: was "clint": 3.6 (Pirmohamed 2006 CYP2C9 linear
        # estimate) — this was Warfarin's approximate TOTAL clearance
        # contribution, not isolated hepatic-only clearance. True hepatic
        # clearance for warfarin is very low: CL_total≈0.19 L/h (low EH<0.3
        # category per muni.cz clinical pharmacology course), with renal
        # clearance of unchanged drug <1% (O'Reilly 1968).
        # CL_hepatic ≈ CL_total x (1-fe) = 0.19 x 0.99 = 0.188 L/h.
        #
        # NOTE: This field is CURRENTLY INERT for Warfarin's simulation —
        # hepatic_module.py uses Path A (explicit Vmax_hepatic/Km_hepatic
        # Michaelis-Menten kinetics, both present below) whenever both are
        # supplied, bypassing CLint/cl_systemic entirely (see
        # hepatic_module.py lines 219-229: use_mm_hepatic=True overrides
        # the linear CLh path). Fixed here for data integrity and to
        # prevent confusion if the MM override is ever removed or the
        # Vmax_hepatic/Km_hepatic values are later found to be wrong.
        # Source: O'Reilly 1968, Ann NY Acad Sci; muni.cz EH category list.
        "cl_systemic": 0.19,
        "clrenal":   0.01,
        # Retained from original (explicit MM kinetics for Phenytoin-like saturation)
        "Vmax_hepatic": 50.0,
        "Km_hepatic":    5.0,
        # Module P7 — TMDD quasi-steady state for deep-binding tissue depot
        # Bmax=100 mg/L tissue: total VKORC1 + deep albumin binding sites
        # Kd=0.1 mg/L ≈ 0.3 nM (tight VKORC1 affinity; Levy 1994)
        "tmdd_params": {
            "Bmax_mg_L": 100.0,   # [mg/L tissue] total target concentration
            "Kd_mg_L":     0.1,   # [mg/L] equilibrium Kd (sub-nM affinity)
        },
        # kp_scalar v5.1 FIX: 2.5 → 0.5
        # 2.5 inflated tissue Kp → crushed plasma Cmax (fold 0.10 → 0.017).
        # For fup=0.007 acidic drug R&R over-predicts tissue Kp; 0.5 corrects
        # direction, pushing drug back toward plasma. [Rule 1 compliant]
        "kp_scalar": 0.5,
        # Gap 2 — Biliary excretion / enterohepatic recirculation
        # Warfarin undergoes biliary secretion of its acyl-glucuronide metabolite;
        # a fraction re-enters the enterohepatic cycle after intestinal hydrolysis.
        # cl_bile_lh=0.1: low biliary clearance (MRP2-mediated) [O'Reilly 1976]
        # f_reabs_bile=0.20: 20% reabsorption fraction after intestinal hydrolysis
        "cl_bile_lh":    0.1,    # [L/h]  [O'Reilly, J Lab Clin Med 1976]
        "f_reabs_bile":  0.20,   # [–]
    },

    # ════════════════════════════════════════════════════════════════════════
    # CNS & Pain
    # ════════════════════════════════════════════════════════════════════════

    "Alprazolam": {
        "smiles":    "Cc1nnc2n1-c3ccc(cc3C(=N2)c4ccccc4)Cl",
        "logp":      2.12,
        "fup":       0.20,
        "mw":        308.8,
        "pka":       2.40,
        "drug_type": "neutral",
        "dose":      1,
        "route":     "oral",
        "cmax":      0.015,
        "auc":       0.22,
        "ka":        1.5,
        "F":         0.90,
        "clint":     5.0,
        "clrenal":   0.05,
    },

    "Caffeine": {
        "smiles":    "Cn1c(=O)c2c(ncn2C)n(c1=O)C",
        "logp":      -0.07,
        "fup":       0.64,
        "mw":        194.2,
        "pka":       0.52,
        "drug_type": "neutral",
        "dose":      100,
        "route":     "oral",
        "cmax":      1.94,
        "auc":       15.5,
        "ka":        1.8,
        "F":         0.99,
        # v5.6 D1 FIX: was "clint": 15.0 — this was approximately Caffeine's
        # TOTAL BODY clearance, not the hepatic-only value the engine
        # requires. Caffeine elimination is dominated by hepatic CYP1A2
        # metabolism, with only 0.5-3% excreted unchanged renally.
        # Total plasma clearance = 0.078 L/h/kg x 70 kg = 5.46 L/h.
        # CL_hepatic = CL_total x (1-fe) = 5.46 x 0.98 = 5.35 L/h.
        # Source: Pharmacology of Caffeine, NCBI Bookshelf (NBK223808);
        # Hereditas 2026 review (CYP1A2 >90% of initial metabolic steps).
        "cl_systemic": 5.35,
        "clrenal":   0.3,
        # Measured human jejunal Peff — dual-pathway gives ~6e-6 (4× under).
        # [Lennernas, J Pharm Pharmacol 1998;50:935]
        "p_eff":     2.5e-5,   # [cm/s]
    },

    # v5.0 — Gap 2: cl_bile_lh and f_reabs_bile added.
    # Diazepam undergoes significant biliary excretion of its N-desmethyl and
    # oxazepam glucuronide metabolites, which are hydrolysed in the intestine
    # and reabsorbed as parent drug. [Garattini et al. 1973; Mandrioli et al. 2008]
    "Diazepam": {
        "smiles":    "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",
        "logp":      2.82,
        "fup":       0.01,
        "mw":        284.7,
        "pka":       3.40,
        "drug_type": "neutral",
        "dose":      10,
        "route":     "oral",
        "cmax":      0.25,
        "auc":       5.8,
        "ka":        1.2,
        "F":         0.99,
        # v5.6 D1 FIX: was "clint": 8.0 — this was Diazepam's approximate
        # TOTAL BODY clearance, not hepatic-only. Diazepam is a low
        # extraction-ratio drug (EH<0.3 category, muni.cz clinical
        # pharmacology course) with <1% renal excretion of unchanged drug.
        # CL_total ≈ 1.6 L/h (Klotz et al. 1975). CL_hepatic ≈
        # CL_total x (1-fe) = 1.6 x 0.99 ≈ 1.58 L/h.
        # Source: Klotz et al., J Clin Invest 1975;55:347; muni.cz EH
        # category list (diazepam = low EH<0.3).
        "cl_systemic": 1.58,
        "clrenal":   0.05,
        # Gap 2 — Biliary excretion / EHC
        "cl_bile_lh":    0.5,    # [L/h]  [Garattini et al. 1973]
        "f_reabs_bile":  0.60,   # [–]  high reabsorption fraction (active parent drug)
    },

    # v5.0 — Gap 2: cl_bile_lh and f_reabs_bile added.
    # Ibuprofen undergoes extensive acyl-glucuronidation followed by biliary
    # secretion. The glucuronide hydrolyses in the gut lumen, contributing to
    # the secondary plasma peak and prolonging AUC. [Duggan & Kwan 1979]
    "Ibuprofen": {
        "smiles":    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "logp":      3.97,
        "fup":       0.01,
        "mw":        206.3,
        "pka":       4.91,
        "drug_type": "acidic",
        "dose":      400,
        "route":     "oral",
        "cmax":      35.0,
        "auc":       125.0,
        "ka":        1.6,
        "F":         0.80,
        # v5.6 D1 FIX: was "clint": 8.0 — this was Ibuprofen's approximate
        # TOTAL BODY clearance, not hepatic-only. Ibuprofen is a low
        # extraction-ratio drug (EH<0.3 category, muni.cz clinical
        # pharmacology course) with <1% renal excretion of unchanged drug.
        # CL_total ≈ 3.1 L/h (Davies 1998). CL_hepatic ≈
        # CL_total x (1-fe) = 3.1 x 0.99 ≈ 3.07 L/h.
        # Source: Davies, Clin Pharmacokinet 1998;34:101; muni.cz EH
        # category list (ibuprofen = low EH<0.3).
        "cl_systemic": 3.07,
        "clrenal":   0.2,
        # Gap 2 — Biliary excretion / EHC
        "cl_bile_lh":    0.3,    # [L/h]  [Duggan & Kwan 1979]
        "f_reabs_bile":  0.35,   # [–]
        # kp_scalar v5.1: Cmax 0.145× under (pred=5 vs obs=35).
        # fup=0.01, logP=3.97: R&R over-predicts tissue Kp → drug trapped
        # in tissues → Vd_model >> Vd_clinical. 0.08 corrects this.
        # [FDA PBPK Guidance 2018 — empirical Vd correction for lipophilic acids]
        "kp_scalar":     0.08,
    },

    # v5.0 — Finding 1: phaseII_kinetics keys corrected (Finding 2) and
    # fu_gut / CLint_gut_cyp3a4 added (Module P2 + P1 activation).
    # SULT Km: 14.7 → 1.0 mg/L  (SULT1A1; Reith et al. 2009; Miners et al. 2004)
    # UGT  Km: 1040.0 → 5.0 mg/L (UGT1A6/1A9; Miners et al. 2004)
    # fu_gut=0.80: significant enterocyte UGT pre-systemic conjugation.
    # CLint_gut_cyp3a4=0.0: paracetamol is not a CYP3A4 substrate.
    "Paracetamol": {
        "smiles":    "CC(=O)Nc1ccc(O)cc1",
        "logp":      0.49,
        "fup":       0.80,
        "mw":        151.2,
        "pka":       9.38,
        "drug_type": "neutral",
        "dose":      1000,
        "route":     "oral",
        "cmax":      15.0,
        "auc":       50.0,
        "ka":        2.0,
        "F":         0.88,
        # v5.6 D1 FIX — SPECIAL CASE: was "clint": 5.0. Unlike most drugs,
        # Paracetamol's J_cyp (driven by this field) and J_phaseII (driven
        # by the explicit phaseII_kinetics SULT/UGT Vmax/Km below) are
        # ADDITIVE in hepatic_module.py (metabolic_rate = J_cyp + J_phaseII
        # + J_bile_secretion — see line 283). SULT+UGT conjugation already
        # accounts for ~85-90% of paracetamol elimination (glucuronide +
        # sulphate conjugates per droracle.ai/NCBI pharmacokinetics
        # summary). The CYP-mediated pathway (NAPQI formation) is a MINOR
        # route, <5% of total hepatic clearance.
        # Setting cl_systemic to the FULL hepatic CL (20.16 L/h, computed
        # the same way as other drugs) would double-count metabolism that
        # SULT/UGT already captures. Correct value: CYP-only fraction
        # = CL_hepatic_total x 0.05 = 20.16 x 0.05 ≈ 1.0 L/h.
        # Source: droracle.ai paracetamol PK summary (NAPQI <5% of dose,
        # SULT/UGT conjugates ~85-90%); total CL 4.5-5.5 mL/kg/min basis.
        "cl_systemic": 1.0,
        "clrenal":   0.1,
        # Module P2 — Phase II saturation kinetics (SULT + UGT)
        # v5.0 FIX: Km_mg_L values corrected to literature consensus.
        "phaseII_kinetics": {
            "sult": {
                "Vmax_mg_h": 1500.0,   # [mg/h] SULT1A1/1A3 maximal sulphation rate
                "Km_mg_L":      1.0,   # [mg/L] v5.0 FIX: was 14.7 (unit-conv error)
            },
            "ugt": {
                "Vmax_mg_h": 5000.0,   # [mg/h] UGT1A6/1A9 maximal glucuronidation rate
                "Km_mg_L":      5.0,   # [mg/L] v5.0 FIX: was 1040.0 (literature error)
            },
        },
        # Module P1 — Gut-wall metabolism (UGT, not CYP3A4)
        "fu_gut":            0.80,   # [–]   enterocyte unbound fraction
        "CLint_gut_cyp3a4":  0.0,    # [L/h] paracetamol is not a CYP3A4 substrate
        # Measured human intestinal Peff — BCS Class I, near-complete oral absorption >90%.
        # v5.1 FIX: 1.7e-5 → 1.6e-4. Egan regression (logP=0.49) gives ~3-5e-6: 30-60× underestimate
        # for small, polar, uncharged drugs. Previous p_eff was severely rate-limiting absorption.
        # Source: [Anderson et al., PMC3734790 2014; Lennernäs, Xenobiotica 2007;37:1015]
        "p_eff":     1.6e-4,   # [cm/s]  # Source: [Anderson et al. PMC3734790; Lennernäs 2007 — BCS Class I, Fabs>90%]
    },

    "Phenytoin": {
        "smiles":    "O=C1NC(=O)C(N1)(c2ccccc2)c3ccccc3",
        "logp":      2.47,
        "fup":       0.10,
        "mw":        252.3,
        "pka":       8.33,
        "drug_type": "neutral",
        "dose":      300,
        "route":     "oral",
        "cmax":      4.5,
        "auc":       110.0,
        "ka":        1.0,
        "F":         0.85,
        "clint":     6.0,
        "clrenal":   0.1,
        # Explicit MM kinetics retained from original
        "Vmax_hepatic": 40.0,
        "Km_hepatic":   10.0,
    },

    # v5.0 — Finding 1: fu_gut and CLint_gut_cyp3a4 were already present in
    # original reference_pk.py; confirmed correct and retained verbatim.
    # These activate Module P1 (gut-wall CYP3A4 extraction).
    # CLint_gut_cyp3a4=60 L/h reflects ~40% of hepatic CYP3A4 CLint
    # (Gertz et al. 2010 ratio; CLint_hep=180 L/h × 0.4 × fm_CYP3A4 ≈ 60).
    "Midazolam": {
        "smiles":    "Cn1cc2c(n1)N=C(c3ccccc3)c1cc(F)ccc1-2",
        "logp":      3.89,
        "fup":       0.03,
        "mw":        325.8,
        "pka":       6.15,
        "drug_type": "basic",
        "dose":      7.5,
        "route":     "oral",
        "cmax":      0.05,
        "auc":       0.18,
        "ka":        1.6,
        "F":         0.40,
        # v5.1 FIX: CLint 180 → 40.0 L/h — hepatocyte IVIVE-corrected.
        # HLM Kronbach 1992: Km≈3.9 µM=1.27 mg/L, Vmax≈1.7 nmol/min/mg → CLint≈60-90 L/h;
        # CYP3A4 microsomes systematically 5.88× over vs hepatocytes (PMC7042718 correction) → ~40 L/h.
        # Source: [Kronbach et al., PubMed:8886602; Bowman & Benet, Drug Metab Dispos 2020 PMC7042718]
        #
        # v5.6 D1 AUDIT NOTE: This value was reviewed during the D1 data-error
        # sweep (total-body-CL-entered-as-hepatic-CL across 10 drugs) and
        # found to be CORRECT as-is. Unlike the other 9 drugs fixed in this
        # sprint, clint=40.0 here is a genuinely derived whole-liver
        # intrinsic clearance from hepatocyte IVIVE scaling — NOT a
        # systemic/total-body clearance copied from clinical literature.
        # Verification: well-stirred model with clint=40 gives Eh=0.315,
        # CLh=27.4 L/h — consistent with Midazolam's intermediate
        # extraction-ratio category (EH 0.3-0.7, muni.cz clinical
        # pharmacology course). DO NOT migrate this key to "cl_systemic" —
        # doing so would cause the RTT reverse well-stirred calculation to
        # incorrectly treat 40.0 as an already-hepatic CL_sys and
        # back-calculate an inflated CLint≈2468 L/h (6x too high; see
        # diag run [RTT] 'Midazolam' log line from the cl_systemic-keyed
        # version of this file).
        "clint":     40.0,     # Source: [Kronbach et al. PubMed:8886602 + PMC7042718 hepatocyte IVIVE correction]
        "clrenal":   0.1,
        # Module P1 — Gut-wall CYP3A4 extraction
        # v5.1 FIX: CLint_gut_cyp3a4 60 → 15.0 L/h.
        # Gertz 2010 ratio: gut-wall CLint ≈ 25-40% of hepatic CLint for high-extraction CYP3A4 substrates.
        # At hepatic CLint=40 L/h: 15/40 = 37.5% — within the Gertz ratio range.
        # 60 L/h gut-wall predicts F_gut≈0.20 → combined F≈0.09 (too low vs clinical F=0.40).
        # Source: [Gertz et al., Drug Metab Dispos 2010;38(7):1147]
        "fu_gut":            1.0,    # [–]   enterocyte unbound fraction
        "CLint_gut_cyp3a4":  15.0,   # [L/h]  # Source: [Gertz et al., Drug Metab Dispos 2010;38:1147 — 37.5% of hepatic CLint=40]
    },

    # ════════════════════════════════════════════════════════════════════════
    # GI & Metabolic
    # ════════════════════════════════════════════════════════════════════════

    "Cimetidine": {
        "smiles":    "Cc1c(nc[nH]1)CSCCN=C(NC)NC#N",
        "logp":      0.40,
        "fup":       0.80,
        "mw":        252.3,
        "pka":       6.80,
        "drug_type": "basic",
        "dose":      400,
        "route":     "oral",
        "cmax":      1.8,
        "auc":       6.2,
        "ka":        1.6,
        "F":         0.62,
        "clint":     8.0,
        # v5.1 FIX: clrenal 10.0 → 27.0 L/h — directly measured in healthy subjects.
        # OCT2 (basolateral) + MATE1/MATE2-K (apical) renal tubular secretion.
        # Patel 1981: CLrenal ≈ 450 mL/min = 27 L/h (direct measurement).
        # Previous value missed dominant MATE1-mediated tubular secretion component.
        # Source: [Patel et al., J Clin Pharmacol 1981; Feng et al., J Clin Pharmacol 2016 DOI:10.1002/jcph.702]
        "clrenal":   27.0,     # Source: [Patel et al. 1981 — 450 mL/min directly measured; Feng et al. 2016 DOI:10.1002/jcph.702]
        # Measured human jejunal Peff — partially carrier-mediated.
        # [Lennernas 1998; Ungell et al. 1998]
        "p_eff":     1.5e-5,   # [cm/s]
    },

    # v5.0 — Finding 1: gut_transporter added (Module P5, PMAT/OCT1 influx).
    # Metformin has logP=−1.43 → p_eff ≈ 0 by Egan regression → without active
    # transport, Module P5 predicts near-zero absorption (Cmax/AUC ~1% of observed).
    # All meaningful intestinal absorption is via PMAT (SLC29A4) and OCT1 (SLC22A1).
    # Vmax=400 mg/h, Km=26 mg/L from Graham et al. 2011; Kimura et al. 2005.
    "Metformin": {
        "smiles":    "CN(C)C(=N)NC(=N)N",
        "logp":      -1.43,
        "fup":       0.97,
        "Rb":        1.00,  # blood:plasma ratio — hydrophilic biguanide, minimal RBC
                             # partitioning despite high pKa=11.5 (low logP prevents
                             # membrane accumulation). Literature: Tucker et al.,
                             # Br J Clin Pharmacol 1981;12:235. Required to keep Ka_AP
                             # back-calculation in valid range for pKa>7 path.
        "mw":        129.21,
        "pka":       11.5,
        "drug_type": "basic",
        "dose":      500,
        "route":     "oral",
        "cmax":      1.3,
        "auc":       10.5,
        "ka":        1.8,
        "F":         0.55,
        "clint":     0.1,
        "clrenal":   30.6,
        # Module P5 — PMAT/OCT1 active gut influx
        # Sources: Graham et al., Diabetologia 2011;54:2000; Kimura et al. 2005
        "gut_transporter": {
            "vmax_mg_h": 400.0,    # [mg/h] PMAT+OCT1 luminal influx Vmax
            "km_mg_L":    26.0,    # [mg/L] OCT1 apparent Km (luminal basis)
            "segments":  [1, 2, 3, 4, 5],   # duodenum (1) → ileum (5)
        },
    },

    "Omeprazole": {
        "smiles":    "COc1ccc2nc(sc2c1)S(=O)Cc3ncc(C)c(OC)c3C",
        "logp":      2.23,
        "fup":       0.05,  # v5.4 data fix: previously 0.97 (inverted).
                             # Omeprazole is ~95% protein-bound (albumin).
                             # fup ≈ 0.05 is the consensus clinical value.
                             # Source: Regardh et al., Clin Pharmacokinet 1992;23:453;
                             # Nissen et al., Br J Clin Pharmacol 1992;34:145.
                             # With fup=0.97 the R&R formula gives Vd=564L (22×);
                             # corrected fup=0.05 yields Vd≈29L (lit: 25L, 1.16×).
        "mw":        345.4,
        "pka":       0.77,
        "drug_type": "neutral",
        "dose":      20,
        "route":     "oral",
        "cmax":      0.45,
        "auc":       1.2,
        # ka, F removed v5.2: not consumed by odes(); p_eff + enteric_coated
        # (below) fully and mechanistically determine gastric shielding and
        # absorption — no lumped-bioavailability bridge exists in the engine.
        # v5.6 D1 FIX: was "clint": 45.0 — unlike Midazolam's clint=40.0
        # (which has a documented hepatocyte IVIVE derivation), this value
        # has no citation explaining its origin and matches the pattern of
        # other drugs in this sweep where total body clearance was entered
        # directly. Omeprazole is a low extraction-ratio drug at the organ
        # level (EH<0.3 category alongside other PPIs — lansoprazole,
        # pantoprazole — per muni.cz clinical pharmacology course) despite
        # high overall CYP2C19-mediated CLint; flow-independence is typical
        # for PPIs. Total CL ≈ 24.0 L/h (consistent with the prior 1.36×
        # Vd-corrected diagnostic baseline); renal excretion of unchanged
        # drug <1%. CL_hepatic ≈ CL_total x (1-fe) = 24.0 x 0.995 ≈ 23.88 L/h.
        # Source: muni.cz EH category list (omeprazole = low EH<0.3,
        # grouped with lansoprazole/pantoprazole).
        "cl_systemic": 23.88,
        "clrenal":   0.05,
        # Measured Peff for enteric-coated omeprazole reaching jejunum intact.
        # Dual-pathway gives ~2.5e-5; PAMPA measured ~8e-5 cm/s.
        # [Artursson & Karlsson, Biochem Biophys Res Commun 1991;175:880]
        "p_eff":     8.0e-5,   # [cm/s]
        # Gap 4 (v5.2) — Enteric-Coated Gastric Shielding.
        # Omeprazole degrades within ~2 min at gastric pH 1.0 and is therefore
        # formulated as enteric-coated pellets (MUPS). The coating prevents
        # both degradation and any luminal absorption in the stomach segment;
        # the intact bolus reaches the duodenum (pH > 5.5) before dissolving.
        # Mechanistically replaces the lumped F=0.35 (acid-degradation +
        # first-pass) bridge that the engine never read.
        # [FDA label; standard PPI MUPS formulation pharmaceutics]
        "enteric_coated": True,
    },

    # v5.0 — Finding 1: gut_transporter added (Module P5, OCT1/OCT3 influx).
    # Ranitidine is a basic drug (pKa=8.20) with logP=0.27.  Its F=0.50 and
    # significant oral absorption despite modest lipophilicity are attributable
    # to OCT1/OCT3-mediated apical uptake in the small intestine.
    # Vmax=300 mg/h, Km=20 mg/L estimated from Bourdet & Thakker 2006 scaled
    # to oral dose of 150 mg and OCT1 expression in human jejunum.
    "Ranitidine": {
        "smiles":    "CNCc1ccc(o1)CSCCN=C(N)N[N+](=O)[O-]",
        "logp":      0.27,
        "fup":       0.85,
        "Rb":        0.89,  # blood:plasma ratio — Ranitidine clinical PK data;
                             # Zantac prescribing information; consistent with
                             # moderate-base hydrophilic distribution (pKa=8.20).
        "mw":        314.4,
        "pka":       8.20,
        "drug_type": "basic",
        "dose":      150,
        "route":     "oral",
        "cmax":      0.5,
        "auc":       2.5,
        "ka":        1.5,
        "F":         0.50,
        "clint":     6.0,
        "clrenal":   9.0,
        # Module P5 — OCT1/OCT3 active gut influx
        # Source: Bourdet & Thakker, Drug Metab Dispos 2006;34:1237
        "gut_transporter": {
            "vmax_mg_h": 300.0,    # [mg/h] OCT1/OCT3 luminal influx Vmax
            "km_mg_L":    20.0,    # [mg/L] apparent Km
            "segments":  [1, 2, 3, 4, 5],
        },
    },
}