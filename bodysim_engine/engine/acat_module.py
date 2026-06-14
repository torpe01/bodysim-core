"""
acat_module.py — ACAT (Advanced Compartmental Absorption and Transit) absorption module.
Extracted from pbpk_model.py v5.0. Contains all luminal transit, ionization, and
gut-wall metabolism logic for the 7-segment ACAT model.
"""

import numpy as np

# ── Module-level ACAT physiological constants ─────────────────────────────────
ACAT_TRANSIT_TIMES = np.array([0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 18.0])

ACAT_SA_FOLDING = np.array([1.0, 1.0, 10.0, 10.0, 8.0, 8.0, 2.0])
_ACAT_SA_NORM   = 10.0
ACAT_SA_FACTORS = ACAT_SA_FOLDING / _ACAT_SA_NORM

ACAT_SEGMENT_NAMES = ["stomach", "duodenum", "jejunum_1", "jejunum_2",
                      "ileum_1", "ileum_2", "cecum_colon"]
N_ACAT_SEGMENTS = 7

# Fallback luminal pH per segment when physiology.py is unavailable.
# Values represent mean fasted-state luminal pH:
#   Stomach ~2.0, Duodenum ~6.0, Jejunum 6.5, Ileum 7.4, Colon ~5.9
ACAT_PH_DEFAULT = np.array([2.0, 6.0, 6.5, 6.5, 7.4, 7.4, 5.9])


class ACATAbsorptionModule:
    """
    Handles the 7-segment ACAT oral absorption model.

    Responsibilities:
      - _build_acat_params: pre-compute all per-segment parameters once at init.
      - calculate_gut_flux:  process the ACAT luminal loop inside odes() and
                             return (total_abs_flux, dydt_lumen_array).
    """

    def _build_acat_params(self, drug: dict, flow: dict) -> dict:
        """
        Pre-compute ACAT model parameters.

        7-segment model: Stomach, Duodenum, Jejunum×2, Ileum×2, Cecum/Colon

        Returns
        -------
        dict with:
          kt                 : np.array(7,) transit rate constants [h⁻¹]
          f_u                : np.array(7,) un-ionized fraction per segment
          k_abs              : np.array(7,) first-order absorption rate constants [h⁻¹]
          F_gut_scalar       : float  gut-wall CYP3A4/UGT availability fraction (0–1)
          Vmax_gut_active    : float  [mg/h] active SLC influx Vmax (0.0 → disabled)
          Km_gut_active      : float  [mg/L] active SLC influx Km
          gut_active_segments: list   ACAT segment indices with transporter expression

        Module P1 — Gut-Wall Metabolism (Yang et al. 2007; Gertz et al. 2010):
          F_gut = Q_gut / (Q_gut + fu_gut × CLint_gut_cyp3a4)

        Module P5 — Active Gut Influx (PMAT/OCT1/PEPT1):
          J_active_i = (Vmax × C_lumen_i) / (Km + C_lumen_i)  [mg/h]
          Active segments default [1,2,3,4,5] (duodenum–ileum; SLC-rich).

        Gap 1 (v5.0) — Zwitterion Two-pKa Ionization:
          f_neutral = f_acid_neutral × f_base_neutral
          Requires pka dict with \"acid\" and \"base\" keys.
          Falls back to paracellular floor 0.02 for legacy scalar pka.

        Gap 3 (v5.2) — Regional Absorption Windows:
          drug["absorption_segments"]: optional list of ACAT segment indices
          (0-6) where permeability-driven absorption (k_abs) is permitted.
          Segments outside this list have k_abs forced to 0.0, reflecting a
          true regional Peff collapse (e.g. furosemide ileal/colonic Peff
          falls to ~0.01x its jejunal value due to complete ionization at
          pH 7.4; Dahan et al. 2020 PMC7761534).  This is APPLIED AFTER the
          existing p_eff x P_EFF_SCALE x fold_raw x f_u computation — purely
          multiplicative gating.  Absent key -> all 7 segments remain active
          (= list(range(7))), so existing drugs are numerically unchanged.

        Gap 4 (v5.2) — Enteric-Coated Gastric Shielding:
          drug["enteric_coated"]: optional bool (default False).  When True,
          k_abs[0] (stomach) is forced to 0.0 — the intact coating prevents
          any luminal absorption in the acidic gastric segment.  The dose
          depot is also re-routed (see calculate_gut_flux) to empty directly
          into LUMEN[1] (duodenum) rather than LUMEN[0] (stomach), modeling
          coating dissolution at duodenal pH (>5.5) before drug release
          (PPI MUPS formulation pharmaceutics).
        """
        # ── Transit rate constants ─────────────────────────────────────────
        kt = 1.0 / ACAT_TRANSIT_TIMES

        # ── Drug ionization parameters ─────────────────────────────────────
        pka       = drug.get("pka",       None)
        drug_type = drug.get("drug_type", "neutral")

        # ── Geometric scaling constant ─────────────────────────────────────
        # P_EFF_SCALE = (2 / r_gut) × 3600 = 4800 h⁻¹/(cm/s); r_gut = 1.5 cm
        # (Amidon et al. Pharm Res 1995; Yu et al. J Pharm Sci 1999)
        P_EFF_SCALE = 4800.0

        # ── Effective permeability ─────────────────────────────────────────────────
        # Use measured p_eff if available (passed via reference_pk "p_eff" key).
        # Otherwise estimate via dual-pathway model:
        #
        #   Transcellular: Egan logP regression (Egan et al. J Med Chem 2000)
        #     p_trans = 10^(0.4×logP − 5.5)  [cm/s]
        #     Good for logP > 1.5; under-estimates hydrophilic compounds.
        #
        #   Paracellular:  MW + H-bond donor corrected (Winiwarter et al.
        #     J Med Chem 1998; Sun et al. Pharm Res 2002)
        #     p_para = 1.5e-5 × exp(−0.010×max(0, MW−100)) × 0.85^HBD  [cm/s]
        #     Dominant for small, polar molecules (MW<300, HBD<3).
        #
        #   Combined: p_eff = sqrt(p_trans^2 + p_para^2)
        #     Geometric combination avoids double-counting the two pathways.
        #     No ionisation correction here — the per-segment f_u loop below
        #     already accounts for pH-dependent neutral fraction.
        logp  = drug.get("logp",  0.0)
        mw    = float(drug.get("mw",   300.0))
        hbd   = int(drug.get("hbd",   0))
        p_eff = drug.get("p_eff", None)
        if p_eff is None:
            p_trans = float(np.clip(10.0 ** (0.4 * logp - 5.5), 1e-8, 1e-3))
            p_para  = float(np.clip(
                1.5e-5 * np.exp(-0.010 * max(0.0, mw - 100.0)) * (0.85 ** hbd),
                1e-9, 1e-4,
            ))
            p_eff = float(np.clip(np.sqrt(p_trans**2 + p_para**2), 1e-8, 5e-4))

        # ── Per-segment pH array ───────────────────────────────────────────
        try:
            from physiology import ACAT_PH as _PHYS_PH
            seg_ph_arr = np.asarray(_PHYS_PH, dtype=float)
        except Exception:
            seg_ph_arr = ACAT_PH_DEFAULT.copy()

        # ── Raw SA folding multipliers (absolute, NOT normalized) ──────────
        # Absolute values preserve the 10× jejunal amplification over stomach.
        try:
            from physiology import ACAT_SA_FACTORS as _PHYS_SA, ACAT_SEGMENT_NAMES as _PHYS_NAMES
            fold_raw = np.array(
                [_PHYS_SA[_PHYS_NAMES.index(s)] for s in ACAT_SEGMENT_NAMES],
                dtype=float,
            ) * _ACAT_SA_NORM
        except Exception:
            fold_raw = ACAT_SA_FOLDING.copy()

        # ── Paracellular / microclimate permeability floor ─────────────────
        # Prevents f_u collapsing to zero for fully ionized drugs.
        # Represents measured paracellular permeability for ions (MW < 500 Da).
        # (Pade & Stavchansky 1998; Daniel & Kottra 2004; Lennernas 1998)
        paracellular_floor = 0.02

        f_u   = np.zeros(N_ACAT_SEGMENTS)
        k_abs = np.zeros(N_ACAT_SEGMENTS)

        for i in range(N_ACAT_SEGMENTS):
            seg_ph = float(seg_ph_arr[i])

            # Henderson-Hasselbalch ionization at anatomically correct seg_ph
            if pka is None or drug_type == "neutral":
                f_u[i] = 1.0

            elif drug_type == "acidic":
                # HA ⇌ H⁺ + A⁻: neutral HA dominates at pH << pKa
                raw_fu = 1.0 / (1.0 + 10.0 ** (seg_ph - pka))
                f_u[i] = float(np.clip(max(raw_fu, paracellular_floor), 0.0, 1.0))

            elif drug_type == "basic":
                # BH⁺ ⇌ B + H⁺: neutral B dominates at pH >> pKa
                raw_fu = 1.0 / (1.0 + 10.0 ** (pka - seg_ph))
                f_u[i] = float(np.clip(max(raw_fu, paracellular_floor), 0.0, 1.0))

            elif drug_type == "zwitterion":
                # Gap 1 (v5.0): Two-pKa model — BOTH groups must be simultaneously
                # neutral for membrane permeation.
                # pka must be {"acid": float, "base": float}
                if isinstance(pka, dict) and "acid" in pka and "base" in pka:
                    pKa_acid = float(pka["acid"])
                    pKa_base = float(pka["base"])
                    f_acid_neutral = 1.0 / (1.0 + 10.0 ** (seg_ph - pKa_acid))
                    f_base_neutral = 1.0 / (1.0 + 10.0 ** (pKa_base - seg_ph))
                    f_neutral_zw   = f_acid_neutral * f_base_neutral
                    f_u[i] = float(np.clip(
                        max(f_neutral_zw, paracellular_floor), 0.0, 1.0
                    ))
                else:
                    # Legacy scalar-pka zwitterion: paracellular floor only.
                    f_u[i] = paracellular_floor

            else:
                f_u[i] = 1.0

            # Absorption rate constant: geometric × anatomical × ionization
            # k_abs[i] [h⁻¹] = p_eff [cm/s] × P_EFF_SCALE [h⁻¹/(cm/s)]
            #                   × fold_raw[i] [–] × f_u[i] [–]
            k_abs[i] = float(p_eff * P_EFF_SCALE * fold_raw[i] * f_u[i])

        # ── Gap 3 (v5.2): Regional Absorption Window Gating ──────────────
        # Restrict permeability-driven absorption to the segments where the
        # drug has measurable Peff (e.g. furosemide: proximal SI only, per
        # Dahan et al. 2020 — Peff collapses ~100x in the ileum due to
        # complete ionization at pH 7.4).  Absent key -> no restriction.
        _absorption_segments = drug.get("absorption_segments", None)
        if _absorption_segments is not None:
            _allowed = set(int(s) for s in _absorption_segments)
            for i in range(N_ACAT_SEGMENTS):
                if i not in _allowed:
                    k_abs[i] = 0.0

        # ── Gap 4 (v5.2): Enteric-Coated Gastric Shielding ───────────────
        # An intact enteric coating prevents any luminal absorption while
        # the dose resides in the acidic stomach segment (segment 0).
        # The coating dissolves only after the bolus reaches the duodenum
        # (pH > 5.5) — see calculate_gut_flux for the corresponding
        # dose-depot routing change.
        enteric_coated = bool(drug.get("enteric_coated", False))
        if enteric_coated:
            k_abs[0] = 0.0

        # ── Module P1: Gut-Wall Metabolism Availability Fraction ──────────
        # F_gut = Q_gut / (Q_gut + fu_gut × CLint_gut_cyp3a4)
        # Default: F_gut_scalar = 1.0 (no gut-wall extraction).
        fu_gut           = float(drug.get("fu_gut",           1.0))
        CLint_gut_cyp3a4 = float(drug.get("CLint_gut_cyp3a4", 0.0))
        Q_gut_flow       = float(flow.get("gut",              0.0))

        _gut_denom = Q_gut_flow + fu_gut * CLint_gut_cyp3a4
        if _gut_denom > 1e-12:
            F_gut_scalar = float(Q_gut_flow / _gut_denom)
        else:
            F_gut_scalar = 1.0
        F_gut_scalar = float(np.clip(F_gut_scalar, 0.0, 1.0))

        # ── Module P5: Active Gut Influx (PMAT/OCT1/PEPT1) ───────────────
        # gut_transporter = {"vmax_mg_h": float, "km_mg_L": float, "segments": list}
        # Absent key → Vmax_gut_active = 0.0 → active branch disabled.
        _gt_raw         = drug.get("gut_transporter", {})
        Vmax_gut_active = float(_gt_raw.get("vmax_mg_h", 0.0))
        Km_gut_active   = float(_gt_raw.get("km_mg_L",   1.0))
        if Km_gut_active < 1e-9:
            Km_gut_active = 1e-9

        # Default active segments: duodenum(1) through ileum(5).
        # Stomach(0) and colon(6) excluded (low SLC expression).
        _default_active_segs = [1, 2, 3, 4, 5]
        gut_active_segments  = list(_gt_raw.get("segments", _default_active_segs))

        return {
            "kt":                  kt,
            "f_u":                 f_u,
            "k_abs":               k_abs,
            "F_gut_scalar":        F_gut_scalar,
            "Vmax_gut_active":     Vmax_gut_active,
            "Km_gut_active":       Km_gut_active,
            "gut_active_segments": gut_active_segments,
            "enteric_coated":      enteric_coated,
        }

    def calculate_gut_flux(
        self,
        y:           np.ndarray,
        drug:        dict,
        acat_params: dict,
        ph_profiles: dict,
    ):
        """
        Process the ACAT luminal transit loop for one ODE evaluation.

        Parameters
        ----------
        y            : full state vector [mg/L or mg depending on compartment]
        drug         : drug profile dict
        acat_params  : pre-computed dict from _build_acat_params()
        ph_profiles  : dict with keys used by the caller:
                         "A_dose_depot"  [mg]   pre-gastric dissolved dose
                         "A_glu_eff"     [mg]   P-gp efflux depot
                         "M_lumen"       array  luminal segment masses [mg]
                         "A_bile"        [mg]   bile pool mass
                         "ka_reabs"      [h⁻¹]  re-absorption rate from efflux depot
                         "LUMEN_BASE"    int    first lumen index in y
                         "BILE_IDX"      int    bile pool index in y

        Returns
        -------
        total_abs_flux : float  net portal absorption flux [mg/h]
        dydt_lumen     : np.array(7,)  derivative for each luminal segment [mg/h]
        dydt_dose_depot: float  derivative for the dose depot state [mg/h]
        dydt_bile      : float  derivative for the bile pool state [mg/h]

        All arithmetic is IDENTICAL to the original monolithic odes() loop.
        """
        # ── Unpack convenience references ─────────────────────────────────
        ac       = acat_params
        kt_arr   = ac["kt"]
        k_abs_arr= ac["k_abs"]

        A_dose_depot = ph_profiles["A_dose_depot"]
        A_glu_eff    = ph_profiles["A_glu_eff"]
        M_lumen      = ph_profiles["M_lumen"]
        A_bile       = ph_profiles["A_bile"]
        ka_reabs     = ph_profiles["ka_reabs"]

        C_tissue_free = ph_profiles["C_tissue_free"]   # needed for bile secretion

        # ── Module P5 active influx pre-fetch ─────────────────────────────
        Vmax_gut_active     = ac["Vmax_gut_active"]
        Km_gut_active       = ac["Km_gut_active"]
        gut_active_segments = ac["gut_active_segments"]

        # Nominal luminal volume per absorptive segment:
        #   250 mL fasted small-intestinal fluid / 6 absorptive segments
        #   (Schiller et al. Aliment Pharmacol Ther 2005)
        V_seg = 0.250 / 6   # ≈ 0.04167 L

        # ── Gap 2 (v5.0): Biliary reabsorption pre-computation ────────────
        # Computed before the loop so J_bile_reabs can be injected into
        # LUMEN[1] (duodenum) as a source term.
        _cl_bile_pre      = float(drug.get("cl_bile_lh",     0.0))
        _k_bile_empty_pre = float(drug.get("k_bile_empty_h", 0.05))
        _f_reabs_pre      = float(drug.get("f_reabs_bile",   0.0))
        _J_bile_emp_pre   = _k_bile_empty_pre * A_bile
        J_bile_reabs      = _f_reabs_pre * _J_bile_emp_pre    # [mg/h] → LUMEN[1]

        # ── Gap 4 (v5.2): Enteric-coated dose-depot routing ───────────────
        # Intact coating survives the stomach; the bolus empties directly
        # into the duodenum (LUMEN[1]) once gastric residence time elapses,
        # rather than into the stomach lumen (LUMEN[0]).  kt_arr[0] (4 h⁻¹)
        # is retained as the physiological gastric-residence transit rate —
        # only the destination segment changes.
        enteric_coated = ac.get("enteric_coated", False)

        # ── Main ACAT loop ─────────────────────────────────────────────────
        total_abs_flux = 0.0
        dydt_lumen     = np.zeros(N_ACAT_SEGMENTS)

        for i in range(N_ACAT_SEGMENTS):
            # Incoming transit mass flux [mg/h]
            if i == 0:
                if enteric_coated:
                    # Stomach receives no dose-depot inflow; only carries
                    # whatever transits in from upstream (none, by
                    # construction) plus any P-gp-reabsorbed efflux.
                    in_transit = ka_reabs * A_glu_eff
                else:
                    # Stomach: physiological gastric emptying (kt[0] = 4 h⁻¹).
                    # P-gp re-absorbed efflux re-enters here as well.
                    in_transit = kt_arr[0] * A_dose_depot + ka_reabs * A_glu_eff
            elif i == 1 and enteric_coated:
                # Duodenum receives the full dose-depot bolus directly
                # (coating dissolves at duodenal pH), plus normal upstream
                # transit from the stomach segment.
                in_transit = kt_arr[0] * A_dose_depot + kt_arr[i - 1] * M_lumen[i - 1]
            else:
                in_transit = kt_arr[i - 1] * M_lumen[i - 1]

            out_transit = kt_arr[i]     * M_lumen[i]   # [mg/h] → next segment / faeces
            j_abs_i     = k_abs_arr[i]  * M_lumen[i]   # [mg/h] → enterocyte (passive)

            # ── Module P5: Saturable Active Influx ────────────────────────
            # J_active_i = (Vmax × C_lumen_i) / (Km + C_lumen_i)  [mg/h]
            # C_lumen_i  = M_lumen[i] / V_seg  [mg/L]
            # NOT multiplied by F_gut_scalar (SLC substrates bypass CYP3A4).
            J_active_i = 0.0
            if Vmax_gut_active > 0.0 and i in gut_active_segments:
                M_i = max(0.0, M_lumen[i])
                if M_i > 0.0:
                    C_lumen_i  = M_i / V_seg
                    J_active_i = (Vmax_gut_active * C_lumen_i) \
                                 / (Km_gut_active  + C_lumen_i)

            # ── Biliary reabsorption source (duodenum only) ────────────────
            # J_bile_reabs added to LUMEN[1] (ampulla of Vater drains to duodenum).
            _bile_src_i = J_bile_reabs if (i == 1) else 0.0

            # ── Lumen ODE assembly ─────────────────────────────────────────
            dydt_lumen[i] = (in_transit - out_transit - j_abs_i
                             - J_active_i + _bile_src_i)

            # ── Module P1: Apply F_gut scalar to passive absorption only ──
            # Active (SLC) flux is credited at full value; passive flux is
            # scaled by F_gut_scalar to account for enterocyte CYP3A4/UGT loss.
            total_abs_flux += (j_abs_i * ac["F_gut_scalar"]) + J_active_i

        # ── Dose depot ODE ────────────────────────────────────────────────
        # Drains at physiological gastric emptying rate kt_arr[0] = 4 h⁻¹.
        dydt_dose_depot = -kt_arr[0] * A_dose_depot

        # ── Bile pool ODE (Gap 2, v5.0) ───────────────────────────────────
        # dA_bile/dt = J_bile_secretion − J_bile_emptying
        # J_bile_secretion computed here using the full drug params.
        cl_bile      = float(drug.get("cl_bile_lh",     0.0))
        k_bile_empty = float(drug.get("k_bile_empty_h", 0.05))
        J_bile_secretion = cl_bile * C_tissue_free      # [L/h × mg/L = mg/h]
        J_bile_emptying  = k_bile_empty * A_bile        # [h⁻¹ × mg  = mg/h]
        dydt_bile = J_bile_secretion - J_bile_emptying  # [mg/h]

        return total_abs_flux, dydt_lumen, dydt_dose_depot, dydt_bile, J_bile_secretion