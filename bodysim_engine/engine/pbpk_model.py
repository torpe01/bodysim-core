"""
pbpk_model.py — Multi-compartment PBPK ODE model for BodySim.

Architecture (13 tissue compartments + gut lumen):
  Arterial blood → [Liver, Kidney, Brain, Heart, Muscle, Fat, Gut, Skin, Bone, Rest]
                              ↓ venous return
                          Venous blood → Lung → Arterial blood

Special routes:
  Gut → Portal vein → Liver  (first-pass metabolism)
  Liver, Kidney              (clearance organs)
  Gut lumen                  (oral absorption depot, amount in mg)

State vector y (14 elements):
  y[0]  = C_arterial  mg/L  arterial blood plasma
  y[1]  = C_venous    mg/L  venous blood plasma
  y[2]  = C_lung      mg/L  lung tissue
  y[3]  = C_liver     mg/L  liver tissue
  y[4]  = C_kidney    mg/L  kidney tissue
  y[5]  = C_brain     mg/L  brain tissue
  y[6]  = C_heart     mg/L  heart tissue
  y[7]  = C_muscle    mg/L  muscle tissue
  y[8]  = C_fat       mg/L  fat tissue
  y[9]  = C_gut       mg/L  gut tissue
  y[10] = C_skin      mg/L  skin tissue
  y[11] = C_bone      mg/L  bone tissue
  y[12] = C_rest      mg/L  rest (remaining tissues)
  y[13] = A_gut_lumen mg    drug amount in gut lumen (oral dosing depot)

All concentrations are total tissue drug concentrations (bound + unbound).
Unbound plasma concentration = C_arterial × fup.
"""

import numpy as np
from scipy.integrate import solve_ivp

# --- State vector indices ---
ART  = 0;  VEN  = 1;  LUNG = 2
LIV  = 3;  KID  = 4;  BRA  = 5
HRT  = 6;  MUS  = 7;  FAT  = 8
GUT  = 9;  SKN  = 10; BON  = 11
RST  = 12; GLU  = 13   # GLU = gut lumen (amount in mg)
N_STATES = 14

ORGAN_NAMES = [
    "arterial", "venous", "lung",
    "liver", "kidney", "brain",
    "heart", "muscle", "fat",
    "gut", "skin", "bone",
    "rest", "gut_lumen"
]

TISSUE_COMPARTMENTS = {
    LIV: "liver", KID: "kidney", BRA: "brain",
    HRT: "heart", MUS: "muscle", FAT: "fat",
    GUT: "gut",   SKN: "skin",   BON: "bone",
    RST: "rest",
}


class PBPKModel:
    """
    13-compartment whole-body PBPK model.

    Parameters
    ----------
    drug    : dict   drug profile from admet.build_drug_profile()
    volumes : dict   organ volumes (L) from physiology.scale_physiology()
    flows   : dict   organ blood flows (L/h) from physiology.scale_physiology()
    params  : dict   subject parameters (egfr, cyp activities, etc.)
    """

    def __init__(self, drug, volumes, flows, params):
        self.drug    = drug
        self.vol     = volumes
        self.flow    = flows
        self.params  = params
        self._validate()

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def _validate(self):
        required_kp = ["liver","kidney","brain","heart","muscle","fat",
                        "gut","skin","bone","lung","rest"]
        for k in required_kp:
            assert k in self.drug["kp"], f"Missing Kp for: {k}"
        assert 0 < self.drug["fup"] <= 1, "fup must be between 0 and 1"
        assert self.drug["CLint"] >= 0, "CLint must be non-negative"
        assert self.drug["CLrenal"] >= 0, "CLrenal must be non-negative"

    # ------------------------------------------------------------------
    # Hepatic clearance via well-stirred model
    # ------------------------------------------------------------------
    def _hepatic_clearance(self):
        """
        Well-stirred liver model:
            CLh = Q_liver × fup × CLint / (Q_liver + fup × CLint)

        Returns CLh in L/h (blood clearance, scaled by Rb).
        """
        Q   = self.flow["liver_hepatic"] + self.flow["liver_portal"]
        fup = self.drug["fup"]
        cl  = self.drug["CLint"]
        Rb  = self.drug["Rb"]
        # Hepatic extraction efficiency
        Eh  = (fup * cl / Rb) / (Q + fup * cl / Rb)
        Eh  = np.clip(Eh, 0.0, 0.99)
        CLh = Q * Eh
        return CLh

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def odes(self, t, y):
        """
        Compute dy/dt for the PBPK system.

        Parameters
        ----------
        t : float  current time (h)
        y : array  current state vector (length N_STATES)

        Returns
        -------
        dydt : array  derivatives (length N_STATES)
        """
        # Unpack state — clip negatives to 0 (mass conservation safety)
        y = np.maximum(y, 0.0)

        C_art  = y[ART];  C_ven = y[VEN];  C_lung = y[LUNG]
        C_liv  = y[LIV];  C_kid = y[KID];  C_bra  = y[BRA]
        C_hrt  = y[HRT];  C_mus = y[MUS];  C_fat  = y[FAT]
        C_gut  = y[GUT];  C_skn = y[SKN];  C_bon  = y[BON]
        C_rst  = y[RST];  A_glu = y[GLU]   # A_glu in mg

        # Shorthand aliases
        v  = self.vol
        q  = self.flow
        kp = self.drug["kp"]
        ka = self.drug["ka"]
        F  = self.drug["F"]
        fup= self.drug["fup"]
        Rb = self.drug["Rb"]

        # Flows used in this model
        Q_ha  = q["liver_hepatic"]   # hepatic artery
        Q_pv  = q["liver_portal"]    # portal vein (= gut flow)
        Q_liv = Q_ha + Q_pv          # total liver inflow
        Q_kid = q["kidney"]
        Q_bra = q["brain"]
        Q_hrt = q["heart"]
        Q_mus = q["muscle"]
        Q_fat = q["fat"]
        Q_gut = q["gut"]
        Q_skn = q["skin"]
        Q_bon = q["bone"]
        Q_rst = q["rest"]
        CO    = q["cardiac_output"]

        # --- Hepatic clearance (well-stirred model) ---
        CLh = self._hepatic_clearance()

        # --- Renal clearance ---
        # Removes drug from systemic circulation proportional to unbound plasma
        CLr = self.drug["CLrenal"]

        # ----------------------------------------------------------
        # dy/dt equations
        # ----------------------------------------------------------
        dydt = np.zeros(N_STATES)

        # [GLU] Gut lumen — first-order oral absorption
        # The absorbed fraction F is embedded; ka applies to the depot
        dydt[GLU] = -ka * A_glu

        # [GUT] Gut tissue — receives portal blood + absorbed drug
        # Absorbed drug enters gut tissue as: ka * A_glu * F / V_gut
        dydt[GUT] = (
            (Q_gut / v["gut"]) * (C_art - C_gut / kp["gut"])
            + ka * A_glu * F / v["gut"]
        )

        # [LIV] Liver — hepatic artery + portal input, clearance via CLh
        # Venous leaving liver = (C_liv / kp_liv) × Q_liv
        dydt[LIV] = (
            (Q_ha * C_art + Q_pv * (C_gut / kp["gut"]) - Q_liv * (C_liv / kp["liver"]))
            / v["liver"]
            - CLh * (C_liv / kp["liver"]) / v["liver"]
        )

        # [KID] Kidney — perfused tissue compartment (distribution only)
        # Renal EXCRETION is handled as a direct venous loss below
        dydt[KID] = (Q_kid / v["kidney"]) * (C_art - C_kid / kp["kidney"])

        # [BRA] Brain
        dydt[BRA] = (Q_bra / v["brain"]) * (C_art - C_bra / kp["brain"])

        # [HRT] Heart
        dydt[HRT] = (Q_hrt / v["heart"]) * (C_art - C_hrt / kp["heart"])

        # [MUS] Muscle
        dydt[MUS] = (Q_mus / v["muscle"]) * (C_art - C_mus / kp["muscle"])

        # [FAT] Fat
        dydt[FAT] = (Q_fat / v["fat"]) * (C_art - C_fat / kp["fat"])

        # [SKN] Skin
        dydt[SKN] = (Q_skn / v["skin"]) * (C_art - C_skn / kp["skin"])

        # [BON] Bone
        dydt[BON] = (Q_bon / v["bone"]) * (C_art - C_bon / kp["bone"])

        # [RST] Rest
        dydt[RST] = (Q_rst / v["rest"]) * (C_art - C_rst / kp["rest"])

        # [VEN] Venous blood — collects from all organ outflows except gut
        # (gut drains via portal to liver, not directly to venous)
        # Renal clearance already accounted in kidney ODE; venous gets the
        # remaining kidney outflow: Q_kid × C_kid/kp_kid
        venous_inflow = (
            Q_liv  * (C_liv / kp["liver"])
            + Q_kid * (C_kid / kp["kidney"])
            + Q_bra * (C_bra / kp["brain"])
            + Q_hrt * (C_hrt / kp["heart"])
            + Q_mus * (C_mus / kp["muscle"])
            + Q_fat * (C_fat / kp["fat"])
            + Q_skn * (C_skn / kp["skin"])
            + Q_bon * (C_bon / kp["bone"])
            + Q_rst * (C_rst / kp["rest"])
        )
        dydt[VEN] = (venous_inflow - CO * C_ven
                     - CLr * fup * C_ven      # renal excretion: loss from systemic venous blood
                    ) / v["venous_blood"]

        # [LUNG] Lung — in series between venous and arterial
        dydt[LUNG] = (CO * C_ven - CO * (C_lung / kp["lung"])) / v["lung"]

        # [ART] Arterial blood — receives oxygenated blood from lung
        dydt[ART] = (CO * (C_lung / kp["lung"]) - CO * C_art) / v["arterial_blood"]

        return dydt

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    def solve(self, dose_mg, route="oral", t_end_h=48.0, n_points=500):
        """
        Solve the PBPK ODE system for a single dose.

        Parameters
        ----------
        dose_mg  : float  administered dose in mg
        route    : str    'oral' or 'iv'
        t_end_h  : float  simulation end time in hours
        n_points : int    number of time points in output

        Returns
        -------
        result : dict with keys:
            't'            : np.array  time (h)
            'plasma'       : np.array  arterial plasma concentration (mg/L)
            'organs'       : dict      {organ_name: concentration array (mg/L)}
            'auc_plasma'   : float     AUC(0-t) for plasma (mg·h/L)
            'auc_organs'   : dict      AUC(0-t) for each organ (mg·h/L)
            'cmax_plasma'  : float     peak plasma concentration (mg/L)
            'tmax_plasma'  : float     time of peak plasma (h)
            'drug'         : dict      drug profile used
            'dose_mg'      : float
            'route'        : str
        """
        y0 = np.zeros(N_STATES)

        # Set initial conditions based on route
        if route == "oral":
            y0[GLU] = dose_mg                             # mg in gut lumen
        elif route == "iv":
            v_art = self.vol["arterial_blood"]
            y0[ART] = dose_mg / v_art / self.drug["Rb"]  # mg/L in blood
        else:
            raise ValueError(f"Unknown route: {route}. Use 'oral' or 'iv'.")

        # Add lag time for oral (shift t_eval)
        tlag = self.drug.get("tlag", 0.0) if route == "oral" else 0.0
        t_span = (0.0, t_end_h)
        t_eval = np.linspace(0.0, t_end_h, n_points)

        # Solve ODE
        sol = solve_ivp(
            fun=self.odes,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method="LSODA",        # stiff-capable solver
            rtol=1e-6,
            atol=1e-9,
            max_step=0.5,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        t = sol.t
        Y = sol.y   # shape (N_STATES, n_points)

        # Plasma = arterial compartment
        plasma = Y[ART]

        # Organ concentrations
        organs = {
            "liver":   Y[LIV],
            "kidney":  Y[KID],
            "brain":   Y[BRA],
            "heart":   Y[HRT],
            "muscle":  Y[MUS],
            "fat":     Y[FAT],
            "gut":     Y[GUT],
            "skin":    Y[SKN],
            "bone":    Y[BON],
            "lung":    Y[LUNG],
            "rest":    Y[RST],
        }

        # AUC using trapezoidal rule — scipy is robust across numpy versions
        from scipy.integrate import trapezoid as _trapz
        auc_plasma = float(_trapz(plasma, t))
        auc_organs = {k: float(_trapz(v, t)) for k, v in organs.items()}

        # Cmax and Tmax
        cmax_idx    = int(np.argmax(plasma))
        cmax_plasma = float(plasma[cmax_idx])
        tmax_plasma = float(t[cmax_idx])

        return {
            "t":           t,
            "plasma":      plasma,
            "organs":      organs,
            "auc_plasma":  auc_plasma,
            "auc_organs":  auc_organs,
            "cmax_plasma": cmax_plasma,
            "tmax_plasma": tmax_plasma,
            "drug":        self.drug,
            "dose_mg":     dose_mg,
            "route":       route,
        }
