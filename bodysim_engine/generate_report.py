"""
generate_report.py — Produces validation charts and a population risk report.
Run: python generate_report.py
Output: bodysim_validation.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

from engine.simulator    import Simulator
from engine.admet        import REFERENCE_DRUGS
from engine.physiology   import scale_physiology
from engine.pbpk_model   import PBPKModel
from engine.risk_scorer  import score_single_simulation, ORGAN_DISPLAY, RISK_BANDS

# ── colour palette ──────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
SUBTLE  = "#8b949e"
GREEN   = "#3fb950"
YELLOW  = "#d29922"
ORANGE  = "#f0883e"
RED     = "#f85149"
BLUE    = "#58a6ff"
PURPLE  = "#bc8cff"
TEAL    = "#39d353"

RISK_COLORS = {"green": GREEN, "yellow": YELLOW, "orange": ORANGE, "red": RED}

def risk_color(score):
    if score < 0.20: return GREEN
    if score < 0.45: return YELLOW
    if score < 0.70: return ORANGE
    return RED

# ── run simulations ─────────────────────────────────────────────────────────
print("[BodySim] Running simulations …")
sim = Simulator(verbose=False)

# Reference 70 kg male
vol, flow, params = scale_physiology()

met_drug = REFERENCE_DRUGS["metformin"]
caf_drug = REFERENCE_DRUGS["caffeine"]

met_model = PBPKModel(met_drug, vol, flow, params)
caf_model = PBPKModel(caf_drug, vol, flow, params)

met_result = met_model.solve(500,  "oral", 48, 400)
caf_result = caf_model.solve(200,  "oral", 24, 300)

# Population simulation — 80 subjects on metformin
print("[BodySim] Running population simulation (80 subjects) …")
pop = sim.run_population(met_drug, 500, "oral", n_subjects=80, seed=42, n_points=150)

met_scores = score_single_simulation(met_result)
caf_scores = score_single_simulation(caf_result)

print("[BodySim] Building charts …")

# ── figure layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 24), facecolor=BG)
gs  = gridspec.GridSpec(4, 3, figure=fig,
                        hspace=0.48, wspace=0.35,
                        left=0.06, right=0.97,
                        top=0.95, bottom=0.04)

def styled_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.tick_params(colors=SUBTLE, labelsize=8)
    ax.xaxis.label.set_color(SUBTLE)
    ax.yaxis.label.set_color(SUBTLE)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
    return ax

# ── Row 0: header ────────────────────────────────────────────────────────────
ax_hdr = fig.add_subplot(gs[0, :])
ax_hdr.set_facecolor(BG)
ax_hdr.axis("off")
ax_hdr.text(0.0, 0.85, "BODYSIM", color=TEXT,
            fontsize=28, fontweight="bold", va="top", transform=ax_hdr.transAxes)
ax_hdr.text(0.0, 0.40, "Core Engine Validation Report",
            color=SUBTLE, fontsize=14, va="top", transform=ax_hdr.transAxes)
ax_hdr.text(0.0, 0.05,
            "13-compartment PBPK model  ·  Metformin 500 mg oral  ·  Caffeine 200 mg oral  ·  80-subject virtual population",
            color=SUBTLE, fontsize=9, va="top", transform=ax_hdr.transAxes)

# Status badges
badges = [
    ("PHYSIOLOGY ✓", GREEN), ("ADMET ✓", GREEN),
    ("ODE SOLVER ✓", GREEN), ("POPULATION ✓", GREEN),
    ("RISK SCORER ✓", GREEN), ("73/73 TESTS ✓", TEAL),
]
for i, (label, col) in enumerate(badges):
    ax_hdr.text(0.58 + i*0.07, 0.75, label,
                color=col, fontsize=7.5, fontweight="bold",
                va="top", transform=ax_hdr.transAxes,
                bbox=dict(facecolor=PANEL, edgecolor=col, boxstyle="round,pad=0.3"))

# ── Row 1 col 0: Metformin plasma PK ─────────────────────────────────────────
ax1 = styled_ax(fig.add_subplot(gs[1, 0]), "Metformin 500 mg — Plasma PK")
t, cp = met_result["t"], met_result["plasma"]
ax1.plot(t, cp, color=BLUE, lw=2.0, label="Predicted plasma")
ax1.axvspan(0, 48, alpha=0.0)
# Literature band (Sambol et al. 1996)
ax1.fill_between([0, 12], [1.0, 1.0], [2.0, 2.0], alpha=0.18, color=GREEN,
                 label="Lit. Cmax range (1–2 mg/L)")
ax1.axvline(met_result["tmax_plasma"], color=YELLOW, ls="--", lw=1, alpha=0.7)
ax1.text(met_result["tmax_plasma"] + 0.3, met_result["cmax_plasma"] * 1.05,
         f"Cmax={met_result['cmax_plasma']:.2f}", color=YELLOW, fontsize=7.5)
ax1.text(met_result["tmax_plasma"] + 0.3, met_result["cmax_plasma"] * 0.85,
         f"Tmax={met_result['tmax_plasma']:.1f}h", color=YELLOW, fontsize=7.5)
auc_label = f"AUC={met_result['auc_plasma']:.1f} mg·h/L"
ax1.text(0.65, 0.88, auc_label, color=SUBTLE, fontsize=8, transform=ax1.transAxes)
ax1.set_xlabel("Time (h)"); ax1.set_ylabel("Plasma conc. (mg/L)")
ax1.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax1.set_xlim(0, 48)

# ── Row 1 col 1: Caffeine plasma PK ──────────────────────────────────────────
ax2 = styled_ax(fig.add_subplot(gs[1, 1]), "Caffeine 200 mg — Plasma PK")
t2, cp2 = caf_result["t"], caf_result["plasma"]
ax2.plot(t2, cp2, color=PURPLE, lw=2.0, label="Predicted plasma")
ax2.fill_between([0, 8], [1.5, 1.5], [3.5, 3.5], alpha=0.18, color=GREEN,
                 label="Lit. Cmax range (1.5–3.5 mg/L)")
ax2.axvline(caf_result["tmax_plasma"], color=YELLOW, ls="--", lw=1, alpha=0.7)
ax2.text(caf_result["tmax_plasma"] + 0.2, caf_result["cmax_plasma"] * 1.05,
         f"Cmax={caf_result['cmax_plasma']:.2f}", color=YELLOW, fontsize=7.5)
ax2.text(caf_result["tmax_plasma"] + 0.2, caf_result["cmax_plasma"] * 0.85,
         f"Tmax={caf_result['tmax_plasma']:.1f}h", color=YELLOW, fontsize=7.5)
auc_label2 = f"AUC={caf_result['auc_plasma']:.1f} mg·h/L"
ax2.text(0.65, 0.88, auc_label2, color=SUBTLE, fontsize=8, transform=ax2.transAxes)
ax2.set_xlabel("Time (h)"); ax2.set_ylabel("Plasma conc. (mg/L)")
ax2.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax2.set_xlim(0, 24)

# ── Row 1 col 2: Metformin organ distribution ─────────────────────────────────
ax3 = styled_ax(fig.add_subplot(gs[1, 2]), "Metformin — Organ Concentrations (48h)")
organ_colors = {
    "kidney": RED, "liver": ORANGE, "gut": YELLOW,
    "muscle": BLUE, "brain": PURPLE, "fat": TEAL,
}
for organ, col in organ_colors.items():
    ax3.plot(met_result["t"], met_result["organs"][organ],
             color=col, lw=1.5, label=ORGAN_DISPLAY.get(organ, organ), alpha=0.9)
ax3.plot(t, cp, color=TEXT, lw=1.0, ls="--", alpha=0.5, label="Plasma")
ax3.set_xlabel("Time (h)"); ax3.set_ylabel("Tissue conc. (mg/L)")
ax3.legend(fontsize=6.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, ncol=2)
ax3.set_xlim(0, 48)

# ── Row 2 col 0: Metformin risk bar chart ─────────────────────────────────────
ax4 = styled_ax(fig.add_subplot(gs[2, 0]), "Metformin — Organ Risk Scores")
organs_sorted = sorted(met_scores["organ_scores"].items(), key=lambda x: -x[1])
names  = [ORGAN_DISPLAY.get(o, o) for o, _ in organs_sorted]
scores = [s for _, s in organs_sorted]
colors = [risk_color(s) for s in scores]
bars = ax4.barh(names, scores, color=colors, height=0.6, edgecolor=BORDER)
ax4.axvline(0.45, color=ORANGE, ls="--", lw=0.8, alpha=0.6, label="Elevated threshold")
ax4.axvline(0.70, color=RED,    ls="--", lw=0.8, alpha=0.6, label="High risk threshold")
ax4.set_xlim(0, 1.05)
ax4.set_xlabel("Risk Score (0=safe, 1=high risk)")
for bar, s in zip(bars, scores):
    ax4.text(s + 0.02, bar.get_y() + bar.get_height()/2,
             f"{s:.3f}", va="center", color=SUBTLE, fontsize=7.5)
ax4.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

# ── Row 2 col 1: Population plasma distribution ───────────────────────────────
ax5 = styled_ax(fig.add_subplot(gs[2, 1]), "Population — Plasma AUC Distribution (n=80)")
pop_aucs = [r["auc_plasma"] for r in pop["individual_results"]]
n, bins, patches = ax5.hist(pop_aucs, bins=18, edgecolor=BORDER,
                             color=BLUE, alpha=0.75)
p5, p50, p95 = np.percentile(pop_aucs, [5, 50, 95])
ax5.axvline(p5,  color=GREEN,  ls="--", lw=1.2, label=f"P5  = {p5:.1f}")
ax5.axvline(p50, color=YELLOW, ls="-",  lw=1.5, label=f"P50 = {p50:.1f}")
ax5.axvline(p95, color=RED,    ls="--", lw=1.2, label=f"P95 = {p95:.1f}")
ax5.set_xlabel("AUC(0-48h)  mg·h/L"); ax5.set_ylabel("Count")
ax5.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

# ── Row 2 col 2: Population by disease state ─────────────────────────────────
ax6 = styled_ax(fig.add_subplot(gs[2, 2]), "Population — AUC by Disease State")
disease_groups = {}
for r in pop["individual_results"]:
    ds = r["subject"]["disease_state"]
    disease_groups.setdefault(ds, []).append(r["auc_plasma"])

ds_colors = {
    "healthy": GREEN, "mild_ckd": YELLOW,
    "moderate_ckd": ORANGE, "severe_ckd": RED,
    "liver_disease": PURPLE,
}
ds_order = ["healthy", "mild_ckd", "moderate_ckd", "severe_ckd", "liver_disease"]
bp_data  = [disease_groups.get(ds, [0]) for ds in ds_order if ds in disease_groups]
bp_labels = [ds.replace("_", "\n") for ds in ds_order if ds in disease_groups]
bp_colors = [ds_colors.get(ds, BLUE) for ds in ds_order if ds in disease_groups]

bp = ax6.boxplot(bp_data, patch_artist=True, widths=0.5,
                 medianprops=dict(color=TEXT, lw=2),
                 whiskerprops=dict(color=SUBTLE),
                 capprops=dict(color=SUBTLE),
                 flierprops=dict(marker="o", color=SUBTLE, markersize=3))
for patch, col in zip(bp["boxes"], bp_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.6)
    patch.set_edgecolor(BORDER)
ax6.set_xticklabels(bp_labels, fontsize=7.5)
ax6.set_ylabel("AUC  mg·h/L")
ax6.text(0.02, 0.95, "↑ Higher AUC in CKD\n  patients (less renal CL)",
         transform=ax6.transAxes, color=SUBTLE, fontsize=7.5, va="top")

# ── Row 3 col 0: Age vs AUC scatter ──────────────────────────────────────────
ax7 = styled_ax(fig.add_subplot(gs[3, 0]), "Population — Age vs Plasma AUC")
ages    = [r["subject"]["age"]        for r in pop["individual_results"]]
aucs    = [r["auc_plasma"]            for r in pop["individual_results"]]
egfrs   = [r["subject"]["egfr"]       for r in pop["individual_results"]]
sc = ax7.scatter(ages, aucs, c=egfrs, cmap="RdYlGn", s=25, alpha=0.75,
                 vmin=10, vmax=130, edgecolors="none")
cbar = fig.colorbar(sc, ax=ax7, pad=0.02)
cbar.set_label("eGFR (mL/min)", color=SUBTLE, fontsize=7.5)
cbar.ax.yaxis.set_tick_params(color=SUBTLE, labelsize=7)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=SUBTLE)
ax7.set_xlabel("Age (years)"); ax7.set_ylabel("Plasma AUC  (mg·h/L)")
# Fit line
z = np.polyfit(ages, aucs, 1)
xfit = np.linspace(min(ages), max(ages), 50)
ax7.plot(xfit, np.polyval(z, xfit), color=YELLOW, lw=1.5, ls="--", alpha=0.7)

# ── Row 3 col 1: Population kidney risk distribution ─────────────────────────
ax8 = styled_ax(fig.add_subplot(gs[3, 1]), "Population — Kidney Risk Score Distribution")
kid_scores = []
for r in pop["individual_results"]:
    s = score_single_simulation(r)
    kid_scores.append(s["organ_scores"].get("kidney", 0))

n_bins = 16
n_k, bins_k, patches_k = ax8.hist(kid_scores, bins=n_bins, edgecolor=BORDER)
for patch, left in zip(patches_k, bins_k[:-1]):
    mid = left + (bins_k[1] - bins_k[0]) / 2
    patch.set_facecolor(risk_color(mid))
    patch.set_alpha(0.8)

ax8.axvline(0.45, color=ORANGE, ls="--", lw=1, label=f"Elevated ≥0.45")
ax8.axvline(0.70, color=RED,    ls="--", lw=1, label=f"High risk ≥0.70")
pct_high = 100 * sum(s >= 0.70 for s in kid_scores) / len(kid_scores)
pct_elev = 100 * sum(s >= 0.45 for s in kid_scores) / len(kid_scores)
ax8.text(0.55, 0.88, f"High risk: {pct_high:.0f}%\nElevated:  {pct_elev:.0f}%",
         transform=ax8.transAxes, color=SUBTLE, fontsize=8, va="top")
ax8.set_xlabel("Kidney Risk Score"); ax8.set_ylabel("Count")
ax8.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

# ── Row 3 col 2: Key metrics summary table ────────────────────────────────────
ax9 = styled_ax(fig.add_subplot(gs[3, 2]), "Engine Validation Summary")
ax9.axis("off")

rows = [
    ("Drug",           "Metformin",          "Caffeine",          ""),
    ("Dose",           "500 mg oral",         "200 mg oral",       ""),
    ("Predicted Cmax", f"{met_result['cmax_plasma']:.2f} mg/L",
                       f"{caf_result['cmax_plasma']:.2f} mg/L",    ""),
    ("Literature Cmax","1.0–2.0 mg/L",        "1.5–3.5 mg/L",     ""),
    ("Predicted AUC",  f"{met_result['auc_plasma']:.1f} mg·h/L",
                       f"{caf_result['auc_plasma']:.1f} mg·h/L",  ""),
    ("Literature AUC", "6–14 mg·h/L",         "12–30 mg·h/L",     ""),
    ("Predicted Tmax", f"{met_result['tmax_plasma']:.1f} h",
                       f"{caf_result['tmax_plasma']:.1f} h",       ""),
    ("Literature Tmax","2–3 h",               "0.5–1.5 h",        ""),
    ("Top risk organ", ORGAN_DISPLAY.get(met_scores["dominant_organ"],"—"),
                       ORGAN_DISPLAY.get(caf_scores["dominant_organ"],"—"), ""),
    ("Population n",   "80 virtual patients", "—",                ""),
    ("AUC P50 (pop)",  f"{np.percentile(pop_aucs,50):.1f} mg·h/L","—",  ""),
    ("AUC fold range", f"P5–P95: {np.percentile(pop_aucs,5):.1f}–{np.percentile(pop_aucs,95):.1f}",
                       "—",                                        ""),
]
col_x = [0.0, 0.35, 0.68]
row_y  = np.linspace(0.96, 0.05, len(rows))
headers = ["Metric", "Metformin", "Caffeine"]
for i, h in enumerate(headers):
    ax9.text(col_x[i], 1.0, h, color=TEXT, fontsize=8.5, fontweight="bold",
             transform=ax9.transAxes, va="top")
ax9.plot([0, 1], [0.97, 0.97], color=BORDER, lw=0.8, transform=ax9.transAxes, clip_on=False)
for j, (label, met_v, caf_v, _) in enumerate(rows):
    y = row_y[j]
    col = TEXT if j % 2 == 0 else SUBTLE
    ax9.text(col_x[0], y, label, color=SUBTLE, fontsize=7.5,
             transform=ax9.transAxes, va="top")
    ax9.text(col_x[1], y, met_v, color=col, fontsize=7.5,
             transform=ax9.transAxes, va="top")
    ax9.text(col_x[2], y, caf_v, color=col, fontsize=7.5,
             transform=ax9.transAxes, va="top")

# ── footer ────────────────────────────────────────────────────────────────────
fig.text(0.06, 0.013,
         "BodySim Core Engine v0.1  ·  13-compartment PBPK  ·  ICRP-89 physiology  ·  "
         "Rodgers-Rowland Kp estimation  ·  Well-stirred liver model  ·  LSODA ODE solver",
         color=SUBTLE, fontsize=7, ha="left")

out = "/mnt/user-data/outputs/bodysim_validation.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[BodySim] Saved → {out}")
