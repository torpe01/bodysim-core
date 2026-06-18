"""
diagnostic_universal_peff_coverage.py — v5.3 diagnostic #5 (universal).

Purpose
-------
Isolation test #4 established that F and ka are dead for ALL 23 drugs (an
architectural fact, not a per-drug bug): acat_module.py's mechanistic ACAT
absorption model never reads drug["ka"] or drug["F"] at all (confirmed by
direct grep — zero matches in acat_module.py). The only parameter that now
actually controls absorption RATE is p_eff (intestinal effective
permeability), with a QSPR fallback (Egan-style transcellular + paracellular
estimate from logp/mw/hbd) used whenever a drug's reference_pk.py entry
omits an explicit, literature-sourced p_eff.

This script is a UNIVERSAL coverage and consistency check, run once across
all 23 drugs simultaneously (per project convention: always build these as
universal sweeps, not single-drug investigations), to answer three
questions for every drug at once:

  1. Does this drug have an EXPLICIT, literature-sourced p_eff in
     reference_pk.py, or is it silently falling through to the generic
     QSPR estimate?
  2. If it IS falling through to QSPR, what value does that estimate
     actually produce, and is it wildly different in magnitude from peer
     drugs with a similar logP that DO have an explicit, sourced p_eff
     (a rough plausibility check, not a rigorous validation)?
  3. Does the now-dead ka value (still present in reference_pk.py and
     still computed by build_drug_profile(), just never consumed) suggest
     a directionally different absorption rate than what p_eff/QSPR is
     actually driving? (A large disagreement here would mean the engine's
     ACTUAL behavior has silently diverged from what the original literature
     calibration intended, even though nobody changed reference_pk.py's
     ka value — because ka stopped being the thing in control.)

This script does NOT call build_drug_profile() or run any simulation — it
inspects reference_pk.py directly and reimplements ONLY the QSPR p_eff
fallback formula (copied verbatim from acat_module.py) for drugs that lack
an explicit p_eff, so it can report what value the engine is actually using
internally for every drug without needing a full simulation run.

Run from the repository root:
    python engine/validation/diagnostic_universal_peff_coverage.py
"""

import sys
import os
import numpy as np

sys.path.append(os.getcwd())

from engine.validation.reference_pk import REFERENCE_PK


def _qspr_p_eff_fallback(logp, mw, hbd=0):
    """
    Verbatim reimplementation of acat_module.py's QSPR p_eff fallback
    (lines ~119-124), used here ONLY for reporting what the engine
    actually computes internally when a drug lacks an explicit p_eff.
    Do not use this as a source of truth for new drug entries — it is a
    coarse generic estimate, not a literature-sourced value.
    """
    p_trans = float(np.clip(10.0 ** (0.4 * logp - 5.5), 1e-8, 1e-3))
    p_para = float(np.clip(
        1.5e-5 * np.exp(-0.010 * max(0.0, mw - 100.0)) * (0.85 ** hbd),
        1e-9, 1e-4,
    ))
    return float(np.clip(np.sqrt(p_trans**2 + p_para**2), 1e-8, 5e-4))


def run_diagnostic():
    print(f"\n{'='*100}")
    print(" v5.3 DIAGNOSTIC #5 (universal): p_eff coverage and ka/F dead-data consistency")
    print(f"{'='*100}\n")

    rows = []
    for drug_name, data in REFERENCE_PK.items():
        explicit_p_eff = data.get("p_eff", None)
        logp = data.get("logp", 0.0)
        mw = data.get("mw", 300.0)
        hbd = data.get("hbd", 0)  # almost certainly always 0 — check below
        dead_ka = data.get("ka", None)
        dead_F = data.get("F", None)
        clint = data.get("clint", None)

        if explicit_p_eff is not None:
            source = "EXPLICIT (literature)"
            effective_p_eff = float(explicit_p_eff)
        else:
            source = "QSPR FALLBACK (generic estimate)"
            effective_p_eff = _qspr_p_eff_fallback(logp, mw, hbd)

        rows.append({
            "drug": drug_name,
            "source": source,
            "p_eff": effective_p_eff,
            "logp": logp,
            "mw": mw,
            "hbd_present": "hbd" in data,
            "dead_ka": dead_ka,
            "dead_F": dead_F,
        })

    # ── Table 1: coverage summary ──────────────────────────────────────
    n_explicit = sum(1 for r in rows if r["source"].startswith("EXPLICIT"))
    n_fallback = len(rows) - n_explicit
    print(f" Coverage: {n_explicit}/{len(rows)} drugs have EXPLICIT literature p_eff;"
          f" {n_fallback}/{len(rows)} are on the QSPR FALLBACK.\n")

    header = f" {'Drug':<16} {'p_eff source':<32} {'p_eff (cm/s)':>14} {'logP':>7} {'MW':>8} {'dead ka':>9} {'dead F':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: x["source"]):
        ka_str = f"{r['dead_ka']:.2f}" if r["dead_ka"] is not None else "—"
        F_str = f"{r['dead_F']:.2f}" if r["dead_F"] is not None else "—"
        print(f" {r['drug']:<16} {r['source']:<32} {r['p_eff']:>14.3e} {r['logp']:>7.2f} "
              f"{r['mw']:>8.1f} {ka_str:>9} {F_str:>8}")

    # ── Table 2: hbd field presence (the QSPR formula depends on hbd, which
    #    may not exist anywhere in reference_pk.py, silently defaulting to 0
    #    for every fallback drug regardless of actual hydrogen-bond count) ──
    n_with_hbd = sum(1 for r in rows if r["hbd_present"])
    print(f"\n Of the {n_fallback} QSPR-fallback drugs, {sum(1 for r in rows if r['hbd_present'] and r['source'].startswith('QSPR'))}"
          f" have an explicit 'hbd' key in reference_pk.py; the rest silently use hbd=0,")
    print(f" meaning the paracellular term of the QSPR estimate ignores hydrogen-bond")
    print(f" donor count entirely for those drugs (the (0.85**hbd) factor reduces to 1.0).")

    # ── Table 3: plausibility cross-check — QSPR fallback drugs vs. nearby
    #    explicit drugs by logP, to spot estimates that look out of family ──
    explicit_rows = [r for r in rows if r["source"].startswith("EXPLICIT")]
    fallback_rows = [r for r in rows if r["source"].startswith("QSPR")]

    print(f"\n{'='*100}")
    print(" Plausibility cross-check: each QSPR-fallback drug vs. its closest-logP")
    print(" EXPLICIT (literature) neighbor — large p_eff ratios flag estimates that")
    print(" may be out of family and worth sourcing a real literature value for.")
    print(f"{'='*100}")
    header2 = f" {'Fallback drug':<16} {'logP':>7} {'p_eff (QSPR)':>14}   {'Nearest explicit':<16} {'logP':>7} {'p_eff (lit)':>14} {'Ratio':>8}"
    print(header2)
    print("-" * len(header2))
    for fb in sorted(fallback_rows, key=lambda x: x["logp"]):
        if not explicit_rows:
            print(f" {fb['drug']:<16} {fb['logp']:>7.2f} {fb['p_eff']:>14.3e}   (no explicit drugs to compare against)")
            continue
        nearest = min(explicit_rows, key=lambda x: abs(x["logp"] - fb["logp"]))
        ratio = fb["p_eff"] / nearest["p_eff"] if nearest["p_eff"] else float("nan")
        flag = "  <-- check" if (ratio > 5.0 or ratio < 0.2) else ""
        print(f" {fb['drug']:<16} {fb['logp']:>7.2f} {fb['p_eff']:>14.3e}   "
              f"{nearest['drug']:<16} {nearest['logp']:>7.2f} {nearest['p_eff']:>14.3e} {ratio:>8.2f}{flag}")

    print(f"\n{'='*100}\n")

    print(" Interpretation guide:")
    print(" - Any drug on QSPR FALLBACK has NO literature backing for the only")
    print("   parameter currently controlling its absorption rate. This is not")
    print("   automatically wrong, but it means that drug's Cmax-timing/shape")
    print("   in validate_drugs.py results is driven by a generic estimate, not")
    print("   a sourced measurement — worth flagging before trusting any")
    print("   fold-error diagnosis for these drugs as a 'real' physiological")
    print("   finding rather than partly an artifact of an unvalidated p_eff.")
    print(" - A large Ratio in the plausibility table (flagged '<-- check') means")
    print("   the QSPR estimate for that drug differs substantially in magnitude")
    print("   from a literature value for a similarly lipophilic drug — worth a")
    print("   manual literature check (Caco-2, PAMPA, or human SPIP data) before")
    print("   trusting that drug's absorption-phase results.")
    print(" - dead_ka / dead_F columns are shown ONLY for reference — they are")
    print("   NOT consumed by the simulation (per isolation test #4). A drug")
    print("   whose dead_ka literature value implies fast absorption (e.g. >1.5")
    print("   /h) while its actual p_eff is small (QSPR or explicit) suggests the")
    print("   original ka-based calibration and the current p_eff-based")
    print("   calibration may disagree about that drug's absorption kinetics —")
    print("   not actionable by itself, but useful context if that drug's")
    print("   Cmax timing looks wrong in validate_drugs.py.")
    print()


if __name__ == "__main__":
    run_diagnostic()