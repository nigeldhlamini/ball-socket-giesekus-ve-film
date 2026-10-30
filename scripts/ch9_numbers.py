#!/usr/bin/env python
r"""Chapter 9 numbers generator: verify every \BsNine... macro, then emit.

CONTRACT.  Every numeral in the Chapter 9 prose is a macro.  This script is the
single place where each macro's value is derived from a persisted artefact and
checked against the value declared in the prose MACRO VALUE BLOCKs
(ch9_results_prose_v1.tex and ch9_scoping_prose_v1.tex).

ABORT-ON-MISMATCH.  A macro whose derived value disagrees with the declared
value beyond its tolerance is a HARD failure.  So is a macro whose derivation
is not available, and so is any macro flagged CONVENTION (declared under an
estimator convention that differs from the one its neighbours use).  If there
is one or more HARD failure the script writes NOTHING and exits non-zero.  It
does not round, retry, widen a tolerance, or fall back to the declared value.

The generator does not decide which convention is correct.  Where two defensible
conventions give different numbers it reports both and fails, because that is an
author's decision and not a computation.
"""
from __future__ import annotations

import glob
import json
import re
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOLVER = ROOT.parent.parent / "giesekus-solver" / "giesekus-solver"
OUT = ROOT / "results" / "ch9"
TEX_OUT = ROOT / "results" / "ch9" / "chapter9_numbers.tex"


# --------------------------------------------------------------------------
# artefact loading
# --------------------------------------------------------------------------

def _load():
    a: dict = {}
    a["audit"] = pd.read_parquet(OUT / "audit_N8192_seed20260727.parquet")
    a["audit_sum"] = json.loads(
        (OUT / "audit_N8192_seed20260727_summary.json").read_text())
    a["ext"] = pd.read_parquet(
        sorted(glob.glob(str(OUT / "audit_N90000_*.parquet")))[-1])
    a["ext_sum"] = json.loads(pathlib.Path(
        sorted(glob.glob(str(OUT / "audit_N90000_*_summary.json")))[-1]).read_text())
    a["oat"] = json.loads((OUT / "oat_block_n256_summary.json").read_text())
    a["e5"] = json.loads((OUT / "e5_sweep_summary.json").read_text())
    a["fidelity"] = json.loads((OUT / "fidelity_T5_extended.json").read_text())
    a["shell"] = json.loads((OUT / "shell_diagnostic.json").read_text())
    a["rescore"] = json.loads((OUT / "classifier_rescore.json").read_text())
    a["recall"] = json.loads((OUT / "boundary_recall.json").read_text())
    a["rnd"] = pd.read_parquet(OUT / "bo_random_arm_n500.parquet")
    a["bo_meta"] = [json.loads(p.read_text())
                    for p in sorted((OUT / "bo_runs").glob("*_meta.json"))]

    # --- slipper artefacts, re-derived not copied -------------------------
    a["slip_conv"] = json.loads(
        (SOLVER / "figures/bo_corrected/fig_bo_convergence_rate_manifest.json"
         ).read_text())
    a["slip_feas"] = json.loads(
        (SOLVER / "results/recompute/bo_feasibility_classification.json"
         ).read_text())
    a["slip_oracle"] = pd.read_csv(SOLVER / "logs/diagnostics/tab_oracle_ceiling.csv")
    return a


def _pct(num, den):
    return 100.0 * float(num) / float(den)


def _median_over_arms(rescore, band, key):
    """Per-metric median over the five seeded arms.  This is the estimator
    convention registered for the slipper (Addendum 3 A3': median over seeds,
    min-max recorded), and it is applied here to every classifier metric."""
    return float(np.median([a[key] for a in rescore[band]["arms"]]))


def _arm_at_median_auc(rescore, band, key):
    """The value of `key` on whichever arm carries the median AUC.  A DIFFERENT
    estimator from _median_over_arms; both are defensible and they disagree."""
    arms = rescore[band]["arms"]
    med_auc = float(np.median([a["auc"] for a in arms]))
    hit = min(arms, key=lambda a: abs(a["auc"] - med_auc))
    return float(hit[key])


# --------------------------------------------------------------------------
# derivations, one per macro
# --------------------------------------------------------------------------

def derive(a: dict) -> dict[str, tuple[float, str]]:
    """macro -> (derived value, provenance string)."""
    d: dict[str, tuple[float, str]] = {}
    s, aud, ext = a["audit_sum"], a["audit"], a["ext"]
    conv = aud[aud.outcome == "converged"]

    # -- audit set ---------------------------------------------------------
    d["BsNineNAudit"] = (s["n"], "audit summary n")

    # --- G1: the registered roughness band, in micrometres --------------------
    # Quoted in S2, in S6's film-parameter arithmetic and in the not-claims
    # list, and it had no macro at all -- so there was nothing for Paper 3 to
    # alias and the band was reaching the prose by hand.  It is a REGISTERED
    # CONSTANT, so this is a derivation from config, never a typed value.
    # Read from the FROZEN box-D artefact, not the live module: the registered
    # value is the hashed one, and a macro sourced from the frozen JSON cannot
    # drift if someone edits the module.
    _frozen = json.loads((ROOT / "ball_socket" / "campaign_frozen_v6.json"
                          ).read_text(encoding="utf-8"))
    _SSB = _frozen["config"]["sigma_star_band_m"]
    d["BsNineSigmaStarLoUm"] = (_SSB[0] * 1e6,
                                "frozen box-D config, sigma* band low end")
    d["BsNineSigmaStarHiUm"] = (_SSB[1] * 1e6,
                                "frozen box-D config, sigma* band high end")

    # --- G2a: median film parameter over the CONVERGED population -------------
    # NOTE THE POPULATION, because it is the whole point of this macro.  A
    # converged row is by construction one that FOUND a full-film equilibrium,
    # so its Lambda is above the criterion and this statistic CANNOT support the
    # S2 claim that the joint runs below the full-film threshold.  It is
    # registered because it is the honest characterisation of the converged
    # minority, and because registering it makes the distinction visible instead
    # of leaving the next reader to reach for the nearest Lambda macro.
    #
    # The statistic S2 needs is Lambda over the BLOW-UP population -- registered
    # expectation E8 -- which does not exist until box D lets those rows reach
    # an equilibrium.  It is deliberately NOT declared here rather than
    # approximated by this one.
    _conv = a["audit"][a["audit"]["converged"] == True]          # noqa: E712
    d["BsNineLamMedConvLo"] = (float(_conv["lambda_ratio_lo"].median()),
                               "median Lambda, CONVERGED rows, sigma* low end")
    d["BsNineLamMedConvHi"] = (float(_conv["lambda_ratio_hi"].median()),
                               "median Lambda, CONVERGED rows, sigma* high end")
    d["BsNineAuditConverged"] = (s["n_converged"], "audit summary")
    d["BsNineAuditConvPct"] = (100.0 * s["convergence_rate"], "audit summary")
    d["BsNineAuditBlowupPct"] = (100.0 * s["nonconvergence_fraction"], "audit summary")
    d["BsNineAuditBlowup"] = (s["n_nonconverged_blowup"], "audit summary")

    # The blow-up rows carry no solver outputs, so F cannot be read off the
    # parquet for them.  It does not need to be: at equilibrium F is
    # deterministic in the inputs, F = p_ch (A_p - A_pocket(theta_min)), which
    # is the same analytic recovery ch9_fidelity._ensure_F uses for the BO arms.
    sys.path.insert(0, str(ROOT))
    from ball_socket.campaign_config_v4 import applied_and_hydrodynamic_load
    fh = np.array([applied_and_hydrodynamic_load(p, t)[2]
                   for p, t in zip(aud["p_ch"], aud["theta_min_deg"])]) / 1e3
    is_conv = (aud.outcome == "converged").to_numpy()
    d["BsNineBlowupLoadMed"] = (float(np.median(fh[~is_conv])),
                                "audit inputs, analytic F, kN, blow-up population")
    d["BsNineConvLoadMed"] = (float(np.median(fh[is_conv])),
                              "audit inputs, analytic F, kN, converged population")
    lo_band = fh < 5.0
    mid_band = (fh >= 5.0) & (fh < 10.0)
    d["BsNineConvBelowFive"] = (
        _pct(is_conv[lo_band].sum(), lo_band.sum()),
        "audit: converged share of the <5 kN band")
    d["BsNineConvFiveTen"] = (
        _pct(is_conv[mid_band].sum(), mid_band.sum()),
        "audit: converged share of the 5-10 kN band")

    # -- extended reference ------------------------------------------------
    es = a["ext_sum"]
    d["BsNineNExtended"] = (es["n"], "extended audit summary")
    d["BsNineExtConvPct"] = (100.0 * es["convergence_rate"], "extended summary")
    p1, n1 = s["convergence_rate"], s["n"]
    p2, n2 = es["convergence_rate"], es["n"]
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    d["BsNineExtConvZ"] = (abs(p1 - p2) / se, "two-proportion z, audit vs extension")

    zs = []
    for cls in ("mineral", "water_glycol"):
        for band in ("lo", "hi"):
            k = f"feasible_frac_of_converged_{cls}_{band}"
            q1, m1 = s[k], s["n_converged"]
            q2, m2 = es[k], es["n_converged"]
            pl = (q1 * m1 + q2 * m2) / (m1 + m2)
            sd = np.sqrt(pl * (1 - pl) * (1 / m1 + 1 / m2))
            zs.append(abs(q1 - q2) / sd if sd > 0 else np.nan)
    d["BsNineExtFeasZmax"] = (float(np.nanmax(zs)),
                              "max |z| over 4 fluid-class x band cells")

    # -- E1 ----------------------------------------------------------------
    for band, tag in (("lo", "Lo"), ("hi", "Hi")):
        g3 = conv[conv[f"binding_mineral_{band}"] == "g3"]
        d[f"BsNineEOneLamGThree{tag}"] = (
            float(np.nanmedian(g3["lam"])), f"audit parquet, g3-binding median lambda ({band})")
        d[f"BsNineEOneLamPct{tag}"] = (
            _pct((g3["lam"] > 1e-3).sum(), len(g3)),
            f"audit parquet, share of g3 bindings above 1e-3 s ({band})")
    d["BsNineEOneLamAll"] = (float(np.nanmedian(conv["lam"])),
                             "audit parquet, median lambda over converged")
    for band, tag in (("lo", "Lo"), ("hi", "Hi")):
        g1 = conv[conv[f"binding_mineral_{band}"] == "g1"]
        d[f"BsNineEOneDesqGOne{tag}"] = (
            float(np.nanmedian(g1["De_sq"])),
            f"audit parquet, median De_sq of the g1-binding population ({band})")

    # -- E3 ----------------------------------------------------------------
    d["BsNineEThreeWgEnv"] = (s["T_max_envelope_C_water_glycol_lo"], "audit summary")
    d["BsNineWgCeiling"] = (60.0, "registered service ceiling, campaign_config_v4")
    d["BsNineEThreeMinCount"] = (s["binding_mineral_lo"].get("g2", 0), "audit summary")
    d["BsNineEThreeMinEnv"] = (s["T_max_envelope_C_mineral_lo"], "audit summary")
    d["BsNineMineralCeiling"] = (110.0, "registered material limit")

    # -- E5 ----------------------------------------------------------------
    d["BsNineEFiveNuMax"] = (9.0, "e5_fluid_spec swept range upper end")

    # -- E6 ----------------------------------------------------------------
    routes = conv["thermal_route"]
    d["BsNineBsQmin"] = (_pct((routes == "L1->L2_qmin").sum(), len(conv)),
                         "audit parquet, through-flow guard share of converged")
    d["BsNineBsBrink"] = (_pct((routes == "L2_brinkman").sum(), len(conv)),
                          "audit parquet, Brinkman share of converged")
    # E6 is registered as a band "across the full range of the friction-driven
    # spin ratio", so only the spin_ratio OAT cells enter it.
    spin_l2 = [c["L2_routing_fraction"] for name, c in a["oat"]["cells"].items()
               if name.startswith("spin_ratio")]
    d["BsNineESixLo"] = (100.0 * min(spin_l2), "OAT spin_ratio cells, min L2 routing")
    d["BsNineESixHi"] = (100.0 * max(spin_l2), "OAT spin_ratio cells, max L2 routing")

    # -- feasibility discovery --------------------------------------------
    rnd_rates = [float((g.outcome == "converged").mean())
                 for _, g in a["rnd"].groupby(["band", "seed"])]
    d["BsNineConvBase"] = (100.0 * float(np.median(rnd_rates)),
                           "random arm, median per-run convergence")
    for band, tag in (("lo", "Lo"), ("hi", "Hi")):
        rates = [m["n_converged"] / m["n_obs_final"] for m in a["bo_meta"]
                 if m.get("band") == band]
        d[f"BsNineConvBo{tag}"] = (100.0 * float(np.median(rates)),
                                   f"BO meta, median per-run convergence ({band})")
    d["BsNineUpliftFactor"] = (
        np.mean([d["BsNineConvBoLo"][0], d["BsNineConvBoHi"][0]])
        / d["BsNineConvBase"][0], "ratio of the two above")

    # -- OAT ---------------------------------------------------------------
    d["BsNineOatN"] = (a["oat"]["n_per_cell"], "OAT summary")
    cells = a["oat"]["cells"]

    def _spread(prefix, key):
        vals = [c[key] for name, c in cells.items() if name.startswith(prefix)]
        return 100.0 * (max(vals) - min(vals))
    d["BsNineOatSpinBlowup"] = (_spread("spin_ratio", "nonconvergence_fraction"),
                                "OAT, blow-up range across spin_ratio cells")
    d["BsNineOatThetaBlowup"] = (_spread("theta_max", "nonconvergence_fraction"),
                                 "OAT, blow-up range across theta_max cells")
    d["BsNineOatSpinRouting"] = (_spread("spin_ratio", "L2_routing_fraction"),
                                 "OAT, L2 routing range across spin_ratio cells")
    bl = [c["nonconvergence_fraction"] for c in cells.values()]
    n_cell = a["oat"]["n_per_cell"]
    within = float(np.mean([np.sqrt(p * (1 - p) / n_cell) for p in bl]))
    # The quantity the prose compares a cell-to-cell MOVEMENT against is the
    # standard error of a DIFFERENCE between two cells, not of one cell.
    d["BsNineOatSE"] = (100.0 * np.sqrt(2.0) * within,
                        "OAT, standard error of a between-cell difference in "
                        "the blow-up fraction")

    # -- reconstruction ----------------------------------------------------
    fid = a["fidelity"]["bands"]
    for band, tag in (("lo", "Lo"), ("hi", "Hi")):
        pr = fid[band]["projections"]
        film = [pr[k] for k in ("F_hmin", "hmin_Tmax")]
        d[f"BsNineIouBoFrac{tag}"] = (
            100.0 * float(np.max([p["arms"]["bo"]["iou_median"]
                                  / p["oracle_ceiling_iou"] for p in film])),
            f"fidelity extended, best film-projection BO fraction of ceiling ({band})")
        d[f"BsNineIouRandFrac{tag}"] = (
            100.0 * float(np.max([p["arms"]["random"]["iou_median"]
                                  / p["oracle_ceiling_iou"] for p in film])),
            f"fidelity extended, best film-projection random fraction of ceiling ({band})")
        d[f"BsNineFeasBo{tag}"] = (
            fid[band]["projections"]["F_hmin"]["arms"]["bo"]["n_feasible_median"],
            f"fidelity extended, median feasible observations, BO ({band})")
        d[f"BsNineFeasRand{tag}"] = (
            fid[band]["projections"]["F_hmin"]["arms"]["random"]["n_feasible_median"],
            f"fidelity extended, median feasible observations, random ({band})")

    pf = [fid[b]["projections"]["Ploss_F"] for b in ("lo", "hi")]
    d["BsNineIouExtractorFrac"] = (
        100.0 * float(np.max([p["arms"][arm]["iou_median"] / p["oracle_ceiling_iou"]
                              for p in pf for arm in ("bo", "random")])),
        "fidelity extended, worst-tier fraction of ceiling, both arms both bands")

    film_ceils = [fid[b]["projections"][k]["oracle_ceiling_iou"]
                  for b in ("lo", "hi") for k in ("F_hmin", "hmin_Tmax")]
    power_ceils = [p["oracle_ceiling_iou"] for p in pf]
    d["BsNineCeilFilmLo"] = (min(film_ceils), "fidelity extended, film-tier ceilings")
    d["BsNineCeilFilmHi"] = (max(film_ceils), "fidelity extended, film-tier ceilings")
    d["BsNineCeilPowerLo"] = (min(power_ceils), "fidelity extended, power-tier ceilings")
    d["BsNineCeilPowerHi"] = (max(power_ceils), "fidelity extended, power-tier ceilings")
    d["BsNineTierRatio"] = (np.mean(film_ceils) / np.mean(power_ceils),
                            "ratio of the tier means")

    # Head-to-head over every evaluable (band, projection) on BOTH registered
    # metrics.  Enumerated once here; the win-count macros and the comparatives
    # both read this enumeration, so a sentence quantifying over cells cannot
    # be checked against a different set than the one the chapter reports.
    cells, won_iou, won_dh, won_both = [], [], [], []
    won_both_meaningful: list[str] = []
    best = None
    for band in ("lo", "hi"):
        for key, pr in fid[band]["projections"].items():
            arms = pr["arms"]
            if "status" in arms["bo"]:
                continue
            cells.append(f"{band}/{key}")
            iou_bo = arms["bo"]["iou_median"] > arms["random"]["iou_median"]
            dh_bo = arms["bo"]["dH_median"] < arms["random"]["dH_median"]
            if iou_bo:
                won_iou.append(f"{band}/{key}")
            if dh_bo:
                won_dh.append(f"{band}/{key}")
            if iou_bo and dh_bo:
                won_both.append(f"{band}/{key}")
                if pr["tier_precommitted"] == "expected_meaningful":
                    won_both_meaningful.append(f"{band}/{key}")
            if dh_bo and (best is None or arms["bo"]["dH_median"] < best[1]):
                best = (f"{band}/{key}", arms["bo"]["dH_median"],
                        arms["random"]["dH_median"])

    d["BsNineProjCells"] = (len(cells), f"evaluable (band, projection) cells: {cells}")
    d["BsNineCellsTotal"] = (2 * len(cells),
                             "evaluable cells x two registered metrics")
    d["BsNineIouRandWins"] = (len(cells) - len(won_iou),
                              f"cells where uniform attains the higher IoU "
                              f"(BO wins {won_iou})")
    d["BsNineDhBoWins"] = (len(won_dh),
                           f"cells where BO attains the smaller d_H: {won_dh}")
    d["BsNineBoWinsTotal"] = (len(won_iou) + len(won_dh),
                              "BO wins across both metrics and all cells")
    d["BsNineDhBoBest"] = (best[1], f"smallest BO d_H over all cells ({best[0]})")
    d["BsNineDhRandBest"] = (best[2], f"uniform d_H in the SAME cell ({best[0]})")

    # audit operands, not chapter macros
    d["_bo_wins_iou_cells"] = (won_iou, "cells where BO wins on IoU")
    d["_bo_wins_dh_cells"] = (won_dh, "cells where BO wins on d_H")
    d["_bo_wins_both_cells"] = (won_both,
                                "cells where BO wins on BOTH metrics at once")
    d["_bo_wins_both_meaningful"] = (
        won_both_meaningful,
        "the same, restricted to the meaningful tier")
    d["_n_cells"] = (len(cells), "evaluable cell count")

    # -- classifier scoping -------------------------------------------------
    rs = a["rescore"]
    for band, tag in (("lo", "Lo"), ("hi", "Hi")):
        d[f"BsNineAuc{tag}"] = (_median_over_arms(rs, band, "auc"),
                                f"rescore, per-metric median over arms ({band})")
        d[f"BsNinePrec{tag}"] = (_median_over_arms(rs, band, "precision"),
                                 f"rescore, per-metric median over arms ({band})")
        d[f"BsNineRecall{tag}"] = (_median_over_arms(rs, band, "recall"),
                                   f"rescore, per-metric median over arms ({band})")
        d[f"BsNineBrier{tag}"] = (_median_over_arms(rs, band, "brier"),
                                  f"rescore, per-metric median over arms ({band})")
    d["BsNineAucThreshold"] = (0.90, "registered criterion, preregistration v4")
    spreads = [a_["auc_spread"] for b in ("lo", "hi") for a_ in rs[b]["arms"]]
    refit = [hi - lo for lo, hi in spreads]
    d["BsNineRefitSpreadMin"] = (min(refit),
                                 "rescore, min within-arm AUC spread over all "
                                 "ten arms (full range, not 'typical')")
    d["BsNineRefitSpreadMax"] = (max(refit),
                                 "rescore, max within-arm AUC spread over all "
                                 "ten arms (full range, not 'typical')")
    d["BsNineRefitSpreadMed"] = (float(np.median(refit)),
                                 "rescore, median within-arm AUC spread over "
                                 "all ten arms")
    # Min/Max, not Lo/Hi: these are RANGE endpoints over the two bands, and
    # Lo/Hi is the roughness band everywhere else in the block (2026-08-01).
    band_spread = {b: max(a_["auc"] for a_ in rs[b]["arms"])
                   - min(a_["auc"] for a_ in rs[b]["arms"]) for b in ("lo", "hi")}
    d["BsNineArmSpreadMin"] = (min(band_spread.values()),
                               "rescore, between-arm AUC spread, smaller of the "
                               "two bands (hi)")
    d["BsNineArmSpreadMax"] = (max(band_spread.values()),
                               "rescore, between-arm AUC spread, larger of the "
                               "two bands (lo)")
    d["BsNineBaseRate"] = (
        _pct(s["feasible_mineral_lo"], s["n_valid"]),
        "audit summary, feasible fraction of the audit, LOWER BAND ONLY "
        "(the upper band is 8.0 per cent)")

    # -- shell diagnostic geometry -----------------------------------------
    # The retracted \BsNineFilmAreaFrac was area/diag^2 in raw coordinates.
    # What replaces it is the area in the extraction pipeline's own normalised
    # space, plus the raw aspect ratio that made the old statistic
    # uninterpretable.  Both are computed, neither asserted.
    sh = a["shell"]["bands"]
    film_keys = ("F_hmin", "hmin_Tmax")
    bbox_fracs = [sh[b][k]["reference_area_over_bbox"]
                  for b in ("lo", "hi") for k in film_keys]
    d["BsNineNormAreaLo"] = (100.0 * min(bbox_fracs),
                             "shell diagnostic, film-projection area as a "
                             "percentage of its bounding box, normalised space")
    d["BsNineNormAreaHi"] = (100.0 * max(bbox_fracs),
                             "shell diagnostic, film-projection area as a "
                             "percentage of its bounding box, normalised space")
    d["BsNineAspectRatio"] = (
        max(sh[b]["F_hmin"]["raw_bbox_aspect_ratio"] for b in ("lo", "hi")),
        "shell diagnostic, raw (F, h_min) bounding-box aspect ratio")

    # Shortfall from unity, split by arm (2026-08-01 correction: the UNIFORM
    # arm saturates, the Bayesian arm does not).
    rec = a["recall"]

    def _gaps(arm):
        return [1.0 - pv["arms"][arm]["recall_overall_median"]
                for b in ("lo", "hi")
                for pv in rec["bands"][b]["projections"].values()]
    # RETIRED: \BsNineRecallSaturationGap.  Reported per band instead, because
    # the uniform arm saturates EXACTLY on (h_min, T_max) at both bands and
    # merely comes close elsewhere, and those are different statements.
    for band, tag in (("lo", "Lo"), ("hi", "Hi")):
        others = [1.0 - pv["arms"]["random"]["recall_overall_median"]
                  for k, pv in rec["bands"][band]["projections"].items()
                  if k != "hmin_Tmax"]
        exact = 1.0 - rec["bands"][band]["projections"]["hmin_Tmax"][
            "arms"]["random"]["recall_overall_median"]
        if exact != 0.0:
            raise AssertionError(
                f"the prose states the uniform arm attains unity EXACTLY on "
                f"(h_min, T_max) at both bands; at {band} the shortfall is "
                f"{exact:.6g}")
        d[f"BsNineRecallGapUnif{tag}"] = (
            float(max(others)),
            f"boundary recall: worst uniform-arm shortfall from unity ({band}), "
            f"excluding (h_min, T_max) where it saturates exactly at 0.0000")
    # Fractions, not points: recall is reported as a fraction throughout the
    # chapter, and this sentence asks the reader to compare four shortfalls
    # (2026-08-01 ruling).  Mixing units inside one comparison is the defect.
    bo_gaps = _gaps("bo")
    d["BsNineRecallGapBoLo"] = (float(min(bo_gaps)),
                                "boundary recall: smallest BO shortfall from "
                                "unity, as a fraction")
    d["BsNineRecallGapBoHi"] = (float(max(bo_gaps)),
                                "boundary recall: largest BO shortfall from "
                                "unity, as a fraction")

    # -- ceiling-validity subsection ---------------------------------------
    fid8 = json.loads((OUT / "fidelity_T5.json").read_text())["bands"]
    draws = []
    for b in ("lo", "hi"):
        n_ref = fid8[b]["n_reference_feasible"]
        draws.append(_pct(min(500, n_ref), n_ref))
    d["BsNineBsOracleDrawLo"] = (min(draws), "fidelity (registered audit), oracle draw share")
    d["BsNineBsOracleDrawHi"] = (max(draws), "fidelity (registered audit), oracle draw share")
    deltas = []
    for b in ("lo", "hi"):
        for k in ("F_hmin", "hmin_Tmax", "Ploss_F"):
            deltas.append(fid8[b]["projections"][k]["oracle_ceiling_iou"]
                          - fid[b]["projections"][k]["oracle_ceiling_iou"])
    d["BsNineCeilDeltaLo"] = (min(deltas), "ceiling drop, registered -> extended reference")
    d["BsNineCeilDeltaHi"] = (max(deltas), "ceiling drop, registered -> extended reference")

    # -- slipper, re-derived ------------------------------------------------
    cm = a["slip_conv"]

    def _norm(k):
        return {"m52": "matern52", "m32": "matern32"}.get(k, k)
    seen, cells_s = set(), []
    for c in cm["ofat_cells"]:
        ck = (_norm(c["kernel"]), int(c["n_init"]), int(c["seed"]))
        if ck in seen:
            continue
        seen.add(ck)
        cells_s.append(c)
    d["BsNineSlipperConvBase"] = (100.0 * cm["exhaustive_baseline_convergence"],
                                  "slipper convergence manifest, exhaustive baseline")
    d["BsNineSlipperConvBo"] = (
        100.0 * float(np.mean([c["conv_rate"] for c in cells_s])),
        f"slipper convergence manifest, mean over {len(cells_s)} deduped OFAT cells")

    sf = a["slip_feas"]["results"]
    d["BsNineSlipperPrec"] = (sf["feasibility_joint_prob"]["precision"]["median"],
                              "slipper feasibility classification, joint-posterior rule")
    d["BsNineSlipperRecall"] = (sf["feasibility_joint_prob"]["recall"]["median"],
                                "slipper feasibility classification, joint-posterior rule")
    d["BsNineSlipperBrier"] = (sf["convergence_only"]["brier"]["median"],
                               "slipper feasibility classification, convergence classifier")

    orc = a["slip_oracle"]
    meds = {k: float(np.median(orc[orc.projection == k]["IoU"]))
            for k in orc.projection.unique()}
    d["BsNineSlipperCeilLo"] = (min(meds["F_hmin"], meds["hmin_Tmax"]),
                                "slipper oracle ceiling csv, meaningful-tier medians")
    d["BsNineSlipperCeilHi"] = (max(meds["F_hmin"], meds["hmin_Tmax"]),
                                "slipper oracle ceiling csv, meaningful-tier medians")
    d["BsNineSlipperOracleDraw"] = (_pct(500, 36232),
                                    "slipper: 500 drawn from 36,232 feasible")

    # -- slipper thermal routing and envelope, re-derived from the corrected
    #    exhaustive parquet (sha 6a96435c), never copied from prose.
    ex = pd.read_parquet(
        SOLVER / "results/recompute/ar_exhaustive_N131072_seed0_corrected.parquet")
    exc = ex[ex.converged == True]                              # noqa: E712
    n_c = len(exc)
    modes = exc["thermal_mode"]
    d["BsNineSlipperLTwo"] = (_pct((modes != "L1").sum(), n_c),
                              "corrected parquet, total L2 share of converged")
    d["BsNineSlipperBrink"] = (_pct((modes == "L2").sum(), n_c),
                               "corrected parquet, Brinkman-routed share")
    d["BsNineSlipperQmin"] = (_pct((modes == "L1->L2").sum(), n_c),
                              "corrected parquet, through-flow-guard reroute share")
    exf = exc[exc.feasible == True]                             # noqa: E712
    d["BsNineSlipperTmax"] = (float(exf["T_max"].max()) - 273.15,
                              "corrected parquet, max T_max over the FEASIBLE "
                              "envelope, degC")
    derive_boxF(d)
    derive_paper3_model(d)
    return d


# --------------------------------------------------------------------------
# BOX F  (R-46, R-47.10.8): every number Paper 3 sections 5.3-5.7 and 6 cite
# from boxes E and F, derived here so that no prose transcribes one.
# --------------------------------------------------------------------------
BOXE_MIXED = "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
BOXE_KIND = "derived_boxE_mixed_32x24_corrected_nonconvergence_kind.parquet"
#: Findings the box F derivations refuse on.  Collected, not raised, so the run
#: still prints everything else and the failure NAMES what moved (R-48.10: a
#: gate that fails silently on a finding is a gate someone will delete).
BOXF_GATE_FAILS: list[str] = []

CELLS = (("sigma_lo|mineral", "LoMin"), ("sigma_lo|water_glycol", "LoWg"),
         ("sigma_hi|mineral", "HiMin"), ("sigma_hi|water_glycol", "HiWg"))


def _registered_E7_threshold_pct() -> float:
    """E7's 5 % is registered in PROSE (prereg v5 sec 4.1), not in a config
    key, so it is read from the frozen text rather than typed here."""
    import re
    txt = (ROOT / "preregistration_v5_FROZEN.md").read_text(encoding="utf-8")
    row = next(ln for ln in txt.splitlines() if ln.startswith("| **E7** |"))
    return float(re.search(r"below (\d+(?:\.\d+)?) % of valid rows", row).group(1))


def derive_boxF(d: dict) -> None:
    M = pd.read_parquet(OUT / BOXE_MIXED, columns=[
        "sample_id", "model_id", "outcome", "h_min_bnd", "c0", "c_equilibrium",
        "T_in", "T_max_bnd", "De_sq_bnd"])
    M["model_id"] = M.model_id.astype(int)
    # Validity partition (R-32.4 / R-37.4), kind from the DERIVED file.  Design
    # rule 3: never df.get(...) -- an absent column must fail, not default.
    K = pd.read_parquet(OUT / BOXE_KIND)[["sample_id", "model_id",
                                          "nonconvergence_kind"]]
    K["model_id"] = K.model_id.astype(int)
    M = M.merge(K, on=["sample_id", "model_id"], how="left", validate="1:1")
    untyped = (M.outcome == "nonconverged") & M.nonconvergence_kind.isna()
    if untyped.any():
        raise AssertionError(f"{int(untyped.sum())} nonconverged rows carry no kind")
    V = M[~((M.nonconvergence_kind == "picard_at_root")
            | M.outcome.isin(["clearance_nonconverged", "error"]))]
    noeq = V.outcome == "contact_limit"
    thr = _registered_E7_threshold_pct()

    # -- E7, registered population (smooth film) ------------------------------
    d["BsNineESevenValid"] = (len(V), "box E mixed, valid rows (R-37.4 partition)")
    d["BsNineESevenNoEq"] = (int(noeq.sum()), "box E mixed, contact_limit over valid")
    d["BsNineESevenPct"] = (100 * noeq.mean(), "E7 registered: no-equilibrium % of valid")
    d["BsNineESevenThreshPct"] = (thr, "prereg v5 FROZEN sec 4.1, E7 row")
    d["BsNineESevenFactor"] = (100 * noeq.mean() / thr, "E7 registered / threshold")

    # -- E7 composite: averaged film outside the mask, smooth inside (R-41.1) --
    F = pd.read_parquet(OUT / "boxF_band_full_resolve.parquet")
    ex, fr = F[F.treatment == "extrapolate"], F[F.treatment == "freeze"]
    key = ["arm_of_campaign", "sample_id", "model_id", "sigma"]
    if not (ex.set_index(key).outcome.sort_index()
            == fr.set_index(key).outcome.sort_index()).all():
        raise AssertionError("box F treatments disagree on an outcome; the "
                             "composite would depend on the treatment")
    Fm = ex[ex.arm_of_campaign == "mixed"][["sample_id", "model_id", "source",
                                             "outcome"]].copy()
    Fm["model_id"] = Fm.model_id.astype(int)
    gone = Fm[Fm.outcome == "clearance_nonconverged"][["sample_id", "model_id"]]
    gain = Fm[(Fm.source == "failure_side") & (Fm.outcome == "converged")]
    Vc = V.merge(gone.assign(_x=1), on=["sample_id", "model_id"], how="left")
    Vc = Vc[Vc._x.isna()].merge(gain[["sample_id", "model_id"]].assign(_g=1),
                                on=["sample_id", "model_id"], how="left")
    noeq_c = (Vc.outcome == "contact_limit") & Vc._g.isna()
    d["BsNineESevenCompValid"] = (len(Vc), "composite: valid rows less box F "
                                           "clearance_nonconverged")
    d["BsNineESevenCompNoEq"] = (int(noeq_c.sum()), "composite no-equilibrium rows")
    d["BsNineESevenCompPct"] = (100 * noeq_c.mean(), "E7 composite %")
    d["BsNineESevenCompFactor"] = (100 * noeq_c.mean() / thr, "E7 composite / threshold")
    d["BsNineSamples"] = (int(M.sample_id.nunique()), "box E samples")
    d["BsNineESevenSamplesNoEq"] = (
        int(noeq.groupby(V.sample_id).all().sum()),
        "registered: samples with no equilibrium on every valid arm")
    d["BsNineESevenCompSamplesNoEq"] = (
        int(noeq_c.groupby(Vc.sample_id).all().sum()),
        "composite: samples with no equilibrium on every valid arm")
    d["BsNineESevenCompSamplesPct"] = (
        _pct(noeq_c.groupby(Vc.sample_id).all().sum(), M.sample_id.nunique()),
        "composite, SAMPLE level: % of samples with no equilibrium on any arm "
        "(clustering diagnostic beside the row-level composite)")

    # -- arm invariance, over samples converging on ANY arm -------------------
    conv = (V.outcome == "converged").groupby(V.sample_id)
    anyc, allc = conv.any(), conv.all()
    d["BsNineArmMixedSamples"] = (int((anyc & ~allc).sum()),
                                  "samples whose verdict differs between arms")
    d["BsNineArmConvAnySamples"] = (int(anyc.sum()), "samples converging on any arm")
    d["BsNineArmInvariantPct"] = (100 * (1 - (anyc & ~allc).sum() / anyc.sum()),
                                  "% of samples converging on any arm whose "
                                  "verdict is the same on all valid arms")

    # -- E7-b: failure-side band rows gaining an equilibrium ------------------
    cap = pd.read_parquet(OUT / "boxF_band_cap_stage.parquet",
                          columns=["treatment"])
    n_fail = int((cap.treatment == "extrapolate").sum())
    fs = ex[(ex.source == "failure_side") & (ex.outcome == "converged")]
    lam_gain = fs.h_min_bnd / fs.sigma
    d["BsNineESevenBRows"] = (n_fail, "box F cap stage: failure-side band rows")
    d["BsNineESevenBGain"] = (len(fs), "failure-side rows gaining an equilibrium")
    d["BsNineESevenBPct"] = (_pct(len(fs), n_fail), "E7-b %")
    d["BsNineESevenBLamMax"] = (float(lam_gain.max()),
                                "largest Lambda among rows gaining an equilibrium")

    # -- E10 ------------------------------------------------------------------
    pb = json.loads((OUT / "boxc_E10_plasticity_band.json").read_text())
    d["BsNineETenPlasticPct"] = (100 * pb["fraction_plastic_psi_gt_1"],
                                 "E10 plasticity band, psi > 1")
    d["BsNineETenDoubtPct"] = (100 * pb["fraction_undetermined_0p6_to_1"],
                               "E10 plasticity band, 0.6 < psi <= 1")
    d["BsNineETenElasticPct"] = (100 * pb["fraction_elastic_psi_le_0p6"],
                                 "E10 plasticity band, psi <= 0.6")
    er = json.loads((OUT / "E10_capped_area_rated_corner.report.json").read_text())
    caf = er["capped_area_fraction"]
    lo, hi = _SSB_frozen()
    d["BsNineETenCapMedLo"] = (caf[f"maximum|sigma*={lo:.3e}|plastic"]["median"],
                               "E10 capped-area median, plastic arms, sigma* lo")
    d["BsNineETenCapMedHi"] = (caf[f"maximum|sigma*={hi:.3e}|plastic"]["median"],
                               "E10 capped-area median, plastic arms, sigma* hi")
    d["BsNineETenCapDoubtMax"] = (max(caf[f"maximum|sigma*={s:.3e}|undetermined"]["max"]
                                      for s in (lo, hi)),
                                  "E10 capped-area MAX over the doubt-band arms")

    # -- Lambda_max per c0 and h_min/c at the rated corner (E10 artefact) -----
    e = pd.read_parquet(OUT / "E10_capped_area_rated_corner.parquet")
    e = e[(e.corner == "maximum") & (e.state == "equilibrium")]
    got_c0 = sorted(e.c0_um.unique())
    if got_c0 != [2.0, 5.0]:
        raise AssertionError(f"converged corner clearances moved: {got_c0}")
    for c0, tag in ((2.0, "Two"), (5.0, "Five")):
        g = e[e.c0_um == c0]
        d[f"BsNineLamMax{tag}Lo"] = (float((g.h_min / lo).max()),
                                     f"rated corner, c0={c0:g} um, sigma* lo")
        d[f"BsNineLamMax{tag}Hi"] = (float((g.h_min / hi).max()),
                                     f"rated corner, c0={c0:g} um, sigma* hi")
        # R-48.10: h_min/c0 is the DESIGN ratio (a designer specifies c0);
        # h_min/c_eq is the MECHANISM diagnostic.  Both, each labelled.
        d[f"BsNineHcCold{tag}"] = (float((g.h_min / g.c0).median()),
                                   f"rated corner, c0={c0:g} um, median h_min/c0 "
                                   f"(design ratio)")
        d[f"BsNineHcEq{tag}"] = (float((g.h_min / g.c).median()),
                                 f"rated corner, c0={c0:g} um, median h_min/c_eq "
                                 f"(mechanism diagnostic)")

    # -- h_min/c across the campaign: STRATUM summaries, not a pooled
    #    correlation (A4, R-47.9).  c = equilibrium clearance.
    cv = M[M.outcome == "converged"]
    bins = pd.cut(cv.c0 * 1e6, [2, 4, 6, 8, 10, 14, 18, 22, 26, 30.0001], right=False)
    for tag, denom, what in (("Cold", cv.c0, "h_min/c0, design ratio"),
                             ("Eq", cv.c_equilibrium, "h_min/c_eq, mechanism diagnostic")):
        med = (cv.h_min_bnd / denom).groupby(bins, observed=True).median()
        for k in np.flatnonzero(np.diff(med.to_numpy()) >= 0):
            BOXF_GATE_FAILS.append(
                f"{what}: bin medians no longer strictly decreasing at bin {k + 1} "
                f"-> {k + 2} ({med.index[k]} = {med.iloc[k]:.4f}, "
                f"{med.index[k + 1]} = {med.iloc[k + 1]:.4f}); the prose's "
                f"'falls monotonically across nine bins' is no longer measured")
        d[f"BsNineHc{tag}BinFirst"] = (float(med.iloc[0]),
                                       f"median {what}, c0 in [2, 4) um")
        d[f"BsNineHc{tag}BinLast"] = (float(med.iloc[-1]),
                                      f"median {what}, c0 in [26, 30] um")
    d["BsNineHcBins"] = (len(med), "c0 bins; strict decrease checked per ratio")

    # -- constant band and feasibility band (R-41, R-42.5) --------------------
    rule = json.loads((OUT / "boxF_R41_band_rule.json").read_text())
    win = rule["constant_band"]["Lambda_bar_window"]
    d["BsNineConstBandSmoothLo"] = (win[0], "R-41 rule, smooth window low edge")
    d["BsNineConstBandHi"] = (win[1], "R-41 rule, window high edge")
    st = json.loads((OUT / "boxF_R41_sigma_band_test.report.json").read_text())
    S = pd.read_parquet(OUT / "boxF_R41_sigma_band_test.parquet",
                        columns=["arm", "unit", "Lb"]).drop_duplicates(["arm", "unit"])
    wmin = min(st["window_widening_Lambda_bar"])
    d["BsNineConstBandLo"] = (float(S[S.Lb < wmin].Lb.max()),
                              "averaged-side edge: largest sampled Lambda_bar "
                              "below the smallest widening unit (R-42.5)")
    d["BsNineConstBandWidenMin"] = (wmin, "smallest Lambda_bar at which widening occurs")
    pw = st["power"]
    d["BsNineFeasBandCount"] = (len(st["feasibility_band_Lambda_bar"]),
                                "feasibility-band units found")
    d["BsNineFeasBandUnits"] = (pw["units_tested"], "feasibility band: units tested")
    d["BsNineFeasBandLbMin"] = (pw["Lambda_bar_range"][0], "Lambda_bar range, low")
    d["BsNineFeasBandLbMax"] = (pw["Lambda_bar_range"][1], "Lambda_bar range, high")
    # R-57.5: the margin is the TIGHT-TOLERANCE value (xtol = rtol = 1e-10 on eps,
    # R56_tolerance_resolve), printed to its remaining uncertainty, the production-mesh
    # error at that row (R57_mesh_error, Newtonian proxy x the solver's coupled slope,
    # confirmed by the row's full-physics refinement).  Default-tolerance value kept in
    # provenance.  The R-56 artefact carries no theta_min_deg, so the h_min value gate
    # (R-54.1) is applied to it HERE, with theta_min joined from its own report.
    tr = json.loads((OUT / "R56_tolerance_resolve.report.json").read_text())
    tp = pd.read_parquet(OUT / "R56_tolerance_resolve.parquet")
    pc = pd.DataFrame(tr["per_case"])[["case", "src_index", "theta_min_deg"]]
    tp = tp.merge(pc, on=["case", "src_index"]).dropna(subset=["h_min_bnd"])
    dev = hmin_deviation(tp[tp.outcome == "converged"])
    if not dev <= HMIN_CLOSED_FORM_RTOL:
        BOXF_GATE_FAILS.append(f"h_min VALUE GATE: R56_tolerance_resolve departs from the "
                               f"closed form by {dev:.3e}; the tight margin macro is at risk")
    me = json.loads((OUT / "R57_mesh_error.report.json").read_text())["margin_row"]
    # R-58.2.4: the margin row at its Richardson value on the D-6 path resolves what the
    # proxy could not; the macro is that, the earlier values stay in provenance.
    mr = json.loads((OUT / "R58_richardson.report.json").read_text())["rows"]["band"]
    d["BsNineFeasBandMarginPct"] = (
        mr["shortfall_to_3_pct"],
        f"smallest |Lambda-3|/3 among tested solves, %, MESH-EXTRAPOLATED (Richardson, signed "
        f"memory, R-58.2): the closest row sits below the threshold; tight-tolerance production "
        f"value {100 * me['margin_tight']:.4f} %, default-tolerance {100 * pw['smallest_g1_margin']:.4f} %")
    d["BsNineFeasBandMarginUncPct"] = (100 * mr["uncertainty"] / 3.0,
                                       "uncertainty of the mesh-extrapolated margin, % of the threshold")
    d["BsNineFeasBandMarginMeshPct"] = (
        100 * abs(me["dLam_mesh"]) / 3.0,
        f"production-mesh error of Lambda at the margin row, as % of the threshold "
        f"(signed dLambda = {me['dLam_mesh']:+.4f}: the solver over-states Lambda; "
        f"full-physics refinement gives {me['full_physics_dLam_32x24']:+.4f})")

    # -- E14 (sec 6): net, its 95 % lower limit, F3 bound, joint worst case ---
    nb = json.loads((OUT / "E14_net_decomposition.json").read_text())["arms"]["mixed"]
    fb = json.loads((OUT / "E14_F3_bound.json").read_text())
    for cell, tag in CELLS:
        d[f"BsNineEFourteenNet{tag}"] = (100 * nb[cell]["net"], f"E14 net, {cell}, pp")
        d[f"BsNineEFourteenNetCiLo{tag}"] = (100 * nb[cell]["net_ci"][0],
                                             f"E14 net 95 % lower limit, {cell}, pp")
        d[f"BsNineEFourteenFThree{tag}"] = (fb["cells"][cell]["F3_bound_pp"],
                                            f"F3 95 % bound, {cell}, pp")
        d[f"BsNineEFourteenJoint{tag}"] = (fb["cells"][cell]["joint_worst_case_pp"],
                                           f"joint worst case, {cell}, pp")
    # -- E14 over the MOVABLE subpopulation (R-53.3, A6) ---------------------
    mv = json.loads((OUT / "E14_movable_subpopulation.json").read_text())["cells"]
    for cell, tag in CELLS:
        c = mv[cell]
        if abs(c["all_rows"]["net_pp"] - d[f"BsNineEFourteenNet{tag}"][0]) > 1e-9:
            BOXF_GATE_FAILS.append(f"E14 {cell}: the movable recomputation does not "
                                   f"reproduce the registered all-rows net")
    wg = mv["sigma_hi|water_glycol"]
    # POPULATION IN THE PROVENANCE (R-54.4): 55.6 and 55.4 are both correct and
    # are different denominators -- E14's VALID rows (196,602) and box E's
    # CONVERGED rows (106,930).  A reader who meets both without the population
    # named reads a typo.
    d["BsNineEFourteenImmovablePct"] = (
        wg["immovable_pct"],
        f"E14: % of the VALID-row denominator (mixed arm, n = {mv['sigma_hi|water_glycol']['all_rows']['n']}) "
        f"whose inlet alone fixes them infeasible under the 60 C ceiling")
    d["BsNineEFourteenNetMovHiWg"] = (wg["movable_rows"]["net_pp"],
                                      "E14 net over MOVABLE rows, sigma hi / 60 C, pp")
    d["BsNineEFourteenFThreeMovHiWg"] = (wg["F3_bound_pp"]["movable_rows"],
                                         "F3 95 % bound over MOVABLE rows, sigma hi / 60 C, pp")
    d["BsNineEFourteenJointMovHiWg"] = (wg["joint_worst_case_pp"]["movable_rows"],
                                        "joint worst case over MOVABLE rows, "
                                        "sigma hi / 60 C, pp")
    # -- E2's geography with the frame-decided rows removed (R-53.3) ---------
    g2r = json.loads((OUT / "E2_restricted_geography.json").read_text())
    if not g2r["finding"]["classes_agree_when_restricted"]:
        BOXF_GATE_FAILS.append("E2 restricted: the classes no longer agree; the "
                               "sentence in sec 5.6 asserts that they do")
    d["BsNineETwoRestrictedPct"] = (
        g2r["restricted_pct_of_converged"],
        f"% of CONVERGED box-E rows (n = {g2r['n_converged']}) whose g2 verdict the "
        f"inlet does not decide (T_in < 60 C); its complement, 55.4 %, is NOT E14's "
        f"55.6 % -- that one is over valid rows (R-54.4)")
    # R-54.5: the gap's ONE channel, decomposed exactly (residual ~1e-15)
    gap = json.loads((OUT / "E2_class_gap_decomposition.json").read_text())
    if not gap["verdict"]["decomposition_closes"]:
        BOXF_GATE_FAILS.append("E2 class gap: the g2 channel does not account for it; "
                               "R-54.5 says STOP, and sec 5.6 must not be drafted")
    d["BsNineETwoCeilingReachPct"] = (
        gap["ceiling_reach_pct_of_restricted"],
        f"% of the RESTRICTED box-E rows (n = {gap['restriction']['n_restricted']}) "
        f"whose rise carries T_max_bnd past 60 C but not past 110 C -- the only "
        f"channel by which the classes can differ in that frame")
    d["BsNineETwoGapPp"] = (g2r["finding"]["largest_share_gap_between_classes_pp"],
                            "largest gap between the classes' dominant-constraint "
                            "shares once restricted, pp")
    for cell, tag in (("sigma_lo|mineral", "MinLo"), ("sigma_hi|mineral", "MinHi"),
                      ("sigma_lo|water_glycol", "WgLo"), ("sigma_hi|water_glycol", "WgHi")):
        c = g2r["cells"][cell]
        d[f"BsNineETwoRestricted{tag}Pct"] = (
            c["restricted"]["dominant_single_pct"],
            f"g1 share of restricted converged rows, {cell}, %")
    # -- E5's VG32 binding clause, measured at last (R-54.7) ------------------
    v32 = json.loads((OUT / "E5_VG32_g1_binding_corner.report.json").read_text())
    if not v32["verdict"].startswith("CONFIRMED"):
        BOXF_GATE_FAILS.append(f"E5 VG32: the artefact no longer says CONFIRMED "
                               f"({v32['verdict'][:60]})")
    if v32["sole_g1_rows"] != v32["equilibrium_rows"]:
        BOXF_GATE_FAILS.append("E5 VG32: g1 is not the sole violation on every "
                               "equilibrium row; the sentence in the register says it is")
    d["BsNineEFiveVgThirtyTwoSolves"] = (
        v32["outcomes"]["converged"] + v32["outcomes"].get("contact_limit", 0),
        "E5 VG32 clause: solves at the registered maximum corner (40 points x 24 arms)")
    d["BsNineEFiveVgThirtyTwoEqRows"] = (
        v32["equilibrium_rows"],
        "E5 VG32 clause: rows reaching an equilibrium (c0 = 2 and 5 um)")
    d["BsNineEFiveVgThirtyTwoSoleGOne"] = (
        v32["sole_g1_rows"],
        "E5 VG32 clause: equilibrium rows on which g1 is the SOLE violated constraint")
    d["BsNineEFiveVgThirtyTwoTMaxMax"] = (
        v32["T_max_bnd"]["max"],
        "E5 VG32 clause: largest T_max_bnd, C, against the 110 C mineral ceiling -- the "
        "land maximum of the FILM-MEAN temperature, not an in-film peak (D-7, R-57.3)")
    d["BsNineEFiveVgThirtyTwoRiseMax"] = (
        v32["T_max_bnd"]["rise_max_K"],
        "E5 VG32 clause: largest dissipative rise over the 100 C inlet, K")
    d["BsNineEFiveVgThirtyTwoDeSqMax"] = (
        v32["arithmetic_premises"]["g3_unreachable"]["De_sq_max_observed"],
        "E5 VG32 clause: largest De_sq_bnd, against the 0.2 g3 limit")
    d["BsNineEFiveLamMaxCorner"] = (
        v32["Lambda"]["max"],
        "E5: largest Lambda at the rated maximum corner, registered boundary extremum "
        "(reproduces the E10-derived value on an independent run)")
    # -- E13 (sec 5.7): the extremum at the admissible corner fluid ---------------
    x13 = json.loads((OUT / "E13_extremum.report.json").read_text())
    s13 = json.loads((OUT / "E13_spec_v2.json").read_text(encoding="utf-8"))
    import hashlib
    if x13["spec_sha256"] != hashlib.sha256(
            (OUT / "E13_spec_v2.json").read_bytes()).hexdigest():
        BOXF_GATE_FAILS.append("E13: extremum report was measured under a "
                               "different spec than E13_spec_v2.json")
    if (x13["where"]["c0_um"], x13["where"]["theta_min_deg"]) != (2.0, 20.0):
        BOXF_GATE_FAILS.append(
            f"E13: the maximum moved off the registered corner to {x13['where']}; "
            f"the prose's 'bounds rather than locates' no longer describes it")
    d["BsNineEThirteenLamMax"] = (x13["Lambda_max"], "E13 extremum, largest Lambda")
    d["BsNineEThirteenShortPct"] = (100 * (1 - x13["Lambda_max"] / x13["full_film_min"]),
                                    "E13: shortfall of the best point below Lambda = 3")
    d["BsNineEThirteenSolves"] = (x13["solves"], "E13 extremum solves")
    d["BsNineEThirteenFullFilm"] = (x13["n_rows_at_or_above_full_film"],
                                    "E13 rows at Lambda >= 3")
    d["BsNineEThirteenEtaMpas"] = (1e3 * s13["fluid"]["eta_0_Pa_s"],
                                   "E13 admissible corner fluid eta_0, mPa s")
    col = x13["per_c0"]
    d["BsNineEThirteenLamMaxFourteen"] = (col["14"]["Lambda_max"],
                                          "E13 column, Lambda_max at c0 = 14 um")
    # R-58.2: the claim rows printed at their full-physics RICHARDSON values, on the D-6
    # (signed) path, tight tolerance, 32x24/64x48/128x96.  The registered-campaign values
    # above stay as the thesis's numbers; Paper 3 aliases point here.
    rr = json.loads((OUT / "R58_richardson.report.json").read_text())
    if rr["memory_sign"] != ["signed"] or not all(rr["registered"].values()):
        BOXF_GATE_FAILS.append("R-58.2 Richardson: not on the signed path, or a registration failed")
    # h_min value gate (R-54.1) on the Richardson artefact, theta_min joined from R-56's cases
    _rp = pd.read_parquet(OUT / "R58_richardson.parquet")
    _pc = pd.DataFrame(json.loads((OUT / "R56_tolerance_resolve.report.json").read_text())["per_case"])
    _rp = _rp.merge(_pc[["case", "src_index", "theta_min_deg"]], on=["case", "src_index"])
    _dev = hmin_deviation(_rp[_rp.outcome == "converged"])
    if not _dev <= HMIN_CLOSED_FORM_RTOL:
        BOXF_GATE_FAILS.append(f"h_min VALUE GATE: R58_richardson departs from the closed form by {_dev:.3e}")
    for key, tag in (("E13", "EThirteen"), ("E5", "EFive")):
        r_ = rr["rows"][key]
        if not (1.5 <= r_["p_obs"] <= 2.5):
            BOXF_GATE_FAILS.append(f"R-58.2 {key}: observed order {r_['p_obs']:.2f} outside 2 +/- 0.5")
        d[f"BsNine{tag}LamMaxR"] = (
            r_["Lambda_R"], f"{key} extremum Lambda, mesh-extrapolated (Richardson, observed order "
                            f"{r_['p_obs']:.3f}), signed memory; uncertainty {r_['uncertainty']:.1e}; "
                            f"production 32x24 value {r_['Lambda_32']:.5f}")
    d["BsNineEThirteenShortPctR"] = (
        rr["rows"]["E13"]["shortfall_to_3_pct"],
        "E13: shortfall of the best point below Lambda = 3, mesh-extrapolated (R-58.2)")

    # -- R-63.6: Section 3 (why load support saturates), chat's draft.  Quadrature values come
    # from the vendored scripts/p3_rim_asymptote.py; solver values from R-56/R-57 artefacts.
    import warnings as _w
    _w.filterwarnings("ignore")
    import p3_rim_asymptote as _RA
    import ball_socket.campaign as _Cc
    _floors = [1 - float(_Cc._registered_eps_cap(np.deg2rad(t))) * np.cos(np.deg2rad(t)) for t in (20, 30, 40)]
    if max(_floors) - min(_floors) > 1e-12:
        BOXF_GATE_FAILS.append("Sec 3: the bracket floor in delta differs across theta_min")
    d["BsNineSatFloorDelta"] = (float(np.mean(_floors)), "production bracket floor in delta = 1 - eps_cap cos theta_min")
    r1 = json.loads((OUT / "R56_R1_verification.report.json").read_text())
    d["BsNineSatFloorLoadErrPct"] = (100 * max(abs(v) for v in r1["production_mesh_load_error_at_delta_0.01"].values()),
                                     "largest |load error| at delta = 0.01, production mesh, over theta_min, % (R1)")
    mo = json.loads((OUT / "R57_memory_off.report.json").read_text())
    _sh = mo["full_shift_all_509"]
    d["BsNineSatShift"] = (_sh["median_of_per_row_differences"],
                           "median per-row exponent difference, solver - closed form, delta 0.1 -> 0.03 (509 rows)")
    d["BsNineSatShiftLo"] = (_sh["bootstrap_95"][0], "  ... 95 % bootstrap, low")
    d["BsNineSatShiftHi"] = (_sh["bootstrap_95"][1], "  ... 95 % bootstrap, high")
    _ns, _q = r1["newtonian_solver_early_exponent"], r1["quadrature_early_exponent"]
    d["BsNineSatNewtonAgree"] = (max(abs(_ns[f"{t}|32"] - _q[str(t)]) for t in (20, 30, 40)),
                                 "largest |Newtonian solver - closed form| early exponent, production mesh (R1)")
    d["BsNineSatMemOff"] = (abs(mo["subsample"]["median_per_row_off_minus_on"]),
                            "median |change in the shift| with memory off (64-row subsample, R-57)")
    _C = lambda t: 2 * np.cos(np.deg2rad(t)) ** 5 / np.sin(np.deg2rad(t)) ** 2
    _ar = _RA.area_proj(40) / _RA.area_proj(20)
    d["BsNineRimRatio"] = (_C(40) / _C(20), "rim coefficient ratio C(40)/C(20), C = 2 cos^5 / sin^2")
    d["BsNineRimRatioArea"] = (_C(40) / _C(20) / _ar, "rim coefficient ratio over the projected-area ratio")
    _dd = np.logspace(np.log10(0.01), np.log10(0.95), 400)
    _rr = np.array([_RA.f_recess(x, 40) / _RA.f_recess(x, 20) / _ar for x in _dd])
    d["BsNineSqAreaRatioLo"] = (float(_rr.min()), "pure-squeeze 40/20 load ratio over area ratio, min over delta in [0.01, 0.95]")
    d["BsNineSqAreaRatioHi"] = (float(_rr.max()), "  ... max over the same range")
    _r3 = json.loads((OUT / "R57_R3.report.json").read_text())["a_theta_min_ratio_pure_squeeze"]["rows"]
    if not all(_rr.min() <= q["solver_ratio_32"] / _ar <= _rr.max() for q in _r3):
        BOXF_GATE_FAILS.append("Sec 3: an R3 solver point lies outside the quadrature's area-normalised range")

    # -- R-64.2: numbers v4 declares as owed (\providecommand ... \owed{...}); generated here so
    # paper3_numbers.tex shadows each declaration.  CITED values carry their key in the string.
    ex = json.loads((OUT / "R57_existence_classification.report.json").read_text())
    d["BsNineExistN"] = (ex["rows"], "no-equilibrium rows classified (R-57 step 4)")
    d["BsNineExistBPct"] = (100 * ex["class_shares"]["B"], "% class B: even a Newtonian film at eta_0 cannot carry the load above 1 % of clearance (run_R57_existence.py:15)")
    d["BsNineExistAPct"] = (100 * ex["class_shares"]["A'"], "% class A': a Newtonian film would carry it; shear-thinning and memory remove the margin "
                                                  "(run_R57_existence.py:17)")
    d["BsNineExistAConf"] = (ex["Aprime_confirmed_by_newtonian_solver"], "A' rows confirmed by the Newtonian solver")
    th = json.loads((OUT / "R57_thermal_by_route.report.json").read_text())
    d["BsNineThetaFlowMed"] = (th["through_flow_group"]["quantiles"]["0.5"], "through-flow number Theta, median")
    d["BsNineThetaFlowAbovePct"] = (100 * th["through_flow_group"]["share_Theta_ge_1"], "% of converged rows with Theta >= 1")
    _c = th["E3_by_route"]
    d["BsNineMeanTinMinLo"] = (_c["lo"]["conduction"]["x1"]["smallest_breaching_inlet_C"],
                               "smallest breaching inlet, conduction rows, x1, sigma* lo, C")
    d["BsNineAdiabTinMinLo"] = (_c["lo"]["conduction"]["x6"]["smallest_breaching_inlet_C"],
                                "the same at x6 (one wall adiabatic), sigma* lo, C")
    d["BsNineAdiabTinMinHi"] = (_c["hi"]["conduction"]["x6"]["smallest_breaching_inlet_C"],
                                "the same at x6, sigma* hi, C")
    d["BsNineNahmeAdiabLoPct"] = (100 * th["nahme"]["x6"]["share_Na_gt_1_at_walther_b_min"],
                                  "% in-domain rows with Na > 1 at x6, Walther b_min (ASTMD341)")
    d["BsNineNahmeAdiabHiPct"] = (100 * th["nahme"]["x6"]["share_Na_gt_1_at_walther_b_max"],
                                  "% in-domain rows with Na > 1 at x6, Walther b_max (ASTMD341)")
    e15p = json.loads((OUT / "R57_E15_prediction.json").read_text())
    d["BsNineThermClearNm"] = (1000 * e15p["thermal_closure"]["dc_um"], "c(T) - c0 at the rated corner, nm (E15 draft)")
    from ball_socket import e5_fluid_spec as _E5
    _cor = _E5.CORNERS["maximum"]
    d["BsNineCornerPBar"] = (_cor["p_ch_bar"], "rated maximum corner pressure, bar -- CITED RE92004 (public data sheet)")
    d["BsNineCornerRpm"] = (_cor["rpm"], "rated maximum corner speed, rpm -- CITED RE92004 (public data sheet)")
    d["BsNineLeeHydWithPct"] = (69.7, "comparator: hydrodynamic share with deformation, % -- CITED LeeShangZhang2026 "
                                      "(as read in reports/boxe_E7_handback.md:89)")
    d["BsNineLeeHydWithoutPct"] = (9.5, "comparator: the same without deformation, % -- CITED LeeShangZhang2026 "
                                        "(reports/boxe_E7_handback.md:89)")
    # R-65.6: Lee et al. as read in Nigel's 13 Sept research update ("New work on the same interface")
    _RU = "Z:/Updates/September 2026/research_update_20260913.tex"
    d["BsNineLeeRbMm"] = (6.475, f"CITED LeeShangZhang2026: ball radius, mm -- {_RU}:221 'ball radius $6.475$\\,mm "
                                 "against our $6.5$' (their Table 2: R = 6.475 mm)")
    d["BsNineLeeLoadKN"] = (5.4, f"CITED LeeShangZhang2026: ball-slipper load, kN -- {_RU}:223-224 'a ball--slipper "
                                 "load of about $5.4$\\,kN against our $5.1$'")
    d["BsNineLeeSigmaUm"] = (0.5, f"CITED LeeShangZhang2026: equivalent roughness, um -- {_RU}:224 'Their equivalent "
                                  "roughness of $0.5\\,\\mu$m' (their Table 2: Rq ball-slipper 0.5 um)")
    d["BsNineLeeTempC"] = (100, f"CITED LeeShangZhang2026: fixed temperature, C -- {_RU}:245 'isothermal at a fixed "
                                "$100\\,^\\circ$C' (their Table 2: 100 C)")
    _sw = pd.read_parquet(OUT / "E4_lambda_sweep.parquet")
    _pf = _sw[(_sw.base == "pf:950") & np.isclose(_sw.lam, 1e-5)].iloc[0]
    d["BsNineDesqAtMemMax"] = (float(_pf.De_sq_bnd), "De_sq at the D2 memory maximum (pf:950)")
    from scipy.optimize import brentq as _bq
    from ball_socket.memory_sign import signed_M_over_etap as _sM
    _w0 = _bq(lambda L: _sM(0.25, L), 0.5, 3.0, xtol=1e-12)
    if abs(_w0 - json.loads((OUT / "R57_subLambda0_shares.report.json").read_text())["Lambda_0_alpha_0.25"]) > 1e-6:
        BOXF_GATE_FAILS.append("Wi_0: the vendored sign law's root no longer matches R-57's Lambda_0")
    d["BsNineWiZero"] = (_w0, "Wi_0: root of the vendored signed memory law at alpha = 0.25")
    _p2 = (pathlib.Path("Z:/Papers/Paper 2/paper2_numbers.tex")).read_text(encoding="utf-8")
    _m2 = re.search(r"\\newcommand\{\\PtwoSecTwoEpsTwoArTwoEightZeroZero\}\{([0-9.eE+-]+)\}", _p2)
    _c2 = re.search(r"\\newcommand\{\\PtwoSecTwoCTwo\}\{([0-9.eE+-]+)\}", _p2)
    if not (_m2 and _c2) or abs(float(_m2.group(1)) - float(_c2.group(1)) * 0.2 ** 2) > 5e-6:
        BOXF_GATE_FAILS.append("Paper 2's second-order share is missing or no longer C2 x 0.2^2")
    d["BsNineSecondOrderPlanar"] = (float(_m2.group(1)) if _m2 else float("nan"),
                                    "planar second-order load share at De_sq = 0.2 -- CITED Dhlamini2026b, "
                                    "Paper 2 macro \\PtwoSecTwoEpsTwoArTwoEightZeroZero (= C2 0.2^2)")
    _rr = json.loads((OUT / "R58_richardson.report.json").read_text())["rows"]
    d["BsNineRichOrderMin"] = (min(_rr["E13"]["p_obs"], _rr["E5"]["p_obs"], json.loads((OUT / "boxG_E15.report.json").read_text(encoding="utf-8"))["richardson_at_argmax"]["p_obs"]),
                               "smallest observed Richardson order over E13, E5 and E15")
    import inspect as _insp
    from ball_socket.solver import BallSocketSolver as _BSS
    _sig = _insp.signature(_BSS.__init__).parameters
    if _sig["brentq_xtol"].default != _sig["brentq_rtol"].default:
        BOXF_GATE_FAILS.append("root finder: xtol and rtol defaults differ")
    d["BsNineRootTol"] = (float(_sig["brentq_xtol"].default), "root-finder xtol = rtol on eps (solver.py:136-137)")
    _Eb = pd.read_parquet(OUT / BOXE_MIXED, columns=["sample_id", "model_id", "h_min_bnd", "lambda_ratio_lo", "lambda_ratio_hi",
                                                      "outcome_mineral_lo", "outcome_water_glycol_lo",
                                                      "outcome_mineral_hi", "outcome_water_glycol_hi"])
    _lo = _Eb.model_id < 12
    _Eb["feas"] = np.where(_lo, (_Eb.outcome_mineral_lo == "feasible") | (_Eb.outcome_water_glycol_lo == "feasible"),
                           (_Eb.outcome_mineral_hi == "feasible") | (_Eb.outcome_water_glycol_hi == "feasible"))
    _Db = pd.read_parquet(OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet",
                          columns=["sample_id", "outcome", "h_min_bnd", "lambda_ratio_lo", "lambda_ratio_hi"])
    _mm = _Eb[_Eb.feas].merge(_Db[_Db.outcome == "converged"], on="sample_id", suffixes=("", "_D"))
    if len(_mm) != int(_Eb.feas.sum()):
        BOXF_GATE_FAILS.append("feasible mixed rows without a converged pure-film counterpart")
    _lE = np.where(_mm.model_id < 12, _mm.lambda_ratio_lo, _mm.lambda_ratio_hi)
    _lD = np.where(_mm.model_id < 12, _mm.lambda_ratio_lo_D, _mm.lambda_ratio_hi_D)
    if ((_lE >= 3) != (_lD >= 3)).any():
        BOXF_GATE_FAILS.append("the contact model moves a feasible point across Lambda = 3")
    d["BsNineFeasHminShiftMaxPct"] = (100 * float((_mm.h_min_bnd / _mm.h_min_bnd_D - 1).abs().max()),
                                      "largest across-arm h_min shift at feasible points vs the pure film, % of h_min")

    # -- R-64.1: the (S, theta_min) collapse on the eta_0 basis and on the effective viscosity
    # eta_G(Wi_char); medians over the three theta_min bands.  Scatter as the factor exp(s_e).
    ce = json.loads((OUT / "R64_collapse_eff.report.json").read_text(encoding="utf-8"))
    if not (ce["registered"]["1_b_ratio_below_1_and_scatter_reduced"] and ce["registered"]["2_a_between_eta0_and_b"]):
        BOXF_GATE_FAILS.append("R-64.1: the effective-viscosity collapse registration no longer holds")
    _med = lambda xs: float(np.median(xs))
    # R-65.1: and on the solvent viscosity eta_s (basis (a), the same committed procedure)
    for _tag, _basis in (("", "eta0"), ("Eff", "b_eta_G_Wi_char"), ("Solv", "a_eta_s")):
        _bb = ce["bases"][_basis]
        d[f"BsNineCollapseScatter{_tag}"] = (_med([np.exp(b["scatter_s_e"]) for b in _bb]),
                                            f"collapse of Phi on (S, theta_min), {_basis} basis: scatter factor exp(s_e), "
                                            "median over theta_min bands")
        d[f"BsNineCollapseSolventSpread{_tag}"] = (_med([b["solvent_stratification_ratio"] for b in _bb]),
                                                  f"collapse, {_basis} basis: eta_s/eta_0 stratification ratio "
                                                  "(spread / scatter), median over bands")

    # -- R-63.3: E15 (box G), confirmed.  Everything below is WITHIN box G (signed path, one mesh
    # family); Lambda* is never set against E13's modulus 2.389 (D-6 alone moves Lambda by ~0.002).
    g15 = json.loads((OUT / "boxG_E15.report.json").read_text(encoding="utf-8"))
    if g15["outcome"] != "confirmed" or "sigma_scaling_check_R63" not in g15:
        BOXF_GATE_FAILS.append("E15: box G report is not 'confirmed' or lacks the sigma* scaling check")
    _lo, _hi = g15["curve_sigma_lo_max_over_arms"], g15["curve_sigma_hi_max_over_arms"]
    _cs = g15["argmax"]["c0_um"]
    d["BsNineEFifteenCZeroStarUm"] = (_cs, "E15: clearance of the Lambda optimum (box G grid, sigma* lo), um")
    d["BsNineEFifteenLamStarR"] = (g15["richardson_at_argmax"]["Lambda_R"],
                                   "E15: Lambda at the optimum, mesh-extrapolated (Richardson), signed path")
    d["BsNineEFifteenLamStarUnc"] = (g15["richardson_at_argmax"]["uncertainty"],
                                     "E15: uncertainty of the Richardson Lambda*")
    d["BsNineEFifteenLamTwoUm"] = (_lo["2"], "E15: Lambda at c0 = 2 um, box G grid (production 32x24), sigma* lo")
    d["BsNineEFifteenLamAtStar"] = (_lo[f"{_cs:g}"], "E15: Lambda at the optimum, box G grid (production), sigma* lo")
    d["BsNineEFifteenGainPct"] = (100 * (_lo[f"{_cs:g}"] / _lo["2"] - 1),
                                  "E15: gain in Lambda from 2 um to the optimum, within box G, %")
    d["BsNineEFifteenLamStarHi"] = (max(_hi.values()), "E15: Lambda peak at sigma* hi (box G grid)")
    d["BsNineEFifteenSigmaScalePct"] = (100 * g15["sigma_scaling_check_R63"]["rel_difference"],
                                        "E15: sigma* hi peak against the film-only scaling of the sigma* lo "
                                        "peak, % difference")

    # -- R-62.4: the film-supported operating space (A3), from its persisted artefact.
    # Modulus campaign rows with the D-6 bound; S50s at sigma* lo.  NOT a design-map claim:
    # the collapse test (R-54.6) has not run, so no (S, theta_min) map may be drawn from these.
    ops = json.loads((OUT / "R62_operating_space.json").read_text(encoding="utf-8"))
    if ops["memory_sign"].split(" ")[0] != "modulus" or "NOT YET RUN" not in ops["collapse_test"]:
        BOXF_GATE_FAILS.append("R-62.4 operating space: memory_sign or collapse-test tag changed")
    # R-65.1: the S50s and the rated corner are re-based on S_s = S eta_0/eta_s (names kept);
    # the class counts and survivors do not depend on S (asserted unchanged by the runner).
    o65 = json.loads((OUT / "R65_ops_solv.json").read_text(encoding="utf-8"))
    if not o65["basis"].startswith("S_s (eta_s)") or not o65["class_counts_unchanged_vs_R62"]             or o65["class_counts"] != ops["class_counts"]:
        BOXF_GATE_FAILS.append("R-65.1 operating space: basis tag or class counts changed")
    for i, band in enumerate(("One", "Two", "Three")):
        rec = o65["S50_sigma_lo"][i]
        for key, tag in (("film_supported_1_2", "Film"), ("full_film_1", "Full")):
            what = "film-supported (classes 1+2)" if tag == "Film" else "full-film (class 1)"
            d[f"BsNineOpsSFifty{tag}{band}"] = (rec[key]["S50"], f"operating space, basis: S_s (eta_s), theta_min band "
                                                f"{band.lower()} {rec['band_deg']}: duty S_s at which the {what} "
                                                "fraction is one half")
            d[f"BsNineOpsSFifty{tag}{band}CiLo"] = (rec[key]["bootstrap95"][0], "  ... its 95 % bootstrap, low")
            d[f"BsNineOpsSFifty{tag}{band}CiHi"] = (rec[key]["bootstrap95"][1], "  ... its 95 % bootstrap, high")
    for end, End in (("lo", "Lo"), ("hi", "Hi")):
        for k, name in (("1", "Full"), ("2", "Mixed"), ("3", "Contact"), ("4", "None")):
            d[f"BsNineOpsCount{name}{End}"] = (ops["class_counts"][end][k],
                                              f"operating space, sigma* {end}: points in class {k}")
    sv = ops["survivors_sigma_lo"]
    n1 = sum(sv["mineral"][k] for k in ("survives", "g2_only", "g3_only", "g2_and_g3"))
    d["BsNineOpsSurviveMinPct"] = (100 * sv["mineral"]["survives"] / n1,
                                   "% of full-film points (sigma* lo) surviving g2 and g3, mineral ceiling")
    d["BsNineOpsSurviveWgPct"] = (100 * sv["water_glycol"]["survives"] / n1,
                                  "% of full-film points (sigma* lo) surviving g2 and g3, water-glycol ceiling")
    rc = o65["rated_corner"]
    d["BsNineOpsCornerPoints"] = (rc["points"], "rated corner (E5 VG32): points placed on the duty axis")
    d["BsNineOpsCornerAboveFull"] = (rc["above_full_film_S50"]["eta_s"],
                                     "basis: S_s (eta_s): rated-corner points beyond their band's full-film S50")
    d["BsNineOpsCornerAboveFilm"] = (rc["above_film_supported_S50"]["eta_s"],
                                     "basis: S_s (eta_s): rated-corner points beyond their band's film-supported S50")
    # R-65.2: the map restricted to Deborah-valid points (g3), sigma* lo, on S_s
    r2 = o65["R65_2"]
    d["BsNineOpsGThreeValidPct"] = (r2["g3"]["lo"]["g3_valid_pct"],
                                    "% of points with an equilibrium (classes 1-3, sigma* lo) whose De_sq satisfies g3; "
                                    "class 4 kept whole (no De_sq without an equilibrium)")
    d["BsNineOpsSFiftyShiftValidPct"] = (r2["max_abs_shift_pct"],
                                         "largest |relative change| of the six S50s (S_s) when classes 1-3 are "
                                         f"restricted to g3-valid points, % (at {r2['max_shift_at']}; registered <= 10)")
    if not r2["registered_at_most_10pct"]:
        BOXF_GATE_FAILS.append("R-65.2: S50 shift on the g3-valid map exceeds 10 % (report; not a STOP)")

    # -- R-65.3: spin friction where P_diss converges (24 class-1 points, 3 meshes, signed, tight).
    # memory_sign signed; basis: smooth film, no Patir-Cheng shear factors; convergence class
    # "order >= 1 and Richardson correction <= 5 %" at the finest mesh (128x96).
    fr = json.loads((OUT / "R65_friction.report.json").read_text(encoding="utf-8"))
    if not fr["registered_delta_ge_0p1_order_ge_1_and_corr_le_5pct"] or fr["memory_sign"] != "signed":
        BOXF_GATE_FAILS.append("R-65.3: friction convergence registration no longer holds")
    _tag = (" [memory_sign signed; basis: smooth film, no Patir-Cheng shear factors; converged class: "
            "order >= 1, Richardson <= 5 %; T_f = P_diss/omega_spin, omega_spin = spin_ratio omega, solver.py:243]")
    d["BsNineFricN"] = (fr["converged_points"], "R-65.3: full-film points (of 24) whose P_diss converges" + _tag)
    d["BsNineFricOrderMin"] = (fr["delta_ge_0p1"]["order_min"], "R-65.3: smallest observed order of P_diss, delta >= 0.1")
    d["BsNineFricCorrMaxPct"] = (fr["delta_ge_0p1"]["corr_max_pct"],
                                 "R-65.3: largest Richardson correction of P_diss, delta >= 0.1, % of finest")
    for k, nm in (("median", "Med"), ("min", "Min"), ("max", "Max")):
        d[f"BsNineFricTorque{nm}Nm"] = (fr["T_f_Nm"][k], f"R-65.3: spin friction torque, {k} over converged points, N m" + _tag)
        d[f"BsNineFricMu{nm}"] = (fr["mu"][k], f"R-65.3: equivalent friction mu = T_f/(F_app R_b), {k} (smooth film: outside ERR-18 scope, boxe_errata.md:331-336)" + _tag)

    # -- R-66.5 (released by R-67.4): three further friction macros, from the R-65.3 artefact
    _fr = pd.read_parquet(OUT / "R65_friction.parquet")
    _nc = _fr[~_fr.converged_class.fillna(False).astype(bool)]
    if len(_fr) != 24 or len(_nc) != 1:
        BOXF_GATE_FAILS.append("R-66.5: friction sample is not 24 with one non-converged point")
    d["BsNineFricSample"] = (len(_fr), "R-65.3: full-film points sampled (theta_min tercile x delta quartile, 2 per cell)")
    d["BsNineFricDeltaSplit"] = (0.1, "R-65.3: the registered delta split (order >= 1 and Richardson <= 5 % for delta >= 0.1)")
    d["BsNineFricDeltaNonconv"] = (float(_nc.delta_boxD.iloc[0]),
                                   f"R-65.3: delta of the one non-converged point (s{int(_nc.sample_id.iloc[0])})")

    # -- R-66.2: E6 from the frozen box-B record (DERIVED from its artefact; not re-emitted
    # under D-8).  The artefact persists thermal_route, and it must equal the hand-back lines
    # campaign_handback_boxB.md:67-80 (1355 / 195 / 44 of 1594) and the OAT band
    # ceiling_extension_handback.md:20-26 (78-86 %).
    _B = pd.read_parquet(OUT / "audit_N8192_seed20260727.parquet", columns=["outcome", "thermal_route", "spin_ratio"])
    _Bc = _B[_B.outcome == "converged"]
    _rt = _Bc.thermal_route.value_counts().to_dict()
    if (len(_Bc), _rt.get("L1->L2_qmin"), _rt.get("L1"), _rt.get("L2_brinkman")) != (1594, 1355, 195, 44) \
            or set(_Bc.spin_ratio) != {1.0}:
        BOXF_GATE_FAILS.append("R-66.2: box-B routing no longer equals campaign_handback_boxB.md:67-80")
    _nl2 = int((_Bc.thermal_route != "L1").sum())
    d["BsNineESixBsN"] = (_nl2, "E6, box B (audit_N8192_seed20260727.parquet, spin_ratio 1.0): converged rows at Level 2")
    d["BsNineESixBsDen"] = (len(_Bc), "E6, box B: converged rows")
    d["BsNineESixBsPct"] = (100 * _nl2 / len(_Bc), "E6, box B: total Level-2 fraction (L2_brinkman + L1->L2_qmin), %")
    d["BsNineESixQminPct"] = (100 * _rt["L1->L2_qmin"] / len(_Bc), "E6, box B: L1->L2_qmin share of converged, %")
    _O = pd.read_parquet(OUT / "oat_block_n256.parquet", columns=["outcome", "thermal_route", "oat_param", "oat_value"])
    _Os = _O[(_O.oat_param == "spin_ratio") & (_O.outcome == "converged")]
    _band = _Os.groupby("oat_value").thermal_route.apply(lambda r: 100 * float((r != "L1").mean()))
    _bn = _Os.groupby("oat_value").size()
    if sorted(_band.index) != [0.0, 0.5, 1.0] or not (_bn == 50).all():
        BOXF_GATE_FAILS.append("R-66.2: the OAT spin cells are not three cells of 50 converged rows")
    d["BsNineESixBandLo"] = (float(_band.min()), "E6 band: OAT spin_ratio cells (oat_block_n256.parquet, a 256-sample "
                             "Sobol prefix per cell, 50 converged each), min Level-2 fraction, %")
    d["BsNineESixBandHi"] = (float(_band.max()), "E6 band: OAT spin_ratio cells, max (spin 1.0: 43/50, +-4.9 pt "
                             "binomial; box B's full population at spin 1.0 gives 87.8 %)")

    # -- R-66.4: the field paragraph (Fig. 5 point), from R65_fields.parquet and the vendored
    # first integral (scripts/p3_lee_fluid.newtonian_peak_deg, the integrals of p3_rim_asymptote._f)
    import p3_lee_fluid as _LF
    _Fd = pd.read_parquet(OUT / "R65_fields.parquet")
    _fm = json.loads(pathlib.Path("Z:/Papers/Paper 3/New/figures/fig_fields.json").read_text(encoding="utf-8"))
    _dl = _fm["h_min_bnd_m"] / _fm["c_equilibrium_m"]
    _i = int(_Fd.p_Pa.values.argmax())
    _cf = np.polyfit(_Fd.theta_deg.values[_i - 1:_i + 2], _Fd.p_Pa.values[_i - 1:_i + 2], 2)
    _pk_th = float(-_cf[1] / (2 * _cf[0]))
    _nw_th = _LF.newtonian_peak_deg(_dl, float(_fm["point"].split("theta_min = ")[1]))
    if abs(_nw_th - 33.8) > 0.5:
        BOXF_GATE_FAILS.append(f"R-66.4: Newtonian peak angle {_nw_th:.2f} deg is > 0.5 deg from 33.8 (STOP for that macro)")
    d["BsNineFieldsDelta"] = (_dl, "Fig. 5 point (E15 optimum): delta = h_min_bnd / c_equilibrium, signed 32x24")
    d["BsNineFieldsPeakThetaDeg"] = (round(_pk_th), f"Fig. 5 point: solver total-pressure peak angle, deg (parabolic "
                                     f"through the three cells about the maximum: {_pk_th:.2f})")
    d["BsNineFieldsPeakThetaNewtDeg"] = (_nw_th, "Fig. 5 point: arcsin sqrt(k), Newtonian squeeze first integral at "
                                         "(delta, theta_min), by quadrature; DERIVED")
    d["BsNineFieldsPeakMPa"] = (float(_Fd.p_Pa.max() / 1e6), "Fig. 5 point: solver peak film pressure, MPa (cell centres)")
    d["BsNineFieldsPeakNewtMPa"] = (float(_Fd.p_newtonian_eta0_Pa.max() / 1e6),
                                    "Fig. 5 point: Newtonian film at eta_0, same (c, eps), peak, MPa")
    d["BsNineFieldsEtaRatioMin"] = (float(_Fd.eta_over_eta0.min()), "Fig. 5 point: eta_bar/eta_0 over the land, min")
    d["BsNineFieldsEtaRatioMax"] = (float(_Fd.eta_over_eta0.max()), "Fig. 5 point: eta_bar/eta_0 over the land, max")
    d["BsNineFieldsSolventFrac"] = (float(_fm["eta_s_ratio"]), "Fig. 5 point: eta_s/eta_0 (sampled input)")
    # R-66.4 / R-67.2: the film-mean rise AFTER D-8, mineral class (scripts/fig_R65_fields.py
    # --thermal mineral, checked against R67_D8_claims.parquet's mineral row to 1e-9)
    _Fm = pd.read_parquet(OUT / "R67_fields_mineral.parquet")
    if set(_Fm.thermal_props) != {"mineral"}:
        BOXF_GATE_FAILS.append("R-67.2: the Fig. 5 mineral field file is not on the mineral basis")
    d["BsNineFieldsRiseMaxmK"] = (1e3 * float(_Fm.dT_film_mean_K.max()),
                                  "Fig. 5 point: largest film-mean rise, mK, thermal_props: by_class (D-8), mineral "
                                  "(rho 828.5, c_p 2000 CITED Totten2011 citation_verified: false, k_f 0.15)")

    # -- R-67.4: Lee et al.'s fluid, CITED / DERIVED (LeeShangZhang2026 Table 1, Eq. 2, T in K)
    _TK = 373.15
    d["BsNineLeeCUm"] = (15, "CITED LeeShangZhang2026: baseline nominal clearance, um (Table 2); radial: Eq. 26 gives "
                             "h_rigid = c - X_ecc.n, c = R_s - R_p")
    d["BsNineLeeEtaMPas"] = (1e3 * float(_LF.mu_lee(_TK)), "DERIVED from LeeShangZhang2026 Table 1 and Eq. (2) at "
                             "373.15 K, p = 0: viscosity, mPa s (39.7 mPa s at 40 C: VG46)")
    d["BsNineLeeAlphaStar"] = (1e9 * _LF.alpha_star(_TK), "DERIVED from LeeShangZhang2026 Eq. (2) at 100 C: Roelands-"
                               "equivalent 1/int eta(0)/eta(p) dp, 1/GPa")
    d["BsNineLeePiezoHundred"] = (float(_LF.roelands_ratio(_TK, 1e8)), "DERIVED from LeeShangZhang2026 Eq. (2) at "
                                  "100 C: eta(100 MPa)/eta(0)")
    d["BsNineLeePiezoPeak"] = (float(_LF.roelands_ratio(_TK, _Fd.p_Pa.max())), "DERIVED from LeeShangZhang2026 Eq. (2) "
                               "at 100 C: eta/eta(0) at the Fig. 5 solver peak pressure")

    # -- R-68.1: D-9 in ratios (R68_piezo_ratio.report.json; Newtonian path at eta_s, exact transform;
    # Barus law in G = alpha p_ch, Roelands VG46 = LeeShangZhang2026 Eq. 2 at 100 C)
    pz = json.loads((OUT / "R68_piezo_ratio.report.json").read_text(encoding="utf-8"))
    _rg = pz["registered"]
    if not (all(_rg["1_kappa_within_3pct_of_fit"].values()) and all(_rg["2_linear_law"].values())
            and _rg["3_ratio_test_below_1e-3"] and _rg["4_c0_optimum_interior"]):
        BOXF_GATE_FAILS.append("R-68.1: a D-9 ratio registration no longer holds")
    _pp = pz["points"]
    d["BsNinePiezoGLee"] = (pz["G_lee_roelands"], "fluid_law roelands_LeeVG46: G* = alpha* p_ch at the rated corner "
                            "(alpha* 1/int eta(0)/eta(p) dp at 100 C, DERIVED from LeeShangZhang2026 Eq. 2)")
    d["BsNinePiezoGLeeBarus"] = (pz["G_lee_barus_equivalent"], "fluid_law barus: the G giving Lee's Roelands lift at E15")
    for _p, _t in (("E15", "EFifteen"), ("E13", "EThirteen"), ("E5", "EFive")):
        d[f"BsNinePiezoGcrit{_t}"] = (_pp[_p]["G_crit_exact"], f"fluid_law barus: G at which Lambda = 3, {_p}, exact sweep")
        d[f"BsNinePiezoFactor{_t}"] = (_pp[_p]["factor_roelands"], f"fluid_law roelands_LeeVG46: Lambda_pv/Lambda_iso, {_p}")
        d[f"BsNinePiezoLam{_t}"] = (_pp[_p]["Lam_roelands"], f"fluid_law roelands_LeeVG46: Lambda, {_p} (32x24)")
    d["BsNinePiezoKappa"] = (_pp["E15"]["kappa"], "fluid_law barus: kappa by the first-order functional, E15")
    d["BsNinePiezoKappaFit"] = (_pp["E15"]["kappa_fit"], "fluid_law barus: slope of the exact sweep (G <= 0.7), E15")
    d["BsNinePiezoLinDevPct"] = (_pp["E15"]["lin_dev_max_pct_G_le_0.7"],
                                 "fluid_law barus: max deviation of 1 + kappa G from the exact sweep, E15, G <= 0.7, %")
    d["BsNinePiezoQPeakFrac"] = (_pp["E15"]["q_peak_over_q_inf_roelands"],
                                 "fluid_law roelands_LeeVG46: q_peak / q_inf at E15")
    _ax = pz["c0_axis"]["roelands_LeeVG46"]
    d["BsNinePiezoCZeroRatio"] = (_ax["c0_ratio_to_iso"], "fluid_law roelands_LeeVG46: c0*_pv / c0*_iso at the E15 arm "
                                  "(grid optima, 0.25 um steps; refined ratio in the report)")
    d["BsNinePiezoLamStar"] = (_ax["Lam_star"], "fluid_law roelands_LeeVG46: Lambda*_pv on the c0 axis (1-6 um)")
    d["BsNinePiezoPeakOverPch"] = (_pp["E15"]["p_peak_over_p_ch_iso"], "fluid_law isoviscous: p_peak / p_ch at E15")

    # -- R-68.3: E6 on the common property basis (R67_E6_common.report.json; box B's stored solutions,
    # Brinkman k_f = 0.13 on both sides, flags independent of routing precedence).  The slipper's
    # q_min flag cannot be recomputed independently: its guard used P_diss_l1 (ar_bo_campaign.py:421-428),
    # which the corrected parquet does not persist, so its 3.53 % stays a reroute share.
    e6c = json.loads((OUT / "R67_E6_common.report.json").read_text(encoding="utf-8"))
    if not (e6c["reproduces_stored_routes_at_k_0.60"] and all(e6c["registered"].values())):
        BOXF_GATE_FAILS.append("R-68.3: the E6 common-basis post-process no longer reproduces box B or its registrations")
    _cb = e6c["k_f_0.13_common_basis"]
    d["BsNineESixCommonTotalPct"] = (_cb["total_L2_pct"], "E6 common basis (k_f 0.13): box B total Level-2 fraction, "
                                     "on the stored solutions, %")
    d["BsNineBsBrinkCommonPct"] = (_cb["brinkman_flag_pct"], "E6 common basis: box B Brinkman-flag share (independent "
                                   "of precedence), %")
    d["BsNineBsQminFlagPct"] = (_cb["qmin_flag_pct"], "E6: box B q_min-flag share (independent of precedence), %")

    # -- R-68.2: D-8 on the mineral class (R68_D8_reemit.report.json; fixed-solution rescale with the
    # route recomputed, the re-routed rows' L2 rise computed, box D's 70 at-risk rows re-solved).
    # thermal_props: by_class (D-8), mineral; water-glycol stays on the defaults (one end of its range)
    # until Totten's density.  Paper 3's unsuffixed aliases of these are withdrawn (R-68.2).
    rm = json.loads((OUT / "R68_D8_reemit.report.json").read_text(encoding="utf-8"))
    if not all(rm["registered"].values()):
        BOXF_GATE_FAILS.append("R-68.2: a D-8 re-emission registration no longer holds (report; not a STOP)")
    _tp = "thermal_props: by_class (D-8), mineral (rho 828.5, c_p 2000, k_f 0.15; c_p and k_f CITED Totten2011, " \
          "citation_verified: false); first_order_fixed_solution except box D's re-solved rows"
    for _k in ("ThetaFlowMed", "ThetaFlowAbovePct", "BsBrink", "BsQmin", "NahmeAdiabLoPct", "NahmeAdiabHiPct",
               "OpsSurviveMinPct", "ETwoCeilingReachPct", "ETwoGapPp", "ThermClearNm"):
        d[f"BsNine{_k}Min"] = (rm[f"{_k}Min"], f"{_k} on the mineral class; {_tp}")
    # R-69.2: the further class changes on the mineral properties (R69_resolve.report.json): class-3 points
    # whose floor arms lose their equilibrium (-> 4), and points whose pure film, failed only after the closure
    # widened c, now carries the load at the floor (-> 2: delta ~ 0.010, Lambda << 3)
    r69 = json.loads((OUT / "R69_resolve.report.json").read_text(encoding="utf-8"))
    _cc = {e: {int(k): int(v) for k, v in rm["class_counts_min"][e].items()} for e in ("lo", "hi")}
    for _c in r69["class_changes"]:
        _new = 4 if _c["new"] == 4 else 2
        _cc[_c["end"]][int(_c["old"])] -= 1
        _cc[_c["end"]][_new] = _cc[_c["end"]].get(_new, 0) + 1
    for _end, _E in (("lo", "Lo"), ("hi", "Hi")):
        for _ki, _nm in ((1, "Full"), (2, "Mixed"), (3, "Contact"), (4, "None")):
            d[f"BsNineOpsCount{_nm}{_E}Min"] = (_cc[_end].get(_ki, 0),
                                                f"operating space, sigma* {_end}, class {_ki}, mineral (R-68.2's 70 re-solved "
                                                "rows and R-69.2's 79 re-solves; no pure-film equilibrium counted 4)")
    for _k, _v in rm["S50_min"].items():
        d[f"BsNine{_k}Min"] = (_v["S50_min"], f"{_k} on S_s with the mineral-class re-solved classes")

    # -- R-69.3: the thermal bracket (R69_thermal_bracket.report.json).  Two wall idealisations:
    # Cond = L2 film-mean rise with both walls at T_in; Adv = P_diss/(rho c_p Q_out), adiabatic walls.
    # Classes: Min = mineral (D-8; c_p, k_f CITED Totten2011 citation_verified false; 110 C);
    # Wg = water defaults (the water-glycol end; 60 C).  Stored solutions; in-domain = converged,
    # De_sq_bnd <= De_sq_max.  The dominant constraint's NAME is in the provenance.
    tb = json.loads((OUT / "R69_thermal_bracket.report.json").read_text(encoding="utf-8"))
    _QTY = [('G2BreachPct', 'GTwoBreachPct', 'share of in-domain evaluations breaching g2, %'), ('G2SolePct', 'GTwoSolePct', 'E3 construction (T_in below the ceiling, g1 and g3 met at sigma* lo): g2 the sole violation, %'), ('NahmeLoPct', 'NahmeLoPct', 'Nahme share on the mean rise, Walther b_min, %'), ('NahmeHiPct', 'NahmeHiPct', 'Nahme share on the mean rise, Walther b_max, %'), ('BindDomPct_lo', 'BindDomPctLo', 'E2 construction (box E, T_in < 60 C), sigma* lo: share of the most frequent single binding constraint, %'), ('BindDomPct_hi', 'BindDomPctHi', 'the same at sigma* hi, %'), ('RiseMed', 'RiseMed', 'median rise over in-domain evaluations, K'), ('OpsSurvivePct', 'OpsSurvivePct', 'box D class 1 (sigma* lo) surviving g2 and g3, %')]
    _SIX = [('NahmeSixLoPct', 'NahmeSixLoPct', 'Nahme share at x6 (one adiabatic wall), Walther b_min, %'), ('NahmeSixHiPct', 'NahmeSixHiPct', 'Nahme share at x6, Walther b_max, %')]
    for _ce, _st in tb["stats"].items():
        for _k, _nm, _what in _QTY + (_SIX if _ce.endswith("Cond") else []):
            _src = f"thermal bracket {_ce}: {_what}"
            if _k.startswith("BindDomPct"):
                _src += f" (dominant: {_st['BindDom_' + _k[-2:]]})"
            d[f"BsNineTherm{_nm}{_ce}"] = (_st[_k], _src)
    d["BsNineThetaFlowMedWg"] = (tb["ThetaFlow"]["Wg"]["med"], "through-flow number Theta, median, water defaults (as run)")
    d["BsNineThetaFlowAbovePctWg"] = (100 * json.loads((OUT / "R57_thermal_by_route.report.json").read_text())
                                      ["through_flow_group"]["share_Theta_ge_1"], "% of converged rows with Theta >= 1, water defaults")

    # -- R-70.1: the water-glycol pressure-viscosity coefficient.
    # (a) DERIVED (digitised) from Wang2024 Fig. 4(b), 2 C, 0-115 MPa: end-to-end Barus equivalents of
    #     curves that are not exponential (data/wang2024_fig4b_digitised.csv, read by eye at 300 dpi).
    _wd = pd.read_csv(ROOT / "data" / "wang2024_fig4b_digitised.csv", comment="#")
    def _barus(fl):
        _f = _wd[_wd.fluid == fl].sort_values("pressure_MPa")
        return float(np.log(_f.nu_mm2_s.iloc[-1] / _f.nu_mm2_s.iloc[0]) / (_f.pressure_MPa.iloc[-1] - _f.pressure_MPa.iloc[0]) * 1e3)
    _pch = 5e7
    d["BsNineWangTempC"] = (2, "Wang2024 Fig. 4(b): temperature of the pressure series, C")
    d["BsNineWangPMaxMPa"] = (115, "Wang2024 Fig. 4(b): largest pressure of the series, MPa")
    d["BsNinePiezoAlphaWgDig"] = (_barus("water_glycol"), "DERIVED (digitised) Wang2024 Fig. 4(b), 2 C, 0-115 MPa: "
                                  "water-glycol, end-to-end Barus equivalent (kinematic), 1/GPa")
    d["BsNinePiezoGWgDig"] = (_barus("water_glycol") * 1e-9 * _pch, "DERIVED (digitised): alpha_WG p_ch at p_ch = 50 MPa")
    d["BsNinePiezoAlphaTenDig"] = (_barus("oil_10"), "DERIVED (digitised) Wang2024 Fig. 4(b): 10# mineral oil, 1/GPa")
    d["BsNinePiezoAlphaTwentyTwoDig"] = (_barus("oil_22"), "DERIVED (digitised) Wang2024 Fig. 4(b): 22# mineral oil, 1/GPa")
    # (b) CITED Hasan2025 Table 2 (falling-body high-pressure viscometer, 40 C, to 0.5 GPa): glycol-water
    #     A1 and B1 (50-60 % water, Table 1), reciprocal asymptotic isoviscous coefficient alpha*_0.5GPa.
    d["BsNineHasanTempC"] = (40, "CITED Hasan2025 Table 2: temperature of the viscometer coefficients, C")
    d["BsNineHasanPMaxGPa"] = (0.5, "CITED Hasan2025 Sec. 2.3: viscometer pressure range, GPa")
    d["BsNinePiezoAlphaWgLo"] = (2.2, "CITED Hasan2025 Table 2: alpha*_0.5GPa, glycol-water A1 (50-60 % water), 1/GPa")
    d["BsNinePiezoAlphaWgHi"] = (2.93, "CITED Hasan2025 Table 2: alpha*_0.5GPa, glycol-water B1 (50-60 % water), 1/GPa")
    d["BsNinePiezoGWg"] = (2.2e-9 * _pch, "CITED Hasan2025 Table 2 (A1): alpha*_WG p_ch at p_ch = 50 MPa")
    d["BsNinePiezoGWgHi"] = (2.93e-9 * _pch, "CITED Hasan2025 Table 2 (B1): alpha*_WG p_ch at p_ch = 50 MPa")

    # -- R-70.2: model-section thresholds (file:line)
    from ball_socket import thermal as _bt
    from ball_socket.campaign_config_v4 import ConstraintSet as _CS
    d["BsNineBrinkThresh"] = (_bt.BR_THRESHOLD, "Brinkman L1/L2 cutoff, ball_socket/thermal.py:27")
    d["BsNineBrinkDeltaT"] = (_bt.BR_DT_REF_K, "Brinkman reference rise, K, ball_socket/thermal.py:28")
    d["BsNineThetaDeltaTRef"] = (_bt.DELTA_T_SAFETY_K, "through-flow reference rise, K, ball_socket/thermal.py:26")
    d["BsNineDesqMax"] = (_CS().De_sq_max, "Deborah validity limit g3, ball_socket/campaign_config_v4.py:116")

    # -- R-72: the macros v7 calls, the claim-point inputs, counts and denominators, E16 (and E17 below)
    x72 = json.loads((OUT / "R72_extract.report.json").read_text(encoding="utf-8"))
    from ball_socket import mixed as mx
    for _k, (_v, _src) in x72["numerals"].items():                     # R-72.4 typed numerals, file:line
        d[f"BsNine{_k}"] = (float(_v), f"DERIVED (registered numeral): {_src}")
    d["BsNineAdiabFactor"] = (6, d["BsNineAdiabFactor"][1] + " -- adiabatic-wall bound over the L2 mean rise")
    _cp = x72["claim_points"]
    for _n, _P in (("EFifteen", "E15"), ("EThirteen", "E13"), ("EFive", "E5")):
        _x = _cp[_P]["x"]
        d[f"BsNine{_n}Desq"] = (_cp[_P]["De_sq"], f"DERIVED: De_sq at the {_P} claim point -- {_cp[_P]['source']}")
        d[f"BsNine{_n}CZeroUm"] = (_x["c0"] * 1e6, f"DERIVED: c0 at the {_P} claim point, um ({_cp[_P]['source']})")
        d[f"BsNine{_n}ThetaMinDeg"] = (_x["theta_min_deg"], f"DERIVED: theta_min at the {_P} claim point, deg")
    _x = _cp["E15"]["x"]
    d["BsNineEFifteenEtaZeroMPas"] = (_x["eta_0"] * 1e3, "DERIVED: eta_0 at the E15 claim point (ASTM D341 at T_in), mPa s")
    d["BsNineEFifteenLamS"] = (_x["lam"], "DERIVED: lambda at the E15 claim point, s")
    d["BsNineEFifteenTinC"] = (_x["T_in"], "DERIVED: T_in at the E15 claim point, C")
    d["BsNineEFifteenPchMPa"] = (_x["p_ch"] / 1e6, "DERIVED: p_ch at the E15 claim point (and E13, E5), MPa")
    d["BsNineEFifteenOmega"] = (_x["omega"], "DERIVED: omega at the E15 claim point (and E13, E5), rad/s")
    d["BsNineEFifteenSolventFrac"] = (_x["eta_s_ratio"], "DERIVED: eta_s/eta_0 at the E15 claim point (and E13, E5)")
    if any(_cp[_P]["x"][_a] != _x[_a] for _P in ("E13", "E5") for _a in ("omega", "p_ch", "T_in", "eta_0", "lam", "eta_s_ratio")):
        BOXF_GATE_FAILS.append("R-72.2: the three claim points no longer share omega, p_ch, T_in, eta_0, lambda, eta_s/eta_0")
    _pz = x72["E15_desq_piezo"]
    d["BsNineEFifteenDesqPiezo"] = (_pz["De_sq_piezo"], "DERIVED: E15 De_sq x the Roelands factor at the Fig. 5 peak "
                                    f"({_pz['roelands_factor_at_fig5_peak']:.4g}), {_pz['note']}")
    _fo = x72["film_only_both"]
    if not _fo["lo_subset_of_hi"]:
        BOXF_GATE_FAILS.append("R-72.4: the class-3 film-only set at sigma lo is not a subset of sigma hi")
    d["BsNineOpsFilmOnlyBoth"] = (_fo["n"], "DERIVED: box D points film-only (class 1) at BOTH sigma* ends")
    d["BsNineOpsClassThreeLo"] = (_fo["class3_lo"], "DERIVED: box D class-3 points at sigma* lo")
    d["BsNineOpsClassThreeHi"] = (_fo["class3_hi"], "DERIVED: box D class-3 points at sigma* hi")
    _va = x72["valid"]
    d["BsNineValidRows"] = (_va["n_rows"], "DERIVED: box E corrected evaluations (rows)")
    d["BsNineValidN"] = (_va["n_valid"], "DERIVED: valid evaluations -- " + _va["criterion"])
    d["BsNineValidExcl"] = (_va["excluded"], "DERIVED: excluded evaluations, all nonconverged/picard_at_root")
    d["BsNineValidCompExcl"] = (_va["composite_further_exclusions"], "DERIVED: composite, further exclusions -- "
                                + _va["composite_criterion"])
    _td = x72["table2_denominators"]
    d["BsNineTabDenInDomain"] = (_td["rows_1_2_6_in_domain"], "DERIVED: Table 2 rows 1, 2, 6 denominator (in-domain)")
    d["BsNineTabDenFrameMin"] = (_td["row_3_frame"]["Min"], "DERIVED: Table 2 row 3 frame, mineral")
    d["BsNineTabDenFrameWg"] = (_td["row_3_frame"]["Wg"], "DERIVED: Table 2 row 3 frame, water-glycol")
    d["BsNineTabDenRestricted"] = (_td["rows_4_5_restricted"], "DERIVED: Table 2 rows 4-5 denominator (T_in < 60 C)")
    d["BsNineTabDenClassOne"] = (_td["row_7_class1_smooth"]["Min"], "DERIVED: Table 2 row 7, box D class 1 at sigma* lo")
    _rc = x72["rated_corner_set"]
    d["BsNineRatedPoints"] = (_rc["points"], "DERIVED: rated-corner set, points -- " + _rc["definition"])
    d["BsNineRatedRows"] = (_rc["rows"], "DERIVED: rated-corner set, rows (points x arms)")
    d["BsNineContactModels"] = (len(mx.registered_contact_models("psi", "derived")),
                                "DERIVED: registered contact models, ball_socket/mixed.py:651 (2 x 2 x 2 x 2)")
    _f6 = x72["fig6b_pair"]
    d["BsNineOpsSurviveMinPctRun"] = (_f6["mineral_pct_plotted"], "DERIVED: Fig. 6b mineral %, " + _f6["basis"])
    d["BsNineOpsSurviveWgPctRun"] = (_f6["water_glycol_pct_plotted"], "DERIVED: Fig. 6b water-glycol %, " + _f6["basis"])
    e16 = json.loads((OUT / "R72_E16.report.json").read_text(encoding="utf-8"))
    if e16["STOP"]:
        BOXF_GATE_FAILS.append("R-72.5 E16: (a) no longer holds (the local maximum is off the rim or off De_sq/delta)")
    d["BsNineDesqLocMed"] = (e16["b_median"], "DERIVED: E16 local Deborah lambda|h_dot|/h at the rim, median over the "
                             "137 in-domain E4 points (R72_E16.report.json)")
    d["BsNineDesqLocQThree"] = (e16["b_q3"], "DERIVED: E16 local Deborah, upper quartile")
    d["BsNineDesqLocMax"] = (e16["b_max"], "DERIVED: E16 local Deborah, maximum")
    d["BsNineDesqLocAbovePct"] = (e16["b_above_De_sq_max_pct"], "DERIVED: E16, % of the 137 points with the local "
                                  "Deborah above De_sq_max = 0.2")
    # R-72.6, E17: the stroke-integrated squeeze (R72_E17_stroke.report.json; rigid, isothermal, Newtonian at
    # eta_s, F_req constant over omega t in [0, pi]; angles in degrees of omega t)
    e17 = json.loads((OUT / "R72_E17_stroke.report.json").read_text(encoding="utf-8"))
    if e17["STOP"]:
        BOXF_GATE_FAILS.append("R-72.6 E17: (a) consistency with the v6 snapshot no longer holds")
    _src17 = "R72_E17_stroke.report.json"
    _deg = 180.0 / np.pi
    for _n, _P in (("EFifteen", "E15"), ("EThirteen", "E13"), ("EFive", "E5")):
        _pr = e17["points"][_P]
        _iso, _roe = _pr["laws"]["isoviscous"], _pr["laws"]["roelands_LeeVG46"]
        d[f"BsNineStroke{_n}SnapDevPct"] = (100 * _pr["a_rel_diff"], f"DERIVED: E17 (a), {_P}: delta at r = eps (G = 0) "
                                            f"against the v6 snapshot delta, % ({_src17})")
        d[f"BsNineStroke{_n}TSnapDeg"] = (_deg * _iso["concentric"]["t_pass_snapshot"], f"DERIVED: E17, {_P}, isoviscous "
                                          "from concentric: omega t at which the snapshot film is passed, deg")
        d[f"BsNineStroke{_n}TcDeg"] = (_deg * _pr["closure_G0"]["t_c"], f"DERIVED: E17 (b), {_P}: isoviscous closure "
                                       "omega t_c from concentric (grid floor plus power-law tail), deg")
        d[f"BsNineStroke{_n}TLamThreeRoelDeg"] = (_deg * _roe["concentric"]["t_Lambda3"], f"DERIVED: E17, {_P}, Roelands "
                                                  "VG46 from concentric: omega t at Lambda = 3, deg")
        d[f"BsNineStroke{_n}LamPiRoel"] = (_roe["concentric"]["Lambda_pi"], f"DERIVED: E17 (d), {_P}, Roelands VG46 from "
                                           "concentric: Lambda at omega t = pi")
        d[f"BsNineStroke{_n}LamPiRoelSnap"] = (_roe["snapshot"]["Lambda_pi"], f"DERIVED: E17, {_P}, Roelands VG46 from "
                                               "the snapshot: Lambda at omega t = pi")
        d[f"BsNineStroke{_n}LamPiGTwo"] = (_pr["laws"]["barus_G2.0"]["concentric"]["Lambda_pi"], f"DERIVED: E17 (e), "
                                           f"{_P}, Barus G = 2 (the largest swept) from concentric: Lambda at omega t = pi; "
                                           f"smallest G with Lambda(pi) >= 3: {_pr['smallest_G_Lambda_pi_ge_3_concentric']}")
        if _pr["smallest_G_Lambda_pi_ge_3_concentric"] != "none <= 2":
            BOXF_GATE_FAILS.append(f"R-72.6 E17 (e): at {_P} a swept G now reaches Lambda(pi) >= 3; StrokeLamPiGTwo no longer "
                                   "carries 'none <= 2'")
    _pr = e17["points"]["E15"]
    _b63 = _pr["laws"]["barus_G0.63"]
    d["BsNineStrokeEFifteenTHundredthDeg"] = (_deg * _pr["laws"]["isoviscous"]["concentric"]["t_delta_0p01"],
                                              "DERIVED: E17 (b), E15, isoviscous from concentric: omega t at delta = 0.01, deg")
    d["BsNineStrokeEFifteenLamPiIso"] = (_pr["laws"]["isoviscous"]["concentric"]["Lambda_pi"],
                                         "DERIVED: E17, E15, isoviscous from concentric: Lambda at omega t = pi")
    d["BsNineStrokeEFifteenLamPiBarus"] = (_b63["concentric"]["Lambda_pi"], "DERIVED: E17, E15, Barus G = 0.63 from "
                                           "concentric: Lambda at omega t = pi")
    d["BsNineStrokeEFifteenTLamThreeBarusDeg"] = (_deg * _b63["concentric"]["t_Lambda3"], "DERIVED: E17, E15, Barus "
                                                  "G = 0.63 from concentric: omega t at Lambda = 3, deg")
    d["BsNineStrokeEFifteenRDeltaBarus"] = (_b63["r_over_delta_decade_above"], "DERIVED: E17 (c), E15, Barus G = 0.63: "
                                            "r/delta at delta = 0.02 (one decade above the grid floor)")
    d["BsNineStrokeEFifteenRDeltaBarusFloor"] = (_b63["r_over_delta_at_min"], "DERIVED: E17 (c), E15, Barus G = 0.63: "
                                                 "r/delta at the grid floor delta = 0.002")
    d["BsNineStrokeTailPower"] = (_pr["closure_G0"]["tail_power_b"], "DERIVED: E17, E15, isoviscous: exponent b of "
                                  "r ~ delta^b on delta <= 0.02 (b < 1: closure in finite time)")
    d["BsNineStrokeGridFloor"] = (e17["delta_grid"][1], "DERIVED: E17 delta grid floor (scripts/run_R72_E17_stroke.py:53)")
    d["BsNineStrokeGMax"] = (2.0, "DERIVED: E17 largest swept Barus G (scripts/run_R72_E17_stroke.py:52)")

    # -- E3 restated (R-48.1): how much of 'g2 binds water-glycol' is the inlet --
    from ball_socket.campaign_config_v4 import ConstraintSet
    cs = ConstraintSet()
    wg = cs.T_max_C_by_class["water_glycol"]
    cvv = M[M.outcome == "converged"]
    g2f = cvv.T_max_bnd > wg
    d["BsNineEThreeWgFails"] = (int(g2f.sum()), "converged rows failing the 60 C ceiling")
    d["BsNineEThreeInletPct"] = (_pct((g2f & (cvv.T_in >= wg)).sum(), g2f.sum()),
                                 "% of water-glycol g2 failures whose INLET is >= 60 C")
    only = (g2f & (cvv.h_min_bnd / lo >= cs.lambda_ratio_min)
            & (cvv.De_sq_bnd <= cs.De_sq_max))
    d["BsNineEThreeOnlyGTwoLo"] = (int(only.sum()),
                                   "sigma lo: rows failing ONLY the water-glycol ceiling")
    d["BsNineEThreeHeatedLo"] = (int((only & (cvv.T_in < wg)).sum()),
                                 "of those, heated across 60 C from a sub-60 inlet")

    # -- E3 restricted (R-49.2): the population where E3 can be refuted --------
    e3 = json.loads((OUT / "E3_restricted.json").read_text())
    if not e3["premise_rise_nonnegative"]["holds"]:
        BOXF_GATE_FAILS.append("E3: a negative dissipative rise exists, so the "
                               "'fails by arithmetic' exclusion no longer holds")
    rl, rh = (e3["bands"][b]["restricted_T_in_below_ceiling"] for b in ("lo", "hi"))
    ul, uh = (e3["bands"][b]["unrestricted"] for b in ("lo", "hi"))
    d["BsNineEThreeForcedLo"] = (ul["g2_sole_binding_with_T_in_at_or_above_ceiling"],
                                 "sigma lo: sole-g2 rows forced by T_in >= 60 C")
    d["BsNineEThreeForcedHi"] = (uh["g2_sole_binding_with_T_in_at_or_above_ceiling"],
                                 "sigma hi: sole-g2 rows forced by T_in >= 60 C")
    d["BsNineEThreeSoleHi"] = (rh["g2_sole_binding"],
                               "sigma hi, T_in < 60 C: sole-g2 rows")
    d["BsNineEThreeSoleTinMin"] = (min(rl["min_T_in_among_g2_sole_C"],
                                       rh["min_T_in_among_g2_sole_C"]),
                                   "lowest inlet among restricted sole-g2 rows, C")
    d["BsNineEThreeRiseMax"] = (max(rl["max_rise_among_g2_sole_K"],
                                    rh["max_rise_among_g2_sole_K"]),
                                "largest dissipative rise among restricted sole-g2 rows, K")
    # A1 (R-50 hand-back): the 24 arms of a sample agree on T_max to < 0.01 K,
    # so the restricted sole-g2 rows are whole samples.  Quote samples.
    bs = json.loads((OUT / "E3_bias_sensitivity.json").read_text())["bands"]
    d["BsNineEThreeSoleSamplesLo"] = (bs["lo"]["as_computed"]["samples"],
                                      "sigma lo, T_in < 60 C: SAMPLES with sole-g2 binding")
    d["BsNineEThreeSoleSamplesHi"] = (bs["hi"]["as_computed"]["samples"],
                                      "sigma hi, T_in < 60 C: SAMPLES with sole-g2 binding")
    d["BsNineEThreeBiasDownLo"] = (bs["lo"]["down_2x"]["samples"],
                                   "sigma lo: samples surviving the rise scaled DOWN by 2x the bias")
    d["BsNineEThreeBiasDownHi"] = (bs["hi"]["down_2x"]["samples"],
                                   "sigma hi: samples surviving the rise scaled DOWN by 2x the bias")
    d["BsNineTinAboveWgPct"] = (100 * e3["T_in_axis_share_at_or_above_ceiling"],
                                "% of the registered T_in axis at or above 60 C")

    # -- R-48.6 / R-49.5: load capacity vs area at a common rim gap ------------
    fz_ = pd.read_parquet(OUT / "R48_film_at_common_eps.parquet")
    fz_ = fz_[fz_["mode"] == "common_psi"]

    def _ratio(c0, psi, area):
        g = fz_[(fz_.c0_um == c0) & (fz_.level == psi)].set_index("theta_min_deg")
        return float((g.F_film[40.0] / g.F_film[20.0]) / (g[area][40.0] / g[area][20.0]))
    d["BsNineFilmAreaClosed"] = (_ratio(2.0, 0.7, "A_proj"),
                                 "F_film ratio over projected-area ratio, 40 vs 20 deg, "
                                 "c0 = 2 um, psi = 0.7 (closed rim, corner-like)")
    d["BsNineFilmAreaOpen"] = (_ratio(10.0, 0.1, "A_sph"),
                               "F_film ratio over spherical-area ratio, 40 vs 20 deg, "
                               "c0 = 10 um, psi = 0.1 (thick-film control) -- SUPERSEDED for "
                               "the paper by BsNineFilmAreaOpenProj (R-55.2 R3: one area basis)")
    d["BsNineFilmAreaOpenProj"] = (_ratio(10.0, 0.1, "A_proj"),
                                   "F_film ratio over PROJECTED-area ratio, 40 vs 20 deg, c0 = 10 um, "
                                   "psi = 0.1 (thick-film control; same basis as FilmAreaClosed; "
                                   "spherical-basis value kept as BsNineFilmAreaOpen)")
    r3 = json.loads((OUT / "R57_R3.report.json").read_text())["b_thick_film_control"]
    if abs(r3["ratio_full_over_sph_area"] / d["BsNineFilmAreaOpen"][0] - 1) > 1e-9:
        BOXF_GATE_FAILS.append("R3: the decomposition does not reproduce BsNineFilmAreaOpen")
    d["BsNineFilmAreaOpenHydPct"] = (100 * min(r3["20"]["hyd_share"], r3["40"]["hyd_share"]),
                                     "thick-film control: smaller hydrostatic share of F_film "
                                     "over theta_min = 20, 40 deg, % (R-55.2 R3)")

    if fb["binding_cell"] != "sigma_hi|water_glycol":
        raise AssertionError(f"the prose names sigma_hi/water-glycol as the binding "
                             f"cell; the artefact says {fb['binding_cell']}")


def derive_paper3_model(d: dict) -> None:
    """Numbers Paper 3 sections 2, 3 and 5.1-5.2 cite that no box F macro
    carried: geometry, the arm count, E4's load split, the saturation
    exponents and the memory share.  Surfaced by drafting (R-48.12.6)."""
    import ball_socket.mixed as mx
    from ball_socket import representative as rep_
    d["BsNineRbMm"] = (1e3 * rep_.R_B_M, "representative ball radius, mm")
    fz = json.loads((ROOT / "ball_socket" / "campaign_frozen_v6.json"
                     ).read_text(encoding="utf-8"))
    ax = {a["name"]: a for a in fz["config"]["input_space"]}
    d["BsNineThetaMinLoDeg"] = (ax["theta_min_deg"]["lower"], "registered theta_min floor")
    d["BsNineThetaMinHiDeg"] = (ax["theta_min_deg"]["upper"], "registered theta_min ceiling")
    d["BsNineCZeroLoUm"] = (1e6 * ax["c0"]["lower"], "registered c0 floor, um")
    d["BsNineCZeroHiUm"] = (1e6 * ax["c0"]["upper"], "registered c0 ceiling, um")
    from ball_socket.geometry import BallSocketGeometry
    d["BsNineThetaMaxDeg"] = (float(np.rad2deg(BallSocketGeometry(c0=1e-5).theta_max)),
                              "land outer edge theta_max, deg")
    for deg, tag in ((20, "Twenty"), (30, "Thirty"), (40, "Forty")):
        d[f"BsNineEpsCeil{tag}"] = (float(1 / np.cos(np.deg2rad(deg))),
                                    f"geometric closure eps = 1/cos({deg} deg)")
    d["BsNineContactArms"] = (len(mx.expand_for_solving(mx.registered_contact_models(
        branch="psi", summit_convention="derived"))), "registered contact arms")

    # E4 at the cap: squeeze vs hydrostatic share of the film load (509 rows
    # whose full solve reproduces the stored film load to 1e-9).
    r = pd.read_parquet(OUT / "analysis_E7_load_decomposition_32x24.parquet")
    r = r[np.abs(r.F_full / r.F_film_stored - 1) <= 1e-9]
    d["BsNineEFourRows"] = (len(r), "load decomposition rows reproducing to 1e-9")
    d["BsNineEFourSqueezePct"] = (100 * float((r.F_squeeze / r.F_full).median()),
                                  "median squeeze share of the film load at the cap, %")
    d["BsNineEFourHydroPct"] = (100 * float((r.F_hydrostatic / r.F_full).median()),
                                "median hydrostatic share of the film load at the cap, %")
    d["BsNineEFourSpinMaxN"] = (float(r.F_spin.abs().max()), "max |F_spin|, N")

    def _exp(a, b):
        return float((np.log(r[f"F_sq_d{b}"] / r[f"F_sq_d{a}"]) / np.log(b / a)).median())
    d["BsNineExpSqFirst"] = (_exp(0.1, 0.03), "squeeze load exponent, delta 0.1 -> 0.03 -- DIAGNOSTIC: no longer cited in Paper 3 sec 3 (R-63.6)")
    d["BsNineExpSqLast"] = (_exp(0.003, 0.001), "squeeze load exponent, delta 0.003 -> 0.001 -- DIAGNOSTIC: below the production floor (R-56.3.2)")

    j = json.loads((OUT / "boxc_phase6_decomposition.json").read_text())
    vals: list[float] = []

    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "share_memory" and isinstance(v, (int, float)):
                    vals.append(abs(float(v)))
                else:
                    _walk(v)
        elif isinstance(o, list):
            for x in o:
                _walk(x)
    _walk(j["points"])
    # Regime thresholds, from the registered key (never typed).
    import ball_socket.campaign_config_v6 as v6
    th = v6.config_dict()["regime_thresholds"]
    d["BsNineFullFilmMin"] = (th["full_film_min"], "regime_thresholds: full film at or above")
    d["BsNineMixedMin"] = (th["mixed_min"], "regime_thresholds: mixed at or above")

    # Inertia contribution at the PUBLIC rated maximum corner (ERR-20, R-48.8):
    # m_eff R_p tan(beta_max) omega^2 / F_app, with 2 R_p tan(beta_max) the
    # pinned stroke.  Reported as a BAND, never as a bound.
    from ball_socket import e5_fluid_spec as E5
    cor = E5.CORNERS["maximum"]
    w = cor["rpm"] * 2 * np.pi / 60.0
    F_app = cor["p_ch_bar"] * 1e5 * rep_.A_P_M2
    m_lo, m_hi = v6.FRAME_BAND["m_eff"]["value"]
    k = 0.5 * v6.PINNED_STROKE_M * w ** 2 / F_app * 100
    d["BsNineInertiaLoPct"] = (m_lo * k, "inertia / F_app at the public rated corner, m_eff low, %")
    d["BsNineInertiaHiPct"] = (m_hi * k, "inertia / F_app at the public rated corner, m_eff high, %")
    d["BsNineInertiaMidPct"] = (0.5 * (m_lo + m_hi) * k,
                                "inertia / F_app at the band midpoint of m_eff, %")
    # E4 across the registered lambda axis (R-50.2), registered bracket passed.
    sw = pd.read_parquet(OUT / "E4_lambda_sweep.parquet")
    if (sw.status != "ok").any():
        BOXF_GATE_FAILS.append(f"E4 sweep: {(sw.status != 'ok').sum()} decompositions "
                               f"failed; the bound would be over a filtered set")
    ind = sw[(sw.status == "ok") & sw.in_g3_domain].copy()
    # R-63.1: E4 RELEASED on the SIGNED path (D-6 fixed): the share columns come from the
    # signed re-run of every in-domain point (R58_E4_signed.parquet), joined one-to-one.
    # The modulus sweep keeps the inputs, the domain flag and Wi_char.
    _sg = pd.read_parquet(OUT / "R58_E4_signed.parquet")[["base", "lam", "share_D1_signed", "share_D2_signed"]]
    ind = ind.merge(_sg, on=["base", "lam"], how="left", validate="1:1")
    if ind.share_D1_signed.isna().any():
        BOXF_GATE_FAILS.append("E4 sweep: an in-domain point has no signed re-run")
    ind["a"] = ind.share_D1_signed.abs()
    if not ((ind.share_D1_signed > 0).all() and (ind.share_D2_signed > 0).all()):
        BOXF_GATE_FAILS.append("E4 sweep (signed): a non-positive memory share in domain -- the "
                               "'memory raises load at every in-domain point' statement no longer holds")
    d["BsNineEFourSweepN"] = (len(sw), "E4 sweep: decompositions")
    d["BsNineEFourSweepBase"] = (int(sw.base.nunique()), "E4 sweep: base rows")
    d["BsNineEFourSweepInDomain"] = (len(ind), "E4 sweep: decompositions in the g3 domain")
    d["BsNineEFourSweepRaisesN"] = (int(((ind.share_D1_signed > 0) & (ind.share_D2_signed > 0)).sum()),
                                    "E4 sweep (signed): in-domain points where memory RAISES the load, D1 and D2")
    d["BsNineEFourSweepMax"] = (float(ind.a.max()), "E4 sweep (signed): largest in-domain memory share, D1")
    d["BsNineEFourSweepMaxDTwo"] = (float(ind.share_D2_signed.abs().max()),
                                    "E4 sweep (signed): largest in-domain memory share, D2 re-solved (pf:950)")
    d["BsNineEFourSweepMedian"] = (float(ind.a.median()), "E4 sweep (signed): median in-domain share, D1")
    _p60 = pd.read_parquet(OUT / "R60_parity.parquet")
    _pf = _p60[(_p60.base == "pf:950") & np.isclose(_p60.lam, 1e-5)].iloc[0]
    d["BsNineEFourEvenPct"] = (100 * abs(abs(_pf.share_D2_signed_at_eps_mod) / abs(_pf.share_D2_mod) - 1),
                               "O(M^2) (model-truncation) part of the D2 share at pf:950, % of the share "
                               "(R-60.4): limits the printed precision")
    floor = float(sw.lam.min())
    at_floor = ind.groupby("base").apply(
        lambda g: bool(g.sort_values("lam").a.iloc[0] >= g.a.max() - 1e-15))
    d["BsNineEFourSweepFloorRows"] = (int(at_floor.sum()),
                                      "base rows whose in-domain max share sits at the "
                                      "lambda axis floor (gradient points outward)")
    d["BsNineEFourSweepWiMin"] = (float(ind.Wi_char.min()),
                                  "smallest characteristic Weissenberg number in domain")
    # E1 on the BOX-E artefact (R-50 hand-back): the existing BsNineEOne* macros
    # read the box-B audit, whose g3 used the pre-correction interior De_sq.
    # Box D's rim-inclusive De_sq_bnd is the registered quantity now.
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M as _S, ConstraintSet as _CS
    _cs = _CS()
    E = pd.read_parquet(OUT / BOXE_MIXED, columns=["sample_id", "outcome", "lam", "De_sq_bnd",
                                                   "h_min_bnd", "T_max_bnd"])
    E = E[E.outcome == "converged"]
    viol = E.De_sq_bnd > _cs.De_sq_max
    d["BsNineEOneViolPct"] = (100 * float(viol.mean()),
                              "box E: % of converged rows violating g3 (De_sq_bnd > 0.2)")
    d["BsNineEOneLamMinViol"] = (float(E.lam[viol].min()),
                                 "box E: smallest lambda among g3-violating rows, s")
    d["BsNineEOneAboveMilliPct"] = (100 * float((E.lam[viol] > 1e-3).mean()),
                                    "box E: % of g3-violating rows with lambda > 1e-3 s")
    sole = []
    for s_ in _S:
        for lim in _cs.T_max_C_by_class.values():
            g1 = E.h_min_bnd / s_ >= _cs.lambda_ratio_min
            g2 = E.T_max_bnd <= lim
            sole.append(100 * float((g1 & g2 & viol).mean()))
    d["BsNineEOneSoleMinPct"] = (min(sole), "box E: g3 sole-binding, % of converged, smallest cell")
    d["BsNineEOneSoleMaxPct"] = (max(sole), "box E: g3 sole-binding, % of converged, largest cell")
    # E1 against its FROZEN text (R-52.3), from the adjudication artefact, which
    # asserts the quote against preregistration_FROZEN.md:117 and the HFC-E
    # population equality before writing.
    e1 = json.loads((OUT / "E1_adjudication.json").read_text(encoding="utf-8"))
    if e1["verdict"] != "REFUTED" or not e1["population_check"][
            "hfce_labelled_converged_rows_equal_converged_rows"]:
        BOXF_GATE_FAILS.append("E1: adjudication artefact no longer says REFUTED on "
                               "the HFC-E population")
    bE = e1["box_E"]
    if abs(bE["violating_g3_pct"] - d["BsNineEOneViolPct"][0]) > 1e-9:
        BOXF_GATE_FAILS.append("E1: adjudication and campaign disagree on the g3 share")
    d["BsNineEOneInRangePct"] = (bE["in_registered_range_pct"],
                                 "E1: % of HFC-E rows with De_sq in the registered 0.02-0.06")
    d["BsNineEOneDesqMedian"] = (bE["median"], "E1: median De_sq_bnd, converged rows")
    d["BsNineEOneDesqQThree"] = (bE["q75"], "E1: upper-quartile De_sq_bnd, converged rows")
    d["BsNineEOneQThreeRatio"] = (bE["q75_over_registered_top"],
                                  "E1: upper quartile / registered top of range (0.06)")
    wg = bE["g3_sole_binding_hfce_pct"]
    d["BsNineEOneSoleWgMinPct"] = (wg[0], "E1: g3 sole-binding % under the HFC-E (60 C) "
                                          "ceiling, smaller sigma* end")
    d["BsNineEOneSoleWgMaxPct"] = (wg[1], "E1: g3 sole-binding % under the HFC-E ceiling, "
                                          "larger sigma* end")
    d["BsNineEOneBoxBInRangePct"] = (e1["own_artefacts"]["box B"]["in_registered_range_pct"],
                                     "E1: % in the registered range on box B's own audit")
    d["BsNineEOneBoxBViolPct"] = (e1["own_artefacts"]["box B"]["violating_g3_pct"],
                                  "E1: % violating g3 on box B's own audit")
    fm = json.loads((OUT / "fluid_to_band_map.json").read_text())
    d["BsNineLamGThreeCorner"] = (fm["lambda_at_g3_limit_s"],
                                  "lambda at which De_sq reaches the g3 limit at the rated "
                                  "corner (eps = 1, theta_min = 20 deg), s")
    # R-63.1: the six phase-6 points on the SIGNED path (R63_E4_phase6_signed.parquet); the
    # modulus values in `vals` are kept only for the sign-flip and magnitude checks.
    _p6 = pd.read_parquet(OUT / "R63_E4_phase6_signed.parquet")
    if not ((_p6.share_signed > 0).all() and _p6.flipped.all() and (_p6.rel_mag <= 0.01).all()
            and sorted(np.round(_p6.share_stored.abs(), 15)) == sorted(np.round(vals, 15))):
        BOXF_GATE_FAILS.append("E4 phase-6 signed: a point did not flip, moved > 1 %, or is not the phase-6 set")
    d["BsNineMemShareMin"] = (float(_p6.share_signed.min()), "E4 (signed): smallest memory share of the load")
    d["BsNineMemShareMax"] = (float(_p6.share_signed.max()), "E4 (signed): largest memory share of the load")


def _SSB_frozen() -> tuple[float, float]:
    fz = json.loads((ROOT / "ball_socket" / "campaign_frozen_v6.json"
                     ).read_text(encoding="utf-8"))
    return tuple(float(v) for v in fz["config"]["sigma_star_band_m"])


# --------------------------------------------------------------------------
# declared values, read straight out of the prose value blocks
# --------------------------------------------------------------------------

#: Sentinel for a macro the prose has not yet fixed a value for.  Standing
#: convention (2026-08-01): an ASSERTED order of magnitude is an unverified
#: number.  Write TBV here rather than a plausible-looking numeral, and the
#: generator will hard-fail while printing the value it derived, so the
#: assertion is replaced by a measurement instead of surviving review.  The
#: retracted \BsNineFilmAreaFrac is the worked example: "~1e-9" read as a
#: description rather than a measurement and survived two review passes.
TBV = "TBV"

#: R-54.1 VALUE GATE.  Reading a column's NAME cannot distinguish a correctly
#: named column from a mislabelled one; reading its VALUES can.  Box D sec 3.3
#: registers the reduction of the boundary extremum under axial eccentricity,
#:     h(theta_min) = c (1 - eps cos theta_min),
#: and an interior-node extremum departs from it by a grid-step term of order
#: eps sin(theta_min) * dtheta -- many orders above this tolerance.  So every
#: artefact behind an h_min or Lambda macro is checked BY VALUE, and a failure
#: NAMES the macros at risk from it.  ERR-9's reach (the E10 run's cell-centre
#: h_min, which reached E5's Lambda_max) is the defect this gate exists for.
HMIN_CLOSED_FORM_RTOL = 1e-10
#: a macro is "at risk" from a failing artefact when its description names the
#: film minimum or the film parameter; derived from the descriptions, not listed
HMIN_MACRO_PATTERN = re.compile(r"h_min|Lambda|film parameter|film minimum", re.I)


def hmin_deviation(df) -> float:
    """Largest relative departure of h_min_bnd from c (1 - eps cos theta_min).

    Split out so the gate can be armed against a frame that violates it: the
    gate's own artefacts all pass, which is what a positive control is for
    (design rule 4, R-54.2 -- a check validated only on cases it passes has an
    unmeasured failure mode).
    """
    closed = df.c_equilibrium * (1.0 - df.eps_ecc
                                 * np.cos(np.deg2rad(df.theta_min_deg)))
    return float(((df.h_min_bnd - closed).abs() / df.h_min_bnd.abs()).max())


def hmin_closed_form_gate(d: dict, paths=None) -> dict:
    """Check h_min_bnd against the registered closed form on every artefact that
    can carry it, over every row that has an equilibrium -- which includes, a
    fortiori, the row that attains any extremum macro."""
    import pyarrow.parquet as _pq
    at_risk = sorted(k for k, v in d.items()
                     if HMIN_MACRO_PATTERN.search(str(v[1])))
    need = {"h_min_bnd", "eps_ecc", "theta_min_deg", "c_equilibrium"}
    checked, interior_only = [], []
    for f in (sorted(OUT.glob("*.parquet")) if paths is None else list(paths)):
        try:
            cols = set(_pq.ParquetFile(f).schema.names)
        except Exception:                       # not a readable parquet
            continue
        if not need <= cols:
            # an artefact carrying an INTERIOR minimum and no corrected column is
            # a place a future Lambda macro could repeat ERR-9; record it.
            if {"h_min", "eps_ecc", "theta_min_deg"} <= cols and "h_min_bnd" not in cols:
                interior_only.append(f.name)
            continue
        use = sorted(need | ({"outcome"} & cols) | ({"state"} & cols))
        df = pd.read_parquet(f, columns=use)
        if "outcome" in df:
            df = df[df.outcome == "converged"]
        elif "state" in df:
            df = df[df.state == "equilibrium"]
        df = df.dropna(subset=["h_min_bnd", "c_equilibrium", "eps_ecc", "theta_min_deg"])
        if df.empty:
            continue
        dev = hmin_deviation(df)
        checked.append({"artefact": f.name, "rows": int(len(df)), "max_rel_dev": dev})
        if not (dev <= HMIN_CLOSED_FORM_RTOL):
            BOXF_GATE_FAILS.append(
                f"h_min VALUE GATE: {f.name} departs from c(1 - eps cos theta_min) "
                f"by {dev:.3e} (> {HMIN_CLOSED_FORM_RTOL:.0e}) on {len(df)} rows -- "
                f"an interior extremum or a mislabelled column.  Macros at risk: "
                f"{', '.join(at_risk) if at_risk else 'none named'}")
    return {"tolerance": HMIN_CLOSED_FORM_RTOL, "artefacts_checked": checked,
            "macros_covered": at_risk,
            "interior_only_artefacts": interior_only}


DECLARED: dict[str, tuple[float, float]] = {
    # macro: (declared value, absolute tolerance)
    "BsNineNAudit": (8192, 0), "BsNineAuditConverged": (1594, 0),
    # G1/G2a, registered 2026-08-31 (ruling R-20.5, surfaced by drafting S1/S2/S6)
    "BsNineSigmaStarLoUm": (0.336, 0.001),
    "BsNineSigmaStarHiUm": (0.524, 0.001),
    "BsNineLamMedConvLo": (5.569, 0.001),
    "BsNineLamMedConvHi": (3.571, 0.001),
    "BsNineAuditConvPct": (19.46, 0.01), "BsNineAuditBlowupPct": (80.54, 0.01),
    "BsNineAuditBlowup": (6598, 0),
    "BsNineBlowupLoadMed": (4.62, 0.01), "BsNineConvLoadMed": (2.81, 0.01),
    "BsNineConvBelowFive": (24.3, 0.05), "BsNineConvFiveTen": (12.5, 0.05),
    "BsNineNExtended": (90000, 0), "BsNineExtConvPct": (19.34, 0.01),
    "BsNineExtConvZ": (0.26, 0.02), "BsNineExtFeasZmax": (0.82, 0.02),
    "BsNineEOneLamGThreeLo": (3.24e-3, 5e-6), "BsNineEOneLamGThreeHi": (3.11e-3, 5e-6),
    "BsNineEOneLamAll": (2.95e-4, 5e-7),
    "BsNineEOneLamPctLo": (93, 0.5), "BsNineEOneLamPctHi": (92, 0.5),
    "BsNineEOneDesqGOneLo": (0.0157, 5e-5),
    "BsNineEOneDesqGOneHi": (0.0179, 5e-5),
    "BsNineEThreeWgEnv": (59.4, 0.05), "BsNineWgCeiling": (60, 0),
    "BsNineEThreeMinCount": (12, 0), "BsNineEThreeMinEnv": (110.0, 0.05),
    "BsNineMineralCeiling": (110, 0), "BsNineEFiveNuMax": (9.0, 0),
    "BsNineESixLo": (78, 0.5), "BsNineESixHi": (86, 0.5),
    "BsNineBsQmin": (85.0, 0.05), "BsNineBsBrink": (2.8, 0.05),
    "BsNineConvBase": (19.6, 0.05),
    "BsNineConvBoLo": (93.2, 0.05), "BsNineConvBoHi": (95.8, 0.05),
    "BsNineUpliftFactor": (4.8, 0.05),
    "BsNineOatN": (256, 0), "BsNineOatSpinBlowup": (0.0, 0.05),
    "BsNineOatThetaBlowup": (2.7, 0.05), "BsNineOatSE": (3.5, 0.05),
    "BsNineOatSpinRouting": (8.0, 0.05),
    "BsNineIouBoFracLo": (34, 0.5), "BsNineIouBoFracHi": (15, 0.5),
    "BsNineIouRandFracLo": (69, 0.5), "BsNineIouRandFracHi": (63, 0.5),
    "BsNineIouExtractorFrac": (7, 0.5),
    "BsNineIouRandWins": (5, 0), "BsNineDhBoWins": (4, 0),
    "BsNineProjCells": (6, 0), "BsNineBoWinsTotal": (5, 0),
    "BsNineCellsTotal": (12, 0),
    "BsNineDhBoBest": (0.0128, 5e-5), "BsNineDhRandBest": (0.0876, 5e-5),
    "BsNineFeasBoLo": (84, 0), "BsNineFeasRandLo": (56, 0),
    "BsNineFeasBoHi": (112, 0), "BsNineFeasRandHi": (42, 0),
    "BsNineCeilFilmLo": (0.679, 5e-4), "BsNineCeilFilmHi": (0.779, 5e-4),
    "BsNineCeilPowerLo": (0.330, 5e-4), "BsNineCeilPowerHi": (0.346, 5e-4),
    "BsNineTierRatio": (2.2, 0.05),
    # scoping block
    "BsNineAucLo": (0.9212, 5e-5), "BsNineAucHi": (0.8609, 5e-5),
    "BsNineAucThreshold": (0.90, 0),
    "BsNineBrierLo": (0.0935, 5e-5), "BsNineBrierHi": (0.1275, 5e-5),
    "BsNineRefitSpreadMin": (0.0009, 5e-5), "BsNineRefitSpreadMax": (0.0348, 5e-5),
    "BsNineRefitSpreadMed": (0.0065, 5e-5),
    "BsNineArmSpreadMin": (0.0782, 5e-5), "BsNineArmSpreadMax": (0.0833, 5e-5),
    "BsNinePrecLo": (0.950, 5e-4), "BsNinePrecHi": (0.951, 5e-4),
    "BsNineBaseRate": (11, 0.5),
    "BsNineRecallLo": (0.537, 5e-4), "BsNineRecallHi": (0.449, 5e-4),
    "BsNineBsOracleDrawLo": (55, 0.5), "BsNineBsOracleDrawHi": (76, 0.5),
    "BsNineCeilDeltaLo": (0.17, 0.005), "BsNineCeilDeltaHi": (0.43, 0.005),
    "BsNineRecallGapUnifLo": (0.0102, 5e-5),
    "BsNineRecallGapUnifHi": (0.0471, 5e-5),
    "BsNineRecallGapBoLo": (0.065, 5e-4), "BsNineRecallGapBoHi": (0.165, 5e-4),
    "BsNineNormAreaLo": (38, 0.5), "BsNineNormAreaHi": (84, 0.5),
    "BsNineAspectRatio": (7e8, 5e7),
    # slipper
    "BsNineSlipperConvBase": (48.1, 0.05), "BsNineSlipperConvBo": (98.1, 0.05),
    "BsNineSlipperPrec": (0.9172, 5e-5), "BsNineSlipperRecall": (0.8499, 5e-5),
    "BsNineSlipperBrier": (0.047, 5e-4),
    "BsNineSlipperCeilLo": (0.2432, 5e-5), "BsNineSlipperCeilHi": (0.4235, 5e-5),
    "BsNineSlipperOracleDraw": (1.4, 0.05),
    "BsNineSlipperLTwo": (64.95, 0.01), "BsNineSlipperBrink": (61.4, 0.05),
    "BsNineSlipperQmin": (3.5, 0.05), "BsNineSlipperTmax": (89.2, 0.05),
    # --- BOX F (R-46 / R-47.10.8).  Declared at the precision Paper 3 quotes;
    # the tolerance is the rounding licensed.  Counts are exact.
    "BsNineESevenValid": (196602, 0), "BsNineESevenNoEq": (89672, 0),
    "BsNineESevenPct": (45.61, 0.005), "BsNineESevenThreshPct": (5, 0),
    "BsNineESevenFactor": (9.12, 0.005),
    "BsNineESevenCompValid": (196592, 0), "BsNineESevenCompNoEq": (84998, 0),
    "BsNineESevenCompPct": (43.24, 0.005), "BsNineESevenCompFactor": (8.65, 0.005),
    "BsNineSamples": (8192, 0),
    "BsNineESevenSamplesNoEq": (3720, 0), "BsNineESevenCompSamplesNoEq": (3450, 0),
    "BsNineESevenCompSamplesPct": (42.1, 0.05),
    # 99.49, NOT the 99.46 of the E7 mechanism hand-back: that count predates
    # R-37.4 and scored sample 6271's six picard_at_root rows as a verdict.
    "BsNineArmMixedSamples": (23, 0), "BsNineArmConvAnySamples": (4472, 0),
    "BsNineArmInvariantPct": (99.49, 0.005),
    "BsNineESevenBRows": (24740, 0), "BsNineESevenBGain": (5064, 0),
    "BsNineESevenBPct": (20.5, 0.05), "BsNineESevenBLamMax": (1.03, 0.005),
    "BsNineETenPlasticPct": (78.3, 0.05), "BsNineETenDoubtPct": (21.7, 0.05),
    "BsNineETenElasticPct": (0, 0),
    "BsNineETenCapMedLo": (0.457, 5e-4), "BsNineETenCapMedHi": (0.716, 5e-4),
    "BsNineETenCapDoubtMax": (0, 0),
    "BsNineLamMaxTwoLo": (1.909, 5e-4), "BsNineLamMaxTwoHi": (1.224, 5e-4),
    "BsNineLamMaxFiveLo": (1.334, 5e-4), "BsNineLamMaxFiveHi": (0.855, 6e-4),
    # R-48.10: design ratio h_min/c0 and mechanism diagnostic h_min/c_eq.
    "BsNineHcColdTwo": (0.256, 5e-4), "BsNineHcColdFive": (0.062, 5e-4),
    "BsNineHcEqTwo": (0.256, 5e-4), "BsNineHcEqFive": (0.062, 5e-4),
    "BsNineHcColdBinFirst": (0.306, 5e-4), "BsNineHcColdBinLast": (0.033, 5e-4),
    "BsNineHcEqBinFirst": (0.306, 5e-4), "BsNineHcEqBinLast": (0.033, 5e-4),
    "BsNineHcBins": (9, 0),
    # Paper 3 sections 2-3, 5.1-5.2 (R-48.12.6, surfaced by drafting)
    "BsNineRbMm": (6.5, 0), "BsNineThetaMaxDeg": (73, 1e-9),
    "BsNineThetaMinLoDeg": (20, 0), "BsNineThetaMinHiDeg": (40, 0),
    "BsNineCZeroLoUm": (2, 1e-9), "BsNineCZeroHiUm": (30, 1e-9),
    "BsNineEpsCeilTwenty": (1.064, 5e-4), "BsNineEpsCeilThirty": (1.155, 5e-4),
    "BsNineEpsCeilForty": (1.305, 5e-4), "BsNineContactArms": (24, 0),
    "BsNineEFourRows": (509, 0),
    "BsNineEFourSqueezePct": (94.3, 0.05), "BsNineEFourHydroPct": (6.0, 0.05),
    "BsNineEFourSpinMaxN": (2e-14, 1e-14),
    "BsNineExpSqFirst": (-0.94, 0.005), "BsNineExpSqLast": (-0.30, 0.005),
    "BsNineMemShareMin": (1.6e-6, 5e-8), "BsNineMemShareMax": (1.2e-4, 5e-6),
    "BsNineFullFilmMin": (3, 0), "BsNineMixedMin": (1, 0),
    "BsNineEFourSweepN": (220, 0), "BsNineEFourSweepBase": (22, 0),
    "BsNineEFourSweepInDomain": (137, 0),
    "BsNineEFourSweepMax": (9.1e-4, 5e-6), "BsNineEFourSweepMaxDTwo": (1.0e-3, 5e-6),
    # R-63.1: E4 on the signed path -- memory raises the load at every in-domain point
    "BsNineEFourSweepRaisesN": (137, 0), "BsNineEFourEvenPct": (0.63, 0.005),
    "BsNineEFourSweepMedian": (2.5e-5, 5e-7), "BsNineEFourSweepFloorRows": (18, 0),
    "BsNineEFourSweepWiMin": (2.2, 0.05),
    "BsNineEOneViolPct": (34.4, 0.05), "BsNineEOneLamMinViol": (4.5e-4, 5e-6),
    "BsNineEOneAboveMilliPct": (88.3, 0.05),
    "BsNineEOneSoleMinPct": (3.6, 0.05), "BsNineEOneSoleMaxPct": (14.5, 0.05),
    # E1 refuted on the frozen text (R-52.3)
    "BsNineEOneInRangePct": (15.6, 0.05), "BsNineEOneDesqMedian": (0.067, 0.0005),
    "BsNineEOneDesqQThree": (0.38, 0.005), "BsNineEOneQThreeRatio": (6.3, 0.05),
    "BsNineEOneSoleWgMinPct": (3.6, 0.05), "BsNineEOneSoleWgMaxPct": (6.4, 0.05),
    "BsNineEOneBoxBInRangePct": (15.2, 0.05), "BsNineEOneBoxBViolPct": (31.1, 0.05),
    "BsNineLamGThreeCorner": (4.5e-4, 5e-6),
    "BsNineInertiaLoPct": (1.7, 0.05), "BsNineInertiaHiPct": (2.8, 0.05),
    "BsNineInertiaMidPct": (2.26, 0.005),
    # E13 (R-47/R-48.2) and E3 restated (R-48.1)
    "BsNineEThirteenLamMax": (2.393, 5e-4), "BsNineEThirteenShortPct": (20.2, 0.05),
    "BsNineEThirteenSolves": (360, 0), "BsNineEThirteenFullFilm": (0, 0),
    "BsNineEThirteenEtaMpas": (7.33, 0.005),
    "BsNineEThirteenLamMaxFourteen": (0.557, 5e-4),
    "BsNineEThreeWgFails": (60326, 0), "BsNineEThreeInletPct": (98.2, 0.05),
    "BsNineEThreeOnlyGTwoLo": (18096, 0), "BsNineEThreeHeatedLo": (408, 0),
    "BsNineEThreeForcedLo": (17688, 0), "BsNineEThreeForcedHi": (10464, 0),
    "BsNineEThreeSoleHi": (216, 0), "BsNineEThreeSoleTinMin": (54.4, 0.05),
    "BsNineEThreeRiseMax": (9.6, 0.05),
    "BsNineEThreeSoleSamplesLo": (17, 0), "BsNineEThreeSoleSamplesHi": (9, 0),
    "BsNineEThreeBiasDownLo": (15, 0), "BsNineEThreeBiasDownHi": (9, 0), "BsNineTinAboveWgPct": (55.6, 0.05),
    "BsNineFilmAreaClosed": (0.49, 0.005), "BsNineFilmAreaOpen": (1.05, 0.005),
    "BsNineConstBandSmoothLo": (2.4024, 5e-5), "BsNineConstBandHi": (3.7463, 5e-5),
    "BsNineConstBandLo": (2.3306, 5e-5), "BsNineConstBandWidenMin": (2.3527, 5e-5),
    "BsNineFeasBandCount": (0, 0), "BsNineFeasBandUnits": (350, 0),
    "BsNineFeasBandLbMin": (0.22, 0.005), "BsNineFeasBandLbMax": (23.9, 0.05),
    # R-57.5: tight-tolerance margin, printed to its mesh uncertainty (0.9 % of 3)
    "BsNineFeasBandMarginPct": (1.11, 0.005), "BsNineFeasBandMarginMeshPct": (0.9, 0.05),
    "BsNineFeasBandMarginUncPct": (0.05, 0.005),
    # R-58.2: claim rows at their Richardson values (signed path)
    "BsNineEThirteenLamMaxR": (2.389, 5e-4), "BsNineEFiveLamMaxR": (1.874, 5e-4),
    "BsNineEThirteenShortPctR": (20.4, 0.05),
    "BsNineFilmAreaOpenProj": (1.22, 0.005), "BsNineFilmAreaOpenHydPct": (99.4, 0.05),
    "BsNineEFourteenNetLoMin": (4.871, 5e-4), "BsNineEFourteenNetLoWg": (2.136, 5e-4),
    "BsNineEFourteenNetHiMin": (1.733, 5e-4), "BsNineEFourteenNetHiWg": (0.903, 5e-4),
    "BsNineEFourteenNetCiLoLoMin": (4.37, 0.005),
    "BsNineEFourteenNetCiLoLoWg": (1.83, 0.005),
    "BsNineEFourteenNetCiLoHiMin": (1.43, 0.005),
    "BsNineEFourteenNetCiLoHiWg": (0.70, 0.005),
    "BsNineEFourteenFThreeLoMin": (0.469, 5e-4),
    "BsNineEFourteenFThreeLoWg": (0.739, 5e-4),
    "BsNineEFourteenFThreeHiMin": (0.317, 5e-4),
    "BsNineEFourteenFThreeHiWg": (0.475, 5e-4),
    "BsNineEFourteenJointLoMin": (3.90, 0.005), "BsNineEFourteenJointLoWg": (1.09, 0.005),
    "BsNineEFourteenJointHiMin": (1.11, 0.005), "BsNineEFourteenJointHiWg": (0.22, 0.005),
    # E14 over the movable subpopulation (R-53.3)
    "BsNineEFourteenImmovablePct": (55.6, 0.05),
    "BsNineEFourteenNetMovHiWg": (2.03, 0.005),
    "BsNineEFourteenFThreeMovHiWg": (1.29, 0.005),
    "BsNineEFourteenJointMovHiWg": (0.30, 0.005),
    # E2's geography restricted to the rows the inlet does not decide (R-53.3)
    "BsNineETwoRestrictedPct": (44.6, 0.05), "BsNineETwoGapPp": (1.06, 0.005),
    "BsNineETwoCeilingReachPct": (2.31, 0.005),
    # E5's VG32 binding clause (R-54.7)
    "BsNineEFiveVgThirtyTwoSolves": (960, 0),
    "BsNineEFiveVgThirtyTwoEqRows": (480, 0),
    "BsNineEFiveVgThirtyTwoSoleGOne": (480, 0),
    "BsNineEFiveVgThirtyTwoTMaxMax": (100.12, 0.005),
    "BsNineEFiveVgThirtyTwoRiseMax": (0.12, 0.005),
    "BsNineEFiveVgThirtyTwoDeSqMax": (0.0045, 5e-5),
    "BsNineEFiveLamMaxCorner": (1.877, 0.0005),
    "BsNineETwoRestrictedMinLoPct": (36.1, 0.05),
    "BsNineETwoRestrictedMinHiPct": (46.4, 0.05),
    "BsNineETwoRestrictedWgLoPct": (35.4, 0.05),
    "BsNineETwoRestrictedWgHiPct": (45.4, 0.05),
    # R-64.2: numbers v4 declared as owed
    "BsNineExistN": (72, 0), "BsNineExistBPct": (68.1, 0.05), "BsNineExistAPct": (31.9, 0.05),
    "BsNineExistAConf": (23, 0), "BsNineThetaFlowMed": (3.41, 0.005), "BsNineThetaFlowAbovePct": (88.1, 0.05),
    "BsNineMeanTinMinLo": (59.5, 0.05), "BsNineAdiabTinMinLo": (52.5, 0.05), "BsNineAdiabTinMinHi": (55.9, 0.05),
    "BsNineNahmeAdiabLoPct": (4.1, 0.05), "BsNineNahmeAdiabHiPct": (9.3, 0.05), "BsNineThermClearNm": (0.34, 0.005),
    "BsNineCornerPBar": (500, 0), "BsNineCornerRpm": (4500, 0),
    "BsNineLeeHydWithPct": (69.7, 0.05), "BsNineLeeHydWithoutPct": (9.5, 0.05),
    "BsNineFricN": (23, 0), "BsNineFricOrderMin": (1.46, 0.005), "BsNineFricCorrMaxPct": (0.64, 0.005),
    "BsNineFricTorqueMedNm": (0.092, 5e-4), "BsNineFricTorqueMinNm": (0.015, 5e-4), "BsNineFricTorqueMaxNm": (0.60, 0.005),
    "BsNineFricMuMed": (0.0049, 5e-5), "BsNineFricMuMin": (0.00075, 5e-6), "BsNineFricMuMax": (0.012, 5e-4),
    # R-66.5 / R-66.2 / R-66.4 / R-67.4 (released items)
    "BsNineESixCommonTotalPct": (91.0, 0.05), "BsNineBsBrinkCommonPct": (42.2, 0.05),
    "BsNineBsQminFlagPct": (87.6, 0.05),
    # R-68.2: mineral class (D-8)
    "BsNineThetaFlowMedMin": (8.24, 0.005),
    "BsNineThetaFlowAbovePctMin": (98.4, 0.05),
    "BsNineBsBrinkMin": (37.6, 0.05),
    "BsNineBsQminMin": (61.3, 0.05),
    "BsNineNahmeAdiabLoPctMin": (1.97, 0.005),
    "BsNineNahmeAdiabHiPctMin": (6.6, 0.005),
    "BsNineOpsSurviveMinPctMin": (66.0, 0.05),
    "BsNineETwoCeilingReachPctMin": (2.31, 0.005),
    "BsNineETwoGapPpMin": (1.06, 0.005),
    "BsNineThermClearNmMin": (1.36, 0.005),
    "BsNineOpsCountFullLoMin": (1972, 0), "BsNineOpsCountContactLoMin": (4, 0),
    "BsNineOpsCountMixedLoMin": (2479, 0),
    "BsNineOpsCountNoneLoMin": (3737, 0),
    "BsNineOpsCountFullHiMin": (1194, 0), "BsNineOpsCountContactHiMin": (15, 0),
    "BsNineOpsCountMixedHiMin": (3257, 0),
    "BsNineOpsCountNoneHiMin": (3726, 0),
    "BsNineOpsSFiftyFilmOneMin": (35.2, 0.05),
    "BsNineOpsSFiftyFullOneMin": (5.19, 0.005),
    "BsNineOpsSFiftyFilmTwoMin": (23.3, 0.05),
    "BsNineOpsSFiftyFullTwoMin": (3.46, 0.005),
    "BsNineOpsSFiftyFilmThreeMin": (15.3, 0.05),
    "BsNineOpsSFiftyFullThreeMin": (2.0, 0.005),
    # R-69.3: thermal bracket
    "BsNineThermGTwoBreachPctMinCond": (1.3, 0.005),
    "BsNineThermGTwoSolePctMinCond": (1.06, 0.005),
    "BsNineThermNahmeLoPctMinCond": (0.0, 0.005),
    "BsNineThermNahmeHiPctMinCond": (0.0, 0.005),
    "BsNineThermBindDomPctLoMinCond": (36.1, 0.05),
    "BsNineThermBindDomPctHiMinCond": (46.4, 0.05),
    "BsNineThermRiseMedMinCond": (0.405, 0.0005),
    "BsNineThermOpsSurvivePctMinCond": (66.0, 0.05),
    "BsNineThermNahmeSixLoPctMinCond": (0.924, 0.0005),
    "BsNineThermNahmeSixHiPctMinCond": (5.0, 0.005),
    "BsNineThermGTwoBreachPctMinAdv": (73.9, 0.05),
    "BsNineThermGTwoSolePctMinAdv": (69.1, 0.05),
    "BsNineThermNahmeLoPctMinAdv": (73.5, 0.05),
    "BsNineThermNahmeHiPctMinAdv": (86.4, 0.05),
    "BsNineThermBindDomPctLoMinAdv": (14.6, 0.05),
    "BsNineThermBindDomPctHiMinAdv": (18.2, 0.05),
    "BsNineThermRiseMedMinAdv": (82.5, 0.05),
    "BsNineThermOpsSurvivePctMinAdv": (20.7, 0.05),
    "BsNineThermGTwoBreachPctWgCond": (55.9, 0.05),
    "BsNineThermGTwoSolePctWgCond": (0.518, 0.0005),
    "BsNineThermNahmeLoPctWgCond": (0.0, 0.005),
    "BsNineThermNahmeHiPctWgCond": (0.0, 0.005),
    "BsNineThermBindDomPctLoWgCond": (35.9, 0.05),
    "BsNineThermBindDomPctHiWgCond": (46.2, 0.05),
    "BsNineThermRiseMedWgCond": (0.101, 0.0005),
    "BsNineThermOpsSurvivePctWgCond": (29.2, 0.05),
    "BsNineThermNahmeSixLoPctWgCond": (0.0, 0.005),
    "BsNineThermNahmeSixHiPctWgCond": (0.0342, 5e-05),
    "BsNineThermGTwoBreachPctWgAdv": (87.5, 0.05),
    "BsNineThermGTwoSolePctWgAdv": (65.8, 0.05),
    "BsNineThermNahmeLoPctWgAdv": (36.3, 0.05),
    "BsNineThermNahmeHiPctWgAdv": (63.0, 0.05),
    "BsNineThermBindDomPctLoWgAdv": (19.2, 0.05),
    "BsNineThermBindDomPctHiWgAdv": (11.8, 0.05),
    "BsNineThermRiseMedWgAdv": (34.2, 0.05),
    "BsNineThermOpsSurvivePctWgAdv": (10.0, 0.05),
    "BsNineThetaFlowMedWg": (3.42, 0.005),
    "BsNineThetaFlowAbovePctWg": (88.1, 0.05),
    # R-70.1 / R-70.2
    "BsNineWangTempC": (2, 0), "BsNineWangPMaxMPa": (115, 0), "BsNinePiezoAlphaWgDig": (0.77, 0.005),
    "BsNinePiezoGWgDig": (0.04, 0.005), "BsNinePiezoAlphaTenDig": (16, 0.5), "BsNinePiezoAlphaTwentyTwoDig": (23, 0.5),
    "BsNineHasanTempC": (40, 0), "BsNineHasanPMaxGPa": (0.5, 0), "BsNinePiezoAlphaWgLo": (2.2, 0),
    "BsNinePiezoAlphaWgHi": (2.93, 0), "BsNinePiezoGWg": (0.11, 0.005), "BsNinePiezoGWgHi": (0.15, 0.005),
    "BsNineBrinkThresh": (0.1, 0), "BsNineBrinkDeltaT": (20, 0), "BsNineThetaDeltaTRef": (10, 0), "BsNineDesqMax": (0.2, 0),
    # R-72: numerals, claim points, counts, denominators, E16
    "BsNineSatDeltaHi": (0.1, 0), "BsNineSatDeltaLo": (0.03, 0), "BsNineVerifDeltaHi": (0.3, 0),
    "BsNineFilmAreaClosedDelta": (0.3, 0), "BsNineLamAxisLo": (1e-5, 0), "BsNineLamAxisHi": (1e-2, 0),
    "BsNineLamSplit": (1e-3, 0), "BsNineAdiabFactor": (6, 0), "BsNinePCLamLo": (2.4, 0.005), "BsNinePCLamHi": (3.75, 0.005),
    "BsNineEFifteenDesq": (0.00303, 5e-6), "BsNineEThirteenDesq": (0.00282, 5e-6), "BsNineEFiveDesq": (0.00323, 5e-6),
    "BsNineEFifteenCZeroUm": (2.25, 0), "BsNineEThirteenCZeroUm": (2, 1e-9), "BsNineEFiveCZeroUm": (2, 1e-9),
    "BsNineEFifteenThetaMinDeg": (20, 0), "BsNineEThirteenThetaMinDeg": (20, 0), "BsNineEFiveThetaMinDeg": (30, 0),
    "BsNineEFifteenEtaZeroMPas": (7.33, 0.005), "BsNineEFifteenLamS": (1e-5, 0), "BsNineEFifteenTinC": (100, 0),
    "BsNineEFifteenPchMPa": (50, 0), "BsNineEFifteenOmega": (471, 0.5), "BsNineEFifteenSolventFrac": (0.95, 0),
    "BsNineEFifteenDesqPiezo": (0.017, 5e-4),
    "BsNineOpsFilmOnlyBoth": (4449, 0), "BsNineOpsClassThreeLo": (7, 0), "BsNineOpsClassThreeHi": (23, 0),
    "BsNineValidRows": (196608, 0), "BsNineValidN": (196602, 0), "BsNineValidExcl": (6, 0), "BsNineValidCompExcl": (10, 0),
    "BsNineTabDenInDomain": (70128, 0), "BsNineTabDenFrameMin": (31584, 0), "BsNineTabDenFrameWg": (13896, 0),
    "BsNineTabDenRestricted": (47708, 0), "BsNineTabDenClassOne": (1972, 0),
    "BsNineRatedPoints": (40, 0), "BsNineRatedRows": (960, 0), "BsNineContactModels": (16, 0),
    "BsNineOpsSurviveMinPctRun": (65.8, 0.05), "BsNineOpsSurviveWgPctRun": (28.5, 0.05),
    "BsNineDesqLocMed": (0.153, 5e-4), "BsNineDesqLocQThree": (0.637, 5e-4), "BsNineDesqLocMax": (9.03, 0.005),
    "BsNineDesqLocAbovePct": (45.3, 0.05),
    # R-72.6: E17
    "BsNineStrokeEFifteenSnapDevPct": (-0.27, 0.005), "BsNineStrokeEThirteenSnapDevPct": (-0.24, 0.005),
    "BsNineStrokeEFiveSnapDevPct": (-0.27, 0.005),
    "BsNineStrokeEFifteenTSnapDeg": (25, 0.5), "BsNineStrokeEThirteenTSnapDeg": (27, 0.5), "BsNineStrokeEFiveTSnapDeg": (22, 0.5),
    "BsNineStrokeEFifteenTcDeg": (216, 0.5), "BsNineStrokeEThirteenTcDeg": (273, 0.5), "BsNineStrokeEFiveTcDeg": (155, 0.5),
    "BsNineStrokeEFifteenTLamThreeRoelDeg": (37, 0.5), "BsNineStrokeEThirteenTLamThreeRoelDeg": (38, 0.5),
    "BsNineStrokeEFiveTLamThreeRoelDeg": (24, 0.5),
    "BsNineStrokeEFifteenLamPiRoel": (0.61, 0.005), "BsNineStrokeEThirteenLamPiRoel": (0.75, 0.005),
    "BsNineStrokeEFiveLamPiRoel": (0.38, 0.005),
    "BsNineStrokeEFifteenLamPiRoelSnap": (0.43, 0.005), "BsNineStrokeEThirteenLamPiRoelSnap": (0.52, 0.005),
    "BsNineStrokeEFiveLamPiRoelSnap": (0.255, 5e-4),
    "BsNineStrokeEFifteenLamPiGTwo": (2.54, 0.005), "BsNineStrokeEThirteenLamPiGTwo": (2.63, 0.005),
    "BsNineStrokeEFiveLamPiGTwo": (2.0, 0.005),
    "BsNineStrokeEFifteenTHundredthDeg": (156, 0.5), "BsNineStrokeEFifteenLamPiIso": (0.021, 5e-4),
    "BsNineStrokeEFifteenLamPiBarus": (0.64, 0.005), "BsNineStrokeEFifteenTLamThreeBarusDeg": (37, 0.5),
    "BsNineStrokeEFifteenRDeltaBarus": (0.23, 0.005), "BsNineStrokeEFifteenRDeltaBarusFloor": (0.14, 0.005),
    "BsNineStrokeTailPower": (0.56, 0.005), "BsNineStrokeGridFloor": (0.002, 0), "BsNineStrokeGMax": (2, 0),
    "BsNinePiezoGLee": (0.59, 0.005), "BsNinePiezoGLeeBarus": (0.63, 0.005),
    "BsNinePiezoGcritEFifteen": (0.41, 0.005), "BsNinePiezoGcritEThirteen": (0.46, 0.005),
    "BsNinePiezoGcritEFive": (0.84, 0.005), "BsNinePiezoFactorEFifteen": (1.38, 0.005),
    "BsNinePiezoFactorEThirteen": (1.34, 0.005), "BsNinePiezoFactorEFive": (1.44, 0.005),
    "BsNinePiezoLamEFifteen": (3.31, 0.005), "BsNinePiezoLamEThirteen": (3.21, 0.005), "BsNinePiezoLamEFive": (2.71, 0.005),
    "BsNinePiezoKappa": (0.61, 0.005), "BsNinePiezoKappaFit": (0.61, 0.005), "BsNinePiezoLinDevPct": (0.12, 0.005),
    "BsNinePiezoQPeakFrac": (0.83, 0.005), "BsNinePiezoCZeroRatio": (1.56, 0.005), "BsNinePiezoLamStar": (3.49, 0.005),
    "BsNinePiezoPeakOverPch": (2.89, 0.005),
    "BsNineFricSample": (24, 0), "BsNineFricDeltaSplit": (0.1, 0), "BsNineFricDeltaNonconv": (0.047, 5e-4),
    "BsNineESixBsN": (1399, 0), "BsNineESixBsDen": (1594, 0), "BsNineESixBsPct": (87.8, 0.05),
    "BsNineESixQminPct": (85.0, 0.05), "BsNineESixBandLo": (78, 0.5), "BsNineESixBandHi": (86, 0.5),
    "BsNineFieldsDelta": (0.36, 0.005), "BsNineFieldsPeakThetaDeg": (32, 0), "BsNineFieldsPeakThetaNewtDeg": (33.8, 0.05),
    "BsNineFieldsPeakMPa": (144, 0.5), "BsNineFieldsPeakNewtMPa": (151, 0.5),
    "BsNineFieldsRiseMaxmK": (170, 0.5), "BsNineFieldsEtaRatioMin": (0.951, 5e-4), "BsNineFieldsEtaRatioMax": (0.954, 5e-4), "BsNineFieldsSolventFrac": (0.95, 0),
    "BsNineLeeCUm": (15, 0), "BsNineLeeEtaMPas": (5.72, 0.005), "BsNineLeeAlphaStar": (11.8, 0.05),
    "BsNineLeePiezoHundred": (3.44, 0.005), "BsNineLeePiezoPeak": (5.62, 0.005),
    "BsNineLeeRbMm": (6.475, 0), "BsNineLeeLoadKN": (5.4, 0), "BsNineLeeSigmaUm": (0.5, 0), "BsNineLeeTempC": (100, 0),
    "BsNineDesqAtMemMax": (0.0045, 5e-5), "BsNineWiZero": (1.25, 0.005), "BsNineSecondOrderPlanar": (2.5e-3, 5e-5),
    "BsNineRichOrderMin": (2.00, 0.005), "BsNineRootTol": (1e-4, 0), "BsNineFeasHminShiftMaxPct": (0.03, 0.005),
    # R-64.1: the collapse, eta_0 basis and effective-viscosity basis
    "BsNineCollapseScatter": (1.54, 0.005), "BsNineCollapseSolventSpread": (2.3, 0.05),
    "BsNineCollapseScatterEff": (1.12, 0.005), "BsNineCollapseSolventSpreadEff": (0.31, 0.005),
    # R-65.1: the collapse on the solvent viscosity eta_s
    "BsNineCollapseScatterSolv": (1.12, 0.005), "BsNineCollapseSolventSpreadSolv": (0.37, 0.005),
    # R-63.6: Section 3 macros
    "BsNineSatFloorDelta": (0.01, 5e-6), "BsNineSatFloorLoadErrPct": (2.4, 0.05),
    "BsNineSatShift": (-0.056, 5e-4), "BsNineSatShiftLo": (-0.057, 5e-4), "BsNineSatShiftHi": (-0.054, 5e-4),
    "BsNineSatNewtonAgree": (0.004, 5e-4), "BsNineSatMemOff": (2e-5, 5e-6),
    "BsNineRimRatio": (0.102, 5e-4), "BsNineRimRatioArea": (0.162, 5e-4),
    "BsNineSqAreaRatioLo": (0.29, 0.005), "BsNineSqAreaRatioHi": (0.35, 0.005),
    # R-63.3: E15 (box G), within-box-G values, signed path
    "BsNineEFifteenCZeroStarUm": (2.25, 0.005), "BsNineEFifteenLamStarR": (2.395, 5e-4),
    "BsNineEFifteenLamStarUnc": (0.0002, 5e-5), "BsNineEFifteenLamTwoUm": (2.393, 5e-4),
    "BsNineEFifteenLamAtStar": (2.399, 5e-4), "BsNineEFifteenGainPct": (0.25, 0.005),
    "BsNineEFifteenLamStarHi": (1.540, 5e-4), "BsNineEFifteenSigmaScalePct": (0.09, 0.005),
    # R-65.1: operating space re-based on S_s (eta_s), sigma* lo S50s with 95 % bootstrap (names kept)
    "BsNineOpsSFiftyFilmOne": (35.2, 0.05), "BsNineOpsSFiftyFilmOneCiLo": (34.4, 0.05),
    "BsNineOpsSFiftyFilmOneCiHi": (36.0, 0.05),
    "BsNineOpsSFiftyFilmTwo": (23.4, 0.05), "BsNineOpsSFiftyFilmTwoCiLo": (22.8, 0.05),
    "BsNineOpsSFiftyFilmTwoCiHi": (23.9, 0.05),
    "BsNineOpsSFiftyFilmThree": (15.3, 0.05), "BsNineOpsSFiftyFilmThreeCiLo": (15.0, 0.05),
    "BsNineOpsSFiftyFilmThreeCiHi": (15.7, 0.05),
    "BsNineOpsSFiftyFullOne": (5.19, 0.005), "BsNineOpsSFiftyFullOneCiLo": (4.80, 0.005),
    "BsNineOpsSFiftyFullOneCiHi": (5.61, 0.005),
    "BsNineOpsSFiftyFullTwo": (3.46, 0.005), "BsNineOpsSFiftyFullTwoCiLo": (3.15, 0.005),
    "BsNineOpsSFiftyFullTwoCiHi": (3.78, 0.005),
    "BsNineOpsSFiftyFullThree": (2.00, 0.005), "BsNineOpsSFiftyFullThreeCiLo": (1.78, 0.005),
    "BsNineOpsSFiftyFullThreeCiHi": (2.20, 0.005),
    "BsNineOpsGThreeValidPct": (65.6, 0.05), "BsNineOpsSFiftyShiftValidPct": (4.6, 0.05),
    "BsNineOpsCountFullLo": (1972, 0), "BsNineOpsCountMixedLo": (2477, 0),
    "BsNineOpsCountContactLo": (7, 0), "BsNineOpsCountNoneLo": (3736, 0),
    "BsNineOpsCountFullHi": (1193, 0), "BsNineOpsCountMixedHi": (3256, 0),
    "BsNineOpsCountContactHi": (23, 0), "BsNineOpsCountNoneHi": (3720, 0),
    "BsNineOpsSurviveMinPct": (65.8, 0.05), "BsNineOpsSurviveWgPct": (28.5, 0.05),
    "BsNineOpsCornerPoints": (40, 0), "BsNineOpsCornerAboveFull": (30, 0),
    "BsNineOpsCornerAboveFilm": (20, 0),
}

def _thin_thousands() -> None:
    """Exact counts of five or more digits typeset with a thin-space
    separator (196\,602, not 196602).  Presentation only: the declared value
    is still what is checked.  Four-digit counts are left unspaced, which is
    the convention the chapter already follows (8192)."""
    for k, (v, tol) in DECLARED.items():
        if (k not in DISPLAY and tol == 0 and isinstance(v, int)
                and not isinstance(v, bool) and abs(v) >= 10000):
            DISPLAY[k] = f"{v:,}".replace(",", r"\,")


#: LaTeX rendering for macros whose plain decimal form is wrong in the chapter.
#: Bodies use \ensuremath, never bare $...$: the same macro is consumed in text
#: mode by the prose and inside math by the tables, and a $-delimited body
#: closes the surrounding math and fails with 'Missing $ inserted'.
#: The VALUE is still whatever DECLARED says and is still checked against the
#: artefact; this only controls how it is typeset.  Every entry here is a
#: presentation proposal awaiting sign-off, listed in the hand-back.
DISPLAY: dict[str, str] = {
    # Exponent form, as the prose value block writes them.  \ensuremath, never
    # bare $...$: the prose consumes these macros in TEXT mode and the tables
    # consume the same macros inside math, and a $-delimited body closes the
    # surrounding math and fails with "Missing $ inserted".  Found by compiling,
    # 2026-08-03.
    "BsNineEOneLamGThreeLo": r"\ensuremath{3.24 \times 10^{-3}}",
    "BsNineEOneLamGThreeHi": r"\ensuremath{3.11 \times 10^{-3}}",
    "BsNineEOneLamAll":  r"\ensuremath{2.95 \times 10^{-4}}",
    # "an aspect ratio of order \BsNineAspectRatio{}" must not read "7e8"
    "BsNineAspectRatio": r"\ensuremath{7 \times 10^{8}}",
    # R-72: lambda axis, split and the E15 relaxation time
    "BsNineLamAxisLo": r"\ensuremath{10^{-5}}", "BsNineLamAxisHi": r"\ensuremath{10^{-2}}",
    "BsNineLamSplit": r"\ensuremath{10^{-3}}", "BsNineEFifteenLamS": r"\ensuremath{10^{-5}}",
    "BsNinePCLamLo": "2.40",
    # Chapter 8 groups integers of five digits or more (63\,013, 131\,072)
    "BsNineNExtended":   r"90\,000",
    # Box F: declared at two decimals; f"{want}" would drop the trailing zero
    # and print "3.9 pp" / "0.7 pp", quietly quoting less precision than declared.
    "BsNineEFourteenJointLoMin": "3.90",
    "BsNineFricTorqueMaxNm": "0.60",
    "BsNineOpsSFiftyFullThreeMin": "2.00",
    # R-65.1: S_s S50s declared at three figures
    "BsNineOpsSFiftyFullOneCiLo": "4.80", "BsNineOpsSFiftyFullThree": "2.00", "BsNineOpsSFiftyFullThreeCiHi": "2.20",
    "BsNineEFourHydroPct": "6.0",
    "BsNineExpSqFirst": r"\ensuremath{-0.94}",
    "BsNineExpSqLast": r"\ensuremath{-0.30}",
    "BsNineMemShareMin": r"\ensuremath{1.6 \times 10^{-6}}",
    "BsNineEFourSweepMax": r"\ensuremath{9.1 \times 10^{-4}}",
    "BsNineEFourSweepMaxDTwo": r"\ensuremath{1.0 \times 10^{-3}}",
    "BsNineEFifteenLamStarHi": "1.540",
    "BsNineSecondOrderPlanar": r"\ensuremath{2.5 \times 10^{-3}}", "BsNineRichOrderMin": "2.00",
    "BsNineRootTol": r"\ensuremath{10^{-4}}", "BsNineDesqAtMemMax": "0.0045",
    "BsNineSatShift": r"\ensuremath{-0.056}", "BsNineSatShiftLo": r"\ensuremath{-0.057}",
    "BsNineSatShiftHi": r"\ensuremath{-0.054}", "BsNineSatMemOff": r"\ensuremath{2 \times 10^{-5}}",
    "BsNineEFourSweepMedian": r"\ensuremath{2.5 \times 10^{-5}}",
    "BsNineEOneLamMinViol": r"\ensuremath{4.5 \times 10^{-4}}",
    "BsNineLamGThreeCorner": r"\ensuremath{4.5 \times 10^{-4}}",
    "BsNineMemShareMax": r"\ensuremath{1.2 \times 10^{-4}}",
    "BsNineEFourSpinMaxN": r"\ensuremath{2 \times 10^{-14}}",
    "BsNineEFourteenNetCiLoHiWg": "0.70",
}
_thin_thousands()

#: COMPARATIVE AUDIT (convention added 2026-08-01).  A comparative between two
#: macros is itself a claim requiring verification.  The macro pipeline checks
#: numerals, so a comparative sitting BETWEEN two verified numerals passes
#: straight through; every failure of the last three passes was of exactly this
#: kind ("an order of magnitude below", "compares slivers rather than regions",
#: "both samplers saturate").  Each entry names the prose clause, the relation
#: it asserts, and a predicate over derived values.  A false relation is a HARD
#: failure with the same standing as a wrong numeral.
#:
#: Prefer relations that cannot be reversed by a choice of summary statistic.
#: C6 is the model: non-overlap of two distributions settles the question
#: outright, where a ratio only settles it for the statistic chosen.
COMPARATIVES: list[tuple[str, str, str]] = [
    ("C1", "sec:disc_geography: 'one constraint dominates by nearly an order "
           "of magnitude' at the slipper",
     "ArSevenGThreeOnly / ArSevenGOneOnly >= 7"),
    ("C2", "sec:disc_geography: the slipper envelope sits below its material "
           "limit",
     "BsNineSlipperTmax < BsNineMineralCeiling"),
    ("C3", "sec:disc_thermal: the routing routes invert between interfaces",
     "BsNineSlipperBrink > BsNineSlipperQmin and "
     "BsNineBsQmin > BsNineBsBrink"),
    ("C4", "sec:disc_transfer: 'the relative gain is larger at the harder "
           "problem' (load-bearing)",
     "(BsNineConvBoLo - BsNineConvBase) / BsNineConvBase > "
     "(BsNineSlipperConvBo - BsNineSlipperConvBase) / BsNineSlipperConvBase"),
    ("C5", "sec:disc_transfer: ball-socket recall is 'roughly half' the "
           "slipper's",
     "0.5 <= BsNineRecallLo / BsNineSlipperRecall <= 0.7 and "
     "0.5 <= BsNineRecallHi / BsNineSlipperRecall <= 0.7"),
    ("C6", "sec:bs_classifier: the refit and between-arm spreads do NOT "
           "overlap, so no summary statistic can reverse the comparison",
     "BsNineRefitSpreadMax < BsNineArmSpreadMin"),
    ("C7", "sec:bs_expectations: the E6 band clears the slipper's L2 fraction "
           "entirely",
     "BsNineESixLo > BsNineSlipperLTwo"),
    ("C8", "sec:bs_reconstruction: the uniform arm beats the Bayesian arm on "
           "fraction of ceiling on the FILM projections at both bands",
     "BsNineIouRandFracLo > BsNineIouBoFracLo and "
     "BsNineIouRandFracHi > BsNineIouBoFracHi"),
    # The blanket claim C8b was written to check is gone from the prose
    # (2026-08-01 ruling); what replaces it is a set of counted claims, and
    # each is checked against the SAME enumeration the counts are drawn from.
    ("C8b", "sec:bs_reconstruction: uniform wins BsNineIouRandWins of "
            "BsNineProjCells overlap cells",
     "BsNineIouRandWins == _n_cells - len(_bo_wins_iou_cells) "
     "and BsNineProjCells == _n_cells"),
    ("C8c", "sec:bs_reconstruction: BO wins BsNineDhBoWins of BsNineProjCells "
            "distance cells, and BsNineBoWinsTotal of BsNineCellsTotal overall",
     "BsNineDhBoWins == len(_bo_wins_dh_cells) and "
     "BsNineBoWinsTotal == len(_bo_wins_iou_cells) + len(_bo_wins_dh_cells) "
     "and BsNineCellsTotal == 2 * _n_cells"),
    ("C8d", "sec:bs_reconstruction: 'the Bayesian arm reconstructs the region "
            "less well and its boundary better' -- a majority each way",
     "len(_bo_wins_iou_cells) < _n_cells / 2 and "
     "len(_bo_wins_dh_cells) > _n_cells / 2"),
    ("C8e", "sec:bs_reconstruction: the quoted strongest d_H pair is the same "
            "cell and BO is the smaller",
     "BsNineDhBoBest < BsNineDhRandBest"),
    # Quantifier-derived, per the 2026-08-01 convention: the Ch.10 sentence
    # quantifies over configurations AND over both metrics simultaneously, so
    # the check must enumerate cells and test the conjunction, not each metric
    # separately.
    # SCOPED 2026-08-01 by ruling.  The unscoped sentence was false in exactly
    # one cell, lo/(P_loss, F), which sits in the extractor-limited tier where
    # both arms are below 7 per cent of ceiling.  The sentence is now scoped to
    # the meaningful tier and the relation is scoped with it.  The unscoped
    # relation is retained as C12b so the exception stays visible rather than
    # disappearing behind the narrower claim.
    ("C12", "sec:disc_transfer (Ch.10): 'no configuration of the Bayesian loop "
            "improved on uniform sampling by both registered metrics at once' "
            "-- scoped to the MEANINGFUL tier",
     "len(_bo_wins_both_meaningful) == 0"),
    ("C12b", "the same claim unscoped, recorded so the extractor-limited "
             "exception is not lost: it is FALSE, in one cell",
     "len(_bo_wins_both_cells) == 1"),
    ("C9", "sec:bs_reconstruction: the Bayesian arm retains MORE feasible "
           "observations, so the result is not an acquisition artefact",
     "BsNineFeasBoLo > BsNineFeasRandLo and BsNineFeasBoHi > BsNineFeasRandHi"),
    ("C10", "sec:bs_ceiling_validity: the ball-socket oracle draw is far "
            "larger a share of its reference than the slipper's",
     "BsNineBsOracleDrawLo > 10 * BsNineSlipperOracleDraw"),
]

#: Slipper macros the comparatives need, read from the thesis macro file so a
#: Chapter 10 comparative cannot drift from what Chapter 7 actually prints.
CH78_MACROS = ("ArSevenGThreeOnly", "ArSevenGOneOnly")


CH78_PATH = pathlib.Path("Z:/PhD Thesis/ch78_numbers.tex")

#: Why each external macro is missing, if it is: "source unreadable" (the file
#: or its drive is absent) is an ENVIRONMENT failure; "not defined in the
#: readable source" is a CLAIM failure.  R-42.10: a HARD failure that cannot
#: tell a missing drive from a missing macro is a gate that cries wolf -- the
#: 2026-09-19 C1 failure was the former and read as the latter.
CH78_STATUS: dict[str, str] = {}


def _load_ch78() -> dict[str, float]:
    import re
    CH78_STATUS.clear()
    path = CH78_PATH
    if not path.exists():
        for name in CH78_MACROS:
            CH78_STATUS[name] = (f"SOURCE UNREADABLE: {path} is not present "
                                 f"(drive {path.drive or '?'} unmounted?) -- an "
                                 f"environment failure, not a claim failure")
        return {}
    txt = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, float] = {}
    for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", txt):
        if name not in CH78_MACROS:
            continue
        clean = val.replace("\\,", "").replace(",", "").strip()
        try:
            out[name] = float(clean)
        except ValueError:
            CH78_STATUS[name] = f"defined in {path} but not numeric: {val!r}"
    for name in CH78_MACROS:
        if name not in out and name not in CH78_STATUS:
            CH78_STATUS[name] = (f"CLAIM UNRESOLVED: {path} is readable but does "
                                 f"not define \{name}")
    return out


def check_comparatives(derived: dict) -> tuple[list[str], list[tuple]]:
    """Evaluate each relation over the DERIVED values.  Returns (failures, rows)."""
    env = {k: v[0] for k, v in derived.items()}
    env.update(_load_ch78())
    failures, rows = [], []
    for tag, clause, relation in COMPARATIVES:
        try:
            ok = bool(eval(relation, {"__builtins__": {"len": len}}, env))  # noqa: S307
            detail = ""
        except NameError as exc:
            missing = getattr(exc, "name", None)
            why = CH78_STATUS.get(missing) if missing else None
            ok, detail = False, (f" ({why})" if why else
                                 f" (CLAIM UNRESOLVED -- no derivation for: {exc})")
        rows.append((tag, relation, "holds" if ok else "FALSE"))
        if not ok:
            failures.append(f"{tag}: relation FALSE -- {relation}{detail}\n"
                            f"        clause: {clause}")
    return failures, rows


#: Macros the prose declares that this generator refuses to emit, with the
#: reason.  Each is a HARD failure until an author's ruling resolves it.
#:
#: \BsNineFilmAreaFrac was blocked here on 2026-08-01 and is now RETRACTED from
#: the prose entirely (ledger L-18), so it no longer appears.  Its replacements,
#: \BsNineNormAreaLo/Hi and \BsNineAspectRatio, are ordinary derived macros.
BLOCKED: dict[str, str] = {}


# ---------------------------------------------------------------------------
# CLAIMS REGISTER  (ruling R-18.3)
#
# Abort-on-mismatch protects NUMERALS.  The last four rulings were about
# SENTENCES, and a sentence has no generator: every macro in
# "the water-glycol envelope reaches \BsNineEThreeWgEnv against a
# \BsNineWgCeiling ceiling" will keep resolving perfectly after box D, while
# the caveat that makes the sentence TRUE -- that the margin is set by inlet
# temperature, not by the dissipative rise -- has nowhere to live.
#
# This register is CHECKED, not merely written down.  A register that nothing
# reads is precisely the defect it exists to fix: box D's own R-17 catch was a
# corrected quantity that was written and read by nothing.  The gate below
# makes a claim whose truth conditions have moved fail LOUDLY at the moment the
# box-D artefacts land, exactly as a numeral does.
#
# STATUS values:
#   STANDS          verdict and basis both survive box D unchanged
#   CONDITIONAL     still true, but only with a stated caveat that MUST travel
#                   with it; the caveat is named in `caveat`
#   PENDING_BOXD    truth conditions moved; must be re-adjudicated against the
#                   box-D artefacts before the sentence may be published
#   WITHDRAWN       retracted; kept visible with its reason, never deleted
#   PROHIBITED      a class of number that may not be reported; enforced by
#                   check_prohibitions() (ERR-18), not by this entry alone
#
# The form follows \BsNineFilmAreaFrac, which is already recorded in the prose
# as ASSERTED / RETRACTED with its reason and a note that the paragraph it
# supported was removed.  A retracted claim kept visible with its reason is
# worth more than a deleted one; it should not stay the only one.
# ---------------------------------------------------------------------------

CLAIMS: dict[str, dict] = {
    "S2_needs_blowup_population_lambda": {
        "sentence": ("the joint runs below the full-film threshold at the "
                     "load-carrying limit (S2, in its corrected film-thickness "
                     "form)"),
        "status": "PENDING_BOXD",
        "moved_by": "R-20.4 (corrected route), E8 (the statistic it needs)",
        "why": ("S2's corrected form is a claim about Lambda, and the ONLY "
                "Lambda currently derivable is over the CONVERGED population, "
                "whose median is above the full-film criterion BY "
                "CONSTRUCTION -- a converged row is one that found a full-film "
                "equilibrium.  The statistic S2 requires is median Lambda over "
                "the BLOW-UP population, registered expectation E8, which does "
                "not exist until box D lets those rows reach equilibrium.  "
                "Registered as owed rather than approximated by the "
                "converged-population figure, which would make S2 look "
                "supported when it is not."),
        "macros_affected": (r"\BsNineLamMedConvLo", r"\BsNineLamMedConvHi"),
        "alias_to_add_when_box_D_lands": ("PThreeLamMedBlowupLo / "
                                          "PThreeLamMedBlowupHi"),
    },
    "outcome_taxonomy_blowup_is_film_failure": {
        "sentence": ("blow-up rows are reported as film failure, and the "
                     "binding tallies count them as such"),
        "status": "PENDING_BOXD",
        "moved_by": "ERR-8 (eps_cap reach-back)",
        "why": ("at the corrected bracket many blow-up rows resolve to an "
                "equilibrium.  Every binding tally conditioned on convergence "
                "therefore changes meaning, and the box-B blow-up figures "
                "become UPPER BOUNDS"),
        "macros_affected": (r"\BsNineAuditBlowupPct", r"\BsNineAuditBlowup",
                            r"\BsNineAuditConverged", r"\BsNineAuditConvPct"),
    },
    "E5_confirmed_strong_form": {
        "sentence": ("the rated corner lies outside the full-film feasible "
                     "set, confirmed in its strong form"),
        "status": "PENDING_BOXD",
        "moved_by": "ERR-8",
        "why": ("its box-B basis rested PARTLY on the capped bracket.  The "
                "verdict may well survive, but when it is quoted the "
                "statement must say WHICH PART OF ITS BASIS MOVED, rather "
                "than being inherited silently"),
        "macros_affected": (),
    },
    "contact_never_engages": {
        "sentence": "asperity contact never engages anywhere in the envelope",
        "status": "WITHDRAWN",
        "moved_by": "R-11, R-13",
        "why": ("an artefact of the capped bracket: the search was stopped "
                "before it could reach the eccentricities at which contact "
                "engages.  The claim was never true of the joint, only of the "
                "search"),
        "macros_affected": (),
    },
    "E3_envelope_presses_the_limit": {
        "sentence": ("the water-glycol thermal envelope presses against its "
                     "class ceiling"),
        "status": "CONDITIONAL",
        "moved_by": "R-14.4",
        "caveat": ("THE MARGIN IS SET BY INLET TEMPERATURE, NOT BY THE "
                   "DISSIPATIVE RISE.  The T_max bias lives in the RISE, so "
                   "the verdict survives it by ~142x -- but it is a property "
                   "of THIS population and must be re-checked, never "
                   "inherited"),
        "why": ("both macros keep resolving after box D while the caveat that "
                "makes the sentence true has no macro of its own.  This is "
                "the sharp case that motivated the register"),
        "macros_affected": (r"\BsNineEThreeWgEnv", r"\BsNineWgCeiling"),
    },
    "corrections_are_one_directional": {
        "sentence": ("the corrections are systematic and one-directional -- "
                     "the reporting convention was optimistic about "
                     "feasibility throughout"),
        "status": "WITHDRAWN",
        "moved_by": "R-16.6, restated R-17.6",
        "why": ("true of the DEFECTS, false of the CORRECTIONS, and only the "
                "corrections are measurable.  One correction (eps_cap) "
                "ENLARGES the feasible set; four (h_min, De_sq, T_max, F3) "
                "SHRINK it.  The net is registered as box-D expectation E14 "
                "and must be MEASURED, not asserted"),
        "macros_affected": (),
    },
    "E18_averaged_film_power_loss": {
        "sentence": ("any power-loss or efficiency number taken from an "
                     "averaged-film (Patir-Cheng) run"),
        "status": "PROHIBITED",
        "moved_by": "R-41.8, defect register ERR-18",
        "why": ("P_loss is computed with the SMOOTH-film shear relation applied "
                "to AVERAGED pressure fields; the Patir-Cheng shear-stress "
                "factors (phi_f, phi_fs, phi_fp) are not implemented.  Box F's "
                "monotonicity test saw P_loss rise on 160/160 rows while eps* "
                "fell on 160/160 -- thermodynamically odd on its face, and the "
                "signature of the missing factors.  No such number may be "
                "reported until the factors are a registered component, or it "
                "carries a smooth-film-only qualifier at the point of use.  "
                "ENFORCED by check_prohibitions(), not only recorded here.  "
                "SCOPE (R-42.2): averaged-film runs ONLY -- the smooth shear "
                "relation is exact for a smooth film, so boxes B-E's thermal "
                "numbers and binding labels are outside this prohibition"),
        "macros_affected": (),
    },
}

#: ERR-18 (R-41.8).  A prohibition that lives only in an addendum is one a future
#: draft walks past, so it is enforced at two independent points:
#:
#:   ARTEFACT  -- any loaded frame that carries BOTH an averaged-film
#:                realised_film_model and a power-loss column is refused: the
#:                loader must drop the column before the frame reaches derive(),
#:                so the number cannot be derived at all.
#:   MACRO     -- any derived macro whose name or provenance names power loss /
#:                efficiency AND whose provenance names an averaged-film source
#:                is refused.
#:
#: The macro check alone would trust free-text provenance; the artefact check
#: alone would miss a number computed elsewhere and passed in.  Either catches it.
POWER_PATTERN = r"(?i)(p_?loss|power|effic)"
AVERAGED_PATTERN = r"(?i)(averag|boxF_band|boxF_R41|patir)"
POWER_COLUMNS = ("P_loss", "P_loss_bnd", "efficiency")


def check_prohibitions(artefacts: dict, derived: dict) -> list[str]:
    """HARD failures for any ERR-18 violation (see POWER_PATTERN above)."""
    import re
    hard: list[str] = []
    for key, v in artefacts.items():
        cols = getattr(v, "columns", None)
        if cols is None or "realised_film_model" not in cols:
            continue
        averaged = v["realised_film_model"].astype(str).str.startswith("averaged").any()
        leaked = [c for c in POWER_COLUMNS if c in cols]
        if averaged and leaked:
            hard.append(f"ERR-18: artefact {key!r} is an averaged-film run and still "
                        f"carries {leaked}; drop the column(s) at load")
    for macro, (_, src) in derived.items():
        if (re.search(POWER_PATTERN, macro) or re.search(POWER_PATTERN, str(src)))                 and re.search(AVERAGED_PATTERN, str(src)):
            hard.append(f"ERR-18: {macro} is a power-loss/efficiency number from an "
                        f"averaged-film source [{src}]; PROHIBITED")
    return hard

#: Artefact whose presence means box D has landed.  Keyed off the audit file
#: the loader actually reads, not off a flag someone must remember to set.
#:
#: THE MATCH IS NORMALISED, AND THAT IS NOT FUSSINESS (ruling R-22.2).  Until
#: ``write_parquet`` began stripping ``df.attrs``, this gate had TWO arming
#: routes: the artefact key and the attrs blob.  The attrs strip was correct and
#: necessary -- timing provenance was leaking into the artefact's sha256 -- but
#: it silently removed the redundant route, leaving the FILENAME as the only
#: thing that arms the gate.
#:
#: A literal contiguous substring test then made three plausible names disarm it:
#:
#:     audit_N8192_boxD_seed...   -- marker split by the N8192 segment
#:     audit_boxd_N8192_...       -- lower-case d
#:     audit_BoxD_N8192           -- different capitalisation
#:
#: A gate one underscore away from being silently off is not a gate.  Matching
#: is therefore done on a normalised key -- case-folded, separators removed --
#: so any name carrying "box d" in any casing or separation arms it.  Widening
#: is the FAIL-SAFE direction: an over-armed gate costs a spurious HARD failure
#: that a human resolves in a minute, while an under-armed one ships stale
#: claims against corrected numbers, which is what the gate exists to prevent.
BOXD_ARTEFACT_MARKER = "audit_boxD"


def _arms_boxd_gate(key: str) -> bool:
    """True when an artefact key denotes a box-D artefact."""
    norm = "".join(ch for ch in str(key).lower() if ch.isalnum())
    return "boxd" in norm


def check_claims(artefacts: dict) -> tuple[list[str], list[tuple]]:
    """Gate the claims register.  Returns (hard_failures, table_rows).

    PENDING_BOXD claims become HARD failures the moment a box-D artefact is
    present, so a sentence whose truth conditions moved cannot be published
    against corrected numbers without being re-adjudicated first.  Before box D
    lands they are reported but do not block, because the corrected numbers do
    not yet exist and nothing can be re-adjudicated against them.

    CONDITIONAL claims are ALWAYS reported, box D or not: a caveat that is only
    printed when something breaks is a caveat that will be forgotten.
    """
    # NOTE: the attrs route below is retained but is now DEAD for anything
    # written through write_parquet, which strips attrs (R-22.2).  It is kept
    # only for in-memory frames that never round-tripped through parquet.  The
    # KEY route is the live one, and it is normalised for the reason above.
    boxd_present = any(_arms_boxd_gate(k) for k in artefacts) or \
        any(BOXD_ARTEFACT_MARKER in str(getattr(v, "attrs", {}))
            for v in artefacts.values() if hasattr(v, "attrs"))

    hard: list[str] = []
    rows: list[tuple[str, str, str]] = []
    for cid, c in sorted(CLAIMS.items()):
        st = c["status"]
        note = c.get("caveat") or c["why"]
        rows.append((cid, st, note))
        if st == "PENDING_BOXD" and boxd_present:
            hard.append(
                f"CLAIM {cid}: PENDING_BOXD and box-D artefacts are present.  "
                f"Truth conditions moved ({c['moved_by']}).  Re-adjudicate the "
                f"sentence against the corrected numbers before publishing it, "
                f"then set status to STANDS or WITHDRAWN.")
        if st == "CONDITIONAL" and not c.get("caveat"):
            hard.append(f"CLAIM {cid}: CONDITIONAL with no caveat recorded.  "
                        f"A conditional claim without its condition is a "
                        f"claim, not a conditional.")
    return hard, rows


# ---------------------------------------------------------------------------
# PAPER 3 MACRO SET  (ruling R-18.2, Nigel's decision 2026-08-31)
#
# THE PROBLEM THIS SOLVES.  The re-scope inverted which campaign Paper 3 rests
# on -- from the AR/BO reconstruction to the ball-socket result -- and the
# number pipeline did not follow.  giesekus-solver/scripts/paper3_numbers.py is
# a good generator, but its canonical artefact is the AR exhaustive parquet and
# it reads ZERO ball-socket artefacts, so when box D re-runs it notices
# nothing.  As wired before this block, the re-scoped Paper 3 had no generator
# at all and its numbers would have been hand-transcribed out of Chapter 9 --
# into a document bound in the SAME thesis, where an examiner is most likely to
# cross-check.
#
# THE FIX, AND WHY IT IS AN ALIAS AND NOT A COPY.  A Paper 3 macro does not
# hold a value.  It names a Chapter 9 DERIVATION KEY, and both macros are
# emitted from that one derivation in the same pass.  The two documents cannot
# disagree because there is only one number: agreement is STRUCTURAL, not
# checked after the fact.  A cross-check between two generators could only ever
# report a disagreement that had already happened.
#
# WHY THIS GENERATOR AND NOT THE OTHER.  The artefacts live here, and this
# module ALREADY reads across into giesekus-solver for the slipper comparator
# (see _load: figures/bo_corrected, results/recompute, logs/diagnostics).  The
# ball-socket -> solver direction is proven; the reverse has never been wired.
#
# SCOPE, STATED HONESTLY.  No Paper 3 manuscript exists on this machine, so the
# SET below cannot be complete -- nobody knows yet which numbers the re-scoped
# paper quotes.  What is delivered now is the MECHANISM plus the quantities
# that are unambiguously the re-scoped paper's headline results.  Adding a
# macro later is one line; retro-fitting a pipeline after two documents quote
# 8192 corrected rows independently is not.  The set is PROVISIONAL; the
# guarantee that Paper 3 and Chapter 9 agree is not.
# ---------------------------------------------------------------------------

#: Paper 3's own folder on the share, following Paper 2's convention
#: (paper2_numbers.tex + paper2_macros.tex live inside the paper's directory).
#: Authorised destination 2026-08-31.  Falls back to the repo staging path when
#: the share is unreachable, so the generator still runs off-network -- but the
#: staging-hop check below then reports that the live copy was not written.
PAPER3_DIR_LIVE = pathlib.Path("Z:/Papers/Paper 3/New")
PAPER3_TEX_LIVE = PAPER3_DIR_LIVE / "paper3_numbers.tex"
PAPER3_TEX_OUT = ROOT / "results" / "ch9" / "paper3_numbers_from_ch9.tex"

#: Paper 3 macro -> Chapter 9 macro whose derivation it aliases.
#: EVERY entry must name a key that DECLARED already carries, so a Paper 3
#: macro can never introduce a number that Chapter 9 has not checked.
PAPER3_ALIASES: dict[str, str] = {
    # Completed 2026-08-31 against paper3_rescope_memo.tex (18 Aug 2026), which
    # states the claim structure S1-S6 and the eight-figure plan.  Every entry
    # below backs a numeral the memo actually uses, so the set is no longer a
    # guess at what the paper quotes.

    # --- S2, the load-carrying limit: the paper's headline result ----------
    "PThreeNAudit":            "BsNineNAudit",
    "PThreeAuditConverged":    "BsNineAuditConverged",
    "PThreeAuditConvPct":      "BsNineAuditConvPct",
    "PThreeAuditBlowup":       "BsNineAuditBlowup",
    "PThreeAuditBlowupPct":    "BsNineAuditBlowupPct",
    "PThreeBlowupLoadMed":     "BsNineBlowupLoadMed",
    "PThreeConvLoadMed":       "BsNineConvLoadMed",

    # --- S1, the mechanism: inverted thermal routing (F7) ------------------
    "PThreeBsQmin":            "BsNineBsQmin",
    "PThreeBsBrink":           "BsNineBsBrink",
    "PThreeSlipperQmin":       "BsNineSlipperQmin",
    "PThreeSlipperBrink":      "BsNineSlipperBrink",

    # --- S3, the rated corner (E5) ----------------------------------------
    "PThreeEFiveNuMax":        "BsNineEFiveNuMax",

    # --- S4, fluid class sets the thermal limit (E3) ----------------------
    "PThreeFullFilmMin":             "BsNineFullFilmMin",
    "PThreeEOneViolPct":                 "BsNineEOneViolPct",
    "PThreeEOneLamMinViol":              "BsNineEOneLamMinViol",
    "PThreeEOneAboveMilliPct":           "BsNineEOneAboveMilliPct",
    "PThreeEOneSoleMinPct":              "BsNineEOneSoleMinPct",
    "PThreeEOneSoleMaxPct":              "BsNineEOneSoleMaxPct",
    "PThreeLamGThreeCorner":             "BsNineLamGThreeCorner",
    "PThreeEOneInRangePct":              "BsNineEOneInRangePct",
    "PThreeEOneDesqMedian":              "BsNineEOneDesqMedian",
    "PThreeEOneDesqQThree":              "BsNineEOneDesqQThree",
    "PThreeEOneQThreeRatio":             "BsNineEOneQThreeRatio",
    "PThreeEOneBoxBInRangePct":          "BsNineEOneBoxBInRangePct",
    "PThreeEOneBoxBViolPct":             "BsNineEOneBoxBViolPct",
    "PThreeEFourSweepN":                 "BsNineEFourSweepN",
    "PThreeEFourSweepBase":              "BsNineEFourSweepBase",
    "PThreeEFourSweepInDomain":          "BsNineEFourSweepInDomain",
    "PThreeEFourSweepMax":               "BsNineEFourSweepMax",
    "PThreeEFourSweepMaxDTwo":           "BsNineEFourSweepMaxDTwo",
    "PThreeEFourSweepMedian":            "BsNineEFourSweepMedian",
    "PThreeEFourSweepFloorRows":         "BsNineEFourSweepFloorRows",
    "PThreeEFourSweepWiMin":             "BsNineEFourSweepWiMin",
    "PThreeEFourSweepRaisesN":           "BsNineEFourSweepRaisesN",
    "PThreeEFourEvenPct":                "BsNineEFourEvenPct",
    # R-64.1: collapse
    **{f"PThree{s}": f"BsNine{s}" for s in ("CollapseScatter", "CollapseSolventSpread", "CollapseScatterEff",
                                            "CollapseSolventSpreadEff", "CollapseScatterSolv",
                                            "CollapseSolventSpreadSolv", "OpsGThreeValidPct",
                                            "OpsSFiftyShiftValidPct", "LeeRbMm", "LeeLoadKN",
                                            "LeeSigmaUm", "LeeTempC", "FricN", "FricOrderMin",
                                            "FricCorrMaxPct", "FricTorqueMedNm", "FricTorqueMinNm",
                                            "FricTorqueMaxNm", "FricMuMed", "FricMuMin", "FricMuMax",
                                            "FricSample", "FricDeltaSplit", "FricDeltaNonconv",
                                            "ESixBsN", "ESixBsDen", "ESixBsPct", "ESixQminPct",
                                            "ESixBandLo", "ESixBandHi", "FieldsDelta",
                                            "FieldsPeakThetaDeg", "FieldsPeakThetaNewtDeg",
                                            "FieldsPeakMPa", "FieldsPeakNewtMPa", "FieldsEtaRatioMin",
                                            "FieldsEtaRatioMax", "FieldsSolventFrac", "FieldsRiseMaxmK", "LeeCUm",
                                            "LeeEtaMPas", "LeeAlphaStar", "LeePiezoHundred",
                                            "LeePiezoPeak")},
    "PThreeESixSlipperPct": "BsNineSlipperLTwo",
    "PThreeESixCommonTotalPct": "BsNineESixCommonTotalPct", "PThreeBsBrinkCommonPct": "BsNineBsBrinkCommonPct",
    "PThreeBsQminFlagPct": "BsNineBsQminFlagPct",
    # R-68.2: D-8, mineral class
    **{f"PThree{s}": f"BsNine{s}" for s in "ThetaFlowMedMin ThetaFlowAbovePctMin BsBrinkMin BsQminMin NahmeAdiabLoPctMin NahmeAdiabHiPctMin OpsSurviveMinPctMin ETwoCeilingReachPctMin ETwoGapPpMin ThermClearNmMin OpsCountFullLoMin OpsCountMixedLoMin OpsCountNoneLoMin OpsCountFullHiMin OpsCountMixedHiMin OpsCountNoneHiMin OpsCountContactLoMin OpsCountContactHiMin OpsSFiftyFilmOneMin OpsSFiftyFullOneMin OpsSFiftyFilmTwoMin OpsSFiftyFullTwoMin OpsSFiftyFilmThreeMin OpsSFiftyFullThreeMin".split()},
    # R-70.1 / R-70.2
    **{f"PThree{s}": f"BsNine{s}" for s in "WangTempC WangPMaxMPa PiezoAlphaWgDig PiezoGWgDig PiezoAlphaTenDig PiezoAlphaTwentyTwoDig HasanTempC HasanPMaxGPa PiezoAlphaWgLo PiezoAlphaWgHi PiezoGWg PiezoGWgHi BrinkThresh BrinkDeltaT ThetaDeltaTRef DesqMax".split()},
    # R-72: the macros v7 calls, claim points, counts, E16
    **{f"PThree{s}": f"BsNine{s}" for s in "StrokeEFifteenSnapDevPct StrokeEThirteenSnapDevPct StrokeEFiveSnapDevPct StrokeEFifteenTSnapDeg StrokeEThirteenTSnapDeg StrokeEFiveTSnapDeg StrokeEFifteenTcDeg StrokeEThirteenTcDeg StrokeEFiveTcDeg StrokeEFifteenTLamThreeRoelDeg StrokeEThirteenTLamThreeRoelDeg StrokeEFiveTLamThreeRoelDeg StrokeEFifteenLamPiRoel StrokeEThirteenLamPiRoel StrokeEFiveLamPiRoel StrokeEFifteenLamPiRoelSnap StrokeEThirteenLamPiRoelSnap StrokeEFiveLamPiRoelSnap StrokeEFifteenLamPiGTwo StrokeEThirteenLamPiGTwo StrokeEFiveLamPiGTwo StrokeEFifteenTHundredthDeg StrokeEFifteenLamPiIso StrokeEFifteenLamPiBarus StrokeEFifteenTLamThreeBarusDeg StrokeEFifteenRDeltaBarus StrokeEFifteenRDeltaBarusFloor StrokeTailPower StrokeGridFloor StrokeGMax".split()},
    **{f"PThree{s}": f"BsNine{s}" for s in "SatDeltaHi SatDeltaLo VerifDeltaHi FilmAreaClosedDelta LamAxisLo LamAxisHi LamSplit AdiabFactor PCLamLo PCLamHi EFifteenDesq EThirteenDesq EFiveDesq EFifteenCZeroUm EThirteenCZeroUm EFiveCZeroUm EFifteenThetaMinDeg EThirteenThetaMinDeg EFiveThetaMinDeg EFifteenEtaZeroMPas EFifteenLamS EFifteenTinC EFifteenPchMPa EFifteenOmega EFifteenSolventFrac EFifteenDesqPiezo OpsFilmOnlyBoth OpsClassThreeLo OpsClassThreeHi ValidRows ValidN ValidExcl ValidCompExcl TabDenInDomain TabDenFrameMin TabDenFrameWg TabDenRestricted TabDenClassOne RatedPoints RatedRows ContactModels OpsSurviveMinPctRun OpsSurviveWgPctRun DesqLocMed DesqLocQThree DesqLocMax DesqLocAbovePct".split()},
    # R-69.3: thermal bracket
    **{f"PThree{s}": f"BsNine{s}" for s in "ThermGTwoBreachPctMinCond ThermGTwoSolePctMinCond ThermNahmeLoPctMinCond ThermNahmeHiPctMinCond ThermBindDomPctLoMinCond ThermBindDomPctHiMinCond ThermRiseMedMinCond ThermOpsSurvivePctMinCond ThermNahmeSixLoPctMinCond ThermNahmeSixHiPctMinCond ThermGTwoBreachPctMinAdv ThermGTwoSolePctMinAdv ThermNahmeLoPctMinAdv ThermNahmeHiPctMinAdv ThermBindDomPctLoMinAdv ThermBindDomPctHiMinAdv ThermRiseMedMinAdv ThermOpsSurvivePctMinAdv ThermGTwoBreachPctWgCond ThermGTwoSolePctWgCond ThermNahmeLoPctWgCond ThermNahmeHiPctWgCond ThermBindDomPctLoWgCond ThermBindDomPctHiWgCond ThermRiseMedWgCond ThermOpsSurvivePctWgCond ThermNahmeSixLoPctWgCond ThermNahmeSixHiPctWgCond ThermGTwoBreachPctWgAdv ThermGTwoSolePctWgAdv ThermNahmeLoPctWgAdv ThermNahmeHiPctWgAdv ThermBindDomPctLoWgAdv ThermBindDomPctHiWgAdv ThermRiseMedWgAdv ThermOpsSurvivePctWgAdv ThetaFlowMedWg ThetaFlowAbovePctWg".split()},
    # R-68.1: D-9 in ratios
    **{f"PThree{s}": f"BsNine{s}" for s in "PiezoGLee PiezoGLeeBarus PiezoGcritEFifteen PiezoGcritEThirteen PiezoGcritEFive PiezoFactorEFifteen PiezoFactorEThirteen PiezoFactorEFive PiezoLamEFifteen PiezoLamEThirteen PiezoLamEFive PiezoKappa PiezoKappaFit PiezoLinDevPct PiezoQPeakFrac PiezoCZeroRatio PiezoLamStar PiezoPeakOverPch".split()},
    # R-64.2: numbers v4 declared as owed
    **{f"PThree{s}": f"BsNine{s}" for s in "ExistN ExistBPct ExistAPct ExistAConf ThetaFlowMed ThetaFlowAbovePct MeanTinMinLo AdiabTinMinLo AdiabTinMinHi NahmeAdiabLoPct NahmeAdiabHiPct ThermClearNm CornerPBar CornerRpm LeeHydWithPct LeeHydWithoutPct DesqAtMemMax WiZero SecondOrderPlanar RichOrderMin RootTol FeasHminShiftMaxPct".split()},
    # R-63.6: Section 3
    **{f"PThree{s}": f"BsNine{s}" for s in
       ("SatFloorDelta", "SatFloorLoadErrPct", "SatShift", "SatShiftLo", "SatShiftHi", "SatNewtonAgree",
        "SatMemOff", "RimRatio", "RimRatioArea", "SqAreaRatioLo", "SqAreaRatioHi")},
    **{f"PThreeEFifteen{s}": f"BsNineEFifteen{s}" for s in
       ("CZeroStarUm", "LamStarR", "LamStarUnc", "LamTwoUm", "LamAtStar", "GainPct", "LamStarHi", "SigmaScalePct")},
    "PThreeESevenCompSamplesPct":    "BsNineESevenCompSamplesPct",
    "PThreeEThreeForcedLo":          "BsNineEThreeForcedLo",
    "PThreeEThreeForcedHi":          "BsNineEThreeForcedHi",
    "PThreeEThreeSoleHi":            "BsNineEThreeSoleHi",
    "PThreeEThreeSoleSamplesLo":     "BsNineEThreeSoleSamplesLo",
    "PThreeEThreeSoleSamplesHi":     "BsNineEThreeSoleSamplesHi",
    "PThreeEThreeBiasDownLo":        "BsNineEThreeBiasDownLo",
    "PThreeEThreeBiasDownHi":        "BsNineEThreeBiasDownHi",
    "PThreeEThreeSoleTinMin":        "BsNineEThreeSoleTinMin",
    "PThreeEThreeRiseMax":           "BsNineEThreeRiseMax",
    "PThreeTinAboveWgPct":           "BsNineTinAboveWgPct",
    "PThreeFilmAreaClosed":          "BsNineFilmAreaClosed",
    "PThreeFilmAreaOpen":            "BsNineFilmAreaOpen",
    "PThreeMixedMin":                "BsNineMixedMin",
    "PThreeInertiaLoPct":            "BsNineInertiaLoPct",
    "PThreeInertiaHiPct":            "BsNineInertiaHiPct",
    "PThreeInertiaMidPct":           "BsNineInertiaMidPct",
    "PThreeEThreeWgEnv":       "BsNineEThreeWgEnv",
    "PThreeWgCeiling":         "BsNineWgCeiling",
    "PThreeEThreeMinCount":    "BsNineEThreeMinCount",
    "PThreeMineralCeiling":    "BsNineMineralCeiling",
    "PThreeSlipperTmax":       "BsNineSlipperTmax",

    # --- G1/G2a: registered 2026-08-31, surfaced by drafting S1/S2/S6 ------
    "PThreeSigmaStarLoUm":     "BsNineSigmaStarLoUm",
    "PThreeSigmaStarHiUm":     "BsNineSigmaStarHiUm",
    # The CONVERGED-population Lambda.  Not S2's evidence -- see the derivation
    # comment.  Aliased so the paper can characterise the converged minority
    # without anyone reaching for it as though it supported S2.
    "PThreeLamMedConvLo":      "BsNineLamMedConvLo",
    "PThreeLamMedConvHi":      "BsNineLamMedConvHi",

    # --- the NOT-claims, which the memo says must be carried verbatim ------
    # A referee who finds one unstated will not trust the rest, so the
    # precision/recall pair is aliased rather than retyped into the paper.
    "PThreePrecLo":            "BsNinePrecLo",
    "PThreePrecHi":            "BsNinePrecHi",
    "PThreeRecallLo":          "BsNineRecallLo",
    "PThreeRecallHi":          "BsNineRecallHi",
    # --- re-scoped Paper 3, sections 1-6 (R-48.12.6), aliased at drafting ---
    "PThreeRbMm":                     "BsNineRbMm",
    "PThreeThetaMaxDeg":              "BsNineThetaMaxDeg",
    "PThreeThetaMinLoDeg":            "BsNineThetaMinLoDeg",
    "PThreeThetaMinHiDeg":            "BsNineThetaMinHiDeg",
    "PThreeCZeroLoUm":                "BsNineCZeroLoUm",
    "PThreeCZeroHiUm":                "BsNineCZeroHiUm",
    "PThreeEpsCeilTwenty":            "BsNineEpsCeilTwenty",
    "PThreeEpsCeilThirty":            "BsNineEpsCeilThirty",
    "PThreeEpsCeilForty":             "BsNineEpsCeilForty",
    "PThreeContactArms":              "BsNineContactArms",
    "PThreeEFourSqueezePct":          "BsNineEFourSqueezePct",
    "PThreeEFourHydroPct":            "BsNineEFourHydroPct",
    "PThreeEFourSpinMaxN":            "BsNineEFourSpinMaxN",
    "PThreeExpSqFirst":               "BsNineExpSqFirst",
    "PThreeExpSqLast":                "BsNineExpSqLast",
    "PThreeMemShareMin":              "BsNineMemShareMin",
    "PThreeMemShareMax":              "BsNineMemShareMax",
    "PThreeETenPlasticPct":           "BsNineETenPlasticPct",
    "PThreeETenDoubtPct":             "BsNineETenDoubtPct",
    "PThreeETenElasticPct":           "BsNineETenElasticPct",
    "PThreeETenCapMedLo":             "BsNineETenCapMedLo",
    "PThreeETenCapMedHi":             "BsNineETenCapMedHi",
    "PThreeETenCapDoubtMax":          "BsNineETenCapDoubtMax",
    "PThreeESevenValid":              "BsNineESevenValid",
    "PThreeESevenNoEq":               "BsNineESevenNoEq",
    "PThreeESevenPct":                "BsNineESevenPct",
    "PThreeESevenThreshPct":          "BsNineESevenThreshPct",
    "PThreeESevenFactor":             "BsNineESevenFactor",
    "PThreeESevenCompValid":          "BsNineESevenCompValid",
    "PThreeESevenCompNoEq":           "BsNineESevenCompNoEq",
    "PThreeESevenCompPct":            "BsNineESevenCompPct",
    "PThreeESevenCompFactor":         "BsNineESevenCompFactor",
    "PThreeSamples":                  "BsNineSamples",
    "PThreeESevenSamplesNoEq":        "BsNineESevenSamplesNoEq",
    "PThreeESevenCompSamplesNoEq":    "BsNineESevenCompSamplesNoEq",
    "PThreeArmMixedSamples":          "BsNineArmMixedSamples",
    "PThreeArmConvAnySamples":        "BsNineArmConvAnySamples",
    "PThreeArmInvariantPct":          "BsNineArmInvariantPct",
    "PThreeESevenBRows":              "BsNineESevenBRows",
    "PThreeESevenBGain":              "BsNineESevenBGain",
    "PThreeESevenBPct":               "BsNineESevenBPct",
    "PThreeESevenBLamMax":            "BsNineESevenBLamMax",
    "PThreeFeasBandCount":            "BsNineFeasBandCount",
    "PThreeFeasBandUnits":            "BsNineFeasBandUnits",
    "PThreeFeasBandLbMin":            "BsNineFeasBandLbMin",
    "PThreeFeasBandLbMax":            "BsNineFeasBandLbMax",
    "PThreeFeasBandMarginPct":        "BsNineFeasBandMarginPct",
    "PThreeEThreeWgFails":            "BsNineEThreeWgFails",
    "PThreeEThreeInletPct":           "BsNineEThreeInletPct",
    "PThreeEThreeOnlyGTwoLo":         "BsNineEThreeOnlyGTwoLo",
    "PThreeEThreeHeatedLo":           "BsNineEThreeHeatedLo",
    "PThreeEFourteenNetLoMin":        "BsNineEFourteenNetLoMin",
    "PThreeEFourteenNetCiLoLoMin":    "BsNineEFourteenNetCiLoLoMin",
    "PThreeEFourteenFThreeLoMin":     "BsNineEFourteenFThreeLoMin",
    "PThreeEFourteenJointLoMin":      "BsNineEFourteenJointLoMin",
    "PThreeEFourteenNetLoWg":         "BsNineEFourteenNetLoWg",
    "PThreeEFourteenNetCiLoLoWg":     "BsNineEFourteenNetCiLoLoWg",
    "PThreeEFourteenFThreeLoWg":      "BsNineEFourteenFThreeLoWg",
    "PThreeEFourteenJointLoWg":       "BsNineEFourteenJointLoWg",
    "PThreeEFourteenNetHiMin":        "BsNineEFourteenNetHiMin",
    "PThreeEFourteenNetCiLoHiMin":    "BsNineEFourteenNetCiLoHiMin",
    "PThreeEFourteenFThreeHiMin":     "BsNineEFourteenFThreeHiMin",
    "PThreeEFourteenJointHiMin":      "BsNineEFourteenJointHiMin",
    "PThreeEFourteenNetHiWg":         "BsNineEFourteenNetHiWg",
    "PThreeEFourteenNetCiLoHiWg":     "BsNineEFourteenNetCiLoHiWg",
    "PThreeEFourteenFThreeHiWg":      "BsNineEFourteenFThreeHiWg",
    "PThreeEFourteenJointHiWg":       "BsNineEFourteenJointHiWg",
    "PThreeEFourteenImmovablePct":    "BsNineEFourteenImmovablePct",
    "PThreeEFourteenNetMovHiWg":      "BsNineEFourteenNetMovHiWg",
    "PThreeEFourteenFThreeMovHiWg":   "BsNineEFourteenFThreeMovHiWg",
    "PThreeEFourteenJointMovHiWg":    "BsNineEFourteenJointMovHiWg",
    "PThreeETwoRestrictedPct":        "BsNineETwoRestrictedPct",
    "PThreeETwoGapPp":                "BsNineETwoGapPp",
    "PThreeETwoCeilingReachPct":      "BsNineETwoCeilingReachPct",
    "PThreeEFiveVgThirtyTwoSolves":   "BsNineEFiveVgThirtyTwoSolves",
    "PThreeEFiveVgThirtyTwoEqRows":   "BsNineEFiveVgThirtyTwoEqRows",
    "PThreeEFiveVgThirtyTwoSoleGOne":   "BsNineEFiveVgThirtyTwoSoleGOne",
    "PThreeEFiveVgThirtyTwoTMaxMax":  "BsNineEFiveVgThirtyTwoTMaxMax",
    # D-7 (R-57.3): the registered "T_max" is the land maximum of the FILM-MEAN
    # temperature; the film-mean name is the one the paper uses, the old name is kept
    # so citing artefacts still resolve.
    "PThreeEFiveVgThirtyTwoTFilmMeanMax": "BsNineEFiveVgThirtyTwoTMaxMax",
    "PThreeFeasBandMarginMeshPct":    "BsNineFeasBandMarginMeshPct",
    "PThreeFilmAreaOpenProj":         "BsNineFilmAreaOpenProj",
    "PThreeFilmAreaOpenHydPct":       "BsNineFilmAreaOpenHydPct",
    "PThreeEFiveVgThirtyTwoRiseMax":  "BsNineEFiveVgThirtyTwoRiseMax",
    # R-58.2: Paper 3 prints the mesh-extrapolated values; the production values keep
    # their own aliases (…Prod) so a reader can see both.
    "PThreeEFiveLamMaxCorner":        "BsNineEFiveLamMaxR",
    "PThreeEFiveLamMaxCornerProd":    "BsNineEFiveLamMaxCorner",
    "PThreeEThirteenLamMax":          "BsNineEThirteenLamMaxR",
    "PThreeEThirteenLamMaxProd":      "BsNineEThirteenLamMax",
    "PThreeEThirteenShortPct":        "BsNineEThirteenShortPctR",
    "PThreeFeasBandMarginUncPct":     "BsNineFeasBandMarginUncPct",
    "PThreeETwoRestrictedMinLoPct":   "BsNineETwoRestrictedMinLoPct",
    "PThreeETwoRestrictedMinHiPct":   "BsNineETwoRestrictedMinHiPct",
    "PThreeETwoRestrictedWgLoPct":    "BsNineETwoRestrictedWgLoPct",
    "PThreeETwoRestrictedWgHiPct":    "BsNineETwoRestrictedWgHiPct",
    # R-62.4: the film-supported operating space, aliased for section 5 (no prose yet; the
    # collapse test must precede any (S, theta_min) map).
    **{f"PThreeOps{s}": f"BsNineOps{s}" for s in
       [f"SFifty{t}{b}{c}" for t in ("Film", "Full") for b in ("One", "Two", "Three") for c in ("", "CiLo", "CiHi")]
       + [f"Count{n}{e}" for n in ("Full", "Mixed", "Contact", "None") for e in ("Lo", "Hi")]
       + ["SurviveMinPct", "SurviveWgPct", "CornerPoints", "CornerAboveFull", "CornerAboveFilm"]},
}


#: R-68.2: the unsuffixed thermal aliases are withdrawn from Paper 3; the mineral-class values carry the
#: suffix Min (v5 shows the unsuffixed names as owed until chat rewrites those sentences).
for _k in ('ThetaFlowMed', 'ThetaFlowAbovePct', 'BsBrink', 'BsQmin', 'NahmeAdiabLoPct', 'NahmeAdiabHiPct', 'OpsSurviveMinPct', 'ETwoCeilingReachPct', 'ETwoGapPp', 'ThermClearNm'):
    PAPER3_ALIASES.pop(f"PThree{_k}", None)


#: R-59.3: memory-sign provenance of every Paper 3 macro.  The campaign macros (boxes D/E,
#: E1-E14 rows, E13/E5 production values) are MODULUS-derived and carry the D-6 row-wise
#: bound; the claim rows printed at their Richardson values are SIGNED (R-58.2).  The mix is
#: legitimate only if visible, so every alias is classified here and the class is written
#: into its provenance comment.  A new alias that is not classified fails check_paper3.
MS_SIGNED = "signed"
MS_MODULUS = "modulus + D-6 bound"
MS_HELD = "modulus, HELD (R-59.1 D2 STOP)"
MS_NO_SOLVE = "none (input or count, no film solve)"
MS_NEWTONIAN = "none (Newtonian quadrature or Newtonian solver; no memory)"
MS_CITED = "none (CITED value; the citation key is in the provenance)"
D6_BOUND_REF = ("R58_E4_signed.report.json", "dLam_D6_bound")
PAPER3_SIGNED = {"PThreeEFiveLamMaxCorner", "PThreeEThirteenLamMax", "PThreeEThirteenShortPct",
                 "PThreeFeasBandMarginPct", "PThreeFeasBandMarginUncPct",
                 # R-63.1: E4 released on the signed D2/D1 measurements
                 "PThreeEFourSweepMax", "PThreeEFourSweepMaxDTwo", "PThreeEFourSweepMedian",
                 "PThreeMemShareMin", "PThreeMemShareMax", "PThreeEFourSweepRaisesN", "PThreeEFourEvenPct",
                 # R-63.3: E15 (box G, signed grid and Richardson)
                 "PThreeEFifteenCZeroStarUm", "PThreeEFifteenLamStarR", "PThreeEFifteenLamStarUnc",
                 "PThreeEFifteenLamTwoUm", "PThreeEFifteenLamAtStar", "PThreeEFifteenGainPct",
                 "PThreeEFifteenLamStarHi", "PThreeEFifteenSigmaScalePct",
                 "PThreeRichOrderMin",
                 # R-65.3 friction (signed solves) -- R-66's tag fix
                 "PThreeFricN", "PThreeFricOrderMin", "PThreeFricCorrMaxPct", "PThreeFricTorqueMedNm",
                 "PThreeFricTorqueMinNm", "PThreeFricTorqueMaxNm", "PThreeFricMuMed", "PThreeFricMuMin",
                 "PThreeFricMuMax", "PThreeFricDeltaNonconv",
                 # R-65.4 / R-66.4: the Fig. 5 point (signed 32x24 solve)
                 "PThreeFieldsDelta", "PThreeFieldsPeakThetaDeg", "PThreeFieldsPeakMPa",
                 "PThreeFieldsEtaRatioMin", "PThreeFieldsEtaRatioMax", "PThreeFieldsRiseMaxmK",
                 # R-72: De_sq at the claim points (box G argmax row; R58_richardson, signed 32x24)
                 "PThreeEFifteenDesq", "PThreeEThirteenDesq", "PThreeEFiveDesq", "PThreeEFifteenDesqPiezo"}
#: R-63.1: E4 released on the signed path; nothing is held.
PAPER3_HELD: set = set()
PAPER3_NO_SOLVE = {"PThreeRbMm", "PThreeThetaMaxDeg", "PThreeThetaMinLoDeg", "PThreeThetaMinHiDeg",
                   "PThreeCZeroLoUm", "PThreeCZeroHiUm", "PThreeSigmaStarLoUm", "PThreeSigmaStarHiUm",
                   "PThreeContactArms", "PThreeWgCeiling", "PThreeMineralCeiling",
                   "PThreeInertiaLoPct", "PThreeInertiaHiPct", "PThreeInertiaMidPct",
                   "PThreeEpsCeilTwenty", "PThreeEpsCeilThirty", "PThreeEpsCeilForty",
                   "PThreeSamples", "PThreeNAudit", "PThreeEFourSweepN", "PThreeEFourSweepBase",
                   "PThreeEFourSweepInDomain", "PThreeWiZero", "PThreeRootTol",
                   "PThreeFricSample", "PThreeFricDeltaSplit", "PThreeFieldsSolventFrac",
                   "PThreeBrinkThresh", "PThreeBrinkDeltaT", "PThreeThetaDeltaTRef", "PThreeDesqMax",
                   # R-72: registered numerals, claim-point inputs, counts of rows and models
                   *(f"PThree{s}" for s in "SatDeltaHi SatDeltaLo VerifDeltaHi FilmAreaClosedDelta LamAxisLo LamAxisHi "
                     "LamSplit AdiabFactor PCLamLo PCLamHi EFifteenCZeroUm EThirteenCZeroUm EFiveCZeroUm "
                     "EFifteenThetaMinDeg EThirteenThetaMinDeg EFiveThetaMinDeg EFifteenEtaZeroMPas EFifteenLamS "
                     "EFifteenTinC EFifteenPchMPa EFifteenOmega EFifteenSolventFrac ValidRows RatedPoints RatedRows "
                     "ContactModels StrokeGridFloor StrokeGMax".split())}
#: R-63.6: closed-form / Newtonian-solver values (no memory enters)
PAPER3_CITED = {"PThreeCornerPBar", "PThreeCornerRpm", "PThreeLeeHydWithPct", "PThreeLeeHydWithoutPct",
                "PThreeLeeRbMm", "PThreeLeeLoadKN", "PThreeLeeSigmaUm", "PThreeLeeTempC",
                "PThreeLeeCUm", "PThreeLeeEtaMPas", "PThreeLeeAlphaStar", "PThreeLeePiezoHundred",
                "PThreeLeePiezoPeak",
                "PThreeWangTempC", "PThreeWangPMaxMPa", "PThreePiezoAlphaWgDig", "PThreePiezoGWgDig",
                "PThreePiezoAlphaTenDig", "PThreePiezoAlphaTwentyTwoDig", "PThreeHasanTempC", "PThreeHasanPMaxGPa",
                "PThreePiezoAlphaWgLo", "PThreePiezoAlphaWgHi", "PThreePiezoGWg", "PThreePiezoGWgHi",
                "PThreeSecondOrderPlanar"}
PAPER3_NEWTONIAN = {*(f"PThree{s}" for s in "PiezoGLee PiezoGLeeBarus PiezoGcritEFifteen PiezoGcritEThirteen PiezoGcritEFive PiezoFactorEFifteen PiezoFactorEThirteen PiezoFactorEFive PiezoLamEFifteen PiezoLamEThirteen PiezoLamEFive PiezoKappa PiezoKappaFit PiezoLinDevPct PiezoQPeakFrac PiezoCZeroRatio PiezoLamStar PiezoPeakOverPch".split()),
                    "PThreeFieldsPeakThetaNewtDeg", "PThreeFieldsPeakNewtMPa", "PThreeThermClearNmMin", "PThreeSatFloorDelta", "PThreeSatFloorLoadErrPct", "PThreeSatNewtonAgree", "PThreeRimRatio",
                    "PThreeRimRatioArea", "PThreeSqAreaRatioLo", "PThreeSqAreaRatioHi",
                    # R-72.6: E17, Newtonian at eta_s (stroke-integrated squeeze)
                    *(f"PThree{s}" for s in "StrokeEFifteenSnapDevPct StrokeEThirteenSnapDevPct StrokeEFiveSnapDevPct StrokeEFifteenTSnapDeg StrokeEThirteenTSnapDeg StrokeEFiveTSnapDeg StrokeEFifteenTcDeg StrokeEThirteenTcDeg StrokeEFiveTcDeg StrokeEFifteenTLamThreeRoelDeg StrokeEThirteenTLamThreeRoelDeg StrokeEFiveTLamThreeRoelDeg StrokeEFifteenLamPiRoel StrokeEThirteenLamPiRoel StrokeEFiveLamPiRoel StrokeEFifteenLamPiRoelSnap StrokeEThirteenLamPiRoelSnap StrokeEFiveLamPiRoelSnap StrokeEFifteenLamPiGTwo StrokeEThirteenLamPiGTwo StrokeEFiveLamPiGTwo StrokeEFifteenTHundredthDeg StrokeEFifteenLamPiIso StrokeEFifteenLamPiBarus StrokeEFifteenTLamThreeBarusDeg StrokeEFifteenRDeltaBarus StrokeEFifteenRDeltaBarusFloor StrokeTailPower".split())}
#: the h_min / Lambda CLAIM macros (R-59.3: each must be signed or name the D-6 bound)
PAPER3_LAMBDA_CLAIMS = {"PThreeLamMedConvLo", "PThreeLamMedConvHi", "PThreeESevenBLamMax",
                        "PThreeFeasBandLbMin", "PThreeFeasBandLbMax", "PThreeFeasBandMarginPct",
                        "PThreeFeasBandMarginUncPct", "PThreeFeasBandMarginMeshPct",
                        "PThreeEFiveLamMaxCorner", "PThreeEFiveLamMaxCornerProd",
                        "PThreeEThirteenLamMax", "PThreeEThirteenLamMaxProd",
                        "PThreeEThirteenShortPct"}


def paper3_memory_sign(p3: str) -> str:
    if p3 in PAPER3_SIGNED:
        return MS_SIGNED
    if p3 in PAPER3_HELD:
        return MS_HELD
    if p3 in PAPER3_NO_SOLVE:
        return MS_NO_SOLVE
    if p3 in PAPER3_NEWTONIAN:
        return MS_NEWTONIAN
    if p3 in PAPER3_CITED:
        return MS_CITED
    return MS_MODULUS


def check_memory_sign_provenance() -> list[str]:
    """R-59.3 generator assertion.  Hard failures, empty if clean."""
    hard = []
    sets = [PAPER3_SIGNED, PAPER3_HELD, PAPER3_NO_SOLVE, PAPER3_NEWTONIAN, PAPER3_CITED]
    for s in sets + [PAPER3_LAMBDA_CLAIMS]:
        for k in s - set(PAPER3_ALIASES):
            hard.append(f"MEMORY SIGN: {k} is classified but is not a Paper 3 alias")
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            for k in a & b:
                hard.append(f"MEMORY SIGN: {k} is in two classes")
    bound = OUT / D6_BOUND_REF[0]
    bound_ok = bound.exists() and D6_BOUND_REF[1] in json.loads(bound.read_text())
    for k in sorted(PAPER3_LAMBDA_CLAIMS & set(PAPER3_ALIASES)):
        ms = paper3_memory_sign(k)
        if ms not in (MS_SIGNED, MS_MODULUS):
            hard.append(f"MEMORY SIGN: h_min/Lambda claim {k} is '{ms}', neither signed nor bounded")
        elif ms == MS_MODULUS and not bound_ok:
            hard.append(f"MEMORY SIGN: {k} is modulus-derived but the D-6 bound "
                        f"{D6_BOUND_REF[0]}:{D6_BOUND_REF[1]} does not resolve")
    return hard


def check_paper3(derived: dict) -> tuple[list[str], list[tuple]]:
    """Every Paper 3 alias must name a DECLARED, derived Chapter 9 macro.

    HARD-fails otherwise.  This is what makes the alias guarantee real: a
    Paper 3 macro pointing at a key that does not exist, or at one Chapter 9
    does not itself check, would be a transcribed number wearing an alias's
    clothes.
    """
    hard: list[str] = []
    rows: list[tuple[str, str, str]] = []
    for p3, ch9 in sorted(PAPER3_ALIASES.items()):
        if ch9 not in DECLARED:
            hard.append(f"PAPER3 {p3}: aliases {ch9}, which is NOT in DECLARED. "
                        f"A Paper 3 macro may only alias a Chapter 9 macro that "
                        f"Chapter 9 itself checks.")
            rows.append((p3, ch9, "NOT DECLARED"))
            continue
        if ch9 not in derived:
            hard.append(f"PAPER3 {p3}: aliases {ch9}, which has NO DERIVATION.")
            rows.append((p3, ch9, "NO DERIVATION"))
            continue
        rows.append((p3, ch9, "ok"))
    hard += check_memory_sign_provenance()
    return hard, rows


def emit_paper3(derived: dict) -> None:
    """Write the Paper 3 macro file from the SAME derivations as Chapter 9."""
    lines = [
        "% paper3_numbers_from_ch9.tex -- GENERATED by scripts/ch9_numbers.py.",
        "% Do not hand-edit.",
        "%",
        "% Paper 3 is bound in the SAME thesis as Chapter 9, so a transcribed",
        "% number is a contradiction waiting where an examiner is most likely",
        "% to check.  Every macro below is an ALIAS of a Chapter 9 macro,",
        "% emitted from the SAME derivation in the SAME pass.  The two",
        "% documents cannot disagree because there is only one number.",
        "%",
        "% This SUPERSEDES giesekus-solver/scripts/paper3_numbers.py as the",
        "% number source for the RE-SCOPED Paper 3.  That generator's canonical",
        "% artefact is the AR exhaustive parquet, which box D does not touch;",
        "% it reads no ball-socket artefact and would notice no correction.",
        "% It is retained, unchanged and superseded, for the AR-based numbers.",
        "%",
        "% Load guard: same convention and same reasoning as chapter9_numbers.tex",
        "% -- a file-level flag, not \\providecommand, so a genuine cross-file",
        "% collision fails the build loudly instead of silently keeping a stale",
        "% first definition.",
        "\\ifdefined\\pThreeNumbersLoaded\\endinput\\fi",
        "\\let\\pThreeNumbersLoaded\\relax",
        "",
    ]
    for p3, ch9 in sorted(PAPER3_ALIASES.items()):
        want, tol = DECLARED[ch9]
        got, src = derived[ch9]
        shown = DISPLAY.get(ch9, f"{want}")
        lines.append(f"\\newcommand{{\\{p3}}}{{{shown}}}  "
                     f"% ALIAS of \\{ch9}; derived {got:.6g} <- {src}; "
                     f"memory_sign: {paper3_memory_sign(p3)}"
                     + (f" ({D6_BOUND_REF[0]}:{D6_BOUND_REF[1]})"
                        if paper3_memory_sign(p3) == MS_MODULUS else ""))
    payload = "\n".join(lines) + "\n"
    PAPER3_TEX_OUT.write_text(payload, encoding="utf-8")
    print(f"wrote {PAPER3_TEX_OUT}")
    # Write the LIVE copy directly, which is what removes the manual hop for
    # Paper 3 (R-19.5).  Chapter 9's file keeps its staging hop because the
    # thesis directory is under version control on the share and a numbers run
    # must not mutate it silently; Paper 3's folder has no such constraint and
    # the destination was authorised explicitly.
    try:
        PAPER3_DIR_LIVE.mkdir(parents=True, exist_ok=True)
        PAPER3_TEX_LIVE.write_text(payload, encoding="utf-8")
        print(f"wrote {PAPER3_TEX_LIVE}   [LIVE -- no manual hop]")
    except OSError as exc:
        print(f"LIVE COPY NOT WRITTEN ({PAPER3_TEX_LIVE}): {exc}.  "
              f"The share is unreachable; copy the staging file across before "
              f"compiling Paper 3.")


# ---------------------------------------------------------------------------
# THE STAGING HOP  (ruling R-19.5)
#
# This generator writes to a STAGING path inside the repository.  The live
# thesis consumes the file from the network share:
#
#     writes  ->  results/ch9/chapter9_numbers.tex          (here)
#     reads   <-  Z:/PhD Thesis/chapter9_numbers.tex        (live thesis)
#
# Between them is a HUMAN COPY.  Abort-on-mismatch guarantees that the staging
# file agrees with the artefacts; it guarantees nothing about the file the
# thesis actually compiles.  The alias guarantee for Paper 3 inherits the same
# hop.  A guarantee that stops at a manual step is a guarantee with a hole in
# it, and this is where guarantees usually stop being structural.
#
# WHY A CHECK AND NOT A WRITE.  Writing directly into the live thesis from a
# numbers generator would make every run mutate a document under version
# control on a share, including exploratory runs.  The conservative closure is
# to DETECT divergence and refuse, which makes the copy step mandatory instead
# of optional without giving this script write access to the thesis.
#
# WHEN THE SHARE IS UNREACHABLE the check reports and does NOT fail: a check
# cannot assert anything about a file it cannot see, and pretending otherwise
# would be the same error this project keeps cataloguing.
# ---------------------------------------------------------------------------

LIVE_THESIS_TEX = pathlib.Path("Z:/PhD Thesis/chapter9_numbers.tex")


def check_staging_hop() -> tuple[list[str], dict]:
    """Compare the staging macro file against the live thesis copy.

    Returns (hard_failures, verify_slot).

    THE VERIFY SLOT, AND WHY A STATUS LINE WAS NOT ENOUGH (ruling R-20.2).
    The share-unreachable path and the verified-identical path both produce NO
    HARD FAILURE, so in the result they are indistinguishable: a run performed
    off-network passes exactly as cleanly as a run that checked and found the
    live copy correct.  Nothing recorded which of the two a given run was, so a
    later claim that Chapter 9's numbers are current would rest on an
    ASSUMPTION ABOUT WHERE THE LAST RUN HAPPENED rather than on a verification.

    The fix is box D's own construction.  ``blocking_unverified()`` does not
    fail the run; it records an unverified state that blocks PUBLICATION, and
    prereg v6 sec 1 states the one remaining slot in terms "so no reader infers
    from a clean freeze that every slot is filled".  Same pattern, applied to
    the last manual step in the chain: the outcome is PERSISTED, so
    unverifiability is visible instead of silent.

    States:
      VERIFIED_IDENTICAL          checked, live copy matches -- publishable
      STALE                       checked, live copy differs -- HARD failure
      LIVE_ABSENT                 checked, no live copy       -- HARD failure
      NOT_CHECKED_SHARE_ABSENT    could not check             -- UNVERIFIED
    """
    import datetime
    import hashlib

    slot = {
        "check": "staging_hop",
        "ruling": "R-19.5, VERIFY slot per R-20.2",
        "staging_path": str(TEX_OUT),
        "live_path": str(LIVE_THESIS_TEX),
        "checked_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "state": None,
        "verified": False,
        "blocks_publication": False,
        "staging_sha256": None,
        "live_sha256": None,
    }

    if not TEX_OUT.exists():
        slot["state"] = "NOT_CHECKED_SHARE_ABSENT"
        slot["detail"] = "staging file not yet written"
        slot["blocks_publication"] = True
        return [], slot

    stage_bytes = TEX_OUT.read_bytes()
    slot["staging_sha256"] = hashlib.sha256(stage_bytes).hexdigest()

    # REACHABILITY IS TESTED SEPARATELY FROM EXISTENCE, and it has to be.
    # On Windows an unreachable UNC path or an unmapped drive returns False
    # from .exists() rather than raising, so "the share is down" and "the file
    # was deleted from a share that is up" collapse into one answer.  They are
    # opposite situations: the first is UNVERIFIABLE and must not fail the run,
    # the second is a real defect and must.  Probing the CONTAINING DIRECTORY
    # separates them.
    try:
        share_up = LIVE_THESIS_TEX.parent.is_dir()
    except OSError:
        share_up = False

    if not share_up:
        slot["state"] = "NOT_CHECKED_SHARE_ABSENT"
        slot["detail"] = (f"share unreachable ({LIVE_THESIS_TEX.parent}); "
                          f"the live copy was NOT checked")
        slot["blocks_publication"] = True
        return [], slot          # UNVERIFIED, not failed

    try:
        live_exists = LIVE_THESIS_TEX.exists()
    except OSError as exc:
        slot["state"] = "NOT_CHECKED_SHARE_ABSENT"
        slot["detail"] = f"live copy unreadable: {exc}"
        slot["blocks_publication"] = True
        return [], slot

    if not live_exists:
        slot["state"] = "LIVE_ABSENT"
        slot["detail"] = ("share is reachable but there is no live copy at the "
                          "recorded path")
        slot["blocks_publication"] = True
        return ([f"STAGING HOP: share is up but no live copy at "
                 f"{LIVE_THESIS_TEX}.  Copy {TEX_OUT} there before compiling "
                 f"the thesis."], slot)

    live_bytes = LIVE_THESIS_TEX.read_bytes()
    slot["live_sha256"] = hashlib.sha256(live_bytes).hexdigest()

    if live_bytes == stage_bytes:
        slot["state"] = "VERIFIED_IDENTICAL"
        slot["verified"] = True
        slot["detail"] = "live thesis copy verified identical to staging"
        return [], slot

    slot["state"] = "STALE"
    slot["blocks_publication"] = True
    slot["detail"] = "live thesis copy differs from the staging file"
    return ([f"STAGING HOP: {LIVE_THESIS_TEX} DIFFERS from the staging file "
             f"just written.  The thesis is compiling stale macros.  Copy "
             f"{TEX_OUT} -> {LIVE_THESIS_TEX} and re-run."], slot)


#: R-66.3: the generated macro files are scanned like committed files.  They change every run,
#: so a hit is classified by the MACRO its line defines (not by line number or file hash), with
#: the reason recorded here.  Any other hit is HARD.
GENERATED_SCAN_ALLOWLIST = {
    "BsNineOpsSFiftyFullThreeCiHi": "95 % bootstrap bound of a duty-number S50 (2.197 at its declared three "
                                    "figures), an unrelated quantity; kept by R-66.3",
    "PThreeOpsSFiftyFullThreeCiHi": "alias of the same bound; kept by R-66.3",
}


def check_generated_scan(paths) -> list[str]:
    import re
    import barred_token_scan as B
    hits = B.scan([str(p) for p in paths if pathlib.Path(p).exists()], deep=True)
    bad = []
    for f, n, k, txt in hits:
        m = re.match(r"\\newcommand\{\\(\w+)\}", txt.strip())
        if not (m and m.group(1) in GENERATED_SCAN_ALLOWLIST):
            bad.append(f"BARRED SCAN: {pathlib.Path(f).name}:{n}: [{k}] unclassified in a generated macro file")
    return bad


def staging_hop_unverified(slot: dict) -> list[str]:
    """The publication guard, mirroring ``blocking_unverified()``.

    Does NOT fail the run.  Returns a non-empty list when the hop's state is
    anything other than verified, so that a downstream publication step can
    refuse on a RECORDED unverified state rather than on an assumption.
    """
    if slot.get("verified"):
        return []
    return [f"staging_hop: {slot.get('state')} -- {slot.get('detail')}"]



#: TeX control sequences are letters only.  `\PThree...SoleG1` compiled to
#: "Missing egin{document}" in the DRAFT, not in the generator, which is the
#: worst place to find it (R-54 execution, 2026-09-23).
_TEX_NAME = re.compile(r"^[A-Za-z]+$")


def tex_name_gate(names) -> list[str]:
    """Names that would not survive as a control sequence.

    Keys beginning with '_' are the generator's own intermediates (they feed
    comparatives and are never emitted), so they are out of scope.
    """
    return sorted(n for n in names
                  if not n.startswith("_") and not _TEX_NAME.match(n))


def main() -> int:
    a = _load()
    derived = derive(a)
    bad_names = tex_name_gate(list(derived) + list(PAPER3_ALIASES))
    if bad_names:
        BOXF_GATE_FAILS.append(
            f"macro names are not TeX-safe (letters only): {', '.join(bad_names)}")
    # R-54.1: the h_min / Lambda value gate.  Runs after derive() so it can name
    # the macros at risk, and before any macro is printed.
    hmin_gate = hmin_closed_form_gate(derived)

    hard: list[str] = []
    rows: list[tuple[str, str, str, str]] = []

    for macro, (want, tol) in sorted(DECLARED.items()):
        if macro not in derived:
            hard.append(f"{macro}: NO DERIVATION AVAILABLE")
            rows.append((macro, f"{want}", "-", "NO DERIVATION"))
            continue
        got, src = derived[macro]
        if want is TBV:
            hard.append(f"{macro}: declared TBV, not yet fixed in the prose.  "
                        f"Derived {got:.6g}  [{src}]")
            rows.append((macro, "TBV", f"{got:.6g}", "TBV -> fill in"))
            continue
        ok = abs(float(got) - float(want)) <= tol
        rows.append((macro, f"{want}", f"{got:.6g}", "ok" if ok else "MISMATCH"))
        if not ok:
            hard.append(f"{macro}: declared {want}, derived {got:.6g} "
                        f"(tol {tol})  [{src}]")

    for macro, why in BLOCKED.items():
        hard.append(f"{macro}: BLOCKED. {why}")
        rows.append((macro, "declared in prose", "-", "BLOCKED"))

    width = max(len(r[0]) for r in rows) + 2
    for macro, want, got, status in rows:
        flag = "" if status == "ok" else "   <<<"
        print(f"{macro:<{width}} declared {want:>12}  derived {got:>12}  "
              f"{status}{flag}")

    p3_fail, p3_rows = check_paper3(derived)
    print("\nPAPER 3 ALIASES  (R-18.2 -- one generator, one artefact lineage)")
    for p3, ch9, st in p3_rows:
        flag = "" if st == "ok" else "   <<<"
        print(f"  {p3:<26} -> {ch9:<26} {st}{flag}")

    claim_fail, claim_rows = check_claims(a)
    print("\nCLAIMS REGISTER  (R-18.3 -- macros catch numerals, this catches sentences)")
    for cid, st, note in claim_rows:
        flag = "   <<<" if st == "PENDING_BOXD" else ""
        print(f"  {st:<13} {cid}{flag}")
        print(f"                {note[:96]}")

    comp_fail, comp_rows = check_comparatives(derived)
    print("\nCOMPARATIVE AUDIT")
    for tag, relation, status in comp_rows:
        flag = "" if status == "holds" else "   <<<"
        print(f"  {tag:<4} {status:<6} {relation}{flag}")
    hard.extend(comp_fail)
    hard.extend(claim_fail)
    hard.extend(check_prohibitions(a, derived))
    hard.extend(f"BOX F: {m}" for m in BOXF_GATE_FAILS)
    hard.extend(p3_fail)

    worst = max((c["max_rel_dev"] for c in hmin_gate["artefacts_checked"]), default=0.0)
    print(f"\nh_min VALUE GATE (R-54.1): {len(hmin_gate['artefacts_checked'])} artefacts "
          f"checked by VALUE against c(1 - eps cos theta_min) at "
          f"{hmin_gate['tolerance']:.0e}, covering "
          f"{len(hmin_gate['macros_covered'])} h_min / Lambda macros; "
          f"worst relative deviation {worst:.2e}")
    if hmin_gate["interior_only_artefacts"]:
        print(f"  interior-only artefacts (no corrected column; a Lambda macro built "
              f"on one would repeat ERR-9): "
              f"{', '.join(hmin_gate['interior_only_artefacts'])}")

    print(f"\n{len(rows)} macros checked, {len(comp_rows)} comparatives "
          f"checked, {len(claim_rows)} claims checked, "
          f"{len(p3_rows)} Paper 3 aliases checked, "
          f"{len(hard)} HARD failure(s).")
    if hard:
        print("\nHARD failures:")
        for h in hard:
            print(f"  * {h}")
        print(f"\nNOT writing {TEX_OUT}.  Abort-on-mismatch is working as "
              f"specified: a chapter cannot be built from unverified numbers.")
        return 1

    # Load guard, copied in form and in reasoning from ch78_numbers.tex.
    # Chapters 9 and 10 both consume these macros and a chapter must be able to
    # \input them without knowing whether another already has; a bare
    # \newcommand raises "already defined" on the second load and aborts the
    # build.  It is a FILE-level flag rather than \providecommand on each macro
    # for the reason ch78_numbers.tex gives: \providecommand would silently keep
    # the first definition, so a genuine collision between two DIFFERENT
    # generated files would become a silent stale-value acceptance instead of a
    # loud abort.
    lines = ["% chapter9_numbers.tex -- GENERATED by scripts/ch9_numbers.py.",
             "% Do not hand-edit.  Every value is derived from a persisted "
             "artefact and",
             "% checked against the prose value block, and every comparative "
             "asserted in",
             "% the prose is checked as a RELATION, before this file is "
             "written.",
             "%",
             "% Load guard: more than one chapter consumes these macros.  "
             "File-level flag,",
             "% not \\providecommand, so that a real cross-file name collision "
             "still fails",
             "% the build loudly instead of silently keeping a stale first "
             "definition.",
             "% Same convention as ch78_numbers.tex.",
             "\\ifdefined\\chNineNumbersLoaded\\endinput\\fi",
             "\\let\\chNineNumbersLoaded\\relax",
             ""]
    # Emit the DECLARED value, not the derived one.  The declared value is the
    # author's chosen presentation precision; the check above is what licenses
    # it, and the tolerance is the size of rounding that was licensed.  Writing
    # the raw derived value instead would put 19.458 and 0.0935282 into the
    # chapter, which is not what the prose block asked for.  The derivation
    # travels in the comment so the provenance is never lost.
    for macro, (want, tol) in sorted(DECLARED.items()):
        got, src = derived[macro]
        shown = DISPLAY.get(macro, f"{want}")
        note = "" if macro not in DISPLAY else f" [typeset as {shown}]"
        lines.append(f"\\newcommand{{\\{macro}}}{{{shown}}}  "
                     f"% declared {want}, derived {got:.6g} (tol {tol:g}) "
                     f"<- {src}{note}")
    TEX_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TEX_OUT}")
    emit_paper3(derived)

    gen_scan = check_generated_scan([TEX_OUT, PAPER3_TEX_OUT, PAPER3_TEX_LIVE])
    print(f"\nBARRED SCAN of the generated macro files (R-66.3): "
          f"{'clean (every hit classified)' if not gen_scan else f'{len(gen_scan)} UNCLASSIFIED'}")
    hop_fail, hop_slot = check_staging_hop()
    hop_fail = gen_scan + hop_fail
    unver = staging_hop_unverified(hop_slot)
    print(f"\nSTAGING HOP (R-19.5): {hop_slot['state']} -- {hop_slot['detail']}")
    if unver:
        # Not a failure.  An UNVERIFIED state, recorded, that blocks
        # publication -- exactly what blocking_unverified() does for a
        # citation slot that is still VERIFY (prereg v6 sec 1).
        print(f"  UNVERIFIED (blocks publication, not this run): {unver[0]}")
    manifest_path = OUT / "ch9_numbers_run_manifest.json"
    manifest_path.write_text(
        json.dumps({"staging_hop": hop_slot,
                    "macros_checked": len(rows),
                    "comparatives_checked": len(comp_rows),
                    "claims_checked": len(claim_rows),
                    "paper3_aliases_checked": len(p3_rows),
                    "hard_failures": len(hard)},
                   indent=2), encoding="utf-8")
    print(f"  verification recorded in {manifest_path}")
    if hop_fail:
        for h in hop_fail:
            print(f"  HARD: {h}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
