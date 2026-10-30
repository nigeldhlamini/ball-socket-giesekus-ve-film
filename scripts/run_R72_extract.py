"""R-72.2 and R-72.4: extractions for the v7 draft.  No solves; every value is read from a stored
artefact or a registered module and written to results/ch9/R72_extract.report.json with its source.

R-72.2  input vectors at the claim points (E15 = box G argmax row; E13, E5 = run_R56_tolerance.cases()
        with De_sq_bnd from R58_richardson.parquet at 32x24), the 40-point rated-corner set
        (E5_VG32_g1_binding_corner.parquet), and De_sq x Lee's Roelands factor at the Fig. 5 peak (E15)
        as a bound under time-pressure superposition.
R-72.4  4.1 film-alone points at both ends, with the class-3 subset assertion (STOP if the smooth-end
            rescues are not a subset of the rough-end ones); 4.2 the valid-row criterion and the
            composite's further exclusions; 4.3 Table 2 denominators; 4.5 the survivor pair Fig. 6b plots;
            4.4 typed numerals with file:line.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
REPORT = OUT / "R72_extract.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]


def line_of(path, pattern):
    for i, ln in enumerate((ROOT / path).read_text(encoding="utf-8").splitlines(), 1):
        if re.search(pattern, ln):
            return f"{path}:{i}"
    raise AssertionError((path, pattern))


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    from ball_socket.campaign_config_v4 import ConstraintSet
    import run_R56_tolerance as T
    import p3_lee_fluid as LF
    prov = require_clean_tree("R-72.2/R-72.4 extraction")
    cs = ConstraintSet()
    rep = {"provenance": prov}
    # ---- R-72.2 claim points
    B = pd.read_parquet(OUT / "boxG_E15.parquet")
    g15 = json.loads((OUT / "boxG_E15.report.json").read_text(encoding="utf-8"))["argmax"]
    r = B[(B.model_id == g15["arm"]) & np.isclose(B.c0 * 1e6, g15["c0_um"]) & (B.memory_sign == "signed")].iloc[0]
    pts = {"E15": dict(x={k: float(r[k]) for k in AXES}, De_sq=float(r.De_sq_bnd), sigma=float(r.sigma_star),
                       source="boxG_E15.parquet, argmax row (arm %d, c0 %.2f um), signed 32x24" % (g15["arm"], g15["c0_um"]))}
    R58 = pd.read_parquet(OUT / "R58_richardson.parquet")
    for c in T.cases():
        if c["case"] in ("E13", "E5"):
            s = R58[(R58.case == c["case"]) & (R58.Ntheta == 32)].iloc[0]
            pts[c["case"]] = dict(x={k: float(v) for k, v in c["x"].items()}, De_sq=float(s.De_sq_bnd),
                                  sigma=float(c["sigma"]), source=f"run_R56_tolerance.cases() {c['case']}; "
                                  "De_sq_bnd from R58_richardson.parquet (32x24, signed, tight)")
    for p in pts.values():
        p["sigma_end"] = "lo" if np.isclose(p["sigma"], min(q["sigma"] for q in pts.values())) else "hi"
    rep["claim_points"] = pts
    Fd = pd.read_parquet(OUT / "R65_fields.parquet")
    g_pk = float(LF.roelands_ratio(373.15, Fd.p_Pa.max()))
    rep["E15_desq_piezo"] = {"De_sq": pts["E15"]["De_sq"], "roelands_factor_at_fig5_peak": g_pk,
                             "De_sq_piezo": pts["E15"]["De_sq"] * g_pk,
                             "note": "lambda(p) = lambda(0) eta(p)/eta(0) under time-pressure superposition; "
                                     "the peak-pressure factor bounds De_sq from above"}
    C = pd.read_parquet(OUT / "E5_VG32_g1_binding_corner.parquet")
    pts40 = C.drop_duplicates("point")
    var, fix = {}, {}
    for k in AXES + ["nu_100", "sigma_star"]:
        u = sorted(set(map(float, C[k].unique())))
        (var if len(u) > 1 else fix)[k] = u if len(u) <= 12 else [u[0], u[-1]]
    rep["rated_corner_set"] = {"points": int(len(pts40)), "rows": int(len(C)), "arms_per_point": int(C.model_id.nunique()),
                               "grade": sorted(map(int, C.grade.unique())), "corner": sorted(C.corner.unique()),
                               "varies": var, "fixed": fix,
                               "definition": "VG32, nu_100 swept (eta_0 from ASTM D341 at T_in) x c0; every other input "
                                             "fixed at the rated maximum corner, theta_min = 30 deg; each point on the 24 "
                                             "contact arms (both sigma* ends)"}
    # ---- R-72.4.1
    O = pd.read_parquet(OUT / "R65_ops_solv.parquet")
    lo, hi = int(O.cls_lo.isin([1, 2]).sum()), int(O.cls_hi.isin([1, 2]).sum())
    D = pd.read_parquet(OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet", columns=["outcome"])
    nconv = int((D.outcome == "converged").sum())
    c3lo, c3hi = set(O.sample_id[O.cls_lo == 3]), set(O.sample_id[O.cls_hi == 3])
    assert lo == hi == nconv, (lo, hi, nconv)
    assert c3lo <= c3hi, "STOP R-72.4.1: the smooth-end contact rescues are not a subset of the rough-end ones"
    rep["film_only_both"] = {"n": lo, "boxD_converged": nconv, "class3_lo": len(c3lo), "class3_hi": len(c3hi),
                             "lo_subset_of_hi": True}
    # ---- R-72.4.2
    src = (ROOT / "scripts" / "ch9_numbers.py").read_text(encoding="utf-8")
    bm = re.search(r'BOXE_MIXED\s*=\s*"([^"]+)"', src).group(1)
    bk = re.search(r'BOXE_KIND\s*=\s*"([^"]+)"', src).group(1)
    M = pd.read_parquet(OUT / bm, columns=["sample_id", "model_id", "outcome"])
    K = pd.read_parquet(OUT / bk)[["sample_id", "model_id", "nonconvergence_kind"]]
    M = M.merge(K, on=["sample_id", "model_id"], how="left")
    exc = M[(M.nonconvergence_kind == "picard_at_root") | M.outcome.isin(["clearance_nonconverged", "error"])]
    F = pd.read_parquet(OUT / "boxF_band_full_resolve.parquet")
    Fm = F[(F.treatment == "extrapolate") & (F.arm_of_campaign == "mixed")]
    gone = Fm[Fm.outcome == "clearance_nonconverged"][["sample_id", "model_id"]].drop_duplicates()
    rep["valid"] = {"n_rows": int(len(M)), "n_valid": int(len(M) - len(exc)), "excluded": int(len(exc)),
                    "excluded_kinds": {f"{a}/{b}": int(n) for (a, b), n in
                                       exc.groupby(["outcome", "nonconvergence_kind"], dropna=False).size().items()},
                    "criterion": ("valid unless the evaluation is not a determination: the equilibrium was bracketed "
                                  "but the viscoelastic Picard iteration did not converge at the root "
                                  "(nonconvergence_kind picard_at_root, ball_socket/solver.py raise at 'Picard "
                                  "non-convergence at eps'), or the thermal-clearance fixed point did not converge, "
                                  "or an infrastructure error (ch9_numbers.derive_boxF, the R-37.4 partition)"),
                    "composite_further_exclusions": int(len(gone)),
                    "composite_criterion": "box F mixed re-solves (averaged film) whose thermal-clearance fixed point "
                                           "did not converge (outcome clearance_nonconverged)"}
    # ---- R-72.4.3: Table 2 denominators
    tb = json.loads((OUT / "R69_thermal_bracket.report.json").read_text(encoding="utf-8"))
    E = pd.read_parquet(OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet",
                        columns=["outcome", "T_in", "De_sq_bnd", "lambda_ratio_lo"])
    Ec = E[E.outcome == "converged"]
    ind = Ec[Ec.De_sq_bnd <= cs.De_sq_max]
    ceil = {"Min": cs.T_max_C_by_class["mineral"], "Wg": cs.T_max_C_by_class["water_glycol"]}
    den = {"rows_1_2_6_in_domain": int(len(ind)),
           "row_3_frame": {c: int(((Ec.T_in < t) & (Ec.lambda_ratio_lo >= cs.lambda_ratio_min)
                                   & (Ec.De_sq_bnd <= cs.De_sq_max)).sum()) for c, t in ceil.items()},
           "rows_4_5_restricted": int((Ec.T_in < ceil["Wg"]).sum()),
           "row_7_class1_smooth": {"Min": int(json.loads((OUT / "R68_D8_reemit.report.json").read_text())
                                              ["OpsSurvive_n_class1"]), "Wg": int((O.cls_lo == 1).sum())},
           "population": "box E corrected, converged evaluations (rows 1-6); box D class 1 at sigma* lo (row 7)"}
    rep["table2_denominators"] = den
    # ---- R-72.4.5
    fj = json.loads(pathlib.Path("Z:/Papers/Paper 3/New/figures/fig_ops.json").read_text(encoding="utf-8"))
    pb = fj["panel_b_sigma_lo"]
    n1 = sum(pb["mineral"][k] for k in ("survives", "g2_only", "g3_only", "g2_and_g3"))
    rep["fig6b_pair"] = {"mineral_pct_plotted": 100 * pb["mineral"]["survives"] / n1,
                         "water_glycol_pct_plotted": 100 * pb["water_glycol"]["survives"] / n1,
                         "basis": "fig_u3 panel (b): box D binding labels as stored, i.e. the as-run (water) "
                                  "thermal properties for both ceilings",
                         "OpsSurviveMinPctMin_D8": float(json.loads((OUT / "R68_D8_reemit.report.json").read_text())
                                                         ["OpsSurviveMinPctMin"])}
    # ---- R-72.4.4: typed numerals, file:line
    import run_R57_thermal_by_route as R57
    import run_R56_R1 as R1
    rule = json.loads((OUT / "boxF_R41_band_rule.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "ball_socket" / "campaign_frozen_v7.json").read_text(encoding="utf-8"))["config"]
    lam = next(a for a in cfg["input_space"] if a["name"] == "lam")
    rep["numerals"] = {
        "SatDeltaHi": (0.1, line_of("scripts/run_R57_memory_off.py", r"for d in \(0\.1, 0\.03\)")),
        "SatDeltaLo": (0.03, line_of("scripts/run_R57_memory_off.py", r"for d in \(0\.1, 0\.03\)")),
        "VerifDeltaHi": (max(R1.DELTAS), line_of("scripts/run_R56_R1.py", r"^DELTAS = ")),
        "FilmAreaClosedDelta": (round(1.0 - 0.7, 12), line_of("scripts/run_R48_film_at_common_eps.py", r"^PSI = ")
                                + " (psi = eps cos theta_min = 1 - delta; psi = 0.7, "
                                + line_of("scripts/ch9_numbers.py", r'"BsNineFilmAreaClosed"\] = \(_ratio\(2\.0, 0\.7') + ")"),
        "LamAxisLo": (float(lam["lower"]), "ball_socket/campaign_frozen_v7.json config.input_space lam"),
        "LamAxisHi": (float(lam["upper"]), "ball_socket/campaign_frozen_v7.json config.input_space lam"),
        "LamSplit": (1e-3, line_of("scripts/ch9_numbers.py", r'"BsNineEOneAboveMilliPct"\] = ')),
        "AdiabFactor": (max(R57.FACTORS), line_of("scripts/run_R57_thermal_by_route.py", r"^FACTORS = ")),
        "PCLamLo": (float(rule["constant_band"]["Lambda_bar_window"][0]), "results/ch9/boxF_R41_band_rule.json "
                    "constant_band.Lambda_bar_window (3 sigma_lo/sigma_bar)"),
        "PCLamHi": (float(rule["constant_band"]["Lambda_bar_window"][1]), "the same, 3 sigma_hi/sigma_bar")}
    from ball_socket.immutable import guarded_path
    guarded_path(str(REPORT), what="R-72 extraction")
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "provenance"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
