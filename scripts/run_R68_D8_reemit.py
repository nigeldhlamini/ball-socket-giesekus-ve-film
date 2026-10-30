"""R-68.2: the thermal quantities on the mineral class, and registrations 3 and 4.  No solves (it
reads the re-solve, R68_D8_resolve.parquet, and the re-routed L2 rises, R68_D8_l2rise.parquet).

Per converged row of boxes D and E (corrected), at the STORED solution (tag
first_order_fixed_solution), except box D's 70 re-solved rows (tag resolved):
  route_min  recomputed with the mineral properties, in the solver's precedence (Brinkman first,
             then q_min; thermal.py:37-48, solver.py:473-484); at the defaults the same rule
             reproduces every stored route (asserted);
  rise_min   L1 -> L1   rise_def x rho c_p,def / rho c_p,min  (L1 = P / (rho c_p Q), thermal.py:31-34)
             L2 -> L2   rise_def x k_def / k_min (the L2 rise is Phi h^2 / 12 k_f at fixed fields: the
                        advection term vanishes, phi-spread asserted in R-67.2)
             L1 -> L2   the L2 rise computed at the stored fields with the mineral properties
                        (R68_D8_l2rise.parquet).
Because of the clearance feedback, fixed-solution rises are upper bounds.

Macros (mineral class; water-glycol stays on the defaults, one end of its range, until Totten's
density): ThetaFlowMed/AbovePct (box E, Theta x rho c_p ratio), NahmeAdiabLo/HiPct (box E, g3 domain,
x6), OpsSurviveMinPct (box D class 1 at sigma* lo with the re-solved classes), ETwoCeilingReachPct and
ETwoGapPp (box E, restricted to T_in < 60 C, mineral labels with the mineral rise), BsBrink/BsQmin (box
B's stored solutions, routes recomputed), ThermClearNm (the E15-draft closure, conduction route,
x k ratio).  The mechanical moves: the class counts and the six S50s (on S_s) with the re-solved classes.
Registered (R-68.2): 3. no S50 moves by more than 3 %; 4. ThetaFlowAbovePct (mineral) >= 88.1 %.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
DEST = OUT / "R68_D8_mineral_rows.parquet"
REPORT = OUT / "R68_D8_reemit.report.json"
BOXD = OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet"
BOXE = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
BOXB = OUT / "audit_N8192_seed20260727.parquet"


def routes(X, k_f, rho_cp):
    from ball_socket import thermal as bt
    from ball_socket import representative as rep
    V = X.spin_ratio * X.omega * rep.R_B_M * np.sin(np.deg2rad(X.get("theta_max_deg", 73.0)))
    br = X.eta_0 * V ** 2 / (k_f * bt.BR_DT_REF_K) > bt.BR_THRESHOLD
    q = X.Q_leak <= X.P_loss / (rho_cp * bt.DELTA_T_SAFETY_K)
    return np.where(br, "L2_brinkman", np.where(q, "L1->L2_qmin", "L1"))


def mineral_rows(X, box, l2):
    import bs_thermal_props as W
    from src.thermal import ThermalProperties
    th = ThermalProperties()
    rc_def, rc_min = th.rho * th.c_p, W.MINERAL["rho"] * W.MINERAL["c_p"]
    r0 = routes(X, th.k_f, rc_def)
    assert (r0 == X.thermal_route.values).all(), f"box {box}: default routes not reproduced"
    X = X.assign(route_min=routes(X, W.MINERAL["k_f"], rc_min), rise_def=X.T_max_bnd - X.T_in)
    L1d, L1m = X.thermal_route == "L1", X.route_min == "L1"
    X["rise_min"] = np.where(L1d & L1m, X.rise_def * rc_def / rc_min,
                             np.where(~L1d, X.rise_def * th.k_f / W.MINERAL["k_f"], np.nan))
    key = ["sample_id"] + (["model_id"] if box == "E" else [])
    l2b = l2[l2.box == box].drop(columns=["box"])
    if box == "D":
        l2b = l2b.drop(columns=["model_id"])
    X = X.merge(l2b[key + ["rise_L2_min_bnd"]], on=key, how="left")
    sel = L1d.values & ~L1m.values
    assert X.loc[sel, "rise_L2_min_bnd"].notna().all(), f"box {box}: a re-routed row lacks its L2 rise"
    X.loc[sel, "rise_min"] = X.loc[sel, "rise_L2_min_bnd"]
    X["tag"] = "first_order_fixed_solution"
    return X, rc_def / rc_min, th.k_f / W.MINERAL["k_f"]


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    from ball_socket.campaign_config_v4 import ConstraintSet
    import fig_R61_update as G
    from run_R57_thermal_by_route import walther_b_band
    prov = require_clean_tree("R-68.2 D-8 re-emission")
    cs = ConstraintSet()
    MIN_C, WG_C, DE = cs.T_max_C_by_class["mineral"], cs.T_max_C_by_class["water_glycol"], cs.De_sq_max
    l2 = pd.read_parquet(OUT / "R68_D8_l2rise.parquet")
    # the forward check summed film + asperity load; box E's stored F is the FILM load (F_asp separate),
    # so the reproduction residual is |F rel - F_asp| / F (box D: F_asp = 0)
    _Fs = pd.concat([pd.read_parquet(BOXD, columns=["sample_id", "F"]).assign(box="D", model_id=-1, F_asp=0.0),
                     pd.read_parquet(BOXE, columns=["sample_id", "model_id", "F", "F_asp"]).assign(box="E")])
    _ck = l2.merge(_Fs, on=["box", "sample_id", "model_id"], how="left")
    _res = (_ck.F * _ck.F_repro_rel - _ck.F_asp.fillna(0.0)).abs() / _ck.F
    assert float(_res.max()) <= 1e-9, float(_res.max())
    L2_REPRO_MAX = float(_res.max())
    rs = pd.read_parquet(OUT / "R68_D8_resolve.parquet").set_index("sample_id")
    base = ["sample_id", "outcome", "T_in", "T_max_bnd", "De_sq_bnd", "h_min_bnd", "lambda_ratio_lo", "lambda_ratio_hi",
            "thermal_route", "P_loss", "Q_leak", "omega", "eta_0", "spin_ratio"]
    D = pd.read_parquet(BOXD, columns=base)
    D = D[D.outcome == "converged"].reset_index(drop=True)
    E = pd.read_parquet(BOXE, columns=base + ["model_id"])
    E = E[E.outcome == "converged"].reset_index(drop=True)
    D, r_rc, r_k = mineral_rows(D, "D", l2)
    E, _, _ = mineral_rows(E, "E", l2)
    # box D's re-solved rows replace their fixed-solution values
    D["lambda_ratio_lo_min"], D["lambda_ratio_hi_min"], D["De_sq_min"] = D.lambda_ratio_lo, D.lambda_ratio_hi, D.De_sq_bnd
    for sid, r in rs.iterrows():
        i = D.index[D.sample_id == sid]
        if r.outcome == "converged":
            D.loc[i, ["rise_min", "lambda_ratio_lo_min", "lambda_ratio_hi_min", "De_sq_min", "tag"]] = \
                [r.T_max_bnd - D.loc[i, "T_in"].iloc[0], r.lam_lo, r.lam_hi, r.De_sq_bnd, "resolved"]
        else:
            D.loc[i, "tag"] = "resolved_no_equilibrium"
    rep = {"provenance": prov, "thermal_props": "by_class (D-8), mineral", "ratio_rho_cp": r_rc, "ratio_k": r_k,
           "l2rise_film_load_repro_max_rel": L2_REPRO_MAX,
           "routes_min": {"D": pd.Series(D.route_min).value_counts().to_dict(),
                          "E": pd.Series(E.route_min).value_counts().to_dict()},
           "tags_D": D.tag.value_counts().to_dict()}
    # ---- through-flow group (box E), exact at fixed solution
    Q = E.Q_leak.astype(float)
    th_def = E.P_loss / ((1000.0 * 4000.0) * Q.clip(lower=1e-30) * 10.0)
    th_def[Q <= 0] = np.inf
    th_min = th_def * r_rc
    rep["ThetaFlowMedMin"] = float(np.quantile(th_min[np.isfinite(th_min)], 0.5))
    rep["ThetaFlowAbovePctMin"] = 100 * float((th_min >= 1).mean())
    rep["ThetaFlowAbovePctDef_check"] = 100 * float((th_def >= 1).mean())
    # ---- Nahme (box E, g3 domain), x6 on the mineral rise
    ind = E[E.De_sq_bnd <= DE]
    b_lo, b_hi = walther_b_band(ind.T_in.values)
    dT6 = 6.0 * ind.rise_min.values
    rep["NahmeAdiabLoPctMin"] = 100 * float((b_lo * dT6 > 1).mean())
    rep["NahmeAdiabHiPctMin"] = 100 * float((b_hi * dT6 > 1).mean())
    # ---- operating space: classes with the re-solved rows; survivors on the mineral rise
    O = pd.read_parquet(OUT / "R65_ops_solv.parquet").set_index("sample_id")
    cls = O.cls_lo.copy()
    cls_hi = O.cls_hi.copy()
    for sid, r in rs.iterrows():
        if r.outcome == "converged":
            cls.loc[sid] = 1 if r.lam_lo >= cs.lambda_ratio_min else 2
            cls_hi.loc[sid] = 1 if r.lam_hi >= cs.lambda_ratio_min else 2
        else:
            cls.loc[sid], cls_hi.loc[sid] = 4, 4            # no pure-film equilibrium; mixed arms not re-solved
    rep["class_counts_min"] = {"lo": {int(k): int(v) for k, v in cls.value_counts().sort_index().items()},
                               "hi": {int(k): int(v) for k, v in cls_hi.value_counts().sort_index().items()}}
    rep["class_changes"] = {"lo": int((cls != O.cls_lo).sum()), "hi": int((cls_hi != O.cls_hi).sum())}
    c1 = D[D.sample_id.isin(cls.index[cls == 1])]
    surv = (c1.T_in + c1.rise_min <= MIN_C) & (c1.De_sq_min <= DE)
    rep["OpsSurviveMinPctMin"] = 100 * float(surv.mean())
    rep["OpsSurvive_n_class1"] = int(len(c1))
    # the six S50s on S_s with the updated classes (fig_u3's procedure and seed)
    o65 = json.loads((OUT / "R65_ops_solv.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(G.SEED)
    S = O.S_s
    s50, shifts = [], {}
    for i, ((t0, t1), band) in enumerate(zip(G.THETA_BANDS, ("One", "Two", "Three"))):
        m = (O.theta_min_deg >= t0) & (O.theta_min_deg < t1)
        logS = np.log10(S[m].values)
        for key, tag in (("film_supported_1_2", "Film"), ("full_film_1", "Full")):
            ok = np.isin(cls[m].values, [1, 2]) if tag == "Film" else cls[m].values == 1
            v, ci, _ = G._logit_s50(logS, ok.astype(float), rng)
            old = o65["S50_sigma_lo"][i][key]["S50"]
            shifts[f"OpsSFifty{tag}{band}"] = {"S50_def": old, "S50_min": v, "ci_min": ci, "shift_pct": 100 * (v / old - 1)}
    rep["S50_min"] = shifts
    rep["S50_max_abs_shift_pct"] = max(abs(v["shift_pct"]) for v in shifts.values())
    # ---- E2 (box E restricted to T_in < 60 C): mineral labels on the mineral rise
    rest = E[E.T_in < WG_C]
    rep["ETwoCeilingReachPctMin"] = 100 * float(((rest.T_in + rest.rise_def > WG_C)
                                                 & (rest.T_in + rest.rise_min <= MIN_C)).mean())
    gaps = {}
    for end in ("lo", "hi"):
        g1 = rest[f"lambda_ratio_{end}"] < cs.lambda_ratio_min
        g3 = rest.De_sq_bnd > DE

        def lab(g2):
            n = g1.astype(int) + g2.astype(int) + g3.astype(int)
            return pd.Series(np.where(n == 0, "none", np.where(n > 1, "multi", np.where(g1, "g1", np.where(g2, "g2", "g3")))))
        lm = lab(rest.T_in + rest.rise_min > MIN_C)
        lw = lab(rest.T_max_bnd > WG_C)
        def dom(s):
            v = s.value_counts().drop(labels=["none", "multi"], errors="ignore")
            return str(v.idxmax()), float(100 * v.max() / len(s))
        (dm, pm), (dw, pw) = dom(lm), dom(lw)
        gaps[end] = {"mineral_dominant": dm, "mineral_pct": pm, "wg_dominant": dw, "wg_pct": pw, "agree": dm == dw}
    rep["ETwo_restricted_min"] = gaps
    rep["ETwoGapPpMin"] = max(abs(g["mineral_pct"] - g["wg_pct"]) for g in gaps.values())
    rep["ETwo_classes_agree_min"] = all(g["agree"] for g in gaps.values())
    # ---- box B routes on the mineral properties (stored solutions)
    B = pd.read_parquet(BOXB, columns=["outcome", "thermal_route", "P_loss", "Q_leak", "omega", "eta_0", "spin_ratio",
                                       "theta_max_deg"])
    B = B[B.outcome == "converged"]
    import bs_thermal_props as W
    rb = routes(B, W.MINERAL["k_f"], W.MINERAL["rho"] * W.MINERAL["c_p"])
    rep["BsBrinkMin"] = 100 * float((rb == "L2_brinkman").mean())
    rep["BsQminMin"] = 100 * float((rb == "L1->L2_qmin").mean())
    # ---- the E15-draft thermal closure (conduction route): Delta c x k ratio at fixed solution
    e15p = json.loads((OUT / "R57_E15_prediction.json").read_text(encoding="utf-8"))
    assert "conduction route" in e15p["thermal_closure"]["note"]
    rep["ThermClearNmMin"] = 1000 * e15p["thermal_closure"]["dc_um"] * r_k
    rep["registered"] = {"3_S50_shift_le_3pct": bool(rep["S50_max_abs_shift_pct"] <= 3.0),
                         "4_ThetaFlowAbovePct_min_ge_88.1": bool(rep["ThetaFlowAbovePctMin"] >= 88.1)}
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-68.2 mineral rows")
    guarded_path(str(REPORT), what="R-68.2 re-emission report")
    pd.concat([D.assign(box="D", model_id=-1), E.assign(box="E")], ignore_index=True)[
        ["box", "sample_id", "model_id", "thermal_route", "route_min", "rise_def", "rise_min", "tag"]].assign(
        git_commit=prov["commit"], thermal_props="by_class (D-8), mineral").to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
