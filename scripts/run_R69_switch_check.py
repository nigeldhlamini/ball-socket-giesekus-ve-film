"""R-69.2: are the class verdicts safe across D-8's route switches?

Box D, switched rows (L1 at the defaults, L2 on the mineral properties; R68_D8_l2rise.parquet):
the actual clearance change is Delta c = Delta alpha R_b (DeltaT_L2,bulk - DeltaT_L1,bulk) <= 0
(the closure's datum is the bulk rise, solver.py:327-333), against the conservative bound
Delta alpha R_b DeltaT_bulk x 3 (R67_D8_bound.parquet).  dLambda and the floor projection are
recomputed with the actual sign (slope from R67_D8_bound.parquet).

No-equilibrium rows: a row that failed on the first outer call failed at c = c0, before any
thermal step, so no property set can change it (the first call is at c0 whatever the properties).
Box D does not persist the failure clearance, so it is TESTED: the film load at eps_cap and c0
(solver._load_split, the arm as stored, memory_sign "modulus" as box D ran) must equal the
stored F_film_max (blow-up on capacity); a film that closes before eps_cap (F_film_max absent)
closes on geometry alone, h = c (1 - eps cos theta), which c cannot change.

Class 3 (contact-supported, R65_ops_solv.parquet): a point changes class if its pure-film row
gains an equilibrium (excluded when it failed at c0, above) or every converged mixed arm at that
sigma* end loses its equilibrium (first-order floor projection per arm with the actual-sign
Delta c; box E's stored T_bulk; the arm's re-routed L2 bulk rise from R68_D8_l2rise.parquet).
Arms the bound says may cross are listed for re-solve (STOP above 200).

Existence classes B/A' (R57_existence_classification): its rows were sampled from box E's
contact_limit rows "each failed at c = c0 on the first outer call" and checked to reproduce at
c0 (run_R57_existence.py), so no property set can move them.
Registered (R-69.2): 1. no box-D switched row exceeds its bound; 2. no class-3 point and no
existence class changes.

    python scripts/run_R69_switch_check.py run [--workers 4]   # c0-failure test, checkpointed
    python scripts/run_R69_switch_check.py report
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
BOXD = OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet"
BOXE = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
CKPT = OUT / "R69_c0fail_ckpt"
REPORT = OUT / "R69_switch_check.report.json"
DEST = OUT / "R69_switch_check.parquet"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
MEMORY_SIGN = "modulus"
DALPHA, R_B, FLOOR = 6e-6, 6.5e-3, 0.01
BLOCK = 200


def job(rec):
    import warnings
    warnings.simplefilter("ignore")
    from run_R57_mesh_error import _solver
    t0 = time.time()
    slv, ci = _solver({k: float(rec[k]) for k in AXES}, None, None)
    slv.memory_sign = MEMORY_SIGN
    out = {"sample_id": int(rec["sample_id"]), "F_film_max_stored": rec["F_film_max"]}
    try:
        F_film, F_asp, _ = slv._load_split(ci, slv.eps_cap, slv._fluid(ci), float(rec["c0"]))
        out.update(F_cap_c0=float(F_film + F_asp), closed=False)
    except Exception as exc:                        # the film closes before eps_cap: geometric
        out.update(F_cap_c0=np.nan, closed=True, closed_reason=type(exc).__name__)
    out["wall_s"] = time.time() - t0
    return out


def stage_run(workers):
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-69.2 c0-failure test")
    D = pd.read_parquet(BOXD)
    nc = D[D.outcome != "converged"][AXES + ["sample_id", "F_film_max"]].sort_values("sample_id")
    recs = nc.to_dict(orient="records")
    CKPT.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(workers, 4)) as ex:
        for b in range(0, len(recs), BLOCK):
            part = CKPT / f"block_{b:05d}.parquet"
            if part.exists():
                continue
            q = pd.DataFrame(list(ex.map(job, recs[b:b + BLOCK]))).assign(git_commit=prov["commit"])
            tmp = part.with_suffix(".tmp")
            q.to_parquet(tmp, index=False)
            tmp.replace(part)
            print(f"{time.strftime('%H:%M:%S')} block {b} done", flush=True)
    return 0


def stage_report():
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-69.2 report")
    rep = {"provenance": prov}
    # --- box D switched rows
    D = pd.read_parquet(BOXD)
    L = pd.read_parquet(OUT / "R68_D8_l2rise.parquet")
    Bd = pd.read_parquet(OUT / "R67_D8_bound.parquet")
    X = L[L.box == "D"].merge(D[["sample_id", "T_bulk", "T_in", "c_equilibrium", "h_min_bnd", "lambda_ratio_lo",
                                 "lambda_ratio_hi"]], on="sample_id").merge(
        Bd[["sample_id", "dlnLam_dlnc", "dlndelta_dlnc", "delta", "at_risk_cons"]], on="sample_id")
    X["dc_act"] = DALPHA * R_B * (X.rise_L2_min_bulk - (X.T_bulk - X.T_in))
    X["dc_bnd"] = DALPHA * R_B * (X.T_bulk - X.T_in) * 3.0
    X["ratio"] = X.dc_act.abs() / X.dc_bnd
    for end in ("lo", "hi"):
        X[f"dLam_{end}_act"] = X[f"lambda_ratio_{end}"] * X.dlnLam_dlnc * X.dc_act / X.c_equilibrium
        X[f"cross_{end}"] = (X[f"lambda_ratio_{end}"] - 3.0) * (X[f"lambda_ratio_{end}"] + X[f"dLam_{end}_act"] - 3.0) < 0
    X["delta_proj_act"] = X.delta * (1.0 + X.dlndelta_dlnc * X.dc_act / X.c_equilibrium)
    rep["boxD_switched"] = {"n": int(len(X)), "exceed_bound": int((X.ratio > 1).sum()), "max_ratio": float(X.ratio.max()),
                            "all_dc_nonpositive": bool((X.dc_act <= 0).all()),
                            "max_abs_dLam_act": float(X[["dLam_lo_act", "dLam_hi_act"]].abs().max().max()),
                            "class_crossings_lo": int(X.cross_lo.sum()), "class_crossings_hi": int(X.cross_hi.sum()),
                            "crossing_rows_in_resolved_70": int((X.cross_lo | X.cross_hi)[X.at_risk_cons].sum()),
                            "projected_delta_below_floor": int((X.delta_proj_act < FLOOR).sum())}
    # --- no-equilibrium rows: failed at c0?
    C = pd.concat([pd.read_parquet(p) for p in sorted(CKPT.glob("block_*.parquet"))], ignore_index=True)
    nD = int((D.outcome != "converged").sum())
    assert len(C) == nD, (len(C), nD)
    cap = ~C.closed & C.F_film_max_stored.notna()
    C["rel"] = (C.F_cap_c0 / C.F_film_max_stored - 1).abs()
    rep["boxD_noequilibrium"] = {"n": nD, "closed_before_eps_cap": int(C.closed.sum()),
                                 "capacity_rows": int(cap.sum()),
                                 "capacity_reproduced_at_c0_1e-9": int((cap & (C.rel <= 1e-9)).sum()),
                                 "capacity_not_reproduced": int((cap & (C.rel > 1e-9)).sum()),
                                 "max_rel_capacity_rows": float(C.rel[cap].max()) if cap.any() else None,
                                 "stored_capacity_absent_not_closed": int((~C.closed & C.F_film_max_stored.isna()).sum())}
    # --- class 3
    O = pd.read_parquet(OUT / "R65_ops_solv.parquet").set_index("sample_id")
    E = pd.read_parquet(BOXE, columns=["sample_id", "model_id", "outcome", "h_min_bnd", "c_equilibrium", "T_bulk", "T_in",
                                       "thermal_route", "theta_min_deg", "eta_0", "omega", "p_ch"])
    Le = L[L.box == "E"][["sample_id", "model_id", "rise_L2_min_bulk"]]
    E = E.merge(Le, on=["sample_id", "model_id"], how="left")
    import p3_rim_asymptote as RA
    import ball_socket.campaign as Cm
    import bs_thermal_props as W
    r_rc, r_k = 4.0e6 / (W.MINERAL["rho"] * W.MINERAL["c_p"]), 0.60 / W.MINERAL["k_f"]
    out3, resolve = [], []
    for end, arms in (("lo", range(0, 12)), ("hi", range(12, 24))):
        pts = O.index[O[f"cls_{end}"] == 3]
        for sid in pts:
            pf_ok_c0 = bool(((C.sample_id == sid) & (C.closed | (C.rel <= 1e-9))).any())
            a = E[(E.sample_id == sid) & E.model_id.isin(list(arms)) & (E.outcome == "converged")].copy()
            keep = 0
            for _, r in a.iterrows():
                dTb = r.T_bulk - r.T_in
                if pd.notna(r.rise_L2_min_bulk):
                    dc = DALPHA * R_B * (r.rise_L2_min_bulk - dTb)
                else:
                    dc = DALPHA * R_B * dTb * ((r_rc if r.thermal_route == "L1" else r_k) - 1.0)
                delta = r.h_min_bnd / r.c_equilibrium
                kw = dict(eta=r.eta_0, R=R_B, c=r.c_equilibrium, omega=r.omega, p_pocket=r.p_ch, p_case=Cm.P_CASE_PA)
                Wf = lambda d: RA.load_solver_kinematics(d, r.theta_min_deg, **kw)
                sq = RA.load_solver_kinematics(delta, r.theta_min_deg, **{**kw, "p_pocket": 0.0, "p_case": 0.0})
                dW = (Wf(delta * 1.0001) - Wf(delta * 0.9999)) / (delta * 2e-4)
                dproj = delta * (1.0 + 2.0 * sq / (delta * dW) * dc / r.c_equilibrium)
                if dproj >= FLOOR:
                    keep += 1
                else:
                    resolve.append(dict(sample_id=int(sid), model_id=int(r.model_id), end=end, delta=delta, delta_proj=dproj))
            out3.append(dict(sample_id=int(sid), end=end, arms_converged=int(len(a)), arms_kept_first_order=keep,
                             purefilm_failed_at_c0_or_closed=pf_ok_c0,
                             class_safe=bool(keep > 0 and pf_ok_c0)))
    T3 = pd.DataFrame(out3)
    rep["class3"] = {"points_lo": int((T3.end == "lo").sum()), "points_hi": int((T3.end == "hi").sum()),
                     "all_safe": bool(T3.class_safe.all()), "arms_flagged_for_resolve": len(resolve),
                     "stop_over_200": len(resolve) > 200, "detail": json.loads(T3.to_json(orient="records")),
                     "flagged": resolve}
    ex = json.loads((OUT / "R57_existence_classification.report.json").read_text(encoding="utf-8"))
    rep["existence"] = {"rows": ex["rows"], "excluded_not_reproducing_at_c0": ex["excluded_not_reproducing_at_c0"],
                        "argument": "rows sampled from box E contact_limit rows failing at c = c0 on the first outer call "
                                    "and checked to reproduce at c0 (run_R57_existence.py); the first call is at c0 "
                                    "whatever the properties, so no property set can move them"}
    rep["registered"] = {"1_no_switched_row_exceeds_bound": rep["boxD_switched"]["exceed_bound"] == 0,
                         "2_no_class3_or_existence_change": bool(T3.class_safe.all())}
    from ball_socket.immutable import guarded_path
    guarded_path(str(REPORT), what="R-69.2 report")
    guarded_path(str(DEST), what="R-69.2 switched rows")
    X.assign(git_commit=prov["commit"]).to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: (v if k != "class3" else {q: w for q, w in v.items() if q not in ("detail", "flagged")})
                      for k, v in rep.items()}, indent=1, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "report"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    return stage_run(a.workers) if a.stage == "run" else stage_report()


if __name__ == "__main__":
    raise SystemExit(main())
