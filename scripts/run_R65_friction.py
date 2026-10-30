"""R-65.3: does the dissipated power converge where the film is full?

Sample (registered by R-65.3): 24 full-film points (class 1, sigma* lo, R65_ops_solv.parquet),
stratified by theta_min tercile x delta quartile (12 cells, 2 points per cell, drawn without
replacement with SEED).  delta = h_min_bnd / c_equilibrium from box D (the production floor's
delta = h_min/c).  Each point is solved to equilibrium on the pure film (contact None, as box D)
at three meshes, 32x24, 64x48 and 128x96, root-finder xtol = rtol = 1e-10 (box G's tight call,
run_E15_boxG.rich_job), memory_sign="signed".  P_diss is the solver's P_loss
(ball_socket/solver.py:467: surface integral of eta_bar gdot^2 h).

Observed order p = log2((P32 - P64) / (P64 - P128)); Richardson correction (P128 - P64)/(2^p - 1),
relative to P128.  Undefined when the two differences change sign (oscillatory): not converged.
Registered: for delta >= 0.1, p >= 1 and |correction| <= 5 % of P128.  For delta < 0.1 it need
not converge, and is reported.

If it holds: T_f = P_diss / omega_spin, omega_spin = spin_ratio * omega (solver.py:243,
spin_ratio = 1.0 campaign_config_v4.py:90), and mu = T_f / (F_app R_b), R_b = 6.5e-3 m
(ball_socket/representative.py:20).  Values at the finest mesh.

    python scripts/run_R65_friction.py run [--workers 4]    # checkpointed; rerun resumes
    python scripts/run_R65_friction.py report
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
SRC = OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet"
CLS = OUT / "R65_ops_solv.parquet"
CKPT = OUT / "R65_friction_ckpt"
SAMPLE = OUT / "R65_friction_sample.json"
DEST = OUT / "R65_friction.parquet"
REPORT = OUT / "R65_friction.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
GRIDS = ((128, 96), (64, 48), (32, 24))      # largest first, for load balance
SEED = 20261009
MEMORY_SIGN = "signed"
TOL = 1e-10
R_B = 6.5e-3


def draw_sample() -> dict:
    D = pd.read_parquet(SRC)
    C = pd.read_parquet(CLS, columns=["sample_id", "cls_lo"])
    D = D.merge(C, on="sample_id")
    c1 = D[D.cls_lo == 1].copy()
    c1["delta"] = c1.h_min_bnd / c1.c_equilibrium
    c1["th_t"] = pd.qcut(c1.theta_min_deg, 3, labels=False)
    c1["d_q"] = pd.qcut(c1.delta, 4, labels=False)
    rng = np.random.default_rng(SEED)
    pts = []
    for t in range(3):
        for q in range(4):
            cell = c1[(c1.th_t == t) & (c1.d_q == q)].sort_values("sample_id")
            for i in rng.choice(len(cell), 2, replace=False):
                r = cell.iloc[int(i)]
                pts.append({"sample_id": int(r.sample_id), "theta_tercile": t, "delta_quartile": q,
                            "delta_boxD": float(r.delta), "P_loss_boxD": float(r.P_loss),
                            "x": {k: float(r[k]) for k in AXES}})
    return {"seed": SEED, "rule": "class 1 sigma* lo; theta_min tercile x delta quartile; 2 per cell",
            "theta_edges": [float(v) for v in pd.qcut(c1.theta_min_deg, 3, retbins=True)[1]],
            "delta_edges": [float(v) for v in pd.qcut(c1.delta, 4, retbins=True)[1]],
            "points": sorted(pts, key=lambda p: p["sample_id"])}


def job(args):
    sid, x, grid = args
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.campaign as C
    t0 = time.time()
    row = C.evaluate_x(x, contact=None, clearance_closure=True, delta_T_datum="bulk",
                       Ntheta=grid[0], Nphi=grid[1], brentq_xtol=TOL, brentq_rtol=TOL,
                       memory_sign=MEMORY_SIGN)
    # R-53.5: keep every registered-constraint quantity.  Added after the R-65.3 run: its
    # checkpoints (commit ec14f8b) lack T_max_bnd and De_sq_bnd, which R-65.3 does not use.
    from ball_socket.campaign import REGISTERED_CONSTRAINT_COLUMNS
    return {"sample_id": sid, "Ntheta": grid[0], "Nphi": grid[1],
            **{k: row.get(k) for k in REGISTERED_CONSTRAINT_COLUMNS},
            "P_loss": row.get("P_loss"), "F_app": row.get("F_app"),
            "c_equilibrium": row.get("c_equilibrium"),
            "realised_memory_sign": row.get("realised_memory_sign"), "wall_s": time.time() - t0}


def stage_run(workers: int) -> int:
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-65.3 friction convergence")
    smp = draw_sample()
    if SAMPLE.exists():
        assert json.loads(SAMPLE.read_text(encoding="utf-8"))["points"] == smp["points"], "sample changed"
    else:
        SAMPLE.write_text(json.dumps({**smp, "provenance": prov}, indent=1), encoding="utf-8")
    CKPT.mkdir(parents=True, exist_ok=True)
    todo = [(p["sample_id"], p["x"], g) for g in GRIDS for p in smp["points"]
            if not (CKPT / f"s{p['sample_id']:05d}_{g[0]}.parquet").exists()]
    print(f"{len(todo)} solves to do", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job, a): a for a in todo}
        for f in as_completed(futs):
            r = f.result()
            r["git_commit"] = prov["commit"]
            part = CKPT / f"s{r['sample_id']:05d}_{r['Ntheta']}.parquet"
            tmp = part.with_suffix(".tmp")
            pd.DataFrame([r]).to_parquet(tmp, index=False)
            tmp.replace(part)
            print(f"{time.strftime('%H:%M:%S')} s{r['sample_id']} {r['Ntheta']} {r['outcome']} "
                  f"P={r['P_loss']} {r['wall_s']:.0f}s", flush=True)
    return 0


def stage_report() -> int:
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-65.3 friction report")
    smp = json.loads(SAMPLE.read_text(encoding="utf-8"))
    R = pd.concat([pd.read_parquet(p) for p in sorted(CKPT.glob("s*.parquet"))], ignore_index=True)
    assert len(R) == 3 * len(smp["points"]) and set(R.realised_memory_sign.dropna()) <= {MEMORY_SIGN}
    from ball_socket.campaign_config_v4 import SPIN_RATIO
    rows = []
    for p in smp["points"]:
        r = R[R.sample_id == p["sample_id"]].set_index("Ntheta")
        ok = all(r.loc[n, "outcome"] == "converged" for n in (32, 64, 128))
        rec = {**{k: p[k] for k in ("sample_id", "theta_tercile", "delta_quartile", "delta_boxD", "P_loss_boxD")},
               "theta_min_deg": p["x"]["theta_min_deg"], "omega": p["x"]["omega"], "all_converged": ok}
        if ok:
            P = [float(r.loc[n, "P_loss"]) for n in (32, 64, 128)]
            d1, d2 = P[0] - P[1], P[1] - P[2]
            rec.update({"P32": P[0], "P64": P[1], "P128": P[2],
                        "delta_128": float(r.loc[128, "h_min_bnd"] / r.loc[128, "c_equilibrium"]),
                        "F_app": float(r.loc[128, "F_app"])})
            if d1 * d2 > 0:
                pp = float(np.log2(d1 / d2))
                corr = d2 / (2 ** pp - 1) if pp > 0 else np.inf
                rec.update({"order": pp, "rich_corr_rel": float(abs(corr) / abs(P[2])) if np.isfinite(corr) else np.inf,
                            "monotone": True})
            else:
                rec.update({"order": np.nan, "rich_corr_rel": np.nan, "monotone": False})
            rec["converged_class"] = bool(rec["monotone"] and rec["order"] >= 1 and rec["rich_corr_rel"] <= 0.05)
            ws = SPIN_RATIO * p["x"]["omega"]
            rec["omega_spin"] = ws
            rec["T_f"] = P[2] / ws
            rec["mu"] = rec["T_f"] / (rec["F_app"] * R_B)
        rows.append(rec)
    T = pd.DataFrame(rows)
    hi = T[T.delta_boxD >= 0.1]
    lo = T[T.delta_boxD < 0.1]
    reg = bool(hi.all_converged.all() and hi.converged_class.fillna(False).astype(bool).all())
    conv = T[T.converged_class.fillna(False).astype(bool)]
    rep = {"provenance": prov, "memory_sign": MEMORY_SIGN,
           "basis": "smooth film, no Patir-Cheng shear factors",
           "spin_rate": "omega_spin = spin_ratio * omega (ball_socket/solver.py:243), spin_ratio = "
                        f"{SPIN_RATIO} (campaign_config_v4.py:90)",
           "P_diss": "P_loss, ball_socket/solver.py:467 (all viscous dissipation in the film)",
           "n": len(T), "n_delta_ge_0p1": int(len(hi)), "n_delta_lt_0p1": int(len(lo)),
           "registered_delta_ge_0p1_order_ge_1_and_corr_le_5pct": reg,
           "delta_ge_0p1": {"converged": int(hi.converged_class.fillna(False).astype(bool).sum()),
                            "order_min": float(hi.order.min()), "order_median": float(hi.order.median()),
                            "corr_max_pct": float(100 * hi.rich_corr_rel.max())},
           "delta_lt_0p1": {"converged": int(lo.converged_class.fillna(False).astype(bool).sum()),
                            "orders": [None if not np.isfinite(v) else float(v) for v in lo.order],
                            "corr_pct": [None if not np.isfinite(v) else float(100 * v) for v in lo.rich_corr_rel]},
           "converged_points": int(len(conv))}
    if reg and len(conv):
        rep["T_f_Nm"] = {"median": float(conv.T_f.median()), "min": float(conv.T_f.min()), "max": float(conv.T_f.max())}
        rep["mu"] = {"median": float(conv.mu.median()), "min": float(conv.mu.min()), "max": float(conv.mu.max())}
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-65.3 friction")
    guarded_path(str(REPORT), what="R-65.3 friction report")
    T.assign(git_commit=prov["commit"], memory_sign=MEMORY_SIGN).to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "report", "sample"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.stage == "sample":
        print(json.dumps(draw_sample(), indent=1))
        return 0
    return stage_run(min(a.workers, 4)) if a.stage == "run" else stage_report()


if __name__ == "__main__":
    raise SystemExit(main())
