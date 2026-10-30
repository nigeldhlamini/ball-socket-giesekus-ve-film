"""R-68.2: re-solve the 70 rows at risk under D-8 (R67_D8_bound.parquet, conservative variant r = 4),
on the mineral properties, through the wrapper (scripts/bs_thermal_props.py).

Each row is box D's pure-film row (contact None) solved as box D solved it: production mesh
32x24, production root-finder, clearance closure on the bulk rise, memory_sign "modulus" (box D's
rows are stored on the modulus path; using it isolates D-8 from D-6).  Lambda at both sigma* ends
is h_min_bnd / sigma*_end, with sigma*_end = h_min_bnd / lambda_ratio_end from the stored row.
Four rows are also solved on the DEFAULT properties as a spot check of the wrapper on box D
(stored h_min_bnd to 1e-9).

Registered (R-68.2): 1. on every re-solved row |dLambda_resolved| <= 1.5 |dLambda_D8| (conservative),
at both ends; 2. class changes at most among these 70 rows; (3, 4: in run_R68_D8_reemit.py).

    python scripts/run_R68_D8_resolve.py run [--workers 4]    # checkpointed; rerun resumes
    python scripts/run_R68_D8_resolve.py report
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
BOUND = OUT / "R67_D8_bound.parquet"
CKPT = OUT / "R68_D8_resolve_ckpt"
DEST = OUT / "R68_D8_resolve.parquet"
REPORT = OUT / "R68_D8_resolve.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
MEMORY_SIGN = "modulus"     # box D's rows as stored (R-59.3: explicit)
N_SPOT = 4


def jobs():
    Bd = pd.read_parquet(BOUND)
    ids = sorted(Bd[Bd.at_risk_cons].sample_id.astype(int))
    D = pd.read_parquet(SRC).set_index("sample_id")
    out = []
    for i, sid in enumerate(ids):
        r = D.loc[sid]
        base = dict(sample_id=sid, x={k: float(r[k]) for k in AXES}, stored_h=float(r.h_min_bnd),
                    sig_lo=float(r.h_min_bnd / r.lambda_ratio_lo), sig_hi=float(r.h_min_bnd / r.lambda_ratio_hi),
                    lam_lo=float(r.lambda_ratio_lo), lam_hi=float(r.lambda_ratio_hi),
                    route_stored=str(r.thermal_route))
        out.append((base, "mineral"))
        if i < N_SPOT:
            out.append((base, "default"))
    return out


def job(args):
    p, props = args
    import warnings
    warnings.simplefilter("ignore")
    import bs_thermal_props as W
    t0 = time.time()
    slv, ci = W.build(p["x"], None, None, thermal_props=W.MINERAL if props == "mineral" else None)
    slv.memory_sign = MEMORY_SIGN
    rec = dict(sample_id=p["sample_id"], props=props, lam_lo_stored=p["lam_lo"], lam_hi_stored=p["lam_hi"],
               route_stored=p["route_stored"])
    try:
        perf = slv.solve_equilibrium(ci)
    except Exception as exc:
        rec.update(outcome=type(exc).__name__, h_min_bnd=np.nan, T_max_bnd=np.nan, De_sq_bnd=np.nan, eps_ecc=np.nan,
                   wall_s=time.time() - t0)
        return rec
    rec.update(outcome="converged", h_min_bnd=perf.h_min_bnd, T_max_bnd=perf.T_max_bnd, De_sq_bnd=perf.De_sq_bnd,
               eps_ecc=perf.eps_ecc, c_equilibrium=perf.c_equilibrium, T_bulk=perf.T_bulk,
               thermal_route=perf.thermal_route, lam_lo=perf.h_min_bnd / p["sig_lo"],
               lam_hi=perf.h_min_bnd / p["sig_hi"], repro_rel=abs(perf.h_min_bnd / p["stored_h"] - 1),
               wall_s=time.time() - t0)
    return rec


def stage_run(workers):
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-68.2 D-8 re-solve")
    CKPT.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs() if not (CKPT / f"s{j[0]['sample_id']:05d}_{j[1]}.json").exists()]
    print(f"{len(todo)} solves to do", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job, j): j for j in todo}
        for f in as_completed(futs):
            r = f.result()
            r["git_commit"] = prov["commit"]
            part = CKPT / f"s{r['sample_id']:05d}_{r['props']}.json"
            tmp = part.with_suffix(".tmp")
            tmp.write_text(json.dumps(r, default=float), encoding="utf-8")
            tmp.replace(part)
            print(f"{time.strftime('%H:%M:%S')} s{r['sample_id']} {r['props']} {r['outcome']} {r['wall_s']:.0f}s", flush=True)
    return 0


def stage_report():
    from ball_socket.provenance import require_clean_tree
    from ball_socket.campaign_config_v4 import ConstraintSet
    prov = require_clean_tree("R-68.2 D-8 re-solve report")
    cs = ConstraintSet()
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CKPT.glob("*.json"))]
    assert len(rows) == len(jobs()), len(rows)
    R = pd.DataFrame(rows)
    Bd = pd.read_parquet(BOUND).set_index("sample_id")
    M = R[R.props == "mineral"].set_index("sample_id").join(Bd[["dLam_lo_cons", "dLam_hi_cons", "dLam_lo", "dLam_hi"]])
    M["dLam_lo_res"], M["dLam_hi_res"] = M.lam_lo - M.lam_lo_stored, M.lam_hi - M.lam_hi_stored
    conv = M.outcome == "converged"
    ok1 = conv & (M.dLam_lo_res.abs() <= 1.5 * M.dLam_lo_cons.abs()) & (M.dLam_hi_res.abs() <= 1.5 * M.dLam_hi_cons.abs())
    thr = cs.lambda_ratio_min
    for end in ("lo", "hi"):
        M[f"class_{end}_stored"] = np.where(M[f"lam_{end}_stored"] >= thr, 1, 2)
        M[f"class_{end}_min"] = np.where(~conv, 0, np.where(M[f"lam_{end}"] >= thr, 1, 2))   # 0: no pure-film eq.
    S = R[R.props == "default"]
    rep = {"provenance": prov, "memory_sign": MEMORY_SIGN, "thermal_props": "by_class (D-8), mineral",
           "n_rows": int(len(M)), "converged": int(conv.sum()),
           "nonconverged": M[~conv].outcome.value_counts().to_dict(),
           "spot_check_default_repro_max_rel": float(S.repro_rel.max()) if len(S) else None,
           "spot_check_passes_1e-9": bool(len(S) and S.repro_rel.max() <= 1e-9),
           "registered_1_resolved_within_1.5x_bound": bool(ok1.all()),
           "registered_1_failing_rows": [int(i) for i in M.index[~ok1]],
           "class_changes_lo": int((M.class_lo_min != M.class_lo_stored).sum()),
           "class_changes_hi": int((M.class_hi_min != M.class_hi_stored).sum()),
           "class_change_detail": json.loads(M[(M.class_lo_min != M.class_lo_stored) | (M.class_hi_min != M.class_hi_stored)]
                                             [["lam_lo_stored", "lam_lo", "lam_hi_stored", "lam_hi", "outcome"]]
                                             .reset_index().to_json(orient="records")),
           "max_abs_dLam_resolved": float(np.nanmax(np.abs(M[["dLam_lo_res", "dLam_hi_res"]].values))),
           "ratio_resolved_to_bound_max": float(np.nanmax((M.dLam_lo_res.abs() / M.dLam_lo_cons.abs()).values)),
           "route_changes": int((conv & (M.thermal_route != M.route_stored)).sum())}
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-68.2 D-8 re-solve")
    guarded_path(str(REPORT), what="R-68.2 D-8 re-solve report")
    M.reset_index().assign(git_commit=prov["commit"], thermal_props="by_class (D-8), mineral").to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "class_change_detail"}, indent=1, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "report"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    return stage_run(min(a.workers, 4)) if a.stage == "run" else stage_report()


if __name__ == "__main__":
    raise SystemExit(main())
