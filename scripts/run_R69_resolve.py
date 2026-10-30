"""R-69.2: re-solve, on the mineral properties, the solves where the first-order bound can change a
point's class (79 <= 200):
  * the 72 box-E mixed arms of class-3 points whose first-order floor projection falls below
    delta = 0.01 (R69_switch_check.report.json "flagged"; every one sits on the floor);
  * the 7 box-D pure-film rows whose stored capacity does NOT reproduce at c0 (they failed on a later
    outer iteration, with c widened by the closure, and carry their load at c0: a lower rise could
    give them a pure-film equilibrium).
Wrapper scripts/bs_thermal_props.py, memory_sign "modulus" (boxes D/E as stored), production mesh and
root-finder, closure on.

    python scripts/run_R69_resolve.py run [--workers 4]    # checkpointed
    python scripts/run_R69_resolve.py report
"""
from __future__ import annotations

import argparse
import glob
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
CKPT = OUT / "R69_resolve_ckpt"
REPORT = OUT / "R69_resolve.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
MEMORY_SIGN = "modulus"


def jobs():
    sw = json.loads((OUT / "R69_switch_check.report.json").read_text(encoding="utf-8"))
    F = pd.DataFrame(sw["class3"]["flagged"])
    E = pd.read_parquet(BOXE, columns=AXES + ["sample_id", "model_id", "lambda_ratio_lo", "lambda_ratio_hi", "h_min_bnd"])
    out = []
    for _, f in F.iterrows():
        r = E[(E.sample_id == f.sample_id) & (E.model_id == f.model_id)].iloc[0]
        out.append(dict(kind="arm", sample_id=int(f.sample_id), model_id=int(f.model_id), end=f.end,
                        x={k: float(r[k]) for k in AXES}))
    C = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(OUT / "R69_c0fail_ckpt" / "block_*.parquet")))])
    bad = C[(C.F_cap_c0 / C.F_film_max_stored - 1).abs() > 1e-9].sample_id.astype(int).tolist()
    D = pd.read_parquet(BOXD).set_index("sample_id")
    for sid in bad:
        r = D.loc[sid]
        out.append(dict(kind="purefilm", sample_id=sid, model_id=-1, end="both", x={k: float(r[k]) for k in AXES}))
    return out


def job(p):
    import warnings
    warnings.simplefilter("ignore")
    import bs_thermal_props as W
    from run_R57_mesh_error import _contact
    t0 = time.time()
    slv, ci = W.build(p["x"], _contact(p["model_id"]), None, thermal_props=W.MINERAL)
    slv.memory_sign = MEMORY_SIGN
    rec = {k: p[k] for k in ("kind", "sample_id", "model_id", "end")}
    try:
        perf = slv.solve_equilibrium(ci)
        rec.update(outcome="converged", h_min_bnd=perf.h_min_bnd, c_equilibrium=perf.c_equilibrium,
                   delta=perf.h_min_bnd / perf.c_equilibrium, thermal_route=perf.thermal_route,
                   De_sq_bnd=perf.De_sq_bnd, T_max_bnd=perf.T_max_bnd, eps_ecc=perf.eps_ecc)
    except Exception as exc:
        rec.update(outcome=type(exc).__name__, h_min_bnd=np.nan, De_sq_bnd=np.nan, T_max_bnd=np.nan, eps_ecc=np.nan)
    rec["wall_s"] = time.time() - t0
    return rec


def stage_run(workers):
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-69.2 re-solves")
    CKPT.mkdir(parents=True, exist_ok=True)
    J = [j for j in jobs() if not (CKPT / f"{j['kind']}_{j['sample_id']}_{j['model_id']}.json").exists()]
    print(f"{len(J)} solves to do", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=min(workers, 4)) as ex:
        for f in as_completed([ex.submit(job, j) for j in J]):
            r = f.result()
            r["git_commit"] = prov["commit"]
            part = CKPT / f"{r['kind']}_{r['sample_id']}_{r['model_id']}.json"
            part.with_suffix(".tmp").write_text(json.dumps(r, default=float), encoding="utf-8")
            part.with_suffix(".tmp").replace(part)
            print(f"{time.strftime('%H:%M:%S')} {r['kind']} s{r['sample_id']} m{r['model_id']} {r['outcome']}", flush=True)
    return 0


def stage_report():
    from ball_socket.provenance import require_clean_tree
    from ball_socket.campaign_config_v4 import ConstraintSet
    prov = require_clean_tree("R-69.2 re-solve report")
    cs = ConstraintSet()
    R = pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in sorted(CKPT.glob("*.json"))])
    assert len(R) == len(jobs())
    sw = json.loads((OUT / "R69_switch_check.report.json").read_text(encoding="utf-8"))
    T3 = pd.DataFrame(sw["class3"]["detail"])
    O = pd.read_parquet(OUT / "R65_ops_solv.parquet").set_index("sample_id")
    D = pd.read_parquet(BOXD).set_index("sample_id")
    pf = R[R.kind == "purefilm"].set_index("sample_id")
    arms = R[R.kind == "arm"]
    changes = []
    # class-3 points
    for _, t in T3.iterrows():
        sid, end = int(t.sample_id), t.end
        new = 3
        if sid in pf.index and pf.loc[sid, "outcome"] == "converged":
            lam = pf.loc[sid, "h_min_bnd"] / float(D.loc[sid, "h_min_bnd"] / D.loc[sid, f"lambda_ratio_{end}"]) \
                if pd.notna(D.loc[sid, "h_min_bnd"]) else np.nan
            new = "1 or 2 (pure film converges)"
        else:
            a = arms[(arms.sample_id == sid) & (arms.end == end)]
            kept = int(t.arms_kept_first_order) + int((a.outcome == "converged").sum())
            if kept == 0:
                new = 4
        if new != 3:
            changes.append(dict(sample_id=sid, end=end, old=3, new=new))
    # class-4 points among the 7 pure-film rows
    for sid, r in pf.iterrows():
        for end in ("lo", "hi"):
            if O.loc[sid, f"cls_{end}"] == 4 and r.outcome == "converged":
                changes.append(dict(sample_id=int(sid), end=end, old=4, new="1 or 2 (pure film converges)"))
    rep = {"provenance": prov, "solves": int(len(R)),
           "arms": {"n": int(len(arms)), "converged": int((arms.outcome == "converged").sum()),
                    "outcomes": arms.outcome.value_counts().to_dict()},
           "purefilm_7": json.loads(pf[["outcome", "delta", "thermal_route"]].reset_index().to_json(orient="records")),
           "class_changes": changes, "n_class_changes": len(changes),
           "registered_2_no_class3_or_existence_change": len([c for c in changes if c["old"] == 3]) == 0}
    from ball_socket.immutable import guarded_path
    guarded_path(str(REPORT), what="R-69.2 re-solve report")
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "report"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    return stage_run(a.workers) if a.stage == "run" else stage_report()


if __name__ == "__main__":
    raise SystemExit(main())
