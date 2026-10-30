"""R-50.2: E4 across the registered relaxation-time axis.

E4's viscoelastic result -- the squeeze-memory share of the film load -- rested
on FOUR in-domain points (the box-C Phase 6 decomposition, six points, two
outside the g3 domain).  Every non-viscoelastic result in the paper rests on
thousands of rows.  This extends the SAME measurement across lambda.

METHOD: identical to scripts/boxc_decomposition.py, whose functions are
imported rather than re-implemented:
  D1 frozen-coefficient: one linear solve at the converged full-problem state,
     memory on vs off -> share_memory = (F_all - F_no_mem) / F_all;
  D2 re-solved: full Picard re-solve with memory off -> the coupled delta.
Normal stresses ON, as in E4's registered decomposition.

DESIGN: base rows x a log grid over the registered lambda axis [1e-5, 1e-2] s,
every other input held at the base row's value.  Base rows are (a) the
six E4 points -- all six, because the g3 domain is re-evaluated at every
lambda, so the two excluded at their own lambda can re-enter at a smaller one
-- and (b) a seeded sample of converged rows of the box-D pure-film campaign.  Each (row, lambda) carries its squeeze Deborah number
De_sq = lambda max|hdot| / c on the boundary-inclusive extremum (the registered
g3 quantity) and its g3 domain flag; out-of-domain points are REPORTED and
excluded from the in-domain curve, exactly as E4 excluded two.

WHAT A FAILING RESULT LOOKS LIKE (R-48.4, gate 6): E4 is registered as "squeeze
memory the leading viscoelastic correction" and the finding it supports is a
BOUND on viscoelastic significance.  The bound fails if the in-domain memory
share reaches a material fraction of the load (the four-point maximum was
1.2e-4) anywhere on the axis.  Nothing in the design prevents that: De_sq
reaches the g3 limit on the axis.

Stages: `chunk1` (one base row x full lambda grid, timed -> projection),
`sweep` (everything, checkpointed per base row).
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
DEST = OUT / "E4_lambda_sweep.parquet"
CKPT = OUT / "E4_lambda_sweep_ckpt"
LAMS = tuple(float(v) for v in np.logspace(-5, -2, 10))
N_SAMPLE = 16
SEED = 20260921
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]


def base_rows() -> list[tuple[str, dict]]:
    fields = ROOT / "exports" / "cme" / "fields"
    dec = json.loads((OUT / "boxc_phase6_decomposition.json").read_text())
    rows = []
    for p in dec["points"]:
        # all six E4 points; the g3 domain is re-evaluated per lambda
        tag = p["point_tag"]
        x = json.loads((fields / f"{tag}.json").read_text())["inputs"]
        rows.append((f"E4:{tag}", {k: float(x[k]) for k in AXES}))
    pf = pd.read_parquet(OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet",
                         columns=["sample_id", "outcome"] + AXES)
    pick = pf[pf.outcome == "converged"].sample(N_SAMPLE, random_state=SEED)
    for r in pick.itertuples():
        rows.append((f"pf:{int(r.sample_id)}", {k: float(getattr(r, k)) for k in AXES}))
    return rows


def one(args):
    tag, x = args
    import warnings
    warnings.simplefilter("ignore")
    from scipy.sparse.linalg import spsolve
    import boxc_decomposition as B
    from ball_socket.reynolds import solve_reynolds
    from src.solver import SolverConfig
    out = []
    for lam in LAMS:
        xx = dict(x, lam=lam)
        t0 = time.time()
        rec = {"base": tag, "lam": lam, **{f"x_{k}": v for k, v in x.items() if k != "lam"}}
        try:
            slv, ci = B._build(xx, True)
            # THE REGISTERED BRACKET (R-33.1 / ERR-13).  boxc_decomposition._build
            # predates box D and leaves the solver at its default eps_cap = 0.99,
            # below the geometric ceiling 1/cos(theta_min).  The six E4 points sit
            # at eps 0.5-0.94 and are unaffected; box-D rows needing eps > 0.99
            # "blew up" in the first run of this sweep (88 of 220).  Set here,
            # read back, and recorded -- not by editing the Phase-6 script.
            import ball_socket.campaign as _C
            slv.eps_cap = float(_C._registered_eps_cap(np.deg2rad(xx["theta_min_deg"])))
            rec["realised_eps_cap"] = float(slv.eps_cap)
            sph, fluid, eta_bar, gdot, pressure, perf = B._state(slv, ci)

            def F_frozen(mem):
                sysm = sph.assemble("viscoelastic", fluid, eta_bar, gdot, pressure=pressure,
                                    include_memory=mem, include_normal_stress=True,
                                    n1_scale=1.0, hoop_scale=1.0)
                return float(sph.load_capacity(sph.to_2d(spsolve(sysm.A, sysm.b))))

            def F_resolved(mem):
                cfg = SolverConfig(include_normal_stress=True, max_iter=80, tol=1e-9, omega=0.7)
                cfg.include_memory = mem
                r = solve_reynolds(sph, fluid, cfg, model="viscoelastic", n1_mult=1.0, hoop_mult=1.0)
                return float(sph.load_capacity(r.pressure))

            Fa, Fn = F_frozen(True), F_frozen(False)
            Ra, Rn = F_resolved(True), F_resolved(False)
            De = float(lam * sph.gap_rate_max_boundary() / ci.c0)
            rec.update(status="ok", eps=float(perf.eps_ecc), F_all=Fa,
                       share_memory_D1=(Fa - Fn) / Fa, share_memory_D2=(Ra - Rn) / Ra,
                       De_sq_bnd=De, Wi_char=float(lam * np.max(gdot)),
                       in_g3_domain=bool(De <= 0.2))
        except Exception as exc:                      # recorded, never swallowed
            rec.update(status=f"{type(exc).__name__}: {str(exc)[:160]}")
        rec["wall_s"] = time.time() - t0
        out.append(rec)
    return out


def _run(rows, workers):
    from concurrent.futures import ProcessPoolExecutor
    CKPT.mkdir(exist_ok=True)
    todo = [r for r in rows if not (CKPT / f"{r[0].replace(':', '_')}.json").exists()]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for got in ex.map(one, todo):
            (CKPT / f"{got[0]['base'].replace(':', '_')}.json").write_text(json.dumps(got))
            print(f"  {got[0]['base']}: {sum(g['status'] == 'ok' for g in got)}/{len(got)} ok, "
                  f"{sum(g['wall_s'] for g in got):.0f} s", flush=True)
    return time.time() - t0


def stage_chunk1(args):
    rows = base_rows()
    t0 = time.time()
    got = one(rows[0])
    el = time.time() - t0
    print(json.dumps(got, indent=1, default=str)[:3000])
    n = len(rows) * len(LAMS)
    print(f"chunk-1: {len(LAMS)} decompositions in {el:.0f} s ({el / len(LAMS):.1f} s each); "
          f"full sweep {n} decompositions ~ {n * el / len(LAMS) / 3600 / max(args.workers, 1):.2f} h "
          f"on {args.workers} workers")
    return 0


def stage_sweep(args):
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="E4 lambda sweep")
    rows = base_rows()
    el = _run(rows, args.workers)
    recs = []
    for f in sorted(CKPT.glob("*.json")):
        recs += json.loads(f.read_text())
    df = pd.DataFrame(recs)
    df.to_parquet(DEST, index=False)
    ok = df[df.status == "ok"]
    ind = ok[ok.in_g3_domain]
    rep = {"base_rows": len(rows), "lambda_grid": list(LAMS), "decompositions": len(df),
           "ok": int(len(ok)), "failed": int((df.status != "ok").sum()),
           "in_g3_domain": int(len(ind)), "wall_min": round(el / 60, 1),
           "share_memory_D1_in_domain": {
               "max_abs": float(ind.share_memory_D1.abs().max()),
               "median_abs": float(ind.share_memory_D1.abs().median()),
               "all_negative": bool((ind.share_memory_D1 < 0).all())},
           "share_memory_D2_in_domain_max_abs": float(ind.share_memory_D2.abs().max()),
           "De_sq_range_in_domain": [float(ind.De_sq_bnd.min()), float(ind.De_sq_bnd.max())],
           "per_lambda": {f"{l:.3g}": {
               "n_in_domain": int((ind.lam == l).sum()),
               "max_abs_share": (float(ind[ind.lam == l].share_memory_D1.abs().max())
                                 if (ind.lam == l).any() else None)} for l in LAMS}}
    DEST.with_suffix(".report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["chunk1", "sweep"])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    return {"chunk1": stage_chunk1, "sweep": stage_sweep}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
