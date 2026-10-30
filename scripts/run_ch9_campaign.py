#!/usr/bin/env python
"""Ch.9 ball-socket campaign runner (frozen configuration).

Stages:
  audit  -- Phase 3 Sobol audit set (N = 2048, independent seed, held out)
  e5     -- E5 rated-corner sweep (160 solves)
  oat    -- one-at-a-time sensitivity block

Usage:  python scripts/run_ch9_campaign.py <stage> [--n N] [--workers W]
Writes parquet + a JSON summary under results/ch9/.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ball_socket import campaign as C                      # noqa: E402
from ball_socket.campaign_config_v4 import (N_AUDIT, OAT_BLOCK,  # noqa: E402
                                            OAT_N, INPUT_SPACE,
                                            E5_LAMBDA, E5_ETA_S_RATIO)
from ball_socket import e5_fluid_spec as E5                # noqa: E402

OUT = ROOT / "results" / "ch9"
AUDIT_SEED = 20260727            # independent of any model-fitting seed

# theta_min is an AXIS in box B but is not part of the E5 point-set spec;
# E5 is evaluated at the axis midpoint.  Gap-fill, flagged in the hand-back.
E5_THETA_MIN_DEG = 30.0


def _save(df, summary, name):
    # THIS FUNCTION IS THE LIVE HAZARD THE SHARED GUARD EXISTS FOR (R-27.3).
    # It constructs `audit_N{n}_seed{AUDIT_SEED}` -- box B's EXACT artefact
    # name -- so running this driver at N=8192 would overwrite a parquet cited
    # by sha256 in Chapter 9 and in the delivered CME export, and the one
    # erratum ERR-8 attaches to.  The box-D driver was written separately to
    # avoid the collision; that avoided it without removing it.
    from ball_socket.immutable import assert_writable
    OUT.mkdir(parents=True, exist_ok=True)
    pq = OUT / f"{name}.parquet"
    assert_writable(str(pq), what="campaign artefact")
    df.to_parquet(pq, index=False)
    summary["parquet"] = str(pq)
    (OUT / f"{name}_summary.json").write_text(json.dumps(summary, indent=2,
                                                         default=str))
    print(f"  wrote {pq}  ({len(df)} rows)")
    return pq


def stage_audit(n: int, workers: int):
    print(f"Phase 3 audit: N = {n}, seed = {AUDIT_SEED}")
    U = C.sobol_unit(n, AUDIT_SEED)
    xs = [C.unit_to_x(u) for u in U]
    df = C.run_batch(xs, workers=workers)
    s = C.summarise(df)
    s.update(stage="audit", n_requested=n, seed=AUDIT_SEED,
             wall_seconds=df.attrs.get("wall_seconds"))
    _save(df, s, f"audit_N{n}_seed{AUDIT_SEED}")
    return df, s


def assert_label_alignment(df, xs) -> None:
    """Row i of `df` must be the solve of `xs[i]` (ERR-26, R-53.1).

    Positional attachment is correct only while the runner returns submission
    order.  It did not on 2026-07-29, and 150 of e5_sweep.parquet's 160 rows
    carry another row's labels.  This turns that guarantee into a check.
    """
    for i, x in enumerate(xs):
        bad = [k for k, v in x.items() if float(df.iloc[i][k]) != float(v)]
        if bad:
            raise SystemExit(
                f"label alignment FAILED at row {i} on {bad}: the runner did not "
                f"return submission order, so labels would land on the wrong rows "
                f"(ERR-26).  Attach them by input instead.")


def stage_e5(workers: int):
    pts = E5.build_e5_points()
    print(f"E5 sweep: {len(pts)} solves")
    xs, meta = [], []
    for p in pts:
        # Gap-fill (dated ledger note, 2026-07-27; NOT an amendment): the E5
        # registration fixes grade, nu_100, c0 and corner but leaves the rest of
        # the fluid triplet unspecified.  A mineral oil is mostly solvent, so
        # eta_s/eta_0 = 0.95 with lambda at the axis floor; the earlier 0.3
        # (70 % polymer) was not mineral-oil-like.
        xs.append(dict(omega=p.omega_rad_s, p_ch=p.p_ch_bar * 1e5,
                       c0=p.c0_um * 1e-6, T_in=p.T_C,
                       eta_0=p.eta0, lam=E5_LAMBDA, eta_s_ratio=E5_ETA_S_RATIO,
                       theta_min_deg=E5_THETA_MIN_DEG))
        meta.append(dict(grade=p.grade, nu_100=p.nu_100, corner=p.corner,
                         thick_arm=p.thick_arm,
                         in_box=E5.AXES["eta0"][0] <= p.eta0 <= E5.AXES["eta0"][1]))
    # BOX B: the load is coupled by construction (F_app = p_ch * A_p), so E5
    # needs no derived load coordinate at all -- the point-set mapping that box
    # A required is gone.
    df = C.run_batch(xs, workers=workers)
    # ERR-26 / R-53.1.  These labels are attached BY POSITION.  That is correct
    # only while run_batch returns rows in submission order, which it has since
    # R-21.6 and did NOT in the 2026-07-29 run that produced e5_sweep.parquet:
    # 150 of its 160 rows carry a label belonging to another row.  The guarantee
    # is now CHECKED here rather than relied on -- every row's own inputs must be
    # the inputs of the point its label describes.
    assert_label_alignment(df, xs)
    for k in ("grade", "nu_100", "corner", "thick_arm", "in_box"):
        df[k] = [m[k] for m in meta]
    s = C.summarise(df)
    s.update(stage="e5", n=len(df), wall_seconds=df.attrs.get("wall_seconds"))
    _save(df, s, "e5_sweep")
    return df, s


def stage_oat(workers: int, oat_n: int | None = None):
    """OAT block: spin_ratio, theta_max, alpha_th, each on a fixed Sobol prefix
    so only the swept parameter differs.

    ``oat_n`` overrides the registered OAT_N.  Amended 32 -> 256 per cell by
    dated ledger entry L-10 (2026-07-29): at box B's ~25 % convergence, 32
    samples yield ~8 converged and ~4 infeasible per cell, which cannot resolve
    a dominant constraint.  The amendment is outcome-neutral -- a larger N could
    as easily produce a genuine theta_max STOP as dissolve one.
    """
    n_cell = int(oat_n or OAT_N)
    U = C.sobol_unit(n_cell, AUDIT_SEED + 1)
    xs = [C.unit_to_x(u) for u in U]
    frames, summaries = [], {}
    for param, values in OAT_BLOCK.items():
        for v in values:
            kw = {}
            if param == "spin_ratio":
                kw["spin_ratio"] = float(v)
            elif param == "theta_max_deg":
                kw["theta_max_deg"] = float(v)
            elif param == "theta_min_deg":
                kw["theta_min_deg"] = float(v)
            elif param == "alpha_th_rel":
                kw["alpha_th_rel"] = float(v)
            print(f"  OAT {param} = {v}")
            df = C.run_batch(xs, workers=workers, **kw)
            df["oat_param"], df["oat_value"] = param, float(v)
            frames.append(df)
            summaries[f"{param}={v}"] = C.summarise(df)
    import pandas as pd
    allc = pd.concat(frames, ignore_index=True)
    s = {"stage": "oat", "n_per_cell": n_cell,
         "registered_n_per_cell": OAT_N,
         "amendment": ("OAT_N 32 -> 256 per cell, ledger L-10 (2026-07-29); "
                       "statistical power, outcome-neutral"),
         "cells": summaries}
    _save(allc, s, f"oat_block_n{n_cell}")
    return allc, s


def _bo_one(band: str, seed: int, n_max: int) -> dict:
    """One BO arm (module-level so it is picklable under Windows spawn)."""
    import time as _t
    from ball_socket.bo_problem import make_problem
    from ball_socket.campaign_config_v4 import N_INIT, INPUT_SPACE
    from src.bo.campaign import BOCampaign
    t0 = _t.time()
    prob = make_problem(fluid_class="mineral", band=band)
    camp = BOCampaign(prob, n_init=N_INIT, n_max=n_max, kernel="matern52",
                      use_classifier=True, entropy_only=True,
                      seed=1000 + seed, sobol_seed=seed,
                      persist_dir=str(OUT / "bo_runs"),
                      campaign_name=f"bs_bo_{band}_s{seed}")
    res = camp.run()
    # PERSIST THE PAYLOAD.  BOCampaign.run() returns the observations in memory
    # only (persist_dir just mkdirs), so a worker that returns a summary throws
    # the campaign away.  Everything needed downstream is written here.
    import numpy as _np
    import pandas as _pd
    outdir = OUT / "bo_runs"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"bs_bo_{band}_s{seed}"

    X = _np.asarray(res["X_obs"])                       # unit cube
    df = _pd.DataFrame(
        {d.name: [float(d.from_unit(u)) for u in X[:, i]]
         for i, d in enumerate(INPUT_SPACE)})
    for i, d in enumerate(INPUT_SPACE):                  # keep unit coords too
        df[f"u_{d.name}"] = X[:, i]
    for k, v in res.get("Y_obs", {}).items():
        v = _np.asarray(v).ravel()
        if len(v) == len(df):
            df[k] = v
    df["converged"] = _np.asarray(res["converged_obs"], dtype=bool)
    df["feasible"] = _np.asarray(res["feasible_obs"], dtype=bool)
    df["iteration"] = _np.arange(len(df))
    df["arm"], df["band"], df["seed"] = "bo", band, seed
    df.to_parquet(outdir / f"{tag}_obs.parquet", index=False)

    meta = {"tag": tag, "band": band, "seed": seed, "n_max": n_max,
            "n_obs_final": int(res.get("n_obs_final", len(df))),
            "terminated_reason": res.get("terminated_reason"),
            "best_feasible_history": _np.asarray(
                res.get("best_feasible_history", [])).tolist(),
            "hausdorff_history": list(res.get("hausdorff_history", [])),
            "n_converged": int(df["converged"].sum()),
            "n_feasible": int(df["feasible"].sum()),
            "wall_s": _t.time() - t0}
    (outdir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2,
                                                        default=str))
    return dict(arm="bo", band=band, seed=seed, n_max=n_max,
                wall_s=meta["wall_s"], n_converged=meta["n_converged"],
                n_feasible=meta["n_feasible"],
                obs_parquet=str(outdir / f"{tag}_obs.parquet"))


def stage_bo(workers: int, n_max: int, seeds: int, bands: tuple[str, ...]):
    """Phase 4: BO arm + matched random arm, per seed, at BOTH sigma* band ends
    (ruling 2026-07-27: two AR constructions reported side by side rather than
    picking a band end)."""
    import time as _t
    import pandas as pd
    from ball_socket.bo_problem import make_problem
    from ball_socket.campaign_config_v4 import N_INIT
    from src.bo.campaign import BOCampaign

    rows, runs = [], []
    # Each BO arm is SEQUENTIAL by construction (every iteration refits the GPs
    # and picks the next point), so the parallelism available is ACROSS the
    # (band, seed) runs, not within them.
    tasks = [(b, s) for b in bands for s in range(seeds)]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    n_par = max(1, min(4, len(tasks)))          # torch is memory-hungry per proc
    t_all = _t.time()
    with ProcessPoolExecutor(max_workers=n_par) as ex:
        futs = {ex.submit(_bo_one, b, s, n_max): (b, s) for b, s in tasks}
        for fut in as_completed(futs):
            b, s = futs[fut]
            try:
                runs.append(fut.result())
                print(f"  BO  band={b} seed={s}: {runs[-1]['wall_s']:.0f} s")
            except BaseException as exc:
                runs.append(dict(arm="bo", band=b, seed=s, n_max=n_max,
                                 error=f"{type(exc).__name__}: {exc}"[:200]))
                print(f"  BO  band={b} seed={s}: FAILED {exc}")
    print(f"  all BO arms: {_t.time()-t_all:.0f} s wall on {n_par} concurrent")

    # --- matched random arms (batch-parallel, cheap) ---
    for band in bands:
        for seed in range(seeds):
            t0 = _t.time()
            U = C.sobol_unit(n_max, 90000 + seed)
            xs = [C.unit_to_x(u) for u in U]
            df = C.run_batch(xs, workers=workers)
            df["arm"], df["band"], df["seed"] = "random", band, seed
            rows.append(df)
            runs.append(dict(arm="random", band=band, seed=seed, n_max=n_max,
                             wall_s=_t.time() - t0,
                             n_converged=int((df.outcome == "converged").sum())))
            print(f"  RND band={band} seed={seed}: {runs[-1]['wall_s']:.0f} s")

    allr = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    s = {"stage": "bo", "n_max": n_max, "seeds": seeds,
         "bands": list(bands), "runs": runs}
    if len(allr):
        _save(allr, s, f"bo_random_arm_n{n_max}")
    (OUT / f"bo_runs_n{n_max}.json").write_text(json.dumps(s, indent=2, default=str))
    return allr, s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["audit", "e5", "oat", "bo"])
    ap.add_argument("--n-max", dest="n_max", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--bands", type=str, default="lo,hi")
    ap.add_argument("--oat-n", dest="oat_n", type=int, default=None)
    ap.add_argument("--n", type=int, default=N_AUDIT)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    if a.stage == "audit":
        _, s = stage_audit(a.n, a.workers)
    elif a.stage == "e5":
        _, s = stage_e5(a.workers)
    elif a.stage == "bo":
        _, s = stage_bo(a.workers, a.n_max, a.seeds,
                        tuple(a.bands.split(",")))
    else:
        _, s = stage_oat(a.workers, a.oat_n)
    print(f"\nstage '{a.stage}' done in {time.time()-t0:.1f} s")
    print(json.dumps({k: v for k, v in s.items() if not isinstance(v, dict)},
                     indent=2, default=str))
