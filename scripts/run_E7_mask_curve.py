"""Resolved below-floor load fraction as a function of the ANALYTIC rim H.

At 32x24 the below-floor load fraction is bimodal (0 or >= 28 %) and switches at
rim H ~ 0.29-0.32 rather than at the fit floor 0.5: it records whether the FIRST
cell ring's centre lies below the floor, and that ring carries the squeeze peak.
A grid-level flag cannot be a registered criterion.  The rim H at the cap is
analytic -- h(theta_min) = EPS_CAP_MARGIN * c -- so H_rim = 0.01 c / sigma* needs
no grid.  This measures the resolved curve load_frac(H_rim) on finer grids so a
mask threshold can be read off it before the chart is computed.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(OUT / "analysis_E7_mask_curve.parquet"))
    args = ap.parse_args()
    from ball_socket.immutable import guarded_path
    guarded_path(args.out, what="mask-curve artefact")
    from run_E7_load_decomposition import X_COLS
    from run_E7_load_weighted_domain import one
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M as BAND

    sub = pd.read_parquet(OUT / "analysis_E7_averaged_film_subsample_32x24.parquet")
    # analytic rim H at the upper sigma* end; spread the picks evenly over 0.15-0.60
    sub["H_rim_analytic"] = 0.01 * sub.c0 / float(BAND[1])
    grid_H = np.linspace(0.15, 0.57, args.n)
    picks = []
    pool = sub.copy()
    for h in grid_H:
        i = (pool.H_rim_analytic - h).abs().idxmin()
        picks.append(pool.loc[i]); pool = pool.drop(i)
    picks = pd.DataFrame(picks)
    jobs = [(int(r.sample_id), {k: float(r[k]) for k in X_COLS}, g, ("hi",))
            for _, r in picks.iterrows() for g in ((64, 48), (96, 72))]
    print(f"{len(jobs)} jobs: {len(picks)} samples x 2 grids, rim H "
          f"{picks.H_rim_analytic.min():.3f}-{picks.H_rim_analytic.max():.3f}", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [row for batch in ex.map(one, jobs) for row in batch]
    res = pd.DataFrame(rows)
    res["H_rim_analytic"] = 0.01 * res.sample_id.map(sub.set_index("sample_id").c0) / res.sigma_m
    res.to_parquet(args.out, index=False)
    print(f"wrote {args.out} ({len(res)} rows, {(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
