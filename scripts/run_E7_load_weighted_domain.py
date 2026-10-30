"""R-38.4: what fraction of the film LOAD comes from cells below Patir-Cheng's fit floor?

The previous measure counted cells ("cell centres below the floor, median 0 %").
Cells are not what validity depends on, and a cell count has a perverse grid
dependence: the out-of-fit strip at the rim is sub-cell at 32x24, so refining
the mesh puts MORE cells below the floor -- a cell-count criterion would make
the model more valid as the mesh got worse.  The load-weighted form is the
quantity that decides whether the extrapolation matters.

Per (sample, grid, sigma* end), on the averaged film at the registered cap:

  cell_frac        area fraction of the land with H <= floor
  load_frac        fraction of the averaged film load carried by those cells
  sq_load_frac     the same for the SQUEEZE-ONLY averaged film (boundaries at
                   zero gauge) -- the part the flow factor actually changes
  treatment_sens   |F_extrapolate - F_freeze| / F_extrapolate -- the direct
                   sensitivity of the load to how the out-of-fit strip is treated

The Dirichlet rim sits at POCKET pressure, not zero, so load_frac can be
non-trivial while the model choice is irrelevant there: pocket pressure is
imposed, not computed.  sq_load_frac and treatment_sens separate the two.

Grid study: the lowest-H samples on 32x24, 48x36, 64x48, 96x72.
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
SUB = OUT / "analysis_E7_averaged_film_subsample_32x24.parquet"
GRIDS = ((32, 24), (48, 36), (64, 48), (96, 72))


def one(args):
    sid, x, grid, tags = args
    import warnings
    warnings.simplefilter("ignore")
    from run_E7_load_decomposition import _solver
    from ball_socket.spherical import Spherical, SphericalConditions
    from ball_socket.reynolds import solve_reynolds
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M as BAND

    slv, ci = _solver(x, *grid)
    eps, c = float(slv.eps_cap), float(ci.c0)
    fluid = slv._fluid(ci)

    def solve(ff, *, hyd=True, spin=True):
        cond = SphericalConditions(
            p_ch=ci.p_ch if hyd else 0.0, p_case=ci.p_case if hyd else 0.0,
            e_vec=(0.0, 0.0, -eps * c),
            e_dot=(0.0, 0.0, -slv.squeeze_phase * ci.omega * eps * c),
            omega_b=(0.0, 0.0, slv.spin_ratio * ci.omega) if spin else (0.0, 0.0, 0.0))
        sph = Spherical(slv.geom.R_b, c, slv.geom.theta_max, cond, grid[0],
                        grid[1], theta_min=slv.geom.theta_min)
        sph.flow_factors = ff
        res = solve_reynolds(sph, fluid, slv.config, model=slv.model)
        nz, dA = sph.load_axis_factors()
        return sph, np.asarray(res.pressure, float) * nz * dA, bool(res.converged)

    rows = []
    _, q_smooth, ok0 = solve(None)
    for tag in tags:
        sig = float(BAND[0] if tag == "lo" else BAND[1])
        ffe = {"sigma": sig, "below_floor": "extrapolate"}
        sph, q_e, ok_e = solve(ffe)
        _, q_f, ok_f = solve({"sigma": sig, "below_floor": "freeze"})
        _, q_sq, ok_s = solve(ffe, hyd=False, spin=False)
        below = sph._flow_factor_fields()[2]
        dA = sph.load_axis_factors()[1]
        Fe, Ff = q_e.sum(), q_f.sum()
        rows.append({
            "sample_id": sid, "Ntheta": grid[0], "Nphi": grid[1], "sigma": tag,
            "cell_frac": float(dA[below].sum() / dA.sum()),
            "load_frac": float(q_e[below].sum() / Fe),
            "load_frac_smooth": float(q_smooth[below].sum() / q_smooth.sum()),
            "sq_load_frac": float(q_sq[below].sum() / q_sq.sum()),
            "treatment_sens": float(abs(Fe - Ff) / abs(Fe)),
            "F_smooth": float(q_smooth.sum()), "F_extrapolate": float(Fe),
            "F_freeze": float(Ff), "F_squeeze_avg": float(q_sq.sum()),
            "h_min_rim": float(sph.h_min_boundary()), "sigma_m": sig,
            "converged": bool(ok0 and ok_e and ok_f and ok_s),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-study-n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(OUT / "analysis_E7_load_weighted_domain.parquet"))
    args = ap.parse_args()
    from ball_socket.immutable import guarded_path
    guarded_path(args.out, what="load-weighted domain artefact")
    from run_E7_load_decomposition import X_COLS

    sub = pd.read_parquet(SUB)
    X = lambda r: {k: float(getattr(r, k)) for k in X_COLS}
    jobs = [(int(r.sample_id), X(r), (32, 24), ("lo", "hi")) for r in sub.itertuples()]
    low = sub.nsmallest(args.grid_study_n, "H_rim_hi")
    for g in GRIDS[1:]:
        jobs += [(int(r.sample_id), X(r), g, ("hi",)) for r in low.itertuples()]
    print(f"{len(jobs)} jobs: 300 samples at 32x24 (both sigma* ends) + "
          f"{len(low)} lowest-H samples at {GRIDS[1:]} (sigma* hi)", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [row for batch in ex.map(one, jobs) for row in batch]
    res = pd.DataFrame(rows)
    res["grid_study"] = res.sample_id.isin(low.sample_id)
    res.to_parquet(args.out, index=False)
    print(f"wrote {args.out}  ({len(res)} rows, {(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
