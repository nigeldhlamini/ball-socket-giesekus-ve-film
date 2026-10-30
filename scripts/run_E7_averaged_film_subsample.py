"""R-37.2: does the averaged (Patir-Cheng) film change E7?  Measure before deciding.

Box C registered PC1978 Eq. 10 as the film model; the campaign solved smooth
Reynolds.  Before committing a two-week re-run, this solves the averaged film
AT THE CAP on a stratified subsample of no-equilibrium samples and reports

  * the film-load ratio averaged / smooth, per sigma* band end;
  * how many no-equilibrium samples (pure film) and arms (mixed) it flips;
  * WHERE the fit is extrapolated, twice: analytically at the rim, and on the
    cell centres the solver actually evaluates.

R-37.2's scaling estimate predicts a ratio of ~1.2-2.1 over H = 1.5-0.3 --
i.e. the averaged film carries MORE and the smooth result is conservative.
That is the prediction this run can falsify.

Below the fit floor (H <= 0.5, gamma = 1) there are two defensible treatments
and neither is picked here: both are solved.

Scope limits, stated rather than discovered later:
  * the film is solved at c = c0.  A row that failed on a later outer iteration
    (clearance grown) does not reproduce at c0; the smooth solve is compared
    with the stored film load and such samples are EXCLUDED and counted.
  * the flip test uses the stored asperity load at the cap, which does not
    depend on the film model (the gap is fixed there).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
MIXED = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
PURE = OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet"
MODES = ("extrapolate", "freeze")


def _sigma_band():
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M
    return tuple(float(s) for s in SIGMA_STAR_BAND_M)


def one(args):
    sid, x, F_film_stored, Ntheta, Nphi = args
    import warnings
    warnings.simplefilter("ignore")
    from run_E7_load_decomposition import _solver
    from ball_socket.spherical import Spherical, SphericalConditions
    from ball_socket.reynolds import solve_reynolds
    from ball_socket import mixed as mx

    slv, ci = _solver(x, Ntheta, Nphi)
    eps, c = float(slv.eps_cap), float(ci.c0)
    fluid = slv._fluid(ci)

    def film(ff):
        cond = SphericalConditions(
            p_ch=ci.p_ch, p_case=ci.p_case, e_vec=(0.0, 0.0, -eps * c),
            e_dot=(0.0, 0.0, -slv.squeeze_phase * ci.omega * eps * c),
            omega_b=(0.0, 0.0, slv.spin_ratio * ci.omega))
        sph = Spherical(slv.geom.R_b, c, slv.geom.theta_max, cond, slv.Ntheta,
                        slv.Nphi, theta_min=slv.geom.theta_min)
        sph.flow_factors = ff
        res = solve_reynolds(sph, fluid, slv.config, model=slv.model)
        cells_below = None
        if ff is not None:
            m = sph._flow_factor_fields()[2].astype(float)
            cells_below = float(sph.mesh.integrate(m)
                                / sph.mesh.integrate(np.ones(sph.shape)))
        return float(sph.load_capacity(res.pressure)), bool(res.converged), \
            cells_below, float(sph.h_min_boundary())

    out = {"sample_id": int(sid), **x, "eps_cap": eps,
           "F_target": float(slv._applied_load(ci) - slv._pocket_relief(ci)),
           "F_film_stored": F_film_stored}
    F0, ok0, _, hmin_rim = film(None)
    out.update(F_smooth=F0, smooth_converged=ok0, h_min_rim=hmin_rim,
               reproduces=bool(abs(F0 - F_film_stored) <= 1e-9 * abs(F_film_stored)))
    floor = mx.flow_factor_valid_above(1.0)
    for i, sig in enumerate(_sigma_band()):
        tag = "lo" if i == 0 else "hi"
        out[f"H_rim_{tag}"] = hmin_rim / sig
        out[f"rim_below_floor_{tag}"] = bool(hmin_rim / sig <= floor)
        for mode in MODES:
            F, ok, below, _ = film({"sigma": sig, "below_floor": mode})
            out[f"F_avg_{tag}_{mode}"] = F
            out[f"conv_{tag}_{mode}"] = ok
            out[f"cells_below_floor_{tag}"] = below
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-bin", type=int, default=50)
    ap.add_argument("--bins", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--grid", type=int, nargs=2, default=(32, 24))
    ap.add_argument("--out", default=str(
        OUT / "analysis_E7_averaged_film_subsample_32x24.parquet"))
    args = ap.parse_args()

    from ball_socket.immutable import guarded_path
    guarded_path(args.out, what="averaged-film subsample")
    from run_E7_load_decomposition import X_COLS

    pf = pd.read_parquet(PURE, columns=X_COLS + ["sample_id", "outcome"])
    noeq = pf[pf.outcome == "nonconverged"].copy()
    mx_ = pd.read_parquet(MIXED, columns=["sample_id", "outcome", "F_film_at_limit"])
    film_at = (mx_[mx_.outcome == "contact_limit"]
               .groupby("sample_id").F_film_at_limit.first())
    noeq = noeq[noeq.sample_id.isin(film_at.index)]
    # stratify on the analytic rim H at the upper sigma* end: 0.01 c0 / sigma
    noeq["H_rim_hi_analytic"] = 0.01 * noeq.c0 / _sigma_band()[1]
    noeq["bin"] = pd.qcut(noeq.H_rim_hi_analytic, args.bins, labels=False)
    pick = (noeq.groupby("bin", group_keys=False)
                .apply(lambda g: g.sample(min(args.n_per_bin, len(g)),
                                          random_state=args.seed)))
    jobs = [(int(r.sample_id), {k: float(getattr(r, k)) for k in X_COLS},
             float(film_at.loc[r.sample_id]), args.grid[0], args.grid[1])
            for r in pick.itertuples()]
    print(f"{len(jobs)} no-equilibrium samples in {args.bins} rim-H bins, "
          f"grid {tuple(args.grid)}, 5 film solves each", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one, jobs))
    res = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    res.to_parquet(args.out, index=False)
    print(f"wrote {args.out}  ({len(res)} rows, {(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
