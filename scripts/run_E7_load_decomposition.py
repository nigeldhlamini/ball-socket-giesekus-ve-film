"""R-36.4: why is the film's load ceiling finite at h_min = 0.01 c?

At the registered cap the median deficit is ~3.1 kN against ~5.1 kN applied, so
most of the load is unbalanced with the film at one per cent of clearance.  A
lubrication reader expects h^-3 divergence.  This decomposes the film load AT
THE CAP, on contact_limit samples, into its sources:

    hydrostatic  -- the Dirichlet boundary pressures alone (p_ch at the inner
                    rim, p_case at the mouth), no motion
    squeeze      -- e_dot alone, boundaries at zero gauge
    spin/wedge   -- omega_b alone, boundaries at zero gauge

and checks three things a reader would ask before believing the split:

  * REPRODUCTION -- the full solve here equals the F_film_at_limit the campaign
    stored for that sample.  Otherwise this is a decomposition of something
    else.
  * SUPERPOSITION -- F_full - (F_hyd + F_sq + F_spin).  The operator is not
    exactly linear (shear-thinning viscosity, squeeze memory), so the residual
    is reported rather than assumed away.
  * SCALING -- F_sq and F_full at eps = (1 - delta)/cos(theta_min) for delta
    from 0.1 to 0.001, i.e. h_min = delta c at the inner rim.  The exponent is
    what answers "why not h^-3".

Every contact_limit row was evaluated at c = c0: the clearance closure's first
outer call raises ContactLimit before any clearance update.  So this reproduces
the campaign's film exactly, and the film does not depend on the contact model,
so one solve per SAMPLE covers all of its arms.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "ch9"

MIXED = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
X_COLS = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio",
          "theta_min_deg"]
DELTAS = (0.1, 0.03, 0.01, 0.003, 0.001)


def _solver(x: dict, Ntheta: int, Nphi: int):
    """Constructed exactly as campaign._evaluate_x_inner does, contact=None."""
    from ball_socket.campaign_config_v4 import (ALPHA_FIXED, SQUEEZE_PHASE,
                                                SPIN_RATIO, P_CASE_PA)
    from ball_socket.campaign_config_v6 import eps_cap
    from ball_socket.geometry import BallSocketGeometry
    from ball_socket.solver import BallSocketSolver, CampaignInput, SolverConfig

    geom = BallSocketGeometry(c0=x["c0"], theta_min=np.deg2rad(x["theta_min_deg"]))
    slv = BallSocketSolver(
        geom, model="viscoelastic", Ntheta=Ntheta, Nphi=Nphi,
        alpha=ALPHA_FIXED, squeeze_phase=SQUEEZE_PHASE, spin_ratio=SPIN_RATIO,
        config=SolverConfig(include_normal_stress=False, max_iter=80, tol=1e-9,
                            omega=0.7),
        eps_cap=float(eps_cap(geom.theta_min)), clearance_closure=True,
        delta_T_datum="bulk", contact=None)
    ci = CampaignInput(omega=x["omega"], p_ch=x["p_ch"], c0=x["c0"],
                       T_in=x["T_in"], eta_0=x["eta_0"], lam=x["lam"],
                       eta_s_over_eta0=x["eta_s_ratio"], k_F_bs=None,
                       p_case=P_CASE_PA)
    return slv, ci


def _film_load(slv, ci, eps: float, *, hyd=True, sq=True, spin=True):
    """Film load at eps with a chosen subset of sources switched on.

    Mirrors BallSocketSolver._forward exactly, with sources zeroed on request.
    """
    from ball_socket.spherical import Spherical, SphericalConditions
    from ball_socket.reynolds import solve_reynolds

    c = float(ci.c0)
    e = (0.0, 0.0, -eps * c)
    e_dot = (0.0, 0.0, -slv.squeeze_phase * ci.omega * eps * c)
    omega_b = (0.0, 0.0, slv.spin_ratio * ci.omega)
    cond = SphericalConditions(
        p_ch=ci.p_ch if hyd else 0.0, p_case=ci.p_case if hyd else 0.0,
        e_vec=e, e_dot=e_dot if sq else (0.0, 0.0, 0.0),
        omega_b=omega_b if spin else (0.0, 0.0, 0.0))
    sph = Spherical(slv.geom.R_b, c, slv.geom.theta_max, cond,
                    slv.Ntheta, slv.Nphi, theta_min=slv.geom.theta_min)
    res = solve_reynolds(sph, slv._fluid(ci), slv.config, model=slv.model)
    return float(sph.load_capacity(res.pressure)), bool(res.converged)


def one(args):
    sid, x, F_film_stored, F_target_stored, Ntheta, Nphi = args
    import warnings
    warnings.simplefilter("ignore")
    t0 = time.time()
    slv, ci = _solver(x, Ntheta, Nphi)
    eps = float(slv.eps_cap)
    ceiling = 1.0 / np.cos(slv.geom.theta_min)
    out = {"sample_id": int(sid), **{k: float(v) for k, v in x.items()},
           "eps_cap": eps, "F_film_stored": F_film_stored,
           "F_target": float(slv._applied_load(ci) - slv._pocket_relief(ci)),
           "F_target_stored": F_target_stored,
           "F_app": float(slv._applied_load(ci)),
           "F_pocket": float(slv._pocket_relief(ci))}
    F_all, ok_all = _film_load(slv, ci, eps)
    F_hyd, ok_h = _film_load(slv, ci, eps, sq=False, spin=False)
    F_sq, ok_s = _film_load(slv, ci, eps, hyd=False, spin=False)
    F_sp, ok_w = _film_load(slv, ci, eps, hyd=False, sq=False)
    out.update(F_full=F_all, F_hydrostatic=F_hyd, F_squeeze=F_sq, F_spin=F_sp,
               superposition_residual=F_all - (F_hyd + F_sq + F_sp),
               all_converged=bool(ok_all and ok_h and ok_s and ok_w))
    for d in DELTAS:
        e_d = (1.0 - d) * ceiling
        out[f"F_sq_d{d}"], _ = _film_load(slv, ci, e_d, hyd=False, spin=False)
        out[f"F_full_d{d}"], _ = _film_load(slv, ci, e_d)
    out["wall_s"] = time.time() - t0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--grid", type=int, nargs=2, default=(32, 24))
    ap.add_argument("--out", default=str(OUT / "analysis_E7_load_decomposition_32x24.parquet"))
    args = ap.parse_args()

    from ball_socket.immutable import guarded_path
    guarded_path(args.out, what="E7 decomposition artefact")

    cols = X_COLS + ["sample_id", "outcome", "F_film_at_limit", "F_target_at_failure"]
    df = pd.read_parquet(MIXED, columns=cols)
    cl = (df[df.outcome == "contact_limit"]
          .drop_duplicates("sample_id").sort_values("sample_id"))
    pick = cl.sample(n=min(args.n, len(cl)), random_state=args.seed)
    jobs = [(r.sample_id, {k: float(getattr(r, k)) for k in X_COLS},
             float(r.F_film_at_limit), float(r.F_target_at_failure),
             args.grid[0], args.grid[1]) for r in pick.itertuples()]
    print(f"{len(jobs)} contact_limit samples of {cl.sample_id.nunique()}, "
          f"grid {tuple(args.grid)}, {1 + 3 + 2 * len(DELTAS)} solves each",
          flush=True)

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
