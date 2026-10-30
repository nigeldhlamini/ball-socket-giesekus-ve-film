"""R-55.2 R1 as restated by R-56.3.1: verification of the solver against the
vendored quadrature, Newtonian, isothermal, pure squeeze.

Configuration (asserted on the CONSTRUCTED objects, design rule 2 -- the fixture
asserts its realised configuration, not its arguments):
  model 'newtonian' (constant eta, so no shear-thinning and no memory); no
  asperities (no contact model on this path); pocket and case at ZERO gauge; spin
  ZERO; smooth film (flow_factors None); no thermal model (Reynolds solve only);
  c fixed.  e = -eps c z, e_dot = -omega eps c z, exactly as solver.py:232-233.

Grid: production 32 x 24 and two refinements (64 x 48, 128 x 96).
theta_min in {20, 30, 40} deg; delta in {0.3, 0.1, 0.03, 0.01} (R-56.3.1: the
range the production solver uses; the bracket top is delta = 0.01).

Registered (R-55.2 R1): error falls at the scheme's formal order, 2 (BS-V1/BS-V2,
gates.py:174, 188), observed order within +/-0.5, and |error| < 1 % at the finest
mesh for delta >= 0.01.  Failing: plateau/growth, or order outside [1.5, 2.5] -> STOP.
Added (R-56.3.1): the production-mesh load error at delta = 0.01, with sign, per
theta_min -- the existence analogue of xtol.
Positive control: a pocket gauge pressure sized so the quadrature predicts a load
change >= 5 % at EVERY test point; the solver WITH it, compared against the
UNPERTURBED quadrature, must FAIL the 1 % criterion at every point.

Also reported, for R-56.3.2's discriminator: the Newtonian solver's own early
exponent (delta 0.1 -> 0.03) at each mesh.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import p3_rim_asymptote as RA  # noqa: E402

OUT = ROOT / "results" / "ch9"
DEST = OUT / "R56_R1_verification.parquet"
REPORT = OUT / "R56_R1_verification.report.json"
THETAS = (20.0, 30.0, 40.0)
DELTAS = (0.3, 0.1, 0.03, 0.01)
GRIDS = ((32, 24), (64, 48), (128, 96))
R_B, C, ETA, OMEGA = 6.5e-3, 10e-6, 0.02, 300.0
FORMAL_ORDER = 2.0


def solve(theta_min_deg, delta, grid, p_pocket=0.0):
    from ball_socket.geometry import BallSocketGeometry
    from ball_socket.spherical import Spherical, SphericalConditions
    from ball_socket.reynolds import solve_reynolds
    from src.rheology import create_newtonian
    from src.solver import SolverConfig
    geom = BallSocketGeometry(c0=C, theta_min=np.deg2rad(theta_min_deg))
    tmin = geom.theta_min
    eps = (1.0 - delta) / np.cos(tmin)
    cond = SphericalConditions(p_ch=p_pocket, p_case=0.0, e_vec=(0.0, 0.0, -eps * C),
                               e_dot=(0.0, 0.0, -OMEGA * eps * C), omega_b=(0.0, 0.0, 0.0))
    sph = Spherical(R_B, C, geom.theta_max, cond, grid[0], grid[1], theta_min=tmin)
    fluid = create_newtonian(ETA)
    # ---- REALISED configuration, asserted on the constructed objects ----
    assert sph.flow_factors is None, "smooth film"
    assert sph.conditions.p_case == 0.0 and sph.conditions.p_ch == p_pocket
    assert not np.any(sph.conditions.omega_b) and not np.any(sph.conditions.omega_art), "spin off"
    assert fluid.is_newtonian, "constant viscosity"
    assert abs(float(sph.h_min_boundary()) / C - delta) < 1e-12, "rim film is delta c"
    assert np.isclose(sph.theta_max, np.deg2rad(73.0)), "theta_max 73 deg"
    res = solve_reynolds(sph, fluid, SolverConfig(linear_solver="direct"), model="newtonian")
    F = float(sph.load_capacity(res.pressure))
    return F, eps


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")
    from ball_socket.immutable import guarded_path
    if "--report-only" in sys.argv:          # rebuild the report from the persisted solves
        return report(pd.read_parquet(DEST), t0=time.time())
    guarded_path(str(DEST), what="R-56 R1 verification")
    # positive-control pocket pressure: >= 5 % predicted change at every point
    need = []
    for tm in THETAS:
        for d in DELTAS:
            eps = (1 - d) / np.cos(np.deg2rad(tm))
            Fsq = 12 * np.pi * ETA * R_B ** 4 * OMEGA * eps / C ** 2 * RA.f_recess(d, tm)
            need.append(0.05 * Fsq / (2 * np.pi * R_B ** 2 * RA.g_pocket(d, tm)))
    p_pk = 1.25 * max(need)
    rows, t0 = [], time.time()
    for tm in THETAS:
        for d in DELTAS:
            fq = RA.f_recess(d, tm)
            gq = RA.g_pocket(d, tm)
            for g in GRIDS:
                F, eps = solve(tm, d, g)
                Fp, _ = solve(tm, d, g, p_pocket=p_pk)
                f = RA.solver_f(F, c=C, eta=ETA, R=R_B, omega=OMEGA, eps=eps)
                fp = RA.solver_f(Fp, c=C, eta=ETA, R=R_B, omega=OMEGA, eps=eps)
                pred_change = (2 * np.pi * R_B ** 2 * p_pk * gq) / (
                    12 * np.pi * ETA * R_B ** 4 * OMEGA * eps / C ** 2 * fq)
                rows.append(dict(theta_min_deg=tm, delta=d, Ntheta=g[0], Nphi=g[1],
                                 f_solver=f, f_quad=fq, err=f / fq - 1,
                                 f_solver_pocket=fp, err_pocket_vs_unperturbed=fp / fq - 1,
                                 err_pocket_vs_superposed=fp / (fq + pred_change * fq) - 1,
                                 predicted_pocket_change=pred_change))
            print(f"theta_min {tm:.0f}  delta {d:<5}  err " +
                  "  ".join(f"{r['err']:+.3e}" for r in rows[-3:]), flush=True)
    D = pd.DataFrame(rows)
    D["p_pocket_Pa"] = p_pk
    D.to_parquet(DEST, index=False)
    return report(D, t0)


def report(D, t0) -> int:
    from ball_socket.immutable import guarded_path
    p_pk = float(D.p_pocket_Pa.iloc[0])
    out = {"config": {"R_b": R_B, "c": C, "eta": ETA, "omega": OMEGA, "p_pocket_control_Pa": p_pk,
                      "grids": [list(g) for g in GRIDS], "formal_order": FORMAL_ORDER},
           "wall_s": time.time() - t0, "points": []}
    ok_all = True
    for (tm, d), g in D.groupby(["theta_min_deg", "delta"]):
        g = g.sort_values("Ntheta")
        e = g.err.abs().values
        orders = [float(np.log2(e[i] / e[i + 1])) if e[i + 1] > 0 else float("nan") for i in range(2)]
        conv = all(e[i + 1] < e[i] for i in range(2))
        order_ok = all(abs(o - FORMAL_ORDER) <= 0.5 for o in orders)
        fine_ok = e[-1] < 0.01
        ctrl_fails = bool((g.err_pocket_vs_unperturbed.abs() >= 0.01).all())
        passed = conv and order_ok and fine_ok
        ok_all &= passed
        out["points"].append(dict(theta_min_deg=tm, delta=d, errors=[float(v) for v in g.err],
                                  observed_orders=orders, decreasing=conv, order_within=order_ok,
                                  finest_below_1pct=fine_ok, passed=passed,
                                  control_predicted_change=float(g.predicted_pocket_change.iloc[0]),
                                  control_err_vs_unperturbed=[float(v) for v in g.err_pocket_vs_unperturbed],
                                  control_fails_1pct_at_every_mesh=ctrl_fails,
                                  control_superposed_err_finest=float(g.err_pocket_vs_superposed.iloc[-1])))
    out["R1_passed"] = bool(ok_all)
    out["STOP"] = not ok_all
    out["control_armed"] = bool(all(p["control_fails_1pct_at_every_mesh"] for p in out["points"]))
    prod = D[(D.Ntheta == 32) & (D.delta == 0.01)]
    out["production_mesh_load_error_at_delta_0.01"] = {
        f"{tm:.0f}": float(e) for tm, e in zip(prod.theta_min_deg, prod.err)}
    ex = {}
    for (tm, n), g in D.groupby(["theta_min_deg", "Ntheta"]):
        s = g.set_index("delta").f_solver
        ex[f"{tm:.0f}|{n}"] = float(np.log(s[0.03] / s[0.1]) / np.log(0.3))
    out["newtonian_solver_early_exponent"] = ex
    out["quadrature_early_exponent"] = {f"{tm:.0f}": float(RA.local_exp(lambda x: RA.f_recess(x, tm), 0.1, 0.03))
                                        for tm in THETAS}
    guarded_path(str(REPORT), what="R-56 R1 report")
    REPORT.write_text(json.dumps(out, indent=2, default=_jsonable))
    print(json.dumps({k: v for k, v in out.items() if k != "points"}, indent=2, default=_jsonable))
    for p in out["points"]:
        print(p["theta_min_deg"], p["delta"], [f"{v:+.2e}" for v in p["errors"]],
              [f"{o:.2f}" for o in p["observed_orders"]], "PASS" if p["passed"] else "FAIL",
              "ctrl-armed" if p["control_fails_1pct_at_every_mesh"] else "CTRL-NOT-ARMED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
