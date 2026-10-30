"""R-55.2 R3 (run under R-57.7 step 7): the theta_min comparison and its control.

(a) Pure squeeze, Newtonian, isothermal (R1 machinery, production mesh 32 x 24 and
    128 x 96): f(40 deg)/f(20 deg) at common psi = eps cos(theta_min) in
    {0.1, 0.5, 0.7, 0.9}, i.e. delta = 1 - psi.  f is the load per unit squeeze rate
    (solver_f), so the different eps at common psi does not enter the ratio.
    Registered (R-55.2 R3): solver vs quadrature within R1's tolerance (1 %); and the
    quadrature ratio / projected-area ratio lies in 0.29-0.36 over that psi range.
(b) The thick-film control (\\PThreeFilmAreaOpen = 1.05: c0 = 10 um, psi = 0.1, E13's
    corner fluid, 40 vs 20 deg), decomposed exactly as run_R48_film_at_common_eps
    built it (closure off, smooth film, no contact, 32 x 24) into hydrostatic (pocket
    and case only), squeeze only and spin only, through run_E7_load_decomposition's
    _film_load.  Registered: hydrostatic carries MORE THAN HALF of F_film there.
    Failing: less than half -> the review's explanation is wrong and the control
    stands as a squeeze comparison.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
REPORT = OUT / "R57_R3.report.json"
PSI = (0.1, 0.5, 0.7, 0.9)


def part_a():
    import p3_rim_asymptote as RA
    from run_R56_R1 import solve, C, ETA, R_B, OMEGA
    rows = []
    for psi in PSI:
        d = 1.0 - psi
        fq = {t: RA.f_recess(d, t) for t in (20.0, 40.0)}
        rq = fq[40.0] / fq[20.0]
        ar = RA.area_proj(40.0) / RA.area_proj(20.0)
        rec = dict(psi=psi, delta=d, quad_ratio=rq, quad_ratio_over_proj_area=rq / ar)
        for g in ((32, 24), (128, 96)):
            fs = {}
            for t in (20.0, 40.0):
                F, eps = solve(t, d, g)
                fs[t] = RA.solver_f(F, c=C, eta=ETA, R=R_B, omega=OMEGA, eps=eps)
            rs = fs[40.0] / fs[20.0]
            rec[f"solver_ratio_{g[0]}"] = rs
            rec[f"rel_err_{g[0]}"] = rs / rq - 1
        rows.append(rec)
    return {"rows": rows,
            "registered_solver_within_1pct_32": bool(all(abs(r["rel_err_32"]) < 0.01 for r in rows)),
            "registered_solver_within_1pct_128": bool(all(abs(r["rel_err_128"]) < 0.01 for r in rows)),
            "registered_quad_over_proj_area_in_0.29_0.36": bool(all(0.29 <= r["quad_ratio_over_proj_area"] <= 0.36 for r in rows))}


def part_b():
    import ball_socket.campaign as Cm
    import run_E13_extremum as E13
    import run_E7_load_decomposition as E7
    from ball_socket.geometry import BallSocketGeometry
    from ball_socket.solver import BallSocketSolver, CampaignInput
    from src.solver import SolverConfig
    spec, _ = E13._load_spec()
    f = spec["fluid"]
    c0, psi = 10e-6, 0.1
    out = {}
    for th in (20.0, 40.0):
        geom = BallSocketGeometry(c0=c0, theta_min=np.deg2rad(th))
        slv = BallSocketSolver(geom, model="viscoelastic", Ntheta=32, Nphi=24, alpha=Cm.ALPHA_FIXED,
                               squeeze_phase=Cm.SQUEEZE_PHASE, spin_ratio=Cm.SPIN_RATIO,
                               config=SolverConfig(include_normal_stress=False, max_iter=80,
                                                   tol=1e-9, omega=0.7),
                               eps_cap=float(Cm._registered_eps_cap(geom.theta_min)), contact=None)
        ci = CampaignInput(omega=spec["corner"]["omega_rad_s"], p_ch=spec["corner"]["p_ch_Pa"], c0=c0,
                           T_in=f["T_in_C"], eta_0=f["eta_0_Pa_s"], lam=f["lam_s"],
                           eta_s_over_eta0=f["eta_s_ratio"])   # p_case default, as run_R48 built it
        eps = psi / np.cos(geom.theta_min)
        full, _ = E7._film_load(slv, ci, eps)
        hyd, _ = E7._film_load(slv, ci, eps, sq=False, spin=False)
        sq, _ = E7._film_load(slv, ci, eps, hyd=False, spin=False)
        sp, _ = E7._film_load(slv, ci, eps, hyd=False, sq=False)
        A_proj = np.pi * geom.R_b ** 2 * (np.sin(geom.theta_max) ** 2 - np.sin(geom.theta_min) ** 2)
        A_sph = 2 * np.pi * geom.R_b ** 2 * (np.cos(geom.theta_min) - np.cos(geom.theta_max))
        out[f"{th:.0f}"] = dict(F_full=full, F_hyd=hyd, F_sq=sq, F_spin=sp,
                                hyd_share=hyd / full, sq_share=sq / full,
                                superposition_residual=(full - hyd - sq - sp) / full,
                                A_proj=A_proj, A_sph=A_sph)
    a, b = out["40"], out["20"]
    out["ratio_full_over_sph_area"] = (a["F_full"] / b["F_full"]) / (a["A_sph"] / b["A_sph"])
    out["ratio_full_over_proj_area"] = (a["F_full"] / b["F_full"]) / (a["A_proj"] / b["A_proj"])
    out["ratio_hyd_over_proj_area"] = (a["F_hyd"] / b["F_hyd"]) / (a["A_proj"] / b["A_proj"])
    out["ratio_sq_over_proj_area"] = (a["F_sq"] / b["F_sq"]) / (a["A_proj"] / b["A_proj"])
    out["registered_hydrostatic_more_than_half"] = bool(a["hyd_share"] > 0.5 and b["hyd_share"] > 0.5)
    return out


def main() -> int:
    import warnings
    warnings.simplefilter("ignore")
    from ball_socket.immutable import guarded_path
    guarded_path(str(REPORT), what="R-57 R3")
    rep = {"a_theta_min_ratio_pure_squeeze": part_a(), "b_thick_film_control": part_b()}
    REPORT.write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps(rep, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
