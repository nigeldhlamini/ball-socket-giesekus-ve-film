"""R-57.1: production-mesh error carried to Lambda at every claim-bearing row.

Rows: the 32 band rows nearest g1, the E13 extremum, the E5 extremum (all from
R56_tolerance_resolve, at their TIGHT-tolerance equilibria).

Per row:
  e_F    load error of the production mesh (32 x 24), by the Newtonian proxy at the
         row's delta = h_min_bnd / c and theta_min: solver_f / f_quad - 1 (R1
         machinery).  A PROXY: it carries no shear-thinning, memory, pocket or
         asperity load -- said in App. C, and checked by the full-physics
         refinement of the margin row below.
  s      the SOLVER'S OWN coupled slope d ln F / d ln delta at the row, by central
         finite difference of the total load (film + asperity, exactly what the load
         balance uses) at delta (1 +/- 2 %), at the row's realised clearance c.
         Not the quadrature's slope (R-57.1).
  dLam   Lambda_true - Lambda_solver = Lambda * e_F / s   (signed).
         Derivation: the solver's load is F (1 + e_F); balance at W gives
         F(delta_true) = W, F(delta_hat) = W / (1 + e_F), so
         ln delta_true - ln delta_hat = ln(1 + e_F) / s ~ e_F / s.
  repro  F_total at the stored (eps, c) over the row's F_target -- the
         construction check: the rebuilt solver IS the row's solver.

Also: R1's Newtonian check at delta = 0.5 (R-57.1: measured, not extrapolated), and
the margin row's full-physics refinement, 32x24 / 64x48 / 128x96, tight tolerance.
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
DEST = OUT / "R57_mesh_error.parquet"
REPORT = OUT / "R57_mesh_error.report.json"
GRID = (32, 24)
LAMBDA_MIN = 3.0
FD = 0.02


def _solver(x, contact, flow_factors, grid=GRID):
    """Constructed exactly as campaign._evaluate_x_inner constructs it."""
    import ball_socket.campaign as C
    from ball_socket.geometry import BallSocketGeometry
    from ball_socket.solver import BallSocketSolver, CampaignInput
    from src.solver import SolverConfig
    geom = BallSocketGeometry(c0=x["c0"], theta_min=np.deg2rad(x["theta_min_deg"]))
    slv = BallSocketSolver(
        geom, model="viscoelastic", Ntheta=grid[0], Nphi=grid[1],
        alpha=C.ALPHA_FIXED, squeeze_phase=C.SQUEEZE_PHASE, spin_ratio=C.SPIN_RATIO,
        config=SolverConfig(include_normal_stress=False, max_iter=80, tol=1e-9, omega=0.7),
        eps_cap=float(C._registered_eps_cap(geom.theta_min)),
        clearance_closure=True, delta_T_datum="bulk", contact=contact)
    slv.flow_factors = None if flow_factors is None else dict(flow_factors)
    ci = CampaignInput(omega=x["omega"], p_ch=x["p_ch"], c0=x["c0"], T_in=x["T_in"],
                       eta_0=x["eta_0"], lam=x["lam"], eta_s_over_eta0=x["eta_s_ratio"],
                       k_F_bs=None, p_case=C.P_CASE_PA)
    return slv, ci


def _contact(model_id):
    import ball_socket.mixed as mx
    if model_id < 0:
        return None
    return mx.expand_for_solving(mx.registered_contact_models(
        branch="psi", summit_convention="derived"))[model_id][0]


def row_job(c):
    import warnings
    warnings.simplefilter("ignore")
    import p3_rim_asymptote as RA
    from run_R56_R1 import solve as newt, C as Cn, ETA, R_B, OMEGA
    slv, ci = _solver(c["x"], _contact(c["model_id"]), c["flow_factors"])
    fluid = slv._fluid(ci)
    cc, eps = c["c_tight"], c["eps_tight"]
    tmin = slv.geom.theta_min
    delta = 1.0 - eps * np.cos(tmin)
    F_tgt = slv._applied_load(ci) - slv._pocket_relief(ci)
    F0 = slv._load_at(ci, eps, fluid, cc)

    def F_at(d):
        return slv._load_at(ci, (1.0 - d) / np.cos(tmin), fluid, cc)
    Fp, Fm = F_at(delta * (1 + FD)), F_at(delta * (1 - FD))
    s = float(np.log(Fp / Fm) / np.log((1 + FD) / (1 - FD)))
    Fn, en = newt(float(np.rad2deg(tmin)), float(delta), GRID)
    e_F = RA.solver_f(Fn, c=Cn, eta=ETA, R=R_B, omega=OMEGA, eps=en) / RA.f_recess(float(delta), float(np.rad2deg(tmin))) - 1
    Lam = delta * cc / c["sigma"]
    return dict(case=c["case"], src_index=c["src_index"], theta_min_deg=float(np.rad2deg(tmin)),
                c0_um=1e6 * c["x"]["c0"], delta=float(delta), Lam_tight=float(Lam),
                Lam_tight_stored=float(c["Lam_tight"]), repro=float(F0 / F_tgt - 1),
                slope=s, e_F=float(e_F), dLam_mesh=float(Lam * e_F / s),
                dist_to_3=float(abs(Lam - LAMBDA_MIN)))


def refine_job(args):
    c, grid = args
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.campaign as C
    kw = dict(contact=_contact(c["model_id"]), clearance_closure=True, delta_T_datum="bulk",
              Ntheta=grid[0], Nphi=grid[1], brentq_xtol=1e-10, brentq_rtol=1e-10)
    if c["flow_factors"] is not None:
        kw["flow_factors"] = c["flow_factors"]
    from ball_socket.campaign import REGISTERED_CONSTRAINT_COLUMNS
    t0 = time.time()
    row = C.evaluate_x(c["x"], **kw)
    # R-53.5: every registered constraint quantity is kept, read or not.
    return dict(grid=f"{grid[0]}x{grid[1]}", Ntheta=grid[0],
                **{k: row.get(k) for k in REGISTERED_CONSTRAINT_COLUMNS},
                c_equilibrium=row.get("c_equilibrium"),
                Lam=float(row.get("h_min_bnd")) / c["sigma"], wall_s=time.time() - t0)


REFINE = OUT / "R57_mesh_error_margin_refine"     # one JSON per grid, written on completion


def _cases():
    import run_R56_tolerance as T
    tight = json.loads((OUT / "R56_tolerance_resolve.report.json").read_text())["per_case"]
    R = pd.read_parquet(OUT / "R56_tolerance_resolve.parquet")
    R = R[R.tol != "default"].set_index(["case", "src_index"])
    tl = {(p["case"], p["src_index"]): p for p in tight}
    cases = []
    for c in T.cases():
        k = (c["case"], c["src_index"])
        r = R.loc[k]
        cases.append(dict(c, c_tight=float(r.c_equilibrium), eps_tight=float(r.eps_ecc),
                          Lam_tight=float(tl[k]["Lam_tight"])))
    margin = min((c for c in cases if c["case"] == "band"), key=lambda c: c["margin"])
    return cases, margin


def stage_rows(workers):
    """Persisted on completion: the 34 claim rows (cheap, ~1 solve-set each)."""
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-57.1 mesh error at claim rows")
    cases, _ = _cases()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(row_job, cases))
    pd.DataFrame(rows).to_parquet(DEST, index=False)
    print("rows written", DEST, flush=True)


def stage_refine(workers):
    """The margin row at three grids; each grid's result written as it completes."""
    _, margin = _cases()
    REFINE.mkdir(exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    grids = [g for g in ((32, 24), (64, 48), (128, 96))
             if not (REFINE / f"{g[0]}x{g[1]}.json").exists()]
    with ProcessPoolExecutor(max_workers=min(workers, len(grids) or 1)) as ex:
        futs = {ex.submit(refine_job, (margin, g)): g for g in grids}
        for f in as_completed(futs):
            g = futs[f]
            (REFINE / f"{g[0]}x{g[1]}.json").write_text(json.dumps(f.result(), default=float))
            print("refine done", g, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage", choices=["rows", "refine", "report"], required=True)
    a = ap.parse_args()
    import warnings
    warnings.simplefilter("ignore")
    if a.stage == "rows":
        stage_rows(a.workers)
        return 0
    if a.stage == "refine":
        stage_refine(a.workers)
        return 0
    from ball_socket.immutable import guarded_path
    t0 = time.time()
    cases, margin = _cases()
    D = pd.read_parquet(DEST)
    ref = [json.loads((REFINE / f"{g}.json").read_text()) for g in ("32x24", "64x48", "128x96")]
    # R1 at delta = 0.5 (Newtonian, three meshes) -- measured, not extrapolated
    import p3_rim_asymptote as RA
    from run_R56_R1 import solve as newt, C as Cn, ETA, R_B, OMEGA
    r1 = {}
    for tm in (20.0, 30.0, 40.0):
        errs = []
        for g in ((32, 24), (64, 48), (128, 96)):
            F, e = newt(tm, 0.5, g)
            errs.append(RA.solver_f(F, c=Cn, eta=ETA, R=R_B, omega=OMEGA, eps=e) / RA.f_recess(0.5, tm) - 1)
        r1[f"{tm:.0f}"] = {"errors": errs, "orders": [float(np.log2(abs(errs[i]) / abs(errs[i + 1]))) for i in range(2)]}

    band = D[D.case == "band"]
    m = D[(D.case == "band") & (D.src_index == margin["src_index"])].iloc[0]
    Rf = pd.DataFrame(ref).sort_values("Ntheta")
    L = Rf.Lam.values
    rich = float(L[-1] + (L[-1] - L[-2]) / 3.0)                    # 2nd-order Richardson
    rep = {
        "rows": int(len(D)), "workers": a.workers, "wall_min": (time.time() - t0) / 60,
        "construction_check_max_rel": float(D.repro.abs().max()),
        "E13": D[D.case == "E13"].iloc[0].to_dict(), "E5": D[D.case == "E5"].iloc[0].to_dict(),
        "registered": {
            "E13_dLam_below_0.001": bool(abs(D[D.case == "E13"].dLam_mesh.iloc[0]) < 1e-3),
            "E5_dLam_below_0.001": bool(abs(D[D.case == "E5"].dLam_mesh.iloc[0]) < 1e-3),
            "margin_row_dLam_below_0.01": bool(abs(m.dLam_mesh) < 0.01)},
        "margin_row": {**m.to_dict(), "margin_tight": float(abs(m.Lam_tight / LAMBDA_MIN - 1)),
                       "full_physics_refinement": ref, "richardson_Lambda": rich,
                       "full_physics_dLam_32x24": float(rich - L[0])},
        "band_32": {"mesh_unresolved": int((band.dLam_mesh.abs() > band.dist_to_3).sum()),
                    "mesh_unresolved_rows": band[band.dLam_mesh.abs() > band.dist_to_3][
                        ["src_index", "c0_um", "theta_min_deg", "delta", "Lam_tight", "dLam_mesh", "dist_to_3"]
                    ].to_dict(orient="records"),
                    "max_abs_dLam_mesh": float(band.dLam_mesh.abs().max())},
        "R1_delta_0.5": r1,
        "note": ("e_F is a Newtonian proxy (squeeze only, no shear-thinning, memory, pocket or "
                 "asperity); the slope is the solver's own coupled slope at the row"),
    }
    guarded_path(str(REPORT), what="R-57.1 report")
    REPORT.write_text(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("E13", "E5")}, indent=1,
                     default=lambda o: o.item() if hasattr(o, "item") else str(o))[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
