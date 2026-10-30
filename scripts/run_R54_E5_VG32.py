"""R-54.7 / R-52.5: measure E5's g1-binding clause on the VG32 rated-corner rows.

THE OWED ITEM.  E5's registered claim (preregistration_FROZEN.md:126-129) is that
at the rated maximum corner the operating point lies outside the full-film
feasible set WITH g1 BINDING.  The VG46 half was STRUCK (ERR-25): its registered
inlet, 110 C, equals the mineral g2 ceiling, so any heating fails g2 and g1 can
never be the sole failure.  The VG32 half is adjudicable -- inlet 100 C, ten
kelvin of headroom -- and no box-D or box-E artefact carries T_max_bnd at those
rows, because the E10 run computed it and did not persist it (R-53.5, F-A14).

WHAT IS SOLVED.  E5's registered point set, restricted to the rows the clause is
about: grade VG32, the REGISTERED maximum corner, 10 swept nu_100 x 4 clearances
= 40 points, each over all 24 registered contact arms = 960 solves.  Same solver
configuration as the E10 corner run (box D: clearance closure, bulk datum, 32x24,
registered bracket), so the two are comparable row for row.

WHAT IT DECIDES, and what it cannot.
  * g3 cannot bind at E5's lambda = 1e-5 s: De_sq <= lambda*omega*eps_cap*cos(30)
    = 0.0047 < 0.2.  That is arithmetic, not a measurement, and it is asserted.
  * g1 fails on every equilibrium row at the corner (Lambda_max 1.877, R-52).
  * So the clause reduces to whether g2 holds: T_max_bnd <= 110 C.  THIS is what
    the run measures.
A row supports the clause when it has an equilibrium, fails g1, and satisfies g2
and g3 -- g1 the SOLE violated constraint.

Every registered-constraint quantity is persisted (R-53.5 convention), whether or
not this adjudication reads it.
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
OUT = ROOT / "results" / "ch9"
DEST = OUT / "E5_VG32_g1_binding_corner.parquet"
REPORT = OUT / "E5_VG32_g1_binding_corner.report.json"
GRID = (32, 24)
E5_THETA_MIN_DEG = 30.0          # E5's registered gap-fill (2026-07-27 ledger note)
GRADE = 32


def _points():
    from ball_socket import e5_fluid_spec as E5
    from ball_socket.campaign_config_v4 import E5_LAMBDA, E5_ETA_S_RATIO
    out = []
    for i, p in enumerate(E5.build_e5_points()):
        if p.grade != GRADE or not E5.CORNERS[p.corner]["registered"]:
            continue
        out.append(dict(point=i, corner=p.corner, grade=p.grade, nu_100=p.nu_100,
                        c0_um=p.c0_um, thick_arm=bool(p.thick_arm),
                        in_box=bool(E5.AXES["eta0"][0] <= p.eta0 <= E5.AXES["eta0"][1]),
                        x=dict(omega=p.omega_rad_s, p_ch=p.p_ch_bar * 1e5,
                               c0=p.c0_um * 1e-6, T_in=p.T_C, eta_0=p.eta0,
                               lam=E5_LAMBDA, eta_s_ratio=E5_ETA_S_RATIO,
                               theta_min_deg=E5_THETA_MIN_DEG)))
    return out


def job(args):
    pt, model_id = args
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.campaign as C
    import ball_socket.mixed as mx
    model, prov = mx.expand_for_solving(mx.registered_contact_models(
        branch="psi", summit_convention="derived"))[model_id]
    x = pt["x"]
    t0 = time.time()
    row = C.evaluate_x(x, contact=model, clearance_closure=True, delta_T_datum="bulk",
                       Ntheta=GRID[0], Nphi=GRID[1])
    rec = {k: v for k, v in pt.items() if k != "x"}
    rec.update({k: x[k] for k in x})
    rec.update(model_id=int(model_id), arm=prov, sigma_star=float(model.sigma_star),
               psi=float(model.psi()), psi_regime=model.psi_regime(),
               wall_s=time.time() - t0,
               realised_film_model=row["realised_film_model"],
               realised_eps_cap=row["realised_eps_cap"],
               realised_contact_model=row["realised_contact_model"],
               c_equilibrium=row.get("c_equilibrium"),
               eps_cap_at_failure=row.get("eps_cap_at_failure"))
    # R-53.5: persist every registered-constraint quantity, read or not.
    rec.update({k: row.get(k) for k in C.REGISTERED_CONSTRAINT_COLUMNS})
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    from ball_socket.immutable import guarded_path
    from ball_socket.campaign_config_v4 import ConstraintSet
    from ball_socket import e5_fluid_spec as E5
    import ball_socket.mixed as mx
    guarded_path(str(DEST), what="E5 VG32 g1-binding corner run")
    cs = ConstraintSet()
    ceiling = cs.T_max_C_by_class["mineral"]

    pts = _points()
    n_models = len(mx.expand_for_solving(mx.registered_contact_models(
        branch="psi", summit_convention="derived")))
    expect = len(E5.NU100_SWEEP) * len(E5.CLEARANCES_UM)
    if len(pts) != expect:
        raise SystemExit(f"point set is {len(pts)}, expected {expect} "
                         f"(VG32 x {len(E5.NU100_SWEEP)} nu_100 x "
                         f"{len(E5.CLEARANCES_UM)} c0 at the registered corner)")
    jobs = [(p, m) for p in pts for m in range(n_models)]
    print(f"E5 VG32: {len(pts)} points x {n_models} arms = {len(jobs)} solves",
          flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0, res = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            res.append(f.result())
            if i % 96 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {el / 60:.1f} min, "
                      f"~{el / i * (len(jobs) - i) / 60:.1f} min left", flush=True)
    # labels travel inside each record (R-53.1); sort for a deterministic hash
    df = pd.DataFrame(res).sort_values(["point", "model_id"]).reset_index(drop=True)
    df.to_parquet(DEST, index=False)

    eq = df[df.outcome == "converged"].copy()
    eq["Lambda"] = eq.h_min_bnd / eq.sigma_star
    g1 = eq.Lambda >= cs.lambda_ratio_min
    g2 = eq.T_max_bnd <= ceiling
    g3 = eq.De_sq_bnd <= cs.De_sq_max
    if not g3.all():
        raise SystemExit("g3 binds at E5's lambda, which the arithmetic excludes; "
                         "the adjudication's premise is wrong")
    sole_g1 = (~g1) & g2 & g3
    rep = {
        "clause": ("E5: at the rated maximum corner the operating point lies outside "
                   "the full-film feasible set WITH g1 BINDING "
                   "(preregistration_FROZEN.md:126-129)"),
        "population": (f"VG32, registered maximum corner, {len(pts)} points x "
                       f"{n_models} arms = {len(jobs)} solves"),
        "solver": {"grid": list(GRID), "clearance_closure": True,
                   "delta_T_datum": "bulk", "theta_min_deg": E5_THETA_MIN_DEG,
                   "film_models": sorted(set(df.realised_film_model))},
        "arithmetic_premises": {
            "g3_unreachable": {"De_sq_max_observed": float(eq.De_sq_bnd.max()),
                               "limit": cs.De_sq_max},
            "inlet_headroom_K": ceiling - E5.SERVICE_T[GRADE]},
        "outcomes": df.outcome.value_counts().to_dict(),
        "equilibrium_rows": int(len(eq)),
        "g1_satisfied_rows": int(g1.sum()),
        "g2_satisfied_rows": int(g2.sum()),
        "sole_g1_rows": int(sole_g1.sum()),
        "sole_g1_pct_of_equilibrium": float(100 * sole_g1.mean()) if len(eq) else None,
        "T_max_bnd": {"max": float(eq.T_max_bnd.max()), "median": float(eq.T_max_bnd.median()),
                      "rise_max_K": float((eq.T_max_bnd - eq.T_in).max())},
        "Lambda": {"max": float(eq.Lambda.max()), "median": float(eq.Lambda.median())},
        "c0_without_equilibrium_um": sorted(set(df.c0_um) - set(eq.c0_um)),
        "verdict": ("CONFIRMED for VG32: g1 is the sole violated constraint on every "
                    "equilibrium row" if len(eq) and sole_g1.all() else
                    "CONFIRMED in part" if sole_g1.any() else
                    "REFUTED for VG32: g1 is never the sole violated constraint"),
        "per_nu_100": [],
    }
    for nu, g in eq.groupby("nu_100"):
        s = (~(g.h_min_bnd / g.sigma_star >= cs.lambda_ratio_min)) & (g.T_max_bnd <= ceiling)
        rep["per_nu_100"].append({"nu_100": float(nu), "equilibrium_rows": int(len(g)),
                                  "sole_g1_rows": int(s.sum()),
                                  "T_max_bnd_max": float(g.T_max_bnd.max())})
    REPORT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_nu_100"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
