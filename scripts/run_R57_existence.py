"""R-56.3.3 / R-57.6: what the no-equilibrium population means, at the bracket floor.

The production bracket stops at h_min = 0.01 c (campaign_config_v6.py:1047, 1059).
For a stratified sample of box-E contact_limit rows (each failed at c = c0 on the
first outer call), compare at delta = 0.01:

  F_target   the hydrodynamic share the film (+asperities) must carry, F_app - F_pocket
             (stored: F_target_at_failure)
  F_N        the Newtonian, isothermal smooth film at eta_0 with the SOLVER's kinematics
             (eps_dot = omega eps, A5) plus the pocket/case hydrostatic superposition --
             the quadrature, not the solver
  F_solver   the solver's FILM load at the cap (stored: F_film_at_limit)

Classes (R-56.3.3):
  (B)   F_target > F_N        even a Newtonian film at eta_0 cannot carry the load
                              above 1 % of clearance;
  (A')  F_N >= F_target > F_solver   the Newtonian film would carry it; the couplings
                              (shear-thinning, memory) remove the margin -- confirmed
                              by re-solving the film at the cap with the Newtonian model;
  undetermined  |F_solver - F_target| / F_target below the production-mesh load error at
                delta = 0.01 for the row's theta_min (R1 machinery, signed, per row).
Registered: >= 90 % classified; (B) the majority; the (A') share rises across lambda
terciles.  STOP if > 10 % undetermined.

Strata: theta_min tercile x c0 tercile x sigma* end (contact arm < 12 = sigma lo).
fluid_class is never read by the solver (R-50.1), so every row stands for both classes.
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
SRC = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
DEST = OUT / "R57_existence_classification.parquet"
REPORT = OUT / "R57_existence_classification.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
PER_STRATUM = 4
SEED = 20261002
DELTA_FLOOR = 0.01


def sample() -> pd.DataFrame:
    cols = AXES + ["sample_id", "model_id", "outcome", "F_film_at_limit", "F_asp_at_limit",
                   "F_target_at_failure", "eps_cap_at_failure", "n_outer_at_failure"]
    E = pd.read_parquet(SRC, columns=cols)
    E = E[E.outcome == "contact_limit"].copy()
    # ERR-16(b): this artefact predates c_at_failure and n_outer_at_failure is NaN on
    # every contact_limit row.  Most failed at c = c0 on the first outer call; a few
    # failed after the clearance grew.  As in run_E7_load_decomposition (509/512), each
    # sampled row's film load at the cap is re-solved at c = c0 and the row is used only
    # if it REPRODUCES the stored F_film_at_limit to 1e-9; the rest are counted out.
    E["c_at_failure"] = E.c0
    E["t_tmin"] = pd.qcut(E.theta_min_deg, 3, labels=False)
    E["t_c0"] = pd.qcut(E.c0, 3, labels=False)
    E["sigma_end"] = np.where(E.model_id.astype(int) < 12, "lo", "hi")
    return (E.groupby(["t_tmin", "t_c0", "sigma_end"])
             .sample(n=PER_STRATUM, random_state=SEED).reset_index(drop=True))


def job(r: dict) -> dict:
    import warnings
    warnings.simplefilter("ignore")
    import p3_rim_asymptote as RA
    from run_R56_R1 import solve as newt, C as Cn, ETA, R_B, OMEGA
    from run_R57_mesh_error import _solver
    import ball_socket.campaign as C
    x = {k: float(r[k]) for k in AXES}
    tm = x["theta_min_deg"]
    slv, ci = _solver(x, None, None)
    R, c = slv.geom.R_b, float(r["c_at_failure"]) if np.isfinite(r["c_at_failure"]) else x["c0"]
    eps = (1 - DELTA_FLOOR) / np.cos(np.deg2rad(tm))
    assert abs(eps - float(r["eps_cap_at_failure"])) < 1e-12, "the cap is delta = 0.01"
    F_N = RA.load_solver_kinematics(DELTA_FLOOR, tm, eta=x["eta_0"], R=R, c=c, omega=x["omega"],
                                    p_pocket=x["p_ch"], p_case=C.P_CASE_PA)
    # production-mesh load error at the floor for this theta_min (Newtonian proxy, signed)
    Fn, en = newt(tm, DELTA_FLOOR, (32, 24))
    e_F = RA.solver_f(Fn, c=Cn, eta=ETA, R=R_B, omega=OMEGA, eps=en) / RA.f_recess(DELTA_FLOOR, tm) - 1
    # reproduction: the registered film (viscoelastic, smooth) at the cap at c = c0
    F_repro = slv._load_split(ci, eps, slv._fluid(ci), c)[0]
    # the solver's own film at the cap, Newtonian at eta_0 (shear-thinning and memory off)
    slv.model = "newtonian"
    F_N_solver = slv._load_split(ci, eps, slv._fluid(ci), c)[0]
    return {"F_N": float(F_N), "F_N_solver": float(F_N_solver), "e_F_floor": float(e_F),
            "repro_rel": float(F_repro / float(r["F_film_at_limit"]) - 1)}


def classify(d: pd.DataFrame) -> pd.Series:
    T, N, S = d.F_target_at_failure, d.F_N, d.F_film_at_limit
    undet = (S - T).abs() / T < d.e_F_floor.abs()
    cls = np.where(undet, "undetermined", np.where(T > N, "B", np.where(T > S, "A'", "unexplained")))
    return pd.Series(cls, index=d.index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-57 existence classification")
    S = sample()
    print(f"{len(S)} contact_limit rows sampled over "
          f"{S.groupby(['t_tmin', 't_c0', 'sigma_end']).ngroups} strata", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        got = list(ex.map(job, S.to_dict(orient="records")))
    D = pd.concat([S, pd.DataFrame(got)], axis=1)
    n_sampled = len(D)
    D["reproduces"] = D.repro_rel.abs() <= 1e-9
    excluded = D[~D.reproduces][["sample_id", "model_id", "repro_rel"]].to_dict(orient="records")
    D = D[D.reproduces].copy()
    D["cls"] = classify(D)
    D["N_over_T"] = D.F_N / D.F_target_at_failure
    D["S_over_T"] = D.F_film_at_limit / D.F_target_at_failure
    D["NS_solver_vs_quad"] = D.F_N_solver / D.F_N - 1
    D["lam_tercile"] = pd.qcut(D.lam, 3, labels=["low", "mid", "high"])
    D.to_parquet(DEST, index=False)
    share = D.cls.value_counts(normalize=True)
    ap_by_lam = D.groupby("lam_tercile", observed=True).cls.apply(lambda s: float((s == "A'").mean()))
    rep = {"rows_sampled": n_sampled, "excluded_not_reproducing_at_c0": excluded,
           "rows": int(len(D)), "wall_min": (time.time() - t0) / 60, "workers": a.workers,
           "class_counts": D.cls.value_counts().to_dict(),
           "class_shares": {k: float(v) for k, v in share.items()},
           "registered": {
               "classified_ge_90pct": bool(share.get("undetermined", 0) + share.get("unexplained", 0) <= 0.10),
               "B_majority": bool(share.get("B", 0) > 0.5),
               "Aprime_share_by_lambda_tercile": ap_by_lam.to_dict(),
               "Aprime_rises_across_lambda_terciles": bool(ap_by_lam.is_monotonic_increasing
                                                           and ap_by_lam.iloc[-1] > ap_by_lam.iloc[0]),
               "STOP_undetermined_gt_10pct": bool(share.get("undetermined", 0) > 0.10)},
           "F_N_over_F_target": {"median": float(D.N_over_T.median()), "min": float(D.N_over_T.min()),
                                 "max": float(D.N_over_T.max())},
           "F_solver_over_F_target": {"median": float(D.S_over_T.median()), "max": float(D.S_over_T.max())},
           "newtonian_solver_vs_quadrature_at_floor": {
               "median": float(D.NS_solver_vs_quad.median()),
               "max_abs": float(D.NS_solver_vs_quad.abs().max())},
           "Aprime_confirmed_by_newtonian_solver": int(((D.cls == "A'") & (D.F_N_solver >= D.F_target_at_failure)).sum()),
           "Aprime_rows": int((D.cls == "A'").sum())}
    guarded_path(str(REPORT), what="R-57 existence report")
    REPORT.write_text(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
