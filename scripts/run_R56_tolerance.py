"""R-56.1: root-finder resolution, re-registered on the variable the solver uses.

The equilibrium search is brentq ON ECCENTRICITY eps with xtol = rtol = 1e-4
(ball_socket/solver.py:136-137, 421-426).  Per row the registered resolution is

    dLambda_row = c_row cos(theta_min) (xtol + rtol eps_row) / sigma*,

c_row the clearance at which the row was solved (closure on: c_equilibrium).

Cases (each re-solved TWICE through the identical evaluate_x call that produced it):
  * band  -- the 32 converged solves of the R-41 feasibility-band test nearest
             g1 (smallest |Lambda/3 - 1|); the smallest is the 0.31 % macro;
  * E13   -- the extremum row (Lambda_max 2.393);
  * E5    -- the VG32 rated-corner extremum row (Lambda_max 1.877).
  default tolerance  -> REPRODUCTION check (gate-5 spirit: the re-solve must be
                        the artefact row before its tight twin is believed);
  1e-10 / 1e-10      -> the resolution study box D's "resolution-study provision"
                        allows, value recorded per row (realised_brentq_*).

Registered (R-56.1), evaluated in the report:
  1. margin row: dLambda_row < 0.0093 => resolved, and |tight - stored| < dLambda_row;
     else the macro is printed no finer than dLambda_row.
  2. E13, E5: |tight - stored| < own dLambda_row; 2 in the third decimal is a STOP
     for the printed precision, not the verdict.
  3. the 32 band rows: no g1 flip under the tight re-solve; any flip is a STOP.
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
DEST = OUT / "R56_tolerance_resolve.parquet"
REPORT = OUT / "R56_tolerance_resolve.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
GRID = (32, 24)
XTOL0 = RTOL0 = 1e-4
TIGHT = 1e-10
LAMBDA_MIN = 3.0


def cases(n_band: int = 32) -> list[dict]:
    out = []
    B = pd.read_parquet(OUT / "boxF_R41_sigma_band_test.parquet")
    B = B[B.outcome == "converged"].copy()
    B["Lam"] = B.h_min_bnd.astype(float) / B.sigma
    B["margin"] = (B.Lam / LAMBDA_MIN - 1).abs()
    for r in B.nsmallest(n_band, "margin").itertuples():
        out.append(dict(case="band", src_index=int(r.Index), x={k: float(getattr(r, k)) for k in AXES},
                        model_id=int(r.model_id), sigma=float(r.sigma),
                        flow_factors={"sigma": float(r.sigma), "below_floor": r.treatment},
                        stored_h=float(r.h_min_bnd), stored_eps=float(r.eps_ecc),
                        stored_c=float(r.c_equilibrium), margin=float(r.margin)))
    for name, fn, scol in (("E13", "E13_extremum.parquet", "sigma_star"),
                           ("E5", "E5_VG32_g1_binding_corner.parquet", "sigma_star")):
        D = pd.read_parquet(OUT / fn)
        D = D[D.outcome == "converged"].copy()
        D["Lam"] = D.h_min_bnd.astype(float) / D[scol]
        r = D.loc[D.Lam.idxmax()]
        out.append(dict(case=name, src_index=int(r.name), x={k: float(r[k]) for k in AXES},
                        model_id=int(r.model_id), sigma=float(r[scol]), flow_factors=None,
                        stored_h=float(r.h_min_bnd), stored_eps=float(r.eps_ecc),
                        stored_c=float(r.c_equilibrium), margin=float(abs(r.Lam / LAMBDA_MIN - 1))))
    return out


def job(args):
    c, tol = args
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.campaign as C
    import ball_socket.mixed as mx
    # model_id -1 is the PURE-FILM arm (run_boxF_R41_bands.py:143-146): contact None.
    # Indexing the arm list with -1 would silently take the last contact arm.
    model = None
    if c["model_id"] >= 0:
        model = mx.expand_for_solving(mx.registered_contact_models(
            branch="psi", summit_convention="derived"))[c["model_id"]][0]
    kw = dict(contact=model, clearance_closure=True, delta_T_datum="bulk",
              Ntheta=GRID[0], Nphi=GRID[1])
    if c["flow_factors"] is not None:
        kw["flow_factors"] = c["flow_factors"]
    if tol is not None:
        kw.update(brentq_xtol=tol, brentq_rtol=tol)
    t0 = time.time()
    row = C.evaluate_x(c["x"], **kw)
    from ball_socket.campaign import REGISTERED_CONSTRAINT_COLUMNS
    # R-53.5: every registered constraint quantity is persisted, read or not.
    return {"case": c["case"], "src_index": c["src_index"], "tol": "default" if tol is None else str(tol),
            **{k: row.get(k) for k in REGISTERED_CONSTRAINT_COLUMNS},
            "c_equilibrium": row.get("c_equilibrium"), "T_max": row.get("T_max"),
            "realised_brentq_xtol": row.get("realised_brentq_xtol"),
            "realised_brentq_rtol": row.get("realised_brentq_rtol"),
            "realised_film_model": row.get("realised_film_model"),
            "wall_s": time.time() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-56.1 tolerance re-solve")
    cs = cases()
    jobs = [(c, t) for c in cs for t in (None, TIGHT)]
    print(f"{len(cs)} cases, {len(jobs)} solves, {a.workers} workers", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        got = list(ex.map(job, jobs))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    R = pd.DataFrame(got)
    R.to_parquet(DEST, index=False)          # per-solve rows: the artefact (gate-compliant)
    meta =pd.DataFrame([{k: v for k, v in c.items() if k not in ("x", "flow_factors")}
                         | {k: v for k, v in c["x"].items()} for c in cs])
    D = R[R.tol == "default"].merge(R[R.tol != "default"], on=["case", "src_index"],
                                     suffixes=("_def", "_tight")).merge(meta, on=["case", "src_index"])
    th = np.deg2rad(D.theta_min_deg)
    D["Lam_stored"] = D.stored_h / D.sigma
    D["Lam_def"] = D.h_min_bnd_def.astype(float) / D.sigma
    D["Lam_tight"] = D.h_min_bnd_tight.astype(float) / D.sigma
    D["dLam_row"] = D.stored_c * np.cos(th) * (XTOL0 + RTOL0 * D.stored_eps) / D.sigma
    D["repro_rel"] = (D.Lam_def / D.Lam_stored - 1).abs()
    D["move"] = (D.Lam_tight - D.Lam_stored).abs()
    D["g1_flip"] = (D.Lam_stored >= LAMBDA_MIN) != (D.Lam_tight >= LAMBDA_MIN)
    D["third_decimal_change"] = (np.round(D.Lam_tight, 3) - np.round(D.Lam_stored, 3)).abs() * 1000

    band = D[D.case == "band"]
    m = band.loc[band.margin.idxmin()]
    rep = {"what": "R-56.1 tolerance re-solve; xtol = rtol = 1e-10 on eps vs registered 1e-4",
           "solves": int(len(R)), "workers": a.workers,
           "reproduction_max_rel": float(D.repro_rel.max()),
           "margin_row": {"c0_um": 1e6 * float(m.c0), "theta_min_deg": float(m.theta_min_deg),
                          "sigma_um": 1e6 * float(m.sigma), "Lambda_stored": float(m.Lam_stored),
                          "margin_pct": 100 * float(m.margin), "margin_Lambda": float(m.margin * 3),
                          "dLam_row": float(m.dLam_row), "move": float(m.move),
                          "resolved": bool(m.dLam_row < m.margin * 3),
                          "registered_holds": bool((m.dLam_row < m.margin * 3 and m.move < m.dLam_row)
                                                   or m.dLam_row >= m.margin * 3)},
           "extrema": {r.case: {"Lambda_stored": float(r.Lam_stored), "Lambda_tight": float(r.Lam_tight),
                                "move": float(r.move), "dLam_row": float(r.dLam_row),
                                "within_own_resolution": bool(r.move < r.dLam_row),
                                "third_decimal_change": int(round(r.third_decimal_change)),
                                "STOP_printed_precision": bool(r.third_decimal_change >= 2)}
                       for r in D[D.case != "band"].itertuples()},
           "band_32": {"g1_flips": int(band.g1_flip.sum()), "STOP": bool(band.g1_flip.any()),
                       "max_move": float(band.move.max()), "max_dLam_row": float(band.dLam_row.max()),
                       "moves_within_own_resolution": int((band.move < band.dLam_row).sum())},
           "outcomes_tight": R[R.tol != "default"].outcome.value_counts().to_dict(),
           "per_case": D[["case", "src_index", "c0", "theta_min_deg", "sigma", "margin",
                          "Lam_stored", "Lam_def", "Lam_tight", "dLam_row", "repro_rel",
                          "move", "g1_flip"]].to_dict(orient="records"),
           "note": ("tightened tolerance is box D's registered resolution-study provision; "
                    "gate 4 compares realised brentq against the DEFAULT and is not applied to "
                    "this artefact; realised values recorded per row")}
    guarded_path(str(REPORT), what="R-56.1 tolerance report")
    REPORT.write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps(rep, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
