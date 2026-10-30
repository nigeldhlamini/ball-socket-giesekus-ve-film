"""E10: capped-area fraction at the rated corner (box C sec 5, box D/F carried).

REGISTERED (preregistration_v5 E10): *psi exceeds 1 over the majority of the
Greenwood-Tripp band, so the elastic contact model requires a plastic cap, and
the land-area fraction at the cap at the rated corner is GREATER THAN ZERO.*
Reported as the capped-area fraction at the rated corner, PER BAND END.  The
first clause was measured at box C (psi > 1 over 78.3 % of the band, 0 %
clearly elastic); this script measures the second.

DEFINITIONS, fixed here before the run:

* RATED CORNER -- E5's registered public corner, "maximum" (500 bar, 4500 rpm),
  on E5's registered point set: ISO VG 32/46 at their service temperatures,
  nu_100 swept 4.5-9.0 mm^2/s (10 values), c0 in {2, 5, 15, 30} um, E5 fluid
  triplet, theta_min at the axis midpoint (E5's documented gap-fill).  80 points.
  E5's "nominal" corner (450 bar, 4300 rpm) is solved too and labelled
  EXPLORATORY, as E5 labels it.
* PROCEDURE -- the registered campaign path: evaluate_x, contact on (all 24
  registered arms), thermal closure on (bulk datum), 32x24, the registered eps_cap,
  and the registered SMOOTH film (box F sec 3.1).
* OPERATING STATE -- converged: (eps*, c_equilibrium).  contact_limit: the cap
  at the clearance where the limit was reached (eps_cap_at_failure,
  c_at_failure).  Any other outcome carries no state and is counted, not used.
* CAPPED -- a land cell is capped where GT's elastic asperity pressure exceeds
  GT's plastic-branch pressure at that cell (the R-36.5 definition, box E
  mechanism hand-back sec 3): the plastic branch is the model's cap (GT Eq. 13;
  it replaced a hard min(p, H_ind), see mixed.asperity_pressure_plastic).
* CAPPED-AREA FRACTION -- land-area-weighted: integral of the capped indicator
  over the land divided by the land area, through the mesh's own quadrature.
  This is the REGISTERED metric.  It is set by H and psi alone, so it counts a
  cell as capped however small its asperity load (smoke, 2026-09-19: c0 = 2 um,
  psi = 3.06 -> the whole land capped while real contact covers 4e-4 of it).
* SUPPLEMENTARY, not the registered metric: CAPPED-CONTACT FRACTION -- the
  share of REAL contact area (GT Eq. 8, A_c/A_n) lying in capped cells.  Carried
  so the registered number is not read as a load statement.

Both band ends of every GT parameter are carried as separate rows (rule 15);
the verdict is reported per sigma* end and per psi regime, never pooled.
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
DEST = OUT / "E10_capped_area_rated_corner.parquet"
GRID = (32, 24)
E5_THETA_MIN_DEG = 30.0


def _points():
    from ball_socket import e5_fluid_spec as E5
    from ball_socket.campaign_config_v4 import E5_LAMBDA, E5_ETA_S_RATIO
    out = []
    for i, p in enumerate(E5.build_e5_points()):
        out.append(dict(point=i, corner=p.corner, grade=p.grade, nu_100=p.nu_100,
                        c0_um=p.c0_um, thick_arm=bool(p.thick_arm),
                        in_box=bool(E5.AXES["eta0"][0] <= p.eta0 <= E5.AXES["eta0"][1]),
                        registered=bool(E5.CORNERS[p.corner]["registered"]),
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
    from ball_socket.geometry import BallSocketGeometry
    from ball_socket.spherical import Spherical, SphericalConditions
    model, prov = mx.expand_for_solving(mx.registered_contact_models(
        branch="psi", summit_convention="derived"))[model_id]
    x = pt["x"]
    row = C.evaluate_x(x, contact=model, clearance_closure=True, delta_T_datum="bulk",
                       Ntheta=GRID[0], Nphi=GRID[1])
    rec = {k: v for k, v in pt.items() if k != "x"}
    rec.update({k: x[k] for k in x})
    rec.update(model_id=int(model_id), arm=prov, sigma_star=float(model.sigma_star),
               psi=float(model.psi()), psi_regime=model.psi_regime(),
               outcome=row["outcome"], realised_film_model=row["realised_film_model"],
               realised_eps_cap=row["realised_eps_cap"],
               realised_contact_model=row["realised_contact_model"])
    # R-53.5: persist EVERY registered-constraint quantity, not only the ones
    # E10 reads.  The first version kept E10's own columns, so T_max_bnd was
    # computed at all 3,840 rated-corner rows and discarded -- which is why E5's
    # VG32 clause (ERR-25) now costs 960 solves to recover it.  The set comes
    # from campaign.REGISTERED_CONSTRAINT_COLUMNS, never hand-listed here.
    rec.update({k: row.get(k) for k in C.REGISTERED_CONSTRAINT_COLUMNS})
    if row["outcome"] == "converged":
        eps, c, state = float(row["eps_ecc"]), float(row["c_equilibrium"]), "equilibrium"
    elif row["outcome"] == "contact_limit":
        eps, c, state = float(row["eps_cap_at_failure"]), float(row["c_at_failure"]), "cap"
    else:
        rec.update(state="none", capped_area_frac=np.nan)
        return rec
    geom = BallSocketGeometry(c0=x["c0"], theta_min=np.deg2rad(x["theta_min_deg"]))
    cond = SphericalConditions(p_ch=x["p_ch"], p_case=1e5, e_vec=(0.0, 0.0, -eps * c),
                               e_dot=(0.0, 0.0, 0.0), omega_b=(0.0, 0.0, 0.0))
    sph = Spherical(geom.R_b, c, geom.theta_max, cond, GRID[0], GRID[1],
                    theta_min=geom.theta_min)
    h = sph.gap()
    H = model.gap_ratio(h)
    p_el = mx.asperity_pressure_elastic(H, model.eta_beta_sigma, mx._E_STAR(),
                                        model.sigma_s_over_beta_a)
    p_pl = mx.asperity_pressure_plastic(H, model.eta_beta_sigma, model.H_ind)
    capped = (np.asarray(p_el) > np.asarray(p_pl)).astype(float)
    area = float(sph.mesh.integrate(np.ones(sph.shape)))
    Ac = np.asarray(model.area_fraction(h), dtype=float)
    rec.update(state=state, eps=eps, c=c, h_min=float(h.min()),
               capped_area_frac=float(sph.mesh.integrate(capped) / area),
               capped_contact_frac=float(sph.mesh.integrate(capped * Ac)
                                         / sph.mesh.integrate(Ac)),
               peak_contact_area=float(np.max(model.area_fraction(h))),
               mean_contact_area=float(sph.mesh.integrate(model.area_fraction(h)) / area))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="E10 capped-area fraction")
    import ball_socket.mixed as mx
    n_models = len(mx.expand_for_solving(mx.registered_contact_models(
        branch="psi", summit_convention="derived")))
    pts = _points()
    jobs = [(p, m) for p in pts for m in range(n_models)]
    print(f"E10: {len(pts)} E5 points x {n_models} arms = {len(jobs)} solves", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0, res = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            res.append(f.result())
            if i % 192 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {el / 60:.1f} min, "
                      f"~{el / i * (len(jobs) - i) / 60:.1f} min left", flush=True)
    df = pd.DataFrame(res).sort_values(["point", "model_id"]).reset_index(drop=True)
    df.to_parquet(DEST, index=False)

    rep = {"n_solves": int(len(df)),
           "outcomes_by_corner": {c: g.outcome.value_counts().to_dict()
                                  for c, g in df.groupby("corner")},
           "realised_film_models": df.realised_film_model.unique().tolist(),
           "realised_eps_cap_range": [float(df.realised_eps_cap.min()),
                                      float(df.realised_eps_cap.max())]}
    tab = {}
    for (corner, s, reg), g in df[df.state != "none"].groupby(
            ["corner", "sigma_star", "psi_regime"]):
        v = g.capped_area_frac
        tab[f"{corner}|sigma*={s:.4g}|{reg}"] = {
            "n": int(len(v)), "n_positive": int((v > 0).sum()),
            "median": float(v.median()), "min": float(v.min()), "max": float(v.max()),
            "capped_contact_frac_median": float(g.capped_contact_frac.median()),
            "by_state": {k: int(n) for k, n in g.state.value_counts().items()}}
    rep["capped_area_fraction"] = tab
    reg = df[(df.corner == "maximum") & (df.state != "none")]
    rep["E10_clause_2"] = {
        "statement": "capped-area fraction at the rated corner > 0, per band end",
        "per_sigma_end": {f"{s:.4g}": {"rows_with_state": int(len(g)),
                                       "rows_positive": int((g.capped_area_frac > 0).sum()),
                                       "median": float(g.capped_area_frac.median())}
                          for s, g in reg.groupby("sigma_star")}}
    path = DEST.with_suffix(".report.json")
    guarded_path(str(path), what="E10 report")
    path.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
