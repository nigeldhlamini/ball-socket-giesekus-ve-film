"""Box G / E15 (R-55.7, R-58.4): E13 mirrored with ONLY the c0 axis changed.

Stages
  spec    build the box-G spec from E13_spec_v2.json, change the c0 axis, add E15's
          registration, write results/ch9/boxG_E15_spec.json + .sha256 (frozen).  No solve.
  run     REFUSES unless ball_socket.signatures carries an attestation for box G bound to
          that sha256 (R-58.4.1: "freeze it, sign it ... before any solve").  Then:
            overlap  E13's 2 um row, all 24 arms, memory_sign="modulus", E13's exact call:
                     must reproduce E13's stored h_min_bnd to 1e-9 (like-for-like check;
                     modulus because E13 was solved before D-6);
            grid     13 c0 x 24 arms at theta_min = 20, memory_sign="signed" (post-D-6).
  report  E15's three registrations; the end-point uncertainties use R-57.1's machinery
          (Newtonian-proxy e_F x the solver's coupled slope) plus brentq's bound.
"""
from __future__ import annotations

import argparse
import hashlib
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
E13_SPEC = OUT / "E13_spec_v2.json"
SPEC = OUT / "boxG_E15_spec.json"
DEST = OUT / "boxG_E15.parquet"
OVERLAP = OUT / "boxG_E15_overlap.parquet"
REPORT = OUT / "boxG_E15.report.json"
CKPT = OUT / "boxG_E15_grid_ckpt"
C0_UM = [round(1.0 + 0.25 * i, 2) for i in range(13)]          # 1.00 ... 4.00
THETA_MIN = 20.0


def build_spec() -> dict:
    s = json.loads(E13_SPEC.read_text(encoding="utf-8"))
    parent = hashlib.sha256(E13_SPEC.read_bytes()).hexdigest()
    s["expectation"] = "E15"
    s["box"] = "G"
    s["mirror_of"] = {"file": E13_SPEC.name, "sha256": parent,
                      "changed": "axes.c0_search_um and axes.c0_m ONLY (R-58.4.1); "
                                 "theta_min fixed at 20 deg (E13's endpoint mode, its extremum)"}
    s["axes"]["c0_search_um"] = C0_UM
    s["axes"]["c0_m"] = [C0_UM[0] * 1e-6, C0_UM[-1] * 1e-6]
    s["axes"]["theta_min_deg"] = [THETA_MIN]
    s["memory_sign"] = {"grid": "signed (D-6 fixed, R-58.1)",
                        "overlap": "modulus (E13 was solved before D-6; like-for-like check)"}
    s["overlap_row"] = {"c0_um": 2.0, "reproduce": "E13_extremum.parquet h_min_bnd, every arm",
                        "tolerance_rel": 1e-9}
    s["sigma_star"] = ("both ends (all 24 arms, as E13): E13's h_min varies across arms by up to "
                       "9e-4 relative at sigma hi, so contact enters its equilibrium (R-58.4.2)")
    s["prediction_stated_before_any_box_G_solve"] = {
        "source": "results/ch9/R57_E15_prediction.json (quadrature + thermal closure, A5)",
        "c0_star_um": 2.26, "Lambda_star_sigma_lo": 2.446}
    s["registered"] = {
        "R1_location": "argmax over the grid (max over arms) in [1.76, 2.76] um",
        "R2_value": "Lambda* (Richardson-corrected, R-58.2) in [2.250, 2.495] (-8 % / +2 % of 2.446), sigma lo",
        "R3_resolved": ("Lambda at c0 = 1.0 and 4.0 um each below Lambda* by more than 3x the combined "
                        "mesh (R-57.1 proxy) and root-finder (xtol + rtol eps) uncertainty at that point")}
    s["outcomes"] = {"confirmed": "R1, R2 and R3 all hold",
                     "refuted": "argmax outside [1.76, 2.76] um, or the maximum on a grid end",
                     "unresolved": "R3 fails: the peak is flatter than the uncertainty -> quote the plateau"}
    s["gate6_reachability"] = {
        "confirm": "quadrature predicts an interior peak at 2.26 um, Lambda(1.0) 18 % below",
        "refute": "a monotone solver response, or a peak outside the window, is reachable: the "
                  "solver's Lambda falls 2-4 % below the quadrature, an error that could move the peak",
        "unresolved": "reachable if the solver's curve is flatter than ~0.2 % near the ends"}
    s["rule18_fork"] = ("R-58.4: box G may reach below E13's 2 um floor; E13's verdict and region are "
                        "untouched; the hypothesis came from analysis; prediction fixed before any solve")
    return s


def stage_spec():
    from ball_socket.immutable import guarded_path
    guarded_path(str(SPEC), what="box G spec")
    txt = json.dumps(build_spec(), indent=2, sort_keys=True, default=float)
    SPEC.write_text(txt, encoding="utf-8")
    h = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    SPEC.with_suffix(".sha256").write_text(h + "\n")
    print("box G spec frozen, sha256", h)


def _require_signature() -> str:
    h = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    if h != SPEC.with_suffix(".sha256").read_text().strip():
        raise SystemExit("box G spec changed after hashing")
    from ball_socket import signatures as S
    a = S.ATTESTATIONS.get("G")
    if a is None or a.get("frozen_sha256") != h:
        raise SystemExit("box G is not signed: no attestation in ball_socket/signatures.py bound "
                         f"to spec sha256 {h} (R-58.4.1).  No solve is run.")
    return h


def job(args):
    x, model_id, sign = args
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.campaign as C
    from ball_socket.campaign import REGISTERED_CONSTRAINT_COLUMNS
    import ball_socket.mixed as mx
    model = mx.expand_for_solving(mx.registered_contact_models(
        branch="psi", summit_convention="derived"))[model_id][0]
    t0 = time.time()
    row = C.evaluate_x(x, contact=model, clearance_closure=True, delta_T_datum="bulk",
                       Ntheta=32, Nphi=24, memory_sign=sign)
    return {"model_id": model_id, "sigma_star": float(model.sigma_star), "memory_sign": sign,
            **{k: row.get(k) for k in REGISTERED_CONSTRAINT_COLUMNS},
            "c_equilibrium": row.get("c_equilibrium"),
            "realised_memory_sign": row.get("realised_memory_sign"),
            "wall_s": time.time() - t0, **x}


def stage_run(workers):
    h = _require_signature()
    # R-59.2.4/5: refuse from a dirty tree; the commit goes into RUN provenance (the parquet
    # columns below), never into the hashed spec, so the spec and its signature stand.
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("box G run")
    import run_E13_extremum as E13
    from ball_socket.immutable import guarded_path
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    guarded_path(str(DEST), what="box G grid")
    from concurrent.futures import ProcessPoolExecutor
    # RESTARTABLE (R-62, after the first grid run was killed with its process tree at ~17 h and
    # nothing persisted): the overlap is reused when it exists and verifies, and the grid is
    # checkpointed per clearance, so a kill loses at most one clearance's 24 solves.
    CKPT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        if OVERLAP.exists():
            ov = pd.read_parquet(OVERLAP)
            assert (ov.spec_sha256 == h).all() and len(ov) == 24 and set(ov.memory_sign) == {"modulus"}, \
                "existing overlap artefact does not verify; move it aside and rerun"
            print("overlap reused:", OVERLAP.name, "at", ov.git_commit.iloc[0][:10])
        else:
            guarded_path(str(OVERLAP), what="box G overlap")
            ov = pd.DataFrame(list(ex.map(job, [(E13._x_at(spec, THETA_MIN, 2e-6), m, "modulus")
                                                for m in range(24)])))
            ov["spec_sha256"] = h
            ov["git_commit"] = prov["commit"]
            ov.to_parquet(OVERLAP, index=False)
        for c in C0_UM:
            part = CKPT / f"c0_{c:.2f}um.parquet"
            if part.exists():
                q = pd.read_parquet(part)
                assert (q.spec_sha256 == h).all() and len(q) == 24, f"checkpoint {part.name} does not verify"
                continue
            q = pd.DataFrame(list(ex.map(job, [(E13._x_at(spec, THETA_MIN, c * 1e-6), m, "signed")
                                               for m in range(24)])))
            q["spec_sha256"] = h
            q["git_commit"] = prov["commit"]
            tmp = part.with_suffix(".tmp")
            q.to_parquet(tmp, index=False)
            tmp.replace(part)
            print(f"c0 = {c:.2f} um done", flush=True)
    g = pd.concat([pd.read_parquet(CKPT / f"c0_{c:.2f}um.parquet") for c in C0_UM], ignore_index=True)
    g.to_parquet(DEST, index=False)
    print("box G run done", len(g), "solves + 24 overlap")


RICH = OUT / "boxG_E15_richardson.parquet"
PROXY = OUT / "boxG_E15_mesh_proxy.parquet"
GRIDS = ((32, 24), (64, 48), (128, 96))
FD = 0.02
BRENTQ_TOL = 1e-4          # the production root-finder's xtol = rtol on eps (R-56)


def rich_job(args):
    """The argmax row on the SIGNED path at three meshes, tight root-finder (R-58.2's call)."""
    x, model_id, grid = args
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.campaign as C
    from ball_socket.campaign import REGISTERED_CONSTRAINT_COLUMNS
    import ball_socket.mixed as mx
    model = mx.expand_for_solving(mx.registered_contact_models(
        branch="psi", summit_convention="derived"))[model_id][0]
    t0 = time.time()
    row = C.evaluate_x(x, contact=model, clearance_closure=True, delta_T_datum="bulk",
                       Ntheta=grid[0], Nphi=grid[1], brentq_xtol=1e-10, brentq_rtol=1e-10,
                       memory_sign="signed")
    return {"model_id": model_id, "Ntheta": grid[0], "sigma_star": float(model.sigma_star),
            **{k: row.get(k) for k in REGISTERED_CONSTRAINT_COLUMNS},
            "c_equilibrium": row.get("c_equilibrium"), "realised_memory_sign": row.get("realised_memory_sign"),
            "Lam": float(row["h_min_bnd"]) / float(model.sigma_star), "wall_s": time.time() - t0, **x}


def proxy_job(args):
    """R-57.1's mesh-error proxy at one grid row, on the SIGNED path: dLam = Lam e_F / s, with
    e_F the Newtonian solver-vs-quadrature load error at the row's (delta, theta_min) and s the
    solver's own coupled slope d ln F / d ln delta (central difference, delta +/- 2 %)."""
    x, model_id, c_eq, h_min, sigma = args
    import warnings
    warnings.simplefilter("ignore")
    import p3_rim_asymptote as RA
    from run_R56_R1 import solve as newt, C as Cn, ETA, R_B, OMEGA
    from run_R57_mesh_error import _solver, _contact, GRID
    slv, ci = _solver(x, _contact(model_id), None)
    slv.memory_sign = "signed"                       # R-59.3: explicit; box G's grid is signed
    fluid = slv._fluid(ci)
    tmin = slv.geom.theta_min
    delta = h_min / c_eq
    F_at = lambda d: slv._load_at(ci, (1.0 - d) / np.cos(tmin), fluid, c_eq)
    s = float(np.log(F_at(delta * (1 + FD)) / F_at(delta * (1 - FD))) / np.log((1 + FD) / (1 - FD)))
    Fn, en = newt(float(np.rad2deg(tmin)), float(delta), GRID)
    e_F = RA.solver_f(Fn, c=Cn, eta=ETA, R=R_B, omega=OMEGA, eps=en) / RA.f_recess(float(delta),
                                                                                   float(np.rad2deg(tmin))) - 1
    lam = h_min / sigma
    eps = (1.0 - delta) / np.cos(tmin)
    d_eps = BRENTQ_TOL + BRENTQ_TOL * eps            # xtol + rtol * eps
    return {"c0_um": round(1e6 * x["c0"], 2), "model_id": model_id, "delta": float(delta), "Lam": float(lam),
            "slope": s, "e_F": float(e_F), "dLam_mesh": float(lam * e_F / s),
            "dLam_root": float(lam * np.cos(tmin) * d_eps / delta)}


def stage_report(workers):
    """E15's three registrations (R-58.4.3), from the persisted grid.  Solves only what the
    registrations need: the argmax row at three meshes (Richardson, R2) and the mesh proxy at
    the two grid ends (R3).  Each step persists before the next."""
    h = _require_signature()
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("box G report")
    import run_E13_extremum as E13
    from ball_socket.immutable import guarded_path
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    G = pd.read_parquet(DEST)
    assert (G.spec_sha256 == h).all() and set(G.realised_memory_sign) == {"signed"}
    G["c0_um"] = (G.c0 * 1e6).round(2)
    G["Lam"] = G.h_min_bnd / G.sigma_star
    lo = G[np.isclose(G.sigma_star, SIGMA_STAR_BAND_M[0]) & (G.outcome == "converged")]
    hi = G[np.isclose(G.sigma_star, SIGMA_STAR_BAND_M[1]) & (G.outcome == "converged")]
    curve = lo.loc[lo.groupby("c0_um").Lam.idxmax()].set_index("c0_um").sort_index()
    curve_hi = hi.groupby("c0_um").Lam.max()
    top_c0 = float(curve.Lam.idxmax())
    top = curve.loc[top_c0]
    from concurrent.futures import ProcessPoolExecutor
    if not RICH.exists():
        guarded_path(str(RICH), what="box G Richardson")
        with ProcessPoolExecutor(max_workers=min(workers, 3)) as ex:
            R = pd.DataFrame(list(ex.map(rich_job, [(E13._x_at(spec, THETA_MIN, top_c0 * 1e-6),
                                                     int(top.model_id), g) for g in GRIDS])))
        R["git_commit"] = prov["commit"]
        R.to_parquet(RICH, index=False)
    R = pd.read_parquet(RICH).sort_values("Ntheta")
    L1, L2, L3 = R.Lam.values
    LR = L3 + (L3 - L2) / 3.0
    p = float(np.log2((L1 - L2) / (L2 - L3))) if (L1 - L2) * (L2 - L3) > 0 else float("nan")
    LRp = L3 + (L3 - L2) / (2 ** p - 1) if np.isfinite(p) else float("nan")
    unc = max(abs(LR - L3), abs(LRp - LR) if np.isfinite(LRp) else 0.0)
    ends = [C0_UM[0], 4.0]
    if not PROXY.exists():
        guarded_path(str(PROXY), what="box G mesh proxy")
        args = [(E13._x_at(spec, THETA_MIN, c * 1e-6), int(curve.loc[c].model_id), float(curve.loc[c].c_equilibrium),
                 float(curve.loc[c].h_min_bnd), float(curve.loc[c].sigma_star)) for c in ends + [top_c0]]
        with ProcessPoolExecutor(max_workers=min(workers, 3)) as ex:
            Pq = pd.DataFrame(list(ex.map(proxy_job, args)))
        Pq["git_commit"] = prov["commit"]
        Pq.to_parquet(PROXY, index=False)
    Pq = pd.read_parquet(PROXY).set_index("c0_um")
    r1 = 1.76 <= top_c0 <= 2.76 and top_c0 not in (C0_UM[0], C0_UM[-1])
    r2 = 2.250 <= LR <= 2.495
    r3_pts = {}
    for c in ends:
        u = abs(Pq.loc[c].dLam_mesh) + abs(Pq.loc[c].dLam_root)
        gap_prod = float(top.Lam - curve.loc[c].Lam)
        r3_pts[f"{c:g}um"] = {"Lambda": float(curve.loc[c].Lam), "gap_to_Lambda_star_production": gap_prod,
                              "gap_to_Lambda_star_richardson": float(LR - curve.loc[c].Lam),
                              "dLam_mesh": float(Pq.loc[c].dLam_mesh), "dLam_root": float(Pq.loc[c].dLam_root),
                              "combined_uncertainty": float(u), "gap_over_uncertainty": gap_prod / u,
                              "holds": bool(gap_prod > 3 * u)}
    r3 = all(v["holds"] for v in r3_pts.values())
    outcome = ("confirmed" if (r1 and r2 and r3) else
               "refuted" if not r1 else "unresolved" if not r3 else "R2 failed (value outside its window)")
    rep = {"provenance": prov, "spec_sha256": h, "attested_by": "box G, ball_socket.signatures['G']",
           "grid_solves": int(len(G)), "outcomes": G.outcome.value_counts().to_dict(),
           "curve_sigma_lo_max_over_arms": {f"{c:g}": float(v) for c, v in curve.Lam.items()},
           "curve_sigma_hi_max_over_arms": {f"{c:g}": float(v) for c, v in curve_hi.items()},
           "argmax": {"c0_um": top_c0, "arm": int(top.model_id), "Lambda_production": float(top.Lam)},
           "richardson_at_argmax": {"Lambda_32": L1, "Lambda_64": L2, "Lambda_128": L3, "Lambda_R": LR,
                                    "p_obs": p, "uncertainty": unc,
                                    "argmax_proxy_dLam_mesh": float(Pq.loc[top_c0].dLam_mesh)},
           "R1_location": {"holds": bool(r1), "window_um": [1.76, 2.76], "argmax_um": top_c0},
           "R2_value": {"holds": bool(r2), "window": [2.250, 2.495], "Lambda_star_R": LR},
           "R3_resolved": {"holds": bool(r3), "points": r3_pts},
           "prediction_compared": spec["prediction_stated_before_any_box_G_solve"],
           "outcome": outcome}
    path = OUT / "boxG_E15.report.json"
    guarded_path(str(path), what="box G report")
    path.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("argmax", "richardson_at_argmax", "R1_location", "R2_value",
                                          "R3_resolved", "outcome")}, indent=2, default=float))


def stage_amend():
    """R-63.3: add the film-only sigma* scaling check at the optimum to the EXISTING report,
    leaving every other key unchanged:  Lambda*(sigma* hi) against Lambda*(sigma* lo) x
    sigma*_lo / sigma*_hi.  If the film alone carries the load at the optimum, h_min does not
    depend on the roughness arm and the two agree; a departure measures the contact model's
    reach into the equilibrium there.  No solves."""
    from ball_socket.provenance import require_clean_tree
    from ball_socket.immutable import guarded_path
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M as S
    prov = require_clean_tree("box G report amendment")
    path = OUT / "boxG_E15.report.json"
    rep = json.loads(path.read_text(encoding="utf-8"))
    lo, hi = rep["curve_sigma_lo_max_over_arms"], rep["curve_sigma_hi_max_over_arms"]
    c_lo = max(lo, key=lo.get)
    c_hi = max(hi, key=hi.get)
    pred = lo[c_lo] * S[0] / S[1]
    rep["sigma_scaling_check_R63"] = {
        "Lambda_star_hi": hi[c_hi], "c0_star_hi_um": float(c_hi),
        "Lambda_star_lo_times_sigma_lo_over_hi": pred, "sigma_lo_over_hi": S[0] / S[1],
        "rel_difference": hi[c_hi] / pred - 1, "same_argmax": c_hi == c_lo,
        "basis": "production 32x24 values of the grid, max over arms at each sigma* end",
        "added_at_commit": prov["commit"]}
    guarded_path(str(path), what="box G report (R-63.3 amendment)", overwrite=True)
    path.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps(rep["sigma_scaling_check_R63"], indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["spec", "run", "report", "amend"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"spec": lambda: stage_spec(), "run": lambda: stage_run(a.workers),
     "report": lambda: stage_report(a.workers), "amend": stage_amend}[a.stage]()
