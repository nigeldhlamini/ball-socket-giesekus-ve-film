#!/usr/bin/env python
r"""CME/FEM handover items 2.2 and 2.3, NARROWED per the 2026-08-03 ruling.

Six named operating points rather than the full 1323 full-film rows:

    inside-1, inside-2   comfortably inside the full-film set (highest Lambda)
    edge-1,  edge-2      nearest the full-film threshold Lambda = 3
    load-1,  load-2      highest applied load among full-film points

Each is a re-solve of a row that already exists in the registered audit, at the
same inputs, with the field arrays retained instead of discarded.  The campaign
is not re-run and no registered artefact is modified.

Per point:
  2.2  film pressure p(theta, phi) and traction on the solver grid, with the
       grid definition written explicitly so the FEM side can map it.
  2.3  film temperature field, wall heat flux into the socket, bulk rise.

The grid is a CELL-CENTRED finite-volume grid on the annulus; node coordinates
are the cell centres and are exported as arrays, not as a formula for the
reader to re-derive.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "ch9"
EXPORT = ROOT / "exports" / "cme" / "fields"
sys.path.insert(0, str(ROOT))

from ball_socket import thermal as bs_thermal                   # noqa: E402
from ball_socket.campaign import NPHI, NTHETA                   # noqa: E402
from ball_socket.campaign_config_v4 import (                    # noqa: E402
    ALPHA_FIXED, P_CASE_PA, SPIN_RATIO, SQUEEZE_PHASE)
from ball_socket.geometry import BallSocketGeometry             # noqa: E402
from src.solver import SolverConfig                             # noqa: E402
from ball_socket.solver import BallSocketSolver, CampaignInput  # noqa: E402

LAMBDA_FULL_FILM = 3.0
INPUT_COLS = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio",
              "theta_min_deg"]


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def select(df: pd.DataFrame, band: str = "lo") -> pd.DataFrame:
    """Six points spanning the regime boundary, at the given sigma* band."""
    conv = df[df.outcome == "converged"].copy()
    lam = conv[f"lambda_ratio_{band}"].astype(float)
    ff = conv[lam >= LAMBDA_FULL_FILM].copy()
    ff["Lambda"] = lam[lam >= LAMBDA_FULL_FILM]
    ff["F_app"] = ff["F_app"].astype(float)

    inside = ff.nlargest(2, "Lambda")
    edge = ff.nsmallest(2, "Lambda")
    load = ff.nlargest(2, "F_app")
    picks, tags = [], []
    for frame, tag in ((inside, "inside"), (edge, "edge"), (load, "load")):
        for k, (_, r) in enumerate(frame.iterrows(), start=1):
            picks.append(r)
            tags.append(f"{tag}-{k}")
    out = pd.DataFrame(picks)
    out.insert(0, "point_tag", tags)
    # A point can qualify twice (e.g. highest load also nearest the edge); the
    # duplicate is kept and labelled rather than silently dropped, so the six
    # slots stay traceable to the criterion that chose them.
    out["duplicate_of"] = [
        next((tags[j] for j in range(i) if picks[j].name == picks[i].name), "")
        for i in range(len(picks))]
    return out


def solve_fields(x: dict) -> dict:
    """Re-solve one operating point and retain the fields.

    The solver is constructed exactly as campaign.evaluate_x constructs it, so
    the re-solve reproduces the campaign row rather than a nearby one; that
    reproduction is asserted against the persisted h_min by the caller.
    """
    geom = BallSocketGeometry(c0=x["c0"],
                              theta_min=np.deg2rad(x["theta_min_deg"]))
    slv = BallSocketSolver(
        geom, model="viscoelastic", Ntheta=NTHETA, Nphi=NPHI,
        alpha=ALPHA_FIXED, squeeze_phase=SQUEEZE_PHASE, spin_ratio=SPIN_RATIO,
        config=SolverConfig(include_normal_stress=False, max_iter=80, tol=1e-9,
                            omega=0.7))
    ci = CampaignInput(omega=x["omega"], p_ch=x["p_ch"], c0=x["c0"],
                       T_in=x["T_in"], eta_0=x["eta_0"], lam=x["lam"],
                       eta_s_over_eta0=x["eta_s_ratio"],
                       k_F_bs=None, p_case=P_CASE_PA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perf = slv.solve_equilibrium(ci)
        fluid = slv._fluid(ci)
        res, sph = slv._forward(ci, perf.eps_ecc, fluid, ci.c0)

    h = sph.gap()
    eta_bar = res.eta_bar if res.eta_bar is not None else np.full_like(h, ci.eta_0)
    gdot = sph.gap_averaged_shear_rate(res.pressure, eta_bar)
    msh = sph.mesh
    p = (res.pressure.reshape(msh.Ntheta, msh.Nphi)
         if res.pressure.ndim == 1 else res.pressure)

    th = slv.thermal
    T_wall_K = ci.T_in + 273.15
    T_field_K = bs_thermal.solve_l2_spherical(sph, eta_bar, gdot, th, T_wall_K)
    # Cross-film conduction to the bounding solids IS the wall heat flux in the
    # L2 closure: q_wall = k_eff (T - T_wall), positive into the socket.
    k_eff = th.k_f / np.maximum(h, 1e-12)
    q_wall = k_eff * (T_field_K - T_wall_K)

    return {
        "performance": {
            "F_N": perf.F, "h_min_m": perf.h_min, "T_max_C": perf.T_max,
            "P_loss_W": perf.P_loss, "Q_leak_m3_s": perf.Q_leak,
            "De_sq": perf.De_sq, "eps_ecc": perf.eps_ecc,
            "thermal_mode": perf.thermal_mode,
            "thermal_route": perf.thermal_route,
            "F_app_N": perf.F_app, "F_pocket_N": perf.F_pocket,
            "F_hydrodynamic_N": perf.F_hydrodynamic,
            "bulk_temperature_rise_C": bs_thermal.l1_bulk_delta_T(
                perf.P_loss, perf.Q_leak, th),
        },
        "grid": {
            "kind": "cell-centred finite volume on the annulus",
            "Ntheta": int(msh.Ntheta), "Nphi": int(msh.Nphi),
            "theta_min_rad": float(msh.theta_min),
            "theta_max_rad": float(msh.theta_max),
            "dtheta_rad": float(msh.dtheta), "dphi_rad": float(msh.dphi),
            "R_b_m": float(slv.geom.R_b),
            "theta_centres_rad": msh.theta.tolist(),
            "phi_centres_rad": msh.phi.tolist(),
            "note": ("theta measured from the load pole; phi is azimuth.  "
                     "Cartesian node positions are "
                     "R_b*(sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)) "
                     "on the ball surface."),
        },
        "arrays": {
            "pressure_Pa": p,
            "film_thickness_m": h,
            "shear_rate_1_s": gdot,
            "viscosity_Pa_s": eta_bar,
            "shear_traction_Pa": eta_bar * gdot,
            "temperature_K": T_field_K,
            "wall_heat_flux_W_m2": q_wall,
        },
    }


def main() -> int:
    EXPORT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(OUT / "audit_N8192_seed20260727.parquet")
    band = "lo"
    sel = select(df, band)

    index, failures = [], []
    for _, r in sel.iterrows():
        tag = r["point_tag"]
        x = {c: float(r[c]) for c in INPUT_COLS}
        try:
            out = solve_fields(x)
        except Exception as exc:                       # never silently skipped
            failures.append({"point_tag": tag, "error": f"{type(exc).__name__}: {exc}"})
            continue
        # The re-solve must reproduce the persisted campaign row, or the fields
        # describe a different operating point from the one the regime map and
        # the contact dataset describe.  Checked, not assumed.
        repro = {
            "h_min_persisted_m": float(r["h_min"]),
            "h_min_resolved_m": out["performance"]["h_min_m"],
            "T_max_persisted_C": float(r["T_max"]),
            "T_max_resolved_C": out["performance"]["T_max_C"],
        }
        repro["h_min_rel_diff"] = abs(
            repro["h_min_resolved_m"] - repro["h_min_persisted_m"]) / max(
                abs(repro["h_min_persisted_m"]), 1e-30)
        repro["reproduces"] = bool(repro["h_min_rel_diff"] < 1e-6)
        if not repro["reproduces"]:
            failures.append({"point_tag": tag,
                             "error": "re-solve does not reproduce the "
                                      "persisted row", **repro})
            continue

        npz = EXPORT / f"{tag}.npz"
        np.savez_compressed(
            npz,
            theta_centres_rad=np.asarray(out["grid"]["theta_centres_rad"]),
            phi_centres_rad=np.asarray(out["grid"]["phi_centres_rad"]),
            **{k: np.asarray(v) for k, v in out["arrays"].items()})
        meta = {"point_tag": tag,
                "selection_criterion": tag.split("-")[0],
                "duplicate_of": r["duplicate_of"],
                "sigma_star_band": band,
                "source_row_id": int(r.name),
                "film_parameter_Lambda": float(r["Lambda"]),
                "inputs": x, "reproduction_check": repro,
                **{k: out[k] for k in ("performance", "grid")},
                "arrays_in_npz": {k: list(np.asarray(v).shape)
                                  for k, v in out["arrays"].items()},
                "npz_sha256": _sha(npz)}
        (EXPORT / f"{tag}.json").write_text(json.dumps(meta, indent=2, default=str))
        index.append({"point_tag": tag, "Lambda": float(r["Lambda"]),
                      "F_app_N": float(r["F_app"]),
                      "h_min_m": out["performance"]["h_min_m"],
                      "T_max_C": out["performance"]["T_max_C"],
                      "p_max_Pa": float(np.max(out["arrays"]["pressure_Pa"])),
                      "q_wall_max_W_m2": float(np.max(out["arrays"]["wall_heat_flux_W_m2"])),
                      "npz": npz.name, "sha256": _sha(npz)})
        print(f"  {tag:<9} Lambda {r['Lambda']:7.2f}  F_app {r['F_app']:9.1f} N  "
              f"p_max {np.max(out['arrays']['pressure_Pa'])/1e6:7.2f} MPa  "
              f"{npz.name}")

    (EXPORT / "index.json").write_text(json.dumps(
        {"band": band, "n_points": len(index),
         "scope": ("NARROWED per the 2026-08-03 ruling: six points spanning "
                   "the regime boundary, not the full 1323 full-film rows"),
         "criteria": {"inside": "highest film parameter Lambda",
                      "edge": "nearest the full-film threshold Lambda = 3",
                      "load": "highest applied load among full-film points"},
         "points": index, "failures": failures}, indent=2))
    if failures:
        print("\nFAILURES (reported, not skipped):")
        for f in failures:
            print(f"  {f['point_tag']}: {f['error']}")
    total = sum((EXPORT / i["npz"]).stat().st_size for i in index)
    print(f"\n{len(index)} points, {total/1e6:.1f} MB total")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
