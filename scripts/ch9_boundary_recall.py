#!/usr/bin/env python
"""Composite-boundary recall for the ball-socket BO and random arms.

DIAGNOSTIC, NOT A VERDICT.  The registered reconstruction metrics are IoU and
d_H, and on those BO does not beat random at this interface.  This script
applies the SLIPPER CAMPAIGN'S OWN registered instrument -- same definition,
same parameters -- to the second interface, to test the mechanism hypothesis
that the acquisition optimises the boundary while region metrics reward uniform
interior coverage.  It appears after the IoU/d_H verdict, never in place of it.

Instrument, carried unchanged from scripts/bo_composite_validation.py:
  * observed feasibility boundary = feasible observation with >= 1
    infeasible/non-converged neighbour, k = 10 nearest neighbours in unit-cube
    INPUT space among the arm's own 500 observations;
  * reference labelled boundary = the same construction on the audit set, each
    boundary point labelled by the majority cause among its bad neighbours;
  * per-label recall = fraction of reference labelled boundary points lying
    within r = 0.04 x plane diagonal of an observed boundary point, per
    projection.
Only the cause labels are re-specified for this interface (blow-up replaces
lift-off; thresholds are the registered ball-socket ones).
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_composite_boundary import _boundary_mask_labels  # noqa: E402  (identical instrument)
from ball_socket.campaign_config_v4 import INPUT_SPACE, ConstraintSet  # noqa: E402
from ball_socket.campaign import BANDS  # noqa: E402

OUT = ROOT / "results" / "ch9"
K_NN = 10               # slipper parameter, unchanged
R_FRAC = 0.04           # slipper parameter, unchanged (fraction of plane diagonal)
PROJ = {"F_hmin": ("F", "h_min"), "hmin_Tmax": ("h_min", "T_max"),
        "Ploss_F": ("P_loss", "F")}
_C = ConstraintSet()
NAMES = [d.name for d in INPUT_SPACE]


def _unit(df):
    U = np.empty((len(df), len(INPUT_SPACE)))
    for i, d in enumerate(INPUT_SPACE):
        v = df[d.name].to_numpy(float)
        U[:, i] = (np.log(v / d.lower) / np.log(d.upper / d.lower)
                   if d.scale == "log" else (v - d.lower) / (d.upper - d.lower))
    return np.clip(U, 0.0, 1.0)


def _add_F(df):
    if "F" in df.columns and df["F"].notna().any():
        return df
    from ball_socket.campaign_config_v4 import applied_and_hydrodynamic_load
    df = df.copy()
    df["F"] = [applied_and_hydrodynamic_load(p, t)[2]
               for p, t in zip(df["p_ch"], df["theta_min_deg"])]
    return df


def _cause_bs(row, sigma, ceiling):
    """Ball-socket cause labels; same structure as the slipper's _cause."""
    if not bool(row["converged"]):
        return "blow_up"
    if row["h_min"] / sigma < _C.lambda_ratio_min:
        return "g1"
    if row["De_sq"] > _C.De_sq_max:
        return "g3"
    if np.isfinite(row["T_max"]) and row["T_max"] > ceiling:
        return "g2"
    return "g3"


def main() -> int:
    aud = _add_F(pd.read_parquet(
        sorted(glob.glob(str(OUT / "audit_N90000_*.parquet")))[-1]))
    rnd = _add_F(pd.read_parquet(OUT / "bo_random_arm_n500.parquet"))
    report = {
        "role": ("DIAGNOSTIC explaining the registered IoU/d_H verdict; not a "
                 "substitute for it and reported after it"),
        "instrument": ("slipper composite-boundary recall, carried unchanged: "
                       f"k={K_NN} NN in unit-cube input space, "
                       f"r={R_FRAC} x plane diagonal"),
        "bands": {},
    }

    for band in ("lo", "hi"):
        sigma = BANDS[band]
        ceiling = _C.T_max_C_by_class["mineral"]
        feas_col = f"outcome_mineral_{band}"

        # --- reference labelled boundary from the audit -------------------
        U_ref = _unit(aud)
        feas_ref = (aud[feas_col] == "feasible").to_numpy()
        causes = np.array([_cause_bs(r, sigma, ceiling)
                           for _, r in aud.iterrows()], dtype=object)
        bmask, labels = _boundary_mask_labels(U_ref, feas_ref, causes, K_NN)
        ref_idx = np.where(bmask)[0]
        band_out = {"n_reference_boundary": int(len(ref_idx)),
                    "label_counts": {str(k): int(v) for k, v in
                                     zip(*np.unique(labels[ref_idx],
                                                    return_counts=True))},
                    "projections": {}}

        for pname, (fx, fy) in PROJ.items():
            rx = aud[fx].to_numpy(float)[ref_idx]
            ry = aud[fy].to_numpy(float)[ref_idx]
            ok = np.isfinite(rx) & np.isfinite(ry)
            rx, ry, rlab = rx[ok], ry[ok], labels[ref_idx][ok]
            if len(rx) < 4:
                continue
            diag = float(np.hypot(rx.max() - rx.min(), ry.max() - ry.min()))
            r = R_FRAC * diag
            ref_pts = np.column_stack([rx, ry])

            arms = {}
            for arm in ("bo", "random"):
                per_lab = {}
                overall = []
                for s in range(5):
                    if arm == "bo":
                        d = _add_F(pd.read_parquet(
                            OUT / "bo_runs" / f"bs_bo_{band}_s{s}_obs.parquet"))
                        U = d[[f"u_{n}" for n in NAMES]].to_numpy()
                        f_obs = d.get("feasible")
                        f_obs = (f_obs.fillna(False).astype(bool).to_numpy()
                                 if f_obs is not None
                                 else np.zeros(len(d), bool))
                    else:
                        d = rnd[(rnd.band == band) & (rnd.seed == s)]
                        U = _unit(d)
                        f_obs = (d[feas_col] == "feasible").to_numpy()
                    if f_obs.sum() < 2 or (~f_obs).sum() < 1:
                        continue
                    obm, _ = _boundary_mask_labels(
                        U, f_obs, np.array(["x"] * len(U), dtype=object), K_NN)
                    ox = d[fx].to_numpy(float)[obm]
                    oy = d[fy].to_numpy(float)[obm]
                    m = np.isfinite(ox) & np.isfinite(oy)
                    if m.sum() < 1:
                        continue
                    tree = cKDTree(np.column_stack([ox[m], oy[m]]))
                    dist, _ = tree.query(ref_pts, k=1)
                    hit = dist <= r
                    overall.append(float(hit.mean()))
                    for lab in np.unique(rlab):
                        sel = rlab == lab
                        per_lab.setdefault(str(lab), []).append(
                            float(hit[sel].mean()))
                arms[arm] = {
                    "recall_overall_median": float(np.median(overall)) if overall else float("nan"),
                    "recall_by_label_median": {k: float(np.median(v))
                                               for k, v in per_lab.items()},
                }
            band_out["projections"][pname] = {
                "n_reference_boundary_pts": int(len(rx)),
                "radius": r, "arms": arms}
            bo_r = arms.get("bo", {}).get("recall_overall_median", float("nan"))
            rd_r = arms.get("random", {}).get("recall_overall_median", float("nan"))
            print(f"  {band}/{pname:10s} ref_bnd {len(rx):5d}  "
                  f"BO recall {bo_r:.3f}  random {rd_r:.3f}  "
                  f"{'BO wins' if bo_r > rd_r else 'random wins'}")
        report["bands"][band] = band_out

    (OUT / "boundary_recall.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT / 'boundary_recall.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
