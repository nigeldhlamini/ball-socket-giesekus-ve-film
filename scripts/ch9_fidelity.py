#!/usr/bin/env python
"""Reconstruction fidelity for the Ch.9 ball-socket deployment (T5, F6, F7).

BRANCH-(b) SCOPING, STATED BEFORE THE NUMBERS.  The registered STOP on the
convergence classifier's global discrimination fired and stands (AUC hi 0.8609
against the registered 0.90 bar).  The classifier is validated as an in-loop
guidance instrument and is NOT validated as a global safety instrument at this
interface.  Every figure produced here inherits that scoping.

METHOD.  The DIRECT point-reconstruction route is used throughout, exactly as
at the slipper: the oracle is then a true point-selection ceiling, and surrogate
prediction is deliberately excluded because a surrogate is a denser,
extrapolating reconstruction that is not commensurable with the oracle.

There is no exhaustive ground truth at this interface -- that is the point of an
audit-only deployment.  The REFERENCE is therefore the held-out audit's feasible
set, and the CEILING is the matched-budget oracle: N_bo points drawn uniformly
from the audit-feasible set and passed through the identical alpha-shape
pipeline.

Two-tier pre-commitment (registered, reported either way): the film-thickness
projections (F, h_min) and (h_min, T_max) were pre-committed as expected
meaningful; the power/leakage projections (P_loss, F) and (Q_leak, F) as
expected extractor-limited.
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from shapely.ops import unary_union

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_F_bo_ar_overlay import _alpha_shape_boundary_poly  # noqa: E402

OUT = ROOT / "results" / "ch9"
N_BO = 500
N_ORACLE_SEEDS = 10

# Ball-socket projections.  F is an OBSERVED solver output here (the load at
# equilibrium), not the slipper's analytic k_F * F_hyd(p_s).
PROJ = {
    "F_hmin":    dict(x="F", y="h_min", alpha_frac=2.0,  x_log=True,  y_log=True,
                      tier="expected_meaningful"),
    "hmin_Tmax": dict(x="h_min", y="T_max", alpha_frac=5.0, x_log=True, y_log=False,
                      tier="expected_meaningful"),
    "Ploss_F":   dict(x="P_loss", y="F", alpha_frac=20.0, x_log=True, y_log=True,
                      tier="expected_extractor_limited"),
    "Qleak_F":   dict(x="Q_leak", y="F", alpha_frac=10.0, x_log=True, y_log=True,
                      tier="expected_extractor_limited"),
}


def _ensure_F(df):
    """F at equilibrium equals the hydrodynamic share by construction, and that
    share is deterministic in the inputs: F = p_ch (A_p - A_pocket(theta_min)).

    The BO arms persist only the Problem's declared channels (objective +
    constraints + auxiliaries), so F is not stored for them.  It is recovered
    analytically here rather than GP-modelled -- the same treatment the slipper
    gave F_applied, and for the same reason: modelling a quantity already known
    exactly from the inputs would only add noise.
    """
    if "F" in df.columns:
        return df
    from ball_socket.campaign_config_v4 import applied_and_hydrodynamic_load
    df = df.copy()
    df["F"] = [applied_and_hydrodynamic_load(p, t)[2]
               for p, t in zip(df["p_ch"], df["theta_min_deg"])]
    return df


def _pts(df, spec):
    df = _ensure_F(df)
    if spec["x"] not in df.columns or spec["y"] not in df.columns:
        return np.zeros((0, 2))
    x = df[spec["x"]].to_numpy(float)
    y = df[spec["y"]].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    if spec["x_log"]:
        m &= x > 0
    if spec["y_log"]:
        m &= y > 0
    return np.column_stack([x[m], y[m]])


def _poly(pts, alpha_frac):
    if len(pts) < 4:
        return []
    try:
        return _alpha_shape_boundary_poly(pts, alpha_frac)
    except Exception:
        return []


def _iou(polys_a, polys_b):
    if not polys_a or not polys_b:
        return float("nan")
    a, b = unary_union(polys_a), unary_union(polys_b)
    u = unary_union([a, b])
    return float(a.intersection(b).area / u.area) if u.area > 0 else float("nan")


def _hausdorff(polys_a, polys_b):
    if not polys_a or not polys_b:
        return float("nan")
    a, b = unary_union(polys_a), unary_union(polys_b)
    return float(a.boundary.hausdorff_distance(b.boundary))


def _norm_hausdorff(polys_a, polys_b, ref):
    """d_H normalised by the reference bounding-box diagonal (chapter
    convention), so values are comparable across projections."""
    d = _hausdorff(polys_a, polys_b)
    if not np.isfinite(d) or not ref:
        return float("nan")
    x0, y0, x1, y1 = unary_union(ref).bounds
    diag = float(np.hypot(x1 - x0, y1 - y0))
    return d / diag if diag > 0 else float("nan")


def main(ref_glob: str = "audit_N8192_*.parquet",
         tag: str = "fidelity_T5") -> int:
    ref_path = sorted(glob.glob(str(OUT / ref_glob)))[-1]
    aud = pd.read_parquet(ref_path)
    aud = aud[aud.outcome == "converged"]
    print(f"reference: {pathlib.Path(ref_path).name}")
    rnd = pd.read_parquet(OUT / "bo_random_arm_n500.parquet")

    report: dict = {
        "scoping": ("Branch (b): the registered STOP on the convergence "
                    "classifier's global discrimination fired and stands "
                    "(AUC hi 0.8609 < 0.90). The classifier is validated as an "
                    "in-loop guidance instrument, NOT as a global safety "
                    "instrument at this interface. These fidelity numbers "
                    "inherit that scoping."),
        "method": ("direct point reconstruction; surrogate prediction excluded "
                   "as non-commensurable with the oracle ceiling"),
        "reference": "held-out audit feasible set (no exhaustive ground truth)",
        "ceiling_validity_condition": (
            "The matched-budget oracle assumes reference size >> evaluation "
            "budget.  At the slipper, 500 drawn from 36,232 feasible = 1.4 %.  "
            "On the current reference the oracle draws 55-76 %, so it is "
            "REFERENCE-LIMITED, not budget-limited: the ceiling it produces "
            "measures extractor fidelity given near-complete information, which "
            "is a DIFFERENT OBJECT from the slipper's ceiling and is NOT "
            "comparable to it.  Ceilings below carry that label."),
        "ceiling_status": "REFERENCE-LIMITED (non-comparable to the slipper)",
        "two_tier_precommitment_status": (
            "PARTIALLY TESTED.  Of the two projections pre-committed as expected "
            "extractor-limited, (P_loss, F) is evaluable and (Q_leak, F) is NOT "
            "COMPUTED for the BO arm: Q_leak was not among the Problem's "
            "declared output channels at run time, so BOCampaign never persisted "
            "it (fixed for future runs; not worth an eight-hour re-run).  The "
            "pre-commitment must not be presented as fully tested."),
        "bands": {},
    }

    for band in ("lo", "hi"):
        feas_col = f"outcome_mineral_{band}"
        ref_df = aud[aud[feas_col] == "feasible"]
        band_out = {"n_reference_feasible": int(len(ref_df)), "projections": {}}

        for key, spec in PROJ.items():
            ref_pts = _pts(ref_df, spec)
            ref_poly = _poly(ref_pts, spec["alpha_frac"])

            # --- matched-budget oracle ceiling -------------------------------
            ious, dhs = [], []
            rs = np.random.default_rng(20260729)
            for _ in range(N_ORACLE_SEEDS):
                if len(ref_pts) < 4:
                    break
                idx = rs.choice(len(ref_pts), size=min(N_BO, len(ref_pts)),
                                replace=False)
                op = _poly(ref_pts[idx], spec["alpha_frac"])
                ious.append(_iou(op, ref_poly))
                dhs.append(_norm_hausdorff(op, ref_poly, ref_poly))
            ceil_iou = float(np.nanmedian(ious)) if ious else float("nan")
            ceil_dh = float(np.nanmedian(dhs)) if dhs else float("nan")

            # --- BO arm and matched random arm, per seed ---------------------
            arm = {}
            for arm_name in ("bo", "random"):
                vals_iou, vals_dh, ns = [], [], []
                for s in range(5):
                    if arm_name == "bo":
                        d = pd.read_parquet(
                            OUT / "bo_runs" / f"bs_bo_{band}_s{s}_obs.parquet")
                        sub = d[d.get("feasible", pd.Series(False, index=d.index))
                                .fillna(False).astype(bool)]
                    else:
                        d = rnd[(rnd.band == band) & (rnd.seed == s)]
                        sub = d[d[feas_col] == "feasible"]
                    pts = _pts(sub, spec)
                    if len(pts) == 0:
                        vals_iou.append(np.nan); vals_dh.append(np.nan)
                        ns.append(0)
                        continue
                    p = _poly(pts, spec["alpha_frac"])
                    vals_iou.append(_iou(p, ref_poly))
                    vals_dh.append(_norm_hausdorff(p, ref_poly, ref_poly))
                    ns.append(int(len(sub)))
                unavailable = (arm_name == "bo" and spec["x"] == "Q_leak")
                arm[arm_name] = {
                    "iou_median": float(np.nanmedian(vals_iou)),
                    "dH_median": float(np.nanmedian(vals_dh)),
                    "n_feasible_median": float(np.median(ns)),
                    **({"status": "UNAVAILABLE: Q_leak is a solver output and "
                                  "was not among the Problem's declared "
                                  "channels, so the BO arms did not persist it. "
                                  "Recoverable only by re-running."}
                       if unavailable else {}),
                }

            band_out["projections"][key] = {
                "tier_precommitted": spec["tier"],
                "n_reference_points": int(len(ref_pts)),
                "oracle_ceiling_iou": ceil_iou,
                "oracle_ceiling_dH": ceil_dh,
                "arms": arm,
            }
            print(f"  {band}/{key:10s} ceiling IoU {ceil_iou:.4f}  "
                  f"BO {arm['bo']['iou_median']:.4f}  "
                  f"rand {arm['random']['iou_median']:.4f}   "
                  f"[{spec['tier']}]")
        report["bands"][band] = band_out

    report["reference_file"] = pathlib.Path(ref_path).name
    (OUT / f"{tag}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT / (tag + '.json')}")
    return 0


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "extended":
        raise SystemExit(main("audit_N90000_*.parquet", "fidelity_T5_extended"))
    raise SystemExit(main())
