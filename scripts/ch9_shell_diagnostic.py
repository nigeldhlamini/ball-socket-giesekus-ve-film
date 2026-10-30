#!/usr/bin/env python
"""EXPLORATORY POST-HOC DIAGNOSTIC (2026-07-29): distance-to-boundary shell test.

Role.  The registered reconstruction verdict is IoU and d_H, and on those the
Bayesian arm does not beat the uniform arm at this interface.  That verdict
stands and is NOT modified by anything here.  This diagnostic asks only whether
the mechanism hypothesis -- that boundary-targeted acquisition concentrates
observations in a thin shell about the feasibility boundary, which an area
measure penalises -- is supported.

Why this construction.  The slipper's boundary-recall instrument proved unsound
at this interface (degenerate boundary set, saturated at the registered radius).
Re-specifying its radius after observing that saturation would be an
unrecoverable retrofit.  This diagnostic instead has NO radius and NO neighbour
rule: it compares, between arms, the distribution of distances from each
feasible observation to the reference feasible-set boundary, normalised by the
same bounding-box diagonal already used for d_H.  Both the reference boundary
and the normalisation are inherited from the registered IoU computation, so
there is no free parameter to tune.

Pre-committed reading, fixed before the numbers existed (ruling 2026-07-29):
  * BO median materially below random's across seeds  -> mechanism CONFIRMED;
  * BO median not below random's                      -> mechanism REFUTED.
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_F_bo_ar_overlay import _alpha_shape_boundary_poly  # noqa: E402
from ch9_fidelity import PROJ, _pts, _ensure_F, OUT              # noqa: E402


def main() -> int:
    aud = pd.read_parquet(
        sorted(glob.glob(str(OUT / "audit_N90000_*.parquet")))[-1])
    aud = aud[aud.outcome == "converged"]
    rnd = pd.read_parquet(OUT / "bo_random_arm_n500.parquet")

    report = {
        "status": "EXPLORATORY POST-HOC DIAGNOSTIC (2026-07-29)",
        "role": ("supports or refutes the mechanism hypothesis only; does NOT "
                 "modify the registered IoU/d_H verdict, and is reported after it"),
        "construction": ("distance from each feasible observation to the "
                         "reference feasible-set boundary, normalised by the "
                         "bounding-box diagonal used for d_H; no radius, no "
                         "neighbour rule, no free parameter"),
        "precommitted_reading": ("BO median materially below random's -> "
                                 "mechanism CONFIRMED; not below -> REFUTED"),
        "bands": {},
    }

    for band in ("lo", "hi"):
        feas_col = f"outcome_mineral_{band}"
        ref_df = aud[aud[feas_col] == "feasible"]
        band_out = {}
        for pname, spec in PROJ.items():
            if spec["x"] == "Q_leak":
                continue                      # no BO arm; excluded throughout
            ref_pts = _pts(ref_df, spec)
            polys = _alpha_shape_boundary_poly(ref_pts, spec["alpha_frac"])
            if not polys:
                continue
            ref = unary_union(polys)
            bnd = ref.boundary
            x0, y0, x1, y1 = ref.bounds
            diag = float(np.hypot(x1 - x0, y1 - y0))

            # --- interior test (added 2026-08-01) ----------------------------
            # The shell comparison presumes the reference region HAS an
            # interior for observations to be inside or outside of.  That
            # precondition is measured, not asserted: the region's area
            # normalised by diag^2, and the largest distance any reference
            # point sits from its own boundary.  A region that is effectively
            # a curve returns ~0 on both, and the arm comparison below is then
            # undefined rather than unfavourable.
            area_frac = float(ref.area / diag ** 2) if diag > 0 else float("nan")
            bbox_area = (x1 - x0) * (y1 - y0)
            area_frac_bbox = (float(ref.area / bbox_area) if bbox_area > 0
                              else float("nan"))
            # The raw-coordinate aspect ratio is the quantity that made
            # area/diag^2 uninterpretable, so it is persisted rather than
            # asserted (2026-08-01 ruling).
            aspect = (float((x1 - x0) / (y1 - y0)) if (y1 - y0) > 0
                      else float("nan"))
            self_d = np.asarray(shapely.distance(
                bnd, MultiPoint(ref_pts).geoms), dtype=float) / diag
            max_interior = float(self_d.max()) if self_d.size else float("nan")

            arms = {}
            for arm in ("bo", "random"):
                meds, fracs = [], []
                for s in range(5):
                    if arm == "bo":
                        d = _ensure_F(pd.read_parquet(
                            OUT / "bo_runs" / f"bs_bo_{band}_s{s}_obs.parquet"))
                        sub = d[d["feasible"].fillna(False).astype(bool)]
                    else:
                        d = rnd[(rnd.band == band) & (rnd.seed == s)]
                        sub = d[d[feas_col] == "feasible"]
                    pts = _pts(sub, spec)
                    if len(pts) < 3:
                        continue
                    dist = shapely.distance(
                        bnd, MultiPoint(pts).geoms) / diag
                    dist = np.asarray(dist, dtype=float)
                    meds.append(float(np.median(dist)))
                    # share of observations lying within the outer 10 % of the
                    # region's scale, i.e. in a shell -- descriptive only
                    fracs.append(float((dist < 0.10).mean()))
                arms[arm] = {
                    "median_norm_distance": float(np.median(meds)) if meds else float("nan"),
                    "per_seed_medians": meds,
                    "shell_share_lt_0p10": float(np.median(fracs)) if fracs else float("nan"),
                }
            bo_m = arms["bo"]["median_norm_distance"]
            rd_m = arms["random"]["median_norm_distance"]
            ratio = bo_m / rd_m if rd_m else float("nan")
            band_out[pname] = {"reference_area_over_diag_sq": area_frac,
                               "reference_area_over_bbox": area_frac_bbox,
                               "raw_bbox_aspect_ratio": aspect,
                               "reference_max_interior_distance": max_interior,
                               "has_interior": bool(max_interior > 1e-6),
                               "arms": arms, "bo_over_random": ratio,
                               "bo_below_random": bool(bo_m < rd_m)}
            print(f"  {band}/{pname:10s} area/diag^2 {area_frac:.3e} "
                  f"max interior d {max_interior:.4f}")
            print(f"  {band}/{pname:10s} BO median d/diag {bo_m:.4f}  "
                  f"random {rd_m:.4f}  ratio {ratio:.2f}  "
                  f"shell<0.10: BO {arms['bo']['shell_share_lt_0p10']:.2f} "
                  f"rand {arms['random']['shell_share_lt_0p10']:.2f}")
        report["bands"][band] = band_out

    (OUT / "shell_diagnostic.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT / 'shell_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
