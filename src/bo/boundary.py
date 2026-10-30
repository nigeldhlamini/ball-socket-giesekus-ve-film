"""AR-boundary extraction from BO surrogates: alpha_shape and gp_level_set.

This module supplies two complementary boundary-extraction methods for the
attainable-region reconstruction:

  "alpha_shape"   — the classical post-hoc method used in Chapter~7 and
                     Chapter~8 of the thesis up to the BO refinement
                     campaign. Takes the BO observations themselves and
                     extracts the alpha-shape of their feasible subset at
                     a fixed alpha_frac per projection.

  "gp_level_set"  — the Stage-A refinement of the BO campaign. Takes the
                     fitted classifier + constraint + regression GPs and
                     extracts the boundary of the set
                       { x : P_classifier(x) * Π_j P(g_j(x) >= 0) >= 0.5 }
                     projected onto the chosen performance plane. Uses a
                     Sobol grid in the 8-D input space, regression-GP
                     posterior means to embed grid points into the
                     performance plane, a 2-D histogram, and the
                     marching-squares contour at probability 0.5.

The two methods return the same data structure (a list of (M, 2) ndarrays,
one per polygon, in the projection's natural physical units) so downstream
set-comparison code (IoU/precision/recall via shapely) operates identically
on either output.

This module is consumed by BOCampaign.evaluate_ar_boundary as well as by
the analysis scripts that operate on a refit SurrogateManager.

References for the level-set construction:
  Bryan, R., et al. "Active learning for identifying function threshold
    boundaries." NeurIPS 18 (2005).
  Bogunovic, I., et al. "Truncated variance reduction: a unified approach
    to Bayesian optimization and level-set estimation." NeurIPS 29 (2016).
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Optional

import numpy as np
import torch
from scipy.stats import norm
from scipy.spatial import ConvexHull

# Repo root on path so we can import ar_bo_campaign.
_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ar_bo_campaign import (  # noqa: E402
    INPUT_SPACE,
    alpha_shape_boundary_v2,
    hydrostatic_floor,
)

# F_applied = k_F * F_hyd(p_s) is analytic from inputs; we cache the
# per-unit-pressure floor (F_hyd scales linearly in p_s).
F_HYD_PER_PS = hydrostatic_floor(p_s=1.0)


# =============================================================================
# Projection specification
# =============================================================================


@dataclasses.dataclass
class ProjectionSpec:
    """Describes a 2-D performance projection.

    name           — short label (e.g. "F_hmin", "Qleak_F").
    x_field        — performance metric on the x-axis.
    y_field        — performance metric on the y-axis.
    alpha_frac     — the chapter's selected alpha-shape tightness for the
                     EXHAUSTIVE feasible set on this projection.
    x_field_source — "gp" if the field comes from a regression GP, "analytic"
                     if it is computed from inputs analytically (used for
                     F_applied which is exactly k_F * F_hyd(p_s)).
    y_field_source — same, for the y-axis.
    """
    name: str
    x_field: str
    y_field: str
    alpha_frac: float
    x_field_source: str = "gp"
    y_field_source: str = "gp"


# Chapter-canonical four projections.
CANONICAL_PROJECTIONS = (
    ProjectionSpec(
        name="F_hmin",
        x_field="F_applied", y_field="h_min", alpha_frac=2.0,
        x_field_source="analytic", y_field_source="gp",
    ),
    ProjectionSpec(
        name="hmin_Tmax",
        x_field="h_min", y_field="T_max", alpha_frac=5.0,
        x_field_source="gp", y_field_source="gp",
    ),
    ProjectionSpec(
        name="Ploss_F",
        x_field="P_loss", y_field="F_applied", alpha_frac=20.0,
        x_field_source="gp", y_field_source="analytic",
    ),
    ProjectionSpec(
        name="Qleak_F",
        x_field="Q_leak", y_field="F_applied", alpha_frac=10.0,
        x_field_source="gp", y_field_source="analytic",
    ),
)


def _analytic_field(name: str, X_unit: np.ndarray) -> np.ndarray:
    """Compute an analytically-known field from unit-cube inputs."""
    if name == "F_applied":
        k_F = INPUT_SPACE[7].from_unit(X_unit[:, 7])
        p_s = INPUT_SPACE[1].from_unit(X_unit[:, 1])
        return k_F * F_HYD_PER_PS * p_s
    raise KeyError(f"unknown analytic field '{name}'")


# =============================================================================
# Shared step: Sobol grid + GP prediction + per-projection coordinates
# =============================================================================


def _grid_predict_and_project(
    surrogates,
    spec: ProjectionSpec,
    *,
    n_grid: int,
    sobol_seed: int,
    constraint_names: list[str],
    objective_name: str,
    auxiliary_names: list[str],
    batch_size: int,
) -> dict:
    """Evaluate GPs on a Sobol grid and project to the spec's plane.

    This is the shared building block for both `alpha_shape` and
    `gp_level_set` boundary methods (they differ only in the boundary
    extractor applied to the result).

    Returns dict with:
        U_np         : (N, d) Sobol unit-cube grid
        grid         : dict of GP-evaluation outputs (P_cls, PF_{c}, *_mean)
        P_joint      : (N,) joint feasibility probability
        x_vals       : (N,) x-axis coordinate (GP-mean or analytic)
        y_vals       : (N,) y-axis coordinate (GP-mean or analytic)
    """
    from torch.quasirandom import SobolEngine
    d = len(INPUT_SPACE)
    sob = SobolEngine(dimension=d, scramble=True, seed=sobol_seed)
    U = sob.draw(n_grid).to(dtype=torch.float64)
    U_np = U.cpu().numpy()

    grid = _evaluate_gp_grid(
        surrogates, U,
        constraint_names=constraint_names,
        objective_name=objective_name,
        auxiliary_names=auxiliary_names,
        batch_size=batch_size,
    )
    P_joint = grid["P_cls"].copy()
    for c in constraint_names:
        P_joint *= grid[f"PF_{c}"]

    def _grid_field(name: str, source: str) -> np.ndarray:
        if source == "analytic":
            return _analytic_field(name, U_np)
        if source == "gp":
            key = f"{name}_mean"
            if key not in grid:
                raise KeyError(
                    f"Regression GP for '{name}' not fitted; "
                    f"need it for projection {spec.name!r}"
                )
            return grid[key]
        raise ValueError(f"unknown field source {source!r}")

    x_vals = _grid_field(spec.x_field, spec.x_field_source)
    y_vals = _grid_field(spec.y_field, spec.y_field_source)
    return {
        "U_np": U_np, "grid": grid, "P_joint": P_joint,
        "x_vals": x_vals, "y_vals": y_vals,
    }


# =============================================================================
# Mode 1: alpha_shape on the GP-predicted feasible Sobol-grid projection
#
# Mirrors analysis_F's pipeline (the "existing behaviour" of the chapter's
# BO AR overlay reporting): take the Sobol grid where GP-predicted feasibility
# holds, project onto (x_field, y_field), and extract the alpha-shape at the
# projection's selected tightness.
# =============================================================================


def _alpha_shape_branch(
    surrogates,
    spec: ProjectionSpec,
    *,
    n_grid: int,
    sobol_seed: int,
    p_threshold: float,
    constraint_names: list[str],
    objective_name: str,
    auxiliary_names: list[str],
    batch_size: int,
) -> dict:
    """Compute the AR alpha-shape on the GP-predicted-feasible Sobol-grid
    projection. This is the chapter's published "BO AR overlay" baseline.
    """
    pred = _grid_predict_and_project(
        surrogates, spec, n_grid=n_grid, sobol_seed=sobol_seed,
        constraint_names=constraint_names, objective_name=objective_name,
        auxiliary_names=auxiliary_names, batch_size=batch_size,
    )
    keep = (pred["P_joint"] >= p_threshold) \
        & np.isfinite(pred["x_vals"]) & np.isfinite(pred["y_vals"])
    n_keep = int(keep.sum())
    if n_keep < 4:
        return {
            "method": "alpha_shape", "projection": spec.name,
            "status": "insufficient_above_threshold",
            "n_grid": int(n_grid), "n_above_threshold": n_keep,
            "p_threshold": p_threshold,
            "boundary_vertices": np.zeros((0, 2)),
            "edges": np.zeros((0, 2), dtype=int),
            "n_components": 0,
            "alpha_frac": spec.alpha_frac,
        }
    pts = np.column_stack([pred["x_vals"][keep], pred["y_vals"][keep]])
    res = alpha_shape_boundary_v2(pts, alpha_frac=spec.alpha_frac)
    return {
        "method": "alpha_shape",
        "projection": spec.name,
        "status": "ok",
        "n_grid": int(n_grid),
        "n_above_threshold": n_keep,
        "p_threshold": p_threshold,
        "boundary_vertices": pts[res.vertex_idx],
        "edges": res.edges,
        "n_components": int(res.n_components),
        "alpha_frac": spec.alpha_frac,
        "bbox_min": res.bbox_min,
        "bbox_max": res.bbox_max,
        "_points": pts,
    }


# =============================================================================
# Mode 2: gp_level_set
# =============================================================================


def _evaluate_gp_grid(
    surrogates,
    X_unit_torch: torch.Tensor,
    constraint_names: list[str],
    objective_name: str,
    auxiliary_names: list[str],
    batch_size: int = 10_000,
) -> dict[str, np.ndarray]:
    """Evaluate the classifier GP + constraint GPs (with feasibility prob)
    + regression GPs (posterior means) on the input grid, in batches to
    avoid memory spikes.

    Returns
    -------
    dict with keys:
      "P_cls"            : (N,) classifier-GP P(converged).
      "PF_<g>"           : (N,) P(g >= 0) for each constraint name.
      "<name>_mean"      : (N,) regression-GP posterior mean for each
                           regression GP (objective + each aux).
    """
    N = X_unit_torch.shape[0]
    out: dict[str, np.ndarray] = {}
    # Allocate buffers
    out["P_cls"] = np.empty(N, dtype=np.float64)
    for c in constraint_names:
        out[f"PF_{c}"] = np.empty(N, dtype=np.float64)
    regression_names = [objective_name] + auxiliary_names
    for n in regression_names:
        out[f"{n}_mean"] = np.empty(N, dtype=np.float64)
        out[f"{n}_var"] = np.empty(N, dtype=np.float64)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            Xb = X_unit_torch[i:j]
            out["P_cls"][i:j] = surrogates.predict_feasibility(Xb).cpu().numpy()
            for c in constraint_names:
                mu, var = surrogates.predict(c, Xb)
                mu = mu.cpu().numpy(); var = var.cpu().numpy()
                # Constraint slack >= 0 probability under Gaussian approx
                # (mean=mu, std=sqrt(var)). P(g >= 0) = 1 - Phi(-mu/std).
                std = np.sqrt(np.maximum(var, 1e-30))
                out[f"PF_{c}"][i:j] = 1.0 - norm.cdf(-mu / std)
            for n in regression_names:
                mu, var = surrogates.predict(n, Xb)
                out[f"{n}_mean"][i:j] = mu.cpu().numpy()
                out[f"{n}_var"][i:j] = var.cpu().numpy()
    return out


def _gp_level_set_branch(
    surrogates,
    spec: ProjectionSpec,
    *,
    n_grid: int = 100_000,
    sobol_seed: int = 0,
    p_threshold: float = 0.5,
    n_hist: int = 200,
    constraint_names: Optional[list[str]] = None,
    objective_name: Optional[str] = None,
    auxiliary_names: Optional[list[str]] = None,
    batch_size: int = 10_000,
) -> dict:
    """Compute the GP-implied feasibility level-set boundary on the
    spec's projection via marching-squares on a 2-D histogram.

    Uses the same Sobol grid + GP prediction + projection step as the
    alpha_shape branch; differs only in the boundary extractor
    (marching-squares contour at the 0.5 occupancy level, rather than
    post-hoc alpha-shape on the projected points).
    """
    pred = _grid_predict_and_project(
        surrogates, spec, n_grid=n_grid, sobol_seed=sobol_seed,
        constraint_names=constraint_names or list(surrogates.constraint_names),
        objective_name=objective_name or (
            surrogates.objective_names[0] if surrogates.objective_names else None
        ),
        auxiliary_names=auxiliary_names or list(surrogates.auxiliary_names),
        batch_size=batch_size,
    )
    keep = (pred["P_joint"] >= p_threshold) \
        & np.isfinite(pred["x_vals"]) & np.isfinite(pred["y_vals"])
    n_keep = int(keep.sum())
    if n_keep < 10:
        return {
            "method": "gp_level_set",
            "projection": spec.name,
            "status": "insufficient_above_threshold",
            "n_grid": int(n_grid),
            "n_above_threshold": n_keep,
            "p_threshold": p_threshold,
            "boundary_polygons": [],
            "contour_arrays": [],
        }
    pts = np.column_stack([pred["x_vals"][keep], pred["y_vals"][keep]])

    # Boolean-occupancy histogram + marching-squares at 0.5: each cell
    # is "feasible" if at least one P_joint>=0.5 grid point falls in it.
    from skimage.measure import find_contours
    x_edges = np.linspace(pts[:, 0].min(), pts[:, 0].max(), n_hist + 1)
    y_edges = np.linspace(pts[:, 1].min(), pts[:, 1].max(), n_hist + 1)
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_edges, y_edges])
    occupancy = (H > 0).astype(float)
    contours = find_contours(occupancy, level=0.5)

    # Convert contour (row, col) coordinates to physical (x, y) via bin
    # edges. find_contours returns (row, col) with row in [0, H.shape[0]-1]
    # and col in [0, H.shape[1]-1]. Bin k spans [edges[k], edges[k+1]].
    def _idx_to_phys(idx: float, edges: np.ndarray) -> float:
        # Linear interp from bin-edge array. idx 0.0 → edges[0]; idx
        # (len-1)*1.0 → edges[-2] + 0.5*(edges[-1]-edges[-2]) etc.
        # Easiest: treat idx as fractional bin index in [0, n_bins-1] and
        # interpolate against bin centres.
        n_bins = len(edges) - 1
        centres = 0.5 * (edges[:-1] + edges[1:])
        # Allow extrapolation: clamp first/last.
        return float(np.interp(idx, np.arange(n_bins), centres))

    contour_arrays: list[np.ndarray] = []
    for c in contours:
        # c shape (M, 2) with (row_idx, col_idx). Map to (x, y).
        xs = np.array([_idx_to_phys(c[k, 0], x_edges) for k in range(len(c))])
        ys = np.array([_idx_to_phys(c[k, 1], y_edges) for k in range(len(c))])
        if len(xs) >= 3:
            contour_arrays.append(np.column_stack([xs, ys]))

    return {
        "method": "gp_level_set",
        "projection": spec.name,
        "status": "ok",
        "n_grid": int(n_grid),
        "n_above_threshold": n_keep,
        "p_threshold": p_threshold,
        "n_hist": n_hist,
        "n_contours": len(contour_arrays),
        "contour_arrays": contour_arrays,
        "pts_filtered": pts,  # for downstream visualisation
        "bbox_min": np.array([x_edges[0], y_edges[0]]),
        "bbox_max": np.array([x_edges[-1], y_edges[-1]]),
        "_grid_eval_summary": {
            "P_joint_above_0.5": int(np.sum(pred["P_joint"] >= 0.5)),
            "P_joint_median": float(np.median(pred["P_joint"])),
            "P_joint_max": float(pred["P_joint"].max()),
        },
    }


# =============================================================================
# Top-level dispatch
# =============================================================================


def evaluate_ar_boundary(
    *,
    spec: ProjectionSpec,
    boundary_method: str,
    surrogates,
    n_grid: int = 100_000,
    sobol_seed: int = 0,
    p_threshold: float = 0.5,
    n_hist: int = 200,
    batch_size: int = 10_000,
    constraint_names: Optional[list[str]] = None,
    objective_name: Optional[str] = None,
    auxiliary_names: Optional[list[str]] = None,
) -> dict:
    """Dispatch to alpha_shape or gp_level_set boundary extractor.

    Both branches consume a fitted SurrogateManager. They share the same
    Sobol-grid GP-prediction + projection step (see
    `_grid_predict_and_project`). The only difference is the boundary
    extractor applied to the GP-predicted-feasible projected points:

      - "alpha_shape": post-hoc alpha-shape at the projection's selected
        tightness. This is the chapter's existing behaviour, retained
        for backward compatibility and direct comparison.
      - "gp_level_set": marching-squares contour at the 0.5-occupancy
        level of a 2-D histogram. Replaces the alpha-shape's gap-
        bridging artefact with a parameter-free contour extractor.
    """
    if surrogates is None:
        raise ValueError(
            "evaluate_ar_boundary requires a fitted SurrogateManager via "
            "the `surrogates` argument."
        )
    cn = constraint_names or list(surrogates.constraint_names)
    on = objective_name or (
        surrogates.objective_names[0] if surrogates.objective_names else None
    )
    an = auxiliary_names or list(surrogates.auxiliary_names)
    if boundary_method == "alpha_shape":
        return _alpha_shape_branch(
            surrogates, spec,
            n_grid=n_grid, sobol_seed=sobol_seed,
            p_threshold=p_threshold,
            constraint_names=cn, objective_name=on, auxiliary_names=an,
            batch_size=batch_size,
        )
    if boundary_method == "gp_level_set":
        return _gp_level_set_branch(
            surrogates, spec,
            n_grid=n_grid, sobol_seed=sobol_seed,
            p_threshold=p_threshold, n_hist=n_hist,
            constraint_names=cn, objective_name=on, auxiliary_names=an,
            batch_size=batch_size,
        )
    raise ValueError(
        f"unknown boundary_method {boundary_method!r}; "
        f"expected 'alpha_shape' or 'gp_level_set'"
    )


# =============================================================================
# Diagnostic: alpha-shape on the raw BO observations (kept for chapter-7
# reproducibility, used by the AR-boundary bootstrap deferred analysis).
# Not part of the Stage A boundary_method dispatch — that is between the
# two GP-grid-based methods above.
# =============================================================================


def obs_alpha_shape_boundary(
    X_obs_unit: np.ndarray,
    Y_obs: dict[str, np.ndarray],
    feasible: np.ndarray,
    spec: ProjectionSpec,
) -> dict:
    """Direct alpha-shape on the BO observations themselves (i.e. without
    any surrogate prediction). Provided as a diagnostic baseline."""
    def _field(name: str, source: str) -> np.ndarray:
        if source == "analytic":
            return _analytic_field(name, X_obs_unit)
        if name not in Y_obs:
            raise KeyError(f"Y_obs missing '{name}' required for projection "
                           f"{spec.name!r}")
        return Y_obs[name]

    x_vals = _field(spec.x_field, spec.x_field_source)
    y_vals = _field(spec.y_field, spec.y_field_source)
    mask = feasible & np.isfinite(x_vals) & np.isfinite(y_vals)
    pts = np.column_stack([x_vals[mask], y_vals[mask]])
    if mask.sum() < 4:
        return {
            "method": "obs_alpha_shape", "projection": spec.name,
            "status": "insufficient_feasible_points",
            "n_feasible": int(mask.sum()),
            "boundary_vertices": np.zeros((0, 2)),
            "edges": np.zeros((0, 2), dtype=int),
            "n_components": 0,
        }
    res = alpha_shape_boundary_v2(pts, alpha_frac=spec.alpha_frac)
    return {
        "method": "obs_alpha_shape", "projection": spec.name, "status": "ok",
        "n_feasible": int(mask.sum()),
        "boundary_vertices": pts[res.vertex_idx],
        "edges": res.edges, "n_components": int(res.n_components),
        "alpha_frac": spec.alpha_frac,
        "_points": pts,
    }
