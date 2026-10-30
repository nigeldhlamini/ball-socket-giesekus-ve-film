"""Analysis F — BO-constructed AR vs exhaustive AR side-by-side (Chapter 8 BO).

Target figure: figures/bo/fig_bo_ar_overlay.pdf

For the baseline BO cell at N_BO=500, reconstruct the BO-feasible set
by:
  1. Re-fitting a SurrogateManager on the full 500-observation trajectory
     (Matern-5/2, N_init=18, seed 2026)
  2. Drawing ~50,000 Sobol points in input space
  3. Predicting posterior means of the three constraint GPs (g1_gap,
     g2_thermal, g3_validity) at each point
  4. Classifying as BO-feasible if all three means are >= 0
  5. Computing posterior means of the performance GPs (h_min, T_max,
     P_loss) at the BO-feasible points; F_applied is a closed-form
     function of (k_F, p_s) so it is computed analytically
  6. Alpha-shape each of the three performance projections of the
     BO-feasible set, using the same alpha selected for the exhaustive
     boundary

Three-panel overlay:
    exhaustive AR boundary (solid black)
    BO-constructed AR boundary (dashed black)
    BO observation points (small markers)

This figure makes Chapter 8's "BO reproduces the AR with O(500)
evaluations" claim concrete — and the gap between the two boundaries
visualises the residual Hausdorff distance.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize

_REPO = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ar_bo_campaign import INPUT_SPACE, alpha_shape_boundary_v2, hydrostatic_floor  # noqa: E402
from src.bo.surrogates import SurrogateManager  # noqa: E402
from src.figure_utils import safe_savefig  # noqa: E402

REPO = _REPO
_CORR = REPO / "results/recompute/ar_exhaustive_N131072_seed0_corrected.parquet"
EXHAUSTIVE_PARQUET = _CORR if _CORR.exists() else REPO / "results/ar_exhaustive_N131072_seed0.parquet"
BASELINE_DIR = REPO / "logs/bo_phase9b_ofat/matern52_init18_seed2026"
OUT_DIR = REPO / "figures/bo_corrected"

N_GRID = 50_000   # Sobol points in unit cube
SOBOL_SEED = 314  # not 0 (baseline) or 1 (Saltelli) or 2 (Newtonian-to-be)

# Pre-computed: the hydrostatic floor coefficient
# F_applied = k_F * hydrostatic_floor(p_s); use INPUT_SPACE-default geometry
F_HYD_PER_PS = hydrostatic_floor(p_s=1.0)   # m^2 (since hydrostatic_floor scales linearly in p_s)

# Projection specs: must match exhaustive build_ar_maps choices
PROJ = {
    "F_hmin": {
        "x_field": "F_applied",   # derived analytically
        "y_field": "h_min",       # h_min regression GP
        "x_scale_plot": 1e-3,     # N -> kN
        "y_scale_plot": 1e6,      # m -> micron
        "x_log": True, "y_log": True,
        "x_label": r"$F$ [kN]",
        "y_label": r"$h_{\min}$ [$\mu$m]",
        "alpha_frac": 2.0,
        "title": r"(a) $F$--$h_{\min}$",
    },
    "hmin_Tmax": {
        "x_field": "h_min",
        "y_field": "T_max",
        "x_scale_plot": 1e6,
        "y_scale_plot": 1.0,      # K -> degC handled below
        "x_log": True, "y_log": False,
        "x_label": r"$h_{\min}$ [$\mu$m]",
        "y_label": r"$T_{\max}$ [$^\circ$C]",
        "alpha_frac": 10.0,
        "title": r"(b) $h_{\min}$--$T_{\max}$",
    },
    "Ploss_F": {
        "x_field": "P_loss",
        "y_field": "F_applied",
        "x_scale_plot": 1e-3,
        "y_scale_plot": 1e-3,
        "x_log": True, "y_log": True,
        "x_label": r"$\mathcal{P}_\mathrm{loss}$ [kW]",
        "y_label": r"$F$ [kN]",
        "alpha_frac": 20.0,
        "title": r"(c) $\mathcal{P}_\mathrm{loss}$--$F$",
    },
}


# --------------------------------------------------------------------------
# Polygon helper (copied from build_ar_maps to keep this script standalone)
# --------------------------------------------------------------------------

def _build_polygons_from_edges(points: np.ndarray,
                               edges: np.ndarray) -> list[Polygon]:
    if len(edges) == 0:
        return []
    segments = [
        LineString([tuple(points[int(i)]), tuple(points[int(j)])])
        for i, j in edges.tolist()
    ]
    polys = list(polygonize(MultiLineString(segments)))
    if not polys:
        return []
    poly_exts = [Polygon(p.exterior) for p in polys]
    keep = []
    for k, (p, p_ext) in enumerate(zip(polys, poly_exts)):
        is_inner_twin = False
        for j, other_ext in enumerate(poly_exts):
            if j == k:
                continue
            if other_ext.contains(p_ext) and other_ext.area > p_ext.area:
                is_inner_twin = True
                break
        if not is_inner_twin:
            keep.append(p)
    return keep


def _alpha_shape_boundary_poly(pts: np.ndarray,
                               alpha_frac: float) -> list[Polygon]:
    """Wrap alpha_shape_boundary_v2 + polygonize into a list of Polygons."""
    res = alpha_shape_boundary_v2(pts, alpha_frac=alpha_frac)
    return _build_polygons_from_edges(pts, res.edges)


# --------------------------------------------------------------------------
# Surrogate refit on baseline data
# --------------------------------------------------------------------------

def _refit_baseline_surrogate() -> tuple[SurrogateManager, np.ndarray]:
    """Re-fit the SurrogateManager on the full 500-observation baseline
    trajectory. Returns (sm, X_obs_unit)."""
    iter_records = json.load(open(BASELINE_DIR / "iter_records.json"))
    X_obs = np.array([r["X_unit"] for r in iter_records])
    converged = np.array([r["converged"] for r in iter_records], dtype=bool)
    traj = np.load(BASELINE_DIR / "trajectory.npz")
    output_names = ["h_min", "T_max", "P_loss", "g1_gap", "g2_thermal", "g3_validity"]

    sm = SurrogateManager(
        d=8,
        objective_names=["h_min"],
        constraint_names=["g1_gap", "g2_thermal", "g3_validity"],
        auxiliary_names=["T_max", "P_loss"],
        kernel="matern52", refit_every=1,
    )
    X_t = torch.tensor(X_obs, dtype=torch.float64)
    Y_t = {n: torch.tensor(traj[f"Y_{n}"], dtype=torch.float64)
           for n in output_names}
    conv_t = torch.tensor(converged, dtype=torch.bool)
    sm.fit(X=X_t, Y=Y_t, converged=conv_t)
    return sm, X_obs


# --------------------------------------------------------------------------
# Input-space helpers
# --------------------------------------------------------------------------

def _unit_to_physical(u: np.ndarray) -> dict[str, np.ndarray]:
    """Map (N, d) unit-cube points to physical inputs using INPUT_SPACE.from_unit."""
    return {dim.name: dim.from_unit(u[:, i]) for i, dim in enumerate(INPUT_SPACE)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: re-fit baseline surrogate on N=500...")
    sm, X_obs_unit = _refit_baseline_surrogate()
    print(f"  fitted GPs: {list(sm._gps.keys())}")

    print(f"Step 2: generate {N_GRID} Sobol points in unit cube (seed={SOBOL_SEED})...")
    sob = qmc.Sobol(d=8, scramble=True, seed=SOBOL_SEED)
    # qmc.Sobol.random(n) gives an unbalanced base; use random_base2 for
    # better balance — but N_GRID = 50,000 isn't a power of 2 so just use
    # random(n).
    U = sob.random(N_GRID)
    print(f"  U.shape = {U.shape}")

    print("Step 3: evaluate constraint GPs at the grid...")
    X_t = torch.tensor(U, dtype=torch.float64)
    with torch.no_grad():
        g1_mean, _ = sm.predict("g1_gap", X_t)
        g2_mean, _ = sm.predict("g2_thermal", X_t)
        g3_mean, _ = sm.predict("g3_validity", X_t)
        h_min_mean, _ = sm.predict("h_min", X_t)
        t_max_mean, _ = sm.predict("T_max", X_t)
        p_loss_mean, _ = sm.predict("P_loss", X_t)
    g1 = g1_mean.cpu().numpy()
    g2 = g2_mean.cpu().numpy()
    g3 = g3_mean.cpu().numpy()
    h_pred = h_min_mean.cpu().numpy()
    t_pred = t_max_mean.cpu().numpy()
    p_pred = p_loss_mean.cpu().numpy()

    bo_feas = (g1 >= 0) & (g2 >= 0) & (g3 >= 0)
    print(f"  BO-feasible: {bo_feas.sum()}/{N_GRID} "
          f"({100 * bo_feas.mean():.1f}%)")
    print(f"    g1>=0: {(g1>=0).sum()}, g2>=0: {(g2>=0).sum()}, "
          f"g3>=0: {(g3>=0).sum()}")

    # F_applied analytic from inputs
    phys = _unit_to_physical(U)
    F_app_pred = phys["k_F"] * F_HYD_PER_PS * phys["p_s"]
    print(f"  F_applied analytic: min={F_app_pred.min():.0f} N, "
          f"max={F_app_pred.max():.0f} N")

    # Sanity check: compare h_min posterior at observation locations
    # against the training values (should be near-exact since we fitted
    # there). This catches refit drift.
    with torch.no_grad():
        h_at_obs, _ = sm.predict("h_min", torch.tensor(X_obs_unit, dtype=torch.float64))
    traj = np.load(BASELINE_DIR / "trajectory.npz")
    Y_h_obs = traj["Y_h_min"]
    converged_at_obs = np.array(
        [r["converged"] for r in json.load(open(BASELINE_DIR / "iter_records.json"))],
        dtype=bool,
    )
    pred_obs = h_at_obs.cpu().numpy()
    diff = (pred_obs - Y_h_obs)[converged_at_obs]
    rel = np.abs(diff) / (np.abs(Y_h_obs[converged_at_obs]) + 1e-30)
    print(f"  Refit sanity check (h_min at training pts, converged only): "
          f"median rel-err = {np.median(rel):.2e}, max = {rel.max():.2e}")

    print("Step 4: load exhaustive data and compute boundaries...")
    df_exh = pd.read_parquet(EXHAUSTIVE_PARQUET)
    df_conv_feas = df_exh[(df_exh["converged"]) & (df_exh["feasible"])].copy()
    print(f"  exhaustive: {len(df_conv_feas)} converged-feasible rows")

    # Per-projection boundary computation
    proj_data = {}
    for key, spec in PROJ.items():
        x_col = spec["x_field"]
        y_col = spec["y_field"]
        # Exhaustive
        valid_exh = (np.isfinite(df_conv_feas[x_col].values)
                     & np.isfinite(df_conv_feas[y_col].values))
        if x_col == "F_applied":
            x_exh = df_conv_feas["F_applied"].values
        else:
            x_exh = df_conv_feas[x_col].values
        y_exh = df_conv_feas[y_col].values
        exh_pts = np.column_stack([x_exh[valid_exh], y_exh[valid_exh]])
        # log-axes: exclude non-positive
        if spec["x_log"]:
            mask = exh_pts[:, 0] > 0
            exh_pts = exh_pts[mask]
        if spec["y_log"]:
            mask = exh_pts[:, 1] > 0
            exh_pts = exh_pts[mask]
        print(f"  [{key}] exhaustive AR pts: {len(exh_pts)}")

        # BO-feasible
        # x BO-pred:
        x_bo_full = (F_app_pred if x_col == "F_applied"
                     else h_pred if x_col == "h_min"
                     else p_pred if x_col == "P_loss"
                     else t_pred)
        y_bo_full = (F_app_pred if y_col == "F_applied"
                     else h_pred if y_col == "h_min"
                     else p_pred if y_col == "P_loss"
                     else t_pred)
        x_bo = x_bo_full[bo_feas]
        y_bo = y_bo_full[bo_feas]
        finite_bo = np.isfinite(x_bo) & np.isfinite(y_bo)
        bo_pts = np.column_stack([x_bo[finite_bo], y_bo[finite_bo]])
        if spec["x_log"]:
            bo_pts = bo_pts[bo_pts[:, 0] > 0]
        if spec["y_log"]:
            bo_pts = bo_pts[bo_pts[:, 1] > 0]
        print(f"  [{key}] BO-feasible AR pts (after finiteness filter): "
              f"{len(bo_pts)}")

        try:
            exh_polys = _alpha_shape_boundary_poly(exh_pts, spec["alpha_frac"])
        except Exception as exc:
            print(f"    exhaustive alpha-shape failed: {exc}")
            exh_polys = []
        try:
            bo_polys = _alpha_shape_boundary_poly(bo_pts, spec["alpha_frac"])
        except Exception as exc:
            print(f"    BO alpha-shape failed: {exc}")
            bo_polys = []

        proj_data[key] = {
            "exh_pts": exh_pts,
            "exh_polys": exh_polys,
            "bo_pts": bo_pts,
            "bo_polys": bo_polys,
        }

    print("Step 5: render figure...")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    # Sample-point overlay: convert observation X_unit to physical inputs
    obs_phys = _unit_to_physical(X_obs_unit)
    obs_F_applied = obs_phys["k_F"] * F_HYD_PER_PS * obs_phys["p_s"]
    Y_T_max_obs = traj["Y_T_max"]
    Y_P_loss_obs = traj["Y_P_loss"]
    # Filter chaotic-Picard cosmic values from the OBSERVED training set.
    # The trajectory has a few rows with Y_T_max ~ 1e20 K and
    # Y_P_loss ~ 1e23 W from chaotic-Picard regions; these are real
    # observations the BO collected but they are unphysical and would
    # blow up the overplot scale. Keep T_max < 1000 K (T_FAIL=383K so
    # 1000 K is far beyond physical operation) and P_loss < 1e7 W
    # (10 MW; the bearing is ~10 cm OD).
    cosmic_obs = ((Y_T_max_obs > 1000.0)
                  | (np.abs(Y_P_loss_obs) > 1e7)
                  | (~np.isfinite(Y_T_max_obs))
                  | (~np.isfinite(Y_P_loss_obs)))
    print(f"  Observed cosmic-Picard rows: {int(cosmic_obs.sum())}/{len(Y_T_max_obs)} "
          f"(masked from overplot)")
    obs_data = {
        "F_applied": obs_F_applied,
        "h_min": Y_h_obs,
        "T_max": np.where(cosmic_obs, np.nan, Y_T_max_obs),
        "P_loss": np.where(cosmic_obs, np.nan, Y_P_loss_obs),
    }

    for ax, (key, spec) in zip(axes, PROJ.items()):
        d = proj_data[key]
        # Convert all points to plot units (special handling for T_max in degC)
        def to_plot(arr, col):
            if col == "T_max":
                return arr - 273.15
            return arr * (spec["x_scale_plot"] if col == spec["x_field"]
                          else spec["y_scale_plot"])
        x_exh_plot = (d["exh_pts"][:, 0] - 273.15
                      if spec["x_field"] == "T_max"
                      else d["exh_pts"][:, 0] * spec["x_scale_plot"])
        y_exh_plot = (d["exh_pts"][:, 1] - 273.15
                      if spec["y_field"] == "T_max"
                      else d["exh_pts"][:, 1] * spec["y_scale_plot"])
        x_bo_plot = (d["bo_pts"][:, 0] - 273.15
                     if spec["x_field"] == "T_max"
                     else d["bo_pts"][:, 0] * spec["x_scale_plot"])
        y_bo_plot = (d["bo_pts"][:, 1] - 273.15
                     if spec["y_field"] == "T_max"
                     else d["bo_pts"][:, 1] * spec["y_scale_plot"])

        # Faint background: exhaustive feasible scatter
        ax.scatter(x_exh_plot, y_exh_plot, s=1, c="#cfcfcf", alpha=0.25,
                   linewidths=0, rasterized=True,
                   label="Exhaustive feasible (8.1)")

        # Faint BO-feasible scatter
        ax.scatter(x_bo_plot, y_bo_plot, s=1, c="#f5b88a", alpha=0.20,
                   linewidths=0, rasterized=True,
                   label="BO-feasible (posterior)")

        # Exhaustive boundary (solid black)
        for i, poly in enumerate(d["exh_polys"]):
            xs, ys = poly.exterior.xy
            xs_arr = np.asarray(xs, dtype=float)
            ys_arr = np.asarray(ys, dtype=float)
            xs_p = (xs_arr - 273.15 if spec["x_field"] == "T_max"
                    else xs_arr * spec["x_scale_plot"])
            ys_p = (ys_arr - 273.15 if spec["y_field"] == "T_max"
                    else ys_arr * spec["y_scale_plot"])
            ax.plot(xs_p, ys_p, color="black", linewidth=1.6,
                    label="Exhaustive AR" if i == 0 else None)

        # BO boundary (dashed red)
        for i, poly in enumerate(d["bo_polys"]):
            xs, ys = poly.exterior.xy
            xs_arr = np.asarray(xs, dtype=float)
            ys_arr = np.asarray(ys, dtype=float)
            xs_p = (xs_arr - 273.15 if spec["x_field"] == "T_max"
                    else xs_arr * spec["x_scale_plot"])
            ys_p = (ys_arr - 273.15 if spec["y_field"] == "T_max"
                    else ys_arr * spec["y_scale_plot"])
            ax.plot(xs_p, ys_p, color="#cc3324", linewidth=1.6,
                    linestyle="--",
                    label="BO-constructed AR" if i == 0 else None)

        # BO observation points
        x_obs = (obs_data[spec["x_field"]] - 273.15
                 if spec["x_field"] == "T_max"
                 else obs_data[spec["x_field"]] * spec["x_scale_plot"])
        y_obs = (obs_data[spec["y_field"]] - 273.15
                 if spec["y_field"] == "T_max"
                 else obs_data[spec["y_field"]] * spec["y_scale_plot"])
        fin = np.isfinite(x_obs) & np.isfinite(y_obs)
        ax.scatter(x_obs[fin], y_obs[fin], s=10, c="#1f4e79",
                   marker="o", linewidths=0.4, edgecolor="white",
                   alpha=0.9, label=r"BO observations ($N=500$)",
                   rasterized=True, zorder=5)

        if spec["x_log"]:
            ax.set_xscale("log")
        if spec["y_log"]:
            ax.set_yscale("log")
        ax.set_xlabel(spec["x_label"])
        ax.set_ylabel(spec["y_label"])
        ax.set_title(spec["title"], fontsize=9)
        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    fig.suptitle(r"BO-constructed AR (dashed red) vs exhaustive AR "
                 r"(solid black) at $N_\mathrm{BO}=500$",
                 fontsize=10)
    fig.tight_layout()
    pdf_path = OUT_DIR / "fig_bo_ar_overlay.pdf"
    png_path = OUT_DIR / "fig_bo_ar_overlay.png"
    pdf_path = safe_savefig(fig, pdf_path)
    png_path = safe_savefig(fig, png_path, dpi=200)
    plt.close(fig)
    print(f"  Wrote {pdf_path}")
    print(f"  Wrote {png_path}")

    # Manifest
    parquet_sha = hashlib.sha256(EXHAUSTIVE_PARQUET.read_bytes()).hexdigest()[:16]
    manifest = {
        "analysis": "F",
        "description": "BO-constructed AR vs exhaustive AR side-by-side, "
                       "three performance projections at N_BO=500",
        "chapter": "8 (BO, new numbering)",
        "input_exhaustive_parquet": str(EXHAUSTIVE_PARQUET),
        "input_exhaustive_sha256_prefix": parquet_sha,
        "bo_baseline_dir": str(BASELINE_DIR),
        "bo_config": {
            "kernel": "matern52",
            "n_init": 18,
            "seed": 2026,
            "refit_every": 5,
            "acquisition": "entropy_only",
            "classifier": True,
        },
        "n_grid": N_GRID,
        "sobol_seed": SOBOL_SEED,
        "projections": {
            k: {
                "alpha_frac": spec["alpha_frac"],
                "n_exhaustive_pts": int(len(proj_data[k]["exh_pts"])),
                "n_bo_pts": int(len(proj_data[k]["bo_pts"])),
            }
            for k, spec in PROJ.items()
        },
        "figures": {"pdf": str(pdf_path), "png": str(png_path)},
        "reproducibility_note": (
            "BO baseline cell: Matern-5/2 ARD kernel, N_init=18, seed=2026, "
            "refit_every=5, entropy_only acquisition with multiplicative "
            "classifier gating. Surrogate is re-fitted in this analysis "
            "by replaying the full 500-observation trajectory through "
            "SurrogateManager.fit; result is bit-identical to the original "
            "run modulo BoTorch's L-BFGS multi-restart stochasticity."
        ),
    }
    manifest_path = OUT_DIR / "fig_bo_ar_overlay_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote {manifest_path}")

    # Findings summary with area coverage
    print()
    print("=== FINDINGS SUMMARY (Analysis F) ===")
    print(f"BO-feasible at N_BO=500: {bo_feas.sum()}/{N_GRID} "
          f"({100 * bo_feas.mean():.1f}%) of input space "
          f"(g1: {(g1>=0).sum()/N_GRID:.1%}, "
          f"g2: {(g2>=0).sum()/N_GRID:.1%}, "
          f"g3: {(g3>=0).sum()/N_GRID:.1%})")
    coverage_summary = {}
    for key, spec in PROJ.items():
        d = proj_data[key]
        exh_area = sum(p.area for p in d["exh_polys"])
        bo_area = sum(p.area for p in d["bo_polys"])
        from shapely.ops import unary_union
        if d["exh_polys"] and d["bo_polys"]:
            exh_union = unary_union(d["exh_polys"])
            bo_union = unary_union(d["bo_polys"])
            intersect_area = exh_union.intersection(bo_union).area
            # IoU and coverage ratios
            iou = intersect_area / unary_union([exh_union, bo_union]).area
            recall = intersect_area / exh_union.area
            precision = intersect_area / bo_union.area if bo_union.area > 0 else float("nan")
        else:
            iou = recall = precision = float("nan")
        coverage_summary[key] = {
            "exhaustive_area": float(exh_area),
            "bo_area": float(bo_area),
            "area_ratio_bo_over_exh": (
                float(bo_area / exh_area) if exh_area > 0 else None),
            "IoU": float(iou),
            "recall": float(recall),
            "precision": float(precision),
        }
        print(f"  [{key}] exh_area={exh_area:.3e}, bo_area={bo_area:.3e}, "
              f"BO/exh={bo_area/exh_area:.2f}, IoU={iou:.3f}, "
              f"recall={recall:.3f}, precision={precision:.3f}")
    # Attach to manifest
    manifest["coverage"] = coverage_summary
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
