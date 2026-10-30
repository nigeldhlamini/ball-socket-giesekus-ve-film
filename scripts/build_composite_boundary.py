"""A1: composite physical AR boundary (parameter-free, physically labelled).

Motivation: the alpha-shape boundary rewards uniform coverage and is the
reconstruction bottleneck (Task-6 oracle ceiling IoU(500)~0.43-0.49; Task-7
BO-vs-random worse on 2/3 projections). The composite boundary is
parameter-free and physically labelled:

  1. FEASIBILITY BOUNDARY: a feasible sample is a boundary point if any of its
     k=10 nearest neighbours (in UNIT-CUBE INPUT space, over all 131,072 rows)
     is infeasible or non-converged. Each boundary point is labelled by the
     neighbour-majority cause: g1 (gap), g3 (validity), lift-off. g2 (thermal)
     is empty on corrected data -- asserted and recorded.
  2. PARETO FRONTIER: non-dominated feasible set per projection's optimisation
     sense (Kung-style O(n log n)).
  3. COMPOSITE = feasibility boundary + Pareto arc.

Per-projection JSON manifest (label counts, arc lengths) + overlay figures
(composite coloured by cause + distinct Pareto arc vs the alpha-shape boundary,
feasible cloud greyed). Stability: k in {6,10,15} boundary-set Jaccard; bootstrap
B=50 composite d_H band.

Output: figures/ar_corrected/composite/ + results/recompute/composite_boundary_manifest.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString
from shapely.ops import polygonize

_REPO = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ar_bo_campaign import (  # noqa: E402
    INPUT_SPACE, H_SAFE, DESQ_VALID, T_FAIL_K, hausdorff_distance,
    alpha_shape_boundary_v2,
)
from scripts.build_ar_maps import PROJECTIONS  # noqa: E402

PARQUET = _REPO / "results/recompute/ar_exhaustive_N131072_seed0_corrected.parquet"
OUTDIR = _REPO / "figures/ar_corrected/composite"
MANIFEST = _REPO / "results/recompute/composite_boundary_manifest.json"
PROJ = ["F_hmin", "hmin_Tmax", "Ploss_F", "Qleak_F"]
LABEL_COLOR = {"g1": "#e89a3c", "g3": "#3c6ee8", "lift_off": "#7a7a7a", "g2": "#cc3324"}
# Rim-trace (silhouette) cause colours: gap (g1) / validity (g3) / lift-off / Pareto.
CAUSE_NAME = {"g1": "gap", "g3": "validity", "lift_off": "lift-off", "pareto": "Pareto (trade-off)"}
CAUSE_COLOR = {"g1": "#e89a3c", "g3": "#3c6ee8", "lift_off": "#7a7a7a", "pareto": "#111111"}
K_MAIN = 10
K_STAB = [6, 15]
B_BOOT = 50
SEED = 20260709
K_SIL = 25           # nearest labelled composite points per rim segment
SIL_THRESH_FRAC = 0.06   # rim segment is Pareto if kNN mean dist > frac x plane diagonal


def _unit_matrix(df):
    return np.clip(np.column_stack([d.to_unit(df[d.name].to_numpy(float)) for d in INPUT_SPACE]),
                   0.0, 1.0)


def _cause(row):
    """Physical cause label for an infeasible/non-converged row."""
    if not bool(row["converged"]):
        return "lift_off"
    if row["h_min"] < H_SAFE:
        return "g1"
    if row["De_sq"] > DESQ_VALID:
        return "g3"
    if np.isfinite(row["T_max"]) and row["T_max"] > T_FAIL_K:
        return "g2"
    return "g3"   # residual (validity dominant)


def _boundary_mask_labels(U, feas, cause_idx, k):
    """Boundary = feasible row with >=1 infeasible/nonconv neighbour (excl self).
    Label = majority cause among the infeasible neighbours."""
    tree = cKDTree(U)
    _, nbr = tree.query(U, k=k + 1)          # includes self at col 0
    nbr = nbr[:, 1:]
    is_bad = ~feas                            # infeasible OR non-converged
    bmask = np.zeros(len(U), bool)
    labels = np.empty(len(U), object)
    for i in np.where(feas)[0]:
        bad_nbr = nbr[i][is_bad[nbr[i]]]
        if bad_nbr.size:
            bmask[i] = True
            causes = cause_idx[bad_nbr]
            vals, cnts = np.unique(causes, return_counts=True)
            labels[i] = vals[np.argmax(cnts)]
    return bmask, labels


def _pareto(xy, sense):
    """Non-dominated frontier indices for 2D points under (x_sense,y_sense)."""
    sign = np.array([1.0 if s == "max" else -1.0 for s in sense])
    s = xy * sign
    order = np.argsort(-s[:, 0], kind="mergesort")
    keep, best_y = [], -np.inf
    for idx in order:
        if s[idx, 1] > best_y:
            keep.append(idx); best_y = s[idx, 1]
    return np.array(keep)


def _plotted(spec, x, y):
    xp = np.log10(x) if spec.x_log else x.astype(float)
    yp = np.log10(y) if spec.y_log else y.astype(float)
    return xp, yp


def _arc_len(pts):
    if len(pts) < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt((d ** 2).sum(1))))


def _jaccard(a, b):
    a, b = set(np.where(a)[0]), set(np.where(b)[0])
    return len(a & b) / len(a | b) if (a | b) else 1.0


def _ordered_rim(pts, edges):
    """Order an alpha-shape edge set into the vertices of its largest closed
    loop (corrected shapes are single-component, no holes -> one loop).
    Returns (N,2) ordered vertices (closed) or empty array."""
    if len(edges) == 0:
        return np.empty((0, 2))
    segs = [LineString([tuple(pts[int(i)]), tuple(pts[int(j)])]) for i, j in edges]
    polys = list(polygonize(MultiLineString(segs)))
    if not polys:
        return np.empty((0, 2))
    poly = max(polys, key=lambda p: p.area)     # main loop
    return np.asarray(poly.exterior.coords)


def _colour_rim(rim, bd_pts, bd_labels, diag):
    """Label each rim segment by the majority cause of the K_SIL nearest
    labelled composite boundary points to the segment midpoint; segments whose
    neighbourhood is farther than SIL_THRESH_FRAC x diag are Pareto arcs.
    Returns list of (p0, p1, cause_key) and a per-cause segment count."""
    thresh = SIL_THRESH_FRAC * diag
    out, counts = [], {}
    if len(rim) < 2 or len(bd_pts) == 0:
        return out, counts, thresh
    tree = cKDTree(bd_pts)
    k = min(K_SIL, len(bd_pts))
    for a, b in zip(rim[:-1], rim[1:]):
        mid = 0.5 * (a + b)
        d, idx = tree.query(mid, k=k)
        d = np.atleast_1d(d); idx = np.atleast_1d(idx)
        if float(np.mean(d)) > thresh:
            cause = "pareto"
        else:
            labs = bd_labels[idx]
            vals, cnts = np.unique(labs, return_counts=True)
            cause = str(vals[np.argmax(cnts)])
        out.append((a, b, cause))
        counts[cause] = counts.get(cause, 0) + 1
    return out, counts, thresh


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQUET).reset_index(drop=True)
    feas = df["feasible"].fillna(False).to_numpy(bool)
    U = _unit_matrix(df)
    # per-row cause label (only meaningful for non-feasible rows)
    bad = ~feas
    causes = np.full(len(df), "", object)
    sub = df.loc[bad, ["converged", "h_min", "De_sq", "T_max"]]
    causes[np.where(bad)[0]] = [_cause(r) for _, r in sub.iterrows()]

    bmask, blabels = _boundary_mask_labels(U, feas, causes, K_MAIN)
    n_g2 = int(np.sum(blabels[bmask] == "g2"))
    assert n_g2 == 0, f"g2 boundary points found ({n_g2}) -- expected 0 on corrected data"

    # k-stability
    jac = {}
    for k in K_STAB:
        bm_k, _ = _boundary_mask_labels(U, feas, causes, k)
        jac[k] = round(_jaccard(bmask, bm_k), 4)

    rng = np.random.default_rng(SEED)
    # precompute B bootstrap boundary sets ONCE (global input-space boundary),
    # reused across all projections (avoids 4x redundant KDTree passes).
    print(f"precomputing {B_BOOT} bootstrap boundaries...", flush=True)
    BOOT = []
    for _ in range(B_BOOT):
        samp = rng.choice(len(df), len(df), replace=True)
        bm_s, _ = _boundary_mask_labels(U[samp], feas[samp], causes[samp], K_MAIN)
        BOOT.append((samp, bm_s))

    manifest = {"parquet": str(PARQUET.name), "n_feasible": int(feas.sum()),
                "n_boundary_points": int(bmask.sum()),
                "boundary_label_counts": {c: int(np.sum(blabels[bmask] == c))
                                          for c in ("g1", "g3", "lift_off")},
                "g2_boundary_points": n_g2, "k_main": K_MAIN,
                "k_stability_jaccard_vs_k10": jac, "projections": {}}

    for key in PROJ:
        spec = PROJECTIONS[key]
        x = pd.to_numeric(df[spec.x_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[spec.y_col], errors="coerce").to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        if spec.x_log: finite &= x > 0
        if spec.y_log: finite &= y > 0
        fe = feas & finite
        bd = bmask & finite
        xp, yp = _plotted(spec, x, y)

        # Pareto on feasible-finite
        fe_idx = np.where(fe)[0]
        par_local = _pareto(np.column_stack([xp[fe_idx], yp[fe_idx]]), spec.pareto_sense)
        par_idx = fe_idx[par_local]
        par_xy = np.column_stack([xp[par_idx], yp[par_idx]])
        par_order = np.argsort(par_xy[:, 0]); par_xy = par_xy[par_order]

        # composite point cloud (plotted coords)
        comp = np.column_stack([np.r_[xp[bd], par_xy[:, 0]], np.r_[yp[bd], par_xy[:, 1]]])

        # alpha-shape reference boundary (for contrast), ordered into ONE closed
        # rim (plotted coords); reuse its polygon as the silhouette to colour.
        try:
            fe_pts = np.column_stack([xp[fe_idx], yp[fe_idx]])
            ash = alpha_shape_boundary_v2(fe_pts, alpha_frac=spec_alpha(key), log_axes=(False, False))
            rim = _ordered_rim(fe_pts, ash.edges)
        except Exception:
            rim = np.empty((0, 2))

        # bootstrap composite d_H band (reuse the precomputed boundary sets)
        dH = []
        for samp, bm_s in BOOT:
            xs, ys = xp[samp], yp[samp]
            fin_s = np.isfinite(xs) & np.isfinite(ys)
            bset = bm_s & fin_s
            if bset.sum() < 8:
                continue
            comp_s = np.column_stack([xs[bset], ys[bset]])
            dH.append(hausdorff_distance(comp_s, comp))
        dH = np.array(dH) if dH else np.array([np.nan])

        # cause-coloured rim trace: label each silhouette segment by the
        # nearest labelled composite boundary points.
        bd_pts = np.column_stack([xp[bd], yp[bd]])
        bd_labels = blabels[bd].astype(str)
        allx, ally = xp[fe_idx], yp[fe_idx]
        diag = float(np.hypot(allx.max() - allx.min(), ally.max() - ally.min())) if fe.sum() else 1.0
        rim_segs, sil_counts, sil_thresh = _colour_rim(rim, bd_pts, bd_labels, diag)

        # figure
        fig, ax = plt.subplots(figsize=(7, 5.2))
        ax.scatter(allx, ally, s=3, c="#e8e8e8", linewidths=0, rasterized=True, label="feasible")
        # labelled composite point cloud: smaller, transparent (colour = cause)
        for lab in ("lift_off", "g1", "g3"):
            m = bd & (blabels == lab)
            if m.any():
                ax.scatter(xp[m], yp[m], s=4, c=LABEL_COLOR[lab], alpha=0.5,
                           linewidths=0, rasterized=True)
        # cause-coloured rim silhouette (thick, under the alpha-shape line)
        seen = set()
        for a, b, cause in rim_segs:
            lbl = None
            if cause not in seen:
                lbl = f"boundary cause: {CAUSE_NAME[cause]}"; seen.add(cause)
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=CAUSE_COLOR[cause],
                    lw=3.0, solid_capstyle="round", label=lbl, zorder=4)
        # Pareto arc (as before)
        if len(par_xy):
            ax.plot(par_xy[:, 0], par_xy[:, 1], "-", color="#111111", lw=1.4,
                    label="Pareto arc", zorder=5)
        # alpha-shape as ONE thin solid line (single legend entry)
        if len(rim):
            ax.plot(rim[:, 0], rim[:, 1], "-", color="#9a3fbf", lw=0.8,
                    label=r"$\alpha$-shape boundary", zorder=6)
        ax.set_xlabel(("log10 " if spec.x_log else "") + spec.x_col)
        ax.set_ylabel(("log10 " if spec.y_log else "") + spec.y_col)
        ax.set_title(f"Composite physical boundary: {key}")
        ax.legend(fontsize=7, loc="best"); ax.grid(True, ls=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(OUTDIR / f"fig_composite_{key}.pdf"); fig.savefig(OUTDIR / f"fig_composite_{key}.png", dpi=180)
        plt.close(fig)

        # sanity: thermal (g2) must never colour the rim; record the distribution.
        assert "g2" not in sil_counts, f"{key}: g2/thermal coloured the rim -- STOP"
        sil_named = {CAUSE_NAME.get(c, c): int(n) for c, n in sil_counts.items()}
        print(f"[{key}] rim segments by cause: {sil_named}", flush=True)

        manifest["projections"][key] = {
            "n_feasible_finite": int(fe.sum()), "n_boundary_finite": int(bd.sum()),
            "n_pareto_points": int(len(par_idx)),
            "pareto_arc_len_plotted": round(_arc_len(par_xy), 4),
            "pareto_sense": list(spec.pareto_sense),
            "boundary_label_counts": {c: int(np.sum((bd) & (blabels == c))) for c in ("g1", "g3", "lift_off")},
            "bootstrap_dH": {"median": float(np.nanmedian(dH)), "p95": float(np.nanpercentile(dH, 95)),
                             "B": B_BOOT},
            "pareto_near_degenerate": bool(len(par_idx) <= 5 or _arc_len(par_xy) < 0.05),
            "rim_trace": {"k_sil": K_SIL, "pareto_thresh_frac": SIL_THRESH_FRAC,
                          "pareto_thresh_plotted": round(sil_thresh, 4),
                          "n_rim_segments": int(len(rim_segs)),
                          "segments_by_cause": {CAUSE_NAME.get(c, c): int(n)
                                                for c, n in sil_counts.items()}},
        }
        print(f"[{key}] boundary={int(bd.sum())} (g1 {manifest['projections'][key]['boundary_label_counts']['g1']}"
              f"/g3 {manifest['projections'][key]['boundary_label_counts']['g3']}"
              f"/lift {manifest['projections'][key]['boundary_label_counts']['lift_off']}) "
              f"pareto={len(par_idx)} dH_med={np.nanmedian(dH):.3f}", flush=True)

    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\ng2 boundary points: {n_g2} (asserted 0) | k-Jaccard {jac}")
    print(f"wrote {MANIFEST} + figures/ar_corrected/composite/")


def spec_alpha(key):
    return {"F_hmin": 20.0, "hmin_Tmax": 10.0, "Ploss_F": 20.0, "Qleak_F": 20.0}[key]


def _close(v):
    v = np.vstack([v, v[:1]])
    return v[:, 0], v[:, 1]


if __name__ == "__main__":
    main()
