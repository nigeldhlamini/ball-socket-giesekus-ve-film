"""A2: re-targeted BO validation against the composite physical boundary (A1).

The alpha-shape metric rewards uniform coverage, which BO does not target;
BO's acquisition targets the FEASIBILITY BOUNDARY. Score each cell against the
exhaustive composite boundary (A1) on the metric BO actually optimises.

Per 500-obs cell (and random 500-row baselines, seeds 0-9):
  - observed feasibility boundary = feasible obs with >=1 infeasible neighbour
    (k=10 NN in unit-cube input space among the 500 obs); project to each plane.
  - d_H(observed boundary, exhaustive composite boundary) per projection.
  - per-label recall: fraction of exhaustive labelled boundary points within
    radius r of an observed boundary point (r = 0.04 x plane diagonal).
  - Pareto recovery: d_H(cell feasible-obs Pareto arc, exhaustive Pareto arc).
Projections: F_hmin, hmin_Tmax, Ploss_F (Q_leak not recorded in trajectory;
its Pareto is degenerate per A1). Appends to results_digest.md.

Output: results/recompute/bo_composite_validation.json
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_REPO = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ar_bo_campaign import (  # noqa: E402
    INPUT_SPACE, hydrostatic_floor, hausdorff_distance,
)
from scripts.build_composite_boundary import (  # noqa: E402
    _unit_matrix, _boundary_mask_labels, _cause, _pareto, PROJECTIONS,
)

PARQUET = _REPO / "results/recompute/ar_exhaustive_N131072_seed0_corrected.parquet"
CELLS = sorted(glob.glob(str(_REPO / "logs/bo_phase9b_ofat/*_seed20*/trajectory.npz")))
OUT = _REPO / "results/recompute/bo_composite_validation.json"
PROJ = ["F_hmin", "hmin_Tmax", "Ploss_F"]
KNN = 10
R_FRAC = 0.04
N_RANDOM = 10


def _proj_xy(spec, feats):
    """(x,y) in plotted coords from a feats dict of arrays."""
    def col(name):
        return feats[name]
    x = col(spec.x_col); y = col(spec.y_col)
    xp = np.log10(x) if spec.x_log else x
    yp = np.log10(y) if spec.y_log else y
    return xp, yp


def _obs_boundary(Uobs, feas_obs):
    """Indices of observed boundary points (feasible w/ >=1 infeasible NN)."""
    if feas_obs.sum() < 4 or (~feas_obs).sum() == 0:
        return np.array([], int)
    tree = cKDTree(Uobs)
    k = min(KNN + 1, len(Uobs))
    _, nbr = tree.query(Uobs, k=k)
    nbr = nbr[:, 1:]
    bad = ~feas_obs
    idx = [i for i in np.where(feas_obs)[0] if bad[nbr[i]].any()]
    return np.array(idx, int)


def _feats_from_unit(Xu):
    """Map unit X (n,8) -> physical feats needed for the 3 projections."""
    phys = {d.name: d.from_unit(Xu[:, j]) for j, d in enumerate(INPUT_SPACE)}
    F = phys["k_F"] * np.array([hydrostatic_floor(p) for p in phys["p_s"]])
    return phys, F


def _score_subset(Xu, feas, conv, hmin, Tmax, Ploss, F, ref, ref_par):
    """d_H + per-label recall + Pareto d_H for one subset, per projection."""
    Uobs = Xu
    feas_ok = feas & conv
    bidx = _obs_boundary(Uobs, feas_ok)
    out = {}
    feats = {"h_min": hmin, "T_max": Tmax, "P_loss": Ploss, "F_applied": F}
    for key in PROJ:
        spec = PROJECTIONS[key]
        xp, yp = _proj_xy(spec, feats)
        fin = np.isfinite(xp) & np.isfinite(yp)
        if spec.x_log: fin &= feats[spec.x_col] > 0
        if spec.y_log: fin &= feats[spec.y_col] > 0
        bsel = np.array([i for i in bidx if fin[i]], int)
        if bsel.size < 3:
            out[key] = {"d_H": float("nan"), "recall": {}, "pareto_dH": float("nan"), "n_bd": int(bsel.size)}
            continue
        obs_pts = np.column_stack([xp[bsel], yp[bsel]])
        rpts = ref[key]["pts"]
        dH = hausdorff_distance(obs_pts, rpts)
        # per-label recall
        diag = np.hypot(*(rpts.max(0) - rpts.min(0)))
        r = R_FRAC * diag
        tree = cKDTree(obs_pts)
        recall = {}
        for lab, lpts in ref[key]["by_label"].items():
            if len(lpts) == 0:
                continue
            d, _ = tree.query(lpts, k=1)
            recall[lab] = round(float((d <= r).mean()), 3)
        # Pareto recovery
        fe = np.where(feas_ok & fin)[0]
        if fe.size >= 3:
            par = fe[_pareto(np.column_stack([xp[fe], yp[fe]]), spec.pareto_sense)]
            po = np.column_stack([xp[par], yp[par]])
            po = po[np.argsort(po[:, 0])]
            pdH = hausdorff_distance(po, ref_par[key]) if len(po) >= 2 else float("nan")
        else:
            pdH = float("nan")
        out[key] = {"d_H": float(dH), "recall": recall, "pareto_dH": float(pdH), "n_bd": int(bsel.size)}
    return out


def main():
    df = pd.read_parquet(PARQUET).reset_index(drop=True)
    feas = df["feasible"].fillna(False).to_numpy(bool)
    U = _unit_matrix(df)
    bad = ~feas
    causes = np.full(len(df), "", object)
    causes[np.where(bad)[0]] = [_cause(r) for _, r in df.loc[bad, ["converged","h_min","De_sq","T_max"]].iterrows()]
    bmask, blabels = _boundary_mask_labels(U, feas, causes, KNN)

    # exhaustive reference boundary + Pareto per projection (plotted coords)
    ref = {}; ref_par = {}
    hmin_e = df["h_min"].to_numpy(float); Tmax_e = df["T_max"].to_numpy(float)
    Ploss_e = df["P_loss"].to_numpy(float); F_e = df["F_applied"].to_numpy(float)
    feats_e = {"h_min": hmin_e, "T_max": Tmax_e, "P_loss": Ploss_e, "F_applied": F_e}
    for key in PROJ:
        spec = PROJECTIONS[key]
        xp, yp = _proj_xy(spec, feats_e)
        fin = np.isfinite(xp) & np.isfinite(yp)
        if spec.x_log: fin &= feats_e[spec.x_col] > 0
        if spec.y_log: fin &= feats_e[spec.y_col] > 0
        bsel = np.where(bmask & fin)[0]
        pts = np.column_stack([xp[bsel], yp[bsel]])
        ref[key] = {"pts": pts, "by_label": {lab: pts[blabels[bsel] == lab] for lab in ("g1","g3","lift_off")}}
        fe = np.where(feas & fin)[0]
        par = fe[_pareto(np.column_stack([xp[fe], yp[fe]]), spec.pareto_sense)]
        po = np.column_stack([xp[par], yp[par]]); ref_par[key] = po[np.argsort(po[:, 0])]

    # BO cells
    bo = {k: [] for k in PROJ}; bo_rec = {k: {l: [] for l in ("g1","g3","lift_off")} for k in PROJ}; bo_par = {k: [] for k in PROJ}
    for f in CELLS:
        d = np.load(f, allow_pickle=True)
        Xu = d["X_obs"]; phys, F = _feats_from_unit(Xu)
        s = _score_subset(Xu, d["feasible"], d["converged"],
                          d["Y_h_min"].astype(float), d["Y_T_max"].astype(float),
                          d["Y_P_loss"].astype(float), F, ref, ref_par)
        for key in PROJ:
            bo[key].append(s[key]["d_H"]); bo_par[key].append(s[key]["pareto_dH"])
            for lab, v in s[key]["recall"].items():
                bo_rec[key][lab].append(v)

    # random baselines
    rng = np.random.default_rng(0)
    rnd = {k: [] for k in PROJ}; rnd_rec = {k: {l: [] for l in ("g1","g3","lift_off")} for k in PROJ}; rnd_par = {k: [] for k in PROJ}
    for seed in range(N_RANDOM):
        idx = np.random.default_rng(seed).choice(len(df), 500, replace=False)
        s = _score_subset(U[idx], feas[idx], df["converged"].to_numpy(bool)[idx],
                          hmin_e[idx], Tmax_e[idx], Ploss_e[idx], F_e[idx], ref, ref_par)
        for key in PROJ:
            rnd[key].append(s[key]["d_H"]); rnd_par[key].append(s[key]["pareto_dH"])
            for lab, v in s[key]["recall"].items():
                rnd_rec[key][lab].append(v)

    def med(a): return round(float(np.nanmedian(a)), 4) if len(a) else float("nan")
    report = {"n_cells": len(CELLS), "n_random": N_RANDOM, "projections": {}}
    stop_flag = False
    for key in PROJ:
        bo_dH, rnd_dH = med(bo[key]), med(rnd[key])
        better = bo_dH < rnd_dH
        if not better:
            stop_flag = True
        report["projections"][key] = {
            "BO_feasboundary_dH_median": bo_dH, "random_dH_median": rnd_dH,
            "BO_better": bool(better),
            "BO_recall": {l: med(bo_rec[key][l]) for l in ("g1","g3","lift_off")},
            "random_recall": {l: med(rnd_rec[key][l]) for l in ("g1","g3","lift_off")},
            "BO_pareto_dH_median": med(bo_par[key]), "random_pareto_dH_median": med(rnd_par[key]),
        }
        print(f"[{key}] feas-boundary d_H: BO {bo_dH} vs random {rnd_dH} -> BO {'BETTER' if better else 'NOT better (FLAG)'}; "
              f"BO recall {report['projections'][key]['BO_recall']} vs rnd {report['projections'][key]['random_recall']}; "
              f"Pareto d_H BO {med(bo_par[key])} vs rnd {med(rnd_par[key])}", flush=True)
    report["STOP_bo_not_better_on_feasboundary"] = bool(stop_flag)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nSTOP flag (BO not better on feas-boundary any projection): {stop_flag}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
