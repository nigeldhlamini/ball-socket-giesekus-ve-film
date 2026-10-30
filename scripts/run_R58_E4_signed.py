"""R-58.1 step 2: E4's 137 in-domain points re-run with the SIGNED memory viscosity (D-6),
then the row-wise bound on what the flip can do to Lambda across the campaign.

Each point is rebuilt exactly as run_R50_E4_lambda_sweep.one() built it
(boxc_decomposition._build(x, True), the registered eps_cap, boxc_decomposition._state),
with ONE change: slv.memory_sign = "signed".  D1 (frozen-coefficient, memory on vs off)
and D2 (full Picard re-solve, memory on vs off) as in the sweep.

Registered (R-58.1):
 1. the 136 all-negative points: share flips sign, |share| within 1 % of stored -> else STOP
 2. E4:inside-1 at the lambda floor (mixed): reported, not predicted; cell breakdown
 3. E4's max |share| changes only if that floor point was or becomes the maximum -> else STOP
Bound (R-58.1 "Exposure beyond E4"): dLambda_D6 = 2 |share_max| Lambda / |s| per row; the
campaign does not persist a per-row memory share, so the E4 maximum is used for every row
(as ruled), and |s| is not persisted either, so the SMALLEST coupled slope measured anywhere
in R-57.1 / R-57.1's floor statement (0.52) is used: a bound, not an estimate.
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
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
SRC = OUT / "E4_lambda_sweep.parquet"
DEST = OUT / "R58_E4_signed.parquet"
REPORT = OUT / "R58_E4_signed.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
L0 = 1.2496941918153979          # R57_subLambda0_shares.report.json
S_MIN = 0.52                     # smallest |d ln F / d ln delta| on record (R-57.1, at the floor)
MIXED = ("E4:inside-1", 1e-5)


def job(rec):
    import warnings
    warnings.simplefilter("ignore")
    from scipy.sparse.linalg import spsolve
    import boxc_decomposition as B
    import ball_socket.campaign as C
    from ball_socket.reynolds import solve_reynolds
    from src.solver import SolverConfig
    x = {k: float(rec[f"x_{k}"]) for k in AXES if k != "lam"}
    x["lam"] = float(rec["lam"])
    slv, ci = B._build(x, True)
    slv.eps_cap = float(C._registered_eps_cap(np.deg2rad(x["theta_min_deg"])))
    slv.memory_sign = "signed"
    sph, fluid, eta_bar, gdot, pressure, perf = B._state(slv, ci)

    def F_frozen(mem):
        sysm = sph.assemble("viscoelastic", fluid, eta_bar, gdot, pressure=pressure,
                            include_memory=mem, include_normal_stress=True,
                            n1_scale=1.0, hoop_scale=1.0)
        return float(sph.load_capacity(sph.to_2d(spsolve(sysm.A, sysm.b))))

    def F_resolved(mem):
        cfg = SolverConfig(include_normal_stress=True, max_iter=80, tol=1e-9, omega=0.7)
        cfg.include_memory = mem
        r = solve_reynolds(sph, fluid, cfg, model="viscoelastic", n1_mult=1.0, hoop_mult=1.0)
        return float(sph.load_capacity(r.pressure))

    Fa, Fn = F_frozen(True), F_frozen(False)
    Ra, Rn = F_resolved(True), F_resolved(False)
    Lam = fluid.lambda_ * np.abs(gdot)
    out = {"base": rec["base"], "lam": float(rec["lam"]), "eps_signed": float(perf.eps_ecc),
           "eps_stored": float(rec["eps"]),
           "share_D1_signed": (Fa - Fn) / Fa, "share_D2_signed": (Ra - Rn) / Ra,
           "share_D1_stored": float(rec["share_memory_D1"]),
           "share_D2_stored": float(rec["share_memory_D2"]),
           "cells_below_L0": int((Lam < L0).sum()), "cells": int(Lam.size),
           "Lam_cell_min": float(Lam.min()), "Lam_cell_max": float(Lam.max())}
    if (rec["base"], float(rec["lam"])) == MIXED:
        Ms = np.asarray(fluid.memory_viscosity(gdot))
        out["mixed_breakdown"] = {"cells_positive_M": int((Ms > 0).sum()),
                                  "cells_negative_M": int((Ms < 0).sum()),
                                  "theta_of_below_cells_deg": sorted(set(np.round(np.rad2deg(
                                      sph.mesh.TH[Lam < L0]), 2).tolist()))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-58 E4 signed re-run")
    sw = pd.read_parquet(SRC)
    ind = sw[(sw.status == "ok") & sw.in_g3_domain]
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        got = list(ex.map(job, ind.to_dict(orient="records")))
    D = pd.DataFrame([{k: v for k, v in g.items() if k != "mixed_breakdown"} for g in got])
    mixed = next((g["mixed_breakdown"] for g in got if "mixed_breakdown" in g), None)
    D.to_parquet(DEST, index=False)
    D["is_mixed"] = [(b, l) == MIXED for b, l in zip(D.base, D.lam)]
    D["flipped_D1"] = np.sign(D.share_D1_signed) == -np.sign(D.share_D1_stored)
    D["rel_D1"] = (D.share_D1_signed.abs() / D.share_D1_stored.abs() - 1).abs()
    D["rel_D2"] = (D.share_D2_signed.abs() / D.share_D2_stored.abs() - 1).abs()
    neg = D[~D.is_mixed]
    m = D[D.is_mixed].iloc[0]
    old_max_i = D.share_D1_stored.abs().idxmax()
    new_max_i = D.share_D1_signed.abs().idxmax()
    reg3_ok = bool(new_max_i == old_max_i or D.is_mixed[new_max_i] or D.is_mixed[old_max_i])
    rep = {"points": int(len(D)), "workers": a.workers, "wall_min": (time.time() - t0) / 60,
           "all_cells_above_L0_points": int((D.cells_below_L0 == 0).sum()),
           "reg1_136": {"n": int(len(neg)), "flipped_D1": int(neg.flipped_D1.sum()),
                        "flipped_D2": int((np.sign(neg.share_D2_signed) == -np.sign(neg.share_D2_stored)).sum()),
                        "max_rel_D1": float(neg.rel_D1.max()), "max_rel_D2": float(neg.rel_D2.max()),
                        "holds": bool(neg.flipped_D1.all() and (neg.rel_D1 <= 0.01).all()),
                        "STOP": bool(not (neg.flipped_D1.all() and (neg.rel_D1 <= 0.01).all()))},
           "reg2_mixed_point": {"base": m.base, "lam": float(m.lam),
                                "share_D1_stored": float(m.share_D1_stored), "share_D1_signed": float(m.share_D1_signed),
                                "share_D2_stored": float(m.share_D2_stored), "share_D2_signed": float(m.share_D2_signed),
                                "cells_below_L0": int(m.cells_below_L0), "cells": int(m.cells),
                                "breakdown": mixed},
           "reg3_max": {"old_max_abs_D1": float(D.share_D1_stored.abs().max()),
                        "new_max_abs_D1": float(D.share_D1_signed.abs().max()),
                        "old_argmax": f"{D.base[old_max_i]}@{D.lam[old_max_i]:.3g}",
                        "new_argmax": f"{D.base[new_max_i]}@{D.lam[new_max_i]:.3g}",
                        "old_max_abs_D2": float(D.share_D2_stored.abs().max()),
                        "new_max_abs_D2": float(D.share_D2_signed.abs().max()),
                        "holds": reg3_ok, "STOP": not reg3_ok},
           "new_median_abs_D1": float(D.share_D1_signed.abs().median()),
           "sign_now": {"positive_D1": int((D.share_D1_signed > 0).sum()),
                        "negative_D1": int((D.share_D1_signed < 0).sum())}}
    # ---- row-wise bound across the corrected campaign (R-58.1) ----
    share = max(rep["reg3_max"]["new_max_abs_D1"], rep["reg3_max"]["new_max_abs_D2"])
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M as S
    E = pd.read_parquet(OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet",
                        columns=["sample_id", "model_id", "outcome", "h_min_bnd"])
    E = E[E.outcome == "converged"]
    near = {}
    for b, s in (("lo", S[0]), ("hi", S[1])):
        lam_ = E.h_min_bnd / s
        dl = 2 * share * lam_ / S_MIN
        hit = E[dl > (lam_ - 3).abs()]
        near[b] = {"rows_within_bound": int(len(hit)), "samples": int(hit.sample_id.nunique()),
                   "max_dLam_D6": float(dl.max())}
    M57 = pd.read_parquet(OUT / "R57_mesh_error.parquet")
    M57["dLam_D6"] = 2 * share * M57.Lam_tight / M57.slope.abs()
    M57["dist"] = (M57.Lam_tight - 3).abs()
    unres = set(M57[M57.dLam_mesh.abs() > M57.dist].src_index[M57.case == "band"])
    over = M57[(M57.dLam_D6 > M57.dist)]
    rep["dLam_D6_bound"] = {
        "share_used": share, "s_min_used": S_MIN,
        "campaign_rows_within_bound": near,
        "claim_rows_over_bound": over[["case", "src_index", "Lam_tight", "dLam_D6", "dist"]].to_dict(orient="records"),
        "claim_rows_over_bound_not_already_mesh_unresolved": [
            int(i) for i in over.src_index[(over.case == "band") & ~over.src_index.isin(unres)]],
        "E13_E5_dLam_D6": {c: float(M57[M57.case == c].dLam_D6.iloc[0]) for c in ("E13", "E5")}}
    guarded_path(str(REPORT), what="R-58 E4 signed report")
    REPORT.write_text(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
