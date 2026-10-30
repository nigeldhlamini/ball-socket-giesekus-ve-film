"""R-57.4 step 6, FIRST HALF ONLY: the load-weighted share of land cells with
lambda * gdot_bar < Lambda_0 at each in-domain E4 point.  Handed back before any
D-6 registration or fix (R-57.0 guard: "to be registered on return").

Each in-domain row of E4_lambda_sweep.parquet is rebuilt exactly as
run_R50_E4_lambda_sweep.one() built it -- boxc_decomposition._build(x, True), the
registered eps_cap set on the solver, boxc_decomposition._state -- and accepted only if
the rebuilt equilibrium eps reproduces the sweep's stored eps to 1e-9.

Lambda_0(alpha = 0.25): the zero of Paper 2's signed M, computed with Paper 2's own
function, giesekus-paper2/scripts/paper2_appendixC_asymptotics.py:74
(signed_M_over_etap), bracketed as Paper 2 brackets it (:105, [0.3, 8]).

Weights reported: load (p n_z dA, the integrand of Spherical.load_capacity), area (dA),
and memory-flux (|mem_factor| h^2 M / eta_bar^2 |grad p| dA, the size of the term D-6
changes the sign of).
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
DEST = OUT / "R57_subLambda0_shares.parquet"
REPORT = OUT / "R57_subLambda0_shares.report.json"
P2 = pathlib.Path(r"C:\Users\Administrator\Documents\giesekus-paper2\scripts")
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]


def lambda_0(alpha=0.25):
    sys.path.insert(0, str(P2))
    from paper2_appendixC_asymptotics import signed_M_over_etap
    from scipy.optimize import brentq
    return float(brentq(lambda L: signed_M_over_etap(alpha, L), 0.3, 8.0, xtol=1e-12))


def job(args):
    rec, L0 = args
    import warnings
    warnings.simplefilter("ignore")
    import boxc_decomposition as B
    import ball_socket.campaign as C
    x = {k: float(rec[f"x_{k}"]) for k in AXES if k != "lam"}
    x["lam"] = float(rec["lam"])
    slv, ci = B._build(x, True)
    slv.eps_cap = float(C._registered_eps_cap(np.deg2rad(x["theta_min_deg"])))
    sph, fluid, eta_bar, gdot, p, perf = B._state(slv, ci)
    Lam = fluid.lambda_ * np.abs(gdot)
    below = Lam < L0
    nz, dA = sph.load_axis_factors()
    w_load = p * nz * dA
    gpt, gpp = sph.grad_s(p)
    M = fluid.memory_viscosity(gdot)
    w_mem = np.abs(fluid.lambda_ * sph.gap_rate() / 12.0) * sph.gap() ** 2 * M / eta_bar ** 2 \
        * np.hypot(gpt, gpp) * dA
    return {"base": rec["base"], "lam": float(rec["lam"]), "eps_rebuilt": float(perf.eps_ecc),
            "eps_stored": float(rec["eps"]), "Wi_char": float(rec["Wi_char"]),
            "share_load": float(w_load[below].sum() / w_load.sum()),
            "share_area": float(dA[below].sum() / dA.sum()),
            "share_memflux": float(w_mem[below].sum() / w_mem.sum()),
            "Lam_cell_min": float(Lam.min()), "Lam_cell_max": float(Lam.max()),
            "share_memory_D1": float(rec["share_memory_D1"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-57.4 sub-Lambda0 shares")
    L0 = lambda_0()
    sw = pd.read_parquet(SRC)
    ind = sw[(sw.status == "ok") & sw.in_g3_domain]
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        got = list(ex.map(job, [(r, L0) for r in ind.to_dict(orient="records")]))
    D = pd.DataFrame(got)
    D["reproduces"] = (D.eps_rebuilt / D.eps_stored - 1).abs() <= 1e-9
    D.to_parquet(DEST, index=False)
    R = D[D.reproduces]
    per_base = R.groupby("base").share_load.max()
    rep = {"Lambda_0_alpha_0.25": L0, "source_Lambda_0": str(P2 / "paper2_appendixC_asymptotics.py") + ":74",
           "in_domain_points": int(len(D)), "reproducing": int(len(R)), "wall_min": (time.time() - t0) / 60,
           "share_load": {"median": float(R.share_load.median()), "max": float(R.share_load.max()),
                          "points_ge_10pct": int((R.share_load >= 0.10).sum())},
           "share_area": {"median": float(R.share_area.median()), "max": float(R.share_area.max())},
           "share_memflux": {"median": float(R.share_memflux.median()), "max": float(R.share_memflux.max()),
                             "points_ge_10pct": int((R.share_memflux >= 0.10).sum())},
           "per_base_max_share_load": per_base.to_dict(),
           "below_10pct_load_everywhere": bool((R.share_load < 0.10).all()),
           "Lam_cell_min_over_points": float(R.Lam_cell_min.min())}
    guarded_path(str(REPORT), what="R-57.4 sub-Lambda0 report")
    REPORT.write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps(rep, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
