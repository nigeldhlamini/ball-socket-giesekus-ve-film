"""R-57.2: is the early-interval exponent shift shear-thinning, as elimination says?

Stratified 64 of the 509 reproducing rows of analysis_E7_load_decomposition_32x24
(theta_min quartile x lambda quartile x 4).  For each, the squeeze-only film load at
delta = 0.1 and 0.03 (pocket, case and spin off, c = c0, 32 x 24), through the E7
script's own construction (run_E7_load_decomposition._solver/_film_load):

  memory ON   -- reproduces the stored F_sq_d0.1 / F_sq_d0.03 (checked, 1e-9)
  memory OFF  -- SolverConfig.include_memory = False, shear-thinning left on

Registered: the shift with memory off (vs the quadrature at the row's theta_min) is
within 0.005 of the full shift, as a median of per-row differences.  Failing: it is
not -> memory or the D-6 sign is doing more than its magnitude allows -> STOP on E4.
Also: the full shift over all 509 rows as a MEDIAN OF PER-ROW DIFFERENCES with a
bootstrap 95 % interval (R-57.2.2).
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
SRC = OUT / "analysis_E7_load_decomposition_32x24.parquet"
DEST = OUT / "R57_memory_off.parquet"
REPORT = OUT / "R57_memory_off.report.json"
X_COLS = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
SEED = 20261002


def job(r):
    import warnings
    warnings.simplefilter("ignore")
    import run_E7_load_decomposition as E7
    x = {k: float(r[k]) for k in X_COLS}
    slv, ci = E7._solver(x, 32, 24)
    ceiling = 1.0 / np.cos(slv.geom.theta_min)
    out = {"sample_id": int(r["sample_id"])}
    for mem in (True, False):
        slv.config.include_memory = mem
        for d in (0.1, 0.03):
            out[f"F_{'on' if mem else 'off'}_{d}"], _ = E7._film_load(
                slv, ci, (1.0 - d) * ceiling, hyd=False, spin=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import warnings
    warnings.simplefilter("ignore")
    import p3_rim_asymptote as RA
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-57.2 memory-off decomposition")
    r = pd.read_parquet(SRC)
    r = r[np.abs(r.F_full / r.F_film_stored - 1) <= 1e-9].copy()
    assert len(r) == 509
    q = {}
    qe = lambda t: q.setdefault(round(t, 10), RA.local_exp(lambda d: RA.f_recess(d, t), 0.1, 0.03))
    r["e_on_stored"] = np.log(r["F_sq_d0.03"] / r["F_sq_d0.1"]) / np.log(0.3)
    r["e_quad"] = [qe(t) for t in r.theta_min_deg]
    r["shift_full"] = r.e_on_stored - r.e_quad
    rng = np.random.default_rng(SEED)
    boots = [float(np.median(rng.choice(r.shift_full.values, len(r)))) for _ in range(4000)]
    r["q_t"] = pd.qcut(r.theta_min_deg, 4, labels=False)
    r["q_l"] = pd.qcut(r.lam, 4, labels=False)
    pick = r.groupby(["q_t", "q_l"]).sample(n=4, random_state=SEED).reset_index(drop=True)
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        got = list(ex.map(job, pick.to_dict(orient="records")))
    D = pick.merge(pd.DataFrame(got), on="sample_id")
    D["repro_01"] = D["F_on_0.1"] / D["F_sq_d0.1"] - 1
    D["repro_003"] = D["F_on_0.03"] / D["F_sq_d0.03"] - 1
    D["e_on"] = np.log(D["F_on_0.03"] / D["F_on_0.1"]) / np.log(0.3)
    D["e_off"] = np.log(D["F_off_0.03"] / D["F_off_0.1"]) / np.log(0.3)
    D["shift_on"] = D.e_on - D.e_quad
    D["shift_off"] = D.e_off - D.e_quad
    D["off_minus_on"] = D.shift_off - D.shift_on
    D["mem_share_01"] = D["F_on_0.1"] / D["F_off_0.1"] - 1
    D.to_parquet(DEST, index=False)
    med = float(D.off_minus_on.median())
    rep = {"rows": int(len(D)), "wall_min": (time.time() - t0) / 60, "workers": a.workers,
           "reproduction_max_rel": float(max(D.repro_01.abs().max(), D.repro_003.abs().max())),
           "full_shift_all_509": {"median_of_per_row_differences": float(r.shift_full.median()),
                                  "bootstrap_95": [float(np.percentile(boots, 2.5)),
                                                   float(np.percentile(boots, 97.5))]},
           "subsample": {"median_shift_on": float(D.shift_on.median()),
                         "median_shift_off": float(D.shift_off.median()),
                         "median_per_row_off_minus_on": med,
                         "max_abs_off_minus_on": float(D.off_minus_on.abs().max()),
                         "memory_share_of_squeeze_load_at_0.1": {
                             "median": float(D.mem_share_01.median()),
                             "max_abs": float(D.mem_share_01.abs().max())}},
           "registered_within_0.005": bool(abs(med) <= 0.005),
           "STOP_on_E4": bool(abs(med) > 0.005)}
    guarded_path(str(REPORT), what="R-57.2 report")
    REPORT.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
