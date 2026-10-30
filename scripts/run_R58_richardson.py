"""R-58.2 step 3: full-physics Richardson at the claim rows, on the D-6 (signed) path.

Rows: the E13 extremum, the E5 extremum, the feasibility-band margin row (R-56 cases).
Each solved through its own original evaluate_x call, tight tolerance (1e-10 on eps),
memory_sign = "signed", at 32x24, 64x48, 128x96.  9 solves.

  Lambda_R   = L3 + (L3 - L2) / 3                (second order, the scheme's formal order)
  p_obs      = log2((L1 - L2) / (L2 - L3))
  uncertainty = max(|Lambda_R - L3|, |Lambda_R(p_obs) - Lambda_R(2)|)

Registered (R-58.2): E13's shortfall to 3 stays 20 % to two figures; E5's verdict does not
move; each Lambda_R lies within +/-0.001 of (production + dLambda_mesh proxy), the proxy
being R-57.1's at the same row (it is a load-error property, so the sign law does not
enter it; the production value used is this run's signed 32x24 value).
Also: the sign of e_F at the 32 rows nearest g1 (R57_mesh_error.parquet).
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
DEST = OUT / "R58_richardson.parquet"
REPORT = OUT / "R58_richardson.report.json"
GRIDS = ((32, 24), (64, 48), (128, 96))


def job(args):
    c, grid = args
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.campaign as C
    from ball_socket.campaign import REGISTERED_CONSTRAINT_COLUMNS
    from run_R57_mesh_error import _contact
    kw = dict(contact=_contact(c["model_id"]), clearance_closure=True, delta_T_datum="bulk",
              Ntheta=grid[0], Nphi=grid[1], brentq_xtol=1e-10, brentq_rtol=1e-10,
              memory_sign="signed")
    if c["flow_factors"] is not None:
        kw["flow_factors"] = c["flow_factors"]
    t0 = time.time()
    row = C.evaluate_x(c["x"], **kw)
    return {"case": c["case"], "src_index": c["src_index"], "Ntheta": grid[0],
            **{k: row.get(k) for k in REGISTERED_CONSTRAINT_COLUMNS},
            "c_equilibrium": row.get("c_equilibrium"), "sigma": c["sigma"],
            "realised_memory_sign": row.get("realised_memory_sign"),
            "Lam": float(row["h_min_bnd"]) / c["sigma"], "wall_s": time.time() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-58.2 Richardson")
    import run_R56_tolerance as T
    m57 = json.loads((OUT / "R57_mesh_error.report.json").read_text())
    margin_idx = int(m57["margin_row"]["src_index"])
    cs = [c for c in T.cases() if c["case"] in ("E13", "E5") or
          (c["case"] == "band" and c["src_index"] == margin_idx)]
    assert len(cs) == 3
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        got = list(ex.map(job, [(c, g) for c in cs for g in GRIDS]))
    D = pd.DataFrame(got)
    D.to_parquet(DEST, index=False)
    proxy = pd.read_parquet(OUT / "R57_mesh_error.parquet").set_index(["case", "src_index"])
    rep = {"rows": {}, "memory_sign": sorted(D.realised_memory_sign.unique().tolist())}
    for (case, si), g in D.groupby(["case", "src_index"]):
        g = g.sort_values("Ntheta")
        L1, L2, L3 = g.Lam.values
        LR = L3 + (L3 - L2) / 3.0
        p = float(np.log2((L1 - L2) / (L2 - L3))) if (L1 - L2) * (L2 - L3) > 0 else float("nan")
        LRp = L3 + (L3 - L2) / (2 ** p - 1) if np.isfinite(p) else float("nan")
        unc = max(abs(LR - L3), abs(LRp - LR) if np.isfinite(LRp) else 0.0)
        pr = proxy.loc[(case, si)]
        pred = L1 + float(pr.dLam_mesh)
        rep["rows"][case] = {"src_index": int(si), "Lambda_32": L1, "Lambda_64": L2, "Lambda_128": L3,
                             "Lambda_R": LR, "p_obs": p, "uncertainty": unc,
                             "production_plus_proxy": pred, "R_minus_proxy_pred": LR - pred,
                             "within_0.001_of_proxy": bool(abs(LR - pred) <= 1e-3),
                             "shortfall_to_3_pct": 100 * (1 - LR / 3.0),
                             "verdict_g1": "full film" if LR >= 3 else "below full film"}
    rep["registered"] = {
        "E13_shortfall_20pct_two_figures": bool(round(rep["rows"]["E13"]["shortfall_to_3_pct"]) == 20),
        "E5_verdict_unmoved": bool(rep["rows"]["E5"]["Lambda_R"] < 3),
        "all_within_0.001_of_proxy": bool(all(r["within_0.001_of_proxy"] for r in rep["rows"].values()))}
    M = pd.read_parquet(OUT / "R57_mesh_error.parquet")
    b = M[M.case == "band"]
    rep["e_F_sign_32_rows"] = {"positive": int((b.e_F > 0).sum()), "negative": int((b.e_F < 0).sum()),
                               "negative_rows": b[b.e_F < 0][["src_index", "theta_min_deg", "delta",
                                                              "Lam_tight", "e_F", "dLam_mesh"]].to_dict(orient="records")}
    guarded_path(str(REPORT), what="R-58.2 report")
    REPORT.write_text(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
