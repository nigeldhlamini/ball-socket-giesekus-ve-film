"""DIAGNOSTIC for the R1 STOP (R-56): does the solver reach its formal order at
delta <= 0.03 once the rim band is resolved?  Theta-only refinement; the film is
axisymmetric, so Nphi is held small after checking it does not move the load.
Does NOT lift the STOP -- that is the adjudicator's -- it says where it comes from.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import p3_rim_asymptote as RA  # noqa: E402
from run_R56_R1 import solve, C, ETA, R_B, OMEGA  # noqa: E402

OUT = ROOT / "results" / "ch9" / "R56_R1_theta_refine.diagnostic.json"
NT = (32, 64, 128, 256, 512, 1024)


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")
    rep = {"phi_check": {}, "points": []}
    F96, e = solve(40.0, 0.01, (128, 96))
    F8, _ = solve(40.0, 0.01, (128, 8))
    rep["phi_check"] = {"F(128x96)/F(128x8) - 1": F96 / F8 - 1}
    for tm in (20.0, 30.0, 40.0):
        for d in (0.03, 0.01):
            fq = RA.f_recess(d, tm)
            errs = []
            for n in NT:
                F, eps = solve(tm, d, (n, 8))
                errs.append(RA.solver_f(F, c=C, eta=ETA, R=R_B, omega=OMEGA, eps=eps) / fq - 1)
            band_cells = [d / np.tan(np.deg2rad(tm)) / (np.deg2rad(73 - tm) / n) for n in NT]
            orders = [float(np.log2(abs(errs[i]) / abs(errs[i + 1]))) for i in range(len(NT) - 1)]
            rep["points"].append(dict(theta_min_deg=tm, delta=d, Ntheta=list(NT), errors=errs,
                                      orders=orders, rim_band_cells=band_cells))
            print(tm, d, " ".join(f"{x:+.2e}" for x in errs), "| orders", " ".join(f"{o:.2f}" for o in orders), flush=True)
    OUT.write_text(json.dumps(rep, indent=2, default=float))
    print(rep["phi_check"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
