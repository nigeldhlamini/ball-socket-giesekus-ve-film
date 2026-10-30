"""R-61 B1: the parity clause re-registered with a measured noise floor, at E4's 136 points.

Per point, rebuilt in MODULUS mode exactly as run_R60_parity.job() rebuilt it (stored eps
reproduced to 1e-9 first), on the modulus run's own fields:
  * frozen-field loads F(0), F(+/-s M), s = 1, 1/2, 1/4, under TWO column orderings of the
    sparse LU: the primary is R-60's own call (spsolve(A, b), default ordering), which must
    reproduce R60_parity.parquet's loads BIT-FOR-BIT; the second is MMD_AT_PLUS_A.
  * r(s) = share(+s) + share(-s), share(s) = (F(sM) - F(0)) / F(sM); the floor is
    dr = |r_primary(1/4) - r_second(1/4)|, and the clause applies where |r(1/4)| > 10 dr.

Registered (R-61 B1):
  1. pf:1831 @ 2.15e-3 and pf:6229 @ 4.64e-4 (the two R-60 misses) fall BELOW their floor;
  2. every point above its floor has r(1)/r(1/2) and r(1/2)/r(1/4) in 4.00 +/- 0.04 and the
     relative ratios (r/|share|) in 2.00 +/- 0.02.
Failing result: an above-floor point outside either band -> STOP, cause unknown.
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
R60 = OUT / "R60_parity.parquet"
DEST = OUT / "R61_parity_floor.parquet"
REPORT = OUT / "R61_parity_floor.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
MIXED = ("E4:inside-1", 1e-5)
MISSES = [("pf:1831", 0.002154), ("pf:6229", 0.000464)]
SCALES = (1.0, 0.5, 0.25)
EPS_REPRO = 1e-9
FLOOR_K = 10.0
MEMORY_SIGN = "modulus"            # the fields; the clause is about the frozen operator


def job(rec):
    import warnings
    warnings.simplefilter("ignore")
    from scipy.sparse.linalg import spsolve
    import boxc_decomposition as B
    import ball_socket.campaign as C
    x = {k: float(rec[f"x_{k}"]) for k in AXES if k != "lam"}
    x["lam"] = float(rec["lam"])
    slv, ci = B._build(x, True)
    slv.eps_cap = float(C._registered_eps_cap(np.deg2rad(x["theta_min_deg"])))
    slv.memory_sign = MEMORY_SIGN
    t0 = time.time()
    sph, fluid, eta_bar, gdot, pressure, perf = B._state(slv, ci)
    out = {"base": rec["base"], "lam": float(rec["lam"]), "eps_mod": float(perf.eps_ecc),
           "eps_repro_abs": abs(float(perf.eps_ecc) - float(rec["eps"]))}
    if out["eps_repro_abs"] > EPS_REPRO:
        out["status"] = "eps_not_reproduced"
        return out

    def F(scale, order):
        s = sph.assemble("viscoelastic", fluid, eta_bar, gdot, pressure=pressure,
                         include_memory=scale != 0.0, include_normal_stress=True,
                         n1_scale=1.0, hoop_scale=1.0,
                         memory_scale=None if scale in (0.0, 1.0) else np.full(sph.shape, scale))
        p = spsolve(s.A, s.b) if order == "primary" else spsolve(s.A.tocsc(), s.b, permc_spec=order)
        return float(sph.load_capacity(sph.to_2d(p)))

    for order in ("primary", "MMD_AT_PLUS_A"):
        tag = "" if order == "primary" else "_mmd"
        out[f"F0{tag}"] = F(0.0, order)
        for s in SCALES:
            out[f"Fp_{s}{tag}"] = F(s, order)
            out[f"Fm_{s}{tag}"] = F(-s, order)
    out.update(status="ok", wall_s=time.time() - t0)
    return out


def derive(D: pd.DataFrame) -> pd.DataFrame:
    D = D.copy()
    for tag in ("", "_mmd"):
        for s in SCALES:
            sp = (D[f"Fp_{s}{tag}"] - D[f"F0{tag}"]) / D[f"Fp_{s}{tag}"]
            sm = (D[f"Fm_{s}{tag}"] - D[f"F0{tag}"]) / D[f"Fm_{s}{tag}"]
            D[f"share_{s}{tag}"], D[f"r_{s}{tag}"] = sp, sp + sm
    D["dr"] = (D["r_0.25"] - D["r_0.25_mmd"]).abs()
    D["above_floor"] = D["r_0.25"].abs() > FLOOR_K * D.dr
    D["abs_1"], D["abs_2"] = D["r_1.0"] / D["r_0.5"], D["r_0.5"] / D["r_0.25"]
    rel = {s: D[f"r_{s}"] / D[f"share_{s}"].abs() for s in SCALES}
    D["rel_1"], D["rel_2"] = rel[1.0] / rel[0.5], rel[0.5] / rel[0.25]
    D["in_abs"] = ((D.abs_1 - 4).abs() <= 0.04) & ((D.abs_2 - 4).abs() <= 0.04)
    D["in_rel"] = ((D.rel_1 - 2).abs() <= 0.02) & ((D.rel_2 - 2).abs() <= 0.02)
    return D


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-61 B1 parity-floor runner")
    from ball_socket.immutable import guarded_path
    if not a.report_only:
        guarded_path(str(DEST), what="R-61 B1 parity floor")
        sw = pd.read_parquet(SRC)
        ind = sw[(sw.status == "ok") & sw.in_g3_domain]
        ind = ind[[(b, float(l)) != MIXED for b, l in zip(ind.base, ind.lam)]]
        from concurrent.futures import ProcessPoolExecutor
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            got = list(ex.map(job, ind.to_dict(orient="records")))
        D = pd.DataFrame(got)
        D["memory_sign_fields"] = MEMORY_SIGN
        D["git_commit"] = prov["commit"]
        D.to_parquet(DEST, index=False)
        wall = (time.time() - t0) / 60
    else:
        D, wall = pd.read_parquet(DEST), None
    ok = derive(D[D.status == "ok"])
    # primary ordering must be R-60's own loads, bit for bit
    P = pd.read_parquet(R60).merge(ok, on=["base", "lam"], suffixes=("_r60", ""))
    cols = ["F0"] + [f"F{pm}_{s}" for pm in "pm" for s in SCALES]
    bitwise = bool(all((P[f"{c}_r60"] == P[c]).all() for c in cols))
    miss = []
    for b, l in MISSES:
        row = ok[(ok.base == b) & np.isclose(ok.lam, l, rtol=1e-3)].iloc[0]
        miss.append({"point": f"{b}@{l:g}", "r_quarter": float(row["r_0.25"]), "dr": float(row.dr),
                     "ratio_to_floor": float(abs(row["r_0.25"]) / (FLOOR_K * row.dr)) if row.dr > 0 else None,
                     "below_floor": bool(not row.above_floor)})
    up = ok[ok.above_floor]
    reg1 = all(m["below_floor"] for m in miss)
    reg2 = bool(up.in_abs.all() and up.in_rel.all())
    rep = {"provenance": prov, "points": int(len(D)), "workers": a.workers, "wall_min": wall,
           "eps_reproduced": int((D.status == "ok").sum()), "eps_repro_max_abs": float(D.eps_repro_abs.max()),
           "primary_reproduces_R60_loads_bitwise": bitwise,
           "floor": f"|r(1/4)| > {FLOOR_K:g} x |r_primary(1/4) - r_MMD_AT_PLUS_A(1/4)|",
           "reg1_misses_below_floor": {"holds": reg1, "points": miss},
           "reg2_above_floor_in_bands": {
               "n_above_floor": int(len(up)), "n_below_floor": int((~ok.above_floor).sum()),
               "abs_ratio_range": [float(min(up.abs_1.min(), up.abs_2.min())), float(max(up.abs_1.max(), up.abs_2.max()))],
               "rel_ratio_range": [float(min(up.rel_1.min(), up.rel_2.min())), float(max(up.rel_1.max(), up.rel_2.max()))],
               "n_outside_abs": int((~up.in_abs).sum()), "n_outside_rel": int((~up.in_rel).sum()),
               "outside": up[~(up.in_abs & up.in_rel)][["base", "lam", "abs_1", "abs_2", "rel_1", "rel_2", "r_0.25", "dr"]]
                   .to_dict(orient="records"),
               "holds": reg2},
           "below_floor_points": ok[~ok.above_floor][["base", "lam", "r_0.25", "dr", "rel_1", "rel_2"]]
               .to_dict(orient="records"),
           "B1_holds": bool(reg1 and reg2), "STOP": not bool(reg1 and reg2)}
    guarded_path(str(REPORT), what="R-61 B1 report")
    REPORT.write_text(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
