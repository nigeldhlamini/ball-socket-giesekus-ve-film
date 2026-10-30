"""R-62.2: B1 re-registered on the M = 0 baseline, at E4's 136 all-negative points.

Per point, rebuilt in MODULUS mode exactly as run_R61_parity_floor.job() rebuilt it (stored
eps reproduced to 1e-9 first), on the modulus run's own fields:
  * F0 = the reference load (memory path OFF), R-60/R-61's own call;
  * F_zero = the memory path ON with M set EXACTLY to zero (a delegating fluid whose
    memory_viscosity returns zeros of the right shape); share(0) = (F_zero - F0) / F_zero;
  * r(s) = share(+s) + share(-s) for s = 1, 1/2, 1/4 (R-60's call; asserted bitwise equal to
    R61_parity_floor.parquet), and r~(s) = r(s) - 2 share(0);
  * at pf:1831 @ 2.15e-3 and pf:6229 @ 4.64e-4 only: the fitted offset c of r(s) = a s^2 + c
    over s = 1 ... 1/32 (the R-61 diagnostic, now persisted).

Registered (R-62.2), before running:
  1. |share(0)| <= 1e-13 at every point;
  2. c = 2 share(0) at the two misses, within 20 % or 2e-15 absolute, whichever is larger;
  3. r~(1)/r~(1/2) and r~(1/2)/r~(1/4) in 4.00 +/- 0.04 and r~/|share| ratios in 2.00 +/- 0.02
     at EVERY point, the two misses included.  No floor.
Outcomes: all hold -> B1 passes, E4 released.  Item 2 or 3 fails -> STOP on E4 (box G is
decoupled, R-62.1).
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
R61 = OUT / "R61_parity_floor.parquet"
DEST = OUT / "R62_share0.parquet"
REPORT = OUT / "R62_share0.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
MIXED = ("E4:inside-1", 1e-5)
MISSES = [("pf:1831", 0.002154), ("pf:6229", 0.000464)]
SCALES = (1.0, 0.5, 0.25)
LADDER = (1.0, 0.5, 0.25, 0.125, 0.0625, 1 / 32)
EPS_REPRO = 1e-9
MEMORY_SIGN = "modulus"


class _ZeroMemory:
    """The base fluid with memory_viscosity identically zero (M = 0 exactly)."""

    def __init__(self, base):
        self._b = base

    def __getattr__(self, n):
        return getattr(self._b, n)

    def memory_viscosity(self, gdot):
        return np.zeros_like(np.asarray(self._b.memory_viscosity(gdot), dtype=float))


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

    def F(scale, fl=fluid, mem=None):
        s = sph.assemble("viscoelastic", fl, eta_bar, gdot, pressure=pressure,
                         include_memory=(scale != 0.0) if mem is None else mem,
                         include_normal_stress=True, n1_scale=1.0, hoop_scale=1.0,
                         memory_scale=None if scale in (0.0, 1.0) else np.full(sph.shape, scale))
        return float(sph.load_capacity(sph.to_2d(spsolve(s.A, s.b))))

    F0 = F(0.0)
    Fz = F(1.0, fl=_ZeroMemory(fluid), mem=True)
    out.update(F0=F0, F_zero=Fz, share0=(Fz - F0) / Fz, F_zero_bit_equal_F0=bool(Fz == F0))
    for s in SCALES:
        out[f"Fp_{s}"], out[f"Fm_{s}"] = F(s), F(-s)
    if any(rec["base"] == b and np.isclose(rec["lam"], l, rtol=1e-3) for b, l in MISSES):
        r = []
        for s in LADDER:
            a, b = F(s), F(-s)
            r.append((a - F0) / a + (b - F0) / b)
        A = np.vstack([np.array(LADDER) ** 2, np.ones(len(LADDER))]).T
        (ca, cc), *_ = np.linalg.lstsq(A, np.array(r), rcond=None)
        out.update(ladder_r=json.dumps(r), ladder_a=float(ca), ladder_c=float(cc))
    out.update(status="ok", wall_s=time.time() - t0)
    return out


def derive(D: pd.DataFrame) -> pd.DataFrame:
    D = D.copy()
    for s in SCALES:
        sp = (D[f"Fp_{s}"] - D.F0) / D[f"Fp_{s}"]
        sm = (D[f"Fm_{s}"] - D.F0) / D[f"Fm_{s}"]
        D[f"share_{s}"], D[f"r_{s}"] = sp, sp + sm
        D[f"rt_{s}"] = D[f"r_{s}"] - 2 * D.share0
    D["abs_1"], D["abs_2"] = D["rt_1.0"] / D["rt_0.5"], D["rt_0.5"] / D["rt_0.25"]
    rel = {s: D[f"rt_{s}"] / D[f"share_{s}"].abs() for s in SCALES}
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
    prov = require_clean_tree("R-62.2 share(0) runner")
    from ball_socket.immutable import guarded_path
    if not a.report_only:
        guarded_path(str(DEST), what="R-62.2 share(0)")
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
    P = pd.read_parquet(R61).merge(ok, on=["base", "lam"], suffixes=("_r61", ""))
    cols = ["F0"] + [f"F{pm}_{s}" for pm in "pm" for s in SCALES]
    bitwise = bool(all((P[f"{c}_r61"] == P[c]).all() for c in cols))
    reg1 = bool((ok.share0.abs() <= 1e-13).all())
    item2 = []
    for b, l in MISSES:
        row = ok[(ok.base == b) & np.isclose(ok.lam, l, rtol=1e-3)].iloc[0]
        c, pred = float(row.ladder_c), float(2 * row.share0)
        tol = max(0.2 * abs(c), 2e-15)
        item2.append({"point": f"{b}@{l:g}", "fitted_c": c, "two_share0": pred, "abs_diff": abs(c - pred),
                      "tolerance": tol, "holds": bool(abs(c - pred) <= tol),
                      "ladder_r": json.loads(row.ladder_r), "ladder_a": float(row.ladder_a)})
    reg2 = all(i["holds"] for i in item2)
    reg3 = bool(ok.in_abs.all() and ok.in_rel.all())
    rep = {"provenance": prov, "points": int(len(D)), "workers": a.workers, "wall_min": wall,
           "eps_reproduced": int((D.status == "ok").sum()), "eps_repro_max_abs": float(D.eps_repro_abs.max()),
           "loads_bitwise_equal_R61": bitwise,
           "reg1_share0_le_1e-13": {"holds": reg1, "max_abs_share0": float(ok.share0.abs().max()),
                                    "n_share0_exactly_zero": int((ok.share0 == 0).sum()),
                                    "n_F_zero_bit_equal_F0": int(ok.F_zero_bit_equal_F0.sum())},
           "reg2_c_equals_2share0": {"holds": reg2, "points": item2},
           "reg3_corrected_ratios_every_point": {
               "holds": reg3, "n_outside_abs": int((~ok.in_abs).sum()), "n_outside_rel": int((~ok.in_rel).sum()),
               "abs_ratio_range": [float(min(ok.abs_1.min(), ok.abs_2.min())), float(max(ok.abs_1.max(), ok.abs_2.max()))],
               "rel_ratio_range": [float(min(ok.rel_1.min(), ok.rel_2.min())), float(max(ok.rel_1.max(), ok.rel_2.max()))],
               "outside": ok[~(ok.in_abs & ok.in_rel)][["base", "lam", "abs_1", "abs_2", "rel_1", "rel_2", "share0"]]
                   .to_dict(orient="records")},
           "B1_holds": bool(reg1 and reg2 and reg3),
           "outcome": ("B1 passes; E4 released (R-61 B2)" if (reg1 and reg2 and reg3) else
                       "STOP on E4: " + ("item 2 failed -- the offset is not the M = 0 baseline; " if not reg2 else "")
                       + ("item 3 failed; " if not reg3 else "") + ("item 1 failed; " if not reg1 else ""))}
    guarded_path(str(REPORT), what="R-62.2 report")
    REPORT.write_text(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
