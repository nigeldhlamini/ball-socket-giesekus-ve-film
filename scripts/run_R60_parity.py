"""R-60.3 (a, re-registered) and R-60.4 (b, re-registered) at E4's 136 all-negative points.

Each point is rebuilt as run_R58_E4_signed.job() rebuilt it, in MODULUS mode, and must
reproduce its stored E4 eps to 1e-9 first.  Then, on the modulus run's own fields:

R-60.3.1  cell level: M_signed == -M_modulus BIT-EXACTLY in every cell (np.array_equal).
R-60.3.2  parity of the residual: share(s) = (F(s M) - F(0)) / F(s M) on frozen fields (D1's
          assembly; s M via memory_scale), r(s) = share(+s) + share(-s), for s = 1, 1/2, 1/4.
          Registered: r(1)/r(1/2) and r(1/2)/r(1/4) in 2.00 +/- 0.02 wherever |r| exceeds
          1e3 machine epsilons of the share.
          PREMISE NOTE: r as written is the ABSOLUTE even part of the share.  The share is
          O(M), its even part O(M^2), so halving M divides r by 4, not 2.  The 2.00 the ruling
          cites (R-59 probe, 2.002 / 2.001) is the RELATIVE residual r(s)/|share(+s)|, which is
          O(M).  Both are computed from the same F values; the report states both and does not
          choose.
R-60.4    on the 135 points other than pf:950: (even + shift) within 20 % of the measured
          signed-vs-modulus D2 deviation wherever |deviation| > 0.1 %; sign agreement there;
          Spearman >= 0.9.  shift = d|share_D2|/d eps (central difference, delta +/- 2 %) x the
          point's measured d_eps / |share_D2|.  even: the ruling's text says "measured directly
          on frozen fields (from R-60.3.2)"; its quoted pf:950 figure (13 %) is the D2 even part
          at fixed eps (signed vs modulus RE-SOLVED share at the modulus equilibrium).  Both are
          computed; both outcomes are reported.
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
R58 = OUT / "R58_E4_signed.parquet"
DEST = OUT / "R60_parity.parquet"
REPORT = OUT / "R60_parity.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
MIXED = ("E4:inside-1", 1e-5)
SEEN = ("pf:950", 1e-5)            # seen before re-registration (R-60.4): reported, not counted
FD = 0.02
EPS_REPRO = 1e-9
SCALES = (1.0, 0.5, 0.25)
MEMORY_SIGN = "modulus"            # the run's fields; the signed fluid comes from wrap() below


def job(rec):
    import warnings
    warnings.simplefilter("ignore")
    from scipy.sparse.linalg import spsolve
    import boxc_decomposition as B
    import ball_socket.campaign as C
    from ball_socket.memory_sign import wrap
    from ball_socket.reynolds import solve_reynolds
    from src.solver import SolverConfig
    x = {k: float(rec[f"x_{k}"]) for k in AXES if k != "lam"}
    x["lam"] = float(rec["lam"])
    slv, ci = B._build(x, True)
    slv.eps_cap = float(C._registered_eps_cap(np.deg2rad(x["theta_min_deg"])))
    slv.memory_sign = MEMORY_SIGN
    t0 = time.time()
    sph, fluid, eta_bar, gdot, pressure, perf = B._state(slv, ci)
    eps_m = float(perf.eps_ecc)
    out = {"base": rec["base"], "lam": float(rec["lam"]), "eps_mod": eps_m,
           "eps_stored": float(rec["eps"]), "eps_repro_abs": abs(eps_m - float(rec["eps"]))}
    if out["eps_repro_abs"] > EPS_REPRO:
        out["status"] = "eps_not_reproduced"
        return out
    fs = wrap(fluid, "signed")
    Mm = np.asarray(fluid.memory_viscosity(gdot), float)
    Ms = np.asarray(fs.memory_viscosity(gdot), float)
    out["a1_bit_exact"] = bool(np.array_equal(Ms, -Mm))
    out["a1_cells_not_exact"] = int(np.sum(Ms != -Mm))
    out["cells"] = int(Mm.size)

    def F(scale):
        s = sph.assemble("viscoelastic", fluid, eta_bar, gdot, pressure=pressure,
                         include_memory=scale != 0.0, include_normal_stress=True,
                         n1_scale=1.0, hoop_scale=1.0,
                         memory_scale=None if scale in (0.0, 1.0) else np.full(sph.shape, scale))
        return float(sph.load_capacity(sph.to_2d(spsolve(s.A, s.b))))

    def F_signed():
        s = sph.assemble("viscoelastic", fs, eta_bar, gdot, pressure=pressure, include_memory=True,
                         include_normal_stress=True, n1_scale=1.0, hoop_scale=1.0)
        return float(sph.load_capacity(sph.to_2d(spsolve(s.A, s.b))))

    out["F0"] = F(0.0)
    for s in SCALES:
        out[f"Fp_{s}"] = F(s)
        out[f"Fm_{s}"] = F(-s)
    out["F_signed_wrapper"] = F_signed()          # must equal Fm_1.0 if the wrapper is a sign flip

    def share_D2(eps, fl):
        _, sp = slv._forward(ci, eps, fluid, ci.c0)
        cfg = SolverConfig(include_normal_stress=True, max_iter=80, tol=1e-9, omega=0.7)
        cfg.include_memory = True
        Ra = float(sp.load_capacity(solve_reynolds(sp, fl, cfg, model="viscoelastic",
                                                   n1_mult=1.0, hoop_mult=1.0).pressure))
        cfg.include_memory = False
        Rn = float(sp.load_capacity(solve_reynolds(sp, fl, cfg, model="viscoelastic",
                                                   n1_mult=1.0, hoop_mult=1.0).pressure))
        return (Ra - Rn) / Ra
    ct = np.cos(np.deg2rad(x["theta_min_deg"]))
    delta = 1.0 - eps_m * ct
    e_p, e_m = (1.0 - delta * (1 - FD)) / ct, (1.0 - delta * (1 + FD)) / ct
    s0 = share_D2(eps_m, fluid)
    out.update(delta=delta, share_D2_mod=s0, share_D2_mod_stored=float(rec["share_memory_D2"]),
               share_D2_signed_at_eps_mod=share_D2(eps_m, fs),
               dabs_share_deps=(abs(share_D2(e_p, fluid)) - abs(share_D2(e_m, fluid))) / (e_p - e_m),
               status="ok", wall_s=time.time() - t0)
    return out


def _derive(D: pd.DataFrame) -> pd.DataFrame:
    D = D.copy()
    for s in SCALES:
        sp = (D[f"Fp_{s}"] - D.F0) / D[f"Fp_{s}"]
        sm = (D[f"Fm_{s}"] - D.F0) / D[f"Fm_{s}"]
        D[f"share_p_{s}"], D[f"r_{s}"] = sp, sp + sm
        D[f"rrel_{s}"] = (sp + sm) / sp.abs()
    D["ratio_abs_1"] = D["r_1.0"] / D["r_0.5"]
    D["ratio_abs_2"] = D["r_0.5"] / D["r_0.25"]
    D["ratio_rel_1"] = D["rrel_1.0"] / D["rrel_0.5"]
    D["ratio_rel_2"] = D["rrel_0.5"] / D["rrel_0.25"]
    D["above_floor"] = D["r_0.25"].abs() > 1e3 * np.finfo(float).eps * D["share_p_0.25"].abs()
    D["wrapper_force_equal"] = D.F_signed_wrapper == D["Fm_1.0"]
    D["d_eps"] = D.eps_signed - D.eps_stored
    D["meas_rel"] = D.share_D2_signed.abs() / D.share_D2_stored.abs() - 1
    D["shift_rel"] = D.dabs_share_deps * D.d_eps / D.share_D2_mod.abs()
    D["even_D2_fixed_eps"] = D.share_D2_signed_at_eps_mod.abs() / D.share_D2_mod.abs() - 1
    # |share_s| / |share_m| - 1 on frozen fields, with share_s = share(-M) = r - share(+M)
    D["even_frozen"] = (D["r_1.0"] - D["share_p_1.0"]).abs() / D["share_p_1.0"].abs() - 1
    D["pred_D2even"] = D.shift_rel + D.even_D2_fixed_eps
    D["pred_frozeneven"] = D.shift_rel + D.even_frozen
    return D


def _b(D, col):
    from scipy.stats import spearmanr
    big = D[D.meas_rel.abs() > 1e-3]
    rel_err = (big[col] / big.meas_rel - 1).abs()
    return {"n": int(len(D)), "n_meas_above_0p1pct": int(len(big)),
            "within_20pct_all": bool((rel_err <= 0.20).all()),
            "n_outside_20pct": int((rel_err > 0.20).sum()),
            "max_rel_err": float(rel_err.max()) if len(big) else None,
            "sign_agree_all": bool((np.sign(big[col]) == np.sign(big.meas_rel)).all()),
            "spearman_rho": float(spearmanr(D[col], D.meas_rel).statistic),
            "holds": bool((rel_err <= 0.20).all() and (np.sign(big[col]) == np.sign(big.meas_rel)).all()
                          and spearmanr(D[col], D.meas_rel).statistic >= 0.9)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-60 parity runner")
    from ball_socket.immutable import guarded_path
    if not a.report_only:
        guarded_path(str(DEST), what="R-60 parity")
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
    S = pd.read_parquet(R58)[["base", "lam", "eps_signed", "eps_stored", "share_D2_signed", "share_D2_stored"]]
    D = D.merge(S, on=["base", "lam"], how="left", suffixes=("", "_r58"))
    ok = _derive(D[D.status == "ok"])
    fl = ok[ok.above_floor]
    band = lambda c: bool(((fl[c] - 2.0).abs() <= 0.02).all())
    a1 = {"points": int(len(ok)), "bit_exact_all": bool(ok.a1_bit_exact.all()),
          "cells_not_exact_total": int(ok.a1_cells_not_exact.sum()),
          "holds": bool(ok.a1_bit_exact.all()), "STOP": not bool(ok.a1_bit_exact.all())}
    a2 = {"points_above_floor": int(len(fl)),
          "as_written_absolute_r": {
              "ratio_1_range": [float(fl.ratio_abs_1.min()), float(fl.ratio_abs_1.max())],
              "ratio_2_range": [float(fl.ratio_abs_2.min()), float(fl.ratio_abs_2.max())],
              "in_2.00+/-0.02": band("ratio_abs_1") and band("ratio_abs_2"),
              "in_4.00+/-0.04_(O(M^2)_absolute)": bool(((fl.ratio_abs_1 - 4).abs() <= 0.04).all()
                                                       and ((fl.ratio_abs_2 - 4).abs() <= 0.04).all())},
          "relative_r_over_share": {
              "ratio_1_range": [float(fl.ratio_rel_1.min()), float(fl.ratio_rel_1.max())],
              "ratio_2_range": [float(fl.ratio_rel_2.min()), float(fl.ratio_rel_2.max())],
              "in_2.00+/-0.02": band("ratio_rel_1") and band("ratio_rel_2")},
          "wrapper_force_bit_equal_to_scale_minus1": bool(ok.wrapper_force_equal.all())}
    seen = ok[[(b, float(l)) == SEEN for b, l in zip(ok.base, ok.lam)]]
    rest = ok[[(b, float(l)) != SEEN for b, l in zip(ok.base, ok.lam)]]
    rep = {"provenance": prov, "points": int(len(D)), "workers": a.workers, "wall_min": wall,
           "eps_reproduced": int((D.status == "ok").sum()),
           "eps_repro_max_abs": float(D.eps_repro_abs.max()),
           "share_D2_mod_vs_stored_max_rel": float((ok.share_D2_mod / ok.share_D2_mod_stored - 1).abs().max()),
           "R60_3_1_cell_level": a1, "R60_3_2_parity": a2,
           "R60_4_on_135": {"even_D2_at_fixed_eps": _b(rest, "pred_D2even"),
                            "even_frozen_fields": _b(rest, "pred_frozeneven")},
           "pf950_seen_before_registration": seen[["meas_rel", "shift_rel", "even_D2_fixed_eps", "even_frozen",
                                                   "pred_D2even", "pred_frozeneven"]].to_dict(orient="records"),
           "max_abs_rrel_1": float(ok["rrel_1.0"].abs().max())}
    guarded_path(str(REPORT), what="R-60 parity report")
    REPORT.write_text(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
