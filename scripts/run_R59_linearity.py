"""R-59.1 (a) and (b): the D2 STOP's two tests, at E4's 136 all-negative points.

Each point is rebuilt exactly as run_R58_E4_signed.job() rebuilt it (boxc_decomposition._build,
the registered eps_cap), in MODULUS mode, and its equilibrium must reproduce the stored
E4 eps to 1e-9 before anything else is computed (R-59.1(a), last bullet: the modulus fields
are not persisted, so they are re-solved).

(a) AS REGISTERED.  On the modulus run's own converged fields (eps, p, gdot, eta_bar), the
    memory force dF = F(memory on) - F(memory off) of the frozen-coefficient assembly (D1's
    assembly), modulus vs signed fluid: registered |dF_s + dF_m| / |dF_m| <= 1e-12.
    PREMISE NOTE, quoted from ball_socket/spherical.py:616-636: memory enters the MOBILITY,
    D_total = h^3/(12 eta_bar) + (-lambda hdot/12) h^2 M / eta_bar^2, and the pressure is the
    solution of div(D_total grad p) = S.  F is therefore not linear in M, and dF is not odd
    in M even with a perfect wrapper.  Two further quantities separate "wrapper wrong" from
    "F nonlinear in M":
      a_wrapper  cellwise max |M_s + M_m| / max |M_m| and the same for the assembled memory
                 mobility: exact (0) for a correct wrapper.
      a_scaling  the same odd-residual with M scaled by 0.5 and 0.25 (memory_scale): a residual
                 that is the quadratic term of the mobility form halves with M (ratio 2.00).
(b) AS REGISTERED.  Prediction of the signed-vs-modulus D2 share deviation from the point's
    own measured d_eps (R58_E4_signed.parquet eps_signed - eps_stored) and a central
    difference of the modulus D2 share in eps at the modulus equilibrium, +/- 2 % in delta
    (R-57.1's construction, delta = 1 - eps cos theta_min).  Registered: within 20 % of the
    measured 1.034 % at pf:950; sign agreement wherever |measured| > 0.1 %; Spearman >= 0.9.
    Reported alongside, NOT registered: the same prediction plus the D2 nonlinear term at
    the modulus equilibrium (signed vs modulus D2 share at fixed eps).
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
DEST = OUT / "R59_linearity.parquet"
REPORT = OUT / "R59_linearity.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
MIXED = ("E4:inside-1", 1e-5)
FD = 0.02
EPS_REPRO = 1e-9
A_TOL = 1e-12
MEMORY_SIGN = "modulus"          # the run's fields; the signed fluid is built by wrap() below


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

    # ---- a_wrapper: cellwise, exact -----------------------------------------------------
    Mm = np.asarray(fluid.memory_viscosity(gdot), float)
    Ms = np.asarray(fs.memory_viscosity(gdot), float)
    out["a_wrapper_M"] = float(np.max(np.abs(Ms + Mm)) / np.max(np.abs(Mm)))
    out["cells_M_negative_signed"] = int((Ms < 0).sum())

    # ---- (a) frozen-field memory force ------------------------------------------------
    def F(fl, mem, scale=None):
        s = sph.assemble("viscoelastic", fl, eta_bar, gdot, pressure=pressure, include_memory=mem,
                         include_normal_stress=True, n1_scale=1.0, hoop_scale=1.0,
                         memory_scale=None if scale is None else np.full(sph.shape, scale))
        return float(sph.load_capacity(sph.to_2d(spsolve(s.A, s.b))))
    Fn = F(fluid, False)
    dFm, dFs = F(fluid, True) - Fn, F(fs, True) - Fn
    out.update(F_off=Fn, dF_mod=dFm, dF_signed=dFs, a_odd_residual=abs(dFs + dFm) / abs(dFm))
    for k in (0.5, 0.25):
        dp, dq = F(fluid, True, k) - Fn, F(fluid, True, -k) - Fn
        out[f"a_odd_residual_scale{k}"] = abs(dp + dq) / abs(dp)

    # ---- D2 share at fixed eps (full re-solve memory on vs off) ------------------------
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
    e_p, e_m = (1.0 - delta * (1 - FD)) / ct, (1.0 - delta * (1 + FD)) / ct   # delta -/+ 2 %
    s0 = share_D2(eps_m, fluid)
    sp_, sm_ = share_D2(e_p, fluid), share_D2(e_m, fluid)
    s0_signed = share_D2(eps_m, fs)
    out.update(delta=delta, share_D2_mod=s0, share_D2_mod_stored=float(rec["share_memory_D2"]),
               share_D2_signed_at_eps_mod=s0_signed,
               dabs_share_deps=(abs(sp_) - abs(sm_)) / (e_p - e_m),
               nonlin_D2_rel=abs(s0_signed) / abs(s0) - 1,
               status="ok", wall_s=time.time() - t0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-59.1 linearity runner")
    from ball_socket.immutable import guarded_path
    if not a.report_only:
        guarded_path(str(DEST), what="R-59.1 linearity")
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
        D = pd.read_parquet(DEST)
        wall = None
    S = pd.read_parquet(R58)[["base", "lam", "eps_signed", "eps_stored", "share_D2_signed", "share_D2_stored"]]
    D = D.merge(S, on=["base", "lam"], how="left", suffixes=("", "_r58"))
    ok = D[D.status == "ok"].copy()
    ok["d_eps"] = ok.eps_signed - ok.eps_stored
    ok["meas_rel"] = ok.share_D2_signed.abs() / ok.share_D2_stored.abs() - 1
    ok["pred_shift_rel"] = ok.dabs_share_deps * ok.d_eps / ok.share_D2_mod.abs()
    ok["pred_total_rel"] = ok.pred_shift_rel + ok.nonlin_D2_rel
    from scipy.stats import spearmanr
    pf = ok.loc[ok.meas_rel.abs().idxmax()]
    big = ok[ok.meas_rel.abs() > 1e-3]

    def b_reg(col):
        return {"at_max_point": f"{pf.base}@{pf.lam:.3g}", "measured_rel": float(pf.meas_rel),
                "predicted_rel": float(pf[col]),
                "within_20pct": bool(abs(pf[col] / pf.meas_rel - 1) <= 0.20),
                "n_meas_above_0p1pct": int(len(big)),
                "sign_agree_all": bool((np.sign(big[col]) == np.sign(big.meas_rel)).all()),
                "sign_disagree": int((np.sign(big[col]) != np.sign(big.meas_rel)).sum()),
                "spearman_rho": float(spearmanr(ok[col], ok.meas_rel).statistic)}
    breg = b_reg("pred_shift_rel")
    breg["holds"] = bool(breg["within_20pct"] and breg["sign_agree_all"] and breg["spearman_rho"] >= 0.9)
    bdiag = b_reg("pred_total_rel")
    rep = {"provenance": prov, "points": int(len(D)), "workers": a.workers, "wall_min": wall,
           "eps_reproduced": int((D.status == "ok").sum()),
           "eps_repro_max_abs": float(D.eps_repro_abs.max()),
           "share_D2_mod_vs_stored_max_rel": float((ok.share_D2_mod / ok.share_D2_mod_stored - 1).abs().max()),
           "a_registered": {"tol": A_TOL, "max_odd_residual": float(ok.a_odd_residual.max()),
                            "min_odd_residual": float(ok.a_odd_residual.min()),
                            "points_above_tol": int((ok.a_odd_residual > A_TOL).sum()),
                            "holds": bool((ok.a_odd_residual <= A_TOL).all())},
           "a_wrapper": {"max_cellwise_M_residual": float(ok.a_wrapper_M.max()),
                         "exact_all": bool((ok.a_wrapper_M == 0).all())},
           "a_scaling": {"ratio_full_to_half_median": float((ok.a_odd_residual / ok["a_odd_residual_scale0.5"]).median()),
                         "ratio_full_to_half_range": [float((ok.a_odd_residual / ok["a_odd_residual_scale0.5"]).min()),
                                                      float((ok.a_odd_residual / ok["a_odd_residual_scale0.5"]).max())],
                         "ratio_half_to_quarter_median": float((ok["a_odd_residual_scale0.5"] / ok["a_odd_residual_scale0.25"]).median())},
           "b_registered": breg,
           "b_diagnostic_shift_plus_nonlinear_NOT_registered": bdiag,
           "max_point": {k: (float(pf[k]) if isinstance(pf[k], (float, np.floating)) else str(pf[k]))
                         for k in ["base", "lam", "eps_mod", "eps_signed", "d_eps", "delta", "share_D2_mod",
                                   "share_D2_signed", "meas_rel", "pred_shift_rel", "nonlin_D2_rel",
                                   "pred_total_rel", "a_odd_residual"]}}
    guarded_path(str(REPORT), what="R-59.1 linearity report")
    REPORT.write_text(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
