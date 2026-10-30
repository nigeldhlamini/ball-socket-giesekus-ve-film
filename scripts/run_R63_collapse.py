"""R-63.4: the collapse test of R-54.6, run as registered.  No solves.  No figure.

R-54.6, verbatim (RULINGS_BOXC_ADDENDUM_R54.md, quote block under "Rule-18 fork"):
  > The collapse test decides whether the chart is drawn, and it can fail. Plot Phi against S
  > at each theta_min, stratified by eta_0 decade and by c0 bin, once with eta(T_in) and once
  > with eta_0. Report the residual dispersion of ln Phi about a smooth fit in ln S per
  > theta_min. Failing looks like: systematic stratification by eta_0 at fixed S beyond the
  > scatter -- S is missing a group and a chart drawn on it would mislead. If both viscosities
  > collapse equally, the sample's thermal spread is too narrow to discriminate; keep eta(T_in)
  > on design grounds and say so.
R-55.8 adds: "Registered: collapse within R-54.6's criterion over the smooth-film rows;
departures concentrated at the highest Nahme numbers."

TWO FACTS THAT SHAPE THE RUN, quoted from the code before computing:
  * eta(T_in) and eta_0 COINCIDE in this model: the solver takes eta_0 directly
    (ball_socket/solver.py:204-205), and T_in enters only the thermal path (clearance closure,
    g2).  There is no reference-temperature viscosity to substitute, so the two runs R-54.6 asks
    for are one run; its "both collapse equally" clause is met trivially and says nothing.
  * The Nahme number is carried per unit b as the film-mean rise T_max_bnd - T_in
    (scripts/run_R57_thermal_by_route.py:22-26); "highest Nahme" = largest rise.

OPERATIONALISATION (R-54.6 gives no number; this one is CODE'S, written and committed before
the first computation, and the hand-back flags it for the adjudicator):
  population  box D corrected pure-film rows with outcome 'converged' (the smooth film);
              Phi = h_min_bnd / c0; S from R62_operating_space.parquet (asserted equal to
              fig_R61_update.duty_S); theta_min in the three bands of A3.
  smooth fit  per band, least-squares cubic spline of ln Phi in ln S, interior knots at the
              20/40/60/80 % quantiles of ln S.  Residual e; scatter s_e = 1.4826 MAD(e).
  strata      eta_0 decade = floor(log10 eta_0) (three: 1e-3.., 1e-2.., 1e-1..); c0 bin = nine
              equal-count bins of c0.  Within each of ten equal-count ln S bins (per band), the
              median residual of each stratum with n >= 10.
  FAIL        for a stratifier in a band: the median over S bins of (max - min stratum median)
              exceeds s_e, AND the ordering of the extreme strata is the same in >= 80 % of the
              S bins that have both.  "Systematic ... beyond the scatter."
  verdict     collapses iff no stratifier fails in any band.
  R-55.8      "concentrated at the highest Nahme numbers" iff the top decile of rise holds at
              least twice its share (i.e. >= 20 %) of the top decile of |e|.
Reported, not registered: partial effects of each input on e (eta_0, omega, p_ch, c0, lam,
eta_s_ratio, T_in, theta_min within band), as slope x IQR of the input in units of s_e;
|e| against rise, against lam, and against a Weissenberg proxy lam*omega*R_b/h_min.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
DP = OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet"
OPS = OUT / "R62_operating_space.parquet"
DEST = OUT / "R63_collapse.parquet"
REPORT = OUT / "R63_collapse.report.json"
BANDS = [(20.0, 26.7), (26.7, 33.3), (33.3, 40.0001)]
R_B = 6.5e-3


def strat(sub, key, nsb=10, nmin=10):
    """Median over S bins of the spread of stratum-median residuals, and sign consistency."""
    sb = pd.qcut(sub.lnS, nsb, labels=False, duplicates="drop")
    spreads, signs = [], []
    levels = sorted(sub[key].unique())
    for _, g in sub.groupby(sb):
        med = g.groupby(key).e.agg(["median", "size"])
        med = med[med["size"] >= nmin]["median"]
        if len(med) >= 2:
            spreads.append(float(med.max() - med.min()))
        if levels[0] in med.index and levels[-1] in med.index:
            signs.append(np.sign(med[levels[-1]] - med[levels[0]]))
    signs = np.array(signs)
    cons = float(max((signs > 0).mean(), (signs < 0).mean())) if len(signs) else 0.0
    return {"median_spread": float(np.median(spreads)) if spreads else None, "n_S_bins": len(spreads),
            "extreme_order_consistency": cons, "n_bins_with_extremes": int(len(signs)),
            "levels": [int(x) if isinstance(x, (np.integer, int)) else str(x) for x in levels]}


def main() -> int:
    from scipy.interpolate import LSQUnivariateSpline
    from scipy.stats import spearmanr
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-63.4 collapse test")
    import fig_R61_update as G
    D = pd.read_parquet(DP).sort_values("sample_id").reset_index(drop=True)
    O = pd.read_parquet(OPS).sort_values("sample_id").reset_index(drop=True)
    assert np.array_equal(D.sample_id.values, O.sample_id.values)
    assert np.allclose(O.S.values, G.duty_S(D.p_ch.values, D.c0.values, D.eta_0.values, D.omega.values),
                       rtol=1e-12, atol=0)
    D["S"] = O.S.values
    D = D[D.outcome == "converged"].copy()
    D["Phi"] = D.h_min_bnd / D.c0
    D["lnS"], D["lnPhi"] = np.log(D.S), np.log(D.Phi)
    D["eta_decade"] = np.floor(np.log10(D.eta_0)).astype(int)
    D["c0_bin"] = pd.qcut(D.c0, 9, labels=False)
    D["rise"] = D.T_max_bnd - D.T_in
    D["Wi_proxy"] = D.lam * D.omega * R_B / D.h_min_bnd
    D["band"] = -1
    rep_b, fails = [], []
    for i, (t0, t1) in enumerate(BANDS):
        m = (D.theta_min_deg >= t0) & (D.theta_min_deg < t1)
        sub = D[m].sort_values("lnS")
        x, y = sub.lnS.values, sub.lnPhi.values
        knots = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
        sp = LSQUnivariateSpline(x, y, knots, k=3)
        e = y - sp(x)
        D.loc[sub.index, "e"], D.loc[sub.index, "band"] = e, i
        sub = D.loc[sub.index]
        se = float(1.4826 * np.median(np.abs(e - np.median(e))))
        st = {k: strat(sub, k) for k in ("eta_decade", "c0_bin")}
        verdict = {}
        for k, v in st.items():
            f = bool(v["median_spread"] is not None and v["median_spread"] > se
                     and v["extreme_order_consistency"] >= 0.8)
            verdict[k] = {"fails": f, "spread_over_scatter": (v["median_spread"] / se) if v["median_spread"] else None}
            if f:
                fails.append(f"band {i + 1}: {k}")
        # partial effects (reported, not registered)
        cols = {"ln_eta_0": np.log(sub.eta_0), "ln_omega": np.log(sub.omega), "ln_p_ch": np.log(sub.p_ch),
                "ln_c0": np.log(sub.c0), "ln_lam": np.log(sub.lam), "eta_s_ratio": sub.eta_s_ratio,
                "T_in": sub.T_in, "theta_min": sub.theta_min_deg}
        X = np.column_stack([np.ones(len(sub))] + [np.asarray(c, float) for c in cols.values()])
        beta, *_ = np.linalg.lstsq(X, sub.e.values, rcond=None)
        partial = {k: float(b * (np.subtract(*np.percentile(np.asarray(c, float), [75, 25]))) / se)
                   for (k, c), b in zip(cols.items(), beta[1:])}
        rep_b.append({"band_deg": [t0, min(t1, 40.0)], "n": int(len(sub)), "scatter_s_e": se,
                      "residual_sd": float(e.std()), "ln_Phi_range": [float(y.min()), float(y.max())],
                      "stratification": st, "verdict": verdict,
                      "partial_effect_IQR_over_s_e": partial})
    absE = D.e.abs()
    top_e = absE >= absE.quantile(0.9)
    top_rise = D.rise >= D.rise.quantile(0.9)
    share = float((top_e & top_rise).sum() / top_e.sum())
    conc = {"share_of_top_decile_abs_e_in_top_decile_rise": share, "concentrated": bool(share >= 0.20),
            "spearman_abs_e_vs_rise": float(spearmanr(absE, D.rise).statistic),
            "diagnostic_spearman_abs_e_vs_lam": float(spearmanr(absE, D.lam).statistic),
            "diagnostic_spearman_abs_e_vs_Wi_proxy": float(spearmanr(absE, D.Wi_proxy).statistic),
            "diagnostic_spearman_e_vs_Wi_proxy": float(spearmanr(D.e, D.Wi_proxy).statistic),
            "diagnostic_spearman_e_vs_lam": float(spearmanr(D.e, D.lam).statistic)}
    rep = {"provenance": prov, "population": "box D corrected pure-film, converged", "n": int(len(D)),
           "eta_T_in_equals_eta_0": True,
           "eta_note": "the solver takes eta_0 as the service viscosity (solver.py:204-205); R-54.6's two "
                       "variants coincide, so its 'both collapse equally' clause is met trivially",
           "operationalisation": "see the runner's docstring (Code's, committed before computing)",
           "bands": rep_b, "failing_stratifiers": fails, "collapses": not fails,
           "R55_8_nahme_concentration": conc, "memory_sign": "modulus + D-6 bound (box D rows as stored)"}
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-63.4 collapse residuals")
    guarded_path(str(REPORT), what="R-63.4 collapse report")
    D[["sample_id", "band", "theta_min_deg", "S", "Phi", "e", "eta_decade", "c0_bin", "rise", "Wi_proxy"]].assign(
        git_commit=prov["commit"]).to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n", "failing_stratifiers", "collapses", "R55_8_nahme_concentration")},
                     indent=2, default=float))
    for b in rep_b:
        print(b["band_deg"], "s_e=%.4f" % b["scatter_s_e"], {k: v for k, v in b["verdict"].items()},
              {k: round(v, 2) for k, v in b["partial_effect_IQR_over_s_e"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
