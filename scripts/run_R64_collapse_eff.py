"""R-64.1: the collapse on the effective viscosity, registered before running.  No solves.

Procedure: EXACTLY the committed R-63.4 procedure (scripts/run_R63_collapse.py: population, cubic
spline of ln Phi in ln S per theta_min band, s_e = 1.4826 MAD, ten ln-S bins), with eta_0 in S
replaced by an effective viscosity:
  eta0  S itself (the R-63.4 basis), reproduced here as the control;
  (a)   eta_e = eta_s = (eta_s/eta_0) eta_0, the high-shear plateau: S_a = S / (eta_s/eta_0);
  (b)   eta_e = eta_G(Wi_char), the solver's Giesekus viscometric viscosity at the characteristic
        Weissenberg number (R64_wichar.parquet; definition quoted there): S_b = S eta_0 / eta_G.
Discriminator (R-64.1): the STRATIFICATION RATIO = the median over ln-S bins of the spread of the
residual across eta_s/eta_0 terciles, divided by s_e (2.3 on the eta_0 basis at R-63).

Registered (R-64.1), read here as holding in EVERY theta_min band (Code's reading, stated before
running):
  1. (b): ratio < 1.0, and s_e smaller than on the eta_0 basis;
  2. (a): ratio < the eta_0 basis ratio, but > (b)'s;
  3. failing: (b) at or above 1.0 in any band -> STOP for section 5's map; the residual is then
     reported stratified by lam, alpha (fixed at 0.25: not a varying input) and Wi_char.
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
WI = OUT / "R64_wichar.parquet"
DEST = OUT / "R64_collapse_eff.parquet"
REPORT = OUT / "R64_collapse_eff.report.json"


def fit_band(sub, col):
    from scipy.interpolate import LSQUnivariateSpline
    s = sub.sort_values(col)
    x, y = np.log(s[col].values), s.lnPhi.values
    sp = LSQUnivariateSpline(x, y, np.quantile(x, [0.2, 0.4, 0.6, 0.8]), k=3)
    e = pd.Series(y - sp(x), index=s.index)
    se = float(1.4826 * np.median(np.abs(e - np.median(e))))
    return e, se


def main() -> int:
    import run_R63_collapse as C
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-64.1 effective-viscosity collapse")
    D = pd.read_parquet(C.DP).sort_values("sample_id").reset_index(drop=True)
    O = pd.read_parquet(C.OPS).sort_values("sample_id").reset_index(drop=True)
    D["S"] = O.S.values
    D = D[D.outcome == "converged"].copy()
    W = pd.read_parquet(WI).set_index("sample_id")
    D = D.join(W[["Wi_char", "eta_G_char", "F_repro_rel"]], on="sample_id")
    assert D.Wi_char.notna().all() and float(D.F_repro_rel.max()) <= 1e-9
    D["Phi"] = D.h_min_bnd / D.c0
    D["lnPhi"] = np.log(D.Phi)
    D["S_eta0"] = D.S
    D["S_a"] = D.S / D.eta_s_ratio
    D["S_b"] = D.S * D.eta_0 / D.eta_G_char
    D["es_q"] = pd.qcut(D.eta_s_ratio, 3, labels=False)
    out, ratios, scat = {}, {}, {}
    for basis, col in (("eta0", "S_eta0"), ("a_eta_s", "S_a"), ("b_eta_G_Wi_char", "S_b")):
        bands = []
        for i, (t0, t1) in enumerate(C.BANDS):
            m = (D.theta_min_deg >= t0) & (D.theta_min_deg < t1)
            sub = D[m].copy()
            e, se = fit_band(sub, col)
            sub["e"] = e
            sub["lnS"] = np.log(sub[col])
            D.loc[sub.index, f"e_{basis}"] = e
            st = C.strat(sub, "es_q")
            r = st["median_spread"] / se
            bands.append({"band_deg": [t0, min(t1, 40.0)], "n": int(len(sub)), "scatter_s_e": se,
                          "solvent_stratification_ratio": r, "order_consistency": st["extreme_order_consistency"],
                          "eta0_decade_ratio": C.strat(sub.assign(eta_decade=np.floor(np.log10(sub.eta_0)).astype(int)),
                                                       "eta_decade")["median_spread"] / se,
                          "c0_bin_ratio": C.strat(sub.assign(c0_bin=pd.qcut(sub.c0, 9, labels=False)),
                                                  "c0_bin")["median_spread"] / se})
        out[basis] = bands
        ratios[basis] = [b["solvent_stratification_ratio"] for b in bands]
        scat[basis] = [b["scatter_s_e"] for b in bands]
    r0, ra, rb = (np.array(ratios[k]) for k in ("eta0", "a_eta_s", "b_eta_G_Wi_char"))
    s0, sb = np.array(scat["eta0"]), np.array(scat["b_eta_G_Wi_char"])
    reg1 = bool((rb < 1.0).all() and (sb < s0).all())
    reg2 = bool((ra < r0).all() and (ra > rb).all())
    rep = {"provenance": prov, "n": int(len(D)), "bases": out,
           "registered": {"1_b_ratio_below_1_and_scatter_reduced": reg1,
                          "2_a_between_eta0_and_b": reg2,
                          "STOP_for_section5_map": not bool((rb < 1.0).all())},
           "scatter_factor_exp_s_e": {k: [float(np.exp(v)) for v in scat[k]] for k in scat},
           "memory_sign": "modulus + D-6 bound (box D rows as stored)",
           "Wi_char_range": [float(D.Wi_char.min()), float(D.Wi_char.max())],
           "eta_G_over_eta0_range": [float((D.eta_G_char / D.eta_0).min()), float((D.eta_G_char / D.eta_0).max())]}
    if not (rb < 1.0).all():
        from scipy.stats import spearmanr
        e = D["e_b_eta_G_Wi_char"]
        rep["stop_diagnostics"] = {"spearman_e_vs_ln_lam": float(spearmanr(e, np.log(D.lam)).statistic),
                                   "spearman_e_vs_ln_Wi_char": float(spearmanr(e, np.log(D.Wi_char)).statistic),
                                   "spearman_e_vs_eta_s_ratio": float(spearmanr(e, D.eta_s_ratio).statistic),
                                   "alpha": "fixed at 0.25 (campaign_config_v4.ALPHA_FIXED): not a varying input"}
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-64.1 collapse residuals")
    guarded_path(str(REPORT), what="R-64.1 collapse report")
    D[["sample_id", "theta_min_deg", "S_eta0", "S_a", "S_b", "Wi_char", "eta_G_char", "Phi",
       "e_eta0", "e_a_eta_s", "e_b_eta_G_Wi_char"]].assign(git_commit=prov["commit"]).to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("registered", "scatter_factor_exp_s_e")}, indent=2, default=float))
    for k, v in out.items():
        print(k, [(round(b["solvent_stratification_ratio"], 2), round(b["scatter_s_e"], 3)) for b in v])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
