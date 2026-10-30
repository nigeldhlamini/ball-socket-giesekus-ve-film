"""R-68.1: D-9 in ratios.  G = alpha p_ch (Barus; G* = alpha* p_ch for Roelands) and the linear lift
law Lambda_pv/Lambda_iso = 1 + kappa G + O(G^2).  Newtonian path, as R-67.1 (run_R67_piezo.Point:
ball-socket assembly, model="newtonian", eta_a = eta_s, solver._forward at the stored clearance,
target F_app - F_pocket, production mesh 32x24); no frozen file is edited.

kappa (first-order functional, R-68.1).  With p = q + alpha q^2/2 + O(alpha^2) and the rim value
q(p_ch) = p_ch - alpha p_ch^2/2, the isoviscous q-solution is q_iso - (alpha p_ch^2/2) Pi_h, where
Pi_h is the hydrostatic field with unit rim pressure, zero mouth pressure and no motion.  Then
p = p_iso + (alpha p_ch^2/2)(P^2 - Pi_h), P = p_iso/p_ch, so at fixed c the load balance gives
    Lambda_pv/Lambda_iso - 1 = -(alpha p_ch^2/2) LC[P^2 - Pi_h] / (delta dF/ddelta) = kappa G,
    kappa = -(p_ch/2) LC[P^2 - Pi_h] / (delta dF/ddelta),
LC the solver's load projection (sph.load_capacity), dF/ddelta the total (film + asperity) load
along the equilibrium path (central difference in eps, eps_dot = omega eps through _forward).
Pi_h is _forward with omega = 0 (no squeeze, no spin), p_ch = 1, p_case = 0.

Runs: (1) kappa at E15, E13, E5; kappa_fit = least-squares slope through the origin of the Barus
sweep for G <= 0.7.  (2) The Barus sweep G in {0.1, ..., 1.0}, alpha = G / p_ch.  (3) The ratio test
at E15: p_ch and eta_s halved (S unchanged), alpha doubled (G unchanged).  (4) The c0 axis at the
E15 arm, 1-6 um in 0.25 um steps, Roelands VG46 and isoviscous; the clearance is box G's stored
c_equilibrium where the grid has it (1-4 um) and c0 + (a + b c0) beyond, (a, b) fitted to the
grid's c_equilibrium - c0 (a few nm).
Registered (R-68.1): 1. kappa within 3 % of kappa_fit at each point; 2. 1 + kappa G within 1 % of
the sweep for G <= 0.7 at E15 and E13, 2 % at E5; 3. the ratio test moves Lambda_pv/Lambda_iso by
< 1e-3; 4. the Roelands c0 optimum in [3.25, 4.25] um (ratio [1.44, 1.89]) and Lambda(6 um) below
Lambda*_pv by more than 3x the production root-finder uncertainty.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
DEST = OUT / "R68_piezo_ratio.parquet"
REPORT = OUT / "R68_piezo_ratio.report.json"
GS = [round(0.1 * k, 1) for k in range(1, 11)]
C0_AXIS = [round(1.0 + 0.25 * k, 2) for k in range(21)]
MEMORY_SIGN = "signed"     # R-59.3: explicit; the Newtonian path carries no memory term
ID = (lambda p: p, lambda q: q)


def barus(G, p_ch):
    import p3_lee_fluid as LF
    a = G / p_ch
    return (lambda p: LF.barus_q(p, a), lambda q: LF.barus_p(q, a))


def kappa(P):
    iso = P.solve(*ID)
    eps, pch = iso["eps"], P.ci.p_ch
    res, sph = P.slv._forward(P.ci, eps, P.fl, P.c)
    Pn = np.asarray(res.pressure) / pch
    ci_h = dataclasses.replace(P.ci, omega=0.0, p_ch=1.0, p_case=0.0)
    res_h, _ = P.slv._forward(ci_h, eps, P.fl, P.c)
    Pi_h = np.asarray(res_h.pressure)
    LC = float(sph.load_capacity(Pn ** 2 - Pi_h))
    h = 1e-4
    sp, sm = P.state(eps * (1 + h), *ID), P.state(eps * (1 - h), *ID)
    dF = (sp["F_film"] + sp["F_asp"]) - (sm["F_film"] + sm["F_asp"])
    dd = (sp["h_min_bnd"] - sm["h_min_bnd"]) / P.c
    return iso, -(pch / 2.0) * LC / (iso["delta"] * dF / dd), dict(LC_m2=LC, dFddelta=dF / dd,
                                                                    P_peak=float(Pn.max()), Pi_h_max=float(Pi_h.max()))


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    from scipy.optimize import brentq
    import run_R67_piezo as R7
    import p3_lee_fluid as LF
    prov = require_clean_tree("R-68.1 D-9 in ratios")
    pts, B, g15 = R7.points()
    rows, rep = [], {"provenance": prov, "points": {}}
    for P in pts:
        iso, kap, kx = kappa(P)
        L0, pch = iso["Lam"], P.ci.p_ch
        sweep = {}
        for G in GS:
            s = P.solve(*barus(G, pch))
            sweep[G] = s["Lam"] / L0
            rows.append(dict(point=P.name, fluid_law=f"barus_G{G}", G=G, ratio=s["Lam"] / L0, ratio_lin=1 + kap * G,
                             **{k: s[k] for k in ("eps", "delta", "Lam", "p_peak", "q_peak", "F_film", "F_asp")}))
        g = np.array([G for G in GS if G <= 0.7 + 1e-9])
        y = np.array([sweep[G] - 1 for G in g])
        kfit = float((g @ y) / (g @ g))
        dev = max(abs(1 + kap * G - sweep[G]) / sweep[G] for G in g)
        gc = brentq(lambda G: P.solve(*barus(G, pch))["Lam"] - 3.0, 1e-4, 2.0, xtol=1e-7)
        L = R7.laws(P.x["T_in"] + 273.15)
        ro = P.solve(*L["roelands_LeeVG46"][:2])
        qinf = L["roelands_LeeVG46"][2]
        rows.append(dict(point=P.name, fluid_law="roelands_LeeVG46", G=pch / qinf, ratio=ro["Lam"] / L0,
                         **{k: ro[k] for k in ("eps", "delta", "Lam", "p_peak", "q_peak", "F_film", "F_asp")}))
        rows.append(dict(point=P.name, fluid_law="isoviscous", G=0.0, ratio=1.0,
                         **{k: iso[k] for k in ("eps", "delta", "Lam", "p_peak", "q_peak", "F_film", "F_asp")}))
        rep["points"][P.name] = {
            "p_ch_Pa": pch, "Lam_iso": L0, "kappa": kap, "kappa_fit": kfit, "kappa_rel_diff": kap / kfit - 1,
            "lin_dev_max_pct_G_le_0.7": 100 * dev, "sweep": sweep, "G_crit_exact": gc,
            "G_crit_law": (3.0 / L0 - 1.0) / kap, "Lam_roelands": ro["Lam"], "factor_roelands": ro["Lam"] / L0,
            "G_star_roelands": pch / qinf, "q_peak_over_q_inf_roelands": ro["q_peak"] / qinf,
            "p_peak_over_p_ch_iso": iso["p_peak"] / pch, **kx}
    e = pts[0]
    pch = e.ci.p_ch
    rf = rep["points"]["E15"]["factor_roelands"]
    rep["G_lee_barus_equivalent"] = brentq(lambda G: e.solve(*barus(G, pch))["Lam"] / rep["points"]["E15"]["Lam_iso"] - rf,
                                           0.05, 2.0, xtol=1e-7)
    rep["G_lee_roelands"] = LF.alpha_star(373.15) * pch
    # (3) the ratio test at E15
    Gt = 0.5
    x2 = dict(e.x, p_ch=e.x["p_ch"] / 2, eta_0=e.x["eta_0"] / 2)
    P2 = R7.Point("E15_ratio", x2, int(g15["arm"]), None, e.sigma, e.c, e.delta_solver, e.Lam_solver)
    r1 = e.solve(*barus(Gt, pch))["Lam"] / e.solve(*ID)["Lam"]
    r2 = P2.solve(*barus(Gt, pch / 2))["Lam"] / P2.solve(*ID)["Lam"]
    rep["ratio_test"] = {"G": Gt, "ratio_base": r1, "ratio_halved_pch_eta": r2, "abs_change": abs(r2 - r1),
                         "differs_only_in": "p_case/p_ch (doubled) and the contact arm's absolute asperity load"}
    # (4) the c0 axis at the E15 arm
    Bg = B[(B.model_id == g15["arm"]) & (B.memory_sign == "signed") & (B.outcome == "converged")].sort_values("c0")
    ab = np.polyfit(Bg.c0.values, (Bg.c_equilibrium - Bg.c0).values, 1)
    stored = {round(1e6 * r.c0, 2): float(r.c_equilibrium) for _, r in Bg.iterrows()}
    L = R7.laws(e.x["T_in"] + 273.15)
    axis = []
    for c0 in C0_AXIS:
        c = stored.get(c0, c0 * 1e-6 + np.polyval(ab, c0 * 1e-6))
        x = dict(e.x, c0=c0 * 1e-6)
        P = R7.Point("E15_axis", x, int(g15["arm"]), None, e.sigma, c, np.nan, np.nan)
        for name in ("isoviscous", "roelands_LeeVG46"):
            s = P.solve(*L[name][:2])
            rec = dict(point="E15_axis", fluid_law=name, c0_um=c0, c=c, c_source="box G" if c0 in stored else "fit",
                       outcome="converged" if s else "no equilibrium")
            if s:
                tmin = np.deg2rad(x["theta_min_deg"])
                rec.update({k: s[k] for k in ("eps", "delta", "Lam", "p_peak", "q_peak", "F_film", "F_asp")})
                rec["u_root"] = s["Lam"] * np.cos(tmin) * (1e-4 + 1e-4 * s["eps"]) / s["delta"]
            axis.append(rec)
            rows.append(rec)
    A = pd.DataFrame(axis)
    ax = {}
    for name in ("isoviscous", "roelands_LeeVG46"):
        a = A[(A.fluid_law == name) & (A.outcome == "converged")].sort_values("c0_um").reset_index(drop=True)
        i = int(a.Lam.values.argmax())
        c0s = float(a.c0_um[i])
        if 0 < i < len(a) - 1:      # parabolic refinement through the three grid points about the maximum
            cf = np.polyfit(a.c0_um[i - 1:i + 2], a.Lam[i - 1:i + 2], 2)
            c0s = float(-cf[1] / (2 * cf[0]))
        ax[name] = {"c0_star_grid_um": float(a.c0_um[i]), "c0_star_refined_um": c0s, "Lam_star": float(a.Lam[i]),
                    "Lam_at_6um": float(a.Lam.iloc[-1]), "u_root_at_star": float(a.u_root[i]),
                    "interior": bool(0 < i < len(a) - 1)}
    c0iso = ax["isoviscous"]["c0_star_grid_um"]
    for name in ax:      # grid optimum on both sides (E15's own convention: c0* = 2.25 um is a grid value)
        ax[name]["c0_ratio_to_iso"] = ax[name]["c0_star_grid_um"] / c0iso
        ax[name]["c0_ratio_to_iso_refined"] = ax[name]["c0_star_refined_um"] / ax["isoviscous"]["c0_star_refined_um"]
    rep["c0_axis"] = ax
    pv = ax["roelands_LeeVG46"]
    pE = rep["points"]
    rep["registered"] = {
        "1_kappa_within_3pct_of_fit": {k: abs(v["kappa_rel_diff"]) <= 0.03 for k, v in pE.items()},
        "2_linear_law": {k: bool(v["lin_dev_max_pct_G_le_0.7"] <= (2.0 if k == "E5" else 1.0)) for k, v in pE.items()},
        "3_ratio_test_below_1e-3": bool(rep["ratio_test"]["abs_change"] < 1e-3),
        "4_c0_optimum_interior": bool(pv["interior"] and 3.25 <= pv["c0_star_grid_um"] <= 4.25
                                      and pv["Lam_star"] - pv["Lam_at_6um"] > 3 * pv["u_root_at_star"])}
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-68.1 piezo ratios")
    guarded_path(str(REPORT), what="R-68.1 report")
    pd.DataFrame(rows).assign(git_commit=prov["commit"], memory_sign="none (Newtonian path)").to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("registered", "ratio_test", "c0_axis", "G_lee_roelands",
                                          "G_lee_barus_equivalent")}, indent=1, default=float))
    for k, v in pE.items():
        print(k, {q: (round(v[q], 4) if isinstance(v[q], float) else v[q]) for q in
                  ("Lam_iso", "kappa", "kappa_fit", "lin_dev_max_pct_G_le_0.7", "G_crit_exact", "G_crit_law",
                   "factor_roelands", "q_peak_over_q_inf_roelands", "p_peak_over_p_ch_iso")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
