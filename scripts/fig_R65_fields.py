"""R-65.4: Paper 3's Figure 5, figures/fig_fields.pdf -- the film along the land at the E15 optimum.

The point: box G's argmax (boxG_E15.report.json: arm 9, c0 = 2.25 um, sigma* lo, rated corner,
theta_min = 20), its inputs read from its own row of boxG_E15.parquet.  Solved again on the
SIGNED path at the production mesh 32x24 with the production root-finder, the solver built as
campaign._evaluate_x_inner builds it (run_R57_mesh_error._solver, contact = arm 9), and the
equilibrium must reproduce box G's stored h_min_bnd, eps_ecc and c_equilibrium.  The fields are
then the forward solve at that equilibrium (solver.py:_forward).

Panels against theta from theta_min to theta_max (phi-means; the film is axisymmetric, asserted):
  (a) film pressure: solver (heavy) and the same geometry (same c, eps) solved Newtonian at eta_0
      (light, src/solver.py:681 solve_newtonian with fluid.eta_0); the pocket pressure p_ch at
      the rim marked; the asperity pressure added light and open if non-zero there.
  (b) h(theta);  (c) eta_bar/eta_0;  (d) film-mean temperature rise, by the solver's own route
      (L2: phi-mean of solve_l2_spherical - T_in; L1: the bulk advective rise, which has no field).
Persists results/ch9/R65_fields.parquet with provenance.  The encoding is asserted against v5's
Fig. 5 caption before saving.
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
DEST = OUT / "R65_fields.parquet"
FIGDIR = pathlib.Path("Z:/Papers/Paper 3/New/figures")
V5 = pathlib.Path("Z:/Papers/Paper 3/New/paper3_v5_draft.tex")
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
MEMORY_SIGN = "signed"


def main() -> int:
    import argparse
    import warnings
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--thermal", choices=("default", "mineral"), default="default",
                    help="R-66.4/R-67.2: 'mineral' re-solves with the D-8 class properties "
                         "(scripts/bs_thermal_props.py) and is checked against R67_D8_claims.parquet")
    thermal = ap.parse_args().thermal
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-65.4 field figure")
    import fig_R61_update as G          # house style (rcParams, Okabe-Ito, HEAVY/LIGHT)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from run_R57_mesh_error import _solver, _contact
    import ball_socket.thermal as bs_thermal

    rep = json.loads((OUT / "boxG_E15.report.json").read_text(encoding="utf-8"))
    arm, c0_um = int(rep["argmax"]["arm"]), float(rep["argmax"]["c0_um"])
    B = pd.read_parquet(OUT / "boxG_E15.parquet")
    row = B[(B.model_id == arm) & (np.isclose(B.c0 * 1e6, c0_um)) & (B.memory_sign == MEMORY_SIGN)]
    assert len(row) == 1 and row.outcome.iloc[0] == "converged"
    row = row.iloc[0]
    x = {k: float(row[k]) for k in AXES}
    dest = DEST
    if thermal == "mineral":
        import bs_thermal_props as W
        slv, ci = W.build(x, _contact(arm), None, thermal_props=W.MINERAL)
        Dm = pd.read_parquet(OUT / "R67_D8_claims.parquet")
        row = Dm[(Dm.point == f"E15_c0_{c0_um:g}") & (Dm.props == "mineral")].iloc[0]
        dest = OUT / "R67_fields_mineral.parquet"
    else:
        slv, ci = _solver(x, _contact(arm), None)
    slv.memory_sign = MEMORY_SIGN
    perf = slv.solve_equilibrium(ci)
    for k in ("h_min_bnd", "eps_ecc", "c_equilibrium"):
        assert abs(getattr(perf, k) / float(row[k]) - 1) < 1e-9, (k, getattr(perf, k), row[k])
    fluid = slv._fluid(ci)
    c, eps = perf.c_equilibrium, perf.eps_ecc
    res, sph = slv._forward(ci, eps, fluid, c)
    h = sph.gap()
    eta_bar = res.eta_bar if res.eta_bar is not None else np.full_like(h, x["eta_0"])
    gdot = sph.gap_averaged_shear_rate(res.pressure, eta_bar)
    # Newtonian at eta_0, same geometry (c, eps): the solver's own Newtonian path
    model0 = slv.model
    slv.model = "newtonian"
    res_n, _ = slv._forward(ci, eps, fluid, c)
    slv.model = model0
    p_asp = np.asarray(slv.contact.pressure(h)) if slv.contact is not None else np.zeros_like(h)
    # thermal, by the solver's own routing (solver.py:468-500)
    th = slv.thermal
    V_char = slv.spin_ratio * x["omega"] * slv.geom.R_b * np.sin(slv.geom.theta_max)
    Br = bs_thermal.brinkman(x["eta_0"], V_char, th)
    route_q, _ = bs_thermal.q_min_route_to_l2(perf.P_loss, perf.Q_leak, th)
    if Br > bs_thermal.BR_THRESHOLD or route_q:
        T = bs_thermal.solve_l2_spherical(sph, eta_bar, gdot, th, x["T_in"] + 273.15)
        dT = (T - 273.15 - x["T_in"]).mean(axis=1)
        route = "L2"
    else:
        dT = np.full(sph.mesh.Ntheta, bs_thermal.l1_bulk_delta_T(perf.P_loss, perf.Q_leak, th))
        route = "L1 (bulk advective rise; no field)"
    assert route.split()[0] == perf.thermal_mode, (route, perf.thermal_mode)

    def prof(f):
        f = np.asarray(f, dtype=float)
        m = f.mean(axis=1)
        spread = float(np.max(np.abs(f - m[:, None])) / max(np.max(np.abs(m)), 1e-300))
        return m, spread
    p, s_p = prof(res.pressure)
    pn, _ = prof(res_n.pressure)
    hh, s_h = prof(h)
    er, _ = prof(eta_bar / x["eta_0"])
    pa, _ = prof(p_asp)
    assert s_p < 1e-6 and s_h < 1e-9, ("not axisymmetric", s_p, s_h)
    th_deg = np.rad2deg(sph.mesh.theta)
    F = pd.DataFrame({"theta_deg": th_deg, "p_Pa": p, "p_newtonian_eta0_Pa": pn, "p_asperity_Pa": pa,
                      "h_m": hh, "eta_over_eta0": er, "dT_film_mean_K": dT})
    meta = {"point": f"E15 optimum: arm {arm}, c0 = {c0_um} um, sigma* lo, theta_min = {x['theta_min_deg']}",
            "mesh": f"{slv.Ntheta}x{slv.Nphi}", "memory_sign": MEMORY_SIGN, "thermal_route": route,
            "p_pocket_Pa": x["p_ch"], "c_equilibrium_m": c, "eps_ecc": eps, "h_min_bnd_m": perf.h_min_bnd,
            "theta_max_deg": float(np.rad2deg(slv.geom.theta_max)), "phi_spread_p": s_p,
            "eta_s_ratio": x["eta_s_ratio"], "F_asp_N": float(perf.F_asp), "F_film_N": float(perf.F_film)}

    # ---- figure
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.6), sharex=True, constrained_layout=True)
    (a, b), (cc, d) = axs
    a.plot(th_deg, p / 1e6, color=G.C_BLUE, lw=G.HEAVY, label="solver (Giesekus, signed)")
    a.plot(th_deg, pn / 1e6, color=G.C_ORANGE, lw=G.LIGHT, ls="--", label=r"Newtonian at $\eta_0$")
    a.plot([float(np.rad2deg(slv.geom.theta_min))], [x["p_ch"] / 1e6], marker="o", ms=8, mfc=G.C_VERM,
           mec="k", ls="none", label=r"pocket pressure $p_{ch}$ at the rim", clip_on=False)
    asp = bool(np.max(pa) > 0)
    if asp:
        a.plot(th_deg, pa / 1e6, color=G.C_PURPLE, lw=G.LIGHT, marker="o", mfc="none", ms=3,
               label=f"asperity pressure (max {pa.max() / 1e6:.2g} MPa)")
    a.set_ylabel("film pressure (MPa)")
    a.set_title("(a)", loc="left")
    fig.legend(handles=a.lines, loc="outside upper center", ncol=2, frameon=False, fontsize=7)
    b.plot(th_deg, hh * 1e6, color=G.C_BLUE, lw=G.HEAVY)
    b.set_ylabel(r"$h$ ($\mu$m)")
    b.set_title("(b)", loc="left")
    cc.plot(th_deg, er, color=G.C_BLUE, lw=G.HEAVY)
    cc.axhline(x["eta_s_ratio"], color=G.C_GREY, lw=G.LIGHT, ls=":", label=r"plateau $\eta_s/\eta_0$")
    cc.set_ylabel(r"$\bar\eta/\eta_0$")
    cc.set_title("(c)", loc="left")
    cc.legend(frameon=False, fontsize=6.5)
    d.plot(th_deg, dT * 1e3, color=G.C_BLUE, lw=G.HEAVY)
    d.set_ylabel(r"film-mean $\Delta T$ (mK)")
    d.set_title("(d)", loc="left")
    for ax in (cc, d):
        ax.set_xlabel(r"polar angle $\theta$ (deg)")
    for ax in axs.flat:
        ax.set_xlim(float(np.rad2deg(slv.geom.theta_min)), meta["theta_max_deg"])

    # ---- encoding asserted against v5's Fig. 5 caption
    tex = V5.read_text(encoding="utf-8")
    cap = tex[tex.index("\\label{fig:fields}") - 900: tex.index("\\label{fig:fields}")]
    cap = " ".join(cap[cap.index("\\caption{"):].split())
    for phrase in ("(a) film pressure, solver (heavy) and Newtonian film at $\\eta_0$ (light)",
                   "with the pocket pressure at the rim", "(b) film thickness",
                   "(c) shear-thinned viscosity relative to $\\eta_0$", "(d) film-mean temperature rise",
                   "against polar angle from the rim to the mouth",
                   # R-66.4: the caption now names the asperity pressure and the plateau
                   "the asperity pressure (light, open), which lies on zero at this scale",
                   "with the high-shear plateau $\\eta_s/\\eta_0$ dotted"):
        assert phrase in cap, phrase
    plateau = [ln for ln in cc.lines if ln.get_label().startswith("plateau")]
    assert len(plateau) == 1 and plateau[0].get_linestyle() == ":" and plateau[0].get_linewidth() == G.LIGHT
    sol, newt = a.lines[0], a.lines[1]
    assert sol.get_linewidth() == G.HEAVY and newt.get_linewidth() == G.LIGHT and newt.get_linestyle() == "--"
    assert a.lines[2].get_marker() == "o" and a.lines[2].get_xdata()[0] == float(np.rad2deg(slv.geom.theta_min))
    assert all(ax.get_xlim()[0] < ax.get_xlim()[1] for ax in axs.flat)   # rim (theta_min) left, mouth right
    if asp:
        assert a.lines[3].get_markerfacecolor() == "none" and a.lines[3].get_linewidth() == G.LIGHT

    from ball_socket.immutable import guarded_path
    guarded_path(str(dest), what=f"R-65.4 fields ({thermal} thermal properties)")
    F.assign(git_commit=prov["commit"], memory_sign=MEMORY_SIGN, point=meta["point"],
             mesh=meta["mesh"], thermal_route=route, thermal_props=thermal).to_parquet(dest, index=False)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig_fields.pdf")
    (FIGDIR / "fig_fields.json").write_text(json.dumps({"figure": "fig_fields.pdf", "provenance": prov,
                                                        "source": str(dest), "thermal_props": thermal, **meta,
                                                        "asperity_pressure_nonzero": asp,
                                                        "eta_ratio_range": [float(er.min()), float(er.max())],
                                                        "dT_range_K": [float(dT.min()), float(dT.max())],
                                                        "p_peak_MPa": float(p.max() / 1e6),
                                                        "p_newtonian_peak_MPa": float(pn.max() / 1e6)},
                                                       indent=1, default=float), encoding="utf-8")
    print(json.dumps(meta, indent=1, default=float))
    print("eta ratio", float(er.min()), float(er.max()), "dT K", float(dT.min()), float(dT.max()), "asp", asp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
