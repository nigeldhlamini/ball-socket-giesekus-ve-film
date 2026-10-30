"""R-64.4: Paper 3 figures (elsarticle width) into Z:/Papers/Paper 3/New/figures.  No solves.

  fig_joint.pdf      port of scripts/figures/f01_domain_schematic.py (the August F1), geometry from
                     the frozen configuration; the RIM is labelled; the clearance is drawn enlarged
  fig_mechanism.pdf  (a) p~(theta) from the vendored quadrature (scripts/p3_rim_asymptote.py: the
                     first integral and k of _f, lines 66-73) at theta_min = 20 deg and
                     delta in {0.1, 0.03, 0.01}, against x = theta - theta_min on a log axis, with
                     the far-field asymptote 2 cos^4 / (x sin^3) dotted and each band delta/tan
                     marked; (b) a sketch of the rim wedge, the pinned pocket and the flow turning
                     toward the mouth
  fig_clear.pdf      fig_R61_update.fig_u2 at elsarticle width
Each asserts its encoding against v4's caption before saving (fig_R61_update.assert_encoding).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT_DEFAULT = pathlib.Path("Z:/Papers/Paper 3/New/figures")

import fig_R61_update as G  # noqa: E402  (sets the Agg backend and the shared style)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Wedge  # noqa: E402


def fig_joint(out, prov):
    cfg = json.loads((ROOT / "ball_socket" / "campaign_frozen_v4.json").read_text())["config"]
    g = cfg["geometry"]
    R, th_max = g["R_b_m"] * 1e3, g["theta_max_deg"]
    t_lo, t_hi = g["theta_min_axis_deg"]
    gap = 0.10 * R                                       # clearance drawn greatly enlarged
    fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
    ax.add_patch(Wedge((0, 0), R + gap + 0.12 * R, 270 - th_max, 270 + th_max, width=0.12 * R,
                       facecolor="#c8cdd4", edgecolor="#5a6472", lw=1.0, hatch="////", zorder=1))
    ax.add_patch(plt.Circle((0, 0), R, facecolor="#eef1f5", edgecolor="#2b2b2b", lw=1.4, zorder=2))
    ax.add_patch(Wedge((0, 0), R + gap, 270 - t_lo, 270 + t_lo, facecolor="#f6d8b8", edgecolor="none",
                       zorder=1.5))
    ts = np.linspace(np.deg2rad(t_lo), np.deg2rad(th_max), 200)
    for sgn in (1, -1):
        ax.fill_between(sgn * np.sin(ts) * R, -np.cos(ts) * R, -np.cos(ts) * (R + gap), color=G.C_BLUE,
                        alpha=0.45, lw=0, zorder=1.8)
        for th, ls in ((t_lo, "-"), (t_hi, ":")):
            a = np.deg2rad(th)
            ax.plot([0, sgn * (R + gap) * np.sin(a)], [0, -(R + gap) * np.cos(a)], ls, color=G.C_VERM, lw=1.3,
                    label="_rim", zorder=4)
        a = np.deg2rad(th_max)
        ax.plot([0, sgn * (R + gap) * np.sin(a)], [0, -(R + gap) * np.cos(a)], "--", color=G.C_BLUE, lw=1.1,
                label="_mouth", zorder=4)
    a = np.deg2rad(t_lo)
    ax.annotate(r"rim ($\theta_{\min}$)", xy=(R * np.sin(a), -R * np.cos(a)), xytext=(R * 0.55, -R * 1.42),
                fontsize=8, color=G.C_VERM, arrowprops=dict(arrowstyle="->", lw=0.8, color=G.C_VERM))
    ax.text(-R * 1.70, -R * 0.05, rf"$\theta_{{\min}} \in [{t_lo:g}^\circ, {t_hi:g}^\circ]$", fontsize=8,
            color=G.C_VERM)
    ax.text(0, -R * 0.62, "pocket\n($p_{\\mathrm{ch}}$)", ha="center", va="center", fontsize=8, zorder=6)
    a = np.deg2rad(th_max)
    ax.text(R * 1.10, -R * np.cos(a) - 0.05 * R, "mouth\n($\\theta_{\\max}$, $p_{\\mathrm{case}}$)",
            fontsize=8, ha="left", va="center", color=G.C_BLUE)
    ax.text(-R * 1.65, -R * 1.25, "film on the land\n(clearance enlarged)", fontsize=7.5, color=G.C_BLUE)
    ax.add_patch(FancyArrowPatch((0, R * 1.55), (0, R * 0.35), arrowstyle="-|>", mutation_scale=13,
                                 color="#2b2b2b", lw=1.4, zorder=6))
    ax.text(0.06 * R, R * 1.30, r"$F_{\mathrm{app}} = p_{\mathrm{ch}} A_p$", fontsize=8, ha="left")
    ax.text(R * 0.25, R * 0.55, rf"$R_b = {R:g}$ mm", fontsize=8)
    ax.set_xlim(-R * 1.7, R * 1.95); ax.set_ylim(-R * 1.55, R * 1.7)
    ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(out / "fig_joint.pdf")
    return fig, {"figure": "fig_joint.pdf", "provenance": prov, "R_b_mm": R, "theta_max_deg": th_max,
                 "theta_min_axis_deg": [t_lo, t_hi], "clearance_drawn_as_fraction_of_R": gap / R,
                 "source": "ported from scripts/figures/f01_domain_schematic.py"}


def p_tilde(x, delta, tmin_deg, tmax=None):
    """p~(theta_min + x) from the vendored first integral (p3_rim_asymptote.py:66-73)."""
    from scipy.integrate import quad
    import p3_rim_asymptote as RA
    tmax = RA.TMAX if tmax is None else tmax
    tmin = np.deg2rad(tmin_deg)
    eps = (1 - delta) / np.cos(tmin)
    H = lambda t: 1.0 - eps * np.cos(t)
    pts = [tmin + s * (tmax - tmin) for s in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)]
    q = dict(limit=800, epsabs=0, epsrel=1e-11)
    a = quad(lambda t: np.sin(t) / H(t) ** 3, tmin, tmax, points=pts, **q)[0]
    b = quad(lambda t: 1 / (np.sin(t) * H(t) ** 3), tmin, tmax, points=pts, **q)[0]
    k = a / b
    dp = lambda t: (k - np.sin(t) ** 2) / (np.sin(t) * H(t) ** 3)
    out, acc, prev = [], 0.0, tmin
    for xi in x:
        acc += quad(dp, prev, tmin + xi, limit=400, epsabs=0, epsrel=1e-10)[0]
        prev = tmin + xi
        out.append(acc)
    return np.array(out)


def fig_mechanism(out, prov):
    import p3_rim_asymptote as RA
    tmin_deg = 20.0
    tmin = np.deg2rad(tmin_deg)
    xmax = RA.TMAX - tmin
    x = np.logspace(-5, np.log10(0.98 * xmax), 260)
    C = 2 * np.cos(tmin) ** 4 / np.sin(tmin) ** 3
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.9), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    styles = {0.1: (G.C_BLUE, "-"), 0.03: (G.C_ORANGE, "--"), 0.01: (G.C_GREEN, "-.")}
    rep_d = {}
    for dl, (col, ls) in styles.items():
        p = p_tilde(x, dl, tmin_deg)
        ax.plot(x, p, color=col, ls=ls, lw=G.HEAVY, label=rf"$\delta = {dl:g}$")
        xb = dl / np.tan(tmin)
        ax.axvline(xb, color=col, ls=ls, lw=G.LIGHT, label="_band")
        far = (x > 10 * xb) & (x < 0.1 * xmax)
        rep_d[str(dl)] = {"band_x": xb, "peak_p": float(p.max()), "x_at_peak": float(x[p.argmax()]),
                          "far_field_p_times_x_over_C_median": float(np.median(p[far] * x[far] / C)) if far.any() else None}
    xa = x[x > 1e-3]
    ax.plot(xa, C / xa, color="k", ls=":", lw=G.LIGHT, label=r"$2\cos^4\theta_{\min}/(x\sin^3\theta_{\min})$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-5, xmax); ax.set_ylim(1e-1, None)
    ax.set_xlabel(r"$x = \theta - \theta_{\min}$ (rad)"); ax.set_ylabel(r"$\tilde p$")
    ax.set_title(r"(a) $\theta_{\min} = 20^\circ$", loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    G.assert_encoding(ax, heavy=[rf"$\delta = {d:g}$" for d in styles],
                      light=[r"$2\cos^4\theta_{\min}/(x\sin^3\theta_{\min})$"])
    # (b) sketch: the rim wedge.  The pocket is a recess (deep gap) ending at the rim; the land
    # beyond the rim opens as a wedge from the film delta*c.
    tan, h0, dep = 0.30, 0.10, 0.42
    xs = np.linspace(0, 1, 50)
    bx.add_patch(Polygon([[-0.75, 0], [0, 0], [0, dep], [-0.75, dep]], closed=True, facecolor="#f6d8b8",
                         edgecolor="none", hatch="..", zorder=1))
    bx.plot([-0.75, 1], [0, 0], color="#2b2b2b", lw=1.6)                         # ball surface
    bx.plot([-0.75, 0, 0], [dep, dep, h0], color="#5a6472", lw=1.6)             # pocket wall, rim edge
    bx.plot(xs, h0 + tan * xs, color="#5a6472", lw=1.6)                         # land (wedge)
    bx.fill_between(xs, 0, h0 + tan * xs, color=G.C_BLUE, alpha=0.18, lw=0)
    bx.text(-0.70, 0.27, "pocket\n$p = p_{\\mathrm{ch}}$ (pinned)", fontsize=7, va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    bx.annotate("", xy=(0.06, 0), xytext=(0.06, h0), arrowprops=dict(arrowstyle="<->", lw=0.8))
    bx.text(0.09, 0.035, r"$\delta c$", fontsize=8)
    bx.text(0.50, 0.30, r"slope $\tan\theta_{\min}$", fontsize=7, rotation=np.degrees(np.arctan(tan * 1.25)))
    bx.add_patch(FancyArrowPatch((0.30, 0.06), (0.92, 0.22), arrowstyle="-|>", mutation_scale=10,
                                 color=G.C_BLUE, lw=1.3, connectionstyle="arc3,rad=-0.15"))
    bx.text(0.80, 0.08, "to the\nmouth", fontsize=7, color=G.C_BLUE)
    bx.add_patch(FancyArrowPatch((0.26, 0.04), (0.01, 0.04), arrowstyle="-|>", mutation_scale=8,
                                 color=G.C_VERM, lw=0.9, ls="--"))
    bx.text(0.02, -0.07, "rim: flow choked", fontsize=7, color=G.C_VERM)
    bx.set_xlim(-0.8, 1.1); bx.set_ylim(-0.12, 0.56); bx.axis("off")
    bx.set_title("(b) the rim wedge (sketch)", loc="left")
    fig.savefig(out / "fig_mechanism.pdf")
    return fig, {"figure": "fig_mechanism.pdf", "provenance": prov, "theta_min_deg": tmin_deg,
                 "asymptote_C": C, "per_delta": rep_d, "memory_sign": "none (Newtonian quadrature)",
                 "source": "scripts/p3_rim_asymptote.py first integral (lines 66-73)"}


def fig_clear(out, prov):
    """Paper version of fig_u2 after box G: the Newtonian prediction (E15 draft computation, 0.02 um
    grid), the box-G solver grid on the signed path (sigma* lo, max over arms, production mesh), the
    Richardson Lambda* at its optimum, and E13's 6 um point (production, before D-6) for the tail."""
    B = json.loads((ROOT / "results" / "ch9" / "boxG_E15.report.json").read_text(encoding="utf-8"))
    import pandas as pd
    cv = G.e15_curve(np.round(np.arange(1.0, 6.0001, 0.02), 4))
    cv = cv[cv.bracketed]
    i = int(cv.Lambda_lo.values.argmax())
    qa, qb, qc = np.polyfit(cv.c0_um.values[i - 1:i + 2], cv.Lambda_lo.values[i - 1:i + 2], 2)
    lstar = float(qc - qb ** 2 / (4 * qa))
    plat = cv[cv.Lambda_lo >= 0.99 * lstar]
    gc = {float(k): v for k, v in B["curve_sigma_lo_max_over_arms"].items()}
    E13 = pd.read_parquet(ROOT / "results" / "ch9" / "E13_extremum.parquet")
    arm = json.loads((ROOT / "results" / "ch9" / "E13_extremum.report.json").read_text())["where"]["arm"]
    e6 = E13[(E13.theta_min_deg == 20) & (E13.arm == arm) & (E13.c0_um == 6.0)].Lambda_g1.iloc[0]
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    ax.axvspan(plat.c0_um.min(), plat.c0_um.max(), color=G.C_SKY, alpha=0.25, lw=0, label="_plateau")
    ax.plot(cv.c0_um, cv.Lambda_lo, color=G.C_BLUE, lw=G.HEAVY, ls="-", label="Newtonian prediction (closed form)")
    ax.plot(list(gc), list(gc.values()), ls="none", marker="o", ms=5.5, color=G.C_VERM, mec="k", mew=0.5,
            label=r"solver, clearance sweep ($32\times24$)")
    ax.plot([6.0], [e6], ls="none", marker="s", ms=6, color=G.C_ORANGE, mec="k", mew=0.5,
            label=r"solver, $c_0 = 6\,\mu$m ($32\times24$)")
    rr = B["richardson_at_argmax"]
    ax.plot([B["argmax"]["c0_um"]], [rr["Lambda_R"]], ls="none", marker="*", ms=12, color=G.C_PURPLE, mec="k",
            mew=0.6, label="solver optimum, mesh-extrapolated")
    ax.axhline(3.0, color="k", lw=G.LIGHT, ls="-", label="_full film")
    ax.text(5.95, 3.03, r"full film, $\Lambda=3$", ha="right", va="bottom", fontsize=7)
    ax.set_xlim(1, 6); ax.set_ylim(1.0, 3.3)
    ax.set_xlabel(r"radial clearance $c_0$ ($\mu$m)"); ax.set_ylabel(r"$\Lambda = h_{\min}/\sigma^{*}$")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    G.assert_encoding(ax, heavy=["Newtonian prediction (closed form)"],
                      filled_markers=[r"solver, clearance sweep ($32\times24$)", r"solver, $c_0 = 6\,\mu$m ($32\times24$)",
                                      "solver optimum, mesh-extrapolated"])
    fig.savefig(out / "fig_clear.pdf")
    return fig, {"figure": "fig_clear.pdf", "provenance": prov,
                 "memory_sign": {"clearance_sweep_and_optimum": "signed (box G)", "c0_6um": "modulus + D-6 bound (E13)",
                                 "prediction": "none (Newtonian)"},
                 "plateau_1pct_um": [float(plat.c0_um.min()), float(plat.c0_um.max())],
                 "Lambda_star_R": rr["Lambda_R"], "c0_star_um": B["argmax"]["c0_um"], "E13_6um": float(e6)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-64.4 paper figures")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [fig_joint, fig_mechanism, fig_clear]
    for f in jobs:
        fig, rep = f(out, prov)
        (out / rep["figure"].replace(".pdf", ".json")).write_text(json.dumps(rep, indent=1, default=float),
                                                                   encoding="utf-8")
        if a.png:
            fig.savefig(pathlib.Path(a.png) / rep["figure"].replace(".pdf", ".png"), dpi=170)
        plt.close(fig)
        print("wrote", rep["figure"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
