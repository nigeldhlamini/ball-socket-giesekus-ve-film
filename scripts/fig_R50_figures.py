"""R-50.8: two figures.

FM1  Lambda_max against cold clearance at the public rated maximum corner, at the
     registered theta_min floor, both sigma* band ends, with the full-film
     (Lambda = 3) and boundary (Lambda = 1) thresholds drawn.  The memo figure:
     it carries E13 at a glance.  Source: results/ch9/E13_extremum.parquet.
FE4  |memory share| of the film load against the squeeze Deborah number, one
     line per base row, across the registered relaxation-time axis, with the g3
     limit drawn.  Source: results/ch9/E4_lambda_sweep.parquet.

One code path for a number that is both computed and drawn: each figure
ASSERTS that its plotted extremum equals the value the generator emits as a
macro, and the manifest records both.

Colour: the reference categorical slots 1-2 (blue, orange).  The palette
validator needs Node, which this host lacks, so identity is carried twice --
marker shape and line style as well as hue -- and the figures read in greyscale.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "ch9"
FIG = ROOT / "figures" / "ch9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
                     "font.family": "serif"})


def _save(fig, name, manifest):
    paths = {}
    for ext in ("pdf", "png"):
        p = FIG / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths[ext] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest.update({"figure": name, "sha256": paths})
    (FIG / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    plt.close(fig)


def fm1():
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M as S
    import ball_socket.campaign_config_v6 as v6
    th = v6.config_dict()["regime_thresholds"]
    df = pd.read_parquet(OUT / "E13_extremum.parquet")
    conv = df[df.outcome == "converged"]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    drawn_max = -np.inf
    for s, col, mk, ls, lab, off in ((S[0], BLUE, "o", "-", r"$\sigma^*$ = %.3f µm" % (S[0] * 1e6), (6, 5)),
                                     (S[1], ORANGE, "s", "--", r"$\sigma^*$ = %.3f µm" % (S[1] * 1e6), (4, -15))):
        g = conv[np.isclose(conv.sigma_star, s)].groupby("c0_um").Lambda_g1.max()
        ax.plot(g.index, g.values, ls, color=col, lw=2, marker=mk, ms=5, label=lab,
                markeredgecolor="#fcfcfb", markeredgewidth=1)
        # direct label at the curve's right end, where the two curves are apart
        ax.annotate(lab, (g.index[-1], g.values[-1]), xytext=(7, -3),
                    textcoords="offset points", fontsize=8, color=INK)
        drawn_max = max(drawn_max, float(g.max()))
    ax.axhline(th["full_film_min"], color=INK, lw=1)
    ax.text(29.5, th["full_film_min"] + 0.06, r"full film, $\Lambda$ = 3", ha="right",
            va="bottom", fontsize=8, color=INK)
    ax.axhline(th["mixed_min"], color=INK2, lw=0.8, ls=":")
    ax.text(29.5, th["mixed_min"] + 0.06, r"boundary below $\Lambda$ = 1", ha="right",
            va="bottom", fontsize=8, color=INK2)
    no_eq = sorted(set(df.c0_um) - set(conv.c0_um))
    if no_eq:
        ax.axvspan(min(no_eq) - 1, 31, color="#f0efec", zorder=0)
        ax.text((min(no_eq) + 30) / 2, 1.9, "no equilibrium", ha="center", fontsize=8,
                color=INK2)
    ax.set_xlim(1, 31)
    ax.set_ylim(0, 3.4)
    ax.set_xlabel(r"cold radial clearance $c_0$ (µm)")
    ax.set_ylabel(r"largest film parameter $\Lambda_{\max}$")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    rep = json.loads((OUT / "E13_extremum.report.json").read_text())
    assert abs(drawn_max - rep["Lambda_max"]) < 1e-12, (drawn_max, rep["Lambda_max"])
    _save(fig, "FM1_lambda_max_vs_c0", {
        "caption": ("Largest film parameter reached at the public rated maximum corner "
                    "(500 bar, 4500 rpm) at the registered pocket-edge floor theta_min = 20 "
                    "deg, over all 24 contact arms, against cold clearance, at both ends of "
                    "the registered roughness band.  The admissible corner fluid is the "
                    "thickest registered one.  No clearance reaches full film; the best "
                    "point is the tightest registered clearance, and above 14 um no "
                    "equilibrium exists."),
        "artefacts_read": ["results/ch9/E13_extremum.parquet",
                           "results/ch9/E13_extremum.report.json"],
        "drawn_max": drawn_max, "emitted_macro": "BsNineEThirteenLamMax",
        "emitted_value": rep["Lambda_max"]})


def fe4():
    sw = pd.read_parquet(OUT / "E4_lambda_sweep.parquet")
    sw = sw[sw.status == "ok"].copy()
    sw["a"] = sw.share_memory_D1.abs()
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.grid(True, color=GRID, lw=0.6, which="major")
    ax.set_axisbelow(True)
    for base, g in sw.groupby("base"):
        g = g.sort_values("De_sq_bnd")
        e4 = base.startswith("E4:")
        col = ORANGE if e4 else BLUE
        ind = g[g.in_g3_domain]
        ax.plot(ind.De_sq_bnd, ind.a, "-", color=col, lw=1.2 if e4 else 0.9,
                alpha=1.0 if e4 else 0.55)
        ax.plot(g.De_sq_bnd, g.a, ":", color=col, lw=0.8, alpha=0.4)
    ax.axvline(0.2, color=INK, lw=1)
    ax.text(0.2, 6e-7, r"$g_3$ limit, De$_{sq}$ = 0.2", fontsize=8, color=INK,
            ha="center", va="center",
            bbox=dict(boxstyle="square,pad=0.25", fc="#fcfcfb", ec="none"))
    ind_all = sw[sw.in_g3_domain]
    drawn_max = float(ind_all.a.max())
    ax.axhline(drawn_max, color=INK2, lw=0.8, ls="--")
    ax.text(1.2e-3, drawn_max * 1.15, r"in-domain maximum %.1f $\times 10^{-4}$" % (drawn_max * 1e4),
            fontsize=8, color=INK2)
    ax.plot([], [], "-", color=ORANGE, lw=1.2, label="the six E4 points")
    ax.plot([], [], "-", color=BLUE, lw=0.9, alpha=0.7, label="16 sampled campaign rows")
    ax.plot([], [], ":", color=INK2, lw=0.8, label=r"outside the $g_3$ domain")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"squeeze Deborah number De$_{sq}$ (boundary maximum)")
    ax.set_ylabel(r"|memory share| of film load")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    rep = json.loads((OUT / "E4_lambda_sweep.report.json").read_text())
    assert abs(drawn_max - rep["share_memory_D1_in_domain"]["max_abs"]) < 1e-15
    _save(fig, "FE4_memory_share_vs_Desq", {
        "caption": ("Squeeze-memory share of the film load across the registered "
                    "relaxation-time axis [1e-5, 1e-2] s, one line per base row, every other "
                    "input held.  Memory reduces load everywhere (share negative; magnitude "
                    "drawn).  Inside the registered validity domain the share stays below "
                    "about 1e-3; for most rows it is largest at the axis floor, and in every "
                    "row it falls monotonically past its maximum.  No mechanism is claimed "
                    "for the uniform sign (R-51.5)."),
        "artefacts_read": ["results/ch9/E4_lambda_sweep.parquet",
                           "results/ch9/E4_lambda_sweep.report.json"],
        "drawn_max": drawn_max, "emitted_macro": "BsNineEFourSweepMax",
        "emitted_value": rep["share_memory_D1_in_domain"]["max_abs"]})


if __name__ == "__main__":
    fm1()
    fe4()
    print("ok")
