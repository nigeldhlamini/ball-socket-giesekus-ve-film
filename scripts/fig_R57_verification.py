"""R-57.1 App. C verification figure: |load error| of the ball-socket solver against the
vendored quadrature (Newtonian, isothermal, pure squeeze) versus theta-cell count.

Small multiples, one panel per theta_min (20, 30, 40 deg); in each, one series per
delta in {0.3, 0.1, 0.03, 0.01}.  Production mesh (32 cells) marked; a second-order
reference line.  Nphi = 8 (the film is axisymmetric: 128x96 vs 128x8 differ by 4e-16).

Encoding (greyscale-safe: identity never rests on colour alone):
  delta  -> marker shape AND line style AND colour (reference categorical order)
  sign   -> filled marker = solver over-predicts, open marker = under-predicts
The generator asserts that the encoding it draws is the one the caption states.
Colours are the dataviz reference palette's first four categorical slots (pre-validated
there; the JS validator could not be run on this machine -- no node).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
DATA = OUT / "R57_verification_figure_data.json"
FIG = ROOT / "figures" / "ch9" / "paper3_appC_verification.pdf"
NT = (32, 64, 128, 256, 512, 1024)
THETAS = (20.0, 30.0, 40.0)
DELTAS = (0.3, 0.1, 0.03, 0.01)
ENC = {0.3: dict(marker="o", ls="-", color="#2a78d6"),
       0.1: dict(marker="s", ls="--", color="#eb6834"),
       0.03: dict(marker="^", ls="-.", color="#1baf7a"),
       0.01: dict(marker="D", ls=":", color="#eda100")}
CAPTION = ("Load error of the solver against the analytical squeeze film (Newtonian, "
           "isothermal), against the number of cells across the land, for "
           "theta_min = 20, 30 and 40 deg. Series: delta = 0.3 (circles, solid), 0.1 "
           "(squares, dashed), 0.03 (triangles, dash-dot), 0.01 (diamonds, dotted). "
           "Filled markers: solver over-predicts; open: under-predicts. Vertical line: "
           "production mesh (32 cells). Dotted grey line: second order.")


def compute():
    import warnings
    warnings.filterwarnings("ignore")
    import p3_rim_asymptote as RA
    from run_R56_R1 import solve, C, ETA, R_B, OMEGA
    data = {}
    for tm in THETAS:
        for d in DELTAS:
            fq = RA.f_recess(d, tm)
            errs = []
            for n in NT:
                F, eps = solve(tm, d, (n, 8))
                errs.append(RA.solver_f(F, c=C, eta=ETA, R=R_B, omega=OMEGA, eps=eps) / fq - 1)
            data[f"{tm:.0f}|{d}"] = errs
            print(tm, d, " ".join(f"{e:+.2e}" for e in errs), flush=True)
    DATA.write_text(json.dumps({"Ntheta": NT, "errors": data}, indent=2))
    return data


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # the caption's stated encoding, parsed back and checked against ENC
    for d, e in ENC.items():
        name = {"o": "circles", "s": "squares", "^": "triangles", "D": "diamonds"}[e["marker"]]
        style = {"-": "solid", "--": "dashed", "-.": "dash-dot", ":": "dotted"}[e["ls"]]
        assert f"{d} ({name}, {style})" in CAPTION, (d, name, style)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.6), sharey=True)
    n = np.array(NT, float)
    for k, tm in enumerate(THETAS):
        a = ax[k]
        for d in DELTAS:
            e = np.array(data[f"{tm:.0f}|{d}"])
            st = ENC[d]
            a.plot(n, np.abs(e), ls=st["ls"], color=st["color"], lw=1.4, marker=st["marker"],
                   ms=5, mfc="none", mec="none", label=fr"$\delta={d}$")
            for xi, ei in zip(n, e):
                a.plot(xi, abs(ei), st["marker"], ms=5, color=st["color"],
                       mfc=st["color"] if ei > 0 else "white", mec=st["color"], mew=1.0)
        # second-order reference, offset a decade below the data so it never hides a series
        a.loglog(n, 3e-4 * (n / 32) ** -2, color="0.25", lw=0.9, ls=(0, (1, 1)),
                 label="second order")
        a.axvline(32, color="0.55", lw=0.8, label="production mesh")
        a.set_xscale("log", base=2)
        a.set_xticks(n)
        a.set_xticklabels([str(int(v)) for v in n])
        a.set_title(fr"$\theta_{{\min}} = {tm:.0f}^\circ$", fontsize=8)
        a.set_xlabel(r"cells across the land, $N_\theta$")
        a.grid(True, which="major", color="0.9", lw=0.5)
    ax[0].set_ylabel(r"$|F_{\mathrm{solver}}/F_{\mathrm{exact}} - 1|$")
    ax[0].set_ylim(1e-8, 1e-1)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=ENC[d]["color"], ls=ENC[d]["ls"], marker=ENC[d]["marker"], ms=5,
                lw=1.4, label=fr"$\delta={d}$") for d in DELTAS]
    h += [Line2D([], [], color="0.25", lw=0.9, ls=(0, (1, 1)), label="second order"),
          Line2D([], [], color="0.55", lw=0.8, label="production mesh"),
          Line2D([], [], color="0.3", ls="none", marker="o", ms=5, label="filled: over-predicts"),
          Line2D([], [], color="0.3", ls="none", marker="o", ms=5, mfc="white", label="open: under-predicts")]
    fig.tight_layout(rect=(0, 0, 0.83, 1))
    fig.legend(handles=h, fontsize=6.5, loc="center left", bbox_to_anchor=(0.835, 0.55),
               frameon=False)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    fig.savefig(FIG.with_suffix(".png"), dpi=200)
    FIG.with_suffix(".caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    print("wrote", FIG)


if __name__ == "__main__":
    data = (json.loads(DATA.read_text())["errors"] if DATA.exists() and "--replot" in sys.argv
            else compute())
    plot(data)
