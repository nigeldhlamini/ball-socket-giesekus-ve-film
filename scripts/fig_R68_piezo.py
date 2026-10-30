"""R-68.1: Paper 3's figures/fig_piezo.pdf, from R68_piezo_ratio.parquet / .report.json.  No solves.

(a) Lambda_pv/Lambda_iso against G = alpha p_ch at E15 (best geometry), E13 (smallest design
    clearance) and E5 (mid-range relief angle): markers, the exact Barus solution on the solver;
    lines, 1 + kappa G with kappa from the first-order functional; a horizontal tick per point at
    the full-film requirement 3/Lambda_iso; a vertical line at Lee's VG46 G* (100 C).
(b) Lambda against c0/c0*_iso at the E15 arm: isoviscous (light, open) and Roelands VG46 (heavy,
    filled); Lambda = 3 dashed.
Encoding asserted against v5's caption (fig:piezo) before saving.
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
FIGDIR = pathlib.Path("Z:/Papers/Paper 3/New/figures")
V5 = pathlib.Path("Z:/Papers/Paper 3/New/paper3_v5_draft.tex")
PTS = (("E15", "best geometry", "o"), ("E13", "smallest design clearance", "s"), ("E5", "mid-range relief angle", "^"))


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-68.1 fig_piezo")
    import fig_R61_update as G
    import matplotlib.pyplot as plt
    rep = json.loads((OUT / "R68_piezo_ratio.report.json").read_text(encoding="utf-8"))
    D = pd.read_parquet(OUT / "R68_piezo_ratio.parquet")
    cols = (G.C_BLUE, G.C_VERM, G.C_GREEN)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
    gg = np.linspace(0, 1.0, 101)
    for k, ((p, lab, mk), col) in enumerate(zip(PTS, cols)):
        r = rep["points"][p]
        sw = D[(D.point == p) & D.fluid_law.str.startswith("barus_G")].sort_values("G")
        a.plot(sw.G, sw.ratio, ls="none", marker=mk, ms=5, mfc=col, mec="k", mew=0.5, color=col,
               label=f"{lab} ({p})", gid=f"markers_{p}")
        a.plot(gg, 1 + r["kappa"] * gg, color=col, lw=G.HEAVY * 0.6, gid=f"law_{p}")
        req = 3.0 / r["Lam_iso"]
        # outside the right edge, staggered: E15's and E13's requirements nearly coincide (1.252, 1.255)
        a.plot([1.01 + 0.035 * k, 1.04 + 0.035 * k], [req, req], color=col, lw=G.HEAVY, solid_capstyle="butt",
               clip_on=False, gid=f"tick_{p}")
    a.axvline(rep["G_lee_roelands"], color=G.C_GREY, lw=G.LIGHT, ls=":", gid="lee")
    a.text(rep["G_lee_roelands"] + 0.01, 1.02, "VG46, 100 °C", fontsize=6.5, color=G.C_GREY, rotation=90, va="bottom")
    a.set_xlim(0, 1.0)
    a.set_xlabel(r"$G = \alpha p_{\mathrm{ch}}$")
    a.set_ylabel(r"$\Lambda_{\mathrm{pv}}/\Lambda$")
    a.set_title("(a)", loc="left")
    a.legend(frameon=False, fontsize=6.5, loc="upper left")
    c0i = rep["c0_axis"]["isoviscous"]["c0_star_grid_um"]
    for name, lw, mf, lab, gid in (("isoviscous", G.LIGHT, "none", "isoviscous", "iso"),
                                   ("roelands_LeeVG46", G.HEAVY, G.C_BLUE, "VG46, 100 °C", "pv")):
        ax_ = D[(D.point == "E15_axis") & (D.fluid_law == name) & (D.outcome == "converged")].sort_values("c0_um")
        b.plot(ax_.c0_um / c0i, ax_.Lam, color=G.C_BLUE, lw=lw, marker="o", ms=3, mfc=mf, mec=G.C_BLUE, label=lab, gid=gid)
    b.axhline(3.0, color="k", lw=G.LIGHT, ls="--", gid="fullfilm")
    b.set_xlabel(r"$c_0/c_0^{*}$ (isoviscous optimum)")
    b.set_ylabel(r"$\Lambda$")
    b.set_title("(b)", loc="left")
    b.legend(frameon=False, fontsize=6.5, loc="lower left")
    # ---- encoding asserted against v5's caption
    tex = V5.read_text(encoding="utf-8")
    cap = tex[tex.index("\\label{fig:piezo}") - 1200: tex.index("\\label{fig:piezo}")]
    cap = " ".join(cap[cap.index("\\caption{"):].split())
    for phrase in ("(a) Film-parameter lift", "against the pressure--viscosity number $G = \\alpha p_{\\mathrm{ch}}$",
                   "at three rated-duty points: the best geometry, the smallest design clearance, and the mid-range relief angle",
                   "Markers, the exact Barus solution on the solver", "lines, the linear law",
                   "ticks, the lift each point needs to reach a full film", "the vertical line marks a VG46 mineral oil",
                   "(b) Film parameter against clearance, relative to the isoviscous optimum",
                   "isoviscous (light)", "on the VG46 oil (heavy)", "dashed, full film"):
        assert phrase in cap, phrase
    by = {ln.get_gid(): ln for ax in (a, b) for ln in ax.lines}
    for p, _, _ in PTS:
        assert by[f"markers_{p}"].get_linestyle() == "None" and by[f"law_{p}"].get_linestyle() == "-"
        assert abs(by[f"tick_{p}"].get_ydata()[0] - 3.0 / rep["points"][p]["Lam_iso"]) < 1e-12
    assert abs(by["lee"].get_xdata()[0] - rep["G_lee_roelands"]) < 1e-12
    assert by["iso"].get_linewidth() == G.LIGHT and by["pv"].get_linewidth() == G.HEAVY
    assert by["fullfilm"].get_linestyle() == "--" and by["fullfilm"].get_ydata()[0] == 3.0
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig_piezo.pdf")
    (FIGDIR / "fig_piezo.json").write_text(json.dumps({"figure": "fig_piezo.pdf", "provenance": prov,
                                                       "source": "results/ch9/R68_piezo_ratio.parquet",
                                                       "fluid_law": "barus (a), roelands_LeeVG46 and isoviscous (b)",
                                                       "memory_sign": "none (Newtonian path)"}, indent=1),
                                           encoding="utf-8")
    print("fig_piezo written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
