"""R-72.6: Paper 3's figures/fig_stroke.pdf, from R72_E17_stroke.parquet / .report.json.  No solves.

Lambda = delta c / sigma* against omega t over one delivery stroke, (a) E15, (b) E5.  Isoviscous (light)
and Lee's Roelands VG46 at T_in (heavy); from concentric (solid) and from the v6 snapshot (dotted);
the snapshot film marked (open circle) where the concentric trajectory passes it; Lambda = 3 dashed.
The stroke ends at omega t = pi.  A trajectory stops at the delta grid floor (0.002) if it gets there
first.  The encoding is asserted from the figure's own artists before saving (v7 has no caption yet).
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
PANELS = (("E15", "(a) best geometry (E15)"), ("E5", "(b) mid-range relief angle (E5)"))
LAWS = (("isoviscous", "isoviscous"), ("roelands_LeeVG46", "VG46, Roelands"))


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-72.6 fig_stroke")
    import fig_R61_update as G
    import matplotlib.pyplot as plt
    rep = json.loads((OUT / "R72_E17_stroke.report.json").read_text(encoding="utf-8"))
    D = pd.read_parquet(OUT / "R72_E17_stroke.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True, sharey=True)
    style = {"isoviscous": (G.C_VERM, G.LIGHT), "roelands_LeeVG46": (G.C_BLUE, G.HEAVY)}
    for ax, (p, title) in zip(axes, PANELS):
        pr = rep["points"][p]
        lam_snap = pr["delta_snapshot_v6"] * pr["c"] / pr["sigma"]
        for law, lab in LAWS:
            col, lw = style[law]
            for start, ls in (("concentric", "-"), ("snapshot", ":")):
                s = D[(D.point == p) & (D.fluid_law == law) & (D.start == start)].sort_values("omega_t")
                s = s[np.isfinite(s.omega_t)]
                t, L = s.omega_t.to_numpy(), s.Lambda.to_numpy()
                if t[-1] > np.pi:                       # end the line at omega t = pi
                    Lpi = np.interp(np.pi, t, L)       # as the report's at_time (linear in delta)
                    k = t < np.pi
                    t, L = np.append(t[k], np.pi), np.append(L[k], Lpi)
                ax.plot(t, L, color=col, lw=lw, ls=ls, label=f"{lab}, {'concentric' if start == 'concentric' else 'snapshot'} start",
                        gid=f"{p}_{law}_{start}")
            tp = pr["laws"][law]["concentric"]["t_pass_snapshot"]
            if tp is not None:
                ax.plot([tp], [lam_snap], ls="none", marker="o", ms=5, mfc="none", mec=col, mew=1.0,
                        gid=f"{p}_{law}_snapmark")
        ax.axhline(3.0, color="k", lw=G.LIGHT, ls="--", gid=f"{p}_fullfilm")
        ax.axvline(np.pi, color=G.C_GREY, lw=G.LIGHT, ls="-", gid=f"{p}_pi")
        ax.set_yscale("log")
        ax.set_xlim(0, 3.3)
        ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
        ax.set_xlabel(r"$\omega t$")
        ax.set_title(title, loc="left", fontsize=8)
    axes[0].set_ylabel(r"$\Lambda = \delta c/\sigma^{*}$")
    axes[0].legend(frameon=False, fontsize=6, loc="lower left")
    # ---- encoding asserted from the artists
    by = {ln.get_gid(): ln for ax in axes for ln in ax.lines}
    for p, _ in PANELS:
        assert by[f"{p}_isoviscous_concentric"].get_linewidth() == G.LIGHT
        assert by[f"{p}_roelands_LeeVG46_concentric"].get_linewidth() == G.HEAVY
        assert by[f"{p}_isoviscous_snapshot"].get_linestyle() == ":"
        assert by[f"{p}_fullfilm"].get_linestyle() == "--" and by[f"{p}_fullfilm"].get_ydata()[0] == 3.0
        pr = rep["points"][p]
        for law, _ in LAWS:
            x = by[f"{p}_{law}_concentric"].get_xdata()
            assert x.max() <= np.pi + 1e-12
            lp = pr["laws"][law]["concentric"]["Lambda_pi"]
            if lp is not None:                         # the line ends at Lambda(pi) of the report
                y = by[f"{p}_{law}_concentric"].get_ydata()
                assert abs(y[-1] / lp - 1) < 1e-9, (p, law, y[-1], lp)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig_stroke.pdf")
    (FIGDIR / "fig_stroke.json").write_text(json.dumps({"figure": "fig_stroke.pdf", "provenance": prov,
                                                        "source": "results/ch9/R72_E17_stroke.parquet, R72_E17_stroke.report.json",
                                                        "fluid_law": "isoviscous and roelands_LeeVG46 (T_in of the point)",
                                                        "model": "rigid, isothermal, Newtonian at eta_s; F_req constant over the stroke",
                                                        "memory_sign": "none (Newtonian path)"}, indent=1),
                                            encoding="utf-8")
    print("fig_stroke written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
