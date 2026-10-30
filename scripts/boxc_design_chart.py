"""Phase 4 design chart -- the FREE version (code prompt sec 5.1), laid out per
R-41.4 / R-42.3.

c0 and theta_min are both input axes of the audit, so a first chart is a
re-slice of the campaign in (c0, beta), beta = relief_fraction(theta_min) (the
config helper, not re-derived).  Each cell pools every audit row whose (c0,
beta) falls in it -- the other six axes are MARGINALISED, so a cell value is a
fraction of the sampled operating envelope, not a single operating point.  The
dense corner-power sweep (sec 5.2) is a costed re-run and is not done here.

PANEL (a): the SMOOTH-film chart (R-42.3: one model, one equilibrium, one verdict
per row), one sub-panel per fluid class, because g2's ceiling is per class.
    fill      share of rows DECIDED feasible: converged, g2 and g3 pass, and
              Lambda_bar above the constant band (feasible at both sigma* ends).
    contours  share of rows in the CONSTANT BAND (R-41.2, drawn with the
              averaged-side lower edge of R-42.5): g2 and g3 pass and
              Lambda_bar in [2.3306, 3.7463], so the registered roughness band
              cannot decide g1.  Uncertainty from a CONSTANT.
    hatch     cells containing MASKED rows (R-39.3): no film model validated.
    feasibility band (R-41.3): EMPTY in the tested sample, so it draws no ink;
              its power is stated in the caption.
    No binding attribution (R-42.4).
PANEL (b): the EXISTENCE result on the region composite (R-41.1 / R-42.7):
    fill      share of valid rows with no equilibrium -- averaged film outside
              the mask, smooth inside.
    contours  share of rows whose existence verdict the film model changes
              (E7-b), the existence band.  Uncertainty from the MODEL.
    hatch     masked rows, as in (a).

Mixed arm (box E, E7's population), 196,608 rows.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "ch9"
FIG = ROOT / "figures" / "ch9"
NAME = "F13_design_chart_free"
MIXED = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
COUNT = OUT / "boxF_marginal_band_count.parquet"
FULL = OUT / "boxF_band_full_resolve.parquet"
RULE = OUT / "boxF_R41_band_rule.json"
SIGMA_TEST = OUT / "boxF_R41_sigma_band_test.report.json"
LB_EDGE_AVERAGED = 2.3306          # R-42.5, conservative end of the measured bracket
C0_EDGES_UM = np.arange(2.0, 30.0 + 1e-9, 2.0)
THETA_EDGES = np.linspace(20.0, 40.0, 11)
MASK_LEVELS = [0.05, 0.15, 1.0]      # display bands for a row-level mask pooled into cells


def build() -> tuple[pd.DataFrame, dict]:
    from ball_socket.campaign_config_v4 import relief_fraction, ConstraintSet
    cs = ConstraintSet()
    rule = json.loads(RULE.read_text(encoding="utf-8"))
    sb = rule["constant_band"]["sigma_bar_m"]
    w_hi = rule["constant_band"]["Lambda_bar_window"][1]
    M = pd.read_parquet(MIXED, columns=["sample_id", "model_id", "outcome", "c0",
                                        "theta_min_deg", "h_min_bnd", "T_max_bnd",
                                        "De_sq_bnd"])
    M["model_id"] = M.model_id.astype(int)
    cnt = pd.read_parquet(COUNT)
    cnt = cnt[cnt.arm_of_campaign == "mixed"][["sample_id", "model_id", "masked"]]
    M = M.merge(cnt, on=["sample_id", "model_id"], how="left")
    M["masked"] = M.masked.astype("boolean").fillna(False).astype(bool)
    # composite existence: averaged outcome for re-solved band rows (treatments
    # agree on all of them, F-A4), smooth otherwise
    F = pd.read_parquet(FULL, columns=["arm_of_campaign", "sample_id", "model_id",
                                       "treatment", "outcome"])
    F = F[(F.arm_of_campaign == "mixed") & (F.treatment == "extrapolate")]
    M = M.merge(F[["sample_id", "model_id", "outcome"]].rename(
        columns={"outcome": "outcome_avg"}), on=["sample_id", "model_id"], how="left")
    M["outcome_comp"] = M.outcome_avg.fillna(M.outcome)
    invalid = ("clearance_nonconverged", "error", "nonconverged")   # nonconverged = picard (R-37.4)
    M["valid_comp"] = ~M.outcome_comp.isin(invalid)
    M["noeq_comp"] = M.outcome_comp == "contact_limit"
    # an existence flip is a change between VALID verdicts; the 10 rows that went
    # clearance_nonconverged under the averaged film are invalid, not flips
    M["exist_flip"] = (M.outcome_avg.notna() & (M.outcome_avg != M.outcome)
                       & M.valid_comp)
    conv = M.outcome == "converged"
    Lb = M.h_min_bnd / sb
    g3 = M.De_sq_bnd <= cs.De_sq_max
    for c, lim in cs.T_max_C_by_class.items():
        ok = conv & g3 & (M.T_max_bnd <= lim)
        M[f"feas_{c}"] = ok & (Lb > w_hi)
        M[f"const_{c}"] = ok & Lb.between(LB_EDGE_AVERAGED, w_hi)
    M["c0_um"] = M.c0 * 1e6
    M["ic"] = np.digitize(M.c0_um, C0_EDGES_UM[1:-1])
    M["it"] = np.digitize(M.theta_min_deg, THETA_EDGES[1:-1])
    g = M.groupby(["ic", "it"])
    cells = pd.DataFrame({
        "n": g.size(), "n_valid_comp": g.valid_comp.sum(),
        "feas_mineral": g.feas_mineral.mean(), "feas_water_glycol": g.feas_water_glycol.mean(),
        "const_mineral": g.const_mineral.mean(), "const_water_glycol": g.const_water_glycol.mean(),
        "masked": g.masked.mean(), "exist_flip": g.exist_flip.mean(),
        "noeq_comp": g.noeq_comp.sum() / g.valid_comp.sum()}).reset_index()
    beta_edges = np.array([relief_fraction(t) for t in THETA_EDGES])
    meta = {"sigma_bar_m": sb, "Lambda_bar_band_drawn": [LB_EDGE_AVERAGED, w_hi],
            "beta_edges": beta_edges.tolist(), "c0_edges_um": C0_EDGES_UM.tolist(),
            "rows": int(len(M)),
            "rows_decided_feasible": {c: int(M[f"feas_{c}"].sum()) for c in cs.T_max_C_by_class},
            "rows_constant_band": {c: int(M[f"const_{c}"].sum()) for c in cs.T_max_C_by_class},
            "rows_masked": int(M.masked.sum()),
            "rows_existence_flip": int(M.exist_flip.sum()),
            "noeq_composite": [int(M.noeq_comp.sum()), int(M.valid_comp.sum())]}
    return cells, meta


def draw(cells: pd.DataFrame, meta: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "hatch.linewidth": 0.5, "hatch.color": "#52514e"})
    xe = np.asarray(meta["c0_edges_um"])
    ye = np.asarray(meta["beta_edges"])
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])

    def grid(col):
        Z = np.full((len(yc), len(xc)), np.nan)
        for r in cells.itertuples():
            Z[int(r.it), int(r.ic)] = getattr(r, col)
        return Z

    fig = plt.figure(figsize=(7.6, 2.9))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.05, 0.28, 1, 0.05], wspace=0.12)
    a1 = fig.add_subplot(gs[0, 0])
    axes = [a1, fig.add_subplot(gs[0, 1], sharey=a1), fig.add_subplot(gs[0, 4], sharey=a1)]
    cax_a, cax_b = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 5])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    specs = [("feas_mineral", "const_mineral", "(a1) smooth film, mineral"),
             ("feas_water_glycol", "const_water_glycol", "(a2) smooth film, water-glycol"),
             ("noeq_comp", "exist_flip", "(b) existence, region composite")]
    Zm = grid("masked")
    A_MAX = 0.3                     # (a) fill ceiling: max decided-feasible share is 0.28
    mesh = {}
    for ax, (fill, cont, title) in zip(axes, specs):
        is_b = fill == "noeq_comp"
        mesh[is_b] = ax.pcolormesh(xe, ye, grid(fill), cmap="Greys" if is_b else "Blues",
                                   vmin=0, vmax=1.0 if is_b else A_MAX, shading="flat",
                                   edgecolors="#fcfcfb", linewidth=0.4)
        # masked share, hatch density by band (display bands, not a verdict)
        ax.contourf(xc, yc, Zm, levels=MASK_LEVELS, colors="none",
                    hatches=["...", "////"])
        ax.contour(xc, yc, Zm, levels=MASK_LEVELS[:1], colors="#52514e",
                   linewidths=0.4)
        levels = [0.02, 0.05] if is_b else [0.05, 0.15]
        cs_ = ax.contour(xc, yc, grid(cont), levels=levels,
                         colors="#b3261e" if is_b else "#0b0b0b",
                         linewidths=0.8, linestyles=["dashed", "solid"])
        ax.clabel(cs_, fmt=lambda v: f"{100 * v:.0f}%", fontsize=6)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel(r"cold clearance $c_0$ ($\mu$m)")
    axes[0].set_ylabel(r"relief fraction $\beta_r$")   # R-43.3 / R-55.8: not the swash beta
    cb = fig.colorbar(mesh[False], cax=cax_a, extend="max")
    cb.set_label("(a) share decided feasible", fontsize=7)
    cb2 = fig.colorbar(mesh[True], cax=cax_b)
    cb2.set_label("(b) share with no equilibrium", fontsize=7)
    handles = [Line2D([], [], color="#0b0b0b", lw=0.8, label="(a) constant-band share 5 % / 15 %"),
               Line2D([], [], color="#b3261e", lw=0.8, label="(b) existence-band share 2 % / 5 %"),
               Patch(facecolor="none", hatch="...", edgecolor="#52514e",
                     label="masked share 5-15 %"),
               Patch(facecolor="none", hatch="////", edgecolor="#52514e",
                     label="masked share > 15 % (no model validated)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.28, top=0.9, left=0.07, right=0.93)
    return fig


def main() -> int:
    from ball_socket.immutable import guarded_path
    FIG.mkdir(parents=True, exist_ok=True)
    pdf, png = FIG / f"{NAME}.pdf", FIG / f"{NAME}.png"
    man, data = FIG / f"{NAME}_manifest.json", OUT / f"{NAME}_cells.parquet"
    for p in (pdf, png, man, data):
        guarded_path(str(p), what="design chart (free)")
    cells, meta = build()
    cells.to_parquet(data, index=False)
    fig = draw(cells, meta)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    st = json.loads(SIGMA_TEST.read_text())
    meta["feasibility_band"] = {"status": "EMPTY in the tested sample (R-41.3 null)",
                                "power": st["power"], "rule_sha256": st["rule_sha256"]}
    manifest = {
        "figure": NAME, "path": str(pdf.relative_to(ROOT)),
        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()[:16],
        "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest()[:16],
        "status": "FREE VERSION (code prompt sec 5.1) -- re-slice of the audit; the "
                  "dense corner-power sweep (sec 5.2) is not run",
        "artefacts_read": [str(p.relative_to(ROOT)) for p in (MIXED, COUNT, FULL, RULE, SIGMA_TEST)],
        "caption_draft": (
            "Design chart, free version: the box-E mixed-lubrication audit re-sliced "
            "onto cold clearance and pocket relief fraction, pooling the other six "
            "input axes, so each cell is a share of the sampled operating envelope. "
            "(a) Smooth film, one sub-panel per fluid class: fill is the share of "
            "rows decided feasible at both ends of the registered roughness band; "
            "contours are the share whose g1 verdict the roughness band itself "
            "cannot decide (constant band, uncertainty from a registered constant). "
            "The film-model feasibility band is empty over the tested sample and "
            "draws no ink. (b) Existence on the region composite (averaged film "
            "outside the validity mask, smooth inside): fill is the no-equilibrium "
            "share; contours are the share whose existence verdict the film model "
            "changes (uncertainty from the model). Hatching marks cells containing "
            "masked rows, where no film model is validated. Binding-constraint "
            "attribution is deliberately not shown."),
        "notes": meta,
    }
    man.write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(meta, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
