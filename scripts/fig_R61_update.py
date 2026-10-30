"""R-61 Part A: the three figures for the October 2026 research update.  No solves.

    python scripts/fig_R61_update.py [--out DIR] [--png DIR]

Writes fig_u1_saturation.pdf, fig_u2_clearance.pdf, fig_u3_operating_space.pdf into
--out (default Z:/Updates/October 2026), each with a .json beside it carrying its
numbers, source artefacts (sha256), the commit and the memory_sign tag.  Refuses to run
from a dirty tree (R-59.2.4).  Every figure's drawn encoding is asserted before saving
(R-61 standing rule): colour plus a second cue, primary curves heavy and filled, bounds
light and open.

Sources (all existing artefacts):
  A1  scripts/p3_rim_asymptote.py (vendored quadrature); analysis_E7_load_decomposition_32x24
      (the 509 rows, R-56.3.2 filter); R56_R1_verification.parquet (Newtonian solver);
      R57_R3.report.json (pure-squeeze ratios).
  A2  R57_E15_prediction.json (Newtonian prediction); E13_extremum.parquet (production);
      R58_richardson.report.json (2 um Richardson value).
  A3  audit_boxD_purefilm_..._corrected.parquet (pure film) and
      audit_boxE_mixed_..._corrected.parquet (24 mixed arms), paired by sample_id;
      E5_VG32_g1_binding_corner.parquet (rated corner).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
RES = ROOT / "results" / "ch9"
OUT_DEFAULT = pathlib.Path("Z:/Updates/October 2026")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Okabe-Ito: colour-vision-safe; every colour is paired with a second cue below.
C_BLUE, C_ORANGE, C_GREEN, C_VERM, C_SKY, C_PURPLE, C_GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#7f7f7f")
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
HEAVY, LIGHT = 2.0, 0.9
BOOT, SEED = 2000, 20261005
THETA_BANDS = [(20.0, 26.7), (26.7, 33.3), (33.3, 40.0001)]
BAND_LABEL = [r"$\theta_{\min}\in[20,26.7)^\circ$", r"$[26.7,33.3)^\circ$", r"$[33.3,40]^\circ$"]
A_P, R_B = 2.0e-4, 6.5e-3     # ball_socket/representative.py:19-20


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def boot_median(x, rng):
    x = np.asarray(x)
    m = np.median(rng.choice(x, size=(BOOT, len(x)), replace=True), axis=1)
    return float(np.median(x)), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def assert_encoding(ax, *, heavy=(), light=(), filled_markers=(), open_markers=()):
    """Drawn encoding against the caption.  Lines are found by label."""
    lines = {ln.get_label(): ln for ln in ax.get_lines()}
    for lab in heavy:
        assert lines[lab].get_linewidth() >= HEAVY - 1e-9, f"{lab}: not heavy"
    for lab in light:
        assert lines[lab].get_linewidth() <= LIGHT + 1e-9, f"{lab}: not light"
    for lab in filled_markers:
        ln = lines[lab]
        assert ln.get_marker() not in ("", "None", None) and ln.get_markerfacecolor() not in ("none", "None"), \
            f"{lab}: not a filled marker"
    for lab in open_markers:
        ln = lines[lab]
        assert ln.get_marker() not in ("", "None", None) and ln.get_markerfacecolor() in ("none", "None"), \
            f"{lab}: not an open marker"
    # second cue: no two labelled data lines share BOTH colour and (linestyle, marker)
    seen = {}
    for lab, ln in lines.items():
        if lab.startswith("_"):
            continue
        key = (matplotlib.colors.to_hex(ln.get_color()), ln.get_linestyle(), ln.get_marker())
        assert key not in seen, f"{lab} and {seen.get(key)} are indistinguishable"
        seen[key] = lab


# =============================================================================== A1
def fig_u1(out: pathlib.Path, prov: dict, dmin: float = 1e-3, figsize=(7.0, 3.9),
           name: str = "fig_u1_saturation") -> dict:
    """The update figure (defaults) or, with dmin = 0.01, Paper 3's Figure F-SAT (R-63.6):
    the same generator with nothing below the production floor."""
    import warnings
    warnings.filterwarnings("ignore")
    import p3_rim_asymptote as RA
    rng = np.random.default_rng(SEED)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    th_col = {20: (C_BLUE, "-", "o"), 30: (C_ORANGE, "--", "s"), 40: (C_GREEN, "-.", "^")}
    d = np.logspace(np.log10(dmin), np.log10(0.5), 160)
    quad = {}
    for th, (col, ls, _) in th_col.items():
        e = [RA.local_exp(lambda z: RA.f_recess(z, th), x / 1.03, x * 1.03) for x in d]
        quad[th] = e
        ax.plot(d, e, color=col, ls=ls, lw=HEAVY, label=fr"closed form, $\theta_{{\min}}={th}^\circ$")
    ec = [RA.local_exp(RA.f_closed, x / 1.03, x * 1.03) for x in d]
    ax.plot(d, ec, color="k", ls=":", lw=LIGHT + 0.3, label="no recess (closed pole)")
    # solver: 509 rows, early interval, per theta_min band
    E7 = RES / "analysis_E7_load_decomposition_32x24.parquet"
    r = pd.read_parquet(E7)
    r = r[np.abs(r.F_full / r.F_film_stored - 1) <= 1e-9].copy()
    assert len(r) == 509, len(r)
    r["early"] = np.log(r["F_sq_d0.03"] / r["F_sq_d0.1"]) / np.log(0.03 / 0.1)
    xm = np.sqrt(0.1 * 0.03)
    bands = []
    for i, ((lo, hi), (th, (col, _, mk))) in enumerate(zip(THETA_BANDS, th_col.items())):
        s = r[(r.theta_min_deg >= lo) & (r.theta_min_deg < hi)]
        med, ci = boot_median(s.early, rng)
        tmed = float(s.theta_min_deg.median())
        qref = RA.local_exp(lambda z: RA.f_recess(z, tmed), 0.03, 0.1)
        bands.append({"band_deg": [lo, min(hi, 40.0)], "n": int(len(s)), "theta_min_median_deg": tmed,
                      "solver_median_exponent": med, "bootstrap95": ci,
                      "quadrature_exponent_at_band_median_theta": float(qref),
                      "difference": float(med - qref)})
        x = xm * (1 + 0.09 * (i - 1))
        ax.errorbar([x], [med], yerr=[[med - ci[0]], [ci[1] - med]], color=col, marker=mk, ms=6,
                    mfc=col, mec="k", mew=0.6, lw=1.0, capsize=2, ls="none",
                    label=f"solver median, band {i + 1} (n={len(s)})")
    # R1: Newtonian solver at 32x24, intervals above the floor
    R1 = pd.read_parquet(RES / "R56_R1_verification.parquet")
    r1 = R1[(R1.Ntheta == 32)]
    r1pts = []
    for th, (col, _, mk) in th_col.items():
        q = r1[r1.theta_min_deg == th].set_index("delta").f_solver
        xs, ys = [], []
        for a, b in ((0.3, 0.1), (0.1, 0.03), (0.03, 0.01)):
            xs.append(np.sqrt(a * b)); ys.append(float(np.log(q[b] / q[a]) / np.log(b / a)))
        r1pts.append({"theta_min_deg": th, "midpoints": xs, "exponents": ys})
        ax.plot(xs, ys, ls="none", marker=mk, ms=5.5, mfc="none", mec=col, mew=1.1,
                label=fr"Newtonian solver, ${th}^\circ$")
    if dmin < 0.01:
        ax.axvline(0.01, color=C_GREY, lw=LIGHT, ls=(0, (1, 2)), label="_floor")
        ax.text(0.0092, -0.06, r"production floor $\delta=0.01$", fontsize=6.5, color=C_GREY, va="top",
                ha="right", rotation=90)
    ymin = min(min(min(v) for v in quad.values()), min(ec), min(min(b["exponents"]) for b in r1pts))
    from matplotlib.ticker import NullFormatter
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlim(dmin, 0.5); ax.set_ylim(np.floor(ymin * 10) / 10 - 0.05, 0.02)
    ax.set_xlabel(r"rim closure $\delta = h_{\min}/c$")
    ax.set_ylabel(r"local exponent $\mathrm{d}\ln F/\mathrm{d}\ln\delta$")
    ax.set_title("(a)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6.0,
              handlelength=2.6, columnspacing=1.0)
    assert_encoding(ax, heavy=[fr"closed form, $\theta_{{\min}}={t}^\circ$" for t in th_col],
                    open_markers=[fr"Newtonian solver, ${t}^\circ$" for t in th_col])
    assert min(min(b["midpoints"]) for b in r1pts) >= 0.01 and xm >= 0.01   # nothing below the floor
    # (b) theta ratio over projected-area ratio
    ar = RA.area_proj(40) / RA.area_proj(20)
    db = np.logspace(np.log10(dmin), np.log10(0.95), 160)
    rq = [RA.f_recess(x, 40) / RA.f_recess(x, 20) / ar for x in db]
    bx.plot(db, rq, color=C_BLUE, lw=HEAVY, ls="-", label="closed form")
    R3 = json.loads((RES / "R57_R3.report.json").read_text())["a_theta_min_ratio_pure_squeeze"]["rows"]
    bx.plot([q["delta"] for q in R3], [q["solver_ratio_32"] / ar for q in R3], ls="none", marker="o", ms=6,
            color=C_VERM, mec="k", mew=0.6, label=r"solver, pure squeeze ($32\times24$)")
    rim = (2 * np.cos(np.deg2rad(40)) ** 5 / np.sin(np.deg2rad(40)) ** 2) / \
          (2 * np.cos(np.deg2rad(20)) ** 5 / np.sin(np.deg2rad(20)) ** 2)
    bx.axhline(rim / ar, color="k", ls=":", lw=LIGHT + 0.3, label="rim-coefficient ratio / area ratio")
    bx.set_xscale("log"); bx.xaxis.set_minor_formatter(NullFormatter()); bx.set_xlim(dmin, 1); bx.set_ylim(0, 0.45)
    bx.set_xlabel(r"$\delta = 1-\psi$")
    bx.set_ylabel(r"$[F(40^\circ)/F(20^\circ)]\,/\,[A(40^\circ)/A(20^\circ)]$")
    bx.set_title("(b)", loc="left")
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False, fontsize=6.5)
    assert_encoding(bx, heavy=["closed form"], filled_markers=[r"solver, pure squeeze ($32\times24$)"])
    assert min(d) >= dmin - 1e-15 and min(db) >= dmin - 1e-15
    fig.savefig(out / f"{name}.pdf")
    rep = {"figure": f"{name}.pdf", "provenance": prov, "delta_min": dmin,
           "memory_sign": {"solver_markers": "modulus + D-6 bound (analysis_E7 rows predate D-6)",
                           "closed_form_and_R1_R3": "none (Newtonian, no memory)"},
           "sources": {p.name: sha(p) for p in [E7, RES / "R56_R1_verification.parquet", RES / "R57_R3.report.json",
                                                ROOT / "scripts" / "p3_rim_asymptote.py"]},
           "a_solver_bands_early_interval_0.1_to_0.03": bands, "a_R1_newtonian_32x24": r1pts,
           "b": {"projected_area_ratio_40_over_20": ar, "rim_coefficient_ratio": rim,
                 "dotted_line_value_on_this_axis": rim / ar,
                 "closed_form_at_delta": {str(x): float(RA.f_recess(x, 40) / RA.f_recess(x, 20) / ar)
                                          for x in (1e-3, 0.01, 0.1, 0.5, 0.9)},
                 "R3_solver_over_area": {str(q["psi"]): q["solver_ratio_32"] / ar for q in R3},
                 "note": ("the RAW ratio F(40)/F(20) tends to the rim-coefficient ratio "
                          f"{rim:.3f}; divided by the area ratio {ar:.3f} its limit on this axis is "
                          f"{rim / ar:.3f}, which is where the dotted line is drawn")}}
    (out / f"{name}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return fig, rep


# =============================================================================== A2
def e15_curve(c0s_um):
    """The E15 prediction, evaluated exactly as scripts/run_R57_E15_prediction.py:38-51 does,
    at arbitrary clearances (that script samples every 0.5 um above 2 um)."""
    import warnings
    warnings.filterwarnings("ignore")
    import p3_rim_asymptote as RA
    import ball_socket.campaign as C
    from ball_socket import representative as rep
    from ball_socket.geometry import delta_alpha_th
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M
    from src.thermal import ThermalProperties
    spec = json.loads((RES / "E13_spec_v2.json").read_text(encoding="utf-8"))
    eta, om, p = spec["fluid"]["eta_0_Pa_s"], spec["corner"]["omega_rad_s"], spec["corner"]["p_ch_Pa"]
    R, tm = rep.R_B_M, 20.0
    W = p * rep.A_P_M2 - p * np.pi * R ** 2 * np.sin(np.deg2rad(tm)) ** 2
    dT = eta * (om * R * np.sin(np.deg2rad(73.0))) ** 2 / (12 * ThermalProperties().k_f)
    dc = delta_alpha_th() * R * dT
    rows = []
    for c0 in c0s_um:
        c = c0 * 1e-6 + dc
        d, ok = RA.delta_for_load(W, tm, eta=eta, R=R, c=c, omega=om, p_pocket=p, p_case=C.P_CASE_PA)
        rows.append(dict(c0_um=float(c0), delta=d, bracketed=ok, Lambda_lo=d * c / SIGMA_STAR_BAND_M[0]))
    return pd.DataFrame(rows)


def fig_u2(out: pathlib.Path, prov: dict, name: str = "fig_u2_clearance", figsize=(4.9, 3.0)) -> dict:
    P = json.loads((RES / "R57_E15_prediction.json").read_text())
    import ast
    curve = P["curve"] if isinstance(P["curve"], list) else ast.literal_eval(P["curve"])
    stored = pd.DataFrame(curve)
    stored = stored[(stored.c0_um >= 1.0) & (stored.c0_um <= 6.0)]
    chk = e15_curve(stored.c0_um.values)
    repro = float(np.max(np.abs(chk.Lambda_lo.values / stored.Lambda_lo.values - 1)))
    assert repro < 1e-12, repro                     # same computation as the registered prediction
    cv = e15_curve(np.round(np.arange(1.0, 6.0001, 0.02), 4))
    cv = cv[cv.bracketed]
    E13 = pd.read_parquet(RES / "E13_extremum.parquet")
    rep13 = json.loads((RES / "E13_extremum.report.json").read_text())
    arm = rep13["where"]["arm"]
    e = E13[(E13.theta_min_deg == 20) & (E13.arm == arm) & (E13.c0_um <= 6.0)].sort_values("c0_um")
    assert np.isclose(e.sigma_star.iloc[0], rep13["where"]["sigma_star"])
    rr = json.loads((RES / "R58_richardson.report.json").read_text())["rows"]["E13"]
    cstar, lstar = P["predicted"]["c0_star_um"], P["predicted"]["Lambda_star_lo"]
    i = int(cv.Lambda_lo.values.argmax())
    xx, yy = cv.c0_um.values[i - 1:i + 2], cv.Lambda_lo.values[i - 1:i + 2]
    qa, qb, qc = np.polyfit(xx, yy, 2)
    cstar_f, lstar_f = float(-qb / (2 * qa)), float(qc - qb ** 2 / (4 * qa))
    plat = cv[cv.Lambda_lo >= 0.99 * lstar_f]
    e13_all = e.copy()
    e = e[e.c0_um != 2.0]                           # the 2 um point is drawn at its Richardson value
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.axvspan(plat.c0_um.min(), plat.c0_um.max(), color=C_SKY, alpha=0.25, lw=0, label="_plateau")
    ax.plot(cv.c0_um, cv.Lambda_lo, color=C_BLUE, lw=HEAVY, ls="-", label="Newtonian prediction (closed form)")
    ax.plot(e.c0_um, e.Lambda_g1, ls="none", marker="o", ms=6.5, color=C_VERM, mec="k", mew=0.6,
            label=r"solver, production $32\times24$")
    ax.plot([2.0], [rr["Lambda_R"]], ls="none", marker="*", ms=11, color=C_PURPLE, mec="k", mew=0.6,
            label=r"solver at $2\,\mu$m, mesh-extrapolated")
    ax.axhline(3.0, color="k", lw=LIGHT, ls="-", label="_full film")
    ax.text(5.95, 3.03, r"full film, $\Lambda=3$", ha="right", va="bottom", fontsize=7)
    ax.text((plat.c0_um.min() + plat.c0_um.max()) / 2, 2.62, "within 1 %\nof peak", ha="center",
            fontsize=6.5, color="#1f4e79")
    ax.annotate("solver confirmation below\n" + r"$2\,\mu$m registered (E15)", xy=(1.45, 2.25),
                xytext=(1.12, 1.25), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7, color="k"))
    ax.set_xlim(1, 6); ax.set_ylim(1.0, 3.3)
    ax.set_xlabel(r"radial clearance $c_0$ ($\mu$m)")
    ax.set_ylabel(r"$\Lambda = h_{\min}/\sigma^{*}$")
    ax.legend(loc="lower right", frameon=False)
    assert_encoding(ax, heavy=["Newtonian prediction (closed form)"],
                    filled_markers=[r"solver, production $32\times24$", r"solver at $2\,\mu$m, mesh-extrapolated"])
    fig.savefig(out / f"{name}.pdf")
    pred_at = {c: float(np.interp(c, cv.c0_um, cv.Lambda_lo)) for c in e13_all.c0_um}
    rep = {"figure": f"{name}.pdf", "provenance": prov,
           "memory_sign": {"prediction": "none (Newtonian)",
                           "solver_production": "modulus + D-6 bound (E13 predates D-6)",
                           "solver_2um_Richardson": "signed (R-58.2)"},
           "sources": {p.name: sha(p) for p in [RES / "R57_E15_prediction.json", RES / "E13_extremum.parquet",
                                                RES / "R58_richardson.report.json"]},
           "conditions": {**P["condition"], "arm": arm, "sigma_star_m": float(e.sigma_star.iloc[0])},
           "prediction": {"registered_E15": {"c0_star_um": cstar, "Lambda_star": lstar,
                                             "note": "R57_E15_prediction.json; Lambda* is the max on its "
                                                     "0.05/0.5 um grid, c0* quadratically refined"},
                          "fine_grid_0.02um": {"c0_star_um": cstar_f, "Lambda_star": lstar_f},
                          "reproduces_stored_curve_max_rel": repro,
                          "plateau_1pct_um": [float(plat.c0_um.min()), float(plat.c0_um.max())],
                          "plateau_resolution_um": 0.02},
           "solver_over_prediction": {
               **{f"{c:g}um_production": float(l / pred_at[c]) for c, l in zip(e13_all.c0_um, e13_all.Lambda_g1)},
               "2um_richardson": float(rr["Lambda_R"] / pred_at[2.0])},
           "solver_values": {**{f"{c:g}um": float(l) for c, l in zip(e13_all.c0_um, e13_all.Lambda_g1)},
                             "2um_richardson": rr["Lambda_R"]}}
    (out / f"{name}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return fig, rep


# =============================================================================== A3
def duty_S(p, c0, eta, omega):
    """S = p A_p c0^2 / (3 pi eta omega R_b^4)  (R-47.6, R-54.6; eta = swept service viscosity)."""
    from ball_socket.solver import squeeze_capacity_scale
    return p * A_P / squeeze_capacity_scale(eta, omega, R_B, c0)


#: three rows checked by hand (scalar arithmetic, 5 Oct 2026; row = boxD index)
HAND_S = {0: 97.80, 1000: 210.67, 5000: 39.404}


def _logit_s50(logS, y, rng):
    from scipy.optimize import minimize

    def fit(x, t):
        def nll(b):
            z = b[0] + b[1] * x
            return float(np.sum(np.logaddexp(0, z) - t * z))
        b = minimize(nll, [0.0, -1.0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10,
                                                                       "maxiter": 4000}).x
        return -b[0] / b[1], b[1]
    s50, slope = fit(logS, y)
    bs = []
    n = len(y)
    for _ in range(400):
        i = rng.integers(0, n, n)
        bs.append(fit(logS[i], y[i])[0])
    return float(10 ** s50), [float(10 ** np.percentile(bs, 2.5)), float(10 ** np.percentile(bs, 97.5))], float(slope)


def fig_u3(out: pathlib.Path, prov: dict, eta_basis: str = "eta0", figsize=(7.0, 4.6),
           fname: str = "fig_u3_operating_space") -> dict:
    """eta_basis 'eta0' (R-61/R-62: S on the zero-shear viscosity) or 'eta_s' (R-65.1: S_s = S eta_0/eta_s,
    the high-shear plateau, defined at every sampled point).  Classes do not depend on S."""
    assert eta_basis in ("eta0", "eta_s"), eta_basis
    from ball_socket.campaign_config_v4 import ConstraintSet
    rng = np.random.default_rng(SEED)
    DP = RES / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet"
    EP = RES / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
    CP = RES / "E5_VG32_g1_binding_corner.parquet"
    D = pd.read_parquet(DP).sort_values("sample_id").reset_index(drop=True)
    E = pd.read_parquet(EP, columns=["sample_id", "model_id", "outcome", "F_contact", "F_app"] +
                        ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"])
    AX = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
    e0 = E[E.model_id == 0].sort_values("sample_id").reset_index(drop=True)
    paired = bool(np.array_equal(D.sample_id.values, e0.sample_id.values)
                  and all(np.array_equal(D[a].values, e0[a].values) for a in AX))
    assert paired, "arms are not paired; the per-point classification does not apply"
    D["S"] = duty_S(D.p_ch.values, D.c0.values, D.eta_0.values, D.omega.values)
    for i, v in HAND_S.items():
        assert abs(D.S[i] / v - 1) < 1e-3, (i, D.S[i], v)
    if eta_basis == "eta_s":
        D["S"] = D.S / D.eta_s_ratio
    cs = ConstraintSet()
    pf_eq = D.outcome == "converged"
    shares, rows = {}, {}
    for end, arms in (("lo", range(0, 12)), ("hi", range(12, 24))):
        m = E[E.model_id.isin(list(arms))]
        any_mixed = m.assign(eq=m.outcome == "converged").groupby("sample_id").eq.any().reindex(D.sample_id).values
        lam = D[f"lambda_ratio_{end}"]
        cls = np.where(pf_eq & (lam >= cs.lambda_ratio_min), 1,
                       np.where(pf_eq, 2, np.where(any_mixed, 3, 4)))
        D[f"cls_{end}"] = cls
        shares[end] = {k: float(np.mean(cls == i)) for i, k in
                       enumerate(["full_film", "film_supported_mixed", "contact_supported", "no_equilibrium"], 1)}
        shares[end]["counts"] = {int(i): int(np.sum(cls == i)) for i in range(1, 5)}
        # alternative split (JSON only): an equilibrium with F_con/F_app < 5 %
        mm = m[m.outcome == "converged"].assign(fc=lambda t: t.F_contact.fillna(0) / t.F_app)
        low_any = mm.assign(ok=mm.fc < 0.05).groupby("sample_id").ok.any().reindex(D.sample_id).fillna(False).values
        low_all = mm.assign(ok=mm.fc < 0.05).groupby("sample_id").ok.all().reindex(D.sample_id).fillna(False).values
        has = mm.groupby("sample_id").size().reindex(D.sample_id).fillna(0).values > 0
        rows[end] = {"film_supported_Fcon_lt_5pct_any_arm": float(np.mean(pf_eq | low_any)),
                     "film_supported_Fcon_lt_5pct_all_converged_arms": float(np.mean(pf_eq | (has & low_all)))}
    # rated corner (E5 VG32, 40 points at theta_min = 30)
    C5 = pd.read_parquet(CP)[["omega", "p_ch", "c0", "eta_0", "theta_min_deg", "eta_s_ratio"]].drop_duplicates()
    S5 = duty_S(C5.p_ch.values, C5.c0.values, C5.eta_0.values, C5.omega.values)
    if eta_basis == "eta_s":
        S5 = S5 / C5.eta_s_ratio.values
    th5 = float(C5.theta_min_deg.iloc[0])
    # --- figure
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0])
    cls_style = [(1, "full film", C_BLUE, ""), (2, "film-supported mixed", C_SKY, "////"),
                 (3, "contact-supported", C_ORANGE, "...."), (4, "no equilibrium", "#d9d9d9", "xxxx")]
    s50, axes_a, xr = [], [], []
    for b, ((t0, t1), lab) in enumerate(zip(THETA_BANDS, BAND_LABEL)):
        ax = fig.add_subplot(gs[0, b])
        axes_a.append(ax)
        sub = D[(D.theta_min_deg >= t0) & (D.theta_min_deg < t1)].sort_values("S")
        nb = max(4, min(14, len(sub) // 30))
        edges = np.quantile(np.log10(sub.S), np.linspace(0, 1, nb + 1))
        idx = np.clip(np.searchsorted(edges, np.log10(sub.S), side="right") - 1, 0, nb - 1)
        counts = np.bincount(idx, minlength=nb)
        assert counts.min() >= 30, counts
        xc = np.array([np.median(np.log10(sub.S)[idx == k]) for k in range(nb)])
        xr += [xc.min(), xc.max()]
        fr = np.array([[np.mean(sub.cls_lo.values[idx == k] == c) for k in range(nb)] for c, *_ in cls_style])
        ys = ax.stackplot(xc, fr, colors=[c for *_, c, _ in cls_style], edgecolor="k", linewidth=0.4)
        for poly, (_, _, _, h) in zip(ys, cls_style):
            poly.set_hatch(h)
        for poly, (_, name, _, _) in zip(ys, cls_style):
            poly.set_label(name)
        if t0 <= th5 < t1:
            ax.axvspan(np.log10(S5.min()), np.log10(S5.max()), facecolor="none", edgecolor="k", lw=1.0, ls="--",
                       label="rated corner")
        ax.set_ylim(0, 1)
        ax.set_title(("(a) " if b == 0 else "") + lab, loc="left", fontsize=8)
        ax.set_xlabel(r"$\log_{10} S$" if eta_basis == "eta0" else r"$\log_{10} S$ (on $\eta_s$)")
        if b == 0:
            ax.set_ylabel(r"fraction of points ($\sigma^{*}$ lo)")
        else:
            ax.set_yticklabels([])
        # S50s
        logS = np.log10(sub.S.values)
        rec = {"band_deg": [t0, min(t1, 40.0)], "n": int(len(sub)), "bins": int(nb), "min_bin": int(counts.min())}
        for key, ok in (("film_supported_1_2", np.isin(sub.cls_lo.values, [1, 2])),
                        ("full_film_1", sub.cls_lo.values == 1)):
            v, ci, slope = _logit_s50(logS, ok.astype(float), rng)
            rec[key] = {"S50": v, "bootstrap95": ci, "logit_slope_per_decade": slope,
                        "S50_inside_band_range": bool(sub.S.min() <= v <= sub.S.max()),
                        "fraction_overall": float(ok.mean())}
        s50.append(rec)
        # encoding: every class carries colour AND hatch (except class 1, solid fill with edge)
        assert [p.get_hatch() or "" for p in ys] == [h for *_, h in cls_style]
    x0 = min(min(xr), np.log10(S5.min())) - 0.05
    x1 = max(max(xr), np.log10(S5.max())) + 0.05
    for ax in axes_a:
        ax.set_xlim(x0, x1)
    handles = [matplotlib.patches.Patch(facecolor=c, hatch=h, edgecolor="k", lw=0.4, label=n) for _, n, c, h in cls_style]
    handles.append(Line2D([], [], color="k", ls="--", lw=1.0,
                          label=fr"rated corner, $S$ range of its 40 points ($\theta_{{\min}}={th5:.0f}^\circ$)"))
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=7)
    # (b) class-1 points: which constraint removes them (sigma* lo)
    bx = fig.add_subplot(gs[1, :])
    c1 = D[D.cls_lo == 1]
    T = c1.T_max_bnd.values
    g3 = c1.De_sq_bnd.values > cs.De_sq_max
    bres = {}
    cats = [("survives", C_GREEN, ""), (r"$g_2$ only", C_VERM, "\\\\\\\\"), (r"$g_3$ only", C_PURPLE, "...."),
            (r"$g_2$ and $g_3$", C_GREY, "xxxx")]
    for j, fc in enumerate(["mineral", "water_glycol"]):
        g2 = T > cs.T_max_C_by_class[fc]
        n = [int(np.sum(~g2 & ~g3)), int(np.sum(g2 & ~g3)), int(np.sum(~g2 & g3)), int(np.sum(g2 & g3))]
        # cross-check against the artefact's own labels
        lab = c1[f"binding_{fc}_lo"].values
        assert n[0] == int(np.sum(lab == "none")) and n[1] == int(np.sum(lab == "g2")) \
            and n[2] == int(np.sum(lab == "g3")) and n[3] == int(np.sum(lab == "multi")), (fc, n)
        left = 0
        for k, ((name, col, h), v) in enumerate(zip(cats, n)):
            bx.barh(j, v, left=left, color=col, hatch=h, edgecolor="k", lw=0.4, label=name if j == 0 else "_")
            if v / len(c1) >= 0.04:
                bx.text(left + v / 2, j, f"{v}\n{100 * v / len(c1):.0f}%", ha="center", va="center", fontsize=6.5,
                        color="k", bbox=dict(fc="w", ec="none", alpha=0.85, pad=0.8))
            left += v
        bres[fc] = {"survives": n[0], "g2_only": n[1], "g3_only": n[2], "g2_and_g3": n[3],
                    "shares": [v / len(c1) for v in n], "ceiling_C": cs.T_max_C_by_class[fc]}
    bx.set_yticks([0, 1]); bx.set_yticklabels(["mineral\n(110 °C)", "water–glycol\n(60 °C)"])
    bx.set_xlim(0, len(c1)); bx.set_xlabel(rf"full-film points ($\sigma^*$ lo), n = {len(c1)}")
    bx.set_title(r"(b) which constraint removes full-film points ($g_2$ film temperature, $g_3$ Deborah)", loc="left",
                 fontsize=8)
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=False)
    bx.invert_yaxis()
    fig.savefig(out / f"{fname}.pdf")
    dom = {fc: max(("g2", r["g2_only"] + r["g2_and_g3"]), ("g3", r["g3_only"] + r["g2_and_g3"]), key=lambda t: t[1])
           for fc, r in bres.items()}
    rep = {"figure": f"{fname}.pdf", "provenance": prov, "eta_basis": eta_basis,
           "memory_sign": "modulus + D-6 bound (corrected campaign rows as stored)",
           "sources": {p.name: sha(p) for p in [DP, EP, CP]},
           "quotes": {"paired": paired,
                      "paired_how": "boxD pure-film and all 24 boxE mixed arms share sample_id 0..8191 with "
                                    "bit-identical inputs (checked here)",
                      "duty_inputs_varying": ["p_ch", "c0", "eta_0", "omega", "theta_min_deg",
                                              "(also T_in, lam, eta_s_ratio; not in S)"],
                      "F_con_persisted": "yes: F_contact (and F_asp) on boxE corrected rows; boxD pure-film rows "
                                         "carry F_contact as well"},
           "S_definition": "p_ch A_p c0^2 / (3 pi eta_0 omega R_b^4), A_p = 2e-4 m^2, R_b = 6.5e-3 m; "
                           "ball_socket.solver.squeeze_capacity_scale; eta_0 = swept service viscosity",
           "S_hand_checks": {str(k): {"hand": v, "generator": float(D.S[k])} for k, v in HAND_S.items()},
           "class_shares": shares, "alternative_split": rows,
           "S50_sigma_lo": s50,
           "panel_b_sigma_lo": bres, "dominant_constraint_panel_b": dom,
           "rated_corner": {"theta_min_deg": th5, "S_range": [float(S5.min()), float(S5.max())],
                            "points": int(len(C5)),
                            "band_full_film_S50": s50[1]["full_film_1"]["S50"],
                            "band_film_supported_S50": s50[1]["film_supported_1_2"]["S50"],
                            "points_above_full_film_S50": int(np.sum(S5 > s50[1]["full_film_1"]["S50"])),
                            "points_above_film_supported_S50": int(np.sum(S5 > s50[1]["film_supported_1_2"]["S50"])),
                            "S_median": float(np.median(S5))},
           "plotted": "sigma* lo end in the figure; both ends in class_shares"}
    (out / f"{fname}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    rep["_frame"] = D   # in-memory only (R-65.2 restriction); not in the JSON
    return fig, rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--png", default=None, help="also write PNGs here for inspection")
    ap.add_argument("--paper-sat", action="store_true",
                    help="write only Paper 3's Figure F-SAT (fig_sat.pdf: delta >= 0.01, elsarticle width)")
    a = ap.parse_args()
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-61 figure generator")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.paper_sat:
        fig, rep = fig_u1(out, prov, dmin=0.01, figsize=(6.3, 3.6), name="fig_sat")
        if a.png:
            fig.savefig(pathlib.Path(a.png) / "fig_sat.png", dpi=170)
        print("wrote", rep["figure"])
        return 0
    for f in (fig_u1, fig_u2, fig_u3):
        fig, rep = f(out, prov)
        if a.png:
            fig.savefig(pathlib.Path(a.png) / rep["figure"].replace(".pdf", ".png"), dpi=170)
        plt.close(fig)
        print("wrote", rep["figure"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
