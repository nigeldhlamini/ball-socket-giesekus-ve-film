"""R-65.1 / R-65.2: the operating space re-based on the solvent viscosity, and restricted to
Deborah-valid points.  No solves.

R-65.1: S_s = S eta_0 / eta_s = S / (eta_s/eta_0), the high-shear plateau (defined at every
sampled point).  The classification, binning, logit S50 and bootstrap are fig_R61_update.fig_u3
itself (same code, same seed), called with eta_basis='eta_s'; the R-62.4 artefact
(R62_operating_space.json, eta_0 basis) is the "before".  The class counts do not depend on S
and are asserted unchanged.  Every move is classified.  No registered expectation (descriptive).
Writes Paper 3's Figure 6 (figures/fig_ops.pdf), its encoding asserted against v5's caption.

R-65.2: g3 is De_sq_bnd > ConstraintSet.De_sq_max.  Classes 1-2: the pure-film equilibrium's
De_sq_bnd (box D).  Class 3: valid when ANY converged mixed arm at that sigma* end satisfies g3
(the variant "all converged arms" is reported beside it).  Class 4 has no equilibrium above
the bracket floor, and box D/E persist no De_sq for it (asserted: all NaN), so class 4 is kept
whole.  The restricted map drops the g3-violating points of classes 1-3 and refits the six
S50s on S_s with the same procedure and seed.
Registered (R-65.2): the largest relative S50 shift is at most 10 %.  Failing is NOT a STOP.
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
BEFORE = OUT / "R62_operating_space.json"
DEST_JSON = OUT / "R65_ops_solv.json"
DEST_PQ = OUT / "R65_ops_solv.parquet"
FIGDIR = pathlib.Path("Z:/Papers/Paper 3/New/figures")
V5 = pathlib.Path("Z:/Papers/Paper 3/New/paper3_v5_draft.tex")

TAGS = {
    "basis": "S_s (eta_s): S_s = p_ch A_p c0^2 / (3 pi eta_s omega R_b^4) = S / (eta_s/eta_0)",
    "memory_sign": "modulus + D-6 bound (corrected campaign rows as stored; classes as R-62.4)",
    "floor": "production floor delta = h_min/c >= 0.01 (class 4: no equilibrium above it)",
}
KEYS = (("film_supported_1_2", "Film"), ("full_film_1", "Full"))
BANDS = ("One", "Two", "Three")


def _move(old, new):
    lo, hi = old["bootstrap95"]
    r = new["S50"] / old["S50"]
    where = "inside" if lo <= new["S50"] <= hi else ("above" if new["S50"] > hi else "below")
    return {"S50_eta0": old["S50"], "S50_eta_s": new["S50"], "ratio": r,
            "ci_eta0": old["bootstrap95"], "ci_eta_s": new["bootstrap95"],
            "class": f"{'up' if r > 1 else 'down'} x{r:.3f}; new value {where} the eta_0-basis 95 % interval"}


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-65.1/R-65.2 operating space on S_s")
    import fig_R61_update as G
    from ball_socket.campaign_config_v4 import ConstraintSet
    cs = ConstraintSet()
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, rep = G.fig_u3(FIGDIR, prov, eta_basis="eta_s", figsize=(6.6, 4.4), fname="fig_ops")
    D = rep.pop("_frame")
    (FIGDIR / "fig_ops.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")

    # ---- R-65.1: counts unchanged, moves classified
    for e in ("lo", "hi"):
        assert {int(k): v for k, v in rep["class_shares"][e]["counts"].items()} == \
               {int(k): v for k, v in before["class_counts"][e].items()}, ("class counts moved", e)
    moves = {}
    for i, band in enumerate(BANDS):
        for key, tag in KEYS:
            moves[f"OpsSFifty{tag}{band}"] = _move(before["S50_sigma_lo"][i][key], rep["S50_sigma_lo"][i][key])
    rc0, rc = before["rated_corner"], rep["rated_corner"]
    S5 = np.array(rc["S_range"])
    corner = {"points": rc["points"], "theta_min_deg": rc["theta_min_deg"], "band": "Two",
              "S_s_range": rc["S_range"], "S_s_median": rc["S_median"],
              "eta_s_ratio_at_corner": 0.95,
              "full_film_S50_band_two": rc["band_full_film_S50"],
              "full_film_S50_ci": rep["S50_sigma_lo"][1]["full_film_1"]["bootstrap95"],
              "film_supported_S50_band_two": rc["band_film_supported_S50"],
              "film_supported_S50_ci": rep["S50_sigma_lo"][1]["film_supported_1_2"]["bootstrap95"],
              "above_full_film_S50": {"eta0": rc0["points_above_full_film_S50"], "eta_s": rc["points_above_full_film_S50"]},
              "above_film_supported_S50": {"eta0": rc0["points_above_film_supported_S50"],
                                           "eta_s": rc["points_above_film_supported_S50"]}}
    # each corner point against the band-two boundaries and their intervals
    C5 = pd.read_parquet(OUT / "E5_VG32_g1_binding_corner.parquet")[
        ["omega", "p_ch", "c0", "eta_0", "theta_min_deg", "eta_s_ratio"]].drop_duplicates()
    s5 = G.duty_S(C5.p_ch.values, C5.c0.values, C5.eta_0.values, C5.omega.values) / C5.eta_s_ratio.values
    assert np.isclose(s5.min(), S5[0]) and np.isclose(s5.max(), S5[1])
    for key, nm in (("full_film_1", "full"), ("film_supported_1_2", "film")):
        r = rep["S50_sigma_lo"][1][key]
        lo, hi = r["bootstrap95"]
        corner[f"vs_{nm}"] = {"below_ci": int(np.sum(s5 < lo)), "inside_ci": int(np.sum((s5 >= lo) & (s5 <= hi))),
                              "above_ci": int(np.sum(s5 > hi)),
                              "decades_min_max_from_S50": [float(np.log10(s5.min() / r["S50"])),
                                                           float(np.log10(s5.max() / r["S50"]))]}

    # ---- R-65.2: Deborah-valid points
    E = pd.read_parquet(OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet",
                        columns=["sample_id", "model_id", "outcome", "De_sq_bnd"])
    Dr = pd.read_parquet(OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet",
                         columns=["sample_id", "outcome", "De_sq_bnd"]).set_index("sample_id")
    D = D.join(Dr[["De_sq_bnd"]].rename(columns={"De_sq_bnd": "De_pf"}), on="sample_id")
    De_max = cs.De_sq_max
    g3 = {}
    for end, arms in (("lo", range(0, 12)), ("hi", range(12, 24))):
        m = E[E.model_id.isin(list(arms)) & (E.outcome == "converged")]
        assert m.De_sq_bnd.notna().all()
        ok = m.assign(ok=m.De_sq_bnd <= De_max).groupby("sample_id").ok
        any_ok = ok.any().reindex(D.sample_id).fillna(False).values
        all_ok = ok.all().reindex(D.sample_id).fillna(False).values
        cls = D[f"cls_{end}"].values
        pf_ok = (D.De_pf <= De_max).values
        assert np.isfinite(D.De_pf.values[np.isin(cls, [1, 2])]).all()
        valid_any = np.where(np.isin(cls, [1, 2]), pf_ok, np.where(cls == 3, any_ok, True))
        valid_all = np.where(np.isin(cls, [1, 2]), pf_ok, np.where(cls == 3, all_ok, True))
        D[f"g3_valid_{end}"] = valid_any
        eq = np.isin(cls, [1, 2, 3])
        g3[end] = {"points_with_equilibrium": int(eq.sum()),
                   "g3_valid": int((eq & valid_any).sum()),
                   "g3_valid_pct": 100 * float((eq & valid_any).sum() / eq.sum()),
                   "g3_valid_pct_class3_all_arms": 100 * float((eq & valid_all).sum() / eq.sum()),
                   "by_class": {int(c): {"n": int((cls == c).sum()), "valid": int(((cls == c) & valid_any).sum())}
                                for c in (1, 2, 3)}}
    # class 4: is there a De_sq at the bracket floor?
    nc = Dr[Dr.outcome != "converged"]
    E4 = E[E.outcome != "converged"]
    class4_De = {"boxD_nonconverged_rows": int(len(nc)), "with_De_sq_bnd": int(nc.De_sq_bnd.notna().sum()),
                 "boxE_nonconverged_rows": int(len(E4)), "with_De_sq_bnd_E": int(E4.De_sq_bnd.notna().sum())}
    assert class4_De["with_De_sq_bnd"] == 0 and class4_De["with_De_sq_bnd_E"] == 0
    # restricted S50s (sigma* lo, as plotted), same procedure and seed
    rng = np.random.default_rng(G.SEED)
    restricted, shifts = [], {}
    for i, ((t0, t1), band) in enumerate(zip(G.THETA_BANDS, BANDS)):
        sub = D[(D.theta_min_deg >= t0) & (D.theta_min_deg < t1) & D.g3_valid_lo]
        logS = np.log10(sub.S.values)
        rec = {"band_deg": [t0, min(t1, 40.0)], "n": int(len(sub))}
        for key, tag in KEYS:
            ok = np.isin(sub.cls_lo.values, [1, 2]) if tag == "Film" else sub.cls_lo.values == 1
            v, ci, slope = G._logit_s50(logS, ok.astype(float), rng)
            rec[key] = {"S50": v, "bootstrap95": ci, "logit_slope_per_decade": slope}
            full = rep["S50_sigma_lo"][i][key]["S50"]
            shifts[f"OpsSFifty{tag}{band}"] = {"unrestricted": full, "restricted": v, "shift_pct": 100 * (v / full - 1)}
        restricted.append(rec)
    worst = max(shifts.items(), key=lambda kv: abs(kv[1]["shift_pct"]))
    max_shift = abs(worst[1]["shift_pct"])

    # ---- encoding asserted against v5's Fig. 6 caption
    tex = V5.read_text(encoding="utf-8")
    cap = tex[tex.index("\\label{fig:ops}") - 900: tex.index("\\label{fig:ops}")]
    cap = cap[cap.index("\\caption{"):]
    for phrase in ("(a) Fraction of sampled points in each class", "against the duty number $S$",
                   "three bands of relief angle", "smooth end of the", "dashed, the duty range of the rated corner",
                   "(b) The full-film points of (a)", "mineral and water--glycol ceilings"):
        assert phrase in " ".join(cap.split()), phrase
    axs = fig.axes
    assert len([a for a in axs if a.get_xlabel().startswith(r"$\log_{10} S$")]) == 3
    spans = [p for a in axs for p in a.patches if p.get_linestyle() == "--"]
    assert len(spans) == 1, "exactly one dashed rated-corner span (band two)"
    assert "sigma" in axs[0].get_ylabel() and "lo" in axs[0].get_ylabel()

    out = {"provenance": prov, **TAGS, "source": "scripts/fig_R61_update.py:fig_u3(eta_basis='eta_s')",
           "class_counts": {e: rep["class_shares"][e]["counts"] for e in ("lo", "hi")},
           "class_counts_unchanged_vs_R62": True,
           "S50_sigma_lo": rep["S50_sigma_lo"], "moves": moves, "rated_corner": corner,
           "survivors_sigma_lo": rep["panel_b_sigma_lo"],
           "R65_2": {"g3_rule": f"valid iff De_sq_bnd <= {De_max} (ConstraintSet.De_sq_max); class 3 valid if ANY "
                                "converged mixed arm at that sigma* end is",
                     "g3": g3, "class4_De_sq": class4_De,
                     "class4_handling": "kept whole: no De_sq exists for points with no equilibrium",
                     "restricted_S50_sigma_lo": restricted, "shifts": shifts,
                     "max_abs_shift_pct": max_shift, "max_shift_at": worst[0],
                     "registered_at_most_10pct": bool(max_shift <= 10.0)},
           "figure": str(FIGDIR / "fig_ops.pdf"), "caption_source": "paper3_v5_draft.tex fig:ops (Fig. 6)"}
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST_JSON), what="R-65.1 operating space on S_s")
    guarded_path(str(DEST_PQ), what="R-65.1 classes on S_s")
    D[["sample_id", "theta_min_deg", "S", "cls_lo", "cls_hi", "g3_valid_lo", "g3_valid_hi"]].rename(
        columns={"S": "S_s"}).assign(git_commit=prov["commit"], memory_sign=TAGS["memory_sign"]).to_parquet(DEST_PQ, index=False)
    DEST_JSON.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"moves": {k: v["class"] for k, v in moves.items()}, "corner": corner,
                      "g3": {e: g3[e]["g3_valid_pct"] for e in g3}, "shifts": {k: round(v["shift_pct"], 2) for k, v in shifts.items()},
                      "registered_at_most_10pct": out["R65_2"]["registered_at_most_10pct"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
