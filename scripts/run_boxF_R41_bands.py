"""R-41: the constant band, and the sigma-band / feasibility-band test on its
complement.

Stage ``rule``   writes the selection rule -- window endpoints included -- to
                 ``results/ch9/boxF_R41_band_rule.json`` and prints its SHA-256.
                 R-41.11: "Hash the rule with the window endpoints in it, then
                 compute."  Every later stage refuses to run unless the rule
                 file hashes to the value it was written with.
Stage ``solve``  the averaged film at BOTH ends of the sigma* band on a
                 stratified sample of smooth-converged units OUTSIDE the window.

The window is derived, not chosen (R-41.2):  g1 is  h_min / sigma* >= 3  with
sigma* registered as the band [sigma_lo, sigma_hi], so g1 is undetermined by the
registered constants exactly when  3 sigma_lo <= h_min <= 3 sigma_hi, i.e.
Lambda_bar = h_min / sigma_bar in [3 sigma_lo/sigma_bar, 3 sigma_hi/sigma_bar]
with sigma_bar the geometric mean.  Exact on the smooth solve, where h_min does
not depend on sigma.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
PURE = OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet"
MIXED = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
RULE = OUT / "boxF_R41_band_rule.json"
SOLVE_OUT = OUT / "boxF_R41_sigma_band_test.parquet"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
GRID = (32, 24)
N_PAIR = 12            # models j and j+12 differ only in sigma* (lo, hi)


def _registered():
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M, ConstraintSet
    import ball_socket.mixed as mx
    lo, hi = (float(s) for s in SIGMA_STAR_BAND_M)
    L = float(ConstraintSet().lambda_ratio_min)
    sb = float(np.sqrt(lo * hi))
    return dict(sigma_lo=lo, sigma_hi=hi, sigma_bar=sb, lambda_min=L,
                ratio=hi / sb, window=(L * lo / sb, L * hi / sb),
                h_window_m=(L * lo, L * hi),
                fit_floor_H=float(mx.flow_factor_valid_above(1.0)))


def _strata(win):
    """Lambda_bar strata outside the window; the two window-adjacent strata
    take the endpoints from the derived window, never from a typed number."""
    wlo, whi = win
    return [(0.0, 1.0), (1.0, 1.6), (1.6, 2.0), (2.0, wlo),
            (whi, 4.5), (4.5, 6.0), (6.0, float("inf"))]


def stage_rule(args):
    from ball_socket.immutable import guarded_path
    guarded_path(str(RULE), what="R-41 band rule")
    reg = _registered()
    rule = {
        "ruling": "R-41.2 / R-41.3 / R-41.11",
        "written_before_computation": True,
        "registered_inputs": {"sigma_star_band_m": [reg["sigma_lo"], reg["sigma_hi"]],
                              "g1": f"h_min_bnd / sigma* >= {reg['lambda_min']}",
                              "fit_floor_H": reg["fit_floor_H"],
                              "source": "campaign_config_v4.SIGMA_STAR_BAND_M, "
                                        "ConstraintSet, mixed.flow_factor_valid_above"},
        "constant_band": {
            "sigma_bar_m": reg["sigma_bar"], "ratio": reg["ratio"],
            "lambda_min": reg["lambda_min"],
            "Lambda_bar_window": list(reg["window"]),
            "h_min_window_m": list(reg["h_window_m"]),
            "definition": ("Lambda_bar = h_min_bnd / sigma_bar on the SMOOTH solve; "
                           "a row is in the constant band iff Lambda_bar lies in the "
                           "window (inclusive).  Row count is an OUTPUT."),
            "exact_on": "smooth solve only (averaged h_min depends on sigma via H)"},
        "feasibility_band": {
            "population": "smooth-converged units with Lambda_bar OUTSIDE the window",
            "units": {"pure_film": "sample",
                      "mixed": "(sample, j), j = 0..11: model j at sigma_lo and "
                               "model j+12 at sigma_hi (identical but for sigma*); "
                               "outside the window iff BOTH members are"},
            "solve": ("full registered procedure (evaluate_x, closure on, 32x24), "
                      "averaged film, at BOTH sigma* ends; treatment 'extrapolate'; "
                      "'freeze' is ALSO solved for any solve whose averaged "
                      "h_min_bnd/sigma <= the fit floor (else the two treatments "
                      "are identical by construction: no cell below the floor)"),
            "classification": {
                "window_widening": "averaged g1 differs between the two sigma* ends",
                "feasibility_band": ("averaged g1 is the same at both ends AND differs "
                                     "from the smooth g1 (which is the same at both "
                                     "ends outside the window)"),
                "agree": "otherwise",
                "existence_change": ("averaged outcome is not converged -- outside "
                                     "the existence band this contradicts kappa "
                                     "invariance and is reported, not classified")},
            "widening_rule": ("if any window_widening unit is found the window is "
                              "widened on the averaged side by the MEASURED amount "
                              "(the outermost Lambda_bar at which it occurs), "
                              "reported as a measurement"),
            "null_rule": ("an empty feasibility band is reported WITH ITS POWER: units "
                          "tested, Lambda_bar range covered, and the smallest g1 margin "
                          "|Lambda - 3|/3 among tested solves.  It reads 'no film-model "
                          "uncertainty resolvable at this resolution over this range', "
                          "never 'no uncertainty'"),
            "sample": {"strata_Lambda_bar": [list(s) for s in _strata(reg["window"])],
                       "per_stratum_per_arm": args.n, "seed": args.seed,
                       "stratify_on": "Lambda_bar of the unit (mixed: sigma_lo member)"},
            "if_nonempty": ("the band is drawn only after the at-risk complement is "
                            "enumerated; its cost is brought back first")},
    }
    txt = json.dumps(rule, indent=2, sort_keys=True, default=float)
    RULE.write_text(txt, encoding="utf-8")
    h = hashlib.sha256(RULE.read_bytes()).hexdigest()
    RULE.with_suffix(".sha256").write_text(h + "\n")
    print(txt)
    print("SHA-256", h)
    return 0


def _load_rule():
    h = hashlib.sha256(RULE.read_bytes()).hexdigest()
    rec = RULE.with_suffix(".sha256").read_text().strip()
    if h != rec:
        raise SystemExit(f"rule file changed after hashing: {h} != {rec}")
    return json.loads(RULE.read_text(encoding="utf-8")), h


def job(args):
    key, x, model_id, sigma, treatment = args
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.campaign as C
    import ball_socket.mixed as mx
    model = None
    if model_id >= 0:
        model = mx.expand_for_solving(mx.registered_contact_models(
            branch="psi", summit_convention="derived"))[model_id][0]
    row = C.evaluate_x(x, contact=model, clearance_closure=True, delta_T_datum="bulk",
                       Ntheta=GRID[0], Nphi=GRID[1],
                       flow_factors={"sigma": float(sigma), "below_floor": treatment})
    keep = ["outcome", "realised_film_model", "realised_contact_model", "eps_ecc",
            "h_min_bnd", "De_sq_bnd", "T_max_bnd", "P_loss", "c_equilibrium",
            "fail_reason"]
    out = {k: row.get(k) for k in keep}
    out.update(key=key, model_id=int(model_id), sigma=float(sigma), treatment=treatment)
    return out


def _pool(fn, jobs, workers, label):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0, res = time.time(), []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            res.append(f.result())
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {label}: {i}/{len(jobs)}  {el / 60:.1f} min, "
                      f"~{el / i * (len(jobs) - i) / 60:.1f} min left", flush=True)
    return res


def _units(rule):
    """Smooth-converged units outside the window, with their smooth quantities."""
    cb = rule["constant_band"]
    sb, (wlo, whi) = cb["sigma_bar_m"], cb["Lambda_bar_window"]
    lo, hi = rule["registered_inputs"]["sigma_star_band_m"]
    cols = AXES + ["sample_id", "outcome", "h_min_bnd", "T_max_bnd", "De_sq_bnd"]
    P = pd.read_parquet(PURE, columns=cols)
    P = P[P.outcome == "converged"].copy()
    P["Lb"] = P.h_min_bnd / sb
    P = P[(P.Lb < wlo) | (P.Lb > whi)]
    pu = pd.DataFrame({"arm": "pure_film", "sample_id": P.sample_id.astype(int),
                       "j": -1, "Lb": P.Lb, "h_lo_s": P.h_min_bnd, "h_hi_s": P.h_min_bnd,
                       **{k: P[k] for k in AXES}})
    M = pd.read_parquet(MIXED, columns=cols + ["model_id"])
    M = M[M.outcome == "converged"].copy()
    M["model_id"] = M.model_id.astype(int)
    a = M[M.model_id < N_PAIR].assign(j=lambda d: d.model_id)
    b = M[M.model_id >= N_PAIR].assign(j=lambda d: d.model_id - N_PAIR)
    m = a.merge(b[["sample_id", "j", "h_min_bnd"]], on=["sample_id", "j"],
                suffixes=("", "_hi"))
    m["Lb"], m["Lb_hi"] = m.h_min_bnd / sb, m.h_min_bnd_hi / sb
    out_ = lambda s: (s < wlo) | (s > whi)
    m = m[out_(m.Lb) & out_(m.Lb_hi)]
    mu = pd.DataFrame({"arm": "mixed", "sample_id": m.sample_id.astype(int),
                       "j": m.j.astype(int), "Lb": m.Lb, "h_lo_s": m.h_min_bnd,
                       "h_hi_s": m.h_min_bnd_hi, **{k: m[k] for k in AXES}})
    return pd.concat([pu, mu], ignore_index=True), lo, hi


def stage_solve(args):
    from ball_socket.immutable import guarded_path
    guarded_path(str(SOLVE_OUT), what="R-41 sigma-band test")
    rule, h = _load_rule()
    print("rule", h)
    fb = rule["feasibility_band"]
    units, lo, hi = _units(rule)
    L = rule["constant_band"]["lambda_min"]
    edges = fb["sample"]["strata_Lambda_bar"]
    units["stratum"] = -1
    for i, (a, b) in enumerate(edges):
        units.loc[(units.Lb >= a) & (units.Lb < b), "stratum"] = i
    units = units[units.stratum >= 0]
    n, seed = fb["sample"]["per_stratum_per_arm"], fb["sample"]["seed"]
    pick = (units.groupby(["arm", "stratum"], group_keys=False)
                 .apply(lambda g: g.sample(min(n, len(g)), random_state=seed))
                 .reset_index(drop=True))
    pick["unit"] = np.arange(len(pick))
    print(f"population outside window: {units.groupby('arm').size().to_dict()}; "
          f"sampled {len(pick)} units")

    def jobs_for(treat, which):
        J = []
        for r in pick.itertuples():
            if which is not None and (r.unit, treat) not in which:
                continue
            x = {k: float(getattr(r, k)) for k in AXES}
            for end, s in (("lo", lo), ("hi", hi)):
                mid = -1 if r.arm == "pure_film" else (r.j if end == "lo" else r.j + N_PAIR)
                J.append(((int(r.unit), end), x, mid, s, treat))
        return J
    res = _pool(job, jobs_for("extrapolate", None), args.workers, "extrapolate")
    df = pd.DataFrame(res)
    floor = rule["registered_inputs"]["fit_floor_H"]
    low = df[(df.outcome == "converged") & (df.h_min_bnd / df.sigma <= floor)]
    need = {(k[0], "freeze") for k in low.key}
    if need:
        print(f"{len(need)} units have an averaged solve at/below the fit floor: "
              f"solving 'freeze' for them")
        df = pd.concat([df, pd.DataFrame(_pool(job, jobs_for("freeze", need),
                                               args.workers, "freeze"))])
    df["unit"] = df.key.map(lambda k: k[0])
    df["end"] = df.key.map(lambda k: k[1])
    df = df.drop(columns="key").merge(pick, on="unit")
    df["h_s"] = np.where(df.end == "lo", df.h_lo_s, df.h_hi_s)
    df["g1_s"] = df.h_s / df.sigma >= L
    df["g1_a"] = np.where(df.outcome == "converged", df.h_min_bnd / df.sigma >= L, np.nan)
    df["rule_sha256"] = h
    df.to_parquet(SOLVE_OUT, index=False)
    return _report(df, rule, h)


def _report(df, rule, h):
    L = rule["constant_band"]["lambda_min"]
    rep = {"rule_sha256": h, "solves": int(len(df)),
           "units": int(df.unit.nunique()),
           "units_by_arm": df.groupby("arm").unit.nunique().to_dict(),
           "outcomes": df.outcome.value_counts().to_dict(),
           "Lambda_bar_range": [float(df.Lb.min()), float(df.Lb.max())]}
    per = []
    for (u, t), g in df.groupby(["unit", "treatment"]):
        if set(g.end) != {"lo", "hi"}:
            continue
        a = g.set_index("end")
        conv = (a.outcome == "converged").all()
        s_lo, s_hi = bool(a.loc["lo", "g1_s"]), bool(a.loc["hi", "g1_s"])
        rec = {"unit": u, "treatment": t, "arm": a.arm.iloc[0], "Lb": float(a.Lb.iloc[0]),
               "smooth_same_both_ends": s_lo == s_hi}
        if not conv:
            rec["cls"] = "existence_change"
        else:
            A_lo, A_hi = bool(a.loc["lo", "g1_a"]), bool(a.loc["hi", "g1_a"])
            rec["cls"] = ("window_widening" if A_lo != A_hi else
                          "feasibility_band" if A_lo != s_lo else "agree")
            rec["min_margin"] = float(np.min(np.abs(
                a.h_min_bnd.astype(float) / a.sigma / L - 1)))
        per.append(rec)
    p = pd.DataFrame(per)
    rep["classification"] = {f"{t}|{c}": int(n) for (t, c), n in
                             p.groupby(["treatment", "cls"]).size().items()}
    rep["smooth_determinate_both_ends"] = int(p.smooth_same_both_ends.sum())
    rep["units_classified"] = int(len(p))
    ag = p[p.cls == "agree"]
    rep["power"] = {"units_tested": int(p.unit.nunique()),
                    "Lambda_bar_range": [float(p.Lb.min()), float(p.Lb.max())],
                    "smallest_g1_margin": float(p.min_margin.min())
                    if "min_margin" in p else None}
    ww = p[p.cls == "window_widening"]
    rep["window_widening_Lambda_bar"] = sorted(float(v) for v in ww.Lb)
    rep["feasibility_band_Lambda_bar"] = sorted(float(v) for v in p[p.cls == "feasibility_band"].Lb)
    rep["null"] = (len(ww) == 0) and not (p.cls == "feasibility_band").any()
    rep["relative_h_change"] = {
        "median": float((df.h_min_bnd.astype(float) / df.h_s - 1).median()),
        "min": float((df.h_min_bnd.astype(float) / df.h_s - 1).min()),
        "max": float((df.h_min_bnd.astype(float) / df.h_s - 1).max())}
    path = SOLVE_OUT.with_suffix(".report.json")
    from ball_socket.immutable import guarded_path
    guarded_path(str(path), what="R-41 sigma-band report")
    path.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rule", "solve", "report"])
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.stage == "rule":
        return stage_rule(args)
    if args.stage == "solve":
        return stage_solve(args)
    rule, h = _load_rule()
    return _report(pd.read_parquet(SOLVE_OUT), rule, h)


if __name__ == "__main__":
    raise SystemExit(main())
