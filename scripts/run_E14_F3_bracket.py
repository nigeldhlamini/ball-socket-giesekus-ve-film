"""E14's F3 term, bounded by BRACKETING rather than by a campaign (R-44.2).

F3 changes sigma_s only: box C used ``sigma_s_eq_sigma_star`` (sigma_s = sigma*),
box D uses ``derived`` (sigma_s = r sigma*, r in [0.841, 0.933]).  Smaller sigma_s
means larger H, so the DERIVED arm carries LESS asperity load than box C's at
every gap.

THE ARGUMENT (R-44.2).  At a fixed clearance the equilibrium solves
F_film(eps) + F_asp(eps) = W with both terms increasing in eps.  So if one arm's
F_asp dominates another's AS A FUNCTION OF eps, its equilibrium sits at a smaller
eps, hence a larger h_min.  For a row solved under derived arm j:

    F_asp^{boxC,j}(eps) >= F_asp^{j}(eps)    (same prefactors, smaller sigma_s)
        =>  eps*_{boxC,j} <= eps*_j  =>  h_min^{boxC,j} >= h_min^j

and if some OTHER registered arm A dominates box C's curve,

    F_asp^{A}(eps) >= F_asp^{boxC,j}(eps)    =>  h_min^{boxC,j} <= h_min^A

so h_min under F3's box-C convention is BRACKETED by two rows already solved.
If those two rows carry the same verdict, box C's verdict is the same, and the
row needs no solve.  De_sq = lambda phi omega eps cos(theta_min) is monotone in
eps, so it is bracketed by the same pair; g2 needs no bracketing (its correction
is inert at +0.096 K maximum, E14).

WHAT IS RECOMPUTED, AND WHY IT NEEDS NO SOLVER.  With e = (0, 0, -eps c),
gap() is h = c (1 - eps n_z) -- LINEAR in eps -- and the load projection is a
fixed geometric weight (n_z dA).  Greenwood-Tripp's asperity pressure is closed
form in h.  So every F_asp curve is a recompute on rows already solved:

    F_asp(eps) = prefactor(arm) * sum_cells F_n(h(eps)/sigma_s) n_z dA

and the prefactor does not depend on sigma_s, so F3 moves only the F_n argument.

RESIDUAL.  Rows where no arm dominates, or where the bracket straddles the
threshold, are reported and solved -- they are the only ones that need a solver.

LIMIT, STATED.  The bracket holds at a FIXED clearance.  Arms of one sample
reach clearances that differ by at most 1.03 % (measured, box F cap stage), and
the bracketing pair are rows at their own clearances, so the Lambda bracket
inherits that spread.  Rows whose bracket straddles the threshold by less than
the spread are counted into the residual and solved rather than argued.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
MIXED = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
DEST = OUT / "E14_F3_bracket.parquet"
GRID = (32, 24)
N_EPS = 33
C_SPREAD = 0.0103          # measured max within-sample clearance spread across arms
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]


# ---------------------------------------------------------------------------
# F_n by interpolation, with the error measured against the quadrature
# ---------------------------------------------------------------------------
def _fn_interp(n, hi=40.0, nodes=4001):
    """log F_n on a uniform H grid, cubic-splined.  F_n falls like exp(-H^2/2),
    so the log is smooth and the spline is accurate; the error is MEASURED
    against mixed.F_n rather than assumed."""
    from scipy.interpolate import CubicSpline
    import ball_socket.mixed as mx
    H = np.linspace(0.0, hi, nodes)
    F = np.array([mx.F_n(n, float(h)) for h in H])
    good = F > 0
    sp = CubicSpline(H[good], np.log(F[good]))
    H_lo, H_hi = H[good][0], H[good][-1]

    def f(x):
        x = np.asarray(x, dtype=float)
        out = np.exp(sp(np.clip(x, H_lo, H_hi)))
        return np.where(x > H_hi, 0.0, out)
    return f


def _check_interp(f, n, rng, k=200):
    import ball_socket.mixed as mx
    H = rng.uniform(0.0, 12.0, k)
    ref = np.array([mx.F_n(n, float(h)) for h in H])
    got = f(H)
    return float(np.max(np.abs(got / ref - 1.0)))


# ---------------------------------------------------------------------------
# per-sample recompute
# ---------------------------------------------------------------------------
def _sample_job(args):
    sid, x, rows, eps_cap = args
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.mixed as mx
    from ball_socket.geometry import BallSocketGeometry
    from ball_socket.spherical import Spherical, SphericalConditions
    import run_E14_F3_bracket as S

    f25, f20 = S.F25, S.F20
    geom = BallSocketGeometry(c0=x["c0"], theta_min=np.deg2rad(x["theta_min_deg"]))
    cond = SphericalConditions(p_ch=x["p_ch"], p_case=1e5, e_vec=(0.0, 0.0, 0.0),
                               e_dot=(0.0, 0.0, 0.0), omega_b=(0.0, 0.0, 0.0))
    models = mx.expand_for_solving(mx.registered_contact_models(
        branch="psi", summit_convention="derived"))
    out = []
    E_STAR = mx._E_STAR()
    eps_grid = np.linspace(1e-3, eps_cap, S.N_EPS)
    for r in rows:                       # one entry per solved row of this sample
        c = float(r["c"])
        sph = Spherical(geom.R_b, c, geom.theta_max, cond, *S.GRID,
                        theta_min=geom.theta_min)
        nz, dA = sph.load_axis_factors()
        w = nz * dA
        n_z = sph.mesh.normal()[2]
        # h(eps) = c (1 - eps n_z), linear in eps
        H_of = lambda s: (c * (1.0 - eps_grid[:, None, None] * n_z[None, ...])) / s
        sig_cache = {}

        def proj(sigma_s, order):
            key = (sigma_s, order)
            if key not in sig_cache:
                F = (f25 if order == 2.5 else f20)(H_of(sigma_s))
                sig_cache[key] = np.einsum("eij,ij->e", F, w)
            return sig_cache[key]

        def curve(model, prov, sigma_s):
            """F_asp(eps) for a model whose summit spread is sigma_s."""
            if prov.endswith("elastic"):
                pre = ((16.0 * math.sqrt(2.0) / 15.0) * math.pi
                       * model.eta_beta_sigma ** 2 * E_STAR
                       * math.sqrt(model.sigma_s_over_beta_a))
                return pre * proj(sigma_s, 2.5)
            pre = 2.0 * math.pi * model.eta_beta_sigma ** 2 * float(model.H_ind)
            return pre * proj(sigma_s, 2.0)

        j = int(r["model_id"])
        mj, pj = models[j]
        c_boxC = curve(mj, pj, float(mj.sigma_star))          # F3: sigma_s = sigma*
        c_derived = curve(mj, pj, float(mj.sigma_summit()))
        rec = {"sample_id": int(sid), "model_id": j, "state": r["state"],
               "boxC_above_own": bool(np.all(c_boxC >= c_derived)),
               "max_ratio_boxC_over_own": float(
                   np.max(c_boxC / np.maximum(c_derived, 1e-300)))}
        if r["state"] == "cap":
            # EXACT, and no bracket needed: the row had no equilibrium because
            # film + asperity fell short of the target AT THE CAP.  F3 adds
            # only asperity load, so if the ADDED load at the cap is smaller
            # than the stored deficit, the row still cannot balance anywhere.
            add = float(c_boxC[-1] - c_derived[-1])           # eps_grid ends at the cap
            rec.update(deficit=float(r["deficit"]), added_at_cap=add,
                       verdict_fixed=bool(add < float(r["deficit"])),
                       fixed_by="cap deficit", n_dominating=-1)
            out.append(rec)
            continue
        # UNIVERSAL UPPER BOUND on h_min under F3, exact and needing no other
        # arm.  At box C's equilibrium eps_C <= eps*_j, so
        #     F_asp^C(eps_C) = W - F_film(eps_C) >= W - F_film(eps*_j)
        #                    = F_asp^j(eps*_j),
        # using only that F_film increases with eps.  F_asp^C increases with
        # eps too, so eps_C is at least the eps where box C's own curve reaches
        # arm j's equilibrium asperity load -- invert the closed-form curve.
        # h_min(eps) = c (1 - eps max n_z) is decreasing, so that gives the
        # upper bound on h_min.
        F_eq = float(r["F_asp_eq"])
        nz_max = float(np.max(n_z))
        if F_eq > 0 and np.any(c_boxC >= F_eq):
            eps_min = float(np.interp(F_eq, c_boxC, eps_grid))
        else:
            eps_min = float(eps_grid[0])
        rec.update(eps_lower_bound=eps_min,
                   h_min_upper=float(c * (1.0 - eps_min * nz_max)),
                   h_min_own=float(c * (1.0 - float(r["eps_eq"]) * nz_max)))
        dominates = []
        for a, (ma, pa) in enumerate(models):
            if a == j:
                continue
            ca = curve(ma, pa, float(ma.sigma_summit()))
            if np.all(ca >= c_boxC):
                dominates.append(a)
        rec.update(dominating_arms=dominates, n_dominating=len(dominates),
                   fixed_by="inversion", verdict_fixed=None)
        out.append(rec)
    return out


def stage_bracket(args):
    from ball_socket.immutable import guarded_path, guarded_write
    if not args.overwrite:
        guarded_path(str(DEST), what="E14 F3 bracket")
    rng = np.random.default_rng(args.seed)
    err25, err20 = _check_interp(F25, 2.5, rng), _check_interp(F20, 2.0, rng)
    print(f"F_n interpolant max relative error vs quadrature: "
          f"F_5/2 {err25:.2e}, F_2 {err20:.2e}", flush=True)
    if max(err25, err20) > 1e-6:
        raise SystemExit("interpolant error too large for a verdict")
    M = pd.read_parquet(MIXED, columns=AXES + [
        "sample_id", "model_id", "outcome", "eps_ecc", "c_equilibrium",
        "h_min_bnd", "De_sq_bnd", "realised_eps_cap", "F_film_at_limit",
        "F_asp_at_limit", "F_target_at_failure", "F_asp"])
    M["model_id"] = M.model_id.astype(int)
    M = M[M.outcome.isin(["converged", "contact_limit"])].copy()
    M["state"] = np.where(M.outcome == "converged", "equilibrium", "cap")
    # THE ARTEFACT HAS NO c_at_failure (it predates ERR-16's typed columns), so a
    # contact_limit row's clearance is c0 EXCEPT where the limit was reached on
    # a later outer iteration.  Box F's cap stage measured exactly which samples
    # those are by reproduction; they are routed to the residual, not assumed.
    late = set()
    cap_stage = OUT / "boxF_band_cap_stage.parquet"
    if cap_stage.exists():
        cs = pd.read_parquet(cap_stage, columns=["sample_id", "reproduces"])
        late = set(cs.loc[~cs.reproduces, "sample_id"].astype(int))
    M["late_failure"] = M.sample_id.astype(int).isin(late) & (M.state == "cap")
    M["c"] = np.where(M.state == "equilibrium", M.c_equilibrium, M.c0)
    M["deficit"] = (M.F_target_at_failure - M.F_film_at_limit - M.F_asp_at_limit)
    if args.limit:
        keep = rng.choice(M.sample_id.unique(), args.limit, replace=False)
        M = M[M.sample_id.isin(keep)]
    jobs = []
    for sid, g in M.groupby("sample_id"):
        x = {k: float(g.iloc[0][k]) for k in AXES}
        rows = [{"model_id": int(r.model_id), "c": float(r.c), "state": r.state,
                 "deficit": float(r.deficit) if r.state == "cap" else float("nan"),
                 "F_asp_eq": float(r.F_asp) if r.state == "equilibrium" else float("nan"),
                 "eps_eq": float(r.eps_ecc) if r.state == "equilibrium" else float("nan")}
                for r in g.itertuples()]
        jobs.append((int(sid), x, rows, float(g.iloc[0].realised_eps_cap)))
    print(f"{len(jobs)} samples, {len(M)} rows; recompute only, no solves", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = [r for chunk in ex.map(_sample_job, jobs, chunksize=8) for r in chunk]
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    b = pd.DataFrame(res).merge(
        M[["sample_id", "model_id", "outcome", "h_min_bnd", "eps_ecc", "De_sq_bnd",
           "late_failure"]], on=["sample_id", "model_id"])
    b["verdict_fixed"] = b.verdict_fixed.map(
        lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else False)
    b["dominating_arms"] = b.dominating_arms.map(
        lambda v: list(v) if isinstance(v, (list, np.ndarray)) else [])
    guarded_write(str(DEST), lambda p: b.to_parquet(p, index=False),
                  what="E14 F3 bracket", overwrite=args.overwrite)
    print(b.groupby("state").agg(n=("model_id", "size"),
                                 boxC_above_own=("boxC_above_own", "mean"),
                                 n_dominating_median=("n_dominating", "median")))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bracket", "residual", "residual_report", "bound"])
    ap.add_argument("--n-resid", type=int, default=100,
                    help="rows per Lambda stratum of the residual")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="restrict to N samples (0 = all)")
    args = ap.parse_args()
    if args.stage == "bracket":
        return stage_bracket(args)
    if args.stage == "bound":
        return stage_bound(args)
    if args.stage == "residual_report":
        return _residual_report(pd.read_parquet(RESID_PICK),
                                pd.read_parquet(RESID_RAW), args)
    return stage_residual(args)


# ---------------------------------------------------------------------------
# residual: rows the bracket does not fix, SOLVED at box C's convention
# ---------------------------------------------------------------------------
RESID_OUT = OUT / "E14_F3_residual_solves.parquet"
#: RAW solver output, written the moment the pool returns and BEFORE any
#: reshaping.  The first attempt post-processed in memory, the reshape raised,
#: and 120 minutes of solves were lost.  Solves are expensive and pandas is not:
#: persist first, analyse second, and let the analysis be re-runnable.
RESID_RAW = OUT / "E14_F3_residual_solves_raw.parquet"
RESID_PICK = OUT / "E14_F3_residual_sample.parquet"


def _solve_job(args):
    key, x, model_id, convention = args
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.campaign as C
    import ball_socket.mixed as mx
    model = mx.expand_for_solving(mx.registered_contact_models(
        branch="psi", summit_convention=convention))[model_id][0]
    row = C.evaluate_x(x, contact=model, clearance_closure=True,
                       delta_T_datum="bulk", Ntheta=GRID[0], Nphi=GRID[1])
    return {"key": key, "convention": convention,
            **{k: row.get(k) for k in ("outcome", "eps_ecc", "h_min_bnd",
                                       "T_max_bnd", "De_sq_bnd", "F_asp",
                                       "c_equilibrium", "realised_contact_model")}}


def stage_residual(args):
    """Solve a stratified sample of the residual at sigma_s_eq_sigma_star (box C's
    F3 arm) and at the derived arm, and compare feasibility labels."""
    from ball_socket.immutable import guarded_path
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M, ConstraintSet
    cs = ConstraintSet()
    b = pd.read_parquet(DEST)
    b = b[b.state == "equilibrium"].copy()
    M = pd.read_parquet(MIXED, columns=AXES + ["sample_id", "model_id",
                                               "T_max_bnd", "h_min_bnd", "De_sq_bnd"])
    M["model_id"] = M.model_id.astype(int)
    b = b.merge(M, on=["sample_id", "model_id"], suffixes=("", "_a"))
    lo, hi = (float(s) for s in SIGMA_STAR_BAND_M)
    resid = np.zeros(len(b), bool)
    for s in (lo, hi):
        for lim in cs.T_max_C_by_class.values():
            feas = ((b.h_min_bnd / s >= cs.lambda_ratio_min)
                    & (b.T_max_bnd <= lim) & (b.De_sq_bnd <= cs.De_sq_max))
            fixed = (b.h_min_upper / s) < cs.lambda_ratio_min * (1 - C_SPREAD)
            resid |= (~feas & ~fixed).to_numpy()
    r = b[resid].copy()
    # Stratify by how close the row sits to the g1 threshold at sigma_lo.
    # THE NAME MATTERS: `lam` is the relaxation-time INPUT AXIS.  Writing
    # Lambda into a column called `lam` overwrote it, and 570 rows were then
    # solved at the wrong fluid (caught 2026-09-20 by an input-vs-artefact
    # check).  Lambda is `Lambda_g1` here and nowhere near an axis name --
    # R-43.3's two-quantities-one-symbol hazard, in lambda rather than beta.
    r["Lambda_g1"] = r.h_min_bnd / lo
    r["stratum"] = pd.cut(r.Lambda_g1, [0, 1, 2, 2.9, 3.1, 10, np.inf], labels=False)
    pick = (r.groupby("stratum", group_keys=False)
             .apply(lambda g: g.sample(min(args.n_resid, len(g)), random_state=args.seed)))
    pick = pick.reset_index(drop=True)
    pick["key"] = np.arange(len(pick))
    print(f"residual rows: {len(r)}; solving {len(pick)} at BOTH conventions "
          f"({2 * len(pick)} solves)", flush=True)
    # GATE 5 (R-45.6): reserved names, then row provenance -- both BEFORE any
    # solve.  This is the gate the lambda collision earned.
    from ball_socket.verification import preflight as pf
    bad = pf.reserved_names_free(pick, derived=["Lambda_g1", "stratum", "key",
                                                "h_min_upper", "h_min_own",
                                                "eps_lower_bound"])
    bad += pf.row_provenance(pick, pd.read_parquet(
        MIXED, columns=AXES + ["sample_id", "model_id"]).astype({"model_id": int}),
        key=["sample_id", "model_id"])
    if bad:
        raise SystemExit(f"gate 5: {len(bad)} failure(s), first: {bad[0]}")
    print(f"gate 5: reserved names clear; inputs verified against the artefact "
          f"on all {len(pick)} rows", flush=True)
    jobs = []
    for conv in ("sigma_s_eq_sigma_star", "derived"):
        jobs += [(int(q.key), {k: float(getattr(q, k)) for k in AXES},
                  int(q.model_id), conv) for q in pick.itertuples()]
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        got = pd.DataFrame(list(ex.map(_solve_job, jobs, chunksize=4)))
    print(f"{len(got)} solves in {(time.time() - t0) / 60:.1f} min", flush=True)
    from ball_socket.immutable import guarded_write
    guarded_write(str(RESID_RAW), lambda p: got.to_parquet(p, index=False),
                  what="E14 F3 residual RAW solves", overwrite=args.overwrite)
    guarded_write(str(RESID_PICK), lambda p: pick.drop(
        columns=["dominating_arms"], errors="ignore").to_parquet(p, index=False),
        what="E14 F3 residual sample", overwrite=args.overwrite)
    print(f"raw solves written to {RESID_RAW.name}", flush=True)
    return _residual_report(pick, got, args)


def _residual_report(pick, got, args):
    """Reshape and score.  Re-runnable from the raw files, so a defect here
    never costs solves again."""
    import json as _json
    from ball_socket.immutable import guarded_path
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M, ConstraintSet
    cs = ConstraintSet()
    lo, hi = (float(s) for s in SIGMA_STAR_BAND_M)
    t0 = time.time()
    w = got.pivot(index="key", columns="convention")
    w.columns = [f"{a}|{b}" for a, b in w.columns]
    out = pick.merge(w.reset_index(), on="key")
    from ball_socket.immutable import guarded_write as _gw
    _gw(str(RESID_OUT), lambda p: out.to_parquet(p, index=False),
        what="E14 F3 residual solves", overwrite=getattr(args, "overwrite", False))
    rep = {"solved": int(len(pick)), "flips": {}}
    for s, name in ((lo, "lo"), (hi, "hi")):
        for cls, lim in cs.T_max_C_by_class.items():
            def feas(pre):
                return ((out[f"h_min_bnd|{pre}"] / s >= cs.lambda_ratio_min)
                        & (out[f"T_max_bnd|{pre}"] <= lim)
                        & (out[f"De_sq_bnd|{pre}"] <= cs.De_sq_max)
                        & (out[f"outcome|{pre}"] == "converged"))
            d, c_ = feas("derived"), feas("sigma_s_eq_sigma_star")
            rep["flips"][f"sigma_{name}|{cls}"] = {
                "n": int(len(out)), "derived_feasible": int(d.sum()),
                "boxC_feasible": int(c_.sum()),
                "flips_to_feasible": int((~d & c_).sum()),
                "flips_to_infeasible": int((d & ~c_).sum())}
    rep["h_min_ratio_boxC_over_derived"] = {
        "median": float((out["h_min_bnd|sigma_s_eq_sigma_star"]
                         / out["h_min_bnd|derived"]).median()),
        "max": float((out["h_min_bnd|sigma_s_eq_sigma_star"]
                      / out["h_min_bnd|derived"]).max())}
    path = RESID_OUT.with_suffix(".report.json")
    _gw(str(path), lambda p: pathlib.Path(p).write_text(
        _json.dumps(rep, indent=2, default=str)),
        what="E14 F3 residual report", overwrite=getattr(args, "overwrite", False))
    print(json.dumps(rep, indent=1, default=str))
    return 0


F25 = _fn_interp(2.5)
F20 = _fn_interp(2.0)

BOUND_OUT = OUT / "E14_F3_bound.json"
NET = OUT / "E14_net_decomposition.json"


def stage_bound(args):
    """F3's 95 % bound and the joint worst case (R-44.2, R-45.1), PERSISTED.

    Until 2026-09-21 these numbers existed only in a scratch script and in the
    hand-back's prose -- a quantity inside a formatted message is not saved.
    Rule of three per Lambda stratum (valid ONLY at zero observed flips, so the
    stage refuses otherwise), weighted by the residual's stratum sizes, over the
    valid-row denominator.  The joint worst case subtracts this bound from the
    net's own 95 % lower limit: its confidence is BELOW either component's.
    """
    import json as _json
    from ball_socket.immutable import guarded_write as _gw
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M, ConstraintSet
    cs = ConstraintSet()
    lo, hi = (float(s) for s in SIGMA_STAR_BAND_M)
    net = _json.loads(NET.read_text())["arms"]["mixed"]
    n_valid = {k: v["n_valid_rows"] for k, v in net.items()}
    if len(set(n_valid.values())) != 1:
        raise SystemExit(f"cells disagree on the valid-row denominator: {n_valid}")
    N_VALID = next(iter(n_valid.values()))
    flips = _json.loads(RESID_OUT.with_suffix(".report.json").read_text())["flips"]
    b = pd.read_parquet(DEST)
    b = b[b.state == "equilibrium"]
    M = pd.read_parquet(MIXED, columns=["sample_id", "model_id", "T_max_bnd",
                                        "h_min_bnd", "De_sq_bnd"])
    M["model_id"] = M.model_id.astype(int)
    eq = b.merge(M, on=["sample_id", "model_id"], suffixes=("", "_a"))
    eq["stratum"] = pd.cut(eq.h_min_bnd / lo, [0, 1, 2, 2.9, 3.1, 10, np.inf],
                           labels=False)
    ns = pd.read_parquet(RESID_PICK).stratum.value_counts()
    rep = {"method": ("rule of three per Lambda_g1 stratum (sigma_lo), weighted by "
                      "residual stratum sizes, over the valid-row denominator"),
           "n_valid": int(N_VALID),
           "sample_per_stratum": {int(k): int(v) for k, v in ns.sort_index().items()},
           "cells": {}}
    for s, name in ((lo, "lo"), (hi, "hi")):
        for cls, lim in cs.T_max_C_by_class.items():
            cell = f"sigma_{name}|{cls}"
            f = flips[cell]
            if f["flips_to_feasible"] or f["flips_to_infeasible"]:
                raise SystemExit(f"{cell}: flips observed; the rule of three does "
                                 f"not apply and a different bound is needed")
            feas = ((eq.h_min_bnd / s >= cs.lambda_ratio_min) & (eq.T_max_bnd <= lim)
                    & (eq.De_sq_bnd <= cs.De_sq_max))
            fixed = (eq.h_min_upper / s) < cs.lambda_ratio_min * (1 - C_SPREAD)
            resid = ~feas & ~fixed
            sizes = eq[resid].stratum.value_counts()
            bound = sum(sizes.get(k, 0) * (3.0 / ns[k]) for k in ns.index) / N_VALID
            ncell = net[cell]
            rep["cells"][cell] = {
                # "fixed" as the hand-back's table counts it: every equilibrium
                # row NOT in the residual (feasible, or infeasible and fixed).
                "fixed": int((~resid).sum()), "residual": int(resid.sum()),
                "residual_per_stratum": {int(k): int(v)
                                         for k, v in sizes.sort_index().items()},
                "F3_bound_pp": 100 * bound,
                "net_pp": 100 * ncell["net"],
                "net_ci_lower_pp": 100 * ncell["net_ci"][0],
                "joint_worst_case_pp": 100 * (ncell["net_ci"][0] - bound),
                "bound_over_net": bound / ncell["net"]}
    worst = min(rep["cells"], key=lambda c: rep["cells"][c]["joint_worst_case_pp"])
    rep["binding_cell"] = worst
    _gw(str(BOUND_OUT), lambda p: pathlib.Path(p).write_text(
        _json.dumps(rep, indent=2)), what="E14 F3 bound",
        overwrite=getattr(args, "overwrite", False))
    print(_json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
