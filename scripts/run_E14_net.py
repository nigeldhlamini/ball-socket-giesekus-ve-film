"""E14: the NET effect of box D's corrections on the feasible-set size,
decomposed per correction, per sigma* band end (box D sec 4.5 / E14).

REGISTERED: *the net effect is a REDUCTION at both sigma* band ends.  Four
corrections shrink the feasible set (h_min, De_sq, T_max, F3) and one enlarges it
(eps_cap).  Direction only is predicted; no magnitude is registered.*  Metric:
the signed change in feasible fraction, box C -> box D, decomposed per
correction, with a bootstrap interval, measured on IDENTICAL rows via the
`_legacy` columns -- never as a difference between two campaigns.

WHAT IS MEASURABLE BY RELABELLING, AND WHAT IS NOT.

* h_min, De_sq, T_max -- the boundary-extremum convention.  Both measurements
  are persisted per row (`h_min`/`h_min_bnd` etc.), so each correction is a
  relabel of the same solve.  Exact.
* eps_cap -- the bracket.  NOT a relabel: it changes which rows have an
  equilibrium at all.  It is RECONSTRUCTED per row from the realised
  eccentricity: brentq searches [eps_lo, eps_cap] on a residual that is
  monotone increasing in eps, so a row whose equilibrium sits at eps* > 0.99
  had no equilibrium under box C's bracket, and a row with eps* <= 0.99 had the
  same one.  The reconstruction is VERIFIED, not assumed (stage `verify`):
  a stratified sample is re-solved with the bracket pinned at 0.99 and the
  reconstructed verdict and feasibility label are compared.
* F3 (summit convention) -- NOT measurable either way.  It changes sigma_s,
  hence H, hence the asperity load, hence the EQUILIBRIUM; no stored column
  carries the box-C convention's solve.  Its term is reported as owed, with the
  solve count it needs, rather than estimated.

DECOMPOSITION.  One-at-a-time from the box-C baseline AND leave-one-out from the
box-D labelling are both reported, with the interaction residual, because the
corrections are not additive and a single ordering would hide that.

Denominator: VALID rows under the registered partition (R-32.4/R-37.4), the
denominator E7 uses.  The feasible fraction of CONVERGED rows is reported
alongside, because the two answer different questions.

Bootstrap: resampled over SAMPLES, not rows -- the 24 arms of a sample are not
independent draws (aggregation rule A1-A3).
"""
from __future__ import annotations

import argparse
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
MIXED = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
PURE = OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet"
DEST = OUT / "E14_net_decomposition.json"
VERIFY_OUT = OUT / "E14_bracket_reconstruction_check.parquet"
BOX_C_EPS_CAP = 0.99
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
GRID = (32, 24)
CLASSES = ("mineral", "water_glycol")
CORRECTIONS = ("eps_cap", "h_min", "De_sq", "T_max")


def _labels(df, *, use_bnd_h, use_bnd_T, use_bnd_De, bracket_box_c, sigma, cls):
    """Feasible mask under a named mixture of conventions, on identical rows."""
    from ball_socket.campaign_config_v4 import ConstraintSet
    cs = ConstraintSet()
    h = df.h_min_bnd if use_bnd_h else df.h_min
    T = df.T_max_bnd if use_bnd_T else df.T_max
    D = df.De_sq_bnd if use_bnd_De else df.De_sq
    exists = df.outcome == "converged"
    if bracket_box_c:                      # box C's bracket: no equilibrium above 0.99
        exists = exists & (df.eps_ecc <= BOX_C_EPS_CAP)
    return (exists & (h / sigma >= cs.lambda_ratio_min)
            & (T <= cs.T_max_C_by_class[cls]) & (D <= cs.De_sq_max))


def _state(**kw):
    """Convention state -> the _labels keywords for it."""
    return dict(use_bnd_h=kw["h_min"], use_bnd_T=kw["T_max"],
                use_bnd_De=kw["De_sq"], bracket_box_c=not kw["eps_cap"])


#: The mixed artefact predates the typed `nonconvergence_kind` column, which
#: lives in a DERIVED file (R-37.4).  Reading it with df.get() returns None and
#: silently counts the six picard_at_root rows as valid -- the denominator then
#: reads 196,608 instead of 196,602.  Caught 2026-09-20 by the two arms
#: returning identical feasible fractions, which sent me looking.
KIND = OUT / "derived_boxE_mixed_32x24_corrected_nonconvergence_kind.parquet"


#: Arms whose `nonconverged` rows carry no typed kind, with the convention the
#: CALLER names for them -- the same contract campaign.summarise() enforces
#: through `legacy_nonconverged_kind`.  ERR-16 establishes it by re-solve: every
#: nonconverged row in box B, box D and the corrected pure-film arm is a genuine
#: capacity blow-up, and the picard_at_root kind occurs only in the mixed arm.
LEGACY_KIND = {"pure_film": "blowup_capacity"}


def _invalid(df, arm):
    """The registered validity partition (R-32.4, extended R-37.4).  Refuses an
    untyped `nonconverged` row unless the convention for that arm is NAMED."""
    bad = df.outcome.isin(["clearance_nonconverged", "error"])
    nc = df.outcome == "nonconverged"
    if not nc.any():
        return bad
    if arm != "mixed":
        named = LEGACY_KIND.get(arm)
        if named is None:
            raise SystemExit(f"{arm}: {int(nc.sum())} nonconverged rows and no "
                             f"kind source or named convention for this arm")
        if named.startswith("blowup"):      # valid under the partition
            return bad
        return bad | nc
    k = pd.read_parquet(KIND)
    key = ["sample_id", "model_id"]
    kk = df[key].merge(k[key + ["nonconvergence_kind"]], on=key, how="left")
    kind = kk.nonconvergence_kind.to_numpy()
    untyped = nc.to_numpy() & pd.isna(kind)
    if untyped.any():
        raise SystemExit(f"{int(untyped.sum())} nonconverged rows carry no kind; "
                         f"the partition will not guess one")
    return bad | (nc.to_numpy() & (kind == "picard_at_root"))


def _frac(mask, valid):
    return float(mask.sum()) / float(valid.sum())


def stage_net(args):
    from ball_socket.immutable import guarded_path
    if not args.overwrite:
        guarded_path(str(DEST), what="E14 net decomposition")
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M
    rng = np.random.default_rng(args.seed)
    frames = {"mixed": pd.read_parquet(MIXED), "pure_film": pd.read_parquet(PURE)}
    rep = {"registered": ("net effect of box D's corrections on the feasible-set "
                          "size is a REDUCTION at both sigma* band ends; direction "
                          "only, no magnitude registered"),
           "denominator": "valid rows (registered partition)",
           "kind_source": {"mixed": "derived nonconvergence_kind file (R-37.4)",
                           "pure_film": "convention NAMED by the caller: "
                                        "blowup_capacity, established by ERR-16"},
           "bootstrap": {"resampled_over": "samples", "B": args.boot,
                         "seed": args.seed, "interval": "percentile 2.5/97.5"},
           "F3_term": {"status": "NOT MEASURABLE BY RELABELLING",
                       "why": ("F3 changes sigma_s, hence H, hence the asperity "
                               "load, hence the equilibrium.  No stored column "
                               "carries the box-C summit convention's solve."),
                       "what_it_would_cost": ("one full campaign arm at the box-C "
                                              "summit convention: 8,192 samples x "
                                              "24 arms = 196,608 solves, or a "
                                              "stratified subsample.  Not run."),
                       "direction_registered": "shrinks the feasible set"},
           "arms": {}}
    for arm, df in frames.items():
        valid = ~_invalid(df, arm)
        d = df[valid].copy()
        sid = d.sample_id.to_numpy()
        samples = np.unique(sid)
        idx_by_sample = {s: np.flatnonzero(sid == s) for s in samples}
        out = {}
        for end, sigma in zip(("lo", "hi"), (float(s) for s in SIGMA_STAR_BAND_M)):
            for cls in CLASSES:
                boxC = dict(eps_cap=False, h_min=False, T_max=False, De_sq=False)
                boxD = dict(eps_cap=True, h_min=True, T_max=True, De_sq=True)
                masks = {}
                def mask(state):
                    key = tuple(sorted(state.items()))
                    if key not in masks:
                        masks[key] = _labels(d, sigma=sigma, cls=cls,
                                             **_state(**state)).to_numpy()
                    return masks[key]
                base, full = mask(boxC), mask(boxD)
                terms = {}
                for c in CORRECTIONS:
                    on = dict(boxC); on[c] = True          # one at a time
                    off = dict(boxD); off[c] = False       # leave one out
                    terms[c] = {"one_at_a_time": mask(on), "leave_one_out": full}
                    terms[c]["loo_other"] = mask(off)
                n = len(d)
                res = {"n_valid_rows": int(n),
                       "n_samples": int(len(samples)),
                       "feasible_frac_boxC": _frac(base, np.ones(n, bool)),
                       "feasible_frac_boxD": _frac(full, np.ones(n, bool)),
                       "net": float(full.mean() - base.mean()),
                       "terms": {}}
                for c in CORRECTIONS:
                    oaat = float(terms[c]["one_at_a_time"].mean() - base.mean())
                    loo = float(full.mean() - terms[c]["loo_other"].mean())
                    res["terms"][c] = {"one_at_a_time": oaat, "leave_one_out": loo}
                res["interaction_residual"] = float(
                    res["net"] - sum(v["one_at_a_time"] for v in res["terms"].values()))
                # bootstrap over samples
                boot_net, boot_terms = [], {c: [] for c in CORRECTIONS}
                for _ in range(args.boot):
                    pick = rng.choice(samples, size=len(samples), replace=True)
                    rows = np.concatenate([idx_by_sample[s] for s in pick])
                    b, f = base[rows], full[rows]
                    boot_net.append(f.mean() - b.mean())
                    for c in CORRECTIONS:
                        boot_terms[c].append(
                            terms[c]["one_at_a_time"][rows].mean() - b.mean())
                q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                res["net_ci"] = q(boot_net)
                for c in CORRECTIONS:
                    res["terms"][c]["ci_one_at_a_time"] = q(boot_terms[c])
                conv = (d.outcome == "converged").to_numpy()
                res["feasible_frac_of_converged_boxD"] = float(full.sum() / conv.sum())
                out[f"sigma_{end}|{cls}"] = res
        rep["arms"][arm] = out
    from ball_socket.immutable import guarded_write
    guarded_write(str(DEST), lambda p: pathlib.Path(p).write_text(
        json.dumps(rep, indent=2, default=float)),
        what="E14 net decomposition", overwrite=args.overwrite)
    print(json.dumps(rep, indent=1, default=float)[:4000])
    return 0


# ---------------------------------------------------------------------------
# verification of the bracket reconstruction (R-24.5 / R-40.2 discipline)
# ---------------------------------------------------------------------------
def _job(args):
    key, x, model_id = args
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.campaign as C
    import ball_socket.mixed as mx
    C._registered_eps_cap = lambda theta_min: BOX_C_EPS_CAP   # box C's bracket
    model = None
    if model_id >= 0:
        model = mx.expand_for_solving(mx.registered_contact_models(
            branch="psi", summit_convention="derived"))[model_id][0]
    row = C.evaluate_x(x, contact=model, clearance_closure=True, delta_T_datum="bulk",
                       Ntheta=GRID[0], Nphi=GRID[1])
    return {"key": key, "model_id": model_id,
            **{k: row.get(k) for k in ("outcome", "realised_eps_cap", "eps_ecc",
                                       "h_min", "h_min_bnd", "T_max", "T_max_bnd",
                                       "De_sq", "De_sq_bnd")}}


def stage_verify(args):
    from ball_socket.immutable import guarded_path
    guarded_path(str(VERIFY_OUT), what="E14 bracket reconstruction check")
    rng = np.random.default_rng(args.seed)
    picks = []
    for arm, path in (("mixed", MIXED), ("pure_film", PURE)):
        d = pd.read_parquet(path)
        d = d[d.outcome == "converged"].copy()
        d["above"] = d.eps_ecc > BOX_C_EPS_CAP
        d["arm"] = arm
        if "model_id" not in d:
            d["model_id"] = -1
        for above, g in d.groupby("above"):
            picks.append(g.sample(min(args.n_verify, len(g)), random_state=args.seed))
    pick = pd.concat(picks).reset_index(drop=True)
    pick["key"] = np.arange(len(pick))
    jobs = [(int(r.key), {k: float(getattr(r, k)) for k in AXES}, int(r.model_id))
            for r in pick.itertuples()]
    print(f"verifying the bracket reconstruction on {len(jobs)} rows "
          f"({int(pick.above.sum())} with eps* > {BOX_C_EPS_CAP})", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        got = pd.DataFrame(list(ex.map(_job, jobs, chunksize=4)))
    m = pick.merge(got.drop(columns="model_id"), on="key", suffixes=("", "_099"))
    m["reconstructed_exists"] = ~m.above
    m["actual_exists"] = m.outcome_099 == "converged"
    m["agrees"] = m.reconstructed_exists == m.actual_exists
    m.to_parquet(VERIFY_OUT, index=False)
    rep = {"n": int(len(m)), "agree": int(m.agrees.sum()),
           "wall_min": round((time.time() - t0) / 60, 1),
           "by_stratum": {f"{a}|eps*>{BOX_C_EPS_CAP}={b}":
                          {"n": int(len(g)), "agree": int(g.agrees.sum()),
                           "outcomes": g.outcome_099.value_counts().to_dict()}
                          for (a, b), g in m.groupby(["arm", "above"])},
           "realised_eps_cap": sorted(m.realised_eps_cap_099.unique().tolist())}
    rep["PASS"] = rep["agree"] == rep["n"]
    path = VERIFY_OUT.with_suffix(".report.json")
    guarded_path(str(path), what="E14 reconstruction report")
    path.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["PASS"] else 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["verify", "net"])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-verify", type=int, default=60)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace a previous run's JSON; the shared writer keeps "
                         "the old file with a timestamp")
    args = ap.parse_args()
    return (stage_verify if args.stage == "verify" else stage_net)(args)


if __name__ == "__main__":
    raise SystemExit(main())
