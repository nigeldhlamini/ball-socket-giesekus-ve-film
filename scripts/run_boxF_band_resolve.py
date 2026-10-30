"""Box F sec 5.2 under R-40: the marginal band, re-solved in two halves.

R-40 splits the unmasked marginal rows by outcome class.

FAILURE SIDE (mixed contact_limit, pure-film blow-up) -- stage ``cap``.
    The registered solve's FIRST act at a clearance is the cap residual

        r_hi = F_film(eps_cap) + F_asp(eps_cap) - F_target

    and it raises (contact_limit / blowup_capacity) iff r_hi < 0.  A row whose
    smooth failure happened on the FIRST outer iteration (c = c0) therefore
    keeps its verdict under the averaged film iff the averaged r_hi at c0 is
    still negative: that is the solver's own control flow, evaluated with the
    solver's own function (BallSocketSolver._load_split).  Rows whose smooth
    failure happened LATER (c != c0; detected by the smooth cap load at c0 not
    reproducing the stored one) are NOT decided here -- the averaged run's
    clearance path differs -- and are routed to the full procedure.
    Stage ``verify_cap`` runs the full registered procedure on a stratified
    sample of rows the cap stage calls unchanged (R-24.5 / R-40.2).

CONVERGED SIDE -- stage ``mono``.
    R-40.3's claim, TESTED not assumed: with the averaged film carrying more
    (kappa >= 1) eps* falls, h_min rises, De_sq falls and T_max falls.  Full
    registered procedure, both treatments, on a stratified sample of unmasked
    marginal converged rows -- plus a NON-marginal converged stratum, because
    the registered marginal criterion is a cap (existence) test and says
    nothing about whether a converged row's constraint labels move.

FLIPS -- stage ``full``.
    Full registered procedure on every row a stage has not decided exactly.

Every stage writes through the shared guard and refuses to overwrite.
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
PURE = OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet"
MIXED = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
COUNT = OUT / "boxF_marginal_band_count.parquet"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
TREATMENTS = ("extrapolate", "freeze")
GRID = (32, 24)
CLASSES = ("mineral", "water_glycol")

CAP_OUT = OUT / "boxF_band_cap_stage.parquet"
VERIFY_OUT = OUT / "boxF_band_cap_verification.parquet"
MONO_OUT = OUT / "boxF_band_monotonicity_test.parquet"
FULL_OUT = OUT / "boxF_band_full_resolve.parquet"


def _models():
    import ball_socket.mixed as mx
    return mx.expand_for_solving(
        mx.registered_contact_models(branch="psi", summit_convention="derived"))


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------
def cap_unit(args):
    """One sample at c0: the smooth cap film (reproduction) and the averaged cap
    film for each requested sigma* under both treatments -- all through
    BallSocketSolver._load_split, the function the registered solve calls."""
    sid, x, sigmas = args
    import warnings
    warnings.simplefilter("ignore")
    from run_E7_load_decomposition import _solver
    slv, ci = _solver(x, *GRID)
    fluid, c0, cap = slv._fluid(ci), float(ci.c0), float(slv.eps_cap)
    out = {"sample_id": int(sid), "eps_cap": cap,
           "F_target_c0": float(slv._applied_load(ci) - slv._pocket_relief(ci))}
    slv.flow_factors = None
    out["F_smooth_c0"] = float(slv._load_split(ci, cap, fluid, c0)[0])
    for s in sigmas:
        for t in TREATMENTS:
            slv.flow_factors = {"sigma": float(s), "below_floor": t}
            F, _, sph = slv._load_split(ci, cap, fluid, c0)
            out[f"F_avg|{s!r}|{t}"] = float(F)
    return out


def full_job(args):
    """The full registered procedure (evaluate_x, closure on, 32x24) with the
    averaged film on, at the row's sigma* end, under one treatment."""
    key, x, model_id, sigma, treatment = args
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.campaign as C
    model = None if model_id < 0 else _models()[model_id][0]
    ff = {"sigma": float(sigma), "below_floor": treatment}
    t0 = time.time()
    row = C.evaluate_x(x, contact=model, clearance_closure=True,
                       delta_T_datum="bulk", Ntheta=GRID[0], Nphi=GRID[1],
                       flow_factors=ff)
    keep = ["outcome", "realised_film_model", "realised_contact_model",
            "realised_eps_cap", "realised_Ntheta", "nonconvergence_kind",
            "eps_ecc", "h_min_bnd", "De_sq_bnd", "T_max_bnd", "T_bulk",
            "c_equilibrium", "n_outer", "F_hydrodynamic", "F_asp", "P_loss",
            "F_film_at_limit", "F_asp_at_limit", "F_film_max",
            "F_target_at_failure", "c_at_failure", "n_outer_at_failure",
            "eps_at_failure", "fail_reason"]
    rec = {k: row.get(k) for k in keep}
    rec.update(key=key, model_id=int(model_id), sigma=float(sigma),
               treatment=treatment, wall_s=time.time() - t0)
    return rec


def kappa_at_eq(args):
    """Averaged / smooth film load at the SMOOTH equilibrium (eps*, c_eq): the
    kappa that decides which way the equilibrium moves (R-40.3)."""
    key, x, sigma, eps, c = args
    import warnings
    warnings.simplefilter("ignore")
    from run_E7_load_decomposition import _solver
    slv, ci = _solver(x, *GRID)
    fluid = slv._fluid(ci)
    slv.flow_factors = None
    F0 = float(slv._load_split(ci, eps, fluid, c)[0])
    out = {"key": key, "F_smooth_eq": F0}
    for t in TREATMENTS:
        slv.flow_factors = {"sigma": float(sigma), "below_floor": t}
        out[f"kappa_eq_{t}"] = float(slv._load_split(ci, eps, fluid, c)[0]) / F0
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
                print(f"  {label}: {i}/{len(jobs)}  {el / 60:.1f} min elapsed, "
                      f"~{el / i * (len(jobs) - i) / 60:.1f} min left", flush=True)
    return res


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def _band():
    """Unmasked marginal rows from the registered count, joined to the stored
    failure / equilibrium quantities they need."""
    d = pd.read_parquet(COUNT)
    d = d[d.unmasked_marginal].copy()
    mxd = pd.read_parquet(MIXED)
    mxd["model_id"] = mxd.model_id.astype(int)
    pf = pd.read_parquet(PURE)
    return d, mxd, pf


def _x(r):
    return {k: float(getattr(r, k)) for k in AXES}


# ---------------------------------------------------------------------------
# stage cap
# ---------------------------------------------------------------------------
def stage_cap(args):
    from ball_socket.immutable import guarded_path
    guarded_path(str(CAP_OUT), what="R-40.2 cap stage")
    import ball_socket.campaign_config_v8 as v8
    k_lo, k_hi = v8.MARGINAL_BAND["kappa"]["lo"], v8.MARGINAL_BAND["kappa"]["hi"]
    d, mxd, pf = _band()
    fail = d[d.outcome.isin(["contact_limit", "nonconverged"])].copy()
    print(f"failure-side unmasked marginal: {len(fail)} rows "
          f"({(fail.arm_of_campaign == 'mixed').sum()} mixed contact_limit, "
          f"{(fail.arm_of_campaign == 'pure_film').sum()} pure-film sigma*-rows)")
    # stored per-row failure quantities
    st = mxd.set_index(["sample_id", "model_id"])[
        ["F_film_at_limit", "F_asp_at_limit", "F_target_at_failure"]]
    pst = pf.set_index("sample_id")[["F_film_max", "F_hydrodynamic"]]
    xs = pf.set_index("sample_id")[AXES]
    units = fail.groupby("sample_id").sigma.agg(lambda s: sorted(set(s)))
    jobs = [(int(sid), {k: float(v) for k, v in xs.loc[sid].items()}, sig)
            for sid, sig in units.items()]
    print(f"{len(jobs)} samples, {int(units.map(len).sum())} (sample, sigma*) film "
          f"units, {len(jobs) + 2 * int(units.map(len).sum())} cap solves")
    res = pd.DataFrame(_pool(cap_unit, jobs, args.workers, "cap")).set_index("sample_id")

    rows = []
    for r in fail.itertuples():
        u = res.loc[r.sample_id]
        if r.arm_of_campaign == "mixed":
            s = st.loc[(r.sample_id, r.model_id)]
            F_sm_st, F_asp, F_t = s.F_film_at_limit, s.F_asp_at_limit, s.F_target_at_failure
        else:
            s = pst.loc[r.sample_id]
            F_sm_st, F_asp, F_t = s.F_film_max, 0.0, s.F_hydrodynamic
        # The pure-film artefact stores F_film_max as (r_hi + F_target), which
        # is not bitwise F_film; 1e-12 relative separates "same clearance" from
        # "a later outer iteration" by ~9 orders of magnitude (R-36.4: the
        # non-reproducing rows differ by 1e-3 or more).
        reproduces = bool(abs(u.F_smooth_c0 / F_sm_st - 1) <= 1e-12
                          and u.F_target_c0 == F_t)
        for t in TREATMENTS:
            F_avg = u[f"F_avg|{r.sigma!r}|{t}"]
            r_hi = (F_avg + F_asp) - F_t          # the solver's own arithmetic
            if not reproduces:
                decision = "route_full:failed_after_c0"
            elif r_hi < 0.0:
                decision = "unchanged"
            else:
                decision = "flips:full_solve"
            rows.append({"arm_of_campaign": r.arm_of_campaign,
                         "sample_id": int(r.sample_id), "model_id": int(r.model_id),
                         "sigma": float(r.sigma), "sigma_end": r.sigma_end,
                         "outcome_smooth": r.outcome, "treatment": t,
                         "H_rim": float(r.H_rim), "F_smooth_stored": float(F_sm_st),
                         "F_smooth_c0": float(u.F_smooth_c0), "reproduces": reproduces,
                         "F_avg_cap": float(F_avg), "F_asp_cap": float(F_asp),
                         "F_target": float(F_t),
                         "kappa_cap": float(F_avg / F_sm_st),
                         "r_hi_avg": float(r_hi),
                         "r_hi_rel": float(r_hi / F_t), "decision": decision,
                         **{k: float(getattr(r, k)) for k in AXES}})
    cap = pd.DataFrame(rows)
    cap["kappa_in_band"] = cap.kappa_cap.between(k_lo, k_hi)
    cap.to_parquet(CAP_OUT, index=False)

    return _cap_report(cap)


def _cap_report(cap):
    """Report for the cap stage, built from the written parquet alone."""
    from ball_socket.immutable import guarded_path
    path = CAP_OUT.with_suffix(".report.json")
    guarded_path(str(path), what="R-40.2 cap report")
    rid = ["arm_of_campaign", "sample_id", "model_id", "sigma"]
    dc = (cap.groupby(["arm_of_campaign", "treatment"]).decision.value_counts()
             .unstack(fill_value=0))
    either = cap.groupby(rid).decision.agg(
        lambda s: "unchanged" if (s == "unchanged").all()
        else ("route_full:failed_after_c0" if s.str.startswith("route").any()
              else "flips_under_" + "+".join(sorted(
                  cap.loc[s.index][s != "unchanged"].treatment))))
    rep = {"rows": int(cap.groupby(rid).ngroups), "row_treatments": int(len(cap)),
           "samples": int(cap.sample_id.nunique()),
           "film_units": int(cap.groupby(["sample_id", "sigma"]).ngroups),
           "samples_not_reproducing_at_c0": int(
               cap.loc[~cap.reproduces, "sample_id"].nunique()),
           "rows_not_reproducing_at_c0": int(cap.loc[~cap.reproduces].groupby(rid).ngroups),
           "max_rel_reproduction_error_reproducing": float(
               (cap.loc[cap.reproduces, "F_smooth_c0"]
                / cap.loc[cap.reproduces, "F_smooth_stored"] - 1).abs().max()),
           "decision_counts": {f"{a}|{t}": {k: int(v) for k, v in r.items()}
                               for (a, t), r in dc.iterrows()},
           "row_decision_over_both_treatments": {
               f"{a}|{k}": int(v) for (a, k), v in
               either.groupby(level="arm_of_campaign").value_counts().items()},
           "kappa_cap_range": [float(cap.kappa_cap.min()), float(cap.kappa_cap.max())],
           "kappa_outside_band_rows": int((~cap.kappa_in_band).sum()),
           "min_abs_r_hi_rel_unchanged": float(
               cap.loc[cap.decision == "unchanged", "r_hi_rel"].abs().min())}
    rep["STOP_kappa"] = rep["kappa_outside_band_rows"] > 0
    path.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 4 if rep["STOP_kappa"] else 0


# ---------------------------------------------------------------------------
# stage verify_cap
# ---------------------------------------------------------------------------
def stage_verify_cap(args):
    from ball_socket.immutable import guarded_path
    guarded_path(str(VERIFY_OUT), what="R-40.2 verification")
    cap = pd.read_parquet(CAP_OUT)
    un = cap[cap.decision == "unchanged"].copy()
    un["Hq"] = pd.qcut(un.H_rim, 4, labels=False, duplicates="drop")
    pick = (un.groupby(["arm_of_campaign", "treatment", "Hq"], group_keys=False)
              .apply(lambda g: g.sample(min(args.n_verify, len(g)),
                                        random_state=args.seed)))
    jobs = [((int(r.sample_id), int(r.model_id), float(r.sigma), r.treatment),
             _x(r), int(r.model_id), float(r.sigma), r.treatment)
            for r in pick.itertuples()]
    print(f"verifying {len(jobs)} cap-unchanged rows by the full registered procedure")
    got = pd.DataFrame(_pool(full_job, jobs, args.workers, "verify"))
    got[["sample_id", "model_id", "sigma", "treatment"]] = pd.DataFrame(
        got.key.tolist(), index=got.index)
    m = pick.merge(got.drop(columns="key"),
                   on=["sample_id", "model_id", "sigma", "treatment"])
    want = np.where(m.arm_of_campaign == "mixed", "contact_limit", "nonconverged")
    m["verdict_agrees"] = m.outcome.to_numpy() == want
    F_full = np.where(m.arm_of_campaign == "mixed", m.F_film_at_limit, m.F_film_max)
    m["F_full"] = F_full.astype(float)
    m["film_bitwise"] = m.F_full.to_numpy() == m.F_avg_cap.to_numpy()
    m["at_c0"] = m.c_at_failure.astype(float) == m.c0.astype(float)
    m.to_parquet(VERIFY_OUT, index=False)
    return _verify_report(m)


def _verify_report(m):
    from ball_socket.immutable import guarded_path
    path = VERIFY_OUT.with_suffix(".report.json")
    guarded_path(str(path), what="R-40.2 verification report")
    rep = {"n": int(len(m)),
           "verdict_agrees": int(m.verdict_agrees.sum()),
           "film_load_bitwise": int(m.film_bitwise.sum()),
           "failed_at_c0": int(m.at_c0.sum()),
           "n_outer_at_failure": m.n_outer_at_failure.value_counts(dropna=False)
                                  .to_dict(),
           "max_rel_film_diff": float((m.F_full / m.F_avg_cap - 1).abs().max()),
           "by_stratum": {f"{a}|{t}": {"n": int(g.size), "agree": int(g.sum())}
                          for (a, t), g in m.groupby(["arm_of_campaign", "treatment"])
                          .verdict_agrees}}
    rep["PASS"] = rep["verdict_agrees"] == rep["n"]
    path.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["PASS"] else 5


# ---------------------------------------------------------------------------
# stage mono
# ---------------------------------------------------------------------------
def _labels_for(h, T, D, sigma):
    import ball_socket.campaign as C
    return {c: C._labels(h_min=h, T_max=T, De_sq=D, sigma_star=sigma,
                         fluid_class=c)[0] for c in CLASSES}


def stage_mono(args):
    from ball_socket.immutable import guarded_path
    guarded_path(str(MONO_OUT), what="R-40.3 monotonicity test")
    d_all = pd.read_parquet(COUNT)
    conv = d_all[d_all.outcome == "converged"].copy()
    rng = np.random.default_rng(args.seed)
    conv["stratum"] = np.where(conv.unmasked_marginal, "marginal_unmasked",
                               np.where(conv.marginal, "marginal_masked",
                                        "non_marginal"))
    conv = conv[conv.stratum != "marginal_masked"]
    conv["Hq"] = pd.qcut(conv.H_rim, 3, labels=False, duplicates="drop")
    n_for = {"marginal_unmasked": args.n_mono, "non_marginal": args.n_mono_nonmarg}
    pick = (conv.groupby(["stratum", "arm_of_campaign", "Hq"], group_keys=False)
                .apply(lambda g: g.sample(min(n_for[g.name[0]], len(g)),
                                          random_state=args.seed)))
    mxd = pd.read_parquet(MIXED).set_index(["sample_id", "model_id"])
    pf = pd.read_parquet(PURE).set_index("sample_id")
    base = []
    for r in pick.itertuples():
        s = (mxd.loc[(r.sample_id, r.model_id)] if r.arm_of_campaign == "mixed"
             else pf.loc[r.sample_id])
        base.append({"eps_s": float(s.eps_ecc), "h_s": float(s.h_min_bnd),
                     "De_s": float(s.De_sq_bnd), "T_s": float(s.T_max_bnd),
                     "c_s": float(s.c_equilibrium), "P_s": float(s.P_loss)})
    pick = pd.concat([pick.reset_index(drop=True), pd.DataFrame(base)], axis=1)
    pick["key"] = [f"{a}|{s}|{m}|{g!r}" for a, s, m, g in
                   zip(pick.arm_of_campaign, pick.sample_id, pick.model_id, pick.sigma)]
    print(f"monotonicity test: {len(pick)} converged rows "
          f"({pick.stratum.value_counts().to_dict()}), x2 treatments")
    kjobs = [(r.key, _x(r), r.sigma, r.eps_s, r.c_s) for r in pick.itertuples()]
    kap = pd.DataFrame(_pool(kappa_at_eq, kjobs, args.workers, "kappa_eq"))
    fjobs = [(r.key, _x(r), int(r.model_id), float(r.sigma), t)
             for r in pick.itertuples() for t in TREATMENTS]
    got = pd.DataFrame(_pool(full_job, fjobs, args.workers, "mono"))
    m = (got.merge(pick, on="key", suffixes=("", "_row"))
            .merge(kap, on="key"))
    m = m.drop(columns=[c for c in m.columns if c.endswith("_row")])
    m["kappa_eq"] = [row[f"kappa_eq_{row.treatment}"] for _, row in m.iterrows()]
    ok = m.outcome == "converged"
    for a, b, n in (("eps_ecc", "eps_s", "d_eps"), ("h_min_bnd", "h_s", "d_h"),
                    ("De_sq_bnd", "De_s", "d_De"), ("T_max_bnd", "T_s", "d_T"),
                    ("c_equilibrium", "c_s", "d_c"), ("P_loss", "P_s", "d_P")):
        m[n] = np.where(ok, m[a].astype(float) - m[b], np.nan)
    lab_s, lab_a = [], []
    for row in m.itertuples():
        lab_s.append(_labels_for(row.h_s, row.T_s, row.De_s, row.sigma))
        lab_a.append(_labels_for(row.h_min_bnd, row.T_max_bnd, row.De_sq_bnd, row.sigma)
                     if row.outcome == "converged" else {c: row.outcome for c in CLASSES})
    for c in CLASSES:
        m[f"label_smooth_{c}"] = [l[c] for l in lab_s]
        m[f"label_avg_{c}"] = [l[c] for l in lab_a]
    m.to_parquet(MONO_OUT, index=False)
    return _mono_report(m)


def _mono_report(m):
    from ball_socket.immutable import guarded_path
    path = MONO_OUT.with_suffix(".report.json")
    guarded_path(str(path), what="R-40.3 monotonicity report")

    def legs(g):
        g = g[g.outcome == "converged"]
        up = g.kappa_eq >= 1.0
        out = {"n_converged": int(len(g)), "n_kappa_ge_1": int(up.sum()),
               "n_kappa_lt_1": int((~up).sum())}
        for n, want in (("d_eps", -1), ("d_h", +1), ("d_De", -1), ("d_T", -1),
                        ("d_P", -1)):
            v = g.loc[up, n]
            out[f"{n}_wrong_sign_given_kappa_ge_1"] = int((np.sign(v) == -want).sum())
            out[f"{n}_zero_given_kappa_ge_1"] = int((v == 0).sum())
            w = g.loc[~up, n]
            out[f"{n}_expected_sign_given_kappa_lt_1"] = int((np.sign(w) == -want).sum())
        return out
    rep = {"n_rows": int(m.groupby(["arm_of_campaign", "sample_id", "model_id",
                                    "sigma"]).ngroups), "n_solves": int(len(m)),
           "outcomes": {f"{a}|{t}": {k: int(v) for k, v in g.value_counts().items()}
                        for (a, t), g in m.groupby(["stratum", "treatment"]).outcome},
           "kappa_eq_range": [float(m.kappa_eq.min()), float(m.kappa_eq.max())],
           "n_kappa_eq_lt_1": int((m.kappa_eq < 1).sum()),
           "legs": {f"{s}|{t}": legs(g) for (s, t), g in m.groupby(["stratum", "treatment"])},
           "label_changes": {}}
    for c in CLASSES:
        ch = m[m[f"label_smooth_{c}"] != m[f"label_avg_{c}"]]
        rep["label_changes"][c] = (ch.groupby(["stratum", "treatment",
                                               f"label_smooth_{c}", f"label_avg_{c}"])
                                     .size().rename("n").reset_index()
                                     .to_dict(orient="records"))
    path.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


# ---------------------------------------------------------------------------
# stage full
# ---------------------------------------------------------------------------
def kappa_cap_job(args):
    """Averaged / smooth film at the cap, at the clearance the smooth row
    reached -- box F sec 5.2's STOP quantity for a re-solved converged row."""
    key, x, sigma, c, eps = args
    import warnings
    warnings.simplefilter("ignore")
    from run_E7_load_decomposition import _solver
    slv, ci = _solver(x, *GRID)
    fluid, cap = slv._fluid(ci), float(slv.eps_cap)
    out = {"key": key}
    for where, e in (("cap", cap), ("eq", eps)):
        slv.flow_factors = None
        F0 = float(slv._load_split(ci, e, fluid, c)[0])
        for t in TREATMENTS:
            slv.flow_factors = {"sigma": float(sigma), "below_floor": t}
            out[f"kappa_{where}_{t}"] = float(slv._load_split(ci, e, fluid, c)[0]) / F0
    return out


def stage_full(args):
    """Full registered procedure on every row not decided exactly.

    Failure side: cap-stage 'flips' and 'route_full' rows.  Converged side: per
    --converged, which R-40.3's test decides (kappa_lt_1 | all | none).
    Checkpointed per chunk so a multi-hour run resumes rather than restarts."""
    from ball_socket.immutable import guarded_path
    guarded_path(str(FULL_OUT), what="R-40 full re-solve")
    import ball_socket.campaign_config_v8 as v8
    k_lo, k_hi = v8.MARGINAL_BAND["kappa"]["lo"], v8.MARGINAL_BAND["kappa"]["hi"]
    cap = pd.read_parquet(CAP_OUT)
    todo = cap[cap.decision != "unchanged"][
        ["arm_of_campaign", "sample_id", "model_id", "sigma", "treatment",
         "decision", "H_rim", *AXES]].copy()
    todo["source"] = "failure_side"
    kap = None
    if args.converged != "none":
        d = pd.read_parquet(COUNT)
        cv = d[d.unmasked_marginal & (d.outcome == "converged")].copy()
        mxd = pd.read_parquet(MIXED).set_index(["sample_id", "model_id"])
        pf = pd.read_parquet(PURE).set_index("sample_id")
        src = [(mxd.loc[(r.sample_id, r.model_id)] if r.arm_of_campaign == "mixed"
                else pf.loc[r.sample_id]) for r in cv.itertuples()]
        cv["c_s"] = [float(z.c_equilibrium) for z in src]
        cv["eps_s"] = [float(z.eps_ecc) for z in src]
        cv["key"] = [f"{a}|{s_}|{m}|{g!r}" for a, s_, m, g in
                     zip(cv.arm_of_campaign, cv.sample_id, cv.model_id, cv.sigma)]
        kap = pd.DataFrame(_pool(kappa_cap_job,
                                 [(r.key, _x(r), r.sigma, r.c_s, r.eps_s)
                                  for r in cv.itertuples()],
                                 args.workers, "kappa_cap(converged)"))
        cv = cv.merge(kap, on="key")
        bad = cv[~(cv.kappa_cap_extrapolate.between(k_lo, k_hi)
                   & cv.kappa_cap_freeze.between(k_lo, k_hi))]
        if len(bad):
            print(f"STOP (box F sec 5.2): {len(bad)} converged band rows have a cap "
                  f"ratio outside [{k_lo}, {k_hi}]")
            bad.to_parquet(FULL_OUT.with_name("boxF_band_kappa_STOP.parquet"), index=False)
            return 4
        rows = []
        for t in TREATMENTS:
            sel = cv if args.converged == "all" else cv[cv[f"kappa_eq_{t}"] < 1.0]
            z = sel.assign(treatment=t, decision="converged_side", source="converged_side")
            rows.append(z[[c for c in todo.columns if c in z.columns]])
        todo = pd.concat([todo, *rows], ignore_index=True)
    todo = todo.reset_index(drop=True)
    todo["job"] = np.arange(len(todo))
    print(f"full re-solve: {len(todo)} solves "
          f"({todo.groupby(['source', 'arm_of_campaign']).size().to_dict()})")

    ck = FULL_OUT.with_name(FULL_OUT.stem + "_ckpt")
    ck.mkdir(exist_ok=True)
    done = set()
    for f in ck.glob("chunk_*.parquet"):
        done |= set(pd.read_parquet(f, columns=["job"]).job)
    rest = todo[~todo.job.isin(done)]
    print(f"{len(done)} already checkpointed, {len(rest)} to run")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0, buf, n = time.time(), [], 0
    nchunk = len(list(ck.glob("chunk_*.parquet")))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(full_job, (int(r.job), _x(r), int(r.model_id),
                                     float(r.sigma), r.treatment)): int(r.job)
                for r in rest.itertuples()}
        for f in as_completed(futs):
            rec = f.result()
            rec["job"] = rec.pop("key")
            buf.append(rec)
            n += 1
            if len(buf) >= args.chunk or n == len(rest):
                pd.DataFrame(buf).astype({"c_at_failure": float,
                                          "n_outer_at_failure": float,
                                          "eps_at_failure": float}).to_parquet(
                    ck / f"chunk_{nchunk:04d}.parquet", index=False)
                nchunk += 1
                buf = []
                el = time.time() - t0
                print(f"  full: {n}/{len(rest)}  {el / 3600:.2f} h, "
                      f"~{el / n * (len(rest) - n) / 3600:.2f} h left", flush=True)
    got = pd.concat([pd.read_parquet(f) for f in sorted(ck.glob("chunk_*.parquet"))])
    out = todo.merge(got.drop(columns=["model_id", "sigma", "treatment"]), on="job")
    if kap is not None:
        out["key"] = [f"{a}|{s_}|{m}|{g!r}" for a, s_, m, g in
                      zip(out.arm_of_campaign, out.sample_id, out.model_id, out.sigma)]
        out = out.merge(kap, on="key", how="left")
    out.to_parquet(FULL_OUT, index=False)
    print(out.groupby(["source", "arm_of_campaign", "treatment"]).outcome
             .value_counts().unstack(fill_value=0))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cap", "cap_report", "verify_cap", "verify_report",
                                     "mono", "mono_report", "full"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--n-verify", type=int, default=25,
                    help="rows per (arm, treatment, rim-H quartile) stratum")
    ap.add_argument("--n-mono", type=int, default=40,
                    help="marginal rows per (arm, rim-H tercile)")
    ap.add_argument("--n-mono-nonmarg", type=int, default=20,
                    help="non-marginal rows per (arm, rim-H tercile)")
    ap.add_argument("--converged", choices=["none", "all", "kappa_lt_1"],
                    default="none", help="converged-side scope, per R-40.3's test")
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()
    import ball_socket.campaign_config_v8 as v8
    if v8.DRAFT:
        raise SystemExit("box F is DRAFT")
    return {"cap": stage_cap,
            "cap_report": lambda a: _cap_report(pd.read_parquet(CAP_OUT)),
            "verify_cap": stage_verify_cap,
            "verify_report": lambda a: _verify_report(pd.read_parquet(VERIFY_OUT)),
            "mono_report": lambda a: _mono_report(pd.read_parquet(MONO_OUT)),
            "mono": stage_mono, "full": stage_full}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
