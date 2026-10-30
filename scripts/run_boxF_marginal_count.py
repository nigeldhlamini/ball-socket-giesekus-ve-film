"""Box F sec 5.2: COUNT the marginal band, mask it, and apply the cost gate.

No averaged-film solve happens here.  Every row of both campaign arms is
classified at the registered cap:

    F_film   smooth film load at the cap, at the clearance the row reached
    F_asp    asperity load at the cap (film-independent: the gap is fixed)
    F_target F_app - F_pocket

    MARGINAL iff kappa_lo*F_film + F_asp < F_target <= kappa_hi*F_film + F_asp
    MASKED   iff MARGINAL and H_rim = EPS_CAP_MARGIN*c/sigma* < H_mask

Stored where the artefact has it (contact_limit and blow-up rows carry their
cap loads).  Converged rows do not: one smooth cap solve per (sample, clearance)
supplies F_film, and F_asp is projected analytically per contact arm.

If the band exceeds the registered gate, the script REPORTS its size, the masked
share, and its distribution across the eight design axes, then stops (R-39.5).
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
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]


def cap_job(args):
    """Smooth film at the cap for one sample at one clearance, plus F_asp for
    each requested contact arm on the same gap."""
    sid, x, c, model_ids = args
    import warnings
    warnings.simplefilter("ignore")
    from run_E7_load_decomposition import _solver
    from ball_socket.spherical import Spherical, SphericalConditions
    from ball_socket.reynolds import solve_reynolds
    import ball_socket.mixed as mx

    slv, ci = _solver(x, 32, 24)
    eps = float(slv.eps_cap)
    cond = SphericalConditions(
        p_ch=ci.p_ch, p_case=ci.p_case, e_vec=(0.0, 0.0, -eps * c),
        e_dot=(0.0, 0.0, -slv.squeeze_phase * ci.omega * eps * c),
        omega_b=(0.0, 0.0, slv.spin_ratio * ci.omega))
    sph = Spherical(slv.geom.R_b, c, slv.geom.theta_max, cond, 32, 24,
                    theta_min=slv.geom.theta_min)
    res = solve_reynolds(sph, slv._fluid(ci), slv.config, model=slv.model)
    F_film = float(sph.load_capacity(res.pressure))
    out = {"sample_id": sid, "c": c, "F_film_cap": F_film,
           "F_target": float(slv._applied_load(ci) - slv._pocket_relief(ci)),
           "converged_solve": bool(res.converged), "F_asp_cap": {}}
    if model_ids:
        models = mx.expand_for_solving(
            mx.registered_contact_models(branch="psi", summit_convention="derived"))
        g = sph.gap()
        for mid in model_ids:
            m, prov = models[mid]
            branch = prov.split(":")[1]
            out["F_asp_cap"][int(mid)] = float(
                sph.project_on_load_axis(m.pressure_arm(branch, g)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(OUT / "boxF_marginal_band_count.parquet"))
    args = ap.parse_args()

    from ball_socket.immutable import guarded_path
    import ball_socket.campaign_config_v8 as v8
    if v8.DRAFT:
        raise SystemExit("box F is DRAFT; the count is a registered step and may not run")
    guarded_path(args.out, what="marginal-band count")
    guarded_path(args.out.replace(".parquet", ".report.json"), what="gate report")

    MB, VM = v8.MARGINAL_BAND, v8.VALIDITY_MASK
    k_lo, k_hi = MB["kappa"]["lo"], MB["kappa"]["hi"]
    gate, H_mask = MB["cost_gate_rows"], VM["H_mask"]
    margin = v8.config_dict()["eps_cap_margin"]

    import ball_socket.mixed as mx
    models = mx.expand_for_solving(
        mx.registered_contact_models(branch="psi", summit_convention="derived"))
    sig_of = {i: float(m.sigma_star) for i, (m, _) in enumerate(models)}

    # ---------------- stored cap loads ------------------------------------
    pf = pd.read_parquet(PURE, columns=AXES + ["sample_id", "outcome", "F_film_max",
                                               "F_hydrodynamic", "c_equilibrium"])
    mxd = pd.read_parquet(MIXED, columns=AXES + [
        "sample_id", "model_id", "arm", "outcome", "F_film_at_limit",
        "F_asp_at_limit", "F_target_at_failure", "c_equilibrium",
        "realised_contact_model"])
    mxd["model_id"] = mxd.model_id.astype(int)
    # the model index must name the same model the row ran
    lab = {i: m.label() for i, (m, _) in enumerate(models)}
    bad = (mxd.realised_contact_model.astype(str) != mxd.model_id.map(lab)).sum()
    if bad:
        raise SystemExit(f"{bad} rows: model_id does not map to the model that ran")

    # ---------------- cap solves for converged rows -----------------------
    jobs = []
    pc = pf[pf.outcome == "converged"]
    for r in pc.itertuples():
        jobs.append(("pure", int(r.sample_id), {k: float(getattr(r, k)) for k in AXES},
                     float(r.c_equilibrium), []))
    mc = mxd[mxd.outcome == "converged"]
    spread = (mc.groupby("sample_id").c_equilibrium.agg(lambda s: s.max() / s.min() - 1))
    for sid, g in mc.groupby("sample_id"):
        x = {k: float(g.iloc[0][k]) for k in AXES}
        jobs.append(("mixed", int(sid), x, float(g.c_equilibrium.median()),
                     sorted(g.model_id.unique().tolist())))
    print(f"{len(jobs)} cap solves ({len(pc)} pure-film, {mc.sample_id.nunique()} mixed "
          f"samples); max within-sample c spread over mixed arms "
          f"{100 * spread.max():.4f} %", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(cap_job, [(sid, x, c, mids) for _, sid, x, c, mids in jobs],
                          chunksize=8))
    tags = [j[0] for j in jobs]
    print(f"cap solves done in {(time.time() - t0) / 60:.1f} min", flush=True)

    rows = []
    # pure film
    capP = {r["sample_id"]: r for t, r in zip(tags, res) if t == "pure"}
    for r in pf.itertuples():
        if r.outcome == "converged":
            F_film, F_t = capP[r.sample_id]["F_film_cap"], capP[r.sample_id]["F_target"]
        else:
            F_film, F_t = float(r.F_film_max), float(r.F_hydrodynamic)
        for tag, s in zip(("lo", "hi"), sorted({v for v in sig_of.values()})):
            rows.append({"arm_of_campaign": "pure_film", "sample_id": int(r.sample_id),
                         "model_id": -1, "sigma_end": tag, "sigma": s,
                         "outcome": r.outcome, "F_film": F_film, "F_asp": 0.0,
                         "F_target": F_t, "c0": r.c0,
                         **{k: getattr(r, k) for k in AXES if k != "c0"}})
    # mixed
    capM = {r["sample_id"]: r for t, r in zip(tags, res) if t == "mixed"}
    for r in mxd.itertuples():
        if r.outcome == "converged":
            cm = capM[r.sample_id]
            F_film, F_asp, F_t = cm["F_film_cap"], cm["F_asp_cap"][r.model_id], cm["F_target"]
        elif r.outcome == "contact_limit":
            F_film, F_asp, F_t = r.F_film_at_limit, r.F_asp_at_limit, r.F_target_at_failure
        else:
            continue          # the six picard_at_root rows are INVALID (sec 6)
        rows.append({"arm_of_campaign": "mixed", "sample_id": int(r.sample_id),
                     "model_id": int(r.model_id), "sigma_end": None,
                     "sigma": sig_of[r.model_id], "outcome": r.outcome,
                     "F_film": float(F_film), "F_asp": float(F_asp),
                     "F_target": float(F_t), "c0": r.c0,
                     **{k: getattr(r, k) for k in AXES if k != "c0"}})
    df = pd.DataFrame(rows)
    df["marginal"] = ((k_lo * df.F_film + df.F_asp < df.F_target)
                      & (df.F_target <= k_hi * df.F_film + df.F_asp))
    df["H_rim"] = margin * df.c0 / df.sigma
    df["masked"] = df.marginal & (df.H_rim < H_mask)
    df["unmasked_marginal"] = df.marginal & ~df.masked
    df.to_parquet(args.out, index=False)

    # ---------------- the gate, and its report ---------------------------
    def summary(d):
        return {"rows": int(len(d)), "marginal": int(d.marginal.sum()),
                "masked": int(d.masked.sum()),
                "unmasked_marginal": int(d.unmasked_marginal.sum()),
                "marginal_pct": round(100 * d.marginal.mean(), 2),
                "masked_pct_of_marginal": round(100 * d.masked.sum() / max(d.marginal.sum(), 1), 2)}
    rep = {"kappa": [k_lo, k_hi], "H_mask": H_mask, "cost_gate_rows": gate,
           "by_arm": {a: summary(g) for a, g in df.groupby("arm_of_campaign")},
           "by_outcome": {f"{a}:{o}": summary(g) for (a, o), g in
                          df.groupby(["arm_of_campaign", "outcome"])}}
    # The registered unit is (sample x contact arm).  A pure-film sample is ONE
    # row whatever its sigma* end; the two sigma* rows are re-solve units (the
    # averaged film depends on sigma*) and are reported separately.
    pm = df[df.arm_of_campaign == "pure_film"].groupby("sample_id").agg(
        marginal=("marginal", "any"), masked=("masked", "all"),
        unmasked=("unmasked_marginal", "any"))
    mm = df[df.arm_of_campaign == "mixed"]
    total = int(mm.marginal.sum() + pm.marginal.sum())
    rep["total_marginal_rows"] = total
    rep["total_marginal_rows_composition"] = {
        "mixed_rows": int(mm.marginal.sum()),
        "pure_film_samples": int(pm.marginal.sum())}
    rep["total_unmasked_marginal_rows"] = int(mm.unmasked_marginal.sum() + pm.unmasked.sum())
    rep["re_solve_units_unmasked"] = int(df.unmasked_marginal.sum())
    rep["gate_fired"] = total > gate
    dist = {}
    for ax in AXES:
        q = pd.qcut(df[ax], 5, duplicates="drop")
        t = df.groupby(q, observed=True).agg(marginal_pct=("marginal", "mean"),
                                             masked_pct=("masked", "mean"),
                                             rows=("marginal", "size"))
        dist[ax] = [{"bin": str(i), "rows": int(v.rows),
                     "marginal_pct": round(100 * v.marginal_pct, 1),
                     "masked_pct": round(100 * v.masked_pct, 1)} for i, v in t.iterrows()]
    rep["distribution_across_axes"] = dist
    rep["within_sample_c_spread_max_pct"] = float(100 * spread.max())
    pathlib.Path(args.out.replace(".parquet", ".report.json")).write_text(
        json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("total_marginal_rows",
                                          "total_unmasked_marginal_rows",
                                          "gate_fired", "by_arm")}, indent=1))
    if rep["gate_fired"]:
        print(f"\nCOST GATE FIRED: {total} marginal rows > {gate}.  Report written; "
              f"no averaged solve performed.  Bring it back.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
