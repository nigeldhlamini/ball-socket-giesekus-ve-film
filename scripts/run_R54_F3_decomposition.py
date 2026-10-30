"""R-54.4: decompose E14's F3-bound ratio between the movable rows and all rows.

R-53.3 reported that restricting E14's 60 C cells to the rows that can move raises
the net by 1/f = 2.25 (f = the movable share) but raises the F3 bound by 2.7, and
explained the excess as "restricting also removes part of the verified sample".
R-54.4 shows that is half the algebra: restriction removes rows from the verified
sample AND from the residual it multiplies, and under the rule of three those
cancel exactly in expectation.  The excess, 2.7/2.25 ~ 1.2, is what needs
accounting for.

THE PRODUCTION BOUND, QUOTED BEFORE USE (R-54.4).
`scripts/run_E14_F3_bracket.py:466-471`:

    feas = ((eq.h_min_bnd / s >= cs.lambda_ratio_min) & (eq.T_max_bnd <= lim)
            & (eq.De_sq_bnd <= cs.De_sq_max))
    fixed = (eq.h_min_upper / s) < cs.lambda_ratio_min * (1 - C_SPREAD)
    resid = ~feas & ~fixed
    sizes = eq[resid].stratum.value_counts()
    bound = sum(sizes.get(k, 0) * (3.0 / ns[k]) for k in ns.index) / N_VALID

with k = 0 enforced at `:463-465`.  The form IS rule of three, u(n, 0) = 3/n, so
**M2 is excluded by reading the source**, and the decomposition identity is exact.

AND ONE THING THE QUOTE SHOWS THAT THE RULING'S MODEL DOES NOT ASSUME.
`sizes` is the residual OF THIS CELL, but `ns` comes from
`E14_F3_residual_sample.parquet`, drawn at `:322-338` from the UNION of the four
cells' residuals:

    for s in (lo, hi):
        for lim in cs.T_max_C_by_class.values():
            ...
            resid |= (~feas & ~fixed).to_numpy()

So the verified sample and the residual it weights are DIFFERENT POPULATIONS by
construction.  `f3_bound_decomposition.null_test` draws its hypergeometric from
the same frame it weights by, which is not how the sample was drawn, so the null
is re-implemented here over the union residual and the delivered module is used
for the identity (`decompose`) only.  This is R-54's own near-miss, in the code
rather than in the prose: comparing a verified-sample share against a population
it was not drawn from.

The excess therefore splits in two:

    rho_cell / phi  =  (rho_cell / rho_union)  x  (rho_union / phi)
                       population mismatch        sampling within the union

reported per stratum and weighted.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
DEST = OUT / "E14_F3_bound_decomposition.json"
MIXED = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
BRACKET = OUT / "E14_F3_bracket.parquet"
RESID_SAMPLE = OUT / "E14_F3_residual_sample.parquet"
BOUND = OUT / "E14_F3_bound.json"
MOVABLE = OUT / "E14_movable_subpopulation.json"
NET = OUT / "E14_net_decomposition.json"
FLIPS = OUT / "E14_F3_residual_solves.report.json"
SRC = ROOT / "scripts" / "run_E14_F3_bracket.py"
QUOTE = (466, 471)
UNION_QUOTE = (322, 329)
SEED, DRAWS = 20260923, 20000

_spec = importlib.util.spec_from_file_location(
    "f3dec", ROOT / "scripts" / "f3_bound_decomposition.py")
f3d = importlib.util.module_from_spec(_spec)
sys.modules["f3dec"] = f3d
_spec.loader.exec_module(f3d)


def _null_over_union(R_union, Rm_union, n_all, rho_cell, w, draws, seed):
    """Sampling distribution of SUM w rho_cell/phi when each stratum's sample is
    a uniform draw from the UNION residual -- the population actually sampled."""
    rng = np.random.default_rng(seed)
    obs_terms = []
    sims = np.empty(draws)
    for i in range(draws):
        nm = np.array([rng.hypergeometric(int(rm), int(ru - rm), int(n))
                       for ru, rm, n in zip(R_union, Rm_union, n_all)])
        phi = np.maximum(nm, 1) / n_all
        sims[i] = float(np.sum(w * rho_cell / phi))
    return sims, obs_terms


def main() -> int:
    from ball_socket.immutable import guarded_path
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M, ConstraintSet
    guarded_path(str(DEST), what="E14 F3 bound decomposition")
    cs = ConstraintSet()
    src = SRC.read_text(encoding="utf-8").splitlines()
    quoted = "\n".join(src[QUOTE[0] - 1:QUOTE[1]])
    union_quoted = "\n".join(src[UNION_QUOTE[0] - 1:UNION_QUOTE[1]])
    if "3.0 / ns[k]" not in quoted or "resid |=" not in union_quoted:
        raise SystemExit("the quoted lines are not the bound / union sampler any "
                         "more; re-read the source before decomposing")

    _s2 = importlib.util.spec_from_file_location("e14f3", SRC)
    f3 = importlib.util.module_from_spec(_s2)
    sys.modules["e14f3"] = f3
    _s2.loader.exec_module(f3)

    flips = json.loads(FLIPS.read_text())["flips"]
    bound_ref = json.loads(BOUND.read_text())["cells"]
    mov_ref = json.loads(MOVABLE.read_text())["cells"]
    N_all = json.loads(NET.read_text())["arms"]["mixed"][
        "sigma_hi|water_glycol"]["n_valid_rows"]

    lo, hi = (float(x) for x in SIGMA_STAR_BAND_M)
    b = pd.read_parquet(BRACKET)
    b = b[b.state == "equilibrium"]
    M = pd.read_parquet(MIXED, columns=["sample_id", "model_id", "T_in", "T_max_bnd",
                                        "h_min_bnd", "De_sq_bnd"])
    M["model_id"] = M.model_id.astype(int)
    eq = b.merge(M, on=["sample_id", "model_id"], suffixes=("", "_a"))
    eq["stratum"] = pd.cut(eq.h_min_bnd / lo, [0, 1, 2, 2.9, 3.1, 10, np.inf],
                           labels=False)
    pick = pd.read_parquet(RESID_SAMPLE)
    ns_all = pick.stratum.value_counts()
    strata = sorted(ns_all.index)

    # the UNION residual, rebuilt exactly as stage_residual draws it
    union = np.zeros(len(eq), bool)
    for s_ in (lo, hi):
        for lim_ in cs.T_max_C_by_class.values():
            fe = ((eq.h_min_bnd / s_ >= cs.lambda_ratio_min)
                  & (eq.T_max_bnd <= lim_) & (eq.De_sq_bnd <= cs.De_sq_max))
            fx = (eq.h_min_upper / s_) < cs.lambda_ratio_min * (1 - f3.C_SPREAD)
            union |= (~fe & ~fx).to_numpy()

    rep = {"quoted": {"bound": {"file": f"{SRC.relative_to(ROOT).as_posix()}:"
                                       f"{QUOTE[0]}-{QUOTE[1]}", "source": quoted,
                               "k_zero_enforced_at": f"{SRC.name}:463-465",
                               "form": "rule of three, u(n, k=0) = 3/n",
                               "M2_excluded_by_reading": True},
                      "sample_population": {
                          "file": f"{SRC.relative_to(ROOT).as_posix()}:"
                                  f"{UNION_QUOTE[0]}-{UNION_QUOTE[1]}",
                          "source": union_quoted,
                          "consequence": ("the verified sample is drawn from the UNION "
                                          "of the four cells' residuals, not from the "
                                          "cell the bound is quoted for")}},
           "null": {"draws": DRAWS, "seed": SEED,
                    "population": "the union residual, per stratum (uniform draw)"},
           "cells": {}}

    for end, sigma in (("lo", lo), ("hi", hi)):
        lim = cs.T_max_C_by_class["water_glycol"]
        cell = f"sigma_{end}|water_glycol"
        f = flips[cell]
        if f["flips_to_feasible"] or f["flips_to_infeasible"]:
            raise SystemExit(f"{cell}: flips observed; k = 0 no longer holds")
        feas = ((eq.h_min_bnd / sigma >= cs.lambda_ratio_min)
                & (eq.T_max_bnd <= lim) & (eq.De_sq_bnd <= cs.De_sq_max))
        fixed = (eq.h_min_upper / sigma) < cs.lambda_ratio_min * (1 - f3.C_SPREAD)
        resid = (~feas & ~fixed).to_numpy()
        mov_eq = (eq.T_in <= lim).to_numpy()
        mov_pick = (pick.T_in <= lim).to_numpy()
        st = eq.stratum.to_numpy()

        R_all = np.array([int((resid & (st == k)).sum()) for k in strata])
        R_mov = np.array([int((resid & mov_eq & (st == k)).sum()) for k in strata])
        RU = np.array([int((union & (st == k)).sum()) for k in strata])
        RUm = np.array([int((union & mov_eq & (st == k)).sum()) for k in strata])
        n_all = np.array([int(ns_all[k]) for k in strata])
        n_mov = np.array([int(((pick.stratum.to_numpy() == k) & mov_pick).sum())
                          for k in strata])
        z = np.zeros(len(strata), int)
        N_mov = int(mov_ref[cell]["movable_rows"]["n"])

        s = f3d.Strata(R_all=R_all, R_mov=R_mov, n_all=n_all, n_mov=n_mov,
                       k_all=z, k_mov=z, N_all=int(N_all), N_mov=N_mov)
        dec = f3d.decompose(s, "rule3")
        for got, ref, what in ((dec["bound_all_pp"], bound_ref[cell]["F3_bound_pp"], "all"),
                               (dec["bound_mov_pp"],
                                mov_ref[cell]["F3_bound_pp"]["movable_rows"], "movable")):
            if abs(got - ref) > 5e-3:
                raise SystemExit(f"{cell}: decomposition's {what}-rows bound {got:.4f} "
                                 f"!= production artefact {ref:.4f}")

        w = np.asarray(dec["w"], float)
        phi = n_mov / n_all
        rho_cell = np.divide(R_mov, R_all, out=np.zeros(len(strata)), where=R_all > 0)
        rho_union = np.divide(RUm, RU, out=np.zeros(len(strata)), where=RU > 0)
        # split the excess: population mismatch x sampling within the union
        mismatch = float(np.sum(w * np.divide(rho_cell, rho_union,
                                              out=np.zeros(len(strata)),
                                              where=rho_union > 0)))
        sampling = float(np.sum(w * np.divide(rho_union, phi,
                                              out=np.zeros(len(strata)),
                                              where=phi > 0)))
        sims, _ = _null_over_union(RU, RUm, n_all, rho_cell, w, DRAWS, SEED)
        obs = dec["representativeness_factor"]
        p_upper = float(np.mean(sims >= obs))

        rep["cells"][cell] = {
            "per_stratum": [
                {"stratum": int(k), "R_cell": int(ra), "R_cell_movable": int(rm),
                 "R_union": int(ru), "R_union_movable": int(rum),
                 "n": int(na), "n_movable": int(nm),
                 "rho_cell": float(rc), "rho_union": float(rux), "phi": float(ph),
                 "w": float(wi)}
                for k, ra, rm, ru, rum, na, nm, rc, rux, ph, wi in
                zip(strata, R_all, R_mov, RU, RUm, n_all, n_mov,
                    rho_cell, rho_union, phi, w)],
            "bound_all_pp": dec["bound_all_pp"], "bound_mov_pp": dec["bound_mov_pp"],
            "ratio": dec["ratio"],
            "denominator_factor": dec["denominator_factor"],
            "representativeness_factor": obs,
            "curvature_remainder": dec["curvature_remainder"],
            "excess_split": {"population_mismatch_cell_vs_union": mismatch,
                             "sampling_within_union": sampling,
                             "product": mismatch * sampling},
            "null_test": {"observed": obs, "median": float(np.median(sims)),
                          "q95": float(np.quantile(sims, 0.95)), "p_upper": p_upper},
        }

    c = rep["cells"]["sigma_hi|water_glycol"]
    rep["verdict"] = {
        "M2_bound_form": "EXCLUDED by reading the source; remainder is "
                         f"{c['curvature_remainder']:.6f}",
        "M1_sampling": ("SUPPORTED" if c["null_test"]["p_upper"] < 0.05 else
                        "NOT SUPPORTED: the observed factor is within the null's "
                        f"range (p = {c['null_test']['p_upper']:.3f})"),
        "M3_population_mismatch": ("the sample is drawn from the union residual, so "
                                   "part of the excess is a population mismatch by "
                                   "construction: "
                                   f"{c['excess_split']['population_mismatch_cell_vs_union']:.3f} "
                                   "of the "
                                   f"{c['representativeness_factor']:.3f} factor"),
        "E14_verdict_unchanged": "+0.22 pp, all rows, the registered measurand"}
    DEST.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    for cell, c in rep["cells"].items():
        print(f"\n{cell}: bound {c['bound_all_pp']:.4f} -> {c['bound_mov_pp']:.4f} pp"
              f"  ratio {c['ratio']:.4f}")
        print(f"  = denominator {c['denominator_factor']:.4f} x representativeness "
              f"{c['representativeness_factor']:.4f} x remainder "
              f"{c['curvature_remainder']:.9f}")
        e = c["excess_split"]
        print(f"  excess split: population mismatch (cell vs union) "
              f"{e['population_mismatch_cell_vs_union']:.4f} x sampling within union "
              f"{e['sampling_within_union']:.4f} = {e['product']:.4f}")
        n = c["null_test"]
        print(f"  null over the union residual: observed {n['observed']:.4f}, median "
              f"{n['median']:.4f}, q95 {n['q95']:.4f}, p_upper {n['p_upper']:.4f}")
        print("   s  R_cell Rc_mov  R_union Ru_mov    n  n_mov  rho_cell rho_union   phi      w")
        for r in c["per_stratum"]:
            print(f"  {r['stratum']:2d} {r['R_cell']:7d} {r['R_cell_movable']:6d} "
                  f"{r['R_union']:8d} {r['R_union_movable']:6d} {r['n']:4d} "
                  f"{r['n_movable']:6d} {r['rho_cell']:9.3f} {r['rho_union']:9.3f} "
                  f"{r['phi']:6.3f} {r['w']:6.3f}")
    print("\nverdict:", json.dumps(rep["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
