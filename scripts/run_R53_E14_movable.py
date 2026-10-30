"""R-53.3: E14's cells recomputed over the MOVABLE subpopulation, and A6.

THE DEFECT IN THE PRESENTATION.  E14's four cells are signed changes in the
feasible fraction, in percentage points, over the valid-row denominator.  Under
the 60 C (water-glycol) ceiling, every row whose inlet temperature exceeds 60 C
is infeasible under BOTH conventions, because T_max_bnd >= T_max >= T_in: no
correction can move it.  Those rows are 55.4 % of the denominator.  They enter
the denominator and contribute zero to the numerator, so the two 60 C cells are
diluted while the two 110 C cells (the mineral ceiling is the T_in axis endpoint)
are not.  The +0.22 pp cell and the +3.90 pp cell have been presented side by
side as if they were the same measurement.

> A6 (aggregation catalogue, R-53.3): a change measured in percentage points over
> a population part of which CANNOT change is not comparable to one where all of
> it can.  Report the movable denominator beside it.

WHAT THIS SCRIPT DOES.  No solves.  It recomputes, per cell:
  * the net change over all valid rows (reproducing E14_net_decomposition.json,
    asserted equal, so this is a re-presentation and not a second measurement);
  * the net change over the movable rows alone, with the bootstrap resampled over
    SAMPLES as E14's own bootstrap is (A1-A3);
  * the F3 bound restricted the same way -- residual rows restricted to movable
    ones, and the rule of three taken on the per-stratum counts of the RESIDUAL
    SAMPLE restricted the same way, so the bound rests on evidence drawn from the
    population it is quoted over;
  * the joint worst case from both.

The restricted bound is not the diluted one: restricting the population also
shrinks the verified sample, which widens the rule of three per stratum.  That is
the part an expectation about "the same denominator" would have missed.
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
OUT = ROOT / "results" / "ch9"
DEST = OUT / "E14_movable_subpopulation.json"
NET = OUT / "E14_net_decomposition.json"
BOUND = OUT / "E14_F3_bound.json"
MIXED = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
BRACKET = OUT / "E14_F3_bracket.parquet"
RESID_SAMPLE = OUT / "E14_F3_residual_sample.parquet"
BOOT, SEED = 2000, 20260923

_spec = importlib.util.spec_from_file_location(
    "e14net", ROOT / "scripts" / "run_E14_net.py")
e14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e14)


def main() -> int:
    from ball_socket.immutable import guarded_path
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M, ConstraintSet
    guarded_path(str(DEST), what="E14 movable-subpopulation recomputation")
    cs = ConstraintSet()
    rng = np.random.default_rng(SEED)
    net_ref = json.loads(NET.read_text())["arms"]["mixed"]
    bound_ref = json.loads(BOUND.read_text())["cells"]

    df = pd.read_parquet(MIXED)
    d = df[~e14._invalid(df, "mixed")].copy()
    sid = d.sample_id.to_numpy()
    samples = np.unique(sid)
    idx_by_sample = {s: np.flatnonzero(sid == s) for s in samples}
    N = len(d)

    rep = {"rule": "A6 (R-53.3): a pp change over a partly immovable population is "
                   "not comparable to one where all of it can move",
           "denominator": "valid rows (registered partition), mixed arm",
           "n_valid_rows": int(N),
           "bootstrap": {"resampled_over": "samples", "B": BOOT, "seed": SEED},
           "cells": {}}

    for end, sigma in zip(("lo", "hi"), (float(s) for s in SIGMA_STAR_BAND_M)):
        for cls, lim in cs.T_max_C_by_class.items():
            cell = f"sigma_{end}|{cls}"
            base = e14._labels(d, sigma=sigma, cls=cls,
                               **e14._state(eps_cap=False, h_min=False,
                                            T_max=False, De_sq=False)).to_numpy()
            full = e14._labels(d, sigma=sigma, cls=cls,
                               **e14._state(eps_cap=True, h_min=True,
                                            T_max=True, De_sq=True)).to_numpy()
            # a row is IMMOVABLE when its inlet alone puts it outside the ceiling:
            # T_max_bnd >= T_max >= T_in, so no correction can make it feasible
            immovable = (d.T_in > lim).to_numpy()
            movable = ~immovable
            if (base[immovable].any() or full[immovable].any()):
                raise SystemExit(f"{cell}: an immovable row is labelled feasible; "
                                 f"the arithmetic behind this split is wrong")

            net_all = float(full.mean() - base.mean())
            if abs(net_all - net_ref[cell]["net"]) > 1e-12:
                raise SystemExit(f"{cell}: net {net_all} != registered artefact "
                                 f"{net_ref[cell]['net']}; this is a re-presentation, "
                                 f"so the all-rows figure must reproduce exactly")
            nm = int(movable.sum())
            net_mov = float(full[movable].mean() - base[movable].mean())

            # The all-rows interval is E14's OWN, read from its artefact: a second
            # bootstrap of a published quantity would put two numbers in the record
            # for one measurement (drawn-vs-emitted rule).  Only the movable
            # interval is new, and it uses E14's resampling design (over samples).
            boot_mov = []
            for _ in range(BOOT):
                pick = rng.choice(samples, size=len(samples), replace=True)
                rows = np.concatenate([idx_by_sample[s] for s in pick])
                b, f, mv = base[rows], full[rows], movable[rows]
                boot_mov.append(f[mv].mean() - b[mv].mean() if mv.any() else np.nan)
            q = lambda v: [float(np.nanpercentile(v, 2.5)),
                           float(np.nanpercentile(v, 97.5))]

            rep["cells"][cell] = {
                "immovable_rows": int(immovable.sum()),
                "immovable_pct": float(100 * immovable.mean()),
                "why_immovable": (f"T_in > {lim:g} C, and T_max_bnd >= T_max >= T_in"
                                  if immovable.any() else
                                  f"none: {lim:g} C is the T_in axis endpoint"),
                "all_rows": {"n": N, "net_pp": 100 * net_all,
                             "ci_pp": [100 * x for x in net_ref[cell]["net_ci"]],
                             "ci_source": "E14_net_decomposition.json (E14's own bootstrap)"},
                # When nothing is immovable the movable population IS the whole
                # population, so the interval is the same measurement: reuse
                # E14's rather than reporting bootstrap noise as a difference.
                "movable_rows": {
                    "n": nm,
                    "net_pp": 100 * net_mov,
                    "ci_pp": ([100 * x for x in net_ref[cell]["net_ci"]]
                              if not immovable.any()
                              else [100 * x for x in q(boot_mov)]),
                    "ci_source": ("E14_net_decomposition.json (identical population)"
                                  if not immovable.any() else
                                  f"bootstrap over samples, B={BOOT}, seed={SEED}")},
                "ratio_movable_to_all": float(net_mov / net_all) if net_all else None,
            }

    # ---- the F3 bound, restricted the same way ------------------------------
    b = pd.read_parquet(BRACKET)
    b = b[b.state == "equilibrium"]
    M = pd.read_parquet(MIXED, columns=["sample_id", "model_id", "T_in", "T_max_bnd",
                                        "h_min_bnd", "De_sq_bnd"])
    M["model_id"] = M.model_id.astype(int)
    eq = b.merge(M, on=["sample_id", "model_id"], suffixes=("", "_a"))
    lo = float(SIGMA_STAR_BAND_M[0])
    eq["stratum"] = pd.cut(eq.h_min_bnd / lo, [0, 1, 2, 2.9, 3.1, 10, np.inf],
                           labels=False)
    pick = pd.read_parquet(RESID_SAMPLE)
    C_SPREAD = float(getattr(e14, "C_SPREAD", 0.0)) or None

    import importlib.util as _u
    _s2 = _u.spec_from_file_location("e14f3", ROOT / "scripts" / "run_E14_F3_bracket.py")
    f3 = _u.module_from_spec(_s2)
    _s2.loader.exec_module(f3)
    C_SPREAD = f3.C_SPREAD

    for end, sigma in zip(("lo", "hi"), (float(s) for s in SIGMA_STAR_BAND_M)):
        for cls, lim in cs.T_max_C_by_class.items():
            cell = f"sigma_{end}|{cls}"
            feas = ((eq.h_min_bnd / sigma >= cs.lambda_ratio_min)
                    & (eq.T_max_bnd <= lim) & (eq.De_sq_bnd <= cs.De_sq_max))
            fixed = (eq.h_min_upper / sigma) < cs.lambda_ratio_min * (1 - C_SPREAD)
            resid = ~feas & ~fixed
            mov_eq = eq.T_in <= lim
            mov_pick = pick[pick.T_in <= lim]
            ns_all = pick.stratum.value_counts()
            ns_mov = mov_pick.stratum.value_counts()
            n_mov_rows = rep["cells"][cell]["movable_rows"]["n"]

            sizes_all = eq[resid].stratum.value_counts()
            sizes_mov = eq[resid & mov_eq].stratum.value_counts()
            bound_all = sum(sizes_all.get(k, 0) * (3.0 / ns_all[k])
                            for k in ns_all.index) / N
            missing = [int(k) for k in sizes_mov.index if ns_mov.get(k, 0) == 0]
            if missing:
                bound_mov = None
            else:
                bound_mov = sum(sizes_mov.get(k, 0) * (3.0 / ns_mov[k])
                                for k in ns_mov.index) / n_mov_rows

            c = rep["cells"][cell]
            ref_bound = bound_ref[cell]["F3_bound_pp"]
            if abs(100 * bound_all - ref_bound) > 5e-3:
                raise SystemExit(f"{cell}: F3 bound {100 * bound_all:.4f} != artefact "
                                 f"{ref_bound:.4f}")
            c["F3_bound_pp"] = {"all_rows": 100 * bound_all,
                                "movable_rows": None if bound_mov is None
                                else 100 * bound_mov,
                                "sample_per_stratum_all": {int(k): int(v) for k, v
                                                           in ns_all.sort_index().items()},
                                "sample_per_stratum_movable": {int(k): int(v) for k, v
                                                               in ns_mov.sort_index().items()},
                                "strata_with_residual_but_no_movable_sample": missing}
            c["joint_worst_case_pp"] = {
                "all_rows": c["all_rows"]["ci_pp"][0] - 100 * bound_all,
                "movable_rows": None if bound_mov is None
                else c["movable_rows"]["ci_pp"][0] - 100 * bound_mov}

    DEST.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["cells"], indent=1)[:200])
    hdr = f"{'cell':22s} {'immov%':>7s} {'net all':>9s} {'net mov':>9s} {'F3 all':>7s} {'F3 mov':>7s} {'joint all':>10s} {'joint mov':>10s}"
    print(hdr)
    for k, c in rep["cells"].items():
        fm = c["F3_bound_pp"]["movable_rows"]
        jm = c["joint_worst_case_pp"]["movable_rows"]
        print(f"{k:22s} {c['immovable_pct']:7.1f} {c['all_rows']['net_pp']:9.3f} "
              f"{c['movable_rows']['net_pp']:9.3f} {c['F3_bound_pp']['all_rows']:7.3f} "
              f"{'n/a' if fm is None else f'{fm:7.3f}'} "
              f"{c['joint_worst_case_pp']['all_rows']:10.3f} "
              f"{'n/a' if jm is None else f'{jm:10.3f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
