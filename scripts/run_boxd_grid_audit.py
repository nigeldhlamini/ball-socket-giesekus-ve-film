"""Box-D registered grid-convergence audit (Phase 3).

Supersedes the 2026-08-23 diagnostic probe (``boxd_grid_probe_new_regime.json``),
which was explicitly NOT the registered audit: three grids at two hand-picked
points.

WHAT BOX D REQUIRES OF THIS AUDIT (ledger 2.6, rulings R-12.4, R-14.4):

  * run at the CORRECTED bracket -- eps_cap(theta_min) = (1-margin)/cos(theta_min)
    rather than the box-C hard-coded 0.99.  The corrected bracket admits a
    regime box C could never enter (eps > 1), and a convergence result
    established on the old bracket does NOT transfer to states the old bracket
    could not reach;
  * report the CORRECTED extrema -- h_min_bnd, De_sq_bnd, T_max_bnd -- since
    those are what box D's constraint set consumes (constraint_extremum_source);
  * stratify on the NEW REGIME: eps > 1, smallest theta_min, largest c;
  * include HOT-SPOT LOCATION as a convergence criterion, not only hot-spot
    value.  R-14.4 measured the T_max extremum's LOCATION to be grid-dependent
    -- interior at 32x24, at the rim by 64x48 -- so a criterion on value alone
    can be satisfied while the peak sits in the wrong place.

The grid is registered as a CRITERION rather than a value (box D
SOLVER_NUMERICS["grid"]), so this script does not assert which grid is
production.  It measures which grids satisfy the criterion and reports that.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from ball_socket.verification import convergence as _cv
import sys
import time

import numpy as np
import pandas as pd

_REPO = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ball_socket import campaign as C                       # noqa: E402
from ball_socket.campaign_config_v6 import (                # noqa: E402
    CONSTRAINT_EXTREMUM_SOURCE, eps_cap)

OUT = _REPO / "results" / "ch9"

#: Grids to compare.  The first is box B/C's production grid; convergence is
#: assessed on the SEQUENCE, not against any one of them as truth.
# CANDIDATES EXTENDED FOR THE CORRECTED BRACKET (rulings R-33.5, R-13.4).
#
# This list was chosen while the audit was -- unknowingly -- running at
# eps_cap = 0.99.  R-13.4 predicts near-rim mobility changing by factors of 2 to
# 13 across the FIRST CELL at eps near the true ceiling, against ~30 % in box
# B's regime, so a list whose finest entry is 64x48 may not bracket the answer.
#
# A candidate list cannot be derived -- it is a search, not a registration.
# What CAN be made structural is the refusal below: a grid that is the finest
# candidate is never reported as converged, because the pairwise criterion has
# nothing above it to compare against.  That converts "the list was too short"
# from a silent wrong answer into an explicit NOT ESTABLISHED.
GRIDS = [(32, 24), (48, 36), (64, 48), (96, 72)]

#: Functionals whose grid convergence decides whether the production grid is
#: adequate.  The three corrected extrema are here because they are what the
#: constraint set consumes; the integrals are here because they converge at the
#: operator order and are the control.
METRICS = ["h_min_bnd", "De_sq_bnd", "T_max_bnd", "F", "P_loss", "Q_leak",
           "eps_ecc", "h_min", "De_sq", "T_max"]

#: Registered tolerance for the audit (per cent, relative, successive grids).
TOL_PCT = 2.0


def _strata(audit: pd.DataFrame, n_per: int, seed: int) -> pd.DataFrame:
    """Stratified sample emphasising the regime the corrected bracket opens.

    The point of the stratification is that box C's grid evidence was collected
    where box C could go.  The corrected bracket reaches further, and it is
    exactly the far end -- widest near-rim cells, largest clearance, highest
    eccentricity -- where the near-rim gradients are steepest and a coarse grid
    is least likely to hold.
    """
    rng = np.random.default_rng(seed)
    a = audit.copy()
    if "theta_min_deg" not in a.columns:
        raise SystemExit("audit lacks theta_min_deg; wrong artefact?")

    tm_q = a["theta_min_deg"].quantile(0.25)
    c_q = a["c0"].quantile(0.75)

    strata: dict[str, pd.Index] = {
        # the new regime, in the three directions the ledger names
        "smallest_theta_min": a.index[a["theta_min_deg"] <= tm_q],
        "largest_c": a.index[a["c0"] >= c_q],
        "new_regime_corner": a.index[(a["theta_min_deg"] <= tm_q)
                                     & (a["c0"] >= c_q)],
        # controls
        "converged": a.index[a["converged"] == True],            # noqa: E712
        "blowup": a.index[a["converged"] == False],              # noqa: E712
    }
    picked: list[int] = []
    for name, idx in strata.items():
        if len(idx) == 0:
            print(f"  stratum {name:<20} EMPTY -- skipped")
            continue
        take = min(n_per, len(idx))
        sel = rng.choice(np.asarray(idx), size=take, replace=False)
        picked.extend(int(i) for i in sel)
        print(f"  stratum {name:<20} {len(idx):>5} rows, sampled {take}")
    rows = a.loc[sorted(set(picked))].copy()
    rows["row_id"] = rows.index
    return rows


def _evaluate(x: dict, nth: int, nph: int) -> dict:
    """One sample at one grid, at the CORRECTED bracket, corrected extrema on."""
    # Explicit, not via the module global.  This script happens to run
    # in-process so the global would have worked, but the same line inside a
    # pool would silently run every worker at the default grid.
    t0 = time.time()
    row = C.evaluate_x(x, clearance_closure=True, Ntheta=nth, Nphi=nph)
    row["_wall_s"] = time.time() - t0
    row["Ntheta"], row["Nphi"] = nth, nph
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=str(OUT / "audit_N8192_seed20260727.parquet"))
    ap.add_argument("--n-per-stratum", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--tolerance-pct", type=float, default=TOL_PCT)
    # BRACKET-TAGGED, AND IT IS NOT COSMETIC.  The untagged name was written by
    # the eps_cap = 0.99 run and re-written by this one, destroying the former.
    # An audit's identity includes the bracket it ran at: two audits at two
    # brackets are two measurements, not two attempts at one.
    ap.add_argument("--out",
                    default=str(OUT / "boxd_grid_audit_corrected_bracket.parquet"))
    args = ap.parse_args()

    print("BOX-D GRID-CONVERGENCE AUDIT (registered Phase 3)")
    print(f"  constraint_extremum_source = {CONSTRAINT_EXTREMUM_SOURCE}")
    print(f"  corrected bracket: eps_cap(20 deg)={eps_cap(np.deg2rad(20)):.4f}  "
          f"eps_cap(40 deg)={eps_cap(np.deg2rad(40)):.4f}")
    print(f"  grids: {GRIDS}   tolerance: {args.tolerance_pct} %\n")

    audit = pd.read_parquet(args.audit)
    # REFUSE TO OVERWRITE (parity with run_boxd_campaign.py).  That driver has
    # refused this since box D -- "already exists; refusing to overwrite a
    # campaign artefact" -- and this one did not.  One remedy, two scripts,
    # applied to one of them: the same partial application as R-28.3 and
    # R-23.6, and this time it cost an artefact.
    # THROUGH THE SHARED WRITER, not a local existence check (R-35.7).  A
    # per-script check is one more thing each new script must remember; the
    # guard belongs in one place every writer already passes through.
    from ball_socket.immutable import guarded_path, ArtefactExists
    try:
        guarded_path(str(args.out), what="grid audit artefact")
        guarded_path(str(pathlib.Path(args.out).with_suffix(".summary.json")),
                     what="grid audit summary")
    except ArtefactExists as exc:
        raise SystemExit(str(exc))

    sample = _strata(audit, args.n_per_stratum, args.seed)
    print(f"\n  {len(sample)} distinct samples x {len(GRIDS)} grids "
          f"= {len(sample) * len(GRIDS)} solves\n")

    axes = [d.name for d in C.INPUT_SPACE]
    recs: list[dict] = []
    t_start = time.time()
    for n, (_, r) in enumerate(sample.iterrows(), 1):
        x = {k: float(r[k]) for k in axes}
        for (nth, nph) in GRIDS:
            rec = _evaluate(x, nth, nph)
            rec["row_id"] = int(r["row_id"])
            recs.append(rec)
        if n % 5 == 0 or n == len(sample):
            el = time.time() - t_start
            print(f"  {n}/{len(sample)} samples  ({el/60:.1f} min elapsed, "
                  f"{el/n*(len(sample)-n)/60:.1f} min remaining)")

    df = pd.DataFrame(recs)
    df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows)")

    summary = _analyse(df, args.tolerance_pct)
    sp = pathlib.Path(args.out).with_suffix(".summary.json")
    sp.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"wrote {sp}")
    return 0


def _analyse(df: pd.DataFrame, tol_pct: float) -> dict:
    """Relative change between successive grids, per metric, plus the hot-spot
    location criterion."""
    out: dict = {"artefact": "boxd_grid_audit",
                 "constraint_extremum_source": CONSTRAINT_EXTREMUM_SOURCE,
                 "grids": GRIDS, "tolerance_pct": tol_pct, "metrics": {}}
    piv = {g: df[(df.Ntheta == g[0]) & (df.Nphi == g[1])].set_index("row_id")
           for g in GRIDS}

    print(f"\nCONVERGENCE, successive grids, tolerance {tol_pct} %")
    _rr = [round(GRIDS[k+1][0]/GRIDS[k][0], 3) for k in range(len(GRIDS)-1)]
    print(f"  refinement ratios {_rr} are UNEQUAL, so the raw pairwise")
    print( "  percentages are not a convergence sequence -- read p (R-35.2)")
    for m in METRICS:
        if m not in df.columns:
            continue
        rows = []
        for a, b in zip(GRIDS, GRIDS[1:]):
            xa, xb = piv[a][m], piv[b][m]
            common = xa.index.intersection(xb.index)
            va, vb = xa.loc[common].astype(float), xb.loc[common].astype(float)
            ok = np.isfinite(va) & np.isfinite(vb) & (np.abs(va) > 0)
            if ok.sum() == 0:
                rows.append({"from": a, "to": b, "n": 0}); continue
            rel = np.abs((vb[ok] - va[ok]) / va[ok]) * 100.0
            rows.append({"from": list(a), "to": list(b), "n": int(ok.sum()),
                         "median_pct": float(np.median(rel)),
                         "p95_pct": float(np.percentile(rel, 95)),
                         "max_pct": float(rel.max()),
                         "frac_within_tol": float((rel <= tol_pct).mean())})
        out["metrics"][m] = rows
        # OBSERVED ORDER, NOT RAW PAIRWISE PERCENTAGES (ruling R-35.2).  This
        # list's ratios are 1.5, 1.333, 1.5 -- unequal -- so the three changes
        # are not a comparable sequence, and reading them as one produced a
        # wrong verdict about legacy h_min.  Steps 0 and 2 share r = 1.5 with N
        # doubled, so (1 - r^-p) cancels and p = log2(d0/d2) exactly.
        _ch = [r["p95_pct"] for r in rows if r.get("n")]
        if len(_ch) == len(GRIDS) - 1:
            out.setdefault("observed_order", {})[m] = _cv.order_table(
                [tuple(g) for g in GRIDS], {m: _ch})[m]
        # THE FINEST CANDIDATE CANNOT BE THE ANSWER (R-33.5).  The criterion is
        # pairwise: grid g is converged when the step g -> (next finer) is
        # within tolerance.  For the last entry there is no next finer grid, so
        # a "pass" there is an artefact of where the list stopped -- the same
        # sufficient-to-fail-not-to-pass asymmetry as gate 1's source scan.
        out.setdefault("selection_guard", {})[m] = {
            "finest_candidate": list(GRIDS[-1]),
            "rule": ("a grid is reportable as converged only if a FINER grid "
                     "confirms it; the finest candidate is never reportable"),
            "reportable": [list(r["from"]) for r in rows
                           if r.get("n") and r.get("p95_pct", 9e9) <= tol_pct],
        }
        last = rows[-1]
        if last.get("n"):
            flag = "" if last["p95_pct"] <= tol_pct else "   <<< NOT CONVERGED"
            print(f"  {m:<12} {last['from']}->{last['to']}  "
                  f"median {last['median_pct']:7.3f} %  p95 {last['p95_pct']:7.3f} %"
                  f"{flag}")

    # --- hot-spot LOCATION criterion (R-14.4) -------------------------------
    # T_max_bnd equal to T_max means the peak was interior; a difference means
    # the boundary-aware value won, i.e. the peak is at the rim.  That flag is
    # the coarsest available proxy for LOCATION and it is the one the row
    # carries, so the criterion is: does the peak's location classification
    # change between grids?
    loc = {}
    for g in GRIDS:
        d = piv[g]
        if "T_max_bnd" in d.columns and "T_max" in d.columns:
            at_rim = (d["T_max_bnd"].astype(float)
                      > d["T_max"].astype(float) + 1e-9)
            loc[str(g)] = at_rim
    if len(loc) >= 2:
        keys = list(loc)
        common = loc[keys[0]].index
        for k in keys[1:]:
            common = common.intersection(loc[k].index)
        flips = np.zeros(len(common), dtype=bool)
        for a, b in zip(keys, keys[1:]):
            flips |= (loc[a].loc[common].values != loc[b].loc[common].values)
        out["hotspot_location"] = {
            "criterion": ("peak location classification (rim vs interior) must "
                          "not change between grids"),
            "n": int(len(common)),
            "n_flipped": int(flips.sum()),
            "frac_stable": float(1.0 - flips.mean()) if len(common) else None,
        }
        print(f"\nHOT-SPOT LOCATION (R-14.4 criterion): "
              f"{int(flips.sum())}/{len(common)} samples change classification "
              f"between grids")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
