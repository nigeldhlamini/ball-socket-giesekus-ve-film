#!/usr/bin/env python
"""Box C, Phase 1 audit: thermal-clearance closure, reference arm.

WHY THIS RUN EXISTS (defect D4, amendment A1).  E9 compares the converged
fraction and median h_min WITH and WITHOUT the thermal-clearance closure.  The
Phase 3 box-C audit cannot answer it: by then the asperity load path is active
and boundary friction is feeding the same thermal balance, and the two effects
are not separable in that artefact.  This run isolates the closure.

WHAT IT IS
  * closure ON, reference arm (F1 = bulk / area-averaged film temperature);
  * asperity path INACTIVE -- Phase 2 has not landed, so this is the closure
    against box-B physics and nothing else;
  * same seed lineage and same N_AUDIT as box B, so the box-B audit remains an
    exact row-for-row comparator;
  * box-A and box-B artefacts are neither read for mutation nor written.

Registered by preregistration_v5 sec 3 before running.  Ruling R-6.8 approved
the launch conditional on the reproducibility manifest being wired first
(R-6.2); it is, and this script writes one.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ball_socket import campaign as C                       # noqa: E402
from ball_socket import campaign_config_v5 as v5            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="override N (default: registered N_AUDIT)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--datum", default="bulk", choices=["bulk", "peak"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Rule 10: read the box, never assert it.
    n = args.n or v5.N_AUDIT
    seed = v5.AUDIT_SEED

    # Defect D7 / ruling R-4: every phase entry script checks box-B invariance
    # at its head, and the return is recorded in the hand-back.
    entry = v5.phase_entry_check("phase-1-audit")
    if entry["drift"]:
        print("STOP: box-B invariance drift", entry["drift"])
        return 2

    out = args.out or str(_ROOT / "results" / "ch9" /
                          f"audit_boxC_phase1_N{n}_seed{seed}.parquet")
    print(f"box-C Phase 1 audit: N={n} seed={seed} datum={args.datum} "
          f"workers={args.workers}", flush=True)
    print(f"  registration : signed {v5.SIGNED_ON} by {v5.SIGNED_BY}", flush=True)
    print(f"  box-B check  : drift={entry['drift']} "
          f"keys={entry['keys_checked']}", flush=True)
    print(f"  grid         : {C.NTHETA} x {C.NPHI} (read from config)", flush=True)
    print(f"  closure      : omega_c={v5.CLEARANCE_CLOSURE['omega_c']} "
          f"tol_c={v5.CLEARANCE_CLOSURE['tol_c']} "
          f"max_outer={v5.CLEARANCE_CLOSURE['max_outer']}", flush=True)

    t0 = time.time()
    u = C.sobol_unit(n, seed)
    xs = [C.unit_to_x(uu) for uu in u]
    df = C.run_batch(xs, workers=args.workers,
                     clearance_closure=True, delta_T_datum=args.datum)
    wall = time.time() - t0

    os.makedirs(os.path.dirname(out), exist_ok=True)
    C.write_parquet(df, out)

    summ = C.summarise(df)
    summ.update(stage="boxC_phase1_clearance_closure", n_requested=n, seed=seed,
                wall_seconds=wall, parquet=os.path.basename(out),
                delta_T_datum=args.datum,
                clearance_closure=True, asperity_path="inactive (Phase 2 not landed)")
    # n_outer distribution, over converged rows only.
    conv = df[df["outcome"] == "converged"]
    if len(conv) and "n_outer" in conv:
        no = conv["n_outer"].dropna()
        summ["n_outer"] = {"min": int(no.min()), "median": float(no.median()),
                           "mean": float(no.mean()), "max": int(no.max()),
                           "histogram": {str(int(k)): int(vv) for k, vv
                                         in no.value_counts().sort_index().items()}}
    # STOP condition: clearance fixed point failing on > 1 % of valid rows.
    frac = summ.get("clearance_nonconvergence_fraction", 0.0)
    summ["STOP_clearance_nonconvergence_over_1pct"] = bool(frac > 0.01)

    spath = os.path.splitext(out)[0] + "_summary.json"
    with open(spath, "w") as fh:
        json.dump(summ, fh, indent=2, default=str)

    # Ruling R-6.2: the manifest is written for EVERY artefact, always.
    mpath = C.write_reproducibility_manifest(
        out, config=v5.config_dict(), seed=seed,
        stage="boxC_phase1_clearance_closure",
        extra={"registration": {
            "signed_on": v5.SIGNED_ON, "signed_by": v5.SIGNED_BY,
            "registration_sha256": v5.REGISTRATION_SHA256,
            "frozen": v5.FROZEN_ON is not None},
            "box_b_invariance": entry,
            "delta_T_datum": args.datum,
            "reference_arm": v5.REFERENCE_ARM})

    print(f"\nwall {wall/3600:.2f} h  ({wall/max(len(df),1):.2f} s/row)", flush=True)
    print(f"  parquet  {out}", flush=True)
    print(f"  summary  {spath}", flush=True)
    print(f"  manifest {mpath}", flush=True)
    print(f"  converged {summ['n_converged']}  blowup "
          f"{summ['n_nonconverged_blowup']}  clearance-nonconv "
          f"{summ['n_clearance_nonconverged']}  error "
          f"{summ['n_error_infrastructure']}", flush=True)
    print(f"  clearance non-convergence fraction {frac:.5f}  "
          f"STOP(>1%)={summ['STOP_clearance_nonconvergence_over_1pct']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
