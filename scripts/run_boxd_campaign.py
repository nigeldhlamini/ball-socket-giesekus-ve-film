"""Box-D campaign driver — the registered Phase 2/3 audit at the corrected bracket.

WHY A SEPARATE DRIVER, and not a flag on ``run_ch9_campaign.py``:

  * that driver names its artefact ``audit_N{n}_seed{AUDIT_SEED}``, which is
    EXACTLY the box-B artefact's name.  Box B is retained unchanged and
    superseded, never edited or re-run -- the standing rule is unconditional,
    and its parquet is cited by sha256 in Chapter 9, in the delivered CME
    export, and is what erratum ERR-8 attaches to.  A box-D run through that path
    would overwrite it;
  * its ``_save`` calls ``df.to_parquet`` directly rather than
    ``campaign.write_parquet``, so it neither sorts by ``sample_id`` nor strips
    ``df.attrs`` -- both of which are what make the artefact hash reproducible
    (ruling R-21.6, R-22.5).

THE ARTEFACT NAME IS LOAD-BEARING (ruling R-22.2).  ``ch9_numbers.check_claims``
arms the box-D claims gate off the artefact key.  It used to have a second
route through ``df.attrs``, which the attrs strip correctly killed.  The name is
now the live route, and ``test_boxd_gate_arming.py`` pins it.

PAIRING (ruling R-22.3).  The campaign draws seed 20260727 -- the SAME seed as
box B -- because E14's per-correction decomposition and the box-B/box-D
side-by-side reporting are PAIRED comparisons: same Sobol point, two treatments.
A fresh seed would silently downgrade every one of them to a distributional
comparison, which is a much weaker claim and would not announce itself.  The
join key is ``sample_id``, which is stable by construction.
"""

from __future__ import annotations

# BLAS pinning must precede any NumPy import in this process AND is re-applied
# in each worker (ruling R-22.1).  16 workers each spawning a thread pool
# oversubscribes the box 16-fold and runs slower than 4 workers.
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                              # noqa: E402
import hashlib                                               # noqa: E402
import json                                                  # noqa: E402
import pathlib                                               # noqa: E402
import sys                                                   # noqa: E402
import time                                                  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ball_socket import campaign as C                        # noqa: E402
import ball_socket.campaign_config_v6 as v6                  # noqa: E402

OUT = _REPO / "results" / "ch9"

from ball_socket import mixed as _mx                          # noqa: E402

#: The registered traversal, expanded into the solves it implies.  Arms are
#: passed EXPLICITLY -- registered_contact_models() has no defaults, because a
#: default is a choice that does not look like one (box E sec 2.3).
_MODELS = _mx.expand_for_solving(
    _mx.registered_contact_models(branch="psi", summit_convention="derived"))

#: PRODUCTION GRID, re-selected at the CORRECTED bracket (ruling R-35.5).
#:
#: 48x36 was chosen by the audit that -- like both campaigns -- ran at
#: eps_cap = 0.99 (erratum ERR-13).  Two metrics drove that choice: P_loss
#: (2.263 %) and legacy h_min (2.092 %) at 32x24 -> 48x36.  Neither survives:
#:
#:   * legacy h_min is NOT a criterion functional (R-35.4).  E14's term is
#:     legacy(N) - corrected(N) on the same rows at the same grid; a difference
#:     at fixed grid needs the two terms computed CONSISTENTLY, not either one
#:     independently converged.  It also converges fine -- first order, p = 0.87,
#:     measured ratios 1 : 0.530 : 0.546 against 1 : 0.500 : 0.500 predicted.
#:   * P_loss has observed order p = -0.20: it gets WORSE under refinement, so
#:     it has no converged grid in the candidate set or beyond it and cannot
#:     select 48x36 any more than 32x24.
#:
#: The criterion's functional set is the constraint inputs g1/g2/g3, and at
#: 32x24 -> 48x36 they read 1.447 / 0.324 / 0.091 % against a 2 % tolerance --
#: two of them at second order or better.
#:
#: reports/boxd_grid_audit_corrected_handback.md
GRID = (32, 24)

#: Registered limitations that travel with the 32x24 choice (R-35.6).  Recorded
#: here, next to the value, because a limitation kept only in a report is not
#: attached to the thing it limits.
GRID_LIMITATIONS = {
    "P_loss_not_converged": {
        "observed_order": -0.20,
        "scope": "E6 and every thermal-routing fraction inherit it",
        "not_scoped": ("g2 -- T_max_bnd converges (p = 0.67, 0.324 %) and the "
                       "temperature field is insensitive to P_loss's p95, "
                       "consistent with E3's inlet-dominated margin"),
        "action": ("re-derive E6 on a quantity that converges, or report it "
                   "with the non-convergence stated"),
    },
    "hotspot_location_unsatisfied": {
        "flips": "4/30 samples change rim/interior classification between grids",
        "consequence": ("rim/interior classification is NOT reported as a "
                        "converged quantity; E3 and E11 carry that"),
        "note": "it cannot select a grid either -- unsatisfied at every pair",
    },
    "T_max_bnd_sub_first_order": {
        "observed_order": 0.67,
        "why": ("the weakest of the three, and the one the convention predicts "
                "would be weakest: a two-point extrapolation, not a closed "
                "form like h_min's boundary offset"),
        "action": "record the order; do not assume second-order behaviour",
    },
}

#: The registered artefact name.  Must contain a box-D marker or the claims
#: gate does not arm -- see the module docstring and R-22.2.
ARTEFACT = "audit_boxE_mixed_N8192_seed20260727_32x24_corrected"

#: The paired pure-film control at the same seed, bracket and grid.
PURE_FILM_ARTEFACT = "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--name", default=None)
    # THE PURE-FILM ARM (R-35.9 step 4).  It runs first and it is not optional
    # scaffolding: box D's pure-film control and box E's mixed arm on IDENTICAL
    # Sobol points are what isolate the contact contribution with the bracket
    # held fixed.  The 0.99 pair did exactly that job and is retained (R-33.4);
    # the corrected pair has to be built the same way.
    ap.add_argument("--pure-film", action="store_true",
                    help="contact=None: the paired control arm, no contact models")
    args = ap.parse_args()

    seed = v6.AUDIT_SEED
    if seed != 20260727:
        raise SystemExit(f"registered AUDIT_SEED is {seed}, expected 20260727; "
                         f"pairing with box B would be broken.  STOP.")

    # The grid travels in the PAYLOAD, never via a module global: workers are
    # spawned on Windows and re-import campaign at its defaults, so a parent
    # assignment would run the campaign at 32x24 while reporting 48x36.
    # Verified, not assumed -- see ball_socket/tests/test_grid_reaches_workers.py

    if args.name is None:
        args.name = (PURE_FILM_ARTEFACT if args.pure_film else ARTEFACT)
    pq = OUT / f"{args.name}.parquet"
    ckpt = OUT / f"{args.name}_ckpt"

    print("BOX-E MIXED-LUBRICATION CAMPAIGN", flush=True)
    # THE BANNER MUST DESCRIBE THE RUN, not the module's mixed-arm defaults.
    # Printed unconditionally, it announced a 24-arm traversal and a declined
    # short-circuit while executing a pure-film control that has neither -- the
    # same shape as a manifest asserting a bracket the solve did not have.
    if args.pure_film:
        print(f"  arm               : PURE FILM (contact=None) -- paired control",
              flush=True)
        print(f"  traversal         : none; contact models are not constructed",
              flush=True)
        print(f"  granularity       : one row per sample", flush=True)
    else:
        print(f"  arm               : MIXED (contact models live)", flush=True)
        print(f"  traversal         : {len(_MODELS)} arms/sample (16 corners, doubt band doubled)", flush=True)
        print(f"  short-circuit     : DECLINED -- reports/boxe_errata.md ERR-12", flush=True)
        print(f"  granularity       : one row per (sample, model, arm)", flush=True)
    print(f"  bracket           : eps_cap DERIVED per row (ERR-13 repaired)", flush=True)
    print(f"  box D frozen      : {v6.FROZEN_ON}  DRAFT={v6.DRAFT}", flush=True)
    print(f"  extremum source   : {v6.CONSTRAINT_EXTREMUM_SOURCE}", flush=True)
    print(f"  grid              : {GRID[0]}x{GRID[1]}", flush=True)
    print(f"  seed              : {seed}  (SAME as box B -- paired)", flush=True)
    print(f"  workers           : {args.workers}  (BLAS threads pinned to 1)",
          flush=True)
    print(f"  N                 : {args.n}", flush=True)
    print(f"  artefact          : {pq}", flush=True)
    print(f"  checkpoint        : {ckpt}", flush=True)
    print(f"  resume            : {not args.no_resume}", flush=True)

    # THROUGH THE SHARED WRITER (R-35.7).  This driver's own existence check was
    # correct and had been here since box D -- and that is exactly the problem:
    # it protected this path and no other, so run_boxd_grid_audit.py, which
    # never had one, destroyed an artefact.  One guard, one place, every writer.
    from ball_socket.immutable import guarded_path, ArtefactExists
    try:
        guarded_path(str(pq), what="campaign artefact")
    except ArtefactExists as exc:
        raise SystemExit(str(exc))

    U = C.sobol_unit(args.n, seed)
    xs = [C.unit_to_x(u) for u in U]

    t0 = time.time()
    df = C.run_batch(xs, workers=args.workers, chunk_size=args.chunk_size,
                     checkpoint_dir=str(ckpt), resume=not args.no_resume,
                     progress=True, clearance_closure=True,
                     Ntheta=GRID[0], Nphi=GRID[1],
                     # BOX E: the registered contact-model traversal.  24 arms
                     # per sample -- 16 cross-product corners, with the 8
                     # doubt-band models expanded to both GT branches because
                     # box E sec 2.2 requires both to be carried.
                     # Pure film passes NO models, so the solver's contact
                     # defaults to None -- the same call box D made, now at the
                     # corrected bracket.  Explicit rather than by omission.
                     **({} if args.pure_film else {"models": _MODELS}))
    wall = time.time() - t0

    # write_parquet, NOT df.to_parquet: it sorts by sample_id and strips attrs,
    # which together are what make the sha256 reproducible.
    C.write_parquet(df, str(pq))
    sha = hashlib.sha256(pq.read_bytes()).hexdigest()

    summary = {
        "stage": ("boxD_purefilm_corrected" if args.pure_film
                  else "boxE_mixed_corrected"),
        "arm": "pure_film" if args.pure_film else "mixed",
        "eps_cap": "DERIVED (1 - EPS_CAP_MARGIN)/cos(theta_min) -- see ERR-13",
        "grid_selected_by": "R-35.5, reports/boxd_grid_audit_corrected_handback.md",
        "grid_limitations": GRID_LIMITATIONS,
        "n_requested": args.n, "n_rows": int(len(df)),
        "seed": seed, "grid": {"Ntheta": GRID[0], "Nphi": GRID[1]},
        "workers": args.workers, "wall_seconds": wall,
        "constraint_extremum_source": v6.CONSTRAINT_EXTREMUM_SOURCE,
        "parquet": str(pq), "parquet_sha256": sha,
    }
    summary.update(C.summarise(df) if hasattr(C, "summarise") else {})
    (OUT / f"{args.name}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # THE MANIFEST IS THE HOME FOR PROVENANCE (ruling R-22.5).  With attrs
    # stripped, nothing about the run survives inside the artefact itself, so
    # grid, workers, seed, code version, wall time and the artefact sha must be
    # recorded here or they are not recorded anywhere.
    try:
        C.write_reproducibility_manifest(
            str(pq), config=v6.config_dict(), seed=seed, stage="boxE_mixed",
            extra={"workers": args.workers, "wall_seconds": wall,
                   "grid": {"Ntheta": GRID[0], "Nphi": GRID[1]},
                   "grid_selected_by": "reports/boxd_grid_audit_handback.md",
                   "blas_threads_pinned": 1,
                   "paired_with_box_B_seed": seed,
                   "determinism": ("rows sorted by sample_id; df.attrs stripped "
                                   "at write -- see R-21.6 / R-22.5"),
                   "parquet_sha256": sha})
    except Exception as exc:                                  # never silent
        print(f"  MANIFEST FAILED: {type(exc).__name__}: {exc}", flush=True)

    print(f"\nwrote {pq}  ({len(df)} rows)", flush=True)
    print(f"  sha256      : {sha}", flush=True)
    print(f"  wall        : {wall/3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
