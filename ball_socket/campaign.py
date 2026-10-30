"""ball_socket/campaign.py -- campaign harness for the frozen Ch.9 campaign.

Sobol sampler, evaluator, parallel runner and parquet writer for the ball-socket
interface.  Mirrors the slipper template (`ar_bo_campaign.py`) in structure and
consumes the FROZEN configuration (`campaign_config.py`, DRAFT = False).

Two implementation conventions, both forced by the frozen registration and
neither of them a change to a registered quantity (recorded in the hand-back):

  * **sigma\\* band.**  g1 is registered as a BAND, so every sample is labelled
    at BOTH band ends (`_lo`, `_hi`) and the binding counts are reported for
    each.  One solve, two labels.
  * **fluid class.**  g2 is registered as per-fluid-class, but fluid class is
    not one of the eight input axes: the class changes only the g2 ceiling, not
    the physics.  Every sample is therefore labelled under BOTH ceilings
    (mineral 110 C, water-glycol 60 C).  One solve, two labels.

Outcomes follow the fourth-outcome discipline: ``NonConvergence`` is the only
thing mapped to NONCONVERGED, and nothing is clipped.

Binding attribution uses the STRICT single-constraint convention: a sample is
attributed to a constraint only if that constraint alone is violated; samples
violating more than one are tallied as ``multi``.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# BLAS THREAD PINNING, SET BEFORE NUMPY IS IMPORTED ANYWHERE (ruling R-30.2).
#
# This has to happen HERE, at the top of the module that pulls in NumPy, and
# NOT in the worker function.  ``_pin_blas_threads()`` ran inside the worker
# AFTER the child had already imported this module and therefore NumPy, so its
# ``setdefault`` could not change thread pools that already existed.  It was
# decorative, and its docstring claimed otherwise.
#
# Measured, from inside a worker (threadpool_info):
#     parent env unset -> child sees mkl=20, openmp=28   -- 16x oversubscribed
#     parent env set   -> child sees mkl=1,  openmp=1
#
# Environment variables DO propagate through Windows spawn, so setting them
# before the import is what actually pins the children.  Until now that
# protection was a property of ONE DRIVER (run_boxd_campaign.py sets them at
# its own top); any other entry point calling run_batch got 16 workers each
# running a 20-thread BLAS on a 20-core box.  Putting it here makes the pin a
# property of the code that spawns, not of whoever happens to call it.
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import functools
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.solver import SolverConfig

# BOX B is the live registration (v4, frozen 2026-07-27, config sha 37632225...).
# Box A (`campaign_config.py`) is retained, superseded, and its results are
# already written to parquet; it is not re-run.
from .campaign_config_v4 import (INPUT_SPACE, ConstraintSet, GEOMETRY,
                                 SIGMA_STAR_BAND_M, ALPHA_FIXED, SQUEEZE_PHASE,
                                 SPIN_RATIO, P_CASE_PA)
from .geometry import BallSocketGeometry
from .outcomes import (NonConvergence, ClearanceNonConvergence, ContactLimit,
                       Outcome)
from .solver import BallSocketSolver, CampaignInput, Performance

NTHETA, NPHI = 32, 24
BANDS = {"lo": SIGMA_STAR_BAND_M[0], "hi": SIGMA_STAR_BAND_M[1]}
CLASSES = ("mineral", "water_glycol")
_CONSTRAINTS = ConstraintSet()

# The registered extremum convention (box D).  Imported, never hard-coded:
# campaign_config_v6 is the single source of truth and this module must not be
# able to drift from it.  Falls back to the box-C behaviour if box D is not
# present, so this module still runs against box C unchanged.
try:
    from .campaign_config_v6 import CONSTRAINT_EXTREMUM_SOURCE
except Exception:                                    # pragma: no cover
    CONSTRAINT_EXTREMUM_SOURCE = "legacy_interior"

# THE REGISTERED BRACKET (ruling R-33.1).  Box D registered
#     eps_cap(theta_min) = (1 - EPS_CAP_MARGIN) / cos(theta_min)
# as a DERIVED key and named 0.99 as the value it corrects.  NOTHING CALLED IT.
# BallSocketSolver.__init__ defaults eps_cap=0.99, campaign.py never passed the
# parameter, and so box D, box E and the grid audit all ran at box C's bracket
# under manifests asserting the correction -- 115 hours of it.
#
# NO SILENT FALLBACK HERE, deliberately.  The obvious `except: eps_cap = 0.99`
# would reproduce the defect exactly: a module that runs at the uncorrected
# bracket whenever the registration is unreachable, and says nothing about it.
# If the registration cannot be read, the campaign must not run.
from .campaign_config_v6 import eps_cap as _registered_eps_cap  # noqa: E402


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def sobol_unit(n: int, seed: int) -> np.ndarray:
    """Scrambled Sobol points in the unit cube, d = len(INPUT_SPACE)."""
    return qmc.Sobol(d=len(INPUT_SPACE), scramble=True, seed=seed).random(n)


def unit_to_x(u: Iterable[float]) -> dict[str, float]:
    return {d.name: float(d.from_unit(uu)) for d, uu in zip(INPUT_SPACE, u)}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _labels(*, h_min: float, T_max: float, De_sq: float,
            sigma_star: float, fluid_class: str) -> tuple[str, str]:
    """(outcome, binding) under the strict single-constraint convention.

    THE EXTREMUM TRIPLE IS PASSED EXPLICITLY AND KEYWORD-ONLY (ruling R-17.5).

    This function deliberately does NOT take ``perf`` and does NOT know the name
    of any field on it.  Box D reports each of h_min, De_sq and T_max TWICE --
    the legacy interior-only value and the boundary-corrected one -- and which
    of the two decides feasibility is a REGISTERED CHOICE
    (``constraint_extremum_source`` in campaign_config_v6).  A function that
    reached into ``perf`` for a fixed attribute name would make that choice
    silently, by plumbing, which is the eps_cap defect one level up: an
    unregistered choice determining a registered outcome.

    Keyword-only so no caller can pass the triple in the wrong order, and so
    adding an argument can never silently re-bind an existing one.

    The CONSTRAINT DEFINITION -- the thresholds and the functional form -- is
    unchanged from box B and is still read from ``_CONSTRAINTS``.  What box D
    changes is only WHICH MEASUREMENT is fed to it.
    """
    g1 = (h_min / sigma_star) >= _CONSTRAINTS.lambda_ratio_min
    g2 = T_max <= _CONSTRAINTS.T_max_C_by_class[fluid_class]
    g3 = De_sq <= _CONSTRAINTS.De_sq_max
    if g1 and g2 and g3:
        return Outcome.FEASIBLE.value, "none"
    violated = [n for n, ok in (("g1", g1), ("g2", g2), ("g3", g3)) if not ok]
    return Outcome.INFEASIBLE.value, (violated[0] if len(violated) == 1 else "multi")


#: Which extremum convention decides feasibility.  Registered in box D as
#: ``constraint_extremum_source``; imported here rather than hard-coded so the
#: registration is the single source of truth and this module cannot drift
#: from it.
def _extremum_triple(perf, source: str) -> tuple[float, float, float]:
    """(h_min, T_max, De_sq) under the NAMED extremum convention.

    The two conventions are resolved HERE, in one place, by an explicit name --
    never by an attribute default anywhere else in the codebase.
    """
    if source == "legacy_interior":
        return float(perf.h_min), float(perf.T_max), float(perf.De_sq)
    if source == "boundary_corrected":
        return float(perf.h_min_bnd), float(perf.T_max_bnd), float(perf.De_sq_bnd)
    raise ValueError(
        "unknown constraint_extremum_source %r: expected 'boundary_corrected' "
        "(box D) or 'legacy_interior' (box B/C comparison label)" % (source,))


def _as_row_columns() -> tuple:
    """Every key any outcome exception can put in a row, ENUMERATED from the
    classes rather than hand-listed (ruling R-33.3).

    The hand-list this replaces named seven columns.  The three exception
    classes between them return THIRTEEN keys, and the six it omitted included
    ``eps_cap_at_failure`` -- the one column that would have shown, on all
    158,544 contact_limit rows, that the campaign was running at 0.99 while its
    manifest asserted the corrected bracket.

    That is two partial applications intersecting on one value: R-28.3 said
    derive from the structure and was applied to ``Performance``'s fields and
    not to ``as_row()``; R-23.6 said record what ran and was applied to six
    parameters and not to ``eps_cap``.  Deriving the set is the only fix that
    does not depend on somebody noticing the next omission.
    """
    keys: set = set()
    for cls in (NonConvergence, ClearanceNonConvergence, ContactLimit):
        try:
            keys |= set(cls("probe").as_row().keys())
        except Exception:                              # pragma: no cover
            continue
    # row-name aliases the branches also set, retained so no existing consumer
    # loses a column; purely additive, nothing renamed or removed.
    keys |= {"F_film_max", "F_contact", "c_last", "c_prev", "T_max_last"}
    return tuple(sorted(keys))


#: Columns every row carries whatever its outcome, beyond Performance's own
#: fields.  DERIVED from the exception classes (R-33.3), never hand-listed.
_DERIVED_ROW_COLUMNS = _as_row_columns()


def _ensure_schema(row: dict) -> dict:
    """One schema across every outcome, filled at a SINGLE exit.

    Five return paths each hand-filled their own columns, so converged rows
    carried F_contact/F_film_max/c_last and contact_limit rows carried
    F_film_at_limit/F_asp_at_limit, and neither carried the other's.  A ragged
    parquet makes any tally over outcomes compare a value against a missing
    one -- the same defect that made a resume test report 76/96 spurious
    disagreements.

    Derived from Performance's fields plus the companions above, so a new field
    propagates automatically (R-28.3): never hand-enumerate what the structure
    can yield.
    """
    import dataclasses as _dc
    for f in _dc.fields(Performance):
        row.setdefault(f.name, np.nan)
    for c in _DERIVED_ROW_COLUMNS:
        row.setdefault(c, np.nan)
    return row


def _evaluate_x_inner(x: dict[str, float], *, spin_ratio: float = SPIN_RATIO,
               theta_min_deg: float | None = None,
               theta_max_deg: float | None = None,
               alpha_th_rel: float = 1.0,
               clearance_closure: bool = False,
               delta_T_datum: str = "bulk",
               Ntheta: int | None = None,
               Nphi: int | None = None,
               contact: Any = None,
               flow_factors: dict | None = None,
               brentq_xtol: float | None = None,
               brentq_rtol: float | None = None,
               memory_sign: str | None = None) -> dict[str, Any]:
    """Evaluate one campaign sample.  Never raises: the fourth outcome is
    recorded as a row, exactly as the classifier consumes it."""
    geom_kw: dict[str, Any] = {}
    # BOX B: theta_min is an INPUT AXIS, so it arrives inside x.  An explicit
    # keyword (box-A OAT sweep) still overrides.
    if theta_min_deg is None and "theta_min_deg" in x:
        theta_min_deg = float(x["theta_min_deg"])
    if theta_min_deg is not None:
        geom_kw["theta_min"] = np.deg2rad(theta_min_deg)
    if theta_max_deg is not None:
        geom_kw["theta_max"] = np.deg2rad(theta_max_deg)
    geom = BallSocketGeometry(c0=x["c0"], **geom_kw)
    _eps_cap_registered = float(_registered_eps_cap(geom.theta_min))

    slv = BallSocketSolver(
        # GRID PASSED EXPLICITLY, NOT READ FROM A MODULE GLOBAL.
        # ProcessPoolExecutor on Windows SPAWNS: each worker re-imports this
        # module and sees NTHETA/NPHI at their module-level defaults.  A parent
        # that assigns `campaign.NTHETA = 48` therefore runs its workers at
        # 32x24 while believing it set 48x36 -- silently, hash-stably, and with
        # the manifest recording the grid the parent intended.  The value must
        # travel in the payload to reach a worker.
        geom, model="viscoelastic",
        Ntheta=(NTHETA if Ntheta is None else int(Ntheta)),
        Nphi=(NPHI if Nphi is None else int(Nphi)),
        alpha=ALPHA_FIXED, squeeze_phase=SQUEEZE_PHASE, spin_ratio=spin_ratio,
        config=SolverConfig(include_normal_stress=False, max_iter=80, tol=1e-9,
                            omega=0.7),
        # THE REGISTERED BRACKET, PASSED (R-33.1).  Not a default, not a module
        # constant: read from box D's DERIVED formula for THIS row's theta_min.
        # Registering a value and never passing it is what produced 115 hours at
        # the wrong bracket, and a key in config_dict() proved only that
        # somebody had written it down.
        eps_cap=_eps_cap_registered,
        # BOX C.  Default False reproduces the box-B call exactly.
        clearance_closure=clearance_closure, delta_T_datum=delta_T_datum,
        # BOX E.  The asperity load path, which box C built, box C/D registered
        # and NOTHING ON THE CAMPAIGN PATH CALLED.  evaluate_x had no contact
        # parameter at all, so the solver defaulted to contact=None and the
        # box-D run was pure film against a registration that reserved a
        # `contact_limit` outcome.  Explicit and defaulting to None, so the
        # pure-film control arm remains reproducible by omission.
        contact=contact,
        # R-56.1: box D's "resolution-study provision" (campaign_config_v6
        # SOLVER_NUMERICS["brentq"]).  None -> the solver's registered default
        # (1e-4 / 1e-4), so every existing call is unchanged; the value that ran
        # is recorded below as realised_brentq_xtol / _rtol.
        **({} if brentq_xtol is None else {"brentq_xtol": float(brentq_xtol)}),
        **({} if brentq_rtol is None else {"brentq_rtol": float(brentq_rtol)}))
    # BOX F sec 5.2 / R-40: the film model, SET ON THE SOLVER and read back
    # off it below (realised_film_model).  None is the registered smooth film;
    # a dict ({"sigma", "below_floor"}) is the averaged film used only for the
    # marginal-band re-solve.  Explicit and defaulting to None, so every
    # campaign row is reproduced by omission -- the same pattern as `contact`.
    slv.flow_factors = (None if flow_factors is None else dict(flow_factors))
    # D-6 (R-58.1): None leaves the solver's default ("modulus", every registered
    # campaign); "signed" applies Paper 2's sign law.  Read back off the solver below.
    if memory_sign is not None:
        slv.memory_sign = str(memory_sign)
    if alpha_th_rel != 1.0:                     # OAT robustness on the one
        from .geometry import SocketMaterial    # material channel
        base = SocketMaterial()
        slv.socket_alpha_th = base.alpha_th * alpha_th_rel

    # ---- REALISED PARAMETERS, RECORDED FROM INSIDE THE WORKER (R-23.6) ----
    # Record what RAN, not what the parent set.  The spawn bug wrote a manifest
    # saying 48x36 while every worker ran 32x24: hash-stable, reproducible and
    # wrong, at the grid Phase 3 had just rejected.  It was caught only because
    # serial and pooled runs disagreed on this platform -- had every path
    # spawned, all runs would have agreed at the wrong grid and the determinism
    # check would have passed.  A gate on agreement cannot detect a consensus
    # error; a column recording the realised value can.
    #
    # These are read back OFF THE CONSTRUCTED OBJECTS, not off the arguments,
    # so a value that failed to travel is visible in the artefact itself rather
    # than inferable only by comparing two runs on two platforms.
    _realised = {
        # READ OFF THE SOLVER, WITH NO FALLBACK TO THE ARGUMENT (R-33.2).
        # This previously read `slv.mesh.Ntheta if hasattr(slv, "mesh") else
        # <the argument>`.  BallSocketSolver has no `mesh` attribute -- the grid
        # lives at slv.Ntheta -- so the guard was ALWAYS false and the column
        # always echoed the argument it was written to cross-check.  It reported
        # 48x36 correctly, but by restating the input, which is precisely what
        # the comment above it says this column must never do.  Gate 4 found it.
        "realised_Ntheta": int(slv.Ntheta),
        "realised_Nphi": int(slv.Nphi),
        "realised_extremum_source": CONSTRAINT_EXTREMUM_SOURCE,
        "realised_contact_model": (contact.label() if contact is not None
                                   else "none"),
        "realised_clearance_closure": bool(clearance_closure),
        "realised_delta_T_datum": str(delta_T_datum),
        # RECORD WHAT RAN, not what was set.  The pin was decorative in the
        # worker and real only at import; a column makes the difference visible
        # in the artefact instead of inferable from a timing anomaly.
        "realised_blas_threads": _realised_blas_threads(),
        # THE SEVENTH PARAMETER (R-23.6, applied fully this time).  Read off the
        # CONSTRUCTED SOLVER, not off the argument: the whole point is that a
        # value which failed to arrive is visible in the artefact rather than
        # reconstructible only by re-solving after the fact.
        "realised_eps_cap": float(getattr(slv, "eps_cap", np.nan)),
        "registered_eps_cap": _eps_cap_registered,
        "realised_eps_lo": float(getattr(slv, "eps_lo", np.nan)),
        "realised_brentq_xtol": float(getattr(slv, "brentq_xtol", np.nan)),
        "realised_brentq_rtol": float(getattr(slv, "brentq_rtol", np.nan)),
        "realised_picard_max_iter": int(getattr(slv.config, "max_iter", -1)),
        "realised_picard_tol": float(getattr(slv.config, "tol", np.nan)),
        "realised_picard_omega": float(getattr(slv.config, "omega", np.nan)),
        # BOX F sec 3.4: the film model, read off the constructed solver.
        "realised_film_model": ("smooth" if getattr(slv, "flow_factors", None) is None
                                else "averaged:%s:sigma=%.6e" % (
                                    slv.flow_factors.get("below_floor"),
                                    float(slv.flow_factors.get("sigma")))),
        "realised_include_normal_stress":
            bool(getattr(slv.config, "include_normal_stress", True)),
        "realised_memory_sign": str(getattr(slv, "memory_sign", "modulus")),
    }

    row: dict[str, Any] = dict(x)
    row.update(**_realised)
    row.update(spin_ratio=spin_ratio, alpha_th_rel=alpha_th_rel,
               theta_min_deg=float(np.degrees(geom.theta_min)),
               theta_max_deg=float(np.degrees(geom.theta_max)),
               clearance_closure=clearance_closure,
               delta_T_datum=(delta_T_datum if clearance_closure else "none"))
    ci = CampaignInput(omega=x["omega"], p_ch=x["p_ch"], c0=x["c0"],
                       T_in=x["T_in"], eta_0=x["eta_0"], lam=x["lam"],
                       eta_s_over_eta0=x["eta_s_ratio"],
                       # BOX B couples the load: k_F_bs absent -> None ->
                       # F_app = p_ch * A_p.  BOX A passes it through.
                       k_F_bs=x.get("k_F_bs"),
                       p_case=P_CASE_PA)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perf = slv.solve_equilibrium(ci)
    except ClearanceNonConvergence as exc:
        # A DISTINCT outcome from nonconverged, never pooled with it: one is a
        # film that cannot carry the load, the other is an outer iteration that
        # did not settle.  Typed columns, not a formatted string (rule 9).
        _cf = exc.as_row()
        row.update(_cf)                      # whole payload first (R-33.3)
        row.update(outcome="clearance_nonconverged", converged=False,
                   fail_reason=str(exc)[:120],
                   F=np.nan, h_min=np.nan, T_max=np.nan, P_loss=np.nan,
                   Q_leak=np.nan, De_sq=np.nan, eps_ecc=np.nan,
                   thermal_mode="", thermal_route="",
                   F_app=np.nan, F_pocket=np.nan, F_hydrodynamic=np.nan,
                   F_film_max=np.nan, F_contact=np.nan,
                   c_equilibrium=np.nan, T_bulk=np.nan,
                   n_outer=_cf.get("n_outer_at_failure", np.nan),
                   c_last=_cf.get("c_last", np.nan),
                   c_prev=_cf.get("c_prev", np.nan),
                   T_max_last=_cf.get("T_max_last_at_failure", np.nan))
        row.update(h_min_bnd=np.nan, T_max_bnd=np.nan, De_sq_bnd=np.nan)
        for b in BANDS:
            row[f"lambda_ratio_{b}"] = np.nan
            row[f"lambda_ratio_{b}_legacy"] = np.nan
            for c in CLASSES:
                # A non-converged row fails in the EQUILIBRIUM SEARCH, not in
                # extremum reporting, so it is non-converged under BOTH
                # conventions.  The _legacy columns are mirrored rather than
                # omitted: a ragged schema would make E14 compare a label
                # against a missing value and count it as a disagreement.
                row[f"outcome_{c}_{b}"] = "clearance_nonconverged"
                row[f"binding_{c}_{b}"] = ""
                row[f"outcome_{c}_{b}_legacy"] = "clearance_nonconverged"
                row[f"binding_{c}_{b}_legacy"] = ""
        return row
    except NonConvergence as exc:
        # The load state at the failure is recorded as TYPED COLUMNS, not left
        # to survive inside a truncated message (2026-08-03).  fail_reason is
        # still truncated because it is diagnostic prose; the numbers no longer
        # depend on it.  Where the film closed before eps_cap there is no
        # F_film_max and the column is NaN, which is distinguishable from zero.
        _fail = getattr(exc, "as_row", lambda: {})()
        row.update(_fail)                    # whole payload first (R-33.3)
        row.update(outcome="nonconverged", converged=False, fail_reason=str(exc)[:120],
                   F=np.nan, h_min=np.nan, T_max=np.nan, P_loss=np.nan,
                   Q_leak=np.nan, De_sq=np.nan, eps_ecc=np.nan,
                   thermal_mode="", thermal_route="",
                   F_app=_fail.get("F_app_at_failure", np.nan),
                   F_pocket=_fail.get("F_pocket_at_failure", np.nan),
                   F_hydrodynamic=_fail.get("F_target_at_failure", np.nan),
                   F_film_max=(_fail.get("F_film_max")
                               if _fail.get("F_film_max") is not None else np.nan),
                   F_contact=((_fail["F_target_at_failure"] - _fail["F_film_max"])
                              if _fail.get("F_film_max") is not None else np.nan),
                   c_equilibrium=np.nan, T_bulk=np.nan, n_outer=np.nan,
                   c_last=np.nan, c_prev=np.nan, T_max_last=np.nan)
        row.update(h_min_bnd=np.nan, T_max_bnd=np.nan, De_sq_bnd=np.nan)
        for b in BANDS:
            row[f"lambda_ratio_{b}"] = np.nan
            row[f"lambda_ratio_{b}_legacy"] = np.nan
            for c in CLASSES:
                # A non-converged row fails in the EQUILIBRIUM SEARCH, not in
                # extremum reporting, so it is non-converged under BOTH
                # conventions.  The _legacy columns are mirrored rather than
                # omitted: a ragged schema would make E14 compare a label
                # against a missing value and count it as a disagreement.
                row[f"outcome_{c}_{b}"] = "nonconverged"
                row[f"binding_{c}_{b}"] = ""
                row[f"outcome_{c}_{b}_legacy"] = "nonconverged"
                row[f"binding_{c}_{b}_legacy"] = ""
        return row
    except ContactLimit as exc:
        # THE REGISTERED OUTCOME, NOT `error` (ruling R-28.2).  Box C, D and E
        # all carry `contact_limit` as a DISTINCT taxonomy value, strictly
        # stronger than box B's blow-up: the load exceeds what the film PLUS
        # fully developed, plastically capped asperity contact can carry.
        #
        # Without this branch it fell through to `except Exception` and landed
        # as `error` -- and `error` is precisely the label a reader discards.
        # The campaign's central physical finding would have been filed as a
        # crash.
        _cl = getattr(exc, "as_row", lambda: {})()
        # CARRY THE WHOLE PAYLOAD (R-33.3).  This previously hand-picked two of
        # the seven keys as_row() returns and dropped the rest on the next line
        # -- including eps_cap_at_failure, which is the column R-32.3 then had
        # to reconstruct by re-solving because the artefact could not answer it.
        row.update(_cl)
        row.update(outcome="contact_limit", converged=False,
                   fail_reason=str(exc)[:160],
                   F_film_at_limit=_cl.get("F_film_at_limit", np.nan),
                   F_asp_at_limit=_cl.get("F_asp_at_limit", np.nan))
        row.update({f.name: np.nan for f in __import__("dataclasses").fields(Performance)
                    if f.name not in row})
        for b in BANDS:
            row[f"lambda_ratio_{b}"] = np.nan
            row[f"lambda_ratio_{b}_legacy"] = np.nan
            for c in CLASSES:
                row[f"outcome_{c}_{b}"] = "contact_limit"
                row[f"binding_{c}_{b}"] = ""
                row[f"outcome_{c}_{b}_legacy"] = "contact_limit"
                row[f"binding_{c}_{b}_legacy"] = ""
        return row
    except Exception as exc:                    # never silently swallowed
        row.update(outcome="error", converged=False, fail_reason=f"{type(exc).__name__}: {exc}"[:160])
        row.update(h_min_bnd=np.nan, T_max_bnd=np.nan, De_sq_bnd=np.nan)
        for b in BANDS:
            row[f"lambda_ratio_{b}"] = np.nan
            row[f"lambda_ratio_{b}_legacy"] = np.nan
            for c in CLASSES:
                row[f"outcome_{c}_{b}"] = "error"
                row[f"binding_{c}_{b}"] = ""
                row[f"outcome_{c}_{b}_legacy"] = "error"
                row[f"binding_{c}_{b}_legacy"] = ""
        return row

    # EVERY Performance FIELD REACHES THE ROW, BY ENUMERATION (ruling R-28.3).
    #
    # This used to be a hand-written field-by-field copy, and that list is the
    # same defect three times over.  The solver computed the corrected extrema
    # and the constraints read the legacy ones (R-17); it computed h_min_bnd and
    # nothing read it (R-22); it computed F_asp, area_frac_contact and
    # contact_model and the row carried NONE of them, so a contact-limited
    # campaign would have recorded no asperity load at all.
    #
    # Three failures at one seam -- between what the solver produces and what
    # the row records -- is structural, not coincidence.  Enumerating the
    # dataclass closes it: a new field on Performance now reaches the artefact
    # automatically and CANNOT be forgotten.  Never hand-enumerate what the
    # structure can yield.
    import dataclasses as _dc
    row.update(outcome="converged", converged=True, fail_reason="")
    for _f in _dc.fields(perf):
        row[_f.name] = getattr(perf, _f.name)
    row.update(
        # Derived companions the failure path also carries, so the contact
        # dataset is ONE SCHEMA across every outcome: at equilibrium the film
        # carries the whole hydrodynamic share and the residual is exactly zero.
        F_film_max=perf.F_hydrodynamic, F_contact=0.0,
        c_last=np.nan, c_prev=np.nan, T_max_last=np.nan)
    # ONE SOLVE, TWO LABELS -- the phrase prereg v6 sec 8 already uses for the
    # sigma* band ends, now applied to the extremum convention (ruling R-17.5).
    # The REGISTERED convention (boundary_corrected) supplies the unsuffixed
    # columns, so every downstream consumer gets the corrected labelling by
    # default; the legacy labelling is carried alongside under an explicit
    # _legacy suffix, on IDENTICAL rows, so E14's per-correction decomposition
    # is a genuine measurement rather than a difference between two campaigns.
    h_c, T_c, D_c = _extremum_triple(perf, CONSTRAINT_EXTREMUM_SOURCE)
    h_l, T_l, D_l = _extremum_triple(perf, "legacy_interior")
    for b, sig in BANDS.items():
        row[f"lambda_ratio_{b}"] = h_c / sig
        row[f"lambda_ratio_{b}_legacy"] = h_l / sig
        for c in CLASSES:
            out, bind = _labels(h_min=h_c, T_max=T_c, De_sq=D_c,
                                sigma_star=sig, fluid_class=c)
            row[f"outcome_{c}_{b}"] = out
            row[f"binding_{c}_{b}"] = bind
            out_l, bind_l = _labels(h_min=h_l, T_max=T_l, De_sq=D_l,
                                    sigma_star=sig, fluid_class=c)
            row[f"outcome_{c}_{b}_legacy"] = out_l
            row[f"binding_{c}_{b}_legacy"] = bind_l
    return row



#: The columns that uniquely identify a row once the campaign runs a traversal
#: (ruling R-29.1).  With 24 rows per sample, ``sample_id`` alone is NOT unique
#: and pandas' stable sort then leaves the within-group order equal to the
#: INPUT order -- which under ``as_completed`` is nondeterministic.  Sorting on
#: sample_id alone would silently undo R-21/R-22's whole reproducibility chain,
#: and it would read as a regression in the sort rather than as a consequence
#: of the granularity decision.
ROW_SORT_KEY = ("sample_id", "model_id", "arm")

#: Every registered constraint's quantity, under the convention box D sec 3.5
#: registered (`constraint_extremum_source = "boundary_corrected"`), plus the
#: outcome and the eccentricity a bracket verdict is reconstructed from.
#:
#: CONVENTION (R-53.5): a bespoke runner that calls `evaluate_x` PERSISTS ALL OF
#: THESE, whether or not the expectation it was written for needs them.  The
#: constraints are registered before the run, so the set worth keeping is known
#: before the run; the cost is disk.  `run_E10_capped_area` kept only its own
#: columns, so T_max_bnd was computed at all 3,840 rated-corner rows and
#: discarded -- which is why E5's VG32 clause now costs 960 solves to recover a
#: quantity that had already been solved (ERR-25).
#: `scripts/run_R53_persist_audit.py` reads this set and audits every runner.
REGISTERED_CONSTRAINT_COLUMNS = ("outcome", "h_min_bnd", "T_max_bnd", "De_sq_bnd",
                                 "eps_ecc")


def evaluate_x_traversal(x: dict[str, float], *, models,
                         short_circuit: bool = False, **kw) -> list[dict]:
    """One sample, evaluated across the registered contact-model traversal.

    Returns ONE ROW PER (sample, model, arm) -- box E's registered granularity.
    Each row therefore has exactly one unambiguous outcome, and the doubt-band
    pair form (lower/upper feasible counts) becomes a tally-time operation over
    rows rather than an ambiguity inside one.

    THE REGISTERED SHORT-CIRCUIT (box E sec 2.1), and why it is exact.
    Greenwood-Tripp's F_5/2 is negligible by H ~ 4.  Solve first at the
    SOONEST-ENGAGING model -- the largest sigma_summit, hence the smallest
    H = h/sigma_s at any film, hence the largest asperity load of the set.  If
    THAT model carries no asperity load at equilibrium, every other model in
    the cross-product engages later still and carries none either, so all of
    them reduce to the same pure-film solve.  Exact, not approximate.

    It also collapses the doubt band on those rows: with F_asp identically zero
    the elastic and plastic arms give the same answer, so the 8 undetermined
    models stop being 16 solves.  The short-circuit removes the doubling
    exactly where the doubt is immaterial.

    Registered as MANDATORY to verify rather than to argue -- a short-circuit
    that is NEARLY exact is the defect class this project keeps finding.
    """
    from . import mixed as _mx

    # THE SHORT-CIRCUIT IS DISABLED, AND MEASUREMENT IS WHY (box E sec 2.1).
    #
    # Box E registers it as an optimisation whose exactness verification is
    # MANDATORY.  Probed on 24 stratified samples at 32x24, NO FIRING CASE WAS
    # OBSERVED -- results/ch9/boxe_shortcircuit_verification.json.
    #
    # THAT IS NOT A CLAIM THAT THE RATE IS ZERO.  Zero observations in 24 puts
    # the 95 per cent upper bound at 11.7 per cent -- as many as ~960 of 8192
    # samples could fire.  The decision does not depend on the rate, which is
    # precisely why the weaker statement is the one recorded: a conclusion
    # should not be stated more strongly than its sample supports, least of all
    # when the outcome is unaffected either way.
    #
    # Two independent legs, either sufficient:
    #   * with no firing case the MANDATORY exactness verification could not be
    #     performed at all, and shipping an unverified short-circuit is exactly
    #     the "nearly exact" defect the registration names;
    #   * it costs a probe solve per sample, so enabled it makes the campaign
    #     ~4 per cent SLOWER than solving all arms unconditionally.
    #
    # Recorded as a deviation in reports/boxe_errata.md (E-12), where box E's
    # readers will look -- not only in a results JSON.  A registered choice
    # declined on evidence gets erratum treatment.
    #
    # The flag is retained: if a later run finds firing samples the
    # verification becomes possible and the optimisation can be adopted under
    # its existing registration, without a new box.
    if short_circuit:
        pivot = _mx.soonest_engaging([m for m, _ in models])
        probe = evaluate_x(x, contact=pivot, **kw)
        engaged = bool(probe.get("F_asp") or probe.get("area_frac_contact"))
    else:
        probe, engaged = None, True

    rows: list[dict] = []
    for idx, (model, provenance) in enumerate(models):
        if short_circuit and not engaged and probe.get("converged"):
            row = dict(probe)                      # exact: all models coincide
            row["short_circuited"] = True
        else:
            row = evaluate_x(x, contact=model, **kw)
            row["short_circuited"] = False
        row["model_id"] = idx
        row["arm"] = provenance
        row["contact_model_label"] = model.label()
        rows.append(row)
    return rows


def _realised_blas_threads() -> str:
    """The thread counts this process ACTUALLY has, read from the runtime.

    Not the environment variable -- the pool.  An env var says what was asked
    for; threadpool_info says what exists.  Cheap enough to record per row and
    it is the only evidence that survives into the artefact.
    """
    try:
        import threadpoolctl
        return ";".join(f"{p['internal_api']}={p['num_threads']}"
                        for p in threadpoolctl.threadpool_info())
    except Exception:
        return "|".join(f"{v}={os.environ.get(v, '?')}" for v in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS"))


def _worker(args):
    x, kw = args
    return evaluate_x(x, **kw)


def _pin_blas_threads() -> None:
    """Belt-and-braces only -- THE REAL PIN IS AT MODULE IMPORT (R-30.2).

    Kept because it is harmless and documents intent, but it CANNOT be what
    protects the run: by the time a worker calls it, importing this module has
    already imported NumPy and created its thread pools, and ``setdefault`` on
    an environment variable does not resize an existing pool.  Verified from
    inside a worker with ``threadpool_info()``.

    Original note follows (ruling R-22.1).

    With N worker processes each spawning its own BLAS thread pool, the box is
    oversubscribed N-fold and a 16-worker run can be SLOWER than a 4-worker one:
    the pools fight for the same cores and thrash cache. Setting these costs
    nothing if the operators never reach threaded BLAS, and costs the entire
    speed-up if they do and it is left unset.

    Must run BEFORE NumPy is imported in the child to take effect, which on a
    spawn-based pool means at the top of the worker rather than in the parent.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


def _worker_ided(args):
    """Worker that stamps the row with its own sample_id.

    The id is assigned at SUBMIT time from the sample's position in the input
    list, so it is a property of the input rather than of the schedule.  Two
    runs of the same inputs therefore produce the same ids regardless of how
    the pool happens to interleave them -- which is what makes the artefact
    hash reproducible (ruling R-21.6) and what makes resume able to tell which
    samples are already done.
    """
    sid, x, kw = args
    _pin_blas_threads()
    # COPY, NEVER POP.  `payload` shares ONE kw dict across every sample, so
    # popping mutated it for all of them: the first serial call consumed
    # "models" and every later call saw None and returned a single pure-film
    # row instead of the traversal -- 24 + 1 + 1 + 1 = 27 rows where 96 were
    # expected.  Invisible in the pooled path, where each worker gets a pickled
    # COPY, so serial and parallel silently diverged.  Second time the
    # determinism check has caught a serial/parallel split; a gate comparing
    # runs to each other earns its keep exactly here.
    kw = dict(kw)
    models = kw.pop("models", None)
    if models is None:
        rows = [evaluate_x(x, **kw)]
    else:
        # A SAMPLE, NOT A ROW, is the unit of work.  That keeps chunks
        # sample-aligned by construction (R-29.1): part files are named by
        # their first sample_id, which stays unique only if a sample is never
        # split across a chunk boundary.  Chunking on rows would break resume's
        # part naming silently.
        rows = evaluate_x_traversal(x, models=models, **kw)
    for r in rows:
        r["sample_id"] = int(sid)
    return rows


def _sort_rows(df: "pd.DataFrame") -> "pd.DataFrame":
    """Canonical row order, on as much of ROW_SORT_KEY as the frame carries.

    Sorting on ``sample_id`` alone was correct while a sample produced one row.
    Under box E's (sample, model, arm) granularity it is not unique, and
    pandas' stable sort then preserves the INPUT order within each group --
    which under ``as_completed`` is nondeterministic.  Identical content, two
    hashes, exactly the failure R-22 fixed.
    """
    keys = [k for k in ROW_SORT_KEY if k in df.columns]
    if not keys:
        return df.copy()
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _completed_ids(ckpt_dir: str) -> tuple[set[int], list[str]]:
    """sample_ids already persisted in a checkpoint directory, and the parts."""
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return set(), []
    parts = sorted(glob.glob(os.path.join(ckpt_dir, "part-*.parquet")))
    done: set[int] = set()
    good: list[str] = []
    for f in parts:
        try:
            ids = pd.read_parquet(f, columns=["sample_id"])["sample_id"]
        except Exception as exc:               # truncated part from a hard kill
            print(f"  checkpoint: DISCARDING unreadable part {os.path.basename(f)}"
                  f" ({type(exc).__name__}); its samples will be re-run")
            continue
        done.update(int(i) for i in ids)
        good.append(f)
    return done, good


def run_batch(xs: list[dict[str, float]], *, workers: int = 12,
              chunk_size: int | None = None, checkpoint_dir: str | None = None,
              resume: bool = True, progress: bool = True,
              **kw) -> pd.DataFrame:
    """Evaluate a list of samples in parallel; returns a DataFrame.

    THREE PROPERTIES THIS FUNCTION NOW GUARANTEES (ruling R-21.6):

    **Deterministic row order.**  Rows are sorted by ``sample_id`` before being
    returned, and ``sample_id`` is assigned at submit time from the input's
    position.  Before this, rows were appended in ``as_completed`` order, so two
    runs over identical inputs produced parquets with identical CONTENT and a
    different sha256.  This repository cites box B's audit BY HASH, so a
    parallel campaign was not hash-citable at all.

    **Incremental persistence.**  With ``checkpoint_dir`` set, each chunk is
    written to its own part file as it completes.  Previously ``rows``
    accumulated in memory and were written once at the end, so a 30-hour run
    dying at hour 25 lost twenty-five hours and discovered it at the worst
    possible moment.

    **Resume.**  On restart, ``sample_id``s already present in the checkpoint
    are skipped.  Unreadable parts -- a part truncated by a hard kill mid-write
    -- are DISCARDED rather than trusted, and their samples re-run.

    ``workers <= 1`` keeps the serial path, which is what the gates and the
    smoke tests use.
    """
    payload = [(i, x, kw) for i, x in enumerate(xs)]

    done, parts = (set(), [])
    if checkpoint_dir and resume:
        done, parts = _completed_ids(checkpoint_dir)
        if done:
            print(f"  resume: {len(done)} of {len(xs)} samples already "
                  f"checkpointed in {checkpoint_dir}")
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    todo = [pl for pl in payload if pl[0] not in done]

    if chunk_size is None:
        chunk_size = len(todo) if not checkpoint_dir else max(1, min(256, len(todo)))

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    n_done = 0
    for c0 in range(0, len(todo), max(1, chunk_size)):
        chunk = todo[c0:c0 + max(1, chunk_size)]
        crows: list[dict[str, Any]] = []
        if workers <= 1:
            crows = [r for pl in chunk for r in _worker_ided(pl)]
        else:
            # submit/as_completed rather than map: a worker that dies abruptly
            # (BrokenProcessPool) then costs one sample, recorded as an error
            # row, instead of destroying the whole batch.  Completion ORDER is
            # nondeterministic, which is why the sort below is not optional.
            from concurrent.futures import as_completed
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_worker_ided, pl): pl for pl in chunk}
                for fut in as_completed(futs):
                    pl = futs[fut]
                    try:
                        crows.extend(fut.result())
                    except BaseException as exc:        # worker death included
                        row = dict(pl[1])
                        row.update(outcome="error", converged=False,
                                   sample_id=int(pl[0]),
                                   fail_reason=f"worker: {type(exc).__name__}: {exc}"[:160])
                        crows.append(row)
        if checkpoint_dir:
            # NAME THE PART BY ITS FIRST SAMPLE_ID, never by the position in
            # `todo`.  On resume `todo` is shorter, so position-based names
            # restart at 0 and OVERWRITE the very parts resume just recovered --
            # silently, and the loss only shows up as a short artefact at the
            # end.  Sample ids are stable across runs, so ids cannot collide
            # with a part holding different samples.
            part = os.path.join(checkpoint_dir, f"part-{chunk[0][0]:07d}.parquet")
            tmp = part + ".tmp"
            # write-then-rename: a kill mid-write leaves the .tmp, never a
            # half-written part that resume would read as complete.
            _sort_rows(pd.DataFrame(crows)).to_parquet(tmp, index=False)
            os.replace(tmp, part)
            parts.append(part)
        else:
            rows.extend(crows)
        n_done += len(chunk)
        if progress and todo:
            el = time.time() - t0
            rate = el / max(n_done, 1)
            print(f"  {n_done}/{len(todo)} samples  ({el/60:.1f} min elapsed, "
                  f"{rate*(len(todo)-n_done)/60:.1f} min remaining)", flush=True)

    if checkpoint_dir:
        frames = []
        for f in sorted(set(parts)):
            try:
                frames.append(pd.read_parquet(f))
            except Exception:
                continue
        df = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(rows))
    else:
        df = pd.DataFrame(rows)

    # THE DETERMINISM GUARANTEE.  Sort before returning, so every downstream
    # consumer -- write_parquet, the manifest, any hash -- sees one canonical
    # order for a given input list.
    df = _sort_rows(df)
    # DO NOT PUT PROVENANCE HERE EXPECTING IT TO SURVIVE (ruling R-22.5).
    # pandas serialises df.attrs into the parquet footer, so anything stored
    # here changes the artefact's sha256 without changing a single value in it.
    # write_parquet therefore STRIPS attrs before writing, and it must: this key
    # alone made two runs over identical inputs produce different hashes, which
    # broke hash-citation of the artefact.
    #
    # wall_seconds lives here only as a convenience for the CALLER, which copies
    # it into the reproducibility manifest.  THE MANIFEST IS THE HOME FOR
    # PROVENANCE -- grid, workers, seed, code version, wall_seconds, artefact
    # sha256.  A key added here instead will be silently dropped at write.
    df.attrs["wall_seconds"] = time.time() - t0
    return df


def write_parquet(df: pd.DataFrame, path: str) -> str:
    """Write the artefact in a DETERMINISTIC row order (ruling R-21.6).

    run_batch already sorts, but the sort is repeated here because this is the
    function that produces the artefact whose sha256 gets cited, and a caller
    that assembles a DataFrame by some other route must not be able to write a
    differently-ordered parquet through the same door.
    """
    # Write-side immutability, shared rather than local (ruling R-27.3).
    from .immutable import assert_writable
    assert_writable(path, what="campaign artefact")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = _sort_rows(df)
    # STRIP df.attrs BEFORE WRITING.  pandas serialises attrs into the parquet
    # footer, so run_batch's wall_seconds -- which differs on every run by
    # construction -- changed the artefact's sha256 even when every column was
    # bitwise identical.  Sorting the rows was necessary and NOT sufficient:
    # timing provenance was leaking into the artefact's IDENTITY.
    #
    # wall_seconds is provenance, not data.  It belongs in the reproducibility
    # manifest, and it is still on the DataFrame the caller holds -- only the
    # artefact is cleaned, so the hash depends on the data and nothing else.
    df.attrs = {}
    df.to_parquet(path, index=False)
    return path



# ---------------------------------------------------------------------------
# reproducibility manifest (ruling R-6.2; brief sec 4.5)
# ---------------------------------------------------------------------------
# BOX B WROTE NO MANIFEST AT ALL.  Its summary carries stage, n_requested, seed,
# wall_seconds and a path, and nothing else -- no packages, no commit, no
# platform, no BLAS -- which is why its ~1e-12 reproduction drift can be
# attributed to an environment change only by inference and never established.
# See reports/boxb_errata.md E-6.  Box C does not repeat that.
#
# WHY THIS IS NOT "A SECOND MANIFEST".  `ar_bo_campaign.write_reproducibility_
# manifest` exists, but its schema does not fit: its `constants` and `ar_review`
# blocks are slipper/AR-specific (H_LOWER, T_FAIL_K, P_RATIO_VALID, Gate-B ...),
# it writes one fixed filename that would collide across box-C artefacts, and it
# records NO package, platform or BLAS block -- the very thing R-6.2 requires.
# It is also on the slipper/AR track, which brief sec 9.1 and 9.3 protect from
# modification.  This function therefore EXTENDS the same convention onto the
# ball-socket path, deliberately mirroring its key names (generated_at,
# solver_commit, config, parquet_files) so the two are readable side by side.

def _solver_commit() -> str | None:
    """Git commit of the working tree, or None if this is not a repository."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=os.path.dirname(os.path.dirname(
                                 os.path.abspath(__file__))),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _sha256(path: str) -> str | None:
    import hashlib
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def environment_block() -> dict:
    """Everything needed to decide whether a later replay is same-environment.

    This is the block box B lacks.  Ruling R-6.3 makes the reproduction gate
    two-tier -- bitwise WITHIN an environment, a registered multiple of a
    measured floor ACROSS -- and this block is what tells a reader which tier
    applies to the artefact in front of them.
    """
    import platform
    import sys
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {},
        "blas": None,
    }
    for mod in ("numpy", "scipy", "pandas", "pyarrow"):
        try:
            env["packages"][mod] = __import__(mod).__version__
        except Exception:
            env["packages"][mod] = None
    # The BLAS numpy actually reports.  This is the most likely origin of a
    # cross-environment drift and the single thing box B most needed.
    try:
        cfg = np.show_config(mode="dicts")
        env["blas"] = {k: cfg["Build Dependencies"][k]
                       for k in ("blas", "lapack")
                       if k in cfg.get("Build Dependencies", {})}
    except Exception:
        try:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                np.show_config()
            env["blas"] = buf.getvalue().strip() or None
        except Exception:
            env["blas"] = None
    return env


def write_reproducibility_manifest(parquet_path: str, *, config: dict,
                                   seed: int, stage: str,
                                   extra: dict | None = None) -> str:
    """Write ``<parquet stem>_manifest.json`` beside the artefact.

    Records configuration, seed, git commit, GRID READ FROM THE CONFIG (never
    asserted -- standing rule 10), package versions, platform, BLAS, and the
    content sha256 of the parquet itself.
    """
    import json
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": stage,
        "solver_commit": _solver_commit(),
        "seed": seed,
        "grid": {"Ntheta": NTHETA, "Nphi": NPHI},
        "environment": environment_block(),
        "config": config,
        "parquet_files": {os.path.basename(parquet_path): _sha256(parquet_path)},
        "reproduction_tier": (
            "Replay is BITWISE only under an environment matching this block; "
            "across environments the gate is a registered multiple of the "
            "measured floor (ruling R-6.3, results/ch9/"
            "boxc_reproduction_floor.json).  A bare reproduction tolerance is "
            "the numerical equivalent of a bare IoU."),
    }
    if extra:
        manifest.update(extra)
    out = os.path.splitext(parquet_path)[0] + "_manifest.json"
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------
# summaries (strict single-constraint convention)
# ---------------------------------------------------------------------------

def dominant_binding(counts: dict[str, int]) -> dict[str, Any]:
    """Resolve the dominant binding constraint, reporting TIES AS TIES.

    Defect fixed here (ruling 2.1, 2026-07-29): using ``max(counts, key=...)``
    resolves a tie by dict insertion order, which silently manufactures a
    dominant constraint out of an n-way tie and would have produced a false
    binding-geography STOP.  Ties are never silently resolved.
    """
    real = {k: int(v) for k, v in (counts or {}).items()
            if k not in ("", "none")}
    if not real:
        return {"dominant": None, "tied": [], "is_tie": False, "counts": {}}
    top = max(real.values())
    at_top = sorted(k for k, v in real.items() if v == top)
    return {"dominant": at_top[0] if len(at_top) == 1 else None,
            "tied": at_top, "is_tie": len(at_top) > 1,
            "max_count": top, "counts": real}


def summarise(df: pd.DataFrame, *,
              legacy_nonconverged_kind: str | None = None) -> dict[str, Any]:
    """Label statistics.

    PRE-COMMITTED CONVENTION.  Error rows are INFRASTRUCTURE failures (e.g. a
    worker death), not the fourth outcome.  They are excluded from every label
    statistic and reported separately; conflating them with eccentricity
    blow-up would poison the feasibility label.  All rates below are therefore
    taken over VALID rows, never over n.

    The validity partition is REGISTERED (R-32.4) rather than inferred from
    whichever outcomes a given dataset happens to contain -- see the comment on
    VALID_OUTCOMES below.
    """
    n = len(df)
    n_err = int((df["outcome"] == "error").sum())
    # THE REGISTERED VALIDITY PARTITION (ruling R-32.4).
    #
    #   valid   = the solve ANSWERED:  converged, nonconverged, contact_limit
    #   invalid = the solve did not:   clearance_nonconverged, error
    #
    # Validity is whether the solve answered, NOT whether the answer was
    # favourable.  A contact_limit row ran to completion, reached the registered
    # ceiling and returned a determinate physical result -- no equilibrium
    # exists in the bracket for this configuration.  It is the box-E analogue of
    # box-B blow-up, which box B already ruled is "the limiting case of film
    # failure rather than missing data", and it inherits that treatment.
    #
    # WHY THE ALTERNATIVE IS NOT A READING.  E7's threshold is expressed over
    # VALID ROWS, and the no-equilibrium rows ARE the contact_limit rows.
    # Exclude them from the denominator and the numerator leaves with them:
    # E7 becomes 0 of 38,064, converting a refutation into a trivial pass.  A
    # convention whose effect is to make a result unreportable is not a
    # convention.
    #
    # clearance_nonconverged MOVES from valid to invalid under this partition.
    # That is a change to box C's convention, so the box-C denominator is
    # retained and reported alongside rather than replaced -- same construction
    # as convergence_rate_boxB_denominator below.  Neither replaces the other.
    VALID_OUTCOMES = ("converged", "nonconverged", "contact_limit")
    INVALID_OUTCOMES = ("clearance_nonconverged", "error")

    # R-37.4: THE PARTITION EXTENDS ALONG THE TYPED FAILURE KIND.
    #   nonconverged, kind = blowup_*        -> VALID   (no root in the bracket)
    #   nonconverged, kind = picard_at_root  -> INVALID (a root exists; the field
    #                                           solve at it did not converge)
    # Classing the second as no-equilibrium inflates E7's numerator -- the
    # direction this project has had to correct four times.
    #
    # An artefact written before the kind was typed has no column to read.  For
    # those the caller must NAME the convention; there is no default, because a
    # silent default for an unrecorded quantity is how every recent defect got in.
    _nc = df["outcome"] == "nonconverged"
    if "nonconvergence_kind" in df.columns:
        _kind = df["nonconvergence_kind"].astype(object)
        _untyped = _nc & _kind.isna()
        if _untyped.any() and legacy_nonconverged_kind is None:
            raise ValueError(
                f"{int(_untyped.sum())} nonconverged row(s) carry no "
                f"nonconvergence_kind.  Every raise site is typed, so this is "
                f"either a pre-typing row or a defect; pass "
                f"legacy_nonconverged_kind='blowup' or 'picard_at_root' to say "
                f"which, explicitly.")
        _kind = _kind.where(~_untyped, legacy_nonconverged_kind)
        _kind_source = ("typed column" if not _untyped.any() else
                        f"typed column; {int(_untyped.sum())} untyped row(s) "
                        f"classed as {legacy_nonconverged_kind!r} by the caller")
    else:
        if _nc.any() and legacy_nonconverged_kind is None:
            raise ValueError(
                f"this frame has {int(_nc.sum())} nonconverged row(s) and no "
                f"nonconvergence_kind column (written before the kind was "
                f"typed).  R-37.4 makes Picard-at-root rows INVALID, so the "
                f"partition cannot be computed without knowing which these are. "
                f"Pass legacy_nonconverged_kind='blowup' only if that has been "
                f"verified for this artefact, or join a derived kind column.")
        _kind = pd.Series(legacy_nonconverged_kind, index=df.index, dtype=object)
        _kind_source = (f"no typed column; all nonconverged rows classed as "
                        f"{legacy_nonconverged_kind!r} by the caller")
    if legacy_nonconverged_kind not in (None, "blowup", "picard_at_root"):
        raise ValueError(f"legacy_nonconverged_kind must be 'blowup' or "
                         f"'picard_at_root', got {legacy_nonconverged_kind!r}")
    _picard = _nc & (_kind.astype(str) == "picard_at_root")
    n_picard = int(_picard.sum())

    _valid_mask = df["outcome"].isin(VALID_OUTCOMES) & ~_picard
    valid = df[_valid_mask]
    n_valid = len(valid)
    n_invalid = int(df["outcome"].isin(INVALID_OUTCOMES).sum()) + n_picard
    conv = valid[valid["outcome"] == "converged"]
    n_nonconv = int((valid["outcome"] == "nonconverged").sum())
    n_contact_limit = int((valid["outcome"] == "contact_limit").sum())
    n_clear = int((df["outcome"] == "clearance_nonconverged").sum())
    # EVERY ROW LANDS IN EXACTLY ONE CLASS.  This assertion fired on the box-E
    # artefact because contact_limit was in neither -- 158,544 rows in a class
    # the partition did not name.  It did its job: the alternative was a
    # convergence_rate reported over a denominator missing 80 % of the run.
    _unclassified = sorted(set(df["outcome"].astype(str))
                           - set(VALID_OUTCOMES) - set(INVALID_OUTCOMES))
    assert not _unclassified, (
        f"outcome(s) {_unclassified} are in neither the valid nor the invalid "
        f"class: the validity partition does not cover the taxonomy")
    assert n_valid + n_invalid == n, "row accounting lost samples"
    # Composition of the VALID class, restated under R-32.4: clearance rows are
    # no longer in it, contact_limit rows are.
    assert n_valid == len(conv) + n_nonconv + n_contact_limit,         "outcome accounting lost rows"
    assert not (df["outcome"] == "error").isin([True]).all() or n_err < n, \
        "all rows are infrastructure errors; nothing to summarise"
    out: dict[str, Any] = {
        "n": int(n),
        "n_error_infrastructure": n_err,          # reported separately, always
        "n_valid": int(n_valid),
        "n_invalid": int(n_invalid),
        "n_contact_limit": n_contact_limit,
        "validity_partition": {
            "valid": ["converged", "contact_limit", "nonconverged[kind=blowup*]"],
            "invalid": ["clearance_nonconverged", "error",
                        "nonconverged[kind=picard_at_root]"],
            "registration": "R-32.4, extended by R-37.4",
            "kind_source": _kind_source},
        "n_nonconverged_picard_at_root": n_picard,
        # BOX-C-COMPARABLE denominator, retained because R-32.4 moves
        # clearance_nonconverged out of the valid set.  Box C and box D numbers
        # remain reproducible from this key; it is not replaced by the above.
        "n_valid_boxC_denominator": int(len(df[df["outcome"].isin(
            ["converged", "nonconverged", "clearance_nonconverged"])])),
        "n_converged": int(len(conv)),
        "n_nonconverged_blowup": n_nonconv,
        # rates over VALID rows, excluding infrastructure errors
        "n_clearance_nonconverged": n_clear,
        "convergence_rate": float(len(conv) / n_valid) if n_valid else float("nan"),
        "nonconvergence_fraction": float(n_nonconv / n_valid) if n_valid else float("nan"),
        "clearance_nonconvergence_fraction":
            float(n_clear / n_valid) if n_valid else float("nan"),
        # BOX-B-COMPARABLE denominator.  When clearance rows exist the valid
        # set is larger than box B's, so convergence_rate above is NOT directly
        # comparable across boxes.  This one is: it uses box B's denominator
        # (converged + blow-up) exactly.  Both are reported; neither replaces
        # the other.  On a box-B frame the two are identical by construction.
        "convergence_rate_boxB_denominator":
            float(len(conv) / (len(conv) + n_nonconv))
            if (len(conv) + n_nonconv) else float("nan"),
        "denominator_note": (
            "convergence_rate is over ALL valid rows including "
            "clearance_nonconverged; convergence_rate_boxB_denominator "
            "excludes them and is the figure comparable with box B"),
    }
    if len(conv):
        # Diagnostic spread over ALL converged samples.  Note this includes
        # infeasible box corners (very small clearance or very high speed) whose
        # dissipation is enormous and which g2/g3 reject; it is NOT the
        # "T_max envelope" deliverable.
        out["T_max_max_all_converged_C"] = float(conv["T_max"].max())
        out["T_max_median_all_converged_C"] = float(conv["T_max"].median())
        out["De_sq_median"] = float(conv["De_sq"].median())
        out["De_sq_p95"] = float(conv["De_sq"].quantile(0.95))
        out["L2_routing_fraction"] = float((conv["thermal_mode"] == "L2").mean())
    for b in BANDS:
        for c in CLASSES:
            oc, bc = f"outcome_{c}_{b}", f"binding_{c}_{b}"
            feas = int((conv[oc] == "feasible").sum()) if len(conv) else 0
            key = f"{c}_{b}"
            out[f"feasible_{key}"] = feas
            out[f"feasible_frac_of_converged_{key}"] = (
                float(feas / len(conv)) if len(conv) else float("nan"))
            counts = conv[bc].value_counts().to_dict() if len(conv) else {}
            out[f"binding_{key}"] = {k: int(v) for k, v in counts.items()
                                     if k not in ("", "none")}
            # Ties reported explicitly, never resolved by insertion order.
            out[f"dominant_{key}"] = dominant_binding(counts)
            # The registered "T_max envelope" and De_sq distribution are over
            # the FEASIBLE set (as at the slipper, where the envelope ran cool
            # at T_max <= 89.2 C against a 110 C limit).
            sub = conv[conv[oc] == "feasible"] if len(conv) else conv
            if len(sub):
                out[f"T_max_envelope_C_{key}"] = float(sub["T_max"].max())
                out[f"T_max_median_feasible_C_{key}"] = float(sub["T_max"].median())
                out[f"De_sq_max_feasible_{key}"] = float(sub["De_sq"].max())
                out[f"L2_routing_fraction_feasible_{key}"] = float(
                    (sub["thermal_mode"] == "L2").mean())
    return out


@functools.wraps(_evaluate_x_inner)
def evaluate_x(x: dict[str, float], **kw) -> dict[str, Any]:
    """Evaluate one campaign sample, with the row schema normalised at one exit.

    functools.wraps is NOT cosmetic here.  Without it the wrapper's signature
    collapses to (x, **kw) and Ntheta/Nphi/contact vanish from introspection --
    which defeats the SIGNATURE-DERIVED coverage gate (R-25.4): a parameter
    hidden behind **kw is invisible in exactly the way a default is, and for
    the same reason.  Removing the arm-A defaults to make choices visible, then
    hiding them again behind a wrapper, would undo that work silently.

    Thin wrapper so every return path in the solve logic -- converged, the two
    non-convergence branches, contact_limit and error -- yields the SAME
    columns.  Normalising inside each branch is how the schema went ragged in
    the first place.
    """
    return _ensure_schema(_evaluate_x_inner(x, **kw))
