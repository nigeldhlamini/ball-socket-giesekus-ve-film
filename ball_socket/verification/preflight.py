"""Pre-flight gates: run BEFORE a campaign, not after it.

Two gates, duals of each other, and the pair is the point.  Each catches a class
the other is blind to, and the hash test catches a third.

    gate            asks                                    catches
    -------------   -------------------------------------   ---------------------
    taxonomy        can every REGISTERED OUTCOME             the contact path being
    (R-23.4)        be produced by the code about to run?    registered but never
                                                             reachable
    coverage        is every PRODUCED CHOICE backed by a     traversal, fork arms,
    (R-24.2)        key in config_dict()?                    undetermined-row load
    hash            did anything NEW enter config_dict()?    genuine scope creep

WHY THE HASH TEST ALONE IS NOT ENOUGH, stated here because it was proposed as
the mechanical procedure that would settle a rule-4 question in one command:

    A hash test detects ADDITION.  It cannot detect OMISSION.  An unchanged hash
    is what you get when nothing new is registered, and equally what you get
    when a required choice was never registered at all.  The absence of a key
    has no hash footprint, so the instrument returns a confident pass on a
    question it cannot address.

The box-D campaign ran four hours of pure film while its registration reserved a
``contact_limit`` taxonomy slot the code could not produce.  Both gates below
would have caught that from the registration alone, in under a second, before
the run rather than after it.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Gate 1 -- taxonomy producibility (R-23.4)
# ---------------------------------------------------------------------------

def taxonomy_producible(taxonomy: dict | tuple | list,
                        code_paths: list[Any]) -> tuple[list[str], dict]:
    """Every registered outcome value must be producible by the code about to run.

    Returns ``(failures, report)``.

    THE CHECK IS DELIBERATELY CRUDE and that is defensible here: it asks whether
    each taxonomy value appears as a literal in the source of the modules that
    produce rows.  A value that appears nowhere in the producing code cannot be
    produced, which is a one-line proof that the campaign is not the registered
    one.  The converse is not proven -- a literal may appear on an unreachable
    branch -- so this gate is SUFFICIENT to fail and not sufficient to pass.
    That asymmetry is stated rather than hidden: it is a smoke alarm, not a
    correctness proof.

    (Contrast the registration-audit convention, which forbids grep-based
    checks: that rule is about whether a quantity is REGISTERED, where a
    substring in prose proves nothing.  Here the question is whether a string is
    ever ASSIGNED by the code, which is what a source scan actually answers.)
    """
    # EXTRACT THE OUTCOME VALUES, NOT THE PROSE.  The registered taxonomy is a
    # dict mixing explanatory sentences with the actual outcome strings, and a
    # first version of this gate checked EVERY string in it -- producing twelve
    # false failures out of fifteen because it was asking whether a paragraph of
    # justification appeared verbatim in the solver.
    #
    # A gate with an 80 %% false-positive rate is a gate that gets ignored, which
    # is worse than no gate: it trains the reader to skip the one real failure
    # sitting among the noise.  Only entries under VALUE_FIELDS are outcome
    # values; everything else in the dict is annotation.
    VALUE_FIELDS = ("new_outcomes", "typed_columns", "values", "outcomes")
    if isinstance(taxonomy, dict):
        values = []
        for k, v in taxonomy.items():
            if k in VALUE_FIELDS and isinstance(v, (list, tuple)):
                values.extend(str(x) for x in v)
    else:
        values = [str(v) for v in taxonomy]
    if not values:
        return (["taxonomy contained no extractable outcome values; the gate "
                 "cannot answer the question and must not report a pass"], {})

    src = "\n".join(inspect.getsource(m) for m in code_paths)
    failures: list[str] = []
    report: dict[str, bool] = {}
    for val in sorted(set(values)):
        # look for the value as a quoted literal, which is how an outcome is
        # assigned; a bare word in a comment does not count
        found = bool(re.search(r"""['"]%s['"]""" % re.escape(val), src))
        report[val] = found
        if not found:
            failures.append(
                f"taxonomy value {val!r} is registered but appears as a literal "
                f"in NO producing module: the code path about to run cannot "
                f"emit it, so this campaign is not the registered one")
    return failures, report


# ---------------------------------------------------------------------------
# Gate 2 -- choice coverage (R-24.2)
# ---------------------------------------------------------------------------

#: Every choice the campaign entry must make, and the ``config_dict()`` key that
#: backs it.  A choice with no key is an unregistered choice determining a
#: registered outcome -- the defect class this project keeps meeting.
#:
#: THIS LIST IS THE GATE'S WEAK POINT and it is stated plainly: the gate can
#: only check choices somebody thought to enumerate.  It is still worth having,
#: because enumerating them is exactly the step that was skipped, and because a
#: missing entry here is visible in review whereas a missing key is not.
REQUIRED_CHOICES: dict[str, str] = {
    "grid":                      "solver_numerics",
    "eps_cap":                   "eps_cap_margin",
    "extremum_source":           "constraint_extremum_source",
    "seed":                      "audit_seed",
    "sigma_star_band":           "sigma_star_band_m",
    "constraint_thresholds":     "constraints",
    # --- the four the box-D campaign needed and could not resolve ------------
    "contact_model_traversal":   "contact_model_traversal",
    "F3_summit_consumption":     "summit_statistics_derived",
    "F5_branch_selection":       "f5_branch_selection",
    "undetermined_row_load":     "undetermined_row_policy",
}


def choices_registered(config: dict,
                       required: dict[str, str] | None = None
                       ) -> tuple[list[str], dict]:
    """Every choice the campaign entry must make is backed by a registered key.

    Returns ``(failures, report)``.  This is the gate whose absence made box E
    necessary: the hash test could not see a key that was never there.
    """
    required = REQUIRED_CHOICES if required is None else required
    failures: list[str] = []
    report: dict[str, bool] = {}
    for choice, key in sorted(required.items()):
        present = key in config
        report[choice] = present
        if not present:
            failures.append(
                f"choice {choice!r} has no registered key (expected "
                f"config_dict()[{key!r}]): an unregistered choice determining a "
                f"registered outcome")
    return failures, report


# ---------------------------------------------------------------------------

def run_preflight(config: dict, taxonomy: Any, code_paths: list[Any],
                  verbose: bool = True, config_module: Any = None) -> list[str]:
    """All three gates.  Returns the combined failure list; empty means clear."""
    tax_fail, tax_rep = taxonomy_producible(taxonomy, code_paths)
    cov_fail, cov_rep = choices_registered(config)
    inh_fail, inh_rep = (inherited_globals_audit(config_module)
                         if config_module is not None else ([], {}))

    if verbose:
        print("PRE-FLIGHT GATE 1 -- taxonomy producibility (R-23.4)")
        for k, ok in sorted(tax_rep.items()):
            print(f"  {'ok  ' if ok else 'FAIL'}  {k}")
        print("\nPRE-FLIGHT GATE 2 -- choice coverage (R-24.2)")
        for k, ok in sorted(cov_rep.items()):
            print(f"  {'ok  ' if ok else 'FAIL'}  {k}")

    if verbose and inh_rep:
        print("\nPRE-FLIGHT GATE 3 -- inherited globals (R-27.2)")
        for k, v in sorted(inh_rep.items()):
            bad = k in {f.split("'")[1] for f in inh_fail if "'" in f}
            print(f"  {'FAIL' if bad else 'ok  '}  {k}: {v}")

    fails = tax_fail + cov_fail + inh_fail
    if verbose:
        print(f"\n{len(fails)} pre-flight failure(s).")
        for f in fails:
            print(f"  - {f}")
    return fails

# ---------------------------------------------------------------------------
# Gate 3 -- inherited-globals audit (R-27.2)
# ---------------------------------------------------------------------------

def inherited_globals_audit(module: Any,
                            verified: dict[str, str] | None = None
                            ) -> tuple[list[str], dict]:
    """No callable in a box's namespace reads a superseded box's globals unless
    it has been verified equivalent and that verification is recorded in-module.

    WHY THIS GATE EXISTS.  ``from campaign_config_v6 import *`` binds box D's
    FUNCTION OBJECTS, and a function carries its own ``__globals__``.  So an
    inherited ``freeze_and_hash`` reads box D's ``DRAFT`` and box D's
    ``config_dict()`` and defaults to writing box D's frozen artefact --
    regardless of anything box E defines.  Overriding ``config_dict`` does not
    help, because the inherited function never calls it.  Freezing box E would
    have silently re-frozen box D and overwritten a hashed, cited artefact.

    THE AUDIT IS ENUMERATED, NOT LISTED, and that distinction is the gate.  An
    audit over functions someone thought of has gate 2's weakness -- it checks
    what was remembered.  Enumerating ``vars(module)`` also answers the question
    TRANSITIVELY: box D itself imports from v4/v5, so a callable box E inherits
    THROUGH box D may read box C's or box B's namespace.  A hand-written list
    would not catch that; this does, without anyone anticipating it.

    ``verified`` is the module's own record of callables that have been checked
    and found equivalent -- the escape hatch, deliberately requiring the
    verification to be written down where the next reader will find it.
    """
    import inspect

    home = getattr(module, "__name__", "?")
    verified = verified or getattr(module, "_INHERITED_STATE_READERS_CHECKED", {})

    failures: list[str] = []
    report: dict[str, str] = {}
    for name, obj in sorted(vars(module).items()):
        if name.startswith("__") or not inspect.isfunction(obj):
            continue
        origin = obj.__globals__.get("__name__", "?")
        if origin == home:
            continue
        # It is inherited.  Does it READ its own module's state?
        try:
            src = inspect.getsource(obj)
        except Exception:
            src = ""
        reads = sorted({g for g in ("DRAFT", "FROZEN_ON", "config_dict")
                        if g in src})
        if not reads:
            report[name] = f"inherited from {origin.split('.')[-1]}, reads no state"
            continue
        if name in verified:
            report[name] = f"inherited, reads {reads} -- VERIFIED: {verified[name]}"
            continue
        report[name] = f"inherited from {origin.split('.')[-1]}, READS {reads}"
        failures.append(
            f"callable {name!r} is inherited from {origin} and reads {reads} "
            f"from THAT module's namespace, not {home}'s.  Either override it "
            f"or record the equivalence in "
            f"_INHERITED_STATE_READERS_CHECKED.")
    return failures, report


# ---------------------------------------------------------------------------
# Gate 1, SECOND STAGE -- artefact scan (box E sec 3)
# ---------------------------------------------------------------------------

def taxonomy_in_artefact(taxonomy: dict | tuple | list, rows,
                         exercisable: set | None = None
                         ) -> tuple[list[str], dict]:
    """Does every registered outcome actually APPEAR in produced rows?

    Gate 1's source scan is sufficient to fail and NOT sufficient to pass: a
    literal on an unreachable branch satisfies it.  It proved that in practice
    -- it went to zero while the row schema still had no ``F_asp`` column at
    all, because the literal existed in a module the scan happened to include.

    This stage asks the artefact instead of the source: check what the run
    PRODUCED, not what the code says.  Same construction as the
    realised-parameter columns.

    ``exercisable`` names the values a smoke sample can plausibly hit; anything
    outside it is REPORTED as not-exercised rather than failed, so the residual
    is enumerated instead of waved at.
    """
    VALUE_FIELDS = ("new_outcomes", "typed_columns", "values", "outcomes")
    if isinstance(taxonomy, dict):
        values = []
        for k, v in taxonomy.items():
            if k in VALUE_FIELDS and isinstance(v, (list, tuple)):
                values.extend(str(x) for x in v)
    else:
        values = [str(v) for v in taxonomy]

    seen_outcomes = {str(r.get("outcome")) for r in rows}
    columns = set().union(*(set(r) for r in rows)) if rows else set()

    failures: list[str] = []
    report: dict[str, str] = {}
    for val in sorted(set(values)):
        if val in seen_outcomes:
            report[val] = "present as an outcome value"
        elif val in columns:
            report[val] = "present as a column"
        elif exercisable is not None and val not in exercisable:
            report[val] = "NOT EXERCISED by this sample (declared, not failed)"
        else:
            report[val] = "ABSENT"
            failures.append(
                f"registered value {val!r} appears in no produced row: the "
                f"source scan passed but the run does not emit it")
    return failures, report


# ---------------------------------------------------------------------------
# Gate 4 -- registered-but-not-applied (ruling R-33.2)
# ---------------------------------------------------------------------------

#: How a registered solver numeric is read back off a CONSTRUCTED solver.
#: Convention first (``eps_cap`` -> ``slv.eps_cap``), with nested blocks named
#: explicitly because their registered key and their attribute path differ.
#:
#: A registered numeric that this cannot resolve is a FAILURE, never a skip.
#: That is the must-be-armed lesson applied to gate 4 itself: a gate that
#: silently passes what it cannot check is worse than no gate, because its zero
#: is read as coverage.
_REALISED_PATHS: dict = {
    "eps_cap":               ("slv.eps_cap",),
    "eps_lo":                ("slv.eps_lo",),
    "grid":                  ("slv.Ntheta", "slv.Nphi"),
    "brentq":                ("slv.brentq_xtol", "slv.brentq_rtol"),
    "picard":                ("slv.config.max_iter", "slv.config.tol",
                              "slv.config.omega"),
    "include_normal_stress": ("slv.config.include_normal_stress",),
}

#: Registered-value keys, by status, that carry the number to compare against.
_VALUE_KEYS = ("value", "default_xtol", "default_rtol", "max_iter", "tol",
               "omega")


def _dig(obj, path: str):
    """Follow ``slv.config.max_iter`` down a constructed object."""
    cur = obj
    for part in path.split(".")[1:]:
        cur = getattr(cur, part)
    return cur


def registered_solver_numerics(config: dict) -> dict:
    """The solver numerics the registration carries, read FROM the config.

    Derived, not listed: whatever ``config_dict()["solver_numerics"]`` holds is
    what gate 4 checks.  Adding a numeric to the registration automatically
    adds it to the gate, which is the property the hand-lists lacked.
    """
    return dict(config.get("solver_numerics") or {})


def realised_matches_registered(config: dict, solver, *, theta_min_rad=None,
                                derived: dict | None = None
                                ) -> tuple[list[str], dict]:
    """Every registered solver numeric equals what the constructed solver has.

    WHY THIS GATE EXISTS.  Gate 2 asks whether a choice has a KEY in
    ``config_dict()``.  ``eps_cap_margin`` was there, so gate 2 passed -- and the
    campaign still ran 115 hours at 0.99, because a key proves the value was
    written down and NOT that any code reads it.  That is a distinct mode:

        registered-but-not-applied.

    It is invisible to every gate the project had.  The taxonomy gate asks about
    outcomes, not parameters.  The coverage gate asks about keys, not values.
    The hash test detects addition.  The determinism check compares runs to each
    other, and two runs at the same wrong bracket agree perfectly.

    ``derived`` supplies the resolved value for DERIVED keys (e.g. ``eps_cap``
    at this row's ``theta_min``), because a formula has no single scalar to
    compare against.
    """
    derived = derived or {}
    numerics = registered_solver_numerics(config)
    if not numerics:
        return (["config carries no solver_numerics block: gate 4 cannot "
                 "answer the question and must not report a pass"], {})

    failures: list[str] = []
    report: dict[str, str] = {}

    for name, spec in sorted(numerics.items()):
        paths = _REALISED_PATHS.get(name)
        if paths is None:
            report[name] = "UNMAPPED"
            failures.append(
                f"registered solver numeric {name!r} has no realised path: "
                f"gate 4 cannot confirm the running solver uses it, and an "
                f"unchecked parameter is exactly how eps_cap survived")
            continue

        status = str(spec.get("status", "")) if isinstance(spec, dict) else ""

        # ---- CRITERION: no scalar to compare; record what ran --------------
        if status.startswith("CRITERION"):
            got = tuple(_dig(solver, q) for q in paths)
            report[name] = f"CRITERION -- realised {got}"
            continue

        # ---- DERIVED: compare against the resolved formula -----------------
        if status == "DERIVED":
            if name not in derived:
                report[name] = "DERIVED, unresolved"
                failures.append(
                    f"registered numeric {name!r} is DERIVED and no resolved "
                    f"value was supplied: the gate cannot check a formula "
                    f"against a solver without evaluating it")
                continue
            want = float(derived[name])
            got = float(_dig(solver, paths[0]))
            ok = abs(want - got) <= 1e-12 * max(1.0, abs(want))
            report[name] = (f"DERIVED want {want:.9g} got {got:.9g} "
                            f"{'ok' if ok else 'DIVERGED'}")
            if not ok:
                failures.append(
                    f"registered numeric {name!r} is DERIVED as {want:.9g} but "
                    f"the constructed solver has {got:.9g}: the registration is "
                    f"not reaching the solve")
            continue

        # ---- everything else: pull the scalar(s) out of the spec -----------
        if not isinstance(spec, dict):
            wants = [spec]
        else:
            wants = [spec[k] for k in _VALUE_KEYS if k in spec]
        if not wants:
            report[name] = f"{status or 'no status'} -- no comparable value"
            failures.append(
                f"registered numeric {name!r} carries no value gate 4 can "
                f"compare (status {status!r}); either register one or give it "
                f"an explicit status the gate understands")
            continue

        gots = [_dig(solver, q) for q in paths]
        pairs = list(zip(wants, gots))
        bad = [(w, g) for w, g in pairs
               if (abs(float(w) - float(g)) > 1e-12 * max(1.0, abs(float(w)))
                   if isinstance(w, (int, float)) and not isinstance(w, bool)
                   else w != g)]
        report[name] = (f"{status.split(',')[0] or 'value'} "
                        + ", ".join(f"want {w!r} got {g!r}" for w, g in pairs)
                        + (" DIVERGED" if bad else " ok"))
        for w, g in bad:
            failures.append(
                f"registered numeric {name!r} is registered as {w!r} but the "
                f"constructed solver has {g!r}: registered-but-not-applied")

    return failures, report


def realised_columns_cover_registration(config: dict, row: dict
                                        ) -> tuple[list[str], dict]:
    """Every registered solver numeric has a ``realised_*`` column in the row.

    The second half of R-33.2, and the half that makes the mode impossible
    rather than merely detectable: a registered parameter with no realised
    column cannot be confirmed from the artefact afterwards.  ``eps_cap`` had no
    column, which is why R-32.3's check needed re-solves instead of one query
    against 196,608 rows that were sitting right there.
    """
    numerics = registered_solver_numerics(config)
    expected = {
        "eps_cap": ["realised_eps_cap"],
        "eps_lo": ["realised_eps_lo"],
        "grid": ["realised_Ntheta", "realised_Nphi"],
        "brentq": ["realised_brentq_xtol", "realised_brentq_rtol"],
        "picard": ["realised_picard_max_iter", "realised_picard_tol",
                   "realised_picard_omega"],
        "include_normal_stress": ["realised_include_normal_stress"],
    }
    failures: list[str] = []
    report: dict[str, str] = {}
    for name in sorted(numerics):
        cols = expected.get(name)
        if cols is None:
            report[name] = "NO EXPECTED COLUMN"
            failures.append(
                f"registered numeric {name!r} has no realised column mapping: "
                f"it cannot be confirmed from an artefact after the fact")
            continue
        missing = [c for c in cols if c not in row]
        report[name] = ("ok -- " + ", ".join(cols)) if not missing else                        ("MISSING " + ", ".join(missing))
        if missing:
            failures.append(
                f"registered numeric {name!r} produces no column(s) "
                f"{missing}: the artefact cannot answer what actually ran")
    # Box F sec 3.4: model components that declare a realised column are held
    # to the same standard as solver numerics.
    comps = ((config.get("model_components") or {}).get("components") or {})
    for name, comp in sorted(comps.items()):
        col = comp.get("realised_column")
        if not col:
            continue
        ok = col in row
        report[f"component:{name}"] = ("ok -- " if ok else "MISSING ") + col
        if not ok:
            failures.append(f"model component {name!r} declares realised column "
                            f"{col!r} and the row does not carry it")
    return failures, report


# ---------------------------------------------------------------------------
# Gate 4, component level -- every registered model component, reachable or not
# (ruling R-39.6)
# ---------------------------------------------------------------------------

#: The roles a ``config_dict()`` key may have.  Every key must carry one; a key
#: with no role is a FAILURE, so a new registration cannot slip past the gate by
#: not being listed.  That is the derived-coverage half of the gate.
KEY_ROLES = ("model_component", "solver_numeric", "sampling", "reporting",
             "fork_record", "descoped", "registration_meta")

#: Component statuses the gate understands.
COMPONENT_STATUSES = ("applied", "not_applied")


def trace_campaign_entry(evaluate, *args, **kwargs):
    """Run ``evaluate(*args, **kwargs)`` and return the set of package functions
    it CALLED, as ``module:qualname``.

    Reachability measured, not asserted: a component whose consumer never
    appears here was not exercised by the campaign entry, whatever the
    registration says.  Returns ``(result, called)``.
    """
    import sys
    called: set = set()

    def prof(frame, event, arg):
        if event == "call":
            m = frame.f_globals.get("__name__", "")
            if m.startswith("ball_socket"):
                called.add(m + ":" + frame.f_code.co_qualname)

    old = sys.getprofile()
    sys.setprofile(prof)
    try:
        result = evaluate(*args, **kwargs)
    finally:
        sys.setprofile(old)
    return result, called


def model_components_reachable(config: dict, registry: dict,
                               called: set) -> tuple[list[str], dict]:
    """Every registered model component is applied where it says, and only there.

    ``registry`` carries two blocks:

      * ``key_roles``   -- {config key: role} for EVERY key in ``config``;
      * ``components``  -- {name: {"config_keys": [...], "status": ...,
                                   "consumers": ["module:qualname", ...]}}.

    Failures:
      * a config key with no role, or an unknown role;
      * a ``model_component`` key covered by no component;
      * an ``applied`` component whose consumers did not ALL run
        -- registered-but-not-applied, the mode that let the contact path,
        ``eps_cap`` and the Patir-Cheng factors sit uncalled;
      * a ``not_applied`` component any of whose consumers DID run
        -- the registration and the code disagree in the other direction;
      * a component with no consumers named, or an unknown status.
    """
    roles = registry.get("key_roles") or {}
    comps = registry.get("components") or {}
    failures: list[str] = []
    report: dict[str, str] = {}
    if not roles or not comps:
        return (["component registry is empty: the gate cannot answer and must "
                 "not report a pass"], {})

    for k in sorted(config):
        role = roles.get(k)
        if role is None:
            failures.append(f"config key {k!r} has no role: an unclassified "
                            f"registration is invisible to this gate")
        elif role not in KEY_ROLES:
            failures.append(f"config key {k!r} has unknown role {role!r}")
    for k in sorted(set(roles) - set(config)):
        failures.append(f"role given for {k!r}, which is not in config_dict(): "
                        f"the classification is stale")

    covered = set()
    for name, comp in sorted(comps.items()):
        status = comp.get("status")
        cons = list(comp.get("consumers") or [])
        keys = list(comp.get("config_keys") or [])
        covered |= set(keys)
        for k in keys:
            if roles.get(k) != "model_component":
                failures.append(f"component {name!r} claims key {k!r}, whose "
                                f"role is {roles.get(k)!r}, not model_component")
        if status not in COMPONENT_STATUSES:
            failures.append(f"component {name!r} has unknown status {status!r}")
            continue
        if not cons:
            failures.append(f"component {name!r} names no consumer: reachability "
                            f"cannot be checked, so it is not a pass")
            continue
        ran = [c for c in cons if c in called]
        missing = [c for c in cons if c not in called]
        if status == "applied":
            if missing:
                failures.append(
                    f"component {name!r} is registered as APPLIED but "
                    f"{missing} never ran from the campaign entry: "
                    f"registered-but-not-applied")
            report[name] = f"applied: {len(ran)}/{len(cons)} consumers ran"
        else:
            if ran:
                failures.append(
                    f"component {name!r} is registered as NOT applied but "
                    f"{ran} ran from the campaign entry: the registration and "
                    f"the code disagree")
            report[name] = f"not applied: {len(ran)}/{len(cons)} consumers ran"

    for k in sorted(k for k, r in roles.items() if r == "model_component"):
        if k not in covered:
            failures.append(f"model_component key {k!r} is covered by no "
                            f"component: its reachability is unchecked")
    return failures, report


# ---------------------------------------------------------------------------
# GATE 5 -- ROW PROVENANCE (R-45.6)
#
# Gates 1-4 police the REGISTRATION/CAMPAIGN boundary: taxonomy producibility,
# registered-choice coverage, inherited callables, realised parameters and model
# components.  None of them look at a ROW'S IDENTITY, because until box F
# nothing re-solved rows outside a campaign.
#
# Then something did, and a derived column named `lam` -- the relaxation-time
# INPUT AXIS -- was overwritten with Lambda = h_min/sigma*.  570 rows were
# re-solved at the wrong fluid, and nothing in the stack noticed: the inputs
# were internally consistent, just not the row's.  R-43.3 had named the hazard
# (two quantities, one symbol) against beta; it landed in lambda, which is the
# argument for naming the CLASS rather than enumerating instances.
#
# Gate 5 is therefore two assertions, and the first is the cheap one:
#
#   * RESERVED NAMES -- a frame that is about to be handed to a solver may not
#     carry a derived column whose name is an input axis, checked at WRITE time;
#   * ROW PROVENANCE -- every row re-solved outside its campaign has all input
#     axes compared against the artefact row it claims to be, keyed by its
#     identity columns, BEFORE the solve.
# ---------------------------------------------------------------------------

def input_axis_names() -> tuple:
    """The registered input axes -- the reserved names, read from box B."""
    from ..campaign_config_v4 import INPUT_SPACE
    return tuple(d.name for d in INPUT_SPACE)


def reserved_names_free(frame, *, derived: list[str]) -> list[str]:
    """Fail if any DERIVED column takes the name of an input axis.

    `derived` names the columns this frame computed, so the check cannot be
    satisfied by simply not listing one: a derived name that collides is a
    defect whether or not the author remembered it.
    """
    axes = set(input_axis_names())
    clash = sorted(set(derived) & axes)
    return [f"derived column {c!r} shadows the input axis of the same name; "
            f"rename it (Lambda is not `lam`)" for c in clash]


def row_provenance(frame, artefact, *, key: list[str],
                   axes: tuple | None = None, atol: float = 0.0) -> list[str]:
    """Every row's input axes must equal the artefact row it claims to be.

    `frame` carries the rows about to be solved, `artefact` the campaign they
    came from, `key` their shared identity columns (sample_id, model_id, ...).
    Compared BITWISE by default: an input axis that has moved at all is a
    different row, not a rounded one.
    """
    import pandas as _pd
    axes = tuple(axes or input_axis_names())
    missing = [c for c in list(key) + list(axes) if c not in frame.columns]
    if missing:
        return [f"frame lacks {missing}; row provenance cannot be checked"]
    ref = artefact.set_index(list(key))
    fails = []
    for row in frame.itertuples(index=False):
        ident = tuple(getattr(row, k) for k in key)
        try:
            r = ref.loc[ident]
        except KeyError:
            fails.append(f"{ident}: no such row in the artefact")
            continue
        if isinstance(r, _pd.DataFrame):
            fails.append(f"{ident}: identity is not unique in the artefact")
            continue
        for a in axes:
            got, want = float(getattr(row, a)), float(r[a])
            if not (got == want or abs(got - want) <= atol):
                fails.append(f"{ident}: input axis {a} is {got!r}, artefact "
                             f"has {want!r}")
    return fails


# ---------------------------------------------------------------------------
# A/B INVARIANT (R-45.7)
#
# In a comparison whose arms differ in exactly ONE convention, anything that
# moves IDENTICALLY in both arms cannot have been caused by that convention --
# so it indicts the harness, not the treatment.  It needs no registered
# expectation and no reference value, only the comparison's own logic, which is
# what makes it the first instrument here aimed at the analysis layer rather
# than at the code under test.
#
# Found the lambda collision above: 24 of 570 rows came back non-converged
# under BOTH conventions while the artefact had them converged.
# ---------------------------------------------------------------------------

def ab_invariant(baseline, arm_a, arm_b, *, name: str = "outcome") -> list[str]:
    """Rows where BOTH arms depart from the baseline in the same way.

    Such a row cannot be evidence about the convention that distinguishes the
    arms; it is evidence about the harness.  Returns one message per offending
    row (empty when the invariant holds).
    """
    import numpy as _np
    base = _np.asarray(baseline)
    a, b = _np.asarray(arm_a), _np.asarray(arm_b)
    if not (len(base) == len(a) == len(b)):
        return [f"ab_invariant: lengths differ ({len(base)}, {len(a)}, {len(b)})"]
    both = (a != base) & (b != base) & (a == b)
    return [f"row {i}: {name} moved identically in both arms "
            f"({base[i]!r} -> {a[i]!r}); the harness is implicated, not the "
            f"convention" for i in _np.flatnonzero(both)]
