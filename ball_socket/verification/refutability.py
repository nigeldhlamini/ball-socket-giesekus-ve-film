"""Gate 6 -- refutability (R-49.4).

The apparatus checked that expectations were stated in advance, hashed and
frozen.  It had no check that an expectation could FAIL given the registered
sampling.  E3 passed every gate and could not have been refuted over the
registered inlet range: a water-glycol row with T_in >= 60 C has
T_max >= T_in >= 60 C, so g2 fails by arithmetic before any solve.

Gate 6: every expectation names a concrete refuting outcome, and states whether
that outcome is REACHABLE within the registered design, with the basis.  An
expectation (or clause) whose refuting outcome is unreachable is a definition,
not a hypothesis, and must carry a disposition.

The mirror image is caught by the same entry: a clause whose CONFIRMING outcome
is unreachable (decided false by construction) is equally not a test.

AMENDED (R-50.4): BOTH DIRECTIONS.  Each expectation (and clause) must have a
reachable CONFIRMING outcome as well as a reachable refuting one.  E2's location
clause passed the refuting check and fails the confirming one: geometry puts
h_min at theta_min, so "toward theta_max" can only fail.  An expectation that
cannot be confirmed is as defective as one that cannot be refuted, and the two
failures look nothing alike in the artefacts.  `confirmable` takes the same
values as `reachable`; an entry or clause without its own `confirmable`
inherits "yes" only where the claim is a measured quantity with both signs
possible, and that inheritance is stated per entry below, not defaulted.

Retrofitted 2026-09-21 over E1-E14.  `reachable` values:
  "yes"         -- the refuting outcome can occur in the registered design;
  "no"          -- it cannot (decided by construction or arithmetic);
  "not_tested"  -- reachable in principle, but the registered design cannot
                   resolve it (e.g. a phase clause in a phase-free campaign);
  "registration_time" -- decided by the registered constants alone, before any
                   solve (a calculation, not a measurement).

AMENDED (R-52.6): BOUNDARY COINCIDENCE, a check that needs no data.  Flag any
expectation whose population includes a registered operating condition that
sits ON a registered constraint boundary -- a point equal to it, or a range
that reaches or contains it -- where the claim concerns that constraint.  Every
constraint-activity claim over such a condition is decided in advance.  It
compares two registered numbers, so it runs at registration time.  Only
constraints with an INPUT-LEVEL bound qualify; in this campaign that is g2,
because T_max >= T_in on every row (rise >= 0, machine-checked for E3), so
T_in at a ceiling decides g2 on any heating and T_in above it decides g2
outright.  g1 and g3 have no registered input equal to their boundaries.
Each flagged entry must carry a `boundary_disposition`, as an unreachable
direction must carry a `disposition`.

REGISTERED-TEXT RULE (R-52.2): an entry's `claim` is the frozen text VERBATIM,
with `claim_source` naming file:line.  E1's first entry paraphrased the
unfrozen draft and was adjudicated against it (ERR-24).
"""
from __future__ import annotations

REACHABLE = ("yes", "no", "not_tested", "registration_time")

REGISTRY: dict[str, dict] = {
    "E1": {
        "claim": "g3 essentially never binds for HFC-E draws (De_sq ~ 0.02–0.06).",
        "claim_source": "preregistration_FROZEN.md:117 (= signed FINAL_DRAFT_v3.md:96)",
        "confirming_outcome": ("De_sq of HFC-E rows concentrated in [0.02, 0.06], "
                               "g3 violated on essentially no row"),
        "confirmable": 'yes',
        "refuting_outcome": ("De_sq of HFC-E rows mostly outside [0.02, 0.06], or g3 "
                             "violated on a material share of rows"),
        "reachable": "yes",
        "basis": ("De_sq = lambda*phi*omega*eps*cos(theta_min) spans ~0 to ~4 on the "
                  "registered axes, so both outcomes are reachable"),
        "population": ["campaign_T_in"], "constraints": ["g3"],
        "caveat": ("'HFC-E draws' = rows under the water-glycol ceiling (R-48.1); "
                   "measured identical to all converged rows for every De_sq "
                   "statistic (one column per row).  The confinement clause 'binds "
                   "only in the high-lambda corner' was never frozen "
                   "(preregistration_DRAFT.md:43-44); it survives as an "
                   "UNREGISTERED, descriptive observation (R-52.4)."),
        "adjudication_defect": ("ERR-24: boxes A and B adjudicated E1 against the "
                                "unfrozen confinement clause"),
        "observed": ("REFUTED (R-52.3), both readings of the parenthetical: 15.6 % of "
                     "HFC-E rows in [0.02, 0.06]; 34.4 % violate g3; median 0.067, "
                     "upper quartile 6.3x the registered top.  Also refuted on box A's "
                     "and box B's own artefacts (7.3 / 15.2 % in range, 14.7 / 31.1 % "
                     "violating)")},
    "E2": {
        "confirming_outcome": 'g1 the dominant binding constraint',
        "confirmable": 'yes',
        "claim": "g1 dominates binding; g1-critical region on the outer land toward theta_max; near load reversal",
        "refuting_outcome": "another constraint dominates (clause 1)",
        "reachable": "yes",
        "clauses": {
            "location": {"reachable": "no", "confirmable": "no",
                         "disposition": ("ERR-22: STRUCK, not reported as refuted -- "
                                         "geometry decided it before box A ran, so "
                                         "no measurement did"),
                         "basis": ("with axial eccentricity h = c(1 - eps cos theta) "
                                   "is minimised at theta_MIN for every eps > 0, so "
                                   "a g1-critical region 'toward theta_max' is "
                                   "excluded by construction: the clause could "
                                   "only fail"),
                         "machine_check": "location_clause_by_construction"},
            "phase": {"reachable": "not_tested", "confirmable": "not_tested",
                      "basis": "the campaign is steady; no phase is resolved (Phase 5)"}},
        "population": ["campaign_T_in"], "constraints": ["g1", "g2", "g3"],
        "boundary_disposition": (
            "ERR-21 REACH (R-52.6 retrospective): the T_in axis [20, 110] C contains "
            "the 60 C water-glycol ceiling, so under that ceiling every row with "
            "T_in > 60 C is g2-violated by arithmetic -- 537 of box A's 626 and 515 "
            "of box B's 528 water-glycol g2-sole rows (sigma lo).  The water-glycol "
            "dominant constraints printed by boxes A and B (g2; multi) were decided "
            "by the frame.  Restricted to T_in < 60 C, box B's water-glycol cells "
            "flip g3 (168 vs g1 84) -> g1 (180 vs g3 120) across the sigma* band, "
            "exactly as its mineral cells do, and box A's still refute dominance "
            "(g2 89 > g3 64 > g1 22).  Neither verdict moves: box B's E2 verdict is "
            "read from the mineral cells, and box A's refutation holds in them "
            "(g1 the smallest category).  The mineral ceiling is the axis endpoint, "
            "a set of measure zero.  FINDING (R-53.3), on box E and so on the "
            "campaign the paper reports: restricted to the 44.6 % of converged rows "
            "the inlet does not decide, g1 is the most common single binding "
            "constraint in ALL FOUR cells and the classes agree to within 1.06 pp.  "
            "The apparent difference between the classes was the frame.  DECOMPOSED "
            "(R-54.5): the agreement is ENTAILED, not measured -- fluid_class is "
            "never read by the solver, so within the restricted frame the classes "
            "can differ ONLY on rows whose rise carries T_max past 60 C but not past "
            "110 C (2.31 % of restricted rows).  That channel accounts for the gap "
            "EXACTLY, residual ~1e-15 pp, with no sampling term because the two "
            "classes label the same rows.  The substantive finding is g1's "
            "dominance; the gap measures the ceiling's reach.  "
            "results/ch9/E2_class_gap_decomposition.json"),
        "observed": "dominance clause refuted (box A), first order confirmed later; "
                    "phase clause untested"},
    "E3": {
        "confirming_outcome": 'some row with the 60 C ceiling the SOLE violated constraint',
        "confirmable": 'yes',
        "claim": "the envelope may genuinely bind g2 for water-glycol entries",
        "refuting_outcome": ("no row in which the 60 C ceiling is the SOLE violated "
                             "constraint (box B's existential reading; no share "
                             "threshold registered)"),
        "reachable": "no",
        "basis": ("over the registered T_in range [20, 110] C, rows with T_in >= 60 C "
                  "and g1, g3 satisfied exist, and on them g2 fails by arithmetic "
                  "(rise >= 0 on every row), so the refuting outcome is unreachable"),
        "machine_check": "e3_arithmetic_exclusion",
        "restricted": {"population": "T_in < 60 C", "reachable": "yes",
                       "observed": ("confirmed existentially: g2 sole-binding in 17 "
                                    "(sigma lo) / 9 (sigma hi) SAMPLES (408 / 216 rows, "
                                    "24 arms each, arm-invariant in T_max), every one "
                                    "with T_in >= 54.4 C; max rise 9.6 K; survives the "
                                    "rise scaled down by 2x the bias (15 / 9 samples)")},
        "disposition": "ERR-21",
        "population": ["campaign_T_in"], "constraints": ["g2_water_glycol"],
        "boundary_disposition": "ERR-21 (the case this check was built to catch)",
        "observed": "confirmed (unrestricted, unrefutable); confirmed (restricted)"},
    "E4": {
        "confirming_outcome": 'memory the largest viscoelastic correction',
        "confirmable": 'yes',
        "claim": "squeeze memory the leading viscoelastic correction (inversion is a STOP)",
        "refuting_outcome": "another viscoelastic term exceeds the memory term",
        "reachable": "yes",
        "basis": "term-by-term decomposition can order the terms either way",
        "observed": ("four in-domain points (1.6e-6 to 1.2e-4), extended across the "
                     "registered lambda axis (R-50.2): 137 in-domain decompositions, "
                     "|share| <= 9.1e-4, negative everywhere; maximum on the axis "
                     "floor for 18 of 22 rows (a bound over the registered range)")},
    "E5": {
        "confirming_outcome": 'rated maximum corner outside the full-film feasible set at every swept nu_100',
        "confirmable": 'yes',
        "claim": "rated maximum corner outside the full-film feasible set, g1 binding",
        "refuting_outcome": "full film at some swept nu_100 (the threshold above which E5 is refuted)",
        "reachable": "yes",
        "basis": "the nu_100 sweep's thick arm lies inside the box",
        "caveat": ("corner results are at the registered mineral service inlet "
                   "temperatures (R-50.1)"),
        "population": ["E5_T_in_VG32", "E5_T_in_VG46"], "constraints": ["g1"],
        "clauses": {
            "g1_binding_VG46": {
                "reachable": "yes", "confirmable": "no",
                "population": ["E5_T_in_VG46"],
                "constraints": ["g1", "g2_mineral", "g3"],
                "disposition": ("ERR-25: STRUCK for VG46 draws -- not refuted (no data "
                                "did it) and not open (no data could settle it)"),
                "boundary_disposition": "ERR-25",
                "basis": ("VG46's registered corner inlet is 110 C, equal to the mineral "
                          "g2 ceiling; any positive rise fails g2 by arithmetic, so g1 as "
                          "the SOLE binding constraint is unreachable on every VG46 row. "
                          "The refuting outcome (another constraint binds) is reachable.")},
            "g1_binding_VG32": {
                "reachable": "yes", "confirmable": "yes",
                "population": ["E5_T_in_VG32"],
                "constraints": ["g1", "g2_mineral", "g3"],
                "basis": ("inlet 100 C, 10 K below the mineral ceiling, so g2 can hold or "
                          "fail; g3 cannot bind at E5's lambda = 1e-5 s (De_sq <= "
                          "lambda*omega*eps_cap*cos(theta_min) = 0.0047 < 0.2), and g1 "
                          "fails on every equilibrium row (Lambda_max 1.877), so the "
                          "clause reduces to whether T_max_bnd <= 110 C"),
                "observed": ("CONFIRMED (R-54.7, measured 2026-09-23): 960 solves at the "
                             "registered corner, smooth film on every row.  On all 480 "
                             "equilibrium rows g1 is the SOLE violated constraint -- g2 holds "
                             "(T_max_bnd max 100.12 C against the 110 C ceiling, rise <= "
                             "0.12 K) and g3 maxes at 0.0045 against 0.2.  c0 = 15 and 30 um "
                             "reach no equilibrium, as in the E10 corner run.  Lambda_max "
                             "1.8768 reproduces the E10-derived 1.877 on an independent run.  "
                             "results/ch9/E5_VG32_g1_binding_corner.parquet")}},
        "observed": ("box D, from the E10 corner run (R-51.4, extremum corrected R-52): "
                     "outside the full-film feasible set at every swept nu_100, "
                     "Lambda_max 1.877.  Binding clause, per grade: VG46 STRUCK "
                     "(ERR-25, inlet on the ceiling); VG32 CONFIRMED (R-54.7, 960 "
                     "solves): g1 the sole violated constraint on every equilibrium "
                     "row.  No clause of E5 is now owed")},
    "E6": {
        "confirming_outcome": "the joint's L2 fraction substantially above the slipper's",
        "confirmable": 'yes',
        "claim": "L2 routing fraction substantially exceeds the slipper's",
        "refuting_outcome": "the joint's L2 fraction at or below the slipper's",
        "reachable": "yes",
        "basis": "routing fraction is a measured share in [0, 1]",
        "observed": "refuted (45.7 % vs 64.95 %)"},
    "E7": {
        "confirming_outcome": 'no-equilibrium fraction below 5 %',
        "confirmable": 'yes',
        "claim": "no-equilibrium fraction below 5 % of valid rows",
        "refuting_outcome": "no-equilibrium fraction at or above 5 %",
        "reachable": "yes", "basis": "a measured fraction",
        "observed": "refuted (45.61 %, 9.12x; composite 43.24 %)"},
    "E8": {
        "confirming_outcome": 'median Lambda below 1 at both band ends',
        "confirmable": 'yes',
        "claim": "median Lambda below 1 over the box-B blow-up population",
        "refuting_outcome": "median Lambda >= 1 at a band end",
        "reachable": "yes", "basis": "a measured median",
        "observed": "not yet measured"},
    "E9": {
        "confirming_outcome": 'both quantities rise',
        "confirmable": 'yes',
        "claim": "the clearance closure raises converged fraction and median h_min",
        "refuting_outcome": "either quantity falls",
        "reachable": "yes", "basis": "signed changes, row for row",
        "observed": "clause 1 refuted; clause 2 below resolution"},
    "E10": {
        # SPLIT (R-51.3, ERR-23).  The premise -- psi > 1 over the majority of the
        # GT band -- is ENTAILED by the constants dossier (E', the sigma_s/beta_a
        # decade, H_ind) before any solve, so it is neither confirmable nor
        # refutable by the campaign.  It left the expectations register for the
        # dossier as a derived property.  What remains registered is the
        # measurement.
        "confirming_outcome": "positive capped-area fraction at the rated corner",
        "confirmable": "yes",
        "claim": "capped-area fraction > 0 at the rated corner, per band end",
        "refuting_outcome": "zero capped area at the rated corner",
        "reachable": "yes",
        "basis": "exactly zero was measured on the doubt-band arms, so zero is reachable",
        "moved_to_dossier": {"premise": "psi > 1 over the majority of the GT band",
                             "disposition": "ERR-23"},
        "observed": "holds on the clearly plastic arms; exactly zero on the doubt-band arms"},
    "E11": {
        "confirming_outcome": 'time fraction above 50 %',
        "confirmable": 'not_tested',
        "claim": "boundary-regime time fraction per revolution above 50 %",
        "refuting_outcome": "time fraction at or below 50 %",
        "reachable": "not_tested", "basis": "duty cycle is Phase 5, out of Paper 3's scope",
        "observed": "not measured"},
    "E12": {
        "confirming_outcome": 'joint share above the re-derived slipper comparator',
        "confirmable": 'yes',
        "claim": "joint memory share exceeds the slipper's elastic fraction",
        "refuting_outcome": "joint share at or below the re-derived slipper comparator",
        "reachable": "yes",
        "caveat": "ill-posed as registered: a term compared against a package",
        "observed": "not adjudicated (both readings owed)"},
    "E13": {
        "confirming_outcome": 'no (c0, theta_min, arm) reaches Lambda >= 3',
        "confirmable": 'yes',
        "claim": "no achievable land geometry recovers full film at the corner-power condition",
        "refuting_outcome": "one (c0, theta_min, arm) with Lambda >= 3",
        "reachable": "yes",
        "basis": ("Lambda up to 12.5 was reached at an off-condition fluid, so "
                  "Lambda >= 3 is attainable by the solver in this geometry"),
        "caveat": "mineral fluids only; bounds rather than locates the optimum",
        "population": ["E13_T_in_admissible"], "constraints": ["g1"],
        "observed": "confirmed (Lambda_max 2.393, 20.2 % short)"},
    "E14": {
        "confirming_outcome": 'a net reduction at both band ends',
        "confirmable": 'yes',
        "claim": "net effect of box D's corrections is a reduction",
        "refuting_outcome": "a net increase at a band end",
        "reachable": "yes", "basis": "a signed, bootstrapped difference",
        "population": ["campaign_T_in"], "constraints": ["g1", "g2", "g3"],
        "boundary_disposition": (
            "partial, sign-preserving (R-52.6 retrospective): under the water-glycol "
            "ceiling, rows with T_in > 60 C are g2-infeasible under BOTH extremum "
            "conventions (T_max_bnd >= T_max >= T_in), so they cannot change "
            "feasibility; they dilute the two water-glycol cells' signed change and "
            "cannot flip it: 55.6 % of E14's valid-row denominator.  The two mineral "
            "cells contain no decided row, and E14 is refuted in those as well.  "
            "RECOMPUTED over the movable rows (R-53.3, A6): the binding cell's net is "
            "+2.03 pp against +0.903 over all rows, its F3 bound widens 0.475 -> 1.294 "
            "(restricting also removes part of the verified sample), so the worst case "
            "improves only +0.22 -> +0.30 pp.  Sign and binding cell survive on either "
            "denominator.  results/ch9/E14_movable_subpopulation.json.  "
            "DECOMPOSED (R-54.4): 2.72 = denominator 2.25 x representativeness 1.21 "
            "x remainder 1.000 exactly (the bound IS rule-of-three, read at "
            "run_E14_F3_bracket.py:471, so the form candidate is excluded).  The "
            "1.21 is 1.047 population mismatch -- the verified sample is drawn from "
            "the UNION of the four cells' residuals (:322-329), not from this cell "
            "-- times 1.154 sampling scatter, which a hypergeometric null over the "
            "union cannot distinguish from a uniform draw (p = 0.17).  "
            "results/ch9/E14_F3_bound_decomposition.json"),
        "observed": "refuted in all four cells"},
}


#: Registered operating conditions an expectation's population is drawn from,
#: each with its registered source.  Points are (v, v); ranges are (lo, hi).
CONDITIONS: dict[str, dict] = {
    "campaign_T_in": {"var": "T_in_C", "range": (20.0, 110.0),
                      "source": "campaign_config_v4.py:73 (T_in axis, unchanged since box A)"},
    "E5_T_in_VG32": {"var": "T_in_C", "range": (100.0, 100.0),
                     "source": "preregistration_FROZEN.md:138; e5_fluid_spec.SERVICE_T[32]"},
    "E5_T_in_VG46": {"var": "T_in_C", "range": (110.0, 110.0),
                     "source": "preregistration_FROZEN.md:138; e5_fluid_spec.SERVICE_T[46]"},
    "E13_T_in_admissible": {"var": "T_in_C", "range": (100.0, 110.0),
                            "source": "E13_spec_v2.json admissible_corner_fluids (E5 SERVICE_T)"},
}

#: Registered constraint boundaries that an INPUT bounds (see the docstring).
BOUNDARIES: dict[str, dict] = {
    "g2_mineral": {"var": "T_in_C", "value": 110.0, "family": "g2",
                   "bound": "T_max >= T_in", "source": "preregistration_FROZEN.md:111"},
    "g2_water_glycol": {"var": "T_in_C", "value": 60.0, "family": "g2",
                        "bound": "T_max >= T_in", "source": "preregistration_FROZEN.md:111"},
}


def _concerns(constraints: list, bname: str, boundaries: dict) -> bool:
    """A claim about "g2" concerns both ceilings; "g2_mineral" only its own."""
    return bname in constraints or boundaries[bname]["family"] in constraints


def boundary_coincidences(registry: dict = None, conditions: dict = None,
                          boundaries: dict = None) -> list:
    """R-52.6: every (expectation or clause, condition, boundary) where a
    registered condition sits on or spans a registered boundary of a constraint
    the claim concerns.  Needs no artefact -- two registered numbers."""
    registry = REGISTRY if registry is None else registry
    conditions = CONDITIONS if conditions is None else conditions
    boundaries = BOUNDARIES if boundaries is None else boundaries
    out = []

    def _one(label, node):
        for cname in node.get("population", []) or []:
            c = conditions[cname]
            lo, hi = c["range"]
            for bname, b in boundaries.items():
                if b["var"] != c["var"] or not _concerns(node.get("constraints", []),
                                                         bname, boundaries):
                    continue
                v = b["value"]
                if lo == hi == v:
                    kind = "sits on"               # a point on the boundary
                elif lo <= v <= hi:
                    kind = "spans"                 # a range reaching it
                else:
                    continue
                out.append({"label": label, "condition": cname, "boundary": bname,
                            "kind": kind, "value": v,
                            "disposed": bool(node.get("boundary_disposition"))})

    for k, e in registry.items():
        _one(k, e)
        # a clause is checked on its OWN population and constraints, never on
        # its parent's: E5's principal claim is g1-only, its binding clause is not
        for cname, c in e.get("clauses", {}).items():
            _one(f"{k}.{cname}", c)
    return out


#: A direction that no solve can reach.  `not_tested` is different: the outcome
#: is reachable in principle and the design cannot resolve it -- reported, not
#: a registration defect.
UNREACHABLE = ("no", "registration_time")


def check(registry: dict = REGISTRY) -> list[str]:
    """Structural gate, BOTH directions (R-50.4): every expectation names a
    refuting and a confirming outcome, each with a reachability status, and
    every direction that no solve can reach carries a disposition."""
    fails = []

    def _dir(label, node, key_out, key_st, what):
        if key_out and not str(node.get(key_out, "")).strip():
            fails.append(f"{label}: no {what} outcome named")
        st = node.get(key_st)
        if st not in REACHABLE:
            fails.append(f"{label}: {what} reachability {st!r} not in {REACHABLE}")
        elif st in UNREACHABLE and not node.get("disposition"):
            fails.append(f"{label}: {what} outcome unreachable ({st}) and no "
                         f"disposition recorded")

    for k, e in registry.items():
        _dir(k, e, "refuting_outcome", "reachable", "refuting")
        _dir(k, e, "confirming_outcome", "confirmable", "confirming")
        for cname, c in e.get("clauses", {}).items():
            _dir(f"{k}.{cname}", c, None, "reachable", "refuting")
            _dir(f"{k}.{cname}", c, None, "confirmable", "confirming")
    for hit in boundary_coincidences(registry):
        if not hit["disposed"]:
            fails.append(f"{hit['label']}: registered condition {hit['condition']} "
                         f"{hit['kind']} the {hit['boundary']} boundary "
                         f"({hit['value']:g}) -- decided before any solve -- and no "
                         f"boundary_disposition recorded")
    return fails


def e3_arithmetic_exclusion(frame, *, ceiling: float, lambda_min: float,
                            desq_max: float, sigma: float) -> dict:
    """Machine check for E3: are there rows on which g2 is the sole violation
    FORCED by the inlet alone?  If so, 'no sole-g2 row' is unreachable."""
    rise_ok = bool(((frame.T_max_bnd - frame.T_in) >= 0).all())
    forced = ((frame.T_in >= ceiling) & (frame.h_min_bnd / sigma >= lambda_min)
              & (frame.De_sq_bnd <= desq_max))
    return {"rise_nonnegative": rise_ok, "forced_sole_g2_rows": int(forced.sum()),
            "refuting_outcome_reachable": not (rise_ok and forced.any())}


def location_clause_by_construction(theta_min: float, theta_max: float,
                                    eps: float, n: int = 400) -> bool:
    """True if h = c(1 - eps cos theta) is minimised at theta_min on the land."""
    import numpy as np
    th = np.linspace(theta_min, theta_max, n)
    h = 1.0 - eps * np.cos(th)
    return int(np.argmin(h)) == 0
