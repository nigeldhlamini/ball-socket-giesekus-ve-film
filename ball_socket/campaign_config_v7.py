"""campaign_config_v7.py -- Chapter 9 ball-socket campaign, BOX E.

DRAFT.  Awaiting Nigel's signature.  Box D (``campaign_config_v6.py``, frozen
2026-08-31, config sha256 ``0ea6081033342f30e59a49511b0e034dd472fbdc760af59c3a14e11882f79e14``)
is retained unchanged and superseded, never edited.

WHY BOX E EXISTS, in two sentences.  Box D registered a mixed-lubrication
campaign -- ``contact_limit`` in its taxonomy, E7 "with load sharing", forks F3
and F5 adjudicated for a contact model -- but the contact path was unreachable
from the campaign entry, so the box-D run was pure film.  Wiring it is
execution, not scope; but three choices the wired campaign must make had no key
in ``config_dict()``, and giving them one moves the hash, which fires box D
sec 0.3.

Everything box D registered is imported unchanged.  Box E adds three keys, two
errata and two STOP conditions.  It is deliberately short: box D needed length
because fourteen constants and eight unregistered numerics were in play.
"""

from __future__ import annotations

import math

# Box D, imported wholesale and NOT restated.  A box-E reader who wants box D's
# content reads box D; copying it here would create two sources of truth for
# values that have not changed.
# WHY A WILDCARD IMPORT IS SAFE HERE, AND THE TWO HAZARDS IT CARRIES.
#
# Hazard 1 -- LATER EXPORTS.  A wildcard silently absorbs whatever the source
# module later exports.  Neutralised here, and only here, because box D is
# FROZEN AND HASHED: there is no "later" for it to absorb, and any change to
# box D would break its own recorded sha256 first.  Copied into a module that
# can still change, this line is a bug.
#
# Hazard 2 -- IMPORTED FUNCTIONS KEEP THEIR OWN GLOBALS, and this one is live.
# ``from v6 import *`` binds v6's FUNCTION OBJECTS, whose ``__globals__`` remain
# v6's module namespace.  So an inherited ``freeze_and_hash`` reads ``v6.DRAFT``
# and ``v6.config_dict()`` and defaults to writing ``campaign_frozen_v6.json``,
# regardless of what box E defines.  Calling it from box E would have
# re-frozen BOX D and OVERWRITTEN A HASHED ARTEFACT that Chapter 9 cites.
#
# Overriding ``config_dict`` is therefore NOT sufficient: every inherited
# function that reads module state must be overridden too, or shadowed
# deliberately.  ``freeze_and_hash`` is overridden below.  The others
# (``unverified``, ``blocking_unverified``, ``assert_box_b_invariants``) read
# only box-D content that box E does not change, so inheriting them is correct
# -- but that is a fact to be CHECKED per function, not assumed for the module.
from .campaign_config_v6 import *                              # noqa: F401,F403
from . import campaign_config_v6 as _v6

BOX_D_CONFIG_SHA256 = (
    "0ea6081033342f30e59a49511b0e034dd472fbdc760af59c3a14e11882f79e14")


# ---------------------------------------------------------------------------
# KEY 1 -- how the campaign traverses the contact-model cross-product
# ---------------------------------------------------------------------------
# mixed.py refuses to pick a band end (rule 15), so the campaign MUST state how
# it traverses.  Box D did not, which made the traversal an unregistered choice
# determining a registered outcome -- the defect class this project keeps
# meeting, and the one the coverage gate now catches.
#
# WHY THE FULL CROSS-PRODUCT AND NOT A REGISTERED BASE.  The alternative is to
# name one configuration as base and explore the ends on the sensitivity prefix.
# There is no justification available for WHICH configuration is base, so
# registering one would invent exactly the kind of unmotivated choice this
# project keeps removing.  Sixteen corners needs no base and satisfies rule 15
# without argument.

CONTACT_MODEL_TRAVERSAL = {
    "rule": "full cross-product of every registered band end",
    "axes": ("sigma_star_band_m", "eta_a_beta_a_sigma_s",
             "sigma_s_over_beta_a", "H_ind_band_Pa"),
    "n_configurations": 16,                      # 2^4
    "status": "REGISTERED CHOICE",
    "ruling": "R-24.5, R-25.7",
    "no_base_configuration": ("deliberate: naming a base would require a "
                              "justification that does not exist"),
    #: An EXACT short-circuit, not an approximation, and it is an optimisation
    #: rather than a design choice -- it changes no result.  Greenwood-Tripp's
    #: F_5/2 is negligible by H ~ 4, so on any row where the film never
    #: approaches the asperities the asperity load is identically zero under
    #: EVERY configuration and all sixteen coincide.  Solve once at the
    #: soonest-engaging configuration (largest sigma_s, smallest H); if the
    #: asperity load is identically zero across the whole bracket, the other
    #: fifteen are known equal without solving.
    "short_circuit": ("solve once at the soonest-engaging configuration; if the "
                      "asperity load is identically zero across the bracket, "
                      "the other fifteen are known equal"),
    #: REQUIRED, not optional.  A short-circuit that is NEARLY exact is a defect
    #: of the class this project keeps finding, so it is verified rather than
    #: argued: take a sample of short-circuited rows, solve them under all
    #: sixteen anyway, and confirm bit-identical results.
    "short_circuit_verification": ("MANDATORY: a sample of short-circuited rows "
                                   "re-solved under all sixteen must be "
                                   "bit-identical"),
}


# ---------------------------------------------------------------------------
# KEY 2 -- what asperity load an UNDETERMINED row carries
# ---------------------------------------------------------------------------
# Box D closed F5 as three-state: elastic, plastic, and UNDETERMINED for
# 0.6 < psi < 1.0 (Greenwood & Williamson 1966 p. 309, CITED).
#
# THREE-STATE IS A LABELLING RULE.  It says what to CALL a row; it does not say
# what asperity load the solver gives it, and the solver needs a number rather
# than a label.  That the doubt band would carry both arms was said in a ruling
# and never registered, and a sentence in a ruling is not a registration.

UNDETERMINED_ROW_POLICY = {
    "rule": ("a row with 0.6 < psi < 1.0 is solved under BOTH arms and both "
             "results are carried; neither is merged into a single number"),
    "status": "REGISTERED CHOICE",
    "ruling": "R-24.3 item 4, R-25.7",
    "why_not_a_default": ("picking one arm for the doubt band would assert a "
                          "deformation mode over the exact range GW call 'in "
                          "doubt'.  The direct analogue of rule 15's both-ends "
                          "discipline is to carry both and report both."),
    "labelling": ("feasibility is reported per arm and NEVER merged: a row in "
                  "the doubt band carries an elastic-arm label and a "
                  "plastic-arm label, exactly as the sigma* band ends give two "
                  "labels"),
    "reporting": ("any tally over the doubt band states which arm produced it; "
                  "a tally that silently picks one is inadmissible"),
    "cost": ("two solves per undetermined row, on top of the traversal.  Only "
             "the doubt band pays it."),
    #: THE REPORTING FORM, registered as part of the choice rather than left as
    #: a consequence of it.  The cost above is COMPUTE; this is the consequence
    #: that actually reaches the reader.  A doubt-band row carries two asperity
    #: loads, hence two equilibria, two h_min, two Lambda -- and TWO FEASIBILITY
    #: LABELS.  So the feasible-set count is not a number on those rows, and
    #: every tally in the campaign would otherwise be ambiguous on an unstated
    #: subset.
    "reporting_form": ("feasible counts are reported as a PAIR: a LOWER count "
                       "in which doubt-band rows must be feasible under BOTH "
                       "arms, and an UPPER count in which EITHER arm suffices. "
                       "Where the two coincide, say so; where they do not, the "
                       "INTERVAL is the result."),
    "reporting_form_why": ("a policy registered without the form in which its "
                           "output is read is the defect this project keeps "
                           "meeting.  Registering the pair means no tally can "
                           "quietly collapse the doubt band to a single "
                           "number."),
}


# ---------------------------------------------------------------------------
# KEY 3 -- how the F5 branch is selected
# ---------------------------------------------------------------------------
# Box D registered the DISPOSITION (DERIVED per sample, three-state) and the
# THRESHOLDS (CITED, GW1966 p. 308-309).  What it did not register is the
# selection mechanism the campaign entry must implement, and ContactModel as
# built takes a single fixed branch that cannot express it.

F5_BRANCH_SELECTION = {
    "rule": ("per sample, from psi = (E'/H_ind) sqrt(sigma_s/beta_a): "
             "psi < 0.6 elastic (GT Eq. 7); psi > 1.0 plastic (GT Eq. 13); "
             "0.6 <= psi <= 1.0 undetermined, see UNDETERMINED_ROW_POLICY"),
    "status": "REGISTERED CHOICE",
    "ruling": "R-24.3 item 3, R-25.7",
    "thresholds_source": "Greenwood & Williamson (1966) p. 309 -- CITED verbatim",
    "threshold_caveat": ("GW's own qualification travels with the numbers: "
                         "these values hold only for the particular surface "
                         "model considered"),
    "no_default_arm": ("ContactModel's arm-A defaults are REMOVED.  A default "
                       "is precisely a choice that does not look like one, and "
                       "it is how box C's answer survived into box D's "
                       "campaign.  Removal is also what makes the coverage "
                       "gate self-populating (R-25.4)."),
}


# ---------------------------------------------------------------------------
# STOP conditions -- the pre-flight gates, REGISTERED not remembered
# ---------------------------------------------------------------------------
# Built as one-off diagnostics these get run once, when someone remembers.  Box
# D's own claims gate carries the reasoning: a gate that depends on remembering
# to arm it is a gate that will be found disarmed.
#
# Promoting them converts "did we wire it correctly this time?" from a
# judgement made after the fact into a check made in under a second before the
# run.  7 -> 0 on both gates is the measurable acceptance criterion.

PREFLIGHT_STOP_CONDITIONS = {
    "gate_1_taxonomy": {
        "asks": "can every registered outcome value be produced by the code?",
        "module": "ball_socket.verification.preflight.taxonomy_producible",
        "stop_rule": "the campaign does not launch while gate 1 reports a failure",
        #: THE ZERO MUST ARRIVE UNASSISTED.  contact_limit, F_film_at_limit and
        #: F_asp_at_limit are absent for ONE reason -- the contact path does not
        #: run -- so wiring it should make all three appear without anyone
        #: adding a column.  A gate satisfied by editing the thing it measures
        #: is not satisfied.
        "zero_must_be_unassisted": True,
        "known_limitation": ("sufficient to fail, not sufficient to pass: a "
                             "literal on an unreachable branch would satisfy "
                             "the source scan.  Closed by the second stage."),
        "second_stage": ("after the smoke run, scan the ARTEFACT: every "
                         "registered outcome the smoke sample can exercise must "
                         "appear, or its absence must be explained.  Values a "
                         "smoke sample cannot exercise are enumerated rather "
                         "than waved at.  Same construction as the "
                         "realised-parameter columns: check what the run "
                         "produced, not what the source says."),
    },
    "gate_2_coverage": {
        "asks": "is every choice the campaign entry makes backed by a key?",
        "module": "ball_socket.verification.preflight.choices_registered",
        "stop_rule": "the campaign does not launch while gate 2 reports a failure",
        "known_limitation": ("checks only choices someone enumerated in "
                             "REQUIRED_CHOICES"),
        "self_population": ("derive the list from the signatures the campaign "
                            "entry calls: a keyword-only parameter with no "
                            "default IS a choice the caller must make.  This is "
                            "why removing the arm-A defaults is worth more than "
                            "rule-15 hygiene -- with a default present, the "
                            "choice is invisible to a signature-derived gate."),
    },
    "standing": ("same as assert_f3_consistency() raising or the F2 trip-wire "
                 "firing: a STOP, not a warning"),
}


# ---------------------------------------------------------------------------
# ERRATA against box D -- box D is NOT edited
# ---------------------------------------------------------------------------

BOX_D_ERRATA = {
    "E-10_clearance_outcome_spelling": {
        "registered": "clearance_nonconvergence",
        "emitted": "clearance_nonconverged",
        "resolution": "ADOPT THE EMITTED SPELLING",
        "why": ("the pure-film artefact on disk already contains it and is "
                "hashed; the Outcome enum and the campaign code already agree "
                "with it; and nothing substantive is at stake -- both names "
                "denote the same category.  A naming erratum, not a scope "
                "question."),
        "failure_shape": ("the contact-path defect in its quieter direction: a "
                          "tally filtered on box D's own vocabulary returns "
                          "zero ALWAYS, and a reader concludes the event never "
                          "occurs rather than that they spelled it the "
                          "registered way."),
        "found_by": "pre-flight gate 1",
    },
    "E-11_one_solve_two_labels_false": {
        "registered": ("box D sec 8: 'Both sigma* band ends continue to be "
                       "reported always -- one solve, two labels.'"),
        "resolution": "FALSE once the contact path is live",
        "why": ("that held while the campaign was pure film, because sigma* "
                "entered only g1's post-hoc labelling.  With contact live, F3 "
                "derives sigma_s = r sigma*, Greenwood-Tripp's argument is "
                "H = d/sigma_s, and the asperity load feeds the EQUILIBRIUM.  "
                "The two sigma* ends are two SOLVES."),
        "consequence": ("campaign cost doubles for a structural reason rather "
                        "than a chosen one, and sec 8's reporting sentence "
                        "cannot be quoted as written"),
        "nothing_re_posed": ("no registered quantity changes; only a statement "
                             "about cost and reporting becomes false"),
    },
}


DRAFT = False                  # SIGNED by Nigel C. Dhlamini, 2026-09-01
FROZEN_ON = "2026-09-01"        # signed and frozen in one event


def summit_ratio_for(eta_beta_sigma_joint: float) -> float:
    """sigma_s/sigma* for a model carrying the JOINT (eta beta sigma) value.

    THE CONVERSION IS THE WHOLE POINT OF THIS FUNCTION, and getting it wrong is
    the exact trap R-16.3 recorded.

    McCool Eq. 35 is derived for ONE surface's summit statistics.  ContactModel
    carries the JOINT group, which is the single-surface value times sqrt(2),
    because the sum of two Gaussian asperity-height distributions has
    sigma = sigma_1 sqrt(2).  Feeding the joint value straight into
    ``alpha_from_gt_group`` would DOUBLE-COUNT that sqrt(2) and return an alpha
    -- and therefore a ratio -- that belongs to no surface.

    Converting back is legitimate rather than a fudge, and R-16.3 is why: the
    ratio is INVARIANT under surface combination.  If both surfaces share a
    bandwidth alpha then sigma_s,i = r sigma*,i with the same r, so

        sigma_s,eq / sigma*_eq
            = sqrt(ss1^2 + ss2^2) / sqrt(s*1^2 + s*2^2) = r

    for ANY pair of amplitudes.  The ratio is a property of the BANDWIDTH, not
    of the roughness, so the single-surface computation transfers to the joint
    quantity unchanged.  The surfaces need not be equally rough -- only equally
    BANDED, which is the assumption box D registers under
    ``summit_statistics["assumption"]``.
    """
    single = float(eta_beta_sigma_joint) / math.sqrt(2.0)
    lo, hi = GT_PARAM_BAND["eta_a_beta_a_sigma_s"]["single_surface_reading"]
    if not (lo - 1e-12 <= single <= hi + 1e-12):
        raise ValueError(
            f"joint (eta beta sigma) = {eta_beta_sigma_joint:g} implies a "
            f"single-surface value {single:g} outside the CITED band "
            f"[{lo:g}, {hi:g}].  Either the wrong band end was passed, or the "
            f"joint value was passed where a single-surface one was expected.")
    ratio = sigma_s_over_sigma_star(alpha_from_gt_group(single))
    r_lo, r_hi = f3_operating_band()["sigma_s_over_sigma_star"]
    if not (min(r_lo, r_hi) - 1e-9 <= ratio <= max(r_lo, r_hi) + 1e-9):
        raise ValueError(
            f"derived ratio {ratio:g} falls outside box D's registered "
            f"operating band [{r_lo:g}, {r_hi:g}]; the derivation and the "
            f"registration disagree.  STOP.")
    return ratio

def corrected_outcome_taxonomy() -> dict:
    """Box D's taxonomy with box E's erratum E-10 applied.

    Box D registers `clearance_nonconvergence`; the code and the Outcome enum
    emit `clearance_nonconverged`, and box E adopts the EMITTED spelling.  Box
    D is not edited, so the correction is applied HERE, as a function -- which
    is what an erratum is for.

    Gate 1 must check against this rather than against box D's raw taxonomy, or
    it reports a permanent failure for a value box E has already ruled on.
    """
    import copy
    tax = copy.deepcopy(OUTCOME_TAXONOMY)
    fix = BOX_D_ERRATA["E-10_clearance_outcome_spelling"]
    old, new = fix["registered"], fix["emitted"]
    for k, v in tax.items():
        if isinstance(v, (list, tuple)):
            tax[k] = type(v)(new if x == old else x for x in v)
    return tax


#: The modules that PRODUCE rows.  Gate 1 scans these for taxonomy literals,
#: and the list is registered here rather than passed ad hoc because choosing
#: it too narrowly makes the gate report failures that are artefacts of the
#: scan.  That happened: campaign+solver alone omitted outcomes.py, where the
#: Outcome enum and the typed at-limit columns are defined, so three values
#: read as unproducible when they were merely unscanned.  The gate answers
#: "do these appear in the modules I was given", which is only the question
#: needed if the module list is right.
#: Retained only as the fallback if package walking fails.  It is NOT the
#: source of truth: a hand-written scope is a choice determining a registered
#: outcome, which by this repository's own convention must be registered or
#: DERIVED (ruling R-29.2).  A check's scope is part of the check.
ROW_PRODUCING_MODULES = ("ball_socket.campaign", "ball_socket.solver",
                         "ball_socket.outcomes", "ball_socket.mixed")


def row_producing_modules() -> list:
    """Every module in the ball_socket package, DERIVED by walking it.

    Gate 1 reported four failures that were artefacts of ITS OWN SCOPE: three
    taxonomy values lived in ``outcomes.py``, which the hand-written list
    omitted, so they read as unproducible when they were merely unscanned.  A
    conclusion about the code was drawn from a gate's configuration.

    Walking the package removes the parameter rather than correcting it -- a
    derived scope has nothing to be wrong about.  Sub-packages included, test
    modules excluded (a literal in a test proves nothing about production).
    """
    import importlib
    import pkgutil
    import ball_socket as _pkg

    mods = []
    for info in pkgutil.walk_packages(_pkg.__path__, _pkg.__name__ + "."):
        if ".tests" in info.name or info.name.endswith(".tests"):
            continue
        try:
            mods.append(importlib.import_module(info.name))
        except Exception:
            continue                      # unimportable module cannot produce
    if not mods:                          # pragma: no cover
        mods = [importlib.import_module(m) for m in ROW_PRODUCING_MODULES]
    return mods


def phase_entry_check(phase: str) -> dict:
    """Box E's phase gate, reading BOX E's state.

    The inherited version reads v6.DRAFT.  With box D frozen and box E in
    DRAFT it would have reported the box as signed and permitted phase entry --
    the same defect class as the inherited freeze_and_hash, found by the
    cross-box audit that defect prompted.  Latent rather than live only because
    both boxes happen to be frozen now.
    """
    if DRAFT:
        raise RuntimeError(
            f"box E is still DRAFT; phase {phase!r} may not be entered")
    return _v6.phase_entry_check(phase)


#: VERIFIED EQUIVALENT, not assumed.  assert_box_b_invariants and
#: blocking_unverified also read box D's config_dict() through their own
#: globals, but box E imports box D's shared subset WHOLESALE and changes none
#: of it, so the box-D answer and the box-E answer coincide by construction.
#: That is a per-function fact established by the audit, not a module-level
#: assumption -- which is the whole lesson of the freeze_and_hash defect.
_INHERITED_STATE_READERS_CHECKED = {
    "freeze_and_hash": "OVERRIDDEN -- would have re-frozen box D",
    "phase_entry_check": "OVERRIDDEN -- would have read box D's DRAFT",
    "assert_box_b_invariants": "inherited: shared subset is box D's, unchanged",
    "blocking_unverified": "inherited: box E adds no VERIFY constants",
}


def freeze_and_hash(path: str | None = None) -> str:
    """Digest box E's ``config_dict()`` and stamp ``campaign_frozen_v7.json``.

    THIS OVERRIDE IS NOT COSMETIC.  ``from .campaign_config_v6 import *`` binds
    box D's ``freeze_and_hash`` FUNCTION OBJECT, whose ``__globals__`` are box
    D's namespace.  Inherited unchanged it would read ``v6.DRAFT``, hash box D's
    ``config_dict()``, and default to writing ``campaign_frozen_v6.json`` --
    silently re-freezing box D and overwriting a hashed artefact that Chapter 9
    and the errata cite.  Overriding ``config_dict`` alone does not help,
    because the inherited function does not call box E's.

    Guards, both refusing rather than warning:
      * ``DRAFT`` is True until Nigel signs (standing rule 3);
      * box D's frozen hash must still verify, so box E cannot supersede an
        artefact that has drifted since it was frozen.
    """
    import hashlib
    import json
    import os

    if DRAFT:
        raise RuntimeError("v7 (box E) is still DRAFT; refusing to freeze")

    missing = blocking_unverified()
    if missing:
        raise RuntimeError(
            f"refusing to freeze box E: still VERIFY (NaN): {missing}")

    # Superseding an artefact whose hash has not been checked is superseding an
    # unknown.  Box D is verified here, before box E claims to supersede it.
    d_recomputed = hashlib.sha256(
        json.dumps(_v6.config_dict(), sort_keys=True, default=str).encode()
    ).hexdigest()
    if d_recomputed != BOX_D_CONFIG_SHA256:
        raise RuntimeError(
            f"box D's config now hashes to {d_recomputed}, not "
            f"{BOX_D_CONFIG_SHA256}; box D has drifted and box E's supersession "
            f"statement is invalid.  STOP.")

    assert_box_b_invariants()

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "campaign_frozen_v7.json")
    # Delegates to the SHARED guard (R-27.3).  This check began here, local to
    # one function; a guard only one caller invokes protects only that caller.
    from .immutable import assert_writable
    assert_writable(path, what="box-E frozen config")

    cfg = config_dict()
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    digest = hashlib.sha256(blob).hexdigest()
    with open(path, "w") as fh:
        json.dump({"sha256": digest, "config": cfg}, fh, indent=2, default=str)
    return digest


def config_dict() -> dict:
    """Box D's registered box, plus box E's three keys.

    Box D's content is carried through unchanged, so the supersession is
    additive and a box-E hash difference is exactly the three new keys.
    """
    cfg = _v6.config_dict()
    cfg.update({
        "DRAFT": DRAFT, "box": "E",
        "supersedes": {
            "box": "D",
            "module": "ball_socket/campaign_config_v6.py",
            "registration": "preregistration_v6_FROZEN.md",
            "frozen_config": "ball_socket/campaign_frozen_v6.json",
            "config_sha256": BOX_D_CONFIG_SHA256,
            "retained": "unchanged and superseded, never edited",
            "why": ("box D registered a mixed-lubrication campaign whose "
                    "contact path was unreachable from the campaign entry.  "
                    "Wiring it is execution, not scope -- but three choices the "
                    "wired campaign must make had no key, and registering them "
                    "moves the hash, which fires box D sec 0.3."),
            "prior_chain": _v6.config_dict().get("supersedes"),
        },
        "contact_model_traversal": dict(CONTACT_MODEL_TRAVERSAL),
        "undetermined_row_policy": dict(UNDETERMINED_ROW_POLICY),
        "f5_branch_selection": dict(F5_BRANCH_SELECTION),
        "preflight_stop_conditions": dict(PREFLIGHT_STOP_CONDITIONS),
        "box_d_errata": dict(BOX_D_ERRATA),
    })
    return cfg
