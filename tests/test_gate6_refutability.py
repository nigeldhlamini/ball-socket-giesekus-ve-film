"""Gate 6 (R-49.4): every registered expectation names a refuting outcome and
whether it is reachable in the registered design.  Each test below can fail:
the positive controls construct the failing case."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ball_socket.verification import refutability as rf


#: The retrofit's unresolved items, pinned EXACTLY (R-50.4).  E10's premise,
#: the one owed item at R-50, was ruled at R-51.3: split out of the register
#: into the constants dossier (ERR-23).  The set is now EMPTY, and pinning it
#: empty means any new unresolved item fails this test.
KNOWN_OWED: set = set()   # R-52.5 struck E5's VG46 clause (ERR-25); its VG32
                          # half is reachable both ways, so it is OWED DATA,
                          # not a gate failure


def test_the_retrofit_covers_E1_to_E14_and_only_the_known_items_are_owed():
    assert set(rf.REGISTRY) == {f"E{i}" for i in range(1, 15)}
    owed = {f.split(":")[0] for f in rf.check()}
    assert owed == KNOWN_OWED


def test_e10_premise_left_the_register_for_the_dossier():
    e = rf.REGISTRY["E10"]
    assert "premise" not in e.get("clauses", {})
    assert e["moved_to_dossier"]["disposition"] == "ERR-23"


def test_every_expectation_names_both_outcomes_explicitly():
    """R-50.4: the confirming direction is stated per entry, never defaulted."""
    for k, e in rf.REGISTRY.items():
        assert str(e.get("confirming_outcome", "")).strip(), k
        assert e.get("confirmable") in rf.REACHABLE, k


def test_an_unconfirmable_expectation_without_a_disposition_fails():
    """POSITIVE CONTROL for the amended direction: E2's location clause shape."""
    bad = {"EX": {"claim": "x", "refuting_outcome": "y", "reachable": "yes",
                  "confirming_outcome": "z", "confirmable": "no"}}
    assert any("confirming outcome unreachable" in f for f in rf.check(bad))


def test_e2_location_clause_carries_the_e22_disposition():
    loc = rf.REGISTRY["E2"]["clauses"]["location"]
    assert loc["confirmable"] == "no" and loc["disposition"].startswith("ERR-22")


def test_an_expectation_without_a_refuting_outcome_fails():
    """POSITIVE CONTROL."""
    bad = {"EX": {"claim": "x", "refuting_outcome": "  ", "reachable": "yes",
                  "confirming_outcome": "z", "confirmable": "yes"}}
    assert any("no refuting outcome" in f for f in rf.check(bad))


def test_an_unreachable_refutation_without_a_disposition_fails():
    """POSITIVE CONTROL: E3's own shape, with its disposition removed."""
    bad = {"EX": {"claim": "x", "refuting_outcome": "y", "reachable": "no",
                  "confirming_outcome": "z", "confirmable": "yes"}}
    assert any("no disposition" in f for f in rf.check(bad))


def _rows(T_in, T_max, lam_ok=True):
    return pd.DataFrame({"T_in": T_in, "T_max_bnd": T_max,
                         "h_min_bnd": [5e-6 if lam_ok else 1e-7] * len(T_in),
                         "De_sq_bnd": [0.01] * len(T_in)})


def test_e3_exclusion_detects_rows_forced_by_the_inlet():
    f = _rows([70.0, 40.0], [70.5, 40.2])
    got = rf.e3_arithmetic_exclusion(f, ceiling=60.0, lambda_min=3.0,
                                     desq_max=0.2, sigma=3.36e-7)
    assert got["forced_sole_g2_rows"] == 1
    assert got["refuting_outcome_reachable"] is False


def test_e3_exclusion_is_quiet_when_every_inlet_is_below_the_ceiling():
    """The restricted design: the check must be able to return 'reachable'."""
    f = _rows([40.0, 55.0], [40.2, 58.0])
    got = rf.e3_arithmetic_exclusion(f, ceiling=60.0, lambda_min=3.0,
                                     desq_max=0.2, sigma=3.36e-7)
    assert got["refuting_outcome_reachable"] is True


def test_e3_exclusion_refuses_the_arithmetic_if_any_rise_is_negative():
    """The exclusion rests on T_max >= T_in; a cooling row breaks it."""
    f = _rows([70.0], [65.0])
    got = rf.e3_arithmetic_exclusion(f, ceiling=60.0, lambda_min=3.0,
                                     desq_max=0.2, sigma=3.36e-7)
    assert got["rise_nonnegative"] is False
    assert got["refuting_outcome_reachable"] is True


@pytest.mark.parametrize("theta_min_deg", [20.0, 30.0, 40.0])
@pytest.mark.parametrize("eps", [0.05, 0.5, 1.0])
def test_e2_location_clause_is_decided_by_construction(theta_min_deg, eps):
    assert rf.location_clause_by_construction(np.deg2rad(theta_min_deg),
                                              np.deg2rad(73.0), eps)


def test_the_location_check_can_return_false():
    """POSITIVE CONTROL: with the eccentricity reversed the minimum moves to
    theta_max, so the check is not true by its own construction."""
    assert not rf.location_clause_by_construction(np.deg2rad(20.0),
                                                  np.deg2rad(73.0), -0.5)


# ---- R-52.2: the claim is the frozen text, quoted -------------------------------

def test_e1_claim_is_the_frozen_text_verbatim():
    import pathlib
    root = pathlib.Path(rf.__file__).resolve().parents[2]
    line = (root / "preregistration_FROZEN.md").read_text(encoding="utf-8").splitlines()[116]
    assert line == "- **E1** — " + rf.REGISTRY["E1"]["claim"]
    assert rf.REGISTRY["E1"]["claim_source"].startswith("preregistration_FROZEN.md:117")
    assert "confined" not in rf.REGISTRY["E1"]["confirming_outcome"]


def test_e5_vg46_clause_is_struck_and_vg32_is_a_live_test():
    c = rf.REGISTRY["E5"]["clauses"]
    assert c["g1_binding_VG46"]["disposition"].startswith("ERR-25")
    assert c["g1_binding_VG32"]["reachable"] == c["g1_binding_VG32"]["confirmable"] == "yes"


# ---- R-52.6: boundary coincidence ------------------------------------------------

#: Retrospective result over E1-E14, pinned EXACTLY.  E3 and E5's VG46 clause are
#: the two the ruling named; E2 and E14 are what running it found.
KNOWN_BOUNDARY_HITS = {"E2", "E3", "E5.g1_binding_VG46", "E14"}


def test_retrospective_boundary_hits_are_exactly_the_known_set():
    hits = rf.boundary_coincidences()
    assert {h["label"] for h in hits} == KNOWN_BOUNDARY_HITS
    assert all(h["disposed"] for h in hits)


def test_registered_numbers_are_read_from_their_sources():
    """The check compares registered numbers; typing them would defeat it."""
    from ball_socket.campaign_config_v4 import ConstraintSet, INPUT_SPACE
    from ball_socket import e5_fluid_spec as E5
    cs = ConstraintSet()
    assert rf.BOUNDARIES["g2_mineral"]["value"] == cs.T_max_C_by_class["mineral"]
    assert rf.BOUNDARIES["g2_water_glycol"]["value"] == cs.T_max_C_by_class["water_glycol"]
    t = [d for d in INPUT_SPACE if d.name == "T_in"][0]
    assert rf.CONDITIONS["campaign_T_in"]["range"] == (t.lower, t.upper)
    assert rf.CONDITIONS["E5_T_in_VG32"]["range"] == (E5.SERVICE_T[32],) * 2
    assert rf.CONDITIONS["E5_T_in_VG46"]["range"] == (E5.SERVICE_T[46],) * 2


def _reg(pop, con, disposed=False):
    e = {"claim": "x", "refuting_outcome": "y", "reachable": "yes",
         "confirming_outcome": "z", "confirmable": "yes",
         "population": pop, "constraints": con}
    if disposed:
        e["boundary_disposition"] = "noted"
    return {"EX": e}


COND = {"on": {"var": "T_in_C", "range": (110.0, 110.0)},
        "below": {"var": "T_in_C", "range": (100.0, 100.0)},
        "span": {"var": "T_in_C", "range": (20.0, 110.0)}}


def test_a_point_on_a_boundary_is_flagged_and_fails_the_gate():
    """POSITIVE CONTROL: VG46's shape."""
    reg = _reg(["on"], ["g2_mineral"])
    hits = rf.boundary_coincidences(reg, COND)
    assert [h["kind"] for h in hits] == ["sits on"]
    assert not hits[0]["disposed"]


def test_a_range_spanning_a_boundary_is_flagged():
    """POSITIVE CONTROL: E3's shape."""
    hits = rf.boundary_coincidences(_reg(["span"], ["g2"]), COND)
    assert {(h["boundary"], h["kind"]) for h in hits} == {("g2_mineral", "spans"),
                                                         ("g2_water_glycol", "spans")}


def test_a_point_below_the_boundary_is_quiet():
    """VG32's shape: ten kelvin of headroom decides nothing."""
    assert rf.boundary_coincidences(_reg(["below"], ["g2_mineral"]), COND) == []


def test_a_claim_not_about_the_constraint_is_quiet():
    """E13's shape: on the boundary, but the claim is g1 alone."""
    assert rf.boundary_coincidences(_reg(["on"], ["g1"]), COND) == []


def test_an_undisposed_hit_fails_the_gate():
    reg = _reg(["E5_T_in_VG46"], ["g2_mineral"])
    assert any("no boundary_disposition" in f for f in rf.check(reg))
    reg = _reg(["E5_T_in_VG46"], ["g2_mineral"], disposed=True)
    assert rf.check(reg) == []
