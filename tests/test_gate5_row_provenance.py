"""Gate 5 (R-45.6) and the A/B invariant (R-45.7).

Both exist because of one defect: a derived column named `lam` -- the
relaxation-time INPUT AXIS -- was overwritten with Lambda = h_min/sigma*, and
570 rows were re-solved at the wrong fluid.  Gates 1-4 could not see it: they
police the registration/campaign boundary, and this was a row's IDENTITY.

Every test here is a POSITIVE CONTROL or its paired negative: each biting test
replays a defect that actually happened and fails if the gate is removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ball_socket.verification import preflight as pf


@pytest.fixture
def artefact():
    return pd.DataFrame({
        "sample_id": [1, 1, 2], "model_id": [0, 1, 0],
        "omega": [100.0, 100.0, 200.0], "p_ch": [2e7, 2e7, 3e7],
        "c0": [5e-6, 5e-6, 9e-6], "T_in": [60.0, 60.0, 70.0],
        "eta_0": [0.02, 0.02, 0.04], "lam": [1e-4, 1e-4, 2e-3],
        "eta_s_ratio": [0.5, 0.5, 0.3], "theta_min_deg": [30.0, 30.0, 28.0]})


def test_the_reserved_names_are_the_registered_input_axes():
    assert pf.input_axis_names() == ("omega", "p_ch", "c0", "T_in", "eta_0",
                                     "lam", "eta_s_ratio", "theta_min_deg")


def test_a_derived_column_may_not_shadow_an_input_axis():
    """THE DEFECT ITSELF: Lambda written into a column called `lam`."""
    fails = pf.reserved_names_free(None, derived=["lam", "stratum"])
    assert fails and "lam" in fails[0]
    assert pf.reserved_names_free(None, derived=["Lambda_g1", "stratum"]) == []


def test_row_provenance_passes_on_untouched_rows(artefact):
    assert pf.row_provenance(artefact.copy(), artefact,
                             key=["sample_id", "model_id"]) == []


def test_row_provenance_catches_the_overwritten_axis(artefact):
    """Replay of the real defect: `lam` overwritten by a derived Lambda."""
    frame = artefact.copy()
    frame["lam"] = frame.c0 / 1.7e-6          # a Lambda-like quantity
    fails = pf.row_provenance(frame, artefact, key=["sample_id", "model_id"])
    assert len(fails) == len(frame)
    assert "input axis lam" in fails[0]


def test_row_provenance_is_bitwise_by_default(artefact):
    frame = artefact.copy()
    frame.loc[0, "eta_0"] = frame.loc[0, "eta_0"] * (1 + 1e-15)
    assert pf.row_provenance(frame, artefact, key=["sample_id", "model_id"])
    assert pf.row_provenance(frame, artefact, key=["sample_id", "model_id"],
                             atol=1e-12) == []


def test_row_provenance_refuses_an_unknown_or_ambiguous_identity(artefact):
    frame = artefact.copy()
    frame.loc[0, "sample_id"] = 99
    assert any("no such row" in f for f in
               pf.row_provenance(frame, artefact, key=["sample_id", "model_id"]))
    dup = pd.concat([artefact, artefact.iloc[[0]]], ignore_index=True)
    assert any("not unique" in f for f in
               pf.row_provenance(artefact, dup, key=["sample_id", "model_id"]))


def test_ab_invariant_fires_when_both_arms_move_together():
    """THE DIAGNOSTIC THAT FOUND IT: 24 of 570 rows went non-converged under
    BOTH conventions while the artefact had them converged."""
    base = np.array(["converged"] * 5)
    a = np.array(["converged", "contact_limit", "converged", "converged", "converged"])
    b = np.array(["converged", "contact_limit", "converged", "contact_limit", "converged"])
    fails = pf.ab_invariant(base, a, b)
    assert len(fails) == 1 and "row 1" in fails[0]


def test_ab_invariant_is_quiet_when_only_one_arm_moves():
    base = np.array(["converged"] * 3)
    a = np.array(["converged", "contact_limit", "converged"])
    b = np.array(["converged", "converged", "converged"])
    assert pf.ab_invariant(base, a, b) == []
