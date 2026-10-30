"""Observed order, its domain, and the shared write guard.

R-35.2: the box-D candidate list 32 -> 48 -> 64 -> 96 has ratios 1.5, 1.333,
1.5.  Three pairwise changes across unequal ratios are not a convergence
sequence, and reading them as one produced the wrong verdict that legacy
``h_min`` was not converging.  It converges at first order.

R-35.7: a hand-list of protected basenames protects only what somebody named.
The shared writer refuses on EXISTENCE, which needs no foresight.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from ball_socket.verification import convergence as cv
from ball_socket import immutable as im


GRIDS = [(32, 24), (48, 36), (64, 48), (96, 72)]

#: The corrected-bracket audit, p95 per successive step.
MEASURED = {
    "h_min_bnd": [1.447, 0.209, 0.169],
    "De_sq_bnd": [0.091, 0.032, 0.023],
    "T_max_bnd": [0.324, 0.188, 0.203],
    "h_min":     [11.572, 6.139, 6.324],
    "P_loss":    [2.496, 1.977, 2.871],
}


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------

def test_the_list_admits_exactly_one_equal_ratio_pair():
    """Steps 0 and 2 both refine by 1.5; step 1 refines by 1.333."""
    assert cv.equal_ratio_pairs(GRIDS) == [(0, 2)]


@pytest.mark.parametrize("metric,want", [
    ("h_min_bnd", 3.10), ("De_sq_bnd", 1.98), ("T_max_bnd", 0.67),
    ("h_min", 0.87), ("P_loss", -0.20),
])
def test_observed_order_matches_the_ruling(metric, want):
    est = cv.observed_order(GRIDS, MEASURED[metric])
    assert est["p"] == pytest.approx(want, abs=0.01)


def test_legacy_h_min_matches_the_first_order_signature():
    """1 : 0.500 : 0.500 predicted; 1 : 0.530 : 0.546 measured.

    This is the confirmation that makes the order estimate a claim rather than
    an assertion -- and it is why "not converging" was withdrawn.
    """
    pred = cv.predicted_changes(GRIDS, 1.0)
    assert pred == pytest.approx([1.0, 0.5, 0.5], abs=1e-9)
    d = MEASURED["h_min"]
    meas = [x / d[0] for x in d]
    assert meas == pytest.approx([1.0, 0.530, 0.546], abs=0.002)
    for m, p in zip(meas[1:], pred[1:]):
        assert abs(m - p) < 0.06


def test_p_loss_is_genuinely_diverging():
    """Negative observed order: refinement makes it worse.  Not an artefact."""
    est = cv.observed_order(GRIDS, MEASURED["P_loss"])
    assert est["p"] < 0
    assert "DIVERG" in cv.classify(est["p"])


def test_the_estimator_is_exact_and_independent_of_amplitude():
    """d0/d2 = 2^p exactly, whatever A is: the (1 - r^-p) factor cancels."""
    for p_true in (0.5, 1.0, 2.0, 3.0):
        for A in (1.0, 137.0):
            d = [A * x for x in cv.predicted_changes(GRIDS, p_true)]
            assert cv.observed_order(GRIDS, d)["p"] == pytest.approx(p_true, abs=1e-9)


# ---------------------------------------------------------------------------
# the estimator's DOMAIN -- a refusal, not a zero
# ---------------------------------------------------------------------------

def test_no_equal_ratio_pair_means_no_order():
    """A list with all-distinct ratios must refuse, not guess."""
    est = cv.observed_order([(32, 24), (48, 36), (64, 48)], [1.0, 0.5])
    assert est["p"] is None
    assert "share a refinement ratio" in est["why"]


def test_changes_at_the_noise_floor_refuse():
    """``F`` moves 0.037, 0.006, 0.029 % and yields p = 0.34 from nothing.

    An order taken from round-off is a ratio of two noise figures.  The useful
    statement is that the metric is grid-insensitive.
    """
    est = cv.observed_order(GRIDS, [0.037, 0.006, 0.029])
    assert est["p"] is None
    assert est.get("at_noise_floor") is True


def test_anisotropic_refinement_is_rejected():
    """A single order is undefined if the two axes refine differently."""
    with pytest.raises(ValueError, match="anisotropic"):
        cv._ratio((32, 24), (48, 24))


# ---------------------------------------------------------------------------
# the shared writer (R-35.7)
# ---------------------------------------------------------------------------

def test_existing_file_refuses_without_foresight():
    """The half that was missing: no name list consulted, just existence."""
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "some_artefact_nobody_named.parquet")
        open(f, "w").write("x")
        with pytest.raises(im.ArtefactExists):
            im.guarded_path(f)


def test_overwrite_flag_allows_an_unnamed_file():
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "scratch.parquet")
        open(f, "w").write("x")
        assert im.guarded_path(f, overwrite=True) == f


def test_overwrite_flag_does_NOT_lift_protection():
    """A cited-by-hash artefact needs a deliberate move, never a flag."""
    with pytest.raises(im.ArtefactExists, match="never overwritten"):
        im.guarded_path(
            "results/ch9/audit_boxD_N8192_seed20260727_48x36.parquet",
            overwrite=True)


def test_authorised_overwrite_keeps_a_backup():
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "a.parquet")
        open(f, "w").write("old")
        im.guarded_write(f, lambda q: open(q, "w").write("new"), overwrite=True)
        assert open(f).read() == "new"
        assert any(x.startswith("a.parquet.superseded-") for x in os.listdir(d))


def test_new_path_is_allowed_and_parent_created():
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "deep", "new.parquet")
        assert im.guarded_path(f) == f
        assert os.path.isdir(os.path.dirname(f))


def test_a_protected_name_may_be_created_once_but_never_overwritten(tmp_path, monkeypatch):
    """freeze_and_hash must be able to write a frozen config the first time."""
    f = tmp_path / "campaign_frozen_vX.json"
    monkeypatch.setitem(im.PROTECTED_BASENAMES, f.name, "test frozen config")
    assert im.guarded_path(str(f)) == str(f)            # creation allowed
    f.write_text("{}")
    with pytest.raises(im.ArtefactExists, match="never overwritten"):
        im.guarded_path(str(f), overwrite=True)          # overwrite refused
