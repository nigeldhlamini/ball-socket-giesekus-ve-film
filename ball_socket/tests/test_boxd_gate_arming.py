"""The claims gate must arm on the ACTUAL production artefact name.

Ruling R-22.2.  Until ``write_parquet`` began stripping ``df.attrs``, the box-D
claims gate had two arming routes: the artefact key and the attrs blob.  The
attrs strip was correct -- timing provenance was leaking into the artefact's
sha256 -- but it silently removed the redundant route.  The FILENAME then became
the only thing arming the gate, and a literal substring match left plausible
names like ``audit_N8192_boxD_...`` disarming it.

These tests exist so that a rename cannot silently switch the gate off.  A gate
that fails open is worse than no gate, because it is trusted.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]

#: The registered production artefact for the box-D campaign.  If this string
#: changes, the campaign driver's default must change with it AND this test
#: must be updated deliberately -- which is the point.
PRODUCTION_ARTEFACT = "audit_boxD_N8192_seed20260727_48x36.parquet"


def _ch9():
    spec = importlib.util.spec_from_file_location(
        "ch9n", _REPO / "scripts" / "ch9_numbers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gate_arms_on_the_production_filename():
    """The one that matters: the real name must arm the gate."""
    m = _ch9()
    hard, rows = m.check_claims({PRODUCTION_ARTEFACT: None})
    assert hard, (
        f"the claims gate did NOT arm on the production artefact name "
        f"{PRODUCTION_ARTEFACT!r}.  Box-D numbers would be published against "
        f"PENDING_BOXD claims with no failure raised.")
    assert any(r[1] == "PENDING_BOXD" for r in rows)


@pytest.mark.parametrize("name", [
    "audit_boxD_N8192_seed20260727_48x36.parquet",
    "audit_N8192_boxD_seed20260727.parquet",     # marker split by a segment
    "audit_boxd_N8192_seed20260727.parquet",     # lower-case d
    "audit_BoxD_N8192.parquet",                  # different capitalisation
    "audit_box-d_N8192.parquet",                 # hyphen separator
])
def test_gate_arms_on_plausible_boxd_names(name):
    """Near-misses must arm too.  Widening is the fail-safe direction: an
    over-armed gate costs a spurious HARD failure a human clears in a minute,
    an under-armed one ships stale claims."""
    assert _ch9().check_claims({name: None})[0], f"{name!r} failed to arm"


@pytest.mark.parametrize("name", [
    "audit_N8192_seed20260727.parquet",          # the box-B artefact
    "audit_boxC_phase1_N8192.parquet",           # box C
    "oat_block_n256_summary.json",
])
def test_gate_does_not_arm_on_non_boxd_artefacts(name):
    """It must not fire on box-B or box-C artefacts, or the generator cannot
    run at all before the campaign lands."""
    assert not _ch9().check_claims({name: None})[0], f"{name!r} armed wrongly"


def test_attrs_route_is_documented_as_dead():
    """The attrs arming route no longer works for anything written through
    write_parquet.  If someone re-enables attrs writing, this test's premise
    changes and the comment must be revisited."""
    from ball_socket import campaign as C
    import inspect
    src = inspect.getsource(C.write_parquet)
    assert "df.attrs = {}" in src, (
        "write_parquet no longer strips attrs; the claims gate's attrs route "
        "may be live again and R-22.2's reasoning needs re-checking")
