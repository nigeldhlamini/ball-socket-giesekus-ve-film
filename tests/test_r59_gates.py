"""R-59.2.4 (dirty-tree gate), R-59.3 (explicit memory_sign in new runners), R-59.4 (box G
attestation bound to the spec hash).

The runner audit is a SOURCE audit, the same pattern as the persist gate: every runner from
R-59 on must (i) call ``require_clean_tree`` before solving and (ii) pass the memory sign
explicitly wherever it builds or calls the solver.  The controls are runner texts that omit
each and must fail.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ball_socket import provenance as P  # noqa: E402
from ball_socket import signatures as S  # noqa: E402

#: runners written under the R-59 regime, beyond the run_R<nn>_ naming rule below
EXTRA_NEW_RUNNERS = {"run_E15_boxG.py"}
SOLVES = re.compile(r"_build\(|BallSocketSolver\(|\bevaluate_x\(|_evaluate_x_inner\(")
SIGN = re.compile(r"memory_sign\s*=|memory_sign=")


def _is_new_runner(name: str) -> bool:
    m = re.match(r"run_R(\d+)_", name)
    return bool(m and int(m.group(1)) >= 59) or name in EXTRA_NEW_RUNNERS


def audit(text: str) -> list[str]:
    """Violations of the R-59 runner rules in a runner's source."""
    bad = []
    if "require_clean_tree(" not in text:
        bad.append("no require_clean_tree (R-59.2.4)")
    if SOLVES.search(text) and not SIGN.search(text):
        bad.append("solves without an explicit memory_sign (R-59.3)")
    return bad


NEW_RUNNERS = sorted(p for p in (ROOT / "scripts").glob("run_*.py") if _is_new_runner(p.name))


def test_new_runner_set_is_not_empty():
    names = {p.name for p in NEW_RUNNERS}
    assert {"run_E15_boxG.py", "run_R59_linearity.py"} <= names


@pytest.mark.parametrize("path", NEW_RUNNERS, ids=lambda p: p.name)
def test_new_runners_obey_r59(path):
    assert audit(path.read_text(encoding="utf-8")) == []


def test_control_runner_without_sign_fails():
    txt = "from ball_socket.provenance import require_clean_tree\nrequire_clean_tree('x')\nslv, ci = B._build(x, True)\n"
    assert audit(txt) == ["solves without an explicit memory_sign (R-59.3)"]


def test_control_runner_without_clean_tree_fails():
    txt = "slv, ci = B._build(x, True)\nslv.memory_sign = 'signed'\n"
    assert audit(txt) == ["no require_clean_tree (R-59.2.4)"]


def test_dirty_tree_refuses(monkeypatch):
    def fake_git(*a):
        return "deadbeef\n" if a[0] == "rev-parse" else "?? scripts/run_R99_new.py\n"
    monkeypatch.setattr(P, "_git", fake_git)
    monkeypatch.setattr(P, "quarantine_state", lambda: {})
    with pytest.raises(SystemExit, match="dirty"):
        P.require_clean_tree("control")


def test_changed_quarantined_file_is_dirty(monkeypatch, tmp_path):
    f = tmp_path / "q.py"
    f.write_text("x = 1\n")
    man = tmp_path / "QUARANTINE_MANIFEST.json"
    man.write_text(json.dumps({"files": {"q.py": {"sha256": "0" * 64}}}))
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(P, "MANIFEST", man)
    assert P.quarantine_state() == {"q.py": "changed"}
    monkeypatch.setattr(P, "_git", lambda *a: "deadbeef\n" if a[0] == "rev-parse" else "")
    assert P.tree_state()["dirty"] is True


def test_quarantine_manifest_matches_disk():
    assert P.quarantine_state() and set(P.quarantine_state().values()) == {"ok"}


def test_box_G_attestation_bound_to_spec_hash():
    a = S.ATTESTATIONS["G"]
    sha = (ROOT / "results" / "ch9" / "boxG_E15_spec.sha256").read_text().strip()
    assert a["frozen_sha256"] == sha == S.frozen_sha("G")
    assert S.signed_by("G") == S.SIGNED_BY


def test_box_G_attestation_voided_by_changed_spec(monkeypatch):
    monkeypatch.setitem(S.ATTESTATIONS, "G", {**S.ATTESTATIONS["G"], "frozen_sha256": "0" * 64})
    with pytest.raises(RuntimeError):
        S.signed_by("G")


def _ch9():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ch9num_r59", ROOT / "scripts" / "ch9_numbers.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_memory_sign_provenance_clean():
    """R-59.3: every h_min/Lambda claim macro is signed or names the D-6 bound."""
    assert _ch9().check_memory_sign_provenance() == []


def test_memory_sign_control_held_claim_fails(monkeypatch):
    m = _ch9()
    monkeypatch.setattr(m, "PAPER3_HELD", m.PAPER3_HELD | {"PThreeEThirteenLamMaxProd"})
    bad = m.check_memory_sign_provenance()
    assert any("PThreeEThirteenLamMaxProd" in b and "neither signed nor bounded" in b for b in bad)


def test_err27_erratum_binds_box_D_to_its_registration_document():
    """R-61 B4: the erratum binds box D's config hash to preregistration_v6_DRAFT.md's hash."""
    b = S.erratum_binding("ERR-27")
    assert b["config_sha256"] == S.ATTESTATIONS["D"]["frozen_sha256"]
    assert b["registration_sha256"] != S.ERRATA["ERR-27"]["frozen_value"]


def test_err27_voided_by_a_moved_hash(monkeypatch):
    monkeypatch.setitem(S.ERRATA, "ERR-27", {**S.ERRATA["ERR-27"], "registration_sha256": "0" * 64})
    with pytest.raises(RuntimeError):
        S.erratum_binding("ERR-27")
