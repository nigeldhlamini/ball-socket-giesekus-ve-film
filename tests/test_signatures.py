"""Boxes D and E: the signature is attested against the frozen hashes, and the
frozen artefacts themselves still carry null (they are not edited)."""
import json
import pathlib

import pytest

from ball_socket import signatures as S

BS = pathlib.Path(S.__file__).resolve().parent


@pytest.mark.parametrize("box", ["D", "E"])
def test_attestation_matches_frozen_hash(box):
    assert S.frozen_sha(box) == S.ATTESTATIONS[box]["frozen_sha256"]
    assert S.signed_by(box) == "Nigel C. Dhlamini"


@pytest.mark.parametrize("box", ["D", "E"])
def test_frozen_json_not_edited(box):
    j = json.loads((BS / S.ATTESTATIONS[box]["frozen_json"]).read_text(encoding="utf-8"))
    assert j["config"]["SIGNED_BY"] is None


def test_attestation_refuses_a_changed_hash(monkeypatch):
    """Control: a different frozen digest voids the attestation."""
    monkeypatch.setattr(S, "frozen_sha", lambda box: "0" * 64)
    with pytest.raises(RuntimeError):
        S.signed_by("D")
