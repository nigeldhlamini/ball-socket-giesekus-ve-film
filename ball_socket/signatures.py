"""Signature attestations for frozen boxes whose ``SIGNED_BY`` was left null.

Boxes D (``campaign_config_v6``) and E (``campaign_config_v7``, which inherits box
D's ``config_dict()``) were frozen with ``SIGNED_BY = None``.  ``SIGNED_BY`` sits
INSIDE ``config_dict()``, so writing a name into the frozen module would change
each box's sha256 and break every artefact that cites it.  The frozen modules and
JSON are therefore not edited (standing rule: errata for frozen artefacts).  The
signature is recorded here instead, BOUND TO THE HASHES it attests: if either
frozen JSON's sha256 ever differs from the value below, the attestation no longer
applies and ``tests/test_signatures.py`` fails.

Instruction: Nigel C. Dhlamini, 2026-10-01, with ruling R-56 ("Consider boxes D
and E signed").  The attestation is retroactive and is dated when it was given.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent

ATTESTATIONS = {
    "D": {"module": "campaign_config_v6", "frozen_json": "campaign_frozen_v6.json",
          # config digest stamped in the frozen JSON (not v6.REGISTRATION_SHA256,
          # which digests the signed DRAFT text and is a different quantity)
          "frozen_sha256": "0ea6081033342f30e59a49511b0e034dd472fbdc760af59c3a14e11882f79e14",
          "frozen_on": "2026-08-31"},
    "E": {"module": "campaign_config_v7", "frozen_json": "campaign_frozen_v7.json",
          "frozen_sha256": "9b24371fcad90db16c4a4e733ee8a77a82b8996728b2c769004959c51ae04c04",
          "frozen_on": "2026-09-01"},
    # Box G (E15): the spec is a JSON file hashed whole (scripts/run_E15_boxG.py stage
    # 'spec'); the binding is that FILE's sha256.  Instruction: Nigel C. Dhlamini,
    # 2026-10-04, with ruling R-59 ("Consider box G signed").  Signed before any box-G
    # solve (R-58.4.1); the run's commit goes in run provenance, not here (R-59.2.5).
    "G": {"spec_file": "../results/ch9/boxG_E15_spec.json",
          "frozen_sha256": "376bfb06d2261d5d27cc1ce37d41fc52b0f4efcc7b8bcdf12f6fa63e3e4d1afc",
          "frozen_on": "2026-10-04", "attested_on": "2026-10-04",
          "instruction": "R-59 hand-off: 'Consider box G signed.'"},
}
SIGNED_BY = "Nigel C. Dhlamini"
ATTESTED_ON = "2026-10-01"
INSTRUCTION = "R-56 hand-off: 'Consider boxes D and E signed.'"


def frozen_sha(box: str) -> str:
    a = ATTESTATIONS[box]
    if "spec_file" in a:
        return hashlib.sha256((_HERE / a["spec_file"]).read_bytes()).hexdigest()
    j = json.loads((_HERE / ATTESTATIONS[box]["frozen_json"]).read_text(encoding="utf-8"))
    return str(j["sha256"])


def signed_by(box: str) -> str:
    """The signatory of ``box``, valid only while its frozen hash is the attested one."""
    a = ATTESTATIONS[box]
    if frozen_sha(box) != a["frozen_sha256"]:
        raise RuntimeError(f"box {box}: frozen sha256 is no longer the attested one; "
                           f"the signature attestation does not apply")
    return SIGNED_BY

#: Signed erratum records (R-61 B4).  Frozen code is never edited; a correction to a frozen
#: constant is recorded here, BOUND to the hashes it relates, and is valid only while both
#: still match the bytes on disk.  ``attested_by`` is None until Nigel attests the record.
ERRATA = {
    "ERR-27": {
        "box": "D",
        "frozen_constant": "campaign_config_v6.REGISTRATION_SHA256",
        "frozen_value": "bdf3a4b0b9faf5a67c5e87c0cbe14632b9d515e4d6ea708eed2dde1d2663854a",
        "frozen_value_is": "sha256 of preregistration_v5_DRAFT.md (box C's registered content)",
        "config_sha256": "0ea6081033342f30e59a49511b0e034dd472fbdc760af59c3a14e11882f79e14",
        "registration_document": "preregistration_v6_DRAFT.md",
        "registration_sha256": "64c88b4682e1b3976d3a0dace8e70eda4e0d896672b2e9a4e031fda04bef2923",
        "record": "reports/boxe_errata.md ERR-27",
        "attested_by": "Nigel C. Dhlamini",
        "attested_on": "2026-10-06",
        "instruction": "R-62.6 hand-off: 'attest ERR-27'",
    },
}


def erratum_binding(err: str) -> dict:
    """Check an erratum's bound hashes against the bytes on disk.  Raises if either moved."""
    e = ERRATA[err]
    cfg = frozen_sha(e["box"])
    if cfg != e["config_sha256"]:
        raise RuntimeError(f"{err}: box {e['box']} config hash moved; the erratum does not apply")
    doc = _HERE.parent / e["registration_document"]
    got = hashlib.sha256(doc.read_bytes()).hexdigest() if doc.exists() else None
    if got != e["registration_sha256"]:
        raise RuntimeError(f"{err}: {e['registration_document']} is missing or its hash moved")
    return {"config_sha256": cfg, "registration_sha256": got, "attested_by": e["attested_by"]}
