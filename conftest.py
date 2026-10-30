"""Tests that need files withheld from this repository (QUARANTINE_MANIFEST.json) or campaign outputs
that are not distributed.  They run in the working repository; here they are skipped, each with the
file it needs."""
import pytest

WITHHELD = {
    "ball_socket/tests/test_boxd_gate_arming.py::test_attrs_route_is_documented_as_dead":
        "ball_socket/campaign_config_v6.py (withheld)",
    "tests/test_metadata_attachment_sweep.py::test_the_alignment_assertion_catches_a_shuffled_frame":
        "ball_socket/campaign_config_v6.py (withheld)",
    "tests/test_memory_sign.py::test_default_is_the_registered_modulus_path":
        "ball_socket/campaign_config_v5.py (withheld)",
    "tests/test_convergence_order.py::test_overwrite_flag_does_NOT_lift_protection":
        "an existing campaign artefact under results/ (not distributed)",
    "tests/test_r59_gates.py::test_quarantine_manifest_matches_disk": "the withheld files themselves",
    "tests/test_r59_gates.py::test_box_G_attestation_bound_to_spec_hash": "results/ch9/boxG_E15_spec.json (withheld)",
    "tests/test_r59_gates.py::test_box_G_attestation_voided_by_changed_spec": "results/ch9/boxG_E15_spec.json (withheld)",
    "tests/test_r59_gates.py::test_memory_sign_provenance_clean": "campaign artefacts under results/ (not distributed)",
    "tests/test_r59_gates.py::test_err27_erratum_binds_box_D_to_its_registration_document":
        "ball_socket/campaign_frozen_v6.json (withheld)",
    "tests/test_r59_gates.py::test_err27_voided_by_a_moved_hash": "ball_socket/campaign_frozen_v6.json (withheld)",
    "tests/test_signatures.py::test_attestation_matches_frozen_hash[D]": "ball_socket/campaign_frozen_v6.json (withheld)",
    "tests/test_signatures.py::test_attestation_matches_frozen_hash[E]": "ball_socket/campaign_frozen_v7.json (withheld)",
    "tests/test_signatures.py::test_frozen_json_not_edited[D]": "ball_socket/campaign_frozen_v6.json (withheld)",
    "tests/test_signatures.py::test_frozen_json_not_edited[E]": "ball_socket/campaign_frozen_v7.json (withheld)",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        need = WITHHELD.get(item.nodeid.replace("\\", "/"))
        if need:
            item.add_marker(pytest.mark.skip(reason=f"needs {need}"))
