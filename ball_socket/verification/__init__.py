"""Formal verification gates for the ball-socket spherical solver.

Each gate writes a JSON verdict under this directory; the gate names (BS-V1..V6)
are the citable identifiers for Ch.9 prose.  BS-V1/V3/V5 physics is additionally
covered by ``ball_socket/tests/test_spherical.py`` (11/11 passing).
"""
from .gates import GATES, run_gate, run_all

__all__ = ["GATES", "run_gate", "run_all"]
