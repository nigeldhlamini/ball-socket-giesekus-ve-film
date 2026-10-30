"""ball_socket -- Chapter 9 ball-and-socket viscoelastic Reynolds extension.

Rebuilt on the corrected ``thesis-validation`` solver core (branch
``ball-socket-v2``, replay-per-scaffold; not a cherry-pick of the old-physics
``feat/ball-socket`` reorg).  The constitutive layer (``GiesekusFluid``:
viscosity, Psi1/Psi2, memory viscosity ``M(alpha, Lambda)`` via
``memory_viscosity``, ``is_newtonian``) is IMPORTED from ``src.rheology`` and
never re-implemented; only the spherical geometry, differential operators,
boundary conditions, kinematic source, and the eccentricity equilibrium search
are new.

References: CODE_BRIEF_CH9_BALLSOCKET_2.md, ball_socket_scaffold_2.py,
CODE_PROMPT_PHASE0_1.md.  The spherical numerics were adapted from the
``feat/ball-socket`` reference (``src/geometry/spherical.py``,
``src/reynolds.py``) and re-wired onto the corrected core.
"""
from __future__ import annotations

from .outcomes import NonConvergence, Outcome
from . import representative

# NOTE: ``geometry_provenance`` is deliberately NOT re-exported here.  It is
# INDUSTRY-REPORT TRACK ONLY (drawing-exact values); the thesis campaign path
# imports ``representative`` (class values).  Ruling 2026-07-26.
__all__ = ["NonConvergence", "Outcome", "representative"]
