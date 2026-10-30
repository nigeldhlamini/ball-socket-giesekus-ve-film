"""D-6 fix (R-57.4, R-58.1): the SIGNED squeeze-memory viscosity on the ball-socket path.

`src/rheology.py:429` returns M = eta_p |[J^-1 g]_xy| / Lambda.  Paper 2 derived the sign
law (M negative above Lambda_0(alpha); Lambda_0(0.25) = 1.2497) and applies it with a
delegating wrapper around the shared modulus; the ball-socket path did not.  src/ is NOT
touched -- it is shared with Papers 1 and 2 and the running Ch. 7 re-sweep.

VENDORED, not imported (R-58.1).  Importing Paper 2's modules executes, at module level,
more than definitions.  In paper2_appendixC_asymptotics.py: a global matplotlib backend
switch (:36-37), `sys.path` insertion of Paper 2's repository (:40-42),
`sys.stdout/stderr.reconfigure` (:47-51) and a process-wide
`warnings.filterwarnings("ignore")` (:52); paper2_numbers.py likewise inserts its
repository on `sys.path` and imports Paper 2's own `src` (:47-51).  R-58.1's rule for that
case is to vendor the two objects, with a test asserting their outputs equal Paper 2's on an
(alpha, Lambda) grid (tests/test_memory_sign.py).  The texts below are Paper 2's, verbatim:

  signed_M_over_etap  giesekus-paper2/scripts/paper2_appendixC_asymptotics.py:74-98
                      (with `_LAM_FIXED` :62 and `_fluid` :65-66)
  _SignedMemoryFluid  giesekus-paper2/scripts/paper2_numbers.py:1402-1419
                      (branch paper2-analysis, 34bcf4d)

and `src.rheology` is asserted to be this repository's copy, byte-identical to Paper 2's
(both at 7d5d518), before anything here is used: the vendored text calls src's
`_stress_components`, so a different src would change the sign law silently.
"""
from __future__ import annotations

import hashlib
import pathlib

import numpy as np

import src.rheology as _rheology
from src.rheology import GiesekusFluid

_HERE_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "rheology.py"
PAPER2_RHEOLOGY = pathlib.Path(r"C:\Users\Administrator\Documents\giesekus-paper2\src\rheology.py")


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def assert_src_identity() -> str:
    """Fail loudly unless `src.rheology` is THIS repository's file and is byte-identical
    to Paper 2's copy.  Returns the common sha256."""
    loaded = pathlib.Path(_rheology.__file__).resolve()
    if loaded != _HERE_SRC.resolve():
        raise ImportError(f"src.rheology resolved to {loaded}, not the ball-socket copy "
                          f"{_HERE_SRC}; the vendored sign law would run on another src")
    here = _sha(_HERE_SRC)
    if PAPER2_RHEOLOGY.exists() and _sha(PAPER2_RHEOLOGY) != here:
        raise ImportError("src/rheology.py differs from Paper 2's copy; the vendored "
                          "signed_M_over_etap is no longer Paper 2's sign law")
    return here


# ---------------------------------------------------------------------------
# VERBATIM from giesekus-paper2/scripts/paper2_appendixC_asymptotics.py:62-98
# ---------------------------------------------------------------------------
_LAM_FIXED = 3.3e-3               # reference lambda (scale-free in M/eta_p)


def _fluid(alpha):
    return GiesekusFluid(eta_s=0.03, eta_p=0.06, lambda_=_LAM_FIXED, alpha=alpha)


def signed_M_over_etap(alpha, Lam, eps=1e-6):
    """SIGNED M/eta_p (CP-6 Task 1). The solver returns eta_p*|[J^-1 g]_xy|/Lambda
    (src/rheology.py:429); the abs() hides a genuine zero crossing. With
    M := -[J^-1 g]_xy/Lambda the documented low-shear limit M -> +eta_p is
    recovered and M is NEGATIVE above Lambda_0(alpha). src/ is frozen -- this is
    a paper-side sign convention, computed here."""
    f = _fluid(alpha)
    a = alpha
    out = []
    for Lv in np.atleast_1d(np.asarray(Lam, float)):
        L = max(float(Lv), 1e-10)
        S = f._stress_components(L)
        Sxx, Syy, Sxy = (float(np.atleast_1d(x)[0]) for x in S)
        dL = max(eps * L, 1e-14)
        p = f._stress_components(L + dL)
        m = f._stress_components(max(L - dL, 1e-14))
        g = np.array([(float(np.atleast_1d(p[i])[0]) - float(np.atleast_1d(m[i])[0]))
                      / (2 * dL) * L for i in range(3)])
        J = np.array([[-1 - 2 * a * Sxx, 0.0, 2 * L - 2 * a * Sxy],
                      [0.0, -1 - 2 * a * Syy, -2 * a * Sxy],
                      [-a * Sxy, L - a * Sxy, -1 - a * (Sxx + Syy)]])
        out.append(-np.linalg.solve(J, g)[2] / L)
    r = np.array(out)
    return float(r[0]) if np.ndim(Lam) == 0 else r


# ---------------------------------------------------------------------------
# VERBATIM from giesekus-paper2/scripts/paper2_numbers.py:1402-1419
# ---------------------------------------------------------------------------
class _SignedMemoryFluid:
    """Delegating wrapper that returns the SIGNED memory viscosity (CP-7 Task 2).
    Everything else is the base fluid, so the only change to the solve is the sign
    of the squeeze-memory flux. src/ stays frozen."""

    def __init__(self, base):
        self._b = base

    def __getattr__(self, n):
        return getattr(self._b, n)

    def memory_viscosity(self, gdot):
        gd = np.asarray(gdot, dtype=float)
        Lam = self._b.lambda_ * np.abs(gd)
        sgn = np.sign(signed_M_over_etap(self._b.alpha, np.atleast_1d(Lam).ravel()))
        mag = np.asarray(self._b.memory_viscosity(gd), dtype=float)
        out = mag * sgn.reshape(np.shape(mag)) if np.ndim(mag) else float(mag) * float(sgn[0])
        return out
# ---------------------------------------------------------------------------


MEMORY_SIGN_CHOICES = ("modulus", "signed")


def wrap(fluid, memory_sign: str):
    """The fluid the solver uses.  'modulus' = the registered campaigns (boxes A-F, E13,
    E5 ...), bit-for-bit; 'signed' = D-6 fixed."""
    if memory_sign not in MEMORY_SIGN_CHOICES:
        raise ValueError(f"memory_sign must be one of {MEMORY_SIGN_CHOICES}, got {memory_sign!r}")
    if memory_sign == "modulus":
        return fluid
    assert_src_identity()
    return _SignedMemoryFluid(fluid)
