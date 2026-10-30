"""Stage D: coordinate-transform layer for the BO surrogates.

Maps the BO's working coordinates from the literal unit-cube
(V_T, p_s, |h_dot|, T_in, eta_0, lambda, eta_s/eta_0, k_F) to a set in
which the ARD-mode-collapse-prone dimensions (k_F and |h_dot|) are
re-parameterised:

  xi_1 = logit((k_F - k_F_min) / (k_F_max - k_F_min))  clipped to [-6, 6]
  xi_2 = log10(|h_dot| / |h_dot|_ref)  with |h_dot|_ref = 1e-3 m/s

The other six dimensions pass through unchanged.

Rationale (from the BO refinement campaign prompt's Stage D):
  The current parameterisation k_F = F_applied / F_hyd(p_s) is the
  variable that makes the failure-mode geography in (k_F, |h_dot|) space
  simple (lift-off for k_F<1, gap-collapse for k_F>1.5), which is also
  what makes k_F the dominant ARD direction and drives the mode-collapse.
  A logit-transform concentrates GP resolution near the lift-off and
  gap-collapse thresholds.

API:
  The TransformedSurrogateManager wraps a SurrogateManager. It maintains
  TWO sets of training-input coordinates:
    - Native space: (V_T, p_s, |h_dot|, T_in, eta_0, lambda, eta_s/eta_0, k_F)
      in the unit cube [0, 1]^8. This is what the SolverAdapter consumes.
    - Transformed space: same 8 dims with k_F and |h_dot| replaced by
      (xi_1, xi_2) on (still mostly) [0, 1] ranges after rescaling.
      This is what the GP sees.

  Forward (native -> transformed): used to compute training inputs the GP
  fits on.
  Backward (transformed -> native): used to compute X_next from any point
  the acquisition selects in transformed space before handing it to the
  solver.

  All BoTorch acquisition operates in transformed space; the
  SurrogateManager's GPs are fit on transformed inputs; only the
  solver, INPUT_SPACE, and Sobol-index reporting see native coordinates.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Optional

import numpy as np
import torch

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ar_bo_campaign import INPUT_SPACE  # noqa: E402

# k_F bounds from INPUT_SPACE: log-scaled (0.3, 3.0).  The unit-cube
# value u_kF maps to k_F = 0.3 * (3.0 / 0.3)^u_kF = 0.3 * 10^u_kF.
# We re-parameterise the unit-cube column directly (the inputs
# the GP sees are unit-cube; the transform operates on those).
KF_DIM_INDEX = 7      # index of k_F in INPUT_SPACE
HDOT_DIM_INDEX = 2    # index of |h_dot| in INPUT_SPACE
XI1_CLIP = (-6.0, 6.0)


def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def transform_inputs(X_native: torch.Tensor) -> torch.Tensor:
    """Apply the Stage D coordinate transform.

    Inputs: X_native of shape (N, 8) in the unit cube [0, 1]^8.
    Outputs: X_xi of shape (N, 8) where columns KF_DIM_INDEX and
    HDOT_DIM_INDEX have been replaced with xi_1 and xi_2 respectively.
    """
    out = X_native.clone()
    # k_F: logit-transform of the unit-cube column (concentrates
    # resolution near the lift-off/gap-collapse thresholds).
    u_kF = X_native[:, KF_DIM_INDEX]
    xi1 = _logit(u_kF)
    xi1 = torch.clamp(xi1, *XI1_CLIP)
    out[:, KF_DIM_INDEX] = xi1
    # |h_dot|: log-transform centred on midpoint. The unit cube already
    # makes |h_dot| log-scaled (1e-4 to 1e-1), so the unit-cube
    # coordinate IS log-scaled. We map to a Cartesian band centred at
    # zero by shifting (u - 0.5) * span, with span chosen to put the
    # midpoint at 0. The span here is 3 (matching the 3-decade log range
    # of |h_dot|).
    u_h = X_native[:, HDOT_DIM_INDEX]
    xi2 = 3.0 * (u_h - 0.5)
    out[:, HDOT_DIM_INDEX] = xi2
    return out


def untransform_inputs(X_xi: torch.Tensor) -> torch.Tensor:
    """Inverse of `transform_inputs`."""
    out = X_xi.clone()
    xi1 = X_xi[:, KF_DIM_INDEX]
    xi1 = torch.clamp(xi1, *XI1_CLIP)
    u_kF = _sigmoid(xi1)
    out[:, KF_DIM_INDEX] = u_kF
    xi2 = X_xi[:, HDOT_DIM_INDEX]
    u_h = xi2 / 3.0 + 0.5
    u_h = torch.clamp(u_h, 0.0, 1.0)
    out[:, HDOT_DIM_INDEX] = u_h
    return out


def transformed_unit_bounds(dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Acquisition bounds in transformed space. The non-transformed
    dimensions stay [0, 1]; xi_1 in [-6, 6]; xi_2 in [-1.5, 1.5].
    """
    d = len(INPUT_SPACE)
    lo = torch.zeros(d, dtype=dtype, device=device)
    hi = torch.ones(d, dtype=dtype, device=device)
    lo[KF_DIM_INDEX] = XI1_CLIP[0]
    hi[KF_DIM_INDEX] = XI1_CLIP[1]
    lo[HDOT_DIM_INDEX] = -1.5
    hi[HDOT_DIM_INDEX] = 1.5
    return torch.stack([lo, hi])


def clip_rate(X_native: torch.Tensor, eps: float = 1e-4) -> float:
    """Fraction of rows where u_kF is within eps of 0 or 1 (would be
    clipped by the logit). Diagnostic for the Stage D stop-and-flag."""
    u = X_native[:, KF_DIM_INDEX]
    return float(((u < eps) | (u > 1.0 - eps)).float().mean().item())


# ---------------------------------------------------------------------------
# Wrapper SurrogateManager
# ---------------------------------------------------------------------------


class TransformedSurrogateManager:
    """Wrap a SurrogateManager and present a transformed-input interface
    to the BO acquisition/loop.

    External callers (BOCampaign, AcquisitionManager) interact with this
    object the same way they would with a plain SurrogateManager, but
    every input they hand in is treated as living in transformed space.
    The wrapper detransforms before calling the underlying surrogates.

    For Stage D, the underlying surrogates ARE the ones fitting in
    transformed space (so the GP lengthscales are on (xi_1, xi_2)). The
    wrapper just makes this explicit at the API boundary.
    """

    def __init__(self, surrogates):
        self._sm = surrogates

    # Pass-through attributes
    @property
    def objective_names(self): return self._sm.objective_names
    @property
    def constraint_names(self): return self._sm.constraint_names
    @property
    def auxiliary_names(self): return self._sm.auxiliary_names
    @property
    def output_names(self): return self._sm.output_names
    @property
    def d(self): return self._sm.d
    @property
    def _gps(self): return self._sm._gps  # for analysis_F / Stage A consumers
    @property
    def _classifier(self): return self._sm._classifier

    def fit(self, X: torch.Tensor, Y: dict, *, converged=None) -> None:
        """Fit on transformed inputs. X is in NATIVE coordinates."""
        X_xi = transform_inputs(X)
        self._sm.fit(X_xi, Y, converged=converged)

    def predict(self, output: str, X_test: torch.Tensor):
        """X_test is in NATIVE coordinates."""
        X_xi = transform_inputs(X_test)
        return self._sm.predict(output, X_xi)

    def predict_feasibility(self, X_test: torch.Tensor):
        """X_test is in NATIVE coordinates."""
        X_xi = transform_inputs(X_test)
        return self._sm.predict_feasibility(X_xi)

    def should_refit(self, n_obs_now: int) -> bool:
        return self._sm.should_refit(n_obs_now)

    def ard_lengthscales(self, output: str):
        """Lengthscales are reported in TRANSFORMED coordinates (so they
        are directly comparable across the eight dims after Stage D)."""
        return self._sm.ard_lengthscales(output)

    def stats(self) -> dict:
        s = self._sm.stats()
        s["coordinate_space"] = "transformed (xi_1, xi_2)"
        return s

    def group_of(self, name: str) -> str:
        return self._sm.group_of(name)
