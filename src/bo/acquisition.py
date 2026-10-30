"""
AcquisitionManager: LogEIC + entropy-regularised mixed acquisition.

Implements the mixed acquisition of chapter 9 eq:mixed_acquisition:

    a_AR(x) = (1 - gamma) * EIC(x) + gamma * H(x)

with a numerically-stable log-space treatment.  EIC is the constrained
log-expected-improvement:

    log EIC(x) = log EI_obj(x)
                 + sum_j log P[g_j(x) >= 0]
                 + log P[evaluation succeeds | x]

where the first term comes from BoTorch's qLogExpectedImprovement on
the objective GP, the middle terms from the constraint regression GPs
via the analytic log-CDF identity for Gaussians, and the last term
from the classifier GP (Step 3's variational Bernoulli model).

H(x) is the additive Shannon entropy across constraint-feasibility
indicators in nats:

    H(x) = -sum_j [PF_j(x) log PF_j(x) + (1 - PF_j(x)) log (1 - PF_j(x))]

H(x) peaks at the AR boundary (where each PF_j approaches 0.5) and is
near-zero in the interior or far exterior of the feasible set.

The mix is computed in log space:

    log a_AR(x) = logsumexp(
        log(1 - gamma) + log EIC(x),
        log(gamma)     + log H(x),
    )

which is monotonically equivalent to the chapter's linear mix for
argmax purposes and stays numerically defined when log EIC underflows
(the very pathology Ament 2023 LogEI is designed to avoid).

The gamma schedule is a piecewise-linear ramp from 0 at the start of
the schedule window to a maximum (default 0.5) at the end:

    gamma(iter) = clip( gamma_max * (iter - start) / (end - start),
                        0, gamma_max )
"""

from __future__ import annotations

import dataclasses
import logging
import math
import warnings
from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.probability.utils import log_ndtr
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Gamma schedule
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GammaSchedule:
    """Piecewise-linear ramp 0 -> gamma_max from iter `start` to `end`."""
    start: int = 50
    end: int = 250
    gamma_max: float = 0.5

    def __call__(self, iteration: int) -> float:
        if iteration <= self.start:
            return 0.0
        if iteration >= self.end:
            return float(self.gamma_max)
        frac = (iteration - self.start) / (self.end - self.start)
        return float(self.gamma_max * frac)


# ---------------------------------------------------------------------
# Internal: mixed-log-acquisition AcquisitionFunction
# ---------------------------------------------------------------------


class _MixedLogAR(AcquisitionFunction):
    """log( (1-gamma) * EIC(x) + gamma * H(x) ) computed stably.

    Used internally by AcquisitionManager.optimize().  Wraps a
    pre-built qLogExpectedImprovement on the objective GP plus the
    SurrogateManager for constraint and classifier predictions.

    Forward expects X of shape (b, q, d) per the BoTorch convention,
    and returns a tensor of shape (b,) suitable for `optimize_acqf`.
    Internally we treat q=1 only for now (one new candidate per
    iteration), which simplifies the constraint factorisation; a future
    q>1 extension would need to handle the per-q feasibility products.

    Entropy-only mode (degenerate-EIC limit, sec:bo_ar_reconstruction)
    -----------------------------------------------------------------
    When `entropy_only=True`, the EIC contribution is treated as
    structurally zero ("a degenerate constant objective") and the
    acquisition reduces to

        log a(x) = log H(x) + log P_classifier(x)

    The constraint feasibility product Π PF_j(x) is INTENTIONALLY
    omitted because it would dampen the boundary signal: PF_j(x) ~ 0.5
    on the boundary makes log PF_j(x) ~ -0.69, which subtracts from
    log H(x) precisely where H peaks.  The classifier weighting is
    retained because it concentrates samples on regions where
    evaluation actually succeeds (avoids the chaotic-Picard failure
    corners surfaced in Phase 9-A Step 2).  This matches chapter 9's
    "gamma -> 1 limit" interpretation: the entropy term dominates the
    acquisition from iteration 1 and the BO becomes pure boundary
    pursuit.
    """

    def __init__(
        self,
        log_ei: Optional[qLogExpectedImprovement],
        surrogates,                       # SurrogateManager (Step 3)
        constraint_names: list[str],
        gamma: float,
        *,
        entropy_only: bool = False,
        log_zero_floor: float = -50.0,
    ) -> None:
        self._entropy_only = bool(entropy_only)
        if self._entropy_only and log_ei is not None:
            raise ValueError(
                "_MixedLogAR: entropy_only=True precludes log_ei (got non-None)"
            )
        if not self._entropy_only and log_ei is None:
            raise ValueError(
                "_MixedLogAR: log_ei is required when entropy_only=False"
            )
        # AcquisitionFunction parent needs a `model` for device/dtype
        # propagation. In entropy_only mode reuse any fitted GP from
        # the SurrogateManager.
        if log_ei is not None:
            model = log_ei.model
        else:
            if not surrogates._gps:
                raise ValueError(
                    "_MixedLogAR(entropy_only=True): surrogates has no fitted "
                    "GP to use as a model for device/dtype propagation"
                )
            model = next(iter(surrogates._gps.values())).gp
        super().__init__(model=model)
        self._log_ei = log_ei
        self._surrogates = surrogates
        self._constraint_names = list(constraint_names)
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1]; got {gamma}")
        self._gamma = float(gamma)
        # Below this floor we treat log-probabilities as -inf for the
        # purpose of logsumexp.  Plain -inf is fine arithmetically but
        # produces NaN gradients in some BoTorch internals; clamping
        # keeps the optimiser well-behaved.
        self._log_zero_floor = float(log_zero_floor)

    @property
    def gamma(self) -> float:
        return self._gamma

    def forward(self, X: Tensor) -> Tensor:
        # BoTorch passes X as (b, q, d).  We support q=1 only.
        if X.dim() != 3 or X.shape[1] != 1:
            raise ValueError(
                f"_MixedLogAR.forward: expected (b, q=1, d) input, got {tuple(X.shape)}"
            )
        b, _q, d = X.shape
        X_flat = X.squeeze(1)        # (b, d)

        # Entropy-only fast path (degenerate-EIC limit)
        if self._entropy_only:
            log_H = self._log_entropy(X_flat)
            log_p_classifier = self._log_classifier_p(X_flat)
            return log_H + log_p_classifier

        # 1) log EI on the objective
        log_ei = self._log_ei(X)     # (b,)

        # 2) Sum_j log P(g_j(x) >= 0) for each physical constraint GP
        log_pf_constraints = self._sum_log_pf_constraints(X_flat)   # (b,)

        # 3) log P(evaluation succeeds | x) from classifier
        log_pf_classifier = self._log_classifier_p(X_flat)          # (b,)

        log_eic = log_ei + log_pf_constraints + log_pf_classifier   # (b,)

        if self._gamma == 0.0:
            # Pure LogEIC; no need to mix in entropy
            return log_eic

        # 4) Entropy in log space
        log_H = self._log_entropy(X_flat)   # (b,)

        # 5) Mix in log space:
        #    log a_AR(x) = logsumexp(log(1-gamma)+log_eic, log(gamma)+log_H)
        log_one_minus_g = math.log(max(1.0 - self._gamma, 1e-300))
        log_g = math.log(max(self._gamma, 1e-300))
        stack = torch.stack([
            log_one_minus_g + log_eic,
            log_g + log_H,
        ], dim=0)   # (2, b)
        return torch.logsumexp(stack, dim=0)

    # ----- helpers -----

    def _sum_log_pf_constraints(self, X_flat: Tensor) -> Tensor:
        """Sum log Phi(mu_g_j / sigma_g_j) over physical constraint GPs.

        Convention (matches eq:ar_constraints): each constraint slack g_j
        is feasible when g_j >= 0, so PF_j(x) = P[g_j(x) >= 0] =
        Phi(mu_g_j / sigma_g_j).  The numerically-stable log_ndtr is
        used to avoid underflow when sigma is small or mu is far below
        zero (deeply infeasible regions).
        """
        device, dtype = X_flat.device, X_flat.dtype
        out = torch.zeros(X_flat.shape[0], dtype=dtype, device=device)
        for cn in self._constraint_names:
            mu, var = self._surrogates.predict(cn, X_flat)
            sd = var.clamp_min(1e-30).sqrt()
            z = (mu / sd).clamp(min=-30.0, max=30.0)
            log_phi = log_ndtr(z)        # log Phi(z), stable
            out = out + log_phi.clamp(min=self._log_zero_floor)
        return out

    def _log_classifier_p(self, X_flat: Tensor) -> Tensor:
        p = self._surrogates.predict_feasibility(X_flat).clamp(min=1e-12, max=1.0)
        return p.log()

    def _log_entropy(self, X_flat: Tensor) -> Tensor:
        """log H(x) with H = sum_j h(PF_j(x)) in nats.

        h(p) = -p log p - (1-p) log (1-p) is the binary entropy.
        We compute H linearly (it's bounded and well-behaved away from
        0/1) and then take log; this is stable because H >= 0 and we
        floor it to a tiny value to handle the deep-interior case
        where every PF is 1 and h(1) = 0.
        """
        H = torch.zeros(X_flat.shape[0], dtype=X_flat.dtype, device=X_flat.device)
        for cn in self._constraint_names:
            mu, var = self._surrogates.predict(cn, X_flat)
            sd = var.clamp_min(1e-30).sqrt()
            z = (mu / sd).clamp(min=-30.0, max=30.0)
            # PF = Phi(z); use exp(log_ndtr(z)) for stability
            log_pf = log_ndtr(z)
            log_one_minus_pf = log_ndtr(-z)
            pf = log_pf.exp().clamp(min=1e-12, max=1.0 - 1e-12)
            one_minus_pf = (1.0 - pf).clamp(min=1e-12, max=1.0 - 1e-12)
            h = -(pf * log_pf + one_minus_pf * log_one_minus_pf)
            H = H + h.clamp(min=0.0)
        return H.clamp(min=1e-30).log()


# ---------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------


class AcquisitionManager:
    """LogEIC + entropy-regularised mixed acquisition orchestrator.

    Builds a fresh _MixedLogAR every iteration (qLogExpectedImprovement
    instances are tied to a fixed best_f and model snapshot), then
    optimises it via BoTorch's `optimize_acqf` over the unit cube.
    """

    def __init__(
        self,
        surrogates,                                       # SurrogateManager
        objective_name: str,
        constraint_names: list[str],
        *,
        objective_sense: str = "min",                     # "min" or "max"
        entropy_only: bool = False,
        gamma_schedule: Optional[GammaSchedule] = None,
        n_restarts: int = 20,
        raw_samples: int = 512,
        sampler_seed: int = 0,
        sampler_n: int = 128,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        if objective_sense not in ("min", "max"):
            raise ValueError(f"objective_sense must be 'min' or 'max'; got {objective_sense!r}")
        self.surrogates = surrogates
        self.objective_name = str(objective_name)
        self.constraint_names = list(constraint_names)
        self.objective_sense = objective_sense
        self.entropy_only = bool(entropy_only)
        self.gamma_schedule = gamma_schedule or GammaSchedule()
        self.n_restarts = int(n_restarts)
        self.raw_samples = int(raw_samples)
        self.sampler_seed = int(sampler_seed)
        self.sampler_n = int(sampler_n)
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

    # --- gamma access ---

    def gamma(self, iteration: int) -> float:
        return self.gamma_schedule(iteration)

    # --- best_f computation ---

    def best_feasible_objective(
        self,
        X_train: Tensor,
        y_obj: Tensor,
        feasible_mask: Tensor,
    ) -> float:
        """Return the best (lowest if min, highest if max) objective
        observed on a feasible row.  If no feasible rows exist, fall
        back to the global best across all rows so qLogEI has a valid
        baseline (the constraint factorisation will still suppress
        infeasible candidates).

        In entropy_only mode the return value is unused (qLogEI is not
        built) but we still compute it cheaply for symmetry with the
        non-degenerate code path; callers may pass `feasible_mask`
        unchanged.
        """
        if self.entropy_only:
            # Provide a sentinel; not consumed by the acquisition.
            return float("nan")
        feasible_mask = feasible_mask.bool().to(y_obj.device)
        finite = torch.isfinite(y_obj)
        if (feasible_mask & finite).any():
            sel = y_obj[feasible_mask & finite]
        elif finite.any():
            logger.info("AcquisitionManager: no feasible-and-finite points; "
                        "using best across all converged samples for best_f")
            sel = y_obj[finite]
        else:
            raise ValueError("best_feasible_objective: no finite y_obj rows")
        if self.objective_sense == "min":
            return float(sel.min().item())
        return float(sel.max().item())

    # --- acquisition build + optimize ---

    def build(self, best_f: float, gamma: float) -> _MixedLogAR:
        """Construct the BoTorch acquisition for one iteration."""
        if self.entropy_only:
            # Degenerate-EIC limit: skip qLogEI entirely.
            return _MixedLogAR(
                log_ei=None,
                surrogates=self.surrogates,
                constraint_names=self.constraint_names,
                gamma=gamma,
                entropy_only=True,
            )

        # qLogExpectedImprovement maximises; for a minimisation problem
        # we negate the model output via best_f sign-flip (the GP itself
        # is unchanged).  But the standard BoTorch idiom is to fit on
        # negated y for minimisation; since SurrogateManager fits the
        # raw objective, we work around this by passing a negative
        # best_f and using maximise=True equivalence: argmax of
        # E[max(f-f*,0)] is unchanged by sign convention as long as the
        # sign of (f - f*) matches the desired direction of improvement.
        # Cleaner approach: use the analytic LogExpectedImprovement
        # which has a `maximize` flag, but the user spec asks for
        # qLogExpectedImprovement throughout.  We therefore build a
        # PosteriorTransform that flips sign for minimisation.
        from botorch.acquisition.objective import ScalarizedPosteriorTransform
        objective_gp = self.surrogates._gps[self.objective_name].gp
        if self.objective_sense == "min":
            transform = ScalarizedPosteriorTransform(
                weights=torch.tensor([-1.0], dtype=self.dtype, device=self.device)
            )
            best_f_signed = -best_f
        else:
            transform = None
            best_f_signed = best_f

        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.sampler_n]),
            seed=self.sampler_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            log_ei = qLogExpectedImprovement(
                model=objective_gp,
                best_f=best_f_signed,
                sampler=sampler,
                posterior_transform=transform,
            )
        return _MixedLogAR(
            log_ei=log_ei,
            surrogates=self.surrogates,
            constraint_names=self.constraint_names,
            gamma=gamma,
        )

    def optimize(
        self,
        bounds: Tensor,
        best_f: float,
        gamma: float,
        *,
        q: int = 1,
    ) -> tuple[Tensor, float]:
        """Optimise the mixed acquisition over `bounds` of shape (2, d).

        Returns (X_next, acq_value), where X_next has shape (q, d) and
        acq_value is the log-acquisition at the chosen point.
        """
        if q != 1:
            raise NotImplementedError(
                "AcquisitionManager.optimize only supports q=1 for now"
            )
        acq = self.build(best_f=best_f, gamma=gamma)
        bounds = bounds.to(dtype=self.dtype, device=self.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            X_next, acq_value = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=q,
                num_restarts=self.n_restarts,
                raw_samples=self.raw_samples,
            )
        return X_next, float(acq_value.item())

    # --- diagnostics: evaluate at a grid ---

    def evaluate_grid(
        self,
        X_grid: Tensor,
        best_f: float,
        gamma: float,
    ) -> Tensor:
        """Evaluate the mixed log-acquisition at every row of X_grid.

        X_grid is (M, d).  Returns a (M,) tensor of log-acquisition
        values.  Useful for plotting/sanity checks; do not use in the
        BO loop hot path (the grid is wasteful — use optimize()).
        """
        acq = self.build(best_f=best_f, gamma=gamma)
        X_grid = X_grid.to(dtype=self.dtype, device=self.device)
        if X_grid.dim() == 2:
            X_grid = X_grid.unsqueeze(1)        # (M, 1, d)
        with torch.no_grad():
            return acq(X_grid)
