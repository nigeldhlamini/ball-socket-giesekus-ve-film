"""
SurrogateManager: per-output GP regression + variational classifier GP.

For each scalar output (objective + each constraint slack) the manager
maintains an exact `SingleTaskGP` with a Matern-5/2 ARD kernel and a
`Standardize` outcome transform.  For the binary "did the solver
converge" flag, a `gpytorch.ApproximateGP` is fitted with a Bernoulli
likelihood under a variational ELBO objective.

Two design choices, both informed by the chaotic-Picard finding of
Step 2 (1-ULP input perturbations can flip the Brent bracket and
produce cosmic NaN/Inf values in the regression metrics):

  1. Regression GPs are fitted ONLY on rows where `converged` is True,
     and any remaining NaN/Inf in the y-vector is masked out as well.
     This prevents a single non-converged row from corrupting an
     entire surrogate via the `Standardize` step or the Cholesky.

  2. The classifier GP is fitted on ALL rows.  Its job is precisely to
     learn the regions of input space where evaluation succeeds vs
     fails, and the chaotic-Picard regions are valuable training
     signal for it.

Hyperparameters (kernel scale, lengthscales, noise) are refitted by
maximising the type-II marginal log-likelihood every `refit_every`
new observations.  The classifier GP is refit on the same schedule.

API:
    sm = SurrogateManager(output_names=["h_min", ...], d=8)
    sm.fit(X, Y={"h_min": y_h_min, ...}, converged=converged_bool)
    mean, var = sm.predict("h_min", X_test)
    p_feas    = sm.predict_feasibility(X_test)
    ls        = sm.ard_lengthscales("h_min")
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from typing import Any, Optional

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Variational Bernoulli classifier GP (gpytorch primitive — BoTorch does not
# expose a ready-made wrapper for binary classification at the layer of
# abstraction we want).
# =============================================================================


class _BernoulliClassifierGP(ApproximateGP):
    """ApproximateGP with Matern-5/2 ARD covariance, learnable inducing
    points, and Bernoulli likelihood (downstream)."""

    def __init__(self, inducing_points: torch.Tensor, d: int,
                 kernel: str = "matern52") -> None:
        var_dist = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        var_strat = VariationalStrategy(
            self, inducing_points, var_dist,
            learn_inducing_locations=True,
        )
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(_make_kernel(kernel, d))

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def _make_kernel(kernel_name: str, d: int) -> gpytorch.kernels.Kernel:
    if kernel_name == "matern52":
        return MaternKernel(nu=2.5, ard_num_dims=d)
    if kernel_name == "matern32":
        return MaternKernel(nu=1.5, ard_num_dims=d)
    if kernel_name == "rbf":
        return RBFKernel(ard_num_dims=d)
    raise ValueError(f"unknown kernel '{kernel_name}'; expected one of "
                     f"'matern52', 'matern32', 'rbf'")


# =============================================================================
# Regression GP factory + fitter
# =============================================================================


def _fit_regression_gp(
    X: torch.Tensor, y: torch.Tensor, d: int, *,
    kernel: str = "matern52",
) -> SingleTaskGP:
    """Fit a SingleTaskGP with Matern-5/2 ARD + Standardize outcome.

    `X` is (N, d) in the unit cube; `y` is (N,) or (N, 1).  Returns a
    fitted GP in eval mode.
    """
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    covar = ScaleKernel(_make_kernel(kernel, d))
    gp = SingleTaskGP(
        train_X=X,
        train_Y=y,
        outcome_transform=Standardize(m=1),
        covar_module=covar,
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fit_gpytorch_mll(mll)
    gp.eval()
    return gp


def _fit_classifier_gp(
    X: torch.Tensor, y_bool: torch.Tensor, d: int, *,
    kernel: str = "matern52",
    n_inducing: int = 64,
    n_iter: int = 200,
    lr: float = 0.05,
) -> tuple[_BernoulliClassifierGP, BernoulliLikelihood]:
    """Fit a variational Bernoulli GP via ELBO maximisation.

    The inducing points are initialised by random subsampling of `X` (or
    by using all points if N <= n_inducing).
    """
    n = X.shape[0]
    n_ind = min(n_inducing, n)
    if n_ind < n:
        idx = torch.randperm(n, device=X.device)[:n_ind]
        inducing = X[idx].clone()
    else:
        inducing = X.clone()
    model = _BernoulliClassifierGP(inducing, d, kernel=kernel).to(X)
    likelihood = BernoulliLikelihood().to(X)

    model.train()
    likelihood.train()
    optimiser = torch.optim.Adam(
        [{"params": model.parameters()},
         {"params": likelihood.parameters()}],
        lr=lr,
    )
    mll = VariationalELBO(likelihood, model, num_data=y_bool.numel())
    y_float = y_bool.float()
    last_loss = float("inf")
    for it in range(n_iter):
        optimiser.zero_grad()
        out = model(X)
        loss = -mll(out, y_float)
        loss.backward()
        optimiser.step()
        last_loss = float(loss.detach().item())

    model.eval()
    likelihood.eval()
    logger.debug(f"_fit_classifier_gp: final ELBO loss = {last_loss:.4f} "
                 f"after {n_iter} iter, n_inducing={n_ind}")
    return model, likelihood


# =============================================================================
# Manager
# =============================================================================


@dataclasses.dataclass
class _GPState:
    gp: SingleTaskGP
    n_train: int
    train_mask: torch.Tensor


class SurrogateManager:
    """Per-output GP regression plus a variational classifier.

    Outputs are organised into three GROUPS reflecting how downstream
    consumers use them:

      objective_names   — GPs that drive qLogExpectedImprovement.
      constraint_names  — GPs that enter the feasibility product
                          Π PF_j(x) of the LogEIC acquisition.
      auxiliary_names   — GPs fitted alongside the others and queryable
                          via predict(), but invisible to the
                          AcquisitionManager. Used for downstream
                          projections (e.g. predicting T_max at Sobol
                          grid points for Hausdorff against the 8.1
                          (h_min, T_max) projection).

    All three groups share the same kernel, hyperparameter procedure,
    and convergence-row masking. The grouping is metadata for consumers,
    not a difference in fitting behaviour.

    Legacy callers can pass `output_names=[...]` instead of the three
    explicit lists; these are treated as auxiliary by default (i.e.
    fitted but not exposed to acquisition).
    """

    def __init__(
        self,
        d: int,
        *,
        # Three-group API (preferred for chapter-9 use)
        objective_names: Optional[list[str]] = None,
        constraint_names: Optional[list[str]] = None,
        auxiliary_names: Optional[list[str]] = None,
        # Legacy single-list API
        output_names: Optional[list[str]] = None,
        # Common
        kernel: str = "matern52",
        refit_every: int = 5,
        classifier_kernel: Optional[str] = None,
        classifier_n_inducing: int = 64,
        classifier_n_iter: int = 200,
        classifier_lr: float = 0.05,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        # Resolve grouping
        groups_specified = any(
            x is not None for x in (objective_names, constraint_names, auxiliary_names)
        )
        if output_names is not None and groups_specified:
            raise ValueError(
                "SurrogateManager: pass either `output_names` (legacy) "
                "or the three-group lists, not both"
            )
        if output_names is not None:
            self.objective_names: list[str] = []
            self.constraint_names: list[str] = []
            self.auxiliary_names: list[str] = list(output_names)
        else:
            self.objective_names = list(objective_names) if objective_names else []
            self.constraint_names = list(constraint_names) if constraint_names else []
            self.auxiliary_names = list(auxiliary_names) if auxiliary_names else []
        # Flat list (preserves insertion order: objective, constraint, auxiliary)
        self.output_names = (
            self.objective_names + self.constraint_names + self.auxiliary_names
        )
        if len(set(self.output_names)) != len(self.output_names):
            raise ValueError(
                f"SurrogateManager: output names must be unique across groups; "
                f"got duplicates in {self.output_names}"
            )
        self.d = int(d)
        self.kernel = str(kernel)
        self.classifier_kernel = classifier_kernel or kernel
        self.refit_every = int(refit_every)
        self.classifier_n_inducing = int(classifier_n_inducing)
        self.classifier_n_iter = int(classifier_n_iter)
        self.classifier_lr = float(classifier_lr)
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        self._gps: dict[str, _GPState] = {}
        self._classifier: Optional[_BernoulliClassifierGP] = None
        self._classifier_likelihood: Optional[BernoulliLikelihood] = None
        self._n_obs_at_last_fit = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: torch.Tensor,
        Y: dict[str, torch.Tensor],
        *,
        converged: Optional[torch.Tensor] = None,
    ) -> None:
        """Fit all regression GPs and the classifier from scratch.

        Parameters
        ----------
        X : torch.Tensor, shape (N, d)
            Unit-cube training inputs.
        Y : dict[str, torch.Tensor of shape (N,)]
            One y-vector per output name in `self.output_names`.
            Non-converged rows may carry NaN/Inf — they are masked out
            of the regression-GP training set automatically.
        converged : torch.Tensor of shape (N,), optional
            Bool indicator of solver convergence.  When provided, the
            classifier GP is fitted on this; rows with `converged=False`
            are also excluded from the regression-GP training sets.
            When None, all rows are used for regression and no
            classifier is fitted.
        """
        X = self._cast(X)
        N = X.shape[0]
        if X.shape[1] != self.d:
            raise ValueError(f"SurrogateManager.fit: expected X with d={self.d}, "
                             f"got shape {tuple(X.shape)}")

        if converged is not None:
            converged = converged.to(device=self.device).bool()
            if converged.shape != (N,):
                raise ValueError(f"converged shape {tuple(converged.shape)} != ({N},)")
        else:
            converged = torch.ones(N, dtype=torch.bool, device=self.device)

        for name in self.output_names:
            if name not in Y:
                raise KeyError(f"output '{name}' missing from Y")
            y = self._cast(Y[name])
            if y.shape != (N,) and y.shape != (N, 1):
                raise ValueError(f"Y['{name}'] shape {tuple(y.shape)} != ({N},)")
            y = y.reshape(-1)
            mask = converged & torch.isfinite(y)
            n_train = int(mask.sum().item())
            if n_train < 3:
                logger.warning(f"SurrogateManager: output '{name}' has only "
                               f"{n_train} usable training rows; skipping fit "
                               f"(GP will not be available until more data arrive)")
                continue
            gp = _fit_regression_gp(
                X[mask], y[mask], self.d, kernel=self.kernel,
            )
            self._gps[name] = _GPState(gp=gp, n_train=n_train, train_mask=mask)

        # Classifier
        if converged is not None and len(set(converged.tolist())) >= 2:
            cls_model, cls_lik = _fit_classifier_gp(
                X, converged, self.d,
                kernel=self.classifier_kernel,
                n_inducing=self.classifier_n_inducing,
                n_iter=self.classifier_n_iter,
                lr=self.classifier_lr,
            )
            self._classifier = cls_model
            self._classifier_likelihood = cls_lik
        else:
            # Constant convergence (all True or all False) — classifier
            # is degenerate; record the constant.
            self._classifier = None
            self._classifier_likelihood = None
            logger.info("SurrogateManager: classifier not fitted (no class diversity)")

        self._n_obs_at_last_fit = N

    def should_refit(self, n_obs_now: int) -> bool:
        """Whether the next observation triggers a refit.

        Returns True if the number of new observations since the last
        fit is >= `refit_every`.
        """
        return (n_obs_now - self._n_obs_at_last_fit) >= self.refit_every

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, output: str, X_test: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior (mean, variance) at X_test for the named output.

        Returns tensors of shape (M,) where M = X_test.shape[0].
        Raises KeyError if the output GP has not been fitted yet.
        """
        if output not in self._gps:
            raise KeyError(
                f"GP for output '{output}' is not available "
                f"(known fitted: {list(self._gps.keys())})"
            )
        gp = self._gps[output].gp
        X_test = self._cast(X_test)
        if X_test.dim() == 1:
            X_test = X_test.unsqueeze(0)
        # NB: do NOT wrap in torch.no_grad() here.  When called from
        # _MixedLogAR.forward (inside optimize_acqf's L-BFGS-B path), the
        # acquisition function needs ∂(log H, log P_classifier)/∂X to
        # exist; stripping gradients at this level breaks L-BFGS in
        # entropy_only mode.  Callers who genuinely don't need grads
        # (eval-grid diagnostics, post-hoc Hausdorff) wrap their own
        # call site in torch.no_grad() — see AcquisitionManager.evaluate_grid.
        posterior = gp.posterior(X_test)
        mean = posterior.mean.squeeze(-1)
        var = posterior.variance.squeeze(-1)
        return mean, var

    def predict_feasibility(self, X_test: torch.Tensor) -> torch.Tensor:
        """Probability that evaluation succeeds, per the classifier GP.

        Returns a (M,) tensor of probabilities in [0, 1].  If no
        classifier was fitted, returns all-ones.

        See the comment in predict() about gradient flow — same rule
        applies here, and breaking it was the bug that turned the
        Phase 9-B baseline into a 500-sample random Sobol run.
        """
        X_test = self._cast(X_test)
        if X_test.dim() == 1:
            X_test = X_test.unsqueeze(0)
        if self._classifier is None or self._classifier_likelihood is None:
            return torch.ones(X_test.shape[0], dtype=self.dtype, device=self.device)
        with gpytorch.settings.fast_pred_var():
            f_dist = self._classifier(X_test)
            obs_dist = self._classifier_likelihood(f_dist)
            probs = obs_dist.mean
        return probs.to(dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def ard_lengthscales(self, output: str) -> torch.Tensor:
        """Return the (d,) ARD lengthscales for the named regression GP."""
        if output not in self._gps:
            raise KeyError(f"GP for output '{output}' not available")
        gp = self._gps[output].gp
        # ScaleKernel(MaternKernel(...)).base_kernel.lengthscale -> (1, d)
        ls = gp.covar_module.base_kernel.lengthscale.detach().squeeze(0)
        return ls.to(dtype=self.dtype, device=self.device)

    def stats(self) -> dict[str, Any]:
        return {
            "outputs_fitted": list(self._gps.keys()),
            "n_train_per_output": {
                name: state.n_train for name, state in self._gps.items()
            },
            "classifier_fitted": self._classifier is not None,
            "n_obs_at_last_fit": self._n_obs_at_last_fit,
            "objective_names": list(self.objective_names),
            "constraint_names": list(self.constraint_names),
            "auxiliary_names": list(self.auxiliary_names),
        }

    def group_of(self, name: str) -> str:
        """Return 'objective', 'constraint', 'auxiliary', or raise KeyError."""
        if name in self.objective_names:
            return "objective"
        if name in self.constraint_names:
            return "constraint"
        if name in self.auxiliary_names:
            return "auxiliary"
        raise KeyError(f"output '{name}' not registered with SurrogateManager")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cast(self, t) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=self.dtype, device=self.device)
        return t.to(dtype=self.dtype, device=self.device)
