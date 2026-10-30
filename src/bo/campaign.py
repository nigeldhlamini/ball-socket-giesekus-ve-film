"""
BOCampaign: Bayesian-optimisation orchestration loop.

Wires SolverAdapter (or any Problem-compatible evaluator), SurrogateManager
and AcquisitionManager into the chapter-9 loop:

  1. Draw N_init Sobol points in [0, 1]^d, evaluate the problem.
  2. For iter = 1 .. N_max - N_init:
       a. (Re)fit the SurrogateManager every refit_every observations.
       b. Optimise the mixed log-acquisition over [0, 1]^d.
       c. Evaluate the problem at the chosen point.
       d. Append to the running training set.
       e. Every hausdorff_eval_every iterations, if a reference boundary
          is supplied, compute the normalised Hausdorff distance from
          the BO-implied feasible set to the reference and append to
          the trajectory.
       f. Persist (in memory always; to Parquet if persist_dir given).
       g. Stop if budget exhausted, the Hausdorff threshold is met, or
          the no-progress criterion fires.

Problem interface
-----------------
A Problem provides
    .evaluate(X_unit: Tensor) -> dict[str, Tensor]
returning per-row tensors keyed by output name plus 'converged' (bool)
and 'feasible' (bool); the dict's keys must include `objective_name`
and every `constraint_name`. A `Problem` dataclass is provided; the
slipper-bearing SolverAdapter is a Problem in spirit (the validation
gate wires it explicitly).

This module is deliberately problem-agnostic.  The slipper-bearing-
specific persistence schema (BOSampleRecord/BOIterationState in
src/bo_schema.py) is invoked only when the campaign is told to
persist; for analytical sanity tests (Hartmann-6 etc.), persistence is
skipped and the in-memory trajectory dict is the artefact.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import time
from typing import Any, Callable, Optional

import numpy as np
import torch
from scipy.stats import qmc

from src.bo.acquisition import AcquisitionManager, GammaSchedule
from src.bo.surrogates import SurrogateManager

logger = logging.getLogger(__name__)


# =============================================================================
# Problem
# =============================================================================


@dataclasses.dataclass
class Problem:
    """Generic BO problem definition.

    The `evaluate_fn` takes a (q, d) torch tensor in the unit cube and
    returns a dict with at least:
      * `objective_name`               -> (q,) float tensor
      * each name in `constraint_names`-> (q,) float tensor (g >= 0 = feasible)
      * each name in `auxiliary_names` -> (q,) float tensor
      * 'converged'                    -> (q,) bool tensor
      * 'feasible'                     -> (q,) bool tensor (optional;
                                         defaults to all-True if absent)

    Auxiliary outputs are fitted as regression GPs alongside the
    objective and constraints but are excluded from the LogEIC
    acquisition's feasibility product.  They exist so downstream
    consumers (Hausdorff projections, multi-output diagnostics) can
    query GP predictions for performance metrics that don't drive
    the BO loop directly.
    """
    evaluate_fn: Callable[[torch.Tensor], dict[str, torch.Tensor]]
    d: int
    objective_name: str
    objective_sense: str = "min"
    constraint_names: list[str] = dataclasses.field(default_factory=list)
    auxiliary_names: list[str] = dataclasses.field(default_factory=list)
    name: str = "problem"

    def evaluate(self, X_unit: torch.Tensor) -> dict[str, torch.Tensor]:
        if X_unit.dim() == 1:
            X_unit = X_unit.unsqueeze(0)
        out = self.evaluate_fn(X_unit)
        # Required keys
        required = (self.objective_name,
                    *self.constraint_names,
                    *self.auxiliary_names,
                    "converged")
        for k in required:
            if k not in out:
                raise KeyError(f"Problem.evaluate: missing key '{k}' in evaluator output")
        if "feasible" not in out:
            out["feasible"] = torch.ones_like(out["converged"], dtype=torch.bool)
        return out


# =============================================================================
# Helper to wrap a SolverAdapter as a Problem
# =============================================================================


def solver_adapter_problem(
    adapter,
    *,
    objective_name: str = "h_min",
    objective_sense: str = "min",
    constraint_names: tuple[str, ...] = ("g1_gap", "g2_thermal", "g3_validity"),
    auxiliary_output_names: tuple[str, ...] = ("T_max", "P_loss"),
    name: str = "slipper_bearing",
) -> Problem:
    """Adapter-to-Problem wiring for the slipper-bearing campaign.

    `auxiliary_output_names` are fitted as regression GPs by BOCampaign
    but excluded from the AcquisitionManager's feasibility product. The
    chapter-9 default is ("T_max", "P_loss") because the (h_min, T_max)
    and (P_loss, F_applied) Hausdorff projections need GP predictions
    of those metrics at Sobol grid points.

    Note that F_applied is intentionally NOT in the auxiliary list:
    F_applied = k_F * F_hyd(p_s) is a deterministic function of the
    inputs and is computed analytically by downstream consumers (the
    Hausdorff replay) rather than being GP-modelled. Including it as
    an auxiliary GP would just add noise to a quantity we already
    know exactly from the unit-cube coordinates.
    """
    from src.bo.adapter import SolverAdapter   # local import to avoid cycle
    if not isinstance(adapter, SolverAdapter):
        raise TypeError(f"expected SolverAdapter, got {type(adapter).__name__}")

    def _eval(X_unit: torch.Tensor) -> dict[str, torch.Tensor]:
        ev = adapter.evaluate(X_unit)
        d: dict[str, torch.Tensor] = {
            "h_min": ev.h_min, "T_max": ev.T_max, "Q_leak": ev.Q_leak,
            "P_loss": ev.P_loss, "F_applied": ev.F_applied, "De_sq": ev.De_sq,
            "DeltaF_elastic_over_F": ev.DeltaF_elastic_over_F,
            "g1_gap": ev.g1_gap, "g2_thermal": ev.g2_thermal,
            "g3_validity": ev.g3_validity,
            "converged": ev.converged, "feasible": ev.feasible,
        }
        return d

    return Problem(
        evaluate_fn=_eval,
        d=adapter.d,
        objective_name=objective_name,
        objective_sense=objective_sense,
        constraint_names=list(constraint_names),
        auxiliary_names=list(auxiliary_output_names),
        name=name,
    )


# =============================================================================
# BOCampaign
# =============================================================================


class BOCampaign:
    """Bayesian-optimisation orchestration loop."""

    def __init__(
        self,
        problem: Problem,
        *,
        n_init: int = 18,
        n_max: int = 500,
        # SurrogateManager config
        kernel: str = "matern52",
        refit_every: int = 5,
        use_classifier: bool = True,
        # AcquisitionManager config
        entropy_only: bool = False,
        gamma_schedule: Optional[GammaSchedule] = None,
        acq_n_restarts: int = 20,
        acq_raw_samples: int = 512,
        # Hausdorff convergence (slipper-bearing only)
        reference_boundary: Optional[np.ndarray] = None,
        hausdorff_eval_every: int = 10,
        hausdorff_threshold: float = 0.02,
        # Persistence
        persist_dir: Optional[pathlib.Path | str] = None,
        campaign_name: str = "bo_run",
        # RNG
        seed: int = 42,
        sobol_seed: int = 0,
        # Initial design override (Stage C of the BO refinement campaign).
        # When provided, this (n_init, d) tensor REPLACES the default Sobol
        # initial draw. The launch script is responsible for orchestrating
        # any stratified pilot pre-screen that produced it; BOCampaign
        # just consumes the override as-is. The shape must be (n_init, d).
        initial_design_override: Optional[torch.Tensor] = None,
        # Repulsion ranker config (Stage B). When provided, the BO main
        # loop uses this for candidate selection INSTEAD of the default
        # AcquisitionManager.optimize path.
        repulsion_config=None,
        # Stage D: when True, wrap the surrogates in a coordinate-
        # transform layer (logit on k_F, log10 on |h_dot|). The wrapper
        # presents the same external API as the plain SurrogateManager
        # but the GPs fit in transformed space.
        coord_transform: bool = False,
        # Device
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        self.problem = problem
        self.d = problem.d
        self.n_init = int(n_init)
        self.n_max = int(n_max)
        if self.n_init >= self.n_max:
            raise ValueError(f"n_init ({n_init}) must be < n_max ({n_max})")
        self.use_classifier = bool(use_classifier)
        self.gamma_schedule = gamma_schedule or GammaSchedule()
        self.reference_boundary = (
            np.asarray(reference_boundary) if reference_boundary is not None else None
        )
        self.hausdorff_eval_every = int(hausdorff_eval_every)
        self.hausdorff_threshold = float(hausdorff_threshold)
        self.persist_dir = pathlib.Path(persist_dir) if persist_dir is not None else None
        self.campaign_name = str(campaign_name)
        self.seed = int(seed)
        self.sobol_seed = int(sobol_seed)
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        # SurrogateManager: regression GPs for objective + each constraint
        # + each auxiliary. The acquisition only sees objective + constraints;
        # auxiliary GPs are fitted-but-invisible-to-acquisition.
        #
        # Stage D wiring (coord_transform=True): the GPs fit in TRANSFORMED
        # space (xi_1 = logit-k_F, xi_2 = log-scaled |h_dot|). To avoid
        # silently misinterpreting candidate coordinates inside botorch's
        # qLogExpectedImprovement (which accesses the SingleTaskGP
        # directly), the BO loop stores _X_obs in TRANSFORMED space when
        # coord_transform is on, untransforming only at the solver
        # boundary. self._unit_bounds() returns transformed bounds in
        # this mode.
        self.surrogates = SurrogateManager(
            d=self.d,
            objective_names=[problem.objective_name],
            constraint_names=list(problem.constraint_names),
            auxiliary_names=list(problem.auxiliary_names),
            kernel=kernel, refit_every=refit_every,
            dtype=dtype, device=device,
        )
        self._coord_transform = bool(coord_transform)
        self.entropy_only = bool(entropy_only)
        self.acquisition = AcquisitionManager(
            surrogates=self.surrogates,
            objective_name=problem.objective_name,
            constraint_names=problem.constraint_names,
            objective_sense=problem.objective_sense,
            entropy_only=self.entropy_only,
            gamma_schedule=self.gamma_schedule,
            n_restarts=acq_n_restarts,
            raw_samples=acq_raw_samples,
            dtype=dtype, device=device,
        )

        # Trajectory accumulators — Y_obs holds ALL fitted outputs
        # (objective + constraints + auxiliary). The auxiliary outputs
        # are exposed to downstream consumers (e.g. Hausdorff replay)
        # via the same Y_obs dict; only the AcquisitionManager treats
        # them differently.
        sm_outputs = self.surrogates.output_names
        self._X_obs: list[torch.Tensor] = []
        self._Y_obs: dict[str, list[torch.Tensor]] = {
            n: [] for n in sm_outputs
        }
        self._converged_obs: list[bool] = []
        self._feasible_obs: list[bool] = []
        self._iter_records: list[dict[str, Any]] = []
        self._hausdorff_history: list[dict[str, float]] = []
        self._best_feasible_history: list[float] = []

        self._n_obs = 0
        self._iteration = 0

        # Stage B: optional performance-space coverage repulsion ranker.
        # When set, the BO main loop uses this ranker INSTEAD OF
        # self.acquisition.optimize. The ranker generates a Sobol
        # candidate batch, applies entropy+classifier scoring, and
        # re-ranks by a KDE-based repulsion penalty in the active
        # projection plane (rotated round-robin across iterations).
        self._repulsion_ranker = None  # populated lazily after _fit_surrogates

        # Stage B: stash the repulsion config.
        self._repulsion_config = repulsion_config

        # Stage C: stash an optional initial-design override.
        if initial_design_override is not None:
            if initial_design_override.shape != (self.n_init, self.d):
                raise ValueError(
                    f"initial_design_override shape "
                    f"{tuple(initial_design_override.shape)} does not match "
                    f"expected ({self.n_init}, {self.d})"
                )
            self._initial_design_override: Optional[torch.Tensor] = (
                initial_design_override.to(dtype=self.dtype, device=self.device)
            )
        else:
            self._initial_design_override = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.persist_dir is not None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initial design: Stage C override or plain Sobol.
        # The initial design is always drawn / provided in NATIVE space
        # ([0, 1]^d unit cube). Stage D's transform is applied here once
        # so the rest of the BO loop runs in TRANSFORMED space.
        if self._initial_design_override is not None:
            X_init_native = self._initial_design_override
            logger.info(f"BOCampaign[{self.problem.name}]: using "
                        f"initial-design override of shape "
                        f"{tuple(X_init_native.shape)} (Stage C stratified design)")
        else:
            X_init_native = self._draw_sobol(self.n_init, seed=self.sobol_seed)
            logger.info(f"BOCampaign[{self.problem.name}]: drawing initial Sobol "
                        f"design of {self.n_init} points")
        if self._coord_transform:
            from src.bo.coord_transform import (
                transform_inputs, clip_rate,
            )
            X_init = transform_inputs(X_init_native)
            cr = clip_rate(X_init_native)
            logger.info(f"BOCampaign: Stage D coord_transform ON; "
                        f"initial-design k_F clip rate = {cr:.3f}")
        else:
            X_init = X_init_native
        self._evaluate_and_append(X_init, is_initial=True)
        self._fit_surrogates()

        # Main BO loop
        n_remaining = self.n_max - self.n_init
        terminated_reason = "budget_exhausted"
        for k in range(n_remaining):
            self._iteration = k
            t0 = time.time()

            if self.surrogates.should_refit(self._n_obs):
                self._fit_surrogates()

            # Pick the next point
            best_f = self._best_feasible_objective()
            gamma = self.gamma_schedule(self._n_obs)
            if self._repulsion_config is not None:
                # Stage B: post-hoc repulsion ranker. Lazy-init the
                # ranker once the surrogates exist.
                if self._repulsion_ranker is None:
                    from src.bo.repulsion import RepulsionRanker
                    self._repulsion_ranker = RepulsionRanker(
                        self._repulsion_config, self.surrogates,
                    )
                X_train_unit = torch.stack(self._X_obs).cpu().numpy()
                Y_train = {n: np.asarray(ys, dtype=np.float64)
                           for n, ys in self._Y_obs.items()}
                feas_arr = np.asarray(self._feasible_obs, dtype=bool)
                try:
                    rank_res = self._repulsion_ranker.rank_candidates(
                        iteration=k,
                        X_train_unit=X_train_unit, Y_train=Y_train,
                        feasible=feas_arr,
                        sobol_seed=self.sobol_seed,
                    )
                    X_next = rank_res["X_next"]
                    acq_val = rank_res["penalised_log_acq"]
                except Exception as exc:
                    logger.warning(f"repulsion-ranker failed at iter {k}: "
                                   f"{exc!r}; falling back to random Sobol point")
                    X_next = self._draw_sobol(1, seed=self.sobol_seed + 100 + k)
                    acq_val = float("nan")
            else:
                try:
                    X_next, acq_val = self.acquisition.optimize(
                        bounds=self._unit_bounds(),
                        best_f=best_f,
                        gamma=gamma,
                        q=1,
                    )
                except Exception as exc:
                    logger.warning(f"acquisition optimisation failed at iter {k}: "
                                   f"{exc!r}; falling back to random Sobol point")
                    X_next = self._draw_sobol(1, seed=self.sobol_seed + 100 + k)
                    acq_val = float("nan")

            # Evaluate
            self._evaluate_and_append(X_next, is_initial=False,
                                      acq_value=acq_val, gamma=gamma)
            self._best_feasible_history.append(best_f)

            wall = time.time() - t0
            logger.info(f"  iter {k:>4d}  N={self._n_obs}  gamma={gamma:.3f}  "
                        f"acq={acq_val:.4f}  best_f={best_f:.4g}  "
                        f"wall={wall:.1f}s")

            # Hausdorff every K iterations
            if (self.reference_boundary is not None
                    and (k + 1) % self.hausdorff_eval_every == 0):
                d_H = self._compute_hausdorff()
                self._hausdorff_history.append({"iter": k, "n": self._n_obs,
                                                 "hausdorff": d_H})
                logger.info(f"    Hausdorff(BO_implied, reference) = {d_H:.4f}")
                if d_H < self.hausdorff_threshold:
                    terminated_reason = "hausdorff_threshold_met"
                    logger.info(f"BOCampaign: Hausdorff threshold "
                                f"({self.hausdorff_threshold}) met at iter {k}")
                    break

        # On output, expose X_obs in NATIVE space regardless of whether
        # Stage D's transform was active. Downstream analysis
        # (boundary extraction, Stage A overlay, alpha-shape, etc.)
        # operates in native unit-cube coordinates.
        X_obs_internal = torch.stack(self._X_obs)
        if self._coord_transform:
            from src.bo.coord_transform import untransform_inputs
            X_obs_native = untransform_inputs(X_obs_internal)
        else:
            X_obs_native = X_obs_internal
        return {
            "X_obs": X_obs_native.cpu().numpy(),
            "X_obs_internal": X_obs_internal.cpu().numpy(),  # for diagnostics
            "Y_obs": {n: torch.tensor(ys).cpu().numpy()
                      for n, ys in self._Y_obs.items() if ys},
            "converged_obs": np.asarray(self._converged_obs, dtype=bool),
            "feasible_obs": np.asarray(self._feasible_obs, dtype=bool),
            "best_feasible_history": np.asarray(self._best_feasible_history),
            "hausdorff_history": list(self._hausdorff_history),
            "iter_records": list(self._iter_records),
            "terminated_reason": terminated_reason,
            "n_obs_final": self._n_obs,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _unit_bounds(self) -> torch.Tensor:
        if self._coord_transform:
            from src.bo.coord_transform import transformed_unit_bounds
            return transformed_unit_bounds(dtype=self.dtype, device=self.device)
        return torch.stack([
            torch.zeros(self.d, dtype=self.dtype, device=self.device),
            torch.ones(self.d, dtype=self.dtype, device=self.device),
        ])

    def _draw_sobol(self, n: int, seed: int) -> torch.Tensor:
        sampler = qmc.Sobol(d=self.d, scramble=True, seed=seed)
        u = sampler.random(n=n)
        return torch.tensor(u, dtype=self.dtype, device=self.device)

    def _evaluate_and_append(
        self, X: torch.Tensor, *, is_initial: bool,
        acq_value: float = float("nan"), gamma: float = float("nan"),
    ) -> None:
        # Stage D: X is in TRANSFORMED space if coord_transform is on.
        # The solver expects NATIVE coords, so untransform before
        # evaluating; store the TRANSFORMED X in _X_obs so the
        # SurrogateManager fits on transformed coords.
        if self._coord_transform:
            from src.bo.coord_transform import untransform_inputs
            X_native = untransform_inputs(X)
        else:
            X_native = X
        out = self.problem.evaluate(X_native)
        q = X.shape[0]
        for i in range(q):
            self._X_obs.append(X[i].detach())
            for name in self._Y_obs:
                yi = float(out[name][i].item())
                self._Y_obs[name].append(yi)
            conv_i = bool(out["converged"][i].item())
            feas_i = bool(out["feasible"][i].item())
            self._converged_obs.append(conv_i)
            self._feasible_obs.append(feas_i)
            self._iter_records.append({
                "iter": self._iteration if not is_initial else -1,
                "n_obs_after": self._n_obs + i + 1,
                "is_initial_design": is_initial,
                "acq_value_at_proposal": acq_value if not is_initial else float("nan"),
                "gamma_at_proposal": gamma if not is_initial else float("nan"),
                "X_unit": X[i].detach().cpu().numpy().tolist(),
                "converged": conv_i,
                "feasible": feas_i,
            })
        self._n_obs += q

    def _fit_surrogates(self) -> None:
        X = torch.stack(self._X_obs)
        Y = {n: torch.tensor(ys, dtype=self.dtype, device=self.device)
             for n, ys in self._Y_obs.items()}
        converged = (
            torch.tensor(self._converged_obs, dtype=torch.bool, device=self.device)
            if self.use_classifier else None
        )
        self.surrogates.fit(X=X, Y=Y, converged=converged)

    def _best_feasible_objective(self) -> float:
        X = torch.stack(self._X_obs)
        y_obj = torch.tensor(
            self._Y_obs[self.problem.objective_name],
            dtype=self.dtype, device=self.device,
        )
        feas_mask = torch.tensor(self._feasible_obs, dtype=torch.bool,
                                  device=self.device)
        return self.acquisition.best_feasible_objective(
            X_train=X, y_obj=y_obj, feasible_mask=feas_mask,
        )

    def _compute_hausdorff(self) -> float:
        """Hausdorff between BO-implied feasible boundary and `self.reference_boundary`.

        Slipper-bearing-only.  For Phase 9-A the implementation is a
        placeholder that returns NaN; the real implementation will land
        in the validation gate (Step 6) where we have a concrete
        reference boundary from build_ar_maps' manifest.
        """
        if self.reference_boundary is None:
            return float("nan")
        # Placeholder — Step 6 will compute properly from GP posterior + alpha-shape
        return float("nan")
