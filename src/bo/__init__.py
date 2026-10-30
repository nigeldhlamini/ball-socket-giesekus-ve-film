"""
Bayesian-optimisation framework for chapter 9 of the thesis.

Layout:
  adapter.py       SolverAdapter — BoTorch-compatible wrapper around
                   ar_bo_campaign.run_sample. Maps unit-cube inputs to
                   physical inputs via INPUT_SPACE and back; returns
                   torch tensors.
  surrogates.py    SurrogateManager — one SingleTaskGP per regression
                   output (Matern-5/2 with ARD), plus a variational
                   Bernoulli GP for the evaluation-failure classifier.
  acquisition.py   AcquisitionManager — LogEIC plus entropy-regularised
                   mixed acquisition with scheduled gamma.
  campaign.py      BOCampaign — orchestration loop with checkpointing
                   to BOIterationState/BOSampleRecord parquets.

Usage:
  from src.bo import SolverAdapter, SurrogateManager, AcquisitionManager, BOCampaign
"""

from src.bo.adapter import SolverAdapter, SolverEvaluation  # noqa: F401
from src.bo.surrogates import SurrogateManager  # noqa: F401
from src.bo.acquisition import AcquisitionManager, GammaSchedule  # noqa: F401
from src.bo.campaign import BOCampaign, Problem, solver_adapter_problem  # noqa: F401
