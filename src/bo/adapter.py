"""
SolverAdapter: BoTorch-compatible wrapper around `ar_bo_campaign.run_sample`.

The slipper-bearing solver is a per-sample function (one Sobol point in,
one performance dict out) that does not natively speak the BoTorch
batched-tensor protocol.  This adapter:

  * Accepts unit-cube tensors X in [0, 1]^d (the natural BO input space)
    and maps each row to physical inputs via INPUT_SPACE.from_unit, so
    the BO loop never sees the dimensional inputs directly.
  * Calls `run_sample` row-by-row (the solver is single-threaded; the
    BO loop typically queries q=1 candidate at a time, so a Python
    for-loop is the right granularity).
  * Returns a SolverEvaluation dataclass holding torch tensors for
    every performance metric, every constraint slack g_j, and the
    feasibility/convergence flags needed by the classifier GP.

A small in-memory cache keyed on the rounded unit-cube coordinates
guards against redundant solver calls when the acquisition optimiser
proposes points within machine-precision of an already-evaluated
location (which can happen during multi-restart L-BFGS).

The adapter does NOT do batching across processes for the BO loop.
That decision lives in BOCampaign — the present class is the per-call
contract and the sanity-check seam.

A note on chaotic Picard regions
--------------------------------
The slipper-bearing Picard iteration has chaotic neighbourhoods in
which a 1-ULP perturbation of a log-scaled input (typically `lam` or
`h_dot`) is amplified into a qualitatively different solver output
(e.g. flipping the Brent bracket from `ok` to `no_root_contact`,
producing cosmic T_max values from a degenerate inner solve). The
production 8.1 sweep's recorded outputs are therefore deterministic
given the EXACT bit-for-bit Sobol unit-cube draw, not given the
recorded physical inputs round-tripped through `INPUT_SPACE.to_unit`.

For the BO loop this is harmless: the production path is
unit-cube -> from_unit -> physical -> solver, with no round-trip in
the loop. SolverAdapter.physical_to_unit() exists for diagnostics
(e.g. comparing a candidate point against a recorded sample) and
should NOT be used to reconstruct unit-cube coordinates that will
then be fed back through the solver and compared bit-for-bit against
production records — use the original `qmc.Sobol(...).random(...)`
draw for that instead. The `_bo_step2_sanity.py` test illustrates the
correct verification pattern.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys
from collections import OrderedDict
from typing import Any, Optional

# Repo root on path so we can import ar_bo_campaign regardless of how
# this module is imported (scripts/, notebooks, or other src/ modules).
_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # type: ignore

from ar_bo_campaign import (  # noqa: E402
    INPUT_SPACE,
    NR_DEFAULT,
    NTHETA_DEFAULT,
    is_feasible,
    run_sample,
)
from src import SolverConfig  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================================
# Output container
# ============================================================================


@dataclasses.dataclass
class SolverEvaluation:
    """One batch of solver outputs, all tensors of shape (q,).

    Fields are organised into three groups:
      * Performance metrics — used to construct BO objectives.
      * Constraint slacks  — g_j >= 0 means the j-th constraint is
                             satisfied. The Saltelli/AR sign convention
                             is preserved (g1_gap = h_min - h_safe etc.)
                             so the BO acquisition can use them directly.
      * Status flags        — bool tensors used by the classifier GP and
                             by feasibility-conditioned objectives.

    All numeric tensors carry the dtype/device of the input X. Status
    tensors are torch.bool on the same device.
    """

    # Performance metrics
    h_min: torch.Tensor                  # film thickness, m
    T_max: torch.Tensor                  # peak temperature, K
    Q_leak: torch.Tensor                 # leakage, m^3/s
    P_loss: torch.Tensor                 # power loss, W
    F_applied: torch.Tensor              # applied force, N (= k_F * F_hyd)
    De_sq: torch.Tensor                  # squeeze Deborah at h_eq
    DeltaF_elastic_over_F: torch.Tensor  # elastic fraction (signed)

    # Constraint slacks (eq:ar_constraints, sign: >= 0 means feasible)
    g1_gap: torch.Tensor       # h_min - h_safe
    g2_thermal: torch.Tensor   # T_fail - T_max
    g3_validity: torch.Tensor  # Desq_valid - De_sq

    # Status
    converged: torch.Tensor    # bool — solver succeeded end-to-end
    feasible: torch.Tensor     # bool — converged AND all g_j >= 0

    # Ancillary (kept for diagnostics / persistence)
    failure_mode: list[Optional[str]]  # one string-or-None per row
    wall_seconds: torch.Tensor         # per-sample wall time, float

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dict view (e.g. for Parquet serialisation)."""
        return dataclasses.asdict(self)


# ============================================================================
# Adapter
# ============================================================================


class SolverAdapter:
    """BoTorch-compatible wrapper around `ar_bo_campaign.run_sample`."""

    def __init__(
        self,
        *,
        nr: int = NR_DEFAULT,
        ntheta: int = NTHETA_DEFAULT,
        config: Optional[SolverConfig] = None,
        do_gnf_compare: bool = True,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        cache_size: int = 4096,
        cache_round_decimals: int = 12,
    ) -> None:
        """
        Parameters
        ----------
        nr, ntheta
            Mesh resolution passed to run_sample (default 30x48).
        config
            SolverConfig override; defaults to the run_sample default
            (max_iter=50, tol=1e-6, omega=0.6, viscoelastic + memory).
        do_gnf_compare
            Run the matched-viscosity GNF re-solve at every sample.
            BO needs DeltaF_elastic_over_F as a possible objective, so
            default True.
        dtype, device
            Target dtype/device for output tensors. GPyTorch generally
            wants float64 on CPU for stable Cholesky factorisations.
        cache_size
            Maximum number of (rounded-coordinate, evaluation) pairs
            kept in memory. LRU eviction.
        cache_round_decimals
            Decimal places used to key the cache. Twelve is enough to
            avoid spurious cache-hits on near-but-not-equal queries
            while still catching the multi-restart-L-BFGS duplicates.
        """
        self.nr = int(nr)
        self.ntheta = int(ntheta)
        self.config = config
        self.do_gnf_compare = bool(do_gnf_compare)
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device
        self.cache_size = int(cache_size)
        self.cache_round_decimals = int(cache_round_decimals)

        self.d = len(INPUT_SPACE)  # 8 dimensions
        self._cache: "OrderedDict[tuple, dict[str, Any]]" = OrderedDict()
        self._n_calls = 0
        self._n_cache_hits = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(self, X_unit: torch.Tensor) -> SolverEvaluation:
        """Evaluate the solver on a (q, d) batch of unit-cube points.

        Parameters
        ----------
        X_unit : torch.Tensor of shape (q, d) or (d,) for q=1
            Unit-cube coordinates in [0, 1]^d. Values outside [0, 1]
            are passed through to INPUT_SPACE.from_unit (which will
            extrapolate); the caller is responsible for clipping if
            that is undesirable.

        Returns
        -------
        SolverEvaluation
            Tensors of shape (q,), in the adapter's dtype/device.
        """
        X_unit = self._to_2d_tensor(X_unit)
        q = X_unit.shape[0]
        if X_unit.shape[1] != self.d:
            raise ValueError(
                f"SolverAdapter.evaluate: expected X_unit of shape (q, {self.d}); "
                f"got {tuple(X_unit.shape)}"
            )

        records: list[dict[str, Any]] = []
        for k in range(q):
            row = X_unit[k].detach().cpu().numpy().astype(float)
            rec = self._evaluate_one(row)
            records.append(rec)
        self._n_calls += q

        return self._stack(records)

    # Convenient short alias
    __call__ = evaluate

    # ------------------------------------------------------------------
    # Per-row evaluation with cache
    # ------------------------------------------------------------------

    def _evaluate_one(self, u_row) -> dict[str, Any]:
        """Run run_sample for one unit-cube row, with cache."""
        key = self._cache_key(u_row)
        if key in self._cache:
            self._n_cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        x_phys = self._unit_to_physical(u_row)
        import time as _t
        t0 = _t.time()
        try:
            perf = run_sample(
                x_phys, nr=self.nr, ntheta=self.ntheta,
                config=self.config, do_gnf_compare=self.do_gnf_compare,
            )
            wall = _t.time() - t0
            cons = {
                "g1_gap":      perf["h_min"] - 3.0e-6,
                "g2_thermal":  (110.0 + 273.15) - perf["T_max"]
                                if perf.get("T_max") is not None
                                and isinstance(perf.get("T_max"), float) else float("nan"),
                "g3_validity": 0.2 - perf["De_sq"],
            }
            rec = {**x_phys, **perf, **cons,
                   "feasible": (is_feasible(perf) if perf.get("converged") else False),
                   "wall_seconds": wall}
        except Exception as exc:  # noqa: BLE001
            wall = _t.time() - t0
            logger.warning(f"SolverAdapter: solver raised on row "
                           f"{u_row.tolist()}: {exc!r}")
            rec = {**x_phys,
                   "converged": False,
                   "feasible": False,
                   "failure_mode": f"exception: {exc!r}",
                   "h_min": float("nan"),
                   "T_max": float("nan"),
                   "Q_leak": float("nan"),
                   "P_loss": float("nan"),
                   "F_applied": float("nan"),
                   "De_sq": float("nan"),
                   "DeltaF_elastic_over_F": float("nan"),
                   "g1_gap": float("nan"),
                   "g2_thermal": float("nan"),
                   "g3_validity": float("nan"),
                   "wall_seconds": wall}
        self._cache_put(key, rec)
        return rec

    def _cache_key(self, u_row) -> tuple:
        rounded = [round(float(v), self.cache_round_decimals) for v in u_row]
        return tuple(rounded)

    def _cache_put(self, key: tuple, rec: dict[str, Any]) -> None:
        self._cache[key] = rec
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def _unit_to_physical(self, u_row) -> dict[str, float]:
        """Map a single unit-cube row to a physical-input dict for run_sample."""
        out: dict[str, float] = {}
        for j, dim in enumerate(INPUT_SPACE):
            out[dim.name] = float(dim.from_unit(float(u_row[j])))
        return out

    def physical_to_unit(self, x_phys: dict[str, float]) -> torch.Tensor:
        """Inverse map. Raises if any required dimension is missing."""
        u = torch.empty(self.d, dtype=self.dtype, device=self.device)
        for j, dim in enumerate(INPUT_SPACE):
            if dim.name not in x_phys:
                raise KeyError(
                    f"physical_to_unit: missing dimension '{dim.name}' "
                    f"in input dict (expected {[d.name for d in INPUT_SPACE]})"
                )
            u[j] = float(dim.to_unit(float(x_phys[dim.name])))
        return u

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_2d_tensor(self, X_unit: torch.Tensor) -> torch.Tensor:
        if not isinstance(X_unit, torch.Tensor):
            X_unit = torch.as_tensor(X_unit, dtype=self.dtype, device=self.device)
        if X_unit.dim() == 1:
            X_unit = X_unit.unsqueeze(0)
        if X_unit.dim() != 2:
            raise ValueError(
                f"SolverAdapter expected 1D or 2D input, got {X_unit.dim()}-D"
            )
        return X_unit.to(dtype=self.dtype, device=self.device)

    def _stack(self, records: list[dict[str, Any]]) -> SolverEvaluation:
        def col(name: str, *, default=float("nan")) -> torch.Tensor:
            return torch.tensor(
                [float(r.get(name, default))
                 if r.get(name) is not None and r.get(name) != "" else default
                 for r in records],
                dtype=self.dtype, device=self.device,
            )

        def boolcol(name: str) -> torch.Tensor:
            return torch.tensor(
                [bool(r.get(name, False)) for r in records],
                dtype=torch.bool, device=self.device,
            )

        return SolverEvaluation(
            h_min=col("h_min"),
            T_max=col("T_max"),
            Q_leak=col("Q_leak"),
            P_loss=col("P_loss"),
            F_applied=col("F_applied"),
            De_sq=col("De_sq"),
            DeltaF_elastic_over_F=col("DeltaF_elastic_over_F"),
            g1_gap=col("g1_gap"),
            g2_thermal=col("g2_thermal"),
            g3_validity=col("g3_validity"),
            converged=boolcol("converged"),
            feasible=boolcol("feasible"),
            failure_mode=[r.get("failure_mode") for r in records],
            wall_seconds=col("wall_seconds", default=0.0),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        return {
            "n_calls": int(self._n_calls),
            "n_cache_hits": int(self._n_cache_hits),
            "cache_size": int(len(self._cache)),
        }
