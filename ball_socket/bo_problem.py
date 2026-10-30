"""ball_socket/bo_problem.py -- wires the ball-socket evaluator into the
slipper-validated BO machinery (`src.bo`).

The framework is carried unchanged from the slipper: Matern-5/2 ARD, N_init=18,
EIC with entropy boundary targeting, a feasibility classifier GP absorbing the
fourth outcome, and the joint-posterior decision rule as the named safety
instrument.  Only the evaluator behind `Problem.evaluate_fn` is new.

Constraints are returned in the framework's convention **g >= 0 means
feasible**, normalised so all three are O(1):

    g1_gap      = h_min / (Lambda_min * sigma*) - 1
    g2_thermal  = (T_ceiling(class) - T_max) / T_ceiling(class)
    g3_validity = 1 - De_sq / De_sq_max

Non-converged samples (eccentricity blow-up) are reported with
``converged = False`` and strongly negative constraint values, so the
classifier GP learns the non-convergence geography while the constraint GPs
are not fed NaN.
"""
from __future__ import annotations

import numpy as np
import torch

from .campaign import BANDS, evaluate_x, unit_to_x
from .campaign_config import ConstraintSet

_C = ConstraintSet()
_FAIL = -1.0          # constraint value assigned to non-converged samples


def make_problem(fluid_class: str = "mineral", band: str = "lo",
                 name: str | None = None):
    """Build a `src.bo.campaign.Problem` for one (fluid class, sigma* band).

    Both are registered as bands/classes rather than axes, so the campaign is
    run per combination and reported per combination.
    """
    from src.bo.campaign import Problem

    sigma = BANDS[band]
    T_ceiling = _C.T_max_C_by_class[fluid_class]

    def _eval(X_unit: torch.Tensor) -> dict[str, torch.Tensor]:
        X = X_unit.detach().cpu().numpy()
        if X.ndim == 1:
            X = X[None, :]
        cols: dict[str, list] = {k: [] for k in
                                 ("h_min", "T_max", "P_loss", "Q_leak", "De_sq",
                                  "F", "g1_gap", "g2_thermal", "g3_validity",
                                  "converged", "feasible")}
        for u in X:
            x = unit_to_x(u)
            r = evaluate_x(x)
            if r.get("outcome") == "error":
                # PRE-COMMITTED: an infrastructure failure is NOT the fourth
                # outcome and must never become a feasibility label.  Retry
                # once, then halt the run rather than poison the classifier.
                r = evaluate_x(x)
                if r.get("outcome") == "error":
                    raise RuntimeError(
                        "infrastructure failure in the BO evaluator, twice at the "
                        f"same point; halting rather than labelling it as "
                        f"non-convergence: {r.get('fail_reason')}")
            ok = bool(r.get("converged", False))
            cols["converged"].append(ok)
            if not ok:
                for k in ("h_min", "T_max", "P_loss", "Q_leak", "De_sq", "F"):
                    cols[k].append(float("nan"))
                for k in ("g1_gap", "g2_thermal", "g3_validity"):
                    cols[k].append(_FAIL)
                cols["feasible"].append(False)
                continue
            h_min, T_max, De = r["h_min"], r["T_max"], r["De_sq"]
            g1 = h_min / (_C.lambda_ratio_min * sigma) - 1.0
            g2 = (T_ceiling - T_max) / T_ceiling
            g3 = 1.0 - De / _C.De_sq_max
            for k, v in (("h_min", h_min), ("T_max", T_max),
                         ("P_loss", r["P_loss"]), ("Q_leak", r["Q_leak"]),
                         ("De_sq", De), ("F", r["F"]),
                         ("g1_gap", g1), ("g2_thermal", g2), ("g3_validity", g3)):
                cols[k].append(float(v))
            cols["feasible"].append(bool(g1 >= 0 and g2 >= 0 and g3 >= 0))

        out = {k: torch.tensor(v, dtype=torch.double)
               for k, v in cols.items() if k not in ("converged", "feasible")}
        out["converged"] = torch.tensor(cols["converged"], dtype=torch.bool)
        out["feasible"] = torch.tensor(cols["feasible"], dtype=torch.bool)
        return out

    return Problem(
        evaluate_fn=_eval,
        d=8,
        objective_name="h_min",
        objective_sense="min",
        constraint_names=["g1_gap", "g2_thermal", "g3_validity"],
        # F and Q_leak are DECLARED so BOCampaign persists them (ruling 3.1,
        # 2026-07-29).  The first campaign declared only T_max and P_loss, so
        # Q_leak was never written and the (Q_leak, F) projection could not be
        # reconstructed for the BO arms without an eight-hour re-run.  Declaring
        # them costs two extra regression GPs and takes effect on the next run.
        #
        # Note that F is additionally recoverable analytically -- at equilibrium
        # the film load equals the hydrodynamic share p_ch(A_p - A_pocket),
        # deterministic in the inputs -- so it is carried here for persistence
        # rather than because it needs to be GP-modelled.
        auxiliary_names=["T_max", "P_loss", "Q_leak", "F"],
        name=name or f"ball_socket_{fluid_class}_{band}",
    )
