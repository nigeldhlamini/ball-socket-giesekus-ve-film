"""
Uncertainty Quantification for Validation
==========================================

Provides tools for computing and propagating uncertainties in the
Giesekus-based Reynolds equation solver, following ASME V&V 10-2006.

Three categories of uncertainty are addressed:

  1. **Numerical uncertainty** (discretisation error):
     Quantified via Grid Convergence Index (GCI) from ``verification.py``.

  2. **Input uncertainty** (parameter sensitivity):
     Propagated via a local sensitivity analysis (one-at-a-time perturbation).

  3. **Model-form uncertainty**:
     Estimated from the spread between Newtonian, GNF, and full
     viscoelastic predictions, and from comparison with experimental data.

References:
    - ASME V&V 10-2006.
    - Oberkampf, W.L. & Roy, C.J. (2010). Verification and Validation
      in Scientific Computing. Cambridge University Press.
    - Coleman, H.W. & Steele, W.G. (2009). Experimentation, Validation,
      and Uncertainty Analysis for Engineers, 3rd ed. Wiley.

Author: Nigel C. Dhlamini
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SensitivityResult:
    """
    Result of a local parameter sensitivity analysis.

    Attributes
    ----------
    parameter_name : str
        Name of the perturbed parameter
    nominal_value : float
        Nominal parameter value
    perturbation : float
        Fractional perturbation applied (e.g. 0.05 for 5%)
    nominal_output : float
        Output at nominal parameter value
    perturbed_plus : float
        Output at (1 + perturbation) * nominal
    perturbed_minus : float
        Output at (1 - perturbation) * nominal
    sensitivity : float
        Central-difference sensitivity: d(output)/d(param) * param / output
        (dimensionless, logarithmic sensitivity)
    """
    parameter_name: str
    nominal_value: float
    perturbation: float
    nominal_output: float
    perturbed_plus: float
    perturbed_minus: float
    sensitivity: float


@dataclass
class UncertaintyBudget:
    """
    Combined uncertainty budget for a scalar output quantity.

    Attributes
    ----------
    quantity_name : str
        Name of the output quantity (e.g. "F_total", "Q_leak")
    nominal_value : float
        Nominal (best-estimate) value
    numerical_uncertainty : float
        From GCI (95% confidence)
    input_uncertainties : dict
        {parameter_name: uncertainty_contribution}
    model_form_uncertainty : float
        Estimated model-form error
    total_uncertainty : float
        Combined (RSS) uncertainty at 95% confidence
    coverage_factor : float
        Coverage factor k (default 2 for 95%)
    """
    quantity_name: str
    nominal_value: float
    numerical_uncertainty: float = 0.0
    input_uncertainties: Dict[str, float] = field(default_factory=dict)
    model_form_uncertainty: float = 0.0
    total_uncertainty: float = 0.0
    coverage_factor: float = 2.0

    def compute_total(self):
        """Compute combined uncertainty (root-sum-of-squares)."""
        u_num = self.numerical_uncertainty
        u_inp = np.sqrt(sum(v**2 for v in self.input_uncertainties.values()))
        u_mod = self.model_form_uncertainty
        self.total_uncertainty = self.coverage_factor * np.sqrt(
            u_num**2 + u_inp**2 + u_mod**2
        )
        return self.total_uncertainty

    @property
    def relative_total(self) -> float:
        """Total uncertainty as fraction of nominal value."""
        if abs(self.nominal_value) < 1e-30:
            return 0.0
        return self.total_uncertainty / abs(self.nominal_value)


def parameter_sensitivity(
    solve_func: Callable[[Dict[str, float]], float],
    params: Dict[str, float],
    perturbation: float = 0.05,
    parameters_to_vary: Optional[List[str]] = None,
) -> List[SensitivityResult]:
    """
    Local one-at-a-time parameter sensitivity analysis.

    Parameters
    ----------
    solve_func : callable
        Function f(params_dict) -> float that solves the problem and
        returns the scalar output of interest.
    params : dict
        Nominal parameter values {name: value}.
    perturbation : float
        Fractional perturbation (e.g. 0.05 = 5%).
    parameters_to_vary : list of str, optional
        Subset of parameter names to perturb.  Default: all.

    Returns
    -------
    list of SensitivityResult
        One entry per perturbed parameter, sorted by |sensitivity|.
    """
    if parameters_to_vary is None:
        parameters_to_vary = list(params.keys())

    # Nominal output
    f_nom = solve_func(params)

    results = []
    for name in parameters_to_vary:
        val = params[name]
        if abs(val) < 1e-30:
            continue  # skip zero parameters

        # Perturbed evaluations
        params_plus = dict(params)
        params_plus[name] = val * (1 + perturbation)
        f_plus = solve_func(params_plus)

        params_minus = dict(params)
        params_minus[name] = val * (1 - perturbation)
        f_minus = solve_func(params_minus)

        # Logarithmic sensitivity (dimensionless)
        if abs(f_nom) > 1e-30:
            sens = (f_plus - f_minus) / (2 * perturbation * f_nom)
        else:
            sens = 0.0

        results.append(SensitivityResult(
            parameter_name=name,
            nominal_value=val,
            perturbation=perturbation,
            nominal_output=f_nom,
            perturbed_plus=f_plus,
            perturbed_minus=f_minus,
            sensitivity=sens,
        ))

    # Sort by absolute sensitivity (most sensitive first)
    results.sort(key=lambda r: -abs(r.sensitivity))
    return results


def propagate_input_uncertainty(
    sensitivities: List[SensitivityResult],
    parameter_uncertainties: Dict[str, float],
) -> Dict[str, float]:
    """
    Propagate input parameter uncertainties to the output.

    Uses linear error propagation:
        u_output_i = |sensitivity_i| * (u_param_i / param_i) * |output|

    Parameters
    ----------
    sensitivities : list of SensitivityResult
        From ``parameter_sensitivity()``.
    parameter_uncertainties : dict
        {parameter_name: fractional_uncertainty} (e.g. 0.05 for 5%).

    Returns
    -------
    dict
        {parameter_name: contribution_to_output_uncertainty}
    """
    contributions = {}
    for s in sensitivities:
        u_param = parameter_uncertainties.get(s.parameter_name, 0.0)
        # u_output_i = |sensitivity_i * u_param_fraction * nominal_output|
        contributions[s.parameter_name] = abs(
            s.sensitivity * u_param * s.nominal_output
        )
    return contributions


def compute_uncertainty_budget(
    quantity_name: str,
    nominal_value: float,
    GCI_fine: float = 0.0,
    sensitivities: Optional[List[SensitivityResult]] = None,
    parameter_uncertainties: Optional[Dict[str, float]] = None,
    model_form_error: float = 0.0,
    coverage_factor: float = 2.0,
) -> UncertaintyBudget:
    """
    Compute a complete uncertainty budget for a scalar output.

    Parameters
    ----------
    quantity_name : str
        Label for the output quantity.
    nominal_value : float
        Best-estimate value.
    GCI_fine : float
        Grid Convergence Index from Richardson extrapolation.
    sensitivities : list, optional
        From ``parameter_sensitivity()``.
    parameter_uncertainties : dict, optional
        {parameter_name: fractional_uncertainty}.
    model_form_error : float
        Estimated model-form error (absolute).
    coverage_factor : float
        k for expanded uncertainty (default 2 = 95%).

    Returns
    -------
    UncertaintyBudget
    """
    budget = UncertaintyBudget(
        quantity_name=quantity_name,
        nominal_value=nominal_value,
        numerical_uncertainty=GCI_fine * abs(nominal_value),
        model_form_uncertainty=model_form_error,
        coverage_factor=coverage_factor,
    )

    if sensitivities is not None and parameter_uncertainties is not None:
        budget.input_uncertainties = propagate_input_uncertainty(
            sensitivities, parameter_uncertainties
        )

    budget.compute_total()
    return budget


def format_uncertainty_budget(budget: UncertaintyBudget) -> str:
    """
    Format an uncertainty budget as a printable table.

    Parameters
    ----------
    budget : UncertaintyBudget

    Returns
    -------
    str
    """
    lines = []
    lines.append(f"Uncertainty Budget: {budget.quantity_name}")
    lines.append("=" * 60)
    lines.append(f"  Nominal value: {budget.nominal_value:.6g}")
    lines.append(f"  Coverage factor k = {budget.coverage_factor}")
    lines.append("-" * 60)
    lines.append(f"  {'Source':30s} {'u_i':>12s} {'% of total':>12s}")
    lines.append("-" * 60)

    total_var = (budget.total_uncertainty / budget.coverage_factor)**2
    if total_var < 1e-30:
        total_var = 1.0

    # Numerical
    u_num = budget.numerical_uncertainty
    pct_num = u_num**2 / total_var * 100
    lines.append(f"  {'Numerical (GCI)':30s} {u_num:>12.2e} {pct_num:>11.1f}%")

    # Input parameters
    for name, u_i in sorted(budget.input_uncertainties.items(),
                             key=lambda x: -x[1]):
        pct_i = u_i**2 / total_var * 100
        lines.append(f"  {name:30s} {u_i:>12.2e} {pct_i:>11.1f}%")

    # Model form
    u_mod = budget.model_form_uncertainty
    pct_mod = u_mod**2 / total_var * 100
    lines.append(f"  {'Model form':30s} {u_mod:>12.2e} {pct_mod:>11.1f}%")

    lines.append("-" * 60)
    lines.append(f"  {'TOTAL (expanded, k='+str(budget.coverage_factor)+')':30s} "
                 f"{budget.total_uncertainty:>12.2e}")
    lines.append(f"  {'Relative':30s} "
                 f"{budget.relative_total*100:>11.2f}%")

    return "\n".join(lines)
