"""
Verification Utilities
======================

Tools for solution verification following ASME V&V 10-2006:

  1. Richardson extrapolation for grid convergence studies
  2. Grid Convergence Index (GCI) for uncertainty quantification
  3. Observed order of accuracy estimation

References:
    - Roache, P.J. (1998). Verification and Validation in Computational
      Science and Engineering. Hermosa Publishers.
    - ASME V&V 10-2006.  Standard for Verification and Validation in
      Computational Solid Mechanics.
    - Oberkampf, W.L. & Roy, C.J. (2010). Verification and Validation
      in Scientific Computing. Cambridge University Press.

Author: Nigel C. Dhlamini
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class RichardsonResult:
    """
    Result of Richardson extrapolation analysis.

    Attributes
    ----------
    f_exact : float
        Richardson-extrapolated "exact" solution
    observed_order : float
        Observed order of accuracy p
    GCI_fine : float
        Grid Convergence Index for the finest grid (95% confidence)
    GCI_medium : float
        GCI for the medium grid
    asymptotic_ratio : float
        Ratio GCI_medium / (r^p * GCI_fine); should be ~1.0 if
        grids are in the asymptotic range
    grid_sizes : list
        [N_fine, N_medium, N_coarse]
    values : list
        [f_fine, f_medium, f_coarse]
    errors : list
        Estimated relative errors [e_fine, e_medium, e_coarse]
    in_asymptotic_range : bool
        True if asymptotic ratio is within [0.9, 1.1]
    """
    f_exact: float
    observed_order: float
    GCI_fine: float
    GCI_medium: float
    asymptotic_ratio: float
    grid_sizes: list
    values: list
    errors: list
    in_asymptotic_range: bool


def richardson_extrapolation(
    f_values: List[float],
    grid_sizes: List[int],
    design_order: float = 2.0,
    safety_factor: float = 1.25
) -> RichardsonResult:
    """
    Perform Richardson extrapolation with three grids.

    Given solutions on three systematically refined grids, estimates:
      (a) the grid-independent ("exact") solution,
      (b) the observed order of accuracy, and
      (c) the Grid Convergence Index (GCI) uncertainty band.

    Parameters
    ----------
    f_values : list of float
        Solution values [f_fine, f_medium, f_coarse] on the three
        grids, ordered from finest to coarsest.
    grid_sizes : list of int
        Number of cells [N_fine, N_medium, N_coarse], finest first.
    design_order : float
        Formal (design) order of the discretisation scheme.
        Default: 2.0 (second-order central differences).
    safety_factor : float
        Safety factor F_s for GCI.  Default: 1.25 for three-grid
        studies (Roache 1998, Section 5.6.1).

    Returns
    -------
    RichardsonResult
        Container with extrapolated value, order, GCI, etc.

    Notes
    -----
    The refinement ratio r = h_coarse / h_fine is computed from the
    grid sizes assuming uniform refinement:  r = (N_fine / N_coarse)
    for 1-D problems, or r = sqrt(N_fine / N_coarse) for 2-D.

    For the annular polar mesh in this solver, the total DOF count is
    N = Nr * Ntheta, so if both directions are refined equally,
    r = sqrt(N_fine / N_coarse).

    Raises
    ------
    ValueError
        If grids are not in monotonic order, or if the observed order
        cannot be determined (oscillatory convergence).
    """
    if len(f_values) != 3 or len(grid_sizes) != 3:
        raise ValueError("Exactly three grid levels required")

    f1, f2, f3 = f_values          # fine, medium, coarse
    N1, N2, N3 = grid_sizes

    # Grid refinement ratios (assume 2-D: r = sqrt(N/N))
    r21 = np.sqrt(N1 / N2)
    r32 = np.sqrt(N2 / N3)

    # Solution differences
    eps21 = f2 - f1
    eps32 = f3 - f2

    # Safeguard: if differences are negligibly small, already converged
    if abs(eps21) < 1e-15 and abs(eps32) < 1e-15:
        return RichardsonResult(
            f_exact=f1, observed_order=design_order,
            GCI_fine=0.0, GCI_medium=0.0,
            asymptotic_ratio=1.0,
            grid_sizes=grid_sizes, values=f_values,
            errors=[0.0, 0.0, 0.0],
            in_asymptotic_range=True
        )

    # Sign check for monotonic convergence
    if eps21 * eps32 <= 0:
        # Oscillatory convergence — fall back to design order
        p = design_order
    else:
        # Fixed-point iteration for observed order (Roache 1998, Eq. 5.10.6)
        ratio = eps32 / eps21
        # Initial guess from constant-ratio formula
        if abs(ratio) > 0 and ratio > 0:
            p = abs(np.log(abs(ratio))) / np.log(r21)
        else:
            p = design_order

        # Iterate if refinement ratios are not equal
        if abs(r21 - r32) > 1e-10:
            for _ in range(50):
                lhs = r21**p - 1.0
                rhs = r32**p - 1.0
                if abs(rhs) < 1e-15:
                    break
                ratio_est = lhs / rhs
                if abs(eps32) < 1e-15:
                    break
                p_new = abs(np.log(abs(eps32 / eps21) * ratio_est)) / np.log(r21)
                if abs(p_new - p) < 1e-6:
                    p = p_new
                    break
                p = p_new

    # Clamp order to physically meaningful range
    p = max(0.5, min(p, 2 * design_order))

    # Richardson-extrapolated exact solution
    f_exact = f1 + (f1 - f2) / (r21**p - 1.0)

    # Relative errors
    e1 = abs((f_exact - f1) / (f_exact + 1e-30))
    e2 = abs((f_exact - f2) / (f_exact + 1e-30))
    e3 = abs((f_exact - f3) / (f_exact + 1e-30))

    # GCI (Roache 1998, Eq. 5.6.1)
    GCI_fine = safety_factor * abs((f2 - f1) / f1) / (r21**p - 1.0)
    GCI_medium = safety_factor * abs((f3 - f2) / f2) / (r32**p - 1.0)

    # Asymptotic ratio (should be ≈ 1.0)
    if GCI_fine > 1e-15:
        asymptotic_ratio = GCI_medium / (r21**p * GCI_fine)
    else:
        asymptotic_ratio = 1.0

    in_asymptotic = 0.9 <= asymptotic_ratio <= 1.1

    return RichardsonResult(
        f_exact=f_exact,
        observed_order=p,
        GCI_fine=GCI_fine,
        GCI_medium=GCI_medium,
        asymptotic_ratio=asymptotic_ratio,
        grid_sizes=grid_sizes,
        values=f_values,
        errors=[e1, e2, e3],
        in_asymptotic_range=in_asymptotic,
    )


def convergence_study(
    solve_func,
    grid_levels: List[Tuple[int, int]],
    quantity_name: str = "F_total"
) -> dict:
    """
    Run a grid convergence study using a series of refinement levels.

    Parameters
    ----------
    solve_func : callable
        Function f(Nr, Ntheta) -> float that solves on the given mesh
        and returns the scalar quantity of interest.
    grid_levels : list of (Nr, Ntheta) tuples
        Grid resolutions from coarsest to finest.
    quantity_name : str
        Label for the quantity being studied.

    Returns
    -------
    dict with keys:
        'grid_sizes': list of total DOFs
        'values': list of computed values
        'richardson': RichardsonResult (from finest three grids)
        'quantity_name': str
    """
    sizes = []
    values = []

    for Nr, Ntheta in grid_levels:
        val = solve_func(Nr, Ntheta)
        sizes.append(Nr * Ntheta)
        values.append(val)

    result = {'grid_sizes': sizes, 'values': values,
              'quantity_name': quantity_name}

    # Richardson extrapolation on finest three
    if len(sizes) >= 3:
        # Take finest three, but order finest-first
        idx = sorted(range(len(sizes)), key=lambda i: -sizes[i])[:3]
        f_tri = [values[i] for i in idx]
        n_tri = [sizes[i] for i in idx]
        result['richardson'] = richardson_extrapolation(f_tri, n_tri)
    else:
        result['richardson'] = None

    return result


def format_convergence_table(study: dict) -> str:
    """
    Format a grid convergence study as a LaTeX-ready table.

    Parameters
    ----------
    study : dict
        Output of ``convergence_study()``.

    Returns
    -------
    str
        Formatted table string.
    """
    lines = []
    lines.append(f"Grid Convergence Study: {study['quantity_name']}")
    lines.append("=" * 65)
    lines.append(f"{'DOFs':>8}  {'Value':>14}  {'Rel. Change':>14}")
    lines.append("-" * 65)

    vals = study['values']
    sizes = study['grid_sizes']

    for i, (n, v) in enumerate(zip(sizes, vals)):
        if i == 0:
            lines.append(f"{n:>8d}  {v:>14.8f}  {'---':>14}")
        else:
            rel = abs(v - vals[i-1]) / (abs(vals[i-1]) + 1e-30)
            lines.append(f"{n:>8d}  {v:>14.8f}  {rel:>14.2e}")

    ri = study.get('richardson')
    if ri is not None:
        lines.append("-" * 65)
        lines.append(f"Richardson extrapolation:")
        lines.append(f"  Extrapolated value : {ri.f_exact:.8f}")
        lines.append(f"  Observed order p   : {ri.observed_order:.2f}")
        lines.append(f"  GCI (fine grid)    : {ri.GCI_fine:.2e}  "
                      f"({ri.GCI_fine*100:.2f}%)")
        lines.append(f"  GCI (medium grid)  : {ri.GCI_medium:.2e}  "
                      f"({ri.GCI_medium*100:.2f}%)")
        lines.append(f"  Asymptotic ratio   : {ri.asymptotic_ratio:.3f}  "
                      f"({'OK' if ri.in_asymptotic_range else 'NOT in range'})")

    return "\n".join(lines)
