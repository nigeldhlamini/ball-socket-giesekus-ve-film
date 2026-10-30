"""
Viscoelastic Reynolds Equation Solver
=====================================

Implements Picard iteration for the nonlinear viscoelastic Reynolds equation.

Algorithm:
    1. Initialize η̄ = η₀ (zero-shear viscosity)
    2. Assemble system A·p = b with current η̄
    3. Solve for pressure p
    4. Update shear rate γ̄̇(p)
    5. Update viscosity η̄ = η(γ̄̇)
    6. Check convergence
    7. Repeat until converged

Physics Notes:
    The Giesekus model predicts LOAD REDUCTION (not enhancement) in squeeze flow.
    This is consistent with Phan-Thien & Tanner (1983) for Maxwell fluids:
    
    "The solutions show wave propagation and show a reduced load capacity 
    relative to the Newtonian case."
    
    Physical explanation: The memory operator adds to the GNF diffusion operator,
    increasing effective "diffusivity" of the pressure equation. For fixed squeeze
    rate, more diffusion → lower pressure → lower load.
    
    Load ENHANCEMENT requires stress overshoot (Modified PTT model), which is
    not present in standard Giesekus. See docs/PHYSICS_NOTES.md for details.

References:
    - Dhlamini, N.C. (2025). PhD Thesis, Chapter 4.
    - Phan-Thien, N. & Tanner, R.I. (1983). J. Fluid Mech. 129, 265–281.
    - Phan-Thien, N., Sugeng, F. & Tanner, R.I. (1987). J. Non-Newtonian Fluid Mech. 24, 97–119.

Author: Nigel C. Dhlamini
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable
import numpy as np
from scipy.sparse.linalg import spsolve, gmres, bicgstab
import warnings

from .rheology import GiesekusFluid
from .geometry import SlipperGeometry, OperatingConditions
from .mesh import PolarMesh
from .fluxes import (
    compute_gap_averaged_shear_rate, 
    FluxCalculator,
    compute_Q_GNF,  # Added for Newtonian/GNF leakage
)
from .assembly import (
    assemble_newtonian_system,
    assemble_GNF_system,
    assemble_viscoelastic_system,
    AssembledSystem
)
from .results import (
    SolverResult,
    FluxDecomposition,
    compute_load_capacity,
    compute_moments,
    compute_leakage,
)


# =============================================================================
# Solver Configuration
# =============================================================================

@dataclass
class SolverConfig:
    """
    Configuration for the Reynolds equation solver.

    Attributes
    ----------
    max_iter : int
        Maximum number of Picard iterations
    tol : float
        Convergence tolerance on relative pressure change
    omega : float
        Under-relaxation factor (0 < ω ≤ 1)
    linear_solver : str
        Linear solver: 'direct', 'gmres', 'bicgstab'
    verbose : bool
        Print iteration progress
    include_memory : bool
        Include memory operator in viscoelastic solve
    include_normal_stress : bool
        Include N₁ and hoop stress terms
    ns_warmup : int
        Number of GNF+memory-only warm-up iterations before
        introducing normal stress RHS contributions.  This allows
        the pressure field to converge to a reasonable baseline
        before the quadratic |∇p|² coupling is activated.
    ns_ramp : int
        Number of iterations over which the normal stress scaling
        factor is linearly ramped from 0 → 1 after warm-up.
    """
    max_iter: int = 50
    tol: float = 1e-6
    omega: float = 0.7
    linear_solver: str = 'direct'
    verbose: bool = False
    include_memory: bool = True
    include_normal_stress: bool = True
    include_alpha_coupling: bool = False  # WP-C: written term is wrong & subdominant
    memory_R_fn: Optional[Callable] = None  # WP-G: R(Lambda) renormalisation of memory flux
    ns_warmup: int = 5
    ns_ramp: int = 5

    def __post_init__(self):
        if not 0 < self.omega <= 1:
            raise ValueError(f"Under-relaxation omega must be in (0, 1], got {self.omega}")
        if self.ns_warmup < 0:
            raise ValueError(f"ns_warmup must be non-negative, got {self.ns_warmup}")
        if self.ns_ramp < 1:
            raise ValueError(f"ns_ramp must be >= 1, got {self.ns_ramp}")


# =============================================================================
# Linear Solver Wrapper
# =============================================================================

def solve_linear_system(A, b, method: str = 'direct') -> np.ndarray:
    """
    Solve the linear system A·x = b.
    
    Parameters
    ----------
    A : sparse matrix
        System matrix
    b : ndarray
        Right-hand side
    method : str
        Solver method: 'direct', 'gmres', 'bicgstab'
    
    Returns
    -------
    x : ndarray
        Solution vector
    """
    if method == 'direct':
        return spsolve(A, b)
    elif method == 'gmres':
        x, info = gmres(A, b, tol=1e-10, maxiter=500)
        if info != 0:
            warnings.warn(f"GMRES did not converge: info = {info}")
        return x
    elif method == 'bicgstab':
        x, info = bicgstab(A, b, tol=1e-10, maxiter=500)
        if info != 0:
            warnings.warn(f"BiCGSTAB did not converge: info = {info}")
        return x
    else:
        raise ValueError(f"Unknown linear solver: {method}")


# =============================================================================
# Helper Function for Computing Leakage from Pressure Field
# =============================================================================

def compute_leakage_from_pressure(
    pressure: np.ndarray,
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions,
    eta_bar: np.ndarray,
    boundary: str = 'outer'
) -> float:
    """
    Compute leakage flow rate from the pressure field.
    
    This function computes the radial flux Q_r and integrates it at the
    specified boundary to get the volumetric leakage rate.
    
    Parameters
    ----------
    pressure : ndarray
        Pressure field [Pa], shape (Nr, Ntheta)
    mesh : PolarMesh
        Computational mesh
    geometry : SlipperGeometry
        Bearing geometry
    conditions : OperatingConditions
        Operating parameters
    eta_bar : ndarray
        Viscosity field [Pa·s], shape (Nr, Ntheta)
    boundary : str
        'outer' or 'inner'
    
    Returns
    -------
    Q_leak : float
        Volumetric leakage rate [m³/s]. Positive = outward flow.
    """
    R, THETA = mesh.R, mesh.THETA
    h = geometry.gap_height(R, THETA)
    
    # Get velocity components
    V_Tr, V_Ttheta = conditions.radial_tangential_velocity(THETA)
    
    # Compute pressure gradients
    dp_dr, dp_dtheta = mesh.compute_field_gradient(pressure)
    
    # Compute GNF (Poiseuille + Couette) flux
    Q_r_GNF, Q_theta_GNF = compute_Q_GNF(
        h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, R, eta_bar
    )
    
    # Integrate flux at boundary
    return compute_leakage(Q_r_GNF, mesh, boundary)


# =============================================================================
# Main Solver Functions
# =============================================================================

def solve_newtonian(
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions,
    eta: float,
    config: Optional[SolverConfig] = None
) -> SolverResult:
    """
    Solve the Newtonian Reynolds equation (constant viscosity).
    
    This is a single linear solve with no iteration required.
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    geometry : SlipperGeometry
        Bearing geometry
    conditions : OperatingConditions
        Operating parameters
    eta : float
        Constant viscosity [Pa·s]
    config : SolverConfig, optional
        Solver configuration
    
    Returns
    -------
    result : SolverResult
        Solution including pressure field and integrated quantities
    """
    if config is None:
        config = SolverConfig()
    
    # Assemble system
    system = assemble_newtonian_system(mesh, geometry, conditions, eta)
    
    # Solve
    p_1d = solve_linear_system(system.A, system.b, config.linear_solver)
    pressure = mesh.to_2d(p_1d)
    
    # Viscosity field (constant for Newtonian)
    eta_bar = eta * np.ones_like(pressure)
    
    # Create result
    result = SolverResult(
        pressure=pressure,
        converged=True,
        iterations=1,
        residual_history=[0.0],
        eta_bar=eta_bar,
        gdot_bar=None,
        solver_info={'type': 'newtonian', 'eta': eta}
    )
    
    # Post-process: Load and moments
    result.F_total = compute_load_capacity(pressure, mesh)
    result.F_GNF = result.F_total
    result.F_elastic = 0.0
    result.M_x, result.M_y = compute_moments(pressure, mesh)
    
    # =========================================================================
    # FIX: Compute leakage for Newtonian case
    # =========================================================================
    result.Q_leak = compute_leakage_from_pressure(
        pressure, mesh, geometry, conditions, eta_bar, boundary='outer'
    )
    
    return result


def solve_GNF(
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions,
    fluid: GiesekusFluid,
    config: Optional[SolverConfig] = None
) -> SolverResult:
    """
    Solve the Generalised Newtonian Fluid Reynolds equation.
    
    Uses Picard iteration to handle the nonlinear viscosity η(γ̇).
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    geometry : SlipperGeometry
        Bearing geometry
    conditions : OperatingConditions
        Operating parameters
    fluid : GiesekusFluid
        Fluid model
    config : SolverConfig, optional
        Solver configuration
    
    Returns
    -------
    result : SolverResult
        Solution including pressure field and integrated quantities
    """
    if config is None:
        config = SolverConfig()
    
    R, THETA = mesh.R, mesh.THETA
    h = geometry.gap_height(R, THETA)
    V_Tr, V_Ttheta = conditions.radial_tangential_velocity(THETA)
    
    # Initialize viscosity with zero-shear value
    eta_bar = fluid.eta_0 * np.ones_like(h)
    
    # Initialize pressure (hydrostatic approximation)
    p_inner = conditions.p_supply if hasattr(conditions, 'p_supply') else 0.0
    p_outer = conditions.p_ambient if hasattr(conditions, 'p_ambient') else 0.0
    frac = (R - geometry.R_in) / (geometry.R_out - geometry.R_in)
    pressure = p_inner + (p_outer - p_inner) * frac
    
    residual_history = []
    converged = False
    
    for iteration in range(config.max_iter):
        # Store previous pressure
        p_old = pressure.copy()
        
        # Assemble system with current viscosity
        system = assemble_GNF_system(mesh, geometry, conditions, fluid, eta_bar)
        
        # Solve
        p_1d = solve_linear_system(system.A, system.b, config.linear_solver)
        p_new = mesh.to_2d(p_1d)
        
        # Under-relaxation
        pressure = config.omega * p_new + (1 - config.omega) * p_old
        
        # Update shear rate (including spin in azimuthal velocity)
        dp_dr, dp_dtheta = mesh.compute_field_gradient(pressure)
        gdot_bar = compute_gap_averaged_shear_rate(
            h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, R, eta_bar,
            omega_s=conditions.omega_s
        )

        # Update viscosity
        eta_bar_new = fluid.viscosity(gdot_bar)

        # Under-relax viscosity too
        eta_bar = config.omega * eta_bar_new + (1 - config.omega) * eta_bar

        # Check convergence
        residual = np.linalg.norm(pressure - p_old) / (np.linalg.norm(p_old) + 1e-15)
        residual_history.append(residual)

        if config.verbose:
            print(f"  Iter {iteration+1:3d}: residual = {residual:.3e}")
        
        if residual < config.tol:
            converged = True
            break
    
    # Create result
    result = SolverResult(
        pressure=pressure,
        converged=converged,
        iterations=iteration + 1,
        residual_history=residual_history,
        eta_bar=eta_bar,
        gdot_bar=gdot_bar,
        solver_info={'type': 'GNF'}
    )
    
    # Post-process: Load and moments
    result.F_total = compute_load_capacity(pressure, mesh)
    result.F_GNF = result.F_total
    result.F_elastic = 0.0
    result.M_x, result.M_y = compute_moments(pressure, mesh)
    
    # =========================================================================
    # FIX: Compute leakage for GNF case
    # =========================================================================
    result.Q_leak = compute_leakage_from_pressure(
        pressure, mesh, geometry, conditions, eta_bar, boundary='outer'
    )
    
    if not converged:
        warnings.warn(f"GNF solver did not converge after {config.max_iter} iterations")

    return result


def _warn_validity_envelope(mesh, geometry, conditions, fluid,
                            pressure, eta_bar, h, V_Tr, V_Ttheta):
    """
    Loud validity guard for the Table-7 envelope (P << 1, De_sq <~ 0.1,
    De <~ 0.3). Replaces the silent Gamma_sq clamp removed 2026-06
    (verification/e1_clamp_evidence.py): the scalarised N1/hoop fluxes and
    the quasi-local memory closure are derived for Couette-dominated
    kinematics and are unreliable outside the envelope -- most severely in
    pure squeeze (V_T = 0, P -> infinity), where the published-equation
    machinery must not be trusted. Never silently alters the solution.
    """
    lam = fluid.lambda_
    # Realised De_sq = lambda*|hdot|/h
    De_sq = lam * abs(conditions.h_dot) / float(np.min(h))
    # Realised P: Poiseuille/Couette shear ratio, h|grad p|/(2 eta) vs V_T/h
    dp_dr, dp_dtheta = mesh.compute_field_gradient(pressure)
    grad_p = np.sqrt(dp_dr**2 + (dp_dtheta / np.maximum(mesh.R, 1e-30))**2)
    V_T_mag = np.sqrt(V_Tr**2 + (V_Ttheta + conditions.omega_s * mesh.R)**2)
    pois_shear = h * grad_p / (2 * np.maximum(eta_bar, 1e-30))
    couette_shear = V_T_mag / h
    with np.errstate(divide='ignore'):
        P_real = float(np.max(np.where(couette_shear > 0,
                                       pois_shear / np.maximum(couette_shear, 1e-300),
                                       np.inf)))
    # Realised De = lambda*V_T/L (advective; L = radial extent)
    L = max(geometry.R_out - geometry.R_in, 1e-30)
    De = lam * float(np.max(V_T_mag)) / L

    breaches = []
    if not np.isfinite(P_real) or P_real > 0.2:
        breaches.append(f"P = {'inf (pure squeeze)' if not np.isfinite(P_real) else f'{P_real:.3g}'} (envelope: P << 1)")
    if De_sq > 0.1:
        breaches.append(f"De_sq = {De_sq:.3g} (envelope: <~ 0.1)")
    if De > 0.3:
        breaches.append(f"De = {De:.3g} (envelope: <~ 0.3)")
    if breaches:
        warnings.warn(
            "VALIDITY ENVELOPE EXCEEDED: " + "; ".join(breaches) + ". "
            "The scalarised normal-stress fluxes and the quasi-local "
            "squeeze-memory closure are derived for Couette-dominated "
            "kinematics (paper Table 7); results outside the envelope are "
            "unreliable and are NOT silently corrected (the pre-2026-06 "
            "Gamma_sq clamp that zeroed normal stresses in pure squeeze "
            "has been removed; see verification/EVIDENCE.md).",
            stacklevel=3)


def solve_viscoelastic(
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions,
    fluid: GiesekusFluid,
    config: Optional[SolverConfig] = None
) -> SolverResult:
    """
    Solve the full viscoelastic Reynolds equation.
    
    Includes:
        - Shear-thinning viscosity η(γ̇)
        - Memory operator (squeeze-film elasticity)
        - Normal stress contributions (N₁, hoop)
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    geometry : SlipperGeometry
        Bearing geometry
    conditions : OperatingConditions
        Operating parameters
    fluid : GiesekusFluid
        Giesekus fluid model
    config : SolverConfig, optional
        Solver configuration
    
    Returns
    -------
    result : SolverResult
        Complete solution with flux decomposition
    """
    if config is None:
        config = SolverConfig()
    
    # Handle Newtonian case
    if fluid.is_newtonian:
        return solve_newtonian(mesh, geometry, conditions, fluid.eta_0, config)
    
    R, THETA = mesh.R, mesh.THETA
    h = geometry.gap_height(R, THETA)
    V_Tr, V_Ttheta = conditions.radial_tangential_velocity(THETA)
    
    # Initialize viscosity and shear rate
    eta_bar = fluid.eta_0 * np.ones_like(h)
    gdot_bar = np.abs(V_Tr) / h  # Initial estimate
    
    # Initialize pressure
    p_inner = conditions.p_supply if hasattr(conditions, 'p_supply') else 0.0
    p_outer = conditions.p_ambient if hasattr(conditions, 'p_ambient') else 0.0
    frac = (R - geometry.R_in) / (geometry.R_out - geometry.R_in)
    pressure = p_inner + (p_outer - p_inner) * frac
    
    residual_history = []
    converged = False
    
    for iteration in range(config.max_iter):
        # Store previous pressure
        p_old = pressure.copy()

        # -----------------------------------------------------------------
        # Normal stress ramp schedule
        # Phase 1 (iter 0 .. ns_warmup-1): ns_scale = 0  (GNF + memory only)
        # Phase 2 (iter ns_warmup .. ns_warmup+ns_ramp-1): linear 0 → 1
        # Phase 3 (iter >= ns_warmup+ns_ramp): ns_scale = 1
        # -----------------------------------------------------------------
        if not (config.include_normal_stress or config.include_alpha_coupling):
            ns_scale = 0.0
        elif iteration < config.ns_warmup:
            ns_scale = 0.0
        elif iteration < config.ns_warmup + config.ns_ramp:
            ns_scale = (iteration - config.ns_warmup + 1) / config.ns_ramp
        else:
            ns_scale = 1.0

        # Assemble viscoelastic system
        system = assemble_viscoelastic_system(
            mesh, geometry, conditions, fluid,
            eta_bar, gdot_bar,
            pressure=pressure if iteration > 0 else None,
            include_memory=config.include_memory,
            include_normal_stress=config.include_normal_stress,
            include_alpha_coupling=config.include_alpha_coupling,
            memory_scale=(config.memory_R_fn(fluid.lambda_ * gdot_bar)
                          if config.memory_R_fn is not None else None),
            ns_scale=ns_scale
        )

        # Solve
        p_1d = solve_linear_system(system.A, system.b, config.linear_solver)
        p_new = mesh.to_2d(p_1d)

        # Check for NaN / Inf in solution – bail out early
        if not np.all(np.isfinite(p_new)):
            warnings.warn(
                f"Viscoelastic solver produced NaN/Inf at iteration {iteration+1}"
            )
            break

        # Under-relaxation
        pressure = config.omega * p_new + (1 - config.omega) * p_old

        # Update shear rate (including spin in azimuthal velocity)
        dp_dr, dp_dtheta = mesh.compute_field_gradient(pressure)
        gdot_bar = compute_gap_averaged_shear_rate(
            h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, R, eta_bar,
            omega_s=conditions.omega_s
        )

        # Update viscosity
        eta_bar_new = fluid.viscosity(gdot_bar)
        eta_bar = config.omega * eta_bar_new + (1 - config.omega) * eta_bar

        # Check convergence
        residual = np.linalg.norm(pressure - p_old) / (np.linalg.norm(p_old) + 1e-15)
        residual_history.append(residual)

        if config.verbose:
            ns_str = f", ns_scale = {ns_scale:.2f}" if config.include_normal_stress else ""
            print(f"  Iter {iteration+1:3d}: residual = {residual:.3e}, "
                  f"\u03b7_min/\u03b7_0 = {eta_bar.min()/fluid.eta_0:.2f}{ns_str}")

        if residual < config.tol:
            converged = True
            break

    # ------------------------------------------------------------------
    # Validity guard (replaces the silent Gamma_sq clamp removed 2026-06,
    # see verification/e1_clamp_evidence.py). The scalarised normal-stress
    # fluxes and the quasi-local memory closure are constructed for
    # Couette-dominated kinematics (Table 7 of the paper: P << 1,
    # De_sq <~ 0.1, De <~ 0.3). Out-of-envelope use is NOT silently
    # corrected: it produces this prominent warning instead.
    # ------------------------------------------------------------------
    _warn_validity_envelope(mesh, geometry, conditions, fluid,
                            pressure, eta_bar, h, V_Tr, V_Ttheta)

    # Compute flux decomposition
    flux_calc = FluxCalculator(fluid, geometry, conditions, mesh)
    fluxes = flux_calc.compute_all_fluxes(pressure, eta_bar, include_viscoelastic=True)
    
    flux_decomp = FluxDecomposition(
        Q_r_GNF=fluxes['Q_r_GNF'],
        Q_r_mem=fluxes['Q_r_mem'],
        Q_r_alpha=fluxes['Q_r_alpha'],
        Q_r_N1=fluxes['Q_r_N1'],
        Q_r_hoop=fluxes['Q_r_hoop'],
        Q_theta_GNF=fluxes['Q_theta_GNF'],
        Q_theta_mem=fluxes['Q_theta_mem'],
    )
    
    # Also solve GNF-only for comparison
    config_gnf = SolverConfig(
        max_iter=config.max_iter,
        tol=config.tol,
        omega=config.omega,
        verbose=False
    )
    result_gnf = solve_GNF(mesh, geometry, conditions, fluid, config_gnf)
    
    # Create result
    result = SolverResult(
        pressure=pressure,
        converged=converged,
        iterations=iteration + 1,
        residual_history=residual_history,
        fluxes=flux_decomp,
        eta_bar=eta_bar,
        gdot_bar=gdot_bar,
        solver_info={
            'type': 'viscoelastic',
            'include_memory': config.include_memory,
            'include_normal_stress': config.include_normal_stress,
        }
    )
    
    # Post-process
    result.F_total = compute_load_capacity(pressure, mesh)
    result.F_GNF = result_gnf.F_total
    result.F_elastic = result.F_total - result.F_GNF
    result.M_x, result.M_y = compute_moments(pressure, mesh)
    
    # Leakage from outer flux (using full flux decomposition)
    result.Q_leak = compute_leakage(flux_decomp.Q_r_total, mesh, 'outer')
    
    if not converged:
        warnings.warn(f"Viscoelastic solver did not converge after {config.max_iter} iterations")
    
    return result


# =============================================================================
# Convenience Wrapper
# =============================================================================

def solve(
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions,
    fluid: GiesekusFluid,
    model: str = 'viscoelastic',
    config: Optional[SolverConfig] = None
) -> SolverResult:
    """
    Unified solver interface.
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    geometry : SlipperGeometry
        Bearing geometry
    conditions : OperatingConditions
        Operating parameters
    fluid : GiesekusFluid
        Fluid model
    model : str
        Model type: 'newtonian', 'GNF', 'viscoelastic'
    config : SolverConfig, optional
        Solver configuration
    
    Returns
    -------
    result : SolverResult
        Solution
    """
    if model == 'newtonian':
        return solve_newtonian(mesh, geometry, conditions, fluid.eta_0, config)
    elif model == 'GNF':
        return solve_GNF(mesh, geometry, conditions, fluid, config)
    elif model == 'viscoelastic':
        return solve_viscoelastic(mesh, geometry, conditions, fluid, config)
    else:
        raise ValueError(f"Unknown model type: {model}")


# =============================================================================
# Dimensionless Number Warnings
# =============================================================================

def check_dimensionless_numbers(
    Wi: float, 
    De_sq: float, 
    fluid: GiesekusFluid = None,
    verbose: bool = True
) -> list:
    """
    Check dimensionless numbers and issue physics warnings.
    
    Parameters
    ----------
    Wi : float
        Weissenberg number λγ̇
    De_sq : float
        Squeeze Deborah number λ|ḣ|/h
    fluid : GiesekusFluid, optional
        Fluid model (for checking α)
    verbose : bool
        Print warnings to stdout
    
    Returns
    -------
    warning_list : list of str
        List of warning messages
        
    Notes
    -----
    High Wi (> 100):
        The Giesekus viscosity function is in its plateau region (η → η_s).
        Spatial viscosity variation will be minimal. This is correct physics,
        not a numerical problem.
    
    Large De_sq (> 0.2):
        The perturbation expansion in De_sq is marginally valid. Truncation
        error may exceed 5%. Consider reducing |ḣ| or using CFD validation.
    """
    warning_list = []
    
    if Wi > 100:
        warning_list.append(
            f"High Wi = {Wi:.0f}: Viscosity in plateau region (η → η_s)"
        )
        warning_list.append(
            "  → Spatial viscosity variation will be minimal"
        )
        warning_list.append(
            "  → This is correct physics, not a numerical problem"
        )
    
    if De_sq > 0.2:
        warning_list.append(
            f"Large De_sq = {De_sq:.3f}: Beyond recommended range (< 0.2)"
        )
        warning_list.append(
            "  → Perturbation truncation error may exceed 5%"
        )
        warning_list.append(
            "  → Consider CFD validation for quantitative predictions"
        )
    
    if De_sq > 1.0:
        warning_list.append(
            f"CAUTION: De_sq = {De_sq:.2f} >> 1: Quasi-steady assumption invalid"
        )
        warning_list.append(
            "  → Transient effects dominate; consider full transient solver"
        )

    if fluid is not None and fluid.alpha > 0.5:
        warning_list.append(
            f"High mobility parameter α = {fluid.alpha:.2f}: "
            "High mobility may cause convergence issues"
        )
    
    if verbose and warning_list:
        for w in warning_list:
            print(f"WARNING: {w}")
    
    return warning_list
