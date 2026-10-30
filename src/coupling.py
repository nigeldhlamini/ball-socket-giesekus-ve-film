"""
Thermal-Rheological Coupling for Viscoelastic Slipper Bearings
===============================================================

Provides an outer coupling loop that wraps the isothermal Reynolds equation
solver with the thermal energy equation, allowing temperature-dependent
fluid properties without modifying the core solver.

Usage
-----
Isothermal (unchanged):
    >>> result = solve_viscoelastic(mesh, geometry, conditions, fluid, config)

Thermal coupling:
    >>> from src.coupling import solve_thermohydrodynamic
    >>> from src.thermal import ThermalProperties
    >>> thermal = ThermalProperties()
    >>> result = solve_thermohydrodynamic(
    ...     mesh, geometry, conditions, fluid, thermal,
    ...     thermal_level=1
    ... )

Architecture:
    - solver.py remains purely isothermal (no changes)
    - ThermallyScaledFluid wraps GiesekusFluid with temperature-dependent
      scaling, transparently providing the same interface via duck typing
    - Level 1: single-pass (isothermal solve -> bulk T -> property ratios)
    - Level 2: iterative (Reynolds <-> energy equation until T converges)

References:
    - Dhlamini, N.C. (2025). PhD Thesis, Chapter 4.
    - Khonsari & Booser (2017). Applied Tribology, 3rd ed.

Author: Nigel C. Dhlamini
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np

from .solver import solve, SolverConfig, SolverResult
from .thermal import (
    ThermalProperties,
    solve_thermal_level1, ThermalLevel1Result,
    solve_thermal_level2, ThermalLevel2Result,
    solve_thermal_level3, ThermalLevel3Result,
    compute_nahme_griffith_z,
)
from .rheology import GiesekusFluid
from .geometry import SlipperGeometry, OperatingConditions
from .mesh import PolarMesh


def _ensure_fields(result, mesh, geometry, conditions, fluid):
    """Ensure eta_bar and gdot_bar are populated on the solver result.

    For Newtonian solves, the solver may leave these as None. We compute
    them from the pressure field so the thermal coupling has the data it needs.
    """
    h = geometry.gap_height(mesh.R, mesh.THETA)

    if result.eta_bar is None:
        eta_0 = fluid.eta_0 if hasattr(fluid, 'eta_0') else getattr(fluid, '_base', fluid).eta_0
        result.eta_bar = eta_0 * np.ones_like(mesh.R)

    if result.gdot_bar is None:
        # Estimate shear rate from Couette: gamma_dot ~ V_T / h
        V_T = conditions.V_T
        result.gdot_bar = np.maximum(V_T / np.maximum(h, 1e-12), 1.0)

    return result


# =============================================================================
# Thermally scaled fluid wrapper
# =============================================================================
class ThermallyScaledFluid:
    """
    Wraps a GiesekusFluid with temperature-dependent PARAMETER scaling (TTS).

    Thermorheological simplicity shifts the material PARAMETERS, not the output:
    eta_s, eta_p -> R_eta * (.) and lambda -> a_T * lambda, with alpha invariant.
    Because every Giesekus material function is a closed form in
    (eta_s, eta_p, alpha, Lambda=lambda*gdot), this induces BOTH a vertical scale
    and a HORIZONTAL shift of the shear-rate argument:

        eta(gdot;T)    = R_eta * eta_ref(a_T*gdot)
        eta_T(gdot;T)  = R_eta * eta_T,ref(a_T*gdot)
        M(gdot;T)      = R_eta * M_ref(a_T*gdot)        (memory viscosity)
        Psi1(gdot;T)   = R_eta*a_T * Psi1_ref(a_T*gdot)
        Psi2(gdot;T)   = R_eta*a_T * Psi2_ref(a_T*gdot)

    The previous implementation multiplied the reference functions by a ratio
    WITHOUT shifting their argument (output scaling). That (a) missed thermal
    de-thinning (heating lowers Lambda -> less shear-thinned at fixed gdot) and,
    after the 2026-06 memory correction, (b) let the solver's new
    memory_viscosity(gdot) call fall through __getattr__ to the BASE fluid at the
    unscaled reference Lambda -- silently computing memory at the wrong shear and
    without R_eta. Parameter scaling via the identities above is exact under TTS,
    covers every material function (incl. memory_viscosity and any future one),
    and makes the solver's validity-envelope monitoring use the local lambda(T).
    See thermal-review handoff A2/A3/I1.

    Implementation note: rather than a 2D bilateral-filter over field-valued
    ratios, we apply the identities directly -- evaluate the base closed form at
    the shifted argument a_T*gdot and apply the vertical R_eta (and a_T for the
    Psi's). This is algebraically identical to constructing a per-node scaled
    GiesekusFluid and works with scalar or 2D-field ratios.

    Parameters
    ----------
    base_fluid : GiesekusFluid
        Reference (isothermal) fluid.
    viscosity_ratio : float or ndarray
        R_eta = eta(T)/eta(T_ref), the Arrhenius vertical factor; scalar or 2D
        field matching the mesh.
    relaxation_ratio : float or ndarray
        a_T = lambda(T)/lambda(T_ref), the (WLF/Rouse) horizontal shift factor.
    """

    def __init__(self, base_fluid: GiesekusFluid,
                 viscosity_ratio: Union[float, np.ndarray] = 1.0,
                 relaxation_ratio: Union[float, np.ndarray] = 1.0):
        self._base = base_fluid
        self._visc_ratio = viscosity_ratio      # R_eta
        self._relax_ratio = relaxation_ratio    # a_T

    def __getattr__(self, name):
        """Delegate non-material-function attributes to the base fluid. Material
        functions are overridden below; this catches things like alpha, lambda_,
        eta_s, eta_p, is_newtonian-helpers, etc. It must NOT be reached for any
        gdot-dependent material function (those are all defined explicitly so the
        TTS shift is always applied)."""
        return getattr(self._base, name)

    @property
    def eta_0(self) -> float:
        """Effective zero-shear viscosity (mean R_eta-scaled for initialisation).
        Zero-shear so the a_T shift is immaterial (Lambda->0)."""
        mean_ratio = float(np.mean(self._visc_ratio))
        return self._base.eta_0 * mean_ratio

    @property
    def is_newtonian(self) -> bool:
        return self._base.is_newtonian

    # --- material functions: vertical R_eta x horizontal a_T shift (TTS) ---
    def viscosity(self, gdot):
        """eta(gdot;T) = R_eta * eta_ref(a_T * gdot)."""
        return self._visc_ratio * self._base.viscosity(self._relax_ratio * gdot)

    def tangent_viscosity(self, gdot):
        """eta_T(gdot;T) = R_eta * eta_T,ref(a_T * gdot)."""
        return self._visc_ratio * self._base.tangent_viscosity(self._relax_ratio * gdot)

    def memory_viscosity(self, gdot):
        """M(gdot;T) = R_eta * M_ref(a_T * gdot). Explicit override so the
        solver's memory term never falls through to the unscaled base method."""
        return self._visc_ratio * self._base.memory_viscosity(self._relax_ratio * gdot)

    def Psi1(self, gdot):
        """Psi1(gdot;T) = R_eta*a_T * Psi1_ref(a_T * gdot)."""
        return (self._visc_ratio * self._relax_ratio
                * self._base.Psi1(self._relax_ratio * gdot))

    def Psi2(self, gdot):
        """Psi2(gdot;T) = R_eta*a_T * Psi2_ref(a_T * gdot)."""
        return (self._visc_ratio * self._relax_ratio
                * self._base.Psi2(self._relax_ratio * gdot))


# =============================================================================
# Result dataclass
# =============================================================================
@dataclass
class ThermohydrodynamicResult:
    """
    Combined result from coupled thermohydrodynamic solve.

    Attributes
    ----------
    isothermal : SolverResult
        Final isothermal solver result (with thermal corrections applied)
    thermal_level1 : ThermalLevel1Result or None
        Level 1 result (if thermal_level == 1)
    thermal_level2 : ThermalLevel2Result or None
        Level 2 result (if thermal_level == 2)
    thermal_level : int
        Thermal model level used (1 or 2)
    outer_iterations : int
        Number of thermal-Reynolds coupling iterations
    converged : bool
        Overall convergence status
    residual_history : list of float
        Temperature residuals per outer iteration
    """
    isothermal: SolverResult
    thermal_level1: Optional[ThermalLevel1Result] = None
    thermal_level2: Optional[ThermalLevel2Result] = None
    thermal_level3: Optional[ThermalLevel3Result] = None
    thermal_level: int = 1
    outer_iterations: int = 1
    converged: bool = True
    residual_history: List[float] = field(default_factory=list)

    @property
    def pressure(self) -> np.ndarray:
        """Pressure field from the isothermal solve."""
        return self.isothermal.pressure

    @property
    def F_total(self) -> float:
        """Total load capacity [N]."""
        return self.isothermal.F_total

    @property
    def T_field(self) -> Optional[np.ndarray]:
        """Temperature field (Level 2 or 3)."""
        if self.thermal_level3 is not None:
            return self.thermal_level3.T_field
        if self.thermal_level2 is not None:
            return self.thermal_level2.T_field
        return None

    @property
    def T_bulk(self) -> Optional[float]:
        """Bulk temperature [K] (Level 1 only)."""
        if self.thermal_level1 is not None:
            return self.thermal_level1.T_bulk
        return None

    @property
    def a1_field(self) -> Optional[np.ndarray]:
        """Cross-film amplitude field (Level 3 only)."""
        if self.thermal_level3 is not None:
            return self.thermal_level3.a1_field
        return None

    def summary(self) -> str:
        """Combined summary of isothermal and thermal results."""
        lines = [self.isothermal.summary()]
        lines.append(f"\nThermal coupling: Level {self.thermal_level}, "
                     f"{self.outer_iterations} outer iteration(s), "
                     f"converged={self.converged}")
        if self.thermal_level1 is not None:
            lines.append(self.thermal_level1.summary())
        if self.thermal_level2 is not None:
            lines.append(self.thermal_level2.summary())
        if self.thermal_level3 is not None:
            lines.append(self.thermal_level3.summary())
        return "\n".join(lines)


# =============================================================================
# Main coupling function
# =============================================================================
def solve_thermohydrodynamic(
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions,
    fluid: GiesekusFluid,
    thermal: ThermalProperties,
    thermal_level: int = 1,
    config: Optional[SolverConfig] = None,
    max_therm_iter: int = 20,
    therm_tol: float = 1e-3,
    therm_omega: float = 0.5,
    model: str = 'viscoelastic',
    verbose: bool = False,
) -> ThermohydrodynamicResult:
    """
    Solve the coupled thermohydrodynamic problem.

    Wraps the isothermal solver with thermal energy equation coupling.
    The isothermal solver (solver.py) is not modified.

    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    geometry : SlipperGeometry
        Bearing geometry
    conditions : OperatingConditions
        Operating parameters
    fluid : GiesekusFluid
        Fluid model (reference/isothermal properties)
    thermal : ThermalProperties
        Thermal property model
    thermal_level : int
        1 = quasi-isothermal (single-pass bulk T),
        2 = gap-averaged 2D energy equation (iterative)
    config : SolverConfig, optional
        Isothermal solver configuration
    max_therm_iter : int
        Maximum outer thermal-Reynolds coupling iterations (Level 2)
    therm_tol : float
        Temperature field convergence tolerance (Level 2)
    therm_omega : float
        Under-relaxation factor for thermal coupling (Level 2)
    model : str
        Isothermal solver model: 'newtonian', 'GNF', 'viscoelastic'
    verbose : bool
        Print progress information

    Returns
    -------
    ThermohydrodynamicResult
        Combined isothermal + thermal result
    """
    if thermal_level == 1:
        return _solve_level1_coupling(
            mesh, geometry, conditions, fluid, thermal,
            config, model, verbose
        )
    elif thermal_level == 2:
        return _solve_level2_coupling(
            mesh, geometry, conditions, fluid, thermal,
            config, max_therm_iter, therm_tol, therm_omega,
            model, verbose
        )
    elif thermal_level == 3:
        return _solve_level3_coupling(
            mesh, geometry, conditions, fluid, thermal,
            config, max_therm_iter, therm_tol, therm_omega,
            model, verbose
        )
    else:
        raise ValueError(f"thermal_level must be 1, 2, or 3, got {thermal_level}")


# =============================================================================
# Level 1: Single-pass coupling
# =============================================================================
def _solve_level1_coupling(
    mesh, geometry, conditions, fluid, thermal,
    config, model, verbose
) -> ThermohydrodynamicResult:
    """
    Level 1: Solve isothermal, then compute bulk temperature post-hoc.

    No iteration needed — the temperature is computed from the energy
    balance after the isothermal solve completes.
    """
    if verbose:
        print("Level 1 thermal coupling: isothermal solve + bulk T...")

    # Step 1: Solve isothermal
    result = solve(mesh, geometry, conditions, fluid, model, config)
    result = _ensure_fields(result, mesh, geometry, conditions, fluid)

    # Step 2: Compute gap height field
    h = geometry.gap_height(mesh.R, mesh.THETA)

    # Step 3: Compute bulk temperature
    thermal_result = solve_thermal_level1(
        eta_bar=result.eta_bar,
        gdot_bar=result.gdot_bar,
        h=h,
        Q_leakage=result.Q_leak,
        thermal=thermal,
        r=mesh.R,
        dr=mesh.dr,
        dtheta=mesh.dtheta,
    )

    if verbose:
        print(thermal_result.summary())

    # Step 4: re-solve the load with the bulk-temperature-scaled viscosity so the
    # Level-1 load REFLECTS T_bulk. Single pass -- T_bulk is computed once from the
    # isothermal dissipation (hence over-predicted, never re-iterated), so Level 1
    # stays the conservative bracket: temperature over-predicted => load
    # under-predicted. (Previously the isothermal load was returned, ignoring T_bulk.)
    # NOTE (copy-only change, flag for production): Level 1 is the AR/BO screening
    # model; this changes its load. Do not propagate to the prod checkout without
    # the author's sign-off.
    scaled_fluid = ThermallyScaledFluid(
        fluid, thermal_result.viscosity_ratio, thermal_result.relaxation_ratio)
    load_result = solve(mesh, geometry, conditions, scaled_fluid, model, config)
    load_result = _ensure_fields(load_result, mesh, geometry, conditions, scaled_fluid)

    return ThermohydrodynamicResult(
        isothermal=load_result,
        thermal_level1=thermal_result,
        thermal_level=1,
        outer_iterations=1,
        converged=True,
    )


# =============================================================================
# Level 2: Iterative coupling
# =============================================================================
def _solve_level2_coupling(
    mesh, geometry, conditions, fluid, thermal,
    config, max_therm_iter, therm_tol, therm_omega,
    model, verbose
) -> ThermohydrodynamicResult:
    """
    Level 2: Iterate between Reynolds and energy equation.

    Outer loop:
        1. Solve Reynolds with current (thermally-scaled) fluid
        2. Solve 2D energy equation for T(r, theta)
        3. Update viscosity and relaxation time ratios
        4. Check temperature convergence
    """
    if verbose:
        print("Level 2 thermal coupling: iterative Reynolds-energy solve...")

    h = geometry.gap_height(mesh.R, mesh.THETA)
    Nr, Ntheta = mesh.Nr, mesh.Ntheta

    # Tangential velocity components
    V_Tr = conditions.V_T * np.sin(mesh.THETA)
    V_Ttheta = conditions.V_T * np.cos(mesh.THETA)

    # Initialise ratio fields
    visc_ratio = np.ones((Nr, Ntheta))
    relax_ratio = np.ones((Nr, Ntheta))
    T_prev = thermal.T_ref * np.ones((Nr, Ntheta))

    residual_history = []
    converged = False
    thermal_result_2 = None

    for outer_iter in range(1, max_therm_iter + 1):
        if verbose:
            print(f"\n  Outer iteration {outer_iter}/{max_therm_iter}")

        # Step 1: Create thermally-scaled fluid
        scaled_fluid = ThermallyScaledFluid(fluid, visc_ratio, relax_ratio)

        # Step 2: Solve isothermal with scaled fluid
        result = solve(mesh, geometry, conditions, scaled_fluid, model, config)
        result = _ensure_fields(result, mesh, geometry, conditions, scaled_fluid)

        # Step 3: Solve 2D energy equation
        thermal_result_2 = solve_thermal_level2(
            eta_bar=result.eta_bar,
            gdot_bar=result.gdot_bar,
            h=h,
            pressure=result.pressure,
            r=mesh.R,
            dr=mesh.dr,
            dtheta=mesh.dtheta,
            V_Tr=V_Tr,
            V_Ttheta=V_Ttheta,
            thermal=thermal,
            max_iter=50,
            tol=1e-4,
            omega=0.5,
            verbose=False,
        )

        # Step 4: Update ratio fields with under-relaxation
        new_visc_ratio = thermal_result_2.eta_ratio_field
        new_relax_ratio = thermal_result_2.lambda_ratio_field

        visc_ratio = (therm_omega * new_visc_ratio
                      + (1 - therm_omega) * visc_ratio)
        relax_ratio = (therm_omega * new_relax_ratio
                       + (1 - therm_omega) * relax_ratio)

        # Step 5: Check temperature convergence
        T_new = thermal_result_2.T_field
        residual = (np.linalg.norm(T_new - T_prev)
                    / (np.linalg.norm(T_prev) + 1e-15))
        residual_history.append(residual)
        T_prev = T_new.copy()

        if verbose:
            print(f"    T_max = {thermal_result_2.T_max_celsius:.1f} C, "
                  f"residual = {residual:.3e}")

        if residual < therm_tol:
            converged = True
            if verbose:
                print(f"  Converged after {outer_iter} outer iterations.")
            break

    if not converged and verbose:
        print(f"  WARNING: Not converged after {max_therm_iter} iterations. "
              f"Final residual: {residual_history[-1]:.3e}")

    return ThermohydrodynamicResult(
        isothermal=result,
        thermal_level2=thermal_result_2,
        thermal_level=2,
        outer_iterations=outer_iter,
        converged=converged,
        residual_history=residual_history,
    )


# =============================================================================
# Level 3: Iterative coupling with cross-film resolution
# =============================================================================
def _solve_level3_coupling(
    mesh, geometry, conditions, fluid, thermal,
    config, max_therm_iter, therm_tol, therm_omega,
    model, verbose
) -> ThermohydrodynamicResult:
    """
    Level 3: Iterate between Reynolds and cross-film resolved energy equation.

    Same outer loop structure as Level 2, but the energy solve produces both
    the gap-averaged temperature T(r,theta) and the amplitude field a_1(r,theta).
    The viscosity ratio is computed from the harmonic-mean effective viscosity
    which accounts for cross-film temperature variation.
    """
    if verbose:
        print("Level 3 thermal coupling: iterative Reynolds-energy (cross-film)...")

    h = geometry.gap_height(mesh.R, mesh.THETA)
    Nr, Ntheta = mesh.Nr, mesh.Ntheta

    V_Tr = conditions.V_T * np.sin(mesh.THETA)
    V_Ttheta = conditions.V_T * np.cos(mesh.THETA)

    visc_ratio = np.ones((Nr, Ntheta))
    relax_ratio = np.ones((Nr, Ntheta))
    T_prev = thermal.T_ref * np.ones((Nr, Ntheta))

    residual_history = []
    converged = False
    thermal_result_3 = None

    for outer_iter in range(1, max_therm_iter + 1):
        if verbose:
            print(f"\n  Outer iteration {outer_iter}/{max_therm_iter}")

        # Step 1: Create thermally-scaled fluid
        scaled_fluid = ThermallyScaledFluid(fluid, visc_ratio, relax_ratio)

        # Step 2: Solve isothermal with scaled fluid
        result = solve(mesh, geometry, conditions, scaled_fluid, model, config)
        result = _ensure_fields(result, mesh, geometry, conditions, scaled_fluid)

        # Step 3: Solve cross-film resolved energy equation
        thermal_result_3 = solve_thermal_level3(
            eta_bar=result.eta_bar,
            gdot_bar=result.gdot_bar,
            h=h,
            pressure=result.pressure,
            r=mesh.R,
            dr=mesh.dr,
            dtheta=mesh.dtheta,
            V_Tr=V_Tr,
            V_Ttheta=V_Ttheta,
            thermal=thermal,
            max_iter=50,
            tol=1e-4,
            omega=0.5,
            verbose=False,
        )

        # Step 4: Update ratio fields with under-relaxation
        # Level 3 uses harmonic-mean effective viscosity ratio
        new_visc_ratio = thermal_result_3.eta_ratio_field
        new_relax_ratio = thermal_result_3.lambda_ratio_field

        visc_ratio = (therm_omega * new_visc_ratio
                      + (1 - therm_omega) * visc_ratio)
        relax_ratio = (therm_omega * new_relax_ratio
                       + (1 - therm_omega) * relax_ratio)

        # Step 5: Check temperature convergence
        T_new = thermal_result_3.T_field
        residual = (np.linalg.norm(T_new - T_prev)
                    / (np.linalg.norm(T_prev) + 1e-15))
        residual_history.append(residual)
        T_prev = T_new.copy()

        if verbose:
            print(f"    T_max = {thermal_result_3.T_max_celsius:.1f} C, "
                  f"T_centre = {thermal_result_3.T_centreline_max_celsius:.1f} C, "
                  f"residual = {residual:.3e}")

        if residual < therm_tol:
            converged = True
            if verbose:
                print(f"  Converged after {outer_iter} outer iterations.")
            break

    if not converged and verbose:
        print(f"  WARNING: Not converged after {max_therm_iter} iterations. "
              f"Final residual: {residual_history[-1]:.3e}")

    return ThermohydrodynamicResult(
        isothermal=result,
        thermal_level3=thermal_result_3,
        thermal_level=3,
        outer_iterations=outer_iter,
        converged=converged,
        residual_history=residual_history,
    )
