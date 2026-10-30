"""
Thermal Extensions for Giesekus-Based Reynolds Equation
========================================================

Implements three levels of thermal coupling for the viscoelastic
slipper bearing lubrication model:

    Level 1 (Quasi-isothermal): Bulk temperature from energy balance
    Level 2 (Gap-averaged): 2D energy equation on polar mesh
    Level 3 (z-resolved): Cross-film temperature profile

Temperature-dependent properties:
    η(T) = η(T_ref) * exp[β_η * (1/T - 1/T_ref)]       (Arrhenius)
    λ(T) = λ(T_ref) * a_T(T)                              (TTS)
    a_T(T) = exp[-C1*(T-T_ref) / (C2 + T - T_ref)]       (WLF)

Theory:
    The gap-averaged energy equation for a thin-film bearing is:

        ρ c_p (ū_r ∂T/∂r + ū_θ/r ∂T/∂θ) = Φ̄_v/h + k_f(T_top + T_bot - 2T)/(h/2)²

    where Φ̄_v ≈ η̄ γ̇² h is the gap-integrated viscous dissipation function.

    For viscoelastic fluids, the dissipation includes both viscous and elastic
    contributions. At leading order, the viscous dissipation dominates:

        Φ_v = η(γ̇) γ̇²

    The elastic dissipation (due to polymer stress relaxation) enters at O(De_sq)
    and is proportional to λ τ:τ / (2η_p), but is typically 1-2 orders of magnitude
    smaller than viscous dissipation for De_sq < 0.2.

References:
    - Khonsari, M.M. & Booser, E.R. (2017). Applied Tribology, 3rd ed., Wiley.
    - Hamrock, B.J. et al. (2004). Fundamentals of Fluid Film Lubrication, 2nd ed.
    - Bair, S. (2019). High Pressure Rheology for Quantitative Elastohydrodynamics.
    - Dhlamini, N.C. (2025). PhD Thesis, Chapter 4, University of Cape Town.

Author: Nigel C. Dhlamini
Affiliation: University of Cape Town, Department of Mechanical Engineering
Contact: dhlnig001@myuct.ac.za | https://github.com/nigeldhlamini
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable
import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


# =============================================================================
# Temperature-Dependent Properties
# =============================================================================

@dataclass
class ThermalProperties:
    """
    Temperature-dependent fluid and thermal properties.

    The Arrhenius model for viscosity:
        η(T) = η_ref * exp[β_η * (1/T - 1/T_ref)]

    where β_η = E_a / R_gas is the activation energy ratio.
    Typical values for polymer-thickened hydraulic fluids: β_η ~ 2000-4000 K.

    The WLF (Williams-Landel-Ferry) equation for time-temperature superposition:
        log(a_T) = -C1 * (T - T_ref) / (C2 + T - T_ref)

    Typical values: C1 ~ 8.86, C2 ~ 101.6 K (universal constants).

    Parameters
    ----------
    T_ref : float
        Reference temperature [K]. Default: 313.15 K (40°C).
    beta_eta : float
        Arrhenius activation parameter E_a/R [K]. Default: 2500 K.
    C1_WLF : float
        WLF constant C1 [-]. Default: 8.86.
    C2_WLF : float
        WLF constant C2 [K]. Default: 101.6 K.
    rho : float
        Fluid density [kg/m³]. Default: 1000 (water-based).
    c_p : float
        Specific heat capacity [J/(kg·K)]. Default: 4000.
    k_f : float
        Thermal conductivity of fluid [W/(m·K)]. Default: 0.6.
    T_top : float
        Temperature of top surface (slipper) [K]. Default: 313.15 K.
    T_bot : float
        Temperature of bottom surface (swash plate) [K]. Default: 313.15 K.
    T_max_failure : float
        Maximum allowable temperature before failure [K]. Default: 383.15 K (110°C).
    """
    T_ref: float = 313.15          # 40°C in Kelvin
    beta_eta: float = 2500.0       # Arrhenius parameter [K]
    C1_WLF: float = 8.86           # WLF constant
    C2_WLF: float = 101.6          # WLF constant [K]
    rho: float = 1000.0            # Density [kg/m³]
    c_p: float = 4000.0            # Specific heat [J/(kg·K)]
    k_f: float = 0.60              # Thermal conductivity [W/(m·K)]
    T_top: float = 313.15          # Top surface temperature [K]
    T_bot: float = 313.15          # Bottom surface temperature [K]
    T_max_failure: float = 383.15  # Failure temperature (110°C) [K]
    # Time-temperature-superposition (TTS) model for the relaxation-time shift
    # factor a_T = lambda(T)/lambda(T_ref). The thermal-review I3 finding: this
    # is the single most consequential thermal modelling choice -- WLF-universal
    # and Arrhenius/Rouse differ by ~2 orders of magnitude at +40 K, and a_T
    # drives both the De_sq suppression and the memory-revival pathways.
    #   'arrhenius_rouse'  : a_T = R_eta * T_ref/T (lambda ~ eta_s/T). DEFAULT --
    #                        defensible for polymer SOLUTIONS far above Tg.
    #   'wlf_universal'    : log10 a_T = -C1 dT/(C2+dT) with universal C1,C2
    #                        (valid only at T_ref ~ Tg+50 K; sensitivity slice).
    #   'wlf_calibrated'   : same WLF form with caller-supplied C1_WLF, C2_WLF.
    tts_model: str = "arrhenius_rouse"

    def viscosity_ratio(self, T: np.ndarray) -> np.ndarray:
        """
        Compute η(T) / η(T_ref) using Arrhenius model.

        η(T) / η(T_ref) = exp[β_η * (1/T - 1/T_ref)]

        Parameters
        ----------
        T : ndarray
            Temperature field [K]

        Returns
        -------
        ratio : ndarray
            Viscosity scaling factor η(T)/η(T_ref)
        """
        return np.exp(self.beta_eta * (1.0 / T - 1.0 / self.T_ref))

    def relaxation_time_ratio(self, T: np.ndarray) -> np.ndarray:
        """
        Compute the TTS shift factor a_T = λ(T)/λ(T_ref) under the configured
        ``tts_model`` (thermal-review I3).

        - 'arrhenius_rouse' (DEFAULT): Rouse/Zimm scaling λ ∝ η_s/T gives
          a_T = R_η · T_ref/T, where R_η = exp[β_η(1/T − 1/T_ref)] is the
          Arrhenius viscosity ratio. Defensible for polymer solutions far above
          Tg; this is the campaign default.
        - 'wlf_universal' / 'wlf_calibrated': log10 a_T = −C1 dT/(C2 + dT) with
          the universal (8.86, 101.6 K) or caller-supplied (C1_WLF, C2_WLF)
          constants. Universal constants are only valid at T_ref ≈ Tg + 50 K;
          use as a sensitivity slice, not the default.

        For T > T_ref, a_T < 1 (faster relaxation = shorter memory).

        Parameters
        ----------
        T : ndarray
            Temperature field [K]

        Returns
        -------
        a_T : ndarray
            Shift factor λ(T)/λ(T_ref)
        """
        model = getattr(self, "tts_model", "arrhenius_rouse")
        if model == "arrhenius_rouse":
            # a_T = R_eta * T_ref/T  (lambda ~ eta_s/T)
            return self.viscosity_ratio(T) * (self.T_ref / T)
        if model in ("wlf_universal", "wlf_calibrated"):
            dT = T - self.T_ref
            # Protect against C2 + dT approaching zero
            denominator = np.maximum(self.C2_WLF + dT, 1.0)
            log10_aT = -self.C1_WLF * dT / denominator
            return np.power(10.0, log10_aT)
        raise ValueError(
            f"unknown tts_model {model!r}; expected one of "
            f"'arrhenius_rouse', 'wlf_universal', 'wlf_calibrated'")

    def nusselt_coefficient(self, h: np.ndarray) -> np.ndarray:
        """
        Effective heat transfer coefficient for cross-film conduction.

        For a parabolic temperature profile across the gap, the effective
        Nusselt number is Nu = 12 (Hamrock et al., 2004, Ch. 7).

        Returns k_eff = Nu * k_f / h² = 12 * k_f / h²

        Parameters
        ----------
        h : ndarray
            Film thickness [m]

        Returns
        -------
        k_eff : ndarray
            Effective heat transfer coefficient [W/(m²·K)]
        """
        return 12.0 * self.k_f / (h**2)


# =============================================================================
# Level 1: Quasi-Isothermal Model
# =============================================================================

@dataclass
class ThermalLevel1Result:
    """Result container for Level 1 (quasi-isothermal) thermal model."""
    T_bulk: float                    # Bulk temperature [K]
    T_bulk_celsius: float            # Bulk temperature [°C]
    delta_T: float                   # Temperature rise [K]
    viscosity_ratio: float           # η(T_bulk)/η(T_ref)
    relaxation_ratio: float          # λ(T_bulk)/λ(T_ref)
    P_dissipation: float             # Total viscous dissipation [W]
    Q_leakage: float                 # Leakage flow rate [m³/s]
    thermal_failure: bool            # True if T_bulk > T_max

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "Level 1: Quasi-Isothermal Thermal Model",
            "=" * 50,
            f"  Bulk temperature   : {self.T_bulk_celsius:.1f} °C ({self.T_bulk:.2f} K)",
            f"  Temperature rise   : {self.delta_T:.1f} K",
            f"  Viscosity ratio    : η(T)/η₀ = {self.viscosity_ratio:.4f}",
            f"  Relaxation ratio   : λ(T)/λ₀ = {self.relaxation_ratio:.4f}",
            f"  Dissipation power  : {self.P_dissipation:.2f} W",
            f"  Leakage flow       : {self.Q_leakage * 1e9:.2f} mm³/s",
            f"  Thermal failure    : {'YES ⚠' if self.thermal_failure else 'No ✓'}",
            "=" * 50,
        ]
        return "\n".join(lines)


def solve_thermal_level1(
    eta_bar: np.ndarray,
    gdot_bar: np.ndarray,
    h: np.ndarray,
    Q_leakage: float,
    thermal: ThermalProperties,
    mesh_areas: Optional[np.ndarray] = None,
    r: Optional[np.ndarray] = None,
    dr: Optional[float] = None,
    dtheta: Optional[float] = None,
) -> ThermalLevel1Result:
    """
    Level 1 thermal model: bulk temperature from global energy balance.

    The energy balance for the lubricant passing through the bearing:
        ρ c_p Q_leak ΔT = P_diss

    where P_diss = ∫∫ η̄ γ̇² h r dr dθ is the total viscous dissipation.

    This gives:
        T_bulk = T_in + P_diss / (ρ c_p Q_leak)

    This is the simplest thermal model and provides a conservative upper
    bound on temperature (assumes all dissipation heats the through-flow).

    Parameters
    ----------
    eta_bar : ndarray
        Gap-averaged viscosity field [Pa·s], shape (Nr, Ntheta)
    gdot_bar : ndarray
        Gap-averaged shear rate field [1/s], shape (Nr, Ntheta)
    h : ndarray
        Film thickness field [m], shape (Nr, Ntheta)
    Q_leakage : float
        Volume flow rate through bearing [m³/s]
    thermal : ThermalProperties
        Thermal property model
    mesh_areas : ndarray, optional
        Cell areas r·dr·dθ [m²], shape (Nr, Ntheta)
    r : ndarray, optional
        Radial coordinate [m], shape (Nr, Ntheta)
    dr : float, optional
        Radial spacing [m]
    dtheta : float, optional
        Angular spacing [rad]

    Returns
    -------
    result : ThermalLevel1Result
    """
    # Compute local dissipation: Φ = η γ̇² (per unit volume)
    Phi_local = eta_bar * gdot_bar**2  # [W/m³]

    # Integrate over the film: P_diss = ∫∫ Φ h r dr dθ
    if mesh_areas is not None:
        P_diss = np.sum(Phi_local * h * mesh_areas)
    elif r is not None and dr is not None and dtheta is not None:
        P_diss = np.sum(Phi_local * h * r * dr * dtheta)
    else:
        raise ValueError("Must provide either mesh_areas or (r, dr, dtheta)")

    # Energy balance: ΔT = P_diss / (ρ c_p Q_leak)
    # Guard against zero leakage (would give infinite temperature)
    Q_eff = max(abs(Q_leakage), 1e-12)
    delta_T = P_diss / (thermal.rho * thermal.c_p * Q_eff)

    # Bulk temperature (assume inlet = T_ref)
    T_bulk = thermal.T_ref + delta_T

    # Property ratios at bulk temperature
    visc_ratio = float(thermal.viscosity_ratio(np.array([T_bulk]))[0])
    relax_ratio = float(thermal.relaxation_time_ratio(np.array([T_bulk]))[0])

    return ThermalLevel1Result(
        T_bulk=T_bulk,
        T_bulk_celsius=T_bulk - 273.15,
        delta_T=delta_T,
        viscosity_ratio=visc_ratio,
        relaxation_ratio=relax_ratio,
        P_dissipation=P_diss,
        Q_leakage=Q_leakage,
        thermal_failure=(T_bulk > thermal.T_max_failure),
    )


# =============================================================================
# Level 2: Gap-Averaged 2D Energy Equation
# =============================================================================

@dataclass
class ThermalLevel2Result:
    """Result container for Level 2 (gap-averaged) thermal model."""
    T_field: np.ndarray              # Temperature field T(r,θ) [K]
    T_max: float                     # Maximum temperature [K]
    T_min: float                     # Minimum temperature [K]
    T_max_celsius: float             # Maximum temperature [°C]
    eta_ratio_field: np.ndarray      # η(T)/η(T_ref) field
    lambda_ratio_field: np.ndarray   # λ(T)/λ(T_ref) field
    P_dissipation: float             # Total dissipation [W]
    iterations: int                  # Number of thermal iterations
    converged: bool                  # Convergence flag
    thermal_failure: bool            # True if T_max > T_max_failure

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "Level 2: Gap-Averaged Thermal Model",
            "=" * 50,
            f"  Temperature range  : {self.T_min - 273.15:.1f} – {self.T_max_celsius:.1f} °C",
            f"  Temperature rise   : {self.T_max - self.T_min:.1f} K",
            f"  η(T)/η₀ range      : {self.eta_ratio_field.min():.4f} – {self.eta_ratio_field.max():.4f}",
            f"  λ(T)/λ₀ range      : {self.lambda_ratio_field.min():.4f} – {self.lambda_ratio_field.max():.4f}",
            f"  Total dissipation  : {self.P_dissipation:.2f} W",
            f"  Converged          : {'Yes' if self.converged else 'No'} ({self.iterations} iters)",
            f"  Thermal failure    : {'YES ⚠' if self.thermal_failure else 'No ✓'}",
            "=" * 50,
        ]
        return "\n".join(lines)


def assemble_energy_equation(
    u_r: np.ndarray,
    u_theta: np.ndarray,
    h: np.ndarray,
    eta_bar: np.ndarray,
    gdot_bar: np.ndarray,
    r: np.ndarray,
    dr: float,
    dtheta: float,
    thermal: ThermalProperties,
    Nr: int,
    Ntheta: int,
) -> Tuple[csr_matrix, np.ndarray]:
    """
    Assemble the gap-averaged energy equation using FVM.

    The governing equation is:
        ρ c_p (ū_r ∂T/∂r + ū_θ/r ∂T/∂θ) = Φ̄_v/h + k_eff*(T_wall - T)

    where k_eff = 12 k_f / h² accounts for cross-film conduction to both
    surfaces (assuming parabolic profile, Nu = 12).

    Discretization: upwind differencing for convection, central for source.

    Boundary conditions:
        r = R_in:  T = T_inlet (Dirichlet, supply temperature)
        r = R_out: ∂T/∂r = 0 (Neumann, adiabatic exit)
        θ: periodic

    Parameters
    ----------
    u_r : ndarray
        Gap-averaged radial velocity [m/s], shape (Nr, Ntheta)
    u_theta : ndarray
        Gap-averaged tangential velocity [m/s], shape (Nr, Ntheta)
    h : ndarray
        Film thickness [m], shape (Nr, Ntheta)
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s], shape (Nr, Ntheta)
    gdot_bar : ndarray
        Gap-averaged shear rate [1/s], shape (Nr, Ntheta)
    r : ndarray
        Radial coordinate [m], shape (Nr, Ntheta)
    dr : float
        Radial spacing [m]
    dtheta : float
        Angular spacing [rad]
    thermal : ThermalProperties
        Thermal model parameters
    Nr, Ntheta : int
        Grid dimensions

    Returns
    -------
    A : csr_matrix
        Coefficient matrix, shape (N, N) where N = Nr * Ntheta
    b : ndarray
        RHS vector, shape (N,)
    """
    N = Nr * Ntheta
    rho_cp = thermal.rho * thermal.c_p

    # Pre-compute coefficients
    Phi_v = eta_bar * gdot_bar**2  # Viscous dissipation [W/m³]
    k_eff = thermal.nusselt_coefficient(h)  # [W/(m²·K)]
    T_wall = 0.5 * (thermal.T_top + thermal.T_bot)  # Mean wall temperature

    # Index mapping: (i, j) -> k = i * Ntheta + j
    def idx(i, j):
        return i * Ntheta + (j % Ntheta)

    # Build sparse matrix using COO format
    rows, cols, vals = [], [], []
    b = np.zeros(N)

    for i in range(Nr):
        for j in range(Ntheta):
            k = idx(i, j)
            r_ij = r[i, j]

            # Inner boundary: Dirichlet T = T_inlet
            if i == 0:
                rows.append(k)
                cols.append(k)
                vals.append(1.0)
                b[k] = thermal.T_ref  # Inlet temperature
                continue

            # Convection coefficients (upwind scheme)
            ur = u_r[i, j]
            uth = u_theta[i, j]

            # Radial convection: ρ c_p u_r ∂T/∂r
            conv_r = rho_cp * ur / dr

            # Azimuthal convection: ρ c_p u_θ/(r) ∂T/∂θ
            conv_theta = rho_cp * uth / (r_ij * dtheta)

            # Cross-film heat loss
            loss = k_eff[i, j]

            # Diagonal coefficient
            diag = loss  # Heat loss term (always positive)

            # Radial upwind
            if ur >= 0:
                # Forward flow: T_i depends on T_{i-1}
                diag += conv_r
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(-conv_r)
            else:
                # Reverse flow: T_i depends on T_{i+1}
                diag += -conv_r
                if i < Nr - 1:
                    rows.append(k); cols.append(idx(i + 1, j)); vals.append(conv_r)
                else:
                    # Neumann BC at outer boundary: T_{Nr} = T_{Nr-1}
                    diag += conv_r  # Add to diagonal

            # Azimuthal upwind (periodic)
            if uth >= 0:
                diag += conv_theta
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(-conv_theta)
            else:
                diag += -conv_theta
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(conv_theta)

            # Outer boundary: Neumann ∂T/∂r = 0
            # Already handled in upwind scheme

            # Diagonal entry
            rows.append(k)
            cols.append(k)
            vals.append(diag)

            # RHS: viscous dissipation + wall heat transfer
            b[k] = Phi_v[i, j] + loss * T_wall

    A = csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A, b


def solve_thermal_level2(
    eta_bar: np.ndarray,
    gdot_bar: np.ndarray,
    h: np.ndarray,
    pressure: np.ndarray,
    r: np.ndarray,
    dr: float,
    dtheta: float,
    V_Tr: np.ndarray,
    V_Ttheta: np.ndarray,
    thermal: ThermalProperties,
    max_iter: int = 50,
    tol: float = 1e-4,
    omega: float = 0.5,
    verbose: bool = True,
) -> ThermalLevel2Result:
    """
    Solve the gap-averaged energy equation iteratively.

    The thermal problem is coupled to the flow through temperature-dependent
    viscosity. The algorithm alternates between:
        1. Solve energy equation for T(r,θ) given η̄(r,θ)
        2. Update η̄(T) using Arrhenius model
        3. Check convergence

    Note: This Level 2 model does NOT re-solve the Reynolds equation with
    updated viscosity. For full coupling, this function should be wrapped
    in an outer iteration that alternates between Reynolds and energy solves.

    Parameters
    ----------
    eta_bar : ndarray
        Initial gap-averaged viscosity field [Pa·s], shape (Nr, Ntheta)
    gdot_bar : ndarray
        Gap-averaged shear rate [1/s], shape (Nr, Ntheta)
    h : ndarray
        Film thickness [m], shape (Nr, Ntheta)
    pressure : ndarray
        Pressure field [Pa], shape (Nr, Ntheta)
    r : ndarray
        Radial coordinate [m], shape (Nr, Ntheta)
    dr, dtheta : float
        Grid spacings
    V_Tr, V_Ttheta : ndarray
        Tangential velocity components [m/s]
    thermal : ThermalProperties
        Thermal model parameters
    max_iter : int
        Maximum thermal iterations
    tol : float
        Convergence tolerance on temperature field
    omega : float
        Under-relaxation factor
    verbose : bool
        Print iteration progress

    Returns
    -------
    result : ThermalLevel2Result
    """
    Nr, Ntheta = eta_bar.shape

    # Gap-averaged velocities
    # Couette: ū_θ = V_Tθ / 2 (linear profile average)
    # Poiseuille: ū_r = -h²/(12η) ∂p/∂r (from Reynolds equation)
    dp_dr = np.gradient(pressure, dr, axis=0)
    u_r = -h**2 / (12 * eta_bar) * dp_dr
    u_theta = V_Ttheta / 2.0

    # Initialize temperature field
    T = thermal.T_ref * np.ones((Nr, Ntheta))

    converged = False
    iteration = 0

    if verbose:
        print("\nLevel 2 Thermal Solver")
        print("-" * 50)

    for iteration in range(max_iter):
        T_old = T.copy()

        # Assemble energy equation
        A, b = assemble_energy_equation(
            u_r, u_theta, h, eta_bar, gdot_bar,
            r, dr, dtheta, thermal, Nr, Ntheta,
        )

        # Solve linear system
        T_1d = spsolve(A, b)
        T_new = T_1d.reshape((Nr, Ntheta))

        # Under-relaxation
        T = omega * T_new + (1 - omega) * T_old

        # Enforce minimum temperature (physical constraint)
        T = np.maximum(T, thermal.T_ref - 10.0)

        # Check convergence
        residual = np.linalg.norm(T - T_old) / (np.linalg.norm(T_old) + 1e-15)

        if verbose:
            print(f"  Iter {iteration + 1:3d}: T_max = {T.max() - 273.15:.1f} °C, "
                  f"residual = {residual:.3e}")

        if residual < tol:
            converged = True
            break

    # Compute property fields at converged temperature
    eta_ratio = thermal.viscosity_ratio(T)
    lambda_ratio = thermal.relaxation_time_ratio(T)

    # Total dissipation
    Phi_v = eta_bar * gdot_bar**2
    P_diss = np.sum(Phi_v * h * r * dr * dtheta)

    return ThermalLevel2Result(
        T_field=T,
        T_max=float(T.max()),
        T_min=float(T.min()),
        T_max_celsius=float(T.max() - 273.15),
        eta_ratio_field=eta_ratio,
        lambda_ratio_field=lambda_ratio,
        P_dissipation=P_diss,
        iterations=iteration + 1,
        converged=converged,
        thermal_failure=(T.max() > thermal.T_max_failure),
    )


# =============================================================================
# Level 3: Cross-Film Resolved Temperature
# =============================================================================

@dataclass
class ThermalLevel3Result:
    """Result container for Level 3 (cross-film resolved) thermal model.

    The Level 3 model resolves the cross-film temperature profile using
    assumed shape functions T(r,theta,z) = T_wall + a_1(r,theta) sin(pi zeta).
    The effective viscosity accounts for the cross-film variation through a
    harmonic mean, which corrects the Poiseuille flow calculation.
    """
    T_field: np.ndarray              # Gap-averaged temperature T(r,theta) [K]
    a1_field: np.ndarray             # Shape function amplitude a_1(r,theta) [K]
    T_max: float                     # Max gap-averaged temperature [K]
    T_max_celsius: float             # Max gap-averaged temperature [C]
    T_centreline_max: float          # Max centreline temperature [K]
    T_centreline_max_celsius: float  # Max centreline temperature [C]
    eta_ratio_field: np.ndarray      # eta(T)/eta(T_ref) field (harmonic mean)
    lambda_ratio_field: np.ndarray   # lambda(T)/lambda(T_ref) field
    P_dissipation: float             # Total dissipation [W]
    Na_z_field: np.ndarray           # Cross-film Nahme-Griffith number
    iterations: int                  # Number of thermal iterations
    converged: bool                  # Convergence flag
    thermal_failure: bool            # True if T_centreline_max > T_max_failure

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "Level 3: Cross-Film Resolved Thermal Model",
            "=" * 50,
            f"  Gap-avg T range    : {self.T_field.min() - 273.15:.1f} - "
            f"{self.T_max_celsius:.1f} C",
            f"  Centreline T_max   : {self.T_centreline_max_celsius:.1f} C",
            f"  Max amplitude a_1  : {self.a1_field.max():.2f} K",
            f"  eta_eff/eta_0 range: {self.eta_ratio_field.min():.4f} - "
            f"{self.eta_ratio_field.max():.4f}",
            f"  lambda/lambda_0    : {self.lambda_ratio_field.min():.4f} - "
            f"{self.lambda_ratio_field.max():.4f}",
            f"  Max Na_z           : {self.Na_z_field.max():.3f}",
            f"  Total dissipation  : {self.P_dissipation:.2f} W",
            f"  Converged          : {'Yes' if self.converged else 'No'} "
            f"({self.iterations} iters)",
            f"  Thermal failure    : "
            f"{'YES' if self.thermal_failure else 'No'}",
            "=" * 50,
        ]
        return "\n".join(lines)


def _galerkin_weights(Nz: int = 16) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Galerkin projection weights for the sine basis phi_1(zeta) = sin(pi*zeta).

    For a Couette+Poiseuille velocity profile, the weighted velocities are:
        u_weighted = int_0^1 u(zeta) phi_1(zeta) dzeta / int_0^1 phi_1^2 dzeta

    Couette profile: u_C(zeta) = zeta  ->  weight_C = int(zeta sin(pi z)) / int(sin^2) = 2/pi / 0.5 = 4/pi
    Poiseuille profile: u_P(zeta) = zeta(1-zeta)  ->  weight_P = 4/(pi^3) / 0.5 = 8/pi^3

    The dissipation weight projects the shear dissipation onto the basis:
        <Phi>_w = int_0^1 Phi(zeta) phi_1(zeta) dzeta / int_0^1 phi_1^2 dzeta

    Returns
    -------
    couette_weight : float
        Velocity weight for Couette component
    poiseuille_weight : float
        Velocity weight for Poiseuille component
    dissipation_weight : float
        Dissipation projection weight
    """
    # Analytical Galerkin integrals for phi_1(z) = sin(pi*z):
    # int_0^1 sin^2(pi*z) dz = 1/2
    norm = 0.5

    # Couette velocity u_C = z:
    # int_0^1 z * sin(pi*z) dz = 1/pi
    couette_weight = (1.0 / np.pi) / norm  # = 2/pi

    # Poiseuille velocity u_P = 6*z*(1-z) (normalized to mean=1):
    # int_0^1 6*z*(1-z) * sin(pi*z) dz = 12/pi^3
    poiseuille_weight = (12.0 / np.pi**3) / norm  # = 24/pi^3

    # Dissipation: For Couette-dominated shear, Phi ~ eta * (du/dz)^2
    # For uniform shear: int_0^1 1 * sin(pi*z) dz = 2/pi
    dissipation_weight = (2.0 / np.pi) / norm  # = 4/pi

    return couette_weight, poiseuille_weight, dissipation_weight


def compute_nahme_griffith_z(
    eta_bar: np.ndarray,
    gdot_bar: np.ndarray,
    h: np.ndarray,
    thermal: ThermalProperties,
) -> np.ndarray:
    """Compute the cross-film Nahme-Griffith number Na_z.

    Na_z = beta_eta * eta_bar * gdot_bar^2 * h^2 / k_f

    When Na_z > 1, cross-film temperature variation is significant
    and Level 3 is recommended over Level 2.

    Parameters
    ----------
    eta_bar : ndarray
        Gap-averaged viscosity [Pa.s]
    gdot_bar : ndarray
        Gap-averaged shear rate [1/s]
    h : ndarray
        Film thickness [m]
    thermal : ThermalProperties
        Thermal model

    Returns
    -------
    Na_z : ndarray
        Cross-film Nahme-Griffith number
    """
    return thermal.beta_eta * eta_bar * gdot_bar**2 * h**2 / thermal.k_f


def _compute_harmonic_mean_viscosity_ratio(
    T_gap_avg: np.ndarray,
    a1: np.ndarray,
    h: np.ndarray,
    thermal: ThermalProperties,
    Nz_quad: int = 16,
) -> np.ndarray:
    """Compute the effective viscosity ratio using harmonic mean across the gap.

    The effective viscosity for Poiseuille flow with z-dependent viscosity:
        eta_eff = [1/h * int_0^h 1/eta(z) dz]^{-1}

    With T(z) = T_wall + a1 * sin(pi * z/h), the viscosity at each z-station
    is eta(z) = eta_ref * exp[beta_eta * (1/T(z) - 1/T_ref)].

    The integral is evaluated by Gauss-Legendre quadrature.

    Parameters
    ----------
    T_gap_avg : ndarray
        Gap-averaged temperature field [K], shape (Nr, Ntheta)
    a1 : ndarray
        Shape function amplitude [K], shape (Nr, Ntheta)
    h : ndarray
        Film thickness [m], shape (Nr, Ntheta)
    thermal : ThermalProperties
        Thermal model
    Nz_quad : int
        Number of quadrature points across the gap

    Returns
    -------
    eta_ratio_eff : ndarray
        Effective viscosity ratio eta_eff/eta_ref, shape (Nr, Ntheta)
    """
    T_wall = 0.5 * (thermal.T_top + thermal.T_bot)

    # Gauss-Legendre quadrature on [0, 1]
    zeta_pts, weights = np.polynomial.legendre.leggauss(Nz_quad)
    # Transform from [-1, 1] to [0, 1]
    zeta_pts = 0.5 * (zeta_pts + 1.0)
    weights = 0.5 * weights

    # Accumulate 1/eta(z) weighted integral
    inv_eta_integral = np.zeros_like(T_gap_avg)
    for zi, wi in zip(zeta_pts, weights):
        phi_z = np.sin(np.pi * zi)
        T_z = T_wall + a1 * phi_z
        # Clamp temperature to physical range
        T_z = np.maximum(T_z, 273.15)
        eta_ratio_z = thermal.viscosity_ratio(T_z)
        inv_eta_integral += wi / eta_ratio_z

    # Harmonic mean: eta_eff_ratio = 1 / (integral of 1/eta_ratio)
    eta_ratio_eff = 1.0 / np.maximum(inv_eta_integral, 1e-15)
    return eta_ratio_eff


def assemble_amplitude_equation(
    u_r: np.ndarray,
    u_theta: np.ndarray,
    h: np.ndarray,
    eta_bar: np.ndarray,
    gdot_bar: np.ndarray,
    r: np.ndarray,
    dr: float,
    dtheta: float,
    thermal: ThermalProperties,
    Nr: int,
    Ntheta: int,
) -> Tuple[csr_matrix, np.ndarray]:
    """Assemble the Galerkin-projected amplitude equation for a_1(r,theta).

    The governing equation (Eq. 6.30 in thesis):
        rho c_p (u_r^(w) da1/dr + u_theta^(w)/r da1/dtheta) = <Phi>_w/h - pi^2 k_f/h^2 a1

    This has the same convection-source structure as the Level 2 energy
    equation but with:
    - Weighted velocities (Galerkin projection of Couette+Poiseuille)
    - Eigenvalue pi^2 instead of Nu=12 for cross-film conduction
    - Weighted dissipation projection as source

    Parameters
    ----------
    u_r, u_theta : ndarray
        Gap-averaged velocities [m/s], shape (Nr, Ntheta)
    h : ndarray
        Film thickness [m], shape (Nr, Ntheta)
    eta_bar : ndarray
        Gap-averaged viscosity [Pa.s], shape (Nr, Ntheta)
    gdot_bar : ndarray
        Gap-averaged shear rate [1/s], shape (Nr, Ntheta)
    r : ndarray
        Radial coordinate [m], shape (Nr, Ntheta)
    dr, dtheta : float
        Grid spacings
    thermal : ThermalProperties
        Thermal model
    Nr, Ntheta : int
        Grid dimensions

    Returns
    -------
    A : csr_matrix
        Coefficient matrix, shape (N, N)
    b : ndarray
        RHS vector, shape (N,)
    """
    N = Nr * Ntheta
    rho_cp = thermal.rho * thermal.c_p

    # Galerkin weights
    w_C, w_P, w_diss = _galerkin_weights()

    # Weighted velocities: u^(w) = w_C * u_Couette + w_P * u_Poiseuille
    # For slipper bearing: u_theta is mostly Couette, u_r is mostly Poiseuille
    u_r_w = w_P * u_r        # Radial is Poiseuille-dominated
    u_theta_w = w_C * u_theta  # Azimuthal is Couette-dominated

    # Dissipation source (weighted projection)
    Phi_v = eta_bar * gdot_bar**2
    Phi_w = w_diss * Phi_v

    # Cross-film conduction eigenvalue: pi^2 k_f / h^2
    k_z = np.pi**2 * thermal.k_f / h**2

    def idx(i, j):
        return i * Ntheta + (j % Ntheta)

    rows, cols, vals = [], [], []
    b = np.zeros(N)

    for i in range(Nr):
        for j in range(Ntheta):
            k = idx(i, j)

            # Inner boundary: a_1 = 0 (inlet at reference temperature)
            if i == 0:
                rows.append(k)
                cols.append(k)
                vals.append(1.0)
                b[k] = 0.0
                continue

            ur_w = u_r_w[i, j]
            uth_w = u_theta_w[i, j]

            conv_r = rho_cp * ur_w / dr
            conv_theta = rho_cp * uth_w / (r[i, j] * dtheta)
            loss = k_z[i, j]

            diag = loss

            # Radial upwind
            if ur_w >= 0:
                diag += conv_r
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(-conv_r)
            else:
                diag += -conv_r
                if i < Nr - 1:
                    rows.append(k); cols.append(idx(i + 1, j)); vals.append(conv_r)
                else:
                    diag += conv_r

            # Azimuthal upwind (periodic)
            if uth_w >= 0:
                diag += conv_theta
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(-conv_theta)
            else:
                diag += -conv_theta
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(conv_theta)

            rows.append(k)
            cols.append(k)
            vals.append(diag)

            # RHS: weighted dissipation source
            b[k] = Phi_w[i, j]

    A = csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A, b


def solve_thermal_level3(
    eta_bar: np.ndarray,
    gdot_bar: np.ndarray,
    h: np.ndarray,
    pressure: np.ndarray,
    r: np.ndarray,
    dr: float,
    dtheta: float,
    V_Tr: np.ndarray,
    V_Ttheta: np.ndarray,
    thermal: ThermalProperties,
    max_iter: int = 50,
    tol: float = 1e-4,
    omega: float = 0.5,
    Nz_quad: int = 16,
    verbose: bool = True,
) -> ThermalLevel3Result:
    """Solve the cross-film resolved thermal problem (Level 3).

    Extends Level 2 by also solving for the shape function amplitude
    a_1(r,theta) which gives the cross-film temperature profile:
        T(r,theta,z) = T_wall + a_1(r,theta) * sin(pi * z/h)

    The effective viscosity is computed as the harmonic mean across the
    gap, which correctly accounts for the Poiseuille flow modification
    due to cross-film viscosity variation.

    The iteration alternates:
        1. Solve Level 2 gap-averaged energy equation for T(r,theta)
        2. Solve Galerkin amplitude equation for a_1(r,theta)
        3. Compute harmonic-mean effective viscosity
        4. Check convergence

    Parameters
    ----------
    eta_bar : ndarray
        Initial gap-averaged viscosity [Pa.s], shape (Nr, Ntheta)
    gdot_bar : ndarray
        Gap-averaged shear rate [1/s], shape (Nr, Ntheta)
    h : ndarray
        Film thickness [m], shape (Nr, Ntheta)
    pressure : ndarray
        Pressure field [Pa], shape (Nr, Ntheta)
    r : ndarray
        Radial coordinate [m], shape (Nr, Ntheta)
    dr, dtheta : float
        Grid spacings
    V_Tr, V_Ttheta : ndarray
        Tangential velocity components [m/s]
    thermal : ThermalProperties
        Thermal model
    max_iter : int
        Maximum thermal iterations
    tol : float
        Convergence tolerance on temperature
    omega : float
        Under-relaxation factor
    Nz_quad : int
        Quadrature points for harmonic mean viscosity integration
    verbose : bool
        Print iteration progress

    Returns
    -------
    result : ThermalLevel3Result
    """
    Nr, Ntheta = eta_bar.shape

    # Gap-averaged velocities
    dp_dr = np.gradient(pressure, dr, axis=0)
    u_r = -h**2 / (12 * eta_bar) * dp_dr
    u_theta = V_Ttheta / 2.0

    T_wall = 0.5 * (thermal.T_top + thermal.T_bot)

    # Initialise
    T = thermal.T_ref * np.ones((Nr, Ntheta))
    a1 = np.zeros((Nr, Ntheta))

    converged = False
    iteration = 0

    if verbose:
        print("\nLevel 3 Thermal Solver (Cross-Film Resolved)")
        print("-" * 50)

    for iteration in range(max_iter):
        T_old = T.copy()
        a1_old = a1.copy()

        # Step 1: Solve gap-averaged energy equation (same as Level 2)
        A_T, b_T = assemble_energy_equation(
            u_r, u_theta, h, eta_bar, gdot_bar,
            r, dr, dtheta, thermal, Nr, Ntheta,
        )
        T_1d = spsolve(A_T, b_T)
        T_new = T_1d.reshape((Nr, Ntheta))

        # Step 2: Solve amplitude equation
        A_a, b_a = assemble_amplitude_equation(
            u_r, u_theta, h, eta_bar, gdot_bar,
            r, dr, dtheta, thermal, Nr, Ntheta,
        )
        a1_1d = spsolve(A_a, b_a)
        a1_new = a1_1d.reshape((Nr, Ntheta))

        # Ensure non-negative amplitude (centreline hotter than walls)
        a1_new = np.maximum(a1_new, 0.0)

        # Under-relaxation
        T = omega * T_new + (1 - omega) * T_old
        a1 = omega * a1_new + (1 - omega) * a1_old

        # Enforce minimum temperature
        T = np.maximum(T, thermal.T_ref - 10.0)

        # Convergence check (combined T and a1)
        res_T = np.linalg.norm(T - T_old) / (np.linalg.norm(T_old) + 1e-15)
        res_a1 = np.linalg.norm(a1 - a1_old) / (np.linalg.norm(a1) + 1.0)
        residual = max(res_T, res_a1)

        if verbose:
            T_centre_max = T_wall + a1.max()
            print(f"  Iter {iteration + 1:3d}: T_max = {T.max() - 273.15:.1f} C, "
                  f"a1_max = {a1.max():.2f} K, "
                  f"T_centre = {T_centre_max - 273.15:.1f} C, "
                  f"res = {residual:.3e}")

        if residual < tol:
            converged = True
            break

    # Compute property fields using harmonic mean viscosity
    eta_ratio = _compute_harmonic_mean_viscosity_ratio(T, a1, h, thermal, Nz_quad)
    lambda_ratio = thermal.relaxation_time_ratio(T)

    # Cross-film Nahme-Griffith number
    Na_z = compute_nahme_griffith_z(eta_bar, gdot_bar, h, thermal)

    # Total dissipation
    Phi_v = eta_bar * gdot_bar**2
    P_diss = np.sum(Phi_v * h * r * dr * dtheta)

    # Centreline temperature
    T_centreline = T_wall + a1
    T_cl_max = float(T_centreline.max())

    return ThermalLevel3Result(
        T_field=T,
        a1_field=a1,
        T_max=float(T.max()),
        T_max_celsius=float(T.max() - 273.15),
        T_centreline_max=T_cl_max,
        T_centreline_max_celsius=T_cl_max - 273.15,
        eta_ratio_field=eta_ratio,
        lambda_ratio_field=lambda_ratio,
        P_dissipation=P_diss,
        Na_z_field=Na_z,
        iterations=iteration + 1,
        converged=converged,
        thermal_failure=(T_cl_max > thermal.T_max_failure),
    )


# =============================================================================
# Demonstration and Validation
# =============================================================================

def demo_thermal_properties():
    """
    Demonstrate temperature-dependent property models.

    Validates against expected physical behaviour:
    - Viscosity decreases with temperature (Arrhenius)
    - Relaxation time decreases with temperature (WLF)
    - Both effects reduce viscoelastic load support
    """
    print("=" * 60)
    print("THERMAL PROPERTY MODEL DEMONSTRATION")
    print("=" * 60)

    thermal = ThermalProperties()
    T_range = np.linspace(300, 400, 11)  # 27°C to 127°C

    print(f"\n{'T [°C]':>8} {'η(T)/η₀':>12} {'λ(T)/λ₀':>12} {'Product':>12}")
    print("-" * 48)

    for T in T_range:
        eta_r = thermal.viscosity_ratio(np.array([T])).item()
        lam_r = thermal.relaxation_time_ratio(np.array([T])).item()
        print(f"  {T - 273.15:5.1f}   {eta_r:10.4f}   {lam_r:10.4f}   {eta_r * lam_r:10.4f}")

    # Physics check: both should decrease with T
    T_hot = np.array([373.15])  # 100°C
    T_cold = np.array([293.15])  # 20°C
    assert thermal.viscosity_ratio(T_hot) < thermal.viscosity_ratio(T_cold), \
        "Viscosity should decrease with temperature"
    assert thermal.relaxation_time_ratio(T_hot) < thermal.relaxation_time_ratio(T_cold), \
        "Relaxation time should decrease with temperature"

    print("\n✓ Physics checks passed: η and λ both decrease with temperature")

    return thermal


def demo_level1():
    """
    Demonstrate Level 1 thermal model with synthetic data.
    """
    print("\n" + "=" * 60)
    print("LEVEL 1: QUASI-ISOTHERMAL MODEL DEMONSTRATION")
    print("=" * 60)

    # Synthetic bearing data (representative of 5% PAM slipper bearing)
    Nr, Ntheta = 40, 60
    R_in, R_out = 0.008, 0.020  # m
    h0 = 5e-6  # m

    r = np.linspace(R_in, R_out, Nr)
    theta = np.linspace(0, 2 * np.pi, Ntheta, endpoint=False)
    R, THETA = np.meshgrid(r, theta, indexing='ij')

    h = h0 * np.ones_like(R)
    dr = (R_out - R_in) / (Nr - 1)
    dtheta = 2 * np.pi / Ntheta

    # Synthetic viscosity and shear rate fields
    eta_0 = 0.15  # Pa·s (5% PAM zero-shear)
    V_T = 10.0  # m/s orbital velocity
    gdot_bar = V_T / h  # Representative shear rate: ~2×10⁶ s⁻¹
    eta_bar = eta_0 * 0.20 * np.ones_like(R)  # Shear-thinned to 20% of η₀

    # Leakage (representative)
    Q_leak = 250e-9  # 250 mm³/s converted to m³/s

    thermal = ThermalProperties()
    result = solve_thermal_level1(
        eta_bar, gdot_bar, h, Q_leak, thermal,
        r=R, dr=dr, dtheta=dtheta,
    )

    print(result.summary())

    # Sensitivity study: effect of supply pressure on thermal state
    print("\nSensitivity: Varying leakage (proxy for supply pressure)")
    print(f"{'Q_leak [mm³/s]':>16} {'ΔT [K]':>10} {'η(T)/η₀':>10} {'Failure?':>10}")
    print("-" * 50)

    for Q_mm3 in [100, 250, 500, 1000, 2000]:
        Q = Q_mm3 * 1e-9
        res = solve_thermal_level1(
            eta_bar, gdot_bar, h, Q, thermal,
            r=R, dr=dr, dtheta=dtheta,
        )
        status = "YES ⚠" if res.thermal_failure else "No"
        print(f"  {Q_mm3:12.0f}   {res.delta_T:8.1f}   {res.viscosity_ratio:8.4f}   {status:>8}")

    return result


if __name__ == "__main__":
    # Run demonstrations
    thermal = demo_thermal_properties()
    result_L1 = demo_level1()

    print("\n" + "=" * 60)
    print("THERMAL MODULE DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Integrate Level 1 into solve_viscoelastic() outer loop")
    print("  2. Validate Level 2 against thermal CFD (ANSYS Fluent)")
    print("  3. Implement Level 3 (cross-film resolved)")
    print("  4. Map thermal failure boundaries in attainable region")
