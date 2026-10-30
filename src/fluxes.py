"""
Flux Components for Viscoelastic Reynolds Equation
===================================================

Implements the v6 four-term volume flux decomposition:

    Q_r = Q_r^GNF + Q_r^mem + Q_r^N₁ + Q_r^hoop

Each component captures a distinct physical mechanism:
    - GNF: Generalised Newtonian (Couette + Poiseuille with η(γ̇))
    - mem: Squeeze-film memory (elastic response to ∂h/∂t); the Giesekus
           quadratic coupling enters HERE through M(α,Λ) per v6.
    - N₁: First normal stress gradient
    - hoop: Hoop stress contribution

NOTE (2026-06-28): the former standalone Giesekus-coupling flux Q_r^α has been
REMOVED from the total. Per v6 the α coupling is folded into the memory
viscosity M(α,Λ) (already inside Q_r^mem); the separately-written Q_r^α term
was dimensionally inconsistent (evaluated to m/s², not m²/s, and carried no
De_sq factor) and, summed into Q_r_total, blew up Q_leak by ~8 orders at high
shear gradient — corrupting the Level-1 bulk-temperature balance. The load
path (assembly.py) already had include_alpha_coupling=False, so loads/h_eq are
unaffected; this fix brings the post-processing flux path into agreement.

References:
    - Dhlamini, N.C. (2025). PhD Thesis, Chapter 4, Eq. 63.
    - Tichy, J.A. (1996). J. Tribol. 118, 344-348.

Author: Nigel C. Dhlamini
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from .rheology import GiesekusFluid, SOFFluid
from .geometry import SlipperGeometry, OperatingConditions
from .mesh import PolarMesh


# =============================================================================
# Gap-Averaged Shear Rate
# =============================================================================

def compute_gap_averaged_shear_rate(
    h: np.ndarray,
    V_Tr: np.ndarray,
    V_Ttheta: np.ndarray,
    dp_dr: np.ndarray,
    dp_dtheta: np.ndarray,
    r: np.ndarray,
    eta_bar: np.ndarray,
    omega_s: float = 0.0
) -> np.ndarray:
    """
    Compute gap-averaged shear rate from Eq. 50.

    γ̄̇² = (V_Tr² + V_Tθ_total²)/h² + (h²/12η̄²)[(∂p/∂r)² + (1/r²)(∂p/∂θ)²]

    where V_Tθ_total = V_Tθ + ω_s·r includes slipper spin.

    The first term is the Couette contribution (wall velocity).
    The second term is the Poiseuille contribution (pressure-driven).

    Parameters
    ----------
    h : ndarray
        Gap height field [m]
    V_Tr : ndarray
        Radial tangential velocity [m/s]
    V_Ttheta : ndarray
        Azimuthal tangential velocity [m/s]
    dp_dr : ndarray
        Radial pressure gradient [Pa/m]
    dp_dtheta : ndarray
        Azimuthal pressure gradient [Pa/rad]
    r : ndarray
        Radial coordinate [m]
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s]
    omega_s : float, optional
        Slipper spin angular velocity [rad/s]. Default: 0.

    Returns
    -------
    gdot_bar : ndarray
        Gap-averaged shear rate [s⁻¹]
    """
    # Couette contribution: total wall velocity / gap height
    # Include spin in azimuthal velocity (Eq. 50)
    V_Ttheta_total = V_Ttheta + omega_s * r
    V_T_sq = V_Tr**2 + V_Ttheta_total**2
    couette_term = V_T_sq / h**2
    
    # Poiseuille contribution: pressure-driven parabolic flow
    # Factor 1/12 comes from integration of parabolic profile
    grad_p_sq = dp_dr**2 + (dp_dtheta / r)**2
    poiseuille_term = (h**2 / (12 * eta_bar**2)) * grad_p_sq
    
    # Total gap-averaged shear rate
    gdot_bar_sq = couette_term + poiseuille_term
    gdot_bar = np.sqrt(np.maximum(gdot_bar_sq, 1e-20))  # Avoid sqrt(0)
    
    return gdot_bar


def compute_Gamma_squared(
    h: np.ndarray,
    V_T: np.ndarray,
    dp_dr: np.ndarray,
    dp_dtheta: np.ndarray,
    r: np.ndarray,
    eta_bar: np.ndarray
) -> np.ndarray:
    """
    Compute Γ̄² for normal stress contributions (Eq. 64).
    
    Γ̄² = V_T²/h² + (h²/12η̄²)|∇p|²
    
    This is similar to γ̄̇² but uses total tangential velocity magnitude.
    
    Parameters
    ----------
    h : ndarray
        Gap height [m]
    V_T : ndarray
        Total tangential velocity magnitude [m/s]
    dp_dr, dp_dtheta : ndarray
        Pressure gradients [Pa/m], [Pa/rad]
    r : ndarray
        Radial coordinate [m]
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s]
    
    Returns
    -------
    Gamma_sq : ndarray
        Γ̄² [s⁻²]
    """
    grad_p_sq = dp_dr**2 + (dp_dtheta / r)**2
    Gamma_sq = (V_T**2 / h**2) + (h**2 / (12 * eta_bar**2)) * grad_p_sq
    
    return Gamma_sq


# =============================================================================
# Individual Flux Components
# =============================================================================

def compute_Q_GNF(
    h: np.ndarray,
    V_Tr: np.ndarray,
    V_Ttheta: np.ndarray,
    dp_dr: np.ndarray,
    dp_dtheta: np.ndarray,
    r: np.ndarray,
    eta_bar: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalised Newtonian Fluid flux (Couette + Poiseuille).
    
    From Eq. 63:
        Q_r^GNF = (h/2)V_Tr - (h³/12η̄)(∂p/∂r)
        Q_θ^GNF = (h/2)V_Tθ - (h³/12η̄)(1/r)(∂p/∂θ)
    
    Parameters
    ----------
    h : ndarray
        Gap height [m]
    V_Tr, V_Ttheta : ndarray
        Tangential velocity components [m/s]
    dp_dr, dp_dtheta : ndarray
        Pressure gradients
    r : ndarray
        Radial coordinate [m]
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s]
    
    Returns
    -------
    Q_r_GNF, Q_theta_GNF : ndarray
        Flux components [m²/s]
    """
    # Couette flux: drag by moving wall
    Q_r_couette = (h / 2) * V_Tr
    Q_theta_couette = (h / 2) * V_Ttheta
    
    # Poiseuille flux: pressure-driven
    coeff = h**3 / (12 * eta_bar)
    Q_r_poiseuille = -coeff * dp_dr
    Q_theta_poiseuille = -coeff * (dp_dtheta / r)
    
    Q_r_GNF = Q_r_couette + Q_r_poiseuille
    Q_theta_GNF = Q_theta_couette + Q_theta_poiseuille
    
    return Q_r_GNF, Q_theta_GNF


def compute_Q_memory(
    h: np.ndarray,
    h_dot: float,
    dp_dr: np.ndarray,
    dp_dtheta: np.ndarray,
    r: np.ndarray,
    eta_bar: np.ndarray,
    memory_visc: np.ndarray,
    lambda_: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute squeeze-film memory flux correction.

    Q_r^mem = +λh²ḣ/(12η̄²) × M(α,Λ) × (∂p/∂r)

    Coefficient and sign from the exact antisymmetric memory kernel
    (λ/η̄)∫₀ʰ(h/2 − z)·∂τ_p,rz⁽⁰⁾/∂t dz with the γ̇ ∝ 1/h closure; derivation
    in verification/e0_kernel_exact.py (replaces the unsupported 1/16 form).
    The viscosity factor is the derived memory viscosity M(α,Λ) =
    GiesekusFluid.memory_viscosity (verification/p1_M_derivation.py): only
    the polymer stress relaxes, and its response passes through the
    linearised constitutive operator (NOT the total tangent viscosity η_T
    used before 2026-06). During gap closure (ḣ < 0) the coefficient is
    negative, the same sign as the Poiseuille coefficient -h³/(12η̄):
    squeeze memory augments the pressure-driven flux and reduces the load.

    This term arises from the upper-convected derivative in the
    constitutive equation. When the gap closes (ḣ < 0), the fluid
    "remembers" the previous weaker shear rate, modifying flux.
    
    Parameters
    ----------
    h : ndarray
        Gap height [m]
    h_dot : float
        Squeeze velocity ∂h/∂t [m/s]. Negative = closing.
    dp_dr, dp_dtheta : ndarray
        Pressure gradients
    r : ndarray
        Radial coordinate [m]
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s]
    memory_visc : ndarray
        Memory viscosity M(α,Λ) from GiesekusFluid.memory_viscosity [Pa·s]
    lambda_ : float
        Relaxation time [s]

    Returns
    -------
    Q_r_mem, Q_theta_mem : ndarray
        Memory flux components [m²/s]
    
    Notes
    -----
    Physics (consistent with Phan-Thien & Tanner 1983):
    
    The memory operator L_mem adds to the GNF diffusion operator L_GN,
    increasing the effective "diffusivity" of the pressure equation.
    For a fixed squeeze rate (fixed source term), stronger diffusion 
    produces LOWER pressure gradients → LOAD REDUCTION.
    
    This is the correct physics for Giesekus/Maxwell/Oldroyd-B fluids.
    Load ENHANCEMENT requires stress overshoot (Modified PTT model).

    Derivation and references: verification/e0_kernel_exact.py and
    verification/EVIDENCE.md.
    """
    # Exact-kernel memory coefficient (verification/e0_kernel_exact.py):
    # Q_r^mem = +λh²ḣ/(12η̄²) × M(α,Λ) × ∂p/∂r
    coeff = (lambda_ * h**2 * h_dot) / (12 * eta_bar**2) * memory_visc
    
    Q_r_mem = coeff * dp_dr
    Q_theta_mem = coeff * (dp_dtheta / r)
    
    return Q_r_mem, Q_theta_mem


def compute_Q_alpha(
    h: np.ndarray,
    gdot_bar: np.ndarray,
    d_gdot_dr: np.ndarray,
    eta_bar: np.ndarray,
    Psi1_bar: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute Giesekus coupling flux (often negligible).
    
    From Eq. 63:
        Q_r^α = -(αh²/4η̄) Ψ̄₁ γ̄̇² (∂γ̄̇/∂r)
    
    This term arises from the nonlinear τ·τ coupling in the
    Giesekus constitutive equation.
    
    Parameters
    ----------
    h : ndarray
        Gap height [m]
    gdot_bar : ndarray
        Gap-averaged shear rate [s⁻¹]
    d_gdot_dr : ndarray
        Radial gradient of shear rate [s⁻¹/m]
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s]
    Psi1_bar : ndarray
        First normal stress coefficient at γ̄̇ [Pa·s²]
    alpha : float
        Giesekus mobility parameter
    
    Returns
    -------
    Q_r_alpha : ndarray
        Giesekus coupling flux [m²/s]
    
    Notes
    -----
    DEPRECATED (2026-06-28). This standalone Giesekus-coupling flux is
    dimensionally inconsistent (the written form evaluates to m/s², not m²/s,
    and carries no De_sq factor) and is NOT part of the v6 four-term
    decomposition: the α coupling is folded into the memory viscosity M(α,Λ)
    inside Q_r^mem. Summing it into Q_r_total blew up Q_leak by ~8 orders at
    high shear gradient. The function is retained only to make the removal
    explicit and now raises rather than silently returning a broken term.
    """
    raise NotImplementedError(
        "compute_Q_alpha is deprecated and removed from the v6 flux "
        "decomposition: the Giesekus coupling is folded into M(alpha,Lambda) "
        "(Q_r_mem). The standalone term was dimensionally inconsistent and "
        "corrupted Q_leak. Do not use."
    )


def compute_Q_N1(
    h: np.ndarray,
    Gamma_sq: np.ndarray,
    d_hPsiGamma_dr: np.ndarray,
    eta_bar: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute first normal stress gradient flux.
    
    From Eq. 63:
        Q_r^N₁ = -(h²(1-α/2) / 6η̄) ∂/∂r(h Ψ̄₁ Γ̄²)
    
    The N₁ contribution generates pressure gradients from
    spatial variations in normal stress.
    
    Parameters
    ----------
    h : ndarray
        Gap height [m]
    Gamma_sq : ndarray
        Gap-averaged squared shear rate Γ̄² [s⁻²]
    d_hPsiGamma_dr : ndarray
        ∂/∂r(h Ψ̄₁ Γ̄²) [Pa/m]
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s]
    alpha : float
        Giesekus mobility parameter
    
    Returns
    -------
    Q_r_N1 : ndarray
        Normal stress gradient flux [m²/s]
    """
    coeff = -(h**2 * (1 - alpha/2)) / (6 * eta_bar)
    Q_r_N1 = coeff * d_hPsiGamma_dr
    
    return Q_r_N1


def compute_Q_hoop(
    h: np.ndarray,
    r: np.ndarray,
    Gamma_sq: np.ndarray,
    Psi1_bar: np.ndarray,
    eta_bar: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute hoop stress flux contribution.
    
    From Eq. 63 (coefficient corrected, WP-N revision1):
        Q_r^hoop = (h³(1−α/2) Ψ̄₁ Γ̄²) / (12η̄ r)

    The hoop stress term (τ_rr − τ_θθ) creates an effective body force in the
    radial direction.  With the correct bookkeeping (neutral τ_θθ = 0;
    N₁ = τ_rr − τ_zz, N₂ = τ_zz − τ_θθ), the hoop stress is
        τ_rr − τ_θθ = N₁ + N₂ = (1 + Ψ₂/Ψ₁) N₁ → (1 − α/2) N₁  (low shear),
    i.e. the SAME (1 − α/2) factor as the N₁-gradient term.  The manuscript's
    (1 + α/2) arose from a sign-flipped N₂ (τ_θθ − τ_rr) and is corrected here.
    
    Parameters
    ----------
    h : ndarray
        Gap height [m]
    r : ndarray
        Radial coordinate [m]
    Gamma_sq : ndarray
        Γ̄² [s⁻²]
    Psi1_bar : ndarray
        Ψ̄₁ at local shear rate [Pa·s²]
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s]
    alpha : float
        Giesekus mobility parameter
    
    Returns
    -------
    Q_r_hoop : ndarray
        Hoop stress flux [m²/s]
    """
    coeff = (h**3 * (1 - alpha/2)) / (12 * eta_bar * r)   # WP-N: (1+a/2)->(1-a/2)
    Q_r_hoop = coeff * Psi1_bar * Gamma_sq
    
    return Q_r_hoop


# =============================================================================
# Source Term
# =============================================================================

def compute_source_term(
    h: np.ndarray,
    h_dot: float,
    V_Tr: np.ndarray,
    V_Ttheta: np.ndarray,
    dh_dr: np.ndarray,
    dh_dtheta: np.ndarray,
    r: np.ndarray,
    omega_s: float = 0.0
) -> np.ndarray:
    """
    Compute the source term (RHS) of Reynolds equation.
    
    From Eq. 71:
        S = ḣ - (1/2r)∂(rhV_Tr)/∂r - (1/2r)∂[h(V_Tθ + ω_s r)]/∂θ
    
    Parameters
    ----------
    h : ndarray
        Gap height [m]
    h_dot : float
        Squeeze velocity [m/s]
    V_Tr, V_Ttheta : ndarray
        Tangential velocity components [m/s]
    dh_dr, dh_dtheta : ndarray
        Gap height gradients
    r : ndarray
        Radial coordinate [m]
    omega_s : float, optional
        Slipper spin angular velocity [rad/s]
    
    Returns
    -------
    S : ndarray
        Source term [m/s]
    """
    # Squeeze term
    squeeze = h_dot * np.ones_like(h)

    # Total azimuthal velocity including spin
    V_theta_total = V_Ttheta + omega_s * r

    # Wedge terms from Eq. 71:
    #   S = ḣ - (V_Tr/2)∂h/∂r - (V_Tθ_total/(2r))∂h/∂θ
    S = squeeze - (V_Tr / 2) * dh_dr - (V_theta_total / (2*r)) * dh_dtheta
    
    return S


# =============================================================================
# Complete Flux Calculator
# =============================================================================

@dataclass
class FluxCalculator:
    """
    Calculator for all flux components in the viscoelastic Reynolds equation.
    
    Computes the decomposition:
        Q_r = Q_r^GNF + Q_r^mem + Q_r^α + Q_r^N₁ + Q_r^hoop
    
    Attributes
    ----------
    fluid : GiesekusFluid or SOFFluid
        Rheological model
    geometry : SlipperGeometry
        Bearing geometry
    conditions : OperatingConditions
        Operating parameters
    mesh : PolarMesh
        Computational mesh
    """
    fluid: GiesekusFluid
    geometry: SlipperGeometry
    conditions: OperatingConditions
    mesh: PolarMesh
    
    def compute_all_fluxes(
        self,
        pressure: np.ndarray,
        eta_bar: Optional[np.ndarray] = None,
        include_viscoelastic: bool = True
    ) -> dict:
        """
        Compute all flux components from pressure field.
        
        Parameters
        ----------
        pressure : ndarray
            Pressure field p(r,θ), shape (Nr, Ntheta) [Pa]
        eta_bar : ndarray, optional
            Gap-averaged viscosity. If None, computed from shear rate.
        include_viscoelastic : bool
            If False, only compute GNF flux (for comparison)
        
        Returns
        -------
        dict
            Dictionary containing all flux components and auxiliary fields
        """
        # Extract mesh coordinates
        R, THETA = self.mesh.R, self.mesh.THETA
        h = self.geometry.gap_height(R, THETA)
        _, dh_dr, dh_dtheta = self.geometry.gap_height_derivatives(R, THETA)
        
        # Operating conditions
        V_Tr, V_Ttheta = self.conditions.radial_tangential_velocity(THETA)
        h_dot = self.conditions.h_dot
        omega_s = self.conditions.omega_s
        
        # Compute pressure gradients
        dp_dr, dp_dtheta = self.mesh.compute_field_gradient(pressure)
        
        # Initial viscosity estimate if not provided
        if eta_bar is None:
            eta_bar = self.fluid.eta_0 * np.ones_like(h)
        
        # Compute gap-averaged shear rate (including spin in azimuthal velocity)
        gdot_bar = compute_gap_averaged_shear_rate(
            h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, R, eta_bar,
            omega_s=omega_s
        )

        # Update viscosity from shear rate
        eta_bar = self.fluid.viscosity(gdot_bar)

        # Total tangential velocity for Γ² (includes spin)
        V_Ttheta_total = V_Ttheta + omega_s * R
        V_T = np.sqrt(V_Tr**2 + V_Ttheta_total**2)
        
        # =====================================================================
        # GNF Flux (always computed)
        # =====================================================================
        Q_r_GNF, Q_theta_GNF = compute_Q_GNF(
            h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, R, eta_bar
        )
        
        # Initialize viscoelastic terms to zero
        Q_r_mem = np.zeros_like(h)
        Q_theta_mem = np.zeros_like(h)
        Q_r_alpha = np.zeros_like(h)
        Q_r_N1 = np.zeros_like(h)
        Q_r_hoop = np.zeros_like(h)
        
        if include_viscoelastic and not self.fluid.is_newtonian:
            # =================================================================
            # Memory Flux (memory viscosity M(alpha, Lambda), not eta_T:
            # verification/p1_M_derivation.py)
            # =================================================================
            M_mem = self.fluid.memory_viscosity(gdot_bar)
            Q_r_mem, Q_theta_mem = compute_Q_memory(
                h, h_dot, dp_dr, dp_dtheta, R,
                eta_bar, M_mem, self.fluid.lambda_
            )
            
            # =================================================================
            # Normal Stress Coefficients
            # =================================================================
            Psi1_bar = self.fluid.Psi1(gdot_bar)
            alpha = self.fluid.alpha
            
            # =================================================================
            # Giesekus Coupling (Q_r^α) -- DEPRECATED / REMOVED (2026-06-28)
            # =================================================================
            # Per v6 the Giesekus quadratic coupling is folded into M(α,Λ) and
            # already contained in Q_r_mem. The standalone Q_r^α term was
            # dimensionally inconsistent and is NO LONGER computed or summed;
            # Q_r_alpha stays the zero array initialised above (kept only as a
            # deprecated diagnostic key in the returned dict).

            # =================================================================
            # Normal Stress Gradient (Q_r^N₁)
            # =================================================================
            Gamma_sq = compute_Gamma_squared(
                h, V_T, dp_dr, dp_dtheta, R, eta_bar
            )
            
            # Compute ∂/∂r(h Ψ̄₁ Γ̄²)
            h_Psi_Gamma = h * Psi1_bar * Gamma_sq
            d_hPsiGamma_dr, _ = self.mesh.compute_field_gradient(h_Psi_Gamma)
            
            Q_r_N1 = compute_Q_N1(h, Gamma_sq, d_hPsiGamma_dr, eta_bar, alpha)
            
            # =================================================================
            # Hoop Stress (Q_r^hoop)
            # =================================================================
            Q_r_hoop = compute_Q_hoop(h, R, Gamma_sq, Psi1_bar, eta_bar, alpha)
        
        # =====================================================================
        # Source Term
        # =====================================================================
        source = compute_source_term(
            h, h_dot, V_Tr, V_Ttheta, dh_dr, dh_dtheta, R, omega_s
        )
        
        # =====================================================================
        # Total Flux
        # =====================================================================
        # v6 four-term total (Q_r_alpha REMOVED 2026-06-28; folded into M, see
        # module header). Q_r_alpha stays zeros and is returned as a deprecated
        # diagnostic key only.
        Q_r_total = Q_r_GNF + Q_r_mem + Q_r_N1 + Q_r_hoop
        Q_theta_total = Q_theta_GNF + Q_theta_mem  # θ-viscoelastic terms small
        
        return {
            # Primary outputs
            'Q_r_total': Q_r_total,
            'Q_theta_total': Q_theta_total,
            'source': source,
            
            # Flux decomposition
            'Q_r_GNF': Q_r_GNF,
            'Q_r_mem': Q_r_mem,
            'Q_r_alpha': Q_r_alpha,
            'Q_r_N1': Q_r_N1,
            'Q_r_hoop': Q_r_hoop,
            
            'Q_theta_GNF': Q_theta_GNF,
            'Q_theta_mem': Q_theta_mem,
            
            # Auxiliary fields
            'eta_bar': eta_bar,
            'gdot_bar': gdot_bar,
            'h': h,
            'dp_dr': dp_dr,
            'dp_dtheta': dp_dtheta,
        }
    
    def compute_flux_ratios(self, fluxes: dict) -> dict:
        """
        Compute relative magnitude of each flux component.
        
        Parameters
        ----------
        fluxes : dict
            Output from compute_all_fluxes()
        
        Returns
        -------
        dict
            Ratios of each component to total (L2 norm)
        """
        Q_total_norm = np.linalg.norm(fluxes['Q_r_total'])
        
        if Q_total_norm < 1e-15:
            return {name: 0.0 for name in 
                    ['GNF', 'memory', 'alpha', 'N1', 'hoop']}
        
        return {
            'GNF': np.linalg.norm(fluxes['Q_r_GNF']) / Q_total_norm,
            'memory': np.linalg.norm(fluxes['Q_r_mem']) / Q_total_norm,
            'alpha': np.linalg.norm(fluxes['Q_r_alpha']) / Q_total_norm,
            'N1': np.linalg.norm(fluxes['Q_r_N1']) / Q_total_norm,
            'hoop': np.linalg.norm(fluxes['Q_r_hoop']) / Q_total_norm,
        }


# =============================================================================
# Newtonian Specialization (for validation)
# =============================================================================

def compute_newtonian_fluxes(
    h: np.ndarray,
    V_Tr: np.ndarray,
    V_Ttheta: np.ndarray,
    dp_dr: np.ndarray,
    dp_dtheta: np.ndarray,
    r: np.ndarray,
    eta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Newtonian flux (constant viscosity).
    
    This is the special case of GNF with η = const.
    Used for validation against analytical solutions.
    
    Parameters
    ----------
    h : ndarray
        Gap height [m]
    V_Tr, V_Ttheta : ndarray
        Tangential velocities [m/s]
    dp_dr, dp_dtheta : ndarray
        Pressure gradients
    r : ndarray
        Radial coordinate [m]
    eta : float
        Constant viscosity [Pa·s]
    
    Returns
    -------
    Q_r, Q_theta : ndarray
        Newtonian flux components [m²/s]
    """
    eta_array = eta * np.ones_like(h)
    return compute_Q_GNF(h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, r, eta_array)
