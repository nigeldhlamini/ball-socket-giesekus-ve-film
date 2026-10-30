"""
Reynolds Equation Assembly
==========================

Builds the sparse matrix system for the viscoelastic Reynolds equation.

The Reynolds equation in operator form (Eq. 66-71):
    L_GN[p] + L_mem[p] + L_α + L_N1[p] + L_hoop = S

This module constructs the discretised form:
    A · p = b

where A is a sparse matrix and b is the right-hand side vector.

References:
    - Dhlamini, N.C. (2025). PhD Thesis, Chapter 4.

Author: Nigel C. Dhlamini
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, diags
from scipy.sparse.linalg import spsolve

from .rheology import GiesekusFluid
from .geometry import SlipperGeometry, OperatingConditions
from .mesh import PolarMesh


# =============================================================================
# Operator Discretisation
# =============================================================================

def build_diffusion_operator(
    mesh: PolarMesh,
    D_coeff: np.ndarray
) -> csr_matrix:
    """
    Build the generalised diffusion operator in polar coordinates.

    Discretises:
        L[p] = (1/r)∂/∂r(r·D·∂p/∂r) + (1/r²)∂/∂θ(D·∂p/∂θ)

    This is the standard form for Reynolds equation with D = h³/(12η).

    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    D_coeff : ndarray
        Diffusion coefficient field, shape (Nr, Ntheta)

    Returns
    -------
    L : csr_matrix
        Sparse diffusion operator, shape (N, N)

    Notes
    -----
    Uses second-order central differences with harmonic averaging of
    diffusion coefficients at cell faces.  Boundary nodes (i = 0 and
    i = Nr-1) use a ghost-node stencil that enforces a Dirichlet value
    at the PHYSICAL boundary r = R_in / R_out to second order.

    For a cell-centred mesh the boundary r = R_in lies exactly
    half-way between the ghost node (i = -1, r = R_in - dr/2) and the
    first interior node (i = 0, r = R_in + dr/2).  Linear interpolation
    gives p(R_in) = (p[-1] + p[0]) / 2 to O(dr²).  Enforcing the
    Dirichlet condition p(R_in) = p_inner yields the ghost value
    p[-1] = 2·p_inner − p[0].  Substituting into the standard 3-point
    radial stencil eliminates p[-1]; the coefficient on p[0] picks up
    an extra factor of 2 on the boundary face contribution, and the
    Dirichlet value moves to the RHS via ``compute_ghost_bc_rhs``.

    With this treatment the truncation error at boundary-adjacent
    nodes is O(dr²), preserving second-order global accuracy.  The
    extrapolated ghost coefficient D[-1] = D[0] is used at the
    boundary face (no harmonic averaging is needed because both
    virtual values coincide).
    """
    Nr, Ntheta = mesh.Nr, mesh.Ntheta
    N = mesh.N
    dr, dtheta = mesh.dr, mesh.dtheta
    r = mesh.R  # 2D array of radial positions

    # Build matrix in LIL format for efficient construction
    L = lil_matrix((N, N), dtype=float)

    for i in range(Nr):
        for j in range(Ntheta):
            k = mesh._index(i, j)
            r_ij = r[i, j]
            D_ij = D_coeff[i, j]

            # Neighbour indices (with periodic BC in theta)
            j_plus = (j + 1) % Ntheta
            j_minus = (j - 1) % Ntheta

            # === Radial diffusion: (1/r)∂/∂r(r·D·∂p/∂r) ===
            r_ip_half = r_ij + dr/2  # r at i+1/2
            r_im_half = r_ij - dr/2  # r at i-1/2

            if i == 0:
                # Inner boundary — ghost-node stencil.
                # Ghost at i = -1 eliminated via p[-1] = 2·p_inner - p[0].
                D_ip = D_coeff[i+1, j]
                D_ip_half = 2 * D_ij * D_ip / (D_ij + D_ip + 1e-30)
                # At the boundary face D[-1] = D[0] (zero-gradient extrapolation
                # of the material property); harmonic mean of equal values = D[0].
                D_im_half = D_ij

                coeff_ip = (r_ip_half * D_ip_half) / (r_ij * dr**2)
                coeff_im = (r_im_half * D_im_half) / (r_ij * dr**2)

                # Substituting p[-1] = 2·p_inner - p[0] into the 3-point stencil:
                #   coeff_ip·p[1] + coeff_im·p[-1] - (coeff_ip+coeff_im)·p[0]
                # = coeff_ip·p[1] - coeff_ip·p[0] - 2·coeff_im·p[0] + 2·coeff_im·p_inner
                # The last term is moved to the RHS by compute_ghost_bc_rhs.
                L[k, mesh._index(i+1, j)] += coeff_ip
                L[k, k] += -coeff_ip - 2 * coeff_im

            elif i == Nr - 1:
                # Outer boundary — ghost-node stencil.
                # Ghost at i = Nr eliminated via p[Nr] = 2·p_outer - p[Nr-1].
                D_im = D_coeff[i-1, j]
                D_im_half = 2 * D_ij * D_im / (D_ij + D_im + 1e-30)
                D_ip_half = D_ij  # D[Nr] = D[Nr-1] extrapolation

                coeff_ip = (r_ip_half * D_ip_half) / (r_ij * dr**2)
                coeff_im = (r_im_half * D_im_half) / (r_ij * dr**2)

                L[k, mesh._index(i-1, j)] += coeff_im
                L[k, k] += -coeff_im - 2 * coeff_ip

            else:
                # Interior point — standard 3-point central stencil.
                D_ip = D_coeff[i+1, j]
                D_im = D_coeff[i-1, j]

                # Harmonic average at faces
                D_ip_half = 2 * D_ij * D_ip / (D_ij + D_ip + 1e-30)
                D_im_half = 2 * D_ij * D_im / (D_ij + D_im + 1e-30)

                # Coefficients
                coeff_ip = (r_ip_half * D_ip_half) / (r_ij * dr**2)
                coeff_im = (r_im_half * D_im_half) / (r_ij * dr**2)
                coeff_center_r = -(coeff_ip + coeff_im)

                L[k, mesh._index(i+1, j)] += coeff_ip
                L[k, mesh._index(i-1, j)] += coeff_im
                L[k, k] += coeff_center_r

            # === Azimuthal diffusion: (1/r²)∂/∂θ(D·∂p/∂θ) ===
            D_jp = D_coeff[i, j_plus]
            D_jm = D_coeff[i, j_minus]

            # Harmonic average at faces
            D_jp_half = 2 * D_ij * D_jp / (D_ij + D_jp + 1e-30)
            D_jm_half = 2 * D_ij * D_jm / (D_ij + D_jm + 1e-30)

            # Coefficients
            coeff_jp = D_jp_half / (r_ij**2 * dtheta**2)
            coeff_jm = D_jm_half / (r_ij**2 * dtheta**2)
            coeff_center_theta = -(coeff_jp + coeff_jm)

            L[k, mesh._index(i, j_plus)] += coeff_jp
            L[k, mesh._index(i, j_minus)] += coeff_jm
            L[k, k] += coeff_center_theta

    return L.tocsr()


def compute_ghost_bc_rhs(
    mesh: PolarMesh,
    D_coeff: np.ndarray,
    bc_inner: float,
    bc_outer: float,
) -> np.ndarray:
    """
    Compute the RHS correction that enforces Dirichlet ghost-node BCs.

    After ``build_diffusion_operator`` folds the ghost unknowns out of
    the boundary rows, the inhomogeneous part of the Dirichlet
    condition becomes a constant source term that must be subtracted
    from the RHS of the linear system.  Specifically, the boundary
    stencils produce

        L[p] + 2·coeff_im·p_inner      at i = 0,
        L[p] + 2·coeff_ip·p_outer      at i = Nr-1,

    so the correction vector ``b_corr`` has

        b_corr[0, j]    = -2·coeff_im(0, j) · p_inner
        b_corr[Nr-1, j] = -2·coeff_ip(Nr-1, j) · p_outer

    and the system is solved as ``L · p = b_source + b_corr``.

    The function is linear in ``D_coeff``, so contributions from
    multiple additive operators (GNF + memory + …) can be combined by
    summing their effective D fields and calling this once.

    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    D_coeff : ndarray
        Effective diffusion coefficient field, shape (Nr, Ntheta),
        matching the operator whose boundary rows need correcting.
    bc_inner : float
        Dirichlet value at r = R_in
    bc_outer : float
        Dirichlet value at r = R_out

    Returns
    -------
    b_corr : ndarray
        RHS correction vector, shape (N,).
    """
    Nr, Ntheta = mesh.Nr, mesh.Ntheta
    dr = mesh.dr
    b_corr = np.zeros(mesh.N)

    # Inner boundary (i = 0): r_{-1/2} = R_in, D[-1] = D[0] extrapolation.
    for j in range(Ntheta):
        k = mesh._index(0, j)
        r_0 = mesh.R[0, j]
        r_im_half = r_0 - dr / 2  # physical boundary R_in
        D_im_half = D_coeff[0, j]
        coeff_im = (r_im_half * D_im_half) / (r_0 * dr**2)
        b_corr[k] -= 2.0 * coeff_im * bc_inner

    # Outer boundary (i = Nr-1): r_{Nr-1/2} = R_out, D[Nr] = D[Nr-1].
    for j in range(Ntheta):
        k = mesh._index(Nr - 1, j)
        r_N = mesh.R[Nr - 1, j]
        r_ip_half = r_N + dr / 2  # physical boundary R_out
        D_ip_half = D_coeff[Nr - 1, j]
        coeff_ip = (r_ip_half * D_ip_half) / (r_N * dr**2)
        b_corr[k] -= 2.0 * coeff_ip * bc_outer

    return b_corr


def build_GNF_operator(
    mesh: PolarMesh,
    h: np.ndarray,
    eta_bar: np.ndarray
) -> csr_matrix:
    """
    Build the Generalised Newtonian Fluid operator.
    
    L_GN[p] = (1/r)∂/∂r(r·h³/(12η̄)·∂p/∂r) + (1/r²)∂/∂θ(h³/(12η̄)·∂p/∂θ)
    
    This is Eq. 66 from the thesis.
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    h : ndarray
        Gap height field [m]
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s]
    
    Returns
    -------
    L_GN : csr_matrix
        GNF diffusion operator
    """
    D_coeff = h**3 / (12 * eta_bar)
    return build_diffusion_operator(mesh, D_coeff)


def build_memory_operator(
    mesh: PolarMesh,
    h: np.ndarray,
    eta_bar: np.ndarray,
    memory_visc: np.ndarray,
    h_dot: float,
    lambda_: float
) -> csr_matrix:
    """
    Build the squeeze-film memory operator.

    L_mem[p] = -(λḣ/12)(1/r)∂/∂r(r·h²M/η̄²·∂p/∂r)

    Coefficient 1/12 from the exact antisymmetric memory kernel
    (λ/η̄)∫₀ʰ(h/2 − z)·∂τ_p,rz⁽⁰⁾/∂t dz with the framework's γ̇ ∝ 1/h closure
    (verification/e0_kernel_exact.py; the previously shipped 1/16 had no
    supporting derivation). The viscosity factor M(α,Λ) is the derived
    memory viscosity GiesekusFluid.memory_viscosity
    (verification/p1_M_derivation.py), replacing the pre-2026-06 total η_T.

    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    h : ndarray
        Gap height [m]
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s]
    memory_visc : ndarray
        Memory viscosity M(α,Λ) [Pa·s]
    h_dot : float
        Squeeze velocity [m/s]
    lambda_ : float
        Relaxation time [s]

    Returns
    -------
    L_mem : csr_matrix
        Memory operator
    """
    # Operator/flux consistency: Reynolds eq is ∇·Q = S with Q = -D∇p + ...,
    # so L_mem = -(λḣ/12)·∇·(D_mem ∇p) is the divergence of the memory flux
    # Q_mem = +λh²ḣM/(12η̄²)·∇p (load-reducing during closure, ḣ < 0).
    prefactor = -lambda_ * h_dot / 12

    # Diffusion coefficient for memory term
    D_mem = h**2 * memory_visc / eta_bar**2
    
    # Build diffusion operator and scale
    L_base = build_diffusion_operator(mesh, D_mem)
    
    return prefactor * L_base


def compute_source_vector(
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions
) -> np.ndarray:
    """
    Compute the source term (RHS) vector.
    
    S = ḣ - (V_Tr/2)∂h/∂r - (V_Tθ/(2r))∂h/∂θ
    
    This is Eq. 71 from the thesis.
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    geometry : SlipperGeometry
        Bearing geometry
    conditions : OperatingConditions
        Operating parameters
    
    Returns
    -------
    S : ndarray
        Source vector, shape (N,)
    """
    R, THETA = mesh.R, mesh.THETA
    
    # Gap height and derivatives
    h = geometry.gap_height(R, THETA)
    _, dh_dr, dh_dtheta = geometry.gap_height_derivatives(R, THETA)
    
    # Velocity components
    V_Tr, V_Ttheta = conditions.radial_tangential_velocity(THETA)
    omega_s = conditions.omega_s
    
    # Squeeze term
    S = conditions.h_dot * np.ones_like(h)
    
    # Wedge terms
    V_theta_total = V_Ttheta + omega_s * R
    S = S - (V_Tr / 2) * dh_dr - (V_theta_total / (2 * R)) * dh_dtheta
    
    return mesh.to_1d(S)


# =============================================================================
# Normal Stress Contributions (explicit, added to RHS)
# =============================================================================

def compute_N1_contribution(
    mesh: PolarMesh,
    h: np.ndarray,
    Gamma_sq: np.ndarray,
    Psi1_bar: np.ndarray,
    eta_bar: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute the N₁ gradient contribution to the source term.
    
    From Eq. 69, this contributes as a pseudo-source:
        S_N1 = (1-α/2)/6 · (1/r)∂/∂r(r·h²/η̄ · ∂(hΨ̄₁Γ̄²)/∂r)
    
    Since this involves ∂(hΨ̄₁Γ̄²)/∂r which depends on pressure,
    we treat it explicitly (lagged) for simplicity.
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    h : ndarray
        Gap height [m]
    Gamma_sq : ndarray
        Γ̄² [s⁻²]
    Psi1_bar : ndarray
        Ψ̄₁ [Pa·s²]
    eta_bar : ndarray
        η̄ [Pa·s]
    alpha : float
        Mobility parameter
    
    Returns
    -------
    S_N1 : ndarray
        N₁ source contribution, shape (N,)
    """
    # Compute h·Ψ₁·Γ²
    h_Psi_Gamma = h * Psi1_bar * Gamma_sq

    # Compute gradient of h·Ψ₁·Γ² (both components)
    d_hPG_dr, d_hPG_dtheta = mesh.compute_field_gradient(h_Psi_Gamma)

    # Coefficient field
    coeff = (1 - alpha/2) / 6 * h**2 / eta_bar

    # Radial divergence: (1/r)∂/∂r(r·coeff·d_hPG_dr)
    flux_r = coeff * d_hPG_dr
    r_flux = mesh.R * flux_r
    d_rflux_dr, _ = mesh.compute_field_gradient(r_flux)
    div_N1_r = d_rflux_dr / mesh.R

    # Azimuthal divergence: (1/r²)∂/∂θ(coeff·d_hPG_dθ)
    # This term is zero for axisymmetric (no-tilt) cases but contributes
    # when tilt creates azimuthal variation in h, Ψ₁, and Γ².
    flux_theta = coeff * d_hPG_dtheta
    _, d_flux_theta_dtheta = mesh.compute_field_gradient(flux_theta)
    div_N1_theta = d_flux_theta_dtheta / mesh.R**2

    div_N1 = div_N1_r + div_N1_theta

    return mesh.to_1d(div_N1)


def compute_hoop_contribution(
    mesh: PolarMesh,
    h: np.ndarray,
    Gamma_sq: np.ndarray,
    Psi1_bar: np.ndarray,
    eta_bar: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute the hoop stress contribution to the source term.

    From Eq. 70, the hoop stress (tau_rr - tau_thetatheta) creates an
    effective radial body force.  The full divergence includes both
    radial and azimuthal components:

        S_hoop = -(1+alpha/2)/12 * [ (1/r) d/dr(h^3 Psi1 Gamma^2 / eta)
                                    + (1/r^2) d/dtheta(h^3 Psi1 Gamma^2 / eta) ]

    The azimuthal term is non-zero for tilted geometries where h, Psi1, and
    Gamma^2 vary with theta.

    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    h : ndarray
        Gap height [m]
    Gamma_sq : ndarray
        Gamma_bar^2 [s^-2]
    Psi1_bar : ndarray
        Psi_1_bar [Pa·s^2]
    eta_bar : ndarray
        eta_bar [Pa·s]
    alpha : float
        Mobility parameter

    Returns
    -------
    S_hoop : ndarray
        Hoop stress source contribution, shape (N,)
    """
    # The quantity inside the derivative
    inner = h**3 * Psi1_bar * Gamma_sq / eta_bar

    # Compute full gradient
    d_inner_dr, d_inner_dtheta = mesh.compute_field_gradient(inner)

    prefactor = -(1 - alpha/2) / 12   # WP-N: (1+a/2)->(1-a/2) (corrected hoop bookkeeping)

    # Radial: -(1+alpha/2)/12 · (1/r)d/dr(inner)
    S_hoop_r = prefactor * d_inner_dr / mesh.R

    # Azimuthal: -(1+alpha/2)/12 · (1/r^2)d/dtheta(inner)
    S_hoop_theta = prefactor * d_inner_dtheta / mesh.R**2

    S_hoop = S_hoop_r + S_hoop_theta

    return mesh.to_1d(S_hoop)


def compute_alpha_contribution(
    mesh: PolarMesh,
    h: np.ndarray,
    gdot_bar: np.ndarray,
    Psi1_bar: np.ndarray,
    eta_bar: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute the Giesekus coupling (L_alpha) contribution to the source term.

    From Eq. 68 (thesis Chapter 4):
        S_alpha = (alpha/4) · (1/r) ∂/∂r(r · h²/eta_bar · Psi1 · gdot² · ∂gdot/∂r)

    This term arises from the nonlinear tau·tau coupling in the Giesekus
    constitutive equation.  It is typically < 5% of the total viscoelastic
    correction and was previously computed only in post-processing flux
    decomposition.  It is now included as an explicit (lagged) RHS term
    for consistency with the full Eq. 4.31.

    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    h : ndarray
        Gap height [m]
    gdot_bar : ndarray
        Gap-averaged shear rate [s^-1]
    Psi1_bar : ndarray
        First normal stress coefficient [Pa·s^2]
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s]
    alpha : float
        Giesekus mobility parameter

    Returns
    -------
    S_alpha : ndarray
        Alpha coupling source contribution, shape (N,)
    """
    # Compute radial gradient of shear rate
    d_gdot_dr, d_gdot_dtheta = mesh.compute_field_gradient(gdot_bar)

    # Inner flux: (alpha/4) · (h^2 / eta_bar) · Psi1 · gdot^2 · d_gdot_dr
    coeff = (alpha / 4) * h**2 / eta_bar * Psi1_bar * gdot_bar**2

    # Radial component
    flux_r = coeff * d_gdot_dr
    r_flux = mesh.R * flux_r

    # Divergence: (1/r) d/dr (r · flux_r)
    d_rflux_dr, _ = mesh.compute_field_gradient(r_flux)
    div_alpha_r = d_rflux_dr / mesh.R

    # Azimuthal component: (alpha/4) · (h^2/eta_bar) · Psi1 · gdot^2 · (1/r^2) d_gdot_dtheta
    #   divergence: (1/r^2) d/dtheta(coeff * d_gdot_dtheta)
    flux_theta = coeff * d_gdot_dtheta
    _, d_flux_theta_dtheta = mesh.compute_field_gradient(flux_theta)
    div_alpha_theta = d_flux_theta_dtheta / mesh.R**2

    S_alpha = div_alpha_r + div_alpha_theta

    return mesh.to_1d(S_alpha)


# =============================================================================
# Complete System Assembly
# =============================================================================

@dataclass
class AssembledSystem:
    """
    Container for the assembled linear system.
    
    Attributes
    ----------
    A : csr_matrix
        System matrix
    b : ndarray
        Right-hand side vector
    A_GNF : csr_matrix
        GNF operator (for diagnostics)
    A_mem : csr_matrix, optional
        Memory operator
    """
    A: csr_matrix
    b: np.ndarray
    A_GNF: csr_matrix
    A_mem: Optional[csr_matrix] = None
    
    # Metadata
    info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.info is None:
            self.info = {}


def assemble_newtonian_system(
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions,
    eta: float
) -> AssembledSystem:
    """
    Assemble the Newtonian Reynolds equation system.
    
    This is the simplest case with constant viscosity and no
    viscoelastic effects. Used for validation.
    
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
    
    Returns
    -------
    system : AssembledSystem
        Assembled linear system
    """
    R, THETA = mesh.R, mesh.THETA
    h = geometry.gap_height(R, THETA)
    eta_bar = eta * np.ones_like(h)

    # Build GNF operator (ghost-node stencils at boundaries — see
    # build_diffusion_operator docstring).  The Dirichlet values enter
    # the linear system through compute_ghost_bc_rhs below, not by
    # overwriting boundary rows.
    D_GNF = h**3 / (12 * eta_bar)
    A_GNF = build_diffusion_operator(mesh, D_GNF)

    # Compute source
    b = compute_source_vector(mesh, geometry, conditions)

    # Apply ghost-node Dirichlet BCs (O(h²) accurate)
    p_inner = conditions.p_supply if hasattr(conditions, 'p_supply') else 0.0
    p_outer = conditions.p_ambient if hasattr(conditions, 'p_ambient') else 0.0
    b = b + compute_ghost_bc_rhs(mesh, D_GNF, p_inner, p_outer)

    return AssembledSystem(
        A=A_GNF, b=b, A_GNF=A_GNF,
        info={'type': 'newtonian', 'eta': eta}
    )


def assemble_GNF_system(
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions,
    fluid: GiesekusFluid,
    eta_bar: np.ndarray
) -> AssembledSystem:
    """
    Assemble the Generalised Newtonian Fluid system.
    
    Uses shear-rate-dependent viscosity but no elastic effects.
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    geometry : SlipperGeometry
        Bearing geometry
    conditions : OperatingConditions
        Operating parameters
    fluid : GiesekusFluid
        Fluid model (for viscosity)
    eta_bar : ndarray
        Current viscosity field [Pa·s]
    
    Returns
    -------
    system : AssembledSystem
        Assembled linear system
    """
    R, THETA = mesh.R, mesh.THETA
    h = geometry.gap_height(R, THETA)

    # Build GNF operator with current viscosity (ghost-node stencils)
    D_GNF = h**3 / (12 * eta_bar)
    A_GNF = build_diffusion_operator(mesh, D_GNF)

    # Compute source
    b = compute_source_vector(mesh, geometry, conditions)

    # Apply ghost-node Dirichlet BCs
    p_inner = conditions.p_supply if hasattr(conditions, 'p_supply') else 0.0
    p_outer = conditions.p_ambient if hasattr(conditions, 'p_ambient') else 0.0
    b = b + compute_ghost_bc_rhs(mesh, D_GNF, p_inner, p_outer)

    return AssembledSystem(
        A=A_GNF, b=b, A_GNF=A_GNF,
        info={'type': 'GNF'}
    )


def assemble_viscoelastic_system(
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions,
    fluid: GiesekusFluid,
    eta_bar: np.ndarray,
    gdot_bar: np.ndarray,
    pressure: Optional[np.ndarray] = None,
    include_memory: bool = True,
    include_normal_stress: bool = True,
    include_alpha_coupling: bool = False,
    memory_scale: Optional[np.ndarray] = None,
    ns_scale: float = 1.0
) -> AssembledSystem:
    """
    Assemble the full viscoelastic Reynolds equation system.

    Includes memory operator in LHS and normal stress terms in RHS.

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
    eta_bar : ndarray
        Current viscosity field [Pa·s]
    gdot_bar : ndarray
        Current shear rate field [s⁻¹]
    pressure : ndarray, optional
        Current pressure field (for explicit N₁, hoop terms)
    include_memory : bool
        Include memory operator in LHS
    include_normal_stress : bool
        Include N₁ and hoop stress in RHS
    ns_scale : float
        Scaling factor for normal stress RHS contributions, in [0, 1].
        Used for gradual ramp-up during Picard iteration to prevent
        divergence from the quadratic |∇p|² coupling.  Default: 1.0.

    Returns
    -------
    system : AssembledSystem
        Assembled linear system
    """
    R, THETA = mesh.R, mesh.THETA
    h = geometry.gap_height(R, THETA)

    # Build GNF operator (ghost-node stencils).  The effective D for
    # the BC correction accumulates GNF + memory contributions.
    D_GNF = h**3 / (12 * eta_bar)
    A_GNF = build_diffusion_operator(mesh, D_GNF)
    A = A_GNF.copy()
    A_mem = None
    D_eff_boundary = D_GNF.copy()

    # Add memory operator if requested
    if include_memory and not fluid.is_newtonian:
        # Memory viscosity M(alpha, Lambda): the linearised polymer-stress
        # response (verification/p1_M_derivation.py). Replaces the total
        # tangent viscosity eta_T used before 2026-06 (the solvent stress
        # is instantaneous; at strong coupling the polymer response is
        # additionally suppressed -- benchmark: verification/p2_gate.py).
        M_mem = fluid.memory_viscosity(gdot_bar)
        # Coefficient 1/12 from the exact antisymmetric kernel
        # (verification/e0_kernel_exact.py); replaces the unsupported 1/16.
        mem_prefactor = -fluid.lambda_ * conditions.h_dot / 12
        D_mem = h**2 * M_mem / eta_bar**2
        # WP-G: optional alpha-renormalisation R(Lambda) of the memory flux.
        if memory_scale is not None:
            D_mem = D_mem * memory_scale
        A_mem = mem_prefactor * build_diffusion_operator(mesh, D_mem)
        A = A + A_mem
        # The combined effective D at the boundary face for the BC
        # correction is D_GNF + mem_prefactor · D_mem.
        D_eff_boundary = D_eff_boundary + mem_prefactor * D_mem
    
    # Compute base source term
    b = compute_source_vector(mesh, geometry, conditions)
    
    # Add normal stress and alpha contributions if requested and pressure available
    if (include_normal_stress and pressure is not None
            and not fluid.is_newtonian and ns_scale > 1e-12):
        # Compute Γ² from current pressure
        dp_dr, dp_dtheta = mesh.compute_field_gradient(pressure)
        V_Tr, V_Ttheta = conditions.radial_tangential_velocity(THETA)
        V_T = np.sqrt(V_Tr**2 + V_Ttheta**2)

        from .fluxes import compute_Gamma_squared
        Gamma_sq = compute_Gamma_squared(h, V_T, dp_dr, dp_dtheta, R, eta_bar)
        # NOTE: a silent cap Gamma_sq <= 100 x max(V_T^2/h^2, 1e-10) lived here
        # from the initial release until 2026-06. In pure squeeze (V_T = 0) it
        # annihilated the N1 + hoop sources, so all published pure-squeeze runs
        # effectively had normal stresses OFF (verification/e1_clamp_evidence.py).
        # It is removed: out-of-envelope use now produces a loud runtime warning
        # from the validity guard in solver.py instead of silent zeroing.

        Psi1_bar = fluid.Psi1(gdot_bar)
        alpha = fluid.alpha

        # N₁ contribution (Eq. 69)
        S_N1 = compute_N1_contribution(mesh, h, Gamma_sq, Psi1_bar, eta_bar, alpha)
        b = b + ns_scale * S_N1

        # Hoop contribution (Eq. 70)
        S_hoop = compute_hoop_contribution(mesh, h, Gamma_sq, Psi1_bar, eta_bar, alpha)
        b = b + ns_scale * S_hoop

    # Giesekus alpha-coupling (Eq. 68) -- DISABLED by default.
    # WP-B/WP-C (revision1) found the written term -(a h^2/4 eta) Psi1 gdot^2
    # d(gdot)/dr is dimensionally m/s^2 (not m^2/s) and carries no De_sq factor;
    # its spuriously large magnitude drove the VE-solve divergence. The
    # re-derived term is O(alpha*De_sq) (subdominant). It stays excluded until
    # the exact prefactor is confirmed; the flag is retained only for the WP-D
    # divergence-isolation diagnostic.
    if (include_alpha_coupling and pressure is not None
            and not fluid.is_newtonian and ns_scale > 1e-12 and fluid.alpha > 1e-10):
        S_alpha = compute_alpha_contribution(
            mesh, h, gdot_bar, fluid.Psi1(gdot_bar), eta_bar, fluid.alpha
        )
        b = b + ns_scale * S_alpha
    
    # Apply ghost-node Dirichlet BCs.  The effective boundary D folds
    # in both GNF and memory contributions so a single correction
    # captures both operators exactly.
    p_inner = conditions.p_supply if hasattr(conditions, 'p_supply') else 0.0
    p_outer = conditions.p_ambient if hasattr(conditions, 'p_ambient') else 0.0
    b = b + compute_ghost_bc_rhs(mesh, D_eff_boundary, p_inner, p_outer)

    return AssembledSystem(
        A=A, b=b, A_GNF=A_GNF, A_mem=A_mem,
        info={
            'type': 'viscoelastic',
            'include_memory': include_memory,
            'include_normal_stress': include_normal_stress
        }
    )
