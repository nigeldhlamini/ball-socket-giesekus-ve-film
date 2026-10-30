"""
Giesekus Material Functions for Viscoelastic Lubrication
=========================================================

Implements the exact analytical expressions for the Giesekus constitutive model
in steady simple shear:

    η(γ̇)   = Viscosity function (Eq. 45)
    Ψ₁(γ̇) = First normal stress coefficient (Eq. 46)  
    Ψ₂(γ̇) = Second normal stress coefficient (Eq. 47)

The auxiliary variable f satisfies a quadratic equation derived from the
steady-state Giesekus constitutive equation.

Key Properties:
    - Low Wi limit: η → η₀, Ψ₁ → Ψ₁⁰ = 2η_p λ, Ψ₂ → Ψ₂⁰ = -αη_p λ
      (so Ψ₂/Ψ₁ → -α/2 only in the *zero-shear* limit)
    - High Wi limit: η → η_s and Ψ₁ → 0 *gradually* (no sharp cut-off); the
      ratio Ψ₂/Ψ₁ is NOT constant — its magnitude decreases below α/2 with
      increasing shear (and is not pinned to -α/2 at finite Wi).
    - The auxiliary variable is obtained from the exact Giesekus steady-shear
      solution (variable χ, below), giving gradual shear thinning. There is no
      finite "critical" Weissenberg number at which η collapses to η_s.

Implementation note (corrected 2026, revision1):
    The earlier auxiliary-variable quadratic
        (1-2α)f² - [1-2α + α²Λ²]f + α²Λ² = 0
    is NOT the Giesekus steady-shear relation: its physical root has the wrong
    small-Λ slope (f ~ α²Λ² instead of f ~ αΛ²) and a perfect-square
    discriminant that forces f→1 at a spurious Λ_crit=√(1-2α)/α, collapsing η
    to η_s and Ψ₁ to 0 with an unphysical sharp cut-off.  The exact solution
    (Bird, Armstrong & Hassager, Vol. 1; Giesekus 1982) is used instead:
        χ² = 2 / (1 + √(1 + 16 α(1-α) Λ²)),   f = (1-χ)/(1 + (1-2α)χ),
    verified against a direct stress-tensor Newton solve to ~1e-12
    (scripts/wp_verify_normal_stress.py).

References:
    - Giesekus, H. (1982). J. Non-Newtonian Fluid Mech. 11, 69-109.
    - Bird et al. (1987). Dynamics of Polymeric Liquids, Vol. 1.
    - Dhlamini, N.C. (2025). PhD Thesis, Chapter 4.

Author: Nigel C. Dhlamini
"""

from dataclasses import dataclass, field
from typing import Union, Tuple
import numpy as np

# Type alias for array-like inputs
ArrayLike = Union[float, np.ndarray]


@dataclass
class GiesekusFluid:
    """
    Giesekus fluid with exact material functions in steady shear.
    
    Parameters
    ----------
    eta_s : float
        Solvent viscosity [Pa·s]
    eta_p : float
        Polymer viscosity contribution [Pa·s]
    lambda_ : float
        Relaxation time [s]
    alpha : float
        Mobility factor, 0 ≤ α ≤ 0.5. Controls shear-thinning strength
        and normal stress ratio Ψ₂/Ψ₁ = -α/2
    
    Attributes
    ----------
    eta_0 : float
        Zero-shear viscosity η₀ = η_s + η_p [Pa·s]
    Psi1_0 : float
        Zero-shear first normal stress coefficient Ψ₁⁰ = 2η_p λ [Pa·s²]
    Psi2_0 : float
        Zero-shear second normal stress coefficient Ψ₂⁰ = -αη_p λ [Pa·s²]
    beta : float
        Viscosity ratio β = η_s/η₀
    
    Examples
    --------
    >>> fluid = GiesekusFluid(eta_s=0.030, eta_p=0.120, lambda_=3.3e-3, alpha=0.25)
    >>> fluid.eta_0
    0.15
    >>> fluid.viscosity(1000.0)  # At γ̇ = 1000 s⁻¹
    ...
    """
    eta_s: float
    eta_p: float
    lambda_: float
    alpha: float
    
    # Derived quantities (computed in __post_init__)
    eta_0: float = field(init=False)
    Psi1_0: float = field(init=False)
    Psi2_0: float = field(init=False)
    beta: float = field(init=False)
    
    def __post_init__(self):
        """Validate parameters and compute derived quantities."""
        # Validation
        if self.eta_s < 0:
            raise ValueError(f"Solvent viscosity must be non-negative: {self.eta_s}")
        if self.eta_p < 0:
            raise ValueError(f"Polymer viscosity must be non-negative: {self.eta_p}")
        if self.lambda_ < 0:
            raise ValueError(f"Relaxation time must be non-negative: {self.lambda_}")
        if not (0 <= self.alpha <= 0.5):
            raise ValueError(f"Mobility factor must be in [0, 0.5]: {self.alpha}")
        
        # Derived quantities
        self.eta_0 = self.eta_s + self.eta_p
        self.Psi1_0 = 2.0 * self.eta_p * self.lambda_
        self.Psi2_0 = -self.alpha * self.eta_p * self.lambda_
        self.beta = self.eta_s / self.eta_0 if self.eta_0 > 0 else 1.0
    
    @property
    def is_newtonian(self) -> bool:
        """Check if fluid is effectively Newtonian (λ = 0 or η_p = 0)."""
        return self.lambda_ < 1e-15 or self.eta_p < 1e-15
    
    def _compute_f(self, Lambda: ArrayLike) -> ArrayLike:
        """
        Auxiliary variable f from the exact Giesekus steady-shear solution.

            χ² = 2 / (1 + √(1 + 16 α(1-α) Λ²))            (numerically stable)
            f  = (1 - χ) / (1 + (1-2α) χ)

        Parameters
        ----------
        Lambda : float or ndarray
            Dimensionless shear rate Λ = λγ̇ (Weissenberg number)

        Returns
        -------
        f : float or ndarray
            Auxiliary variable, 0 ≤ f < 1, with f → 0 as Λ → 0 (f ~ αΛ²) and
            f → 1 only as Λ → ∞.  Valid for all α ∈ [0, 0.5] (no degeneracy at
            α = 0.5, unlike the earlier quadratic).
        """
        Lambda = np.asarray(Lambda)
        scalar_input = Lambda.ndim == 0
        Lambda = np.atleast_1d(Lambda).astype(float)
        alpha = self.alpha

        # Newtonian / UCM (α→0): no shear structure, f ≡ 0
        if self.is_newtonian or alpha < 1e-12:
            f = np.zeros_like(Lambda)
            return float(f[0]) if scalar_input else f

        L = np.abs(Lambda)
        x = 16.0 * alpha * (1.0 - alpha) * L ** 2
        chi = np.sqrt(2.0 / (1.0 + np.sqrt(1.0 + x)))
        f = (1.0 - chi) / (1.0 + (1.0 - 2.0 * alpha) * chi)

        # Low-Λ asymptote f ~ αΛ² avoids (1-χ) cancellation as χ→1
        small = L < 1e-4
        if np.any(small):
            f = np.where(small, alpha * L ** 2, f)

        f = np.clip(f, 0.0, 1.0)
        return float(f[0]) if scalar_input else f

    def _stress_components(self, Lambda: ArrayLike):
        """
        Dimensionless polymer stress components (S = (λ/η_p) τ_p) in steady
        simple shear, from the exact Giesekus solution.  Λ = λγ̇.

        Returns (Sxx, Syy, Sxy) as arrays.  Used by Ψ₁ (∝ Sxx - Syy) and
        Ψ₂ (∝ Syy).  Requires α > 0 (caller handles the UCM limit).
        """
        a = self.alpha
        L = np.abs(np.atleast_1d(np.asarray(Lambda, dtype=float)))
        f = np.atleast_1d(self._compute_f(L))
        D = 1.0 + (1.0 - 2.0 * a) * f
        Sxy = L * (1.0 - f) ** 2 / D
        # a Syy² + Syy + a Sxy² = 0  -> physical root (→0 as Λ→0)
        Syy = (-1.0 + np.sqrt(np.maximum(1.0 - 4.0 * a ** 2 * Sxy ** 2, 0.0))) / (2.0 * a)
        # a Sxx² + Sxx + (a Sxy² - 2 Λ Sxy) = 0  -> physical root
        Sxx = (-1.0 + np.sqrt(np.maximum(
            1.0 - 4.0 * a * (a * Sxy ** 2 - 2.0 * L * Sxy), 0.0))) / (2.0 * a)
        return Sxx, Syy, Sxy
    
    def viscosity(self, gdot: ArrayLike) -> ArrayLike:
        """
        Compute shear-rate-dependent viscosity η(γ̇).
        
        From Eq. 45:
            η(γ̇) = η_s + η_p · (1-f)² / [1 + (1-2α)f]
        
        Parameters
        ----------
        gdot : float or ndarray
            Shear rate γ̇ [s⁻¹]
        
        Returns
        -------
        eta : float or ndarray
            Viscosity [Pa·s]
        
        Limits
        ------
        - γ̇ → 0: η → η₀ = η_s + η_p
        - γ̇ → ∞: η → η_s (shear-thinning plateau)
        """
        gdot = np.asarray(gdot)
        scalar_input = gdot.ndim == 0
        gdot = np.atleast_1d(gdot)
        
        # Handle Newtonian limit
        if self.is_newtonian:
            eta = np.full_like(gdot, self.eta_0, dtype=float)
            return float(eta[0]) if scalar_input else eta
        
        Lambda = self.lambda_ * np.abs(gdot)
        f = self._compute_f(Lambda)
        
        alpha = self.alpha
        
        # Viscosity formula (Eq. 45)
        numerator = (1.0 - f)**2
        denominator = 1.0 + (1.0 - 2.0 * alpha) * f
        
        eta = self.eta_s + self.eta_p * numerator / denominator
        
        return float(eta[0]) if scalar_input else eta
    
    def Psi1(self, gdot: ArrayLike) -> ArrayLike:
        """
        Compute first normal stress coefficient Ψ₁(γ̇).
        
        From the exact Giesekus steady-shear stress solution:
            Ψ₁ = N₁/γ̇² = η_p λ (Sxx - Syy) / Λ²
        with (Sxx, Syy) the dimensionless polymer-stress components.

        Parameters
        ----------
        gdot : float or ndarray
            Shear rate γ̇ [s⁻¹]

        Returns
        -------
        Psi1 : float or ndarray
            First normal stress coefficient [Pa·s²]

        Limits
        ------
        - γ̇ → 0: Ψ₁ → Ψ₁⁰ = 2η_p λ
        - γ̇ → ∞: Ψ₁ → 0 gradually (no sharp cut-off)
        """
        gdot = np.asarray(gdot)
        scalar_input = gdot.ndim == 0
        gdot = np.atleast_1d(gdot).astype(float)

        # Newtonian (Ψ₁ = 0)
        if self.is_newtonian:
            Psi1 = np.zeros_like(gdot)
            return float(Psi1[0]) if scalar_input else Psi1

        Lambda = self.lambda_ * np.abs(gdot)
        Psi1 = np.empty_like(gdot)

        # Low-Λ: Ψ₁ → Ψ₁⁰ (avoids 0/0 in (Sxx-Syy)/Λ²)
        low = Lambda < 1e-4
        if np.any(low):
            Psi1[low] = self.Psi1_0
        high = ~low
        if np.any(high):
            if self.alpha < 1e-12:
                # UCM limit: Sxx-Syy = 2Λ²  ->  Ψ₁ = 2 η_p λ (constant)
                Psi1[high] = self.Psi1_0
            else:
                Sxx, Syy, _ = self._stress_components(Lambda[high])
                Psi1[high] = self.eta_p * self.lambda_ * (Sxx - Syy) / Lambda[high] ** 2

        return float(Psi1[0]) if scalar_input else Psi1
    
    def Psi2(self, gdot: ArrayLike) -> ArrayLike:
        """
        Compute second normal stress coefficient Ψ₂(γ̇).
        
        From the exact Giesekus steady-shear stress solution (Szz = 0 in shear,
        so N₂ = τ_yy):
            Ψ₂ = N₂/γ̇² = η_p λ Syy / Λ²    (Syy < 0)

        Parameters
        ----------
        gdot : float or ndarray
            Shear rate γ̇ [s⁻¹]

        Returns
        -------
        Psi2 : float or ndarray
            Second normal stress coefficient [Pa·s²] (≤ 0)

        Limits
        ------
        - γ̇ → 0: Ψ₂ → Ψ₂⁰ = -αη_p λ = -α/2 · Ψ₁⁰  (so Ψ₂/Ψ₁ → -α/2 ONLY here)
        - finite γ̇: |Ψ₂/Ψ₁| < α/2 and decreasing; the ratio is NOT a constant.
        """
        gdot = np.asarray(gdot)
        scalar_input = gdot.ndim == 0
        gdot = np.atleast_1d(gdot).astype(float)

        # Newtonian / UCM (α→0): Ψ₂ = 0
        if self.is_newtonian or self.alpha < 1e-12:
            Psi2 = np.zeros_like(gdot)
            return float(Psi2[0]) if scalar_input else Psi2

        Lambda = self.lambda_ * np.abs(gdot)
        Psi2 = np.empty_like(gdot)

        low = Lambda < 1e-4
        if np.any(low):
            Psi2[low] = self.Psi2_0
        high = ~low
        if np.any(high):
            _, Syy, _ = self._stress_components(Lambda[high])
            Psi2[high] = self.eta_p * self.lambda_ * Syy / Lambda[high] ** 2

        return float(Psi2[0]) if scalar_input else Psi2
    
    def tangent_viscosity(self, gdot: ArrayLike) -> ArrayLike:
        """
        Compute tangent viscosity η_T(γ̇) = η + γ̇ · dη/dγ̇.
        
        Used in memory flux correction (Eq. 60).
        
        Parameters
        ----------
        gdot : float or ndarray
            Shear rate γ̇ [s⁻¹]
        
        Returns
        -------
        eta_T : float or ndarray
            Tangent viscosity [Pa·s]
        
        Notes
        -----
        For shear-thinning fluids, dη/dγ̇ < 0, so η_T < η.
        For Newtonian fluids, η_T = η.
        """
        gdot = np.asarray(gdot)
        scalar_input = gdot.ndim == 0
        gdot = np.atleast_1d(gdot)
        
        # Numerical differentiation with central differences
        delta = np.maximum(1e-6 * np.abs(gdot), 1e-6)
        
        eta_plus = self.viscosity(gdot + delta)
        eta_minus = self.viscosity(gdot - delta)
        eta = self.viscosity(gdot)
        
        d_eta_d_gdot = (eta_plus - eta_minus) / (2.0 * delta)
        eta_T = eta + gdot * d_eta_d_gdot

        return float(eta_T[0]) if scalar_input else eta_T

    def memory_viscosity(self, gdot: ArrayLike) -> ArrayLike:
        """
        Derived memory viscosity M(α, Λ) for the squeeze-memory flux.

            M = (η_p/Λ) |[J⁻¹ (Λ dS₀/dΛ)]_xy|

        where J is the Giesekus constitutive operator linearised about the
        closed-form steady shear state S₀(α, Λ) and the forcing is the
        Λ-advection of the base state. M is the viscosity factor that
        belongs in the squeeze-memory flux: only the polymer stress
        relaxes, and its first-order response passes through J⁻¹ rather
        than the bare chain rule. Derivation and limit proofs:
        verification/p1_M_derivation.py; benchmark corroboration:
        verification/p2_gate.py (in-domain z-resolved transient).

        Limits
        ------
        - Λ → 0:  M → η_p (the zero-shear polymer tangent viscosity)
        - α → 0:  M → η_p exactly (UCM; [J₀⁻¹g]_xy = -(Λ g_yy + g_xy)
                  with g_yy = 0)
        - α Λ² ≫ 1: M collapses far below the total tangent viscosity η_T
          (the polymer shear stress plateaus and its transient response is
          additionally suppressed); at (α=0.25, Λ=2000), M/η_p ≈ 1.0e-5.

        Notes
        -----
        Replaces the (incorrect) total tangent viscosity η_T used before
        2026-06: the solvent stress is instantaneous and must not appear
        in the memory term. M ≈ (η_T − η_s)·R(α,Λ) only for αΛ² ≲ 1;
        at strong coupling neither factorization holds and M must be
        evaluated directly.
        """
        gdot = np.asarray(gdot)
        scalar_input = gdot.ndim == 0
        gdot = np.atleast_1d(gdot).astype(float)
        shape = gdot.shape

        # UCM / Newtonian limit: M = η_p exactly
        if self.is_newtonian or self.alpha < 1e-12:
            M = np.full(shape, self.eta_p, dtype=float)
            return float(M.ravel()[0]) if scalar_input else M

        a = self.alpha
        L = np.maximum(self.lambda_ * np.abs(gdot).ravel(), 1e-10)

        # closed-form steady state and its Λ-derivative (central FD; the
        # components are smooth in Λ)
        Sxx, Syy, Sxy = self._stress_components(L)
        dL = np.maximum(1e-6 * L, 1e-12)
        Sxx_p, Syy_p, Sxy_p = self._stress_components(L + dL)
        Sxx_m, Syy_m, Sxy_m = self._stress_components(np.maximum(L - dL, 1e-12))
        dSxx = (Sxx_p - Sxx_m) / (2 * dL)
        dSyy = (Syy_p - Syy_m) / (2 * dL)
        dSxy = (Sxy_p - Sxy_m) / (2 * dL)

        # linearised operator J (rows: xx, yy, xy; appendix D of the paper)
        n = L.size
        J = np.zeros((n, 3, 3))
        J[:, 0, 0] = -1.0 - 2.0 * a * Sxx
        J[:, 0, 2] = 2.0 * L - 2.0 * a * Sxy
        J[:, 1, 1] = -1.0 - 2.0 * a * Syy
        J[:, 1, 2] = -2.0 * a * Sxy
        J[:, 2, 0] = -a * Sxy
        J[:, 2, 1] = L - a * Sxy
        J[:, 2, 2] = -1.0 - a * (Sxx + Syy)
        g = np.stack([dSxx * L, dSyy * L, dSxy * L], axis=1)

        s = np.linalg.solve(J, g[..., None])[..., 0]
        M = (self.eta_p * np.abs(s[:, 2]) / L).reshape(shape)
        return float(M.ravel()[0]) if scalar_input else M

    def complex_viscosity(self, omega: ArrayLike) -> Union[complex, np.ndarray]:
        """
        SAOS complex viscosity η*(ω) for the Giesekus model.

        In the linear viscoelastic limit (small amplitude oscillatory shear):
            η*(ω) = η_s + η_p / (1 + iωλ)

        Parameters
        ----------
        omega : float or ndarray
            Angular frequency [rad/s]

        Returns
        -------
        eta_star : complex or ndarray of complex
            Complex viscosity [Pa·s]

        References
        ----------
        Bird et al. (1987), Dynamics of Polymeric Liquids, Vol. 1, Eq. 8.5-6.
        """
        omega = np.asarray(omega, dtype=float)
        scalar_input = omega.ndim == 0
        omega = np.atleast_1d(omega)

        if self.is_newtonian:
            result = np.full_like(omega, complex(self.eta_0, 0.0))
        else:
            result = self.eta_s + self.eta_p / (1.0 + 1j * omega * self.lambda_)

        return complex(result[0]) if scalar_input else result

    def eta_prime(self, omega: ArrayLike) -> ArrayLike:
        """
        In-phase (viscous) component η'(ω) = Re[η*(ω)].

        Parameters
        ----------
        omega : float or ndarray
            Angular frequency [rad/s]

        Returns
        -------
        eta_p : float or ndarray
            Storage viscosity [Pa·s]
        """
        eta_star = self.complex_viscosity(omega)
        result = np.real(eta_star)
        return float(result) if np.ndim(omega) == 0 else result

    def eta_double_prime(self, omega: ArrayLike) -> ArrayLike:
        """
        Out-of-phase (elastic) component η''(ω) = -Im[η*(ω)].

        Parameters
        ----------
        omega : float or ndarray
            Angular frequency [rad/s]

        Returns
        -------
        eta_pp : float or ndarray
            Loss viscosity [Pa·s]
        """
        eta_star = self.complex_viscosity(omega)
        result = -np.imag(eta_star)
        return float(result) if np.ndim(omega) == 0 else result

    def material_functions(self, gdot: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute all material functions at given shear rate(s).
        
        Parameters
        ----------
        gdot : float or ndarray
            Shear rate(s) γ̇ [s⁻¹]
        
        Returns
        -------
        eta : ndarray
            Viscosity [Pa·s]
        Psi1 : ndarray
            First normal stress coefficient [Pa·s²]
        Psi2 : ndarray
            Second normal stress coefficient [Pa·s²]
        """
        return self.viscosity(gdot), self.Psi1(gdot), self.Psi2(gdot)
    
    def normal_stress_ratio(self, gdot: ArrayLike) -> ArrayLike:
        """
        Compute Ψ₂/Ψ₁ ratio.

        Approaches -α/2 in the zero-shear limit only; at finite shear its
        magnitude is smaller than α/2 (the exact Giesekus solution does not
        pin the ratio to a constant).

        Parameters
        ----------
        gdot : float or ndarray
            Shear rate(s) γ̇ [s⁻¹]
        
        Returns
        -------
        ratio : float or ndarray
            Ψ₂/Ψ₁ (should be approximately -α/2)
        """
        Psi1 = self.Psi1(gdot)
        Psi2 = self.Psi2(gdot)
        
        # Avoid division by zero
        ratio = np.where(np.abs(Psi1) > 1e-20, Psi2 / Psi1, -self.alpha / 2.0)
        return ratio
    
    def De(self, U: float, L: float) -> float:
        """Deborah number De = λU/L."""
        return self.lambda_ * U / L
    
    def Wi(self, U: float, h: float) -> float:
        """Weissenberg number Wi = λU/h."""
        return self.lambda_ * U / h
    
    def De_sq(self, h_dot: float, h: float) -> float:
        """Squeeze Deborah number De_sq = λ|ḣ|/h."""
        return self.lambda_ * abs(h_dot) / h
    
    def __repr__(self) -> str:
        return (f"GiesekusFluid(η_s={self.eta_s:.4f}, η_p={self.eta_p:.4f}, "
                f"λ={self.lambda_:.2e}, α={self.alpha:.2f})")
    
    def summary(self) -> str:
        """Return a formatted summary of fluid properties."""
        lines = [
            "Giesekus Fluid Properties",
            "=" * 40,
            f"Solvent viscosity η_s    : {self.eta_s:.4f} Pa·s",
            f"Polymer viscosity η_p    : {self.eta_p:.4f} Pa·s",
            f"Zero-shear viscosity η₀  : {self.eta_0:.4f} Pa·s",
            f"Relaxation time λ        : {self.lambda_:.2e} s",
            f"Mobility factor α        : {self.alpha:.3f}",
            f"Viscosity ratio β        : {self.beta:.3f}",
            "-" * 40,
            "Zero-shear normal stress coefficients:",
            f"  Ψ₁⁰ = 2η_p λ           : {self.Psi1_0:.2e} Pa·s²",
            f"  Ψ₂⁰ = -αη_p λ          : {self.Psi2_0:.2e} Pa·s²",
            f"  Ψ₂⁰/Ψ₁⁰ = -α/2        : {-self.alpha/2:.3f}",
        ]
        return "\n".join(lines)


# =============================================================================
# Predefined Fluids (Table 1 in thesis)
# =============================================================================

def create_newtonian(eta: float = 0.030) -> GiesekusFluid:
    """Create a Newtonian fluid (λ = 0)."""
    return GiesekusFluid(eta_s=eta, eta_p=0.0, lambda_=0.0, alpha=0.0)


def create_PAM_2pct() -> GiesekusFluid:
    """
    Create 2% PAM (polyacrylamide) in water-glycol.

    From Table 1:
        η₀ = 0.050 Pa·s, η_s = 0.030 Pa·s, λ = 1.0 ms, α = 0.25

    Re-added to the corrected (revision1) rheology for call-surface parity with
    the campaign-live infrastructure (src/__init__.py, src/dynamics.py,
    scripts/generate_paper1_figures.py). The GiesekusFluid constructor is
    identical across branches, so this is a pure convenience builder and does
    not touch the corrected physics.
    """
    return GiesekusFluid(eta_s=0.030, eta_p=0.020, lambda_=1.0e-3, alpha=0.25)


def create_PAM_5pct() -> GiesekusFluid:
    """
    Create 5% PAM (polyacrylamide) in water-glycol.

    From Table 1:
        η₀ = 0.150 Pa·s, η_s = 0.030 Pa·s, λ = 3.3 ms, α = 0.25

    See create_PAM_2pct note on call-surface parity.
    """
    return GiesekusFluid(eta_s=0.030, eta_p=0.120, lambda_=3.3e-3, alpha=0.25)


# =============================================================================
# SOF (Second-Order Fluid) as Limiting Case
# =============================================================================

@dataclass
class SOFFluid:
    """
    Second-Order Fluid (SOF) with constant material properties.
    
    The SOF emerges as the Wi → 0 limit of Giesekus.
    Constitutive equation:
        τ = η₀ A₁ - (Ψ₁⁰/2) A₁^∇ + Ψ₂⁰ A₁²
    
    Parameters
    ----------
    eta_0 : float
        Constant viscosity [Pa·s]
    Psi1_0 : float
        First normal stress coefficient [Pa·s²]
    Psi2_0 : float
        Second normal stress coefficient [Pa·s²]
    
    Notes
    -----
    For consistency with Giesekus, use Ψ₂⁰/Ψ₁⁰ = -α/2.
    """
    eta_0: float
    Psi1_0: float
    Psi2_0: float
    
    def viscosity(self, gdot: ArrayLike) -> ArrayLike:
        """Constant viscosity (independent of γ̇)."""
        gdot = np.asarray(gdot)
        return np.full_like(gdot, self.eta_0, dtype=float)
    
    def Psi1(self, gdot: ArrayLike) -> ArrayLike:
        """Constant first normal stress coefficient."""
        gdot = np.asarray(gdot)
        return np.full_like(gdot, self.Psi1_0, dtype=float)
    
    def Psi2(self, gdot: ArrayLike) -> ArrayLike:
        """Constant second normal stress coefficient."""
        gdot = np.asarray(gdot)
        return np.full_like(gdot, self.Psi2_0, dtype=float)
    
    def tangent_viscosity(self, gdot: ArrayLike) -> ArrayLike:
        """Tangent viscosity equals viscosity for constant η."""
        return self.viscosity(gdot)
    
    @property
    def lambda_(self) -> float:
        """Effective relaxation time λ = Ψ₁⁰/(2η₀)."""
        if self.eta_0 > 0:
            return self.Psi1_0 / (2.0 * self.eta_0)
        return 0.0
    
    @classmethod
    def from_giesekus(cls, fluid: GiesekusFluid) -> 'SOFFluid':
        """Create SOF from Giesekus fluid using zero-shear properties."""
        return cls(
            eta_0=fluid.eta_0,
            Psi1_0=fluid.Psi1_0,
            Psi2_0=fluid.Psi2_0
        )


# =============================================================================
# Verification Functions
# =============================================================================

def verify_psi_ratio(fluid: GiesekusFluid, gdot_range: ArrayLike = None) -> dict:
    """
    Verify the *zero-shear* normal-stress ratio Ψ₂/Ψ₁ → -α/2, and that at
    finite shear the ratio's magnitude stays in (0, α/2] (it is NOT constant).

    Parameters
    ----------
    fluid : GiesekusFluid
        Fluid to verify
    gdot_range : array-like, optional
        Shear rates to report the profile over. Default: 10⁰ to 10⁶ s⁻¹

    Returns
    -------
    dict
        Verification results.  ``all_pass`` checks the zero-shear limit and the
        magnitude bound; the full ``ratio`` profile is returned for inspection.
    """
    if gdot_range is None:
        gdot_range = np.logspace(0, 6, 50)

    ratio = fluid.normal_stress_ratio(gdot_range)
    expected = -fluid.alpha / 2.0

    # Zero-shear limit (the true molecular constraint)
    gdot_zero = 1e-6 / fluid.lambda_ if fluid.lambda_ > 0 else 1e-6
    ratio_zero = float(fluid.normal_stress_ratio(gdot_zero))
    zero_shear_error = abs(ratio_zero - expected)

    # Finite-shear ratios must not exceed α/2 in magnitude (within tiny tol)
    magnitude_ok = bool(np.all(np.abs(ratio) <= abs(expected) + 1e-9))

    return {
        'gdot': gdot_range,
        'ratio': ratio,
        'expected_zero_shear': expected,
        'ratio_zero_shear': ratio_zero,
        'zero_shear_error': zero_shear_error,
        'max_error': zero_shear_error,           # back-compat key
        'magnitude_bounded': magnitude_ok,
        'all_pass': (zero_shear_error < 0.01 * abs(expected)) and magnitude_ok,
    }


def verify_sof_limit(fluid: GiesekusFluid, gdot_test: float = 1e-3) -> dict:
    """
    Verify that Giesekus reduces to SOF at low Wi.
    
    Parameters
    ----------
    fluid : GiesekusFluid
        Fluid to verify
    gdot_test : float
        Low shear rate for testing (should give Wi ≪ 1)
    
    Returns
    -------
    dict
        Comparison between Giesekus and SOF at low Wi
    """
    sof = SOFFluid.from_giesekus(fluid)
    
    Wi = fluid.lambda_ * gdot_test
    
    eta_giesekus = fluid.viscosity(gdot_test)
    eta_sof = sof.eta_0
    
    Psi1_giesekus = fluid.Psi1(gdot_test)
    Psi1_sof = sof.Psi1_0
    
    Psi2_giesekus = fluid.Psi2(gdot_test)
    Psi2_sof = sof.Psi2_0
    
    return {
        'Wi': Wi,
        'eta': {'giesekus': eta_giesekus, 'sof': eta_sof, 
                'error': abs(eta_giesekus - eta_sof) / eta_sof},
        'Psi1': {'giesekus': Psi1_giesekus, 'sof': Psi1_sof,
                 'error': abs(Psi1_giesekus - Psi1_sof) / Psi1_sof if Psi1_sof > 0 else 0},
        'Psi2': {'giesekus': Psi2_giesekus, 'sof': Psi2_sof,
                 'error': abs(Psi2_giesekus - Psi2_sof) / abs(Psi2_sof) if abs(Psi2_sof) > 0 else 0}
    }
