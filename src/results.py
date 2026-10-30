"""
Solver Results and Post-Processing
===================================

Dataclasses for structured solver output and post-processing utilities.

Key Classes:
    - FluxDecomposition: Individual flux components (GNF, memory, N₁, hoop)
    - SolverResult: Complete solver output with convergence info

References:
    - Dhlamini, N.C. (2025). PhD Thesis, Chapter 4.

Author: Nigel C. Dhlamini
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class FluxDecomposition:
    """
    Decomposition of radial volume flux into physical mechanisms.
    
    From Eq. 63:
        Q_r = Q_r^GNF + Q_r^mem + Q_r^α + Q_r^N₁ + Q_r^hoop
    
    Attributes
    ----------
    Q_r_GNF : ndarray
        Generalised Newtonian flux (Couette + Poiseuille) [m²/s]
    Q_r_mem : ndarray
        Squeeze-film memory correction [m²/s]
    Q_r_alpha : ndarray
        DEPRECATED (2026-06-28): Giesekus coupling term, no longer summed into
        Q_r_total/Q_r_viscoelastic (folded into M(α,Λ) inside Q_r_mem). Retained
        as a (zeroed) field for back-compat; the standalone term was
        dimensionally inconsistent and corrupted Q_leak. [m²/s]
    Q_r_N1 : ndarray
        First normal stress gradient contribution [m²/s]
    Q_r_hoop : ndarray
        Hoop stress contribution [m²/s]
    
    Notes
    -----
    All arrays have shape (Nr, Ntheta) matching the mesh grid.
    """
    Q_r_GNF: np.ndarray
    Q_r_mem: np.ndarray
    Q_r_alpha: np.ndarray
    Q_r_N1: np.ndarray
    Q_r_hoop: np.ndarray
    
    # Optional: θ-direction fluxes
    Q_theta_GNF: Optional[np.ndarray] = None
    Q_theta_mem: Optional[np.ndarray] = None
    
    @property
    def Q_r_total(self) -> np.ndarray:
        """Total radial flux (v6 four-term; Q_r_alpha REMOVED 2026-06-28).

        The Giesekus coupling is folded into M(α,Λ) inside Q_r_mem; the
        standalone Q_r_alpha term was dimensionally inconsistent and blew up
        Q_leak, so it is excluded here regardless of its stored value."""
        return (self.Q_r_GNF + self.Q_r_mem +
                self.Q_r_N1 + self.Q_r_hoop)

    @property
    def Q_r_viscoelastic(self) -> np.ndarray:
        """Total viscoelastic contribution (everything except GNF; v6 four-term,
        Q_r_alpha excluded -- folded into Q_r_mem)."""
        return self.Q_r_mem + self.Q_r_N1 + self.Q_r_hoop
    
    def flux_ratios(self) -> Dict[str, float]:
        """
        Compute ratio of each flux component to total.
        
        Returns
        -------
        dict
            Ratios (based on L2 norms)
        """
        Q_total_norm = np.linalg.norm(self.Q_r_total)
        
        if Q_total_norm < 1e-15:
            return {name: 0.0 for name in 
                    ['GNF', 'memory', 'alpha', 'N1', 'hoop', 'viscoelastic']}
        
        return {
            'GNF': np.linalg.norm(self.Q_r_GNF) / Q_total_norm,
            'memory': np.linalg.norm(self.Q_r_mem) / Q_total_norm,
            'alpha': np.linalg.norm(self.Q_r_alpha) / Q_total_norm,
            'N1': np.linalg.norm(self.Q_r_N1) / Q_total_norm,
            'hoop': np.linalg.norm(self.Q_r_hoop) / Q_total_norm,
            'viscoelastic': np.linalg.norm(self.Q_r_viscoelastic) / Q_total_norm,
        }


@dataclass
class SolverResult:
    """
    Complete output from the Giesekus Reynolds equation solver.
    
    Attributes
    ----------
    pressure : ndarray
        Pressure field p(r, θ), shape (Nr, Ntheta) [Pa]
    converged : bool
        Whether the solver converged
    iterations : int
        Number of iterations required
    residual_history : list
        Residual at each iteration
    
    fluxes : FluxDecomposition, optional
        Decomposed flux components
    
    F_total : float
        Total load capacity (pressure integral) [N]
    F_GNF : float
        Load from GNF pressure only [N]
    F_elastic : float
        Elastic enhancement F_total - F_GNF [N]
    
    M_x : float
        Moment about x-axis [N·m]
    M_y : float  
        Moment about y-axis [N·m]
    
    Q_leak : float
        Total leakage flow rate [m³/s]
    
    eta_bar : ndarray
        Gap-averaged effective viscosity field [Pa·s]
    gdot_bar : ndarray
        Gap-averaged shear rate field [s⁻¹]
    
    solver_info : dict
        Additional solver diagnostics
    """
    # Primary outputs
    pressure: np.ndarray
    converged: bool
    iterations: int
    residual_history: List[float] = field(default_factory=list)
    
    # Flux decomposition
    fluxes: Optional[FluxDecomposition] = None
    
    # Integrated quantities (computed in post-processing)
    F_total: float = 0.0
    F_GNF: float = 0.0
    F_elastic: float = 0.0
    M_x: float = 0.0
    M_y: float = 0.0
    Q_leak: float = 0.0
    
    # Field quantities
    eta_bar: Optional[np.ndarray] = None
    gdot_bar: Optional[np.ndarray] = None
    
    # Metadata
    solver_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def p_max(self) -> float:
        """Maximum pressure in the domain [Pa]."""
        return float(np.max(self.pressure))
    
    @property
    def p_min(self) -> float:
        """Minimum pressure in the domain [Pa]."""
        return float(np.min(self.pressure))
    
    @property
    def elastic_enhancement(self) -> float:
        """
        Elastic load enhancement ratio.
        
        Returns
        -------
        ratio : float
            F_elastic / F_GNF (fractional enhancement)
        """
        if abs(self.F_GNF) > 1e-10:
            return self.F_elastic / self.F_GNF
        return 0.0
    
    @property
    def final_residual(self) -> float:
        """Final iteration residual."""
        if self.residual_history:
            return self.residual_history[-1]
        return np.nan
    
    def summary(self) -> str:
        """Return formatted summary of results."""
        lines = [
            "Solver Results",
            "=" * 50,
            f"Converged                : {'Yes' if self.converged else 'No'}",
            f"Iterations               : {self.iterations}",
            f"Final residual           : {self.final_residual:.2e}",
            "-" * 50,
            f"Load capacity F          : {self.F_total:.4f} N",
            f"  GNF contribution       : {self.F_GNF:.4f} N",
            f"  Elastic enhancement    : {self.F_elastic:.4f} N ({self.elastic_enhancement*100:.1f}%)",
            "-" * 50,
            f"Moment M_x               : {self.M_x:.6f} N·m",
            f"Moment M_y               : {self.M_y:.6f} N·m",
            f"Leakage Q                : {self.Q_leak*1e9:.4f} mm³/s",
            "-" * 50,
            f"Pressure range           : [{self.p_min/1e6:.3f}, {self.p_max/1e6:.3f}] MPa",
        ]
        return "\n".join(lines)


# =============================================================================
# Post-Processing Functions
# =============================================================================

def compute_load_capacity(pressure: np.ndarray, mesh) -> float:
    """
    Compute total load capacity from pressure field.
    
    F = ∫∫ p(r,θ) r dr dθ
    
    Parameters
    ----------
    pressure : ndarray
        Pressure field, shape (Nr, Ntheta) [Pa]
    mesh : PolarMesh
        Mesh object for integration
    
    Returns
    -------
    F : float
        Total load capacity [N]
    """
    return mesh.integrate_field(pressure)


def compute_moments(pressure: np.ndarray, mesh) -> tuple:
    """
    Compute moments about x and y axes from pressure field.
    
    M_x = ∫∫ p(r,θ) · r · sin(θ) · r dr dθ
    M_y = ∫∫ p(r,θ) · r · cos(θ) · r dr dθ
    
    Parameters
    ----------
    pressure : ndarray
        Pressure field [Pa]
    mesh : PolarMesh
        Mesh object
    
    Returns
    -------
    M_x, M_y : float
        Moments about x and y axes [N·m]
    """
    R, THETA = mesh.R, mesh.THETA
    
    # Moment arms
    moment_arm_x = R * np.sin(THETA)
    moment_arm_y = R * np.cos(THETA)
    
    M_x = mesh.integrate_field(pressure * moment_arm_x)
    M_y = mesh.integrate_field(pressure * moment_arm_y)
    
    return M_x, M_y


def compute_leakage(flux_r: np.ndarray, mesh, boundary: str = 'outer') -> float:
    """
    Compute leakage flow rate at specified boundary.
    
    Q = ∫ Q_r(r_boundary, θ) · r_boundary dθ
    
    Parameters
    ----------
    flux_r : ndarray
        Radial flux field [m²/s]
    mesh : PolarMesh
        Mesh object
    boundary : str
        'outer' or 'inner'
    
    Returns
    -------
    Q : float
        Volumetric flow rate [m³/s]. Positive = outward.
    """
    if boundary == 'outer':
        idx = -1
        r_boundary = mesh.r[-1]
    else:
        idx = 0
        r_boundary = mesh.r[0]
    
    # Line integral
    Q_line = flux_r[idx, :]  # Flux at boundary
    Q = np.sum(Q_line) * r_boundary * mesh.dtheta
    
    return Q


def post_process_result(result: SolverResult, mesh) -> SolverResult:
    """
    Compute integrated quantities from raw solver output.
    
    Parameters
    ----------
    result : SolverResult
        Raw solver result with pressure field
    mesh : PolarMesh
        Mesh object
    
    Returns
    -------
    result : SolverResult
        Result with computed F_total, M_x, M_y, Q_leak
    """
    result.F_total = compute_load_capacity(result.pressure, mesh)
    result.M_x, result.M_y = compute_moments(result.pressure, mesh)
    
    if result.fluxes is not None:
        result.Q_leak = compute_leakage(result.fluxes.Q_r_total, mesh, 'outer')
    
    return result
