"""
Dynamic Analysis Module for Viscoelastic Slipper Bearings
==========================================================

Consolidated frequency-domain and time-domain dynamic analyses for
evaluating the quasi-steady approximation in viscoelastic lubrication.

Provides:
    - Complex Reynolds equation solver (linearised squeeze-film)
    - Frequency sweep for dynamic stiffness K(omega) and damping C(omega)
    - Time-domain squeeze-film solver with memory-corrected diffusion
    - Timescale hierarchy and error decomposition
    - Parameter space mapping of memory effects

Consolidates and replaces the standalone scripts:
    supplementary/dynamic_coefficients.py
    supplementary/frequency_domain_error_analysis.py
    supplementary/dynamic_analysis.py

References:
    - Dhlamini, N.C. (2025). PhD Thesis, Chapter 4.
    - Bird et al. (1987). Dynamics of Polymeric Liquids, Vol. 1.
    - Hamrock et al. (2004). Fundamentals of Fluid Film Lubrication.

Author: Nigel C. Dhlamini
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Union
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .rheology import GiesekusFluid, create_newtonian, create_PAM_2pct, create_PAM_5pct

ArrayLike = Union[float, np.ndarray]


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class DynamicsConfig:
    """
    Configuration for 1D axisymmetric dynamic analyses.

    These analyses use a simplified radial geometry (no tilt, no theta)
    appropriate for linearised squeeze-film problems.

    Parameters
    ----------
    R_in : float
        Inner radius [m]
    R_out : float
        Outer radius [m]
    h_0 : float
        Mean gap height [m]
    U : float
        Tangential sliding velocity [m/s]
    omega_op : float
        Pump angular speed [rad/s]
    rho : float
        Fluid density [kg/m^3]
    Nr : int
        Number of radial grid points
    """
    R_in: float = 0.010
    R_out: float = 0.020
    h_0: float = 5e-6
    U: float = 10.0
    omega_op: float = 157.0
    rho: float = 1050.0
    Nr: int = 300

    @property
    def L(self) -> float:
        """Pad width [m]."""
        return self.R_out - self.R_in

    @property
    def St_op(self) -> float:
        """Operating Strouhal number."""
        return self.omega_op * self.L / self.U

    @classmethod
    def from_bearing(cls, geometry, conditions, rho: float = 1050.0,
                     Nr: int = 300) -> 'DynamicsConfig':
        """
        Create from SlipperGeometry and OperatingConditions.

        Parameters
        ----------
        geometry : SlipperGeometry
            Bearing geometry object
        conditions : OperatingConditions
            Operating conditions object
        rho : float
            Fluid density [kg/m^3]
        Nr : int
            Radial grid points
        """
        return cls(
            R_in=geometry.R_in, R_out=geometry.R_out, h_0=geometry.h_0,
            U=conditions.V_T, omega_op=2 * np.pi * 25,
            rho=rho, Nr=Nr
        )


# =============================================================================
# Result dataclasses
# =============================================================================
@dataclass
class ComplexReynoldsResult:
    """Result from the linearised complex Reynolds equation solver."""
    r: np.ndarray
    p_hat_QS: np.ndarray
    p_hat_VE: np.ndarray
    F_hat_QS: complex
    F_hat_VE: complex
    K_QS: float
    K_VE: float
    C_QS: float
    C_VE: float
    amplitude_ratio: float
    phase_lag: float
    eta_eff: complex
    omega: float


@dataclass
class FrequencySweepResult:
    """Result from a frequency sweep analysis."""
    St: np.ndarray
    omega: np.ndarray
    amp_ratio: np.ndarray
    phase_lag: np.ndarray
    K_QS: np.ndarray
    K_VE: np.ndarray
    C_QS: np.ndarray
    C_VE: np.ndarray


@dataclass
class TimescaleReport:
    """Timescale hierarchy for a given fluid and configuration."""
    lambda_: float
    t_transit: float
    T_pump: float
    De: float
    St: float
    omega_lambda: float
    residual_transit: float
    residual_cycle: float
    Re_sq: float


@dataclass
class OperatingPointResult:
    """Error analysis at a specific operating frequency."""
    amp_mod: float
    phase_lag_deg: float
    omega_lambda: float
    De_sq_truncation: float
    VE_memory_amp: float
    VE_memory_phase: float
    Re_sq: float


@dataclass
class TimeDomainResult:
    """Result from time-domain squeeze-film analysis."""
    t_norm: np.ndarray
    F_sq_GNF: np.ndarray
    F_sq_VE: np.ndarray
    A_GNF: float
    A_VE: float
    phi_GNF: float
    phi_VE: float
    ratio: float
    phase_diff: float
    De_sq_max: float
    h_t: np.ndarray
    h_dot: np.ndarray


@dataclass
class ParameterSpaceResult:
    """Result from parameter space analysis."""
    St_arr: np.ndarray
    eps_arr: np.ndarray
    De_sq_grid: np.ndarray
    memory_effect: np.ndarray


# =============================================================================
# Complex Reynolds equation solver
# =============================================================================
def solve_complex_reynolds(fluid: GiesekusFluid, omega: float,
                           config: Optional[DynamicsConfig] = None
                           ) -> ComplexReynoldsResult:
    """
    Solve the linearised complex Reynolds equation for annular squeeze.

    For small-amplitude gap oscillation h(t) = h_0 + h_1 exp(i omega t),
    the pressure perturbation satisfies:

        (1/r) d/dr [r h_0^3/(12 eta_eff) dp_hat/dr] = i omega h_1

    Solves for both quasi-steady (eta_0) and viscoelastic (eta*(omega))
    cases in a single call.

    Parameters
    ----------
    fluid : GiesekusFluid
        Fluid properties
    omega : float
        Angular frequency [rad/s]
    config : DynamicsConfig, optional
        Configuration (uses defaults if not provided)

    Returns
    -------
    ComplexReynoldsResult
        Contains QS and VE pressure fields, forces, dynamic coefficients
    """
    if config is None:
        config = DynamicsConfig()

    h0, Nr = config.h_0, config.Nr
    h1 = 1e-9  # arbitrary perturbation (cancels in ratios)
    S = 1j * omega * h1

    r = np.linspace(config.R_in, config.R_out, Nr)
    dr = r[1] - r[0]
    N = Nr - 2  # interior points

    results = {}
    for label, eta_eff in [('QS', complex(fluid.eta_0, 0.0)),
                           ('VE', fluid.complex_viscosity(omega))]:
        D = h0**3 / (12.0 * eta_eff)

        # Build sparse tridiagonal system
        lower = np.zeros(N - 1, dtype=complex)
        diag = np.zeros(N, dtype=complex)
        upper = np.zeros(N - 1, dtype=complex)
        b = np.full(N, S, dtype=complex)

        for i in range(N):
            idx = i + 1
            ri = r[idx]
            rp = 0.5 * (r[idx] + r[idx + 1])
            rm = 0.5 * (r[idx - 1] + r[idx])
            cp = D * rp / (ri * dr**2)
            cm = D * rm / (ri * dr**2)
            diag[i] = -(cp + cm)
            if i > 0:
                lower[i - 1] = cm
            if i < N - 1:
                upper[i] = cp

        A = diags([lower, diag, upper], [-1, 0, 1], format='csr')
        p_hat = np.zeros(Nr, dtype=complex)
        p_hat[1:-1] = spsolve(A, b)
        F_hat = 2 * np.pi * np.trapezoid(p_hat * r, r)

        ratio = F_hat / h1
        K = -ratio.real
        C = -ratio.imag / omega if omega > 0 else 0.0

        results[label] = {'p_hat': p_hat, 'F_hat': F_hat, 'K': K, 'C': C,
                          'eta_eff': eta_eff}

    # Relative quantities
    F_QS = results['QS']['F_hat']
    F_VE = results['VE']['F_hat']
    amp_ratio = abs(F_VE) / abs(F_QS) if abs(F_QS) > 0 else 1.0
    phase = np.angle(F_VE) - np.angle(F_QS)
    phase = (phase + np.pi) % (2 * np.pi) - np.pi

    return ComplexReynoldsResult(
        r=r,
        p_hat_QS=results['QS']['p_hat'],
        p_hat_VE=results['VE']['p_hat'],
        F_hat_QS=F_QS, F_hat_VE=F_VE,
        K_QS=results['QS']['K'], K_VE=results['VE']['K'],
        C_QS=results['QS']['C'], C_VE=results['VE']['C'],
        amplitude_ratio=amp_ratio, phase_lag=phase,
        eta_eff=results['VE']['eta_eff'], omega=omega
    )


# =============================================================================
# Squeeze-film solver (time-domain, with memory correction)
# =============================================================================
class SqueezeSolver:
    """
    Solves the squeeze-film Reynolds equation on an annular domain.

    Includes an optional memory-corrected diffusion coefficient from the
    first-order De_sq expansion of the viscoelastic constitutive equation.

    Parameters
    ----------
    R_in : float
        Inner radius [m]
    R_out : float
        Outer radius [m]
    Nr : int
        Number of radial grid points
    """

    def __init__(self, R_in: float, R_out: float, Nr: int = 200):
        self.Nr = Nr
        self.R_in = R_in
        self.R_out = R_out
        self.r = np.linspace(R_in, R_out, Nr)
        self.dr = self.r[1] - self.r[0]

    def solve(self, eta_field: np.ndarray, h: np.ndarray, h_dot: float,
              p_in: float, p_out: float = 0.0,
              lam: float = 0.0, eta_T_field: Optional[np.ndarray] = None,
              dp_dr_prev: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve: (1/r) d/dr [r D_eff dp/dr] = h_dot

        where D_eff = h^3/(12*eta) + memory correction.

        Parameters
        ----------
        eta_field : ndarray
            Viscosity field along radial direction [Pa.s]
        h : ndarray
            Gap height field [m]
        h_dot : float
            Rate of gap change [m/s]
        p_in : float
            Pressure at inner boundary [Pa]
        p_out : float
            Pressure at outer boundary [Pa]
        lam : float
            Relaxation time [s] (0 for GNF)
        eta_T_field : ndarray, optional
            Tangent viscosity field [Pa.s]
        dp_dr_prev : ndarray, optional
            Previous pressure gradient (unused, reserved)

        Returns
        -------
        p : ndarray
            Pressure field [Pa]
        """
        Nr = self.Nr
        r = self.r
        dr = self.dr
        D = h**3 / (12 * eta_field)

        # Memory correction to diffusion (from De_sq expansion)
        if lam > 0 and eta_T_field is not None and h_dot != 0:
            D_mem = lam * h**2 * abs(h_dot) * eta_T_field / (16 * eta_field**2)
            if h_dot < 0:
                D_eff = D + D_mem   # closing: memory adds to diffusion
            else:
                D_eff = D - D_mem   # opening: memory subtracts
        else:
            D_eff = D

        # Build FD operator
        D_hp = 0.5 * (D_eff[:-1] + D_eff[1:])
        r_hp = 0.5 * (r[:-1] + r[1:])
        al = np.zeros(Nr)
        ad = np.zeros(Nr)
        au = np.zeros(Nr)
        for i in range(1, Nr - 1):
            rDp = r_hp[i] * D_hp[i]
            rDm = r_hp[i - 1] * D_hp[i - 1] if i > 0 else r[0] * D_eff[0]
            pf = 1 / (r[i] * dr**2)
            al[i] = pf * rDm
            au[i] = pf * rDp
            ad[i] = -pf * (rDp + rDm)
        ad[0] = 1
        ad[-1] = 1
        A = diags([al[1:], ad, au[:-1]], [-1, 0, 1], format='csr')
        rhs = np.full(Nr, h_dot)
        rhs[0] = p_in
        rhs[-1] = p_out
        return spsolve(A, rhs)

    def load(self, p: np.ndarray) -> float:
        """Integrate pressure to compute axial force [N]."""
        return 2 * np.pi * np.trapezoid(p * self.r, self.r)


# =============================================================================
# Frequency sweep
# =============================================================================
def frequency_sweep(fluid: GiesekusFluid,
                    config: Optional[DynamicsConfig] = None,
                    St_range: Optional[np.ndarray] = None
                    ) -> FrequencySweepResult:
    """
    Sweep over frequency to compute dynamic coefficients.

    Parameters
    ----------
    fluid : GiesekusFluid
        Fluid properties
    config : DynamicsConfig, optional
        Configuration (uses defaults if not provided)
    St_range : ndarray, optional
        Strouhal number range (default: logspace(-2, 1.5, 150))

    Returns
    -------
    FrequencySweepResult
        Amplitude ratios, phase lags, K and C vs frequency
    """
    if config is None:
        config = DynamicsConfig()
    if St_range is None:
        St_range = np.logspace(-2, 1.5, 150)

    omega_range = St_range * config.U / config.L
    n = len(St_range)

    amp_ratio = np.zeros(n)
    phase_lag = np.zeros(n)
    K_QS = np.zeros(n)
    K_VE = np.zeros(n)
    C_QS = np.zeros(n)
    C_VE = np.zeros(n)

    for i, (St, w) in enumerate(zip(St_range, omega_range)):
        res = solve_complex_reynolds(fluid, w, config)
        amp_ratio[i] = res.amplitude_ratio
        phase_lag[i] = res.phase_lag
        K_QS[i] = res.K_QS
        K_VE[i] = res.K_VE
        C_QS[i] = res.C_QS
        C_VE[i] = res.C_VE

    return FrequencySweepResult(
        St=St_range, omega=omega_range,
        amp_ratio=amp_ratio, phase_lag=phase_lag,
        K_QS=K_QS, K_VE=K_VE, C_QS=C_QS, C_VE=C_VE
    )


# =============================================================================
# Timescale analysis
# =============================================================================
def compute_timescales(fluid: GiesekusFluid,
                       config: Optional[DynamicsConfig] = None
                       ) -> TimescaleReport:
    """
    Compute timescale hierarchy for a given fluid and configuration.

    Parameters
    ----------
    fluid : GiesekusFluid
        Fluid properties
    config : DynamicsConfig, optional
        Configuration (uses defaults if not provided)

    Returns
    -------
    TimescaleReport
        All relevant timescales and dimensionless numbers
    """
    if config is None:
        config = DynamicsConfig()

    lam = fluid.lambda_
    L = config.L
    U = config.U
    omega = config.omega_op

    t_transit = L / U
    T_pump = 2 * np.pi / omega
    De = lam * U / L if lam > 0 else 0.0
    St = omega * L / U
    wl = omega * lam
    Re_sq = config.rho * config.h_0**2 * omega / fluid.eta_0

    residual_transit = np.exp(-t_transit / lam) if lam > 0 else 0.0
    residual_cycle = np.exp(-T_pump / lam) if lam > 0 else 0.0

    return TimescaleReport(
        lambda_=lam, t_transit=t_transit, T_pump=T_pump,
        De=De, St=St, omega_lambda=wl,
        residual_transit=residual_transit,
        residual_cycle=residual_cycle,
        Re_sq=Re_sq
    )


# =============================================================================
# Operating point analysis
# =============================================================================
def operating_point_analysis(fluid: GiesekusFluid,
                             config: Optional[DynamicsConfig] = None,
                             omega: Optional[float] = None
                             ) -> OperatingPointResult:
    """
    Compute error decomposition at a specific operating frequency.

    Parameters
    ----------
    fluid : GiesekusFluid
        Fluid properties
    config : DynamicsConfig, optional
        Configuration (uses defaults if not provided)
    omega : float, optional
        Operating frequency [rad/s] (default: config.omega_op)

    Returns
    -------
    OperatingPointResult
        Amplitude modification, phase lag, and error source breakdown
    """
    if config is None:
        config = DynamicsConfig()
    if omega is None:
        omega = config.omega_op

    res = solve_complex_reynolds(fluid, omega, config)
    wl = omega * fluid.lambda_
    Re_sq = config.rho * config.h_0**2 * omega / fluid.eta_0

    amp_mod = (1 - res.amplitude_ratio) * 100
    phase_deg = np.degrees(res.phase_lag)

    # De_sq truncation estimate (from thesis Ch. 4)
    De = fluid.lambda_ * config.U / config.L if fluid.lambda_ > 0 else 0.0
    De_sq_trunc = De**2 * 100 if De > 0 else 0.0

    return OperatingPointResult(
        amp_mod=amp_mod, phase_lag_deg=phase_deg,
        omega_lambda=wl, De_sq_truncation=De_sq_trunc,
        VE_memory_amp=abs(amp_mod), VE_memory_phase=abs(phase_deg),
        Re_sq=Re_sq
    )


# =============================================================================
# Time-domain squeeze-film analysis
# =============================================================================
def time_domain_analysis(fluid: GiesekusFluid,
                         config: Optional[DynamicsConfig] = None,
                         eps_h: float = 0.05,
                         p_supply: float = 200e5,
                         Nt: int = 200) -> TimeDomainResult:
    """
    Compare quasi-steady vs memory-corrected squeeze-film response.

    Computes the load over one sinusoidal squeeze cycle:
        h(t) = h_0 [1 + eps_h * sin(omega * t)]

    Parameters
    ----------
    fluid : GiesekusFluid
        Fluid properties
    config : DynamicsConfig, optional
        Configuration (uses defaults if not provided)
    eps_h : float
        Relative gap oscillation amplitude
    p_supply : float
        Supply pressure [Pa]
    Nt : int
        Number of time steps per cycle

    Returns
    -------
    TimeDomainResult
        GNF and VE squeeze forces, amplitudes, phases
    """
    if config is None:
        config = DynamicsConfig()

    h0 = config.h_0
    omega = config.omega_op
    V_T = config.U
    solver = SqueezeSolver(config.R_in, config.R_out, Nr=min(config.Nr, 200))
    Nr = solver.Nr

    gdot_mean = V_T / h0
    eta_mean = fluid.viscosity(gdot_mean)
    if isinstance(eta_mean, np.ndarray):
        eta_mean = float(eta_mean[0])
    eta_T_mean = float(fluid.tangent_viscosity(gdot_mean))
    if isinstance(eta_T_mean, np.ndarray):
        eta_T_mean = float(eta_T_mean[0]) if len(eta_T_mean) > 0 else eta_mean
    De_sq_max = fluid.lambda_ * eps_h * omega if fluid.lambda_ > 0 else 0.0

    T = 2 * np.pi / omega
    t_arr = np.linspace(0, T, Nt)

    F_GNF = np.zeros(Nt)
    F_VE = np.zeros(Nt)
    F_hydrostatic = np.zeros(Nt)

    for j in range(Nt):
        h_t = h0 * (1 + eps_h * np.sin(omega * t_arr[j]))
        h_dot_t = h0 * eps_h * omega * np.cos(omega * t_arr[j])
        h_field = np.full(Nr, h_t)
        eta_field = np.full(Nr, eta_mean)
        eta_T_field = np.full(Nr, eta_T_mean)

        # Hydrostatic (no squeeze)
        p_static = solver.solve(eta_field, h_field, 0.0, p_supply, 0.0)
        F_hydrostatic[j] = solver.load(p_static)

        # GNF squeeze (no memory)
        p_gnf = solver.solve(eta_field, h_field, h_dot_t, p_supply, 0.0)
        F_GNF[j] = solver.load(p_gnf)

        # VE with memory
        p_ve = solver.solve(eta_field, h_field, h_dot_t, p_supply, 0.0,
                            lam=fluid.lambda_, eta_T_field=eta_T_field)
        F_VE[j] = solver.load(p_ve)

    # Squeeze-only forces
    F_sq_GNF = F_GNF - F_hydrostatic
    F_sq_VE = F_VE - F_hydrostatic

    # Fourier analysis for amplitude and phase
    F_hat_GNF = 2 / T * np.trapezoid(
        F_sq_GNF * np.exp(-1j * omega * t_arr), t_arr)
    F_hat_VE = 2 / T * np.trapezoid(
        F_sq_VE * np.exp(-1j * omega * t_arr), t_arr)

    A_GNF = np.abs(F_hat_GNF)
    phi_GNF = np.angle(F_hat_GNF)
    A_VE = np.abs(F_hat_VE)
    phi_VE = np.angle(F_hat_VE)

    ratio = A_VE / A_GNF if A_GNF > 0 else 1.0
    phase_diff = np.degrees(phi_VE - phi_GNF)

    return TimeDomainResult(
        t_norm=t_arr / T,
        F_sq_GNF=F_sq_GNF, F_sq_VE=F_sq_VE,
        A_GNF=A_GNF, A_VE=A_VE,
        phi_GNF=phi_GNF, phi_VE=phi_VE,
        ratio=ratio, phase_diff=phase_diff,
        De_sq_max=De_sq_max,
        h_t=h0 * (1 + eps_h * np.sin(omega * t_arr)),
        h_dot=h0 * eps_h * omega * np.cos(omega * t_arr)
    )


# =============================================================================
# Parameter space analysis
# =============================================================================
def parameter_space_analysis(fluid: GiesekusFluid,
                             config: Optional[DynamicsConfig] = None,
                             St_arr: Optional[np.ndarray] = None,
                             eps_arr: Optional[np.ndarray] = None,
                             p_supply: float = 200e5
                             ) -> ParameterSpaceResult:
    """
    Map the quasi-steady error across (St, De_sq) parameter space.

    Parameters
    ----------
    fluid : GiesekusFluid
        Fluid properties (should be viscoelastic)
    config : DynamicsConfig, optional
        Configuration (uses defaults if not provided)
    St_arr : ndarray, optional
        Strouhal number array
    eps_arr : ndarray, optional
        Relative gap amplitude array
    p_supply : float
        Supply pressure [Pa]

    Returns
    -------
    ParameterSpaceResult
        Memory effect magnitude across parameter space
    """
    if config is None:
        config = DynamicsConfig()
    if St_arr is None:
        St_arr = np.array([0.01, 0.05, 0.1, 0.157, 0.3, 0.5, 1.0, 2.0, 5.0])
    if eps_arr is None:
        eps_arr = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])

    h0 = config.h_0
    V_T = config.U
    solver = SqueezeSolver(config.R_in, config.R_out, Nr=100)
    Nr = solver.Nr

    gdot_mean = V_T / h0
    eta_mean = fluid.viscosity(gdot_mean)
    if isinstance(eta_mean, np.ndarray):
        eta_mean = float(eta_mean[0])
    eta_T_mean = float(fluid.tangent_viscosity(gdot_mean))
    if isinstance(eta_T_mean, np.ndarray):
        eta_T_mean = float(eta_T_mean[0]) if len(eta_T_mean) > 0 else eta_mean

    memory_effect = np.zeros((len(St_arr), len(eps_arr)))
    De_sq_grid = np.zeros((len(St_arr), len(eps_arr)))

    for i, St in enumerate(St_arr):
        omega = St * V_T / config.L
        for j, eps_h in enumerate(eps_arr):
            De_sq_grid[i, j] = fluid.lambda_ * eps_h * omega
            Nt = 100
            T = 2 * np.pi / omega
            t_arr = np.linspace(0, T, Nt)

            F_sq_GNF = np.zeros(Nt)
            F_sq_VE = np.zeros(Nt)

            for k in range(Nt):
                h_t = h0 * (1 + eps_h * np.sin(omega * t_arr[k]))
                h_dot_t = h0 * eps_h * omega * np.cos(omega * t_arr[k])
                h_f = np.full(Nr, h_t)
                eta_f = np.full(Nr, eta_mean)
                etaT_f = np.full(Nr, eta_T_mean)

                p_static = solver.solve(eta_f, h_f, 0.0, p_supply, 0.0)
                p_gnf = solver.solve(eta_f, h_f, h_dot_t, p_supply, 0.0)
                p_ve = solver.solve(eta_f, h_f, h_dot_t, p_supply, 0.0,
                                    lam=fluid.lambda_, eta_T_field=etaT_f)
                F_sq_GNF[k] = solver.load(p_gnf) - solver.load(p_static)
                F_sq_VE[k] = solver.load(p_ve) - solver.load(p_static)

            A_GNF = np.max(np.abs(F_sq_GNF))
            if A_GNF > 0:
                memory_effect[i, j] = (
                    1 - np.max(np.abs(F_sq_VE)) / A_GNF) * 100
            else:
                memory_effect[i, j] = 0

    return ParameterSpaceResult(
        St_arr=St_arr, eps_arr=eps_arr,
        De_sq_grid=De_sq_grid, memory_effect=memory_effect
    )


# =============================================================================
# Plotting functions
# =============================================================================
def plot_complex_viscosity(fluids: Dict[str, GiesekusFluid],
                           config: Optional[DynamicsConfig] = None,
                           output_dir: Optional[str] = None):
    """Plot complex viscosity components eta'(omega) and eta''(omega)."""
    import matplotlib.pyplot as plt

    if config is None:
        config = DynamicsConfig()

    cols = _default_colors(fluids)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    wl_arr = np.logspace(-3, 2, 500)

    for name, fl in fluids.items():
        if fl.lambda_ == 0:
            continue
        omega_arr = wl_arr / fl.lambda_
        ep = fl.eta_prime(omega_arr)
        epp = fl.eta_double_prime(omega_arr)

        ax1.loglog(wl_arr, ep / fl.eta_0, color=cols[name], lw=2,
                   label=fr"{name}: $\eta'/\eta_0$")
        ax1.loglog(wl_arr, epp / fl.eta_0, color=cols[name], lw=2,
                   ls='--', label=fr"{name}: $\eta''/\eta_0$")

        wl_op = config.omega_op * fl.lambda_
        ax1.axvline(wl_op, color=cols[name], ls=':', alpha=0.4)

        with np.errstate(divide='ignore', invalid='ignore'):
            loss_tangent = np.where(epp > 0, epp / ep, 0)
        ax2.loglog(wl_arr, loss_tangent, color=cols[name], lw=2, label=name)
        ax2.axvline(wl_op, color=cols[name], ls=':', alpha=0.4)

    ax1.set_xlabel(r'$\omega\lambda$')
    ax1.set_ylabel(r'$\eta/\eta_0$')
    ax1.set_title(r"(a) Dynamic viscosity components")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, which='both')
    ax1.set_xlim(1e-3, 100)

    ax2.set_xlabel(r'$\omega\lambda$')
    ax2.set_ylabel(r"$\eta''/\eta'$ (loss tangent)")
    ax2.set_title(r"(b) Loss tangent")
    ax2.axhline(1, color='gray', ls='--', alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, which='both')
    ax2.set_xlim(1e-3, 100)

    fig.tight_layout()
    _save_or_show(fig, 'complex_viscosity', output_dir)
    return fig


def plot_amplitude_phase(sweep_results: Dict[str, FrequencySweepResult],
                         config: Optional[DynamicsConfig] = None,
                         fluids: Optional[Dict[str, GiesekusFluid]] = None,
                         output_dir: Optional[str] = None):
    """Plot amplitude modification and phase lag vs Strouhal number."""
    import matplotlib.pyplot as plt

    if config is None:
        config = DynamicsConfig()

    cols = _default_colors(sweep_results)
    St_op = config.St_op
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, r in sweep_results.items():
        wl_label = ''
        if fluids and name in fluids and fluids[name].lambda_ > 0:
            wl_label = fr" ($\omega\lambda_{{op}}$={config.omega_op * fluids[name].lambda_:.2f})"
        ax1.semilogx(r.St, (1 - r.amp_ratio) * 100, color=cols[name],
                     lw=2, label=f'{name}{wl_label}')
        ax2.semilogx(r.St, np.degrees(r.phase_lag), color=cols[name],
                     lw=2, label=name)

    for ax in [ax1, ax2]:
        ax.axvline(St_op, color='gray', ls=':', lw=1.2, alpha=0.6)
        ax.grid(True, alpha=0.2)

    ax1.set_xlabel(r'St $= \omega L/U$')
    ax1.set_ylabel('Amplitude modification [%]')
    ax1.set_title('(a) Force amplitude change due to VE memory')
    ax1.legend(fontsize=8)

    ax2.set_xlabel(r'St $= \omega L/U$')
    ax2.set_ylabel(r'Phase lag $\phi$ [degrees]')
    ax2.set_title('(b) Phase lag')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, 'amplitude_phase', output_dir)
    return fig


def plot_stiffness_damping(sweep_results: Dict[str, FrequencySweepResult],
                           config: Optional[DynamicsConfig] = None,
                           output_dir: Optional[str] = None):
    """Plot dynamic stiffness K(omega) and damping C(omega)."""
    import matplotlib.pyplot as plt

    if config is None:
        config = DynamicsConfig()

    cols = _default_colors(sweep_results)
    St_op = config.St_op
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, r in sweep_results.items():
        K0 = r.K_QS[0]
        C_ref = abs(r.C_QS[min(10, len(r.C_QS) - 1)])
        if abs(K0) > 0:
            ax1.semilogx(r.St, r.K_QS / K0, color=cols[name], lw=2,
                         label=fr'{name} ($\eta_0$)')
            ax1.semilogx(r.St, r.K_VE / K0, color=cols[name], lw=2,
                         ls='--', label=fr'{name} ($\eta^*$)')
        if C_ref > 0:
            ax2.semilogx(r.St, r.C_QS / C_ref, color=cols[name], lw=2,
                         label=fr'{name} ($\eta_0$)')
            ax2.semilogx(r.St, r.C_VE / C_ref, color=cols[name], lw=2,
                         ls='--', label=fr'{name} ($\eta^*$)')

    for ax in [ax1, ax2]:
        ax.axvline(St_op, color='gray', ls=':', lw=1.2)
        ax.grid(True, alpha=0.2)

    ax1.set_xlabel(r'St $= \omega L/U$')
    ax1.set_ylabel(r'$K(\omega)/K_0$')
    ax1.set_title('(a) Dynamic stiffness')
    ax1.legend(fontsize=8)

    ax2.set_xlabel(r'St $= \omega L/U$')
    ax2.set_ylabel(r'$C(\omega)/C_{\rm ref}$')
    ax2.set_title('(b) Damping coefficient')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, 'stiffness_damping', output_dir)
    return fig


def plot_time_domain(td_results: Dict[str, TimeDomainResult],
                     config: Optional[DynamicsConfig] = None,
                     output_dir: Optional[str] = None):
    """Plot time-domain squeeze-film response comparison."""
    import matplotlib.pyplot as plt

    if config is None:
        config = DynamicsConfig()

    cols = _default_colors(td_results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Gap oscillation (use first result for h(t))
    ax = axes[0, 0]
    first = next(iter(td_results.values()))
    ax.plot(first.t_norm, first.h_t / config.h_0, 'k-', lw=2)
    ax.set_xlabel(r'$t/T$')
    ax.set_ylabel(r'$h/h_0$')
    ax.set_title('(a) Gap oscillation')
    ax.grid(True, alpha=0.3)

    # (b) Squeeze force
    ax = axes[0, 1]
    for name, td in td_results.items():
        ax.plot(td.t_norm, td.F_sq_GNF, color=cols[name], ls='-', lw=1.5,
                label=f'{name} GNF', alpha=0.7)
        if td.De_sq_max > 0:
            ax.plot(td.t_norm, td.F_sq_VE, color=cols[name], ls='--', lw=2,
                    label=f'{name} VE')
    ax.set_xlabel(r'$t/T$')
    ax.set_ylabel('Squeeze force [N]')
    ax.set_title('(b) Squeeze force over one cycle')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) VE/GNF ratio
    ax = axes[1, 0]
    for name, td in td_results.items():
        if td.De_sq_max > 0 and np.any(td.F_sq_GNF != 0):
            mask = np.abs(td.F_sq_GNF) > 0.1 * np.max(np.abs(td.F_sq_GNF))
            ratio_t = np.where(mask, td.F_sq_VE / td.F_sq_GNF, np.nan)
            ax.plot(td.t_norm, ratio_t, color=cols[name], lw=2, label=name)
    ax.axhline(1, color='k', ls=':', alpha=0.5)
    ax.set_xlabel(r'$t/T$')
    ax.set_ylabel(r'$F_{\rm VE}/F_{\rm GNF}$')
    ax.set_title('(c) Viscoelastic memory effect')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) De_sq(t)
    ax = axes[1, 1]
    for name, td in td_results.items():
        if td.De_sq_max > 0:
            De_sq_t = td.De_sq_max * np.abs(np.cos(
                2 * np.pi * td.t_norm))
            ax.plot(td.t_norm, De_sq_t, color=cols[name], lw=2, label=name)
    ax.axhline(0.1, color='k', ls=':', alpha=0.5, label='Validity limit')
    ax.set_xlabel(r'$t/T$')
    ax.set_ylabel(r'$\mathrm{De}_{\rm sq}(t)$')
    ax.set_title(r'(d) Instantaneous $\mathrm{De}_{\rm sq}$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, 'time_domain_squeeze', output_dir)
    return fig


def plot_parameter_space(ps_result: ParameterSpaceResult,
                         output_dir: Optional[str] = None):
    """Plot parameter space map of memory effect."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Memory effect vs De_sq at selected St values
    ax = axes[0]
    select_St = [0.01, 0.1, 0.157, 0.5, 2.0]
    for i, St in enumerate(ps_result.St_arr):
        if St in select_St or np.isclose(St, select_St, atol=0.005).any():
            ax.plot(ps_result.De_sq_grid[i, :],
                    np.abs(ps_result.memory_effect[i, :]),
                    'o-', lw=1.5, ms=5, label=f'St={St:.3f}')

    De_sq_th = np.linspace(0, 0.1, 50)
    ax.plot(De_sq_th, 2 * De_sq_th * 100, 'k--', lw=1.5,
            label=r'$2\,\mathrm{De}_{\rm sq}\times 100$')
    ax.set_xlabel(r'$\mathrm{De}_{\rm sq}^{\max}$')
    ax.set_ylabel('Memory correction [%]')
    ax.set_title(r'(a) Memory effect vs $\mathrm{De}_{\rm sq}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Contour plot
    ax = axes[1]
    St_mesh, eps_mesh = np.meshgrid(
        ps_result.St_arr, ps_result.eps_arr, indexing='ij')
    cs = ax.contourf(St_mesh, ps_result.De_sq_grid,
                     np.abs(ps_result.memory_effect),
                     levels=np.linspace(0, 20, 11), cmap='YlOrRd')
    plt.colorbar(cs, ax=ax, label='|Memory correction| [%]')
    ax.set_xlabel(r'St $= \omega L/U$')
    ax.set_ylabel(r'$\mathrm{De}_{\rm sq}^{\max}$')
    ax.set_title(r'(b) Memory effect in (St, $\mathrm{De}_{\rm sq}$) space')
    ax.set_xscale('log')

    fig.tight_layout()
    _save_or_show(fig, 'parameter_space', output_dir)
    return fig


def plot_advective_memory(fluids: Dict[str, GiesekusFluid],
                          config: Optional[DynamicsConfig] = None,
                          output_dir: Optional[str] = None):
    """Plot advective stress memory decay along streamlines."""
    import matplotlib.pyplot as plt

    if config is None:
        config = DynamicsConfig()

    cols = _default_colors(fluids)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    x_norm = np.linspace(0, 1, 200)
    t_transit = x_norm * config.L / config.U

    for name, fl in fluids.items():
        if fl.lambda_ == 0:
            continue
        De = fl.lambda_ * config.U / config.L
        stress_frac = np.exp(-t_transit / fl.lambda_)
        ax1.plot(x_norm, stress_frac * 100, color=cols[name], lw=2,
                 label=fr'{name} (De = {De:.2f})')
        ax1.fill_between(x_norm, stress_frac * 100, alpha=0.1,
                         color=cols[name])

    ax1.set_xlabel(r'Position along pad $x/L$')
    ax1.set_ylabel('Residual stress [%]')
    ax1.set_title('(a) Stress memory decay along streamline')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(0, 105)

    De_arr = np.logspace(-1, 1, 200)
    residual = np.exp(-1 / De_arr) * 100
    ax2.semilogx(De_arr, residual, 'k-', lw=2)

    for name, fl in fluids.items():
        if fl.lambda_ == 0:
            continue
        De = fl.lambda_ * config.U / config.L
        res = np.exp(-1 / De) * 100
        ax2.plot(De, res, 'o', color=cols[name], ms=10, zorder=5,
                 label=fr'{name}: {res:.0f}% at De={De:.2f}')

    ax2.set_xlabel(r'De $= \lambda U / L$')
    ax2.set_ylabel('Stress at pad exit [%]')
    ax2.set_title('(b) Stress fraction surviving transit')
    ax2.axhline(10, color='gray', ls=':', alpha=0.4)
    ax2.annotate('10% threshold', xy=(0.12, 12), fontsize=8, color='gray')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    _save_or_show(fig, 'advective_memory', output_dir)
    return fig


def plot_error_bar_chart(op_results: Dict[str, OperatingPointResult],
                         config: Optional[DynamicsConfig] = None,
                         output_dir: Optional[str] = None):
    """Plot bar chart comparing error sources at the operating point."""
    import matplotlib.pyplot as plt

    if config is None:
        config = DynamicsConfig()

    names = list(op_results.keys())
    x = np.arange(len(names))
    width = 0.2

    De_sq_trunc = [r.De_sq_truncation for r in op_results.values()]
    VE_amp = [r.VE_memory_amp for r in op_results.values()]
    VE_phase = [r.VE_memory_phase for r in op_results.values()]
    inertia = [r.Re_sq * 100 for r in op_results.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 1.5 * width, De_sq_trunc, width,
           label=r'$\mathrm{De}_{\rm sq}$ truncation [%]',
           color='#BFDBFE', edgecolor='#1E40AF', linewidth=1.2)
    ax.bar(x - 0.5 * width, VE_amp, width,
           label='VE memory (amplitude) [%]',
           color='#FECACA', edgecolor='#B91C1C', linewidth=1.2)
    ax.bar(x + 0.5 * width, VE_phase, width,
           label=r'VE memory (phase) [$^\circ$]',
           color='#FED7AA', edgecolor='#C2410C', linewidth=1.2)
    ax.bar(x + 1.5 * width, inertia, width,
           label='Thin-film inertia [%]',
           color='#D1FAE5', edgecolor='#059669', linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Error magnitude')
    ax.set_title(f'Error source comparison at operating point '
                 f'(omega = {config.omega_op:.0f} rad/s)', fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    _save_or_show(fig, 'error_bar_chart', output_dir)
    return fig


# =============================================================================
# Helpers
# =============================================================================
_DEFAULT_COLORS = {
    'Newtonian': '#2166AC',
    '2% PAM': '#B2182B',
    '5% PAM': '#1B7837',
}


def _default_colors(items: dict) -> dict:
    """Assign default colors to fluid names."""
    fallback = ['#2166AC', '#B2182B', '#1B7837', '#762A83', '#D6604D']
    colors = {}
    for i, name in enumerate(items):
        colors[name] = _DEFAULT_COLORS.get(name, fallback[i % len(fallback)])
    return colors


def _save_or_show(fig, name: str, output_dir: Optional[str]):
    """Save figure if output_dir is provided, otherwise just close."""
    import matplotlib.pyplot as plt
    if output_dir is not None:
        from pathlib import Path
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
