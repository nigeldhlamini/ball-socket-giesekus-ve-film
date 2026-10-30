"""
Slipper Bearing Geometry
========================

Defines the slipper/swash plate geometry for lubrication analysis.

Gap height function (Eq. 1 in thesis):
    h(r, θ, t) = h₀(t) + α_tilt(t) · r · cos[θ - φ(t)]

where:
    - h₀: Central clearance
    - α_tilt: Tilt magnitude  
    - φ: Tilt direction (azimuthal angle)

For simplified analyses:
    h(r, θ) = h₀ + r·(α_x·cos(θ) + α_y·sin(θ))

References:
    - Bergada et al. (2008). ASME J. Fluids Eng.
    - Dhlamini, N.C. (2025). PhD Thesis, Chapter 4.

Author: Nigel C. Dhlamini
"""

from dataclasses import dataclass, field
from typing import Union, Callable, Optional
import numpy as np

ArrayLike = Union[float, np.ndarray]


@dataclass
class SlipperGeometry:
    """
    Slipper bearing geometry parameters.
    
    The slipper is an annular pad with inner radius R_in (pocket edge)
    and outer radius R_out (land edge).
    
    Parameters
    ----------
    R_in : float
        Inner radius (pocket/orifice radius) [m]
    R_out : float
        Outer radius (land edge) [m]
    h_0 : float
        Reference central clearance [m]
    alpha_x : float, optional
        Tilt about x-axis [rad]. Default: 0
    alpha_y : float, optional
        Tilt about y-axis [rad]. Default: 0
    
    Attributes
    ----------
    L : float
        Characteristic length scale = R_out - R_in [m]
    A_land : float
        Land area (lubrication zone) [m²]
    aspect_ratio : float
        ε = h₀/L (should be ≪ 1 for lubrication)
    
    Examples
    --------
    >>> geom = SlipperGeometry(R_in=0.008, R_out=0.020, h_0=5e-6)
    >>> geom.gap_height(0.015, 0.0)  # At r=15mm, θ=0
    5e-06
    """
    R_in: float
    R_out: float
    h_0: float
    alpha_x: float = 0.0
    alpha_y: float = 0.0
    
    # Derived quantities
    L: float = field(init=False)
    A_land: float = field(init=False)
    aspect_ratio: float = field(init=False)
    
    def __post_init__(self):
        """Validate and compute derived quantities."""
        if self.R_in < 0:
            raise ValueError(f"Inner radius must be non-negative: {self.R_in}")
        if self.R_out <= self.R_in:
            raise ValueError(f"Outer radius must exceed inner: R_out={self.R_out}, R_in={self.R_in}")
        if self.h_0 <= 0:
            raise ValueError(f"Central clearance must be positive: {self.h_0}")
        
        self.L = self.R_out - self.R_in
        self.A_land = np.pi * (self.R_out**2 - self.R_in**2)
        self.aspect_ratio = self.h_0 / self.L
    
    @property
    def tilt_magnitude(self) -> float:
        """Total tilt magnitude α_tilt = √(α_x² + α_y²) [rad]."""
        return np.sqrt(self.alpha_x**2 + self.alpha_y**2)
    
    @property
    def tilt_direction(self) -> float:
        """Tilt direction φ = atan2(α_y, α_x) [rad]."""
        return np.arctan2(self.alpha_y, self.alpha_x)
    
    def gap_height(self, r: ArrayLike, theta: ArrayLike, 
                   h_0_override: Optional[float] = None) -> ArrayLike:
        """
        Compute gap height h(r, θ).
        
        h(r, θ) = h₀ + r·(α_x·cos(θ) + α_y·sin(θ))
        
        Parameters
        ----------
        r : float or ndarray
            Radial position(s) [m]
        theta : float or ndarray
            Azimuthal angle(s) [rad]
        h_0_override : float, optional
            Override central clearance (for transient analysis)
        
        Returns
        -------
        h : float or ndarray
            Gap height [m]
        """
        r = np.asarray(r)
        theta = np.asarray(theta)
        h_0 = h_0_override if h_0_override is not None else self.h_0
        
        tilt_contribution = r * (self.alpha_x * np.cos(theta) + 
                                  self.alpha_y * np.sin(theta))
        h = h_0 + tilt_contribution
        
        return h
    
    def gap_height_derivatives(self, r: ArrayLike, theta: ArrayLike,
                                h_0_override: Optional[float] = None) -> tuple:
        """
        Compute gap height and its spatial derivatives.
        
        Returns
        -------
        h : ndarray
            Gap height
        h_r : ndarray
            ∂h/∂r = α_x·cos(θ) + α_y·sin(θ)
        h_theta : ndarray
            ∂h/∂θ = r·(-α_x·sin(θ) + α_y·cos(θ))
        """
        r = np.asarray(r)
        theta = np.asarray(theta)
        
        h = self.gap_height(r, theta, h_0_override)
        h_r = self.alpha_x * np.cos(theta) + self.alpha_y * np.sin(theta)
        h_theta = r * (-self.alpha_x * np.sin(theta) + self.alpha_y * np.cos(theta))
        
        return h, h_r, h_theta
    
    def min_gap(self) -> tuple:
        """
        Find minimum gap height and its location.
        
        Returns
        -------
        h_min : float
            Minimum gap height [m]
        r_min : float
            Radial location of minimum [m]
        theta_min : float
            Azimuthal location of minimum [rad]
        """
        if self.tilt_magnitude < 1e-12:
            # No tilt: uniform gap
            return self.h_0, self.R_out, 0.0
        
        # Minimum is at outer radius in direction of negative tilt
        theta_min = np.arctan2(self.alpha_y, self.alpha_x) + np.pi
        r_min = self.R_out
        h_min = self.gap_height(r_min, theta_min)
        
        return h_min, r_min, theta_min
    
    def max_gap(self) -> tuple:
        """
        Find maximum gap height and its location.
        
        Returns
        -------
        h_max : float
            Maximum gap height [m]
        r_max : float
            Radial location of maximum [m]
        theta_max : float
            Azimuthal location of maximum [rad]
        """
        if self.tilt_magnitude < 1e-12:
            return self.h_0, self.R_out, 0.0
        
        # Maximum is at outer radius in direction of positive tilt
        theta_max = np.arctan2(self.alpha_y, self.alpha_x)
        r_max = self.R_out
        h_max = self.gap_height(r_max, theta_max)
        
        return h_max, r_max, theta_max
    
    def check_lubrication_validity(self) -> dict:
        """
        Check if lubrication assumptions are valid.
        
        Returns
        -------
        dict
            Validity checks with warnings if assumptions are marginal
        """
        h_min, _, _ = self.min_gap()
        h_max, _, _ = self.max_gap()
        
        checks = {
            'aspect_ratio': self.aspect_ratio,
            'aspect_ratio_valid': self.aspect_ratio < 0.01,
            'h_min': h_min,
            'h_min_positive': h_min > 0,
            'h_ratio': h_max / h_min if h_min > 0 else np.inf,
            'warnings': []
        }
        
        if self.aspect_ratio > 0.01:
            checks['warnings'].append(
                f"Aspect ratio ε = {self.aspect_ratio:.2e} > 0.01: "
                "lubrication approximation may be marginal"
            )
        
        if h_min <= 0:
            checks['warnings'].append(
                f"Minimum gap h_min = {h_min:.2e} ≤ 0: contact predicted"
            )
        elif h_min < 1e-6:
            checks['warnings'].append(
                f"Minimum gap h_min = {h_min:.2e} m < 1 μm: "
                "boundary/mixed lubrication likely"
            )
        
        return checks
    
    def summary(self) -> str:
        """Return formatted summary of geometry."""
        h_min, r_min, theta_min = self.min_gap()
        h_max, r_max, theta_max = self.max_gap()
        
        lines = [
            "Slipper Bearing Geometry",
            "=" * 40,
            f"Inner radius R_in        : {self.R_in*1e3:.2f} mm",
            f"Outer radius R_out       : {self.R_out*1e3:.2f} mm",
            f"Land width L             : {self.L*1e3:.2f} mm",
            f"Land area A              : {self.A_land*1e6:.2f} mm²",
            "-" * 40,
            f"Central clearance h₀     : {self.h_0*1e6:.2f} μm",
            f"Tilt α_x                 : {self.alpha_x*1e3:.3f} mrad",
            f"Tilt α_y                 : {self.alpha_y*1e3:.3f} mrad",
            f"Tilt magnitude           : {self.tilt_magnitude*1e3:.3f} mrad",
            "-" * 40,
            f"Minimum gap              : {h_min*1e6:.2f} μm at θ={np.degrees(theta_min):.1f}°",
            f"Maximum gap              : {h_max*1e6:.2f} μm at θ={np.degrees(theta_max):.1f}°",
            f"Aspect ratio ε           : {self.aspect_ratio:.2e}",
        ]
        return "\n".join(lines)


@dataclass
class OperatingConditions:
    """
    Operating conditions for the slipper bearing.
    
    Parameters
    ----------
    p_supply : float
        Supply pressure (pocket pressure) [Pa]
    p_ambient : float
        Ambient pressure [Pa]. Default: 0 (gauge)
    V_T : float, optional
        Tangential sliding velocity magnitude [m/s]. Default: 0
    theta_V : float, optional
        Direction of sliding velocity [rad]. Default: 0
    omega_s : float, optional
        Slipper spin angular velocity [rad/s]. Default: 0
    h_dot : float, optional
        Squeeze velocity dh/dt [m/s]. Negative = approaching. Default: 0
    """
    p_supply: float
    p_ambient: float = 0.0
    V_T: float = 0.0
    theta_V: float = 0.0
    omega_s: float = 0.0
    h_dot: float = 0.0
    
    @property
    def delta_p(self) -> float:
        """Pressure drop across the land [Pa]."""
        return self.p_supply - self.p_ambient
    
    @property
    def is_squeezing(self) -> bool:
        """True if gap is closing (ḣ < 0)."""
        return self.h_dot < 0
    
    @property
    def is_separating(self) -> bool:
        """True if gap is opening (ḣ > 0)."""
        return self.h_dot > 0
    
    def tangential_velocity_components(self) -> tuple:
        """
        Compute Cartesian components of tangential velocity.
        
        Returns
        -------
        V_Tx : float
            x-component of tangential velocity [m/s]
        V_Ty : float  
            y-component of tangential velocity [m/s]
        """
        V_Tx = self.V_T * np.cos(self.theta_V)
        V_Ty = self.V_T * np.sin(self.theta_V)
        return V_Tx, V_Ty
    
    def radial_tangential_velocity(self, theta: ArrayLike) -> tuple:
        """
        Compute radial and circumferential components of tangential velocity.
        
        Parameters
        ----------
        theta : float or ndarray
            Azimuthal position(s) [rad]
        
        Returns
        -------
        V_Tr : float or ndarray
            Radial component of tangential velocity [m/s]
        V_Ttheta : float or ndarray
            Circumferential component [m/s]
        """
        theta = np.asarray(theta)
        # Transform from Cartesian (V_T at angle theta_V) to polar
        V_Tr = self.V_T * np.cos(theta - self.theta_V)
        V_Ttheta = -self.V_T * np.sin(theta - self.theta_V)
        return V_Tr, V_Ttheta
    
    def summary(self) -> str:
        """Return formatted summary of operating conditions."""
        lines = [
            "Operating Conditions",
            "=" * 40,
            f"Supply pressure p_s      : {self.p_supply/1e6:.2f} MPa",
            f"Ambient pressure p_a     : {self.p_ambient/1e6:.2f} MPa",
            f"Pressure drop Δp         : {self.delta_p/1e6:.2f} MPa",
            "-" * 40,
            f"Tangential velocity V_T  : {self.V_T:.2f} m/s",
            f"Velocity direction θ_V   : {np.degrees(self.theta_V):.1f}°",
            f"Slipper spin ω_s         : {self.omega_s:.1f} rad/s",
            "-" * 40,
            f"Squeeze velocity ḣ       : {self.h_dot*1e3:.3f} mm/s",
            f"Squeeze state            : {'Closing' if self.is_squeezing else 'Opening' if self.is_separating else 'Static'}",
        ]
        return "\n".join(lines)


# =============================================================================
# Factory Functions
# =============================================================================

def create_standard_slipper() -> SlipperGeometry:
    """
    Create standard slipper geometry based on thesis reference case.
    
    From Table 1:
        R_in = 8 mm, R_out = 20 mm, h₀ = 10 μm
    """
    return SlipperGeometry(
        R_in=0.008,
        R_out=0.020,
        h_0=10e-6,
        alpha_x=0.0,
        alpha_y=0.0
    )


def create_tilted_slipper(alpha_tilt: float = 1e-4, 
                           phi: float = 0.0) -> SlipperGeometry:
    """
    Create slipper with specified tilt.
    
    Parameters
    ----------
    alpha_tilt : float
        Tilt magnitude [rad]. Default: 0.1 mrad
    phi : float
        Tilt direction [rad]. Default: 0 (tilt about y-axis)
    
    Returns
    -------
    SlipperGeometry
        Slipper with specified tilt decomposed into α_x, α_y
    """
    alpha_x = alpha_tilt * np.cos(phi)
    alpha_y = alpha_tilt * np.sin(phi)
    
    return SlipperGeometry(
        R_in=0.008,
        R_out=0.020,
        h_0=10e-6,
        alpha_x=alpha_x,
        alpha_y=alpha_y
    )


def create_standard_operating_conditions(
    p_supply: float = 25e6,
    V_T: float = 10.0,
    h_dot: float = 0.0
) -> OperatingConditions:
    """
    Create standard operating conditions.
    
    Parameters
    ----------
    p_supply : float
        Supply pressure [Pa]. Default: 25 MPa
    V_T : float
        Tangential velocity [m/s]. Default: 10 m/s
    h_dot : float
        Squeeze velocity [m/s]. Default: 0
    
    Returns
    -------
    OperatingConditions
    """
    return OperatingConditions(
        p_supply=p_supply,
        p_ambient=0.0,
        V_T=V_T,
        theta_V=0.0,
        omega_s=0.0,
        h_dot=h_dot
    )
