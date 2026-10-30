"""Ball-socket geometry and gap function (Ch.9 sec:bs_geometry).

RULING 2026-07-26 (thesis material track): the socket is modelled as the
literature-standard leaded tin bronze (CuSn12Pb class) with published, citable
properties; the ball as generic hardened bearing steel (100Cr6 class).  The
NaN VERIFY guards are retired -- every constant below is CITED.  Proprietary
material identity is industry-report track only (see
``industry_track/PARKED_MATERIAL_CONTENT.md``).

The film domain is an ANNULUS: axial piston machines feed the slipper pocket
through a central passage, so the socket carries a fed recess at its pole near
chamber pressure and the film occupies theta in [theta_min, theta_max].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .outcomes import NonConvergence
from . import representative as rep


# ===========================================================================
# 1. Geometry
# ===========================================================================

@dataclass(frozen=True)
class BallSocketGeometry:
    """Representative geometry of the spherical joint (annular fed film).

    Defaults are REPRESENTATIVE class values from ``representative.py`` (not
    drawing values).  ``c0`` is a campaign input axis.  ``theta_min`` is the
    feed-pocket edge (a pre-registered band, OAT-swept), ``theta_max`` the
    socket mouth (OAT-swept).
    """
    c0: float = float("nan")                          # cold radial clearance [m] (axis)
    R_b: float = rep.R_B_M                            # ball radius [m]
    theta_max: float = np.deg2rad(rep.THETA_MAX_DEG)  # socket mouth [rad]
    theta_min: float = np.deg2rad(rep.THETA_MIN_BAND_DEG[0])   # pocket edge [rad]

    Rq_ball: float = rep.RQ_BALL_M                    # ground/microfinished, Ra 0.05 class
    Rq_socket: float = rep.RQ_SOCKET_BAND_M[0]        # fine-turned band end (swept)

    @property
    def sigma_star(self) -> float:
        """Composite roughness for the lambda ratio (Ch.9)."""
        return float(np.hypot(self.Rq_ball, self.Rq_socket))

    @property
    def aspect_ratio(self) -> float:
        return self.c0 / self.R_b                     # epsilon_sph ~ 1e-3

    @property
    def pocket_relief_area(self) -> float:
        """Load-line projected area of the fed pocket cap [m^2]."""
        return rep.pocket_relief_area(self.R_b, self.theta_min)


# ===========================================================================
# 2. Materials (CITED -- literature-standard interface)
# ===========================================================================

@dataclass(frozen=True)
class SocketMaterial:
    """Leaded tin bronze (CuSn12Pb / CuPb10Sn class) -- the literature-standard
    socket lining.  All values CITED (not VERIFY); the guards are retired per
    the 2026-07-26 ruling.  Sources are the references Chapter 9 already cites
    for these constants.
    """
    E: float = 100.0e9              # Young's modulus [Pa]
    nu: float = 0.34                # Poisson ratio [-]
    alpha_th: float = 18.0e-6       # CTE [1/K]
    HB_min: float = 80.0            # Brinell hardness, lower of the class range
    HB_max: float = 120.0           # Brinell hardness, upper of the class range

    SOURCES = {
        "E": "CITED: Ch.9 bronze baseline (E_s = 100 GPa), eq:Estar",
        "nu": "CITED: Ch.9 bronze baseline (nu_s = 0.34), eq:Estar",
        "alpha_th": "CITED: Ch.9 eq:alpha_bronze (CuSn12 bronze, 18e-6 /K)",
        "HB": "CITED: jiang2019novel (leaded tin bronze, ~80-120 HB)",
        "class": "CITED: hamrock2004fundamentals (conformability, embeddability, "
                 "emergency dry-running of the soft bronze socket)",
    }

    @property
    def HB_range(self) -> tuple[float, float]:
        return (self.HB_min, self.HB_max)


# Generic hardened bearing steel (100Cr6 / AISI 52100 class), Ch.9 frozen.
BALL_STEEL = dict(E=210e9, nu=0.30, alpha_th=12e-6)
BALL_STEEL_SOURCES = {
    "class": "CITED: hardened bearing steel (100Cr6 / AISI~52100 class), Ch.9",
    "E, nu, alpha_th": "CITED: Ch.9 frozen steel constants",
}


def composite_modulus(E_b: float = BALL_STEEL["E"], nu_b: float = BALL_STEEL["nu"],
                      E_s: float = SocketMaterial().E,
                      nu_s: float = SocketMaterial().nu) -> float:
    """1/E* = (1-nu_b^2)/E_b + (1-nu_s^2)/E_s  (Ch.9 Eq. Estar).

    With the cited steel-on-bronze pair this reproduces the chapter's
    E* ~ 76 GPa, which unblocks Phase 5 (Patir-Cheng + Greenwood-Tripp).
    """
    return 1.0 / ((1 - nu_b ** 2) / E_b + (1 - nu_s ** 2) / E_s)


def delta_alpha_th(socket: SocketMaterial | None = None) -> float:
    """Delta_alpha = alpha_socket - alpha_ball [1/K]; the ONLY channel (with
    sigma*) through which material identity enters the campaign."""
    s = socket or SocketMaterial()
    return s.alpha_th - BALL_STEEL["alpha_th"]


# ===========================================================================
# 3. Gap and thermal clearance
# ===========================================================================

def gap_function(theta: np.ndarray, phi: np.ndarray,
                 c: float, e: np.ndarray) -> np.ndarray:
    """h(theta, phi) = c + e_x sin t cos p + e_y sin t sin p + e_z cos t.

    Ch.9 Eq. h_spherical.  Raises ``NonConvergence`` if min(h) <= 0
    (interpenetration): the fourth outcome, never clipped (Paper 3 C1).
    """
    st = np.sin(theta)
    h = (c + e[0] * st * np.cos(phi) + e[1] * st * np.sin(phi)
         + e[2] * np.cos(theta))
    if np.any(h <= 0.0):
        raise NonConvergence("eccentricity blow-up: h <= 0 encountered",
                             kind="blowup_film_closed")
    return h


def thermal_clearance(c0: float, R_b: float, delta_T: float,
                      alpha_ball: float = BALL_STEEL["alpha_th"],
                      alpha_socket: float | None = None) -> float:
    """c(T) = c0 + (alpha_socket - alpha_ball) R_b delta_T  (Ch.9 c_thermal).

    Bronze socket expands faster than the steel ball, so clearance grows under
    heating; the seizure risk is on cold-socket transients.
    """
    a_s = SocketMaterial().alpha_th if alpha_socket is None else alpha_socket
    return c0 + (a_s - alpha_ball) * R_b * delta_T
