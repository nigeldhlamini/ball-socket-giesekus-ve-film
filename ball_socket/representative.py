"""Representative (thesis-track) geometry and finish class for the ball-socket.

RULING 2026-07-26: the THESIS models the literature-standard interface (hardened
bearing steel on leaded tin bronze) with published, citable properties.  This
module holds the REPRESENTATIVE class values the campaign uses.  It deliberately
does NOT import ``geometry_provenance`` (drawing-exact values), which is
INDUSTRY-REPORT TRACK ONLY.  The thesis campaign config must import only this
module.

Nothing here is drawing-derived: R_b, A_p, theta_max and theta_min are stated as
representative of the swashplate-pump slipper-joint geometry class, and the
surface finishes as standard machining finish classes.
"""
from __future__ import annotations

import math

# --- Representative geometry (class values, not drawing values) ------------
R_B_M = 6.5e-3                 # ball radius [m], representative of the class
A_P_M2 = 2.0e-4                # piston area [m^2], representative
THETA_MAX_DEG = 73.0           # socket mouth (ball-neck interference)
THETA_MAX_OAT_DEG = (66.0, 73.0, 80.0)        # +/-~3-7 deg OAT (sec 1.5)

# Feed-pocket edge: the film is an ANNULUS, theta in [theta_min, theta_max].
# Both pocket-edge candidates are carried as a band until the rasterisation
# resolves it; described as the representative feed-pocket range for the class.
THETA_MIN_BAND_DEG = (25.0, 36.0)
THETA_MIN_OAT_DEG = (25.0, 36.0)

# --- Surface finish classes (citation slot: machining-finish reference) ----
# Ball: ground / microfinished, Ra 0.05 class.  Rq ~ 1.25 Ra (Gaussian profile).
BALL_RA_CLASS_UM = 0.05
RQ_BALL_M = 1.25 * BALL_RA_CLASS_UM * 1e-6            # 6.25e-8 m

# Socket: fine-turned bronze, standard Ra 0.3-0.4 finish class.  For a
# deterministic turned profile the Rq/Ra conversion is 1.1-1.3, carried as an
# explicit pre-registered uncertainty.
SOCKET_RA_CLASS_UM = (0.3, 0.4)
RQ_CONVERSION_BAND = (1.1, 1.3)

# RULING A1 (2026-07-26): adopt the WIDE band -- the full cross-product of the
# finish CLASS (Ra 0.3-0.4) with the conversion band (1.1-1.3).  The thesis
# models a machining finish class, not a specific drawing callout; centring on
# Ra 0.4 would reimport the proprietary specificity the material ruling removed.
RQ_SOCKET_BAND_M = (1.1 * 0.3e-6, 1.3 * 0.4e-6)                    # (3.3e-7, 5.2e-7)
# Superseded narrow reading (Ra 0.4 only), retained for traceability.
RQ_SOCKET_BAND_NARROW_M = (1.1 * 0.4e-6, 1.3 * 0.4e-6)             # (4.4e-7, 5.2e-7)


def sigma_star_band(narrow: bool = False) -> tuple[float, float]:
    """Composite roughness sigma* = hypot(Rq_ball, Rq_socket) at both band ends.

    Default is the RULED wide band: sigma* ~ (0.336, 0.524) um.
    """
    band = RQ_SOCKET_BAND_NARROW_M if narrow else RQ_SOCKET_BAND_M
    return (math.hypot(RQ_BALL_M, band[0]), math.hypot(RQ_BALL_M, band[1]))


def hmin_floor_band(lambda_ratio_min: float = 3.0, narrow: bool = False):
    """Full-film h_min floor (Lambda >= lambda_ratio_min) at both sigma* ends.

    Default (wide band, Lambda >= 3): ~ (1.01, 1.57) um.
    """
    lo, hi = sigma_star_band(narrow=narrow)
    return (lambda_ratio_min * lo, lambda_ratio_min * hi)


def pocket_relief_area(R_b: float, theta_min: float) -> float:
    """Load-line projected area of the fed pocket cap theta in [0, theta_min]:

        A_proj = int_0^{theta_min} cos(th) * R^2 sin(th) dth dphi
               = pi R_b^2 sin^2(theta_min)

    The pocket is held near chamber pressure, so it supplies a partial
    hydrostatic relief F_pocket = p_ch * A_proj on the load line; the film
    carries only the balance (Ch.9 annular-domain patch T2).
    """
    return math.pi * R_b ** 2 * math.sin(theta_min) ** 2
