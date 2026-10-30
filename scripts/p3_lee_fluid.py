"""Lee et al.'s fluid (ISO VG46) and the reduced-pressure transform of R-67.1 (D-9).

Source, CITED: Lee, Shang & Zhang, Tribol. Int. 226 (2027) 112637 (``LeeShangZhang2026``),
Table 1 (mu_r = 0.05 Pa s at T_r = 35 C, beta_mu = 0.0476 1/K, alpha_mu = 20 1/GPa,
rho_0 = 865 kg/m3 at T_0 = 35 C) and Eqs. (1)-(2).  Eq. (2) is Roelands' law with the
temperature in KELVIN: read in Celsius it gives 1.56 mPa s at 100 C, which is not a VG46; read
in kelvin it gives 39.7 mPa s at 40 C (45.9 cSt at rho_0), which is (r66_step1_handback.md §2).

Reduced pressure (R-67.1): with eta = eta_a g(p), q(p) = int_0^p dp'/g(p') satisfies the
isoviscous squeeze problem exactly on the axisymmetric Newtonian film.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

#: Table 1 of LeeShangZhang2026
MU_R, T_R_K, BETA_MU, ALPHA_MU = 0.05, 35.0 + 273.15, 0.0476, 20e-9
RHO_0, T_0_C, RHO_T_COEFF = 865.0, 35.0, 0.00065
_L = np.log(MU_R) + 9.67


def mu_lee(T_K: float, p=0.0):
    """Eq. (2), Roelands, Pa s; T in kelvin, p in Pa (gauge)."""
    z = ALPHA_MU / (5.1e-9 * _L)
    s0 = BETA_MU * (T_R_K - 138.0) / _L
    return MU_R * np.exp(_L * (-1.0 + (1.0 + 5.1e-9 * np.asarray(p, float)) ** z
                               * ((T_K - 138.0) / (T_R_K - 138.0)) ** (-s0)))


def rho_lee(T_C: float, p: float = 0.0) -> float:
    """Eq. (1), Dowson-Higginson with the linear thermal term, kg/m3; T in Celsius."""
    return RHO_0 * (1.0 + 0.6e-9 * p / (1.0 + 1.7e-9 * p) - RHO_T_COEFF * (T_C - T_0_C))


def roelands_ratio(T_K: float, p):
    """g(p) = eta(p)/eta(0) at temperature T."""
    return mu_lee(T_K, p) / mu_lee(T_K, 0.0)


def alpha_star(T_K: float) -> float:
    """Roelands-equivalent pressure-viscosity coefficient 1/int_0^inf eta(0)/eta(p) dp, 1/Pa."""
    return 1.0 / quad(lambda p: 1.0 / roelands_ratio(T_K, p), 0.0, 3e9, limit=400)[0]


def roelands_q_table(T_K: float, p_max: float = 3e9, n: int = 300001):
    """(p, q) on a grid: q = int_0^p dp'/g(p'); q is monotone with ceiling 1/alpha_star."""
    p = np.linspace(0.0, p_max, n)
    q = cumulative_trapezoid(1.0 / roelands_ratio(T_K, p), p, initial=0.0)
    return p, q


def barus_q(p, a):
    return (1.0 - np.exp(-a * np.asarray(p, float))) / a


def barus_p(q, a):
    q = np.asarray(q, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a * q < 1.0, -np.log(np.clip(1.0 - a * q, 1e-300, None)) / a, np.inf)


def first_integral_k(delta: float, tmin_deg: float, tmax_deg: float = 73.0) -> float:
    """k of the Newtonian squeeze first integral sin t H^3 p' ∝ (k - sin^2 t), the same
    integrals as the vendored scripts/p3_rim_asymptote._f (k = a/b)."""
    tmin, tmax = np.deg2rad(tmin_deg), np.deg2rad(tmax_deg)
    eps = (1.0 - delta) / np.cos(tmin)
    H = lambda t: 1.0 - eps * np.cos(t)
    pts = [tmin + x * (tmax - tmin) for x in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)]
    q = dict(limit=800, epsabs=0, epsrel=1e-11, points=pts)
    a = quad(lambda t: np.sin(t) / H(t) ** 3, tmin, tmax, **q)[0]
    b = quad(lambda t: 1.0 / (np.sin(t) * H(t) ** 3), tmin, tmax, **q)[0]
    return a / b


def newtonian_peak_deg(delta: float, tmin_deg: float, tmax_deg: float = 73.0) -> float:
    """Angle of the Newtonian squeeze pressure peak, sin^2 theta = k (p' = 0)."""
    return float(np.rad2deg(np.arcsin(np.sqrt(first_integral_k(delta, tmin_deg, tmax_deg)))))
