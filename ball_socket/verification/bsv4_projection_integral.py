r"""
Closed-form geometric projection factor for the squeeze-memory load
coefficient on the annular spherical film.  BS-V4 decomposition.

Model (axisymmetric, concentric limit, linearised in eccentricity):

  Reynolds:  (1/(R^2 sin t)) d/dt [ sin t * D(t) * dp/dt ] = hdot(t)
  squeeze:   h = c + e cos t,  hdot = edot cos t
  memory:    the memory flux is proportional to dp/dt, so it enters as a
             modification of the mobility,
                 D(t) = D0 [ 1 - k cos t ],   k = De_sq_ref * M/eta_bar
             because De_sq(t) = lambda hdot / h = (lambda edot / c) cos t
             carries the SAME cos t weighting as the squeeze source.
             D0 = c^3 / (12 eta_bar).

  BCs: squeeze-generated pressure vanishes at both rims (the hydrostatic
       particular solution is separate and linear, and contributes no
       memory term at this order).

Expansion p = p0 + k p1:
  O(1): d/dt[ sin t D0 p0' ] = R^2 edot cos t sin t
  O(k): d/dt[ sin t D0 (p1' - cos t p0') ] = 0

Load on the polar axis: F = 2 pi R^2 \int p cos t sin t dt.

Geometric projection factor:
  G = [ \int p1 cos t sin t dt ] / [ \int p0 cos t sin t dt ]

so that the leading memory coefficient is  c_lead = (M/eta_bar) * G,
i.e. (kernel factor) x (geometric projection factor).  G -> 1 in the
planar limit.  All integrals below are elementary; quadrature is used
only to evaluate the closed forms and to cross-check them.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# ---------------------------------------------------------------- O(1)
# sin t D0 p0' = R^2 edot sin^2 t / 2 + A
#   => p0'(t) = (R^2 edot/D0) [ sin t / 2 + a / sin t ],  a = A/(R^2 edot)
# p0(t) = (R^2 edot/D0) [ -cos t / 2 + a ln tan(t/2) ] + const
# Units scaled out: set (R^2 edot / D0) = 1.

def p0_shape(t, a):
    return -np.cos(t) / 2.0 + a * np.log(np.tan(t / 2.0))

def solve_a(t1, t2):
    """a fixed by p0(t1) = p0(t2) = 0 (two BCs, one free constant each)."""
    f = lambda a: p0_shape(t2, a) - p0_shape(t1, a)
    return brentq(f, -1e3, 1e3, xtol=1e-14)

def p0(t, t1, a):
    return p0_shape(t, a) - p0_shape(t1, a)

# ---------------------------------------------------------------- O(k)
# p1' = cos t p0' + b / sin t,   b = B/(D0 R^2 edot)
# p1(t) = [ sin^2 t / 4 + a ln sin t ] + b ln tan(t/2) + const
def p1_shape(t, a, b):
    return np.sin(t) ** 2 / 4.0 + a * np.log(np.sin(t)) + b * np.log(np.tan(t / 2.0))

def solve_b(t1, t2, a):
    f = lambda b: p1_shape(t2, a, b) - p1_shape(t1, a, b)
    return brentq(f, -1e3, 1e3, xtol=1e-14)

def p1(t, t1, a, b):
    return p1_shape(t, a, b) - p1_shape(t1, a, b)

# ------------------------------------------------------- projection factor
def G(theta1_deg, theta2_deg):
    t1, t2 = np.radians(theta1_deg), np.radians(theta2_deg)
    a = solve_a(t1, t2)
    b = solve_b(t1, t2, a)
    num = quad(lambda t: p1(t, t1, a, b) * np.cos(t) * np.sin(t), t1, t2,
               limit=200)[0]
    den = quad(lambda t: p0(t, t1, a) * np.cos(t) * np.sin(t), t1, t2,
               limit=200)[0]
    return num / den, a, b

# ---------------------------------------------------------------- checks
def _selfcheck() -> None:
    """Demo / validation output.  Kept out of import so the BS-V4 gate can use
    this module side-effect free."""
    print("Planar-limit check (narrow annulus near the pole, cos t -> 1):")
    for t1 in (5.0, 2.0, 1.0, 0.5):
        g, _, _ = G(t1, t1 * 1.2)
        print(f"  t1={t1:5.2f} deg, t2=1.2*t1 :  G = {g:.6f}")

    print("\nFull-cap-like (t1 -> 0) vs annular, t2 = 73 deg:")
    for t1 in (0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 36.0, 45.0):
        g, a, b = G(t1, 73.0)
        print(f"  theta_min={t1:5.1f} deg :  G = {g:.6f}   (a={a:+.4f}, b={b:+.4f})")

    print("\nSensitivity in theta_max at the two registered theta_min values:")
    for t1 in (25.0, 36.0):
        for t2 in (66.0, 73.0, 80.0):
            g, _, _ = G(t1, t2)
            print(f"  theta_min={t1:4.1f}, theta_max={t2:4.1f} :  G = {g:.6f}")

    print("\nImplied leading coefficient c_lead = (M/eta_bar) * G, with the")
    print("planar kernel factor eta_p/eta_0 = 0.8 used as M/eta_bar:")
    for t1, t2 in ((25.0, 73.0), (36.0, 73.0)):
        g, _, _ = G(t1, t2)
        print(f"  theta_min={t1:4.1f}, theta_max={t2:4.1f} :  "
              f"G={g:.4f}  ->  c_lead = {0.8*g:.4f}")


if __name__ == "__main__":
    _selfcheck()
