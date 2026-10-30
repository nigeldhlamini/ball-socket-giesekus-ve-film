"""R-55: the vendored p3_rim_asymptote.py, checked by something other than itself.

The quadrature's coefficient was wrong by exactly 2 until the quadrature caught it,
so the quadrature cannot be its own witness.  The independent check here is a
direct finite-volume solve of the SAME axisymmetric equation,

    d/dt( sin t H^3 p' ) = -2 sin t cos t      (squeeze, p = P0 p~)
    d/dt( sin t H^3 p' ) = 0                   (pocket, p(t_min) = 1)

on a grid graded into the rim band, with Dirichlet ends -- no shared code with the
module under test beyond the land geometry.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from scipy.linalg import solve_banded

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import p3_rim_asymptote as RA  # noqa: E402


def _fv(delta, tmin_deg, *, source=True, p_pocket=0.0, N=6000):
    """Vertex-centred conservative FD on a graded grid; returns (f_sq_part, g_part)
    as the load integral int p sin t cos t dt of the solved p."""
    tmin = np.deg2rad(tmin_deg)
    eps = (1 - delta) / np.cos(tmin)
    L = RA.TMAX - tmin
    band = delta / np.tan(tmin)
    sig = np.log(1 + L / (band / 40.0)) if band / 40.0 < L else 1.0
    s = np.linspace(0.0, 1.0, N + 1)
    x = L * np.expm1(sig * s) / np.expm1(sig)
    t = tmin + x
    tf = 0.5 * (t[1:] + t[:-1])
    a = np.sin(tf) * (1 - eps * np.cos(tf)) ** 3 / np.diff(t)       # face conductance
    n = N - 1                                                        # interior nodes
    ab = np.zeros((3, n))
    ab[1] = -(a[:-1] + a[1:])
    ab[0, 1:] = a[1:-1]
    ab[2, :-1] = a[1:-1]
    vol = 0.5 * (t[2:] - t[:-2])
    rhs = (-2 * np.sin(t[1:-1]) * np.cos(t[1:-1]) * vol) if source else np.zeros(n)
    rhs = rhs.copy()
    rhs[0] -= a[0] * p_pocket                                         # p(t_min) = p_pocket
    p = np.concatenate([[p_pocket], solve_banded((1, 1), ab, rhs), [0.0]])
    w = np.sin(t) * np.cos(t) * p
    return float(np.sum(0.5 * (w[1:] + w[:-1]) * np.diff(t)))


@pytest.mark.parametrize("tmin", [20, 30, 40])
@pytest.mark.parametrize("delta", [0.3, 0.03, 0.003])
def test_squeeze_quadrature_matches_independent_fv(tmin, delta):
    """f_recess against a direct solve.  Failing looks like: a relative gap above
    1e-4, i.e. a coefficient or boundary-condition error of the kind R-55 records."""
    fq = RA.f_recess(delta, tmin)
    ff = _fv(delta, tmin)
    assert abs(ff / fq - 1) < 1e-4, (tmin, delta, fq, ff)


@pytest.mark.parametrize("tmin", [20, 40])
@pytest.mark.parametrize("delta", [0.3, 0.01])
def test_pocket_term_matches_fv_and_superposes(tmin, delta):
    """A2: g_pocket against a direct solve, and the combined solve equals the sum
    (linearity at constant eta is what licenses the positive control)."""
    gq = RA.g_pocket(delta, tmin)
    gf = _fv(delta, tmin, source=False, p_pocket=1.0)
    assert abs(gf / gq - 1) < 1e-4, (gq, gf)
    P = 7.3
    both = _fv(delta, tmin, source=True, p_pocket=P)
    assert abs(both / (RA.f_recess(delta, tmin) + P * gq) - 1) < 1e-4


@pytest.mark.parametrize("tmin", [20, 30, 40])
def test_log_coefficient(tmin):
    """d f / d ln(1/delta) -> 2 cos^5 / sin^2 (the R-55 box).  A missing factor 2
    fails this by 50 %."""
    t = np.deg2rad(tmin)
    C = 2 * np.cos(t) ** 5 / np.sin(t) ** 2
    v6, v8 = RA.f_recess(1e-6, tmin), RA.f_recess(1e-8, tmin)
    assert abs((v8 - v6) / np.log(100.0) / C - 1) < 1e-3


def test_closed_pole_exponent_tends_to_minus_one():
    """Counterfactual: no recess -> sphere-on-plane, exponent -> -1."""
    e = RA.local_exp(RA.f_closed, 1e-4, 1e-5)
    assert -1.02 < e < -0.98, e


def test_delta_required_roundtrip_and_target_normalisation():
    """A3/A4: delta_required inverts f_recess; solver_f uses eps_dot = omega eps."""
    for tmin, d in ((20, 3e-3), (40, 1e-5)):
        got, ok = RA.delta_required(RA.f_recess(d, tmin), tmin)
        assert ok and abs(got / d - 1) < 1e-8
    F = RA.load_newtonian(0.05, 30, eta=0.02, R=6.5e-3, c=5e-6, eps_dot=300.0 * 1.1)
    assert abs(RA.solver_f(F, c=5e-6, eta=0.02, R=6.5e-3, omega=300.0, eps=1.1)
               / RA.f_recess(0.05, 30) - 1) < 1e-12


def test_A5_hand_case_with_solver_squeeze_rate():
    """R-57.6 hand case: theta_min 30 deg, delta 0.05, so eps = 0.95/cos 30 = 1.09697.
    W is built from the INDEPENDENT finite-volume f, with eps_dot = omega eps; the
    amended inversion must return delta = 0.05.  The delivered delta_required, fed the
    same W at a delta-independent eps_dot (eps fixed at 1), must NOT -- the
    amendment changes the answer."""
    eta, R, c, om = 0.02, 6.5e-3, 10e-6, 300.0
    eps = 0.95 / np.cos(np.deg2rad(30.0))
    assert abs(eps - 1.0969655114602888) < 1e-12
    W = 12 * np.pi * eta * R ** 4 * om * eps * _fv(0.05, 30) / c ** 2
    d, ok = RA.delta_for_load(W, 30, eta=eta, R=R, c=c, omega=om)
    assert ok and abs(d / 0.05 - 1) < 1e-3, d
    d_old, _ = RA.delta_required(RA.f_target(W, c=c, eta=eta, R=R, eps_dot=om), 30)
    assert abs(d_old / 0.05 - 1) > 1e-2, d_old


def test_A5_hydrostatic_superposition_matches_the_solver():
    """The A5 load (squeeze + pocket + case) against the ball-socket solver itself,
    Newtonian, at 128 theta-cells where R1 measured the error below 3e-4."""
    from ball_socket.spherical import Spherical, SphericalConditions
    from ball_socket.reynolds import solve_reynolds
    from src.rheology import create_newtonian
    from src.solver import SolverConfig
    eta, R, c, om, tm, d = 0.02, 6.5e-3, 10e-6, 300.0, 30.0, 0.1
    pp, pc = 2.0e5, 1.0e5
    tmin = np.deg2rad(tm)
    eps = (1 - d) / np.cos(tmin)
    cond = SphericalConditions(p_ch=pp, p_case=pc, e_vec=(0, 0, -eps * c),
                               e_dot=(0, 0, -om * eps * c), omega_b=(0, 0, 0))
    sph = Spherical(R, c, np.deg2rad(73.0), cond, 128, 8, theta_min=tmin)
    res = solve_reynolds(sph, create_newtonian(eta), SolverConfig(linear_solver="direct"),
                         model="newtonian")
    F_solver = sph.load_capacity(res.pressure)
    F_A5 = RA.load_solver_kinematics(d, tm, eta=eta, R=R, c=c, omega=om, p_pocket=pp, p_case=pc)
    assert abs(F_solver / F_A5 - 1) < 1e-3, (F_solver, F_A5)


def test_pocket_vanishes_as_rim_closes():
    """As delta -> 0 the rim conductance vanishes: the land falls to mouth pressure
    and the pocket's land load -> 0.  The positive control must therefore be sized
    at the SMALLEST test delta (R-55.2 R1)."""
    g = [RA.g_pocket(d, 20) for d in (0.3, 0.1, 0.03, 0.01)]
    assert all(b < a for a, b in zip(g, g[1:]))
