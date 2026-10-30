"""
Spherical (ball-and-socket) solver: Phase 3 + Phase 4 gates.
============================================================

Phase 3 (Newtonian / leading order):
  * axisymmetric pure-squeeze pressure matches a fine-quadrature reference and
    converges at 2nd order  -> validates metric, pole regularity, mouth BC;
  * a pure-rotation wedge gives pressure antisymmetric about the load line and a
    purely transverse load resultant (full-film hydrodynamic result).

Phase 4 (viscoelastic via the shared M):
  * Newtonian limit (lambda=0) recovers the Newtonian solve exactly;
  * UCM (alpha->0) and SOF (Wi->0) limits give M -> eta_p;
  * memory sign: gap closure (hdot<0) reduces the load (F_elastic < 0);
  * M is wired, not eta_T: swapping eta_T for M changes the elastic load by O(1)+.
"""
import warnings

import numpy as np
import pytest

from src.rheology import GiesekusFluid, create_PAM_5pct, create_newtonian
from ball_socket.spherical import Spherical, SphericalConditions
from ball_socket.reynolds import solve_reynolds
from src.solver import SolverConfig

R_B, C, THETA_MAX, ETA = 0.02, 20e-6, 1.2, 0.03
DELTA, DELTADOT = 0.5 * C, 1.0e-3


def _axisym_reference(theta_eval):
    tt = np.linspace(0.0, THETA_MAX, 200001)
    h = C - DELTA * np.cos(tt)
    hdot = -DELTADOT * np.cos(tt)
    D = h ** 3 / (12 * ETA)
    iG = np.sin(tt) * hdot
    G = np.concatenate([[0.0], np.cumsum(0.5 * (iG[1:] + iG[:-1]) * np.diff(tt))])
    with np.errstate(divide="ignore", invalid="ignore"):
        pp = R_B ** 2 * G / (np.sin(tt) * D)
    pp[0] = 0.0
    Pcum = np.concatenate([[0.0], np.cumsum(0.5 * (pp[1:] + pp[:-1]) * np.diff(tt))])
    p_ref = Pcum + (0.0 - Pcum[-1])
    return np.interp(theta_eval, tt, p_ref)


def _squeeze_conditions():
    return SphericalConditions(p_ch=0.0, e_vec=(0, 0, -DELTA), e_dot=(0, 0, -DELTADOT))


# --------------------------------------------------------------------------- #
# Phase 3 — Newtonian
# --------------------------------------------------------------------------- #
def test_axisymmetric_squeeze_matches_quadrature():
    fluid = create_newtonian(ETA)
    sph = Spherical(R_B, C, THETA_MAX, _squeeze_conditions(), Ntheta=80, Nphi=8)
    res = solve_reynolds(sph, fluid, SolverConfig(linear_solver="direct"), model="newtonian")
    p = res.pressure
    # phi-independent (axisymmetric) to machine precision
    assert np.max(np.abs(p - p.mean(axis=1, keepdims=True))) / np.max(np.abs(p)) < 1e-10
    p_ref = _axisym_reference(sph.mesh.theta)
    rel = np.max(np.abs(p.mean(axis=1) - p_ref)) / np.max(np.abs(p_ref))
    assert rel < 2e-3, f"squeeze pressure relerr {rel:.2e} too large"


def test_axisymmetric_squeeze_second_order():
    fluid = create_newtonian(ETA)
    errs, hs = [], []
    for Nt in (20, 40, 80):
        sph = Spherical(R_B, C, THETA_MAX, _squeeze_conditions(), Ntheta=Nt, Nphi=8)
        res = solve_reynolds(sph, fluid, SolverConfig(linear_solver="direct"), model="newtonian")
        p_ref = _axisym_reference(sph.mesh.theta)
        errs.append(np.max(np.abs(res.pressure.mean(axis=1) - p_ref)) / np.max(np.abs(p_ref)))
        hs.append(sph.mesh.dtheta)
    order = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert order > 1.9, f"observed order {order:.2f} < 1.9 (not 2nd order)"


def test_rotation_wedge_antisymmetric_about_load_line():
    fluid = create_newtonian(ETA)
    cond = SphericalConditions(p_ch=0.0, e_vec=(0.5 * C, 0, 0), e_dot=(0, 0, 0),
                               omega_b=(0, 0, 200.0))
    sph = Spherical(R_B, C, THETA_MAX, cond, Ntheta=60, Nphi=120)
    res = solve_reynolds(sph, fluid, SolverConfig(linear_solver="direct"), model="newtonian")
    p = res.pressure
    # fully 3D (not axisymmetric)
    assert np.max(np.abs(p - p.mean(axis=1, keepdims=True))) / np.max(np.abs(p)) > 0.5
    # antisymmetric about phi=0 (the load line): p(phi) = -p(-phi)
    anti = np.max(np.abs(p + p[:, ::-1])) / np.max(np.abs(p))
    assert anti < 1e-9, f"antisymmetry residual {anti:.2e}"
    # load resultant transverse to the line of centres (y), x & z ~ 0
    L = sph.load_vector(p)
    assert abs(L[1]) > 1e2 and abs(L[0]) < 1e-6 * abs(L[1]) and abs(L[2]) < 1e-6 * abs(L[1])


# --------------------------------------------------------------------------- #
# Phase 4 — viscoelastic via shared M
# --------------------------------------------------------------------------- #
def test_newtonian_limit_recovers_phase3_exactly():
    nf = create_newtonian(ETA)
    sph = Spherical(R_B, C, THETA_MAX, _squeeze_conditions(), Ntheta=40, Nphi=8)
    rN = solve_reynolds(sph, nf, SolverConfig(linear_solver="direct"), model="newtonian")
    rVE = solve_reynolds(sph, nf, SolverConfig(linear_solver="direct"), model="viscoelastic")
    assert np.array_equal(rVE.pressure, rN.pressure)


def test_M_limits_ucm_and_sof():
    fu = GiesekusFluid(eta_s=0.03, eta_p=0.12, lambda_=3.3e-3, alpha=1e-13)
    assert abs(fu.memory_viscosity(100 / fu.lambda_) / fu.eta_p - 1.0) < 1e-4
    f = create_PAM_5pct()
    assert abs(f.memory_viscosity(1e-3 / f.lambda_) / f.eta_p - 1.0) < 1e-3
    assert abs(f.Psi1(1e-4 / f.lambda_) / (2 * f.eta_p * f.lambda_) - 1.0) < 1e-3


def test_memory_regime_contrast():
    f = create_PAM_5pct()
    M_ball = f.memory_viscosity(2.5 / f.lambda_) / f.eta_p
    M_slip = f.memory_viscosity(2000.0 / f.lambda_) / f.eta_p
    # slipper collapses to ~1e-5; ball-socket is 3-4 orders larger (leading VE term)
    assert M_slip < 1e-4
    assert M_ball > 1e-2
    assert M_ball / M_slip > 1e3


def test_memory_sign_load_reduction_under_squeeze():
    f = create_PAM_5pct()
    cfg = SolverConfig(linear_solver="direct", include_memory=True, include_normal_stress=False)
    sph = Spherical(R_B, C, THETA_MAX, _squeeze_conditions(), Ntheta=48, Nphi=16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve_reynolds(sph, f, cfg, model="viscoelastic")
    assert res.converged
    assert res.F_GNF > 0
    assert res.F_elastic < 0, "memory must reduce load under gap closure"


def test_wedge_load_matches_closed_form_full_sphere_limit():
    # Derived small-eps linearized full-sphere wedge load (spin omega_z, transverse
    # eccentricity e_x = eps c):  F_y = -4 pi eta omega eps R^4 / c^2.
    # As the socket -> full sphere (theta_max -> pi) the solver converges to it.
    eta, omega, eps = 0.03, 100.0, 0.01
    F_closed = -4 * np.pi * eta * omega * eps * R_B ** 4 / C ** 2
    cond = SphericalConditions(p_ch=0.0, e_vec=(eps * C, 0, 0), e_dot=(0, 0, 0),
                               omega_b=(0, 0, omega))
    sph = Spherical(R_B, C, 3.10, cond, Ntheta=140, Nphi=48)  # theta_max ~ 178 deg
    res = solve_reynolds(sph, create_newtonian(eta), SolverConfig(linear_solver="direct"),
                         model="newtonian")
    Fy = sph.load_vector(res.pressure)[1]
    assert abs(Fy / F_closed - 1.0) < 0.01, f"F_y/closed = {Fy/F_closed:.4f}"


def test_wedge_load_linear_in_eccentricity():
    eta, omega = 0.03, 100.0
    fluid = create_newtonian(eta)
    slopes = []
    for eps in (0.005, 0.02):
        cond = SphericalConditions(p_ch=0.0, e_vec=(eps * C, 0, 0), omega_b=(0, 0, omega))
        sph = Spherical(R_B, C, 3.0, cond, Ntheta=120, Nphi=48)
        res = solve_reynolds(sph, fluid, SolverConfig(linear_solver="direct"), model="newtonian")
        slopes.append(sph.load_vector(res.pressure)[1] / eps)
    assert abs(slopes[1] / slopes[0] - 1.0) < 1e-3, f"F_y not linear in eps: {slopes}"


def test_n1_hoop_solver_load_second_order():
    # The full VE solve WITH N1 + hoop converges at 2nd order in the load:
    # halving (Dtheta, Dphi) quarters the successive load change.
    f = create_PAM_5pct()
    sc = SphericalConditions(p_ch=0.0, e_vec=(0, 0, -0.5 * C), e_dot=(0, 0, -1.0e-3),
                             omega_b=(0, 0, 80.0))
    cfg = SolverConfig(linear_solver="direct", include_memory=True, include_normal_stress=True)
    Fs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for Nt in (24, 48, 96):
            sph = Spherical(R_B, C, THETA_MAX, sc, Ntheta=Nt, Nphi=2 * Nt)
            Fs.append(solve_reynolds(sph, f, cfg, model="viscoelastic").F_total)
    d1, d2 = abs(Fs[1] - Fs[0]), abs(Fs[2] - Fs[1])
    assert d2 < d1 and (d1 / d2) > 3.0, f"load not ~2nd order: ratio {d1/d2:.2f}"


def test_M_is_wired_not_eta_T():
    f = create_PAM_5pct()

    class TangentAsMemory:
        def __init__(self, base):
            self._b = base
        def __getattr__(self, n):
            return getattr(self._b, n)
        def memory_viscosity(self, gdot):
            return self._b.tangent_viscosity(gdot)

    cfg = SolverConfig(linear_solver="direct", include_memory=True, include_normal_stress=False)
    sph = Spherical(R_B, C, THETA_MAX, _squeeze_conditions(), Ntheta=48, Nphi=16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rM = solve_reynolds(sph, f, cfg, model="viscoelastic")
        rT = solve_reynolds(sph, TangentAsMemory(f), cfg, model="viscoelastic")
    rel = abs(rT.F_elastic - rM.F_elastic) / abs(rM.F_elastic)
    assert rel > 1.0, f"swapping eta_T for M barely changed load ({rel:.2f}) -> M not wired"
