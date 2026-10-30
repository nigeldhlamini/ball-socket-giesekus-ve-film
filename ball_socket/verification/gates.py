"""Ball-socket verification gates (Ch.6 structural-verification discipline).

RULING 2026-07-26: the production domain is an ANNULUS (fed pocket at
theta_min, socket mouth at theta_max).  Gates that passed on the polar-cap
domain do NOT carry over, so every gate below runs in the ANNULAR production
configuration.  The full-cap (pole) path survives only as the literature
benchmark route: it is exercised by ``ball_socket/tests/test_spherical.py``
(axisymmetric squeeze vs closed-form quadrature, closed-form wedge load) and by
the pole-MMS sub-check of BS-V2.

Run: ``python -m ball_socket.verification.gates`` from the worktree root.
Each gate writes ``BS-VN.json`` next to this file.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Callable, Optional

import numpy as np
from scipy.sparse.linalg import spsolve

from .. import representative as rep

# Representative ANNULAR production geometry
R_B = rep.R_B_M
THETA_MIN = np.deg2rad(rep.THETA_MIN_BAND_DEG[0])
THETA_MAX = np.deg2rad(rep.THETA_MAX_DEG)

GATES = {
    "BS-V1": "Newtonian annular film vs the exact axisymmetric solution "
             "p = A ln(tan(theta/2)) + B; grid order 2.0 +/- 0.1.  Full-cap "
             "literature benchmarks in tests/test_spherical.py",
    "BS-V2": "Manufactured solution, order preserved: ANNULAR (both Dirichlet "
             "ends, production domain) and POLE (cap benchmark path)",
    "BS-V3": "GNF/constitutive layer imported unchanged: Newtonian limit exact "
             "on the annular domain; Picard convergence",
    "BS-V4": "Squeeze-memory: residual exponent De_sq^2 in [1.8, 2.2] on the "
             "annular domain, with the kernel/geometry decomposition reported",
    "BS-V5": "UCM/SOF limits M -> eta_p; ball-socket vs slipper Lambda regime "
             "contrast (constitutive, geometry-independent)",
    "BS-V6": "Thermal L1 isothermal invariant + Q_min/Brinkman routing guard on "
             "the annular domain",
    "BS-V7": "BOX C: thermal-clearance closure.  Delta_alpha=0 reproduces the "
             "closure-off path bitwise; the fixed point matches the closed-form "
             "solution of c = c0 + Dalpha R_b (T(c) - T_in) on a manufactured "
             "T(c) with known slope, and converges monotonically; the sign is "
             "right (heating opens the bronze-on-steel clearance).  FIRST "
             "VERIFICATION, not a regression guard: no prior gate touched "
             "thermal_clearance (see reports/boxb_errata.md E-4)",
    "BS-V8": "BOX C: Patir-Cheng flow factors.  Tables 1-4 complete and typed; "
             "the Eq. 22 closed form agrees with the Table 1 gamma=1 row; "
             "phi_x -> 1 and Phi_s -> 0 as H -> inf; the Eq. 27 identity "
             "phi_y(H,gamma) = phi_x(H,1/gamma) holds on every tabulated "
             "gamma; at gamma=1 phi_theta == phi_phi BITWISE; the two-field "
             "diffusion operator reproduces the scalar operator BITWISE when "
             "D_theta == D_phi, and the MMS order is unchanged by the "
             "extension; h_T_bar >= h with equality in the smooth limit",
    "BS-V9": "BOX C: Greenwood-Tripp.  F_n by quadrature against the EXACT "
             "column of the article's own Appendix 2 table (54 entries, "
             "p.632); the Appendix 2 recurrence; A_c/A_n <= 1 and monotone "
             "decreasing in H; the printed-vs-exact deltas recorded "
             "separately so the article's rounding is never mistaken for "
             "solver error.  The p_a and plastic-cap limbs are BLOCKED and "
             "are recorded as blocked rather than skipped",
}


def run_gate(name: str, fn: Callable[[], dict], outdir: str) -> bool:
    result = fn()
    result.update(gate=name, spec=GATES[name])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{name}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    return bool(result.get("passed", False))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ghost_rhs_phivar(mesh, D, p_out, p_in: Optional[np.ndarray] = None):
    """Dirichlet ghost RHS allowing phi-varying face values at both theta ends."""
    Nt, Np = mesh.Ntheta, mesh.Nphi
    dth, R2, th = mesh.dtheta, mesh.R_b ** 2, mesh.theta
    b = np.zeros(mesh.N)
    i = Nt - 1
    sin_i, sin_ip = np.sin(th[i]), np.sin(th[i] + dth / 2.0)
    for j in range(Np):
        b[mesh._index(i, j)] -= (2.0 * sin_ip * D[i, j]
                                 / (R2 * sin_i * dth ** 2) * p_out[j])
    if p_in is not None and mesh.is_annular:
        i = 0
        sin_i, sin_im = np.sin(th[i]), np.sin(th[i] - dth / 2.0)
        for j in range(Np):
            b[mesh._index(i, j)] -= (2.0 * sin_im * D[i, j]
                                     / (R2 * sin_i * dth ** 2) * p_in[j])
    return b


def _mms_order(theta_min: float) -> dict:
    """Manufactured solution p = a0 + a1 cos t + a2 sin t cos p (regular at the
    pole), exact source f = -2 (a1 cos t + a2 sin t cos p)/R^2."""
    from ..spherical import SphericalMesh, build_spherical_diffusion_operator

    a0, a1, a2 = 1.0, 0.3, 0.2
    R_b = 1.0

    def p_exact(TH, PH):
        return a0 + a1 * np.cos(TH) + a2 * np.sin(TH) * np.cos(PH)

    errs, hs, grids = [], [], []
    for Nt in (24, 48, 96):
        Np = 2 * Nt
        mesh = SphericalMesh(R_b, THETA_MAX, Nt, Np, theta_min=theta_min)
        D = np.ones((Nt, Np))
        A = build_spherical_diffusion_operator(mesh, D)
        f = -2.0 * (a1 * np.cos(mesh.TH)
                    + a2 * np.sin(mesh.TH) * np.cos(mesh.PH)) / R_b ** 2
        p_out = p_exact(np.full(Np, THETA_MAX), mesh.phi)
        p_in = (p_exact(np.full(Np, theta_min), mesh.phi)
                if mesh.is_annular else None)
        b = f.ravel() + _ghost_rhs_phivar(mesh, D, p_out, p_in)
        p = spsolve(A, b).reshape(Nt, Np)
        pex = p_exact(mesh.TH, mesh.PH)
        l2 = np.sqrt(mesh.integrate((p - pex) ** 2)
                     / mesh.integrate(np.ones_like(pex)))
        errs.append(float(l2)); hs.append(mesh.dtheta); grids.append(Nt)
    order = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
    return dict(observed_order=order, grids=grids, l2_errors=errs)


def _annular_case(eps=0.0, hdot=0.0, spin=0.0, p_ch=0.0, p_case=0.0,
                  Nt=48, Np=16, c=15e-6):
    from ..spherical import Spherical, SphericalConditions
    cond = SphericalConditions(p_ch=p_ch, p_case=p_case,
                               e_vec=(0, 0, -eps * c), e_dot=(0, 0, -hdot),
                               omega_b=(0, 0, spin))
    return Spherical(R_B, c, THETA_MAX, cond, Nt, Np, theta_min=THETA_MIN)


# ===========================================================================
# BS-V1 -- Newtonian annular film vs the exact axisymmetric solution
# ===========================================================================

def gate_bs_v1() -> dict:
    """Concentric, source-free, constant-D annulus has the exact solution
        p(theta) = p_in + (p_out - p_in) [L(t) - L(t0)] / [L(t1) - L(t0)],
        L(t) = ln(tan(t/2)),
    from (1/(R^2 sin t)) d/dt(sin t D dp/dt) = 0.  Verifies the metric, both
    Dirichlet closures and the annular grid order.
    """
    from ..reynolds import solve_reynolds
    from src.rheology import create_newtonian
    from src.solver import SolverConfig

    p_in, p_out = 1.0e6, 1.0e5
    fluid, cfg = create_newtonian(0.03), SolverConfig(linear_solver="direct")
    L = lambda t: np.log(np.tan(t / 2.0))

    errs, hs, grids, axi = [], [], [], []
    for Nt in (24, 48, 96):
        sph = _annular_case(p_ch=p_in, p_case=p_out, Nt=Nt, Np=8)
        p = solve_reynolds(sph, fluid, cfg, model="newtonian").pressure
        axi.append(float(np.max(np.abs(p - p.mean(axis=1, keepdims=True)))
                         / np.max(np.abs(p))))
        th = sph.mesh.theta
        ref = p_in + (p_out - p_in) * (L(th) - L(THETA_MIN)) / (L(THETA_MAX) - L(THETA_MIN))
        errs.append(float(np.max(np.abs(p.mean(axis=1) - ref)) / np.max(np.abs(ref))))
        hs.append(sph.mesh.dtheta); grids.append(Nt)

    order = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
    passed = (1.9 <= order <= 2.1) and max(axi) < 1e-10 and errs[-1] < 1e-3
    return dict(passed=bool(passed), observed_order=order, target="2.0 +/- 0.1",
                grids=grids, rel_errors=errs, axisymmetry_dev=axi,
                note="annular production domain; full-cap literature benchmarks "
                     "(quadrature squeeze, closed-form wedge) in tests/test_spherical.py")


# ===========================================================================
# BS-V2 -- manufactured solution, annular (production) + pole (benchmark)
# ===========================================================================

def gate_bs_v2() -> dict:
    ann = _mms_order(THETA_MIN)
    pole = _mms_order(0.0)
    passed = (1.9 <= ann["observed_order"] <= 2.1) and (1.9 <= pole["observed_order"] <= 2.1)
    return dict(passed=bool(passed), target="2.0 +/- 0.1",
                annular=ann, pole_benchmark=pole,
                note="annular = production domain (both Dirichlet ends); pole = "
                     "cap benchmark path, exercised only in verification")


# ===========================================================================
# BS-V3 -- constitutive layer imported unchanged (annular)
# ===========================================================================

def gate_bs_v3() -> dict:
    """A Newtonian fluid driven through the viscoelastic path must reproduce the
    Newtonian solve exactly on the annular domain (the VE path short-circuits to
    Newtonian), and the GNF Picard loop must converge."""
    from ..reynolds import solve_reynolds
    from src.rheology import create_newtonian, GiesekusFluid
    from src.solver import SolverConfig

    cfg = SolverConfig(linear_solver="direct", max_iter=80, tol=1e-10)
    newt = create_newtonian(0.03)
    sph = _annular_case(eps=0.3, hdot=1e-3, spin=100.0, p_ch=1e6, p_case=1e5, Nt=40, Np=16)
    rN = solve_reynolds(sph, newt, cfg, model="newtonian")
    rVE = solve_reynolds(sph, newt, cfg, model="viscoelastic")
    exact = bool(np.array_equal(rVE.pressure, rN.pressure))

    gf = GiesekusFluid(eta_s=0.03, eta_p=0.12, lambda_=3.3e-3, alpha=0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rG = solve_reynolds(sph, gf, cfg, model="GNF")
    thinning = bool(rG.eta_bar.min() < gf.eta_0)
    passed = exact and bool(rG.converged) and thinning
    return dict(passed=bool(passed), newtonian_limit_exact=exact,
                gnf_converged=bool(rG.converged), gnf_iterations=int(rG.iterations),
                shear_thinning_direction_ok=thinning,
                eta_min_over_eta0=float(rG.eta_bar.min() / gf.eta_0))


# ===========================================================================
# BS-V4 -- squeeze-memory residual scaling (annular) + decomposition
# ===========================================================================

def gate_bs_v4() -> dict:
    """Pure-squeeze UCM-ish annulus (alpha -> 0 so M -> eta_p, no shear-thinning
    contamination), normal stress off.  Fit dev = a De + K De^p to the fractional
    memory load reduction dev = 1 - F_VE/F_newt.

    Gate criterion (Paper-1 p2_gate methodology): the RESIDUAL beyond the linear
    closure is quadratic, i.e. p in [1.8, 2.2].

    Decomposition diagnostic (licensed, sec 5): the leading coefficient a
    factorises as (kernel factor M/eta_0 -> eta_p/eta_0 in this limit) x
    (geometric projection integral).  Both are reported; the geometric factor is
    the ball-socket-specific part.  NOTE: the numeric leading coefficient is
    EMBARGOED from thesis prose pending Chat/Nigel review.
    """
    from ..reynolds import solve_reynolds
    from src.rheology import GiesekusFluid, create_newtonian
    from src.solver import SolverConfig
    from scipy.optimize import curve_fit

    C = 15e-6
    ETA0, ETAS, LAM = 0.15, 0.03, 5.0e-4
    fluid = GiesekusFluid(eta_s=ETAS, eta_p=ETA0 - ETAS, lambda_=LAM, alpha=1e-9)
    newt = create_newtonian(ETA0)
    cfg = SolverConfig(linear_solver="direct", include_memory=True,
                       include_normal_stress=False, max_iter=120, tol=1e-11, omega=0.6)

    # The closed-form projection factor is derived in the CONCENTRIC limit, so
    # the fit is taken at eps = 0 to isolate the decomposition from
    # finite-eccentricity contamination.  (The squeeze source e_dot . n_hat is
    # non-zero at eps = 0, so this limit is directly reachable.)  For reference,
    # the same fit at eps = 0.1 gives 0.578 instead of 0.536.
    EPS_FIT = 0.0
    De_sq = np.array([0.001, 0.002, 0.004, 0.008, 0.016])
    devs = []
    for De in De_sq:
        hdot = De * C / LAM
        sph = _annular_case(eps=EPS_FIT, hdot=hdot, Nt=48, Np=8, c=C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rv = solve_reynolds(sph, fluid, cfg, model="viscoelastic")
            rn = solve_reynolds(sph, newt, cfg, model="newtonian")
        devs.append(1.0 - rv.F_total / rn.F_total)
    devs = np.array(devs)

    (a_fit, K_fit, p_fit), _ = curve_fit(
        lambda De, a, K, p: a * De + K * De ** p, De_sq, devs,
        p0=(0.7, -0.5, 2.0), maxfev=40000)
    # RULING A6 (2026-07-26): the fitted leading coefficient must be reproduced
    # by the ANALYTICALLY COMPUTED geometric projection integral to within 5 %:
    #     | c_fitted / (c_kernel * G_analytic) - 1 | <= 0.05
    # G is now DERIVED in closed form (bsv4_projection_integral.py): the
    # concentric axisymmetric squeeze problem gives
    #   p0 ~ -cos t / 2 + a ln tan(t/2),   p1 ~ sin^2 t / 4 + a ln sin t + b ln tan(t/2)
    # and G = int p1 cos t sin t dt / int p0 cos t sin t dt over the annulus.
    # G -> 1 in the planar limit (validated).  G is DOMAIN-DEPENDENT, not a
    # constant: it spans ~0.53-0.67 across the pre-registered theta ranges.
    from .bsv4_projection_integral import G as _G

    # kernel factor MEASURED at the test condition (not assumed): the UCM-like
    # fluid has constant eta_bar = eta_0 and M = eta_p, so M/eta_bar is uniform.
    kernel_factor = float(np.mean(fluid.memory_viscosity(np.ones(4)) / ETA0))
    G_analytic = float(_G(np.degrees(THETA_MIN), np.degrees(THETA_MAX))[0])
    predicted = kernel_factor * G_analytic
    agreement = abs(a_fit / predicted - 1.0)
    residual_ok = bool(1.8 <= p_fit <= 2.2)
    passed = bool(residual_ok and agreement <= 0.05)

    return dict(passed=passed,
                residual_exponent=float(p_fit), residual_target="[1.8, 2.2]",
                residual_criterion_met=residual_ok,
                leading_coefficient=float(a_fit),
                kernel_factor_measured=kernel_factor,
                projection_factor_analytic=G_analytic,
                predicted_leading_coefficient=predicted,
                agreement_vs_analytic=float(agreement),
                agreement_tolerance=0.05,
                theta_range_deg=[float(np.degrees(THETA_MIN)),
                                 float(np.degrees(THETA_MAX))],
                eccentricity_of_fit=EPS_FIT,
                quad_coefficient=float(K_fit),
                De_sq=[float(d) for d in De_sq],
                deviations=[float(d) for d in devs],
                FINDING="The closed-form G is validated EXACTLY (0.0 %) against the "
                        "solver when hdot is placed INSIDE the surface divergence "
                        "(mobility form) in the concentric limit.  The implemented "
                        "operator follows Ch.9 D1 and places hdot OUTSIDE the "
                        "divergence (row scaling), which differs by ~7.8 % at "
                        "eps -> 0 and ~16 % at eps = 0.1.  The two forms coincide "
                        "only when hdot is spatially uniform (the planar slipper); "
                        "on the sphere hdot ~ cos(theta) varies, so they do not.  "
                        "Deciding which form Ch.9 should carry is a physics/chapter "
                        "call for Chat/Nigel, NOT a code change.",
                EMBARGO="leading_coefficient and projection factor remain EMBARGOED "
                        "from thesis prose until Chat clears them",
                note="annular production domain; residual-exponent criterion per "
                     "Paper-1 p2_gate (the linear term IS the closure)")


# ===========================================================================
# BS-V5 -- constitutive limits (geometry-independent)
# ===========================================================================

def gate_bs_v5() -> dict:
    from src.rheology import GiesekusFluid, create_PAM_5pct

    ucm = GiesekusFluid(eta_s=0.03, eta_p=0.12, lambda_=3.3e-3, alpha=1e-13)
    ucm_ratio = float(ucm.memory_viscosity(100.0 / ucm.lambda_) / ucm.eta_p)
    f = create_PAM_5pct()
    sof_ratio = float(f.memory_viscosity(1e-3 / f.lambda_) / f.eta_p)
    M_ball = float(f.memory_viscosity(2.5 / f.lambda_) / f.eta_p)
    M_slip = float(f.memory_viscosity(2000.0 / f.lambda_) / f.eta_p)
    passed = (abs(ucm_ratio - 1.0) < 1e-4 and abs(sof_ratio - 1.0) < 1e-3
              and M_ball > 1e-2 and M_slip < 1e-4 and M_ball / M_slip > 1e3)
    return dict(passed=bool(passed), ucm_M_over_eta_p=ucm_ratio,
                sof_M_over_eta_p=sof_ratio,
                M_over_eta_p_ball_Lambda2p5=M_ball,
                M_over_eta_p_slipper_Lambda2000=M_slip,
                ratio_ball_over_slipper=float(M_ball / M_slip),
                note="constitutive layer imported from src.rheology, not re-derived")


# ===========================================================================
# BS-V6 -- thermal invariant + routing guard (annular)
# ===========================================================================

def gate_bs_v6() -> dict:
    """(1) L1 isothermal invariant: with beta_eta -> 0 the thermally-scaled
    re-solve reproduces the isothermal load exactly; it is non-trivially active
    at beta_eta = 2500 K.  (2) Routing: Q_min guard (Q_out <= Q_min => L2,
    subsuming the 10 K catch) and the Brinkman escalation with V_char =
    omega_b R_b sin(theta_max).  Both on the annular domain.
    """
    from ..reynolds import solve_reynolds
    from .. import thermal as bt
    from src.rheology import create_newtonian
    from src.solver import SolverConfig
    from src.coupling import ThermallyScaledFluid
    from src.thermal import ThermalProperties

    eta0 = 0.05
    cfg = SolverConfig(linear_solver="direct")
    newt = create_newtonian(eta0)
    sph = _annular_case(eps=0.4, hdot=1e-3, spin=150.0, p_ch=1e6, p_case=1e5,
                        Nt=40, Np=24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F_iso = solve_reynolds(sph, newt, cfg, model="newtonian").F_total

    T_bulk = 353.15

    def l1_load(beta_eta):
        th = ThermalProperties(beta_eta=beta_eta)
        scaled = ThermallyScaledFluid(newt,
                                      float(th.viscosity_ratio(T_bulk)),
                                      float(th.relaxation_time_ratio(T_bulk)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return solve_reynolds(sph, scaled, cfg, model="newtonian").F_total

    invariant_rel = abs(l1_load(0.0) - F_iso) / abs(F_iso)
    coupling_rel = abs(l1_load(2500.0) - F_iso) / abs(F_iso)

    th = ThermalProperties()
    # Q_min guard
    P_diss = 30.0
    q_hi, Q_min = bt.q_min_route_to_l2(P_diss, 1.0e-3, th)     # ample outflow -> L1
    q_lo, _ = bt.q_min_route_to_l2(P_diss, 1.0e-9, th)         # starved -> L2
    q_neg, _ = bt.q_min_route_to_l2(P_diss, -1.0e-7, th)       # inward -> L2
    # Brinkman
    Br_lo = bt.brinkman(eta0, 150.0 * R_B * np.sin(THETA_MAX), th)
    Br_hi = bt.brinkman(eta0, 4000.0 * R_B * np.sin(THETA_MAX), th)

    routing_ok = ((not q_hi) and q_lo and q_neg
                  and Br_lo < bt.BR_THRESHOLD < Br_hi)
    passed = invariant_rel < 1e-10 and routing_ok and coupling_rel > 1e-6
    return dict(passed=bool(passed),
                l1_isothermal_invariant_rel_err=float(invariant_rel),
                l1_coupling_active_rel=float(coupling_rel),
                q_min_guard={"Q_min": float(Q_min), "ample_outflow_L2": bool(q_hi),
                             "starved_L2": bool(q_lo), "inward_L2": bool(q_neg)},
                brinkman={"Br_slow": float(Br_lo), "Br_fast": float(Br_hi),
                          "threshold": bt.BR_THRESHOLD},
                V_char="omega_b * R_b * sin(theta_max)",
                note="annular domain; spherical L2 FVM (walls at T_in) wired in "
                     "ball_socket/thermal.py and used by solve_equilibrium")



# ---------------------------------------------------------------------------
# BS-V7 -- BOX C thermal-clearance closure (brief sec 2.4)
# ---------------------------------------------------------------------------

def gate_bs_v7() -> dict:
    import numpy as _np
    from ..solver import BallSocketSolver, CampaignInput, Performance
    from ..geometry import BallSocketGeometry, thermal_clearance, delta_alpha_th
    from .. import representative as _rep

    R_b = _rep.R_B_M
    # PROBE POINTS.  Taken from the registered box-B audit
    # (results/ch9/audit_N8192_seed20260727.parquet, sha f4dd4c00...3b00e) by a
    # stated rule and not by eye: of the 40 converged rows with the lowest
    # p_ch, the second L1-routed row and the first L2-routed row.  Real
    # operating points rather than invented ones, and both converge on the
    # coarse gate grid.  BOTH routes are exercised because fork F1's two arms
    # coincide identically on L1 and can only differ on L2.
    PROBE_L1 = dict(omega=272.15470916125923, p_ch=2014598.5931158066, c0=1.4727968137711285e-05, T_in=65.078886449337, eta_0=0.047755625071446435, lam=0.0014257626576117902, eta_s_ratio=0.4275217922404408, theta_min_deg=23.924135696142912)
    PROBE_L2 = dict(omega=198.84754496160895, p_ch=2044428.8700819016, c0=8.795981619507073e-06, T_in=85.99343705922365, eta_0=0.15081667211715094, lam=4.900394742680383e-05, eta_s_ratio=0.5730807626619935, theta_min_deg=29.29352268576622)

    def _mk(d, **kw):
        g = BallSocketGeometry(c0=d["c0"], R_b=R_b,
                               theta_max=_np.deg2rad(_rep.THETA_MAX_DEG),
                               theta_min=_np.deg2rad(d["theta_min_deg"]))
        slv = BallSocketSolver(g, model="viscoelastic", Ntheta=16, Nphi=12, **kw)
        ci = CampaignInput(omega=d["omega"], p_ch=d["p_ch"], c0=d["c0"],
                           T_in=d["T_in"], eta_0=d["eta_0"], lam=d["lam"],
                           eta_s_over_eta0=d["eta_s_ratio"])
        return slv, ci

    geom = BallSocketGeometry(c0=PROBE_L1["c0"], R_b=R_b,
                              theta_max=_np.deg2rad(_rep.THETA_MAX_DEG),
                              theta_min=_np.deg2rad(PROBE_L1["theta_min_deg"]))
    out = {}

    # -- (1) Delta_alpha = 0 reproduces the closure-off path BITWISE ---------
    # Tested WITHIN the environment.  Bitwise comparison against the stored
    # box-B parquet is NOT attempted here: that artefact was produced on
    # 2026-07-27 under a different numpy/scipy/BLAS and reproduces only to
    # ~1e-12 relative, which is environment drift and not a property of box C.
    # See the Phase 1 hand-back; the tolerance against the stored artefact is
    # an author decision and is deliberately NOT set inside this gate.
    base, x = _mk(PROBE_L1, clearance_closure=False)
    zero, _ = _mk(PROBE_L1, clearance_closure=True, delta_T_datum="bulk")
    zero.socket_alpha_th = 12e-6          # == ball alpha_th  =>  Delta_alpha = 0
    pb, pz = base.solve_equilibrium(x), zero.solve_equilibrium(x)
    bit = {k: (float(getattr(pb, k)) == float(getattr(pz, k)))
           for k in ("h_min", "F", "T_max")}
    out["delta_alpha_zero_bitwise"] = bit
    out["delta_alpha_zero_n_outer"] = pz.n_outer
    out["delta_alpha_zero_c_equals_c0"] = (pz.c_equilibrium == x.c0)

    # -- (2) manufactured T(c) with known slope vs the closed form -----------
    # T(c) = T_in + A + B c  =>  c* = (c0 + k A) / (1 - k B),  k = Dalpha R_b.
    A, B = 50.0, 2.0e5          # 50 K offset, +2e5 K/m (= +0.2 K/um) slope
    k = delta_alpha_th() * R_b
    c_closed = (x.c0 + k * A) / (1.0 - k * B)

    class _Manufactured(BallSocketSolver):
        def _equilibrium_at_clearance(self, xx, c):
            T = xx.T_in + A + B * c
            return Performance(F=0.0, h_min=c, T_max=T, P_loss=0.0, Q_leak=0.0,
                               De_sq=0.0, eps_ecc=0.0, converged=True,
                               T_bulk=T, c_equilibrium=c)

    man = _Manufactured(geom, clearance_closure=True, delta_T_datum="bulk")
    traj = []
    c_it = float(x.c0)
    for _ in range(man.max_outer):
        T = x.T_in + A + B * c_it
        c_star = thermal_clearance(x.c0, R_b, T - x.T_in)
        traj.append(c_it)
        if abs(c_star - c_it) / c_it < man.tol_c:
            break
        c_it = man.omega_c * c_star + (1 - man.omega_c) * c_it
    perf = man.solve_equilibrium(x)

    steps = _np.diff(_np.array(traj))
    # ACCURACY BOUND, DERIVED not fitted.  The iteration terminates on the
    # fixed-point residual |g(c) - c| / c < tol_c, so it cannot deliver an
    # accuracy better than tol_c.  For a contraction with |g'| = L the distance
    # to the true fixed point is bounded by residual / (1 - L), and here
    # L = k B exactly, because the manufactured T(c) has a known slope.  The
    # gate therefore asserts against tol_c / (1 - L) -- a consequence of the
    # REGISTERED tol_c and the manufactured slope, with no number chosen after
    # seeing the result.
    L = k * B
    err_bound = man.tol_c / (1.0 - L)
    rel_err = abs(perf.c_equilibrium - c_closed) / c_closed
    out["manufactured"] = {
        "c_closed_form": c_closed,
        "c_converged": float(perf.c_equilibrium),
        "rel_error": rel_err,
        "contraction_L": float(L),
        "tol_c": man.tol_c,
        "derived_error_bound": float(err_bound),
        "within_derived_bound": bool(rel_err < err_bound),
        "n_outer": int(perf.n_outer),
        "monotone": bool(_np.all(steps > 0) or _np.all(steps < 0)) if steps.size else True,
        "steps_shrink": bool(_np.all(_np.abs(steps[1:]) < _np.abs(steps[:-1])))
                        if steps.size > 1 else True,
    }

    # -- (3) the SIGN, not the magnitude -------------------------------------
    d_alpha = delta_alpha_th()
    c_hot = thermal_clearance(x.c0, R_b, +60.0)
    c_cold = thermal_clearance(x.c0, R_b, -60.0)
    out["sign"] = {
        "delta_alpha_th": d_alpha,
        "delta_alpha_positive": bool(d_alpha > 0.0),
        "heating_opens_clearance": bool(c_hot > x.c0),
        "cooling_closes_clearance": bool(c_cold < x.c0),
    }

    # -- (4) the two F1 arms are both live and coincide on L1 ----------------
    arms = {}
    for tag, probe in (("L1", PROBE_L1), ("L2", PROBE_L2)):
        sb, xb = _mk(probe, clearance_closure=True, delta_T_datum="bulk")
        sp, xp = _mk(probe, clearance_closure=True, delta_T_datum="peak")
        rb, rp = sb.solve_equilibrium(xb), sp.solve_equilibrium(xp)
        arms[tag] = {"route": rb.thermal_mode,
                     "c_bulk": float(rb.c_equilibrium),
                     "c_peak": float(rp.c_equilibrium),
                     "arms_identical": bool(rb.c_equilibrium == rp.c_equilibrium),
                     "T_max_minus_T_bulk_K": float(rp.T_max - rp.T_bulk),
                     "n_outer_bulk": int(rb.n_outer),
                     "n_outer_peak": int(rp.n_outer)}
    out["fork_F1"] = {
        "by_route": arms,
        "arms_coincide_on_L1": bool(arms["L1"]["arms_identical"]),
        "arms_differ_on_L2": bool(not arms["L2"]["arms_identical"]),
        "note": ("both arms are implemented and neither is a default; box C "
                 "reports both (standing rule 18)"),
    }

    m = out["manufactured"]
    out["passed"] = bool(
        all(bit.values()) and out["delta_alpha_zero_c_equals_c0"]
        and m["within_derived_bound"] and m["monotone"] and m["steps_shrink"]
        and out["sign"]["delta_alpha_positive"]
        and out["sign"]["heating_opens_clearance"]
        and out["sign"]["cooling_closes_clearance"])
    return out



# ---------------------------------------------------------------------------
# BS-V8 -- BOX C Patir-Cheng flow factors (brief sec 3.8)
# ---------------------------------------------------------------------------

def gate_bs_v8() -> dict:
    import numpy as _np
    from .. import mixed as M
    from ..spherical import build_spherical_diffusion_operator as _B
    from ..campaign_config_v5 import PATIR_CHENG as _PC
    out = {}

    # -- (1) the transcribed tables are complete and typed -------------------
    t1 = _PC["pressure_flow_factor"]["table_1"]
    t2 = _PC["shear_flow_factor"]["table_2"]
    gammas = ("1/9", "1/6", "1/3", "1", "3", "6", "9")
    out["tables"] = {
        "table_1_rows": sorted(t1), "table_2_rows": sorted(t2),
        "complete": bool(set(t1) == set(gammas) and set(t2) == set(gammas)),
        "table_1_arity_ok": all(len(v) == 3 for v in t1.values()),
        "table_2_arity_ok": all(len(v) == 5 for v in t2.values()),
        "status_cited": _PC["status"] == "CITED",
    }

    # -- (2) Eq. 22 closed form vs the Table 1 gamma = 1 row -----------------
    C, r, _cut = t1["1"]
    Hs = _np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    closed = 1.0 - 0.90 * _np.exp(-0.56 * Hs)
    out["eq22_matches_table1"] = {
        "C": C, "r": r,
        "bitwise": bool(_np.array_equal(M.phi_x(Hs, 1.0), closed))}

    # -- (3) limits ----------------------------------------------------------
    out["limits"] = {
        "phi_x_at_1e3": float(M.phi_x(1e3, 1.0)),
        "phi_x_to_one": bool(abs(float(M.phi_x(1e3, 1.0)) - 1.0) < 1e-12),
        "Phi_s_at_1e3": float(M.Phi_s(1e3, 1.0)),
        "Phi_s_to_zero": bool(abs(float(M.Phi_s(1e3, 1.0))) < 1e-30)}

    # -- (4) Eq. 27 identity on every tabulated gamma -------------------------
    vals = {"1/9": 1/9, "1/6": 1/6, "1/3": 1/3, "1": 1.0, "3": 3.0,
            "6": 6.0, "9": 9.0}
    eq27 = {}
    for k, g in vals.items():
        if abs(1.0 / g - round(1.0 / g, 12)) < 1e-12 or (1.0 / g) in vals.values():
            pass
        try:
            eq27[k] = bool(_np.array_equal(M.phi_y(Hs, g), M.phi_x(Hs, 1.0 / g)))
        except ValueError:
            eq27[k] = "reciprocal not tabulated"
    out["eq27_identity"] = eq27
    out["eq27_holds"] = all(v is True for v in eq27.values())

    # -- (5) gamma = 1 isotropy: phi_theta == phi_phi BITWISE -----------------
    out["isotropic_identity_bitwise"] = bool(
        _np.array_equal(M.phi_x(Hs, 1.0), M.phi_y(Hs, 1.0)))

    # -- (6) two-field operator == scalar operator, BITWISE, on the MMS mesh --
    from ..spherical import SphericalMesh as _SM
    # Same mesh family _mms_order uses: R_b = 1, THETA_MAX, annular at
    # THETA_MIN, Np = 2 Nt.
    mesh = _SM(1.0, THETA_MAX, 24, 48, theta_min=THETA_MIN)
    rng = _np.random.default_rng(20260822)
    D = rng.uniform(1e-7, 1e-5, size=(mesh.Ntheta, mesh.Nphi))
    A_scalar, A_pair = _B(mesh, D), _B(mesh, (D, D))
    Da = rng.uniform(1e-7, 1e-5, size=D.shape)
    out["two_field_operator"] = {
        "scalar_vs_pair_nnz_diff": int((A_scalar - A_pair).nnz),
        "bitwise_identical": bool((A_scalar - A_pair).nnz == 0),
        "anisotropic_actually_differs": bool(abs(A_scalar - _B(mesh, (D, Da))).max() > 0)}

    # -- (7) the extension did not move the MMS order ------------------------
    mms = _mms_order(THETA_MIN)
    order = mms.get("order", mms.get("observed_order"))
    out["mms_order_after_extension"] = mms
    out["mms_order_ok"] = bool(order is not None and 1.8 <= float(order) <= 2.2)

    # -- (8) the variance-ratio combination uses BOTH surfaces ---------------
    s1, s2 = 6.25e-8, 4.4e-7            # ground ball vs fine-turned bronze
    V1 = s1 ** 2 / (s1 ** 2 + s2 ** 2)
    out["variance_ratio"] = {
        "V_r1": V1, "V_r2": 1.0 - V1, "sums_to_one": bool(abs(V1 + (1 - V1) - 1) < 1e-15),
        "phi_s_at_H2": float(M.phi_s(2.0, s1, s2)),
        "differs_from_single_surface": bool(
            abs(float(M.phi_s(2.0, s1, s2)) - float(M.Phi_s(2.0, 1.0))) > 1e-6),
        "note": "steel-on-bronze Rq differ ~10x, so the terms do not cancel"}

    # -- (9) average gap: h_T_bar >= h, -> h in the smooth limit -------------
    sig = 3.36e-7
    hs = _np.array([0.3, 1.0, 3.0, 10.0, 20.0]) * sig
    ratios = [float(M.h_T_bar(float(h), sig)) / float(h) for h in hs]
    out["average_gap"] = {
        "ratios_h_T_over_h": ratios,
        "never_below_one": bool(all(rr >= 1.0 - 1e-12 for rr in ratios)),
        "tends_to_one": bool(abs(ratios[-1] - 1.0) < 1e-9)}

    out["passed"] = bool(
        out["tables"]["complete"] and out["tables"]["status_cited"]
        and out["tables"]["table_1_arity_ok"] and out["tables"]["table_2_arity_ok"]
        and out["eq22_matches_table1"]["bitwise"]
        and out["limits"]["phi_x_to_one"] and out["limits"]["Phi_s_to_zero"]
        and out["eq27_holds"] and out["isotropic_identity_bitwise"]
        and out["two_field_operator"]["bitwise_identical"]
        and out["two_field_operator"]["anisotropic_actually_differs"]
        and out["mms_order_ok"]
        and out["variance_ratio"]["sums_to_one"]
        and out["variance_ratio"]["differs_from_single_surface"]
        and out["average_gap"]["never_below_one"]
        and out["average_gap"]["tends_to_one"])
    return out



# ---------------------------------------------------------------------------
# BS-V9 -- BOX C Greenwood-Tripp (brief sec 3.8)
# ---------------------------------------------------------------------------

def gate_bs_v9() -> dict:
    import hashlib as _hl
    import json as _json
    import os as _os
    import numpy as _np
    from .. import mixed as M

    out = {}
    tpath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "gt1970_appendix2_Fn_table.json")
    raw = open(tpath, "rb").read()
    tab = _json.loads(raw)
    out["table"] = {"sha256": _hl.sha256(raw).hexdigest(),
                    "source": tab["source"]["location_in_source"],
                    "doi": tab["source"]["doi"],
                    "entries": tab["transcription_check"]["entries_checked"]}

    # -- (1) quadrature against the EXACT column, all 54 entries -------------
    # Gated on EXACT, never on PRINTED: the printed column carries the
    # article's own 5-dp rounding and three entries depart from the exact
    # value by more than half a ULP.  Gating on printed would bake the
    # article's rounding into the solver's tolerance.
    orders = {"F_0": 0.0, "F_1/2": 0.5, "F_1": 1.0,
              "F_3/2": 1.5, "F_2": 2.0, "F_5/2": 2.5}
    worst_exact = 0.0
    worst_printed = 0.0
    printed_deltas = []
    for row in tab["rows"]:
        h = row["h"]
        for key, n in orders.items():
            ex, pr = row[key]["exact"], row[key]["printed"]
            mine = float(M.F_n(n, h))
            worst_exact = max(worst_exact, abs(mine - ex))
            dp = abs(pr - ex)
            worst_printed = max(worst_printed, dp)
            if dp > 0.5e-5:
                printed_deltas.append({"h": h, "order": key, "printed": pr,
                                       "exact": ex, "abs_diff": dp})
    out["quadrature_vs_exact"] = {
        "worst_abs_diff": worst_exact, "n_entries": 54, "tolerance": 1e-10,
        "passed": bool(worst_exact < 1e-10)}
    out["printed_vs_exact"] = {
        "worst_abs_diff": worst_printed,
        "entries_beyond_half_ulp": printed_deltas,
        "note": ("the article's own rounding, NOT solver error.  F_1(1.0) "
                 "printed 0.08333 against an exact 0.08331547 is about 3 ULP; "
                 "the other two are half-ULP round-down.  A typesetting error "
                 "cannot be distinguished from a scan misread here, and does "
                 "not need to be: the gate is on the exact column.")}

    # -- (2) the Appendix 2 recurrence, correctly indexed --------------------
    # F_{n+1} = n F_{n-1} - h F_n.  n indexes the MIDDLE term, and the
    # recurrence steps by 1 while the table steps by 1/2, so exactly two
    # triples are checkable from the table itself: n = 1 and n = 3/2.
    # Mapping n = 1/2 onto (F_0, F_1/2, F_1) is WRONG -- it needs F_{-1/2}.
    rec = 0.0
    for row in tab["rows"]:
        h = row["h"]
        rec = max(rec, abs(row["F_2"]["exact"]
                           - (1.0 * row["F_0"]["exact"] - h * row["F_1"]["exact"])))
        rec = max(rec, abs(row["F_5/2"]["exact"]
                           - (1.5 * row["F_1/2"]["exact"] - h * row["F_3/2"]["exact"])))
    solver_rec = max(abs(M.F_n_recurrence_residual(1.5, h)) for h in (1.0, 2.0, 3.0))
    out["recurrence"] = {"on_table_exact": rec, "on_solver": solver_rec,
                         "passed": bool(rec < 1e-15 and solver_rec < 1e-12)}

    # -- (3) contact area: bounded and monotone ------------------------------
    lo, hi = M.eta_beta_sigma_band()
    Hs = _np.linspace(0.0, 4.0, 41)
    vals_hi = [M.contact_area_fraction(float(h), hi) for h in Hs]
    out["contact_area"] = {
        "eta_beta_sigma_band": [lo, hi],
        "max_fraction": max(vals_hi),
        "bounded_by_one": bool(all(0.0 <= v <= 1.0 for v in vals_hi)),
        "monotone_decreasing": bool(all(b <= a for a, b in zip(vals_hi, vals_hi[1:]))),
        "at_H_4": vals_hi[-1]}

    # -- (4) the corrected far-tail exponent ---------------------------------
    ratios = {str(h): float(M.F_n_asymptotic(2.5, h) / M.F_n(2.5, h))
              for h in (6.0, 10.0, 12.0)}
    out["asymptotic_exponent"] = {
        "ratios_to_quadrature": ratios,
        "converges_to_one": bool(abs(ratios["12.0"] - 1.0) < 0.01),
        "note": ("the article prints h^-n; the correct power is h^-(n+1).  "
                 "The h^-n form diverges linearly in h.  Derivation in "
                 "mixed.F_n_asymptotic.")}

    # -- (5) the BLOCKED limbs, recorded as blocked rather than skipped ------
    out["blocked_limbs"] = {
        "entry_points": M.blocked_entry_points(),
        "note": ("p_a, the plastic branch and psi cannot be gated until "
                 "GT_PARAM_BAND.sigma_s_over_beta_a and "
                 "HARDNESS_CONVERSION.H_ind_band_Pa are CITED.  Recorded so "
                 "the gate's coverage is explicit: BS-V9 currently verifies "
                 "the F_n machinery and the contact AREA, not the contact "
                 "LOAD."),
        "owed_when_unblocked": [
            "p_a -> 0 for H >= the cut-off",
            "plastic branch active exactly where GT Eq. 13 exceeds GT Eq. 7",
            "dimensional check on the K-prime prefactor"]}

    out["passed"] = bool(out["quadrature_vs_exact"]["passed"]
                         and out["recurrence"]["passed"]
                         and out["contact_area"]["bounded_by_one"]
                         and out["contact_area"]["monotone_decreasing"]
                         and out["asymptotic_exponent"]["converges_to_one"])
    return out


_GATE_FNS = {"BS-V1": gate_bs_v1, "BS-V2": gate_bs_v2, "BS-V3": gate_bs_v3,
             "BS-V4": gate_bs_v4, "BS-V5": gate_bs_v5, "BS-V6": gate_bs_v6,
             "BS-V7": gate_bs_v7, "BS-V8": gate_bs_v8, "BS-V9": gate_bs_v9}


def run_all(outdir: str | None = None) -> dict:
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    return {name: run_gate(name, fn, outdir) for name, fn in _GATE_FNS.items()}


if __name__ == "__main__":
    for k, v in run_all().items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
