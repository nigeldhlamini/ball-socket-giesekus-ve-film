"""R-67.2 (D-8): pass fluid-class thermal properties into the ball-socket solver without editing
ball_socket/solver.py or src/ (the D-6 precedent: a wrapper in scripts/).

BallSocketSolver.__init__ already takes ``thermal`` (ball_socket/solver.py:133) and stores it
(``self.thermal = thermal or ThermalProperties()``, :165); campaign.evaluate_x never passes it, so
every campaign row ran on src/thermal.py's water-based defaults (rho 1000, c_p 4 kJ/(kg K), k_f 0.60).
``build`` mirrors run_R57_mesh_error._solver (itself "constructed exactly as
campaign._evaluate_x_inner constructs it") and adds the properties.

Mineral class (R-67.2 step 2):
  rho = 828.5 kg/m3 at 100 C, DERIVED from LeeShangZhang2026 Eq. (1) (rho_0 = 865 at 35 C,
        -0.00065 per K): scripts/p3_lee_fluid.rho_lee(100) = 828.45;
  c_p = 2000 J/(kg K), k_f = 0.15 W/(m K): Chapter 9 l. 235 (uncited there), emitted CITED to
        Totten2011 with citation_verified: false until Nigel clears the page.
"""
from __future__ import annotations

import numpy as np

from ball_socket.solver import BallSocketSolver
from src.thermal import ThermalProperties

MINERAL = dict(rho=828.5, c_p=2000.0, k_f=0.15)
MINERAL_SOURCES = {"rho": "DERIVED: LeeShangZhang2026 Eq. (1) at 100 C (828.45)",
                   "c_p": "CITED Totten2011 (Ch. 9 l. 235), citation_verified: false",
                   "k_f": "CITED Totten2011 (Ch. 9 l. 235), citation_verified: false"}


class PropsSolver(BallSocketSolver):
    """BallSocketSolver with explicit fluid thermal properties (None = the defaults)."""

    def __init__(self, *a, thermal_props: dict | None = None, **k):
        th = ThermalProperties(**thermal_props) if thermal_props else None
        super().__init__(*a, thermal=th, **k)


def build(x, contact, flow_factors, thermal_props=None, grid=(32, 24)):
    import ball_socket.campaign as C
    from ball_socket.geometry import BallSocketGeometry
    from ball_socket.solver import CampaignInput
    from src.solver import SolverConfig
    geom = BallSocketGeometry(c0=x["c0"], theta_min=np.deg2rad(x["theta_min_deg"]))
    slv = PropsSolver(
        geom, model="viscoelastic", Ntheta=grid[0], Nphi=grid[1],
        alpha=C.ALPHA_FIXED, squeeze_phase=C.SQUEEZE_PHASE, spin_ratio=C.SPIN_RATIO,
        config=SolverConfig(include_normal_stress=False, max_iter=80, tol=1e-9, omega=0.7),
        eps_cap=float(C._registered_eps_cap(geom.theta_min)),
        clearance_closure=True, delta_T_datum="bulk", contact=contact, thermal_props=thermal_props)
    slv.flow_factors = None if flow_factors is None else dict(flow_factors)
    ci = CampaignInput(omega=x["omega"], p_ch=x["p_ch"], c0=x["c0"], T_in=x["T_in"],
                       eta_0=x["eta_0"], lam=x["lam"], eta_s_over_eta0=x["eta_s_ratio"],
                       k_F_bs=None, p_case=C.P_CASE_PA)
    return slv, ci


def l2_field(slv, ci, perf):
    """The L2 film temperature field at the converged state, by the solver's own call
    (solver.py:486), for the phi-spread assertion and the film-mean profile."""
    import ball_socket.thermal as bs_thermal
    fl = slv._fluid(ci)
    res, sph = slv._forward(ci, perf.eps_ecc, fl, perf.c_equilibrium)
    h = sph.gap()
    eta_bar = res.eta_bar if res.eta_bar is not None else np.full_like(h, ci.eta_0)
    gdot = sph.gap_averaged_shear_rate(res.pressure, eta_bar)
    return bs_thermal.solve_l2_spherical(sph, eta_bar, gdot, slv.thermal, ci.T_in + 273.15), sph


def phi_spread(T):
    m = T.mean(axis=1, keepdims=True)
    return float(np.max(np.abs(T - m)))
