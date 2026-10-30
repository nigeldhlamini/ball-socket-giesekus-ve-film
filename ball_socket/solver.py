"""BallSocketSolver -- eccentricity equilibrium search + performance extraction.

The forward spherical solve (``ball_socket.spherical`` + ``reynolds``) is wrapped
in an axial load-balance search over the eccentricity, following the Ch.9
kinematic model (sec:bs_kinematics): squeeze is the PRIMARY load-support
mechanism with hdot ~ c*omega*eps, spin adds V_spin = omega_b R_b sin(theta),
and the axial piston force F_app is balanced by the squeeze-film load F_z(eps).
Non-convergence is eccentricity blow-up (eps -> 1 under a load the squeeze film
cannot carry); it is raised as ``NonConvergence`` and never clipped.

MODELLING CONVENTIONS -- grounded in Ch.9, but the O(1) prefactors and open axes
are FLAGGED and must be confirmed before any Phase-2 campaign FREEZE:
  * D2 (OPEN): F_app = k_F_bs * W_sq_ref, W_sq_ref = 3 pi eta0 omega R_b^4 / c0^2
    (``squeeze_capacity_scale``) -- decouples the AR from one geometry.
  * ``squeeze_phase`` (PROPOSAL=1.0): |e_dot| = squeeze_phase*omega*|e|, the
    orbit-rate snapshot of hdot ~ c*omega*eps (Ch.9 sec:bs_kinematics); O(1),
    phase-set.
  * ``spin_ratio`` (PROPOSAL=1.0): omega_b = spin_ratio*omega; friction-induced
    spin magnitude is uncertain (Ch.9), and affects shear/dissipation, not the
    leading axial squeeze load.
  * ``alpha`` (=0.25): campaign-fixed Giesekus mobility (scaffold operating point).
  * ``T_max``: STUB = T_in (isothermal).  Thermal coupling (L1/L2 routing) is
    gate BS-V6 and is NOT yet wired -- T_max is a placeholder, not a physical
    value, and feasibility on g2 must not be trusted until it is wired.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from src.rheology import GiesekusFluid
from src.solver import SolverConfig
from src.thermal import ThermalProperties

from .spherical import Spherical, SphericalConditions
from .reynolds import solve_reynolds
from .geometry import BallSocketGeometry, thermal_clearance
from .outcomes import (NonConvergence, ClearanceNonConvergence,
                       ContactLimit)
from . import thermal as bs_thermal


def squeeze_capacity_scale(eta0: float, omega: float, R_b: float, c0: float) -> float:
    """W_sq_ref = 3 pi eta0 omega R_b^4 / c0^2 (Decision D2 proposal; scaffold).

    Dimensional check: [Pa s][1/s][m^4]/[m^2] = [N].  Exact prefactor is a
    convention; what matters is decoupling the AR from one geometry.
    """
    return 3.0 * np.pi * eta0 * omega * R_b ** 4 / c0 ** 2


@dataclass(frozen=True)
class CampaignInput:
    """8-D campaign input (ranges/axes per CODE_BRIEF Phase 2; D1 open)."""
    omega: float             # shaft speed [rad/s]
    p_ch: float              # chamber (socket-mouth) pressure [Pa]
    c0: float                # cold radial clearance [m]
    T_in: float              # inlet temperature [C]
    eta_0: float             # zero-shear viscosity [Pa s]
    lam: float               # relaxation time lambda [s]
    eta_s_over_eta0: float   # solvent fraction beta = eta_s / eta_0
    # BOX A (frozen, retained for reproducibility): k_F_bs is an input axis and
    # F_app = k_F_bs * W_sq_ref.  BOX B (re-posed 2026-07-27, ruling L-9):
    # k_F_bs is REMOVED and the load is coupled, F_app = p_ch * A_p, which is
    # what makes the hydrodynamic share positive by construction.
    # k_F_bs = None selects the coupled (box B) form.
    k_F_bs: float | None = None
    fluid_class: str = "mineral"
    p_case: float = 1.0e5    # socket-mouth (case drain) pressure [Pa]; CONVENTION


@dataclass
class Performance:
    """Raw performance at the equilibrium eccentricity.  Feasibility (Outcome)
    is decided separately by the ConstraintSet (Phase 2), never here."""
    F: float                 # equilibrium load F_z [N] (== F_app to tol)
    h_min: float             # minimum film thickness [m]
    T_max: float             # STUB = T_in until BS-V6 thermal wiring
    P_loss: float            # viscous dissipation [W]
    Q_leak: float            # leakage at the socket mouth [m^3/s]
    De_sq: float             # squeeze Deborah number lambda|hdot|/c ~ lambda*omega*eps (Ch.9)
    eps_ecc: float           # equilibrium eccentricity ratio
    converged: bool
    thermal_mode: str = "L1" # "L1" (advective closure) or "L2" (conduction-dominated FVM)
    thermal_route: str = "L1"  # "L1" | "L2_brinkman" | "L1->L2_qmin" (ruling 2.6)
    F_app: float = float("nan")       # applied piston force [N] = k_F_bs * W_sq_ref
    F_pocket: float = float("nan")    # hydrostatic relief from the fed pocket [N]
    F_hydrodynamic: float = float("nan")  # share the film must carry = F_app - F_pocket
    # ---- BOX C additive fields.  Defaults leave every box-B row unchanged. --
    T_bulk: float = float("nan")      # area-averaged film temperature [degC]
    c_equilibrium: float = float("nan")   # clearance at the converged fixed point [m]
    n_outer: int = 0                  # outer clearance iterations (0 = closure off)
    clearance_closure: bool = False   # whether the thermal closure was applied
    delta_T_datum: str = "none"       # "peak" | "bulk" | "none" (fork F1)
    # ---- BOX C Phase 2, the load split.  Defaults leave box B unchanged. ----
    F_film: float = float("nan")      # film share of the load at equilibrium [N]
    F_asp: float = 0.0                # asperity share [N]; 0.0 with no contact model
    area_frac_contact: float = 0.0    # max A_c/A_n over the land
    contact_model: str = "none"       # every band end and fork arm, as one string
    # ---- BOX D boundary extrema (registered convention R-14.5, erratum E-9).
    # BOTH values are carried so the size of each correction is measurable per
    # row rather than asserted; the legacy interior-only fields above keep their
    # names and values, so box B and box C reproduce bit-for-bit.
    h_min_bnd: float = float("nan")    # incl. theta faces -- h_min biased HIGH
    De_sq_bnd: float = float("nan")    # incl. theta faces -- De_sq biased LOW
    T_max_bnd: float = float("nan")    # incl. theta faces -- T_max biased LOW


class BallSocketSolver:
    """Spherical-joint viscoelastic Reynolds solver with axial load balance.

    Shares the constitutive layer (``GiesekusFluid``) and the Picard core with
    the slipper solver; only the spherical geometry/operators and the
    eccentricity equilibrium search are specific to the ball-socket.
    """

    def __init__(
        self,
        geom: BallSocketGeometry,
        *,
        config: Optional[SolverConfig] = None,
        model: str = "viscoelastic",
        Ntheta: int = 48,
        Nphi: int = 48,
        alpha: float = 0.25,
        squeeze_phase: float = 1.0,
        spin_ratio: float = 1.0,
        eps_cap: float = 0.99,
        eps_lo: float = 1e-3,
        thermal: Optional[ThermalProperties] = None,
        clearance_closure: bool = False,
        delta_T_datum: str = "bulk",
        brentq_xtol: float = 1e-4,
        brentq_rtol: float = 1e-4,
        contact=None,
        omega_c: Optional[float] = None,
        tol_c: Optional[float] = None,
        max_outer: Optional[int] = None,
    ):
        self.geom = geom
        self.config = config or SolverConfig()
        self.model = model
        self.Ntheta, self.Nphi = Ntheta, Nphi
        self.alpha = alpha
        self.squeeze_phase = squeeze_phase
        self.spin_ratio = spin_ratio
        self.eps_cap = eps_cap
        #: FILM MODEL, handed to every Spherical this solver builds (box F
        #: sec 3.1, 3.4).  None is the registered smooth film; a dict turns on
        #: the Patir-Cheng averaged film (see Spherical.flow_factors), which box
        #: F uses only for the marginal-band re-solve.  An attribute rather than
        #: a module default so the realised model is read off the object.
        self.flow_factors = None
        #: D-6 (R-57.4, R-58.1): the sign of the squeeze-memory viscosity.
        #: "modulus" -- src's |M|, what every registered campaign solved; the default,
        #: so boxes A-F, E13, E5 ... reproduce bit-for-bit.  "signed" -- Paper 2's sign
        #: law (ball_socket.memory_sign), used for every artefact made after R-58.
        #: An attribute, so the realised choice is read off the object
        #: (realised_memory_sign).
        self.memory_sign = "modulus"
        self.eps_lo = eps_lo
        self.thermal = thermal or ThermalProperties()
        # ---- BOX C thermal-clearance closure (brief sec 2.2) ---------------
        # DEFAULT OFF.  With clearance_closure=False the solve path is the
        # box-B path exactly, which is what makes the sec 3.1 regression anchor
        # and gate BS-V10 meaningful.
        # ECCENTRICITY SOLVE TOLERANCE.  Defaults are the registered campaign
        # values (1e-4), so every campaign row is unaffected.  They are exposed
        # because a DIFFERENCE of h_min between two configurations has a
        # resolution floor of  rtol * eps / (1 - eps)  -- h = c(1 - eps), so
        # d h / h = -d eps / (1 - eps) -- and at rtol = 1e-4 that floor is
        # 1.2e-4 to 1.2e-3 over the eccentricities this campaign runs at.  A
        # term-off delta below that floor is BELOW RESOLUTION, not small
        # (ruling R-9.1).  Tightening lets the diagnostic answer the question
        # the default tolerance cannot.
        # BOX C, Phase 2.  ``contact`` is a mixed.ContactModel or None.
        # DEFAULT None reproduces the box-B pure-film load balance exactly,
        # which is what the sec 3.1 regression anchor and BS-V10 rest on.
        self.contact = contact
        self.brentq_xtol = float(brentq_xtol)
        self.brentq_rtol = float(brentq_rtol)
        self.clearance_closure = clearance_closure
        if delta_T_datum not in ("peak", "bulk"):
            raise ValueError(
                f"delta_T_datum must be 'peak' or 'bulk' (fork F1), "
                f"got {delta_T_datum!r}.  Neither arm is a default: box C "
                f"implements both and reports both (standing rule 18).")
        self.delta_T_datum = delta_T_datum
        # The numerics are REGISTERED, not hard-coded at the call site
        # (brief sec 2.2).  They are read from the signed box-C config unless
        # explicitly overridden, so there is one source for them.
        if omega_c is None or tol_c is None or max_outer is None:
            from .campaign_config_v5 import CLEARANCE_CLOSURE as _CC
            omega_c = _CC["omega_c"] if omega_c is None else omega_c
            tol_c = _CC["tol_c"] if tol_c is None else tol_c
            max_outer = _CC["max_outer"] if max_outer is None else max_outer
        self.omega_c, self.tol_c, self.max_outer = omega_c, tol_c, int(max_outer)

    # -- fluid / load ---------------------------------------------------------
    def _fluid(self, x: CampaignInput) -> GiesekusFluid:
        eta_s = x.eta_s_over_eta0 * x.eta_0
        eta_p = x.eta_0 - eta_s
        base = GiesekusFluid(eta_s=eta_s, eta_p=eta_p, lambda_=x.lam, alpha=self.alpha)
        from .memory_sign import wrap
        return wrap(base, self.memory_sign)

    def _applied_load(self, x: CampaignInput) -> float:
        """Applied piston force.

        BOX B (coupled, ``k_F_bs is None``): F_app = p_ch * A_p, the quasi-static
        piston force.  Because F_pocket = p_ch * A_pocket with the same p_ch, the
        hydrodynamic share is p_ch (A_p - A_pocket) > 0 for every sampled point.
        Inertial and centrifugal contributions to F_app are neglected
        (acknowledged omission, ruling 2.7).

        BOX A (frozen, retained): F_app = k_F_bs * W_sq_ref (D2).
        """
        if x.k_F_bs is None:
            from . import representative as _rep
            return x.p_ch * _rep.A_P_M2
        return x.k_F_bs * squeeze_capacity_scale(x.eta_0, x.omega, self.geom.R_b, x.c0)

    def _pocket_relief(self, x: CampaignInput) -> float:
        """Partial hydrostatic relief from the fed pocket held at p_ch:
        F_pocket = p_ch * pi R_b^2 sin^2(theta_min)  (Ch.9 patch T2).  The film
        carries only the balance F_app - F_pocket."""
        return x.p_ch * self.geom.pocket_relief_area

    # -- forward solve at a trial eccentricity --------------------------------
    def _forward(self, x: CampaignInput, eps: float, fluid: GiesekusFluid, c: float):
        """Solve the spherical film at axial eccentricity ``eps``.

        e = (0,0,-eps c) puts the minimum film at the loaded pole; the squeeze
        orbit gives e_dot = (0,0,-squeeze_phase*omega*eps*c) (hdot ~ c omega eps),
        and spin omega_b = spin_ratio*omega about the load axis.  ``gap_function``
        raises ``NonConvergence`` if the film closes (h <= 0).
        """
        e = (0.0, 0.0, -eps * c)
        e_dot = (0.0, 0.0, -self.squeeze_phase * x.omega * eps * c)
        omega_b = (0.0, 0.0, self.spin_ratio * x.omega)
        cond = SphericalConditions(p_ch=x.p_ch, p_case=x.p_case,
                                   e_vec=e, e_dot=e_dot, omega_b=omega_b)
        sph = Spherical(self.geom.R_b, c, self.geom.theta_max, cond,
                        self.Ntheta, self.Nphi, theta_min=self.geom.theta_min)
        sph.flow_factors = self.flow_factors
        res = solve_reynolds(sph, fluid, self.config, model=self.model)
        return res, sph

    def _load_at(self, x, eps, fluid, c) -> float:
        """Total load carried at eccentricity ``eps``: film plus asperity."""
        F_film, F_asp, _ = self._load_split(x, eps, fluid, c)
        return F_film + F_asp

    def _load_split(self, x, eps, fluid, c):
        """(F_film, F_asp, sph) at ``eps``.

        BOX C load balance (brief sec 3.5):

            residual(eps) = F_film(eps) + F_asp(eps) - F_target

        ``F_asp`` is the Greenwood-Tripp asperity pressure field projected onto
        the load line **through the same quadrature the film load uses** --
        ``Spherical.project_on_load_axis`` -- so the two shares cannot drift
        apart.  The brief forbids a second projection and there is not one.

        With ``self.contact is None`` this returns F_asp = 0.0 and the balance
        is the box-B pure-film balance, unchanged.
        """
        res, sph = self._forward(x, eps, fluid, c)
        F_film = sph.load_capacity(res.pressure)
        if self.contact is None:
            return F_film, 0.0, sph
        F_asp = sph.project_on_load_axis(self.contact.pressure(sph.gap()))
        return F_film, F_asp, sph


    # -- BOX C: thermal-clearance closure (brief sec 2.2, gate BS-V7) --------
    def solve_equilibrium(self, x: CampaignInput) -> Performance:
        """Equilibrium performance, with the thermal-clearance closure if it is
        enabled.

        CLOSURE OFF (default): one call to ``_equilibrium_at_clearance`` at
        ``c = x.c0``.  This is the box-B path, unchanged, and is what the
        sec 3.1 regression anchor and gate BS-V10 compare against.

        CLOSURE ON: the clearance depends on the film temperature and the film
        temperature depends on the clearance, so this is a FIXED POINT, not a
        substitution.  With  g(c) = c0 + Dalpha R_b (T_datum(c) - T_in),

            c_{k+1} = omega_c g(c_k) + (1 - omega_c) c_k

        under-relaxed, terminating on the FIXED-POINT RESIDUAL
        |g(c_k) - c_k| / c_k < tol_c.

        Note the deviation from the brief's pseudocode, which tests the
        relaxed STEP |c_{k+1} - c_k| / c_k.  Since the step is omega_c times
        the residual, testing the residual is stricter by 1/omega_c (a factor
        of 2 at the registered omega_c = 0.5), and it lets the returned
        Performance be the one solved AT the converged clearance rather than
        one iterate behind it.  Recorded in the Phase 1 hand-back.

        Failure to converge raises ``ClearanceNonConvergence`` -- a distinct
        typed outcome, never a silent fallback to ``c0`` (standing rule 14).
        ``NonConvergence`` from the inner solve propagates unchanged.
        """
        if not self.clearance_closure:
            try:
                perf = self._equilibrium_at_clearance(x, x.c0)
            except (NonConvergence, ContactLimit) as exc:
                exc.n_outer_at_failure = 0           # closure off
                raise
            perf.clearance_closure = False
            perf.n_outer = 0
            perf.delta_T_datum = "none"
            return perf

        R_b = self.geom.R_b
        c = float(x.c0)
        c_prev = float("nan")
        perf = None
        for k in range(1, self.max_outer + 1):
            try:
                perf = self._equilibrium_at_clearance(x, c)
            except (NonConvergence, ContactLimit) as exc:
                exc.n_outer_at_failure = int(k)      # 1 means raised at c0
                raise
            T_datum = perf.T_max if self.delta_T_datum == "peak" else perf.T_bulk
            delta_T = T_datum - x.T_in
            c_star = thermal_clearance(x.c0, R_b, delta_T,
                                       alpha_socket=getattr(self, "socket_alpha_th", None))
            if abs(c_star - c) / c < self.tol_c:
                perf.clearance_closure = True
                perf.n_outer = k
                perf.delta_T_datum = self.delta_T_datum
                perf.c_equilibrium = float(c)
                return perf
            c_prev = c
            c = self.omega_c * c_star + (1.0 - self.omega_c) * c

        raise ClearanceNonConvergence(
            f"thermal-clearance fixed point did not converge in "
            f"{self.max_outer} outer iterations (datum={self.delta_T_datum}): "
            f"c_last={c:.6e} m, c_prev={c_prev:.6e} m, "
            f"rel step={abs(c - c_prev) / c_prev:.3e}, tol_c={self.tol_c:.1e}",
            c_last=float(c), c_prev=float(c_prev),
            T_max_last=float(perf.T_max) if perf is not None else None,
            n_outer=int(self.max_outer), datum=self.delta_T_datum)

    # -- equilibrium ----------------------------------------------------------
    def _equilibrium_at_clearance(self, x: CampaignInput, c: float) -> Performance:
        """Find the axial eccentricity at load balance F_z(eps) = F_app, at a
        GIVEN nominal clearance ``c``, and return the performance.

        Raises ``NonConvergence`` on eccentricity blow-up (the film cannot
        carry F_app below the contact cap).

        ``c`` is a parameter rather than ``x.c0`` so that the box-C thermal
        closure can iterate on it.  Called once with ``c = x.c0`` when the
        closure is off, which is the box-B path exactly.
        """
        try:
            return self._equilibrium_at_clearance_inner(x, c)
        except (NonConvergence, ContactLimit) as exc:
            # THE CLEARANCE AT WHICH IT FAILED, typed (2026-09-16).  A failure
            # can occur on a LATER outer iteration: the film carries the load at
            # c0, the joint heats, the clearance grows, and the limit is hit at
            # c_k > c0.  Without this the stored film load cannot be reproduced
            # from its own row -- R-36.4's decomposition assumed c0 and was wrong
            # on 3 of 512 samples, one by 4 %.
            exc.c_at_failure = float(c)
            raise

    def _equilibrium_at_clearance_inner(self, x: CampaignInput, c: float) -> Performance:
        fluid = self._fluid(x)
        F_app = self._applied_load(x)
        F_pocket = self._pocket_relief(x)
        # The fed pocket carries part of the piston force hydrostatically; the
        # film carries only the balance (Ch.9 patch T2).  If the relief exceeds
        # the applied force the search returns eps_lo (near-concentric, thick
        # film) -- that is the natural bracket outcome, not a blow-up.
        F_target = F_app - F_pocket

        def residual(eps: float) -> float:
            return self._load_at(x, eps, fluid, c) - F_target

        # F_z(eps) is monotone increasing (thinner film + faster squeeze); bracket.
        try:
            r_hi = residual(self.eps_cap)
        except NonConvergence:
            # The film closed before eps_cap, so no load was ever evaluated:
            # F_film_max does not exist for this row and is left None rather
            # than filled with a placeholder.
            raise NonConvergence(
                f"eccentricity blow-up: film closes below eps_cap={self.eps_cap}",
                F_film_max=None, F_target=float(F_target), F_app=float(F_app),
                F_pocket=float(F_pocket), eps_cap=float(self.eps_cap),
                kind="blowup_film_closed", eps_at_failure=float(self.eps_cap))
        if r_hi < 0.0:
            if self.contact is not None:
                # BOX C.  The load exceeds film PLUS fully developed asperity
                # contact.  That is STRICTLY STRONGER than the box-B outcome
                # and is deliberately a different exception and a different
                # outcome string, so no downstream script can pool the two
                # (brief sec 3.5).
                F_film_lim, F_asp_lim, _ = self._load_split(
                    x, self.eps_cap, fluid, c)
                raise ContactLimit(
                    f"contact limit: film {F_film_lim:.3e} N + asperity "
                    f"{F_asp_lim:.3e} N < hydrodynamic share {F_target:.3e} N "
                    f"at eps_cap={self.eps_cap}",
                    F_film_at_limit=float(F_film_lim),
                    F_asp_at_limit=float(F_asp_lim),
                    F_target=float(F_target), F_app=float(F_app),
                    F_pocket=float(F_pocket), eps_cap=float(self.eps_cap),
                    contact_model=self.contact.label())
            # F_film_max is the film's load capacity and is carried as a typed
            # attribute, not only in this message (see NonConvergence).
            raise NonConvergence(
                f"eccentricity blow-up: max squeeze load {r_hi + F_target:.3e} N "
                f"< hydrodynamic share {F_target:.3e} N (F_app {F_app:.3e}, "
                f"pocket relief {F_pocket:.3e}) at eps_cap={self.eps_cap}",
                F_film_max=float(r_hi + F_target), F_target=float(F_target),
                F_app=float(F_app), F_pocket=float(F_pocket),
                eps_cap=float(self.eps_cap),
                kind="blowup_capacity", eps_at_failure=float(self.eps_cap))

        if residual(self.eps_lo) >= 0.0:
            eps_star = self.eps_lo   # load already carried at minimal eccentricity
        else:
            eps_star = brentq(residual, self.eps_lo, self.eps_cap,
                              xtol=self.brentq_xtol, rtol=self.brentq_rtol,
                              maxiter=200)

        res, sph = self._forward(x, eps_star, fluid, c)
        if not res.converged:
            # NOT a blow-up: the equilibrium was found.  Typed, so no consumer
            # has to read the message to tell the two apart.
            raise NonConvergence(
                f"Picard non-convergence at eps={eps_star:.4f}",
                F_target=float(F_target), F_app=float(F_app),
                F_pocket=float(F_pocket), eps_cap=float(self.eps_cap),
                kind="picard_at_root", eps_at_failure=float(eps_star))

        h = sph.gap()
        # The split at the converged eccentricity, through the one projection.
        F_film_eq = float(sph.load_capacity(res.pressure))
        if self.contact is not None:
            F_asp_eq = float(sph.project_on_load_axis(self.contact.pressure(h)))
            area_frac = float(np.max(self.contact.area_fraction(h)))
            contact_label = self.contact.label()
        else:
            F_asp_eq, area_frac, contact_label = 0.0, 0.0, "none"
        eta_bar = res.eta_bar if res.eta_bar is not None else np.full_like(h, x.eta_0)
        gdot = sph.gap_averaged_shear_rate(res.pressure, eta_bar)
        # De_sq uses the NOMINAL clearance c (Ch.9 sec:bs_kinematics: De_sq =
        # lambda|hdot|/c ~ lambda omega eps), NOT the local film h -- this keeps
        # g3 (validity) distinct from g1 (film thickness) and avoids a spurious
        # divergence near contact.
        De_sq = float(fluid.lambda_ * np.max(np.abs(sph.gap_rate())) / c)
        # BOX D: the same two quantities with the theta faces included, where
        # their true extrema lie.  Purely ADDITIVE -- neither line above moves.
        h_min_bnd_v = float(sph.h_min_boundary())
        De_sq_bnd_v = float(fluid.lambda_ * sph.gap_rate_max_boundary() / c)
        P_loss = float(sph.surface_integral(eta_bar * gdot ** 2 * h))

        # --- thermal: slipper L1 advective closure + Q_min/Brinkman routing to L2
        # (Ch.5 hierarchy carried to the sphere; wall datum T_in). ball_socket.thermal.
        th = self.thermal
        T_in_K = x.T_in + 273.15
        V_char = self.spin_ratio * x.omega * self.geom.R_b * np.sin(self.geom.theta_max)
        Br = bs_thermal.brinkman(x.eta_0, V_char, th)
        route_q, _ = bs_thermal.q_min_route_to_l2(P_loss, float(res.Q_leak), th)
        # Routing reason recorded separately (ruling 2.6) so Brinkman-routed and
        # Q_min-routed samples are distinguishable, matching the slipper's
        # three-way L2 / L1 / L1->L2 split.  Brinkman takes precedence when both
        # fire, as at the slipper.
        if Br > bs_thermal.BR_THRESHOLD:
            thermal_mode, thermal_route = "L2", "L2_brinkman"
        elif route_q:
            thermal_mode, thermal_route = "L2", "L1->L2_qmin"
        else:
            thermal_mode, thermal_route = "L1", "L1"
        if thermal_mode == "L2":
            T_field = bs_thermal.solve_l2_spherical(sph, eta_bar, gdot, th, T_in_K)
            T_max_C = float(T_field.max()) - 273.15
            # BOX D: boundary-aware peak.  Only the L2 route can carry this
            # defect -- on L1 there is no field to take an extremum of.
            T_max_bnd_C = float(sph.field_max_boundary(T_field)) - 273.15
            # Area-averaged film temperature, the "bulk" arm of fork F1.
            # Additive: it does not touch T_max_C and so cannot move a box-B row.
            _area = sph.surface_integral(np.ones_like(T_field))
            T_bulk_C = float(sph.surface_integral(T_field) / _area) - 273.15
        else:
            T_max_C = x.T_in + bs_thermal.l1_bulk_delta_T(P_loss, float(res.Q_leak), th)
            # L1 has no field, so the boundary-aware value IS the bulk value and
            # the correction is identically zero on L1-routed rows.
            T_max_bnd_C = T_max_C
            # On the L1 route the closure IS a bulk advective rise, so the two
            # arms of F1 coincide exactly.  The fork therefore only bites on
            # L2-routed rows, which is itself a reportable finding.
            T_bulk_C = T_max_C

        return Performance(
            F=float(res.F_total),
            h_min=float(h.min()),
            T_max=T_max_C,
            P_loss=P_loss,
            Q_leak=float(res.Q_leak),
            De_sq=De_sq,
            eps_ecc=float(eps_star),
            converged=True,
            thermal_mode=thermal_mode,
            thermal_route=thermal_route,
            F_app=float(F_app),
            F_pocket=float(F_pocket),
            F_hydrodynamic=float(F_target),
            T_bulk=T_bulk_C,
            c_equilibrium=float(c),
            h_min_bnd=h_min_bnd_v,
            De_sq_bnd=De_sq_bnd_v,
            T_max_bnd=T_max_bnd_C,
            F_film=F_film_eq,
            F_asp=F_asp_eq,
            area_frac_contact=area_frac,
            contact_model=contact_label,
        )
