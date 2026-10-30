"""
Shared Reynolds solver core (geometry-agnostic Picard iteration).
=================================================================

One Picard core drives every geometry. It speaks only the ``Geometry`` interface
and the constitutive evaluator (the ``fluid``); it contains no planar- or
spherical-specific arithmetic. ``PlanarAnnular`` and ``Spherical`` compose with
it identically -- the spherical solver is a *sibling* of the planar one sharing
this base, not a subclass that inherits planar assumptions.

For the planar geometry this reproduces ``src.solver.solve_viscoelastic`` (and
``solve_GNF`` / ``solve_newtonian``) bit-for-bit, because ``PlanarAnnular``
delegates each primitive to the very functions those solvers call. That
equivalence is the Phase-1 regression anchor (tests/test_planar_geometry_seam.py).
"""

from __future__ import annotations

from typing import Optional, Any
import warnings

import numpy as np

from src.solver import SolverConfig, solve_linear_system
from src.results import SolverResult


def solve_reynolds(
    geometry,
    fluid: Any,
    config: Optional[SolverConfig] = None,
    model: str = "viscoelastic",
    n1_mult: float = 1.0,
    hoop_mult: float = 1.0,
) -> SolverResult:
    """Solve the Reynolds equation on ``geometry`` with the shared Picard core.

    BOX C, Phase 6.  ``n1_mult`` and ``hoop_mult`` are MULTIPLIERS on the
    ns_scale ramp, not replacements for it, so the warm-up and ramp schedule is
    preserved exactly and only the term's weight changes.  Both default to 1.0,
    in which case the assembled system is bitwise what it was before this
    extension.  They exist because the E4 decomposition must separate N1 from
    hoop, and ``ns_scale`` scales the two together.

    This is the ball-socket Picard driver, not the shared slipper core, so the
    extension touches no slipper code (brief sec 9.1, 9.3).
    """
    if config is None:
        config = SolverConfig()
    if model == "newtonian":
        return _solve_newtonian(geometry, fluid, config)
    if model == "GNF":
        return _solve_gnf(geometry, fluid, config)
    if model == "viscoelastic":
        return _solve_viscoelastic(geometry, fluid, config, n1_mult=n1_mult, hoop_mult=hoop_mult)
    raise ValueError(f"Unknown model type: {model}")


def _solve_newtonian(geometry, fluid, config) -> SolverResult:
    system = geometry.assemble("newtonian", fluid, fluid.eta_0)
    p_1d = solve_linear_system(system.A, system.b, config.linear_solver)
    pressure = geometry.to_2d(p_1d)
    eta_bar = fluid.eta_0 * np.ones_like(pressure)

    result = SolverResult(
        pressure=pressure, converged=True, iterations=1,
        residual_history=[0.0], eta_bar=eta_bar, gdot_bar=None,
        solver_info={"type": "newtonian", "eta": fluid.eta_0},
    )
    result.F_total = geometry.load_capacity(pressure)
    result.F_GNF = result.F_total
    result.F_elastic = 0.0
    result.M_x, result.M_y = geometry.moments(pressure)
    result.Q_leak = geometry.leakage(pressure, eta_bar, fluid)
    return result


def _solve_gnf(geometry, fluid, config) -> SolverResult:
    h = geometry.gap()
    eta_bar = fluid.eta_0 * np.ones_like(h)
    pressure = geometry.initial_pressure()

    residual_history = []
    converged = False
    gdot_bar = None
    iteration = 0
    for iteration in range(config.max_iter):
        p_old = pressure.copy()
        system = geometry.assemble("GNF", fluid, eta_bar)
        p_new = geometry.to_2d(solve_linear_system(system.A, system.b, config.linear_solver))
        pressure = config.omega * p_new + (1 - config.omega) * p_old

        gdot_bar = geometry.gap_averaged_shear_rate(pressure, eta_bar)
        eta_bar_new = fluid.viscosity(gdot_bar)
        eta_bar = config.omega * eta_bar_new + (1 - config.omega) * eta_bar

        residual = np.linalg.norm(pressure - p_old) / (np.linalg.norm(p_old) + 1e-15)
        residual_history.append(residual)
        if config.verbose:
            print(f"  Iter {iteration+1:3d}: residual = {residual:.3e}")
        if residual < config.tol:
            converged = True
            break

    result = SolverResult(
        pressure=pressure, converged=converged, iterations=iteration + 1,
        residual_history=residual_history, eta_bar=eta_bar, gdot_bar=gdot_bar,
        solver_info={"type": "GNF"},
    )
    result.F_total = geometry.load_capacity(pressure)
    result.F_GNF = result.F_total
    result.F_elastic = 0.0
    result.M_x, result.M_y = geometry.moments(pressure)
    result.Q_leak = geometry.leakage(pressure, eta_bar, fluid)
    if not converged:
        warnings.warn(f"GNF solver did not converge after {config.max_iter} iterations")
    return result


def _solve_viscoelastic(geometry, fluid, config,
                        n1_mult: float = 1.0,
                        hoop_mult: float = 1.0) -> SolverResult:
    if fluid.is_newtonian:
        return _solve_newtonian(geometry, fluid, config)

    h = geometry.gap()
    eta_bar = fluid.eta_0 * np.ones_like(h)
    gdot_bar = geometry.surface_shear_speed()
    pressure = geometry.initial_pressure()

    residual_history = []
    converged = False
    iteration = 0
    for iteration in range(config.max_iter):
        p_old = pressure.copy()

        if not (config.include_normal_stress or config.include_alpha_coupling):
            ns_scale = 0.0
        elif iteration < config.ns_warmup:
            ns_scale = 0.0
        elif iteration < config.ns_warmup + config.ns_ramp:
            ns_scale = (iteration - config.ns_warmup + 1) / config.ns_ramp
        else:
            ns_scale = 1.0

        memory_scale = (
            config.memory_R_fn(fluid.lambda_ * gdot_bar)
            if config.memory_R_fn is not None else None
        )
        system = geometry.assemble(
            "viscoelastic", fluid, eta_bar, gdot_bar,
            pressure=pressure if iteration > 0 else None,
            include_memory=config.include_memory,
            include_normal_stress=config.include_normal_stress,
            include_alpha_coupling=config.include_alpha_coupling,
            memory_scale=memory_scale,
            ns_scale=ns_scale,
            n1_scale=ns_scale * n1_mult,
            hoop_scale=ns_scale * hoop_mult,
        )
        p_new = geometry.to_2d(solve_linear_system(system.A, system.b, config.linear_solver))
        if not np.all(np.isfinite(p_new)):
            warnings.warn(
                f"Viscoelastic solver produced NaN/Inf at iteration {iteration+1}"
            )
            break

        pressure = config.omega * p_new + (1 - config.omega) * p_old
        gdot_bar = geometry.gap_averaged_shear_rate(pressure, eta_bar)
        eta_bar_new = fluid.viscosity(gdot_bar)
        eta_bar = config.omega * eta_bar_new + (1 - config.omega) * eta_bar

        residual = np.linalg.norm(pressure - p_old) / (np.linalg.norm(p_old) + 1e-15)
        residual_history.append(residual)
        if config.verbose:
            ns_str = f", ns_scale = {ns_scale:.2f}" if config.include_normal_stress else ""
            print(f"  Iter {iteration+1:3d}: residual = {residual:.3e}, "
                  f"η_min/η_0 = {eta_bar.min()/fluid.eta_0:.2f}{ns_str}")
        if residual < config.tol:
            converged = True
            break

    # Out-of-envelope validity guard (geometry-specific; planar -> Table 7).
    warn = getattr(geometry, "warn_validity", None)
    if callable(warn):
        warn(pressure, eta_bar, fluid)

    flux_decomp = geometry.flux_decomposition(pressure, eta_bar, fluid) if hasattr(geometry, "flux_decomposition") else None

    config_gnf = SolverConfig(
        max_iter=config.max_iter, tol=config.tol, omega=config.omega, verbose=False
    )
    result_gnf = _solve_gnf(geometry, fluid, config_gnf)

    result = SolverResult(
        pressure=pressure, converged=converged, iterations=iteration + 1,
        residual_history=residual_history, fluxes=flux_decomp,
        eta_bar=eta_bar, gdot_bar=gdot_bar,
        solver_info={
            "type": "viscoelastic",
            "include_memory": config.include_memory,
            "include_normal_stress": config.include_normal_stress,
        },
    )
    result.F_total = geometry.load_capacity(pressure)
    result.F_GNF = result_gnf.F_total
    result.F_elastic = result.F_total - result.F_GNF
    result.M_x, result.M_y = geometry.moments(pressure)
    result.Q_leak = geometry.leakage(
        pressure, eta_bar, fluid, flux_decomposition=flux_decomp
    )
    if not converged:
        warnings.warn(
            f"Viscoelastic solver did not converge after {config.max_iter} iterations"
        )
    return result
