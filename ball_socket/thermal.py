"""Ball-socket thermal wiring -- Ch.5 hierarchy carried to the sphere.

Nigel ruling (2026-07-25): carry the slipper L1 closure UNCHANGED --
    T_bulk = T_in + P_diss / (rho c_p Q_out),    Q_out = boundary outflow (leakage)
-- with the Q_min routing guard (Q_out <= Q_min => escalate to L2),
    Q_min = P_diss / (rho c_p * DELTA_T_SAFETY_K),
which subsumes the delta_T > 10 K catch and inward/near-zero through-flow.  The
Brinkman escalation REUSES the slipper definition Br = eta_0 V_char^2 /
(k_f * BR_DT_REF_K) with ONLY V_char swapped to the spin-dominated Couette speed
omega_b R_b sin(theta_max).  L2 is the gap-averaged FVM energy solve on the
spherical land with cross-film conduction to the bounding solids held at the wall
datum T_in.  Wall datum = T_in throughout (slipper convention).

A conforming, mostly-closed joint has small articulation-pumped outflow, so the
guard fires often and most samples route to L2 (conduction-dominated); that is the
correct behaviour and the basis of pre-registered expectation E6.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from src.thermal import ThermalProperties

DELTA_T_SAFETY_K = 10.0     # L1 validity: implied rise above which L1 is untrusted
BR_THRESHOLD = 0.1          # Brinkman L1/L2 cutoff (slipper convention)
BR_DT_REF_K = 20.0          # Brinkman reference delta_T (slipper convention)


def l1_bulk_delta_T(P_diss: float, Q_out: float, thermal: ThermalProperties) -> float:
    """ΔT = P_diss / (rho c_p Q_out) -- the slipper advective closure."""
    Q_eff = max(abs(Q_out), 1e-12)
    return P_diss / (thermal.rho * thermal.c_p * Q_eff)


def q_min_route_to_l2(P_diss: float, Q_out: float, thermal: ThermalProperties):
    """Q_min guard: route to L2 when Q_out <= Q_min = P_diss/(rho c_p dT_safety),
    i.e. the implied L1 rise would exceed DELTA_T_SAFETY_K, or the through-flow is
    non-positive.  Returns (route_to_l2: bool, Q_min: float)."""
    rho_cp = thermal.rho * thermal.c_p
    Q_min = P_diss / (rho_cp * DELTA_T_SAFETY_K) if P_diss > 0 else 0.0
    return bool(Q_out <= Q_min), float(Q_min)


def brinkman(eta_0: float, V_char: float, thermal: ThermalProperties) -> float:
    """Br = eta_0 V_char^2 / (k_f * BR_DT_REF_K); slipper definition, V_char swapped."""
    return float(eta_0 * V_char ** 2 / (thermal.k_f * BR_DT_REF_K))


def solve_l2_spherical(sph, eta_bar: np.ndarray, gdot_bar: np.ndarray,
                       thermal: ThermalProperties, T_wall_K: float,
                       advection_frac: float = 0.5) -> np.ndarray:
    """Gap-averaged FVM energy solve on the spherical land:

        rho c_p (ubar . grad_s T) = Phi_v + k_eff (T_wall - T),
        Phi_v = eta gdot^2 [W/m^3],   k_eff = 12 k_f / h^2 [W/(m^3 K)]  (Nu = 12)

    Upwind convection; Dirichlet T = T_wall at the socket mouth (theta_max);
    zero-gradient inflow at the pole; periodic in phi.  ubar = advection_frac *
    ball surface velocity (Couette mean; 0.5 for a stationary socket).  The
    cross-film conduction sink to the bounding solids at T_wall = T_in makes the
    convection-reaction system diagonally dominant.  Returns T [K], shape (Nt,Np).
    """
    mesh = sph.mesh
    Nt, Np = mesh.Ntheta, mesh.Nphi
    dth, dph, R_b = mesh.dtheta, mesh.dphi, mesh.R_b
    th = mesh.theta
    rho_cp = thermal.rho * thermal.c_p

    h = sph.gap()
    Phi_v = eta_bar * gdot_bar ** 2
    k_eff = 12.0 * thermal.k_f / h ** 2
    u_th_w, u_ph_w = sph.surface_velocity()
    u_th = advection_frac * u_th_w
    u_ph = advection_frac * u_ph_w

    N = mesh.N
    rows, cols, vals = [], [], []
    b = np.zeros(N)

    def idx(i, j):
        return i * Np + (j % Np)

    for i in range(Nt):
        sin_i = np.sin(th[i])
        for j in range(Np):
            k = idx(i, j)
            if i == Nt - 1:                 # socket mouth: Dirichlet T = T_wall
                rows.append(k); cols.append(k); vals.append(1.0)
                b[k] = T_wall_K
                continue

            diag = k_eff[i, j]
            b[k] = Phi_v[i, j] + k_eff[i, j] * T_wall_K

            # theta convection (upwind); u_th > 0 flows toward the mouth
            c_th = rho_cp * u_th[i, j] / (R_b * dth)
            if c_th >= 0:
                diag += c_th
                if i > 0:
                    rows.append(k); cols.append(idx(i - 1, j)); vals.append(-c_th)
                else:
                    diag -= c_th             # pole zero-gradient inflow (T_{-1}=T_0)
            else:
                diag += -c_th
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(c_th)

            # phi convection (upwind, periodic); u_ph/(R sin) finite at the pole
            c_ph = rho_cp * u_ph[i, j] / (R_b * max(sin_i, 1e-9) * dph)
            if c_ph >= 0:
                diag += c_ph
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(-c_ph)
            else:
                diag += -c_ph
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(c_ph)

            rows.append(k); cols.append(k); vals.append(diag)

    A = csr_matrix((vals, (rows, cols)), shape=(N, N))
    return spsolve(A, b).reshape(Nt, Np)
