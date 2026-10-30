"""
Spherical (ball-and-socket) geometry.
=====================================

Implements the spherical Reynolds geometry of Ch.9 (spherical_RE) through the
``Geometry`` interface, as a *sibling* of ``PlanarAnnular`` sharing the
``src.reynolds`` Picard core. No planar assumptions are inherited; only the
metric and the gap/wall kinematics change.

Surface coordinates
    theta in (0, theta_max]   polar angle from the load/pole axis (+z)
    phi   in [0, 2 pi)        azimuth (periodic)
    outward unit normal  n_hat = (sin th cos ph, sin th sin ph, cos th)

Conformal gap
    h(theta, phi) = c + e . n_hat,   eps_ecc = |e| / c
  with eccentricity vector ``e`` [m]. SIGN CONVENTION (reported): we take ``e``
  ANTIPARALLEL to the ball-centre displacement, so for an axially loaded ball
  (displaced +z toward the pole) ``e = (0, 0, -|e|)`` gives h = c - |e| cos th,
  whose MINIMUM sits at the pole theta = 0 -- the loaded side. (Equivalently:
  the minimum film is where n_hat is most anti-aligned with e.)

Metric (h_theta = R_b, h_phi = R_b sin th); the surface Laplace-Beltrami /
surface-divergence diffusion operator is
    L[p] = (1/(R_b^2 sin th)) d/dth ( sin th  D  dp/dth )
         + (1/(R_b^2 sin^2 th)) d/dph ( D dp/dph )
discretised in conservative flux form on a cell-centred (theta, phi) grid:
  * the pole face at theta = 0 carries sin(0) = 0, so its flux vanishes
    identically -- this enforces the regularity BC  dp/dtheta|_0 = 0  (an
    excised-cap / zero-inner-flux treatment, equivalent to the L'Hopital
    (sin th)^-1 d/dth(sin th .) -> 2 d/dth limit at the first interior node);
  * the socket mouth theta = theta_max uses a ghost-node Dirichlet stencil
    p(theta_max, phi) = p_ch (second order), exactly as the planar outer edge.

Squeeze source  h_dot = e_dot . n_hat  from the (oscillatory) eccentricity rate
``e_dot`` [m/s]; the consistent squeeze scaling is h_dot ~ c omega eps (NOT the
entrainment speed). Wall (entrainment) velocity from ball spin omega_b and
articulation omega_art: u = (omega_b + omega_art) x (R_b n_hat), tangential part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Any

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

# Geometry ABC dropped: on the thesis-validation core the slipper geometry is
# concrete (src/geometry.py), so Spherical is a standalone sibling class.
MEMORY_KERNEL_DENOM = 12.0  # single-source 1/12 squeeze-memory coefficient (Ch.9 Lmem_sph)


# =============================================================================
# Spherical socket mesh + metric
# =============================================================================

@dataclass
class SphericalMesh:
    """Cell-centred (theta, phi) grid on the spherical socket land."""

    R_b: float                # ball radius [m]
    theta_max: float          # socket half-angle (mouth) [rad]
    Ntheta: int
    Nphi: int
    theta_min: float = 0.0    # feed-pocket edge [rad]; 0 => full cap (pole mode)

    theta: np.ndarray = field(init=False, repr=False)
    phi: np.ndarray = field(init=False, repr=False)
    TH: np.ndarray = field(init=False, repr=False)
    PH: np.ndarray = field(init=False, repr=False)
    dtheta: float = field(init=False)
    dphi: float = field(init=False)
    N: int = field(init=False)

    def __post_init__(self):
        if self.Ntheta < 4:
            raise ValueError(f"Ntheta must be >= 4, got {self.Ntheta}")
        if self.Nphi < 4:
            raise ValueError(f"Nphi must be >= 4, got {self.Nphi}")
        if not (0 < self.theta_max < np.pi):
            raise ValueError(f"theta_max must be in (0, pi), got {self.theta_max}")
        if not (0.0 <= self.theta_min < self.theta_max):
            raise ValueError(f"theta_min must be in [0, theta_max), got {self.theta_min}")

        self.dtheta = (self.theta_max - self.theta_min) / self.Ntheta
        self.dphi = 2.0 * np.pi / self.Nphi
        # cell centres: theta_0 = theta_min + dtheta/2, theta_{Ntheta-1} =
        # theta_max - dtheta/2.  ANNULAR (theta_min > 0): both theta faces are
        # Dirichlet ghosts (pocket edge, socket mouth).  POLE/CAP mode
        # (theta_min = 0): the inner face carries sin(0) = 0 and the zero-flux
        # regularity closure -- exercised only in the full-cap verification
        # benchmarks, not in production runs (Ch.9 patch T2).
        self.theta = self.theta_min + (np.arange(self.Ntheta) + 0.5) * self.dtheta
        self.phi = (np.arange(self.Nphi) + 0.5) * self.dphi
        self.TH, self.PH = np.meshgrid(self.theta, self.phi, indexing="ij")
        self.N = self.Ntheta * self.Nphi

    @property
    def is_annular(self) -> bool:
        """True when the pole is excluded (fed-pocket edge at theta_min > 0)."""
        return self.theta_min > 1e-12

    def _index(self, i: int, j: int) -> int:
        return i * self.Nphi + (j % self.Nphi)

    def to_1d(self, f2d: np.ndarray) -> np.ndarray:
        return f2d.ravel()

    def to_2d(self, f1d: np.ndarray) -> np.ndarray:
        return f1d.reshape((self.Ntheta, self.Nphi))

    def normal(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s, c = np.sin(self.TH), np.cos(self.TH)
        return s * np.cos(self.PH), s * np.sin(self.PH), c

    def normal_at_theta(self, th0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``normal()`` evaluated on the phi line at a single theta.

        Deliberately the same expression as ``normal()`` with ``TH`` replaced by
        a scalar, so a boundary value and an interior value differ ONLY in where
        they are sampled -- which is precisely what the boundary-extremum
        convention asserts.  Used with ``th0 = theta_min`` / ``theta_max``,
        the two domain faces, which sit dtheta/2 outside the end cell centres.
        """
        s, c = np.sin(float(th0)), np.cos(float(th0))
        one = np.ones_like(self.phi)
        return s * np.cos(self.phi), s * np.sin(self.phi), c * one

    def theta_faces(self) -> Tuple[float, float]:
        """The two theta-domain faces (NOT cell centres)."""
        return float(self.theta_min), float(self.theta_max)

    def field_gradient(self, f2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Surface gradient (d/dtheta, (1/sin th) d/dphi) ... returned as the
        raw partials (df/dtheta, df/dphi); metric factors applied by callers."""
        Nt, Np = self.Ntheta, self.Nphi
        df_dth = np.empty_like(f2d)
        # central in interior, one-sided at the two theta ends
        df_dth[1:-1, :] = (f2d[2:, :] - f2d[:-2, :]) / (2 * self.dtheta)
        df_dth[0, :] = (f2d[1, :] - f2d[0, :]) / self.dtheta
        df_dth[-1, :] = (f2d[-1, :] - f2d[-2, :]) / self.dtheta
        jp = (np.arange(Np) + 1) % Np
        jm = (np.arange(Np) - 1) % Np
        df_dph = (f2d[:, jp] - f2d[:, jm]) / (2 * self.dphi)
        return df_dth, df_dph

    def integrate(self, f2d: np.ndarray) -> float:
        """Surface integral over the land: int f dA, dA = R_b^2 sin th dth dph."""
        dA = self.R_b ** 2 * np.sin(self.TH) * self.dtheta * self.dphi
        return float(np.sum(f2d * dA))


def build_spherical_diffusion_operator(
    mesh: SphericalMesh, D_coeff
) -> csr_matrix:
    """
    Conservative discretisation of
        L[p] = (1/(R^2 sin th)) d/dth(sin th D dp/dth)
             + (1/(R^2 sin^2 th)) d/dph(D dp/dph)
    with pole zero-flux regularity (theta=0 face) and a ghost-node Dirichlet
    stencil at the socket mouth (theta=theta_max). The Dirichlet value enters
    the RHS via ``spherical_ghost_bc_rhs``.

    ANISOTROPIC MOBILITY (box C, brief sec 3.2(a)).  ``D_coeff`` is either

      * a single scalar field  -- current behaviour, UNCHANGED and bitwise
        identical to before this extension; or
      * a pair ``(D_theta, D_phi)`` -- separate mobilities on the theta and phi
        faces, with the SAME conservative harmonic face-averaging in each
        direction.

    The pair form exists because Patir-Cheng pressure flow factors are
    direction dependent on the metric: phi_theta multiplies the theta mobility
    and phi_phi the phi mobility.  Per PC1978 Eq. 27,
    ``phi_y(H, gamma) = phi_x(H, 1/gamma)``, so at the registered
    ``PEKLENIK_GAMMA = 1`` the two fields coincide identically and the scalar
    path is exact -- which is what box C actually runs.  The pair path is built
    so the gamma sweep is available to the student work package without a
    rewrite.
    """
    if isinstance(D_coeff, (tuple, list)):
        D_th, D_ph = (np.asarray(D_coeff[0], dtype=float),
                      np.asarray(D_coeff[1], dtype=float))
        if D_th.shape != D_ph.shape:
            raise ValueError(
                f"anisotropic mobility needs matching shapes, got "
                f"{D_th.shape} and {D_ph.shape}")
    else:
        D_th = D_ph = np.asarray(D_coeff, dtype=float)
    Nt, Np = mesh.Ntheta, mesh.Nphi
    dth, dph = mesh.dtheta, mesh.dphi
    R2 = mesh.R_b ** 2
    th = mesh.theta
    L = lil_matrix((mesh.N, mesh.N), dtype=float)

    def harm(a, b):
        return 2.0 * a * b / (a + b + 1e-300)

    for i in range(Nt):
        sin_i = np.sin(th[i])
        sin_ip = np.sin(th[i] + dth / 2.0)   # face i+1/2
        sin_im = np.sin(th[i] - dth / 2.0)   # face i-1/2 (=sin(0)=0 at i=0)
        for j in range(Np):
            k = mesh._index(i, j)
            D_ij = D_th[i, j]                 # theta-face mobility

            # ---- theta diffusion (conservative flux form) ----
            # inner face i-1/2
            if i == 0:
                if mesh.is_annular:
                    # ANNULAR: fed-pocket ghost p[-1] = 2 p_ch - p[0]
                    # contribution coeff_im*(p_ghost - p_i) = coeff_im*(-2 p_i + 2 p_ch)
                    coeff_im = sin_im * D_ij / (R2 * sin_i * dth ** 2)
                    L[k, k] += -2.0 * coeff_im
                    # the +2 coeff_im p_ch term is moved to the RHS
                else:
                    # POLE/CAP: sin(0)=0 -> zero flux (regularity dp/dth|_0 = 0)
                    coeff_im = 0.0
            else:
                D_im_half = harm(D_ij, D_th[i - 1, j])
                coeff_im = sin_im * D_im_half / (R2 * sin_i * dth ** 2)
                L[k, mesh._index(i - 1, j)] += coeff_im

            # outer face i+1/2
            if i == Nt - 1:
                # socket mouth ghost: p[Nt] = 2 p_ch - p[Nt-1]; D at face = D_ij
                coeff_ip = sin_ip * D_ij / (R2 * sin_i * dth ** 2)
                # contribution: coeff_ip*(p_ghost - p_i) = coeff_ip*(-2 p_i + 2 p_ch)
                L[k, k] += -2.0 * coeff_ip
                # the +2 coeff_ip p_ch term is moved to the RHS
            else:
                D_ip_half = harm(D_ij, D_th[i + 1, j])
                coeff_ip = sin_ip * D_ip_half / (R2 * sin_i * dth ** 2)
                L[k, mesh._index(i + 1, j)] += coeff_ip

            if i != Nt - 1:
                L[k, k] += -coeff_ip
            if i != 0:
                L[k, k] += -coeff_im

            # ---- phi diffusion (periodic) ----
            jp, jm = (j + 1) % Np, (j - 1) % Np
            D_ij_ph = D_ph[i, j]              # phi-face mobility
            D_jp = harm(D_ij_ph, D_ph[i, jp])
            D_jm = harm(D_ij_ph, D_ph[i, jm])
            fac = 1.0 / (R2 * sin_i ** 2 * dph ** 2)
            L[k, mesh._index(i, jp)] += D_jp * fac
            L[k, mesh._index(i, jm)] += D_jm * fac
            L[k, k] += -(D_jp + D_jm) * fac

    return L.tocsr()


def surface_div_grad(mesh: SphericalMesh, C: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Conservative, 2nd-order discretisation of  div_s( C grad_s g )  on the grid:
        (1/(R^2 sin th)) d/dth( sin th C dg/dth )
      + (1/(R^2 sin^2 th)) d/dph( C dg/dph )
    with NATURAL (zero-flux) closure at the pole (sin 0 = 0) and the socket mouth
    -- appropriate for an RHS SOURCE field g that is not itself Dirichlet-bound.

    Used for the N1 / hoop sources. The naive composition grad_s -> (.) -> div_s
    of two central first-derivatives is inconsistent on this metric (it diverges
    under refinement); this face-flux form converges at 2nd order (MMS-verified,
    tests/test_spherical.py::test_n1_hoop_operator_second_order).
    """
    Nt, Np = mesh.Ntheta, mesh.Nphi
    dth, dph = mesh.dtheta, mesh.dphi
    R2 = mesh.R_b ** 2
    th = mesh.theta
    sin_i = np.sin(th)[:, None]                       # (Nt,1)

    # theta faces i in 0..Nt (face i at theta = i*dth); pole face 0 and mouth
    # face Nt carry zero flux.
    th_face = mesh.theta_min + np.arange(Nt + 1) * dth
    sin_face = np.sin(th_face)[:, None]               # (Nt+1,1)
    F = np.zeros((Nt + 1, Np))
    Cface = 0.5 * (C[1:, :] + C[:-1, :])              # (Nt-1,Np) interior faces 1..Nt-1
    dg = (g[1:, :] - g[:-1, :]) / dth
    F[1:Nt, :] = sin_face[1:Nt, :] * Cface * dg       # F[0]=F[Nt]=0 (zero flux)
    div_th = (F[1:, :] - F[:-1, :]) / dth             # (Nt,Np)

    # phi faces (periodic)
    jp = (np.arange(Np) + 1) % Np
    jm = (np.arange(Np) - 1) % Np
    Cface_ph = 0.5 * (C + C[:, jp])
    Fph = Cface_ph * (g[:, jp] - g) / dph             # flux at face j+1/2
    div_ph = (Fph - Fph[:, jm]) / dph

    return div_th / (R2 * sin_i) + div_ph / (R2 * sin_i ** 2)


def spherical_ghost_bc_rhs(
    mesh: SphericalMesh, D_coeff: np.ndarray, p_case: float,
    p_pocket: Optional[float] = None,
) -> np.ndarray:
    """RHS correction for the Dirichlet ghost stencils.

    ``p_case`` is the socket-mouth value at theta_max (always applied).
    ``p_pocket`` is the fed-pocket-edge value at theta_min, applied only on an
    ANNULAR mesh; ignored (pole zero-flux closure) on a full cap.
    """
    Nt, Np = mesh.Ntheta, mesh.Nphi
    dth = mesh.dtheta
    R2 = mesh.R_b ** 2
    th = mesh.theta
    b = np.zeros(mesh.N)

    # outer (socket mouth) ghost
    i = Nt - 1
    sin_i = np.sin(th[i])
    sin_ip = np.sin(th[i] + dth / 2.0)
    for j in range(Np):
        k = mesh._index(i, j)
        coeff_ip = sin_ip * D_coeff[i, j] / (R2 * sin_i * dth ** 2)
        # L p = b - 2 coeff_ip p_case  -> move to RHS as -2 coeff_ip p_case
        b[k] -= 2.0 * coeff_ip * p_case

    # inner (fed pocket edge) ghost -- annular only
    if mesh.is_annular and p_pocket is not None:
        i = 0
        sin_i = np.sin(th[i])
        sin_im = np.sin(th[i] - dth / 2.0)
        for j in range(Np):
            k = mesh._index(i, j)
            coeff_im = sin_im * D_coeff[i, j] / (R2 * sin_i * dth ** 2)
            b[k] -= 2.0 * coeff_im * p_pocket
    return b


# =============================================================================
# Spherical geometry (Geometry interface)
# =============================================================================

@dataclass
class SphericalConditions:
    """Operating point for the ball-and-socket bearing.

    BC convention (annular fed film, Ch.9 patch T2):
      * ``p_ch``   -> Dirichlet at theta_min, the fed-pocket edge (chamber
        pressure).  Applied on an ANNULAR mesh only.
      * ``p_case`` -> Dirichlet at theta_max, the socket mouth (case pressure).
        Applied always; this is the value the full-cap benchmarks use at the
        mouth (they leave it at the 0.0 default).
    """
    p_ch: float = 0.0                       # fed-pocket (chamber) pressure [Pa]
    p_case: float = 0.0                     # socket-mouth (case) pressure [Pa]
    e_vec: Tuple[float, float, float] = (0.0, 0.0, 0.0)      # eccentricity [m]
    e_dot: Tuple[float, float, float] = (0.0, 0.0, 0.0)      # ecc. rate [m/s]
    omega_b: Tuple[float, float, float] = (0.0, 0.0, 0.0)    # ball spin [rad/s]
    omega_art: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # articulation [rad/s]


class Spherical:
    """Ball-and-socket spherical-bearing geometry."""

    def __init__(
        self,
        R_b: float,
        c: float,
        theta_max: float,
        conditions: SphericalConditions,
        Ntheta: int,
        Nphi: int,
        theta_min: float = 0.0,
    ):
        self.R_b = float(R_b)
        self.c = float(c)
        self.theta_max = float(theta_max)
        self.theta_min = float(theta_min)
        self.conditions = conditions
        self.mesh = SphericalMesh(R_b, theta_max, Ntheta, Nphi, theta_min=theta_min)
        self.e_vec = np.asarray(conditions.e_vec, dtype=float)
        self.e_dot = np.asarray(conditions.e_dot, dtype=float)
        #: PATIR-CHENG AVERAGED FILM, OPT-IN (ruling R-37.2 measurement).
        #: None -> smooth Reynolds, bitwise what every campaign so far solved.
        #: Otherwise a dict {"sigma": composite surface Rq [m],
        #:                   "below_floor": "extrapolate" | "freeze"}.
        #: Box C registered the averaged equation (PC1978 Eq. 10) and no code
        #: path applied it; this attribute is the smallest change that lets the
        #: two be compared on the same mesh, the same fluid, the same Picard
        #: loop.  It is NOT yet the campaign's film -- that is an author
        #: decision R-37.2 takes only after this measurement.
        self.flow_factors = None

    # -- diagnostics ----------------------------------------------------------
    @property
    def eps_ecc(self) -> float:
        return float(np.linalg.norm(self.e_vec) / self.c)

    # -- grid / reshaping -----------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.mesh.Ntheta, self.mesh.Nphi)

    def to_1d(self, f2d):
        return self.mesh.to_1d(f2d)

    def to_2d(self, f1d):
        return self.mesh.to_2d(f1d)

    # -- geometry / metric ----------------------------------------------------
    def gap(self) -> np.ndarray:
        nx, ny, nz = self.mesh.normal()
        ex, ey, ez = self.e_vec
        return self.c + ex * nx + ey * ny + ez * nz

    def gap_rate(self) -> np.ndarray:
        """h_dot = e_dot . n_hat  (squeeze source from oscillatory eccentricity)."""
        nx, ny, nz = self.mesh.normal()
        ex, ey, ez = self.e_dot
        return ex * nx + ey * ny + ez * nz

    def surface_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tangential wall (ball) velocity (u_theta, u_phi) [m/s] from spin +
        articulation: u = omega x (R_b n_hat)."""
        TH, PH = self.mesh.TH, self.mesh.PH
        nx, ny, nz = self.mesh.normal()
        omega = np.asarray(self.conditions.omega_b, float) + np.asarray(
            self.conditions.omega_art, float
        )
        # u = omega x (R_b n_hat)
        ux = self.R_b * (omega[1] * nz - omega[2] * ny)
        uy = self.R_b * (omega[2] * nx - omega[0] * nz)
        uz = self.R_b * (omega[0] * ny - omega[1] * nx)
        # unit tangent vectors
        eth = (np.cos(TH) * np.cos(PH), np.cos(TH) * np.sin(PH), -np.sin(TH))
        eph = (-np.sin(PH), np.cos(PH), np.zeros_like(PH))
        u_th = ux * eth[0] + uy * eth[1] + uz * eth[2]
        u_ph = ux * eph[0] + uy * eph[1] + uz * eph[2]
        return u_th, u_ph

    def surface_shear_speed(self) -> np.ndarray:
        u_th, u_ph = self.surface_velocity()
        return np.sqrt(u_th ** 2 + u_ph ** 2) / self.gap()

    def field_gradient(self, f2d):
        return self.mesh.field_gradient(f2d)

    # -- Patir-Cheng averaged film (opt-in) ------------------------------------
    def _flow_factor_fields(self):
        """(phi_p, dhT_dh, below_floor_mask) on the gap, or None when smooth.

        phi_p   -- pressure flow factor, gamma = 1 (phi_x = phi_y, Eq. 27), on
                   the TOTAL mobility (fork F2 arm A, as box D closed it).
        dhT_dh  -- d h_T_bar / d h = Phi(H), the standard normal CDF, exactly:
                   differentiating Eq. 2 under the integral leaves the density
                   integrated from -H to infinity.  It multiplies the squeeze
                   source, since d h_T_bar/dt = Phi(H) dh/dt.
        Below the fit floor (H <= 0.5 at gamma = 1) the treatment is NAMED by
        the caller -- extrapolate the fit, or freeze phi at its floor value --
        and the mask is returned so the fraction can be reported.
        """
        ff = self.flow_factors
        if ff is None:
            return None
        from . import mixed as _mx
        sigma = float(ff["sigma"])
        mode = ff.get("below_floor")
        if mode not in ("extrapolate", "freeze"):
            raise ValueError("flow_factors['below_floor'] must be named: "
                             "'extrapolate' or 'freeze'")
        H = self.gap() / sigma
        floor = _mx.flow_factor_valid_above(1.0)
        below = H <= floor
        phi = np.asarray(_mx.phi_x(H, 1.0), dtype=float)
        if mode == "freeze":
            phi = np.where(below, float(_mx.phi_x(floor, 1.0)), phi)
        dhT = _mx.dh_T_bar_dh(self.gap(), sigma)
        return phi, dhT, below

    # -- source ---------------------------------------------------------------
    def source(self) -> np.ndarray:
        """Geometric source S = h_dot + wedge, as a 1D vector.

        Squeeze: S_sq = h_dot = e_dot . n_hat.
        Wedge (entrainment): S_wedge = -(1/2) surface-div(h U_wall), with the
        surface divergence in the spherical metric
            div_s(F) = (1/(R sin th)) d/dth(sin th F_th) + (1/(R sin th)) d/dph(F_ph).
        For pure squeeze (U_wall = 0) the wedge vanishes identically.
        """
        h = self.gap()
        S = self.gap_rate().copy()
        _ff = self._flow_factor_fields()
        if _ff is not None:
            # Eq. 10: the squeeze term takes h_T_bar, so its rate is Phi(H) hdot;
            # the Couette term takes h_T_bar too (below).
            S = S * _ff[1]
            # ONE code path for the average gap: mixed.h_T_bar, now closed form.
            from . import mixed as _mx
            h = _mx.h_T_bar(h, float(self.flow_factors["sigma"]))

        u_th, u_ph = self.surface_velocity()
        if np.any(u_th) or np.any(u_ph):
            mesh = self.mesh
            sin_th = np.sin(mesh.TH)
            F_th = h * u_th
            F_ph = h * u_ph
            dFth_dth, _ = mesh.field_gradient(sin_th * F_th)
            _, dFph_dph = mesh.field_gradient(F_ph)
            div_s = (dFth_dth + dFph_dph) / (self.R_b * sin_th)
            S = S - 0.5 * div_s
        return mesh_to_1d(self.mesh, S)

    # -- assembly / iteration -------------------------------------------------
    def _ghost(self, D: np.ndarray) -> np.ndarray:
        """Dirichlet ghost RHS for the current BC pair (mouth + pocket edge)."""
        return spherical_ghost_bc_rhs(
            self.mesh, D, self.conditions.p_case,
            self.conditions.p_ch if self.mesh.is_annular else None)

    def initial_pressure(self) -> np.ndarray:
        """Linear-in-theta ramp between the two Dirichlet values (annular), or a
        uniform mouth-value field (full cap)."""
        if not self.mesh.is_annular:
            return np.full(self.shape, self.conditions.p_case, dtype=float)
        w = ((self.mesh.TH - self.theta_min)
             / max(self.theta_max - self.theta_min, 1e-30))
        return (1.0 - w) * self.conditions.p_ch + w * self.conditions.p_case

    def _diffusion_coeff(self, model, fluid, eta_bar):
        h = self.gap()
        return h ** 3 / (12.0 * eta_bar)

    def assemble(
        self,
        model: str,
        fluid: Any,
        eta_bar: np.ndarray,
        gdot_bar: Optional[np.ndarray] = None,
        pressure: Optional[np.ndarray] = None,
        *,
        include_memory: bool = True,
        include_normal_stress: bool = True,
        include_alpha_coupling: bool = False,
        memory_scale: Optional[np.ndarray] = None,
        ns_scale: float = 1.0,
        n1_scale: Optional[float] = None,
        hoop_scale: Optional[float] = None,
    ):
        if model == "newtonian":
            eta = float(np.asarray(eta_bar).ravel()[0]) if np.ndim(eta_bar) else float(eta_bar)
            eta_field = np.full(self.shape, eta, dtype=float)
            D = self.gap() ** 3 / (12.0 * eta_field)
        elif model in ("GNF", "viscoelastic"):
            D = self.gap() ** 3 / (12.0 * eta_bar)
        else:
            raise ValueError(f"Unknown model type: {model}")

        if model == "viscoelastic":
            return self._assemble_viscoelastic(
                fluid, eta_bar, gdot_bar, pressure, D,
                include_memory=include_memory,
                include_normal_stress=include_normal_stress,
                memory_scale=memory_scale, ns_scale=ns_scale,
                n1_scale=n1_scale, hoop_scale=hoop_scale,
            )

        _ff = self._flow_factor_fields()
        if _ff is not None:
            D = D * _ff[0]
        A = build_spherical_diffusion_operator(self.mesh, D)
        b = self.source() + self._ghost(D)
        return _SimpleSystem(A=A, b=b)

    # -- surface vector calculus (physical components) ------------------------
    def grad_s(self, f2d) -> Tuple[np.ndarray, np.ndarray]:
        """Physical surface gradient (grad_th, grad_ph) of a scalar field."""
        df_dth, df_dph = self.mesh.field_gradient(f2d)
        grad_th = df_dth / self.R_b
        grad_ph = df_dph / (self.R_b * np.sin(self.mesh.TH))
        return grad_th, grad_ph

    def div_s(self, F_th, F_ph) -> np.ndarray:
        """Surface divergence of a tangential vector field (physical comps):
        div_s F = (1/(R sin th))[ d/dth(sin th F_th) + d/dph(F_ph) ]."""
        sin_th = np.sin(self.mesh.TH)
        dth_term, _ = self.mesh.field_gradient(sin_th * F_th)
        _, dph_term = self.mesh.field_gradient(F_ph)
        return (dth_term + dph_term) / (self.R_b * sin_th)

    def _assemble_viscoelastic(
        self, fluid, eta_bar, gdot_bar, pressure, D_gnf,
        include_memory, include_normal_stress, memory_scale, ns_scale,
        n1_scale=None, hoop_scale=None,
    ):
        """Full spherical VE system  L_ss + L_mem + L_N1 = S - S_hoop.

        MEMORY -- MOBILITY FORM (ruling R1, 2026-07-27).  The Reynolds equation
        is mass conservation, dh/dt + div(q) = 0, with q the TOTAL
        depth-integrated flux.  The squeeze-memory term is a FLUX contribution,
        obtained by integrating the cross-film velocity profile that the
        lambda d(sigma)/dt constitutive term induces:
            q_mem = -(lambda hdot / 12) (h^2 M / eta_bar^2) grad_s p
        Being a flux, the whole of it -- hdot included -- sits INSIDE the
        divergence, so it enters as a modification of the mobility:
            D_total = h^3/(12 eta_bar)  +  [-(lambda hdot/12)] (h^2 M/eta_bar^2)
            L[p] = div_s( D_total grad_s p )
        The earlier transcription placed hdot OUTSIDE the divergence (row
        scaling of the assembled operator), which removes the memory
        contribution from mass balance and has no derivation behind it.  The
        two forms COINCIDE when hdot is spatially uniform -- as in planar
        parallel-plate squeeze -- which is why the ambiguity stayed latent
        until the spherical domain, where hdot = edot cos(theta) varies.

        N1 / HOOP (metric-consistent divergence form; provisional, see Phase-4
        notes): RHS sources with the (1 - alpha/2) prefactor mirroring the planar
        L_N1 / S_hoop, built from Gamma^2 and the shared Psi1.
        """
        mesh = self.mesh
        h = self.gap()

        # ---- total mobility: viscometric + squeeze memory (MOBILITY FORM) ----
        D_total = D_gnf
        if include_memory and not fluid.is_newtonian:
            M = fluid.memory_viscosity(gdot_bar)
            hdot = self.gap_rate()
            mem_factor = -fluid.lambda_ * hdot / MEMORY_KERNEL_DENOM      # field
            D_mem = h ** 2 * M / eta_bar ** 2
            if memory_scale is not None:
                D_mem = D_mem * memory_scale
            # hdot INSIDE the divergence: fold it into the mobility and assemble
            # ONCE, so the conservative face averaging sees the TOTAL flux
            # coefficient (assembling the two operators separately and adding
            # them would face-average each independently, which is not the same
            # discretisation of div(D_total grad p)).
            D_total = D_gnf + mem_factor * D_mem

        _ff = self._flow_factor_fields()
        if _ff is not None:
            # F2 ARM A, as box D closed it: the factor multiplies the TOTAL
            # mobility, memory included.  The ghost BC below sees the same D.
            D_total = D_total * _ff[0]

        A = build_spherical_diffusion_operator(mesh, D_total)
        b = self.source() + self._ghost(D_total)

        # ---- N1 + hoop sources (RHS), (1 - alpha/2) prefactor ----
        # BOX C, Phase 6.  N1 and hoop were scaled TOGETHER by ns_scale, so the
        # E4 decomposition could not separate them.  ``n1_scale`` and
        # ``hoop_scale`` default to None, in which case each falls back to
        # ns_scale and the assembled system is BITWISE what it was before this
        # extension.  Only a caller that passes them explicitly sees a
        # difference.
        _n1 = ns_scale if n1_scale is None else float(n1_scale)
        _hoop = ns_scale if hoop_scale is None else float(hoop_scale)
        if (include_normal_stress and pressure is not None
                and not fluid.is_newtonian
                and max(abs(_n1), abs(_hoop)) > 1e-12):
            gpt, gpp = self.grad_s(pressure)
            grad_p_sq = gpt ** 2 + gpp ** 2
            u_th, u_ph = self.surface_velocity()
            V_T = np.sqrt(u_th ** 2 + u_ph ** 2)
            Gamma_sq = (V_T ** 2 / h ** 2) + (h ** 2 / (12 * eta_bar ** 2)) * grad_p_sq
            Psi1 = fluid.Psi1(gdot_bar)
            alpha = fluid.alpha

            # N1: (1-a/2)/6 div_s( (h^2/eta) grad_s(h Psi1 Gamma^2) )  [conservative]
            g_N1 = h * Psi1 * Gamma_sq
            coeff = h ** 2 / eta_bar
            S_N1 = (1 - alpha / 2) / 6 * surface_div_grad(mesh, coeff, g_N1)

            # hoop: -(1-a/2)/12 div_s( grad_s(h^3 Psi1 Gamma^2 / eta) )  [conservative]
            inner = h ** 3 * Psi1 * Gamma_sq / eta_bar
            ones = np.ones_like(inner)
            S_hoop = -(1 - alpha / 2) / 12 * surface_div_grad(mesh, ones, inner)

            b = b + _n1 * S_N1.ravel() + _hoop * S_hoop.ravel()

        return _SimpleSystem(A=A, b=b)

    def gap_averaged_shear_rate(self, pressure, eta_bar) -> np.ndarray:
        """Metric-consistent gap-averaged shear rate.

        |gdot| ~ sqrt( (u_wall/h)^2 + (h |grad_s p| / (2 eta))^2 ), the Couette +
        Poiseuille shear magnitude, with grad_s p the physical surface gradient.
        """
        h = self.gap()
        u_th, u_ph = self.surface_velocity()
        df_dth, df_dph = self.mesh.field_gradient(pressure)
        gradp_th = df_dth / self.R_b
        gradp_ph = df_dph / (self.R_b * np.sin(self.mesh.TH))
        grad_p = np.sqrt(gradp_th ** 2 + gradp_ph ** 2)
        couette = np.sqrt(u_th ** 2 + u_ph ** 2) / h
        poiseuille = h * grad_p / (2.0 * eta_bar)
        return np.sqrt(couette ** 2 + poiseuille ** 2)

    # -- post-processing ------------------------------------------------------
    # ------------------------------------------------------------------ #
    # BOUNDARY EXTREMA -- box D, registered convention (R-14.5), erratum E-9.
    #
    # ONE RULE, THREE INSTANCES: a reported pointwise extremum of a field
    # whose extremum lies on a domain BOUNDARY is taken over the interior
    # cell centres UNION the two theta faces -- the face value computed
    # analytically where the field has a closed form (h, h_dot) and by an
    # order-consistent extrapolation where it does not (T).
    #
    # Cell centres sit dtheta/2 inside each face, so the legacy interior-only
    # extrema are biased ONE-DIRECTIONALLY: h_min too HIGH, |h_dot|_max too
    # LOW, T_max too LOW.  Each of the three biases flatters feasibility.
    #
    # The union form carries NO monotonicity assumption: it can never report a
    # weaker extremum than the interior-only value, for any eccentricity
    # orientation.  Monotonicity is why the correction has a definite SIGN in
    # the registered regime; it is not required for the code to be correct.
    # ------------------------------------------------------------------ #

    def _face_normals(self):
        """(theta_min, theta_max) face normals, from mesh.normal_at_theta."""
        th_lo, th_hi = self.mesh.theta_faces()
        return (self.mesh.normal_at_theta(th_lo),
                self.mesh.normal_at_theta(th_hi))

    def h_min_boundary(self) -> float:
        """Minimum film over cell centres UNION both theta faces.

        h = c + e . n_hat is closed-form, so the face values carry no
        discretisation error at all.  On an annulus with an axial eccentricity
        h rises monotonically with theta, so the true minimum is at theta_min
        while ``gap().min()`` samples theta_min + dtheta/2 and
        OVERSTATES the rim film.
        """
        ex, ey, ez = self.e_vec
        vals = [float(self.gap().min())]
        for nx, ny, nz in self._face_normals():
            vals.append(float(np.min(self.c + ex * nx + ey * ny + ez * nz)))
        return min(vals)

    def gap_rate_max_boundary(self) -> float:
        """max |h_dot| over cell centres UNION both theta faces (closed form).

        Feeds De_sq = lambda |h_dot|_max / c.  g3 is a HARD threshold, so an
        under-reported |h_dot| labels rows feasible that are not.
        """
        ex, ey, ez = self.e_dot
        vals = [float(np.max(np.abs(self.gap_rate())))]
        for nx, ny, nz in self._face_normals():
            vals.append(float(np.max(np.abs(ex * nx + ey * ny + ez * nz))))
        return max(vals)

    def field_max_boundary(self, f2d: np.ndarray) -> float:
        """max of a field over cell centres UNION both theta faces, the faces
        by two-point linear extrapolation.

        For fields with no closed form -- T.  Faces lie dtheta/2 outside the end
        centres and centres are dtheta apart, so

            f_face = 1.5 f[0] - 0.5 f[1]     (theta_min)
            f_face = 1.5 f[-1] - 0.5 f[-2]   (theta_max)

        which is second-order, matching the operator.  R-14.4 measured this
        extremum's LOCATION to be grid-dependent -- interior at 32x24, at the
        rim by 64x48 -- so an interior-only maximum can miss the peak
        altogether, not merely under-report it.
        """
        F = np.asarray(f2d, dtype=float)
        if F.ndim != 2 or F.shape[0] < 2:
            return float(F.max())
        lo = 1.5 * F[0, :] - 0.5 * F[1, :]
        hi = 1.5 * F[-1, :] - 0.5 * F[-2, :]
        return float(max(F.max(), lo.max(), hi.max()))


    def load_axis_factors(self):
        """The two factors of the load-axis projection, kept SEPARATE.

            n_z = cos(theta)
            dA  = R^2 sin(theta) dtheta dphi

        Returned unmultiplied on purpose.  Floating-point multiplication is not
        associative, so folding them into a single weight and writing
        ``field * weight`` differs from ``field * n_z * dA`` by about one ULP.
        That is far inside the registered reproduction gate, but it is a
        gratuitous change to a number box B already published, so the original
        evaluation ORDER is preserved exactly.
        """
        nz = np.cos(self.mesh.TH)
        dA = (self.R_b ** 2 * np.sin(self.mesh.TH)
              * self.mesh.dtheta * self.mesh.dphi)
        return nz, dA

    def project_on_load_axis(self, surface_pressure: np.ndarray) -> float:
        """Integrate a surface pressure field onto the +z load axis.

        THE SINGLE PROJECTION (brief sec 3.5).  ``load_capacity`` and the box-C
        asperity load ``F_asp`` both go through this, so there is exactly one
        projection quadrature in the code and the film and contact shares of
        the load balance cannot drift apart.

        Extracting it was required rather than optional: the brief says do not
        write a second projection, and ``load_capacity`` previously carried its
        quadrature inline with no reusable form.  The arithmetic and its
        evaluation order are unchanged, and BS-V10 asserts that bitwise.
        """
        nz, dA = self.load_axis_factors()
        return float(np.sum(np.asarray(surface_pressure, dtype=float) * nz * dA))

    def load_capacity(self, pressure: np.ndarray) -> float:
        """Net load along the load axis (+z): F_z = int p cos th dA.

        Pressure acts along -n_hat on the ball; we report the magnitude of the
        z-resultant of the pressure traction on the ball surface.  Delegates to
        ``project_on_load_axis``; bitwise unchanged from the inline form.
        """
        return self.project_on_load_axis(pressure)

    def load_vector(self, pressure: np.ndarray) -> np.ndarray:
        """Full pressure-resultant vector on the ball, int p n_hat dA."""
        nx, ny, nz = self.mesh.normal()
        dA = self.R_b ** 2 * np.sin(self.mesh.TH) * self.mesh.dtheta * self.mesh.dphi
        return np.array([
            float(np.sum(pressure * nx * dA)),
            float(np.sum(pressure * ny * dA)),
            float(np.sum(pressure * nz * dA)),
        ])

    def moments(self, pressure: np.ndarray):
        """Transverse pressure-resultant components (F_x, F_y) [N] -- the
        off-load-line balance the eccentricity equilibrium search nulls.  (True
        torque about the ball centre vanishes: pressure acts radially.)
        """
        lv = self.load_vector(pressure)
        return float(lv[0]), float(lv[1])

    def surface_integral(self, field_2d: np.ndarray) -> float:
        return self.mesh.integrate(field_2d)

    def squeeze_rate_magnitude(self) -> float:
        return float(np.max(np.abs(self.gap_rate())))

    def leakage(self, pressure, eta_bar, fluid, flux_decomposition=None) -> float:
        """Volumetric leakage out of the socket mouth (theta = theta_max).

        Q = - oint_mouth (h^3/(12 eta)) dp/dn ds, with the mouth arc-length
        ds = R_b sin(theta_max) dphi and the outward normal derivative
        dp/dn = (1/R_b) dp/dtheta at theta_max.
        """
        h = self.gap()
        D = h ** 3 / (12.0 * eta_bar)
        i = self.mesh.Ntheta - 1
        # one-sided dp/dtheta at the mouth using the Dirichlet value
        dth = self.mesh.dtheta
        p_face = self.conditions.p_case
        # p at mouth face approximated from ghost: dp/dtheta ~ 2(p_ch - p_i)/dth
        dp_dth_mouth = 2.0 * (p_face - pressure[i, :]) / dth
        q = -D[i, :] * (dp_dth_mouth / self.R_b)
        ds = self.R_b * np.sin(self.theta_max) * self.mesh.dphi
        return float(np.sum(q * ds))


# small helpers ---------------------------------------------------------------

class _SimpleSystem:
    """Minimal assembled-system container (A, b) for the spherical path."""
    __slots__ = ("A", "b")

    def __init__(self, A, b):
        self.A = A
        self.b = b


def mesh_to_1d(mesh: SphericalMesh, f2d: np.ndarray) -> np.ndarray:
    return mesh.to_1d(f2d)
