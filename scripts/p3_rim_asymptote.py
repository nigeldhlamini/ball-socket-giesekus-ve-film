"""
VENDORED from Z:/Papers/Paper 3/New/p3_rim_asymptote.py (delivered with ruling R-55,
30 Sep 2026).  Policy (R-55, as for R-54's script): vendor it, do not trust it;
amendments live here, each documented below and each carried by a test in
tests/test_p3_rim_asymptote.py.

AMENDMENTS (vendored copy only)
  A1  The three interleaved ``if __name__ == "__main__"`` blocks are collected into
      one at the end, so every function is defined before any is called and an
      import runs nothing.  No numerical line is changed.
  A2  ``g_pocket`` added: the HYDROSTATIC part of the same film, pocket (inner rim)
      at unit gauge pressure, mouth at zero, no motion.  R-55.2 R1's positive
      control needs it ("the pocket term superposes linearly; extend the vendored
      script to compute it, with a test").  Same one-integral structure as the
      squeeze part:  d/dt( sin t H^3 p' ) = 0  =>  sin t H^3 p' = K,
      K = -1 / int (sin t)^-1 H^-3 dt,  load  F_hyd = 2 pi R^2 p_pocket g_pocket.
  A3  ``f_target`` added: the dimensionless load the smooth Newtonian film must
      carry, f_target = W c^2 / (12 pi eta R^4 eps_dot), with eps_dot passed in
      explicitly because in the solver it is NOT a constant: e_dot = omega eps c
      (solver.py:233), i.e. eps_dot = squeeze_phase * omega * eps.
  A4  ``solver_f``: the solver's film load converted to this module's f, so a
      comparison states its normalisation in one place.
  A5  (R-57.6) ``delta_for_load``: the load balance with the solver's squeeze rate.
      The solver prescribes eps_dot = omega * eps (solver.py:233), and eps =
      (1 - delta)/cos(theta_min) runs from ~0.1 to ~1.06 over the delta range, so
      ``delta_required``'s delta-INDEPENDENT f_target is the wrong balance for the
      solver.  The balance is
          W = 12 pi eta R^4 omega eps(delta) f(delta, theta_min) / c^2
              + 2 pi R^2 [ p_case (sin^2 tmax - sin^2 tmin)/2 + (p_pocket - p_case) g(delta) ],
      the hydrostatic bracket being the pocket/case superposition of A2 (zero when both
      pressures are zero, which is the ruling's squeeze-only form).  eps f is monotone
      decreasing in delta (both factors are), so the root is unique when bracketed.
      ``delta_required`` is left unchanged (it is the delivered function).

Everything below the amendments is the delivered text, unchanged.

-----------------------------------------------------------------------------------
Axisymmetric Newtonian, isothermal squeeze film on the annular spherical land of a
ball-socket joint.  Candidate analytical backbone for Paper 3's saturation mechanism.

Reynolds on the sphere (radius R, gap h = c H, H = 1 - eps cos(theta)), squeeze only:
    d/dtheta( sin(t) h^3 dp/dtheta ) = 12 eta R^2 sin(t) dh/dt,   dh/dt = -c eps_dot cos(t)
Integrate once:  sin(t) H^3 p' = P0 (k - sin^2 t),  P0 = 6 eta R^2 eps_dot / c^2.
k fixed by the two Dirichlet conditions p(theta_min) = p(theta_max) = 0 (squeeze part;
the hydrostatic pocket part superposes linearly).  Axial load
    F = 2 pi R^2 P0 * f(eps, theta_min),  f = int p~ sin t cos t dt.

Two land topologies:
  RECESS : land [theta_min, theta_max], rim pinned at pocket pressure (the joint).
  CLOSED : land [0, theta_max], regular at the pole (counterfactual, no recess).
Film variable delta = h_min / c: RECESS 1 - eps cos(theta_min); CLOSED 1 - eps.

Predicted leading order (RECESS, delta -> 0), derived in the accompanying note:
    f ~ (2 cos^5 theta_min / sin^2 theta_min) ln(1/delta) + O(1),
(the factor 2 comes from sin^2 t - sin^2 t_min ~ 2 sin t_min cos t_min x; a hand estimate
 without it was wrong by exactly 2 and was caught by this quadrature, 30 Sep 2026),
so d ln f / d ln delta -> -1/ln(1/delta) -> 0 (logarithmic saturation), whereas CLOSED
gives f ~ 1/delta (exponent -> -1, sphere-on-plane).
"""
import numpy as np
from scipy.integrate import quad

TMAX = np.deg2rad(73.0)

def _f(eps, tmin, tmax=TMAX):
    H = lambda t: 1.0 - eps*np.cos(t)
    pts = [tmin + x*(tmax - tmin) for x in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)] if tmin > 0 else None
    q = dict(limit=800, epsabs=0, epsrel=1e-11)
    if tmin > 0:
        a = quad(lambda t: np.sin(t)/H(t)**3, tmin, tmax, points=pts, **q)[0]
        b = quad(lambda t: 1/(np.sin(t)*H(t)**3), tmin, tmax, points=pts, **q)[0]
        k = a/b
        dp = lambda t: (k - np.sin(t)**2)/(np.sin(t)*H(t)**3)
    else:  # closed pole: regularity => k = 0
        dp = lambda t: -np.sin(t)/H(t)**3
    # f = int p sin cos dt; integrate by parts with p(tmax)=0:
    #   f = int_{tmin}^{tmax} p'(t) * (G(t) - G(tmin)) ... use G(t) = sin^2 t / 2 antiderivative
    #   f = [p G] - int p' G = -int p'(t) (sin^2 t - sin^2 tmin)/2 dt   (p(tmin)=0 or regular)
    s0 = np.sin(tmin)**2
    return quad(lambda t: -dp(t)*(np.sin(t)**2 - s0)/2.0, tmin, tmax,
                points=pts, **q)[0]

def f_recess(delta, tmin_deg):
    tmin = np.deg2rad(tmin_deg); eps = (1 - delta)/np.cos(tmin)
    return _f(eps, tmin)

def f_closed(delta):
    return _f(1 - delta, 0.0)

def delta_required(f_target, tmin_deg, lo=1e-14, hi=0.999):
    """Smallest-film equilibrium of the Newtonian isothermal smooth film: the delta at which
    f_recess(delta, tmin) = f_target, where f_target = W c^2 / (12 pi eta R_b^4 eps_dot) and W is
    the film's share of the load.  f is monotone decreasing in delta and diverges (log) as
    delta -> 0, so a root exists for every f_target > f_recess(hi).  Returns (delta, bracketed)."""
    from scipy.optimize import brentq
    g = lambda ld: np.log(f_recess(np.exp(ld), tmin_deg)) - np.log(f_target)
    a, b = np.log(lo), np.log(hi)
    if g(b) > 0:            # load already carried at the thickest film tested
        return hi, False
    if g(a) < 0:            # needs a film thinner than lo
        return lo, False
    return float(np.exp(brentq(g, a, b, xtol=1e-12))), True

def local_exp(fun, d1, d2):
    return (np.log(fun(d2)) - np.log(fun(d1)))/(np.log(d2) - np.log(d1))

def area_proj(tmin_deg, tmax=TMAX):   # projected (axial) land area / (pi R^2)
    return np.sin(tmax)**2 - np.sin(np.deg2rad(tmin_deg))**2
def area_sph(tmin_deg, tmax=TMAX):    # spherical land area / (2 pi R^2)
    return np.cos(np.deg2rad(tmin_deg)) - np.cos(tmax)

def theta_ratio_table():
    print("\nSqueeze-load ratio 40 vs 20 deg at common rim closure psi = eps cos(theta_min) = 1 - delta")
    print(f"{'psi':>5} {'F40/F20':>8} {'/proj-area':>11} {'/sph-area':>10} {'2cos^5/sin^2 ratio':>19}")
    Cr = (np.cos(np.deg2rad(40))**5/np.sin(np.deg2rad(40))**2)/(np.cos(np.deg2rad(20))**5/np.sin(np.deg2rad(20))**2)
    for psi in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999):
        d = 1 - psi
        r = f_recess(d, 40)/f_recess(d, 20)
        print(f"{psi:5.3f} {r:8.3f} {r/(area_proj(40)/area_proj(20)):11.3f} {r/(area_sph(40)/area_sph(20)):10.3f} {Cr:19.3f}")

def make_figure(path="p3_rim_asymptote.pdf"):
    """Two panels, greyscale-safe (marker + line style carry the cue, colour is secondary).
    Solver points (the draft's \\PThreeExpSqFirst/Last) are plotted at the geometric
    mid-point of their delta intervals; theta_min of those points is NOT recorded in the
    macro provenance, so they are shown as reference marks, not as a fit."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.logspace(-6, -0.3, 60)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    sty = {20: ("-", "o", "C0"), 30: ("--", "s", "C1"), 40: ("-.", "^", "C2")}
    for tm, (ls, mk, col) in sty.items():
        lf = np.log([f_recess(x, tm) for x in d])
        ex = np.gradient(lf, np.log(d))
        ax[0].plot(d, ex, ls, color=col, lw=1.6, label=fr"recess, $\theta_{{\min}}={tm}^\circ$")
        ax[0].plot(d[::10], ex[::10], mk, color=col, ms=4, mfc="white")
    lf = np.log([f_closed(x) for x in d]); ex = np.gradient(lf, np.log(d))
    ax[0].plot(d, ex, ":", color="k", lw=1.6, label="closed pole (no recess)")
    ax[0].plot([np.sqrt(0.1*0.03), np.sqrt(3e-3*1e-3)], [-0.94, -0.30], "D", color="k",
               ms=6, label="full solver (draft macros)")
    ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\delta = h_{\min}/c$")
    ax[0].set_ylabel(r"$\mathrm{d}\ln F/\mathrm{d}\ln\delta$"); ax[0].set_ylim(-1.25, 0.05)
    ax[0].axhline(0, color="0.6", lw=0.6); ax[0].legend(fontsize=6, loc="center left", bbox_to_anchor=(0.0, 0.42), framealpha=0.9)
    ax[0].set_title("(a) local load exponent", fontsize=9)
    psi = 1 - np.logspace(-6, -0.05, 50)
    r = np.array([f_recess(1-p, 40)/f_recess(1-p, 20) for p in psi])
    Cr = (np.cos(np.deg2rad(40))**5/np.sin(np.deg2rad(40))**2)/(np.cos(np.deg2rad(20))**5/np.sin(np.deg2rad(20))**2)
    ax[1].plot(1-psi, r/(area_proj(40)/area_proj(20)), "-", color="C0", lw=1.6, label="squeeze ratio / projected-area ratio")
    ax[1].plot(1-psi, r, "--", color="C3", lw=1.6, label=r"squeeze ratio $F_{40}/F_{20}$")
    ax[1].axhline(Cr, color="k", ls=":", lw=1.2, label=r"rim limit $2\cos^5\theta_{\min}/\sin^2\theta_{\min}$ ratio")
    ax[1].axhline(1, color="0.6", lw=0.6)
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"$\delta = 1-\psi$"); ax[1].set_ylim(0, 1.1)
    ax[1].legend(fontsize=6.5, loc="upper left"); ax[1].set_title(r"(b) $40^\circ$ vs $20^\circ$ at common closure", fontsize=9)
    fig.tight_layout(); fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"), dpi=160)


# ------------------------------------------------------------------------------------
# AMENDMENTS A2-A4 (vendored copy).  See the module docstring.
# ------------------------------------------------------------------------------------

def g_pocket(delta, tmin_deg, tmax=TMAX):
    """A2.  Hydrostatic load of the same land per unit pocket gauge pressure, / (2 pi R^2).

    p(theta_min) = 1, p(theta_max) = 0, no motion:  sin t H^3 p' = K,
    K = -1 / int_{tmin}^{tmax} (sin t)^-1 H^-3 dt.  Load by the same integration by
    parts as ``_f`` (p G vanishes at theta_min because G(tmin) = 0 there, and at theta_max
    because p = 0 there):  g = -int p' (sin^2 t - sin^2 tmin) / 2 dt.
    F_hyd = 2 pi R^2 p_pocket g.  As delta -> 0 the rim conductance vanishes and the whole
    land falls to mouth pressure, so g -> 0: the pocket load is carried by the POCKET
    (solver.py:217-221), not by the land, once the rim closes."""
    tmin = np.deg2rad(tmin_deg); eps = (1 - delta)/np.cos(tmin)
    H = lambda t: 1.0 - eps*np.cos(t)
    pts = [tmin + x*(tmax - tmin) for x in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)]
    q = dict(limit=800, epsabs=0, epsrel=1e-11)
    b = quad(lambda t: 1/(np.sin(t)*H(t)**3), tmin, tmax, points=pts, **q)[0]
    K = -1.0/b
    s0 = np.sin(tmin)**2
    return quad(lambda t: -(K/(np.sin(t)*H(t)**3))*(np.sin(t)**2 - s0)/2.0, tmin, tmax,
                points=pts, **q)[0]


def load_newtonian(delta, tmin_deg, *, eta, R, c, eps_dot, p_pocket=0.0):
    """Total axial film load [N] of the Newtonian isothermal smooth land, squeeze plus
    pocket hydrostatic, by superposition (the operator is linear at constant eta)."""
    return (12*np.pi*eta*R**4*eps_dot/c**2 * f_recess(delta, tmin_deg)
            + 2*np.pi*R**2*p_pocket * g_pocket(delta, tmin_deg))


def f_target(W, *, c, eta, R, eps_dot):
    """A3.  Dimensionless film load required: f = W c^2 / (12 pi eta R^4 eps_dot)."""
    return W*c**2/(12*np.pi*eta*R**4*eps_dot)


def solver_f(F_film, *, c, eta, R, omega, eps, squeeze_phase=1.0):
    """A4.  The solver's pure-squeeze film load in this module's normalisation.
    The solver prescribes e_dot = squeeze_phase * omega * eps * c (solver.py:233), so
    eps_dot = squeeze_phase * omega * eps here."""
    return f_target(F_film, c=c, eta=eta, R=R, eps_dot=squeeze_phase*omega*eps)


def load_solver_kinematics(delta, tmin_deg, *, eta, R, c, omega, p_pocket=0.0, p_case=0.0,
                           tmax=TMAX):
    """A5.  Total land load [N] of the Newtonian isothermal smooth film with the
    SOLVER's squeeze rate eps_dot = omega * eps(delta), plus the hydrostatic
    pocket/case superposition."""
    tmin = np.deg2rad(tmin_deg)
    eps = (1 - delta) / np.cos(tmin)
    sq = 12 * np.pi * eta * R ** 4 * omega * eps * f_recess(delta, tmin_deg) / c ** 2
    hyd = 2 * np.pi * R ** 2 * (p_case * (np.sin(tmax) ** 2 - np.sin(tmin) ** 2) / 2.0
                                + (p_pocket - p_case) * g_pocket(delta, tmin_deg))
    return sq + hyd


def delta_for_load(W, tmin_deg, *, eta, R, c, omega, p_pocket=0.0, p_case=0.0,
                   lo=1e-12, hi=0.999):
    """A5.  delta at which ``load_solver_kinematics`` equals W.  Returns (delta, bracketed);
    (hi, False) if W is carried at the thickest film, (lo, False) if a film thinner
    than ``lo`` would be needed."""
    from scipy.optimize import brentq
    kw = dict(eta=eta, R=R, c=c, omega=omega, p_pocket=p_pocket, p_case=p_case)
    g = lambda ld: load_solver_kinematics(np.exp(ld), tmin_deg, **kw) - W
    a, b = np.log(lo), np.log(hi)
    if g(b) > 0:
        return hi, False
    if g(a) < 0:
        return lo, False
    return float(np.exp(brentq(g, a, b, xtol=1e-13))), True


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    print("Local exponent d ln F / d ln delta over the draft's two intervals")
    print(f"{'land':>12} {'0.1->0.03':>10} {'0.003->0.001':>13}")
    for tm in (20, 30, 40):
        g = lambda d, tm=tm: f_recess(d, tm)
        print(f"{'recess '+str(tm)+'deg':>12} {local_exp(g,0.1,0.03):10.3f} {local_exp(g,3e-3,1e-3):13.3f}")
    print(f"{'closed pole':>12} {local_exp(f_closed,0.1,0.03):10.3f} {local_exp(f_closed,3e-3,1e-3):13.3f}")
    print("\nLog-coefficient check: d f / d ln(1/delta) -> 2 cos^5/sin^2 (RECESS)")
    for tm in (20, 30, 40):
        t = np.deg2rad(tm); C = 2*np.cos(t)**5/np.sin(t)**2
        vals = [f_recess(d, tm) for d in (1e-4, 1e-6, 1e-8)]
        slope = (vals[2]-vals[1])/np.log(100.0)      # df / d ln(1/delta)
        print(f"  theta_min={tm}: predicted C={C:.4f}, measured d f/d ln(1/delta) = {slope:.4f}")
    theta_ratio_table()
    make_figure()
