"""ball_socket/mixed.py -- mixed lubrication: Patir-Cheng flow factors and
Greenwood-Tripp asperity contact.

CONSTANT DISCIPLINE (standing rule 19).  Every constant used here is either
CITED in the registered config or VERIFY.  **No function returns a number built
on an unread constant -- it raises**, exactly as ``scripts/kinematics_a4vg.py``
raises on an unread ``FRAME`` entry.  The registered values live in
``campaign_config_v5`` (``PATIR_CHENG``, ``GREENWOOD_TRIPP``, ``GT_PARAM_BAND``)
and are imported, never restated: a registered quantity that is not in
``config_dict()`` is not registered.

WHAT IS AVAILABLE AND WHAT IS NOT, as of the pass-2 primary read:

  AVAILABLE (all Patir-Cheng coefficients are CITED PRIMARY)
    * pressure flow factors phi_x, phi_y, both gamma branches
    * shear flow factor Phi_s and the variance-ratio combination phi_s
    * shear STRESS factors phi_fp, phi_fs
    * the average gap h_T_bar, by quadrature
    * F_n, by quadrature -- a pure integral with no fitted constants
    * the CONTACT AREA fraction, which needs only the CITED (eta beta sigma)

  AVAILABLE since ruling R-10 (every constant closed as a band, a declared
  convention or a class value -- none needed a measurement)
    * asperity LOAD p_a, GT Eq. 7 elastic branch
    * the plastic branch, GT Eq. 13  (fork F5 arm B)
    * the plasticity index psi, over the full registered cross-product

  DESCOPED (raises, and is meant to)
    * boundary friction  -- tau_0 and m describe a boundary chemistry that is
      the OBJECT of study, not an input to it.  Replaced by a bounding sweep.

  BAND DISCIPLINE.  sigma_s_over_beta_a and H_ind are SWEPT BANDS.  Every entry
  point that uses them takes the chosen end EXPLICITLY and will not pick one
  for you, so no result can silently rest on one end of a band (rule 15).

TWO TRAPS, both recorded at the point of use because separating them re-opens
them:

  (1) **The mobility keeps h^3.**  Patir-Cheng Eq. 10 carries the NOMINAL film
      in the Poiseuille term and the AVERAGE gap h_T_bar in the Couette and
      squeeze terms.  The brief's instruction to use h_T in the mobility was
      wrong.  ``poiseuille_mobility`` therefore takes ``h``, and
      ``h_T_bar`` is applied only where Eq. 10 puts it.

  (2) **Two different sigmas.**  Patir-Cheng's sigma is the composite SURFACE
      roughness (what the campaign registers as sigma*); Greenwood-Tripp's is
      the composite SUMMIT height spread, which is smaller.  Every function
      below names which one it wants in its signature.  Passing one for the
      other is a silent error -- see fork F3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import integrate

from .campaign_config_v5 import (PATIR_CHENG, GREENWOOD_TRIPP, GT_PARAM_BAND,
                                 HARDNESS_CONVERSION, BOUNDARY_FRICTION,
                                 PEKLENIK_GAMMA, SIGMA_STAR_BAND_M)


class MixedNotVerified(RuntimeError):
    """Raised when a computation needs a constant that is still VERIFY (NaN).

    Mirrors ``kinematics_a4vg.FrameNotVerified``.  The mixed-lubrication model
    is deliberately half-built: the Patir-Cheng half is CITED and runs, the
    Greenwood-Tripp load half is not and raises.  Returning a placeholder
    number for the second would be the defect rule 19 exists to prevent.
    """


def _need(block: dict, key: str, block_name: str):
    """Return a registered value, or raise if it is still VERIFY."""
    entry = block[key]
    v = entry["value"]
    vals = v if isinstance(v, (tuple, list)) else (v,)
    if any(not math.isfinite(float(x)) for x in vals):
        raise MixedNotVerified(
            f"{block_name}['{key}'] is {entry['status']}: {entry['what']}  "
            f"Read it from its source and mark it CITED before calling this.")
    return v


def _gamma_key(gamma: float) -> str:
    """Table row key for a Peklenik gamma, exact matches only.

    Deliberately NOT interpolating.  Patir and Cheng tabulate seven gammas;
    interpolating between them would invent coefficients that no source states.
    """
    for k, val in (("1/9", 1/9), ("1/6", 1/6), ("1/3", 1/3), ("1", 1.0),
                   ("3", 3.0), ("6", 6.0), ("9", 9.0)):
        if abs(gamma - val) < 1e-12:
            return k
    raise ValueError(
        f"gamma = {gamma} is not a tabulated Patir-Cheng row.  Tabulated: "
        f"1/9, 1/6, 1/3, 1, 3, 6, 9.  Interpolation is refused: it would "
        f"invent coefficients no source states.")


# ---------------------------------------------------------------------------
# Patir-Cheng 1978 -- pressure flow factors (CITED)
# ---------------------------------------------------------------------------

def phi_x(H, gamma: float = PEKLENIK_GAMMA):
    """Pressure flow factor in the x (here: theta) direction.

    ``H = h / sigma`` with sigma the composite **SURFACE** roughness.
    Eq. 22 / Table 1.  Two branches:
        gamma <= 1 : phi_x = 1 - C exp(-r H)
        gamma >  1 : phi_x = 1 + C H^(-r)      (POWER LAW, not exponential)
    """
    C, r, _ = PATIR_CHENG["pressure_flow_factor"]["table_1"][_gamma_key(gamma)]
    H = np.asarray(H, dtype=float)
    if gamma <= 1.0:
        return 1.0 - C * np.exp(-r * H)
    return 1.0 + C * H ** (-r)


def phi_y(H, gamma: float = PEKLENIK_GAMMA):
    """Pressure flow factor in the y (here: phi) direction.

    **PC1978 Eq. 27: phi_y(H, gamma) = phi_x(H, 1/gamma).**  The "two-field
    operator" the brief asked to build as new work is this identity.  At
    gamma = 1 it coincides with ``phi_x`` identically and the scalar path is
    exact, which is what box C runs.
    """
    return phi_x(H, 1.0 / gamma)


def flow_factor_valid_above(gamma: float = PEKLENIK_GAMMA) -> float:
    """Lower cut-off in H below which the tabulated fit is not claimed."""
    return PATIR_CHENG["pressure_flow_factor"]["table_1"][_gamma_key(gamma)][2]


# ---------------------------------------------------------------------------
# Patir-Cheng 1979 -- shear flow factor (CITED)
# ---------------------------------------------------------------------------

def Phi_s(H, gamma: float = PEKLENIK_GAMMA):
    """PER-SURFACE shear flow factor, Table 2.  Positive by construction.

    Note the capital: ``Phi_s`` is one surface's contribution and is positive;
    ``phi_s`` below is the SIGNED DIFFERENCE of two of them.  The secondary
    literature that printed a leading minus was showing one term of a
    particular case.
    """
    A1, a1, a2, a3, A2 = PATIR_CHENG["shear_flow_factor"]["table_2"][_gamma_key(gamma)]
    H = np.asarray(H, dtype=float)
    # np.where evaluates BOTH branches, and the H <= 5 fit has a +a3 H^2 in the
    # exponent that overflows for large H.  Clip the argument of the branch
    # that will be discarded; the selection is unchanged.
    Hc = np.minimum(H, 5.0)
    low = A1 * np.abs(Hc) ** a1 * np.exp(-a2 * Hc + a3 * Hc ** 2)
    return np.where(H <= 5.0, low, A2 * np.exp(-0.25 * H))


def phi_s(H, sigma_1: float, sigma_2: float,
          gamma_1: float = PEKLENIK_GAMMA, gamma_2: float = PEKLENIK_GAMMA):
    """Combined shear flow factor, PC1979 Eqs. 17, 18.

        phi_s = V_r1 Phi_s(H, gamma_1) - V_r2 Phi_s(H, gamma_2)
        V_r1  = (sigma_1/sigma)^2,   V_r2 = 1 - V_r1,   sigma^2 = s1^2 + s2^2

    For steel on bronze the two Rq differ by roughly an order of magnitude, so
    V_r1 and V_r2 are far from equal and the two terms do NOT cancel.
    """
    s2 = sigma_1 ** 2 + sigma_2 ** 2
    V_r1 = sigma_1 ** 2 / s2
    return V_r1 * Phi_s(H, gamma_1) - (1.0 - V_r1) * Phi_s(H, gamma_2)


# ---------------------------------------------------------------------------
# Patir-Cheng 1979 -- shear STRESS factors (CITED)
# ---------------------------------------------------------------------------
# The brief never asked for these.  P_loss under mixed lubrication cannot be
# computed without them: carrying flow factors in the pressure field while
# taking dissipation from smooth-film shear is internally inconsistent.

def phi_fp(H, gamma: float = PEKLENIK_GAMMA):
    """Eq. 25: phi_fp = 1 - D exp(-s H).  Table 3, valid H > 0.5.

    DOMAIN GUARD.  Table 3 states the range for gamma = 1 as "H > .5",
    STRICTLY.  H = 0.5 is therefore OUTSIDE the fitted range, not at its edge:
    1 - 1.40 exp(-0.33) = -0.006493, and the fit only turns positive at
    H ~ 0.51.  A negative shear-stress factor is unphysical, so this raises
    below the tabulated range rather than returning it -- the same treatment
    the H <= 5 / H > 5 split already gets, and the same treatment rule 19
    gives an unread constant.  Never clip silently.
    """
    if _gamma_key(gamma) != "1":
        raise MixedNotVerified(
            "only the gamma = 1 row of Table 3 was transcribed at pass 2; "
            "read the remaining rows before running a gamma sweep.")
    t = PATIR_CHENG["shear_stress_factors"]["phi_fp"]["table_3_gamma_1"]
    H = np.asarray(H, dtype=float)
    lo = flow_factor_valid_above(gamma)          # 0.5 at gamma = 1
    if np.any(H <= lo):
        bad = float(np.min(H))
        raise MixedNotVerified(
            f"phi_fp evaluated at H = {bad} but Patir & Cheng (1979) Table 3 "
            f"states the range for gamma = {gamma} as H > {lo}, strictly.  "
            f"Below it the fit returns a negative shear-stress factor "
            f"({1.0 - t['D'] * math.exp(-t['s'] * bad):.6f} at H = {bad}), "
            f"which is unphysical.  Extrapolating outside a fitted range is "
            f"the same defect as building a number on an unread constant.")
    return 1.0 - t["D"] * np.exp(-t["s"] * H)


def Phi_fs(H, gamma: float = PEKLENIK_GAMMA):
    """Eqs. 26, 27: Phi_fs = A3 H^a4 exp(-a5 H + a6 H^2), zero for H > 7."""
    if _gamma_key(gamma) != "1":
        raise MixedNotVerified(
            "only the gamma = 1 row of Table 4 was transcribed at pass 2; "
            "read the remaining rows before running a gamma sweep.")
    t = PATIR_CHENG["shear_stress_factors"]["phi_fs"]["table_4_gamma_1"]
    H = np.asarray(H, dtype=float)
    # Same both-branch hazard as Phi_s: clip before exponentiating.
    Hc = np.minimum(H, 7.0)
    val = (t["A3"] * np.abs(Hc) ** t["alpha4"]
           * np.exp(-t["alpha5"] * Hc + t["alpha6"] * Hc ** 2))
    return np.where(H > 7.0, 0.0, val)


def phi_fs(H, sigma_1: float, sigma_2: float,
           gamma_1: float = PEKLENIK_GAMMA, gamma_2: float = PEKLENIK_GAMMA):
    """Combined shear-stress factor, same variance-ratio form as ``phi_s``."""
    s2 = sigma_1 ** 2 + sigma_2 ** 2
    V_r1 = sigma_1 ** 2 / s2
    return V_r1 * Phi_fs(H, gamma_1) - (1.0 - V_r1) * Phi_fs(H, gamma_2)


# ---------------------------------------------------------------------------
# The average gap (CITED form, quadrature)
# ---------------------------------------------------------------------------

def h_T_bar(h, sigma: float):
    """PC1978 Eq. 2: the mean gap seen by the flow, IN CLOSED FORM.

        h_T_bar = int_{-h}^{inf} (h + delta) f(delta) d delta
                = sigma [ H Phi(H) + phi(H) ],        H = h / sigma

    with Phi and phi the standard normal CDF and density: the integral of
    t f(t) from -H to infinity is f(H).  Its derivative is exact as well,

        d h_T_bar / d h = Phi(H).

    CORRECTION (ruling R-38.7, 2026-09-17).  This docstring previously said
    "No closed form in erf; computed by quadrature."  That was wrong, and a
    note claiming a quantity has no closed form is the kind that stops anyone
    looking.  The closed form agrees with the cited quadrature -- retained as
    ``h_T_bar_quadrature`` for exactly that check -- to 2.7e-16, and it removes
    a per-cell numerical integral from any inner loop that needs the gap.

    **WHERE THIS GOES.**  Eq. 10 puts h_T_bar in the COUETTE and SQUEEZE terms
    only.  The Poiseuille mobility keeps h^3.  Do not substitute it into the
    mobility -- see ``poiseuille_mobility``.
    """
    from scipy.special import erf
    H = np.asarray(h, dtype=float) / float(sigma)
    Phi = 0.5 * (1.0 + erf(H / math.sqrt(2.0)))
    phi = np.exp(-0.5 * H * H) / math.sqrt(2.0 * math.pi)
    out = float(sigma) * (H * Phi + phi)
    return out if np.ndim(h) else float(out)


def dh_T_bar_dh(h, sigma: float):
    """d h_T_bar / d h = Phi(H), exactly -- the squeeze-source factor."""
    from scipy.special import erf
    H = np.asarray(h, dtype=float) / float(sigma)
    out = 0.5 * (1.0 + erf(H / math.sqrt(2.0)))
    return out if np.ndim(h) else float(out)


def h_T_bar_quadrature(h, sigma: float):
    """PC1978 Eq. 2 by direct quadrature -- the independent check on ``h_T_bar``.

    Kept, not deleted: a closed form is a claim, and this is what it is checked
    against.
    """
    h_arr = np.atleast_1d(np.asarray(h, dtype=float))
    out = np.empty_like(h_arr)
    for i, hh in enumerate(h_arr.ravel()):
        H = hh / sigma
        f = lambda t, H=H: (H + t) * math.exp(-t * t / 2.0) / math.sqrt(2 * math.pi)
        v, _ = integrate.quad(f, -H, 40.0, limit=200,
                              epsabs=1e-300, epsrel=1e-12)
        out.ravel()[i] = v * sigma
    return out.reshape(h_arr.shape) if np.ndim(h) else float(out.ravel()[0])


def poiseuille_mobility(h, eta_bar):
    """D = h^3 / (12 eta_bar) -- the NOMINAL film, deliberately.

    Kept as a named function so that the one place the trap could be sprung is
    the one place it is documented.  PC1978 Eq. 10's Poiseuille term carries
    h^3, not h_T_bar.  The flow factor multiplies THIS.
    """
    return np.asarray(h, dtype=float) ** 3 / (12.0 * np.asarray(eta_bar, dtype=float))


# ---------------------------------------------------------------------------
# Greenwood-Tripp -- F_n by quadrature (no fitted constants at all)
# ---------------------------------------------------------------------------

def F_n(n: float, H, _cut: float = 40.0):
    """GT p. 627: F_n(H) = (1/sqrt(2 pi)) int_H^inf (u-H)^n exp(-u^2/2) du.

    Computed by QUADRATURE.  The widely quoted polynomial fit
    ``F_5/2 ~ A (4-H)^6.804`` is **rejected**, not merely disputed: against
    GT's own Appendix 2 both circulating variants are ~11 % low at H = 0, which
    is precisely the heavy-contact regime this joint operates in.
    """
    H_arr = np.atleast_1d(np.asarray(H, dtype=float))
    out = np.empty_like(H_arr)
    for i, hh in enumerate(H_arr.ravel()):
        f = lambda u, hh=hh, n=n: (u - hh) ** n * math.exp(-u * u / 2.0)
        # epsabs MUST be driven to zero: F_n falls to ~1e-35 by H = 12, and
        # quad's default absolute tolerance of 1.49e-8 would terminate
        # immediately and return noise.  Accuracy here is RELATIVE.
        v, _ = integrate.quad(f, hh, hh + _cut, limit=400,
                              epsabs=1e-300, epsrel=1e-12)
        out.ravel()[i] = v / math.sqrt(2 * math.pi)
    return out.reshape(H_arr.shape) if np.ndim(H) else float(out.ravel()[0])


def F_n_recurrence_residual(n: float, H: float) -> float:
    """GT Appendix 2: F_{n+1}(H) = n F_{n-1}(H) - H F_n(H).  Returns the
    residual, for BS-V9 to assert on.  Verified to close at 4.9e-14."""
    return float(F_n(n + 1, H) - (n * F_n(n - 1, H) - H * F_n(n, H)))


def F_n_asymptotic(n: float, H: float) -> float:
    """Far-tail form, GT Appendix 2, WITH THE EXPONENT CORRECTED.

        F_n(H) ~ (n! / sqrt(2 pi)) H^{-(n+1)} exp(-H^2/2)
                 [1 - (n+1)(n+2)/(2 H^2) + ...]

    The dossier printed ``H^{-n}``.  The correct power is ``H^{-(n+1)}``, from
    u = H + t and int_0^inf t^n e^{-Ht} dt = Gamma(n+1)/H^{n+1}.  Measured: the
    printed form diverges linearly in H (ratio to quadrature 0.69, 5.69, 11.96
    at H = 3, 6, 12) while this one converges (0.2304, 0.9487, 0.9967).
    """
    from scipy.special import gamma as _G
    return float((_G(n + 1) / math.sqrt(2 * math.pi)) * H ** (-(n + 1))
                 * math.exp(-H * H / 2.0)
                 * (1.0 - (n + 1) * (n + 2) / (2.0 * H * H)))


# ---------------------------------------------------------------------------
# Greenwood-Tripp -- contact area (UNBLOCKED) and load (BLOCKED)
# ---------------------------------------------------------------------------

def eta_beta_sigma_band(single_surface: bool = False) -> tuple[float, float]:
    """The CITED (eta beta sigma) band, both ends (standing rule 15).

    Default is the JOINT value, 0.0424 to 0.0707.  GT p. 628 gives 0.03 to 0.05
    for the SINGLE-surface group; the joint value is that times sqrt(2) because
    the sum of two Gaussian asperity-height distributions has sigma = sigma_1
    sqrt(2).  The secondary literature drops that qualifier.
    """
    e = GT_PARAM_BAND["eta_a_beta_a_sigma_s"]
    return tuple(e["single_surface_reading"] if single_surface else e["value"])


def contact_area_fraction(H, eta_beta_sigma: float):
    """GT Eq. 8: A_c/A_n = pi^2 (eta beta sigma)^2 F_2(H).

    **UNBLOCKED**: needs only the CITED group.  ``H = d / sigma`` with sigma the
    composite SUMMIT height spread, not the composite surface roughness.
    """
    return math.pi ** 2 * eta_beta_sigma ** 2 * F_n(2.0, H)


def n_contacts(H, eta_beta_sigma: float, eta_a: float, area: float):
    """GT Eq. 9: N(d) = 4 pi (eta beta sigma) eta A F_1(d/sigma)."""
    return 4.0 * math.pi * eta_beta_sigma * eta_a * area * F_n(1.0, H)


def sigma_s_over_beta_a_band() -> tuple[float, float]:
    """The registered SWEPT decade for the summit-slope group (ruling R-10.3).

    Returned as a band, never a scalar, so that no caller can pick an end by
    accident.  Every entry point below takes the chosen end EXPLICITLY.
    """
    return tuple(GT_PARAM_BAND["sigma_s_over_beta_a"]["value"])


def H_ind_band() -> tuple[float, float]:
    """Socket indentation hardness band, derived from the CITED 80-120 HB.

    LOWER BOUND on asperity-scale hardness: Brinell is a macro-scale test and
    the indentation size effect raises hardness at asperity scale.  The
    direction of that error is known and is registered with the constant.
    """
    return tuple(HARDNESS_CONVERSION["H_ind_band_Pa"]["value"])


def asperity_pressure_elastic(H, eta_beta_sigma: float, E_star: float,
                              sigma_s_over_beta_a: float):
    """GT Eq. 7, elastic branch.

        p_a = (16 sqrt2 / 15) pi (eta beta sigma)^2 E' sqrt(sigma/beta) F_5/2(H)

    ``sigma_s_over_beta_a`` is REQUIRED and SCALAR: it is a swept band and this
    function will not choose an end for you (standing rule 15).  Use
    ``sigma_s_over_beta_a_band()`` and call once per end.

    **THE MODULUS PAIRING IS PART OF THE CONSTANT.**  E' is the HERTZ REDUCED
    modulus, 1/E' = (1-nu1^2)/E1 + (1-nu2^2)/E2, defined on GT p. 627 -- exactly
    what ``geometry.composite_modulus()`` computes.  The ``8 sqrt2/15`` form
    circulating elsewhere goes with ``E* = 2 E_Hertz``; taking one prefactor
    with the other modulus is a clean factor of two.
    """
    sob = float(sigma_s_over_beta_a)
    lo, hi = sigma_s_over_beta_a_band()
    if not (lo <= sob <= hi):
        raise MixedNotVerified(
            f"sigma_s_over_beta_a = {sob:g} is outside the registered swept "
            f"band [{lo:g}, {hi:g}].  Extrapolating outside a registered band "
            f"is the same defect as building a number on an unread constant.")
    return ((16.0 * math.sqrt(2.0) / 15.0) * math.pi * eta_beta_sigma ** 2
            * E_star * math.sqrt(sob) * F_n(2.5, H))


def asperity_pressure_plastic(H, eta_beta_sigma: float, H_ind: float):
    """GT Eq. 13, plastic branch (misaligned asperities), fork F5 arm B.

        P(d) = 2 pi (eta beta sigma)^2 H_ind A F_2(d/sigma)

    This REPLACES the hard cap ``p_a <- min(p_a, H_ind)`` the brief called
    "required": that cap is not a sourced convention, and GT supply their own
    plastic branch.  Eq. 13 over Eq. 8 gives a mean real-contact pressure of
    ``2 H_ind / pi = 0.637 H_ind``, NOT ``H_ind``, so a cap at H_ind is not
    what the model does even fully plastic.

    ``H_ind`` is REQUIRED and SCALAR; see ``H_ind_band()``.
    """
    return 2.0 * math.pi * eta_beta_sigma ** 2 * float(H_ind) * F_n(2.0, H)


def plasticity_index(E_star: float, sigma_s_over_beta_a: float,
                     H_ind: float) -> float:
    """psi = (E*/H_ind) sqrt(sigma_s/beta_a).  Both band inputs explicit.

    Greenwood & Williamson thresholds: psi < 0.6 elastic, psi > 1 plastic, and
    0.6 < psi < 1 an explicitly UNDETERMINED band.

    E10 is registered as psi exceeding 1 over the MAJORITY of the band, so the
    reportable object is psi over the full cross-product, not a single value.
    """
    return (E_star / float(H_ind)) * math.sqrt(float(sigma_s_over_beta_a))


def plasticity_index_cross_product(E_star: float) -> list[dict]:
    """psi at every corner of the registered cross-product -- E10's evidence."""
    out = []
    for sob in sigma_s_over_beta_a_band():
        for h in H_ind_band():
            psi = plasticity_index(E_star, sob, h)
            out.append({"sigma_s_over_beta_a": sob, "H_ind_Pa": h,
                        "psi": psi,
                        "regime": ("plastic" if psi > 1.0
                                   else "undetermined" if psi > 0.6
                                   else "elastic")})
    return out


def boundary_shear(p_a):
    """tau_b = tau_0 + m p_a -- **DESCOPED from this paper** (ruling R-10.5).

    tau_0 and m describe the boundary film of a SPECIFIC additive package on a
    zinc-bearing brass, and the published values are for ZDDP on steel.  The
    zinc-free formulation IS the failure signal, so the difference between
    those chemistries is the object of study, not an input to it.  Replaced by
    the bounding sweep registered in ``BOUNDARY_FRICTION["bounding_sweep"]``.
    """
    raise MixedNotVerified(
        "boundary_shear is DESCOPED from box C (ruling R-10.5).  tau_0 and m "
        "describe a boundary chemistry that is the OBJECT of study, not an "
        "input to it.  Use the registered bounding sweep instead: run the "
        "thermal result with boundary friction off and at a stated upper "
        "range, and report whether the regime map moves.")



# ---------------------------------------------------------------------------
# The contact model as ONE recorded object (box C, Phase 2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContactModel:
    """Every band end and every fork arm the asperity load depends on, in one
    frozen object.

    WHY AN OBJECT RATHER THAN LOOSE ARGUMENTS.  The asperity load depends on
    four registered band ends and two open rule-18 forks.  Passing those
    separately invites a caller to fix some and default others, which is how a
    result comes to rest silently on one end of a band.  Bundling them means
    the choice is made once, is explicit, and can be written into the row as a
    single string -- so every persisted number carries the model that produced
    it.

    THE FORKS ARE NOT RESOLVED HERE.  ``summit_convention`` is fork F3 and
    ``branch`` is fork F5.  Both arms are implemented; neither is a default;
    the caller must choose and the choice is recorded.
    """

    #: composite SURFACE roughness sigma* -- Patir-Cheng's sigma, a band end
    sigma_star: float
    #: (eta beta sigma) for the pair, CITED band end
    eta_beta_sigma: float
    #: summit-slope group, SWEPT band end
    sigma_s_over_beta_a: float
    #: socket indentation hardness, DERIVED band end
    H_ind: float
    #: fork F3, NO DEFAULT (box E sec 2.3).  Box D closed F3 as DERIVED, so
    #: "derived" is the box-E arm; "sigma_s_eq_sigma_star" and "nayak" are box
    #: C's arms, retained so superseded results remain reproducible.
    #:
    #: THE DEFAULTS ARE REMOVED, NOT UPDATED.  With branch="elastic" and
    #: summit_convention="sigma_s_eq_sigma_star" present, wiring the campaign
    #: would have run box C's arm A while the manifest recorded DERIVED -- and,
    #: worse, the choice would have been INVISIBLE to a signature-derived
    #: coverage gate, because a default is precisely a choice that does not look
    #: like one.
    summit_convention: str
    #: fork F5, NO DEFAULT.  "psi" is box D's registered three-state selection;
    #: "elastic"/"plastic" are box C's fixed arms.
    branch: str
    #: Nayak bandwidth parameter alpha = m0 m4 / m2^2.  REQUIRED for F3 arm B,
    #: and obtainable only from an areal scan or a registered band.
    nayak_alpha: float | None = None

    def __post_init__(self):
        if self.summit_convention not in ("sigma_s_eq_sigma_star", "nayak",
                                          "derived"):
            raise ValueError(f"unknown F3 arm {self.summit_convention!r}")
        if self.branch not in ("elastic", "plastic", "psi"):
            raise ValueError(f"unknown F5 arm {self.branch!r}")
        if self.summit_convention == "nayak" and self.nayak_alpha is None:
            raise MixedNotVerified(
                "fork F3 arm B (Nayak/McCool) needs the bandwidth parameter "
                "alpha = m0 m4 / m2^2, which comes from an areal scan or a "
                "registered band.  Neither exists, so arm B is IMPLEMENTED but "
                "not EVALUABLE.  Arm A is the alpha -> infinity limit of this "
                "same relation, which is why arm A is not a neutral "
                "alternative but the extreme end of arm B.")

    def sigma_summit(self) -> float:
        """The composite SUMMIT height spread that Greenwood-Tripp's F_n needs.

        NOT the same as ``sigma_star``, which is the composite SURFACE
        roughness Patir-Cheng needs.  Confusing them is a silent error; fork F3
        is exactly the question of the relation between them.
        """
        if self.summit_convention == "sigma_s_eq_sigma_star":
            return self.sigma_star
        if self.summit_convention == "derived":
            # BOX D's F3 CLOSURE, CONSUMED.  sigma_s/sigma* is DERIVED from the
            # already-CITED (eta beta sigma) band via McCool Eqs. 34-35; the
            # ratio is read from the registered key rather than recomputed here,
            # so the campaign cannot drift from the registration.
            from .campaign_config_v7 import summit_ratio_for
            return self.sigma_star * summit_ratio_for(self.eta_beta_sigma)
        return self.sigma_star * math.sqrt(1.0 - 0.8968 / float(self.nayak_alpha))

    def gap_ratio(self, h) -> np.ndarray:
        """H = h / sigma_summit -- Greenwood-Tripp's argument.

        Factored out because three entry points need it and an inline copy in
        each is three places for the summit convention to diverge.
        """
        return np.asarray(h, dtype=float) / self.sigma_summit()

    def pressure(self, h) -> np.ndarray:
        """Asperity pressure field.  Branch per box E sec 2.3.

        For ``branch == "psi"`` the arm is selected PER SAMPLE from the
        plasticity index against Greenwood & Williamson's CITED thresholds.
        The doubt band 0.6 <= psi <= 1.0 raises rather than choosing: box E sec
        2.2 registers that such a row is solved under BOTH arms and both are
        carried, so a single pressure field is not the right answer and
        returning one silently would assert a deformation mode over exactly the
        range GW call "in doubt".  ``psi_regime()`` and ``pressure_arm()`` are
        the entry points for that case.
        """
        H = self.gap_ratio(h)
        if self.branch == "psi":
            regime = self.psi_regime()
            if regime == "undetermined":
                raise MixedNotVerified(
                    f"psi = {self.psi():.4g} is in GW's doubt band "
                    f"[0.6, 1.0]; box E sec 2.2 requires BOTH arms to be "
                    f"solved and carried.  Call pressure_arm('elastic') and "
                    f"pressure_arm('plastic') -- this entry point will not "
                    f"pick one for you.")
            return self.pressure_arm(regime, h)
        return self.pressure_arm(self.branch, h)

    def psi(self) -> float:
        """Plasticity index for THIS model's band ends."""
        return plasticity_index(_E_STAR(), self.sigma_s_over_beta_a, self.H_ind)

    def psi_regime(self) -> str:
        """GW1966 p. 309, CITED verbatim: < 0.6 elastic, > 1.0 plastic, and
        0.6 to 1.0 a range in which the mode of deformation is 'in doubt'.

        GW's own caveat travels with the numbers: these values hold only for
        the particular surface model considered.
        """
        p = self.psi()
        return "plastic" if p > 1.0 else ("elastic" if p < 0.6
                                          else "undetermined")

    def pressure_arm(self, arm: str, h) -> np.ndarray:
        """Asperity pressure under an EXPLICITLY NAMED arm.

        The only entry point that computes a pressure without consulting psi,
        and it requires the caller to name the arm -- so a doubt-band row's two
        results are always attributable to the arm that produced them.
        """
        if arm not in ("elastic", "plastic"):
            raise ValueError(f"arm must be 'elastic' or 'plastic', got {arm!r}")
        H = self.gap_ratio(h)
        if arm == "elastic":
            return asperity_pressure_elastic(H, self.eta_beta_sigma, _E_STAR(),
                                             self.sigma_s_over_beta_a)
        return asperity_pressure_plastic(H, self.eta_beta_sigma, self.H_ind)

    def area_fraction(self, h) -> np.ndarray:
        """A_c/A_n on the gap field, GT Eq. 8.  Branch-independent."""
        H = np.asarray(h, dtype=float) / self.sigma_summit()
        return math.pi ** 2 * self.eta_beta_sigma ** 2 * F_n(2.0, H)

    def label(self) -> str:
        """One string naming every choice, for the persisted row."""
        return (f"sigma*={self.sigma_star:.4g};nbs={self.eta_beta_sigma:.4g};"
                f"sob={self.sigma_s_over_beta_a:.4g};H_ind={self.H_ind:.4g};"
                f"F3={self.summit_convention};F5={self.branch}")


def _E_STAR() -> float:
    """Hertz reduced modulus, from the CITED pair.  Imported, never restated."""
    from .geometry import composite_modulus
    return composite_modulus()


def registered_contact_models(branch: str,
                              summit_convention: str) -> list["ContactModel"]:
    """Every corner of the registered band cross-product, for one fork choice.

    Standing rule 15: both ends of every band, always.  Returns the full
    cross-product so a caller cannot quietly run one corner -- sixteen models,
    which is box E's registered ``contact_model_traversal``.

    BOTH ARMS ARE NOW REQUIRED POSITIONALLY (box E sec 2.3).  They used to
    default to box C's arm A, so a caller who wrote
    ``registered_contact_models()`` got the arms box D explicitly closed
    AGAINST, while the manifest recorded DERIVED.  Removing the defaults is
    also what lets a signature-derived coverage gate see these as choices at
    all: a default is precisely a choice that does not look like one.
    """
    out = []
    for sig in SIGMA_STAR_BAND_M:
        for nbs in eta_beta_sigma_band():
            for sob in sigma_s_over_beta_a_band():
                for hind in H_ind_band():
                    out.append(ContactModel(
                        sigma_star=sig, eta_beta_sigma=nbs,
                        sigma_s_over_beta_a=sob, H_ind=hind,
                        summit_convention=summit_convention, branch=branch))
    return out



def expand_for_solving(models: list["ContactModel"]) -> list[tuple["ContactModel", str]]:
    """Turn the registered cross-product into the list of solves it implies.

    Box E sec 2.2: a doubt-band row is solved under BOTH arms and both are
    carried.  ``ContactModel.pressure()`` refuses to pick for such a model, so
    the expansion has to happen HERE, once, where it is visible -- rather than
    inside a solver loop where a caller could quietly resolve it.

    Returns ``(model_with_an_explicit_branch, provenance)`` pairs:

        psi < 0.6   -> one solve,  provenance "psi:elastic"
        psi > 1.0   -> one solve,  provenance "psi:plastic"
        in between  -> TWO solves, provenance "doubt:elastic" / "doubt:plastic"

    The provenance string travels into the row, so every doubt-band result is
    attributable to the arm that produced it and no tally can silently collapse
    the doubt band to one number.
    """
    import dataclasses

    out: list[tuple[ContactModel, str]] = []
    for m in models:
        if m.branch != "psi":
            out.append((m, f"fixed:{m.branch}"))
            continue
        regime = m.psi_regime()
        if regime == "undetermined":
            for arm in ("elastic", "plastic"):
                out.append((dataclasses.replace(m, branch=arm), f"doubt:{arm}"))
        else:
            out.append((dataclasses.replace(m, branch=regime), f"psi:{regime}"))
    return out


def soonest_engaging(models: list["ContactModel"]) -> "ContactModel":
    """The model whose asperities engage FIRST -- largest sigma_summit, hence
    smallest H = h/sigma_s at any given film, hence the largest F_5/2.

    Box E sec 2.1's short-circuit rests on this: if THIS model carries no
    asperity load at equilibrium, every other model in the cross-product
    engages later still and carries none either, so all sixteen coincide with
    the pure-film solve.  Exact, not approximate -- but registered as requiring
    verification rather than argument, because a short-circuit that is NEARLY
    exact is the defect class this project keeps finding.
    """
    if not models:
        raise ValueError("no models given")
    return max(models, key=lambda m: m.sigma_summit())

def blocked_entry_points() -> list[str]:
    """Which entry points still refuse, and why.

    After ruling R-10 the Greenwood-Tripp LOAD path is live: every constant it
    needs is a registered band, a declared convention or a class value.  What
    remains is ``boundary_shear``, which is DESCOPED rather than blocked -- a
    different thing, and it stays refusing on purpose.
    """
    out = []
    for name, fn, args in (
            ("asperity_pressure_elastic", asperity_pressure_elastic,
             (1.0, 0.05, 76e9, 5e-4)),
            ("asperity_pressure_plastic", asperity_pressure_plastic,
             (1.0, 0.05, 1.0e9)),
            ("plasticity_index", plasticity_index, (76e9, 5e-4, 1.0e9)),
            ("boundary_shear", boundary_shear, (1.0e6,))):
        try:
            fn(*args)
        except MixedNotVerified as exc:
            out.append(f"{name}: {str(exc).split('.')[0]}")
    return out
