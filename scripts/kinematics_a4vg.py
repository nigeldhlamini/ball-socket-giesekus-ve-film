#!/usr/bin/env python
r"""A4VG piston kinematics and applied load, report Sec. 3.

Implements report Eqs. (x), (xdot), (xddot), (Fapp) and (V):

    x(phi)      = R_p tan(beta) (1 - cos phi)
    xdot(phi)   = R_p omega tan(beta) sin phi
    xddot(phi)  = R_p omega^2 tan(beta) cos phi
    F_app(phi)  = p_ch(phi) A_p + m_eff xddot(phi)      (F_mu neglected)
    V(theta)    = omega_b R_b sin(theta)

FRAME CONSTANTS ARE NOT KNOWN.  Every drawing-derived value in ``FRAME`` is
NaN and flagged VERIFY.  Nothing here invents one.  The functions are written
so that filling FRAME is the only step needed, and every entry point that
would return a number built on an unread constant raises instead.

THE DISPLACEMENT CHECK IS AN ASSERTION.  Nine pistons at beta_max must
reproduce the nominal 45 cm^3/rev:

    V_geo = n_pistons * A_p * 2 R_p tan(beta_max)

It is how we know R_p and beta_max were read correctly, so it fails loudly and
is not downgraded to a warning.  While R_p and beta_max are unread the check
cannot run; what CAN be stated without them is the stroke the identity
requires, and ``required_stroke_m`` returns it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NAN = float("nan")

#: Drawing-derived constants.  Every value is NaN until read from the pump
#: drawings, which are NOT reproduced in this repository; only the scalar
#: values, once read, belong here.
FRAME: dict[str, dict] = {
    "R_p":        {"value": NAN, "units": "m",   "status": "VERIFY",
                   "what": "piston pitch radius"},
    "beta_max":   {"value": NAN, "units": "rad", "status": "VERIFY",
                   "what": "maximum swashplate angle"},
    "m_eff":      {"value": NAN, "units": "kg",  "status": "VERIFY",
                   "what": "reciprocating mass, piston plus pad"},
    "A_p":        {"value": NAN, "units": "m^2", "status": "VERIFY",
                   "what": "piston area; 2.0e-4 is the representative value "
                           "used by the lubrication campaign, but the drawing "
                           "value is what belongs here"},
    "p_boost":    {"value": NAN, "units": "Pa",  "status": "VERIFY",
                   "what": "charge/boost pressure; report says 20 to 30 bar "
                           "for a closed circuit, not atmospheric"},
    "port_timing": {"value": NAN, "units": "rad", "status": "VERIFY",
                    "what": "angular duration of the commutation transitions "
                            "at the dead centres"},
    "n_pistons":  {"value": 9,   "units": "-",   "status": "REPORT",
                   "what": "stated in the report text"},
    "V_nominal":  {"value": 45e-6, "units": "m^3/rev", "status": "REPORT",
                   "what": "nominal displacement, stated in the report text"},
}

#: Tolerance on the displacement identity.  A geometric identity between three
#: drawing scalars should close far tighter than this; the allowance is for
#: rounding in how the drawings state R_p and beta_max, not for slack.
DISPLACEMENT_RTOL = 0.02


class FrameNotVerified(RuntimeError):
    """Raised when a computation needs a FRAME constant that is still NaN."""


def _get(name: str) -> float:
    entry = FRAME[name]
    v = float(entry["value"])
    if not math.isfinite(v):
        raise FrameNotVerified(
            f"FRAME['{name}'] is {entry['status']}: {entry['what']}.  Read it "
            f"from the drawings and set it before calling this.")
    return v


def unverified() -> list[str]:
    """Names of FRAME entries still carrying a NaN."""
    return sorted(k for k, e in FRAME.items()
                  if not math.isfinite(float(e["value"])))


# ---------------------------------------------------------------------------
# piston motion, report Eqs. (x), (xdot), (xddot)
# ---------------------------------------------------------------------------

def x(phi, beta: float | None = None) -> np.ndarray:
    R_p = _get("R_p")
    b = _get("beta_max") if beta is None else beta
    return R_p * math.tan(b) * (1.0 - np.cos(phi))


def xdot(phi, omega: float, beta: float | None = None) -> np.ndarray:
    R_p = _get("R_p")
    b = _get("beta_max") if beta is None else beta
    return R_p * omega * math.tan(b) * np.sin(phi)


def xddot(phi, omega: float, beta: float | None = None) -> np.ndarray:
    R_p = _get("R_p")
    b = _get("beta_max") if beta is None else beta
    return R_p * omega ** 2 * math.tan(b) * np.cos(phi)


def stroke(beta: float | None = None) -> float:
    """2 R_p tan(beta): the axial stroke."""
    R_p = _get("R_p")
    b = _get("beta_max") if beta is None else beta
    return 2.0 * R_p * math.tan(b)


# ---------------------------------------------------------------------------
# the displacement check (prompt 3.1) -- an ASSERTION
# ---------------------------------------------------------------------------

def required_stroke_m() -> float:
    """The stroke the displacement identity forces, given n_pistons, A_p and
    the nominal displacement.

    Computable without R_p or beta_max, and therefore the one quantitative
    statement available before the drawings are read: whatever R_p and
    beta_max turn out to be, 2 R_p tan(beta_max) must equal this.
    """
    n = float(FRAME["n_pistons"]["value"])
    V = float(FRAME["V_nominal"]["value"])
    A_p = FRAME["A_p"]["value"]
    if not math.isfinite(float(A_p)):
        # A_p is itself VERIFY; fall back to the representative value and say so.
        A_p = 2.0e-4
    return V / (n * float(A_p))


def check_displacement(*, strict: bool = True) -> dict:
    """Nine pistons at beta_max must reproduce the nominal 45 cm^3/rev.

    This is how we know R_p and beta_max were read correctly.  It raises on
    failure; it does not warn.
    """
    missing = [k for k in ("R_p", "beta_max", "A_p") if k in unverified()]
    if missing:
        raise FrameNotVerified(
            "the displacement check cannot run while "
            f"{', '.join(missing)} are unread.  It is the check that confirms "
            "R_p and beta_max, so it must not be skipped: fill FRAME and "
            "re-run.  Required stroke 2 R_p tan(beta_max) = "
            f"{required_stroke_m()*1e3:.3f} mm "
            f"({'representative A_p' if 'A_p' in missing else 'drawing A_p'}).")

    n = float(FRAME["n_pistons"]["value"])
    V_nom = float(FRAME["V_nominal"]["value"])
    V_geo = n * _get("A_p") * stroke()
    rel = abs(V_geo - V_nom) / V_nom
    result = {"V_geometric_m3_per_rev": V_geo, "V_nominal_m3_per_rev": V_nom,
              "relative_error": rel, "tolerance": DISPLACEMENT_RTOL,
              "closes": bool(rel <= DISPLACEMENT_RTOL),
              "stroke_m": stroke()}
    if strict and not result["closes"]:
        raise AssertionError(
            f"displacement check FAILED: nine pistons at beta_max give "
            f"{V_geo*1e6:.3f} cm^3/rev against a nominal {V_nom*1e6:.3f}, a "
            f"relative error of {rel:.3%} against a tolerance of "
            f"{DISPLACEMENT_RTOL:.1%}.  R_p or beta_max has been misread from "
            f"the drawings; do not use any number downstream of this.")
    return result


# ---------------------------------------------------------------------------
# chamber pressure and applied load, report Eq. (Fapp)
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    """One test condition.  p_d and omega are operating settings, not drawing
    constants, so they are known; beta is the commanded swash angle."""
    name: str
    p_d: float           # discharge pressure [Pa]
    rpm: float           # shaft speed [min^-1]
    beta: float          # swash angle [rad]
    note: str = ""

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * self.rpm / 60.0


def p_ch(phi, cond: Condition) -> np.ndarray:
    """Square-wave commutation with linear transitions of width `port_timing`
    at the dead centres: discharge over the delivery stroke, boost over the
    suction stroke."""
    p_b = _get("p_boost")
    tau = _get("port_timing")
    phi = np.mod(np.asarray(phi, float), 2.0 * math.pi)
    out = np.where(phi < math.pi, cond.p_d, p_b).astype(float)
    # transitions centred on the dead centres phi = 0 and phi = pi
    for centre, lo, hi in ((0.0, p_b, cond.p_d), (math.pi, cond.p_d, p_b)):
        d = np.abs(np.angle(np.exp(1j * (phi - centre))))
        band = d < tau / 2.0
        frac = 0.5 + (phi[band] - centre) / tau if centre == 0.0 else \
            0.5 + (phi[band] - centre) / tau
        out[band] = lo + (hi - lo) * np.clip(frac, 0.0, 1.0)
    return out


def F_app(phi, cond: Condition) -> dict:
    """Report Eq. (Fapp), with the inertial term returned separately so its
    contribution to F_max and F_min is visible rather than buried in a sum."""
    A_p = _get("A_p")
    m_eff = _get("m_eff")
    phi = np.asarray(phi, float)
    pressure_term = p_ch(phi, cond) * A_p
    inertial_term = m_eff * xddot(phi, cond.omega, cond.beta)
    return {"phi": phi, "pressure_term_N": pressure_term,
            "inertial_term_N": inertial_term,
            "F_app_N": pressure_term + inertial_term}


def inertia_fractions(cond: Condition, n: int = 2048) -> dict:
    """Prompt 3.3: the inertial fraction at F_max and at F_min.

    The report asserts of order one per cent at the peak and a much larger
    fraction at the trough.  Both are returned so the assertion can be checked
    rather than repeated.
    """
    phi = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    r = F_app(phi, cond)
    F, I = r["F_app_N"], r["inertial_term_N"]
    i_max, i_min = int(np.argmax(F)), int(np.argmin(F))
    return {
        "condition": cond.name,
        "F_max_N": float(F[i_max]), "phi_at_F_max_rad": float(phi[i_max]),
        "inertia_fraction_at_F_max": float(abs(I[i_max]) / abs(F[i_max])),
        "F_min_N": float(F[i_min]), "phi_at_F_min_rad": float(phi[i_min]),
        "inertia_fraction_at_F_min": float(abs(I[i_min]) / abs(F[i_min]))
        if F[i_min] != 0 else float("inf"),
        "load_ratio_R": float(F[i_min] / F[i_max]) if F[i_max] else float("nan"),
    }


# ---------------------------------------------------------------------------
# surface speed, report Eq. (V)
# ---------------------------------------------------------------------------

def V_surface(theta, omega_b: float, R_b: float = 6.5e-3) -> np.ndarray:
    """Report Eq. (V): V(theta) = omega_b R_b sin(theta)."""
    return omega_b * R_b * np.sin(np.asarray(theta, float))


# ---------------------------------------------------------------------------

def main() -> int:
    print("FRAME status")
    for k, e in FRAME.items():
        v = float(e["value"])
        shown = "NaN" if not math.isfinite(v) else f"{v:g}"
        print(f"  {k:<12} {shown:>10}  {e['units']:<8} {e['status']}")
    miss = unverified()
    print(f"\nunverified: {', '.join(miss) if miss else 'none'}")
    print(f"\nrequired stroke from the displacement identity: "
          f"{required_stroke_m()*1e3:.3f} mm")
    print("  (V_nominal / (n_pistons * A_p); A_p is the representative "
          "2.0e-4 m^2 until the drawing value is read)")
    try:
        print(check_displacement())
    except FrameNotVerified as exc:
        print(f"\ndisplacement check BLOCKED: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
