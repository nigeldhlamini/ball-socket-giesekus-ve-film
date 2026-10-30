"""e5_fluid_spec.py
================
Public-source fluid specification for the E5 point test (Chapter 9
ball-socket campaign).

Grade viscosities at 40 C come from ISO 3448 (public, cited).  The
temperature dependence is ASTM D341 (Walther), implemented and verified
here.  The 100 C viscosity is **swept**, not derived from a viscosity
index.

WHY THE VISCOSITY INDEX WAS REMOVED (ruling R2, 2026-07-27).  The VI was
never load-bearing: what E5 needs is a thin arm and a genuinely thick arm
at service temperature, which a direct sweep of nu_100 supplies without
any assumed grade property.  Inverting ASTM D2270 would additionally have
required its L/H tables, which are paywalled and copyrighted, so
reproducing them in a published thesis is not acceptable in any case.
The D2270 machinery (L_and_H, viscosity_index, nu_100_from_VI and the
edition requirement) is therefore deleted rather than left blocked.

Severity note (pre-registered): E5 predicts the rated-corner operating
point lies OUTSIDE the full-film feasible set with g1 binding.  Thinner
oil makes that easier to confirm, so the THIN end of the sweep is the
anti-severe case and the THICK end is severity-determining.  The reported
result is the E5 verdict as a FUNCTION of nu_100, identifying the
threshold hot viscosity above which E5 is refuted; no single assumed
viscosity is load-bearing anywhere.

Author of record: N. C. Dhlamini (dhlnig001@myuct.ac.za), CERECAM, UCT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- ISO 3448 kinematic viscosity at 40 C, mm^2/s (public standard) -------
ISO_VG = {32: 32.0, 46: 46.0}

# --- swept 100 C viscosity (mm^2/s): thin arm -> thick arm ----------------
NU100_SWEEP = tuple(np.linspace(4.5, 9.0, 10))
NU100_THICK_ARM_FROM = 6.75      # upper half: the severity-determining region


# ===========================================================================
# ASTM D341 (Walther) -- fully implemented and verified
# ===========================================================================

def _walther_Z(nu: float) -> float:
    """Z = log10(log10(nu + 0.7)), nu in mm^2/s."""
    return np.log10(np.log10(nu + 0.7))


def walther_fit(nu_40: float, nu_100: float) -> tuple[float, float]:
    """Fit Z = A - B log10(T_K) through the 40 C and 100 C points."""
    T1, T2 = 313.15, 373.15
    Z1, Z2 = _walther_Z(nu_40), _walther_Z(nu_100)
    B = (Z1 - Z2) / (np.log10(T2) - np.log10(T1))
    A = Z1 + B * np.log10(T1)
    return A, B


def walther_nu(A: float, B: float, T_C: float) -> float:
    """Kinematic viscosity [mm^2/s] at T_C from a fitted (A, B)."""
    Z = A - B * np.log10(T_C + 273.15)
    return 10.0 ** (10.0 ** Z) - 0.7


def density(T_C: float, rho15: float = 870.0, k: float = 0.65) -> float:
    """Linear mineral-oil density model [kg/m^3]; campaign convention."""
    return rho15 - k * (T_C - 15.0)


def eta_at(nu_40: float, nu_100: float, T_C: float) -> float:
    """Dynamic viscosity [Pa s] at T_C for a grade pinned at (nu_40, nu_100)."""
    A, B = walther_fit(nu_40, nu_100)
    return walther_nu(A, B, T_C) * 1e-6 * density(T_C)


# ===========================================================================
# E5 point set
# ===========================================================================

@dataclass(frozen=True)
class E5Point:
    grade: int                  # 32 or 46
    nu_100: float               # swept [mm^2/s]
    T_C: float                  # service temperature
    c0_um: float
    p_ch_bar: float
    omega_rad_s: float
    eta0: float                 # Pa s, derived via D341
    corner: str                 # "maximum" (registered) or "nominal"
    thick_arm: bool             # True in the severity-determining region


# Public rated corners (manufacturer's published datasheet for the 45 cm^3
# frame; edition string confirmed at proof stage, see
# reports/proof_stage_checklist.md).  The maximum corner is the
# pre-registered E5 prediction point; the nominal corner is exploratory.
CORNERS = {
    "maximum": dict(p_ch_bar=500.0, rpm=4500.0, registered=True),
    "nominal": dict(p_ch_bar=450.0, rpm=4300.0, registered=False),
}

SERVICE_T = {32: 100.0, 46: 110.0}
CLEARANCES_UM = (2.0, 5.0, 15.0, 30.0)


def build_e5_points(clearances=CLEARANCES_UM) -> list[E5Point]:
    """2 grades x 10 nu_100 x len(clearances) x 2 corners (= 160 by default)."""
    pts: list[E5Point] = []
    for grade, nu40 in ISO_VG.items():
        T = SERVICE_T[grade]
        for nu100 in NU100_SWEEP:
            eta0 = eta_at(nu40, float(nu100), T)
            for cname, corner in CORNERS.items():
                for c0 in clearances:
                    pts.append(E5Point(
                        grade=grade, nu_100=float(nu100), T_C=T, c0_um=c0,
                        p_ch_bar=corner["p_ch_bar"],
                        omega_rad_s=corner["rpm"] * 2 * np.pi / 60.0,
                        eta0=eta0, corner=cname,
                        thick_arm=bool(nu100 >= NU100_THICK_ARM_FROM)))
    return pts


# ===========================================================================
# Box coverage
# ===========================================================================

# Post-ruling axes (pre-freeze extension, documented in prereg sec. 0).
AXES = dict(eta0=(4.0e-3, 2.0e-1), omega=(50.0, 475.0),
            c0_um=(2.0, 30.0), p_ch_Pa=(2.0e5, 5.0e7))

# Comparability convention: cross-interface AR comparison with the slipper
# is made ONLY on the common sub-box below.
SLIPPER_COMMON_SUBBOX = dict(eta0=(5.0e-3, 2.0e-1))


def coverage_report(points: list[E5Point]) -> dict:
    """Flag any E5 point outside the campaign box.  Points outside are still
    evaluated, but their results are labelled an EXTRAPOLATION of the
    constructed region in the prereg and in Ch. 9 prose."""
    out = []
    for p in points:
        flags = []
        if not AXES["eta0"][0] <= p.eta0 <= AXES["eta0"][1]:
            flags.append(f"eta0 {p.eta0:.3e} outside {AXES['eta0']}")
        if not AXES["omega"][0] <= p.omega_rad_s <= AXES["omega"][1]:
            flags.append(f"omega {p.omega_rad_s:.1f} outside {AXES['omega']}")
        if flags:
            out.append(dict(grade=p.grade, nu_100=round(p.nu_100, 3),
                            c0=p.c0_um, corner=p.corner, flags=flags))
    return dict(n_points=len(points), n_outside=len(out), detail=out)


def eta0_table() -> list[dict]:
    """eta_0 at service temperature across the sweep, per grade (for the
    pre-registration record)."""
    rows = []
    for grade, nu40 in ISO_VG.items():
        T = SERVICE_T[grade]
        for nu100 in NU100_SWEEP:
            rows.append(dict(grade=grade, T_C=T, nu_100=round(float(nu100), 3),
                             eta0=eta_at(nu40, float(nu100), T),
                             thick_arm=bool(nu100 >= NU100_THICK_ARM_FROM)))
    return rows


if __name__ == "__main__":
    # D341 verification against the independently computed draft values.
    A, B = walther_fit(32.0, 5.4)
    print(f"D341 check: VG32 with nu_100=5.4 -> A={A:.4f}, B={B:.5f} "
          f"(expected 9.5308, 3.74658)")
    print(f"            eta(100 C) = {eta_at(32.0, 5.4, 100.0):.3e} Pa s "
          f"(expected 4.400e-3)\n")

    print("eta_0 at service temperature across the swept nu_100:")
    for r in eta0_table():
        mark = "  <- thick arm" if r["thick_arm"] else ""
        print(f"  VG{r['grade']} @ {r['T_C']:.0f} C  nu_100={r['nu_100']:5.2f} "
              f"-> eta_0 = {r['eta0']:.3e} Pa s{mark}")

    pts = build_e5_points()
    print(f"\nE5 point set: {len(pts)} solves "
          f"(2 grades x {len(NU100_SWEEP)} nu_100 x {len(CLEARANCES_UM)} c0 x 2 corners)")
    cov = coverage_report(pts)
    print(f"box coverage: {cov['n_outside']} of {cov['n_points']} outside the axes")
    for d in cov["detail"][:6]:
        print("   ", d)
