"""E5 rated-corner point test -- specification and viscosity interpolation.

RULING 2026-07-26: E5 is re-based from the (proprietary) partner test programme
onto the PUBLIC rated-corner envelope:
  * nominal corner  450 bar @ 4300 min^-1
  * maximum corner  500 bar @ 4500 min^-1
  (public datasheet RE 92004; Nigel confirms edition)
Fluids: ISO VG32 at 100 C and ISO VG46 at 110 C, with eta(T) from the ASTM D341
(Walther) interpolation of the public 40/100 C grade viscosities.
Clearances: c0 in {2, 5, 15, 30} um.  Total 2 x 2 x 4 = 16 solves.

This module COMPUTES AND RECORDS the interpolation and the box-coverage check
for the pre-registration.  It does NOT run the solves (Phase 3+ is unlicensed;
running them before freeze would pre-empt a pre-registered result).

ASSUMPTION FLAGGED FOR NIGEL: ISO 3448 fixes only the 40 C viscosity of a VG
grade.  The 100 C values below are the typical datasheet figures for a VI ~ 100
solvent-refined mineral hydraulic oil; they must be confirmed against the
specific product datasheet before freeze, since eta(T) at the corner sets
whether the point lies inside the eta_0 axis.
"""
from __future__ import annotations

import numpy as np

# --- public grade data (ISO 3448 at 40 C; 100 C typical, VI ~ 100) ---------
GRADES = {
    "ISO VG32": {"nu40_cSt": 32.0, "nu100_cSt": 5.4},   # nu100 ASSUMPTION
    "ISO VG46": {"nu40_cSt": 46.0, "nu100_cSt": 6.8},   # nu100 ASSUMPTION
}
RHO15 = 870.0          # kg/m^3, typical mineral hydraulic oil
DRHO_DT = -0.65        # kg/(m^3 K)

# rated corners (public envelope)
CORNERS = {
    "nominal": {"p_bar": 450.0, "rpm": 4300.0},
    "maximum": {"p_bar": 500.0, "rpm": 4500.0},
}
FLUID_AT = {"ISO VG32": 100.0, "ISO VG46": 110.0}       # service temperature [C]
C0_LIST_UM = (2.0, 5.0, 15.0, 30.0)


def walther_coeffs(nu1_cSt: float, T1_C: float, nu2_cSt: float, T2_C: float):
    """ASTM D341 / Walther: log10(log10(nu + 0.7)) = A - B log10(T_K)."""
    Z = lambda nu: np.log10(np.log10(nu + 0.7))
    t1, t2 = np.log10(T1_C + 273.15), np.log10(T2_C + 273.15)
    B = (Z(nu1_cSt) - Z(nu2_cSt)) / (t2 - t1)
    A = Z(nu1_cSt) + B * t1
    return float(A), float(B)


def nu_at(nu40: float, nu100: float, T_C: float) -> float:
    """Kinematic viscosity [cSt] at T_C by the Walther interpolation."""
    A, B = walther_coeffs(nu40, 40.0, nu100, 100.0)
    Zt = A - B * np.log10(T_C + 273.15)
    return float(10.0 ** (10.0 ** Zt) - 0.7)


def eta_at(nu40: float, nu100: float, T_C: float) -> float:
    """Dynamic viscosity [Pa s] = nu * rho, with a linear density-temperature law."""
    rho = RHO15 + DRHO_DT * (T_C - 15.0)
    return nu_at(nu40, nu100, T_C) * 1e-6 * rho


def e5_specification() -> dict:
    """Full E5 point-set with the interpolation recorded and the box-coverage
    check against the campaign axes."""
    from .campaign_config_DRAFT import INPUT_SPACE

    axes = {d.name: (d.lower, d.upper) for d in INPUT_SPACE}
    fluids = {}
    for name, g in GRADES.items():
        T = FLUID_AT[name]
        A, B = walther_coeffs(g["nu40_cSt"], 40.0, g["nu100_cSt"], 100.0)
        nu = nu_at(g["nu40_cSt"], g["nu100_cSt"], T)
        eta = eta_at(g["nu40_cSt"], g["nu100_cSt"], T)
        lo, hi = axes["eta_0"]
        fluids[name] = {
            "service_T_C": T, "walther_A": A, "walther_B": B,
            "nu40_cSt": g["nu40_cSt"], "nu100_cSt": g["nu100_cSt"],
            "nu_at_T_cSt": nu, "rho_at_T": RHO15 + DRHO_DT * (T - 15.0),
            "eta_at_T_Pa_s": eta,
            "inside_eta0_axis": bool(lo <= eta <= hi),
            "eta0_axis": [lo, hi],
        }

    corners = {}
    for cname, c in CORNERS.items():
        omega = c["rpm"] * 2.0 * np.pi / 60.0
        p_pa = c["p_bar"] * 1e5
        corners[cname] = {
            "p_ch_Pa": p_pa, "omega_rad_s": omega, "rpm": c["rpm"],
            "inside_p_ch_axis": bool(axes["p_ch"][0] <= p_pa <= axes["p_ch"][1]),
            "inside_omega_axis": bool(axes["omega"][0] <= omega <= axes["omega"][1]),
        }

    coverage = []
    for cname, cc in corners.items():
        if not cc["inside_omega_axis"]:
            coverage.append(f"{cname} corner omega = {cc['omega_rad_s']:.1f} rad/s "
                            f"outside axis {axes['omega']}")
        if not cc["inside_p_ch_axis"]:
            coverage.append(f"{cname} corner p_ch = {cc['p_ch_Pa']:.2e} Pa "
                            f"outside axis {axes['p_ch']}")
    for fname, ff in fluids.items():
        if not ff["inside_eta0_axis"]:
            coverage.append(f"{fname} at {ff['service_T_C']:.0f} C gives eta_0 = "
                            f"{ff['eta_at_T_Pa_s']:.3e} Pa s outside axis "
                            f"{ff['eta0_axis']}")

    return {"fluids": fluids, "corners": corners,
            "c0_list_um": list(C0_LIST_UM),
            "n_solves": len(CORNERS) * len(GRADES) * len(C0_LIST_UM),
            "box_coverage_exceptions": coverage}


if __name__ == "__main__":
    import json
    spec = e5_specification()
    print(json.dumps(spec, indent=2))
