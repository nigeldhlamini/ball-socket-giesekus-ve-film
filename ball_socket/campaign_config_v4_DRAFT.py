"""campaign_config_v4_DRAFT.py -- Chapter 9 ball-socket campaign, BOX B.

DRAFT.  Awaiting Chat's read and Nigel's signature; then hashed and run.
Box A (`campaign_config.py`, frozen, hash 2468949e...41f6) is RETAINED
unchanged and superseded by this configuration, not edited.

RE-POSE (ruling 2026-07-27, ledger L-9).  Box A's D2 normalisation made
``k_F_bs`` an independent axis divided by a squeeze-capacity scale containing no
p_ch, which severed the physical coupling between the applied piston force and
the pocket relief.  Both derive from the SAME chamber pressure:

    F_app    = p_ch * A_p
    F_pocket = p_ch * A_pocket,   A_pocket = pi R_b^2 sin^2(theta_min)
    F_hyd    = p_ch (A_p - A_pocket) > 0   for every sampled point,

with the relief fraction beta = A_pocket / A_p fixed by geometry alone.  Box B
therefore:

  * REMOVES k_F_bs as an input axis and couples the load (2.1);
  * PROMOTES theta_min to the eighth axis (2.2), because theta_min is the
    physical control on beta -- the true analogue of the slipper's balance
    ratio, which k_F measured by dividing by the hydrostatic floor.  It is a
    design variable, not a normalisation.  Range widened to [20, 40] deg
    accordingly, and beta reported as a derived quantity;
  * keeps the dimensionality at 8-D, all other axes unchanged including the
    pre-freeze eta_0 and omega extensions (2.4).

ACKNOWLEDGED OMISSION (2.7): inertial and centrifugal contributions to F_app
are neglected; F_app is the quasi-static piston force.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import BallSocketGeometry, SocketMaterial, BALL_STEEL, delta_alpha_th
from .outcomes import Outcome
from . import representative as rep


@dataclass(frozen=True)
class InputDimension:
    name: str
    lower: float
    upper: float
    scale: str
    units: str
    note: str = ""

    def from_unit(self, u):
        if self.scale == "log":
            return self.lower * (self.upper / self.lower) ** u
        return self.lower + (self.upper - self.lower) * u


INPUT_SPACE: tuple[InputDimension, ...] = (
    InputDimension("omega", 50.0, 475.0, "lin", "rad/s",
                   "rated envelope; pre-freeze extension 470 -> 475 carried from box A"),
    InputDimension("p_ch", 2.0e6, 5.0e7, "lin", "Pa",
                   "NOW SETS THE LOAD: F_app = p_ch * A_p.  Floor raised "
                   "2.0e5 -> 2.0e6 (20 bar): this is a CLOSED-CIRCUIT pump, so "
                   "the chamber never sees 2 bar -- during suction the "
                   "low-pressure side is held at charge/boost pressure "
                   "(~20-30 bar), and a 2 bar floor samples states the machine "
                   "does not occupy.  Scale log -> LINEAR: load enters the "
                   "constraints through the linearly-acting F_app = p_ch*A_p on "
                   "a bounded envelope, so uniform coverage in p_ch is the "
                   "natural choice; log scaling earns its place for "
                   "order-of-magnitude parameters with multiplicative response "
                   "(eta_0, lambda), which this is not"),
    InputDimension("c0", 2.0e-6, 30.0e-6, "lin", "m", "radial cold clearance"),
    InputDimension("T_in", 20.0, 110.0, "lin", "degC",
                   "per-fluid-class g2 ceilings still cap water-glycol"),
    InputDimension("eta_0", 4.0e-3, 2.0e-1, "log", "Pa.s",
                   "pre-freeze extension carried from box A; strict SUPERSET of "
                   "the slipper axis -- see COMMON_SUBBOX_WITH_SLIPPER"),
    InputDimension("lam", 1.0e-5, 1.0e-2, "log", "s", "as slipper INPUT_SPACE"),
    InputDimension("eta_s_ratio", 0.05, 0.95, "lin", "-", "as slipper INPUT_SPACE"),
    # EIGHTH AXIS (replaces k_F_bs)
    InputDimension("theta_min_deg", 20.0, 40.0, "lin", "deg",
                   "feed-pocket edge; the physical control on the relief "
                   "fraction beta = A_pocket/A_p.  Widened from the "
                   "representative [25, 36] because it is now a design axis, "
                   "not a geometric uncertainty"),
)

ALPHA_FIXED = 0.25
SQUEEZE_PHASE = 1.0
SPIN_RATIO = 1.0
P_CASE_PA = 1.0e5

GEOMETRY = BallSocketGeometry(c0=float("nan"), R_b=rep.R_B_M,
                              theta_max=np.deg2rad(rep.THETA_MAX_DEG))
A_P_M2 = rep.A_P_M2

SIGMA_STAR_BAND_M = rep.sigma_star_band()          # wide band (ruling A1)
HMIN_FLOOR_BAND_M = rep.hmin_floor_band(3.0)
FLUID_CLASSES = ("mineral", "water_glycol")
COMMON_SUBBOX_WITH_SLIPPER = {"eta_0": (5.0e-3, 2.0e-1)}

SOCKET = SocketMaterial()
DELTA_ALPHA_TH = delta_alpha_th()

# E5 fluid triplet -- REGISTERED VALUES in v4 (box A carried these as a dated
# gap-fill note; a fresh registration should not leave the gap open twice).
E5_ETA_S_RATIO = 0.95          # mineral oil is mostly solvent
E5_LAMBDA = 1.0e-5             # axis floor


@dataclass(frozen=True)
class ConstraintSet:
    lambda_ratio_min: float = 3.0
    T_max_C_by_class: dict = field(default_factory=lambda: {
        "mineral": 110.0, "water_glycol": 60.0})
    De_sq_max: float = 0.2

    def evaluate(self, perf, fluid_class: str, sigma_star: float) -> Outcome:
        g1 = (perf.h_min / sigma_star) >= self.lambda_ratio_min
        g2 = perf.T_max <= self.T_max_C_by_class[fluid_class]
        g3 = perf.De_sq <= self.De_sq_max
        return Outcome.FEASIBLE if (g1 and g2 and g3) else Outcome.INFEASIBLE


# OAT block loses theta_min (now an axis), retains the rest (2.3)
OAT_BLOCK = {
    "spin_ratio": (0.0, 0.5, 1.0),
    "theta_max_deg": rep.THETA_MAX_OAT_DEG,
    "alpha_th_rel": (0.9, 1.0, 1.1),
}
OAT_N = 32

N_AUDIT = 2 ** 13              # 8192 (statistical power; does not touch the box)
N_BO = 500
N_INIT = 18
N_SEEDS = 5

DRAFT = False                  # FROZEN 2026-07-27 on Nigel's signature
FROZEN_ON = "2026-07-27"
SIGNED_BY = "Nigel C. Dhlamini"


def relief_fraction(theta_min_deg: float, R_b: float = rep.R_B_M,
                    A_p: float = rep.A_P_M2) -> float:
    """beta = A_pocket / A_p = pi R_b^2 sin^2(theta_min) / A_p (derived)."""
    return rep.pocket_relief_area(R_b, np.deg2rad(theta_min_deg)) / A_p


def applied_and_hydrodynamic_load(p_ch: float, theta_min_deg: float):
    """(F_app, F_pocket, F_hyd) in N.  F_hyd > 0 by construction."""
    beta = relief_fraction(theta_min_deg)
    F_app = p_ch * rep.A_P_M2
    return F_app, beta * F_app, F_app * (1.0 - beta)


def config_dict() -> dict:
    return {
        "DRAFT": DRAFT, "box": "B",
        "supersedes": "campaign_config.py (box A, sha 2468949e...41f6)",
        "load_model": "F_app = p_ch * A_p (coupled); "
                      "F_pocket = p_ch * A_pocket; F_hyd > 0 by construction",
        "acknowledged_omission": "inertial and centrifugal contributions to "
                                 "F_app neglected (quasi-static piston force)",
        "input_space": [d.__dict__ for d in INPUT_SPACE],
        "alpha_fixed": ALPHA_FIXED, "squeeze_phase": SQUEEZE_PHASE,
        "spin_ratio": SPIN_RATIO, "p_case_Pa": P_CASE_PA,
        "geometry": {"R_b_m": rep.R_B_M, "A_p_m2": rep.A_P_M2,
                     "theta_max_deg": rep.THETA_MAX_DEG,
                     "theta_min_axis_deg": [20.0, 40.0],
                     "domain": "annular fed film"},
        "materials": {"socket": "leaded tin bronze CuSn12Pb class (CITED)",
                      "socket_E_Pa": SOCKET.E, "socket_nu": SOCKET.nu,
                      "socket_alpha_th": SOCKET.alpha_th,
                      "ball": "hardened bearing steel 100Cr6 class (CITED)",
                      "ball_constants": BALL_STEEL,
                      "delta_alpha_th": DELTA_ALPHA_TH},
        "sigma_star_band_m": list(SIGMA_STAR_BAND_M),
        "hmin_floor_band_m": list(HMIN_FLOOR_BAND_M),
        "fluid_classes": list(FLUID_CLASSES),
        "common_subbox_with_slipper": {k: list(v) for k, v
                                       in COMMON_SUBBOX_WITH_SLIPPER.items()},
        "constraints": {"lambda_ratio_min": 3.0,
                        "T_max_C": {"mineral": 110.0, "water_glycol": 60.0},
                        "De_sq_max": 0.2},
        "e5_fluid_triplet": {"eta_s_ratio": E5_ETA_S_RATIO, "lam": E5_LAMBDA},
        "oat_block": {k: list(v) for k, v in OAT_BLOCK.items()}, "oat_n": OAT_N,
        "N_audit": N_AUDIT, "N_bo": N_BO, "N_init": N_INIT, "N_seeds": N_SEEDS,
        "analysis_convention": (
            "Binding splits conditional on convergence; non-convergence "
            "(blow-up) fraction reported alongside in every table; blow-up "
            "treated as the limiting case of film failure, not missing data; "
            "infrastructure error rows excluded from all label statistics and "
            "reported separately."),
        "FROZEN_ON": FROZEN_ON, "SIGNED_BY": SIGNED_BY,
    }


def freeze_and_hash(path: str | None = None) -> str:
    import hashlib, json, os
    if DRAFT:
        raise RuntimeError("v4 is still DRAFT; refusing to freeze")
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "campaign_frozen_v4.json")
    cfg = config_dict()
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    digest = hashlib.sha256(blob).hexdigest()
    with open(path, "w") as fh:
        json.dump({"sha256": digest, "config": cfg}, fh, indent=2,
                  sort_keys=True, default=str)
    return digest
