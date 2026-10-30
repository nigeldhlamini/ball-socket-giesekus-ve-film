"""campaign_config.py -- Chapter 9 ball-socket campaign configuration (FROZEN).

FROZEN 2026-07-27 on Nigel Dhlamini's signature of the pre-registration
(`preregistration_FROZEN.md`).  ``DRAFT = False``; ``freeze_and_hash`` writes
``campaign_frozen.json`` and returns its SHA256, which is stamped into the
frozen pre-registration header and ledger entry L-7.

IMMUTABLE.  Any change to a registered axis, threshold, convention or STOP
condition after this point requires a NEW dated ledger entry stating what
changed, why, and that it post-dates the freeze.  Amendments are permitted;
silent edits are not.

Predecessor retained: ``campaign_config_DRAFT.py``.

RULING 2026-07-26 (thesis material track): the thesis models the
literature-standard steel-on-bronze interface with published, citable
properties, on a REPRESENTATIVE geometry decoupled from the drawings.  This
module imports ``ball_socket.representative`` ONLY -- never
``geometry_provenance`` (industry-report track).  Material identity enters the
campaign through exactly two channels: ``alpha_th`` (thermal clearance) and
``sigma*`` (g1), both cited/banded.

Earlier rulings retained: D1 axes as tabled; D2 k_F_bs = F_app / W_sq_ref with
axis [0.1, 30] log (the 3*pi prefactor is a convention constant); g1 Lambda >= 3;
g2 per-fluid-class (mineral 110 C / water-glycol 60 C); g3 De_sq <= 0.2;
alpha = 0.25, squeeze_phase = 1, spin_ratio = 1 (with the mandatory OAT
sensitivity); D4 N_bo = 500.  Fluid triplet copies the frozen slipper INPUT_SPACE.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import BallSocketGeometry, SocketMaterial, BALL_STEEL, delta_alpha_th
from .solver import squeeze_capacity_scale
from .outcomes import Outcome
from . import representative as rep


# ===========================================================================
# Input space (8-D)
# ===========================================================================

@dataclass(frozen=True)
class InputDimension:
    name: str
    lower: float
    upper: float
    scale: str                  # "log" or "lin"
    units: str
    note: str = ""

    def from_unit(self, u):
        if self.scale == "log":
            return self.lower * (self.upper / self.lower) ** u
        return self.lower + (self.upper - self.lower) * u


INPUT_SPACE: tuple[InputDimension, ...] = (
    InputDimension("omega", 50.0, 475.0, "lin", "rad/s",
                   "RULING A2(i) pre-freeze extension 470 -> 475: the rated maximum "
                   "corner (4500 min^-1 = 471.2 rad/s) must lie inside a campaign "
                   "intended to cover rated operation"),
    InputDimension("p_ch", 2.0e5, 5.0e7, "log", "Pa",
                   "to the rated maximum corner (500 bar)"),
    InputDimension("c0", 2.0e-6, 30.0e-6, "lin", "m",
                   "radial cold clearance; 2 um floor avoids the degenerate limit"),
    InputDimension("T_in", 20.0, 110.0, "lin", "degC",
                   "per-fluid-class g2 still caps water-glycol entries"),
    InputDimension("eta_0", 4.0e-3, 2.0e-1, "log", "Pa.s",
                   "RULING A2(i) pre-freeze extension 5.0e-3 -> 4.0e-3: hot mineral "
                   "oil at rated temperature is genuinely 4-5 mPa s, so a floor "
                   "excluding rated operation is mis-set.  NOTE: this axis is now a "
                   "strict SUPERSET of the slipper's -- see COMMON_SUBBOX"),
    InputDimension("lam", 1.0e-5, 1.0e-2, "log", "s", "as slipper INPUT_SPACE"),
    InputDimension("eta_s_ratio", 0.05, 0.95, "lin", "-", "as slipper INPUT_SPACE"),
    InputDimension("k_F_bs", 0.1, 30.0, "log", "-",
                   "D2: F_app = k_F_bs * W_sq_ref (3 pi eta0 omega R_b^4 / c0^2); "
                   "the 3 pi prefactor is a convention constant, not physics"),
)

ALPHA_FIXED = 0.25             # Giesekus mobility, campaign-fixed (pre-registered)
SQUEEZE_PHASE = 1.0            # |e_dot| = SQUEEZE_PHASE * omega * |e| (pre-registered)
SPIN_RATIO = 1.0               # omega_b = SPIN_RATIO * omega (+ MANDATORY OAT)
P_CASE_PA = 1.0e5              # socket-mouth (case drain) pressure; CONVENTION


# ===========================================================================
# Geometry -- REPRESENTATIVE class values (annular fed film)
# ===========================================================================

GEOMETRY = BallSocketGeometry(
    c0=float("nan"),                                  # c0 is an INPUT AXIS
    R_b=rep.R_B_M,                                    # 6.5e-3 m (representative)
    theta_max=np.deg2rad(rep.THETA_MAX_DEG),          # 73 deg
    theta_min=np.deg2rad(rep.THETA_MIN_BAND_DEG[0]),  # band end; OAT-swept
)
A_P_M2 = rep.A_P_M2                                   # 2.0e-4 m^2 (representative)

# Pre-registered BANDS (reported at both ends; STOP on binding-geography flip)
# RULING A1: the WIDE band governs.
SIGMA_STAR_BAND_M = rep.sigma_star_band()             # ~ (0.336, 0.524) um
HMIN_FLOOR_BAND_M = rep.hmin_floor_band(3.0)          # ~ (1.01, 1.57) um
SIGMA_STAR_BAND_NARROW_M = rep.sigma_star_band(narrow=True)   # superseded
THETA_MIN_BAND_DEG = rep.THETA_MIN_BAND_DEG           # (25, 36) deg

# MANDATORY CONVENTION (ruling A2(i)): the ball-socket eta_0 axis is a strict
# SUPERSET of the slipper's, so any cross-interface AR comparison is made ONLY
# on the common sub-box below.  This is a convention, not a footnote.
COMMON_SUBBOX_WITH_SLIPPER = {"eta_0": (5.0e-3, 2.0e-1)}

FLUID_CLASSES = ("mineral", "water_glycol")

# Materials: literature-standard, CITED (guards retired by the 2026-07-26 ruling)
SOCKET = SocketMaterial()
DELTA_ALPHA_TH = delta_alpha_th()                     # 6.0e-6 /K (bronze - steel)


# ===========================================================================
# Constraint set (Decision D3)
# ===========================================================================

@dataclass(frozen=True)
class ConstraintSet:
    """g2 is per-fluid-class, never a single global number."""
    lambda_ratio_min: float = 3.0
    T_max_C_by_class: dict = field(default_factory=lambda: {
        "mineral": 110.0,
        "water_glycol": 60.0,        # HFC-E service ceiling (Totten & De Negri 2012)
    })
    De_sq_max: float = 0.2

    def evaluate(self, perf, fluid_class: str, sigma_star: float) -> Outcome:
        """sigma_star is passed explicitly because it is a pre-registered BAND;
        evaluate at BOTH ends and report the binding counts for each."""
        g1 = (perf.h_min / sigma_star) >= self.lambda_ratio_min
        g2 = perf.T_max <= self.T_max_C_by_class[fluid_class]      # degC
        g3 = perf.De_sq <= self.De_sq_max
        return Outcome.FEASIBLE if (g1 and g2 and g3) else Outcome.INFEASIBLE


# ===========================================================================
# One-at-a-time sensitivity block (ruling sec 1.5)
# ===========================================================================
# The alpha_th [19,21]e-6 band and its OAT entries are RETIRED: alpha_th is now
# the cited bronze point value 18e-6 /K.
OAT_BLOCK = {
    "spin_ratio": (0.0, 0.5, 1.0),                    # mandatory (uncertain physics)
    "theta_max_deg": rep.THETA_MAX_OAT_DEG,           # (66, 73, 80)
    "theta_min_deg": rep.THETA_MIN_OAT_DEG,           # (25, 36)
    "alpha_th_rel": (0.9, 1.0, 1.1),                  # cheap robustness, +/-10%
}
OAT_N = 32                     # Sobol prefix per OAT value (16-32 sample harness)


# ===========================================================================
# Campaign meta (slipper-validated; D4)
# ===========================================================================

N_AUDIT = 2 ** 11
N_BO = 500
N_INIT = 18
N_SEEDS = 5

DRAFT = False                  # FROZEN 2026-07-27; freeze_and_hash may run
FROZEN_ON = "2026-07-27"
SIGNED_BY = "Nigel C. Dhlamini"


def draft_config_dict() -> dict:
    return {
        "DRAFT": DRAFT,
        "material_policy": "literature-standard steel-on-bronze (CITED); "
                           "material identity enters only via alpha_th and sigma*",
        "input_space": [d.__dict__ for d in INPUT_SPACE],
        "alpha_fixed": ALPHA_FIXED, "squeeze_phase": SQUEEZE_PHASE,
        "spin_ratio": SPIN_RATIO, "p_case_Pa": P_CASE_PA,
        "geometry": {"R_b_m": rep.R_B_M, "A_p_m2": rep.A_P_M2,
                     "theta_max_deg": rep.THETA_MAX_DEG,
                     "theta_min_band_deg": list(THETA_MIN_BAND_DEG),
                     "domain": "annular fed film (pocket edge -> socket mouth)"},
        "materials": {"socket": "leaded tin bronze CuSn12Pb class (CITED)",
                      "socket_E_Pa": SOCKET.E, "socket_nu": SOCKET.nu,
                      "socket_alpha_th": SOCKET.alpha_th,
                      "socket_HB_range": list(SOCKET.HB_range),
                      "ball": "hardened bearing steel 100Cr6 class (CITED)",
                      "ball_constants": BALL_STEEL,
                      "delta_alpha_th": DELTA_ALPHA_TH},
        "sigma_star_band_m": list(SIGMA_STAR_BAND_M),
        "sigma_star_band_narrow_superseded_m": list(SIGMA_STAR_BAND_NARROW_M),
        "common_subbox_with_slipper": {k: list(v) for k, v
                                       in COMMON_SUBBOX_WITH_SLIPPER.items()},
        "hmin_floor_band_m": list(HMIN_FLOOR_BAND_M),
        "fluid_classes": list(FLUID_CLASSES),
        "constraints": {"lambda_ratio_min": 3.0,
                        "T_max_C": {"mineral": 110.0, "water_glycol": 60.0},
                        "De_sq_max": 0.2},
        "oat_block": {k: list(v) for k, v in OAT_BLOCK.items()}, "oat_n": OAT_N,
        "N_audit": N_AUDIT, "N_bo": N_BO, "N_init": N_INIT, "N_seeds": N_SEEDS,
        "FROZEN_ON": FROZEN_ON, "SIGNED_BY": SIGNED_BY,
    }


config_dict = draft_config_dict          # frozen alias


def freeze_and_hash(path: str | None = None) -> str:
    """Serialise the frozen campaign config, write it, and return its SHA256.

    The hash is quoted in `preregistration_FROZEN.md` and in ledger entry L-7.
    Refuses to run while DRAFT is True.
    """
    import hashlib
    import json
    import os

    if DRAFT:
        raise RuntimeError("config is still DRAFT; refusing to freeze")
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "campaign_frozen.json")
    cfg = draft_config_dict()
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    digest = hashlib.sha256(blob).hexdigest()
    with open(path, "w") as fh:
        json.dump({"sha256": digest, "frozen_on": FROZEN_ON,
                   "signed_by": SIGNED_BY, "config": cfg},
                  fh, indent=2, sort_keys=True, default=str)
    return digest


if __name__ == "__main__":
    print(freeze_and_hash())
