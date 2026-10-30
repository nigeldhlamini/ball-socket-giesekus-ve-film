"""R-49.2: E3 restricted to the population where it can be refuted.

REGISTERED (preregistration_FROZEN.md, box A): "E3 -- the envelope may
genuinely bind g2 for water-glycol entries.  Either outcome reportable; record
which."  NO QUANTITATIVE THRESHOLD is registered.  Box B operationalised it
existentially under the registered strict single-constraint binding convention
(g2 the ONLY violated constraint): "binds" = that count is positive (528 lo /
379 hi, campaign_handback_boxB.md).

WHY RESTRICT (R-49.1(c)).  For T_in >= the 60 C ceiling, T_max >= T_in >= 60 C,
so g2 fails by arithmetic before any solve -- if and only if the dissipative
rise is non-negative on every row, which this script CHECKS rather than
assumes.  Over T_in < 60 C the verdict is not fixed by construction.

WHAT A REFUTING RESULT LOOKS LIKE (R-48.4 / gate 6): zero rows, at a sigma*
band end, in which g2 is the sole violated constraint with T_in < 60 C.  The
script does NOT apply any share threshold: none is registered, and choosing one
after seeing 2.3 % would be a post-hoc criterion (rule 18).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "ch9" / "E3_restricted.json"
MIXED = ROOT / "results" / "ch9" / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"


def main() -> int:
    from ball_socket.immutable import guarded_path
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M, ConstraintSet
    guarded_path(str(OUT), what="E3 restricted test")
    cs = ConstraintSet()
    wg = cs.T_max_C_by_class["water_glycol"]
    M = pd.read_parquet(MIXED, columns=["sample_id", "model_id", "outcome", "T_in",
                                        "T_max_bnd", "h_min_bnd", "De_sq_bnd"])
    c = M[M.outcome == "converged"].copy()
    rise = c.T_max_bnd - c.T_in
    rep = {"registered_text": ("the envelope may genuinely bind g2 for water-glycol "
                               "entries. Either outcome reportable; record which."),
           "registered_threshold": None,
           "operationalisation": ("box B: count of rows where g2 is the ONLY violated "
                                  "constraint (strict single-constraint convention) "
                                  "is positive"),
           "ceiling_C": wg, "converged_rows": int(len(c)),
           "premise_rise_nonnegative": {
               "min_rise_K": float(rise.min()), "n_negative": int((rise < 0).sum()),
               "holds": bool((rise >= 0).all())},
           "T_in_axis_share_at_or_above_ceiling": float((M.T_in >= wg).mean()),
           "bands": {}}
    for s, name in zip(SIGMA_STAR_BAND_M, ("lo", "hi")):
        g1 = c.h_min_bnd / s >= cs.lambda_ratio_min
        g2 = c.T_max_bnd <= wg
        g3 = c.De_sq_bnd <= cs.De_sq_max
        infeasible = ~(g1 & g2 & g3)
        only_g2 = g1 & ~g2 & g3                    # strict single-constraint: g2
        restricted = c.T_in < wg
        rep["bands"][name] = {
            "unrestricted": {
                "infeasible": int(infeasible.sum()),
                "g2_sole_binding": int(only_g2.sum()),
                "g2_sole_binding_with_T_in_at_or_above_ceiling":
                    int((only_g2 & ~restricted).sum())},
            "restricted_T_in_below_ceiling": {
                "converged": int(restricted.sum()),
                "infeasible": int((infeasible & restricted).sum()),
                "g2_sole_binding": int((only_g2 & restricted).sum()),
                "share_of_restricted_infeasible": float(
                    (only_g2 & restricted).sum() / max((infeasible & restricted).sum(), 1)),
                "share_of_unrestricted_g2_sole": float(
                    (only_g2 & restricted).sum() / max(only_g2.sum(), 1)),
                "max_rise_among_g2_sole_K": float(rise[only_g2 & restricted].max())
                if (only_g2 & restricted).any() else None,
                "min_T_in_among_g2_sole_C": float(c.T_in[only_g2 & restricted].min())
                if (only_g2 & restricted).any() else None,
                "refuted_existentially": bool((only_g2 & restricted).sum() == 0)}}
    OUT.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
