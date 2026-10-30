"""R-50.5: E3's restricted verdict against the T_max discretisation bias.

The old 142x survival factor was computed when g2's margin was wide.  On the
restricted population (T_in < 60 C) every sole-g2 row has an inlet >= 54.4 C and
a dissipative rise <= 9.6 K, so the RISE is now the whole margin -- and the bias
lives in the rise.  Question: how many of the 408 / 216 sole-g2 rows survive
when the rise is scaled by the bias factor, both ways?

Bias factor: the registered measurement, box-B errata -- the dissipative rise is
UNDER-reported by 12.91 % at the production 32x24 grid on E3's envelope row.
Scaling the rise DOWN by that factor is the adverse direction for the
confirmation (rows can drop back below 60 C); UP is the direction the
measured bias actually points (rows can be added).  Both are reported, and a
2x stress on each, because one row's bias is not a population bound.

What a failing result looks like: the down-scaled count reaches zero at a band
end, i.e. E3's restricted confirmation depends on the bias.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "ch9" / "E3_bias_sensitivity.json"
MIXED = ROOT / "results" / "ch9" / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
RISE_BIAS = 0.1291          # boxb_errata.md, E3 envelope row, 32x24 vs refined


def main() -> int:
    from ball_socket.immutable import guarded_path
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M, ConstraintSet
    guarded_path(str(OUT), what="E3 bias sensitivity")
    cs = ConstraintSet()
    wg = cs.T_max_C_by_class["water_glycol"]
    c = pd.read_parquet(MIXED, columns=["sample_id", "outcome", "T_in", "T_max_bnd", "h_min_bnd",
                                        "De_sq_bnd"])
    c = c[(c.outcome == "converged") & (c.T_in < wg)]
    rise = c.T_max_bnd - c.T_in
    rep = {"rise_bias": RISE_BIAS, "restricted_rows": int(len(c)), "bands": {}}
    for s, name in zip(SIGMA_STAR_BAND_M, ("lo", "hi")):
        ok13 = (c.h_min_bnd / s >= cs.lambda_ratio_min) & (c.De_sq_bnd <= cs.De_sq_max)
        row = {}
        for label, k in (("as_computed", 1.0), ("down_1x", 1 - RISE_BIAS),
                         ("down_2x", 1 - 2 * RISE_BIAS), ("up_1x", 1 + RISE_BIAS),
                         ("up_2x", 1 + 2 * RISE_BIAS)):
            sole = ok13 & (c.T_in + k * rise > wg)
            row[label] = {"rows": int(sole.sum()),
                          # A1: the 24 arms of a sample agree on T_max to < 0.01 K,
                          # so rows move in blocks of 24 -- quote SAMPLES
                          "samples": int(c.sample_id[sole].nunique())}
        rep["bands"][name] = row
    OUT.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
