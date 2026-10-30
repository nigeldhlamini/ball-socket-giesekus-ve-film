"""R-54.5: decompose the class gap in E2's restricted frame, or STOP.

THE CLAIM UNDER TEST.  R-53.3 reported that once the rows the inlet decides are
removed, g1 is the most frequent single binding constraint in all four cells and
the two classes agree to within 1.06 pp.  R-54.5 rules that the agreement is
largely ENTAILED: `fluid_class` is declared and never read by the solver (R-50.1),
so within the restricted frame the classes can differ through ONE channel only --
rows whose dissipative rise carries T_max past the 60 C ceiling while staying
under 110 C.  The gap measures the ceiling's residual reach, not a resemblance
between fluids.

WHY THERE IS NO SAMPLING TERM.  The ruling allows for one "if the two classes'
cells are independent draws".  They are not: both classes label THE SAME ROWS of
one artefact (`binding_mineral_*` and `binding_water_glycol_*` are two columns of
the same frame), and the restriction T_in < 60 C is applied identically to both.
So the decomposition is not statistical at all -- it is exact, row for row, and
that makes it a check that can fail.

WHAT MUST HOLD, EXACTLY.  A row's labels can differ between classes only if g2
differs, i.e. 60 C < T_max_bnd <= 110 C.  On such a row, if g1 is violated and g3
holds, the row is g1-SOLE under the mineral ceiling and MULTI under the
water-glycol one.  The converse cannot occur, because the water-glycol ceiling is
the stricter of the two.  Therefore

    (mineral g1-sole share) - (water-glycol g1-sole share)
        == |{g1 violated, g3 satisfied, 60 < T_max_bnd <= 110}| / N_restricted

with no residual.  **Failing looks like:** any mismatch, to the row.  That would
mean a class channel other than g2 exists, contradicting R-50.1's verified
statement -- and this script STOPs and reports rather than reconciling.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "ch9"
DEST = OUT / "E2_class_gap_decomposition.json"
MIXED = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
ENDS = ("lo", "hi")


def main() -> int:
    from ball_socket.immutable import guarded_path
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M, ConstraintSet
    guarded_path(str(DEST), what="E2 class-gap decomposition")
    cs = ConstraintSet()
    wg, mineral = (cs.T_max_C_by_class["water_glycol"],
                   cs.T_max_C_by_class["mineral"])

    cols = (["outcome", "T_in", "T_max_bnd", "h_min_bnd", "De_sq_bnd"]
            + [f"binding_{c}_{e}" for c in ("mineral", "water_glycol") for e in ENDS])
    df = pd.read_parquet(MIXED, columns=cols)
    conv = df[df.outcome == "converged"]
    rest = conv[conv.T_in < wg]                      # the SAME rows for both classes
    n = len(rest)

    rep = {"restriction": {"rule": f"T_in < {wg:g} C, applied to the same rows for "
                                   f"both classes (one frame, two label columns)",
                           "n_converged": int(len(conv)), "n_restricted": int(n),
                           "identical_row_sets": True},
           "channel": (f"the only way the classes can differ in this frame: "
                       f"{wg:g} C < T_max_bnd <= {mineral:g} C"),
           "sampling_term": {"value_pp": 0.0,
                             "why": "the classes are two labels on one row set, not "
                                    "independent draws, so there is nothing to resample"},
           "cells": {}}

    reach = (rest.T_max_bnd > wg) & (rest.T_max_bnd <= mineral)
    rep["ceiling_reach_pct_of_restricted"] = float(100 * reach.mean())

    stop = []
    for end, sigma in zip(ENDS, (float(s) for s in SIGMA_STAR_BAND_M)):
        g1_ok = rest.h_min_bnd / sigma >= cs.lambda_ratio_min
        g3_ok = rest.De_sq_bnd <= cs.De_sq_max
        share = {}
        for cls in ("mineral", "water_glycol"):
            share[cls] = float(100 * (rest[f"binding_{cls}_{end}"] == "g1").mean())
        gap = share["mineral"] - share["water_glycol"]
        predicted_rows = int(((~g1_ok) & g3_ok & reach).sum())
        predicted = 100 * predicted_rows / n
        residual = gap - predicted
        if abs(residual) > 100 * 0.5 / n:            # half a row
            stop.append(f"sigma_{end}: gap {gap:.4f} pp, ceiling channel "
                        f"{predicted:.4f} pp, residual {residual:.4f} pp")
        rep["cells"][f"sigma_{end}"] = {
            "g1_sole_share_pct": share,
            "class_gap_pp": gap,
            "ceiling_channel_pp": predicted,
            "ceiling_channel_rows": predicted_rows,
            "sampling_pp": 0.0,
            "residual_pp": residual,
            "closes_exactly": abs(residual) <= 100 * 0.5 / n}

    rep["verdict"] = {
        "decomposition_closes": not stop,
        "reading": ("the gap is the thermal ceiling's residual reach in the "
                    "restricted frame, not a resemblance between the fluid classes"),
        "substantive_finding": ("g1 is the most frequent single binding constraint in "
                                "all four cells -- that is the property of the joint")}
    DEST.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"restricted rows: {n} of {len(conv)} converged "
          f"({100 * n / len(conv):.2f} %)")
    print(f"ceiling reach ({wg:g} < T_max_bnd <= {mineral:g}): "
          f"{rep['ceiling_reach_pct_of_restricted']:.3f} % of restricted rows")
    for cell, c in rep["cells"].items():
        print(f"  {cell}: mineral {c['g1_sole_share_pct']['mineral']:.3f} % vs "
              f"water-glycol {c['g1_sole_share_pct']['water_glycol']:.3f} %  "
              f"gap {c['class_gap_pp']:.4f} pp = ceiling {c['ceiling_channel_pp']:.4f} "
              f"+ sampling 0 + residual {c['residual_pp']:.2e}")
    if stop:
        raise SystemExit("STOP (R-54.5): the class gap is NOT accounted for by the "
                         "g2 channel, which contradicts R-50.1's verified statement "
                         "that fluid_class is never read by the solver:\n  "
                         + "\n  ".join(stop))
    print("\ndecomposition closes exactly; no class channel other than g2 exists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
