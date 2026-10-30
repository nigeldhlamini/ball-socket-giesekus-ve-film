"""R-53.3: the binding geography with the frame-decided rows removed (E2).

Gate 6's boundary-coincidence check flags E2: the registered inlet range spans
the water-glycol ceiling, so under that ceiling every row with T_in > 60 C fails
g2 by arithmetic (T_max_bnd >= T_max >= T_in).  Those rows are not evidence about
the joint; they are evidence about where the inlet range sits.

That is recorded as ERR-21's reach.  But the restricted result is a FINDING, not
bookkeeping: with the decided rows removed, does the water-glycol binding
geography still differ from the mineral one, or was the difference the frame?

Measured on the campaign the paper reports (box E, mixed), not on box A/B, whose
counts the erratum corrects.  No solves.  Reported per sigma* band end and per
class, over converged rows, as counts and as shares of the restricted converged
population, with the dominant single constraint named for each cell.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "ch9"
DEST = OUT / "E2_restricted_geography.json"
MIXED = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
CLASSES = ("mineral", "water_glycol")
ENDS = ("lo", "hi")


def _tally(frame, col):
    v = frame[col].value_counts()
    single = v.drop(labels=["none", "multi"], errors="ignore")
    return {"counts": {str(k): int(n) for k, n in v.items()},
            "dominant_single": str(single.idxmax()),
            "dominant_single_pct": float(100 * single.max() / len(frame))}


def main() -> int:
    from ball_socket.immutable import guarded_path
    from ball_socket.campaign_config_v4 import ConstraintSet
    guarded_path(str(DEST), what="E2 restricted geography")
    cs = ConstraintSet()
    wg_ceiling = cs.T_max_C_by_class["water_glycol"]

    cols = ["outcome", "T_in"] + [f"binding_{c}_{e}" for c in CLASSES for e in ENDS]
    df = pd.read_parquet(MIXED, columns=cols)
    conv = df[df.outcome == "converged"]
    rest = conv[conv.T_in < wg_ceiling]

    rep = {"artefact": MIXED.name,
           "restriction": f"T_in < {wg_ceiling:g} C (rows the ceiling does not decide)",
           "n_converged": int(len(conv)),
           "n_restricted": int(len(rest)),
           "restricted_pct_of_converged": float(100 * len(rest) / len(conv)),
           "cells": {}}
    for cls in CLASSES:
        for end in ENDS:
            col = f"binding_{cls}_{end}"
            rep["cells"][f"sigma_{end}|{cls}"] = {
                "all_converged": _tally(conv, col),
                "restricted": _tally(rest, col)}

    dom_all = {k: v["all_converged"]["dominant_single"] for k, v in rep["cells"].items()}
    dom_res = {k: v["restricted"]["dominant_single"] for k, v in rep["cells"].items()}
    rep["finding"] = {
        "dominant_single_all_converged": dom_all,
        "dominant_single_restricted": dom_res,
        "classes_agree_when_restricted": all(
            dom_res[f"sigma_{e}|mineral"] == dom_res[f"sigma_{e}|water_glycol"]
            for e in ENDS),
        "largest_share_gap_between_classes_pp": max(
            abs(rep["cells"][f"sigma_{e}|mineral"]["restricted"]["dominant_single_pct"]
                - rep["cells"][f"sigma_{e}|water_glycol"]["restricted"]["dominant_single_pct"])
            for e in ENDS)}
    DEST.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"converged {rep['n_converged']}, restricted {rep['n_restricted']} "
          f"({rep['restricted_pct_of_converged']:.1f} %)")
    for k, v in rep["cells"].items():
        print(f"  {k:24s} all -> {v['all_converged']['dominant_single']:5s} "
              f"({v['all_converged']['dominant_single_pct']:5.1f} %)   "
              f"restricted -> {v['restricted']['dominant_single']:5s} "
              f"({v['restricted']['dominant_single_pct']:5.1f} %)")
    print("classes agree when restricted:", rep["finding"]["classes_agree_when_restricted"],
          "| largest gap %.2f pp" % rep["finding"]["largest_share_gap_between_classes_pp"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
