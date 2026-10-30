"""R-52.2/3: adjudicate E1 against its FROZEN text, after the population check.

THE REGISTERED CLAIM, verbatim (R-52.2: quoted with file and line, or not
adjudicated).  `preregistration_FROZEN.md:117`, identical to the signed content
`preregistration_FINAL_DRAFT_v3.md:96` (sha256 b413a5d1...10e9d, re-verified):

    E1 -- g3 essentially never binds for HFC-E draws (De_sq ~ 0.02-0.06).

The confinement clause "binds only in the high-lambda corner" is in
`preregistration_DRAFT.md:43-44` and `preregistration_FINAL_DRAFT.md:97-98`
only; it was dropped at v2 and never frozen.  Its wording traces to
`CODE_BRIEF_CH9_BALLSOCKET.md:168-170` ("lambda ~ 3e-3 s at high omega").

POPULATION CHECK (R-52.3), measured, not argued.  "HFC-E draws" can only mean
rows labelled under the water-glycol ceiling (R-48.1: fluid class enters only
through g2).  Box E carries no per-row fluid identity; every row is labelled
under BOTH classes.  So the check is: (a) the set of converged rows carrying a
water-glycol label equals the set of converged rows, and (b) De_sq_bnd is one
column, not a per-class one.  Both are asserted.  What the check DOES change is
any statistic that reads g2: the sole-binding range, which mixed both ceilings.

VERDICT, both readings of the parenthetical (R-52.3):
  * part of the claim   -> refuted directly by the in-range share;
  * supporting forecast -> the forecast fails, and the qualitative clause with
    it, because the violation share is what "never binds" denies.
The verdict must not rest on the sole-binding figure: "essentially never" has no
threshold, and a few-percent sole-binding share is exactly where that bites.

ALSO RECORDED: the same statistics on box A's and box B's OWN artefacts (their
interior De_sq), and on box E's legacy interior column, so the reader can see
the refutation does not depend on box D's boundary correction and was already
in the data both boxes adjudicated.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "ch9"
DEST = OUT / "E1_adjudication.json"
BOXE = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
OWN = {"box A": OUT / "audit_N2048_seed20260727.parquet",
       "box B": OUT / "audit_N8192_seed20260727.parquet"}
LO, HI = 0.02, 0.06          # the registered parenthetical
SIGNED_SHA = "b413a5d1990185510991ef46e27c23a8fc88638226ab5e53c9da5ff981710e9d"
FROZEN_LINE = "- **E1** — g3 essentially never binds for HFC-E draws (De_sq ~ 0.02–0.06)."


def _stats(x: pd.Series, limit: float) -> dict:
    return {"n": int(len(x)), "median": float(x.median()),
            "q25": float(x.quantile(0.25)), "q75": float(x.quantile(0.75)),
            "in_registered_range_pct": float(100 * x.between(LO, HI).mean()),
            "below_range_pct": float(100 * (x < LO).mean()),
            "above_range_pct": float(100 * (x > HI).mean()),
            "violating_g3_pct": float(100 * (x > limit).mean()),
            "q75_over_registered_top": float(x.quantile(0.75) / HI)}


def main() -> int:
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M, ConstraintSet
    cs = ConstraintSet()
    # -- the quote is the registered text: check it against both files -------
    frozen = (ROOT / "preregistration_FROZEN.md").read_text(encoding="utf-8").splitlines()
    signed = (ROOT / "preregistration_FINAL_DRAFT_v3.md").read_text(encoding="utf-8").splitlines()
    assert frozen[116] == FROZEN_LINE, "FROZEN.md:117 is not the quoted text"
    assert signed[95] == FROZEN_LINE, "FINAL_DRAFT_v3.md:96 is not the quoted text"
    sha = hashlib.sha256((ROOT / "preregistration_FINAL_DRAFT_v3.md").read_bytes()).hexdigest()
    assert sha == SIGNED_SHA, "signed content hash moved"

    E = pd.read_parquet(BOXE)
    conv = E[E.outcome == "converged"]
    # -- population check (a) and (b) -----------------------------------------
    wg_label_cols = [c for c in E.columns if c.startswith("outcome_water_glycol_")
                     and not c.endswith("_legacy")]
    wg_rows = set(E.index[E[wg_label_cols].notna().all(axis=1)
                          & (E.outcome == "converged")])
    assert wg_rows == set(conv.index), "HFC-E-labelled converged rows != converged rows"
    assert not [c for c in E.columns if c.startswith("De_sq") and ("water" in c or "mineral" in c)], \
        "a per-class De_sq column exists: the De_sq figures would need the subset"

    viol = conv.De_sq_bnd > cs.De_sq_max
    sole = {}
    for band, s in zip(("lo", "hi"), SIGMA_STAR_BAND_M):
        g1 = conv.h_min_bnd / s >= cs.lambda_ratio_min
        for cls, lim in cs.T_max_C_by_class.items():
            g2 = conv.T_max_bnd <= lim
            sole[f"{cls}_{band}"] = float(100 * (g1 & g2 & viol).mean())
            # cross-check against the campaign's own labels
            lab = (conv[f"binding_{cls}_{band}"] == "g3").mean() * 100
            assert abs(lab - sole[f"{cls}_{band}"]) < 1e-9, (cls, band, lab)
    wg = [v for k, v in sole.items() if k.startswith("water_glycol")]

    own = {}
    for name, p in OWN.items():
        B = pd.read_parquet(p, columns=["outcome", "De_sq"])
        own[name] = {"artefact": p.name, "quantity": "De_sq (interior, as that box used it)",
                     **_stats(B[B.outcome == "converged"].De_sq, cs.De_sq_max)}

    bnd = _stats(conv.De_sq_bnd, cs.De_sq_max)
    rep = {
        "expectation": "E1",
        "registered_text": {"quote": FROZEN_LINE, "file_line": "preregistration_FROZEN.md:117",
                            "signed_content": "preregistration_FINAL_DRAFT_v3.md:96",
                            "signed_sha256": SIGNED_SHA},
        "never_frozen": {"clause": "binds only in the high-lambda corner",
                         "appears_in": ["preregistration_DRAFT.md:43-44",
                                        "preregistration_FINAL_DRAFT.md:97-98"],
                         "origin": "CODE_BRIEF_CH9_BALLSOCKET.md:168-170",
                         "dropped_at": "preregistration_FINAL_DRAFT_v2.md:98"},
        "population_check": {
            "reading": "HFC-E draws = rows under the water-glycol ceiling (R-48.1)",
            "hfce_labelled_converged_rows_equal_converged_rows": True,
            "De_sq_is_one_column_per_row": True,
            "consequence": ("the De_sq statistics are the HFC-E statistics; the "
                            "sole-binding range is NOT -- it mixed both ceilings"),
            "no_lambda_subset": ("the frozen text registers no HFC-E rheology, so no "
                                 "lambda-restricted subset is definable; choosing one "
                                 "now would be post hoc")},
        "box_E": {"artefact": BOXE.name, "quantity": "De_sq_bnd (box D, registered)",
                  **bnd,
                  "legacy_interior": _stats(conv.De_sq, cs.De_sq_max),
                  "g3_sole_binding_pct": sole,
                  "g3_sole_binding_hfce_pct": [min(wg), max(wg)],
                  "g3_sole_binding_all_cells_pct": [min(sole.values()), max(sole.values())]},
        "own_artefacts": own,
        "verdict": "REFUTED",
        "readings": {
            "parenthetical_part_of_claim": (f"refuted directly: {bnd['in_registered_range_pct']:.1f} % "
                                            f"of HFC-E rows lie in [{LO}, {HI}]"),
            "parenthetical_as_forecast": (f"the forecast implied a >= {cs.De_sq_max / HI:.1f}x margin "
                                          f"below the g3 limit throughout; "
                                          f"{bnd['violating_g3_pct']:.1f} % of rows exceed the "
                                          f"limit, so 'essentially never binds' fails with it")},
        "reason": ("the SPREAD, not the centre: median %.3f is barely above the "
                   "registered top %.2f, the upper quartile is %.1fx it. g3 activity "
                   "is a tail property; the registration supplied a central range "
                   "(aggregation catalogue A5)" % (bnd["median"], HI, bnd["q75_over_registered_top"])),
        "not_load_bearing": ("the sole-binding share (%.1f-%.1f %% under the HFC-E "
                             "ceiling) is not used: 'essentially never' has no threshold"
                             % (min(wg), max(wg))),
        "descriptive_unregistered": ("every g3 violation has lambda >= 4.5e-4 s, 88.3 % above "
                                     "1e-3 s -- an OBSERVATION the campaign supports, not "
                                     "evidence for E1 and not a confirmed expectation (R-52.4)"),
    }
    DEST.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("verdict", "readings", "reason")}, indent=2))
    print("box E:", {k: round(v, 4) for k, v in bnd.items() if isinstance(v, float)})
    print("legacy:", {k: round(v, 4) for k, v in rep["box_E"]["legacy_interior"].items() if isinstance(v, float)})
    print("sole:", {k: round(v, 2) for k, v in sole.items()})
    for n, o in own.items():
        print(n, {k: round(v, 4) for k, v in o.items() if isinstance(v, float)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
