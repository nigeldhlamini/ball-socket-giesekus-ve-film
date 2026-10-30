"""R-62.4: persist A3's film-supported operating-space classification as an artefact.  No solves.

The computation is fig_R61_update.fig_u3 itself (same code, same bootstrap seed), run into a
scratch directory; its numbers are then written to results/ch9/R62_operating_space.json with
floor / basis / memory_sign tags, and the per-sample classes to R62_operating_space.parquet,
so scripts/ch9_numbers.py can emit macros from them.  The figure itself is not re-published.

Asserts: the per-sample classes recomputed here give the same counts as fig_u3; the rated
corner's 31 / 40 and every S50 equal the values the October update used
(Z:/Updates/October 2026/fig_u3_operating_space.json), when that file is reachable.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
DEST_JSON = OUT / "R62_operating_space.json"
DEST_PQ = OUT / "R62_operating_space.parquet"
UPDATE_JSON = pathlib.Path("Z:/Updates/October 2026/fig_u3_operating_space.json")

TAGS = {
    "floor": "production floor delta = h_min/c >= 0.01: an 'equilibrium' is one the solver finds above it; "
             "class 4 is 'no equilibrium above one per cent of the clearance'",
    "basis": "pure-film arm (box D corrected) for classes 1-2, Lambda = h_min_bnd / sigma* at each band end; "
             "class 3 = any of the 12 mixed arms at that sigma* end (box E corrected), paired by sample_id; "
             "S = p_ch A_p c0^2 / (3 pi eta_0 omega R_b^4), eta_0 the swept service viscosity",
    "memory_sign": "modulus + D-6 bound (corrected campaign rows as stored; R58_E4_signed.report.json:dLam_D6_bound)",
    "collapse_test": "NOT YET RUN (R-54.6): S has not been shown to collapse the other inputs; no (S, theta_min) "
                     "map may be drawn in the paper from these numbers until it is",
}


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-62.4 operating-space artefact")
    import matplotlib
    matplotlib.use("Agg")
    import fig_R61_update as G
    from ball_socket.campaign_config_v4 import ConstraintSet
    with tempfile.TemporaryDirectory() as td:
        fig, rep = G.fig_u3(pathlib.Path(td), prov)
    # per-sample classes, recomputed with fig_u3's own rules, and checked against its counts
    D = pd.read_parquet(OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet").sort_values(
        "sample_id").reset_index(drop=True)
    E = pd.read_parquet(OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet",
                        columns=["sample_id", "model_id", "outcome"])
    cs = ConstraintSet()
    D["S"] = G.duty_S(D.p_ch.values, D.c0.values, D.eta_0.values, D.omega.values)
    pf = D.outcome == "converged"
    for end, arms in (("lo", range(0, 12)), ("hi", range(12, 24))):
        m = E[E.model_id.isin(list(arms))]
        anym = m.assign(eq=m.outcome == "converged").groupby("sample_id").eq.any().reindex(D.sample_id).values
        lam = D[f"lambda_ratio_{end}"]
        D[f"class_{end}"] = np.where(pf & (lam >= cs.lambda_ratio_min), 1, np.where(pf, 2, np.where(anym, 3, 4)))
        got = {int(i): int((D[f"class_{end}"] == i).sum()) for i in range(1, 5)}
        assert got == {int(k): v for k, v in rep["class_shares"][end]["counts"].items()}, (end, got)
    if UPDATE_JSON.exists():
        U = json.loads(UPDATE_JSON.read_text(encoding="utf-8"))
        for a, b in zip(U["S50_sigma_lo"], rep["S50_sigma_lo"]):
            for k in ("film_supported_1_2", "full_film_1"):
                assert a[k]["S50"] == b[k]["S50"] and a[k]["bootstrap95"] == b[k]["bootstrap95"], k
        assert U["rated_corner"] == rep["rated_corner"]
    out = {"provenance": prov, **TAGS,
           "source": "scripts/fig_R61_update.py:fig_u3 (same code and seed as the October update figure)",
           "class_counts": {e: rep["class_shares"][e]["counts"] for e in ("lo", "hi")},
           "class_shares": {e: {k: v for k, v in rep["class_shares"][e].items() if k != "counts"}
                            for e in ("lo", "hi")},
           "contact_supported_note": "class 3 is kept and reported (R-62.3): where the film fails, the mixed "
                                     "arms almost never find an equilibrium either",
           "S50_sigma_lo": rep["S50_sigma_lo"],
           "survivors_sigma_lo": rep["panel_b_sigma_lo"],
           "dominant_constraint_sigma_lo": rep["dominant_constraint_panel_b"],
           "rated_corner": rep["rated_corner"],
           "S_definition": rep["S_definition"], "S_hand_checks": rep["S_hand_checks"],
           "sources": rep["sources"]}
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST_JSON), what="R-62.4 operating space")
    guarded_path(str(DEST_PQ), what="R-62.4 operating space classes")
    D[["sample_id", "theta_min_deg", "S", "class_lo", "class_hi"]].assign(
        git_commit=prov["commit"], memory_sign=TAGS["memory_sign"]).to_parquet(DEST_PQ, index=False)
    DEST_JSON.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("class_counts", "rated_corner")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
