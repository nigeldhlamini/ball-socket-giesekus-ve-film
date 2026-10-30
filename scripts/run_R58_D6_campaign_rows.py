"""R-58.1 "Exposure beyond E4", failing branch: the box-E rows the row-wise bound could not
clear are re-solved on the fix; any feasibility flip is a STOP.

Selection is the bound's own (run_R58_E4_signed.py): a converged row at a sigma* end is
flagged when 2 |share_max| Lambda / s_min > |Lambda - 3|, share_max the E4 maximum on the
signed path and s_min = 0.52.  Each flagged (sample, arm) is re-solved ONCE through the
registered evaluate_x call with memory_sign = "signed", and Lambda = h_min_bnd / sigma* is
compared with the stored (modulus) value at every flagged end.  g2 and g3 are not touched by
the sign of the memory term beyond its O(1e-3) load effect; they are recomputed and reported.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
SRC = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
DEST = OUT / "R58_D6_campaign_rows.parquet"
REPORT = OUT / "R58_D6_campaign_rows.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]


def job(r):
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.campaign as C
    from ball_socket.campaign import REGISTERED_CONSTRAINT_COLUMNS
    from run_R57_mesh_error import _contact
    x = {k: float(r[k]) for k in AXES}
    row = C.evaluate_x(x, contact=_contact(int(r["model_id"])), clearance_closure=True,
                       delta_T_datum="bulk", Ntheta=32, Nphi=24, memory_sign="signed")
    return {"sample_id": int(r["sample_id"]), "model_id": int(r["model_id"]),
            **{f"{k}_signed": row.get(k) for k in REGISTERED_CONSTRAINT_COLUMNS},
            "realised_memory_sign": row.get("realised_memory_sign")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from ball_socket.immutable import guarded_path
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M as S, ConstraintSet
    guarded_path(str(DEST), what="R-58 D-6 campaign rows")
    b = json.loads((OUT / "R58_E4_signed.report.json").read_text())["dLam_D6_bound"]
    share, smin = b["share_used"], b["s_min_used"]
    E = pd.read_parquet(SRC, columns=AXES + ["sample_id", "model_id", "outcome", "h_min_bnd",
                                             "T_max_bnd", "De_sq_bnd", "eps_ecc"])
    E = E[E.outcome == "converged"].copy()
    for end, s in (("lo", S[0]), ("hi", S[1])):
        lam = E.h_min_bnd / s
        E[f"flag_{end}"] = 2 * share * lam / smin > (lam - 3).abs()
    F = E[E.flag_lo | E.flag_hi].copy()
    print(f"{len(F)} flagged rows ({int(F.flag_lo.sum())} at sigma lo, {int(F.flag_hi.sum())} at hi), "
          f"{F.sample_id.nunique()} samples", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        got = list(ex.map(job, F.to_dict(orient="records")))
    D = F.merge(pd.DataFrame(got), on=["sample_id", "model_id"])
    D.to_parquet(DEST, index=False)
    cs = ConstraintSet()
    flips, moves = [], []
    for end, s in (("lo", S[0]), ("hi", S[1])):
        g = D[D[f"flag_{end}"] & (D.outcome_signed == "converged")]
        L0_, L1_ = g.h_min_bnd / s, g.h_min_bnd_signed.astype(float) / s
        moves += list((L1_ - L0_).values)
        fl = g[(L0_ >= 3) != (L1_ >= 3)]
        flips += [{"end": end, "sample_id": int(r.sample_id), "model_id": int(r.model_id),
                   "Lambda_modulus": float(r.h_min_bnd / s), "Lambda_signed": float(r.h_min_bnd_signed / s)}
                  for r in fl.itertuples()]
    moves = np.array(moves)
    rep = {"flagged_rows": int(len(F)), "samples": int(F.sample_id.nunique()), "workers": a.workers,
           "wall_min": (time.time() - t0) / 60,
           "outcomes_signed": D.outcome_signed.value_counts().to_dict(),
           "dLambda_signed_minus_modulus": {"median": float(np.median(moves)), "min": float(moves.min()),
                                            "max": float(moves.max()), "max_abs": float(np.abs(moves).max())},
           "bound_used": {"share": share, "s_min": smin},
           "g1_flips": flips, "STOP": bool(flips),
           "g3_change_max": float((D.De_sq_bnd_signed.astype(float) - D.De_sq_bnd).abs().max()),
           "T_max_change_max_K": float((D.T_max_bnd_signed.astype(float) - D.T_max_bnd).abs().max())}
    guarded_path(str(REPORT), what="R-58 D-6 campaign rows report")
    REPORT.write_text(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(rep, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
