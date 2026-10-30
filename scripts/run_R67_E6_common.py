"""R-67.3: E6 on a common property basis, by post-processing box B's stored rows.  No solves.

Box B's frozen files are read, never written (audit_N8192_seed20260727.parquet; its routing
equals campaign_handback_boxB.md:67-80, asserted by scripts/ch9_numbers.py).  The two sides of
E6 differed in one input: the Brinkman k_f (slipper 0.13, giesekus-solver/ar_bo_campaign.py:250;
joint 0.60, ball_socket/thermal.py:46-48 on src/thermal.py's default).  Both through-flow tests
use the same default rho c_p (ball_socket/thermal.py:37-43; ar_bo_campaign.py:425-426).

Flags, independent of routing precedence (the stored route records Brinkman first, so the q_min
flag of a Brinkman-routed row is recomputed rather than read):
  Brinkman  Br = eta_0 V_char^2 / (k_f 20 K) > 0.1, V_char = spin_ratio omega R_b sin(theta_max)
            (solver.py:473-474);
  q_min     Q_out <= Q_min = P_diss / (rho c_p 10 K), from the stored P_loss and Q_leak.
Check first: at k_f = 0.60 the flags reproduce every stored route.  "On the stored solutions":
a route change would alter the row's temperature and clearance (second order, not post-processed).

Registered (R-67.3), the section 6.3 contrast:
  1. the joint's Brinkman-flag share at k_f = 0.13 stays below the slipper's 61.42 %;
  2. the joint's q_min-flag share stays above its own Brinkman-flag share.
By monotonicity (not a hypothesis): the total Level-2 fraction is >= 87.8 %.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "ch9"
SRC = OUT / "audit_N8192_seed20260727.parquet"
DEST = OUT / "R67_E6_common.parquet"
REPORT = OUT / "R67_E6_common.report.json"
K_SLIPPER, K_JOINT = 0.13, 0.60
SLIPPER_BRINK_PCT = 61.42


def flags(B, k_f):
    from ball_socket import thermal as bt
    from ball_socket import representative as rep
    from src.thermal import ThermalProperties
    th = ThermalProperties()
    V = B.spin_ratio * B.omega * rep.R_B_M * np.sin(np.deg2rad(B.theta_max_deg))
    br = B.eta_0 * V ** 2 / (k_f * bt.BR_DT_REF_K) > bt.BR_THRESHOLD
    qmin = B.P_loss / (th.rho * th.c_p * bt.DELTA_T_SAFETY_K)
    qf = (B.Q_leak <= qmin)
    return br, qf


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-67.3 E6 common basis")
    B = pd.read_parquet(SRC)
    B = B[B.outcome == "converged"].copy()
    br6, q6 = flags(B, K_JOINT)
    route6 = np.where(br6, "L2_brinkman", np.where(q6, "L1->L2_qmin", "L1"))
    mism = int((route6 != B.thermal_route.values).sum())
    br13, q13 = flags(B, K_SLIPPER)
    n = len(B)
    pct = lambda m: 100.0 * float(np.sum(m)) / n
    res = {"k_f_0.60_stored_basis": {"brinkman_flag_pct": pct(br6), "qmin_flag_pct": pct(q6),
                                     "brinkman_only_pct": pct(br6 & ~q6), "qmin_only_pct": pct(q6 & ~br6),
                                     "both_pct": pct(br6 & q6), "total_L2_pct": pct(br6 | q6)},
           "k_f_0.13_common_basis": {"brinkman_flag_pct": pct(br13), "qmin_flag_pct": pct(q13),
                                     "brinkman_only_pct": pct(br13 & ~q13), "qmin_only_pct": pct(q13 & ~br13),
                                     "both_pct": pct(br13 & q13), "total_L2_pct": pct(br13 | q13)}}
    c = res["k_f_0.13_common_basis"]
    rep = {"provenance": prov, "source": SRC.name, "n_converged": n,
           "basis": "on the stored solutions (box B); rho c_p default on both sides",
           "reproduces_stored_routes_at_k_0.60": mism == 0, "stored_route_mismatches": mism, **res,
           "monotone_total_ge_stored": bool(c["total_L2_pct"] >= res["k_f_0.60_stored_basis"]["total_L2_pct"]),
           "slipper_brinkman_pct": SLIPPER_BRINK_PCT,
           "registered": {"1_joint_brinkman_below_slipper": bool(c["brinkman_flag_pct"] < SLIPPER_BRINK_PCT),
                          "2_joint_qmin_above_joint_brinkman": bool(c["qmin_flag_pct"] > c["brinkman_flag_pct"])}}
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-67.3 E6 common basis")
    guarded_path(str(REPORT), what="R-67.3 report")
    B.assign(brinkman_flag_k060=br6.values, qmin_flag=q6.values, brinkman_flag_k013=br13.values,
             git_commit=prov["commit"], row_index=B.index.values)[["row_index", "thermal_route", "brinkman_flag_k060", "brinkman_flag_k013",
                                         "qmin_flag", "git_commit"]].to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
