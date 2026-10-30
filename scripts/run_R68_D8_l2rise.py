"""R-68.2: the L2 rise on the mineral properties for every converged row whose route changes from L1
to L2 when the route is recomputed with the class properties (boxes D and E, corrected).

Route recomputation (thermal.py:37-48, the solver's precedence, solver.py:473-484): Brinkman
Br = eta_0 V_char^2 / (k_f 20 K) > 0.1 first, then q_min Q_out <= P_diss / (rho c_p 10 K).  At the
defaults it reproduces every stored route in both boxes (checked in run_R68_D8_reemit.py).  A row
L1 at the defaults but L2 on the mineral properties has no stored L2 field; its L2 rise is computed
here at the STORED solution (first_order_fixed_solution): the solver built as campaign builds it
(run_R57_mesh_error._solver) at the row's contact arm, memory_sign "modulus" (boxes D/E as stored),
one forward solve at the stored eps_ecc and c_equilibrium (solver._forward; the stored load must be
reproduced to 1e-9), then solve_l2_spherical with the mineral ThermalProperties and the boundary-
aware maximum (solver.py:486-491): rise = field_max_boundary(T) - T_in.

    python scripts/run_R68_D8_l2rise.py run [--workers 4]     # checkpointed in blocks; rerun resumes
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
SRCS = {"D": OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet",
        "E": OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"}
CKPT = OUT / "R68_D8_l2rise_ckpt"
DEST = OUT / "R68_D8_l2rise.parquet"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
MEMORY_SIGN = "modulus"
BLOCK = 250


def mineral_routes(c):
    import bs_thermal_props as W
    from ball_socket import thermal as bt
    from ball_socket import representative as rep
    V = c.spin_ratio * c.omega * rep.R_B_M * np.sin(np.deg2rad(73.0))
    br = c.eta_0 * V ** 2 / (W.MINERAL["k_f"] * bt.BR_DT_REF_K) > bt.BR_THRESHOLD
    q = c.Q_leak <= c.P_loss / (W.MINERAL["rho"] * W.MINERAL["c_p"] * bt.DELTA_T_SAFETY_K)
    return np.where(br, "L2_brinkman", np.where(q, "L1->L2_qmin", "L1"))


def rows():
    out = []
    for box, src in SRCS.items():
        cols = AXES + ["sample_id", "outcome", "thermal_route", "eps_ecc", "c_equilibrium", "F", "P_loss", "Q_leak",
                       "spin_ratio"] + (["model_id"] if box == "E" else [])
        X = pd.read_parquet(src, columns=cols)
        c = X[X.outcome == "converged"].copy()
        c["route_min"] = mineral_routes(c)
        ch = c[(c.thermal_route == "L1") & (c.route_min != "L1")]
        ch = ch.assign(box=box, model_id=(ch.model_id if box == "E" else -1))
        out.append(ch)
    return pd.concat(out, ignore_index=True).sort_values(["box", "sample_id", "model_id"]).reset_index(drop=True)


def job(rec):
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.thermal as bs_thermal
    from src.thermal import ThermalProperties
    import bs_thermal_props as W
    from run_R57_mesh_error import _solver, _contact
    t0 = time.time()
    slv, ci = _solver({k: float(rec[k]) for k in AXES}, _contact(int(rec["model_id"])), None)
    slv.memory_sign = MEMORY_SIGN
    fl = slv._fluid(ci)
    res, sph = slv._forward(ci, float(rec["eps_ecc"]), fl, float(rec["c_equilibrium"]))
    h = sph.gap()
    F = float(sph.load_capacity(res.pressure))
    if slv.contact is not None:
        F += float(sph.project_on_load_axis(slv.contact.pressure(h)))
    eta_bar = res.eta_bar if res.eta_bar is not None else np.full_like(h, ci.eta_0)
    gdot = sph.gap_averaged_shear_rate(res.pressure, eta_bar)
    T = bs_thermal.solve_l2_spherical(sph, eta_bar, gdot, ThermalProperties(**W.MINERAL), ci.T_in + 273.15)
    return {"box": rec["box"], "sample_id": int(rec["sample_id"]), "model_id": int(rec["model_id"]),
            "route_min": rec["route_min"], "rise_L2_min_bnd": float(sph.field_max_boundary(T)) - 273.15 - ci.T_in,
            "rise_L2_min_bulk": float(sph.surface_integral(T) / sph.surface_integral(np.ones_like(T))) - 273.15 - ci.T_in,
            "F_repro_rel": abs(F / float(rec["F"]) - 1), "wall_s": time.time() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run",))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-68.2 D-8 L2 rise for re-routed rows")
    R = rows()
    print(f"{len(R)} re-routed rows (D {int((R.box == 'D').sum())}, E {int((R.box == 'E').sum())})", flush=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    recs = R.to_dict(orient="records")
    with ProcessPoolExecutor(max_workers=min(a.workers, 4)) as ex:
        for b in range(0, len(recs), BLOCK):
            part = CKPT / f"block_{b:06d}.parquet"
            if part.exists():
                continue
            got = list(ex.map(job, recs[b:b + BLOCK]))
            q = pd.DataFrame(got).assign(git_commit=prov["commit"])
            tmp = part.with_suffix(".tmp")
            q.to_parquet(tmp, index=False)
            tmp.replace(part)
            print(f"{time.strftime('%H:%M:%S')} block {b} done, max F repro {q.F_repro_rel.max():.2e}", flush=True)
    A = pd.concat([pd.read_parquet(p) for p in sorted(CKPT.glob("block_*.parquet"))], ignore_index=True)
    assert len(A) == len(R)
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-68.2 L2 rise on mineral properties")
    A.assign(thermal_props="by_class (D-8), mineral", tag="first_order_fixed_solution",
             memory_sign=MEMORY_SIGN).to_parquet(DEST, index=False)
    print("rows", len(A), "max F repro", float(A.F_repro_rel.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
