"""R-64.1 step 1: the characteristic Weissenberg number at every converged pure-film row of box D.

Definition (quoted, scripts/run_R50_E4_lambda_sweep.py:117): Wi_char = lam * max(gdot), with gdot
the solver's gap-averaged shear rate (ball_socket/spherical.py:673-687: Couette from the ball's
surface velocity, sqrt(u_th^2 + u_ph^2)/h, plus Poiseuille h |grad_s p| / (2 eta_bar)) on the
converged state.  The spin is omega_b = spin_ratio * omega about the load axis
(ball_socket/solver.py:243), spin_ratio = 1.0 (campaign_config_v4.py:90).

The viscometric viscosity is the solver's own: GiesekusFluid.viscosity (src/rheology.py:185,
Eq. 45), the function the Picard loop applies at gdot (ball_socket/reynolds.py:93-95).
eta_G(Wi_char) = fluid.viscosity(max gdot).

Per row: the solver built exactly as campaign._evaluate_x_inner builds it for the pure film
(scripts/run_R57_mesh_error._solver, contact None), MODULUS memory (box D's rows predate D-6),
one forward solve at the stored eps_ecc and c_equilibrium (no equilibrium search).  The load must
reproduce the stored F to 1e-9 relative.  Checkpointed in blocks; rerunning resumes.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
SRC = OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet"
CKPT = OUT / "R64_wichar_ckpt"
DEST = OUT / "R64_wichar.parquet"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
BLOCK = 200
MEMORY_SIGN = "modulus"


def job(rec):
    import warnings
    warnings.simplefilter("ignore")
    from run_R57_mesh_error import _solver
    t0 = time.time()
    slv, ci = _solver({k: float(rec[k]) for k in AXES}, None, None)
    slv.memory_sign = MEMORY_SIGN
    fl = slv._fluid(ci)
    res, sph = slv._forward(ci, float(rec["eps_ecc"]), fl, float(rec["c_equilibrium"]))
    gd = np.asarray(sph.gap_averaged_shear_rate(res.pressure, res.eta_bar))
    F = float(sph.load_capacity(res.pressure))
    gmax = float(gd.max())
    return {"sample_id": int(rec["sample_id"]), "Wi_char": float(rec["lam"]) * gmax, "gdot_max": gmax,
            "eta_G_char": float(np.asarray(fl.viscosity(gmax))), "eta_s": float(fl.eta_s), "eta_0": float(fl.eta_0),
            "F_fwd": F, "F_repro_rel": abs(F / float(rec["F"]) - 1), "wall_s": time.time() - t0}


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-64.1 Wi_char runner")
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-64.1 Wi_char")
    D = pd.read_parquet(SRC)
    D = D[D.outcome == "converged"].sort_values("sample_id").reset_index(drop=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as ex:
        for b in range(0, len(D), BLOCK):
            part = CKPT / f"block_{b:05d}.parquet"
            if part.exists():
                continue
            got = list(ex.map(job, D.iloc[b:b + BLOCK].to_dict(orient="records")))
            q = pd.DataFrame(got)
            q["git_commit"] = prov["commit"]
            tmp = part.with_suffix(".tmp")
            q.to_parquet(tmp, index=False)
            tmp.replace(part)
            print(f"block {b} done", flush=True)
    W = pd.concat([pd.read_parquet(p) for p in sorted(CKPT.glob("block_*.parquet"))], ignore_index=True)
    assert len(W) == len(D) and W.sample_id.is_unique
    W["memory_sign_fields"] = MEMORY_SIGN
    W.to_parquet(DEST, index=False)
    print("rows", len(W), "max F repro", float(W.F_repro_rel.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
