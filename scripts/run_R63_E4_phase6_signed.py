"""R-63.1: the six phase-6 decomposition points (boxc_phase6_decomposition.json, the source of
\\BsNineMemShareMin/Max) on the SIGNED memory path.  Six equilibria; minutes.

Each point is rebuilt exactly as boxc_decomposition.decompose() built it (_build(x, True), its
own eps_cap) and solved twice:
  * modulus: the D1 frozen-coefficient share must reproduce the stored phase-6 value to 1e-9
    relative (like-for-like check), else the point is reported and not used;
  * signed: the same construction with memory_sign = "signed" (D-6 fixed), D1 share.
Expected (R-63.1): every share changes sign, magnitude within 1 % of the stored value.
"""
from __future__ import annotations

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
SRC = OUT / "boxc_phase6_decomposition.json"
DEST = OUT / "R63_E4_phase6_signed.parquet"
REPRO = 1e-9


def job(p):
    import warnings
    warnings.simplefilter("ignore")
    from scipy.sparse.linalg import spsolve
    import boxc_decomposition as B
    out = {"point_tag": p["point_tag"], "lam": float(p["inputs"]["lam"]),
           "share_stored": float(p["D1_frozen"]["share_memory"])}
    t0 = time.time()
    for sign in ("modulus", "signed"):
        slv, ci = B._build(p["inputs"], True)
        slv.memory_sign = sign
        sph, fluid, eta_bar, gdot, pressure, perf = B._state(slv, ci)

        def F(mem):
            s = sph.assemble("viscoelastic", fluid, eta_bar, gdot, pressure=pressure, include_memory=mem,
                             include_normal_stress=True, n1_scale=1.0, hoop_scale=1.0)
            return float(sph.load_capacity(sph.to_2d(spsolve(s.A, s.b))))
        Fa, Fn = F(True), F(False)
        out[f"share_{sign}"] = (Fa - Fn) / Fa
        out[f"eps_{sign}"] = float(perf.eps_ecc)
    out["repro_rel"] = abs(out["share_modulus"] / out["share_stored"] - 1)
    out["reproduced"] = bool(out["repro_rel"] <= REPRO)
    out["flipped"] = bool(np.sign(out["share_signed"]) == -np.sign(out["share_stored"]))
    out["rel_mag"] = abs(abs(out["share_signed"]) / abs(out["share_stored"]) - 1)
    out["wall_s"] = time.time() - t0
    return out


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-63.1 phase-6 signed")
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-63.1 phase-6 signed")
    pts = json.loads(SRC.read_text())["points"]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as ex:
        D = pd.DataFrame(list(ex.map(job, pts)))
    D["git_commit"] = prov["commit"]
    D.to_parquet(DEST, index=False)
    print(D[["point_tag", "lam", "share_stored", "share_modulus", "share_signed", "repro_rel", "flipped",
             "rel_mag"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
