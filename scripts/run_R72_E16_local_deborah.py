"""R-72.5, registration E16: the local Deborah number lambda |h_dot| / h at the rim.  No Reynolds solve.

v6 normalises De_sq = lambda max|h_dot| / c (solver.py:458-463, boundary-aware gap_rate_max_boundary);
Paper 1's expansion parameter is lambda h_dot / h.  The geometry is the solver's own: Spherical built
as solver._forward builds it (solver.py:241-247: e = (0, 0, -eps c), e_dot = (0, 0, -squeeze_phase
omega eps c), squeeze_phase = 1), the field lambda |gap_rate()| / gap() over the cell centres and,
in closed form, over both theta faces (as spherical.h_min_boundary / gap_rate_max_boundary do).  The
ratio is independent of c (h_dot and h both scale with it), so c = c0.

Population: the 137 in-domain E4 points (E4_lambda_sweep.parquet, in_g3_domain), each at its stored eps.
Registered (E16, R-72.5, frozen):
  (a) the maximum of lambda |h_dot| / h over the land lies on the rim (theta_min face) at every point
      and equals lambda omega eps cos(theta_min) / delta = De_sq / delta within 1 % (else STOP);
  (b) report median, upper quartile and maximum of the rim value and the % above De_sq_max.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
REPORT = OUT / "R72_E16.report.json"
DEST = OUT / "R72_E16.parquet"


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    from ball_socket.campaign_config_v4 import ConstraintSet
    from ball_socket.spherical import Spherical, SphericalConditions
    from ball_socket.geometry import BallSocketGeometry
    import ball_socket.campaign as C
    prov = require_clean_tree("R-72.5 E16 local Deborah")
    cs = ConstraintSet()
    S = pd.read_parquet(OUT / "E4_lambda_sweep.parquet")
    S = S[S.in_g3_domain].reset_index(drop=True)
    rows = []
    for _, r in S.iterrows():
        tmin = np.deg2rad(r.x_theta_min_deg)
        geom = BallSocketGeometry(c0=r.x_c0, theta_min=tmin)
        c, eps, om = float(r.x_c0), float(r.eps), float(r.x_omega)
        cond = SphericalConditions(p_ch=r.x_p_ch, p_case=C.P_CASE_PA, e_vec=(0.0, 0.0, -eps * c),
                                   e_dot=(0.0, 0.0, -C.SQUEEZE_PHASE * om * eps * c), omega_b=(0.0, 0.0, C.SPIN_RATIO * om))
        sph = Spherical(geom.R_b, c, geom.theta_max, cond, 32, 24, theta_min=tmin)
        cell = r.lam * np.abs(sph.gap_rate()) / sph.gap()
        ex, ey, ez = sph.e_vec
        dx, dy, dz = sph.e_dot
        faces = []
        for nx, ny, nz in sph._face_normals():
            faces.append(float(np.max(r.lam * np.abs(dx * nx + dy * ny + dz * nz) / (c + ex * nx + ey * ny + ez * nz))))
        tmin_face, tmax_face = faces[0], faces[1]
        cell_max_at_rim_cell = bool(np.unravel_index(np.argmax(cell), cell.shape)[0] == 0)
        mx = max(float(cell.max()), *faces)
        delta = 1.0 - eps * np.cos(tmin)
        analytic = r.lam * om * eps * np.cos(tmin) / delta
        rows.append(dict(base=r.base, lam=r.lam, eps=eps, theta_min_deg=r.x_theta_min_deg, delta=delta,
                         rim_face=tmin_face, mouth_face=tmax_face, cell_max=float(cell.max()), field_max=mx,
                         max_on_rim=bool(np.isclose(mx, tmin_face) and tmin_face >= tmax_face),
                         cell_max_at_rim_cell=cell_max_at_rim_cell, analytic=analytic,
                         dev_analytic=abs(mx / analytic - 1), De_sq_bnd=r.De_sq_bnd,
                         dev_stored=abs(mx / (r.De_sq_bnd / delta) - 1)))
    T = pd.DataFrame(rows)
    a_ok = bool(T.max_on_rim.all() and (T.dev_analytic <= 0.01).all() and (T.dev_stored <= 0.01).all())
    loc = T.field_max
    rep = {"provenance": prov, "n": int(len(T)),
           "a_max_on_rim_all": bool(T.max_on_rim.all()), "a_cell_max_at_rim_cell_all": bool(T.cell_max_at_rim_cell.all()),
           "a_max_dev_analytic": float(T.dev_analytic.max()), "a_max_dev_stored_De_sq_over_delta": float(T.dev_stored.max()),
           "registered_a_holds": a_ok, "STOP": not a_ok,
           "b_median": float(loc.median()), "b_q3": float(loc.quantile(0.75)), "b_max": float(loc.max()),
           "b_above_De_sq_max_pct": 100 * float((loc > cs.De_sq_max).mean()), "De_sq_max": cs.De_sq_max,
           "De_sq_median_for_reference": float(T.De_sq_bnd.median())}
    from ball_socket.immutable import guarded_path
    guarded_path(str(REPORT), what="R-72.5 E16")
    guarded_path(str(DEST), what="R-72.5 E16 rows")
    T.assign(git_commit=prov["commit"]).to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))
    return 0 if a_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
