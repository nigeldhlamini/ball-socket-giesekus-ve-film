"""R-67.2 step 5 (D-8): the campaign's row-wise first-order bound on Lambda.  No re-solve.

At fixed load, with the clearance moved by the property change Delta c = Delta alpha R_b DeltaT (r - 1)
(ball_socket/geometry.py:137-146; the rise is the row's stored dissipative T_bulk - T_in):

    W(delta, c) = A(delta)/c^2 + B(delta)        A: squeeze, B: land hydrostatic (pocket, case)
    d ln delta / d ln c = 2 (A/c^2) / (delta dW/ddelta)       (dW/ddelta < 0)
    d ln Lambda / d ln c = 1 + d ln delta / d ln c             (Lambda = delta c / sigma*)
    Delta Lambda = Lambda (d ln Lambda / d ln c) Delta c / c

which is R-67.2's (1 + 2 d ln Phi / d ln S) Delta c / c with the pocket kept.  A and B are the
vendored quadrature's (scripts/p3_rim_asymptote.load_solver_kinematics: the solver's squeeze
rate eps_dot = omega eps, the pocket at p_ch, the case pressure), evaluated at the row's own
(delta, theta_min) with eta = eta_G(Wi_char) (R64_wichar.parquet, the solver's viscometric
viscosity at the row's characteristic shear); only the A/B split depends on eta.

Mineral ratio r (R-67.2 step 2): L1 rows r = rho c_p,def / rho c_p,min = 4.0e6 / (828.5 x 2000) = 2.41;
L2 rows r = k_def / k_min = 0.60/0.15 = 4.0.  A conservative variant takes r = 4 on every row
(a row may re-route to L2 under the mineral properties).
A row is AT RISK if |Delta Lambda| >= |Lambda - 3| at either sigma* end, or its projected delta
falls below the 0.01 floor.

Class 4 (no equilibrium above the floor): at fixed delta the squeeze capacity A/c^2 falls as c
grows and B does not depend on c, so a larger clearance cannot create an equilibrium.  Asserted
row-wise: A(0.01) > 0 for every class-4 row, and d ln Lambda / d ln c < 0 at the floor reported.
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
SRC = OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet"
DEST = OUT / "R67_D8_bound.parquet"
REPORT = OUT / "R67_D8_bound.report.json"
DALPHA, R_B, FLOOR = 6e-6, 6.5e-3, 0.01
R_L1, R_L2 = 4.0e6 / (828.5 * 2000.0), 4.0


def slope(delta, tmin, eta, c, omega, p_pocket, p_case, rel=1e-4):
    import p3_rim_asymptote as RA
    kw = dict(eta=eta, R=R_B, c=c, omega=omega, p_pocket=p_pocket, p_case=p_case)
    W = lambda d: RA.load_solver_kinematics(d, tmin, **kw)
    sq = lambda d: RA.load_solver_kinematics(d, tmin, **{**kw, "p_pocket": 0.0, "p_case": 0.0})
    dW = (W(delta * (1 + rel)) - W(delta * (1 - rel))) / (2 * delta * rel)
    dlnd = 2.0 * sq(delta) / (delta * dW)
    return 1.0 + dlnd, dlnd, sq(delta)


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    import ball_socket.campaign as C
    prov = require_clean_tree("R-67.2 step 5 D-8 bound")
    D = pd.read_parquet(SRC)
    W = pd.read_parquet(OUT / "R64_wichar.parquet").set_index("sample_id")
    cv = D[D.outcome == "converged"].join(W[["eta_G_char"]], on="sample_id").copy()
    assert cv.eta_G_char.notna().all()
    out = []
    for _, r in cv.iterrows():
        delta = r.h_min_bnd / r.c_equilibrium
        s_L, s_d, _ = slope(delta, r.theta_min_deg, r.eta_G_char, r.c_equilibrium, r.omega, r.p_ch, C.P_CASE_PA)
        dT = r.T_bulk - r.T_in
        rr = R_L1 if r.thermal_route == "L1" else R_L2
        rec = dict(sample_id=int(r.sample_id), delta=delta, dlnLam_dlnc=s_L, dlndelta_dlnc=s_d, dT=dT,
                   route=r.thermal_route, r=rr)
        for tag, rv in (("", rr), ("_cons", R_L2)):
            dcc = DALPHA * R_B * dT * (rv - 1.0) / r.c_equilibrium
            d_proj = delta * (1.0 + s_d * dcc)
            risk = d_proj < FLOOR
            for end in ("lo", "hi"):
                lam = r[f"lambda_ratio_{end}"]
                dl = lam * s_L * dcc
                rec[f"dLam_{end}{tag}"] = dl
                risk = risk or (abs(dl) >= abs(lam - 3.0))
            rec[f"dc_over_c{tag}"] = dcc
            rec[f"delta_proj{tag}"] = d_proj
            rec[f"at_risk{tag}"] = bool(risk)
        out.append(rec)
    T = pd.DataFrame(out)
    nc = D[D.outcome != "converged"]
    fl = [slope(FLOOR, r.theta_min_deg, r.eta_0, r.c0, r.omega, r.p_ch, C.P_CASE_PA) for _, r in nc.iterrows()]
    A_pos = all(f[2] > 0 for f in fl)
    s_floor = np.array([f[0] for f in fl])
    rate = None
    ck = OUT / "R67_D8_claims_ckpt"
    if ck.exists():
        w = [json.loads(p.read_text(encoding="utf-8"))["wall_s"] for p in ck.glob("*.json")]
        rate = float(np.mean(w)) if w else None
    rep = {"provenance": prov, "thermal_props": "by_class (D-8), mineral", "n_converged": len(T),
           "ratio_L1": R_L1, "ratio_L2": R_L2,
           "at_risk": int(T.at_risk.sum()), "at_risk_conservative": int(T.at_risk_cons.sum()),
           "max_abs_dLam": float(T[["dLam_lo", "dLam_hi"]].abs().max().max()),
           "max_abs_dLam_conservative": float(T[["dLam_lo_cons", "dLam_hi_cons"]].abs().max().max()),
           "median_abs_dLam_lo": float(T.dLam_lo.abs().median()),
           "projected_delta_below_floor": int((T.delta_proj < FLOOR).sum()),
           "class4_rows": len(nc), "class4_A_positive_all": bool(A_pos),
           "class4_dlnLam_dlnc_at_floor_max": float(s_floor.max()),
           "class4_slope_factor_below_minus_half_all": bool((s_floor < -0.5).all()),
           "class4_slope_ge_minus_half_n": int((s_floor >= -0.5).sum()), "class4_slope_positive_n": int((s_floor > 0).sum()),
           "measured_solve_rate_s": rate,
           "resolve_estimate_h_4_workers": (None if rate is None else rate * int(T.at_risk.sum()) / 4 / 3600),
           "resolve_estimate_h_4_workers_conservative": (None if rate is None else
                                                          rate * int(T.at_risk_cons.sum()) / 4 / 3600)}
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-67.2 D-8 bound")
    guarded_path(str(REPORT), what="R-67.2 D-8 bound report")
    T.assign(git_commit=prov["commit"]).to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
