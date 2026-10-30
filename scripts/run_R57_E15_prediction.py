"""R-55.7 / R-57.7 step 8: E15's prediction, from the quadrature plus the thermal
closure, BEFORE any box-G solve.  No solver call is made here.

Condition: E13's registered corner and admissible fluid (E13_spec_v2.json), theta_min
= 20 deg, both sigma* ends.  Load balance (A5, the solver's kinematics eps_dot = omega
eps, pocket at p_ch, case at P_CASE):
    W = p_ch A_p - p_ch pi R^2 sin^2 theta_min = load_solver_kinematics(delta; c, eta_0)
Thermal closure (the code's): c = c0 + Dalpha R_b dT, dT the film-mean rise.  On the
conduction route, uniform spin-Couette dissipation gives dT = eta (omega R sin t)^2 /
(12 k_f), INDEPENDENT of h; its land maximum is at theta_max.  Newtonian at eta_0 (no
shear-thinning): the quadrature's assumption, stated.
Lambda(c0) = delta(c0) c / sigma*.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
REPORT = OUT / "R57_E15_prediction.json"


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")
    import p3_rim_asymptote as RA
    import ball_socket.campaign as C
    from ball_socket import representative as rep
    from ball_socket.geometry import delta_alpha_th
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M
    from src.thermal import ThermalProperties
    spec = json.loads((OUT / "E13_spec_v2.json").read_text(encoding="utf-8"))
    eta, om, p = spec["fluid"]["eta_0_Pa_s"], spec["corner"]["omega_rad_s"], spec["corner"]["p_ch_Pa"]
    R, tm = rep.R_B_M, 20.0
    W = p * rep.A_P_M2 - p * np.pi * R ** 2 * np.sin(np.deg2rad(tm)) ** 2
    th = ThermalProperties()
    dT = eta * (om * R * np.sin(np.deg2rad(73.0))) ** 2 / (12 * th.k_f)
    dc = delta_alpha_th() * R * dT
    c0s = np.round(np.concatenate([np.arange(0.1, 2.0, 0.05), np.arange(2.0, 8.01, 0.5)]), 4)
    rows = []
    for c0 in c0s:
        c = c0 * 1e-6 + dc
        d, ok = RA.delta_for_load(W, tm, eta=eta, R=R, c=c, omega=om, p_pocket=p, p_case=C.P_CASE_PA)
        rows.append(dict(c0_um=float(c0), c_um=c * 1e6, delta=d, bracketed=ok,
                         **{f"Lambda_{b}": d * c / s for b, s in zip(("lo", "hi"), SIGMA_STAR_BAND_M)}))
    lam = np.array([r["Lambda_lo"] for r in rows])
    i = int(lam.argmax())
    # quadratic refinement of the peak on its three neighbours
    x, y = c0s[i - 1:i + 2], lam[i - 1:i + 2]
    a, b, _ = np.polyfit(x, y, 2)
    c0_star = float(-b / (2 * a))
    meas = json.loads((OUT / "E13_extremum.report.json").read_text())["per_c0"]
    calib = {k: {"measured_Lambda_max": v["Lambda_max"],
                 "predicted_Lambda_lo": next(r["Lambda_lo"] for r in rows if abs(r["c0_um"] - float(k)) < 1e-9)}
             for k, v in meas.items() if any(abs(r["c0_um"] - float(k)) < 1e-9 for r in rows)}
    rep_ = {"condition": {"eta_0": eta, "omega": om, "p_ch": p, "theta_min_deg": tm, "W_N": W,
                          "spec": "E13_spec_v2.json"},
            "thermal_closure": {"dT_film_mean_max_K": dT, "dc_um": dc * 1e6,
                                "note": ("h-independent on the conduction route; at this corner it opens the "
                                         "clearance by a few nm, so it does NOT bound the optimum from below "
                                         "-- the review's mechanism for the lower side is not this code's")},
            "predicted": {"c0_star_um": c0_star, "Lambda_star_lo": float(lam.max()),
                          "Lambda_star_hi": float(rows[i]["Lambda_hi"])},
            "curve": rows,
            "post_hoc_comparison_with_E13_measured": calib}
    REPORT.write_text(json.dumps(rep_, indent=2, default=float))
    print(json.dumps({k: v for k, v in rep_.items() if k != "curve"}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
