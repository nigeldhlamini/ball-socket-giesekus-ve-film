"""R-57.3: the thermal constraint is a film-MEAN temperature; E3 by route; through-flow
group; Nahme.  Deterministic post-processing of box E's corrected artefact, no solves.

What the code computes (quoted in reports/r56_handback.md sec 1.3):
  L2 routes  T - T_in = Phi_v h^2 / (12 k_f): the CROSS-FILM MEAN rise of the profile
             between two isothermal walls at T_in, maximised over the land (T_max_bnd).
  L1 route   T - T_in = P / (rho c_p Q): the bulk advective mixed-mean, i.e. all
             dissipated heat leaves with the outflow -- the adiabatic-wall limit FOR THE
             MEAN.  No in-film peak is computed on either route.

Factors against the code's conduction rise (uniform, spin-Couette dissipation):
  x1    the mean (what g2 was evaluated on)
  x1.5  isothermal-wall PEAK      (q h^2/8k over q h^2/12k)
  x6    one-adiabatic-wall PEAK   (q h^2/2k over q h^2/12k)
Applied to the conduction-routed (L2) E3 rows only; bulk-routed rows are reported at
the mean, which is already their adiabatic limit.

Through-flow group (thermal.py:37-43): Theta = P_loss / (rho c_p Q_out dT_safety),
dT_safety = 10 K; the "escalation" to L2 fires iff Q_out <= Q_min, i.e. Theta >= 1
(or Q_out <= 0).  Its distribution over the corrected campaign is the S1 statement.

Nahme: Na = b dT, b = -d ln eta / dT.  The constants dossier carries no viscosity-
temperature coefficient, so Na is reported PER UNIT b (i.e. as dT in K), with the band
of b from the code's CITED law (ASTM D341 Walther on ISO 3448 VG32/VG46, nu_100 swept
4.5-9 mm^2/s, e5_fluid_spec.py) evaluated at each row's inlet as context.  Columns:
mean rise and x6.  (The classical Nahme-Griffith group b eta U^2 / k equals 12 b dT_mean
for Couette heating.)

E3 frame (as R-49.2 / R-54): converged rows, T_in < 60 C, g1 (Lambda >= 3 at the
sigma* end) and g3 (De_sq_bnd <= 0.2) satisfied; sole-g2 means T > 60 C.
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
SRC = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
REPORT = OUT / "R57_thermal_by_route.report.json"
FACTORS = (1.0, 1.5, 6.0)
WG = 60.0


def walther_b_band(T_C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from ball_socket import e5_fluid_spec as E5
    grid = np.arange(15.0, 115.01, 0.5)
    h = 0.01
    bs = []
    for g in (32, 46):
        for nu100 in E5.NU100_SWEEP:
            lo = np.array([E5.eta_at(E5.ISO_VG[g], nu100, t - h) for t in grid])
            hi = np.array([E5.eta_at(E5.ISO_VG[g], nu100, t + h) for t in grid])
            bs.append(-(np.log(hi) - np.log(lo)) / (2 * h))
    bs = np.array(bs)
    return (np.interp(T_C, grid, bs.min(axis=0)), np.interp(T_C, grid, bs.max(axis=0)))


def main() -> int:
    from ball_socket.campaign_config_v4 import SIGMA_STAR_BAND_M as S, ConstraintSet
    from ball_socket import thermal as bt
    from src.thermal import ThermalProperties
    cs = ConstraintSet()
    th = ThermalProperties()
    cols = ["sample_id", "model_id", "outcome", "T_in", "T_max_bnd", "h_min_bnd", "De_sq_bnd",
            "thermal_route", "P_loss", "Q_leak"]
    E = pd.read_parquet(SRC, columns=cols)
    c = E[E.outcome == "converged"].copy()
    c["rise"] = c.T_max_bnd - c.T_in
    c["conduction"] = c.thermal_route.isin(["L1->L2_qmin", "L2_brinkman"])
    rep = {"population": "box E mixed, corrected, converged rows", "n_converged": int(len(c)),
           "routes": c.thermal_route.value_counts().to_dict(), "E3_by_route": {}}
    assert (c.rise >= -1e-9).all(), "rise must be non-negative"
    for b, s in (("lo", S[0]), ("hi", S[1])):
        fr = c[(c.T_in < WG) & (c.h_min_bnd / s >= cs.lambda_ratio_min) & (c.De_sq_bnd <= cs.De_sq_max)]
        out = {"frame_rows": int(len(fr))}
        for name, sub in (("conduction", fr[fr.conduction]), ("bulk", fr[~fr.conduction])):
            facs = FACTORS if name == "conduction" else (1.0,)
            d = {}
            for k in facs:
                brk = sub[sub.T_in + k * sub.rise > WG]
                d[f"x{k:g}"] = {"sole_g2_rows": int(len(brk)), "sole_g2_samples": int(brk.sample_id.nunique()),
                                "smallest_breaching_inlet_C": (float(brk.T_in.min()) if len(brk) else None),
                                "max_rise_K": (float(k * brk.rise.max()) if len(brk) else None)}
            d["rows_in_route"] = int(len(sub))
            out[name] = d
        rep["E3_by_route"][b] = out
    rep["E3_note"] = ("bulk-routed rows: mean only -- it is already the adiabatic-wall limit for "
                      "the mean; the in-film peak is not computed on that route")
    # through-flow group
    rho_cp = th.rho * th.c_p
    Q = c.Q_leak.astype(float)
    theta = c.P_loss / (rho_cp * Q.clip(lower=1e-30) * bt.DELTA_T_SAFETY_K)
    theta[Q <= 0] = np.inf
    rep["through_flow_group"] = {
        "definition": "Theta = P_loss / (rho c_p Q_leak * 10 K); L2 escalation iff Theta >= 1 or Q <= 0 (thermal.py:37-43)",
        "rho_c_p": rho_cp,
        "share_Theta_ge_1": float((theta >= 1).mean()), "share_Q_le_0": float((Q <= 0).mean()),
        "quantiles": {q: float(np.quantile(theta[np.isfinite(theta)], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "consistency_with_route": float(((theta >= 1) == c.thermal_route.isin(["L1->L2_qmin"])).mean())}
    # Nahme per unit b, and the Walther b band at each inlet as context
    ind = c[c.De_sq_bnd <= cs.De_sq_max]
    b_lo, b_hi = walther_b_band(ind.T_in.values)
    na = {}
    for k in (1.0, 6.0):
        dT = k * ind.rise.values
        na[f"x{k:g}"] = {"dT_K_quantiles(=Na/b)": {q: float(np.quantile(dT, q)) for q in (0.5, 0.95, 0.99, 1.0)},
                         "share_Na_gt_1_at_walther_b_max": float((b_hi * dT > 1).mean()),
                         "share_Na_gt_1_at_walther_b_min": float((b_lo * dT > 1).mean())}
    rep["nahme"] = {"population": "converged rows inside the g3 domain", "n": int(len(ind)),
                    "walther_b_per_K_range_over_rows": [float(b_lo.min()), float(b_hi.max())],
                    "source_of_b": "ASTM D341 (Walther) on ISO 3448 VG32/46, nu_100 4.5-9 mm^2/s (e5_fluid_spec.py); NOT a dossier constant",
                    **na}
    from ball_socket.immutable import guarded_path
    guarded_path(str(REPORT), what="R-57.3 thermal by route")
    REPORT.write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps(rep, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
