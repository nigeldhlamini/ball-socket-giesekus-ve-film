"""R-48.6: separate AREA from the POCKET BOUNDARY as the cause of the sinking.

R-47.7's data showed Lambda falls with theta_min because the equilibrium
eccentricity rises (the ball sinks deeper), overwhelming the rim's own geometry,
which works FOR the film.  Open question: is the deeper sinking explained by the
loss of bearing area alone, or is the pressure field degrading faster than the
area -- i.e. is the pocket-pressure (Dirichlet) boundary contributing?

THE TEST, and what a failing result looks like (R-48.4: state it first).
Evaluate the FILM load F_film at a COMMON eccentricity eps across the theta_min
sweep -- one Reynolds solve per point, no equilibrium search -- and compare
F_film(theta)/F_film(20) against the land-area ratio A(theta)/A(20):

  * ratio of ratios ~ 1  -> area explains the sinking;
  * ratio of ratios < 1  -> load capacity falls FASTER than area: the pressure
                            field is degrading beyond area loss;
  * ratio of ratios > 1  -> slower than area.

Either outcome is possible, so the test can fail in either direction.

A CONFOUND IN THE TEST AS PROPOSED, and the second mode that removes it.  At a
common eps, raising theta_min moves the rim to a THICKER gap: h_min = c (1 - eps
cos theta_min) rises.  So "common eps" is not "common film", and part of
F_film's fall is the rim-geometry channel R-47 already isolated.  Mode
`common_psi` holds the RIM GAP fixed instead -- psi = eps cos theta_min, i.e.
common h_min -- so the only thing that changes along the sweep is the land
itself.  Both modes are reported; `common_psi` is the one that separates area
from boundary.

Both area conventions are reported, because they are different quantities and
this script must not pick one (rule 18): PROJECTED  pi R_b^2 (sin^2 th_max -
sin^2 th_min) -- the plane the load acts on -- and SPHERICAL 2 pi R_b^2
(cos th_min - cos th_max) -- the surface the film occupies.

Conditions: the E13 v2 spec's corner and fluid (the admissible corner fluid),
pure film (contact=None: F_film at fixed eps and c does not depend on the
contact model), cold clearance c = c0 (the equilibrium clearance moved <= 0.07 %
across the sweep, R-47 hand-back sec 5), the campaign's solver settings.
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
OUT = ROOT / "results" / "ch9" / "R48_film_at_common_eps.parquet"

THETAS = np.linspace(20.0, 40.0, 11)
EPS = (0.1, 0.3, 0.5, 0.7, 0.9)        # all < 1/cos(20 deg) = 1.064: film open
PSI = (0.1, 0.3, 0.5, 0.7, 0.9)        # psi = eps cos(theta_min) = 1 - h_min/c
C0S = (2e-6, 10e-6)


def main() -> int:
    from ball_socket.immutable import guarded_path
    import ball_socket.campaign as C
    from ball_socket.geometry import BallSocketGeometry
    from ball_socket.solver import BallSocketSolver, CampaignInput, SolverConfig
    import run_E13_extremum as E13
    guarded_path(str(OUT), what="R-48.6 film load at common eps")
    spec, h = E13._load_spec()
    f = spec["fluid"]
    rows = []
    t0 = time.time()
    for c0 in C0S:
        for th in THETAS:
            geom = BallSocketGeometry(c0=c0, theta_min=np.deg2rad(th))
            slv = BallSocketSolver(
                geom, model="viscoelastic", Ntheta=E13.GRID[0], Nphi=E13.GRID[1],
                alpha=C.ALPHA_FIXED, squeeze_phase=C.SQUEEZE_PHASE,
                spin_ratio=C.SPIN_RATIO,
                config=SolverConfig(include_normal_stress=False, max_iter=80,
                                    tol=1e-9, omega=0.7),
                eps_cap=float(C._registered_eps_cap(geom.theta_min)),
                contact=None)
            ci = CampaignInput(omega=spec["corner"]["omega_rad_s"],
                               p_ch=spec["corner"]["p_ch_Pa"], c0=c0,
                               T_in=f["T_in_C"], eta_0=f["eta_0_Pa_s"],
                               lam=f["lam_s"], eta_s_over_eta0=f["eta_s_ratio"])
            fluid = slv._fluid(ci)
            tmax = geom.theta_max
            tmin = geom.theta_min
            for mode, level, eps in ([("common_eps", e, e) for e in EPS]
                                     + [("common_psi", q, q / np.cos(tmin))
                                        for q in PSI]):
                F_film, _, _ = slv._load_split(ci, eps, fluid, c0)
                rows.append({"c0_um": c0 * 1e6, "theta_min_deg": float(th),
                             "mode": mode, "level": level,
                             "eps": float(eps), "F_film": float(F_film),
                             "A_proj": float(np.pi * geom.R_b ** 2
                                             * (np.sin(tmax) ** 2 - np.sin(tmin) ** 2)),
                             "A_sph": float(2 * np.pi * geom.R_b ** 2
                                            * (np.cos(tmin) - np.cos(tmax))),
                             "W_hyd": float(slv._applied_load(ci)
                                            - slv._pocket_relief(ci))})
    df = pd.DataFrame(rows)
    df["spec_sha256"] = h
    df.to_parquet(OUT, index=False)
    print(f"{len(df)} fixed-eps film evaluations in {time.time() - t0:.0f} s")

    ref = df[df.theta_min_deg == 20.0].set_index(["c0_um", "mode", "level"])
    j = df.join(ref[["F_film", "A_proj", "A_sph", "W_hyd"]],
                on=["c0_um", "mode", "level"], rsuffix="_20")
    j["F_rel"] = j.F_film / j.F_film_20
    j["Aproj_rel"] = j.A_proj / j.A_proj_20
    j["Asph_rel"] = j.A_sph / j.A_sph_20
    j["W_rel"] = j.W_hyd / j.W_hyd_20
    j["F_over_Aproj"] = j.F_rel / j.Aproj_rel
    j["F_over_Asph"] = j.F_rel / j.Asph_rel
    end = j[j.theta_min_deg == 40.0]
    rep = {"spec_sha256": h, "evaluations": len(df),
           "at_40_deg_relative_to_20": end[["c0_um", "mode", "level", "F_rel", "Aproj_rel",
                                            "Asph_rel", "W_rel", "F_over_Aproj",
                                            "F_over_Asph"]].round(4)
           .to_dict(orient="records")}
    OUT.with_suffix(".report.json").write_text(json.dumps(rep, indent=2))
    print(end[["c0_um", "mode", "level", "F_rel", "Aproj_rel", "Asph_rel", "W_rel",
               "F_over_Aproj", "F_over_Asph"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
