"""R-51.4: re-adjudicate E5 from the E10 corner run, and record why that reuse is
legitimate.

REUSE LICENCE (written at the time, R-51.4): an existing artefact may discharge
a second expectation only if it was produced BEFORE, and INDEPENDENTLY OF, that
expectation's adjudication.  results/ch9/E10_capped_area_rated_corner.parquet was
run for E10 (capped-area fraction at the rated corner) at E5's point set; its
corner-feasibility content was incidental.  Nothing was selected: it is the only
run at this point set under the box-D solver.

THE TWO CHECKS R-51.4 REQUIRES, asserted here, not assumed:
  1. population: the E10 run's (grade, nu_100, c0, corner) set EQUALS E5's
     registered 160 points (e5_fluid_spec.build_e5_points), each with all 24 arms;
  2. film model: every row was solved on the SMOOTH film (R-42.3 scopes
     feasibility to the smooth solve).

AMENDED (R-52), two defects in this script's own first version, both found
while looking for a VG32 artefact:
  * check 1 compared LABEL sets only.  A run whose labels were attached by
    position to rows returned in completion order passes a set comparison with
    every label on the wrong row -- box B's e5_sweep.parquet is exactly that
    (ERR-26).  Check 1 now also asserts, row by row, that each label is the one
    its solve INPUTS imply (T_in <-> grade, eta_0 <-> (grade, nu_100),
    omega/p_ch <-> corner, c0 <-> c0_um).
  * Lambda used the E10 run's h_min, a cell-centre minimum -- the interior
    extremum box D sec 3.5 registered AGAINST (constraint_extremum_source =
    "boundary_corrected").  It now uses the registered reduction under axial
    eccentricity, h(theta_min) = c (1 - eps cos theta_min) (box D sec 3.3).
    Lambda_max moves 1.905 -> 1.877; the verdict only strengthens, because the
    interior value flattered feasibility.

WHAT IT CAN AND CANNOT ADJUDICATE.  E5: "at the rated maximum corner the
operating point lies outside the full-film feasible set WITH g1 BINDING",
reported as a function of nu_100 with the threshold above which it is refuted.
  * "outside the full-film feasible set": adjudicable -- a row is outside if it
    has no equilibrium or Lambda < 3 at its sigma* end.
  * "with g1 binding" (strict single-constraint): NOT adjudicable here -- the
    E10 run recorded neither T_max nor De_sq.  And for VG46, whose registered
    corner inlet is 110 C = the mineral g2 ceiling, any positive rise fails g2 by
    arithmetic, so sole-g1 binding is UNREACHABLE for every VG46 row (gate 6).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "results" / "ch9" / "E10_capped_area_rated_corner.parquet"
OUT = ROOT / "results" / "ch9" / "E5_readjudication_from_E10.json"


def main() -> int:
    from ball_socket.immutable import guarded_path
    from ball_socket import e5_fluid_spec as E5
    import ball_socket.campaign_config_v6 as v6
    from ball_socket.campaign_config_v4 import ConstraintSet
    guarded_path(str(OUT), what="E5 re-adjudication")
    ff = v6.config_dict()["regime_thresholds"]["full_film_min"]
    e = pd.read_parquet(SRC)
    reg = {(p.grade, round(p.nu_100, 9), p.c0_um, p.corner) for p in E5.build_e5_points()}
    got = {(int(r.grade), round(float(r.nu_100), 9), float(r.c0_um), r.corner)
           for r in e.itertuples()}
    if got != reg:
        raise SystemExit(f"check 1 FAILED: E10 population != E5's ({len(got)} vs {len(reg)})")
    if set(e.groupby(["grade", "nu_100", "c0_um", "corner"]).size()) != {24}:
        raise SystemExit("check 1 FAILED: not every E5 point carries all 24 arms")
    truth = {(round(p.omega_rad_s, 6), round(p.p_ch_bar * 1e5, 1), round(p.c0_um * 1e-6, 12),
              round(p.T_C, 6), float(f"{p.eta0:.9g}")): (p.grade, round(p.nu_100, 9), p.c0_um, p.corner)
             for p in E5.build_e5_points()}
    bad = 0
    for r in e.itertuples():
        k = (round(r.omega, 6), round(r.p_ch, 1), round(r.c0, 12), round(r.T_in, 6),
             float(f"{r.eta_0:.9g}"))
        bad += truth[k] != (int(r.grade), round(float(r.nu_100), 9), float(r.c0_um), r.corner)
    if bad:
        raise SystemExit(f"check 1 FAILED: {bad} rows carry labels their inputs contradict")
    if set(e.realised_film_model) != {"smooth"}:
        raise SystemExit(f"check 2 FAILED: film models {set(e.realised_film_model)}")
    m = e[e.corner == "maximum"].copy()
    # registered extremum (box D sec 3.5), not the run's cell-centre h_min
    m["h_min_bnd"] = m.c * (1.0 - m.eps * np.cos(np.deg2rad(m.theta_min_deg)))
    m["Lambda"] = m.h_min_bnd / m.sigma_star
    m["inside_full_film"] = (m.state == "equilibrium") & (m.Lambda >= ff)
    per_nu = []
    for (g, nu), grp in m.groupby(["grade", "nu_100"]):
        conv = grp[grp.state == "equilibrium"]
        per_nu.append({"grade": int(g), "nu_100": float(nu),
                       "points_inside_full_film": int(grp.inside_full_film.sum()),
                       "Lambda_max": float(conv.Lambda.max()) if len(conv) else None,
                       "c0_without_equilibrium_um": sorted(
                           set(grp.c0_um) - set(conv.c0_um))})
    refuted_at = [r for r in per_nu if r["points_inside_full_film"] > 0]
    mineral_ceiling = ConstraintSet().T_max_C_by_class["mineral"]
    rep = {
        "reuse_licence": ("produced before and independently of E5's box-D "
                          "adjudication (run for E10); nothing selected"),
        "check_1_population": ("E10 point set == E5 registered 160 points, 24 arms each; "
                               "every row's labels match the labels its inputs imply"),
        "extremum": "h_min_bnd = c (1 - eps cos theta_min), registered (box D sec 3.5)",
        "check_2_film_model": "smooth on every row",
        "clause_outside_full_film": {
            "verdict": "CONFIRMED at every swept nu_100" if not refuted_at else "REFUTED",
            "threshold_nu_100_above_which_refuted": None if not refuted_at else
            min(r["nu_100"] for r in refuted_at),
            "Lambda_max_maximum_corner": float(m[m.state == "equilibrium"].Lambda.max()),
            "per_grade_nu_100": per_nu},
        "clause_g1_binding": {
            "VG46": {"disposition": "ERR-25: STRUCK, not refuted and not open (R-52.5)",
                     "why": (f"registered corner inlet {E5.SERVICE_T[46]:g} C equals the "
                             f"mineral g2 ceiling {mineral_ceiling:g} C: any positive rise "
                             f"fails g2 by arithmetic, so no measurement can decide it")},
            "VG32": {"status": "OWED -- adjudicable, no admissible artefact (R-52.5)",
                     "headroom_K": mineral_ceiling - E5.SERVICE_T[32],
                     "why_not_this_artefact": "the E10 run recorded neither T_max nor De_sq"}},
    }
    OUT.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "clause_outside_full_film"}, indent=1))
    c = rep["clause_outside_full_film"]
    print("outside full film:", c["verdict"], "| Lambda_max", round(c["Lambda_max_maximum_corner"], 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
