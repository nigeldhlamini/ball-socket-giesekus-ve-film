"""R-67.2 steps 1 and 4 (D-8): the claim points re-solved exactly with mineral thermal properties.

Wrapper: scripts/bs_thermal_props.py (PropsSolver; ball_socket/solver.py and src/ untouched).
Each point is solved twice with identical settings: DEFAULT properties (src/thermal.py), which
must reproduce the stored row's h_min_bnd to 1e-9 relative (the wrapper test), and MINERAL
(rho 828.5, c_p 2000, k_f 0.15).  Full physics: viscoelastic, memory_sign "signed", clearance
closure on the bulk rise, production mesh 32x24.

Points (R-67.2 step 4): the E15 c0 axis at the argmax arm (box G's 13-row grid, production
root-finder, as box G ran; it includes the Fig. 5 point at c0 = 2.25 um), and the E13 argmax,
the E5 extremum and the feasibility-band margin row (run_R56_tolerance.cases(), tight 1e-10, as
R-58.2 ran them; stored rows: R58_richardson.parquet at 32x24).

Asserted on every re-solve routed to L2: the phi-spread of the L2 temperature field is below
1e-12 K (u is azimuthal and dT/dphi = 0, so the advection term vanishes and the L2 rise scales
as 1/k_f at fixed fields).
Registered (R-67.2 step 4): |dLambda| <= 0.005 at each point, and every verdict unchanged.

    python scripts/run_R67_D8_claims.py run [--workers 4]   # checkpointed; rerun resumes
    python scripts/run_R67_D8_claims.py report
"""
from __future__ import annotations

import argparse
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
CKPT = OUT / "R67_D8_claims_ckpt"
DEST = OUT / "R67_D8_claims.parquet"
REPORT = OUT / "R67_D8_claims.report.json"
FIELDS = OUT / "R67_D8_fields_mineral.parquet"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
MEMORY_SIGN = "signed"


def jobs():
    import run_R56_tolerance as T
    from ball_socket import mixed as mx
    g15 = json.loads((OUT / "boxG_E15.report.json").read_text(encoding="utf-8"))["argmax"]
    B = pd.read_parquet(OUT / "boxG_E15.parquet")
    Bg = B[(B.model_id == g15["arm"]) & (B.memory_sign == "signed") & (B.outcome == "converged")].sort_values("c0")
    sig = float(mx.expand_for_solving(mx.registered_contact_models(branch="psi", summit_convention="derived"))
                [int(g15["arm"])][0].sigma_star)
    pts = [dict(point=f"E15_c0_{1e6 * r.c0:g}", x={k: float(r[k]) for k in AXES}, model_id=int(r.model_id),
                flow_factors=None, sigma=sig, tight=False, stored_h=float(r.h_min_bnd),
                fig5=bool(np.isclose(1e6 * r.c0, g15["c0_um"]))) for _, r in Bg.iterrows()]
    R = pd.read_parquet(OUT / "R58_richardson.parquet")
    for c in T.cases():
        s = R[(R.case == c["case"]) & (R.src_index == c["src_index"]) & (R.Ntheta == 32)]
        if len(s) == 1:
            pts.append(dict(point=c["case"] if c["case"] != "band" else "margin", x=dict(c["x"]),
                            model_id=c["model_id"], flow_factors=c["flow_factors"], sigma=c["sigma"], tight=True,
                            stored_h=float(s.h_min_bnd.iloc[0]), fig5=False))
    assert len(pts) == len(Bg) + 3, len(pts)
    return [(p, props) for p in pts for props in ("default", "mineral")]


def job(args):
    p, props = args
    import warnings
    warnings.simplefilter("ignore")
    import bs_thermal_props as W
    from run_R57_mesh_error import _contact
    t0 = time.time()
    slv, ci = W.build(p["x"], _contact(p["model_id"]), p["flow_factors"],
                      thermal_props=W.MINERAL if props == "mineral" else None)
    slv.memory_sign = MEMORY_SIGN
    if p["tight"]:
        slv.brentq_xtol = slv.brentq_rtol = 1e-10
    try:
        perf = slv.solve_equilibrium(ci)
    except Exception as exc:                       # typed non-convergence: recorded, never dropped
        return dict(point=p["point"], props=props, outcome=type(exc).__name__, h_min_bnd=np.nan,
                    T_max_bnd=np.nan, De_sq_bnd=np.nan, eps_ecc=np.nan, wall_s=time.time() - t0)
    rec = dict(point=p["point"], props=props, outcome="converged", h_min_bnd=perf.h_min_bnd,
               T_max_bnd=perf.T_max_bnd, De_sq_bnd=perf.De_sq_bnd, eps_ecc=perf.eps_ecc,
               c_equilibrium=perf.c_equilibrium, T_bulk=perf.T_bulk, T_in=p["x"]["T_in"],
               thermal_route=perf.thermal_route, Lam=perf.h_min_bnd / p["sigma"], sigma=p["sigma"],
               stored_h=p["stored_h"], repro_rel=abs(perf.h_min_bnd / p["stored_h"] - 1),
               c0_um=1e6 * p["x"]["c0"], theta_min_deg=p["x"]["theta_min_deg"], fig5=p["fig5"],
               phi_spread_T=np.nan, wall_s=time.time() - t0)
    if perf.thermal_mode == "L2":
        T, sph = W.l2_field(slv, ci, perf)
        rec["phi_spread_T"] = W.phi_spread(T)
        if p["fig5"]:
            rec["profile_theta_deg"] = list(np.rad2deg(sph.mesh.theta))
            rec["profile_dT_K"] = list(T.mean(axis=1) - 273.15 - p["x"]["T_in"])
    return rec


def stage_run(workers):
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-67.2 D-8 claim points")
    CKPT.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs() if not (CKPT / f"{j[0]['point']}_{j[1]}.json").exists()]
    print(f"{len(todo)} solves to do", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job, j): j for j in todo}
        for f in as_completed(futs):
            r = f.result()
            r["git_commit"] = prov["commit"]
            part = CKPT / f"{r['point']}_{r['props']}.json"
            tmp = part.with_suffix(".tmp")
            tmp.write_text(json.dumps(r, default=float), encoding="utf-8")
            tmp.replace(part)
            print(f"{time.strftime('%H:%M:%S')} {r['point']} {r['props']} {r['outcome']} "
                  f"Lam={r.get('Lam')} {r['wall_s']:.0f}s", flush=True)
    return 0


def stage_report():
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-67.2 D-8 claim-point report")
    import bs_thermal_props as W
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CKPT.glob("*.json"))]
    assert len(rows) == len(jobs()), len(rows)
    prof = [r for r in rows if r["props"] == "mineral" and r.get("profile_dT_K")]
    D = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("profile_")} for r in rows])
    d, m = (D[D.props == k].set_index("point") for k in ("default", "mineral"))
    P = d[["Lam", "repro_rel", "T_bulk", "T_in", "c_equilibrium", "thermal_route"]].join(
        m[["Lam", "T_bulk", "c_equilibrium", "thermal_route", "phi_spread_T"]], lsuffix="_def", rsuffix="_min")
    P["dLam"] = P.Lam_min - P.Lam_def
    P["dc_nm"] = 1e9 * (P.c_equilibrium_min - P.c_equilibrium_def)
    P["verdict_def"], P["verdict_min"] = P.Lam_def >= 3, P.Lam_min >= 3
    l2 = D[D.thermal_route.astype(str).str.startswith("L") & D.phi_spread_T.notna()]
    rep = {"provenance": prov, "memory_sign": MEMORY_SIGN, "thermal_props": "by_class (D-8)",
           "mineral": W.MINERAL, "mineral_sources": W.MINERAL_SOURCES,
           "all_converged": bool((D.outcome == "converged").all()),
           "wrapper_test_repro_max_rel": float(d.repro_rel.max()),
           "wrapper_test_passes_1e-9": bool(d.repro_rel.max() <= 1e-9),
           "phi_spread_T_max_K": float(l2.phi_spread_T.max()) if len(l2) else None,
           "phi_spread_below_1e-12": bool(len(l2) == 0 or l2.phi_spread_T.max() < 1e-12),
           "max_abs_dLam": float(P.dLam.abs().max()), "max_abs_dLam_at": str(P.dLam.abs().idxmax()),
           "registered_dLam_le_0.005": bool(P.dLam.abs().max() <= 0.005),
           "verdicts_unchanged": bool((P.verdict_def == P.verdict_min).all()),
           "points": json.loads(P.reset_index().to_json(orient="records"))}
    if prof:
        f = prof[0]
        rep["fig5_rise_max_mK"] = 1e3 * float(max(f["profile_dT_K"]))
        from ball_socket.immutable import guarded_path
        guarded_path(str(FIELDS), what="R-67.2 Fig. 5 mineral rise profile")
        pd.DataFrame({"theta_deg": f["profile_theta_deg"], "dT_film_mean_K": f["profile_dT_K"]}).assign(
            git_commit=prov["commit"], thermal_props="mineral (D-8)", memory_sign=MEMORY_SIGN).to_parquet(FIELDS, index=False)
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-67.2 D-8 claim points")
    guarded_path(str(REPORT), what="R-67.2 D-8 report")
    D.assign(thermal_props=D.props.map({"default": "defaults (src/thermal.py)", "mineral": "by_class (D-8)"}),
             git_commit=prov["commit"]).to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "points"}, indent=1, default=float))
    print(P[["Lam_def", "Lam_min", "dLam", "dc_nm", "repro_rel", "thermal_route_def", "thermal_route_min"]].to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "report"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    return stage_run(min(a.workers, 4)) if a.stage == "run" else stage_report()


if __name__ == "__main__":
    raise SystemExit(main())
