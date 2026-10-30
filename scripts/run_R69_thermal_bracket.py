"""R-69.3: every thermal statement at both ends of the wall-condition bracket, per fluid class.

The two routes are two wall idealisations (R-69.1): the ADVECTIVE end puts all the dissipated heat
into the outflow (adiabatic walls), rise = P_diss / (rho c_p Q_out) (thermal.py:31-34; exact from the
stored P_loss and Q_leak; Q_out <= 0 is an unbounded rise); the CONDUCTION end holds both walls at
T_in, rise = the L2 film-mean maximum over the land (solve_l2_spherical, field_max_boundary,
solver.py:486-491), with the one-adiabatic-wall peak x6 as before (R-57.3).  Thresholds no longer
pick the end.  Work is on the stored solutions; the mechanical campaign stays on the routing as run.

Classes: MINERAL (rho 828.5, c_p 2000, k_f 0.15; c_p, k_f CITED Totten2011 citation_verified false;
ceiling 110 C) and the water defaults as the water-glycol end (rho 1000, c_p 4 kJ/(kg K), k_f 0.60;
ceiling 60 C).  Advective rise scales 1/(rho c_p); conduction rise 1/k_f at fixed fields (the
advection term vanishes, R-67.2).  Conduction rises at the defaults: stored where the row ran L2;
for rows that ran L1, the mineral L2 rise of R68_D8_l2rise.parquet / (k_def/k_min), or, where none
exists (rows L1 under both property sets), one forward solve here with the default properties.

In-domain = converged and De_sq_bnd <= De_sq_max (the Nahme population of R-57.3).  Statistics on
box E (g2, E2, E3 as before) and box D class 1 (OpsSurvive).
Registered (R-69.3, chat's estimates): 1. conduction end: g1 the most frequent single binding
constraint for both classes (E2's restriction T_in < 60 C); 2. advective end, mineral: g2 replaces
g1; 3. advective end, mineral: g2 breached on more than half of the in-domain evaluations;
4. conduction end, mineral: Nahme share on the mean rise below 1 %.

    python scripts/run_R69_thermal_bracket.py run [--workers 4]   # forward solves, checkpointed
    python scripts/run_R69_thermal_bracket.py report
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
BOXD = OUT / "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet"
BOXE = OUT / "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet"
CKPT = OUT / "R69_bracket_ckpt"
DEST = OUT / "R69_thermal_bracket.parquet"
REPORT = OUT / "R69_thermal_bracket.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
COLS = AXES + ["sample_id", "outcome", "thermal_route", "T_max_bnd", "De_sq_bnd", "lambda_ratio_lo", "lambda_ratio_hi",
               "P_loss", "Q_leak", "eps_ecc", "c_equilibrium", "F", "omega", "eta_0", "spin_ratio"]
MEMORY_SIGN = "modulus"
BLOCK = 200


def load():
    import run_R68_D8_reemit as RE
    import bs_thermal_props as W
    from src.thermal import ThermalProperties
    th = ThermalProperties()
    out = {}
    for box, src in (("D", BOXD), ("E", BOXE)):
        X = pd.read_parquet(src, columns=list(dict.fromkeys(COLS + (["model_id"] if box == "E" else []))))
        X = X[X.outcome == "converged"].reset_index(drop=True)
        if box == "D":
            X["model_id"] = -1
        X["route_min"] = RE.routes(X, W.MINERAL["k_f"], W.MINERAL["rho"] * W.MINERAL["c_p"])
        out[box] = X
    return out, th


def need_forward(X):
    return X[(X.thermal_route == "L1") & (X.route_min == "L1")]


def job(rec):
    import warnings
    warnings.simplefilter("ignore")
    import ball_socket.thermal as bs_thermal
    from src.thermal import ThermalProperties
    from run_R57_mesh_error import _solver, _contact
    t0 = time.time()
    slv, ci = _solver({k: float(rec[k]) for k in AXES}, _contact(int(rec["model_id"])), None)
    slv.memory_sign = MEMORY_SIGN
    res, sph = slv._forward(ci, float(rec["eps_ecc"]), slv._fluid(ci), float(rec["c_equilibrium"]))
    h = sph.gap()
    F_film = float(sph.load_capacity(res.pressure))
    eta_bar = res.eta_bar if res.eta_bar is not None else np.full_like(h, ci.eta_0)
    gdot = sph.gap_averaged_shear_rate(res.pressure, eta_bar)
    T = bs_thermal.solve_l2_spherical(sph, eta_bar, gdot, ThermalProperties(), ci.T_in + 273.15)
    return {"box": rec["box"], "sample_id": int(rec["sample_id"]), "model_id": int(rec["model_id"]),
            "rise_L2_def_bnd": float(sph.field_max_boundary(T)) - 273.15 - ci.T_in,
            "F_film_repro_rel": abs(F_film / float(rec["F"]) - 1), "wall_s": time.time() - t0}


def stage_run(workers):
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-69.3 conduction-end forward solves")
    data, _ = load()
    recs = pd.concat([need_forward(X).assign(box=b) for b, X in data.items()], ignore_index=True).to_dict(orient="records")
    print(f"{len(recs)} forward solves (rows L1 under both property sets)", flush=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(workers, 4)) as ex:
        for b in range(0, len(recs), BLOCK):
            part = CKPT / f"block_{b:05d}.parquet"
            if part.exists():
                continue
            q = pd.DataFrame(list(ex.map(job, recs[b:b + BLOCK]))).assign(git_commit=prov["commit"])
            tmp = part.with_suffix(".tmp")
            q.to_parquet(tmp, index=False)
            tmp.replace(part)
            print(f"{time.strftime('%H:%M:%S')} block {b} done", flush=True)
    return 0


def rises(X, box, th, l2, fw):
    import bs_thermal_props as W
    rc_def, rc_min = th.rho * th.c_p, W.MINERAL["rho"] * W.MINERAL["c_p"]
    kr = th.k_f / W.MINERAL["k_f"]
    Q = X.Q_leak.astype(float)
    adv_def = np.where(Q > 0, X.P_loss / (rc_def * Q.clip(lower=1e-300)), np.inf)
    X = X.assign(adv_Wg=adv_def, adv_Min=adv_def * rc_def / rc_min)
    key = ["sample_id", "model_id"]
    X = X.merge(l2[l2.box == box][key + ["rise_L2_min_bnd"]], on=key, how="left")
    X = X.merge(fw[fw.box == box][key + ["rise_L2_def_bnd"]], on=key, how="left")
    stored_L2 = X.thermal_route != "L1"
    cond = np.where(stored_L2, X.T_max_bnd - X.T_in,
                    np.where(X.rise_L2_min_bnd.notna(), X.rise_L2_min_bnd / kr, X.rise_L2_def_bnd))
    X["cond_Wg"], X["cond_Min"] = cond, cond * kr
    assert np.isfinite(X.cond_Wg).all(), f"box {box}: a row lacks its conduction rise"
    return X


def stage_report():
    from ball_socket.provenance import require_clean_tree
    from ball_socket.campaign_config_v4 import ConstraintSet
    from run_R57_thermal_by_route import walther_b_band
    prov = require_clean_tree("R-69.3 thermal bracket")
    cs = ConstraintSet()
    CEIL = {"Min": cs.T_max_C_by_class["mineral"], "Wg": cs.T_max_C_by_class["water_glycol"]}
    WG, DE, L3 = CEIL["Wg"], cs.De_sq_max, cs.lambda_ratio_min
    data, th = load()
    l2 = pd.read_parquet(OUT / "R68_D8_l2rise.parquet")
    fw = pd.concat([pd.read_parquet(p) for p in sorted(CKPT.glob("block_*.parquet"))], ignore_index=True)
    nfw = sum(len(need_forward(X)) for X in data.values())
    assert len(fw) == nfw, (len(fw), nfw)
    E = rises(data["E"], "E", th, l2, fw)
    D = rises(data["D"], "D", th, l2, fw)
    rep = {"provenance": prov, "forward_solves": int(len(fw)), "forward_film_repro_max_rel": float(fw.F_film_repro_rel.max()),
           "in_domain": "converged and De_sq_bnd <= De_sq_max", "classes": {"Min": "mineral (D-8), 110 C",
                                                                             "Wg": "water defaults, 60 C"}}
    ind = E[E.De_sq_bnd <= DE]
    b_lo, b_hi = walther_b_band(ind.T_in.values)
    O = pd.read_parquet(OUT / "R65_ops_solv.parquet").set_index("sample_id")
    rs = pd.read_parquet(OUT / "R68_D8_resolve.parquet").set_index("sample_id")
    stats = {}
    for c in ("Min", "Wg"):
        ce = CEIL[c]
        # class 1 at sigma* lo: the mineral class uses the re-solved classes (R-68.2), the water end as stored
        cls = O.cls_lo.copy()
        if c == "Min":
            for sid, r in rs.iterrows():
                cls.loc[sid] = (1 if r.lam_lo >= L3 else 2) if r.outcome == "converged" else 4
        c1 = D[D.sample_id.isin(cls.index[cls == 1])]
        for e in ("Cond", "Adv"):
            col = f"{e.lower()}_{c}"
            rise = ind[col].values
            g2 = ind.T_in.values + rise > ce
            s = {"G2BreachPct": 100 * float(g2.mean()), "RiseMed": float(np.median(rise))}
            # E3 construction: T_in below the ceiling, g1 and g3 satisfied at sigma* lo, g2 the sole violation
            fr = E[(E.T_in < ce) & (E.lambda_ratio_lo >= L3) & (E.De_sq_bnd <= DE)]
            s["G2SolePct"] = 100 * float((fr.T_in + fr[col] > ce).mean())
            s["G2SoleRows"] = int((fr.T_in + fr[col] > ce).sum())
            # Nahme on the mean rise; at the conduction end also x6
            s["NahmeLoPct"] = 100 * float((b_lo * rise > 1).mean())
            s["NahmeHiPct"] = 100 * float((b_hi * rise > 1).mean())
            if e == "Cond":
                s["NahmeSixLoPct"] = 100 * float((b_lo * 6 * rise > 1).mean())
                s["NahmeSixHiPct"] = 100 * float((b_hi * 6 * rise > 1).mean())
            # E2 construction: box E restricted to T_in < 60 C; the most frequent single binding constraint
            rest = E[E.T_in < WG]
            for end in ("lo", "hi"):
                g1 = rest[f"lambda_ratio_{end}"] < L3
                g2r = rest.T_in + rest[col] > ce
                g3 = rest.De_sq_bnd > DE
                n = g1.astype(int) + g2r.astype(int) + g3.astype(int)
                lab = pd.Series(np.where(n == 0, "none", np.where(n > 1, "multi",
                                                                  np.where(g1, "g1", np.where(g2r, "g2", "g3")))))
                v = lab.value_counts().drop(labels=["none", "multi"], errors="ignore")
                s[f"BindDom_{end}"] = str(v.idxmax())
                s[f"BindDomPct_{end}"] = float(100 * v.max() / len(lab))
                s[f"BindShares_{end}"] = {k: float(100 * n_ / len(lab)) for k, n_ in v.items()}
            # OpsSurvive: box D class 1 at sigma* lo surviving g2 and g3
            s["OpsSurvivePct"] = 100 * float(((c1.T_in + c1[col] <= ce) & (c1.De_sq_bnd <= DE)).mean())
            stats[f"{c}{e}"] = s
    rep["stats"] = stats
    rep["ThetaFlow"] = {"Wg": {"med": float(np.median(ind.adv_Wg[np.isfinite(ind.adv_Wg)] / 10.0))}}
    m, w = stats["MinCond"], stats["WgCond"]
    ma = stats["MinAdv"]
    rep["registered"] = {
        "1_cond_g1_dominant_both_classes": bool(all(stats[k][f"BindDom_{e}"] == "g1" for k in ("MinCond", "WgCond")
                                                    for e in ("lo", "hi"))),
        "2_adv_mineral_g2_replaces_g1": bool(all(ma[f"BindDom_{e}"] == "g2" for e in ("lo", "hi"))),
        "3_adv_mineral_g2_breach_gt_half": bool(ma["G2BreachPct"] > 50.0),
        "4_cond_mineral_nahme_mean_lt_1pct": bool(m["NahmeHiPct"] < 1.0)}
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-69.3 thermal bracket")
    guarded_path(str(REPORT), what="R-69.3 report")
    pd.concat([D.assign(box="D"), E.assign(box="E")], ignore_index=True)[
        ["box", "sample_id", "model_id", "thermal_route", "route_min", "adv_Wg", "adv_Min", "cond_Wg", "cond_Min"]].assign(
        git_commit=prov["commit"], tag="first_order_fixed_solution").to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "report"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    return stage_run(a.workers) if a.stage == "run" else stage_report()


if __name__ == "__main__":
    raise SystemExit(main())
