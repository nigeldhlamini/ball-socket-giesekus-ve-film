"""R-72.6, registration E17: the stroke-integrated squeeze at the three claim points (fork F1(b)).

Model (R-72.6): rigid, isothermal, Newtonian at eta_s; geometry and duty of each claim point.  The
unrelieved load F_req = (1 - beta_r) p_ch A_p = F_app - F_pocket (the solver's own target, solver.py:
381-387) is constant over a delivery stroke omega t in [0, pi].  At each delta, the squeeze rate
r = eps_dot / omega that carries F_req with the land's hydrostatic field (Dirichlet p_ch at the rim,
p_case at the mouth) is found, and d delta / d(omega t) = -r cos(theta_min), so
    omega t(delta) = int_delta^delta0 d delta' / (r(delta') cos theta_min)          (exact 1-D quadrature).
Film load only (the rigid model's film; asperity contact is not part of this model).

Machinery: the D-9 transform of R-67/R-68 (run_R67_piezo; scripts/p3_lee_fluid.py).  The Newtonian
problem is linear, so at each delta three basis solves of the ball-socket assembly (model
"newtonian", eta = eta_s, solver._forward at the point's stored clearance) fix every law and rate:
    Pi_rim  rim 1, mouth 0, no motion;   Pi_mouth  rim 0, mouth 1, no motion;   Q_s  zero BCs, unit
    squeeze rate (e_dot = omega c);   q = q(p_ch) Pi_rim + q(p_case) Pi_mouth + r Q_s,  p = p(q),
    F = sph.load_capacity(p).
Mesh 128 x 96 (the rim band delta / tan theta_min is unresolved on 32 x 24 below delta ~ 0.03);
delta grid 0.999 ... 0.002.  Laws: isoviscous; Barus G = alpha p_ch in {0.11, 0.15, 0.41, 0.59, 0.63,
1.0, 2.0}; Lee's Roelands VG46 at the point's T_in.  Starts: delta0 = 0.999 (concentric) and the
snapshot delta (the v6 value).

Registered (E17, R-72.6, frozen):
  (a) consistency: at G = 0 the instant r = eps (the snapshot condition, eps_dot = omega eps) falls at the
      v6 snapshot delta within 0.5 % at all three points -- STOP if not;
  (b) isoviscous, from delta0 = 1, the closure time is finite, and at E15 delta = 0.01 is reached before
      omega t = pi (chat ~2.7);
  (c) for every G > 0, r / delta stays bounded as delta falls (operationalised here, before running: r/delta at
      the smallest delta of the grid is finite and within a factor 1.5 of its value one decade above);
  (d) E15, Roelands, delta0 = 1: Lambda(pi) < 3 (chat ~0.64 under Barus G = 0.63);
  (e) the table of crossing times, Lambda(pi), the G = 0 closure time and the smallest G with Lambda(pi) >= 3.
Refuting (b), (c) or (d) is reported, not a STOP.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "ch9"
DEST = OUT / "R72_E17_stroke.parquet"
REPORT = OUT / "R72_E17_stroke.report.json"
GRID = (128, 96)
GS = [0.11, 0.15, 0.41, 0.59, 0.63, 1.0, 2.0]
DELTAS = np.unique(np.concatenate([np.linspace(0.999, 0.2, 60), np.geomspace(0.2, 0.002, 70)]))[::-1]
MEMORY_SIGN = "signed"     # R-59.3: explicit; the Newtonian path carries no memory term
LAM_TH = 3.0


def bases(slv, ci, c, fl, delta, tmin):
    eps = (1.0 - delta) / np.cos(tmin)
    slv.squeeze_phase = 0.0
    rR, sph = slv._forward(dataclasses.replace(ci, p_ch=1.0, p_case=0.0), eps, fl, c)
    rM, _ = slv._forward(dataclasses.replace(ci, p_ch=0.0, p_case=1.0), eps, fl, c)
    slv.squeeze_phase = 1.0 / eps
    rS, _ = slv._forward(dataclasses.replace(ci, p_ch=0.0, p_case=0.0), eps, fl, c)
    slv.squeeze_phase = 1.0
    return np.asarray(rR.pressure), np.asarray(rM.pressure), np.asarray(rS.pressure), sph, eps


def rate(Fq, F_req, PiR, PiM, Qs, qch, qcase, pq):
    """r carrying F_req: F(r) = LC(p(q)) is increasing in r (Q_s > 0 on the land)."""
    from scipy.optimize import brentq
    base = qch * PiR + qcase * PiM

    def F(r):
        p = pq(base + r * Qs)
        return Fq(p) if np.all(np.isfinite(p)) else np.inf
    f0 = F(0.0) - F_req
    if f0 == 0:
        return 0.0
    if f0 < 0:
        hi = 1e-3
        while F(hi) - F_req < 0:
            hi *= 2
            if hi > 1e8:
                return np.nan
        return brentq(lambda r: min(F(r), 1e30) - F_req, 0.0, hi, xtol=1e-14, rtol=1e-12, maxiter=400)
    lo = -1e-3
    while F(lo) - F_req > 0:
        lo *= 2
        if lo < -1e8:
            return np.nan
    return brentq(lambda r: min(F(r), 1e30) - F_req, lo, 0.0, xtol=1e-14, rtol=1e-12, maxiter=400)


def trajectory(deltas, r, tmin, delta0):
    """omega t(delta) for delta <= delta0 while r > 0 (closing); NaN beyond a stall (r <= 0)."""
    m = deltas <= delta0 + 1e-12
    d, rr = deltas[m], r[m]
    if d[0] < delta0 - 1e-12:            # start exactly at delta0
        r0 = np.interp(delta0, deltas[::-1], r[::-1])
        d, rr = np.concatenate([[delta0], d]), np.concatenate([[r0], rr])
    t = np.zeros_like(d)
    for i in range(1, len(d)):
        if rr[i] <= 0 or rr[i - 1] <= 0 or not np.isfinite(rr[i]):
            t[i:] = np.nan
            break
        t[i] = t[i - 1] + (d[i - 1] - d[i]) / (np.cos(tmin) * 0.5 * (rr[i] + rr[i - 1]))
    return d, t


def at_time(d, t, tq):
    ok = np.isfinite(t)
    d, t = d[ok], t[ok]
    if t[-1] < tq:
        return None                       # the grid floor is reached before tq
    return float(np.interp(tq, t, d))


def time_at(d, t, dq):
    ok = np.isfinite(t)
    d, t = d[ok], t[ok]
    if dq < d[-1] or dq > d[0]:
        return None
    return float(np.interp(-dq, -d, t))


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    import run_R67_piezo as R7
    import p3_lee_fluid as LF
    from run_R57_mesh_error import _solver, _contact
    prov = require_clean_tree("R-72.6 E17 stroke")
    pts, _, _ = R7.points()
    rows, rep = [], {"provenance": prov, "grid": list(GRID), "delta_grid": [float(DELTAS.max()), float(DELTAS.min())],
                     "points": {}}
    for P in pts:
        t0 = time.time()
        slv, ci = _solver(P.x, _contact(-1), None, grid=GRID)
        slv.memory_sign, slv.model = MEMORY_SIGN, "newtonian"
        fl = SimpleNamespace(eta_0=P.x["eta_0"] * P.x["eta_s_ratio"])
        tmin = np.deg2rad(P.x["theta_min_deg"])
        F_req = slv._applied_load(ci) - slv._pocket_relief(ci)
        laws = {"isoviscous": (lambda p: p, lambda q: q)}
        for G in GS:
            a = G / ci.p_ch
            laws[f"barus_G{G}"] = (lambda p, a=a: LF.barus_q(p, a), lambda q, a=a: LF.barus_p(q, a))
        pg, qg = LF.roelands_q_table(P.x["T_in"] + 273.15)
        laws["roelands_LeeVG46"] = (lambda p: np.interp(p, pg, qg), lambda q: np.interp(q, qg, pg, right=np.inf))
        R = {k: np.full(len(DELTAS), np.nan) for k in laws}
        for i, dl in enumerate(DELTAS):
            PiR, PiM, Qs, sph, eps = bases(slv, ci, P.c, fl, dl, tmin)
            Fq = sph.load_capacity
            for k, (qp, pq) in laws.items():
                R[k][i] = rate(Fq, F_req, PiR, PiM, Qs, float(qp(ci.p_ch)), float(qp(ci.p_case)), pq)
        eps_g = (1.0 - DELTAS) / np.cos(tmin)
        # (a) the snapshot instant r = eps at G = 0
        dr = R["isoviscous"] - eps_g
        k = np.where(np.sign(dr[:-1]) != np.sign(dr[1:]))[0]
        d_snap_traj = float(np.interp(0.0, [dr[k[0] + 1], dr[k[0]]], [DELTAS[k[0] + 1], DELTAS[k[0]]])) if len(k) else np.nan
        snap = P.delta_solver
        pr = {"F_req_N": F_req, "delta_snapshot_v6": snap, "delta_r_eq_eps_G0": d_snap_traj,
              "a_rel_diff": d_snap_traj / snap - 1, "sigma": P.sigma, "c": P.c, "wall_s": 0.0, "laws": {}}
        d3 = LAM_TH * P.sigma / P.c
        for name in laws:
            r = R[name]
            for start, d0 in (("concentric", float(DELTAS.max())), ("snapshot", snap)):
                d, t = trajectory(DELTAS, r, tmin, d0)
                dpi = at_time(d, t, np.pi)
                rec = {"t_pass_snapshot": time_at(d, t, snap) if start == "concentric" else 0.0,
                       "t_Lambda3": time_at(d, t, d3), "delta_pi": dpi,
                       "Lambda_pi": None if dpi is None else dpi * P.c / P.sigma,
                       "floor_reached_before_pi": dpi is None,
                       "t_delta_0p01": time_at(d, t, 0.01), "t_at_grid_floor": float(np.nanmax(t))}
                pr["laws"].setdefault(name, {})[start] = rec
                for di, ti in zip(d, t):
                    rows.append(dict(point=P.name, fluid_law=name, start=start, delta=di, omega_t=ti,
                                     Lambda=di * P.c / P.sigma))
            ok = np.isfinite(r) & (r > 0)
            rd = r / DELTAS
            i_min = int(np.where(ok)[0][-1])
            i_dec = int(np.argmin(np.abs(np.log10(DELTAS) - (np.log10(DELTAS[i_min]) + 1))))
            pr["laws"][name]["r_over_delta_at_min"] = float(rd[i_min])
            pr["laws"][name]["r_over_delta_decade_above"] = float(rd[i_dec])
            pr["laws"][name]["delta_min"] = float(DELTAS[i_min])
        # closure time at G = 0 beyond the grid floor: local power law r ~ C delta^b on the last decade
        r0 = R["isoviscous"]
        m = (DELTAS <= 0.02) & np.isfinite(r0) & (r0 > 0)
        b, lnC = np.polyfit(np.log(DELTAS[m]), np.log(r0[m]), 1)
        dmin = float(DELTAS[m].min())
        tail = dmin ** (1 - b) / (np.exp(lnC) * (1 - b) * np.cos(tmin)) if b < 1 else np.inf
        pr["closure_G0"] = {"t_at_grid_floor": pr["laws"]["isoviscous"]["concentric"]["t_at_grid_floor"],
                            "tail_power_b": float(b), "tail_time": float(tail),
                            "t_c": pr["laws"]["isoviscous"]["concentric"]["t_at_grid_floor"] + float(tail)}
        smallest = None
        for G in GS:
            lp = pr["laws"][f"barus_G{G}"]["concentric"]["Lambda_pi"]
            if lp is not None and lp >= LAM_TH:
                smallest = G
                break
        pr["smallest_G_Lambda_pi_ge_3_concentric"] = smallest if smallest is not None else "none <= 2"
        pr["wall_s"] = time.time() - t0
        for k, v in R.items():
            pr["laws"][k]["r_curve"] = [float(x) for x in v]
        rep["points"][P.name] = pr
        print(P.name, "done", round(pr["wall_s"]), "s", flush=True)
    rep["delta_values"] = [float(x) for x in DELTAS]
    e = rep["points"]["E15"]
    reg_c = {}
    for P in pts:
        for name, v in rep["points"][P.name]["laws"].items():
            if name == "isoviscous":
                continue
            reg_c[f"{P.name}:{name}"] = bool(np.isfinite(v["r_over_delta_at_min"])
                                             and v["r_over_delta_at_min"] <= 1.5 * v["r_over_delta_decade_above"])
    rep["registered"] = {
        "a_consistency_within_0.5pct": {k: bool(abs(v["a_rel_diff"]) <= 0.005) for k, v in rep["points"].items()},
        "b_closure_finite": bool(np.isfinite(e["closure_G0"]["t_c"])),
        "b_E15_reaches_0p01_before_pi": bool(e["laws"]["isoviscous"]["concentric"]["t_delta_0p01"] is not None
                                             and e["laws"]["isoviscous"]["concentric"]["t_delta_0p01"] < np.pi),
        "c_r_over_delta_bounded_all_G": bool(all(reg_c.values())), "c_detail": reg_c,
        "d_E15_roelands_Lambda_pi_lt_3": bool((e["laws"]["roelands_LeeVG46"]["concentric"]["Lambda_pi"] or 0) < LAM_TH)}
    rep["STOP"] = not all(rep["registered"]["a_consistency_within_0.5pct"].values())
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-72.6 E17 trajectories")
    guarded_path(str(REPORT), what="R-72.6 E17 report")
    pd.DataFrame(rows).assign(git_commit=prov["commit"], memory_sign="none (Newtonian path)").to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep["registered"], indent=1, default=float))
    return 2 if rep["STOP"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
