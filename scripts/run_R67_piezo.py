"""R-67.1 (D-9): pressure-dependent viscosity at the claim points, by the exact reduced-pressure
transform.  No frozen file is edited.

Physics (R-67.1).  With eta = eta_a g(p) and q(p) = int_0^p dp'/g(p'), the Newtonian film
equation div(h^3/(12 eta) grad p) = squeeze becomes div(h^3/(12 eta_a) grad q) = squeeze: q
solves the ISOVISCOUS problem with the rim value q(p_ch) and the mouth value q(p_case).  Spin
generates no pressure on the axisymmetric film.  The real pressure is p = p(q), and the film
load is the solver's own projection of p(q).

Path (quoted): the ball-socket assembly with model="newtonian" (ball_socket/reynolds.py:50-51,
_solve_newtonian at fluid.eta_0), with eta_a = eta_s, the high-shear plateau (the claim points
sit on it: eta_bar/eta_0 = 0.951-0.954, R-65.4).  The solver is built as campaign._evaluate_x_inner
builds it (run_R57_mesh_error._solver) at the point's contact arm; the forward solve is
solver._forward (solver.py:233-250) at the point's stored clearance c_equilibrium (the closure's
rise there is tens of mK, Delta c ~ nm: c is held fixed); the target is the solver's own
F_app - F_pocket (solver.py:381-387; the pocket is hydrostatic at the real p_ch); the asperity
load is the arm's contact.pressure(h) projected (solver.py:446-448).  eps is found by brentq on
F_film(p(q)) + F_asp - F_target over [eps_lo, eps_cap].

Points (production mesh 32x24): the E15 argmax (box G arm 9, c0 = 2.25 um; the Fig. 5 point),
the E13 argmax and the E5 extremum (run_R56_tolerance.cases(), sigma* lo).  Laws:
  isoviscous   g = 1 (the campaign's fluid, Newtonian limit at eta_s)
  roelands     LeeShangZhang2026 Table 1, Eq. (2), T = T_in in kelvin (scripts/p3_lee_fluid.py)
  barus        g = exp(alpha p), alpha in {2, 4, ..., 14} 1/GPa (sensitivity axis)
alpha_crit: the Barus alpha at which Lambda = 3, by brentq.  The c0 axis: box G's grid at the
argmax arm, Roelands and isoviscous.

Registered (R-67.1, chat's estimate):
  1. the isoviscous Newtonian equilibrium reproduces the solver's delta within 1 % at each point;
  2. E15: the Roelands factor on Lambda in [1.30, 1.48], and Lambda >= 3;
  3. E15: Barus 4 1/GPa factor in [1.08, 1.18];
  4. E15: alpha_crit in [6.5, 9.5] 1/GPa.
Failing 2-4 is not a STOP for the run.
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
DEST = OUT / "R67_piezo.parquet"
REPORT = OUT / "R67_piezo.report.json"
AXES = ["omega", "p_ch", "c0", "T_in", "eta_0", "lam", "eta_s_ratio", "theta_min_deg"]
BARUS = [2, 4, 6, 8, 10, 12, 14]
MEMORY_SIGN = "signed"     # R-59.3: explicit; the Newtonian path carries no memory term


class Point:
    def __init__(self, name, x, model_id, flow_factors, sigma, c_eq, delta_solver, Lam_solver):
        from run_R57_mesh_error import _solver, _contact
        self.name, self.x, self.sigma = name, x, sigma
        self.slv, self.ci = _solver(x, _contact(model_id), flow_factors)
        self.slv.memory_sign = MEMORY_SIGN
        self.slv.model = "newtonian"
        self.fl = SimpleNamespace(eta_0=x["eta_0"] * x["eta_s_ratio"])     # eta_a = eta_s
        self.c = c_eq
        self.F_target = self.slv._applied_load(self.ci) - self.slv._pocket_relief(self.ci)
        self.delta_solver, self.Lam_solver = delta_solver, Lam_solver

    def state(self, eps, q_of_p, p_of_q):
        ci_q = dataclasses.replace(self.ci, p_ch=float(q_of_p(self.ci.p_ch)), p_case=float(q_of_p(self.ci.p_case)))
        res, sph = self.slv._forward(ci_q, eps, self.fl, self.c)
        q = np.asarray(res.pressure)
        p = np.asarray(p_of_q(q), float)
        h = sph.gap()
        F_asp = float(sph.project_on_load_axis(self.slv.contact.pressure(h))) if self.slv.contact is not None else 0.0
        F_film = float(sph.load_capacity(p)) if np.all(np.isfinite(p)) else np.inf
        return dict(F_film=F_film, F_asp=F_asp, q_peak=float(q.max()), p_peak=float(np.max(p)),
                    h_min_bnd=float(sph.h_min_boundary()))

    def solve(self, q_of_p, p_of_q):
        from scipy.optimize import brentq
        lo, hi = self.slv.eps_lo, self.slv.eps_cap

        def r(eps):
            s = self.state(eps, q_of_p, p_of_q)
            return min(s["F_film"] + s["F_asp"] - self.F_target, 1e30)
        if r(lo) >= 0:
            eps = lo
        else:
            if r(hi) < 0:
                return None
            eps = brentq(r, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=300)
        s = self.state(eps, q_of_p, p_of_q)
        return dict(eps=eps, delta=s["h_min_bnd"] / self.c, Lam=s["h_min_bnd"] / self.sigma, **s)


def laws(T_K):
    import p3_lee_fluid as LF
    pg, qg = LF.roelands_q_table(T_K)
    out = {"isoviscous": (lambda p: p, lambda q: q, None),
           "roelands_LeeVG46": (lambda p: np.interp(p, pg, qg),
                                lambda q: np.interp(q, qg, pg, right=np.inf), 1.0 / LF.alpha_star(T_K))}
    for a in BARUS:
        al = a * 1e-9
        out[f"barus_{a}"] = (lambda p, al=al: LF.barus_q(p, al), lambda q, al=al: LF.barus_p(q, al), 1.0 / al)
    return out


def points():
    import run_R56_tolerance as T
    R = pd.read_parquet(OUT / "R58_richardson.parquet")
    pts = []
    B = pd.read_parquet(OUT / "boxG_E15.parquet")
    g15 = json.loads((OUT / "boxG_E15.report.json").read_text(encoding="utf-8"))["argmax"]
    r = B[(B.model_id == g15["arm"]) & np.isclose(B.c0 * 1e6, g15["c0_um"]) & (B.memory_sign == "signed")].iloc[0]
    from ball_socket import mixed as mx
    sig = float(mx.expand_for_solving(mx.registered_contact_models(branch="psi", summit_convention="derived"))
                [int(r.model_id)][0].sigma_star)
    pts.append(Point("E15", {k: float(r[k]) for k in AXES}, int(r.model_id), None, sig, float(r.c_equilibrium),
                     float(r.h_min_bnd / r.c_equilibrium), float(r.h_min_bnd / sig)))
    for c in T.cases():
        if c["case"] in ("E13", "E5"):
            s = R[(R.case == c["case"]) & (R.Ntheta == 32)].iloc[0]
            pts.append(Point(c["case"], dict(c["x"]), c["model_id"], c["flow_factors"], c["sigma"],
                             float(s.c_equilibrium), float(s.h_min_bnd / s.c_equilibrium), float(s.Lam)))
    return pts, B, g15


def main() -> int:
    from ball_socket.provenance import require_clean_tree
    from scipy.optimize import brentq
    prov = require_clean_tree("R-67.1 D-9 piezoviscous transform")
    pts, B, g15 = points()
    rows, rep = [], {"provenance": prov, "points": {}}
    for P in pts:
        L = laws(P.x["T_in"] + 273.15)
        base = None
        for name, (qp, pq, qinf) in L.items():
            t0 = time.time()
            s = P.solve(qp, pq)
            if s is None:
                rows.append(dict(point=P.name, fluid_law=name, outcome="no equilibrium below eps_cap"))
                continue
            if name == "isoviscous":
                base = s
            rows.append(dict(point=P.name, fluid_law=name, outcome="converged", **s, F_target=P.F_target,
                             q_inf=qinf, q_peak_over_q_inf=(s["q_peak"] / qinf if qinf else np.nan),
                             Lam_factor_vs_iso=s["Lam"] / base["Lam"], Lam_solver=P.Lam_solver,
                             delta_solver=P.delta_solver, c=P.c, T_in=P.x["T_in"], wall_s=time.time() - t0))

        def lam_at(a):
            al = a * 1e-9
            import p3_lee_fluid as LF
            s = P.solve(lambda p: LF.barus_q(p, al), lambda q: LF.barus_p(q, al))
            return (s["Lam"] if s else np.inf) - 3.0
        ac = brentq(lam_at, 1e-3, 40.0, xtol=1e-6) if lam_at(1e-3) < 0 < lam_at(40.0) else np.nan  # Barus at alpha -> 0 is the isoviscous limit
        rp = {r["fluid_law"]: r for r in rows if r["point"] == P.name and r.get("outcome") == "converged"}
        rep["points"][P.name] = {
            "delta_solver": P.delta_solver, "delta_newtonian_eta_s": base["delta"],
            "delta_rel_diff": base["delta"] / P.delta_solver - 1, "Lam_solver": P.Lam_solver,
            "Lam_iso": base["Lam"], "Lam_roelands": rp["roelands_LeeVG46"]["Lam"],
            "factor_roelands": rp["roelands_LeeVG46"]["Lam_factor_vs_iso"],
            "factor_barus": {a: rp[f"barus_{a}"]["Lam_factor_vs_iso"] for a in BARUS if f"barus_{a}" in rp},
            "alpha_crit_per_GPa": ac,
            "roelands_q_peak_over_q_inf": rp["roelands_LeeVG46"]["q_peak_over_q_inf"],
            "roelands_p_peak_MPa": rp["roelands_LeeVG46"]["p_peak"] / 1e6, "iso_p_peak_MPa": base["p_peak"] / 1e6}
    # the c0 axis at the argmax arm (box G grid), Roelands and isoviscous
    from ball_socket import mixed as mx
    axis = []
    Bg = B[(B.model_id == g15["arm"]) & (B.memory_sign == "signed") & (B.outcome == "converged")].sort_values("c0")
    sig = pts[0].sigma
    for _, r in Bg.iterrows():
        P = Point("E15_axis", {k: float(r[k]) for k in AXES}, int(r.model_id), None, sig, float(r.c_equilibrium),
                  float(r.h_min_bnd / r.c_equilibrium), float(r.h_min_bnd / sig))
        L = laws(P.x["T_in"] + 273.15)
        for name in ("isoviscous", "roelands_LeeVG46"):
            s = P.solve(*L[name][:2])
            rec = dict(point="E15_axis", fluid_law=name, c0_um=1e6 * P.x["c0"],
                       outcome="converged" if s else "no equilibrium below eps_cap", Lam_solver=P.Lam_solver,
                       delta_solver=P.delta_solver, c=P.c, F_target=P.F_target)
            if s:
                rec.update(s)
            rows.append(rec)
            axis.append(rec)
    A = pd.DataFrame(axis)
    ax = {}
    for name in ("isoviscous", "roelands_LeeVG46"):
        a = A[(A.fluid_law == name) & (A.outcome == "converged")].sort_values("c0_um")
        i = int(a.Lam.values.argmax())
        ax[name] = {"c0_star_um": float(a.c0_um.iloc[i]), "Lam_star": float(a.Lam.iloc[i]),
                    "Lam_at_c0_min": float(a.Lam.iloc[0]), "Lam_at_c0_max": float(a.Lam.iloc[-1]),
                    "c0_min_um": float(a.c0_um.iloc[0]), "c0_max_um": float(a.c0_um.iloc[-1]),
                    "interior": bool(0 < i < len(a) - 1)}
    rep["c0_axis"] = ax
    e = rep["points"]["E15"]
    rep["registered"] = {
        "1_delta_within_1pct": {k: abs(v["delta_rel_diff"]) <= 0.01 for k, v in rep["points"].items()},
        "2_E15_roelands_factor_in_1.30_1.48_and_Lam_ge_3": bool(1.30 <= e["factor_roelands"] <= 1.48 and e["Lam_roelands"] >= 3),
        "3_E15_barus4_factor_in_1.08_1.18": bool(1.08 <= e["factor_barus"][4] <= 1.18),
        "4_E15_alpha_crit_in_6.5_9.5": bool(6.5 <= e["alpha_crit_per_GPa"] <= 9.5)}
    rep["path"] = ("ball-socket assembly, model='newtonian' (ball_socket/reynolds.py:50-51), eta_a = eta_s; "
                   "solver._forward at the stored c_equilibrium; target F_app - F_pocket; production mesh 32x24")
    from ball_socket.immutable import guarded_path
    guarded_path(str(DEST), what="R-67.1 piezoviscous transform")
    guarded_path(str(REPORT), what="R-67.1 report")
    pd.DataFrame(rows).assign(git_commit=prov["commit"], memory_sign="none (Newtonian path)").to_parquet(DEST, index=False)
    REPORT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("points", "c0_axis", "registered")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
