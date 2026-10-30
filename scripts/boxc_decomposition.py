#!/usr/bin/env python
"""Box C, Phase 6: term-by-term load decomposition (registered expectation E4).

E4 has been carried as UNTESTED through box A and box B.  This is the first
test of it.

THREE DECOMPOSITIONS, NOT TWO (ruling R-8.3).  Brief sec 7.2 asks for the
re-solved decomposition and the flux shares and is missing the frozen one.

  D1  frozen-coefficient : ONE linear solve at the converged (eta_bar, gdot_bar)
                           of the full problem, with each term switched off.
                           The term's own linear-response contribution.
  D2  re-solved          : full Picard re-solve with the term off.  The
                           deployed sensitivity, Picard feedback included.
  D3  flux shares        : each flux term evaluated at the full solution and
                           integrated.  An independent solver-side cross-check.

**The D1-to-D2 gap is a RESULT, not a nuisance** (R-8.3): it is the Picard
feedback, and it says how much the constitutive coupling amplifies or damps
each term once the viscosity field responds.  "The bare contribution is X and
the coupled contribution is Y" is a statement about the MODEL rather than about
this joint.

THE ADDITIVITY IDENTITY (R-8.1), asserted not observed.  N1 and hoop are pure
RHS sources and never touch the operator, so p = A^-1 b is AFFINE in their
scales; load_capacity is a weighted surface integral of p, i.e. a LINEAR
functional; therefore F is affine too and

    F(both on) - F(both off) == [F(N1 only) - F(off)] + [F(hoop only) - F(off)]

holds EXACTLY on D1.  It is an algebraic identity, so a violation is a DEFECT
in the assembly, the flux evaluation or the scale plumbing -- never a fork.

THE STOP IS ARMED ON THE MEMORY TERM ONLY (R-8.4).  Memory alone has a genuine
reason to disagree between decompositions, and it has THREE sources of
non-additivity, not one (R-8.2): operator nonlinearity, the ghost RHS
_ghost(D_total), and Picard feedback.  A memory flux share computed from the
mobility alone omits the boundary term and would show a spurious disagreement,
so the ghost contribution is computed and reported explicitly.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np
from scipy.sparse.linalg import spsolve

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ball_socket import campaign_config_v5 as v5          # noqa: E402
from ball_socket import representative as rep             # noqa: E402
from ball_socket.geometry import BallSocketGeometry       # noqa: E402
from ball_socket.reynolds import solve_reynolds           # noqa: E402
from ball_socket.solver import BallSocketSolver, CampaignInput  # noqa: E402
from src.rheology import GiesekusFluid                    # noqa: E402
from src.solver import SolverConfig                       # noqa: E402

FIELDS = _ROOT / "exports" / "cme" / "fields"

#: REGISTERED BEFORE RUNNING (R-8.4).  The additivity identity is exact in
#: exact arithmetic; the residual is the direct sparse solve's own accuracy.
#: Normalised against the LARGER of the two term deltas, because that is the
#: quantity being tested -- normalising against F would hide a total
#: cancellation.  1e-8 is generous for a double-precision direct solve and is
#: still six orders below any physical effect.
ADDITIVITY_RTOL = 1.0e-8

#: The campaign runs include_normal_stress=False.  The decomposition must
#: switch it ON, and the difference is itself a result: it quantifies what the
#: campaign's own configuration omits.
CAMPAIGN_NS = False


def _build(x: dict, include_ns: bool):
    geom = BallSocketGeometry(c0=x["c0"], R_b=rep.R_B_M,
                              theta_max=np.deg2rad(rep.THETA_MAX_DEG),
                              theta_min=np.deg2rad(x["theta_min_deg"]))
    cfg = SolverConfig(include_normal_stress=include_ns, max_iter=80,
                       tol=1e-9, omega=0.7)
    slv = BallSocketSolver(geom, model="viscoelastic",
                           Ntheta=v5.INPUT_SPACE and 32, Nphi=24,
                           alpha=v5.ALPHA_FIXED,
                           squeeze_phase=v5.SQUEEZE_PHASE,
                           spin_ratio=v5.SPIN_RATIO, config=cfg)
    ci = CampaignInput(omega=x["omega"], p_ch=x["p_ch"], c0=x["c0"],
                       T_in=x["T_in"], eta_0=x["eta_0"], lam=x["lam"],
                       eta_s_over_eta0=x["eta_s_ratio"], p_case=v5.P_CASE_PA)
    return slv, ci


def _state(slv, ci):
    """Converged full-problem state: (sph, fluid, eta_bar, gdot, pressure, eps)."""
    perf = slv.solve_equilibrium(ci)
    fluid = slv._fluid(ci)
    res, sph = slv._forward(ci, perf.eps_ecc, fluid, ci.c0)
    h = sph.gap()
    eta_bar = res.eta_bar if res.eta_bar is not None else np.full_like(h, ci.eta_0)
    gdot = sph.gap_averaged_shear_rate(res.pressure, eta_bar)
    return sph, fluid, eta_bar, gdot, res.pressure, perf


def decompose(tag: str, x: dict) -> dict:
    out = {"point_tag": tag, "inputs": x}

    # ---- the campaign's own omission, quantified -------------------------
    slv_off, ci = _build(x, CAMPAIGN_NS)
    slv_on, _ = _build(x, True)
    p_off = slv_off.solve_equilibrium(ci)
    sph, fluid, eta_bar, gdot, pressure, p_on = _state(slv_on, ci)
    # DEFECT FOUND ON THE FIRST RUN AND CORRECTED.  Comparing F between the two
    # configurations measures almost nothing: solve_equilibrium PINS F to
    # F_target by construction, so the load is a CONSTRAINT, not an output, and
    # the difference came back at 1e-8 relative.  Quoting that as "the campaign
    # omits nothing" would have been wrong.  What actually moves is the
    # ECCENTRICITY the film needs to carry the same load, and therefore h_min.
    out["campaign_omission"] = {
        "F_target_is_pinned": True,
        "F_ns_off": float(p_off.F), "F_ns_on": float(p_on.F),
        "F_relative_change": float((p_on.F - p_off.F) / p_off.F),
        "eps_ns_off": float(p_off.eps_ecc), "eps_ns_on": float(p_on.eps_ecc),
        "eps_relative_change": float((p_on.eps_ecc - p_off.eps_ecc)
                                     / p_off.eps_ecc),
        "h_min_ns_off": float(p_off.h_min), "h_min_ns_on": float(p_on.h_min),
        "h_min_relative_change": float((p_on.h_min - p_off.h_min)
                                       / p_off.h_min),
        "note": ("the campaign runs include_normal_stress=False, so N1 and "
                 "hoop are OFF in every registered row.  Read the h_min and "
                 "eps changes, NOT the F change: F is pinned to F_target by "
                 "the equilibrium search, so its invariance is a property of "
                 "the search and not evidence that the terms are negligible.")}

    # ---- D1: frozen-coefficient ------------------------------------------
    def F_frozen(mem: bool, n1: float, hoop: float) -> float:
        sysm = sph.assemble("viscoelastic", fluid, eta_bar, gdot,
                            pressure=pressure, include_memory=mem,
                            include_normal_stress=True,
                            n1_scale=n1, hoop_scale=hoop)
        p = sph.to_2d(spsolve(sysm.A, sysm.b))
        return float(sph.load_capacity(p))

    F_all = F_frozen(True, 1.0, 1.0)
    F_no_mem = F_frozen(False, 1.0, 1.0)
    F_no_n1 = F_frozen(True, 0.0, 1.0)
    F_no_hoop = F_frozen(True, 1.0, 0.0)
    F_bare = F_frozen(True, 0.0, 0.0)          # both sources off
    F_n1_only = F_frozen(True, 1.0, 0.0)
    F_hoop_only = F_frozen(True, 0.0, 1.0)
    out["D1_frozen"] = {
        "F_all": F_all,
        "delta_memory": F_all - F_no_mem,
        "delta_N1": F_all - F_no_n1,
        "delta_hoop": F_all - F_no_hoop,
        "share_memory": (F_all - F_no_mem) / F_all,
        "share_N1": (F_all - F_no_n1) / F_all,
        "share_hoop": (F_all - F_no_hoop) / F_all}

    # ---- the additivity identity (R-8.1): HARD FAILURE -------------------
    lhs = F_all - F_bare
    rhs = (F_n1_only - F_bare) + (F_hoop_only - F_bare)
    scale = max(abs(F_n1_only - F_bare), abs(F_hoop_only - F_bare), 1e-300)
    rel = abs(lhs - rhs) / scale
    out["additivity_identity"] = {
        "lhs_F_all_minus_bare": lhs, "rhs_sum_of_singles": rhs,
        "abs_residual": abs(lhs - rhs), "rel_residual": rel,
        "registered_rtol": ADDITIVITY_RTOL,
        "holds": bool(rel < ADDITIVITY_RTOL),
        "disposition": ("HARD FAILURE if violated -- an exact algebraic "
                        "identity, so a violation is a defect in the assembly, "
                        "the flux evaluation or the scale plumbing, never a "
                        "fork (ruling R-8.4)")}

    # ---- D2: re-solved ----------------------------------------------------
    def F_resolved(mem: bool, n1m: float, hoopm: float) -> float:
        cfg = SolverConfig(include_normal_stress=True, max_iter=80, tol=1e-9,
                           omega=0.7)
        cfg.include_memory = mem
        r = solve_reynolds(sph, fluid, cfg, model="viscoelastic",
                           n1_mult=n1m, hoop_mult=hoopm)
        return float(sph.load_capacity(r.pressure))

    R_all = F_resolved(True, 1.0, 1.0)
    out["D2_resolved"] = {
        "F_all": R_all,
        "delta_memory": R_all - F_resolved(False, 1.0, 1.0),
        "delta_N1": R_all - F_resolved(True, 0.0, 1.0),
        "delta_hoop": R_all - F_resolved(True, 1.0, 0.0)}

    # ---- the D1 -> D2 gap: the Picard feedback, reported as a result ------
    out["picard_feedback"] = {
        term: {"D1_bare": out["D1_frozen"]["delta_" + term],
               "D2_coupled": out["D2_resolved"]["delta_" + term],
               "amplification": (out["D2_resolved"]["delta_" + term]
                                 / out["D1_frozen"]["delta_" + term]
                                 if out["D1_frozen"]["delta_" + term] != 0 else None)}
        for term in ("memory", "N1", "hoop")}

    # ---- D3: flux shares, WITH the ghost term (R-8.2) ---------------------
    h = sph.gap()
    D_gnf = h ** 3 / (12.0 * eta_bar)
    Mv = fluid.memory_viscosity(gdot)
    hdot = sph.gap_rate()
    mem_factor = -fluid.lambda_ * hdot / 12.0
    D_mem = mem_factor * (h ** 2 * Mv / eta_bar ** 2)
    gpt, gpp = sph.grad_s(pressure)
    gmag = np.sqrt(gpt ** 2 + gpp ** 2)
    visc_flux = float(sph.surface_integral(np.abs(D_gnf) * gmag))
    mem_flux = float(sph.surface_integral(np.abs(D_mem) * gmag))
    ghost_full = sph._ghost(D_gnf + D_mem)
    ghost_gnf = sph._ghost(D_gnf)
    ghost_mem_delta = float(np.abs(ghost_full - ghost_gnf).sum())
    out["D3_flux_shares"] = {
        "viscometric_flux": visc_flux, "memory_flux_mobility_only": mem_flux,
        "memory_share_of_poiseuille": mem_flux / (visc_flux + mem_flux),
        "ghost_rhs_memory_contribution_L1": ghost_mem_delta,
        "ghost_rhs_gnf_L1": float(np.abs(ghost_gnf).sum()),
        "ghost_relative": (ghost_mem_delta / float(np.abs(ghost_gnf).sum())
                           if np.abs(ghost_gnf).sum() > 0 else None),
        "note": ("R-8.2: D_total enters _ghost(D_total) as well as A, and the "
                 "ghost RHS is EXACTLY LINEAR in D (verified).  A memory flux "
                 "share from the mobility alone omits this boundary term; it "
                 "is reported here rather than left out.")}

    # ---- the sign gate, with its validity precondition (R-8.5) -----------
    cos_th = np.cos(sph.mesh.TH)
    out["sign_gate"] = {
        "cos_theta_min": float(cos_th.min()), "cos_theta_max": float(cos_th.max()),
        "precondition_cos_positive": bool(cos_th.min() > 0.0),
        "precondition_note": ("valid BECAUSE theta_max = %.1f deg < 90 deg.  "
                              "hdot = -squeeze_phase omega eps c cos(theta), so "
                              "cos > 0 over the whole land makes hdot < 0 "
                              "everywhere and mem_factor > 0 everywhere.  A "
                              "land past 90 deg would split sign(hdot) and "
                              "make a GLOBAL load-reduction claim unsafe."
                              % rep.THETA_MAX_DEG),
        "mem_factor_min": float(mem_factor.min()),
        "mem_factor_all_positive": bool(mem_factor.min() > 0.0),
        "memory_reduces_load": bool(out["D1_frozen"]["delta_memory"] < 0.0),
        "expected": ("load REDUCTION -- PHYSICS_NOTES.md, Phan-Thien & Tanner "
                     "(1983); Phan-Thien, Dudek, Boger & Tirtaatmadja (1985)")}
    return out


def main() -> int:
    idx = json.loads((FIELDS / "index.json").read_text())
    t0 = time.time()
    results, failures = [], []
    for pt in idx["points"]:
        tag = pt["point_tag"]
        x = json.loads((FIELDS / f"{tag}.json").read_text())["inputs"]
        print(f"  {tag} ...", flush=True)
        try:
            r = decompose(tag, x)
        except Exception as exc:                     # never silently swallowed
            failures.append({"point_tag": tag, "error": f"{type(exc).__name__}: {exc}"})
            print(f"    ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        results.append(r)
        a = r["additivity_identity"]
        print("    additivity rel residual %.3e  holds=%s | memory share %.4f"
              % (a["rel_residual"], a["holds"], r["D1_frozen"]["share_memory"]),
              flush=True)

    out = {
        "artefact": "boxc_phase6_decomposition",
        "expectation": "E4",
        "generated": time.strftime("%Y-%m-%d"),
        "band_caveat": ("exports/cme/fields/index.json carries band = 'lo'.  "
                        "These six points are the sigma* LO band end ONLY, so "
                        "they do not satisfy standing rule 15 on their own.  "
                        "Reported as a lo-band result and NOT as if it were "
                        "band-symmetric."),
        "design_chart_corners": ("OWED.  Brief sec 7.2 also asks for the "
                                 "corners of the sec 5 design chart, which "
                                 "does not exist until Phase 4."),
        "additivity_rtol_registered_before_running": ADDITIVITY_RTOL,
        "points": results, "failures": failures,
        "wall_seconds": time.time() - t0,
    }
    # HARD FAILURE, not a fork (R-8.4)
    bad = [r["point_tag"] for r in results if not r["additivity_identity"]["holds"]]
    out["additivity_violations"] = bad
    out["HARD_FAILURE"] = bool(bad)
    signs = [r["point_tag"] for r in results
             if not r["sign_gate"]["memory_reduces_load"]]
    out["sign_gate_violations"] = signs
    out["SIGN_GATE_STOP"] = bool(signs)

    p = _ROOT / "results" / "ch9" / "boxc_phase6_decomposition.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {p}")
    print(f"  points {len(results)}  failures {len(failures)}  "
          f"wall {time.time()-t0:.1f} s")
    print(f"  additivity violations : {bad or 'none'}  -> HARD_FAILURE={out['HARD_FAILURE']}")
    print(f"  sign-gate violations  : {signs or 'none'} -> STOP={out['SIGN_GATE_STOP']}")
    return 2 if (out["HARD_FAILURE"] or out["SIGN_GATE_STOP"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
