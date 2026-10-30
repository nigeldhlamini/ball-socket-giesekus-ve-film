"""R-50.1: the fluid-to-band map.  No solves.

The campaign sweeps the rheological space (eta_0, lambda, eta_s/eta_0) as a
registered band rather than fixing a named product.  A reader with a specific
fluid must be able to LOCATE it in that space.  For each candidate class this
records (eta_0, lambda, eta_s/eta_0) at a stated temperature, where each lands
in the registered band (log-position in [0, 1], or outside), and the squeeze
Deborah number at the public rated maximum corner.

R-51.1: A PLACEMENT RULE, NOT A TABLE OF GUESSED VALUES.  The standing rule is
CITED or VERIFY with no third category, so the DECLARED rows that stood here
(water-glycol and polymer classes from the parametric library) are DELETED.  The
map instead supplies the criterion a reader applies to their own measured fluid:
at the rated corner the registered g3 limit De_sq <= 0.2 is reached at
lambda_limit; a fluid whose measured relaxation time at its service temperature
is below it lies inside the paper's validity domain, and above it outside, where
the paper asserts nothing.  That is DERIVED from a registered constraint and needs
no citation.  Only the mineral rows remain, because their chain is cited.

PROVENANCE OF THE REMAINING ROWS:
  * mineral eta_0: ISO 3448 nu_40 (CITED) and ASTM D341 at service temperature
    via ball_socket.e5_fluid_spec (DERIVED), density by the campaign convention;
  * mineral lambda: NOT CITED -- mineral hydraulic oils are treated as Newtonian
    at these rates; recorded as "below the axis floor", not as a number;

De_sq at the rated corner uses the registered closed form
De_sq = lambda * phi * omega * eps * cos(theta_min), with eps = 1 and
theta_min at the registered floor (20 deg) -- the largest value the corner
reaches short of the geometric ceiling -- so it is an UPPER figure per fluid.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOLVER = ROOT.parent.parent / "giesekus-solver" / "giesekus-solver"
sys.path.insert(0, str(SOLVER))
OUT = ROOT / "results" / "ch9" / "fluid_to_band_map.json"


def _logpos(v, lo, hi):
    if v is None or v <= 0:
        return None
    return (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))


def main() -> int:
    from ball_socket import e5_fluid_spec as E5
    from ball_socket.campaign_config_v4 import INPUT_SPACE
    import ball_socket.campaign as C
    ax = {d.name: (d.lower, d.upper) for d in INPUT_SPACE}
    omega = E5.CORNERS["maximum"]["rpm"] * 2 * math.pi / 60.0
    de_per_lam = C.SQUEEZE_PHASE * omega * 1.0 * math.cos(math.radians(ax["theta_min_deg"][0]))

    rows = []
    for grade, nu40 in E5.ISO_VG.items():
        T = E5.SERVICE_T[grade]
        for nu100 in (min(E5.NU100_SWEEP), max(E5.NU100_SWEEP)):
            rows.append({"class": f"mineral VG{grade}", "T_C": T, "note": f"nu_100 = {nu100:g}",
                         "eta_0": E5.eta_at(nu40, nu100, T), "lam": None, "eta_s_ratio": 1.0,
                         "provenance": "eta_0 CITED (ISO 3448) + DERIVED (ASTM D341); "
                                       "lambda NOT CITED (treated as Newtonian)"})
        rows.append({"class": f"mineral VG{grade}", "T_C": 40.0, "note": "ISO grade point",
                     "eta_0": nu40 * 1e-6 * E5.density(40.0), "lam": None, "eta_s_ratio": 1.0,
                     "provenance": "eta_0 CITED (ISO 3448 nu_40) x campaign density; "
                                   "lambda NOT CITED"})
    for r in rows:
        e, l = r["eta_0"], r["lam"]
        r["eta_0_logpos"] = _logpos(e, *ax["eta_0"])
        r["eta_0_in_band"] = bool(ax["eta_0"][0] <= e <= ax["eta_0"][1])
        r["lam_logpos"] = _logpos(l, *ax["lam"]) if l else None
        r["lam_in_band"] = (bool(ax["lam"][0] <= l <= ax["lam"][1]) if l
                            else "below floor (Newtonian)")
        r["eta_s_ratio_in_band"] = (bool(ax["eta_s_ratio"][0] <= r["eta_s_ratio"] <= ax["eta_s_ratio"][1])
                                    if r["eta_s_ratio"] is not None else None)
        r["De_sq_corner_upper"] = de_per_lam * l if l else 0.0
        r["in_g3_domain_at_corner"] = bool(r["De_sq_corner_upper"] <= 0.2)
    rep = {"registered_band": {k: ax[k] for k in ("eta_0", "lam", "eta_s_ratio")},
           "corner": "public rated maximum, 500 bar / 4500 rpm",
           "De_sq_per_second_of_lambda": de_per_lam,
           "lambda_at_g3_limit_s": 0.2 / de_per_lam,
           "placement_rule": ("a fluid lies inside the validity domain at the rated "
                              "corner if its measured relaxation time at service "
                              "temperature is below lambda_at_g3_limit_s, and outside "
                              "if above; the paper asserts nothing about fluids above "
                              "it.  Derived from the registered g3 constraint at eps = 1 "
                              "and theta_min = 20 deg, the most severe registered "
                              "condition."),
           "rows": rows}
    OUT.write_text(json.dumps(rep, indent=2))
    for r in rows:
        print(f"{r['class']:<44} {r['T_C']:>5.0f}C eta0 {r['eta_0']:.4f} "
              f"({'in' if r['eta_0_in_band'] else 'OUT'}) lam {r['lam'] if r['lam'] else '-':<10} "
              f"eta_s/eta0 {r['eta_s_ratio']:.2f} ({r['eta_s_ratio_in_band']}) "
              f"De_corner {r['De_sq_corner_upper']:.3g} g3ok {r['in_g3_domain_at_corner']}")
    print("lambda at the g3 limit at the corner: %.2e s" % rep["lambda_at_g3_limit_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
