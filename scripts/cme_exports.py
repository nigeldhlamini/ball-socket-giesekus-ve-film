#!/usr/bin/env python
r"""CME/FEM handover exports: prompt items 2.1 and 2.4.

2.1  Regime map over the sampled envelope, both sigma* band ends in one file
     with the band end as a column, plus a README naming units and conventions.
2.4  Pocket relief fraction eps_hs = A_pocket/A_p tabulated over the registered
     theta_min axis.

Items 2.2 (film traction field) and 2.3 (thermal field) are NOT produced here.
They are not exports of existing artefacts: the campaign persisted scalar
performance only, and no pressure or temperature field was ever written to
disk.  Producing them requires re-solving, which the prompt excludes; a costing
is written to cme_field_export_costing.json instead so the decision can be
made on numbers.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "ch9"
EXPORT = ROOT / "exports" / "cme"
sys.path.insert(0, str(ROOT))

import ball_socket.representative as rep                       # noqa: E402
from ball_socket.campaign_config_v4 import relief_fraction     # noqa: E402

#: Registered full-film threshold on the film parameter Lambda = h_min/sigma*.
LAMBDA_FULL_FILM = 3.0


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 2.1 regime map
# ---------------------------------------------------------------------------

def regime_map(df: pd.DataFrame) -> pd.DataFrame:
    sig_lo, sig_hi = rep.sigma_star_band()
    rows = []
    for band, sigma in (("lo", sig_lo), ("hi", sig_hi)):
        lam = df[f"lambda_ratio_{band}"].to_numpy(float)
        conv = (df["outcome"] == "converged").to_numpy()
        regime = np.where(
            ~conv, "no-equilibrium",
            np.where(lam >= LAMBDA_FULL_FILM, "full-film", "mixed-or-boundary"))
        for cls in ("mineral", "water_glycol"):
            rows.append(pd.DataFrame({
                "row_id": np.arange(len(df)),
                "sigma_star_band": band,
                "sigma_star_m": sigma,
                "fluid_class": cls,
                "p_ch_Pa": df["p_ch"].to_numpy(),
                "omega_rad_s": df["omega"].to_numpy(),
                "c0_m": df["c0"].to_numpy(),
                "T_in_C": df["T_in"].to_numpy(),
                "theta_min_deg": df["theta_min_deg"].to_numpy(),
                "eta_0_Pa_s": df["eta_0"].to_numpy(),
                "lambda_s": df["lam"].to_numpy(),
                "eta_s_ratio": df["eta_s_ratio"].to_numpy(),
                "regime": regime,
                "film_parameter_Lambda": lam,
                "h_min_m": df["h_min"].to_numpy(),
                "T_max_C": df["T_max"].to_numpy(),
                "feasible": (df[f"outcome_{cls}_{band}"] == "feasible").to_numpy(),
                "binding_constraint": df[f"binding_{cls}_{band}"].to_numpy(),
            }))
    return pd.concat(rows, ignore_index=True)


README = """# Ball-socket regime map -- CME/FEM handover

Source: registered uniform Sobol audit of the rated envelope, {n_rows} sampled
operating points, parquet sha256 {sha}.

One row per (operating point x sigma* band end x fluid class).  The band end is
a COLUMN, not a separate file: which regime an operating point falls in depends
on where in the composite-roughness band the finish sits, and that dependence is
a finding of the campaign rather than a nuisance parameter.  Neither band end is
a default; use both.

## Columns

| column | units | convention |
|---|---|---|
| row_id | - | index into the source audit; the same operating point appears once per band end and fluid class |
| sigma_star_band | - | `lo` or `hi`, the two ends of the registered composite-roughness band |
| sigma_star_m | m | composite roughness sigma* = hypot(Rq_ball, Rq_socket) at that band end |
| fluid_class | - | `mineral` or `water_glycol`; sets the thermal service ceiling only |
| p_ch_Pa | Pa | chamber pressure; also sets the applied load, F_app = p_ch * A_p |
| omega_rad_s | rad/s | shaft speed |
| c0_m | m | radial cold clearance (isothermal; no thermal clearance correction applied) |
| T_in_C | degC | inlet temperature |
| theta_min_deg | deg | pocket edge; the film is the annulus theta in [theta_min, theta_max] |
| eta_0_Pa_s | Pa s | zero-shear viscosity |
| lambda_s | s | Giesekus relaxation time |
| eta_s_ratio | - | solvent viscosity fraction |
| regime | - | `full-film`, `mixed-or-boundary`, or `no-equilibrium`; see below |
| film_parameter_Lambda | - | h_min / sigma* at that band end |
| h_min_m | m | minimum film thickness at equilibrium; NaN where no equilibrium exists |
| T_max_C | degC | peak film temperature at equilibrium; NaN where no equilibrium exists |
| feasible | bool | satisfies all three registered constraints at that band end and fluid class |
| binding_constraint | - | `g1` film, `g2` thermal, `g3` validity, `multi`, or empty where feasible |

## Regime definition

* `full-film`        an equilibrium exists AND Lambda >= {lam_ff:g}
* `mixed-or-boundary` an equilibrium exists AND Lambda < {lam_ff:g}
* `no-equilibrium`   the equilibrium search found no eccentricity short of
                      contact at which the film carries the applied load

`no-equilibrium` is NOT missing data.  It is the load-carrying limit of the
film, and it is the majority outcome over the rated envelope.  The contact load
for those rows is in `contact_residual_audit_N8192.csv`; this file and that one
share `row_id`.

## What this file does not contain

Pressure, traction and temperature FIELDS are not here.  The campaign persisted
scalar performance only and no field was written to disk, so they cannot be
exported without re-solving.  See `cme_field_export_costing.json`.

## Caveats carried from the campaign

* The clearance is the cold radial clearance; the thermal-expansion correction
  is verified as a gate but is not applied in the campaign rows.
* The applied load is quasi-static: F_app = p_ch * A_p, with inertial and
  centrifugal contributions neglected.  The kinematics module quantifies what
  that omits.
* Rows are a Sobol sample of the rated envelope, not a duty cycle.  Do not read
  the regime proportions as time fractions.
"""


# ---------------------------------------------------------------------------
# 2.4 pocket relief fraction
# ---------------------------------------------------------------------------

def relief_table() -> pd.DataFrame:
    import json as _json
    cfg = _json.loads(
        (ROOT / "ball_socket" / "campaign_frozen_v4.json").read_text())["config"]
    lo, hi = cfg["geometry"]["theta_min_axis_deg"]
    th = np.linspace(lo, hi, 41)
    R_b, A_p = rep.R_B_M, rep.A_P_M2
    A_pocket = np.array([rep.pocket_relief_area(R_b, np.deg2rad(t)) for t in th])
    return pd.DataFrame({
        "theta_min_deg": th,
        "A_pocket_m2": A_pocket,
        "A_p_m2": A_p,
        "relief_fraction_eps_hs": np.array([relief_fraction(t) for t in th]),
        "closed_form_check": np.pi * R_b ** 2 * np.sin(np.deg2rad(th)) ** 2 / A_p,
    })


# ---------------------------------------------------------------------------
# 2.2 / 2.3 costing
# ---------------------------------------------------------------------------

def costing(df: pd.DataFrame) -> dict:
    sig_lo, _ = rep.sigma_star_band()
    conv = (df["outcome"] == "converged").to_numpy()
    ff_lo = conv & (df["lambda_ratio_lo"].to_numpy(float) >= LAMBDA_FULL_FILM)
    ff_hi = conv & (df["lambda_ratio_hi"].to_numpy(float) >= LAMBDA_FULL_FILM)
    ff_any = ff_lo | ff_hi

    # Observed campaign throughput, from the persisted run summaries.
    obs = []
    for f in sorted(OUT.glob("audit_N*_summary.json")):
        s = json.loads(f.read_text())
        if s.get("wall_seconds") and s.get("n"):
            obs.append((s["n"], s["wall_seconds"], s["wall_seconds"] / s["n"]))
    per_row = float(np.median([o[2] for o in obs])) if obs else float("nan")

    ntheta, nphi = 96, 64          # solver grid, ball_socket.solver defaults
    bytes_per_field = ntheta * nphi * 8
    return {
        "why_this_is_not_an_export": (
            "The campaign persisted scalar performance only.  No pressure or "
            "temperature field was written to disk at any point, so items 2.2 "
            "and 2.3 cannot be produced by post-processing.  They require "
            "re-solving the full-film rows with field output enabled, which "
            "the prompt's 'NO new solver runs' excludes.  Costed, not run."),
        "rows_needing_re_solve": {
            "full_film_lo_band": int(ff_lo.sum()),
            "full_film_hi_band": int(ff_hi.sum()),
            "union_either_band": int(ff_any.sum()),
            "all_converged": int(conv.sum()),
        },
        "observed_throughput": {
            "runs_used": [{"n": o[0], "wall_s": round(o[1], 1),
                           "s_per_row": round(o[2], 4)} for o in obs],
            "median_s_per_row": round(per_row, 4),
            "note": "wall time on the campaign's worker pool, not single-core",
        },
        "estimated_wall_time_s": {
            "union_either_band": round(per_row * int(ff_any.sum()), 1),
            "all_converged": round(per_row * int(conv.sum()), 1),
            "note": "solve time only; field serialisation is additional and is "
                    "not in the measured rate",
        },
        "estimated_output_size": {
            "grid": f"{ntheta} x {nphi}",
            "bytes_per_field_per_row": bytes_per_field,
            "fields_per_row": ["pressure", "film_thickness", "temperature",
                               "wall_heat_flux"],
            "total_MB_union_either_band": round(
                4 * bytes_per_field * int(ff_any.sum()) / 1e6, 1),
        },
        "code_change_required": (
            "solver.solve_equilibrium returns a scalar Performance; the field "
            "arrays exist inside _forward and are discarded on return.  A "
            "field-export path would have to persist them, which is a change "
            "to the solve path and therefore not a post-processing task."),
        "decision_requested": (
            "authorise the re-solve with a row budget, or narrow 2.2/2.3 to a "
            "named handful of operating points"),
    }


def main() -> int:
    EXPORT.mkdir(parents=True, exist_ok=True)
    src = OUT / "audit_N8192_seed20260727.parquet"
    df = pd.read_parquet(src)

    rm = regime_map(df)
    rm_path = EXPORT / "regime_map_audit_N8192.csv"
    rm.to_csv(rm_path, index=False, float_format="%.6e")

    readme = README.format(n_rows=len(df), sha=_sha(src)[:16],
                           lam_ff=LAMBDA_FULL_FILM)
    rd_path = EXPORT / "README_regime_map.md"
    rd_path.write_text(readme, encoding="utf-8")

    rt = relief_table()
    rt_path = EXPORT / "pocket_relief_fraction.csv"
    rt.to_csv(rt_path, index=False, float_format="%.6e")

    cost = costing(df)
    cost_path = EXPORT / "cme_field_export_costing.json"
    cost_path.write_text(json.dumps(cost, indent=2))

    counts = rm.groupby(["sigma_star_band", "regime"]).size().unstack(fill_value=0)
    print(counts.to_string())
    print()
    print(f"relief fraction over the registered axis: "
          f"{rt.relief_fraction_eps_hs.min()*100:.1f} to "
          f"{rt.relief_fraction_eps_hs.max()*100:.1f} per cent")
    print(f"\n2.2/2.3 costed, NOT run: "
          f"{cost['rows_needing_re_solve']['union_either_band']} rows, "
          f"~{cost['estimated_wall_time_s']['union_either_band']/60:.0f} min, "
          f"~{cost['estimated_output_size']['total_MB_union_either_band']} MB")
    for p in (rm_path, rd_path, rt_path, cost_path):
        print(f"  {_sha(p)[:16]}  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
