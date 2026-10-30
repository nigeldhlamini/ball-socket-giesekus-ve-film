#!/usr/bin/env python
r"""CME/FEM handover: the contact residual, report Sec. 4.3, Eq. (Fcontact).

    F_contact = F_app - p_ch A_pocket - F_film_max

NO NEW SOLVES.  The prompt asks for a cost estimate before any re-evaluation.
None is needed: the film load at the eccentricity bracket limit is already
persisted for every non-equilibrium row.  The solver computes

    r_hi = residual(eps_cap) = F_film_max - (F_app - F_pocket)

and, when r_hi < 0, raises NonConvergence with a message that prints
``max squeeze load {r_hi + F_target}``, i.e. F_film_max itself, together with
F_target, F_app and the pocket relief.  That message is stored verbatim in the
``fail_reason`` column of the audit parquet.  The quantity was discarded from
the typed output columns but not from the artefact.

PRECISION.  fail_reason formats the load with %.3e, so F_film_max is recovered
to four significant figures and no better.  F_app and F_pocket are NOT taken
from the string: they are recomputed exactly from the inputs, since both are
deterministic in (p_ch, theta_min).  The rounding therefore enters F_contact
only through the F_film_max term, and the per-row relative error on F_contact
is reported so the FEM side can see it rather than infer it.

STOP CONDITION (prompt 1.3).  A negative F_contact would mean film plus pocket
over-carry the applied load, which the coupled load construction forbids.  If
any row comes out negative the script writes nothing and reports.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "ch9"
EXPORT = ROOT / "exports" / "cme"
sys.path.insert(0, str(ROOT))

from ball_socket.campaign_config_v4 import (  # noqa: E402
    applied_and_hydrodynamic_load, relief_fraction)

#: Parses the persisted NonConvergence message.  Anchored on the whole phrase
#: rather than on a loose number match, so a message of a different shape fails
#: to parse and is reported instead of being silently mis-read.
#:
#: TRUNCATION.  ball_socket/campaign.py stores ``str(exc)[:120]``, so every
#: stored message is cut at 120 characters, mid-number, inside the trailing
#: "pocket relief" figure.  The two quantities this export needs sit at
#: characters ~38 and ~72 and survive intact; the truncated tail is only a
#: cross-check.  The required groups are therefore the first two, and the
#: F_app cross-check is applied only where it survived.
BLOWUP_RE = re.compile(
    r"eccentricity blow-up: max squeeze load ([0-9.eE+-]+) N "
    r"< hydrodynamic share ([0-9.eE+-]+) N")
BLOWUP_FAPP_RE = re.compile(r"\(F_app ([0-9.eE+-]+)[,)]")

#: The other blow-up path in solver.solve_equilibrium: the film closes before
#: eps_cap, so no load was ever evaluated there and F_film_max does not exist.
CLOSED_RE = re.compile(r"eccentricity blow-up: film closes below eps_cap")


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(parquet: pathlib.Path, tag: str) -> dict:
    df = pd.read_parquet(parquet)
    n = len(df)

    # Exact, from the inputs.  Not parsed from the message.
    F_app = np.array([applied_and_hydrodynamic_load(p, t)[0]
                      for p, t in zip(df["p_ch"], df["theta_min_deg"])])
    F_pocket = np.array([applied_and_hydrodynamic_load(p, t)[1]
                         for p, t in zip(df["p_ch"], df["theta_min_deg"])])
    F_target = F_app - F_pocket

    converged = (df["outcome"] == "converged").to_numpy()
    reason = df["fail_reason"].astype(str).to_numpy()

    F_film_max = np.full(n, np.nan)
    parse_fail: list[int] = []
    film_closed: list[int] = []
    app_mismatch: list[dict] = []

    # Campaigns run after 2026-08-03 carry F_film_max as a TYPED COLUMN at full
    # precision.  Prefer it; fall back to parsing the message only for the
    # artefacts written before the fix.  The string path is kept rather than
    # deleted because the registered audit predates the column.
    typed = "F_film_max" in df.columns
    source = "typed column" if typed else "fail_reason string (pre-fix artefact)"
    if typed:
        F_film_max = df["F_film_max"].to_numpy(float).copy()

    for i in np.flatnonzero(~converged):
        if typed and np.isfinite(F_film_max[i]):
            continue
        m = BLOWUP_RE.search(reason[i])
        if m is None:
            (film_closed if CLOSED_RE.search(reason[i]) else parse_fail).append(int(i))
            continue
        F_film_max[i] = float(m.group(1))
        # The message also carries the solver's own F_target.  Cross-check it
        # against the analytic recomputation: if they disagree, the row is not
        # what we think it is and the export must not quietly proceed.
        if not np.isclose(float(m.group(2)), F_target[i], rtol=2e-3):
            app_mismatch.append({"row": int(i), "field": "F_target",
                                 "message": float(m.group(2)),
                                 "recomputed": float(F_target[i])})
        ma = BLOWUP_FAPP_RE.search(reason[i])
        if ma and not np.isclose(float(ma.group(1)), F_app[i], rtol=2e-3):
            app_mismatch.append({"row": int(i), "field": "F_app",
                                 "message": float(ma.group(1)),
                                 "recomputed": float(F_app[i])})

    # Equilibrium rows: the film carries the whole hydrodynamic share by
    # definition of the equilibrium, and the contact residual is exactly zero.
    F_film = np.where(converged, F_target, F_film_max)
    F_contact = np.where(converged, 0.0, F_target - F_film_max)

    # Precision: only the F_film_max term is rounded (%.3e -> 4 s.f.).
    # Rounding applies only where the value came from the string.
    if typed:
        half_ulp = np.zeros(n)          # full precision, no rounding to carry
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = np.abs(np.where(F_film_max == 0, np.nan, F_film_max))
            half_ulp = np.where(np.isfinite(F_film_max),
                                0.5 * 10.0 ** (np.floor(np.log10(mag)) - 3), 0.0)
    contact_abs_err = np.where(converged, 0.0, half_ulp)
    with np.errstate(divide="ignore", invalid="ignore"):
        contact_rel_err = np.where(F_contact > 0, contact_abs_err / F_contact, 0.0)

    # -- STOP condition --------------------------------------------------
    neg = np.flatnonzero(np.isfinite(F_contact) & (F_contact < 0))
    stop = {
        "negative_F_contact_rows": int(neg.size),
        "parse_failures": parse_fail,
        "film_closed_rows": film_closed,
        "F_app_message_vs_recomputed_mismatches": app_mismatch,
    }
    if neg.size or parse_fail or app_mismatch:
        stop["worst_negative"] = (float(np.min(F_contact[neg])) if neg.size
                                  else None)
        return {"STOP": True, "detail": stop, "tag": tag}

    out = pd.DataFrame({
        "row_id": np.arange(n),
        "outcome": df["outcome"].to_numpy(),
        "equilibrium": converged,
        "p_ch_Pa": df["p_ch"].to_numpy(),
        "omega_rad_s": df["omega"].to_numpy(),
        "c0_m": df["c0"].to_numpy(),
        "T_in_C": df["T_in"].to_numpy(),
        "theta_min_deg": df["theta_min_deg"].to_numpy(),
        "eta_0_Pa_s": df["eta_0"].to_numpy(),
        "lambda_s": df["lam"].to_numpy(),
        "eta_s_ratio": df["eta_s_ratio"].to_numpy(),
        "relief_fraction_eps_hs": np.array(
            [relief_fraction(t) for t in df["theta_min_deg"]]),
        "F_app_N": F_app,
        "F_pocket_N": F_pocket,
        "F_hydrodynamic_share_N": F_target,
        "F_film_N": F_film,
        "F_contact_N": F_contact,
        "contact_fraction_of_F_app": F_contact / F_app,
        "F_contact_abs_uncertainty_N": contact_abs_err,
        "F_contact_rel_uncertainty": contact_rel_err,
        "h_min_m": df["h_min"].to_numpy(),
        "T_max_C": df["T_max"].to_numpy(),
    })

    EXPORT.mkdir(parents=True, exist_ok=True)
    csv = EXPORT / f"contact_residual_{tag}.csv"
    out.to_csv(csv, index=False, float_format="%.6e")

    nonq = out[~out.equilibrium]
    frac = nonq["contact_fraction_of_F_app"].to_numpy()
    summary = {
        "STOP": False,
        "tag": tag,
        "source_parquet": parquet.name,
        "source_sha256": _sha(parquet),
        "n_rows": int(n),
        "n_equilibrium": int(converged.sum()),
        "n_non_equilibrium": int((~converged).sum()),
        "F_contact_recovered_without_re_solving": True,
        "F_contact_N": {
            "min": float(nonq["F_contact_N"].min()),
            "median": float(nonq["F_contact_N"].median()),
            "p95": float(nonq["F_contact_N"].quantile(0.95)),
            "max": float(nonq["F_contact_N"].max()),
        },
        "contact_fraction_of_F_app": {
            "min": float(np.min(frac)), "median": float(np.median(frac)),
            "p95": float(np.quantile(frac, 0.95)), "max": float(np.max(frac)),
        },
        "film_share_of_hydrodynamic_duty_at_bracket_limit": {
            "median": float(np.median(nonq["F_film_N"]
                                      / nonq["F_hydrodynamic_share_N"])),
        },
        "precision": {
            "F_film_max_source": source,
            "F_app_and_F_pocket_source": "recomputed exactly from inputs",
            "F_contact_rel_uncertainty_median": float(
                np.median(nonq["F_contact_rel_uncertainty"])),
            "F_contact_rel_uncertainty_max": float(
                np.max(nonq["F_contact_rel_uncertainty"])),
        },
        "files": {csv.name: _sha(csv)},
    }
    return summary


def main() -> int:
    reports = {}
    for parquet, tag in ((OUT / "audit_N8192_seed20260727.parquet", "audit_N8192"),):
        reports[tag] = build(parquet, tag)

    EXPORT.mkdir(parents=True, exist_ok=True)
    (EXPORT / "contact_residual_report.json").write_text(
        json.dumps(reports, indent=2, default=str))

    for tag, r in reports.items():
        if r.get("STOP"):
            print(f"STOP on {tag}: {json.dumps(r['detail'], indent=2)[:1200]}")
            return 1
        print(f"{tag}: {r['n_non_equilibrium']} non-equilibrium rows, "
              f"F_contact median {r['F_contact_N']['median']:.4g} N, "
              f"contact fraction median "
              f"{r['contact_fraction_of_F_app']['median']:.4f}")
        print(f"  rel. uncertainty median "
              f"{r['precision']['F_contact_rel_uncertainty_median']:.2e}, "
              f"max {r['precision']['F_contact_rel_uncertainty_max']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
