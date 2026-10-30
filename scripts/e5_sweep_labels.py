"""Read box B's E5 sweep with labels re-derived from each row's solve inputs (ERR-26).

`results/ch9/e5_sweep.parquet` (2026-07-29) predates R-21.6's deterministic row
order.  `run_ch9_campaign.stage_e5` attached grade / nu_100 / corner /
thick_arm / in_box BY POSITION to rows that `run_batch` then returned in
`as_completed` order, so the labels sit on the wrong rows: 150 of 160 rows carry
at least one wrong label (corner 144, nu_100 93, thick_arm 18, grade 6).  The
solve INPUTS (omega, p_ch, c0, T_in, eta_0) travelled with the solve and are
right, and they identify the registered point uniquely.

The artefact is box B's and is not edited (standing rule).  Every consumer
reads it through `load()`, which rebuilds the labels from the inputs and keeps
the stored ones as `*_as_stored` so the defect stays visible.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PATH = ROOT / "results" / "ch9" / "e5_sweep.parquet"
LABELS = ("grade", "nu_100", "corner", "thick_arm", "in_box")


def _key(omega, p_ch, c0, T_in, eta_0):
    return (round(float(omega), 6), round(float(p_ch), 1), round(float(c0), 12),
            round(float(T_in), 6), float(f"{float(eta_0):.9g}"))


def truth_table() -> dict:
    from ball_socket import e5_fluid_spec as E5
    tab = {}
    for p in E5.build_e5_points():
        k = _key(p.omega_rad_s, p.p_ch_bar * 1e5, p.c0_um * 1e-6, p.T_C, p.eta0)
        if k in tab:
            raise ValueError(f"two registered E5 points share inputs {k}")
        tab[k] = dict(grade=p.grade, nu_100=p.nu_100, corner=p.corner,
                      thick_arm=p.thick_arm,
                      in_box=E5.AXES["eta0"][0] <= p.eta0 <= E5.AXES["eta0"][1])
    return tab


def load(path: pathlib.Path = PATH) -> pd.DataFrame:
    df = pd.read_parquet(path).reset_index(drop=True)
    tab = truth_table()
    rows = [tab[_key(r.omega, r.p_ch, r.c0, r.T_in, r.eta_0)] for r in df.itertuples()]
    if len({_key(r.omega, r.p_ch, r.c0, r.T_in, r.eta_0) for r in df.itertuples()}) != len(tab):
        raise ValueError("the sweep does not cover each registered E5 point exactly once")
    fixed = pd.DataFrame(rows)
    for k in LABELS:
        df[f"{k}_as_stored"] = df[k]
        df[k] = fixed[k].values
    df.attrs["mislabelled_rows"] = int(sum(
        (df[f"{k}_as_stored"] != df[k]) for k in ("grade", "nu_100", "corner")).astype(bool).sum())
    return df
