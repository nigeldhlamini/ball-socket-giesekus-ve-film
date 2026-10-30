"""Observed order of convergence, and the domain the estimator is valid on.

WHY THIS MODULE EXISTS (ruling R-35.2).  A grid audit reported three pairwise
percentage changes per metric and they were read as a convergence sequence.
They are not one.  For a refinement N -> rN with error A/N^p the pairwise change
is

    d = (A / N^p) (1 - r^-p)

so d depends on BOTH N and r.  The box-D candidate list is 32 -> 48 -> 64 -> 96,
whose ratios are 1.5, 1.333, 1.5 -- NOT uniform.  Across unequal r the three
numbers are incommensurable, and legacy ``h_min`` reading 11.57, 6.14, 6.32 was
called "not converging" when it is converging at first order: the apparent
uptick is the second and third steps having different r, nothing more.

THE ESTIMATOR HAS A DOMAIN, AND IT IS CHECKED RATHER THAN ASSUMED.  When two
steps share the same r, the ``(1 - r^-p)`` factor cancels exactly and

    d_i / d_j = (N_j / N_i) ** p          exact, independent of A

That is available for this list because steps 1 and 3 both have r = 1.5 with N
doubled.  It is NOT available for an arbitrary list, and a function that
silently returned a number anyway would be the same defect one level up -- a
result reported outside the domain that makes it meaningful.  ``observed_order``
therefore returns ``None`` with a stated reason when no equal-ratio pair exists.

That is the same lesson as the finest-candidate guard: a convergence study's
candidate list is itself a registered choice, and both the list's REFINEMENT
RATIOS and its ENDPOINT can produce a verdict the data does not support.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


def _ratio(a: Sequence[int], b: Sequence[int]) -> float:
    """Refinement ratio between two grids, required consistent on both axes."""
    ra = float(b[0]) / float(a[0])
    rb = float(b[1]) / float(a[1])
    if not math.isclose(ra, rb, rel_tol=1e-9):
        raise ValueError(
            f"anisotropic refinement {tuple(a)} -> {tuple(b)}: ratios "
            f"{ra:.4f} and {rb:.4f} differ, so a single order is not defined")
    return ra


def equal_ratio_pairs(grids: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    """Index pairs of STEPS that share a refinement ratio.

    Step i is grids[i] -> grids[i+1].  Two steps with the same r admit the exact
    estimator; anything else does not.
    """
    steps = list(range(len(grids) - 1))
    rs = {}
    for i in steps:
        try:
            rs[i] = _ratio(grids[i], grids[i + 1])
        except ValueError:
            continue
    out = []
    for i in steps:
        for j in steps:
            if j <= i or i not in rs or j not in rs:
                continue
            if math.isclose(rs[i], rs[j], rel_tol=1e-9):
                out.append((i, j))
    return out


def observed_order(grids: Sequence[Sequence[int]],
                   changes: Sequence[float]) -> dict:
    """Observed order p from two steps sharing a refinement ratio.

    ``changes[i]`` is the pairwise change for step ``grids[i] -> grids[i+1]``.

    Returns a dict with ``p`` (or ``None``), the pair used, and ``why`` when the
    estimator is unavailable.  A ``None`` is a refusal, not a zero.
    """
    pairs = equal_ratio_pairs(grids)
    if not pairs:
        return {"p": None, "pair": None,
                "why": ("no two steps in this candidate list share a refinement "
                        "ratio, so (1 - r^-p) does not cancel and no order can "
                        "be extracted without assuming one")}
    # prefer the widest separation in N: the largest lever arm
    i, j = max(pairs, key=lambda ij: float(grids[ij[1]][0]) / float(grids[ij[0]][0]))
    di, dj = float(changes[i]), float(changes[j])
    if not (math.isfinite(di) and math.isfinite(dj)) or di <= 0 or dj <= 0:
        return {"p": None, "pair": [i, j],
                "why": "a pairwise change is zero or non-finite; log undefined"}
    if max(abs(float(c)) for c in changes) < NOISE_FLOOR_PCT:
        return {"p": None, "pair": [i, j], "at_noise_floor": True,
                "why": (f"every pairwise change is below {NOISE_FLOOR_PCT} %, at "
                        f"round-off / solver-tolerance level: an order taken "
                        f"from these is a ratio of two noise figures.  The "
                        f"metric is grid-insensitive, which is the useful "
                        f"statement -- not that it converges at some order")}
    n_ratio = float(grids[j][0]) / float(grids[i][0])
    p = math.log(di / dj) / math.log(n_ratio)
    return {"p": p, "pair": [list(grids[i]), list(grids[i + 1]),
                             list(grids[j]), list(grids[j + 1])],
            "N_ratio": n_ratio, "d_ratio": di / dj,
            "exact": "the (1 - r^-p) factor cancels: both steps share r"}


def predicted_changes(grids: Sequence[Sequence[int]], p: float) -> list[float]:
    """Pairwise changes a pure order-``p`` error would give, normalised to 1."""
    d = []
    for i in range(len(grids) - 1):
        N = float(grids[i][0])
        r = _ratio(grids[i], grids[i + 1])
        d.append((1.0 / N ** p) * (1.0 - r ** (-p)))
    return [x / d[0] for x in d] if d and d[0] else d


#: Pairwise changes below this (per cent) are at round-off / solver-tolerance
#: level, and an order extracted from them is a ratio of two noise figures.
#: ``F`` on the corrected-bracket audit moves 0.037, 0.006, 0.029 % and yields
#: p = 0.34 from nothing at all.  Reporting that as a convergence order would be
#: the same defect as reading unequal-ratio steps as a sequence: a number
#: produced outside the domain that gives it meaning.
NOISE_FLOOR_PCT = 0.05


def classify(p: float | None) -> str:
    """Plain words for an observed order.

    NOTE the bands cannot distinguish an order that an ERROR MODEL PREDICTS from
    one that merely lands nearby.  Legacy ``h_min`` at p = 0.87 is first order
    because ``h_legacy - h_bnd = c eps sin(theta_min) d_theta / 2`` is O(d_theta)
    with no further grid structure, AND its measured ratios match the
    first-order signature; ``T_max_bnd`` at p = 0.67 has no such model and is
    simply weak.  Use the signature check, not the band, to make that call.
    """
    if p is None:
        return "not determinable on this candidate list"
    if p < 0:
        return "DIVERGING -- refinement makes it worse"
    if p < 0.8:
        return "sub-first order"
    if p < 1.6:
        return "first order"
    if p < 2.6:
        return "second order"
    return "second order or better"


def order_table(grids: Sequence[Sequence[int]],
                metrics: dict[str, Sequence[float]]) -> dict[str, dict]:
    """Observed order plus the order-p signature check, per metric.

    The signature check is the confirmation R-35.2 used: a pure first-order
    error on THIS list predicts pairwise changes in the ratio 1 : 0.500 : 0.500,
    and legacy ``h_min`` measured 1 : 0.531 : 0.546.  Reporting the measured
    ratios beside the predicted ones is what turns an order estimate into a
    claim a reader can check.
    """
    out: dict[str, dict] = {}
    for name, changes in metrics.items():
        est = observed_order(grids, changes)
        row: dict[str, Any] = dict(est)
        row["verdict"] = classify(est["p"])
        d0 = float(changes[0]) if changes and float(changes[0]) else None
        row["measured_ratios"] = ([round(float(c) / d0, 3) for c in changes]
                                  if d0 else None)
        if est["p"] is not None and est["p"] > 0:
            row["predicted_at_observed_p"] = [
                round(x, 3) for x in predicted_changes(grids, est["p"])]
            row["predicted_at_first_order"] = [
                round(x, 3) for x in predicted_changes(grids, 1.0)]
        out[name] = row
    return out
