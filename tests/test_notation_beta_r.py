"""R-55.8 / R-43.3: the relief fraction is beta_r in live code and on figure axes.

The guard scans every script for a matplotlib label/title/text string carrying a
bare TeX beta (``\\beta`` not followed by ``_``).  Failing looks like: a figure
script that labels an axis ``$\\beta$`` -- the notation hazard R-43.3 recorded.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
_LABEL = re.compile(r"set_[xy]?label\(|set_title\(|\.text\(|label\s*=")
_BARE = re.compile(r"\\\\?beta(?![_a-zA-Z])")


def _offending_lines(src: str) -> list[str]:
    return [l for l in src.splitlines()
            if _LABEL.search(l) and _BARE.search(l) and not l.lstrip().startswith("#")]


def test_no_figure_axis_uses_bare_beta():
    bad = []
    for p in sorted((ROOT / "scripts").glob("*.py")):
        for l in _offending_lines(p.read_text(encoding="utf-8", errors="replace")):
            bad.append(f"{p.name}: {l.strip()}")
    assert not bad, "bare beta on a figure label (use beta_r):\n" + "\n".join(bad)


def test_guard_is_armed():
    """Positive and negative controls for the scanner itself."""
    assert _offending_lines('ax.set_ylabel(r"relief fraction $\\beta$")')
    assert not _offending_lines('ax.set_ylabel(r"relief fraction $\\beta_r$")')
    assert not _offending_lines('ax.set_ylabel(r"$\\beta_{\\max}$")')


def test_beta_r_delegates_to_frozen_helper():
    from ball_socket.campaign_config_v4 import relief_fraction
    from ball_socket.notation import beta_r, SYMBOLS
    assert beta_r(30.0) == float(relief_fraction(30.0))
    assert set(SYMBOLS) == {"beta_r", "beta_swash"}
