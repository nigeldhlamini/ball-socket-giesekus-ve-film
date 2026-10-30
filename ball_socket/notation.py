"""Registered symbols that share a letter elsewhere (R-43.3, R-54.6, R-55.8).

Two unrelated quantities are both called beta in the record:

  beta_r      pocket RELIEF FRACTION, pi R_b^2 sin^2(theta_min) / A_p.  DERIVED from
              theta_min; not an independent design axis (R-47.5) and not on the
              design chart (R-54.6).
  beta_swash  SWASHPLATE ANGLE; appears only as ``beta_max`` in the stroke identity
              2 R_p tan(beta_max) (campaign_config_v6.py:487-510) and the inertia
              band.  The swash/kinematic axis itself is descoped.

Frozen modules keep their names (``campaign_config_v4.relief_fraction``; the E13
spec's ``beta_derived`` key is inside a hashed artefact).  Live code, macros and
figure axes use ``beta_r`` / ``$\\beta_r$`` for the relief fraction; the guard in
tests/test_notation_beta_r.py fails on a figure axis labelled with a bare beta.
"""
from __future__ import annotations

SYMBOLS = {
    "beta_r": {"tex": r"\beta_r", "what": "pocket relief fraction A_pocket / A_p",
               "status": "DERIVED from theta_min",
               "source": "ball_socket.campaign_config_v4.relief_fraction"},
    "beta_swash": {"tex": r"\beta_{\max}", "what": "swashplate angle (stroke identity only)",
                   "status": "band; swash axis descoped",
                   "source": "ball_socket.campaign_config_v6.derived_R_p"},
}


def beta_r(theta_min_deg: float) -> float:
    """Relief fraction; delegates to the frozen helper, never re-derived."""
    from .campaign_config_v4 import relief_fraction
    return float(relief_fraction(theta_min_deg))
