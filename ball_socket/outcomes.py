"""Fourth-outcome types for the ball-socket campaign (Ch.9; Paper 3 C1).

The campaign harness catches exactly ``NonConvergence`` and records
``Outcome.NONCONVERGED``; nothing else is ever mapped to it.  Eccentricity
blow-up (h <= 0) must surface as this exception and is never silently clipped.
"""
from __future__ import annotations

from enum import Enum


class NonConvergence(RuntimeError):
    """Fourth outcome: eccentricity blow-up / interpenetration (h <= 0), or a
    failed equilibrium/Picard solve.  The feasibility classifier consumes this
    sentinel.  Never clip eccentricity to avoid raising it.

    Carries the load state as TYPED ATTRIBUTES, not only inside the message.

    Why (2026-08-03): the film load at the eccentricity bracket limit,
    ``F_film_max``, is the film's load capacity and converts the
    non-equilibrium population into a quantified contact-load dataset.  It was
    computed on every blow-up, printed into the exception message, and then
    discarded from the typed columns.  The campaign stores ``str(exc)[:120]``,
    which truncates that message mid-number; the quantity survived only because
    it happens to sit at character 38.  A quantity that exists only inside a
    formatted human-readable string is not persisted.  These attributes are the
    persistence.
    """

    #: WHICH EVENT, as a typed value (R-36 follow-up, 2026-09-16).  This class
    #: carries two different things under one outcome string:
    #:   * "blowup_film_closed"  -- the film closed before eps_cap (physics)
    #:   * "blowup_capacity"     -- film load at eps_cap < share (physics; box
    #:                              B's "limiting case of film failure")
    #:   * "picard_at_root"      -- brentq FOUND the equilibrium and the final
    #:                              forward re-solve's Picard iteration did not
    #:                              converge there.  An equilibrium exists; the
    #:                              solve did not answer.  NUMERICS, not physics.
    #: The corrected mixed arm produced six of the third kind, distinguishable
    #: from blow-up only by the message text -- a string is not persistence.
    KINDS = ("blowup_film_closed", "blowup_capacity", "picard_at_root")

    def __init__(self, message: str, *, F_film_max: float | None = None,
                 F_target: float | None = None, F_app: float | None = None,
                 F_pocket: float | None = None, eps_cap: float | None = None,
                 kind: str | None = None, eps_at_failure: float | None = None):
        super().__init__(message)
        if kind is not None and kind not in self.KINDS:
            raise ValueError(f"unknown NonConvergence kind {kind!r}")
        #: None only for raises that predate the discriminator
        self.kind = kind
        #: the eccentricity at which the failure occurred (the found root, for
        #: picard_at_root; eps_cap for the blow-up kinds)
        self.eps_at_failure = eps_at_failure
        #: film load at the bracket limit [N]; None where the film closed
        #: before eps_cap and no load was ever evaluated.
        self.F_film_max = F_film_max
        #: hydrodynamic share the film had to carry, F_app - F_pocket [N]
        self.F_target = F_target
        self.F_app = F_app
        self.F_pocket = F_pocket
        self.eps_cap = eps_cap

    def as_row(self) -> dict:
        """Typed fields for the campaign row.  Keys are stable column names."""
        return {"F_film_max": self.F_film_max, "F_target_at_failure": self.F_target,
                "F_app_at_failure": self.F_app,
                "F_pocket_at_failure": self.F_pocket,
                "eps_cap_at_failure": self.eps_cap,
                "nonconvergence_kind": self.kind,
                "eps_at_failure": self.eps_at_failure,
                "c_at_failure": getattr(self, "c_at_failure", None),
                "n_outer_at_failure": getattr(self, "n_outer_at_failure", None)}


class ClearanceNonConvergence(RuntimeError):
    """Box C: the thermal-clearance fixed point failed to converge.

    A DISTINCT outcome from ``NonConvergence``, never pooled with it and never
    a silent fallback to ``c0`` (brief sec 2.3; standing rule 14: never clip,
    never fall back).  The two failures mean different things -- one is a film
    that cannot carry the load, the other is an outer iteration that did not
    settle -- and their rates are reported separately.

    Carries the iteration state as TYPED ATTRIBUTES, not only inside the
    message, for the same reason ``NonConvergence`` carries ``F_film_max``: a
    quantity that exists only inside a formatted human-readable string is not
    persisted.
    """

    def __init__(self, message: str, *, c_last: float | None = None,
                 c_prev: float | None = None, T_max_last: float | None = None,
                 n_outer: int | None = None, datum: str | None = None):
        super().__init__(message)
        #: clearance at the last outer iterate [m]
        self.c_last = c_last
        #: clearance at the previous outer iterate [m]
        self.c_prev = c_prev
        #: peak film temperature at the last outer iterate [degC]
        self.T_max_last = T_max_last
        #: number of outer iterations performed before giving up
        self.n_outer = n_outer
        #: which delta_T datum was driving the expansion ("peak" | "bulk"),
        #: because the two arms of fork F1 can converge differently
        self.datum = datum

    def as_row(self) -> dict:
        """Typed fields for the campaign row.  Keys are stable column names."""
        return {"c_last": self.c_last, "c_prev": self.c_prev,
                "T_max_last_at_failure": self.T_max_last,
                "n_outer_at_failure": self.n_outer,
                "delta_T_datum_at_failure": self.datum}


class ContactLimit(RuntimeError):
    """BOX C: the load exceeds what film PLUS asperity contact can carry.

    **NOT the box-B outcome, and deliberately not its string.**  Under box B,
    ``NonConvergence`` meant the FILM could not carry the load.  Under box C
    that is a REGIME LABEL, not a failure: the joint runs mixed or boundary and
    the asperities take the balance.  ``ContactLimit`` is the strictly stronger
    statement -- not even fully developed, plastically capped asperity contact
    can carry it.

    Reusing 'nonconverged' for this would let a downstream script treat two
    different events as one, which is why the outcome value is renamed rather
    than overloaded (brief sec 3.5).

    Typed attributes, one schema across both failure outcomes, exactly as
    ``NonConvergence`` carries ``F_film_max``.
    """

    def __init__(self, message: str, *, F_film_at_limit: float | None = None,
                 F_asp_at_limit: float | None = None,
                 F_target: float | None = None, F_app: float | None = None,
                 F_pocket: float | None = None, eps_cap: float | None = None,
                 contact_model: str | None = None):
        super().__init__(message)
        #: film load at the bracket limit [N]
        self.F_film_at_limit = F_film_at_limit
        #: asperity load at the bracket limit [N]
        self.F_asp_at_limit = F_asp_at_limit
        self.F_target = F_target
        self.F_app = F_app
        self.F_pocket = F_pocket
        self.eps_cap = eps_cap
        #: which band ends and which fork arms produced this row
        self.contact_model = contact_model

    def as_row(self) -> dict:
        return {"F_film_at_limit": self.F_film_at_limit,
                "F_asp_at_limit": self.F_asp_at_limit,
                "F_target_at_failure": self.F_target,
                "F_app_at_failure": self.F_app,
                "F_pocket_at_failure": self.F_pocket,
                "eps_cap_at_failure": self.eps_cap,
                "contact_model_at_failure": self.contact_model,
                "c_at_failure": getattr(self, "c_at_failure", None),
                "n_outer_at_failure": getattr(self, "n_outer_at_failure", None)}


class Outcome(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    NONCONVERGED = "nonconverged"
    #: BOX C only.  The outer thermal-clearance iteration did not settle.
    #: Deliberately NOT folded into NONCONVERGED: the box-B string keeps its
    #: box-B meaning so that no downstream script can silently treat a
    #: clearance failure as a film-load failure (brief sec 2.3).
    CLEARANCE_NONCONVERGED = "clearance_nonconverged"
    #: BOX C only.  Load exceeds film + fully developed asperity contact.
    #: Strictly stronger than NONCONVERGED and deliberately a DIFFERENT string,
    #: so no downstream script can pool the two (brief sec 3.5).
    CONTACT_LIMIT = "contact_limit"
