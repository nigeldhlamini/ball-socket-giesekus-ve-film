"""Write-side enforcement of artefact immutability.

Until this module existed, *"do not modify, move or re-run any box-A or box-B
artefact"* was a **convention enforced by everyone remembering it**.  Box E's
``freeze_and_hash`` was the first write-side guard in the repository, and it was
local to one function.  This lifts it out, because the next thing to overwrite a
hashed artefact will not be ``freeze_and_hash`` -- it will be something that
CONSTRUCTS A FILENAME.

**There is already such a thing in the repository.**
``scripts/run_ch9_campaign.py``'s ``_save`` writes to
``audit_N{n}_seed{AUDIT_SEED}`` -- box B's exact artefact name -- and calls
``df.to_parquet`` on it directly.  Writing a separate box-D driver avoided that
collision; it did not remove it.  Anyone running the old driver at N=8192 would
overwrite a parquet that is cited by sha256 in Chapter 9, in the delivered CME
export, and that erratum E-8 attaches to.

Relying on nobody running it is the same posture that made ``eps_cap`` a defect.
"""

from __future__ import annotations

import os


class ImmutableArtefactError(RuntimeError):
    """Raised on any attempt to write over a protected artefact."""


#: Artefacts that are FROZEN, HASHED and CITED.  A superseded box is retained
#: unchanged and never edited, so every entry here is write-protected for the
#: life of the project.  Matching is on BASENAME, because the hazard is a
#: constructed filename landing in the same directory, not a path typo.
PROTECTED_BASENAMES: dict[str, str] = {
    # --- frozen box configs -------------------------------------------------
    "campaign_frozen_v4.json": "box B frozen config, cited by hash",
    "campaign_frozen_v5.json": "box C frozen config, cited by hash",
    "campaign_frozen_v6.json": "box D frozen config, cited by hash",
    "campaign_frozen_v7.json": "box E frozen config, cited by hash",
    "campaign_frozen_v8.json": "box F frozen config, cited by hash",
    "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet":
        "box F's pure-film campaign artefact (corrected bracket, 32x24)",
    "analysis_E7_averaged_film_subsample_32x24.parquet":
        "box F's measured film bound (R-37.2), cited in FILM_MODEL",
    "analysis_E7_load_weighted_domain.parquet":
        "box F sec 3.3's load-weighted out-of-domain measurement",
    "analysis_E7_mask_curve.parquet":
        "box F sec 5.4's H_mask basis",
    # --- campaign artefacts -------------------------------------------------
    "audit_N8192_seed20260727.parquet":
        ("box B's registered audit: cited by sha256 in Chapter 9 and the CME "
         "export, and the artefact erratum E-8 attaches to"),
    "audit_boxD_N8192_seed20260727_48x36.parquet":
        ("box D's pure-film control arm: the paired baseline against which the "
         "mixed arm isolates load sharing"),
    "audit_boxE_mixed_N8192_seed20260727_48x36.parquet":
        ("box E's mixed arm at eps_cap = 0.99; superseded as a corrected-bracket "
         "result and RETAINED as the 0.99 control arm (R-33.4, erratum E-13)"),
    "boxd_grid_audit_corrected_bracket.parquet":
        ("the grid audit at the corrected bracket -- the untagged predecessor "
         "was destroyed by a re-run because this list did not name it"),
    "boxd_grid_audit_corrected_bracket.summary.json":
        "the corrected-bracket audit's summary, same reason",
}

# WHY THIS LIST KEEPS BEING WRONG, recorded where the next reader will see it.
# It is a HAND-ENUMERATION, which is the failure mode this project has now met
# five times (R-28.3, R-23.6, REQUIRED_CHOICES, ROW_PRODUCING_MODULES,
# _DERIVED_ROW_COLUMNS).  It protected box B's and box D's campaign parquets and
# the frozen configs, and did not protect the grid audit -- which was then
# overwritten by its own re-run.
#
# A derived guard is not available here: "is this artefact cited?" is not a
# property the filesystem carries.  The compensating control is therefore at the
# WRITER: every script that writes a campaign or audit artefact refuses to
# overwrite an existing file at its output path, so protection does not depend
# on this list being complete.  run_boxd_campaign.py has done that since box D;
# run_boxd_grid_audit.py does now.

#: Suffixes that are protected whatever the stem, so a NEW superseded box does
#: not have to be remembered into the table above.
PROTECTED_SUFFIXES: tuple[str, ...] = ()


def is_protected(path: str) -> str | None:
    """Return the reason a path is protected, or None."""
    base = os.path.basename(str(path))
    if base in PROTECTED_BASENAMES:
        return PROTECTED_BASENAMES[base]
    for suf in PROTECTED_SUFFIXES:
        if base.endswith(suf):
            return f"matches protected suffix {suf!r}"
    return None


def assert_writable(path: str, *, what: str = "artefact") -> str:
    """Refuse to write over a protected artefact.  Returns the path if clear.

    EVERY write path that constructs an artefact filename must go through this.
    A guard that only one function calls protects only that function, which is
    what box E's local version did.
    """
    reason = is_protected(path)
    if reason is not None:
        raise ImmutableArtefactError(
            f"refusing to write {what} to {path}: {reason}.  Superseded and "
            f"frozen artefacts are retained unchanged and never edited.  If "
            f"this is a new run, give it a new name; if it is a deliberate "
            f"replacement, move the original aside by hand first.")
    return str(path)


# ---------------------------------------------------------------------------
# THE SHARED WRITER (ruling R-27.3, done properly at R-35.7)
# ---------------------------------------------------------------------------
#
# R-27.3 ruled this remedy and named run_ch9_campaign.py:_save as the live
# target.  What was done instead was to extend PROTECTED_BASENAMES -- a list --
# and the next unnamed writer destroyed the grid audit's parquet.  A list of
# protected names can only protect what somebody remembered to name; a shared
# writer protects what nobody thought of, which is the whole point.
#
# THE INVERSION IS THE DESIGN.  The list answers "is this file precious?", which
# requires foresight.  The writer answers "does this file already exist?", which
# requires none.  Every campaign, audit and summary artefact goes through
# guarded_write / guarded_path, and an existing file stops the write unless the
# caller says overwrite=True in so many words.

import os as _os
import shutil as _shutil
import time as _time
from typing import Callable


class ArtefactExists(RuntimeError):
    """A write would have destroyed an existing artefact."""


def guarded_path(path: str, *, what: str = "artefact",
                 overwrite: bool = False) -> str:
    """Clear ``path`` for writing, or refuse.

    Two independent refusals, in order:

    1. ``PROTECTED_BASENAMES`` -- a named artefact is never overwritten, and
       ``overwrite=True`` does NOT lift this.  Cited-by-hash files require a
       deliberate move, not a flag.
    2. **Existence** -- any other file that already exists stops the write
       unless ``overwrite=True``.  This is the half that needs no foresight and
       the half that was missing.
    """
    # A protected NAME blocks OVERWRITING, not creation: a frozen config has to
    # be written once, by freeze_and_hash, before it can be protected at all.
    # (The first box-F freeze was refused here because the name was listed
    # before the file existed.)
    why = is_protected(path)
    if why and _os.path.exists(path):
        raise ArtefactExists(
            f"{path} is a PROTECTED artefact ({why}).  It is never overwritten, "
            f"and overwrite=True does not lift this.  Move it aside "
            f"deliberately, or write to a different path.")
    if _os.path.exists(path) and not overwrite:
        raise ArtefactExists(
            f"{path} already exists; refusing to overwrite a {what}.  Pass "
            f"overwrite=True only if destroying it is the intent, or write to a "
            f"different path.  (The eps_cap = 0.99 grid audit was lost exactly "
            f"here: a fixed output path, a re-run, and no existence check.)")
    parent = _os.path.dirname(_os.path.abspath(path))
    if parent:
        _os.makedirs(parent, exist_ok=True)
    return path


def guarded_write(path: str, writer: Callable[[str], object], *,
                  what: str = "artefact", overwrite: bool = False,
                  backup: bool = True):
    """Run ``writer(path)`` behind :func:`guarded_path`.

    ``backup`` moves an existing file aside with a timestamp when an overwrite
    IS authorised, so an intentional overwrite is still recoverable.  Belt and
    braces, and cheap.
    """
    guarded_path(path, what=what, overwrite=overwrite)
    if overwrite and _os.path.exists(path):
        stamp = _time.strftime("%Y%m%dT%H%M%S")
        _shutil.move(path, f"{path}.superseded-{stamp}")
    return writer(path)
