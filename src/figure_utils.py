"""Small helpers for figure-writing drivers."""

from __future__ import annotations

import logging
import pathlib

logger = logging.getLogger(__name__)


def safe_savefig(fig, path: pathlib.Path | str, **kwargs) -> pathlib.Path:
    """Save a matplotlib figure with a Windows file-lock fallback.

    Windows holds an exclusive write lock on PDFs that are open in
    viewers (Adobe Acrobat, Sumatra, etc.); a re-run of any figure
    driver while the previous figure is being viewed will fail with
    PermissionError. On that error this helper transparently writes
    to a `<stem>.new<suffix>` sibling and emits a warning, so the
    pipeline finishes and the user can close the viewer + rename
    the new file at leisure.

    Returns the actual path written.
    """
    path = pathlib.Path(path)
    try:
        fig.savefig(path, **kwargs)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}.new{path.suffix}")
        logger.warning(
            f"PermissionError writing {path} (file likely open in a viewer). "
            f"Wrote to {alt} instead. Close the viewer and rename if needed."
        )
        fig.savefig(alt, **kwargs)
        return alt
