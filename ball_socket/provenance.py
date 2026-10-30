"""Run provenance and the dirty-tree gate (R-59.2.4/5).

Every runner written from R-59 on calls ``require_clean_tree(what)`` before its first
solve and stores the returned dict in its artefact's provenance.  The commit hash goes
into RUN provenance only, never into a hashed spec, so a frozen spec and any signature
on it stand.

"Clean" means, over the CODE paths below: no modified tracked file, no untracked file
(ignored files do not count), and every quarantined file -- kept out of git because the
barred-token scan hit it (R-59.2.2) -- still has the sha256 recorded in the committed
QUARANTINE_MANIFEST.json.  Documents (reports/, preregistrations, exports/) are not
code and are not checked.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_PATHS = ("ball_socket", "scripts", "tests", "src", "conftest.py", "pyproject.toml",
              ".gitignore", ".gitattributes", "QUARANTINE_MANIFEST.json")
MANIFEST = ROOT / "QUARANTINE_MANIFEST.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def quarantine_state() -> dict:
    """{path: 'ok' | 'changed' | 'missing'} for every quarantined file."""
    if not MANIFEST.exists():
        return {}
    out = {}
    for rel, rec in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"].items():
        p = ROOT / rel
        out[rel] = "missing" if not p.exists() else ("ok" if _sha(p) == rec["sha256"] else "changed")
    return out


def tree_state() -> dict:
    commit = _git("rev-parse", "HEAD").strip()
    porcelain = [ln for ln in _git("status", "--porcelain", "--untracked-files=all", "--",
                                   *CODE_PATHS).splitlines() if ln.strip()]
    q = quarantine_state()
    q_bad = {k: v for k, v in q.items() if v != "ok"}
    return {"commit": commit, "dirty": bool(porcelain or q_bad),
            "dirty_paths": [ln[3:] for ln in porcelain], "quarantine_not_ok": q_bad,
            "quarantine_files": len(q)}


def require_clean_tree(what: str) -> dict:
    """The gate.  Returns the provenance dict, or raises SystemExit naming what is dirty."""
    st = tree_state()
    if st["dirty"]:
        shown = st["dirty_paths"][:10] + [f"{k} ({v})" for k, v in st["quarantine_not_ok"].items()]
        raise SystemExit(f"{what}: refused -- the code tree is dirty at {st['commit'][:12]} "
                         f"(R-59.2.4).  Commit first.  Dirty: {shown}")
    return st
