"""Cut a release tree for Paper 3 from a COMMITTED revision (R-60.2).  Local only.

    python scripts/export_release.py --rev <tag-or-commit> --out <empty local dir>

What it does, and nothing more:
  1. `git archive` of <rev> with core.autocrlf=false (bytes exactly as committed; the
     .gitattributes -text paths are byte-faithful), restricted to INCLUDE plus the src/
     modules the included Python imports, closed transitively (the shared src/ is vendored
     at the committed bytes; ball_socket/memory_sign.assert_src_identity stays in place and
     SRC_PINS.json records each vendored module's sha256).
  2. Writes RELEASE_MANIFEST.json (every included file, sha256, bytes), WITHHELD.json (the
     quarantined files from QUARANTINE_MANIFEST.json at <rev>: path, sha256, reason --
     the originals' hashes are published beside their absence, R-60.5), and README_RELEASE.md
     naming the commit it was cut from.
  3. Runs the barred-token scan over the tree and the release tests inside it, and prints a
     summary.  The verdict is the caller's.

It NEVER adds a remote, pushes, or touches any repository other than reading <rev>.  The
public repository is not a remote of the working repository (R-60.2).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Paper 3's code and the tests that exercise it.  Directories end in "/".
INCLUDE = [
    "ball_socket/",
    "scripts/p3_rim_asymptote.py",
    "scripts/boxc_decomposition.py",
    "scripts/barred_token_scan.py",
    "scripts/ch9_numbers.py",            # loaded by path in ball_socket/tests/test_boxd_gate_arming.py
    "tests/test_memory_sign.py",
    "tests/test_p3_rim_asymptote.py",
    "tests/test_notation_beta_r.py",
    "pyproject.toml",
    ".gitattributes",
    "LICENSE",
    "QUARANTINE_MANIFEST.json",
]
RELEASE_TESTS = ["ball_socket/tests", "tests/test_memory_sign.py", "tests/test_p3_rim_asymptote.py",
                 "tests/test_notation_beta_r.py"]


def _git(*a, binary=False):
    r = subprocess.run(["git", "-c", "core.autocrlf=false", *a], cwd=ROOT, capture_output=True,
                       check=True)
    return r.stdout if binary else r.stdout.decode()


def _src_imports(py_bytes: bytes, path: str = "") -> set[str]:
    """src.* modules imported by a file, absolute and (inside src/) relative."""
    try:
        tree = ast.parse(py_bytes)
    except SyntaxError:
        return set()
    pkg = path[:-3].split("/")[:-1]          # the package the file lives in
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level > 0 and pkg[:1] == ["src"]:
            base = pkg[:len(pkg) - (n.level - 1)]
            mod = ".".join(base + ([n.module] if n.module else []))
            mods.add(mod)
            mods.update(f"{mod}.{a.name}" for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and n.module.split(".")[0] == "src":
            mods.add(n.module)
            mods.update(f"{n.module}.{a.name}" for a in n.names)
        elif isinstance(n, ast.Import):
            mods.update(a.name for a in n.names if a.name.split(".")[0] == "src")
    return mods


def cut(rev: str, out: pathlib.Path) -> dict:
    commit = _git("rev-parse", rev).strip()
    tracked = _git("ls-tree", "-r", "--name-only", commit).split("\n")
    want = {f for f in tracked if f and any(f == i or (i.endswith("/") and f.startswith(i)) for i in INCLUDE)}
    # transitive closure over src/ imports
    blob = lambda f: _git("show", f"{commit}:{f}", binary=True)
    queue, src_files = [f for f in want if f.endswith(".py")], set()
    while queue:
        cur = queue.pop()
        for m in _src_imports(blob(cur), cur):
            parts = m.split(".")
            for k in range(len(parts), 0, -1):
                for cand in ("/".join(parts[:k]) + ".py", "/".join(parts[:k]) + "/__init__.py"):
                    if cand in tracked and cand not in src_files:
                        src_files.add(cand)
                        queue.append(cand)
    src_files |= {f for f in ("src/__init__.py",) if f in tracked}
    for f in list(src_files):          # package __init__ files along each path
        p = pathlib.PurePosixPath(f).parent
        while str(p) not in (".", ""):
            init = f"{p}/__init__.py"
            if init in tracked and init not in src_files:
                src_files.add(init)
            p = p.parent
    files = sorted(want | src_files)
    out.mkdir(parents=True, exist_ok=False)
    z = zipfile.ZipFile(io.BytesIO(_git("archive", "--format=zip", commit, *files, binary=True)))
    z.extractall(out)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    man = {f: {"sha256": sha(out / f), "bytes": (out / f).stat().st_size} for f in files}
    (out / "RELEASE_MANIFEST.json").write_text(json.dumps({"commit": commit, "files": man}, indent=1) + "\n",
                                               encoding="utf-8", newline="\n")
    (out / "SRC_PINS.json").write_text(json.dumps({f: man[f]["sha256"] for f in sorted(src_files)}, indent=1)
                                       + "\n", encoding="utf-8", newline="\n")
    q = json.loads(blob("QUARANTINE_MANIFEST.json"))["files"]
    (out / "WITHHELD.json").write_text(json.dumps(
        {"statement": "Withheld under the industry collaboration agreement; available to reviewers and examiners on "
                      "request.  Each original's sha256 is given so a supplied copy can be verified.",
         "files": {k: {"sha256": v["sha256"], "reason": v["reason"]} for k, v in q.items()}}, indent=1) + "\n",
        encoding="utf-8", newline="\n")
    (out / "README_RELEASE.md").write_text(
        f"# Paper 3 code release (dry run)\n\nCut from commit `{commit}` by `scripts/export_release.py`.\n"
        f"{len(files)} files ({len(src_files)} vendored from the shared `src/`, pinned in SRC_PINS.json).\n"
        f"Withheld files and their sha256: WITHHELD.json.\n", encoding="utf-8", newline="\n")
    return {"commit": commit, "files": len(files), "src_vendored": sorted(src_files),
            "withheld": len(q)}


GENERATED = {"RELEASE_MANIFEST.json", "SRC_PINS.json", "WITHHELD.json", "README_RELEASE.md"}


def check(out: pathlib.Path) -> dict:
    """R-61 B3: zero UNCLASSIFIED hits.  Committed files are classified against the repository's
    allowlist (their bytes equal the committed bytes, so the entries' hashes apply); the files
    this script generates may name withheld paths and nothing else.  Raster images are listed
    in RELEASE_MANIFEST.json for a visual check before the first public push."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import barred_token_scan as B
    hits = B.scan([str(p) for p in sorted(out.rglob("*")) if p.is_file()], deep=True)
    withheld = set(json.loads((out / "WITHHELD.json").read_text(encoding="utf-8"))["files"])
    gen = [h for h in hits if pathlib.Path(h[0]).name in GENERATED and pathlib.Path(h[0]).parent == out]
    rest = [h for h in hits if h not in gen]
    gen_bad = [h for h in gen if not any(w in h[3] for w in withheld)]
    unc, cls, img, _ = B.classify(rest, out, ROOT / "BARRED_SCAN_ALLOWLIST.json")
    man_p = out / "RELEASE_MANIFEST.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    man["images_for_visual_check"] = sorted(img)
    man_p.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
    r = subprocess.run([sys.executable, "-m", "pytest", *RELEASE_TESTS, "-q", "-p", "no:cacheprovider",
                        "--continue-on-collection-errors", "-o", "addopts="],
                       cwd=out, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = [ln for ln in r.stdout.splitlines() if ln.strip()][-40:]
    return {"scan_unclassified": [f"{rel}:{n}: [{k}] {s[:120]}" for rel, n, k, s in unc]
                                 + [f"{pathlib.Path(f).name}:{n}: [{k}] (generated file, not a withheld path)"
                                    for f, n, k, s in gen_bad],
            "scan_classified": len(cls), "generated_file_hits_naming_withheld_paths": len(gen) - len(gen_bad),
            "images_for_visual_check": sorted(img),
            "pytest_exit": r.returncode, "pytest_tail": tail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out).resolve()
    if ROOT.resolve() in out.parents or out == ROOT.resolve():
        raise SystemExit("--out must be outside the working repository")
    info = cut(a.rev, out)
    res = {**info, **check(out)}
    (out.parent / f"{out.name}.export_report.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
