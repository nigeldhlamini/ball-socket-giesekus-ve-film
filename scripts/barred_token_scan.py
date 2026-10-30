"""Barred-token scan (ERR-20, R-46.1, R-59.2.2, R-61 B3) over files about to be committed
or released.

Searches for the barred conditions by spelling AND for the quantities ERR-20 found
derived at them (the sweep-scope rule: forward from the barred condition's consumers,
not only backward from its spelling).  The patterns themselves spell the barred
conditions, so they live in scripts/barred_tokens.local.json, which is ignored by git
and pinned by sha256 in QUARANTINE_MANIFEST.json.

R-61 B3: the criterion is ZERO UNCLASSIFIED hits.  A hit is classified when the committed
allowlist (BARRED_SCAN_ALLOWLIST.json) carries an entry with its file, line and pattern id
AND the file's current sha256 equals the entry's: a changed file invalidates its entries.
Pattern ids are sha256(pattern name)[:10], so the allowlist never spells a pattern.

--deep extracts text from non-text files (notebooks, Office XML, parquet schema/metadata/
string columns, npz keys and string arrays, PDF literal strings after zlib) and scans it.
Raster images cannot be text-scanned; they are LISTED for a visual check, not passed.

    python scripts/barred_token_scan.py PATH [PATH ...] [--root DIR] [--deep] [--allowlist FILE]
        exit 1 if any unclassified hit
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import re
import sys
import zipfile

PATTERN_FILE = pathlib.Path(__file__).with_name("barred_tokens.local.json")
ALLOWLIST_DEFAULT = pathlib.Path(__file__).resolve().parents[1] / "BARRED_SCAN_ALLOWLIST.json"
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_FILES = {PATTERN_FILE.name}
TEXT_EXT = {".py", ".ps1", ".md", ".toml", ".json", ".txt", ".cfg", ".ini", ".tex", ".yaml", ".yml", ".csv",
            ".sha256", ".bib", ".sty", ".cls", ".bat", ".sh"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}


def patterns() -> dict:
    d = json.loads(PATTERN_FILE.read_text(encoding="utf-8"))
    return {k: re.compile(v, re.I) for k, v in d.items() if not k.startswith("_")}


def pattern_id(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]


def iter_files(paths):
    for p in map(pathlib.Path, paths):
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (SKIP_DIRS & set(f.parts)) and f.name not in SKIP_FILES:
                    yield f


DEEP_EXT = {".ipynb", ".xlsx", ".docx", ".pptx", ".parquet", ".npz", ".pdf"}


def eff_suffix(f: pathlib.Path) -> str:
    """The file's type suffix, looking past a trailing non-type suffix such as
    '.superseded-R52' or '.bak_pre_E2' (R-62.5.3)."""
    for s in reversed([s.lower() for s in f.suffixes]):
        if s in TEXT_EXT | IMAGE_EXT | DEEP_EXT:
            return s
    return f.suffix.lower()


def _is_text(f: pathlib.Path) -> bool:
    return eff_suffix(f) in TEXT_EXT or f.name.startswith(".") or f.suffix == ""


def deep_text(f: pathlib.Path):
    """Text carried by a non-text file, or None when it has none we can extract."""
    s, b = eff_suffix(f), f.read_bytes()
    if s == ".ipynb":
        return b.decode("utf-8", "replace")
    if s in (".xlsx", ".docx", ".pptx"):
        z = zipfile.ZipFile(io.BytesIO(b))
        return "\n".join(z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.endswith(".xml"))
    if s == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(f)
        md = pf.schema_arrow.metadata or {}
        txt = [str(pf.schema_arrow)] + [v.decode("utf-8", "replace") for v in md.values()]
        t = pf.read()
        for c in t.column_names:
            if str(t.schema.field(c).type) in ("string", "large_string"):
                txt += sorted({str(x) for x in t.column(c).to_pylist()})
        return "\n".join(txt)
    if s == ".npz":
        import numpy as np
        d = np.load(f, allow_pickle=False)
        return "\n".join(list(d.files) + [str(d[k]) for k in d.files if d[k].dtype.kind in "US"])
    if s == ".pdf":
        import zlib
        chunks = [b]
        for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", b, re.S):
            try:
                chunks.append(zlib.decompress(m.group(1)))
            except Exception:
                pass
        return "\n".join(x.decode("latin-1") for c in chunks
                         for x in re.findall(rb"\(((?:[^()\\]|\\.){1,300})\)", c))
    try:                                   # an unknown suffix that is really text (e.g. a .bak copy)
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan(paths, deep=False):
    """[(file, line, pattern, text)].  Unscannable files come back as pattern
    'BINARY/UNKNOWN TYPE' (or 'IMAGE (visual check)' under --deep)."""
    hits = []
    rx = patterns()
    for f in iter_files(paths):
        if _is_text(f):
            txt = f.read_text(encoding="utf-8", errors="replace")
        elif deep and eff_suffix(f) in IMAGE_EXT:
            hits.append((str(f), 0, "IMAGE (visual check)", "raster image: not text-scannable"))
            continue
        elif deep:
            try:
                txt = deep_text(f)
            except Exception as exc:                       # noqa: BLE001
                txt, why = None, type(exc).__name__
            else:
                why = "no extractor"
            if txt is None:
                hits.append((str(f), 0, "BINARY/UNKNOWN TYPE", f"not scanned ({why})"))
                continue
        else:
            hits.append((str(f), 0, "BINARY/UNKNOWN TYPE", "not scanned as text"))
            continue
        for n, line in enumerate(txt.splitlines(), 1):
            for k, r in rx.items():
                if r.search(line):
                    hits.append((str(f), n, k, line.strip()[:200]))
    return hits


def classify(hits, root: pathlib.Path, allowlist: pathlib.Path):
    """Split hits into (unclassified, classified, images, stale_entries)."""
    entries = json.loads(allowlist.read_text(encoding="utf-8"))["entries"] if allowlist.exists() else []
    sha_cache = {}

    def fsha(rel):
        if rel not in sha_cache:
            p = root / rel
            sha_cache[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        return sha_cache[rel]
    index = {(e["file"], e["line"], e["pattern_id"]): e for e in entries}
    unc, cls, img, used = [], [], [], set()
    for f, n, k, s in hits:
        rel = pathlib.Path(f).resolve().relative_to(root.resolve()).as_posix()
        if k == "IMAGE (visual check)":
            img.append(rel)
            continue
        key = (rel, n, pattern_id(k))
        e = index.get(key)
        if e is not None and e["file_sha256"] == fsha(rel):
            cls.append((rel, n, k, e["reason"]))
            used.add(key)
        else:
            unc.append((rel, n, k, s))
    stale = [e for k, e in index.items() if k not in used]
    return unc, cls, img, stale


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--root", default=None, help="classify against the allowlist relative to this root")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--allowlist", default=str(ALLOWLIST_DEFAULT))
    a = ap.parse_args()
    h = scan(a.paths, deep=a.deep)
    if a.root is None:
        for f, n, k, s in h:
            print(f"{f}:{n}: [{k}] {s}")
        print(f"{len(h)} hit(s)")
        raise SystemExit(1 if h else 0)
    unc, cls, img, stale = classify(h, pathlib.Path(a.root), pathlib.Path(a.allowlist))
    for rel, n, k, s in unc:
        print(f"UNCLASSIFIED {rel}:{n}: [{k}] {s}")
    print(f"{len(unc)} unclassified, {len(cls)} classified, {len(img)} images listed for visual check, "
          f"{len(stale)} allowlist entries unused")
    raise SystemExit(1 if unc else 0)
