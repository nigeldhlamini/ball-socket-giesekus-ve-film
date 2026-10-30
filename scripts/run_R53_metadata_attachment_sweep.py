"""R-53.1: sweep every site that attaches metadata to a result set.

WHY.  ERR-26 is not a new defect.  Its root cause is R-21.6's -- results arriving
in completion order rather than submission order -- which was repaired once, in
`campaign.run_batch`, the stage where it surfaced.  `run_ch9_campaign.stage_e5`
pairs an ORDERED label list with that result set and was never swept.

THE RULE (R-53.1, extending R-46.1's forward sweep to fixed defects): when a root
cause is repaired, sweep every site that shares the MECHANISM -- anything that
pairs an ordered label array with a result array whose order it does not control
-- not only the site that exposed it.

THIS SCRIPT DERIVES ITS OWN SITE LIST (gate-6 style: derive the scope, never
hand-list it).  It walks every .py file in the repository, finds each function
that runs work in parallel, and classifies how that function attaches labels:

  KEYED      -- the label travels INSIDE the returned record (job returns its own
                key/ids), or the frame is sorted by such a key afterwards.  Safe
                under any completion order.
  ORDER_KEPT -- executor.map / imap (ordered), positional attach.  Correct today,
                but correct only because map preserves order: recorded, not fixed.
  UNGUARDED  -- an unordered API and a positional attach, with no alignment
                assertion covering it.  ERR-26's shape; this count must be zero.
  GUARDED    -- the same shape, with `assert_label_alignment` called on THAT
                frame before the attach, on the same or an enclosing block.
  NO_ATTACH  -- parallel work, no metadata attached to the results.

A site's classification is evidence for a human read, not a verdict: the report
lists SITES CHECKED, not sites fixed (R-53.1 -- "a sweep that reports only what it
changed cannot be distinguished from one that looked in two places").
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "results" / "ch9" / "metadata_attachment_sweep.json"
SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules", "site-packages"}

#: RECALL GAPS FOUND BY R-54.2's INDEPENDENT FIXTURES (2026-09-23), both fixed
#: here and recorded as must-be-armed instances:
#:   * `add_done_callback` / bare `submit` -- concurrency with no as_completed;
#:   * an executor in ANOTHER MODULE (`df = helpers.run_all(jobs)`), where this
#:     function sees no executor API at all but still does not control the order.
#: The sweep had been revised twice until it found `stage_e5`, and `stage_e5` was
#: then its only control: design rule 4's case exactly.
UNORDERED = {"as_completed", "imap_unordered", "add_done_callback", "submit"}
ORDERED = {"map", "imap"}
#: names whose call means "run these in parallel"
POOL_CALLS = {"ProcessPoolExecutor", "ThreadPoolExecutor", "Pool"}
#: PROJECT batch runners.  Without these the sweep cannot see its own positive
#: control: `stage_e5` dispatches through `campaign.run_batch` and touches no
#: executor API itself, which is exactly why it was never swept.  A sweep that
#: cannot flag the defect it was written for is not a sweep (R-49.5).
BATCH_RUNNERS = {"run_batch", "evaluate_batch"}
#: names bound to these calls are DATAFRAMES; a subscript assignment on anything
#: else (a dict keyed by an id, a report dict) is not the ERR-26 shape.
FRAME_SOURCES = {"DataFrame", "read_parquet", "read_csv", "concat", "copy",
                 "run_batch", "evaluate_batch"}
#: Constructors whose ROW ORDER this function itself determines.  A frame bound
#: from anything else came from elsewhere, so this function does not control its
#: order -- attaching a positional column to one is the ERR-26 shape even when no
#: executor is visible here.
ORDER_OWNED_SOURCES = {"DataFrame", "concat", "copy"}


def _calls(node) -> list[str]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.append(f.id)
            elif isinstance(f, ast.Attribute):
                out.append(f.attr)
    return out


def _external_frames(node) -> set:
    """Frame names bound from a call this function does not own the order of."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            f = n.value.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name and name not in ORDER_OWNED_SOURCES:
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        out.add(t.id)
    return out


def _frame_names(node) -> set:
    """Names bound to a DataFrame inside this function.

    Without this the sweep reports every `results[key] = out` -- a dict keyed by
    an identifier, which is the SAFE pattern -- as a defect.  Four such false
    positives came back on the first run.
    """
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            f = n.value.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in FRAME_SOURCES:
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        out.add(t.id)
        # `for i, row in df.iterrows()` etc. do not bind frames; ignore
    # _EXTERNAL_HINT holds FRAME names (not callee names): a frame handed back by
    # a runner in another module is still a frame to attach to.
    return out | _EXTERNAL_HINT


#: filled per-call so a frame from an unknown callee is still treated as a frame
_EXTERNAL_HINT: set = set()


GUARD = "assert_label_alignment"


def _block_paths(fn):
    """Map each statement to its block path, so "dominates" can be checked.

    A guard covers an attach when it runs before it on every path: same block and
    earlier, or in an enclosing block.  Comparing line numbers alone would call a
    guard inside `if debug:` a guard over an attach outside it.
    """
    paths = {}

    def walk(body, path):
        for i, st in enumerate(body):
            paths[st] = path + (i,)
            for f in ("body", "orelse", "finalbody"):
                if hasattr(st, f):
                    walk(getattr(st, f), path + (i,))
            for h in getattr(st, "handlers", []) or []:
                walk(h.body, path + (i,))
    walk(fn.body, ())
    return paths


def _stmt_of(fn, node, paths):
    """The INNERMOST statement containing `node`.

    Taking the outermost instead resolved a guard inside `if check:` to the `if`
    itself, so a guard that does not run on every path was read as dominating the
    attach -- the fixture for that is in the suite.
    """
    best = None
    for st in paths:
        for n in ast.walk(st):
            if n is node:
                if best is None or len(paths[st]) > len(paths[best]):
                    best = st
                break
    return best


def _guard_covers(fn, attach_nodes) -> bool:
    """True when every attach is preceded, on the same or an enclosing block, by
    a GUARD call whose FIRST argument is the frame being attached to.

    Checking that the name merely appears would pass a guard on a different
    frame -- R-54.2(b) requires the same frame object and control-flow dominance,
    and the fixture that guards the wrong frame is in the suite.
    """
    if not attach_nodes:
        return False
    paths = _block_paths(fn)
    guards = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            fname = (n.func.attr if isinstance(n.func, ast.Attribute)
                     else getattr(n.func, "id", None))
            if fname == GUARD and n.args and isinstance(n.args[0], ast.Name):
                st = _stmt_of(fn, n, paths)
                if st is not None:
                    guards.append((n.args[0].id, paths[st]))
    for frame, node in attach_nodes:
        st = _stmt_of(fn, node, paths)
        if st is None:
            return False
        here = paths[st]
        if not any(g_frame == frame and g_path < here
                   and g_path[:len(g_path) - 1] == here[:len(g_path) - 1]
                   for g_frame, g_path in guards):
            return False
    return True


def _attach_nodes(node):
    """(frame name, Assign node) for each positional column attach."""
    frames = _frame_names(node)
    hits = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Assign):
            continue
        for tgt in n.targets:
            if not (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)):
                continue
            frame = tgt.value.id
            if frame not in frames:
                continue                       # a dict keyed by an id, not a column
            # the column name may be a literal OR a loop variable: stage_e5's
            # `for k in (...): df[k] = [m[k] for m in meta]` is the defect itself,
            # and requiring a literal here excluded it on the second run.
            key = tgt.slice
            if not (isinstance(key, (ast.Constant, ast.Name))
                    or isinstance(key, ast.Tuple)):
                continue
            val = n.value
            if isinstance(val, ast.ListComp):
                src = val.generators[0].iter
                if isinstance(src, ast.Name) and src.id != frame:
                    hits.append((frame, n))
                elif isinstance(src, ast.Attribute) and isinstance(src.value, ast.Name) \
                        and src.value.id != frame:
                    hits.append((frame, n))
            elif isinstance(val, (ast.Name, ast.List)):
                hits.append((frame, n))
    return hits


def _is_positional_attach(node) -> list[str]:
    """The same attaches, rendered for the report."""
    return [ast.unparse(n)[:110] for _, n in _attach_nodes(node)]


def _keyed_evidence(node) -> list[str]:
    """Evidence that identity travels with the record: a sort on a returned key,
    a merge, or a dict built from the result itself."""
    ev = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in {"sort_values", "merge", "set_index", "reindex", "join"}:
                ev.append(ast.unparse(n)[:110])
    return ev


#: Reading an artefact is the CONSUMER-side form of the same mechanism: the
#: function pairs its own ordered label list with rows whose order it does not
#: control.  Included so the sweep covers both sides (R-53.1).
READERS = {"read_parquet", "read_csv"}


def classify(fn) -> dict | None:
    global _EXTERNAL_HINT
    _EXTERNAL_HINT = _external_frames(fn)
    names = _calls(fn)
    if not (set(names) & (UNORDERED | ORDERED | POOL_CALLS | BATCH_RUNNERS | READERS)):
        # no concurrency and no reader in sight: still the pattern if a positional
        # column is attached to a frame this function did not build (R-54.2)
        # STRICTER when no concurrency is visible.  Treating every call result
        # as a frame flagged 24 dict/array assignments (`parent_t[rt] = rn`) --
        # mode 4, cries-wolf, which trains readers to ignore the flag.  Here the
        # attach must have a COLUMN NAME (a string literal) and be filled from a
        # comprehension over a foreign iterable: ERR-26's shape, narrowly.
        nodes = [(f, n) for f, n in _attach_nodes(fn)
                 if isinstance(n.targets[0].slice, ast.Constant)
                 and isinstance(n.targets[0].slice.value, str)
                 and isinstance(n.value, ast.ListComp)]
        ext = _external_frames(fn) & {f for f, _ in nodes}
        if not (nodes and ext):
            return None
        return {"function": fn.name, "line": fn.lineno,
                "dispatch": [f"frame from another module ({', '.join(sorted(ext))})"],
                "positional_attaches": _is_positional_attach(fn),
                "keyed_evidence": _keyed_evidence(fn),
                "class": "GUARDED" if _guard_covers(fn, nodes) else "UNGUARDED"}
    if not (set(names) & (UNORDERED | ORDERED | POOL_CALLS | BATCH_RUNNERS)):
        att = _is_positional_attach(fn)
        if not att:
            return None
        return {"function": fn.name, "line": fn.lineno, "dispatch": ["read_parquet"],
                "positional_attaches": att, "keyed_evidence": _keyed_evidence(fn),
                "class": "READ_ATTACH"}
    unordered = sorted(set(names) & UNORDERED)
    ordered = sorted(set(names) & ORDERED)
    delegated = sorted(set(names) & BATCH_RUNNERS)
    if delegated and not (unordered or ordered):
        # the order is whatever the runner guarantees, which this function does
        # not control: treat as unordered for classification
        unordered = [f"via {d}()" for d in delegated]
    if not (unordered or ordered):
        return None                            # a pool opened but nothing dispatched
    nodes = _attach_nodes(fn)
    attaches = _is_positional_attach(fn)
    keyed = _keyed_evidence(fn)
    # an attach onto a frame from another module's runner: no executor API is
    # visible here, but the order is not this function's to rely on
    if attaches and not (unordered or ordered):
        ext = _external_frames(fn) & {f for f, _ in nodes}
        if ext:
            unordered = [f"frame from another module ({', '.join(sorted(ext))})"]
    if not attaches:
        kind = "NO_ATTACH"
    elif unordered:
        # R-54.2(b): a permanently flagged guarded site trains its readers to
        # ignore the flag.  GUARDED is a site that asserts the alignment it
        # relies on; UNGUARDED is the one that must be zero.
        kind = ("KEYED" if keyed else
                "GUARDED" if _guard_covers(fn, nodes) else "UNGUARDED")
    else:
        kind = "ORDER_KEPT"
    return {"function": fn.name, "line": fn.lineno,
            "dispatch": unordered + ordered + delegated,
            "positional_attaches": attaches, "keyed_evidence": keyed, "class": kind}


def main() -> int:
    sites = []
    for p in sorted(ROOT.rglob("*.py")):
        if set(p.parts) & SKIP_DIRS:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:
            sites.append({"file": str(p.relative_to(ROOT)), "class": "UNPARSED",
                          "error": f"{type(exc).__name__}"})
            continue
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            got = classify(fn)
            if got:
                sites.append({"file": str(p.relative_to(ROOT)), **got})
    by = {}
    for s in sites:
        by.setdefault(s["class"], []).append(f"{s['file']}:{s.get('line', 0)} {s.get('function', '')}")
    rep = {"rule": "R-53.1: sweep the mechanism, report sites CHECKED not sites fixed",
           "n_sites_checked": len(sites),
           "counts": {k: len(v) for k, v in sorted(by.items())},
           "by_class": {k: sorted(v) for k, v in sorted(by.items())},
           "sites": sites}
    DEST.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"sites checked: {len(sites)}")
    for k, v in sorted(by.items()):
        print(f"  {k:11s} {len(v)}")
    for s in sites:
        if s["class"] in {"UNGUARDED", "UNPARSED"}:
            print(f"  !! {s['class']}: {s['file']}:{s.get('line')} {s.get('function')}")
            for a in s.get("positional_attaches", []):
                print(f"       {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
