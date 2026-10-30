"""R-53.5: which runners compute a registered-constraint quantity and discard it.

E10's job called `campaign.evaluate_x`, which returns every registered constraint
quantity, and persisted only the columns E10 itself needed.  T_max_bnd was
computed at all 3,840 rated-corner rows and thrown away, so E5's VG32 clause now
costs 960 solves to recover something that had already been solved.

> CONVENTION (R-53.5): persist every registered-constraint quantity on every
> solve, whether or not the current expectation needs it.  The constraints are
> registered in advance, so the set is known before the run; the cost is disk.

This audit finds each function that calls the campaign evaluator and reports which
of the registered-constraint columns it keeps.  It reads the source, not the
artefacts, so it covers stages that have not been run recently as well.
`ball_socket.campaign.REGISTERED_CONSTRAINT_COLUMNS` is the single source for the
required set -- hand-listing it here would be the defect that gate 1 exists for.
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEST = ROOT / "results" / "ch9" / "persist_audit.json"
SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules", "site-packages"}
EVALUATORS = {"evaluate_x", "evaluate_x_traversal"}


def _strings(node) -> set:
    return {n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def main() -> int:
    from ball_socket.campaign import REGISTERED_CONSTRAINT_COLUMNS as REQ
    rows = []
    # SCOPE: runner scripts that PERSIST an artefact.  A test or a core helper
    # that calls the evaluator and keeps three columns persists nothing, so the
    # convention has nothing to bite on; including them buried the one real hit
    # under 20 lines of noise on the first run.
    for p in sorted((ROOT / "scripts").rglob("*.py")):
        if set(p.parts) & SKIP_DIRS:
            continue
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError):
            continue
        if "to_parquet" not in src:
            continue
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            calls = {n.func.attr if isinstance(n.func, ast.Attribute)
                     else getattr(n.func, "id", None)
                     for n in ast.walk(fn) if isinstance(n, ast.Call)}
            if not (calls & EVALUATORS):
                continue
            kept = _strings(fn)
            # A function that keeps the WHOLE row keeps everything.  Two shapes:
            # `{**row, ...}`, and `row = evaluate_x(...); ...; return row`, which
            # `run_boxd_grid_audit._evaluate` uses and which the first version of
            # this audit reported as discarding all five columns.
            evaluated = {t.id for n in ast.walk(fn)
                         if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                         and ((n.value.func.attr if isinstance(n.value.func, ast.Attribute)
                               else getattr(n.value.func, "id", None)) in EVALUATORS)
                         for t in n.targets if isinstance(t, ast.Name)}
            returns_row = any(
                isinstance(n, ast.Return) and isinstance(n.value, ast.Name)
                and n.value.id in evaluated for n in ast.walk(fn))
            splats_row = any(
                isinstance(n, ast.Dict) and any(
                    k is None and isinstance(v, ast.Name) and v.id in evaluated
                    for k, v in zip(n.keys, n.values))
                for n in ast.walk(fn))
            whole = returns_row or splats_row
            # Referencing the shared constant IS the convention: a runner that
            # spreads REGISTERED_CONSTRAINT_COLUMNS keeps whatever that set holds
            # today, which is stronger than naming five columns as literals.
            uses_constant = any(
                (isinstance(n, ast.Name) and n.id == "REGISTERED_CONSTRAINT_COLUMNS")
                or (isinstance(n, ast.Attribute) and n.attr == "REGISTERED_CONSTRAINT_COLUMNS")
                for n in ast.walk(fn))
            missing = [] if (whole or uses_constant) else sorted(set(REQ) - kept)
            rows.append({"file": p.relative_to(ROOT).as_posix(), "function": fn.name,
                         "line": fn.lineno, "keeps_whole_row": whole,
                         "uses_registered_set": uses_constant, "missing": missing,
                         "status": "OK" if not missing else "DISCARDS"})
    rep = {"convention": ("R-53.5: persist every registered-constraint quantity on "
                          "every solve; the registered set is known in advance"),
           "required_columns": list(REQ),
           "source_of_required_set": "ball_socket.campaign.REGISTERED_CONSTRAINT_COLUMNS",
           "n_functions_calling_the_evaluator": len(rows),
           "discarding": [r for r in rows if r["status"] == "DISCARDS"],
           "all": rows}
    DEST.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"required: {list(REQ)}")
    print(f"functions calling the evaluator: {len(rows)}")
    for r in rows:
        mark = "  " if r["status"] == "OK" else "!!"
        if r.get("uses_registered_set"):
            what = "spreads REGISTERED_CONSTRAINT_COLUMNS"
        elif r["keeps_whole_row"]:
            what = "keeps the whole row"
        elif r["missing"]:
            what = "DISCARDS: " + ", ".join(r["missing"])
        else:
            what = "keeps every registered column"
        print(f" {mark} {r['file']}:{r['line']} {r['function']:24s} {what}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
