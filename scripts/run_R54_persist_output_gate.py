"""R-54.3: the persist convention checked on the ARTEFACT, not on the source.

R-53.5's audit read the source and, after the E10 fix, credited any runner that
REFERENCES `REGISTERED_CONSTRAINT_COLUMNS`.  A runner can reference the constant
and still write a subset, so that audit certifies the constant was registered in
the runner, not that it was applied to the product -- instrument mode 5,
registered-but-not-applied, which is the mode the eps_cap defect belonged to.

> **The gate is output-level.**  For every runner that calls the campaign
> evaluator and writes a parquet, read THAT PARQUET'S SCHEMA and require it to
> carry every registered-constraint column.  The source audit stays as a fast
> pre-filter; this is the gate.

Artefacts written BEFORE the convention are expected to fail, and that is the
point: `E10_capped_area_rated_corner.parquet` is exactly the file whose missing
`T_max_bnd` costs 960 solves to recover (ERR-25).  The test pins the historical
set, so a NEW non-compliant artefact fails.
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "ch9"
DEST = OUT / "persist_output_gate.json"
SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules", "site-packages"}
EVALUATORS = {"evaluate_x", "evaluate_x_traversal"}


#: A parquet this module READS is not its product.  The first version attributed
#: every parquet constant in a module to that module, so box E's audit -- an INPUT
#: to four runners -- was checked four times and two aggregate tables were
#: required to carry per-solve columns.  Flagging what the gate cannot be
#: responsible for is mode 4 (cries-wolf).
WRITERS = {"to_parquet", "guarded_write", "guarded_path"}
#: Markers that an artefact is a PER-SOLVE row product rather than an aggregate:
#: a count table has no business carrying a film minimum.
SOLVE_ROW_MARKERS = {"eps_ecc", "realised_eps_cap", "state", "h_min", "h_min_bnd"}


def _is_artefact_name(v: str) -> bool:
    """A parquet FILE NAME -- not a glob (`chunk_*.parquet`) and not the bare
    suffix a name is built from."""
    return v.endswith(".parquet") and "*" not in v and v != ".parquet"


def _written_parquets(tree) -> list[str]:
    """Parquet literals this module WRITES, resolved through its own constants."""
    const = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name):
            lits = [x.value for x in ast.walk(n.value)
                    if isinstance(x, ast.Constant) and isinstance(x.value, str)
                    and x.value.endswith(".parquet")]
            if len(lits) == 1:
                const[n.targets[0].id] = lits[0]
    written = set()
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call)):
            continue
        fname = (n.func.attr if isinstance(n.func, ast.Attribute)
                 else getattr(n.func, "id", None))
        if fname not in WRITERS or not n.args:
            continue
        for x in ast.walk(n.args[0]):
            if isinstance(x, ast.Name) and x.id in const:
                written.add(const[x.id])
            elif isinstance(x, ast.Constant) and isinstance(x.value, str) \
                    and _is_artefact_name(x.value):
                written.add(x.value)
    return sorted(written)


def gate(paths=None) -> dict:
    from ball_socket.campaign import REGISTERED_CONSTRAINT_COLUMNS as REQ
    import pyarrow.parquet as pq
    rows = []
    files = sorted((ROOT / "scripts").rglob("*.py")) if paths is None else list(paths)
    for p in files:
        if set(p.parts) & SKIP_DIRS:
            continue
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError):
            continue
        calls = {n.func.attr if isinstance(n.func, ast.Attribute)
                 else getattr(n.func, "id", None)
                 for n in ast.walk(tree) if isinstance(n, ast.Call)}
        if not (calls & EVALUATORS) or "to_parquet" not in src:
            continue
        # a fixture runner passed in by a test lives outside the repo tree
        try:
            runner_name = p.relative_to(ROOT).as_posix()
        except ValueError:
            runner_name = p.name
        for name in _written_parquets(tree):
            f = OUT / name
            if not f.exists():
                rows.append({"runner": runner_name,
                             "artefact": name, "status": "NOT_WRITTEN"})
                continue
            cols = set(pq.ParquetFile(f).schema.names)
            if not (cols & SOLVE_ROW_MARKERS):
                rows.append({"runner": runner_name,
                             "artefact": name, "status": "NOT_A_SOLVE_ROW_ARTEFACT"})
                continue
            missing = sorted(set(REQ) - cols)
            rows.append({"runner": runner_name, "artefact": name,
                         "missing": missing,
                         "status": "OK" if not missing else "MISSING_COLUMNS"})
    return {"required_columns": list(REQ),
            "source_of_required_set": "ball_socket.campaign.REGISTERED_CONSTRAINT_COLUMNS",
            "checked": rows,
            "non_compliant": [r for r in rows if r["status"] == "MISSING_COLUMNS"]}


def main() -> int:
    rep = gate()
    DEST.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"required: {rep['required_columns']}")
    for r in rep["checked"]:
        mark = "  " if r["status"] == "OK" else "!!"
        extra = ("" if r["status"] == "OK"
                 else f"  ({r['status']}"
                      + (f": {', '.join(r['missing'])}" if r.get("missing") else "")
                      + ")")
        print(f" {mark} {r['artefact'][:62]:62s} {r['runner']}{extra}")
    print(f"\n{len(rep['checked'])} artefact(s) checked, "
          f"{len(rep['non_compliant'])} non-compliant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
