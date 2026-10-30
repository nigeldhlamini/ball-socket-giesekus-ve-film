"""R-53.1: the metadata-attachment sweep, pinned.

ERR-26's root cause is R-21.6's, repaired once in the stage where it surfaced.
`stage_e5` shared the mechanism and was never swept.  These tests pin the sweep's
result over the repository and arm it against its own defect: a sweep that cannot
flag the case it was written for is not a sweep (R-49.5).
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "r53sweep", ROOT / "scripts" / "run_R53_metadata_attachment_sweep.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)


def _classify(src: str):
    fn = [n for n in ast.walk(ast.parse(src))
          if isinstance(n, ast.FunctionDef)][0]
    return sweep.classify(fn)


ERR26_SHAPE = '''
def stage(xs, meta):
    df = run_batch(xs)
    for k in ("grade", "corner"):
        df[k] = [m[k] for m in meta]
    return df
'''

KEYED_SHAPE = '''
def stage(jobs):
    res = []
    for f in as_completed(futs):
        res.append(f.result())
    df = pd.DataFrame(res).sort_values(["point", "model_id"])
    return df
'''

DICT_SHAPE = '''
def stage(jobs):
    results = {}
    for f in as_completed(futs):
        out = f.result()
        results[out["key"]] = out
    return results
'''


def test_the_sweep_flags_the_err26_shape():
    """POSITIVE CONTROL: the defect it was written for, unguarded."""
    assert _classify(ERR26_SHAPE)["class"] == "UNGUARDED"


def test_the_sweep_is_quiet_on_a_keyed_shape():
    assert _classify(KEYED_SHAPE)["class"] == "NO_ATTACH"


def test_a_dict_keyed_by_an_id_is_not_a_column_attach():
    """Four of these came back as false positives on the first run."""
    got = _classify(DICT_SHAPE)
    assert got is None or got["class"] == "NO_ATTACH"


def test_a_scalar_broadcast_is_not_a_positional_attach():
    got = _classify('''
def stage(path, arm):
    d = pd.read_parquet(path)
    d["arm"] = arm
    return d
''')
    assert got is None or got["class"] != "POSITIONAL"


#: R-54.2(b): GUARDED sites assert the alignment they rely on; UNGUARDED is the
#: count that must be ZERO.  A permanently flagged guarded site would train its
#: readers to ignore the flag, so the two are separate classes.
KNOWN_GUARDED = {"scripts/run_ch9_campaign.py:stage_e5"}


def test_the_repository_has_no_unguarded_attachment_site():
    sites, guarded = [], []
    for p in sorted(ROOT.rglob("*.py")):
        if set(p.parts) & sweep.SKIP_DIRS:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            got = sweep.classify(fn)
            if got and got["class"] == "UNGUARDED":
                sites.append(f"{p.relative_to(ROOT).as_posix()}:{fn.name}")
            elif got and got["class"] == "GUARDED":
                guarded.append(f"{p.relative_to(ROOT).as_posix()}:{fn.name}")
    assert sites == []
    assert set(guarded) == KNOWN_GUARDED


def test_the_alignment_assertion_catches_a_shuffled_frame():
    """POSITIVE CONTROL for the guard itself, on ERR-26's actual shuffle."""
    pd = pytest.importorskip("pandas")
    import run_ch9_campaign as rc
    xs = [{"omega": 1.0, "c0": 2e-6}, {"omega": 2.0, "c0": 5e-6}]
    good = pd.DataFrame(xs)
    rc.assert_label_alignment(good, xs)               # aligned: silent
    shuffled = good.iloc[::-1].reset_index(drop=True)
    with pytest.raises(SystemExit, match="label alignment FAILED"):
        rc.assert_label_alignment(shuffled, xs)
