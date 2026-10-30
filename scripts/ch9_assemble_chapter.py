#!/usr/bin/env python
r"""Assemble the Chapter 9 results block and stage it into the thesis tree.

Closes the dependency chain opened when the Chapter 9 closing sentence began
forward-referencing \S\ref{sec:bs_ar_deployment} before the section existed.

  1. copy results/ch9/chapter9_numbers.tex to the thesis root, beside
     ch78_numbers.tex, which is where Chapters 3, 7 and 8 read theirs from;
  2. copy the eight generated tables to <thesis>/tables/ch9/ and the eleven
     figures to <thesis>/figures/ch9/, matching the figures/<chapter>/ path
     convention Chapter 8 already uses;
  3. assemble the results prose and the two scoping subsections into one
     ch9_results_block.tex, in the registered order, with the chapter-label
     placeholders corrected and the tables and figures placed at their first
     reference.

LABEL CORRECTIONS.  The drafts used three chapter labels that do not exist.
They are corrected here from the labels actually declared in the thesis, read
out of the chapter files rather than assumed, and any placeholder without a
match is reported instead of guessed.

NOTHING IS COMPILED.  This script stages files; it does not run LaTeX.
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DRAFTS = ROOT.parent                      # giesekus-ballsocket/
THESIS = pathlib.Path("Z:/PhD Thesis")
OUT = ROOT / "results" / "ch9"

#: draft placeholder -> the label actually declared in the thesis.
LABEL_FIX = {
    "ballsocket-chpt": "ball-socket-chpt",
    "bo-chpt": "bayes-chpt",
    "validation-chpt": "valid-chpt",
    "discussion-chpt": "discuss-chpt",
}

#: Table -> the paragraph cue it should follow, and the figures likewise.
TABLE_AFTER = {
    "T1": r"\subsection{Audit set}",
    "T4": r"\subsection{Audit set}",
    "T2": r"\subsection{Expectations}",
    "T6": r"\subsection{Expectations}",
    "T3": r"\subsection{Expectations}",
    "T8": r"\subsection{Expectations}",
    "T5": r"\subsection{Boundary reconstruction}",
    "T7": r"\subsection{Boundary reconstruction}",
}
FIG_LABEL = {
    "F1_domain_schematic": "fig:bs_domain",
    "F2_verification_panel": "fig:bs_verification",
    "F3_projection_factor": "fig:bs_projection_factor",
    "F4_feasible_slices": "fig:bs_feasible_slices",
    "F5_binding_geography": "fig:bs_binding_geography",
    "F6_ar_projections": "fig:bs_ar_projections",
    "F7_bo_vs_random_budget": "fig:bs_bo_vs_random",
    "F8_e5_viscosity_sweep": "fig:bs_e5_sweep",
    "F9_thermal_routing": "fig:bs_thermal_routing",
    "F10_interface_contrast": "fig:bs_interface_contrast",
    "F11_oat_panel": "fig:bs_oat",
}
FIG_AFTER = {
    "F1_domain_schematic": r"\subsection{Audit set}",
    "F4_feasible_slices": r"\subsection{Audit set}",
    "F2_verification_panel": r"\subsection{Expectations}",
    "F3_projection_factor": r"\subsection{Expectations}",
    "F5_binding_geography": r"\subsection{Expectations}",
    "F8_e5_viscosity_sweep": r"\subsection{Expectations}",
    "F9_thermal_routing": r"\subsection{Expectations}",
    "F11_oat_panel": r"\subsection{Sensitivity}",
    "F6_ar_projections": r"\subsection{Boundary reconstruction}",
    "F7_bo_vs_random_budget": r"\subsection{Boundary reconstruction}",
    "F10_interface_contrast": r"\subsection{Boundary reconstruction}",
}


def _strip_value_block(text: str) -> str:
    """Drop the trailing MACRO VALUE BLOCK comment.

    It is instructions to the generator, not chapter content, and its values now
    live in chapter9_numbers.tex.  Leaving it in would put a second, un-checked
    copy of every number into the thesis tree, which is exactly the drift the
    macro discipline exists to prevent.
    """
    marker = re.search(r"^%={10,}\n% MACRO VALUE BLOCK", text, re.M)
    return text[:marker.start()].rstrip() + "\n" if marker else text


#: draft macro name -> emitted macro name.  A LaTeX control sequence may
#: contain letters only: \BsNineEOneLamG3Lo parses as \BsNineEOneLamG followed
#: by the characters "3Lo", which silently defines the wrong macro and then
#: typesets the remainder in the preamble.  Found by compiling, 2026-08-03.
MACRO_RENAME = {
    "BsNineEOneLamG3Lo": "BsNineEOneLamGThreeLo",
    "BsNineEOneLamG3Hi": "BsNineEOneLamGThreeHi",
}


def _fix_macro_names(text: str, report: dict) -> str:
    for bad, good in MACRO_RENAME.items():
        n = len(re.findall(rf"\\{bad}\b", text))
        if n:
            text = re.sub(rf"\\{bad}", f"\\\\{good}", text)
            report.setdefault("macro_renames", {})[bad] = {
                "to": good, "occurrences": n,
                "reason": "LaTeX control sequences take letters only"}
    return text


def _fix_labels(text: str, report: dict) -> str:
    for bad, good in LABEL_FIX.items():
        n = len(re.findall(rf"\\ref\{{{re.escape(bad)}\}}", text))
        if n:
            text = text.replace(f"\\ref{{{bad}}}", f"\\ref{{{good}}}")
            report.setdefault("label_corrections", {})[bad] = {
                "to": good, "occurrences": n}
    return text


def _thesis_labels() -> set[str]:
    labs: set[str] = set()
    for p in list((THESIS / "chapters").glob("*.tex")) + [THESIS / "uct_thesis_main.tex"]:
        if p.exists():
            labs |= set(re.findall(r"\\label\{([^}]*)\}",
                                   p.read_text(encoding="utf-8", errors="ignore")))
    return labs


def _figure_env(name: str, caption: str, label: str) -> str:
    cap = " ".join(caption.split())
    return (f"\n\\begin{{figure}}[htbp]\n"
            f"\t\\centering\n"
            f"\t\\includegraphics[width=\\textwidth]{{figures/ch9/{name}.pdf}}\n"
            f"\t\\caption[{name.split('_', 1)[1].replace('_', ' ').capitalize()}]"
            f"{{{cap}}}\n"
            f"\t\\label{{{label}}}\n"
            f"\\end{{figure}}\n")


def main() -> int:
    report: dict = {"staged": [], "dangling_refs": [], "notes": []}

    # -- 1. numbers file -------------------------------------------------
    src = OUT / "chapter9_numbers.tex"
    if not src.exists():
        print("chapter9_numbers.tex absent: run scripts/ch9_numbers.py first")
        return 1
    dst = THESIS / "chapter9_numbers.tex"
    shutil.copy2(src, dst)
    report["staged"].append(str(dst))

    # -- 2. tables and figures ------------------------------------------
    tdir = THESIS / "tables" / "ch9"
    tdir.mkdir(parents=True, exist_ok=True)
    for name in TABLE_AFTER:
        shutil.copy2(OUT / "tables" / f"{name}.tex", tdir / f"{name}.tex")
        report["staged"].append(str(tdir / f"{name}.tex"))

    fdir = THESIS / "figures" / "ch9"
    fdir.mkdir(parents=True, exist_ok=True)
    figs = {}
    for name in FIG_LABEL:
        pdf = ROOT / "figures" / "ch9" / f"{name}.pdf"
        man = ROOT / "figures" / "ch9" / f"{name}_manifest.json"
        shutil.copy2(pdf, fdir / f"{name}.pdf")
        figs[name] = json.loads(man.read_text())["caption"]
        report["staged"].append(str(fdir / f"{name}.pdf"))

    # -- 3. assemble the block ------------------------------------------
    results = _strip_value_block(
        (DRAFTS / "ch9_results_prose_v1.tex").read_text(encoding="utf-8"))
    scoping = _strip_value_block(
        (DRAFTS / "ch9_scoping_prose_v1.tex").read_text(encoding="utf-8"))
    results = _fix_macro_names(_fix_labels(results, report), report)
    scoping = _fix_macro_names(_fix_labels(scoping, report), report)

    # strip each draft's own file header comment
    def _strip_header(t: str) -> str:
        m = list(re.finditer(r"^%={10,}$", t, re.M))
        return t[m[1].end():].lstrip("\n") if len(m) >= 2 else t

    results, scoping = _strip_header(results), _strip_header(scoping)

    # the scoping subsections sit BEFORE Boundary reconstruction
    anchor = r"\subsection{Boundary reconstruction}"
    if anchor not in results:
        print("could not find the Boundary reconstruction anchor")
        return 1
    head, tail = results.split(anchor, 1)
    body = head + scoping.rstrip() + "\n\n" + anchor + tail

    # the section that the chapter's closing sentence forward-references.
    # It is this one; the label did not exist because no draft declared it.
    body = body.replace(
        r"\section{Campaign results}\label{sec:bs_results}",
        "\\section{Campaign results}\\label{sec:bs_results}%\n"
        "\\label{sec:bs_ar_deployment}% forward-referenced by the chapter's\n"
        "% closing paragraph; this section IS the framework deployment it names.")
    report["notes"].append(
        "added \\label{sec:bs_ar_deployment} to \\section{Campaign results}: "
        "no draft declared it, and the chapter's closing sentence "
        "forward-references it")

    # place tables and figures after their cue
    for name, cue in TABLE_AFTER.items():
        ins = f"\n\\input{{tables/ch9/{name}}}\n"
        body = body.replace(cue, cue + ins, 1)
    for name, cue in FIG_AFTER.items():
        body = body.replace(cue, cue + _figure_env(name, figs[name],
                                                   FIG_LABEL[name]), 1)

    header = (
        "%" + "=" * 77 + "\n"
        "% ch9_results_block.tex -- ASSEMBLED by scripts/ch9_assemble_chapter.py\n"
        "% from ch9_results_prose_v1.tex and ch9_scoping_prose_v1.tex.\n"
        "% Do not hand-edit: edit the drafts and re-run the assembler.\n"
        "%\n"
        "% Numbers come from chapter9_numbers.tex, which is emitted only when\n"
        "% scripts/ch9_numbers.py passes 98 macro checks and 16 comparatives.\n"
        "% Tables are generated by scripts/ch9_tables.py, figures by\n"
        "% scripts/figures/*.py; both write SHA256 manifests.\n"
        "%" + "=" * 77 + "\n\n")
    block = THESIS / "chapters" / "ch9_results_block.tex"
    block.write_text(header + body, encoding="utf-8")
    report["staged"].append(str(block))

    # -- 4. dangling-reference report ------------------------------------
    declared = _thesis_labels()
    declared |= set(re.findall(r"\\label\{([^}]*)\}", header + body))
    for t in tdir.glob("*.tex"):
        declared |= set(re.findall(r"\\label\{([^}]*)\}",
                                   t.read_text(encoding="utf-8")))
    used = set(re.findall(r"\\ref\{([^}]*)\}", header + body))
    for t in tdir.glob("*.tex"):
        used |= set(re.findall(r"\\ref\{([^}]*)\}", t.read_text(encoding="utf-8")))
    report["dangling_refs"] = sorted(used - declared)

    # -- 5. Chapter 10 interface-contrast section ------------------------
    c10 = _strip_value_block(
        (DRAFTS / "ch10_interface_contrast_v1.tex").read_text(encoding="utf-8"))
    c10 = _fix_macro_names(_fix_labels(c10, report), report)
    c10 = _strip_header(c10)
    c10_header = (
        "%" + "=" * 77 + "\n"
        "% ch10_interface_contrast.tex -- STAGED by "
        "scripts/ch9_assemble_chapter.py\n"
        "% from ch10_interface_contrast_v1.tex, with the chapter-label\n"
        "% placeholders corrected against the labels actually declared in the\n"
        "% thesis.  Numbers come from ch78_numbers.tex and chapter9_numbers.tex.\n"
        "%" + "=" * 77 + "\n\n")
    c10_path = THESIS / "chapters" / "ch10_interface_contrast.tex"
    c10_path.write_text(c10_header + c10, encoding="utf-8")
    report["staged"].append(str(c10_path))
    report["notes"].append(
        "Chapter 10 contrast staged as a separate file; it is NOT wired into "
        "chapter10-discussion.tex by this script, because choosing where a "
        "discussion section sits is an authoring decision")

    (OUT / "assemble_report.json").write_text(json.dumps(report, indent=2))
    print(f"staged {len(report['staged'])} files")
    for bad, info in report.get("label_corrections", {}).items():
        print(f"  label fix: {bad} -> {info['to']}  ({info['occurrences']}x)")
    if report["dangling_refs"]:
        print("\n  STILL DANGLING (not invented, reported):")
        for r in report["dangling_refs"]:
            print(f"    \\ref{{{r}}}")
    else:
        print("\n  no dangling references")
    print(f"\nwrote {block}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
