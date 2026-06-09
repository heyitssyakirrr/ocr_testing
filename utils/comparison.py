"""
utils/comparison.py
===================
Build a cross-engine comparison table and known-field summary for all engines
that ran successfully on the same PDF.
"""

from __future__ import annotations
from pathlib import Path

from engines.base import OCRResult
from metrics.confidence import confidence_stats
from metrics.field_check import check_known_fields


def build_and_write_comparison(
    results:      list[OCRResult],
    known_fields: dict[str, str],
    output_dir:   Path,
    pdf_stem:     str,
) -> Path | None:
    """
    Write a cross-engine comparison file if more than one engine succeeded.

    Args:
        results:      All OCRResult objects for a single PDF (completed only).
        known_fields: Known-field dict for this PDF (may be empty).
        output_dir:   Folder to write the comparison file into.
        pdf_stem:     PDF filename stem used to name the output file.

    Returns:
        Path of the written file, or None if fewer than two engines succeeded.
    """
    completed = [r for r in results if not r.skipped]
    if len(completed) < 2:
        return None

    comp_file = output_dir / ("comparison_%s.txt" % pdf_stem.lower().replace(" ", "_"))
    lines     = _build_table(completed)

    if known_fields:
        lines += _build_field_summary(completed, known_fields)

    comp_file.write_text("\n".join(lines), encoding="utf-8")
    return comp_file


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

_SEP = "=" * 80


def _build_table(results: list[OCRResult]) -> list[str]:
    lines: list[str] = []
    lines.append(_SEP)
    lines.append("CROSS-ENGINE COMPARISON")
    lines.append(_SEP)
    lines.append("")

    header = "%-20s  %8s  %8s  %5s  %6s  %6s  %6s" % (
        "Engine", "Time(s)", "Lines", "Skip?", "HIGH%", "MED%", "LOW%"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        if r.skipped:
            lines.append("%-20s  %8s  %8s  %5s" % (r.engine_key, "—", "—", "YES"))
            continue

        text_lines = [l for l in r.text.split("\n") if l.strip()]
        cs         = confidence_stats(r.text)
        n          = len(text_lines)

        if cs and cs["count"]:
            high_pct = cs["high"]   / cs["count"] * 100
            med_pct  = cs["medium"] / cs["count"] * 100
            low_pct  = cs["low"]    / cs["count"] * 100
        else:
            high_pct = med_pct = low_pct = 0.0

        lines.append("%-20s  %8.1f  %8d  %5s  %6.1f  %6.1f  %6.1f" % (
            r.engine_key, r.elapsed, n, "no",
            high_pct, med_pct, low_pct,
        ))

    lines.append("")
    lines.append("NOTE: HIGH/MED/LOW are confidence DISTRIBUTION bands, not accuracy bands.")
    lines.append("      Use the Known-Field Spot Check results to judge real-world accuracy.")
    lines.append("")
    return lines


def _build_field_summary(
    results: list[OCRResult],
    known_fields: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    lines.append("KNOWN-FIELD SPOT CHECK SUMMARY")
    lines.append("-" * 60)

    for r in results:
        fr      = check_known_fields(r.text, known_fields)
        exact   = sum(1 for x in fr if x.status == "EXACT")
        fuzzy   = sum(1 for x in fr if x.status == "FUZZY")
        missing = sum(1 for x in fr if x.status == "MISSING")
        lines.append("%-20s  EXACT:%-3d  FUZZY:%-3d  MISSING:%-3d" % (
            r.engine_key, exact, fuzzy, missing
        ))

    lines.append("")
    return lines
