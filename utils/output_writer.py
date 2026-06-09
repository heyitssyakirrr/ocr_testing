"""
utils/output_writer.py
======================
Write a structured analysis report for one engine run on one PDF file.
"""

from __future__ import annotations
from pathlib import Path

from engines.base import OCRResult
from metrics.confidence import confidence_stats
from metrics.cer import cer_against_pdf_layer
from metrics.field_check import check_known_fields
from utils.flagging import flag_critical_lines
from config.settings import CONF_HIGH_THRESHOLD, CONF_MEDIUM_THRESHOLD, FUZZY_MATCH_THRESHOLD

_SEP = "=" * 80


def write_output(
    result:          OCRResult,
    device_evidence: list[str],
    dpi:             int,
    output_dir:      Path,
    pdf_text_layer:  str,
    known_fields:    dict[str, str],
) -> Path:
    """
    Write the full analysis report for one engine + one PDF.

    Returns the Path of the written file.
    """
    stem     = Path(result.pdf_name).stem.lower().replace(" ", "_")
    out_file = output_dir / ("%s_%s_output.txt" % (result.engine_key, stem))

    lines    = result.text.split("\n") if not result.skipped else []
    non_empty = [l for l in lines if l.strip()]

    with open(out_file, "w", encoding="utf-8") as f:
        _write_header(f, result, dpi, non_empty)
        if result.skipped:
            f.write("Engine could not run: %s\n" % result.skip_reason)
            return out_file

        f.write("\n".join(device_evidence) + "\n\n")
        _write_metric1(f, result.text)
        _write_metric2(f, result.text, pdf_text_layer)
        _write_metric3(f, result.text, known_fields)
        _write_full_text(f, non_empty)
        _write_flagged(f, result.text)

    return out_file


# ---------------------------------------------------------------------------
# Section writers
# ---------------------------------------------------------------------------

def _write_header(f, result: OCRResult, dpi: int, non_empty: list[str]) -> None:
    f.write(_SEP + "\n")
    f.write("FILE   : %s\n" % result.pdf_name)
    f.write("ENGINE : %s\n" % result.engine_name)
    f.write("DPI    : %d\n" % dpi)
    f.write("TIME   : %.1f seconds\n" % result.elapsed)
    f.write("LINES  : %d\n" % len(non_empty))
    if result.skipped:
        f.write("STATUS : SKIPPED — %s\n" % result.skip_reason)
    f.write(_SEP + "\n\n")


def _write_metric1(f, ocr_text: str) -> None:
    f.write(_SEP + "\n")
    f.write("METRIC 1 — CONFIDENCE DISTRIBUTION\n")
    f.write(_SEP + "\n\n")
    f.write("IMPORTANT: This is NOT accuracy. It is the model's self-reported\n")
    f.write("certainty about its own output. A high score does not mean the\n")
    f.write("text was read correctly — the model can be confidently wrong.\n")
    f.write("Use this to find individual lines that need manual review.\n\n")

    cs = confidence_stats(ocr_text)
    if cs:
        f.write("  Total lines scored : %d\n"   % cs["count"])
        f.write("  Average confidence : %.4f\n" % cs["avg_conf"])
        f.write("  Min / Max          : %.4f / %.4f\n" % (cs["min_conf"], cs["max_conf"]))
        f.write("\n  Band breakdown (for individual line review):\n")
        f.write("    HIGH   (>= %.2f) : %d lines  (%.1f%%)  — model is fairly certain\n" % (
            CONF_HIGH_THRESHOLD, cs["high"], cs["high"] / cs["count"] * 100))
        f.write("    MEDIUM (%.2f–%.2f): %d lines  (%.1f%%)  — model had some difficulty\n" % (
            CONF_MEDIUM_THRESHOLD, CONF_HIGH_THRESHOLD,
            cs["medium"], cs["medium"] / cs["count"] * 100))
        f.write("    LOW    (< %.2f)  : %d lines  (%.1f%%)  — review these manually\n" % (
            CONF_MEDIUM_THRESHOLD, cs["low"], cs["low"] / cs["count"] * 100))
        if cs["low_lines"]:
            f.write("\n  Low-confidence lines (the ones to check first):\n")
            for ll in cs["low_lines"]:
                f.write("    - %s\n" % ll)
    else:
        f.write("  No confidence scores available for this engine.\n")
    f.write("\n")


def _write_metric2(f, ocr_text: str, pdf_text_layer: str) -> None:
    f.write(_SEP + "\n")
    f.write("METRIC 2 — CHARACTER ERROR RATE vs PDF TEXT LAYER\n")
    f.write(_SEP + "\n\n")

    cer_data = cer_against_pdf_layer(ocr_text, pdf_text_layer)
    if cer_data is None:
        f.write("  SKIPPED — this PDF has no embedded text layer (image-only / scanned).\n")
        f.write("  This is expected for documents scanned with CamScanner or similar.\n")
        f.write("  CER can only be measured when the PDF contains selectable text.\n")
    else:
        f.write("  Reference chars (PDF layer) : %d\n" % cer_data["ref_chars"])
        f.write("  OCR output chars            : %d\n" % cer_data["hyp_chars"])
        f.write("  Character edit distance     : %d\n" % cer_data["edit_distance"])
        f.write("  CER                         : %.4f  (%.2f%%)\n" % (
            cer_data["cer"], cer_data["cer_pct"]))
        f.write("\n  Interpretation:\n")
        f.write("    < 2%%   Excellent — nearly perfect text extraction\n")
        f.write("    2–5%%   Good      — minor errors, spot-check critical fields\n")
        f.write("    5–10%%  Fair      — noticeable errors, review key fields\n")
        f.write("    > 10%%  Poor      — significant errors, try higher DPI\n")
    f.write("\n")


def _write_metric3(f, ocr_text: str, known_fields: dict[str, str]) -> None:
    f.write(_SEP + "\n")
    f.write("METRIC 3 — KNOWN-FIELD SPOT CHECK\n")
    f.write(_SEP + "\n\n")

    if not known_fields:
        f.write("  No known fields provided.\n")
        f.write("  Create input_files/known_fields.txt with lines like:\n")
        f.write("    FI Code        : 034707062\n")
        f.write("    Master Account : 0000006072902892170888001\n")
        f.write("    Bank Name      : AFFIN BANK BERHAD\n")
        f.write("\n")
        return

    field_results = check_known_fields(ocr_text, known_fields)
    exact   = [r for r in field_results if r.status == "EXACT"]
    fuzzy   = [r for r in field_results if r.status == "FUZZY"]
    missing = [r for r in field_results if r.status == "MISSING"]

    f.write("  Summary: %d EXACT  |  %d FUZZY  |  %d MISSING  (of %d fields)\n\n" % (
        len(exact), len(fuzzy), len(missing), len(field_results)))

    f.write("  EXACT matches (read correctly):\n")
    if exact:
        for r in exact:
            f.write("    [OK] %-30s : %s\n" % (r.field, r.expected))
    else:
        f.write("    (none)\n")
    f.write("\n")

    f.write("  FUZZY matches (present but with typos — similarity %.0f%%+):\n" % (
        FUZZY_MATCH_THRESHOLD * 100))
    if fuzzy:
        for r in fuzzy:
            f.write("    [~] %-30s\n" % r.field)
            f.write("        Expected : %s\n" % r.expected)
            f.write("        Found as : %s  (similarity: %.1f%%)\n" % (
                r.found_as, r.score * 100))
    else:
        f.write("    (none)\n")
    f.write("\n")

    f.write("  MISSING (not found — score below %.0f%%):\n" % (FUZZY_MATCH_THRESHOLD * 100))
    if missing:
        for r in missing:
            f.write("    [X] %-30s : %s  (best match score: %.1f%%)\n" % (
                r.field, r.expected, r.score * 100))
    else:
        f.write("    (none — all fields found)\n")
    f.write("\n")


def _write_full_text(f, non_empty: list[str]) -> None:
    f.write(_SEP + "\n")
    f.write("FULL OCR TEXT\n")
    f.write(_SEP + "\n\n")
    for line in non_empty:
        f.write(line + "\n")
    f.write("\n")


def _write_flagged(f, ocr_text: str) -> None:
    f.write(_SEP + "\n")
    f.write("CRITICAL LINES — POSSIBLE CHARACTER CONFUSION\n")
    f.write("(Only lines with patterns relevant to field extraction)\n")
    f.write(_SEP + "\n\n")
    flagged = flag_critical_lines(ocr_text)
    if flagged:
        for flag in flagged:
            f.write(flag + "\n")
    else:
        f.write("No suspicious patterns detected.\n")
    f.write("\n")
