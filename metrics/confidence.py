"""
metrics/confidence.py
=====================
Metric 1 — Confidence Distribution

Extracts the model's self-reported certainty scores from OCR output lines
ending with "  (conf:X.XXX)".

IMPORTANT: These scores are NOT real accuracy. They reflect how certain the
model was about its own output. A model can be 0.99 confident and still
output "BUKIL" instead of "BUKIT". Use this to flag lines for manual review,
not to judge overall engine quality.
"""

from __future__ import annotations

from config.settings import CONF_HIGH_THRESHOLD, CONF_MEDIUM_THRESHOLD


def confidence_stats(ocr_text: str) -> dict | None:
    """
    Parse confidence scores embedded in OCR output lines.

    Returns a dict with aggregate statistics and the list of low-confidence
    lines, or None if no scores are present.
    """
    confs: list[float] = []
    for line in ocr_text.split("\n"):
        if "(conf:" in line:
            try:
                val = float(line.split("(conf:")[1].rstrip(")"))
                confs.append(val)
            except (ValueError, IndexError):
                pass

    if not confs:
        return None

    total  = len(confs)
    high   = sum(1 for c in confs if c >= CONF_HIGH_THRESHOLD)
    medium = sum(1 for c in confs if CONF_MEDIUM_THRESHOLD <= c < CONF_HIGH_THRESHOLD)
    low    = sum(1 for c in confs if c < CONF_MEDIUM_THRESHOLD)

    low_lines = [
        line.split("  (conf:")[0].strip()
        for line in ocr_text.split("\n")
        if "(conf:" in line
        and _parse_conf(line) < CONF_MEDIUM_THRESHOLD
    ]

    return {
        "count":     total,
        "avg_conf":  sum(confs) / total,
        "min_conf":  min(confs),
        "max_conf":  max(confs),
        "high":      high,
        "medium":    medium,
        "low":       low,
        "low_lines": low_lines,
    }


def _parse_conf(line: str) -> float:
    try:
        return float(line.split("(conf:")[1].rstrip(")"))
    except (ValueError, IndexError):
        return 1.0
