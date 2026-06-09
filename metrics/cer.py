"""
metrics/cer.py
==============
Metric 2 — Character Error Rate (CER) proxy

Compares OCR output against the PDF's embedded text layer (if it exists).
For scanned / image-only PDFs there is no text layer, so this metric is
skipped automatically.

Formula:   CER = levenshtein_distance / len(reference)
Lower = better.  0.0 = perfect.  0.05 = 5% of characters are wrong.
"""

from __future__ import annotations
import difflib
import re


def cer_against_pdf_layer(ocr_text: str, pdf_text_layer: str) -> dict | None:
    """
    Compute a CER proxy between the OCR output and the PDF text layer.

    Returns None if the PDF has no text layer (scanned image).
    Returns a dict with edit_distance, CER value, and character counts.
    """
    if not pdf_text_layer.strip():
        return None

    # Strip confidence annotations
    clean_ocr = "\n".join(
        line.split("  (conf:")[0].strip()
        for line in ocr_text.split("\n")
        if line.strip() and not line.startswith("[SKIP]")
    )

    # Normalise whitespace for fair comparison
    ref = re.sub(r"\s+", " ", pdf_text_layer).strip().lower()
    hyp = re.sub(r"\s+", " ", clean_ocr).strip().lower()

    matcher = difflib.SequenceMatcher(None, ref, hyp)
    n_errors = sum(
        max(b - a, d - c)
        for tag, a, b, c, d in matcher.get_opcodes()
        if tag != "equal"
    )
    cer = n_errors / max(len(ref), 1)

    return {
        "ref_chars":     len(ref),
        "hyp_chars":     len(hyp),
        "edit_distance": n_errors,
        "cer":           cer,
        "cer_pct":       cer * 100,
    }
