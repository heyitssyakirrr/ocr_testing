"""
metrics/field_check.py
======================
Metric 3 — Known-Field Spot Check

Looks for specific values that MUST appear in the document (account numbers,
FI codes, bank names). Reports each one as:

  EXACT   — value found verbatim (spaces ignored)
  FUZZY   — similarity >= FUZZY_MATCH_THRESHOLD (catches minor OCR typos)
  MISSING — similarity below threshold (field was not read correctly)

This is the most meaningful metric for bank-document field extraction.
"""

from __future__ import annotations
import difflib
import re
from typing import NamedTuple

from config.settings import FUZZY_MATCH_THRESHOLD


class FieldCheckResult(NamedTuple):
    field:    str    # field label, e.g. "FI Code"
    expected: str    # ground-truth value
    status:   str    # "EXACT" | "FUZZY" | "MISSING"
    found_as: str    # what OCR produced (empty if MISSING)
    score:    float  # similarity 0.0–1.0


def check_known_fields(
    ocr_text: str,
    known_fields: dict[str, str],
) -> list[FieldCheckResult]:
    """
    Compare each known field against the full OCR text.

    Args:
        ocr_text:     Raw OCR output with "(conf:X.XXX)" annotations.
        known_fields: Mapping of field_name → expected_value.

    Returns:
        List of FieldCheckResult, one per field.
    """
    # Strip confidence annotations and join everything
    clean_lines = [
        line.split("  (conf:")[0].strip()
        for line in ocr_text.split("\n")
        if line.strip()
    ]
    full_text          = " ".join(clean_lines)
    full_text_nospace  = re.sub(r"\s+", "", full_text)

    results: list[FieldCheckResult] = []

    for field_name, expected in known_fields.items():
        expected_nospace = re.sub(r"\s+", "", expected)

        # 1. Exact match (spaces stripped)
        if expected_nospace in full_text_nospace:
            results.append(FieldCheckResult(
                field=field_name,
                expected=expected,
                status="EXACT",
                found_as=expected,
                score=1.0,
            ))
            continue

        # 2. Fuzzy: slide a window of the same length over the no-space text
        best_score = 0.0
        best_found = ""
        window     = len(expected_nospace)
        for i in range(max(1, len(full_text_nospace) - window + 1)):
            candidate = full_text_nospace[i : i + window]
            score     = difflib.SequenceMatcher(
                None,
                expected_nospace.lower(),
                candidate.lower(),
            ).ratio()
            if score > best_score:
                best_score = score
                best_found = candidate

        if best_score >= FUZZY_MATCH_THRESHOLD:
            results.append(FieldCheckResult(
                field=field_name,
                expected=expected,
                status="FUZZY",
                found_as=best_found,
                score=best_score,
            ))
        else:
            results.append(FieldCheckResult(
                field=field_name,
                expected=expected,
                status="MISSING",
                found_as="",
                score=best_score,
            ))

    return results
