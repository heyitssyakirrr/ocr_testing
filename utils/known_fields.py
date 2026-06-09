"""
utils/known_fields.py
=====================
Load per-PDF known-field files for Metric 3 (Known-Field Spot Check).

Expected file: <pdf_folder>/known_fields.txt
Format (one field per line, colon-separated):
    FI Code : 034707062
    Master Account : 0000006072902892170888001
    Sub Account    : 0000000406070003660
    Bank Name      : AFFIN BANK BERHAD

Lines starting with # are comments and are ignored.
"""

from __future__ import annotations
from pathlib import Path


def load_known_fields(pdf_path: Path) -> dict[str, str]:
    """
    Look for known_fields.txt in the same folder as the given PDF.

    Returns a dict of {field_name: expected_value}.
    Returns an empty dict if the file does not exist.
    """
    kf_path = pdf_path.parent / "known_fields.txt"
    if not kf_path.exists():
        return {}

    fields: dict[str, str] = {}
    with kf_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if key and val:
                    fields[key] = val

    return fields
