"""
utils/flagging.py
=================
Flag OCR lines that contain patterns suggesting character-confusion errors
relevant to bank document field extraction (account numbers, FI codes, etc.).

Only genuinely suspicious patterns are flagged — not every line that happens
to contain common letters next to digits.
"""

from __future__ import annotations
import re

# Each entry: (compiled_regex, human-readable reason)
_CRITICAL_PATTERNS: list[tuple[re.Pattern, str]] = [
    # HSBC O/D suffix — letter O vs digit 0 confusion
    (re.compile(r"\b\d{9}[O0]/[D][0-9]{2}\b"),          "HSBC O/D suffix — check letter O vs digit 0"),
    # Account numbers that should be pure digits but contain letters
    (re.compile(r"\b\d*[A-NP-Z]\d{3,}\b"),              "Account number may contain OCR letter/digit confusion"),
    # FI codes (7–9 pure digits) containing an unexpected letter
    (re.compile(r"\b0[0-9]{1,2}[A-Z][0-9]{4,}\b"),      "FI code contains unexpected letter"),
    # Common single-character confusions inside digit runs
    (re.compile(r"\b[0-9]+[lI][0-9]+\b"),               "Possible 1/l/I confusion in number"),
    (re.compile(r"\b[0-9]+[oO][0-9]+\b"),               "Possible 0/O confusion in number"),
    (re.compile(r"\b[0-9]+[sS][0-9]+\b"),               "Possible 5/S confusion in number"),
    (re.compile(r"\b[0-9]+[bB][0-9]+\b"),               "Possible 6/b confusion in number"),
]


def flag_critical_lines(text: str) -> list[str]:
    """
    Return a list of formatted warning strings for lines that match any of
    the critical patterns.  At most one flag per line.

    Args:
        text: Raw OCR output (may include "(conf:X.XXX)" annotations).

    Returns:
        List of "[!] reason\\n      Line: <line>" strings.
    """
    flagged: list[str] = []
    for line in text.split("\n"):
        clean = line.split("  (conf:")[0].strip()
        if not clean:
            continue
        for pattern, reason in _CRITICAL_PATTERNS:
            if pattern.search(clean):
                flagged.append("  [!] %s\n      Line: %s" % (reason, line))
                break   # one flag per line is enough
    return flagged
