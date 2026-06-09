"""
engines/base.py
===============
Shared data structures used by all engine modules and the rest of the codebase.

Every engine's run() function MUST return an EngineResult:
    (text: str, elapsed: float)

  • text   — OCR lines, each ending with "  (conf:X.XXX)" where available.
             If the engine cannot run, return a string starting with "[SKIP] "
             explaining why, and elapsed=0.0.
  • elapsed — wall-clock seconds for the OCR work (model load excluded when
              the model is cached between calls, included on first call).
"""

from __future__ import annotations
from typing import NamedTuple

# Return type for every engine's run() function
EngineResult = tuple[str, float]   # (text_or_skip_message, elapsed_seconds)


class OCRResult(NamedTuple):
    """Holds everything produced by one engine run on one PDF file."""
    engine_key:  str
    engine_name: str
    pdf_name:    str
    text:        str    # raw OCR lines with (conf:X.XXX) suffixes
    elapsed:     float  # seconds
    skipped:     bool   # True if the engine could not run
    skip_reason: str    # non-empty when skipped=True


def make_result(
    engine_key:  str,
    engine_name: str,
    pdf_name:    str,
    raw:         EngineResult,
) -> OCRResult:
    """
    Convert an EngineResult tuple into a fully-typed OCRResult.
    Handles the [SKIP] sentinel automatically.
    """
    text, elapsed = raw
    skipped = text.startswith("[SKIP]")
    return OCRResult(
        engine_key=engine_key,
        engine_name=engine_name,
        pdf_name=pdf_name,
        text="" if skipped else text,
        elapsed=elapsed,
        skipped=skipped,
        skip_reason=text[7:].strip() if skipped else "",
    )
