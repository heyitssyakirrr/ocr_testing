"""
engines/tesseract_engine.py
===========================
Tesseract 5.x via pytesseract — CPU only, no GPU support.

Requires:
  1. Tesseract binary installed and in PATH.
     Windows: https://github.com/UB-Mannheim/tesseract/wiki
  2. pip install pytesseract

Config:
  --oem 3  — LSTM neural net engine (most accurate)
  --psm 6  — assume a uniform block of text
  -l eng   — English language model
"""

from __future__ import annotations
import time

from engines.base import EngineResult

_TESS_CONFIG = r"--oem 3 --psm 6 -l eng"


def run(images: list, use_gpu: bool = False) -> EngineResult:
    """
    Run Tesseract on a list of (page_num, PIL.Image) pairs.
    use_gpu is accepted but always ignored.
    """
    try:
        import pytesseract
    except ImportError:
        return "[SKIP] pytesseract not installed.  pip install pytesseract", 0.0

    # Verify binary is reachable before processing any pages
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return (
            "[SKIP] Tesseract binary not found in PATH. "
            "Install from https://github.com/UB-Mannheim/tesseract/wiki "
            "and add its folder to your system PATH.",
            0.0,
        )

    if use_gpu:
        print("  [Tesseract] NOTE: Tesseract has no GPU support — running on CPU.")

    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img in images:
        print("  [Tesseract] Processing page %d..." % page_num)
        try:
            data = pytesseract.image_to_data(
                img.convert("RGB"),
                config=_TESS_CONFIG,
                output_type=pytesseract.Output.DICT,
            )
            all_lines.extend(_group_into_lines(data))
        except Exception as exc:
            print("    [Tesseract] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0


def _group_into_lines(data: dict) -> list[str]:
    """Group Tesseract word-level data into lines with average confidence."""
    line_words: dict[tuple, list[tuple[str, float]]] = {}

    for i, word in enumerate(data["text"]):
        word = str(word).strip()
        if not word:
            continue
        conf = int(data["conf"][i])
        if conf < 0:   # -1 = no confidence data (non-text block)
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line_words.setdefault(key, []).append((word, conf / 100.0))

    out = []
    for key in sorted(line_words):
        words_confs = line_words[key]
        line_text   = " ".join(w for w, _ in words_confs)
        avg_conf    = sum(c for _, c in words_confs) / len(words_confs)
        out.append("%s  (conf:%.3f)" % (line_text, avg_conf))
    return out
