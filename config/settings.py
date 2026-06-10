"""
config/settings.py
==================
Central configuration: defaults, engine registry, GPU-capable set.
CPU-only mode — all engines run on CPU.
"""

from __future__ import annotations

DEFAULT_INPUT_FOLDER  = "input_files"
DEFAULT_OUTPUT_DIR    = "ocr_results"
DEFAULT_DPI           = 300
DEFAULT_ENGINE        = "rapidocr"

CONF_HIGH_THRESHOLD   = 0.90
CONF_MEDIUM_THRESHOLD = 0.70
FUZZY_MATCH_THRESHOLD = 0.80

ENGINE_REGISTRY: dict[str, tuple[str, str]] = {
    "surya":     ("Surya OCR",       "engines.surya_engine"),
    "doctr":     ("docTR",           "engines.doctr_engine"),
    "rapidocr":  ("RapidOCR-ONNX",  "engines.rapidocr_engine"),
    "easyocr":   ("EasyOCR",        "engines.easyocr_engine"),
    "tesseract": ("Tesseract 5.x",  "engines.tesseract_engine"),
    "paddle":    ("PaddleOCR",      "engines.paddle_engine"),
}

# CPU-only — no engine uses GPU
GPU_CAPABLE_ENGINES: frozenset[str] = frozenset()