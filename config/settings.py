"""
config/settings.py
==================
Central configuration: defaults, engine registry, GPU-capable set.

To add a new engine:
  1. Implement engines/<name>_engine.py with a run(images, use_gpu) function.
  2. Import it below and add an entry to ENGINE_REGISTRY.
  3. If it supports GPU, add its key to GPU_CAPABLE_ENGINES.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_INPUT_FOLDER  = "input_files"
DEFAULT_OUTPUT_DIR    = "ocr_results"
DEFAULT_DPI           = 300
DEFAULT_ENGINE        = "rapidocr"

# ---------------------------------------------------------------------------
# Confidence band thresholds
# ---------------------------------------------------------------------------
CONF_HIGH_THRESHOLD   = 0.90
CONF_MEDIUM_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Fuzzy-match threshold for known-field spot check
# ---------------------------------------------------------------------------
FUZZY_MATCH_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# Engine registry
# key → (display_name, module_path)
#
# Modules are imported lazily inside each engine file to avoid paying import
# costs (and DLL load failures) for engines that are not being used.
#
# REMOVED engines (DLL / library issues):
#   paddle          — PaddleOCR PP-OCRv5  (Windows CUDA DLL conflicts)
#   rapidocr_paddle — RapidOCR-Paddle     (same paddle DLL issue)
# ---------------------------------------------------------------------------
ENGINE_REGISTRY: dict[str, tuple[str, str]] = {
    "surya":    ("Surya OCR",        "engines.surya_engine"),
    "doctr":    ("docTR",            "engines.doctr_engine"),
    "rapidocr": ("RapidOCR-ONNX",   "engines.rapidocr_engine"),
    "easyocr":  ("EasyOCR",         "engines.easyocr_engine"),
    "tesseract":("Tesseract 5.x",   "engines.tesseract_engine"),
}

# Keys whose run() functions can make use of CUDA when use_gpu=True.
GPU_CAPABLE_ENGINES: frozenset[str] = frozenset({
    "surya",
    "doctr",
    "easyocr",
})