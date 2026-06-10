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
#   rapidocr_paddle — RapidOCR-Paddle     (paddle DLL conflicts)
#
# CPU-ONLY engines (GPU not supported or disabled intentionally):
#   paddle          — PaddleOCR PP-OCRv5  (GPU build has Windows CUDA DLL
#                     conflicts; CPU build with MKL-DNN is stable everywhere)
#   rapidocr        — RapidOCR-ONNX       (ONNX runtime, no GPU path)
#   tesseract       — Tesseract 5.x        (CPU binary, no GPU path)
# ---------------------------------------------------------------------------
ENGINE_REGISTRY: dict[str, tuple[str, str]] = {
    "surya":    ("Surya OCR",        "engines.surya_engine"),
    "doctr":    ("docTR",            "engines.doctr_engine"),
    "rapidocr": ("RapidOCR-ONNX",   "engines.rapidocr_engine"),
    "easyocr":  ("EasyOCR",         "engines.easyocr_engine"),
    "tesseract":("Tesseract 5.x",   "engines.tesseract_engine"),
    "paddle":   ("PaddleOCR",       "engines.paddle_engine"),    # CPU-only, PP-OCRv5
}

# Keys whose run() functions can make use of CUDA when use_gpu=True.
# paddle is intentionally excluded — see note above.
GPU_CAPABLE_ENGINES: frozenset[str] = frozenset({
    "surya",
    "doctr",
    "easyocr",
})