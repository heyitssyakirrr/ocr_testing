"""
engines/surya_engine.py
=======================
Surya OCR v1 — pure PyTorch, no Docker/server required (surya-ocr == 0.16.x).

API for 0.16.x:
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()
    predictions = recognition_predictor([image], det_predictor=detection_predictor)

Output: list of PageOCRResult, each with .text_lines
Each text_line has .text (str) and .confidence (float)

Install:
    pip install surya-ocr==0.16.7

GPU: automatic via torch CUDA — same build already used by EasyOCR/docTR.
"""

from __future__ import annotations
import time

from engines.base import EngineResult


def run(images: list, use_gpu: bool = False) -> EngineResult:
    """
    Run Surya OCR v1 on a list of (page_num, PIL.Image) pairs.

    Returns:
        EngineResult — (ocr_text, elapsed_seconds).
    """
    # 1. Verify package is installed
    try:
        import surya  # noqa: F401
    except ImportError:
        return "[SKIP] surya-ocr not installed.  pip install surya-ocr==0.16.7", 0.0

    # 2. Import the 0.16.x API
    try:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
    except ImportError as exc:
        return "[SKIP] surya-ocr import failed: %s" % exc, 0.0

    # 3. FoundationPredictor is present in some 0.16.x builds — try it, fall back gracefully
    foundation_predictor = _try_load_foundation()

    # 4. Check GPU
    cuda_ok = _cuda_available()
    if use_gpu and not cuda_ok:
        print("  [Surya] WARNING: CUDA not available — falling back to CPU.")
    effective_gpu = use_gpu and cuda_ok
    print("  [Surya v1] Loading models (gpu=%s)..." % effective_gpu)

    # 5. Load predictors
    try:
        if foundation_predictor is not None:
            rec_predictor = RecognitionPredictor(foundation_predictor)
        else:
            rec_predictor = RecognitionPredictor()
        det_predictor = DetectionPredictor()
    except Exception as exc:
        return "[SKIP] Surya model load failed: %s" % exc, 0.0

    # 6. Run OCR page by page
    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img in images:
        print("  [Surya] Processing page %d..." % page_num)
        try:
            pil_img = img.convert("RGB")
            predictions = rec_predictor(
                [pil_img],
                det_predictor=det_predictor,
            )
            for page_pred in predictions:
                for line in page_pred.text_lines:
                    text = str(line.text).strip()
                    conf = float(getattr(line, "confidence", 1.0))
                    if text:
                        all_lines.append("%s  (conf:%.3f)" % (text, conf))
        except Exception as exc:
            print("    [Surya] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_load_foundation():
    """
    Try to load FoundationPredictor — present in some 0.16.x builds.
    Returns None if not available (older sub-versions use no foundation model).
    """
    try:
        from surya.foundation import FoundationPredictor
        return FoundationPredictor()
    except (ImportError, Exception):
        return None


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False