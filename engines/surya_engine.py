"""
engines/surya_engine.py
=======================
Surya OCR v1 — pure PyTorch (surya-ocr == 0.16.x).

Key insight from settings inspection:
  - TORCH_DEVICE_MODEL=cuda  → model IS on GPU when --gpu used
  - RECOGNITION_BATCH_SIZE=None → resolves to a tiny default = bottleneck
  - DETECTOR_BATCH_SIZE=None → same issue

Fix: set RECOGNITION_BATCH_SIZE and DETECTOR_BATCH_SIZE env vars BEFORE
any surya import, since Surya reads them at import time via pydantic-settings.

RTX 2050 (4GB VRAM) safe batch sizes:
  RECOGNITION_BATCH_SIZE=256  — recognition transformer, each item is a
                                 small cropped text line, very low VRAM cost
  DETECTOR_BATCH_SIZE=4       — detection is heavier per item

Install:
    pip install surya-ocr==0.16.7
"""

from __future__ import annotations
import os
import time

from engines.base import EngineResult


def run(images: list, use_gpu: bool = False) -> EngineResult:
    """
    Run Surya OCR v1 on a list of (page_num, PIL.Image) pairs.
    """
    # 1. Resolve device FIRST — before any surya import
    #    Surya reads TORCH_DEVICE and batch sizes at import time
    cuda_ok      = _cuda_available()
    effective_gpu = use_gpu and cuda_ok

    if use_gpu and not cuda_ok:
        print("  [Surya] WARNING: CUDA not available — falling back to CPU.")
    elif not use_gpu:
        print("  [Surya] TIP: Run with --gpu for ~10x speed (RTX 2050 detected).")

    if effective_gpu:
        os.environ["TORCH_DEVICE"]            = "cuda"
        os.environ["RECOGNITION_BATCH_SIZE"]  = "256"  # text lines are tiny crops
        os.environ["DETECTOR_BATCH_SIZE"]     = "4"    # heavier — keep lower
    else:
        os.environ["TORCH_DEVICE"]            = "cpu"
        os.environ["RECOGNITION_BATCH_SIZE"]  = "8"
        os.environ["DETECTOR_BATCH_SIZE"]     = "2"

    device_str = "cuda" if effective_gpu else "cpu"
    print("  [Surya v1] Loading models on %s..." % device_str)

    # 2. Import AFTER env vars are set — pydantic-settings reads them at import
    try:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
    except ImportError as exc:
        return "[SKIP] surya-ocr not installed: %s  pip install surya-ocr==0.16.7" % exc, 0.0

    # 3. Load all three predictors
    try:
        foundation_predictor = FoundationPredictor()
        rec_predictor        = RecognitionPredictor(foundation_predictor)
        det_predictor        = DetectionPredictor()
    except Exception as exc:
        return "[SKIP] Surya model load failed: %s" % exc, 0.0

    print("  [Surya] Recognition batch size : %s" % rec_predictor.get_batch_size())
    print("  [Surya] Detection batch size   : %s" % det_predictor.get_batch_size())

    # 4. Run OCR page by page
    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img in images:
        print("  [Surya] Processing page %d..." % page_num)
        t_page = time.time()
        try:
            pil_img = img.convert("RGB")
            predictions = rec_predictor(
                [pil_img],
                det_predictor=det_predictor,
            )
            page_lines = 0
            for page_pred in predictions:
                for line in page_pred.text_lines:
                    text = str(line.text).strip()
                    conf = float(getattr(line, "confidence", 1.0))
                    if text:
                        all_lines.append("%s  (conf:%.3f)" % (text, conf))
                        page_lines += 1
            print("  [Surya] Page %d done in %.1fs — %d lines" % (
                page_num, time.time() - t_page, page_lines))
        except Exception as exc:
            print("    [Surya] Page %d error: %s" % (page_num, exc))

    elapsed = time.time() - t0
    print("  [Surya] Total: %.1fs on %s" % (elapsed, device_str))
    return "\n".join(all_lines), elapsed


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False