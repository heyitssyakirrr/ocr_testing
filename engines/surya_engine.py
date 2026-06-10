"""
engines/surya_engine.py
=======================
Surya OCR v1 — CPU only (surya-ocr == 0.16.x).
Install: pip install surya-ocr==0.16.7
"""

from __future__ import annotations
import os
import time

from engines.base import EngineResult


def run(images: list, use_gpu: bool = False) -> EngineResult:
    os.environ["TORCH_DEVICE"]           = "cpu"
    os.environ["RECOGNITION_BATCH_SIZE"] = "8"
    os.environ["DETECTOR_BATCH_SIZE"]    = "2"

    print("  [Surya v1] Loading models on CPU...")
    try:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection  import DetectionPredictor
    except ImportError as exc:
        return "[SKIP] surya-ocr not installed: %s  pip install surya-ocr==0.16.7" % exc, 0.0

    try:
        foundation_predictor = FoundationPredictor()
        rec_predictor        = RecognitionPredictor(foundation_predictor)
        det_predictor        = DetectionPredictor()
    except Exception as exc:
        return "[SKIP] Surya model load failed: %s" % exc, 0.0

    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img in images:
        print("  [Surya] Processing page %d..." % page_num)
        t_page = time.time()
        try:
            pil_img     = img.convert("RGB")
            predictions = rec_predictor([pil_img], det_predictor=det_predictor)
            page_lines  = 0
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
    print("  [Surya] Total: %.1fs on CPU" % elapsed)
    return "\n".join(all_lines), elapsed