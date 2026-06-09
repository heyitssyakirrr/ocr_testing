"""
engines/easyocr_engine.py
=========================
EasyOCR — CRAFT text detection + CRNN recognition.
GPU: via PyTorch CUDA (same install as docTR).
"""

from __future__ import annotations
import time

from engines.base import EngineResult


def run(images: list, use_gpu: bool = False) -> EngineResult:
    """
    Run EasyOCR on a list of (page_num, PIL.Image) pairs.
    """
    try:
        import numpy as np
        import easyocr
    except ImportError:
        return "[SKIP] EasyOCR not installed.  pip install easyocr", 0.0

    # Determine effective GPU availability
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False

    effective_gpu = use_gpu and cuda_ok
    if use_gpu and not cuda_ok:
        print("  [EasyOCR] WARNING: CUDA not available — falling back to CPU.")

    print("  [EasyOCR] Loading model (gpu=%s)..." % effective_gpu)
    try:
        reader = easyocr.Reader(["en"], gpu=effective_gpu, verbose=False)
    except Exception as exc:
        return "[SKIP] EasyOCR model load failed: %s" % exc, 0.0

    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img in images:
        print("  [EasyOCR] Processing page %d..." % page_num)
        img_array = np.array(img.convert("RGB"))
        try:
            results = reader.readtext(img_array, detail=1, paragraph=False)
            for (_bbox, text, conf) in results:
                if str(text).strip():
                    all_lines.append("%s  (conf:%.3f)" % (text, float(conf)))
        except Exception as exc:
            print("    [EasyOCR] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0
