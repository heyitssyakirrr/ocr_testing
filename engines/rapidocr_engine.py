"""
engines/rapidocr_engine.py
==========================
RapidOCR — ONNX runtime backend, CPU only by design.

Fast and lightweight; no GPU support regardless of --gpu flag.
"""

from __future__ import annotations
import time

from engines.base import EngineResult


def run(images: list, use_gpu: bool = False) -> EngineResult:
    """
    Run RapidOCR (ONNX) on a list of (page_num, PIL.Image) pairs.

    use_gpu is accepted but always ignored; a notice is printed if True.
    """
    try:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return "[SKIP] RapidOCR not installed.  pip install rapidocr-onnxruntime", 0.0

    if use_gpu:
        print("  [RapidOCR-ONNX] NOTE: RapidOCR has no GPU support — running on CPU.")

    print("  [RapidOCR-ONNX] Loading model...")
    try:
        ocr = RapidOCR()
    except Exception as exc:
        return "[SKIP] RapidOCR model load failed: %s" % exc, 0.0

    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img in images:
        print("  [RapidOCR-ONNX] Processing page %d..." % page_num)
        import numpy as np
        img_array = np.array(img.convert("RGB"))
        try:
            result, _ = ocr(img_array)
            if result is None:
                continue
            for item in result:
                if len(item) >= 3:
                    text, conf = item[1], item[2]
                    if str(text).strip():
                        all_lines.append("%s  (conf:%.3f)" % (text, float(conf)))
        except Exception as exc:
            print("    [RapidOCR-ONNX] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0
