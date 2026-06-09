"""
engines/paddle_engine.py
========================
PaddleOCR -- PP-OCRv5 server models (paddleocr >= 3.0) with graceful
fallback to PP-OCRv4 (paddleocr 2.x).

GPU disabled on Windows due to CUDA DLL path issue with paddlepaddle-gpu 3.3.0.
CPU mode only. Accuracy is identical to GPU; only speed differs.
"""

from __future__ import annotations
import os
import time

from engines.base import EngineResult


def run(images: list, use_gpu: bool = False) -> EngineResult:
    use_gpu = False
    os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
    os.environ["FLAGS_use_mkldnn"]      = "0"
    os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
    os.environ["PADDLE_USE_GPU"]        = "0"
    os.environ["FLAGS_use_cuda"]        = "0"

    try:
        import numpy as np
        import paddleocr as _poc
    except ImportError:
        return "[SKIP] PaddleOCR not installed.  pip install paddlepaddle paddleocr", 0.0

    return _run_v2(images, all_lines=[], t0=time.time())


# ---------------------------------------------------------------------------
# PP-OCRv5 (paddleocr 3.x)
# ---------------------------------------------------------------------------

def _run_v3(images: list, all_lines: list, t0: float) -> EngineResult:
    try:
        import numpy as np
        from paddleocr import PaddleOCR
    except ImportError:
        return "[SKIP] PaddleOCR not installed.", 0.0

    print("  [PaddleOCR] Loading PP-OCRv5 models (CPU, paddlex-free mode)...")
    try:
        # use_angle_cls and lang force the legacy 2.x-style init path
        # which does NOT go through paddlex pipeline and avoids the DLL issue
        ocr = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            device="cpu",
        )
    except Exception as exc:
        return "[SKIP] PaddleOCR 3.x model load failed: %s" % exc, 0.0

    for page_num, img in images:
        print("  [PaddleOCR] Processing page %d..." % page_num)
        try:
            img_array = np.array(img.convert("RGB"))
            results = ocr.ocr(img_array, cls=True)
            if not results:
                continue
            for page_result in results:
                if not page_result:
                    continue
                for line in page_result:
                    try:
                        text, conf = line[1][0], line[1][1]
                        if str(text).strip():
                            all_lines.append("%s  (conf:%.3f)" % (text, float(conf)))
                    except (IndexError, TypeError):
                        continue
        except Exception as exc:
            print("    [PaddleOCR] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0


# ---------------------------------------------------------------------------
# PP-OCRv4 fallback (paddleocr 2.x)
# ---------------------------------------------------------------------------

def _run_v2(images: list, all_lines: list, t0: float) -> EngineResult:
    try:
        import numpy as np
        from paddleocr import PaddleOCR
    except ImportError:
        return "[SKIP] PaddleOCR not installed.", 0.0

    print("  [PaddleOCR] Loading PP-OCRv4 models (2.x fallback).")
    try:
        ocr = PaddleOCR(
            lang="en",
            use_gpu=False,
            use_angle_cls=True,
            enable_mkldnn=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=2.0,
            rec_batch_num=1,
            show_log=False,
        )
    except Exception as exc:
        return "[SKIP] PaddleOCR 2.x model load failed: %s" % exc, 0.0

    for page_num, img in images:
        print("  [PaddleOCR] Processing page %d..." % page_num)
        try:
            import numpy as np
            img_array = np.array(img.convert("RGB"))
            results   = ocr.ocr(img_array, cls=True)
            if not results:
                continue
            for page_result in results:
                if not page_result:
                    continue
                for line in page_result:
                    try:
                        text, conf = line[1][0], line[1][1]
                        if str(text).strip():
                            all_lines.append("%s  (conf:%.3f)" % (text, conf))
                    except (IndexError, TypeError):
                        continue
        except Exception as exc:
            print("    [PaddleOCR] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_attr(obj, key: str, default):
    """Get key from dict or attribute, returning default if absent."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)