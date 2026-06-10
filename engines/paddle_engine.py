"""
engines/paddle_engine.py
========================
PaddleOCR v3 — PP-OCRv5 server detection + English mobile recognition (CPU).

GPU note
--------
PaddleOCR's GPU (paddlepaddle-gpu) build causes CUDA DLL conflicts on Windows
when torch is also installed. This engine is intentionally CPU-only.

Install
-------
    pip install paddlepaddle paddleocr

Critical Windows requirement
-----------------------------
The following env vars MUST be set BEFORE Python starts (i.e. in run_ocr.bat),
NOT via os.environ in Python — paddle reads them at C++ runtime init which
happens at the first `import paddle`, before any Python-level os.environ call
can take effect:

    SET FLAGS_use_mkldnn=0
    SET FLAGS_enable_pir_api=0
    SET FLAGS_use_new_executor=0
    SET PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0

Without these, every ocr.predict() call raises:
    NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
    [pir::ArrayAttribute<pir::DoubleAttribute>]

Orientation models
------------------
use_doc_orientation_classify and use_textline_orientation are disabled.
Both load UVDoc/orientation models that hit the same oneDNN crash path on
Windows CPU even when the above flags are set. Pre-rotate pages with Pillow
before passing to this engine if your documents may be rotated.
"""

from __future__ import annotations
import time
import threading

from engines.base import EngineResult

_ocr_instance = None


def _get_ocr():
    """Return a cached PaddleOCR instance (PP-OCRv5, CPU)."""
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance

    from paddleocr import PaddleOCR

    _ocr_instance = PaddleOCR(
        lang="en",

        # Orientation models disabled — both trigger the oneDNN PIR crash on
        # Windows CPU regardless of FLAGS_enable_pir_api. Pre-rotate with
        # Pillow upstream if needed.
        use_doc_orientation_classify=False,
        use_textline_orientation=False,

        text_det_thresh=0.3,
        text_det_box_thresh=0.5,
        text_det_unclip_ratio=1.8,
        text_recognition_batch_size=6,
        text_rec_score_thresh=0.0,
    )
    return _ocr_instance


def _predict_with_timeout(ocr, img_array, timeout_sec: int = 120):
    """
    Run ocr.predict() in a daemon thread with a hard timeout.

    PaddleOCR on Windows can deadlock silently instead of raising an exception.
    A daemon thread with a join timeout is the safest cross-platform guard —
    the model object is not picklable so multiprocessing is not an option.

    Raises TimeoutError if the call doesn't return within timeout_sec.
    """
    result_holder = [None]
    exc_holder    = [None]

    def _worker():
        try:
            result_holder[0] = ocr.predict(img_array)
        except Exception as e:
            exc_holder[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        raise TimeoutError("ocr.predict() exceeded %ds — likely a Windows oneDNN deadlock" % timeout_sec)
    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result_holder[0]


def run(images: list, use_gpu: bool = False) -> EngineResult:
    """
    Run PaddleOCR (PP-OCRv5, CPU) on a list of (page_num, PIL.Image) pairs.
    """
    try:
        import numpy as np
    except ImportError:
        return "[SKIP] numpy not installed.  pip install numpy", 0.0

    try:
        from paddleocr import PaddleOCR  # noqa: F401
    except ImportError:
        return "[SKIP] PaddleOCR not installed.  pip install paddlepaddle paddleocr", 0.0

    if use_gpu:
        print("  [PaddleOCR] NOTE: GPU build conflicts with torch on Windows — running CPU.")

    print("  [PaddleOCR] Loading PP-OCRv5 model (CPU)...")
    try:
        ocr = _get_ocr()
    except Exception as exc:
        return "[SKIP] PaddleOCR model load failed: %s" % exc, 0.0

    import numpy as np
    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img in images:
        print("  [PaddleOCR] Processing page %d..." % page_num)
        img_array = np.array(img.convert("RGB"))

        try:
            results = _predict_with_timeout(ocr, img_array, timeout_sec=120)
        except TimeoutError as exc:
            print("    [PaddleOCR] Page %d TIMED OUT: %s" % (page_num, exc))
            print("    [PaddleOCR] Ensure run_ocr.bat sets FLAGS_enable_pir_api=0 before Python starts.")
            continue
        except Exception as exc:
            print("    [PaddleOCR] Page %d error: %s" % (page_num, exc))
            continue

        if not results:
            print("    [PaddleOCR] Page %d: no text detected." % page_num)
            continue

        for page_result in results:
            if page_result is None:
                continue
            rec_texts  = page_result.get("rec_texts",  []) or []
            rec_scores = page_result.get("rec_scores", []) or []
            if len(rec_scores) < len(rec_texts):
                rec_scores = list(rec_scores) + [0.0] * (len(rec_texts) - len(rec_scores))
            for text, score in zip(rec_texts, rec_scores):
                text = str(text).strip()
                if text:
                    all_lines.append("%s  (conf:%.3f)" % (text, float(score)))

    elapsed = time.time() - t0
    print("  [PaddleOCR] Total: %.1fs on CPU" % elapsed)
    return "\n".join(all_lines), elapsed