r"""
engines/paddle_engine.py
========================
PaddleOCR -- PP-OCRv5 server models (paddleocr >= 3.0) with graceful
fallback to PP-OCRv4 (paddleocr 2.x).

GPU: requires paddlepaddle-gpu.
     If DLL loading fails on Windows, copy the nvidia DLLs manually:
       copy "venv\Lib\site-packages\nvidia\cudnn\bin\*.dll"        "venv\Lib\site-packages\paddle\"
       copy "venv\Lib\site-packages\nvidia\cublas\bin\*.dll"       "venv\Lib\site-packages\paddle\"
       copy "venv\Lib\site-packages\nvidia\cuda_runtime\bin\*.dll" "venv\Lib\site-packages\paddle\"
"""

from __future__ import annotations
import os
import time

from engines.base import EngineResult


def run(images: list, use_gpu: bool = False) -> EngineResult:
    """
    Run PaddleOCR on a list of (page_num, PIL.Image) pairs.

    Args:
        images:  List of (page_num, PIL.Image) tuples.
        use_gpu: Request CUDA acceleration.

    Returns:
        EngineResult — (ocr_text, elapsed_seconds).
        Returns a "[SKIP] ..." string if the engine cannot initialise.
    """
    try:
        import numpy as np
        from paddleocr import PaddleOCR
        import paddleocr as _poc
        paddle_ver = tuple(int(x) for x in _poc.__version__.split(".")[:2])
    except ImportError:
        return "[SKIP] PaddleOCR not installed.  pip install paddlepaddle paddleocr", 0.0

    # Disable MKL-DNN — it conflicts with CUDA builds on Windows
    os.environ["FLAGS_use_mkldnn"]      = "0"
    os.environ["PADDLE_DISABLE_ONEDNN"] = "1"

    all_lines: list[str] = []
    t0 = time.time()

    if paddle_ver >= (3, 0):
        return _run_v3(images, use_gpu, all_lines, t0)
    else:
        return _run_v2(images, use_gpu, all_lines, t0)


# ---------------------------------------------------------------------------
# PP-OCRv5 (paddleocr 3.x)
# ---------------------------------------------------------------------------

def _run_v3(images: list, use_gpu: bool, all_lines: list[str], t0: float) -> EngineResult:
    from paddleocr import PaddleOCR
    import numpy as np

    print("  [PaddleOCR] Loading PP-OCRv5 server models (3.x API)...")
    try:
        ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device="gpu" if use_gpu else "cpu",
        )
    except Exception as exc:
        return "[SKIP] PaddleOCR 3.x model load failed: %s" % exc, 0.0

    for page_num, img in images:
        print("  [PaddleOCR] Processing page %d..." % page_num)
        img_array = np.array(img.convert("RGB"))
        try:
            results = ocr.predict(img_array)
            if not results:
                continue
            for res in results:
                rec_texts  = _get_attr(res, "rec_texts",  [])
                rec_scores = _get_attr(res, "rec_scores", [])
                for text, conf in zip(rec_texts, rec_scores):
                    if str(text).strip():
                        all_lines.append("%s  (conf:%.3f)" % (text, float(conf)))
        except Exception as exc:
            print("    [PaddleOCR] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0


# ---------------------------------------------------------------------------
# PP-OCRv4 fallback (paddleocr 2.x)
# ---------------------------------------------------------------------------

def _run_v2(images: list, use_gpu: bool, all_lines: list[str], t0: float) -> EngineResult:
    from paddleocr import PaddleOCR
    import numpy as np

    print("  [PaddleOCR] Loading PP-OCRv4 models (2.x fallback).")
    print("  [PaddleOCR] Upgrade to paddlepaddle>=3.0 + paddleocr>=3.0 for PP-OCRv5.")
    try:
        ocr = PaddleOCR(
            lang="en",
            use_gpu=use_gpu,
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

    return "\n".join(all_lines), time.time() - t0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_attr(obj, key: str, default):
    """Get key from dict or attribute, returning default if absent."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
