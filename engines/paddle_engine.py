"""
engines/paddle_engine.py
========================
PaddleOCR v3.x — PP-OCRv5, CPU only.
Install: pip install paddlepaddle==3.1.1 paddleocr>=3.0.0

Windows requirement
-------------------
These flags MUST exist in the OS environment BEFORE Python starts.
Add them once to your PowerShell profile (notepad $PROFILE):

    $env:FLAGS_use_mkldnn                   = "0"
    $env:FLAGS_enable_pir_api               = "0"
    $env:FLAGS_use_new_executor             = "0"
    $env:PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT = "0"

On Linux no flags are needed — just run python ocr_tester.py directly.
"""

from __future__ import annotations
import os
import platform
import threading
import time

from engines.base import EngineResult

_ocr_instance = None

_REQUIRED_FLAGS = {
    "FLAGS_use_mkldnn":                   "0",
    "FLAGS_enable_pir_api":               "0",
    "FLAGS_use_new_executor":             "0",
    "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT": "0",
}


def _check_windows_flags() -> str | None:
    """
    Returns an error message if required flags are missing on Windows.
    Returns None if all good (or if not on Windows).
    """
    if platform.system() != "Windows":
        return None

    missing = [k for k, v in _REQUIRED_FLAGS.items() if os.environ.get(k) != v]
    if not missing:
        return None

    lines = [
        "PaddleOCR v3.x requires these flags set BEFORE Python starts on Windows.",
        "Add them once to your PowerShell profile (run: notepad $PROFILE):",
        "",
    ]
    for k, v in _REQUIRED_FLAGS.items():
        lines.append('    $env:%s = "%s"' % (k, v))
    lines += [
        "",
        "Then restart your terminal. After that, just run: python ocr_tester.py",
        "Missing flags: %s" % ", ".join(missing),
    ]
    return "\n".join(lines)


def _get_ocr():
    """Return a cached PaddleOCR v3.x instance (PP-OCRv5, CPU)."""
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance

    from paddleocr import PaddleOCR
    _ocr_instance = PaddleOCR(
        lang="en",
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
    Guards against the Windows oneDNN deadlock when flags are missing.
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
        raise TimeoutError(
            "ocr.predict() exceeded %ds — Windows oneDNN deadlock detected.\n"
            "    Fix: add the 4 FLAGS to your PowerShell profile (see paddle_engine.py)." % timeout_sec
        )
    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result_holder[0]


def run(images: list, use_gpu: bool = False) -> EngineResult:
    try:
        import numpy as np
        from paddleocr import PaddleOCR  # noqa: F401
    except ImportError:
        return "[SKIP] PaddleOCR not installed.  pip install paddlepaddle==3.1.1 paddleocr>=3.0.0", 0.0

    # Warn early on Windows if flags are missing — but still attempt to run.
    # If flags are missing the predict() call will deadlock and hit the timeout.
    flag_error = _check_windows_flags()
    if flag_error:
        print("\n  [PaddleOCR] WARNING — missing required Windows flags:")
        for line in flag_error.split("\n"):
            print("    %s" % line)
        print()

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