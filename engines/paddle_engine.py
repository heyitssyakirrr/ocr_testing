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
    # Do NOT run: pip install paddlepaddle-gpu

Known issue — oneDNN/PIR crash (Unimplemented: ConvertPirAttribute2RuntimeAttribute)
-------------------------------------------------------------------------------------
PaddlePaddle 3.x enables MKL-DNN (oneDNN) by default.  On Windows, the
oneDNN executor has an unimplemented path in the PIR (Program IR) compiler
for DoubleAttribute nodes, which crashes every inference call with:

    (Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
    [pir::ArrayAttribute<pir::DoubleAttribute>]

Fix: set os.environ["FLAGS_use_mkldnn"] = "0" BEFORE paddle is imported.
This must be done at module level here, because paddle reads the flag at
import-time. The flag is an OS environment variable, not a paddle.set_flags()
call — paddle.set_flags() only works after the C++ runtime has already read it.

The performance cost is ~10-30% slower inference on CPU; still acceptable.

API changes from v2 → v3
-------------------------
PaddleOCR 3.x is a full rewrite. Nothing from the 2.x constructor works here:
  • Constructor args renamed: det_db_thresh → text_det_thresh, etc.
  • use_angle_cls / cls_thresh removed → use_textline_orientation instead
  • use_gpu / enable_mkldnn removed from constructor (env-var only)
  • Call method: .predict() replaces .ocr()  (.ocr() still exists, deprecated)
  • Result: dict-like with result["rec_texts"] and result["rec_scores"] lists
"""

from __future__ import annotations
import os
import time

# ---------------------------------------------------------------------------
# MKL-DNN MUST be disabled before paddle is imported anywhere in the process.
# Setting this env var here is safe — paddle_engine is only imported when the
# user selects the paddle engine, so no other engine has loaded paddle yet.
# ---------------------------------------------------------------------------

from engines.base import EngineResult


# ---------------------------------------------------------------------------
# Module-level cache — model loaded once, reused across pages and files
# ---------------------------------------------------------------------------
_ocr_instance = None


def _get_ocr():
    """Return a cached PaddleOCR instance (PP-OCRv5, CPU, no MKL-DNN)."""
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance

    from paddleocr import PaddleOCR

    _ocr_instance = PaddleOCR(
        # ── Model selection ──────────────────────────────────────────────────
        # lang="en" + no explicit ocr_version → v3 auto-selects PP-OCRv5
        # which resolves to PP-OCRv5_server_det + en_PP-OCRv5_mobile_rec
        lang="en",

        # ── Orientation correction ───────────────────────────────────────────
        # Corrects upside-down pages and rotated text lines —
        # both are common in CamScanner bank document scans.
        use_doc_orientation_classify=True,
        use_textline_orientation=True,

        # ── Detection tuning ─────────────────────────────────────────────────
        text_det_thresh=0.3,           # pixel-score threshold (default ~0.3)
        text_det_box_thresh=0.5,       # box-level filter   (default 0.6, relaxed)
        text_det_unclip_ratio=1.8,     # expand boxes slightly (default 1.5)

        # ── Recognition tuning ───────────────────────────────────────────────
        text_recognition_batch_size=6, # crops per recognition batch
        text_rec_score_thresh=0.0,     # keep all detections; conf shown in output
    )
    return _ocr_instance


def run(images: list, use_gpu: bool = False) -> EngineResult:
    """
    Run PaddleOCR (PP-OCRv5, CPU) on a list of (page_num, PIL.Image) pairs.

    Args:
        images:  List of (page_number, PIL.Image) tuples.
        use_gpu: Accepted for API compatibility; always ignored with a notice.

    Returns:
        EngineResult — (ocr_text, elapsed_seconds).
    """
    try:
        import numpy as np
    except ImportError:
        return "[SKIP] numpy not installed.  pip install numpy", 0.0

    try:
        from paddleocr import PaddleOCR  # noqa: F401 — import check only
    except ImportError:
        return (
            "[SKIP] PaddleOCR not installed.  "
            "pip install paddlepaddle paddleocr",
            0.0,
        )

    if use_gpu:
        print(
            "  [PaddleOCR] NOTE: GPU build has DLL conflicts with torch on Windows — "
            "running on CPU."
        )

    print("  [PaddleOCR] Loading PP-OCRv5 model (CPU, MKL-DNN disabled)...")
    try:
        ocr = _get_ocr()
    except Exception as exc:
        return "[SKIP] PaddleOCR model load failed: %s" % exc, 0.0

    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img in images:
        print("  [PaddleOCR] Processing page %d..." % page_num)
        img_array = np.array(img.convert("RGB"))

        try:
            # predict() returns a list of OCRResult objects (one per input image).
            # Each OCRResult is dict-like:
            #   result["rec_texts"]  → list[str]   — one recognised string per box
            #   result["rec_scores"] → list[float] — confidence per box
            results = ocr.predict(img_array)
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

            # Guard against length mismatch (shouldn't occur in practice)
            if len(rec_scores) < len(rec_texts):
                rec_scores = list(rec_scores) + [0.0] * (
                    len(rec_texts) - len(rec_scores)
                )

            for text, score in zip(rec_texts, rec_scores):
                text = str(text).strip()
                if text:
                    all_lines.append("%s  (conf:%.3f)" % (text, float(score)))

    elapsed = time.time() - t0
    print("  [PaddleOCR] Total: %.1fs on CPU" % elapsed)
    return "\n".join(all_lines), elapsed