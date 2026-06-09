r"""
engines/rapidocr_paddle_engine.py
==================================
RapidOCR -- PaddlePaddle backend.  Supports GPU via paddlepaddle-gpu.

BUG FIX: If the early `get_device_evidence()` call imported paddle and it
failed (DLL not found), the half-initialised paddle module stays cached in
sys.modules and causes this import to fail too.

Fix applied here: purge all paddle* entries from sys.modules before importing
rapidocr_paddle, giving it a clean slate.

If you still see DLL errors, apply the one-time DLL copy fix:
  copy "venv\Lib\site-packages\nvidia\cudnn\bin\*.dll"        "venv\Lib\site-packages\paddle\"
  copy "venv\Lib\site-packages\nvidia\cublas\bin\*.dll"       "venv\Lib\site-packages\paddle\"
  copy "venv\Lib\site-packages\nvidia\cuda_runtime\bin\*.dll" "venv\Lib\site-packages\paddle\"
"""

from __future__ import annotations
import sys
import time

from engines.base import EngineResult


def _purge_stale_paddle_modules() -> None:
    """
    Remove any partially-loaded paddle modules from sys.modules.
    A failed import leaves a broken stub; subsequent imports reuse the stub
    and also fail. Purging forces a clean re-import.
    """
    stale = [key for key in sys.modules if key == "paddle" or key.startswith("paddle.")]
    for key in stale:
        del sys.modules[key]
    if stale:
        print("  [RapidOCR-Paddle] Purged %d stale paddle module(s) from sys.modules." % len(stale))


def run(images: list, use_gpu: bool = False) -> EngineResult:
    """
    Run RapidOCR (Paddle backend) on a list of (page_num, PIL.Image) pairs.
    """
    _purge_stale_paddle_modules()

    # Force CPU before any paddle import — same DLL issue as paddle_engine
    import os
    use_gpu = False
    os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
    os.environ["FLAGS_use_mkldnn"]      = "0"
    os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
    os.environ["PADDLE_USE_GPU"]        = "0"
    os.environ["FLAGS_use_cuda"]        = "0"

    try:
        import numpy as np
    except ImportError:
        return "[SKIP] numpy not installed.", 0.0

    try:
        from rapidocr_paddle import RapidOCR
    except ImportError:
        return "[SKIP] rapidocr-paddle not installed.  pip install rapidocr-paddle", 0.0
    except Exception as exc:
        return "[SKIP] rapidocr-paddle import error: %s" % exc, 0.0

    print("  [RapidOCR-Paddle] Loading model (gpu=%s)..." % use_gpu)
    ocr = _load_model(use_gpu)
    if isinstance(ocr, str):
        return ocr, 0.0   # skip message

    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img in images:
        print("  [RapidOCR-Paddle] Processing page %d..." % page_num)
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
            print("    [RapidOCR-Paddle] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0


def _load_model(use_gpu: bool):
    """
    Attempt to construct RapidOCR with GPU kwargs; fall back to default
    constructor for older versions that don't accept them.
    Returns the RapidOCR instance, or a "[SKIP] ..." string on failure.
    """
    from rapidocr_paddle import RapidOCR
    try:
        return RapidOCR(
            det_use_cuda=use_gpu,
            cls_use_cuda=use_gpu,
            rec_use_cuda=use_gpu,
        )
    except TypeError:
        # Older rapidocr-paddle versions don't accept cuda kwargs
        print("  [RapidOCR-Paddle] Note: GPU kwargs not supported by this version — using defaults.")
        try:
            return RapidOCR()
        except Exception as exc:
            return "[SKIP] RapidOCR-Paddle model load failed: %s" % exc
    except Exception as exc:
        return "[SKIP] RapidOCR-Paddle model load failed: %s" % exc
