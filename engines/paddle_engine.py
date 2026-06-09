"""
engines/paddle_engine.py
========================
PaddleOCR -- PP-OCRv4/v5 with automatic version detection.

GPU: paddlepaddle-gpu 3.0.0 + CUDA 12.3
  pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

DLL fix (Windows): copy nvidia CUDA DLLs into paddle/libs — see README.md.
"""

from __future__ import annotations
import os
import time

from engines.base import EngineResult


def run(images: list, use_gpu: bool = False) -> EngineResult:
    # On Windows, only disable GPU if the DLL fix hasn't been applied.
    # Check by attempting a paddle import to see if CUDA loads cleanly.
    if use_gpu:
        use_gpu = _verify_gpu_usable()

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
        os.environ["FLAGS_use_mkldnn"]      = "0"
        os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
        os.environ["PADDLE_USE_GPU"]        = "0"
        os.environ["FLAGS_use_cuda"]        = "0"

    try:
        import numpy as np
        import paddle
    except ImportError:
        return "[SKIP] PaddleOCR not installed.  pip install paddlepaddle paddleocr", 0.0

    # Detect paddleocr version and route accordingly
    try:
        import paddleocr as _poc
        version = getattr(_poc, "__version__", "2.0.0")
        major = int(version.split(".")[0])
    except Exception:
        major = 2

    all_lines: list[str] = []
    t0 = time.time()

    if major >= 3:
        return _run_v3(images, all_lines, t0, use_gpu)
    else:
        return _run_v2(images, all_lines, t0, use_gpu)


def _verify_gpu_usable() -> bool:
    """
    Try importing paddle to check if CUDA DLLs load cleanly.
    Returns True if GPU is usable, False if DLL errors occur.
    """
    try:
        import paddle
        return paddle.device.is_compiled_with_cuda()
    except Exception as exc:
        print("  [PaddleOCR] WARNING: GPU requested but paddle CUDA check failed: %s" % exc)
        print("  [PaddleOCR] Falling back to CPU. Run the DLL copy fix — see README.md.")
        return False


def _run_v3(images: list, all_lines: list, t0: float, use_gpu: bool) -> EngineResult:
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return "[SKIP] PaddleOCR not installed.", 0.0

    device = "gpu" if use_gpu else "cpu"
    print("  [PaddleOCR] Loading PP-OCRv5 models (device=%s)..." % device)
    try:
        ocr = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            device=device,
        )
    except Exception as exc:
        return "[SKIP] PaddleOCR 3.x model load failed: %s" % exc, 0.0

    import numpy as np
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


def _run_v2(images: list, all_lines: list, t0: float, use_gpu: bool) -> EngineResult:
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return "[SKIP] PaddleOCR not installed.", 0.0

    print("  [PaddleOCR] Loading PP-OCRv4 models (use_gpu=%s)..." % use_gpu)
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

    import numpy as np
    for page_num, img in images:
        print("  [PaddleOCR] Processing page %d..." % page_num)
        try:
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


def _get_attr(obj, key: str, default):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)