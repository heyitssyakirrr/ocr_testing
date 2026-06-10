"""
utils/device_probe.py
=====================
Collect evidence about available CPU libraries for the report header.
"""

from __future__ import annotations
import platform
import os


def get_device_evidence() -> list[str]:
    lines: list[str] = []
    lines.append("--- DEVICE EVIDENCE ---")
    lines.append("Platform  : %s %s" % (platform.system(), platform.release()))
    lines.append("Python    : %s"    % platform.python_version())
    lines.append("Mode      : CPU only")
    _probe_torch(lines)
    _probe_surya(lines)
    _probe_easyocr(lines)
    _probe_tesseract(lines)
    _probe_paddleocr(lines)
    return lines


def _probe_torch(lines: list[str]) -> None:
    try:
        import torch
        lines.append("PyTorch   : %s (CPU)" % torch.__version__)
    except ImportError:
        lines.append("PyTorch   : not installed")


def _probe_surya(lines: list[str]) -> None:
    try:
        import surya
        lines.append("Surya OCR : %s" % getattr(surya, "__version__", "unknown"))
    except ImportError:
        lines.append("Surya OCR : not installed  (pip install surya-ocr)")


def _probe_easyocr(lines: list[str]) -> None:
    try:
        import easyocr  # noqa: F401
        lines.append("EasyOCR   : installed")
    except ImportError:
        lines.append("EasyOCR   : not installed")


def _probe_tesseract(lines: list[str]) -> None:
    try:
        import pytesseract
        ver = pytesseract.get_tesseract_version()
        lines.append("Tesseract : %s" % ver)
    except ImportError:
        lines.append("Tesseract : pytesseract not installed")
    except Exception:
        lines.append("Tesseract : installed but binary not found in PATH")


def _probe_paddleocr(lines: list[str]) -> None:
    try:
        import paddleocr
        lines.append("PaddleOCR : %s" % getattr(paddleocr, "__version__", "unknown"))
    except ImportError:
        lines.append("PaddleOCR : not installed  (pip install paddlepaddle==3.1.1 paddleocr>=3.0.0)")
        return
    try:
        import paddle
        paddle_ver  = paddle.__version__
        paddle_cuda = getattr(paddle, "is_compiled_with_cuda", lambda: False)()
        lines.append("  paddlepaddle : %s  (CUDA build: %s)" % (paddle_ver, paddle_cuda))
        if paddle_cuda:
            lines.append("  *** WARNING: paddlepaddle-gpu detected — uninstall and run: pip install paddlepaddle==3.1.1")
        if platform.system() == "Windows":
            missing = [k for k, v in {
                "FLAGS_use_mkldnn": "0",
                "FLAGS_enable_pir_api": "0",
                "FLAGS_use_new_executor": "0",
                "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT": "0",
            }.items() if os.environ.get(k) != v]
            if missing:
                lines.append("  *** WARNING: Missing Windows flags: %s" % ", ".join(missing))
                lines.append("               Add them to PowerShell profile — see paddle_engine.py")
            else:
                lines.append("  Windows flags : OK")
    except ImportError:
        lines.append("  paddlepaddle : not importable")