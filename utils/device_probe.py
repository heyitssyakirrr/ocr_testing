"""
utils/device_probe.py
=====================
Collect hard evidence about which compute device is actually available.
"""

from __future__ import annotations
import os
import platform


def get_device_evidence(use_gpu: bool) -> list[str]:
    """
    Return a list of human-readable lines describing the detected hardware
    and library state.  Written into every output report for reproducibility.
    """
    lines: list[str] = []
    lines.append("--- DEVICE / GPU EVIDENCE ---")
    lines.append("Platform             : %s %s" % (platform.system(), platform.release()))
    lines.append("Python               : %s" % platform.python_version())
    lines.append("--gpu flag           : %s" % use_gpu)
    lines.append("CUDA_VISIBLE_DEVICES : %s" % os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))

    _probe_torch(lines, use_gpu)
    _probe_surya(lines)
    _probe_easyocr(lines)
    _probe_tesseract(lines)
    _probe_paddleocr(lines)

    return lines


# ---------------------------------------------------------------------------
# Per-library probes — each is isolated so one failure doesn't block others
# ---------------------------------------------------------------------------

def _probe_torch(lines: list[str], use_gpu: bool) -> None:
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        lines.append("PyTorch version           : %s" % torch.__version__)
        lines.append("torch.cuda.is_available() : %s" % cuda_ok)
        if cuda_ok:
            try:
                lines.append("  CUDA device : %s" % torch.cuda.get_device_name(0))
                lines.append("  VRAM total  : %.1f GB" % (
                    torch.cuda.get_device_properties(0).total_memory / 1e9
                ))
            except AssertionError:
                lines.append("  CUDA device : not accessible (CUDA_VISIBLE_DEVICES may be empty)")
        active = "cuda" if (use_gpu and cuda_ok) else "cpu"
        lines.append("Active torch device       : %s" % active)
        if use_gpu and not cuda_ok:
            lines.append("  *** WARNING: --gpu requested but CUDA not available.")
    except ImportError:
        lines.append("PyTorch : not installed")


def _probe_surya(lines: list[str]) -> None:
    try:
        import surya
        ver = getattr(surya, "__version__", "unknown")
        lines.append("Surya OCR version : %s (GPU via torch CUDA)" % ver)
    except ImportError:
        lines.append("Surya OCR : not installed  (pip install surya-ocr)")


def _probe_easyocr(lines: list[str]) -> None:
    try:
        import easyocr   # noqa: F401
        lines.append("EasyOCR : installed (uses torch CUDA if available)")
    except ImportError:
        lines.append("EasyOCR : not installed")


def _probe_tesseract(lines: list[str]) -> None:
    try:
        import pytesseract
        ver = pytesseract.get_tesseract_version()
        lines.append("Tesseract version : %s (CPU only)" % ver)
    except ImportError:
        lines.append("Tesseract : pytesseract not installed")
    except Exception:
        lines.append("Tesseract : pytesseract installed but binary not found in PATH")


def _probe_paddleocr(lines: list[str]) -> None:
    """Probe PaddleOCR and the underlying paddlepaddle runtime."""
    try:
        import paddleocr
        ver = getattr(paddleocr, "__version__", "unknown")
        lines.append("PaddleOCR version : %s (CPU only — PP-OCRv5)" % ver)
    except ImportError:
        lines.append(
            "PaddleOCR : not installed  "
            "(pip install paddlepaddle paddleocr)"
        )
        return

    # Also report the paddlepaddle runtime version and whether it was built
    # with CUDA (it shouldn't be, given our CPU-only install guidance)
    try:
        import paddle
        paddle_ver   = paddle.__version__
        paddle_cuda  = getattr(paddle, "is_compiled_with_cuda", lambda: False)()
        lines.append(
            "  paddlepaddle version  : %s  (CUDA build: %s)" % (paddle_ver, paddle_cuda)
        )
        if paddle_cuda:
            lines.append(
                "  *** WARNING: paddlepaddle-gpu detected — "
                "this may cause DLL conflicts on Windows. "
                "Uninstall and run: pip install paddlepaddle"
            )
    except ImportError:
        lines.append("  paddlepaddle runtime  : not importable (unusual — reinstall paddlepaddle)")