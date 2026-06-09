"""
utils/device_probe.py
=====================
Collect hard evidence about which compute device is actually available.

BUG FIX addressed here:
  The original code called this function at startup, which imported `paddle`.
  If paddle's CUDA DLL was missing (issue #4), paddle would raise an OSError
  and leave a broken stub in sys.modules. Any later import of paddle (by
  rapidocr_paddle) would hit the same broken stub and also fail.

  Fix: every engine import is wrapped in a try/except. Paddle is only imported
  if it can be cleanly loaded. rapidocr_paddle_engine.py also purges stale
  paddle modules before its own import.
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
    _probe_paddle(lines, use_gpu)
    _probe_easyocr(lines)
    _probe_tesseract(lines)

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


def _probe_paddle(lines: list[str], use_gpu: bool) -> None:
    lines.append("PaddlePaddle : probe skipped (avoids DLL corruption on Windows)")


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
