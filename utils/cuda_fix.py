"""
utils/cuda_fix.py
=================
Windows CUDA DLL path fix.

On Windows, PyTorch and other libraries look for CUDA DLLs in the system PATH.
When CUDA libraries are installed as pip packages (nvidia-cudnn-cu12, etc.)
their DLLs live inside the venv and are NOT on PATH by default.

This module adds them via os.add_dll_directory() (Python 3.8+), which works
for PyTorch and most libraries including EasyOCR, docTR, and Surya.

Import this module at the very top of ocr_tester.py, before any torch
imports, to ensure the DLL directories are registered as early as possible.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Sub-paths inside the nvidia pip package tree that hold CUDA DLLs
_CUDA_SUBPATHS = [
    "cudnn/bin",
    "cudnn_ops/bin",
    "cudnn_graph/bin",
    "cublas/bin",
    "cuda_runtime/bin",
    "cufft/bin",
    "curand/bin",
    "cusolver/bin",
    "cusparse/bin",
    "nvjitlink/bin",
    "nccl/bin",
]


def apply() -> None:
    """
    Locate the nvidia pip-package DLL directories and register them with
    the Windows DLL loader via os.add_dll_directory.

    Safe to call on non-Windows platforms — it's a no-op.
    """
    # Disable PaddlePaddle MKL-DNN/oneDNN — must be set before paddle C++ init.
    # Without this, PaddlePaddle 3.x crashes on Windows with:
    #   (Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
    #   [pir::ArrayAttribute<pir::DoubleAttribute>]
    import os
    os.environ["FLAGS_use_mkldnn"] = "0"

    if os.name != "nt":
        return   # Linux/macOS don't need this

    nvidia_base = _find_nvidia_base()
    if nvidia_base is None:
        return   # CUDA not installed as pip packages

    added: list[str] = []
    for subpath in _CUDA_SUBPATHS:
        dll_dir = nvidia_base / subpath.replace("/", os.sep)
        if dll_dir.exists():
            try:
                os.add_dll_directory(str(dll_dir))
                added.append(str(dll_dir))
            except OSError:
                pass   # non-fatal

    if added:
        print("[cuda_fix] Registered %d CUDA DLL director%s with the Windows loader." % (
            len(added), "y" if len(added) == 1 else "ies"
        ))


def _find_nvidia_base() -> Path | None:
    """Search sys.path for the nvidia pip-package base directory."""
    for p in sys.path:
        candidate = Path(p) / "nvidia"
        if candidate.is_dir():
            return candidate
    return None