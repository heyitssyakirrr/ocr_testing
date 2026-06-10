"""
utils/cuda_fix.py
=================
Windows CUDA DLL path fix + Paddle env-var guard.

On Windows, PyTorch and other libraries look for CUDA DLLs in the system PATH.
When CUDA libraries are installed as pip packages (nvidia-cudnn-cu12, etc.)
their DLLs live inside the venv and are NOT on PATH by default.

This module adds them via os.add_dll_directory() (Python 3.8+), which works
for PyTorch and most libraries including EasyOCR, docTR, and Surya.

IMPORTANT — PaddleOCR env vars
--------------------------------
The following flags MUST be set as OS environment variables BEFORE Python
starts — i.e. in run_ocr.bat — NOT via os.environ in Python code.

Paddle reads these at C++ runtime initialisation, which happens the moment
`import paddle` executes for the first time. Any os.environ assignment made
in Python code (including here) is already too late.

Required flags (set in run_ocr.bat):
    SET FLAGS_use_mkldnn=0
    SET FLAGS_enable_pir_api=0
    SET FLAGS_use_new_executor=0
    SET PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0

Without them, every ocr.predict() call raises:
    NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
    [pir::ArrayAttribute<pir::DoubleAttribute>]

Import this module at the very top of ocr_tester.py, before any torch/paddle
imports, to register DLL directories and emit an early warning if the required
Paddle flags are missing.
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

# Paddle flags that MUST be set before Python starts (via run_ocr.bat).
# These cannot be set via os.environ in Python — paddle reads them at
# C++ init which happens at the first `import paddle`.
_REQUIRED_PADDLE_FLAGS: dict[str, str] = {
    "FLAGS_use_mkldnn":                  "0",
    "FLAGS_enable_pir_api":              "0",
    "FLAGS_use_new_executor":            "0",
    "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT": "0",
}


def apply() -> None:
    """
    1. Warn loudly if required Paddle env vars were not set before Python started.
    2. Register CUDA pip-package DLL directories with the Windows loader.

    Safe to call on non-Windows platforms — step 2 is a no-op there.
    Call this at the very top of ocr_tester.py before any other imports.
    """
    _check_paddle_flags()

    if os.name != "nt":
        return   # Linux/macOS don't need DLL path registration

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
        print(
            "[cuda_fix] Registered %d CUDA DLL director%s with the Windows loader."
            % (len(added), "y" if len(added) == 1 else "ies")
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_paddle_flags() -> None:
    """
    Emit a clear warning if any required Paddle env var is missing or wrong.

    These flags must arrive via the OS environment (run_ocr.bat), not via
    os.environ in Python — paddle reads them at C++ init before any Python
    code runs after `import paddle`.
    """
    missing = [
        k for k, v in _REQUIRED_PADDLE_FLAGS.items()
        if os.environ.get(k) != v
    ]
    if not missing:
        return

    border = "!" * 72
    print(border)
    print("  [cuda_fix] WARNING — required Paddle env vars not set:")
    for k in missing:
        current = os.environ.get(k, "(not set)")
        print("    %-44s  current=%s  required=0" % (k, current))
    print()
    print("  These flags MUST be set BEFORE Python starts (in run_ocr.bat).")
    print("  Setting them via os.environ in Python is too late — paddle reads")
    print("  them at C++ runtime init on the first `import paddle`.")
    print()
    print("  PaddleOCR will crash with:")
    print("    NotImplementedError: ConvertPirAttribute2RuntimeAttribute")
    print("    not support [pir::ArrayAttribute<pir::DoubleAttribute>]")
    print()
    print("  Fix: use run_ocr.bat instead of `python ocr_tester.py` directly.")
    print("  run_ocr.bat must contain:")
    for k, v in _REQUIRED_PADDLE_FLAGS.items():
        print("    SET %s=%s" % (k, v))
    print(border)
    print()


def _find_nvidia_base() -> Path | None:
    """Search sys.path for the nvidia pip-package base directory."""
    for p in sys.path:
        candidate = Path(p) / "nvidia"
        if candidate.is_dir():
            return candidate
    return None