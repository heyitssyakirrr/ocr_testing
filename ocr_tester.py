"""
ocr_tester.py
=============
Test OCR engines on scanned PDF documents (bank letters, contracts, etc.).

Supported engines:
    paddle    — PaddleOCR (PP-OCRv5 or PP-OCRv4 fallback)
    doctr     — docTR (db_resnet50 + parseq)
    rapidocr  — RapidOCR (ONNX, CPU-only)
    easyocr   — EasyOCR (CRAFT detection + CRNN recognition)
    tesseract — Tesseract 5.x via pytesseract

Usage examples:
    python ocr_tester.py --folder input_files
    python ocr_tester.py --folder input_files --engines paddle easyocr tesseract
    python ocr_tester.py --folder input_files --engines paddle --dpi 400
    python ocr_tester.py --folder input_files --engines all --gpu
    python ocr_tester.py --folder input_files --engines paddle doctr --output-dir my_results

    # ONNX only (CPU)
    python ocr_tester.py --folder input_files --engines rapidocr

    # Paddle only (CPU or GPU)
    python ocr_tester.py --folder input_files --engines rapidocr_paddle
    python ocr_tester.py --folder input_files --engines rapidocr_paddle --gpu

    # Both side by side for comparison
    python ocr_tester.py --folder input_files --engines rapidocr rapidocr_paddle
    python ocr_tester.py --folder input_files --engines rapidocr rapidocr_paddle --gpu

    # Full suite including both
    python ocr_tester.py --folder input_files --engines all --gpu

==============================================================================
GPU REQUIREMENTS — READ THIS BEFORE USING --gpu
==============================================================================

GPU support requires an NVIDIA GPU with CUDA. AMD and Intel GPUs are NOT
supported by any of these engines.

WHY YOUR GPU SHOWS 0% IN TASK MANAGER:
  You have CPU-only builds installed. The --gpu flag in this script only
  REQUESTS GPU use — the installed library still decides. If your PyTorch or
  PaddlePaddle was installed without CUDA support, GPU will never be used no
  matter what flag you pass. The device evidence section in each output file
  will confirm what is actually running.

TO ENABLE GPU (NVIDIA only):

  Step 1 — Confirm your NVIDIA driver and CUDA version:
    nvidia-smi
    Look for "CUDA Version: X.X" in the top-right corner.

  Step 2 — Install CUDA-enabled PyTorch (for docTR and easyocr):
    # For CUDA 12.1:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    # For CUDA 11.8:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    # Check available versions: https://pytorch.org/get-started/locally/

  Step 3 — Install CUDA-enabled PaddlePaddle (for paddle):
    # For CUDA 12.3:
    pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
    # For CUDA 11.8:
    pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    # Check: https://www.paddlepaddle.org.cn/install/quick

  Step 4 — Verify GPU is detected:
    python -c "import torch; print(torch.cuda.is_available())"   # must print True
    python -c "import paddle; print(paddle.is_compiled_with_cuda())"  # must print True

  Notes:
    - RapidOCR (ONNX) has no GPU support. It always runs on CPU.
    - Tesseract has no GPU support. It always runs on CPU.
    - EasyOCR supports CUDA via PyTorch (same install as docTR).
    - PaddleOCR and docTR give the biggest GPU speedup (5-10x on large PDFs).
    - VRAM requirement: 2 GB minimum; 4 GB recommended for server models.

==============================================================================
ACCURACY METRICS — HOW TO READ THE REPORT
==============================================================================

IMPORTANT: Self-confidence scores from OCR engines are NOT real accuracy.
  A model can say "confidence: 0.99" and still output "BUKIL" instead of "BUKIT".
  Confidence measures how certain the model is about its OWN output — not
  whether that output matches the actual text on the page.

This script uses THREE complementary metrics instead:

  1. CONFIDENCE DISTRIBUTION
     The model's internal certainty. Useful for spotting which individual lines
     the model itself found hard. Low-confidence lines deserve human review.
     Do NOT treat the average as "accuracy percentage".

  2. CHARACTER ERROR RATE PROXY (CER Proxy)
     Compares OCR output against what the PDF's embedded text layer says
     (if it exists). For scanned/image PDFs with no text layer, this is skipped.
     Formula: edit_distance(ocr_text, pdf_text) / len(pdf_text)
     Lower = better. 0.0 = perfect. 0.05 = 5% of characters are wrong.

  3. KNOWN-FIELD SPOT CHECK
     Looks for specific values you KNOW must appear in the document
     (e.g. account numbers, FI codes, bank names). Reports which ones were
     found exactly, which were found with minor typos, and which were missed.
     This is the most meaningful metric for your use case.

==============================================================================
"""

# ---------------------------------------------------------------------------
# DLL PATH fix — must run before any CUDA library is imported
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

def _fix_cuda_dll_paths() -> None:
    """
    On Windows, PaddlePaddle and PyTorch look for CUDA DLLs in system PATH.
    When CUDA libraries are installed as pip packages (nvidia-cudnn-cu12 etc.)
    their DLLs live inside the venv and are NOT on PATH by default.
    This function adds them automatically.
    """
    # Find the site-packages/nvidia folder relative to this venv
    nvidia_base = None
    for p in sys.path:
        candidate = Path(p) / "nvidia"
        if candidate.exists():
            nvidia_base = candidate
            break

    if nvidia_base is None:
        return  # not a pip-installed CUDA setup, skip

    dll_dirs = [
        nvidia_base / "cudnn"        / "bin",
        nvidia_base / "cublas"       / "bin",
        nvidia_base / "cuda_runtime" / "bin",
        nvidia_base / "cufft"        / "bin",
        nvidia_base / "curand"       / "bin",
        nvidia_base / "cusolver"     / "bin",
        nvidia_base / "cusparse"     / "bin",
        nvidia_base / "nvjitlink"    / "bin",
    ]

    added = []
    for d in dll_dirs:
        if d.exists():
            # os.add_dll_directory is the modern Windows way (Python 3.8+)
            # It tells the DLL loader to search this folder
            os.add_dll_directory(str(d))
            added.append(str(d))

    if added:
        print("[INFO] Added %d CUDA DLL directories to loader path." % len(added))

_fix_cuda_dll_paths()

import argparse
import difflib
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Result data structure
# ---------------------------------------------------------------------------

class OCRResult(NamedTuple):
    """Holds everything produced by one engine run on one file."""
    engine_key:   str
    engine_name:  str
    pdf_name:     str
    text:         str          # raw OCR text with (conf:X.XXX) suffixes
    elapsed:      float        # seconds
    skipped:      bool         # True if engine could not run
    skip_reason:  str          # non-empty when skipped=True


# ---------------------------------------------------------------------------
# GPU / CPU setup
# ---------------------------------------------------------------------------

def force_cpu() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"]  = ""
    os.environ["PADDLE_USE_GPU"]        = "0"
    os.environ["FLAGS_use_cuda"]        = "0"
    os.environ["FLAGS_use_mkldnn"]      = "0"
    os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
    print("[INFO] CPU-only mode — GPU disabled.")
    print()


def get_device_evidence(use_gpu: bool) -> list[str]:
    """Collect hard evidence about which device is actually being used."""
    lines = []
    lines.append("--- DEVICE / GPU EVIDENCE ---")
    lines.append("Platform             : %s %s" % (platform.system(), platform.release()))
    lines.append("Python               : %s" % platform.python_version())
    lines.append("--gpu flag           : %s" % use_gpu)
    lines.append("CUDA_VISIBLE_DEVICES : %s" % os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        lines.append("PyTorch version           : %s" % torch.__version__)
        lines.append("torch.cuda.is_available() : %s" % cuda_ok)
        if cuda_ok:
            lines.append("  CUDA device : %s" % torch.cuda.get_device_name(0))
            lines.append("  VRAM total  : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
        active = "cuda" if (use_gpu and cuda_ok) else "cpu"
        lines.append("Active torch device       : %s" % active)
        if use_gpu and not cuda_ok:
            lines.append("  *** WARNING: --gpu requested but CUDA not available.")
            lines.append("  *** Install a CUDA-enabled PyTorch — see script header for instructions.")
    except ImportError:
        lines.append("PyTorch : not installed")

    try:
        import paddle
        paddle_cuda = paddle.is_compiled_with_cuda()
        lines.append("PaddlePaddle version           : %s" % paddle.__version__)
        lines.append("paddle.is_compiled_with_cuda() : %s" % paddle_cuda)
        lines.append("paddle.device.get_device()     : %s" % paddle.device.get_device())
        if use_gpu and not paddle_cuda:
            lines.append("  *** WARNING: --gpu requested but PaddlePaddle has no CUDA support.")
            lines.append("  *** Install paddlepaddle-gpu — see script header for instructions.")
    except ImportError:
        lines.append("PaddlePaddle : not installed")
    except Exception as exc:
        lines.append("PaddlePaddle device check error: %s" % exc)

    try:
        import easyocr  # noqa: F401
        import torch as _t
        lines.append("EasyOCR : installed (uses torch CUDA if available)")
    except ImportError:
        lines.append("EasyOCR : not installed")

    try:
        import pytesseract
        tess_ver = pytesseract.get_tesseract_version()
        lines.append("Tesseract version : %s (CPU only)" % tess_ver)
    except ImportError:
        lines.append("Tesseract : pytesseract not installed")
    except Exception:
        lines.append("Tesseract : installed but tesseract binary not found in PATH")

    return lines


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[tuple[int, object]]:
    """Render every page of a PDF to a PIL Image at the given DPI."""
    try:
        import fitz
    except ImportError:
        sys.exit("[ERROR] PyMuPDF not installed.  pip install PyMuPDF")
    try:
        from PIL import Image
    except ImportError:
        sys.exit("[ERROR] Pillow not installed.  pip install Pillow")
    import io

    doc  = fitz.open(pdf_path)
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    imgs = []
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        imgs.append((page_num + 1, img))
        print("  [PDF] Page %d rendered at %d DPI: %dx%d px" % (
            page_num + 1, dpi, img.size[0], img.size[1]))
    doc.close()
    return imgs


def extract_pdf_text_layer(pdf_path: str) -> str:
    """
    Try to extract the embedded text layer from the PDF (if any).
    Returns empty string for image-only / scanned PDFs.
    """
    try:
        import fitz
    except ImportError:
        return ""
    doc  = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc).strip()
    doc.close()
    return text


# ---------------------------------------------------------------------------
# OCR engine implementations
# ---------------------------------------------------------------------------
# Convention for every run_<name> function:
#   Args   : images [(page_num, PIL.Image)], use_gpu bool
#   Return : (text: str, elapsed: float)
#            text has lines ending with "  (conf:X.XXX)"
#            return ("[SKIP] reason", 0.0) if engine cannot run
# ---------------------------------------------------------------------------

def run_paddle(images: list, use_gpu: bool = False) -> tuple[str, float]:
    """
    PaddleOCR — PP-OCRv5 server models (3.x) or PP-OCRv4 fallback (2.x).
    GPU: requires paddlepaddle-gpu (see script header).
    """
    try:
        import numpy as np
        from paddleocr import PaddleOCR
        import paddleocr as _poc
        paddle_ver = tuple(int(x) for x in _poc.__version__.split(".")[:2])
    except ImportError:
        return "[SKIP] PaddleOCR not installed.  pip install paddlepaddle paddleocr", 0.0

    os.environ["FLAGS_use_mkldnn"]      = "0"
    os.environ["PADDLE_DISABLE_ONEDNN"] = "1"

    all_lines: list[str] = []
    t0 = time.time()

    if paddle_ver >= (3, 0):
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
                    rec_texts  = res.get("rec_texts",  []) if isinstance(res, dict) else getattr(res, "rec_texts",  [])
                    rec_scores = res.get("rec_scores", []) if isinstance(res, dict) else getattr(res, "rec_scores", [])
                    for text, conf in zip(rec_texts, rec_scores):
                        if str(text).strip():
                            all_lines.append("%s  (conf:%.3f)" % (text, float(conf)))
            except Exception as exc:
                print("    [PaddleOCR] Page %d error: %s" % (page_num, exc))

    else:
        print("  [PaddleOCR] Loading PP-OCRv4 models (2.x fallback)...")
        print("  [PaddleOCR] Upgrade to paddlepaddle>=3.0 + paddleocr>=3.0 for PP-OCRv5.")
        try:
            ocr = PaddleOCR(
                lang="en", use_gpu=use_gpu, use_angle_cls=True,
                enable_mkldnn=False, det_db_thresh=0.3,
                det_db_box_thresh=0.5, det_db_unclip_ratio=2.0,
                rec_batch_num=1, show_log=False,
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


def run_doctr(images: list, use_gpu: bool = False) -> tuple[str, float]:
    """
    docTR — db_resnet50 detection + parseq recognition (highest-accuracy arch).
    GPU: requires torch with CUDA (see script header).
    """
    try:
        import torch
        import numpy as np
        from doctr.models import ocr_predictor
    except ImportError:
        return "[SKIP] python-doctr not installed.  pip install python-doctr", 0.0

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    print("  [docTR] Loading model on %s..." % device)
    try:
        model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="parseq",
            pretrained=True,
            assume_straight_pages=False,
        ).to(device)
    except Exception as exc:
        return "[SKIP] docTR model load failed: %s" % exc, 0.0

    all_lines: list[str] = []
    t0 = time.time()
    for page_num, img in images:
        print("  [docTR] Processing page %d..." % page_num)
        img_rgb    = np.array(img.convert("RGB"))
        img_tensor = (img_rgb.astype("float32") / 255.0).transpose(2, 0, 1)
        try:
            result = model([img_tensor])
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        words     = [w.value      for w in line.words if w.value.strip()]
                        confs     = [w.confidence  for w in line.words if w.value.strip()]
                        if words:
                            avg_conf = sum(confs) / len(confs)
                            all_lines.append("%s  (conf:%.3f)" % (" ".join(words), avg_conf))
        except Exception as exc:
            print("    [docTR] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0


def run_rapidocr(images: list, use_gpu: bool = False) -> tuple[str, float]:
    """
    RapidOCR — ONNX runtime, CPU only. Fast, no GPU support.
    """
    try:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return "[SKIP] RapidOCR not installed.  pip install rapidocr-onnxruntime", 0.0

    if use_gpu:
        print("  [RapidOCR] NOTE: RapidOCR has no GPU support — running on CPU.")

    print("  [RapidOCR] Loading model...")
    ocr = RapidOCR()

    all_lines: list[str] = []
    t0 = time.time()
    for page_num, img in images:
        print("  [RapidOCR] Processing page %d..." % page_num)
        import numpy as np
        img_array = np.array(img.convert("RGB"))
        try:
            result, _ = ocr(img_array)
            if result is None:
                continue
            for item in result:
                if len(item) >= 3:
                    text, conf = item[1], item[2]
                    if str(text).strip():
                        all_lines.append("%s  (conf:%.3f)" % (text, conf))
        except Exception as exc:
            print("    [RapidOCR] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0

def run_rapidocr_paddle(images: list, use_gpu: bool = False) -> tuple[str, float]:
    import sys

    # Clear any cached partial paddle imports
    for mod in list(sys.modules.keys()):
        if mod.startswith("paddle"):
            del sys.modules[mod]

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
    try:
        ocr = RapidOCR(
            det_use_cuda=use_gpu,
            cls_use_cuda=use_gpu,
            rec_use_cuda=use_gpu,
        )
    except TypeError:
        # Older rapidocr-paddle versions don't accept cuda kwargs
        print("  [RapidOCR-Paddle] Note: GPU kwargs not supported, loading default...")
        try:
            ocr = RapidOCR()
        except Exception as exc:
            return "[SKIP] RapidOCR-Paddle model load failed: %s" % exc, 0.0
    except Exception as exc:
        return "[SKIP] RapidOCR-Paddle model load failed: %s" % exc, 0.0

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

def run_easyocr(images: list, use_gpu: bool = False) -> tuple[str, float]:
    """
    EasyOCR — CRAFT text detection + CRNN recognition.
    Good balance of accuracy and ease of setup. GPU via PyTorch CUDA.

    Install: pip install easyocr
    GPU    : needs torch with CUDA (same install as docTR).
    """
    try:
        import numpy as np
        import easyocr
    except ImportError:
        return "[SKIP] EasyOCR not installed.  pip install easyocr", 0.0

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False

    effective_gpu = use_gpu and cuda_ok
    if use_gpu and not cuda_ok:
        print("  [EasyOCR] WARNING: CUDA not available — falling back to CPU.")

    print("  [EasyOCR] Loading model (gpu=%s)..." % effective_gpu)
    try:
        reader = easyocr.Reader(["en"], gpu=effective_gpu, verbose=False)
    except Exception as exc:
        return "[SKIP] EasyOCR model load failed: %s" % exc, 0.0

    all_lines: list[str] = []
    t0 = time.time()
    for page_num, img in images:
        print("  [EasyOCR] Processing page %d..." % page_num)
        img_array = np.array(img.convert("RGB"))
        try:
            results = reader.readtext(img_array, detail=1, paragraph=False)
            for (_bbox, text, conf) in results:
                if str(text).strip():
                    all_lines.append("%s  (conf:%.3f)" % (text, float(conf)))
        except Exception as exc:
            print("    [EasyOCR] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0


def run_tesseract(images: list, use_gpu: bool = False) -> tuple[str, float]:
    """
    Tesseract 5.x via pytesseract — CPU only, no GPU support.
    Requires tesseract binary in PATH.

    Install binary : https://github.com/UB-Mannheim/tesseract/wiki  (Windows)
    Install wrapper: pip install pytesseract

    Config used:
      --oem 3   — use LSTM neural net engine (most accurate)
      --psm 6   — assume a uniform block of text
      -l eng    — English language model
    """
    try:
        import pytesseract
        from PIL import Image as PilImage
    except ImportError:
        return "[SKIP] pytesseract not installed.  pip install pytesseract", 0.0

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return (
            "[SKIP] Tesseract binary not found. "
            "Install from https://github.com/UB-Mannheim/tesseract/wiki "
            "and add to PATH.", 0.0
        )

    if use_gpu:
        print("  [Tesseract] NOTE: Tesseract has no GPU support — running on CPU.")

    custom_config = r"--oem 3 --psm 6 -l eng"
    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img in images:
        print("  [Tesseract] Processing page %d..." % page_num)
        try:
            # Get word-level data with confidence scores
            data = pytesseract.image_to_data(
                img.convert("RGB"),
                config=custom_config,
                output_type=pytesseract.Output.DICT,
            )
            # Group words into lines using line_num
            line_words: dict[tuple, list] = {}
            for i, word in enumerate(data["text"]):
                word = str(word).strip()
                if not word:
                    continue
                conf = int(data["conf"][i])
                if conf < 0:        # -1 means no confidence data (non-text block)
                    continue
                line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                line_words.setdefault(line_key, []).append((word, conf / 100.0))

            for line_key in sorted(line_words.keys()):
                words_confs = line_words[line_key]
                line_text   = " ".join(w for w, _ in words_confs)
                avg_conf    = sum(c for _, c in words_confs) / len(words_confs)
                all_lines.append("%s  (conf:%.3f)" % (line_text, avg_conf))

        except Exception as exc:
            print("    [Tesseract] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0


# ---------------------------------------------------------------------------
# Engine registry
# To add a new engine: implement run_<name>() above and add an entry here.
# ---------------------------------------------------------------------------

ENGINE_REGISTRY: dict[str, tuple[str, callable]] = {
    "paddle":           ("PaddleOCR",  run_paddle),
    "doctr":            ("docTR",      run_doctr),
    "rapidocr":         ("RapidOCR-ONNX",   run_rapidocr),
    "rapidocr_paddle":  ("RapidOCR-Paddle", run_rapidocr_paddle),
    "easyocr":          ("EasyOCR",    run_easyocr),
    "tesseract":        ("Tesseract",  run_tesseract),
}

# Engines that support GPU acceleration
GPU_CAPABLE_ENGINES = {"paddle", "doctr", "easyocr", "rapidocr_paddle"}


# ---------------------------------------------------------------------------
# Metric 1: Confidence distribution
# ---------------------------------------------------------------------------

def confidence_stats(ocr_text: str) -> dict | None:
    """
    Extract model self-confidence scores from lines ending with (conf:X.XXX).

    IMPORTANT: This is NOT accuracy. It is the model's own certainty about
    its output. A high score means the model is confident — not that it is
    correct. Use this to identify which individual lines need review, not
    to judge the engine's overall quality.
    """
    confs = []
    for line in ocr_text.split("\n"):
        if "(conf:" in line:
            try:
                val = float(line.split("(conf:")[1].rstrip(")"))
                confs.append(val)
            except ValueError:
                pass
    if not confs:
        return None

    total  = len(confs)
    avg    = sum(confs) / total
    high   = sum(1 for c in confs if c >= 0.90)
    medium = sum(1 for c in confs if 0.70 <= c < 0.90)
    low    = sum(1 for c in confs if c < 0.70)

    return {
        "count":         total,
        "avg_conf":      avg,
        "min_conf":      min(confs),
        "max_conf":      max(confs),
        "high":          high,
        "medium":        medium,
        "low":           low,
        "low_lines":     [
            line.split("  (conf:")[0].strip()
            for line in ocr_text.split("\n")
            if "(conf:" in line and float(line.split("(conf:")[1].rstrip(")")) < 0.70
        ],
    }


# ---------------------------------------------------------------------------
# Metric 2: Character Error Rate proxy against PDF text layer
# ---------------------------------------------------------------------------

def cer_against_pdf_layer(ocr_text: str, pdf_text_layer: str) -> dict | None:
    """
    Compare OCR output against the PDF's embedded text layer.
    Only meaningful when the PDF has a real text layer (not scanned image-only).

    CER = edit_distance / len(reference)
    Lower is better. 0.0 = perfect match.
    """
    if not pdf_text_layer.strip():
        return None

    # Strip confidence annotations from OCR lines
    clean_ocr = "\n".join(
        line.split("  (conf:")[0].strip()
        for line in ocr_text.split("\n")
        if line.strip() and not line.startswith("[SKIP]")
    )

    # Normalise whitespace for fair comparison
    ref = re.sub(r"\s+", " ", pdf_text_layer).strip().lower()
    hyp = re.sub(r"\s+", " ", clean_ocr).strip().lower()

    # Levenshtein distance at character level using difflib
    matcher = difflib.SequenceMatcher(None, ref, hyp)
    edit_distance = sum(
        max(block.size, block.size)
        for block in matcher.get_opcodes()
        if block[0] != "equal"
    )
    # More accurate: count actual char operations
    ops = matcher.get_opcodes()
    n_errors = sum(
        max(b - a, d - c)
        for tag, a, b, c, d in ops
        if tag != "equal"
    )
    cer = n_errors / max(len(ref), 1)

    return {
        "ref_chars":     len(ref),
        "hyp_chars":     len(hyp),
        "edit_distance": n_errors,
        "cer":           cer,
        "cer_pct":       cer * 100,
    }


# ---------------------------------------------------------------------------
# Metric 3: Known-field spot check
# ---------------------------------------------------------------------------

class FieldCheckResult(NamedTuple):
    field:    str
    expected: str
    status:   str    # EXACT | FUZZY | MISSING
    found_as: str    # what the OCR actually produced (empty if MISSING)
    score:    float  # similarity 0.0-1.0


def check_known_fields(
    ocr_text: str,
    known_fields: dict[str, str],
) -> list[FieldCheckResult]:
    """
    Check whether known values (account numbers, FI codes, bank names, etc.)
    appear correctly in the OCR output.

    known_fields: {"FI Code": "034707062", "Master Account": "0000006072902892170888001", ...}

    Matching logic:
      EXACT  — value found verbatim (after stripping spaces)
      FUZZY  — similarity >= 0.80 (catches minor typos like 0/O swaps)
      MISSING — similarity < 0.80
    """
    # Strip conf annotations, join all text
    clean_lines = [
        line.split("  (conf:")[0].strip()
        for line in ocr_text.split("\n")
        if line.strip()
    ]
    full_text = " ".join(clean_lines)
    full_text_nospace = re.sub(r"\s+", "", full_text)

    results = []
    for field_name, expected in known_fields.items():
        expected_nospace = re.sub(r"\s+", "", expected)

        # 1. Exact match (ignore surrounding spaces)
        if expected_nospace in full_text_nospace:
            results.append(FieldCheckResult(
                field=field_name, expected=expected,
                status="EXACT", found_as=expected, score=1.0,
            ))
            continue

        # 2. Fuzzy match — find the closest substring of the same length
        best_score   = 0.0
        best_found   = ""
        window       = len(expected_nospace)
        # Slide a window over no-space text
        for i in range(max(1, len(full_text_nospace) - window + 1)):
            candidate = full_text_nospace[i : i + window]
            score     = difflib.SequenceMatcher(None, expected_nospace.lower(), candidate.lower()).ratio()
            if score > best_score:
                best_score = score
                best_found = candidate

        if best_score >= 0.80:
            results.append(FieldCheckResult(
                field=field_name, expected=expected,
                status="FUZZY", found_as=best_found, score=best_score,
            ))
        else:
            results.append(FieldCheckResult(
                field=field_name, expected=expected,
                status="MISSING", found_as="", score=best_score,
            ))

    return results


# ---------------------------------------------------------------------------
# Flagging — targeted, not everything
# ---------------------------------------------------------------------------

# Patterns that are specifically critical for this bank document use case
_CRITICAL_PATTERNS = [
    # HSBC-style account suffixes where O/0 confusion matters
    (r"\b\d{9}[O0]/[D][0-9]{2}\b",          "HSBC O/D suffix — check letter O vs digit 0"),
    # Account numbers that should be pure digits but contain letters
    (r"\b\d*[A-NP-Z]\d{3,}\b",              "Account number may contain OCR letter/digit confusion"),
    # FI codes (should be 7-9 pure digits)
    (r"\b0[0-9]{1,2}[A-Z][0-9]{4,}\b",      "FI code contains unexpected letter"),
    # Common confused characters in numeric sequences
    (r"\b[0-9]+[lI][0-9]+\b",               "Possible 1/l/I confusion in number"),
    (r"\b[0-9]+[oO][0-9]+\b",               "Possible 0/O confusion in number"),
    (r"\b[0-9]+[sS][0-9]+\b",               "Possible 5/S confusion in number"),
    (r"\b[0-9]+[bB][0-9]+\b",               "Possible 6/b confusion in number"),
]

def flag_critical_lines(text: str) -> list[str]:
    """
    Flag only lines that contain genuinely suspicious patterns relevant
    to bank document field extraction. Much more targeted than the previous
    approach of flagging any line containing common letters.
    """
    flagged = []
    for line in text.split("\n"):
        clean = line.split("  (conf:")[0].strip()
        if not clean:
            continue
        for pattern, reason in _CRITICAL_PATTERNS:
            if re.search(pattern, clean, re.IGNORECASE):
                flagged.append("  [!] %s\n      Line: %s" % (reason, line))
                break  # one flag per line is enough
    return flagged


# ---------------------------------------------------------------------------
# Cross-engine comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(results: list[OCRResult]) -> list[str]:
    """
    Build a side-by-side summary table comparing all engines run on the same file.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-ENGINE COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    header = "%-12s  %8s  %8s  %5s  %6s  %6s  %6s" % (
        "Engine", "Time(s)", "Lines", "Skip?", "HIGH%", "MED%", "LOW%"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        if r.skipped:
            lines.append("%-12s  %8s  %8s  %5s" % (r.engine_key, "—", "—", "YES"))
            continue
        text_lines = [l for l in r.text.split("\n") if l.strip()]
        cs         = confidence_stats(r.text)
        n          = len(text_lines)
        if cs and cs["count"]:
            high_pct = cs["high"]   / cs["count"] * 100
            med_pct  = cs["medium"] / cs["count"] * 100
            low_pct  = cs["low"]    / cs["count"] * 100
        else:
            high_pct = med_pct = low_pct = 0.0
        lines.append("%-12s  %8.1f  %8d  %5s  %6.1f  %6.1f  %6.1f" % (
            r.engine_key, r.elapsed, n, "no",
            high_pct, med_pct, low_pct,
        ))

    lines.append("")
    lines.append("NOTE: HIGH/MED/LOW are confidence DISTRIBUTION bands, not accuracy bands.")
    lines.append("      Use the Known-Field Spot Check results to judge real-world accuracy.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_output(
    result: OCRResult,
    device_evidence: list[str],
    dpi: int,
    output_dir: Path,
    pdf_text_layer: str,
    known_fields: dict[str, str],
) -> Path:
    """Write the full analysis report for one engine + one file."""

    out_filename = "%s_%s_output.txt" % (result.engine_key, Path(result.pdf_name).stem.lower().replace(" ", "_"))
    out_file     = output_dir / out_filename

    lines = result.text.split("\n") if not result.skipped else []
    non_empty_lines = [l for l in lines if l.strip()]

    with open(out_file, "w", encoding="utf-8") as f:

        # ---- Header ----
        f.write("=" * 80 + "\n")
        f.write("FILE   : %s\n" % result.pdf_name)
        f.write("ENGINE : %s\n" % result.engine_name)
        f.write("DPI    : %d\n" % dpi)
        f.write("TIME   : %.1f seconds\n" % result.elapsed)
        f.write("LINES  : %d\n" % len(non_empty_lines))
        if result.skipped:
            f.write("STATUS : SKIPPED — %s\n" % result.skip_reason)
        f.write("=" * 80 + "\n\n")

        if result.skipped:
            f.write("Engine could not run: %s\n" % result.skip_reason)
            return out_file

        # ---- Device evidence ----
        f.write("\n".join(device_evidence) + "\n\n")

        # ================================================================
        # METRIC 1 — Confidence distribution
        # ================================================================
        f.write("=" * 80 + "\n")
        f.write("METRIC 1 — CONFIDENCE DISTRIBUTION\n")
        f.write("=" * 80 + "\n")
        f.write("\n")
        f.write("IMPORTANT: This is NOT accuracy. It is the model's self-reported\n")
        f.write("certainty about its own output. A high score does not mean the\n")
        f.write("text was read correctly — the model can be confidently wrong.\n")
        f.write("Use this to find individual lines that need manual review.\n")
        f.write("\n")

        cs = confidence_stats(result.text)
        if cs:
            f.write("  Total lines scored : %d\n"   % cs["count"])
            f.write("  Average confidence : %.4f\n" % cs["avg_conf"])
            f.write("  Min / Max          : %.4f / %.4f\n" % (cs["min_conf"], cs["max_conf"]))
            f.write("\n")
            f.write("  Band breakdown (for individual line review):\n")
            f.write("    HIGH   (>= 0.90) : %d lines  (%.1f%%)  — model is fairly certain\n" % (
                cs["high"],   cs["high"]   / cs["count"] * 100))
            f.write("    MEDIUM (0.70-0.89): %d lines  (%.1f%%)  — model had some difficulty\n" % (
                cs["medium"], cs["medium"] / cs["count"] * 100))
            f.write("    LOW    (< 0.70)  : %d lines  (%.1f%%)  — review these manually\n" % (
                cs["low"],    cs["low"]    / cs["count"] * 100))
            if cs["low_lines"]:
                f.write("\n  Low-confidence lines (the ones to check first):\n")
                for ll in cs["low_lines"]:
                    f.write("    - %s\n" % ll)
        else:
            f.write("  No confidence scores available for this engine.\n")
        f.write("\n")

        # ================================================================
        # METRIC 2 — CER against PDF text layer
        # ================================================================
        f.write("=" * 80 + "\n")
        f.write("METRIC 2 — CHARACTER ERROR RATE vs PDF TEXT LAYER\n")
        f.write("=" * 80 + "\n\n")

        cer_data = cer_against_pdf_layer(result.text, pdf_text_layer)
        if cer_data is None:
            f.write("  SKIPPED — this PDF has no embedded text layer (image-only / scanned).\n")
            f.write("  This is expected for documents scanned with CamScanner.\n")
            f.write("  CER can only be measured when the PDF contains selectable text.\n")
        else:
            f.write("  Reference chars (PDF layer) : %d\n" % cer_data["ref_chars"])
            f.write("  OCR output chars            : %d\n" % cer_data["hyp_chars"])
            f.write("  Character edit distance     : %d\n" % cer_data["edit_distance"])
            f.write("  CER                         : %.4f  (%.2f%%)\n" % (
                cer_data["cer"], cer_data["cer_pct"]))
            f.write("\n")
            f.write("  Interpretation:\n")
            f.write("    < 2%%   Excellent — nearly perfect text extraction\n")
            f.write("    2-5%%   Good      — minor errors, spot-check critical fields\n")
            f.write("    5-10%%  Fair      — noticeable errors, review key fields\n")
            f.write("    > 10%%  Poor      — significant errors, try higher DPI\n")
        f.write("\n")

        # ================================================================
        # METRIC 3 — Known-field spot check
        # ================================================================
        f.write("=" * 80 + "\n")
        f.write("METRIC 3 — KNOWN-FIELD SPOT CHECK\n")
        f.write("=" * 80 + "\n\n")

        if not known_fields:
            f.write("  No known fields provided.\n")
            f.write("  To use this metric, add a 'known_fields.txt' file in the same\n")
            f.write("  folder as your PDF with lines like:\n")
            f.write("    FI Code : 034707062\n")
            f.write("    Master Account : 0000006072902892170888001\n")
            f.write("    Sub Account : 0000000406070003660\n")
            f.write("    Bank Name : AFFIN BANK BERHAD\n")
        else:
            field_results = check_known_fields(result.text, known_fields)
            exact   = [r for r in field_results if r.status == "EXACT"]
            fuzzy   = [r for r in field_results if r.status == "FUZZY"]
            missing = [r for r in field_results if r.status == "MISSING"]

            f.write("  Summary: %d EXACT  |  %d FUZZY  |  %d MISSING  (of %d fields)\n" % (
                len(exact), len(fuzzy), len(missing), len(field_results)))
            f.write("\n")

            f.write("  EXACT matches (read correctly):\n")
            if exact:
                for r in exact:
                    f.write("    [OK] %-30s : %s\n" % (r.field, r.expected))
            else:
                f.write("    (none)\n")
            f.write("\n")

            f.write("  FUZZY matches (present but with typos — similarity %.0f%%+):\n" % 80)
            if fuzzy:
                for r in fuzzy:
                    f.write("    [~] %-30s\n" % r.field)
                    f.write("        Expected : %s\n" % r.expected)
                    f.write("        Found as : %s  (similarity: %.1f%%)\n" % (
                        r.found_as, r.score * 100))
            else:
                f.write("    (none)\n")
            f.write("\n")

            f.write("  MISSING (not found — score below 80%%):\n")
            if missing:
                for r in missing:
                    f.write("    [X] %-30s : %s  (best match score: %.1f%%)\n" % (
                        r.field, r.expected, r.score * 100))
            else:
                f.write("    (none — all fields found)\n")
        f.write("\n")

        # ================================================================
        # Full text
        # ================================================================
        f.write("=" * 80 + "\n")
        f.write("FULL OCR TEXT\n")
        f.write("=" * 80 + "\n\n")
        for line in non_empty_lines:
            f.write(line + "\n")
        f.write("\n")

        # ================================================================
        # Targeted flagging
        # ================================================================
        f.write("=" * 80 + "\n")
        f.write("CRITICAL LINES — POSSIBLE CHARACTER CONFUSION\n")
        f.write("(Only lines with patterns that matter for field extraction)\n")
        f.write("=" * 80 + "\n\n")
        flagged = flag_critical_lines(result.text)
        if flagged:
            for flag in flagged:
                f.write(flag + "\n")
        else:
            f.write("No suspicious patterns detected.\n")
        f.write("\n")

    return out_file


# ---------------------------------------------------------------------------
# Known-fields loader
# ---------------------------------------------------------------------------

def load_known_fields(pdf_path: Path) -> dict[str, str]:
    """
    Look for a 'known_fields.txt' file alongside the PDF.
    Format (one field per line):
        FI Code : 034707062
        Master Account : 0000006072902892170888001

    Returns empty dict if file not found.
    """
    kf_path = pdf_path.parent / "known_fields.txt"
    if not kf_path.exists():
        return {}
    fields = {}
    with kf_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                fields[key.strip()] = val.strip()
    return fields


# ---------------------------------------------------------------------------
# File processing orchestrator
# ---------------------------------------------------------------------------

def process_file(
    pdf_path:       Path,
    engine_keys:    list[str],
    use_gpu:        bool,
    dpi:            int,
    output_dir:     Path,
    device_evidence: list[str],
) -> None:
    print("\n" + "=" * 80)
    print("  FILE : %s" % pdf_path.name)
    print("=" * 80)

    # Render PDF pages once — shared by all engines
    print("\n  Rendering PDF at %d DPI..." % dpi)
    images = pdf_to_images(str(pdf_path), dpi=dpi)

    # Try to extract PDF text layer for CER metric
    pdf_text_layer = extract_pdf_text_layer(str(pdf_path))
    if pdf_text_layer.strip():
        print("  [INFO] PDF has embedded text layer (%d chars) — CER metric available." % len(pdf_text_layer))
    else:
        print("  [INFO] PDF has no text layer (scanned image) — CER metric will be skipped.")

    # Load known fields for spot-check metric
    known_fields = load_known_fields(pdf_path)
    if known_fields:
        print("  [INFO] Loaded %d known fields from known_fields.txt" % len(known_fields))
    else:
        print("  [INFO] No known_fields.txt found — spot-check metric will be skipped.")
        print("         Create %s to enable it." % (pdf_path.parent / "known_fields.txt"))

    # Run each engine
    all_results: list[OCRResult] = []

    for engine_key in engine_keys:
        engine_name, engine_fn = ENGINE_REGISTRY[engine_key]
        print("\n  Running %s..." % engine_name)
        if use_gpu and engine_key not in GPU_CAPABLE_ENGINES:
            print("  [INFO] %s has no GPU support — will run on CPU." % engine_name)

        try:
            text, elapsed = engine_fn(images, use_gpu=use_gpu)
        except Exception as exc:
            text, elapsed = "[SKIP] Unexpected error: %s" % exc, 0.0

        skipped = text.startswith("[SKIP]")
        result  = OCRResult(
            engine_key=engine_key,
            engine_name=engine_name,
            pdf_name=pdf_path.name,
            text=text if not skipped else "",
            elapsed=elapsed,
            skipped=skipped,
            skip_reason=text[7:] if skipped else "",
        )
        all_results.append(result)

        if skipped:
            print("  SKIPPED: %s" % result.skip_reason)
            continue

        out_file = write_output(
            result, device_evidence, dpi,
            output_dir, pdf_text_layer, known_fields,
        )

        n_lines  = len([l for l in text.split("\n") if l.strip()])
        n_flagged = len(flag_critical_lines(text))
        cs       = confidence_stats(text)
        print("  Done in %.1fs — %d lines, %d critically flagged" % (
            elapsed, n_lines, n_flagged))
        if cs:
            print("  Confidence avg/min: %.3f / %.3f  |  LOW lines: %d" % (
                cs["avg_conf"], cs["min_conf"], cs["low"]))
        print("  Saved: %s" % out_file)

    # Cross-engine comparison (only useful if more than one engine ran)
    completed = [r for r in all_results if not r.skipped]
    if len(completed) > 1:
        comp_file = output_dir / ("comparison_%s.txt" % pdf_path.stem.lower().replace(" ", "_"))
        cmp_lines = build_comparison_table(completed)
        # Append known-field comparison per engine
        if known_fields:
            cmp_lines.append("KNOWN-FIELD SPOT CHECK SUMMARY")
            cmp_lines.append("-" * 60)
            for r in completed:
                fr = check_known_fields(r.text, known_fields)
                exact   = sum(1 for x in fr if x.status == "EXACT")
                fuzzy   = sum(1 for x in fr if x.status == "FUZZY")
                missing = sum(1 for x in fr if x.status == "MISSING")
                cmp_lines.append("%-12s  EXACT:%d  FUZZY:%d  MISSING:%d" % (
                    r.engine_key, exact, fuzzy, missing))
            cmp_lines.append("")
        with open(comp_file, "w", encoding="utf-8") as f:
            f.write("\n".join(cmp_lines))
        print("\n  Cross-engine comparison saved: %s" % comp_file)


# ---------------------------------------------------------------------------
# File selection helpers
# ---------------------------------------------------------------------------

def list_pdfs(folder: str) -> list[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        sys.exit("[ERROR] Folder not found: %s" % folder)
    pdfs = sorted(folder_path.glob("*.pdf"))
    if not pdfs:
        sys.exit("[ERROR] No PDF files found in: %s" % folder)
    return pdfs


def pick_files(pdfs: list[Path]) -> list[Path]:
    print("\nPDF files found:")
    for i, p in enumerate(pdfs):
        print("  [%d] %s" % (i + 1, p.name))
    print("  [A] All files\n")
    choice = input("Select file(s) — number(s) comma-separated, or A for all: ").strip()

    if choice.upper() == "A":
        return pdfs

    selected = []
    for part in choice.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(pdfs):
                selected.append(pdfs[idx])
            else:
                print("  [WARN] Invalid number: %s (skipped)" % part)
        else:
            print("  [WARN] Invalid input: %s (skipped)" % part)

    if not selected:
        sys.exit("[ERROR] No valid files selected.")
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    engine_choices = list(ENGINE_REGISTRY.keys()) + ["all"]

    parser = argparse.ArgumentParser(
        description="Test OCR engines on scanned bank PDF documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ocr_tester.py --folder input_files\n"
            "  python ocr_tester.py --folder input_files --engines paddle easyocr\n"
            "  python ocr_tester.py --folder input_files --engines all --gpu\n"
            "  python ocr_tester.py --folder input_files --engines paddle --dpi 400\n"
        ),
    )
    parser.add_argument("--folder",     type=str, default="input_files",
                        help="Folder containing PDF files (default: input_files)")
    parser.add_argument("--engines",    nargs="+", choices=engine_choices, default=["paddle"],
                        help="Engines to run: %s (default: paddle)" % " | ".join(engine_choices))
    parser.add_argument("--dpi",        type=int, default=300,
                        help="PDF render DPI (default: 300 — try 400 for heavily degraded scans)")
    parser.add_argument("--gpu",        action="store_true", default=False,
                        help=(
                            "Request GPU for supported engines (paddle, doctr, easyocr). "
                            "Requires CUDA-enabled builds — see script header for install instructions."
                        ))
    parser.add_argument("--output-dir", type=str, default="ocr_results",
                        help="Output folder for result files (default: ocr_results)")
    args = parser.parse_args()

    # Expand "all" shortcut
    engine_keys = (
        list(ENGINE_REGISTRY.keys())
        if "all" in args.engines
        else args.engines
    )

    # GPU / CPU mode
    if not args.gpu:
        force_cpu()
    else:
        print("[INFO] GPU mode requested.")
        print("[INFO] Whether GPU is actually used depends on your installed builds.")
        print("[INFO] Check the 'DEVICE / GPU EVIDENCE' section in each output file.\n")

    # Collect device evidence once (written into every output file)
    device_evidence = get_device_evidence(args.gpu)
    print("\n".join(device_evidence))
    print()

    # File selection
    pdfs = list_pdfs(args.folder)
    selected_pdfs = pick_files(pdfs)

    print("\nSelected : %s" % ", ".join(p.name for p in selected_pdfs))
    print("Engines  : %s" % ", ".join(engine_keys))
    print("DPI      : %d" % args.dpi)
    print("GPU      : %s" % args.gpu)
    print()

    # Warn about CPU-only engines when --gpu is set
    if args.gpu:
        cpu_only = [k for k in engine_keys if k not in GPU_CAPABLE_ENGINES]
        if cpu_only:
            print("[INFO] These engines have no GPU support and will run on CPU: %s" % ", ".join(cpu_only))
            print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in selected_pdfs:
        try:
            process_file(
                pdf_path, engine_keys, args.gpu,
                args.dpi, output_dir, device_evidence,
            )
        except Exception as exc:
            print("  [ERROR] Fatal error processing %s: %s" % (pdf_path.name, exc))

    print("\n" + "=" * 80)
    print("  ALL DONE — results saved to: %s\\" % output_dir)
    print("=" * 80)
    for f in sorted(output_dir.glob("*.txt")):
        print("  - %s" % f.name)
    print()


if __name__ == "__main__":
    main()