"""
ocr_tester.py
=============
Test OCR engines on PDF documents.

Supported engines : paddle | doctr | rapidocr
Default engine    : paddle

Usage:
    python ocr_tester.py --folder input_files --engines paddle
    python ocr_tester.py --folder input_files --engines paddle doctr rapidocr
    python ocr_tester.py --folder input_files --engines paddle --dpi 400 --gpu
    python ocr_tester.py --folder input_files --engines paddle --output-dir my_results

Accuracy is estimated from each engine's own confidence scores, broken down into
HIGH / MEDIUM / LOW confidence bands with an overall estimated accuracy percentage.

Results are saved to the --output-dir folder (default: ocr_results).
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# GPU / CPU setup
# ---------------------------------------------------------------------------

def force_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PADDLE_USE_GPU"] = "0"
    os.environ["FLAGS_use_cuda"] = "0"
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
    print("[INFO] GPU disabled - running in CPU-only mode.")
    print()


def get_device_evidence(use_gpu):
    """Collect hard evidence about which device is actually in use."""
    lines = []
    lines.append("--- DEVICE / GPU EVIDENCE ---")
    lines.append("Platform             : %s %s" % (platform.system(), platform.release()))
    lines.append("Python               : %s" % platform.python_version())
    lines.append("--gpu flag           : %s" % use_gpu)
    lines.append("CUDA_VISIBLE_DEVICES : %s" % os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))
    lines.append("PADDLE_USE_GPU       : %s" % os.environ.get("PADDLE_USE_GPU", "(not set)"))

    try:
        import torch
        lines.append("PyTorch version           : %s" % torch.__version__)
        lines.append("torch.cuda.is_available() : %s" % torch.cuda.is_available())
        if torch.cuda.is_available():
            lines.append("  CUDA device : %s" % torch.cuda.get_device_name(0))
        lines.append("Active torch device       : %s" % (
            "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"))
    except ImportError:
        lines.append("PyTorch : not installed")

    try:
        import paddle
        lines.append("PaddlePaddle version           : %s" % paddle.__version__)
        lines.append("paddle.is_compiled_with_cuda() : %s" % paddle.is_compiled_with_cuda())
        lines.append("paddle.device.get_device()     : %s" % paddle.device.get_device())
    except ImportError:
        lines.append("PaddlePaddle : not installed")
    except Exception as e:
        lines.append("PaddlePaddle device check error: %s" % e)

    return lines


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path, dpi=300):
    """Render every page of a PDF to a PIL Image at the given DPI."""
    try:
        import fitz
    except ImportError:
        sys.exit("[ERROR] PyMuPDF not installed. Run: pip install PyMuPDF")
    try:
        from PIL import Image
    except ImportError:
        sys.exit("[ERROR] Pillow not installed. Run: pip install Pillow")
    import io

    doc = fitz.open(pdf_path)
    images = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append((page_num + 1, img))
        print("  [PDF] Page %d rendered at %d DPI: %dx%d px" % (
            page_num + 1, dpi, img.size[0], img.size[1]))
    doc.close()
    return images


# ---------------------------------------------------------------------------
# OCR engines
# ---------------------------------------------------------------------------
# Each engine signature:
#   run_<name>(images, use_gpu=False) -> (text: str, elapsed: float)
#
# images = [(page_num, PIL.Image), ...]
# Return ("[SKIP] reason", 0.0) when engine cannot run.
# Each output line must end with "  (conf:X.XXX)" for accuracy metrics.
# ---------------------------------------------------------------------------

def run_paddle(images, use_gpu=False):
    """
    PaddleOCR 3.x — PP-OCRv5 server models (highest accuracy).

    New 3.x API uses ocr.predict() and returns result objects instead of
    nested lists. Falls back to PP-OCRv4 if paddleocr < 3.0 is installed.

    PP-OCRv5 accuracy settings:
      PP-OCRv5_server_det  — large detection model, best for documents
      PP-OCRv5_server_rec  — large recognition model, best accuracy
      use_doc_orientation_classify=True  — auto-correct rotated pages
      use_textline_orientation=True      — correct individual line tilt
    """
    try:
        import numpy as np
        from paddleocr import PaddleOCR
        import paddleocr as _poc
        paddle_ver = tuple(int(x) for x in _poc.__version__.split(".")[:2])
    except ImportError:
        return "[SKIP] PaddleOCR not installed. Run: pip install paddlepaddle paddleocr", 0.0

    # Disable oneDNN to avoid Windows CPU crash
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["PADDLE_DISABLE_ONEDNN"] = "1"

    all_lines = []
    t0 = time.time()

    if paddle_ver >= (3, 0):
        # ---- PaddleOCR 3.x — PP-OCRv5 API ----
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
        except Exception as e:
            return "[SKIP] PaddleOCR 3.x model load failed: %s" % e, 0.0

        for page_num, img in images:
            print("  [PaddleOCR] Processing page %d..." % page_num)
            import tempfile, os as _os
            # PP-OCRv5 predict() accepts file paths or numpy arrays
            img_array = np.array(img.convert("RGB"))
            try:
                results = ocr.predict(img_array)
                if not results:
                    continue
                for res in results:
                    # 3.x result object: res['rec_texts'], res['rec_scores']
                    rec_texts  = res.get("rec_texts",  []) if isinstance(res, dict) else getattr(res, "rec_texts",  [])
                    rec_scores = res.get("rec_scores", []) if isinstance(res, dict) else getattr(res, "rec_scores", [])
                    for text, conf in zip(rec_texts, rec_scores):
                        if str(text).strip():
                            all_lines.append("%s  (conf:%.3f)" % (text, float(conf)))
            except Exception as e:
                print("    [PaddleOCR] Page %d error: %s" % (page_num, e))
                continue

    else:
        # ---- PaddleOCR 2.x — PP-OCRv4 fallback ----
        print("  [PaddleOCR] Loading PP-OCRv4 models (2.x API fallback)...")
        print("  [PaddleOCR] WARNING: Install paddlepaddle>=3.0 + paddleocr>=3.0 for PP-OCRv5.")
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
        except Exception as e:
            return "[SKIP] PaddleOCR 2.x model load failed: %s" % e, 0.0

        for page_num, img in images:
            print("  [PaddleOCR] Processing page %d..." % page_num)
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
                            all_lines.append("%s  (conf:%.3f)" % (text, conf))
                    except (IndexError, TypeError):
                        continue

    elapsed = time.time() - t0
    return "\n".join(all_lines), elapsed


def run_doctr(images, use_gpu=False):
    """
    docTR — db_resnet50 detection + parseq recognition.
    parseq is the highest-accuracy recognition arch in docTR.
    assume_straight_pages=False handles mild page tilt/skew.
    """
    try:
        import torch
        import numpy as np
        from doctr.models import ocr_predictor
    except ImportError:
        return "[SKIP] python-doctr not installed. Run: pip install python-doctr", 0.0

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    print("  [doctr] Loading model on %s..." % device)
    try:
        model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="parseq",
            pretrained=True,
            assume_straight_pages=False,
        ).to(device)
    except Exception as e:
        return "[SKIP] docTR model load failed: %s" % e, 0.0

    all_lines = []
    t0 = time.time()
    for page_num, img in images:
        print("  [doctr] Processing page %d..." % page_num)
        img_rgb = np.array(img.convert("RGB"))
        # docTR expects float32 CHW in [0, 1]
        img_tensor = (img_rgb.astype("float32") / 255.0).transpose(2, 0, 1)
        try:
            result = model([img_tensor])
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        words = [w.value for w in line.words if w.value.strip()]
                        confs = [w.confidence for w in line.words if w.value.strip()]
                        if words:
                            line_text = " ".join(words)
                            avg_conf = sum(confs) / len(confs) if confs else 0.0
                            all_lines.append("%s  (conf:%.3f)" % (line_text, avg_conf))
        except Exception as e:
            print("    [doctr] Page %d error: %s" % (page_num, e))
            continue
    elapsed = time.time() - t0
    return "\n".join(all_lines), elapsed


def run_rapidocr(images, use_gpu=False):
    """
    RapidOCR — ONNX-runtime based, CPU-only, very fast.
    Best for quick runs; word spacing quality is lower than Paddle/docTR.
    """
    try:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return "[SKIP] RapidOCR not installed. Run: pip install rapidocr-onnxruntime", 0.0

    print("  [RapidOCR] Loading model...")
    ocr = RapidOCR()

    all_lines = []
    t0 = time.time()
    for page_num, img in images:
        print("  [RapidOCR] Processing page %d..." % page_num)
        img_array = np.array(img.convert("RGB"))
        try:
            result, _ = ocr(img_array)
            if result is None:
                continue
            for item in result:
                # item = [bbox, text, confidence]
                if len(item) >= 3:
                    text, conf = item[1], item[2]
                    if str(text).strip():
                        all_lines.append("%s  (conf:%.3f)" % (text, conf))
        except Exception as e:
            print("    [RapidOCR] Page %d error: %s" % (page_num, e))
            continue
    elapsed = time.time() - t0
    return "\n".join(all_lines), elapsed


# ---------------------------------------------------------------------------
# Engine registry
# To add a new engine: implement run_<name>() above and add an entry here.
# ---------------------------------------------------------------------------

ENGINE_REGISTRY = {
    "paddle":   ("PaddleOCR",  run_paddle),
    "doctr":    ("docTR",      run_doctr),
    "rapidocr": ("RapidOCR",   run_rapidocr),
}


# ---------------------------------------------------------------------------
# Confidence-based accuracy metrics
# ---------------------------------------------------------------------------

def confidence_stats(ocr_text):
    """
    Extract confidence scores from lines ending with '  (conf:X.XXX)'.
    Returns per-band breakdown and an estimated accuracy percentage.
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
    estimated_accuracy = avg * 100

    return {
        "count"              : total,
        "avg_conf"           : avg,
        "min_conf"           : min(confs),
        "max_conf"           : max(confs),
        "high_conf_lines"    : high,
        "medium_conf_lines"  : medium,
        "low_conf_lines"     : low,
        "estimated_accuracy" : estimated_accuracy,
    }


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------

def flag_ambiguous_lines(text):
    BANK_PATTERNS  = ["O/D", "0/D", "SLUMYR", "MH3MYR", "/D88", "/D55", "/D90"]
    AMBIGUOUS_CHARS = set("0O1lI5S8B2Z4A6G3E")
    flagged = []
    for line in text.split("\n"):
        clean = line.split("  (conf:")[0].strip()
        if any(p in clean for p in BANK_PATTERNS):
            flagged.append("  >>> BANK PATTERN: %s" % line)
        elif len(clean) >= 8 and sum(c.isalnum() for c in clean) >= 6:
            if any(c in clean for c in AMBIGUOUS_CHARS):
                flagged.append("  [?] AMBIGUOUS: %s" % line)
    return flagged


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_file(pdf_path, engine_key, engine_name, engine_fn,
                 use_gpu, dpi, output_dir, device_evidence):
    print("\n" + "=" * 80)
    print("  FILE   : %s" % pdf_path.name)
    print("  ENGINE : %s" % engine_name)
    print("=" * 80)

    print("\n  Rendering PDF...")
    images = pdf_to_images(str(pdf_path), dpi=dpi)

    print("\n  Running %s..." % engine_name)
    text, elapsed = engine_fn(images, use_gpu=use_gpu)

    pdf_stem = pdf_path.stem.lower().replace(" ", "_")
    out_filename = "%s_%s_output.txt" % (engine_key, pdf_stem)
    out_file = output_dir / out_filename

    if text.startswith("[SKIP]"):
        print("  " + text)
        return

    lines   = [l for l in text.split("\n") if l.strip()]
    flagged = flag_ambiguous_lines(text)
    cs      = confidence_stats(text)

    with open(out_file, "w", encoding="utf-8") as f:

        # Header
        f.write("=" * 80 + "\n")
        f.write("FILE   : %s\n" % pdf_path.name)
        f.write("ENGINE : %s\n" % engine_name)
        f.write("DPI    : %d\n" % dpi)
        f.write("TIME   : %.1f seconds\n" % elapsed)
        f.write("LINES  : %d\n" % len(lines))
        f.write("=" * 80 + "\n\n")

        # Device evidence
        f.write("\n".join(device_evidence) + "\n\n")

        # Confidence-based accuracy
        f.write("--- ACCURACY ESTIMATE (confidence-based) ---\n\n")
        if cs:
            f.write("  NOTE: PDF is image-based so no selectable ground truth exists.\n")
            f.write("        Accuracy is estimated from the model's own confidence scores.\n")
            f.write("        Each line's confidence = model's estimated probability it read that line correctly.\n\n")
            f.write("  Total lines scored     : %d\n"   % cs["count"])
            f.write("  Average confidence     : %.4f\n" % cs["avg_conf"])
            f.write("  Min confidence         : %.4f\n" % cs["min_conf"])
            f.write("  Max confidence         : %.4f\n" % cs["max_conf"])
            f.write("\n")
            f.write("  Confidence breakdown:\n")
            f.write("    HIGH   (>= 0.90) : %d lines  (%.1f%%)  <- very reliable\n" % (
                cs["high_conf_lines"],   cs["high_conf_lines"]   / cs["count"] * 100))
            f.write("    MEDIUM (0.70-0.89): %d lines  (%.1f%%)  <- acceptable\n" % (
                cs["medium_conf_lines"], cs["medium_conf_lines"] / cs["count"] * 100))
            f.write("    LOW    (< 0.70)  : %d lines  (%.1f%%)  <- unreliable, review manually\n" % (
                cs["low_conf_lines"],    cs["low_conf_lines"]    / cs["count"] * 100))
            f.write("\n")
            f.write("  Estimated accuracy     : %.2f%%\n" % cs["estimated_accuracy"])
            f.write("\n")
            f.write("  Interpretation guide:\n")
            f.write("    >= 95%  Excellent  - very few errors expected\n")
            f.write("    90-94%  Good       - minor errors, spot-check LOW lines\n")
            f.write("    80-89%  Acceptable - noticeable errors, review LOW lines\n")
            f.write("    < 80%   Poor       - significant errors, consider higher DPI\n")
        else:
            f.write("  No confidence scores found.\n")
        f.write("\n")

        # Full text
        f.write("--- FULL TEXT ---\n\n")
        for line in lines:
            f.write(line + "\n")

        # Flagged lines
        f.write("\n\n--- FLAGGED LINES (possible 0/O confusion) ---\n\n")
        if flagged:
            for line in flagged:
                f.write(line + "\n")
        else:
            f.write("No ambiguous lines detected.\n")

    # Console summary
    print("  Done in %.1fs — %d lines extracted, %d flagged" % (elapsed, len(lines), len(flagged)))
    if cs:
        print("  Estimated accuracy      : %.2f%%" % cs["estimated_accuracy"])
        print("  Confidence avg/min/max  : %.3f / %.3f / %.3f" % (
            cs["avg_conf"], cs["min_conf"], cs["max_conf"]))
        print("  Lines HIGH/MEDIUM/LOW   : %d / %d / %d" % (
            cs["high_conf_lines"], cs["medium_conf_lines"], cs["low_conf_lines"]))
    print("  Saved: %s" % out_file)


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

def list_pdfs(folder):
    folder_path = Path(folder)
    if not folder_path.exists():
        sys.exit("[ERROR] Folder not found: %s" % folder)
    pdfs = sorted(folder_path.glob("*.pdf"))
    if not pdfs:
        sys.exit("[ERROR] No PDF files found in: %s" % folder)
    return pdfs


def pick_files(pdfs):
    print("\nPDF files found in folder:")
    for i, p in enumerate(pdfs):
        print("  [%d] %s" % (i + 1, p.name))
    print("  [A] All files")
    print()
    choice = input("Which file(s) to test? Enter number(s) separated by comma, or A for all: ").strip()

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

def main():
    parser = argparse.ArgumentParser(
        description="Test OCR engines on PDF documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--folder",     type=str, default="input_files",
                        help="Folder containing PDF files (default: input_files)")
    parser.add_argument("--engines",    nargs="+",
                        choices=list(ENGINE_REGISTRY.keys()),
                        default=["paddle"],
                        help="Engine(s) to run: paddle | doctr | rapidocr (default: paddle)")
    parser.add_argument("--dpi",        type=int, default=300,
                        help="PDF render DPI (default: 300). Higher = more accurate but slower.")
    parser.add_argument("--gpu",        action="store_true", default=False,
                        help="Enable GPU for engines that support it (paddle, doctr)")
    parser.add_argument("--output-dir", type=str, default="ocr_results",
                        help="Folder to save result .txt files (default: ocr_results)")
    args = parser.parse_args()

    # GPU / CPU setup
    if not args.gpu:
        force_cpu()
    else:
        print("[INFO] GPU mode enabled.")

    # Device evidence (printed once, written into every output file)
    device_evidence = get_device_evidence(args.gpu)
    print("\n".join(device_evidence))
    print()

    # List and interactively pick PDFs
    pdfs = list_pdfs(args.folder)
    selected_pdfs = pick_files(pdfs)

    print("\nSelected files:")
    for p in selected_pdfs:
        print("  - %s" % p.name)
    print("\nEngines : %s" % ", ".join(args.engines))
    print("DPI     : %d" % args.dpi)
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in selected_pdfs:
        for engine_key in args.engines:
            engine_name, engine_fn = ENGINE_REGISTRY[engine_key]
            try:
                process_file(
                    pdf_path, engine_key, engine_name, engine_fn,
                    args.gpu, args.dpi, output_dir, device_evidence,
                )
            except Exception as e:
                print("  [ERROR] %s on %s: %s" % (engine_name, pdf_path.name, e))

    print("\n" + "=" * 80)
    print("  ALL DONE")
    print("=" * 80)
    print("  Results saved to: %s\\" % output_dir)
    result_files = sorted(output_dir.glob("*.txt"))
    for f in result_files:
        print("  - %s" % f.name)
    print()


if __name__ == "__main__":
    main()