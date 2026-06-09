"""
ocr_tester.py
=============
Entry point for the OCR engine tester.

This file is intentionally thin — it handles CLI parsing, file selection,
and orchestration only. All logic lives in the subpackages:

  config/   — settings, engine registry
  engines/  — one module per OCR engine
  metrics/  — confidence, CER, field-check
  utils/    — PDF rendering, device probe, output writing, etc.

Run with -h / --help for full usage.
"""

# ---------------------------------------------------------------------------
# CUDA DLL fix — must run before ANY torch/paddle/nvidia import
# ---------------------------------------------------------------------------
from utils.cuda_fix import apply as _apply_cuda_fix
_apply_cuda_fix()

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import argparse
import importlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config.settings import (
    DEFAULT_INPUT_FOLDER,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_DPI,
    DEFAULT_ENGINE,
    ENGINE_REGISTRY,
    GPU_CAPABLE_ENGINES,
)
from engines.base import make_result, OCRResult
from utils.device_probe import get_device_evidence
from utils.pdf_utils import pdf_to_images, extract_pdf_text_layer
from utils.known_fields import load_known_fields
from utils.output_writer import write_output
from utils.comparison import build_and_write_comparison
from utils.flagging import flag_critical_lines
from metrics.confidence import confidence_stats


# ---------------------------------------------------------------------------
# GPU / CPU mode
# ---------------------------------------------------------------------------

def _force_cpu() -> None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]  = ""
    os.environ["PADDLE_USE_GPU"]        = "0"
    os.environ["FLAGS_use_cuda"]        = "0"
    os.environ["FLAGS_use_mkldnn"]      = "0"
    os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
    print("[INFO] CPU-only mode — GPU disabled.\n")


# ---------------------------------------------------------------------------
# File selection helpers
# ---------------------------------------------------------------------------

def _list_pdfs(folder: str) -> list[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        sys.exit("[ERROR] Folder not found: %s" % folder)
    pdfs = sorted(folder_path.glob("*.pdf"))
    if not pdfs:
        sys.exit("[ERROR] No PDF files found in: %s" % folder)
    return pdfs


def _pick_files(pdfs: list[Path]) -> list[Path]:
    print("\nPDF files found:")
    for i, p in enumerate(pdfs, start=1):
        print("  [%d] %s" % (i, p.name))
    print("  [A] All files\n")
    choice = input("Select file(s) — number(s) comma-separated, or A for all: ").strip()

    if choice.upper() == "A":
        return pdfs

    selected: list[Path] = []
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
# Engine runner
# ---------------------------------------------------------------------------

def _load_engine_fn(engine_key: str):
    """Dynamically import and return the run() function for an engine."""
    _, module_path = ENGINE_REGISTRY[engine_key]
    module = importlib.import_module(module_path)
    return module.run


def _run_engines(
    pdf_path:        Path,
    engine_keys:     list[str],
    use_gpu:         bool,
    dpi:             int,
    output_dir:      Path,
    device_evidence: list[str],
) -> None:
    print("\n" + "=" * 80)
    print("  FILE : %s" % pdf_path.name)
    print("=" * 80)

    # Render PDF pages once — reused by every engine
    print("\n  Rendering PDF at %d DPI..." % dpi)
    images = pdf_to_images(str(pdf_path), dpi=dpi)

    # PDF text layer (for CER metric; empty for scanned PDFs)
    pdf_text_layer = extract_pdf_text_layer(str(pdf_path))
    if pdf_text_layer.strip():
        print("  [INFO] PDF has embedded text layer (%d chars) — CER metric available." % len(pdf_text_layer))
    else:
        print("  [INFO] PDF has no text layer (scanned image) — CER metric will be skipped.")

    # Known fields (for spot-check metric)
    known_fields = load_known_fields(pdf_path)
    if known_fields:
        print("  [INFO] Loaded %d known fields from known_fields.txt" % len(known_fields))
    else:
        print("  [INFO] No known_fields.txt found — spot-check metric will be skipped.")
        print("         Create %s to enable it." % (pdf_path.parent / "known_fields.txt"))

    all_results: list[OCRResult] = []

    for engine_key in engine_keys:
        engine_name, _ = ENGINE_REGISTRY[engine_key]
        print("\n  Running %s..." % engine_name)

        if use_gpu and engine_key not in GPU_CAPABLE_ENGINES:
            print("  [INFO] %s has no GPU support — will run on CPU." % engine_name)

        engine_fn = _load_engine_fn(engine_key)

        try:
            raw = engine_fn(images, use_gpu=use_gpu)
        except Exception as exc:
            raw = ("[SKIP] Unexpected error: %s" % exc, 0.0)

        result = make_result(engine_key, engine_name, pdf_path.name, raw)
        all_results.append(result)

        if result.skipped:
            print("  SKIPPED: %s" % result.skip_reason)
            continue

        out_file = write_output(
            result, device_evidence, dpi,
            output_dir, pdf_text_layer, known_fields,
        )

        n_lines   = len([l for l in result.text.split("\n") if l.strip()])
        n_flagged = len(flag_critical_lines(result.text))
        cs        = confidence_stats(result.text)

        print("  Done in %.1fs — %d lines, %d critically flagged" % (
            result.elapsed, n_lines, n_flagged))
        if cs:
            print("  Confidence avg/min: %.3f / %.3f  |  LOW lines: %d" % (
                cs["avg_conf"], cs["min_conf"], cs["low"]))
        print("  Saved: %s" % out_file)

    # Cross-engine comparison
    comp_file = build_and_write_comparison(
        all_results, known_fields, output_dir, pdf_path.stem
    )
    if comp_file:
        print("\n  Cross-engine comparison saved: %s" % comp_file)


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
            "\n"
            "GPU notes:\n"
            "  Requires NVIDIA GPU + CUDA-enabled builds of torch/paddlepaddle-gpu.\n"
            "  See README.md for full setup instructions.\n"
        ),
    )
    parser.add_argument(
        "--folder",
        default=DEFAULT_INPUT_FOLDER,
        help="Folder containing PDF files (default: %s)" % DEFAULT_INPUT_FOLDER,
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=engine_choices,
        default=[DEFAULT_ENGINE],
        help="Engines to run (default: %s)" % DEFAULT_ENGINE,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="PDF render DPI (default: %d; try 400 for heavily degraded scans)" % DEFAULT_DPI,
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Request GPU for supported engines (paddle, doctr, easyocr, rapidocr_paddle).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for result files (default: %s)" % DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    # Expand "all" shortcut
    engine_keys = (
        list(ENGINE_REGISTRY.keys())
        if "all" in args.engines
        else args.engines
    )

    # Set up GPU / CPU environment variables
    if not args.gpu:
        _force_cpu()
    else:
        print("[INFO] GPU mode requested.")
        print("[INFO] Whether GPU is actually used depends on your installed builds.")
        print("[INFO] Check the 'DEVICE / GPU EVIDENCE' section in each output file.\n")

    # Collect device evidence once — written into every report
    device_evidence = get_device_evidence(args.gpu)
    print("\n".join(device_evidence))
    print()

    # File selection
    pdfs          = _list_pdfs(args.folder)
    selected_pdfs = _pick_files(pdfs)

    print("\nSelected : %s" % ", ".join(p.name for p in selected_pdfs))
    print("Engines  : %s" % ", ".join(engine_keys))
    print("DPI      : %d" % args.dpi)
    print("GPU      : %s" % args.gpu)
    print()

    if args.gpu:
        cpu_only = [k for k in engine_keys if k not in GPU_CAPABLE_ENGINES]
        if cpu_only:
            print("[INFO] CPU-only engines (no GPU support): %s\n" % ", ".join(cpu_only))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in selected_pdfs:
        try:
            _run_engines(
                pdf_path, engine_keys, args.gpu,
                args.dpi, output_dir, device_evidence,
            )
        except Exception as exc:
            print("  [ERROR] Fatal error processing %s: %s" % (pdf_path.name, exc))

    # Final summary
    print("\n" + "=" * 80)
    print("  ALL DONE — results saved to: %s\\" % output_dir)
    print("=" * 80)
    for f in sorted(output_dir.glob("*.txt")):
        print("  - %s" % f.name)
    print()


if __name__ == "__main__":
    main()
