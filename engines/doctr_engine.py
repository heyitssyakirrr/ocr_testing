"""
engines/doctr_engine.py
=======================
docTR — db_resnet50 detection + parseq recognition.

GPU: requires torch with CUDA.
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

Note: torchvision must also be a CUDA build (not +cpu).
  pip uninstall torchvision -y
  pip install torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
"""

from __future__ import annotations
import time

from engines.base import EngineResult


def run(images: list, use_gpu: bool = False) -> EngineResult:
    """
    Run docTR on a list of (page_num, PIL.Image) pairs.

    Returns:
        EngineResult — (ocr_text, elapsed_seconds).
    """
    try:
        import torch
        import numpy as np
        from doctr.models import ocr_predictor
    except ImportError:
        return "[SKIP] python-doctr not installed.  pip install python-doctr", 0.0

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    if use_gpu and not torch.cuda.is_available():
        print("  [docTR] WARNING: CUDA not available — falling back to CPU.")

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
        # docTR expects float32 CHW tensors in [0, 1]
        img_tensor = (img_rgb.astype("float32") / 255.0).transpose(2, 0, 1)
        try:
            result = model([img_tensor])
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        words = [w.value     for w in line.words if w.value.strip()]
                        confs = [w.confidence for w in line.words if w.value.strip()]
                        if words:
                            avg_conf = sum(confs) / len(confs)
                            all_lines.append(
                                "%s  (conf:%.3f)" % (" ".join(words), avg_conf)
                            )
        except Exception as exc:
            print("    [docTR] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0
