"""
engines/doctr_engine.py
=======================
docTR — db_resnet50 + parseq, CPU only.
Install: pip install python-doctr torch torchvision
"""

from __future__ import annotations
import time

from engines.base import EngineResult


def run(images: list, use_gpu: bool = False) -> EngineResult:
    try:
        import numpy as np
        from doctr.models import ocr_predictor
    except ImportError:
        return "[SKIP] python-doctr not installed.  pip install python-doctr", 0.0

    print("  [docTR] Loading model on CPU...")
    try:
        model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="parseq",
            pretrained=True,
            assume_straight_pages=False,
        )
    except Exception as exc:
        return "[SKIP] docTR model load failed: %s" % exc, 0.0

    import numpy as np
    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img in images:
        print("  [docTR] Processing page %d..." % page_num)
        img_rgb = np.array(img.convert("RGB"))
        try:
            result = model([img_rgb])
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        words = [w.value      for w in line.words if w.value.strip()]
                        confs = [w.confidence for w in line.words if w.value.strip()]
                        if words:
                            avg_conf = sum(confs) / len(confs)
                            all_lines.append("%s  (conf:%.3f)" % (" ".join(words), avg_conf))
        except Exception as exc:
            print("    [docTR] Page %d error: %s" % (page_num, exc))

    return "\n".join(all_lines), time.time() - t0