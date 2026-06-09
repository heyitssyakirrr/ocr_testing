"""
utils/pdf_utils.py
==================
PDF utilities: render pages to PIL Images and extract the embedded text layer.
"""

from __future__ import annotations
import io
import sys
from pathlib import Path


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[tuple[int, object]]:
    """
    Render every page of a PDF to a PIL Image at the requested DPI.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        dpi:      Render resolution (default 300; try 400 for degraded scans).

    Returns:
        List of (page_number, PIL.Image) tuples (1-indexed page numbers).
    """
    try:
        import fitz
    except ImportError:
        sys.exit("[ERROR] PyMuPDF not installed.  pip install PyMuPDF")
    try:
        from PIL import Image
    except ImportError:
        sys.exit("[ERROR] Pillow not installed.  pip install Pillow")

    doc  = fitz.open(pdf_path)
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    imgs = []

    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        imgs.append((page_num, img))
        print("  [PDF] Page %d rendered at %d DPI: %dx%d px" % (
            page_num, dpi, img.size[0], img.size[1]
        ))

    doc.close()
    return imgs


def extract_pdf_text_layer(pdf_path: str) -> str:
    """
    Extract the embedded text layer from the PDF, if any.

    Returns an empty string for image-only / scanned PDFs.
    """
    try:
        import fitz
    except ImportError:
        return ""

    doc  = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc).strip()
    doc.close()
    return text
