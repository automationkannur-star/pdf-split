from __future__ import annotations

from typing import Iterable

import fitz  # PyMuPDF
from PIL import Image


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return int(doc.page_count)
    finally:
        doc.close()


def render_pdf_to_images(
    pdf_bytes: bytes,
    *,
    dpi: int = 200,
    first_page: int | None = None,
    last_page: int | None = None,
) -> list[Image.Image]:
    """
    Render PDF pages into PIL images using PyMuPDF (no external poppler needed).
    Page indices are 0-based in output list.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        start = first_page if first_page is not None else 0
        end = last_page if last_page is not None else doc.page_count - 1
        start = max(0, start)
        end = min(doc.page_count - 1, end)

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        images: list[Image.Image] = []
        for page_index in range(start, end + 1):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    finally:
        doc.close()


def iter_page_numbers(total_pages: int, first_page: int | None, last_page: int | None) -> Iterable[int]:
    start = first_page if first_page is not None else 0
    end = last_page if last_page is not None else total_pages - 1
    start = max(0, start)
    end = min(total_pages - 1, end)
    for i in range(start, end + 1):
        yield i


def resize_image_max_side(img: Image.Image, *, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    largest = max(w, h)
    if largest <= max_side:
        return img
    ratio = max_side / float(largest)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return img.resize(new_size, Image.Resampling.LANCZOS)

