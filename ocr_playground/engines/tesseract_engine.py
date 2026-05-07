from __future__ import annotations

import os
import time
from typing import Any

from PIL import Image

from ..models import OcrDocumentResult, OcrPageResult
from .base import OcrEngine


class TesseractEngine(OcrEngine):
    @property
    def id(self) -> str:
        return "tesseract"

    @property
    def display_name(self) -> str:
        return "Tesseract OCR"

    def is_available(self) -> tuple[bool, str | None]:
        try:
            import pytesseract  # noqa: F401
        except Exception as e:  # pragma: no cover
            return False, f"Python package missing: pytesseract ({e})"

        # Optional: allow overriding the tesseract binary path.
        cmd = os.environ.get("TESSERACT_CMD")
        if cmd:
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = cmd
            except Exception:
                pass

        try:
            import pytesseract

            _ = pytesseract.get_tesseract_version()
            return True, None
        except Exception as e:
            return (
                False,
                "Tesseract binary not found/working. Install Tesseract and/or set TESSERACT_CMD. "
                f"Underlying error: {e}",
            )

    def extract_from_images(
        self,
        images: list[Image.Image],
        *,
        lang: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> OcrDocumentResult:
        import pytesseract

        tesseract_config = ""
        if config and config.get("tesseract_config"):
            tesseract_config = str(config["tesseract_config"])

        pages: list[OcrPageResult] = []
        for i, img in enumerate(images):
            start = time.perf_counter()
            text = pytesseract.image_to_string(img, lang=lang or None, config=tesseract_config)
            seconds = time.perf_counter() - start
            pages.append(OcrPageResult(page_index=i, text=text or "", seconds=seconds))
        return OcrDocumentResult(pages=pages)

