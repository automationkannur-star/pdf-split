from __future__ import annotations

import time
from typing import Any

from PIL import Image

from ..models import OcrDocumentResult, OcrPageResult
from .base import OcrEngine


class SuryaEngine(OcrEngine):
    @property
    def id(self) -> str:
        return "surya"

    @property
    def display_name(self) -> str:
        return "Surya OCR"

    def is_available(self) -> tuple[bool, str | None]:
        # Surya OCR packaging/API may vary; we check a couple common import shapes.
        try:
            import surya  # noqa: F401

            return True, None
        except Exception as e:
            return False, f"Surya OCR not available. Install `surya-ocr`. Import error: {e}"

    def extract_from_images(
        self,
        images: list[Image.Image],
        *,
        lang: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> OcrDocumentResult:
        """
        Tries to call Surya in a best-effort way. If your installed Surya version
        uses a different API, this will raise a helpful error message shown in the UI.
        """
        try:
            # Common (but not guaranteed) API shapes:
            # - from surya.ocr import run_ocr
            # - from surya import OCR
            from surya.ocr import run_ocr  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Could not import `from surya.ocr import run_ocr`. "
                "Your Surya version likely exposes a different API. "
                f"Import error: {e}"
            ) from e

        pages: list[OcrPageResult] = []
        for i, img in enumerate(images):
            start = time.perf_counter()
            out = run_ocr(img, lang=lang)  # type: ignore[arg-type]
            seconds = time.perf_counter() - start

            # Attempt to normalize to text.
            text = ""
            if isinstance(out, str):
                text = out
            elif isinstance(out, dict) and "text" in out:
                text = str(out.get("text") or "")
            else:
                text = str(out)

            pages.append(OcrPageResult(page_index=i, text=text, seconds=seconds))
        return OcrDocumentResult(pages=pages)

