from __future__ import annotations

import time
from typing import Any

from PIL import Image

from ..models import OcrDocumentResult, OcrPageResult
from .base import OcrEngine


class DoctrEngine(OcrEngine):
    @property
    def id(self) -> str:
        return "doctr"

    @property
    def display_name(self) -> str:
        return "docTR"

    def is_available(self) -> tuple[bool, str | None]:
        try:
            from doctr.models import ocr_predictor  # noqa: F401

            return True, None
        except Exception as e:
            return False, f"docTR not available. Install `python-doctr[torch]`. Import error: {e}"

    def extract_from_images(
        self,
        images: list[Image.Image],
        *,
        lang: str | None = None,  # docTR handles language differently; kept for interface parity
        config: dict[str, Any] | None = None,
    ) -> OcrDocumentResult:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        det_arch = (config or {}).get("det_arch", "db_resnet50")
        reco_arch = (config or {}).get("reco_arch", "crnn_vgg16_bn")

        predictor = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)

        pages: list[OcrPageResult] = []
        for i, img in enumerate(images):
            start = time.perf_counter()
            doc = DocumentFile.from_images([img])
            result = predictor(doc)
            # Render returns a list of pages' text; we pass a single image so take [0].
            text_pages = result.render()
            text = text_pages[0] if text_pages else ""
            seconds = time.perf_counter() - start
            pages.append(OcrPageResult(page_index=i, text=text or "", seconds=seconds))
        return OcrDocumentResult(pages=pages)

