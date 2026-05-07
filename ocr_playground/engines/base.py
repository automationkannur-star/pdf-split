from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from ..models import OcrDocumentResult


class OcrEngine(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> tuple[bool, str | None]:
        """Returns (available, reason_if_not_available)."""

    @abstractmethod
    def extract_from_images(
        self,
        images: list[Image.Image],
        *,
        lang: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> OcrDocumentResult: ...

