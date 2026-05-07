from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OcrPageResult:
    page_index: int
    text: str
    seconds: float


@dataclass(frozen=True)
class OcrDocumentResult:
    pages: list[OcrPageResult]

    @property
    def text(self) -> str:
        return "\n\n".join(p.text.rstrip() for p in self.pages).strip() + "\n"

    @property
    def total_seconds(self) -> float:
        return float(sum(p.seconds for p in self.pages))

