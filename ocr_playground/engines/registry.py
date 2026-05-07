from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .base import OcrEngine
from .doctr_engine import DoctrEngine
from .surya_engine import SuryaEngine
from .textract_engine import TextractEngine
from .tesseract_engine import TesseractEngine

EngineId = Literal["tesseract", "doctr", "surya", "textract"]


@dataclass(frozen=True)
class EngineInfo:
    id: EngineId
    name: str
    available: bool
    reason: str | None = None


def _all_engines() -> list[OcrEngine]:
    return [TesseractEngine(), DoctrEngine(), SuryaEngine(), TextractEngine()]


def list_engines() -> list[EngineInfo]:
    infos: list[EngineInfo] = []
    for e in _all_engines():
        ok, reason = e.is_available()
        infos.append(EngineInfo(id=e.id, name=e.display_name, available=ok, reason=reason))
    return infos


def get_engine(engine_id: EngineId) -> OcrEngine:
    for e in _all_engines():
        if e.id == engine_id:
            return e
    raise KeyError(engine_id)

