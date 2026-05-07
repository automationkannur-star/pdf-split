from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass(frozen=True)
class SplitChunk:
    name: str
    start_page: int  # 0-based absolute page index in source PDF
    end_page: int  # 0-based absolute page index in source PDF
    pdf_bytes: bytes


def find_keyword_page_indices(
    page_texts: list[str],
    keyword: str,
    *,
    case_sensitive: bool = False,
) -> list[int]:
    needle = keyword if case_sensitive else keyword.lower()
    hits: list[int] = []
    for i, text in enumerate(page_texts):
        hay = text if case_sensitive else (text or "").lower()
        if needle and needle in hay:
            hits.append(i)
    return hits


def build_ranges_from_starts(starts: list[int], total_pages: int) -> list[tuple[int, int]]:
    if not starts:
        return []
    sorted_starts = sorted(set(s for s in starts if 0 <= s < total_pages))
    ranges: list[tuple[int, int]] = []
    for i, s in enumerate(sorted_starts):
        e = (sorted_starts[i + 1] - 1) if i + 1 < len(sorted_starts) else (total_pages - 1)
        if e >= s:
            ranges.append((s, e))
    return ranges


def split_pdf_by_ranges(
    pdf_bytes: bytes,
    ranges: list[tuple[int, int]],
    *,
    name_prefix: str = "split",
) -> list[SplitChunk]:
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks: list[SplitChunk] = []
    try:
        for idx, (start, end) in enumerate(ranges, start=1):
            out = fitz.open()
            out.insert_pdf(src, from_page=start, to_page=end)
            bytes_out = out.tobytes(garbage=4, deflate=True)
            out.close()
            chunks.append(
                SplitChunk(
                    name=f"{name_prefix}_{idx:03d}_p{start + 1}-p{end + 1}.pdf",
                    start_page=start,
                    end_page=end,
                    pdf_bytes=bytes_out,
                )
            )
        return chunks
    finally:
        src.close()


def chunks_to_zip_bytes(chunks: list[SplitChunk], *, zip_name_hint: str = "splits") -> tuple[str, bytes]:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for c in chunks:
            zf.writestr(c.name, c.pdf_bytes)
    zip_name = f"{zip_name_hint}_splits.zip"
    return zip_name, mem.getvalue()

