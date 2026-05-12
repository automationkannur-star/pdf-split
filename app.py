from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import streamlit as st

from ocr_playground.engines import get_engine
from ocr_playground.pdf_render import render_pdf_to_images, resize_image_max_side
from ocr_playground.pdf_split import build_ranges_from_starts, find_regex_page_indices


st.set_page_config(page_title="PDF OCR → JSON → split settings", layout="wide")


def _resolved_ocr_json_dir() -> Path:
    """On-disk folder for OCR JSON (not shown in UI). Override with OCR_JSON_OUTPUT_DIR."""
    raw = (os.environ.get("OCR_JSON_OUTPUT_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path(__file__).resolve().parent / "ocr_json_output").resolve()


def _bytes_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _safe_filename_component(name: str) -> str:
    base = name.rsplit(".", 1)[0] if "." in name else name
    base = re.sub(r"[^\w\-]+", "_", base, flags=re.UNICODE).strip("_") or "document"
    return base[:120]


def build_ocr_export_payload(
    *,
    source_filename: str,
    engine_id: str,
    pdf_sha1_hex: str,
    start_page_abs: int,
    end_page_abs: int,
    total_document_pages: int,
    elapsed_seconds: float,
    pages: list[Any],
    split_config: dict[str, Any],
    split_preview: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "source_file": source_filename,
        "engine": engine_id,
        "pdf_sha1": pdf_sha1_hex,
        "total_document_pages": total_document_pages,
        "ocr_window": {
            "start_page_1_indexed": start_page_abs + 1,
            "end_page_1_indexed": end_page_abs + 1,
            "page_count": end_page_abs - start_page_abs + 1,
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
        "split_config": split_config,
        "split_preview": split_preview,
        "pages": [
            {
                "page_index_in_batch": p.page_index,
                "absolute_page_1_indexed": start_page_abs + p.page_index + 1,
                "text": p.text,
                "seconds": round(p.seconds, 4),
            }
            for p in pages
        ],
    }


def write_ocr_extract_json(
    output_dir: str,
    *,
    source_filename: str,
    engine_id: str,
    pdf_sha1_hex: str,
    start_page_abs: int,
    end_page_abs: int,
    total_document_pages: int,
    elapsed_seconds: float,
    pages: list[Any],
    split_config: dict[str, Any],
    split_preview: dict[str, Any],
) -> str:
    """
    Write OCR pages, split configuration, and regex preview to JSON under ``output_dir``.
    """
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename_component(source_filename)
    out_name = f"{stem}__{engine_id}__{pdf_sha1_hex[:12]}.json"
    out_path = root / out_name
    payload = build_ocr_export_payload(
        source_filename=source_filename,
        engine_id=engine_id,
        pdf_sha1_hex=pdf_sha1_hex,
        start_page_abs=start_page_abs,
        end_page_abs=end_page_abs,
        total_document_pages=total_document_pages,
        elapsed_seconds=elapsed_seconds,
        pages=pages,
        split_config=split_config,
        split_preview=split_preview,
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def _build_split_config_and_preview(
    page_texts: list[str],
    *,
    start_page_abs: int,
    end_page_abs: int,
    range_cap_pages: int,
    segment_regex: str,
    regex_ignore_case: bool,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Build ``split_config`` + ``split_preview`` for JSON export. Returns (config, preview, regex_error)."""
    flags = re.IGNORECASE if regex_ignore_case else 0
    local_hits, rx_err = find_regex_page_indices(page_texts, segment_regex, flags=flags)
    if rx_err:
        split_preview: dict[str, Any] = {
            "pattern_valid": False,
            "regex_error": rx_err,
            "local_match_indices_0_based": [],
            "absolute_start_pages_1_indexed": [],
            "suggested_segments_1_indexed": [],
        }
        split_config = {
            "version": 1,
            "segment_starts": {
                "detection": "regex",
                "pattern": segment_regex.strip(),
                "ignore_case": regex_ignore_case,
                "python_regex_flags": int(flags),
            },
            "how_to_split": (
                "For each page index (0-based in full PDF) where `pattern` matches page text, "
                "start a new segment. Close each segment at the page before the next match, "
                "or at the last page of the document. Use build_ranges_from_starts(starts_0_based, total_page_count)."
            ),
        }
        return split_config, split_preview, rx_err

    abs_starts = [start_page_abs + i for i in local_hits]
    ranges = build_ranges_from_starts(abs_starts, total_pages=range_cap_pages)
    segments = [
        {
            "segment_index": idx,
            "start_page_1_indexed": s + 1,
            "end_page_1_indexed": e + 1,
            "start_page_0_indexed": s,
            "end_page_0_indexed": e,
        }
        for idx, (s, e) in enumerate(ranges, start=1)
    ]
    split_config = {
        "version": 1,
        "segment_starts": {
            "detection": "regex",
            "pattern": segment_regex.strip(),
            "ignore_case": regex_ignore_case,
            "python_regex_flags": int(flags),
        },
        "how_to_split": (
            "Collect 0-based page indices where `segment_starts.pattern` matches that page's OCR text. "
            "Sort unique. For each start S_i, end = (S_{i+1} - 1) or (total_document_pages - 1). "
            "Emit one PDF per (S_i, end) inclusive. When OCR covers only a page window, cap the last "
            "segment with the same rule but total_pages = min(document_pages, window_end+1) if you "
            "only trust matches inside the window."
        ),
    }
    split_preview = {
        "pattern_valid": True,
        "local_match_indices_0_based": local_hits,
        "absolute_start_pages_1_indexed": [i + 1 for i in abs_starts],
        "suggested_segments_1_indexed": segments,
        "range_calc_total_pages": range_cap_pages,
        "note": (
            "Segments use range_calc_total_pages so the last segment does not extend past the "
            "OCR window when only part of the PDF was OCR'd."
        ),
    }
    return split_config, split_preview, None


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _first_page_text_in_segment(pages: list[dict[str, Any]], start_page_1: int) -> str:
    for p in pages:
        if int(p.get("absolute_page_1_indexed", 0)) == int(start_page_1):
            return str(p.get("text") or "")
    return ""


def _extract_invoice_number_hint(text: str) -> str | None:
    if not (text or "").strip():
        return None
    patterns = [
        r"(?:INVOICE\s*(?:NUMBER|NO)\.?|INV(?:OICE)?\s*#?)\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-/_]*)",
        r"\b(INV[-\s]?[0-9]{3,})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


def _extract_vendor_hint(text: str) -> str | None:
    m = re.search(
        r"(?:VENDOR|SUPPLIER|FROM|COMPANY|SELLER)\s*[:#]\s*(.{2,120}?)(?:\n|\r|$)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip())[:120]
    return None


def _safe_generated_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def _expand_filename_pattern(pattern: str, *, ctx: dict[str, Any]) -> str:
    out = pattern
    for k, v in ctx.items():
        token = "{" + k + "}"
        if token in out:
            out = out.replace(token, str(v) if v is not None else "")
    return _safe_generated_filename(out)


def build_split_settings_payload(
    ocr_payload: dict[str, Any],
    *,
    document_type: str,
    file_name_pattern: str,
    confidence_auto_threshold: int = 80,
) -> dict[str, Any]:
    """
    Build a split-plan JSON for batch/review from a schema v2 OCR export payload.
    """
    source_file = str(ocr_payload.get("source_file") or "document.pdf")
    pages: list[dict[str, Any]] = list(ocr_payload.get("pages") or [])
    preview = ocr_payload.get("split_preview") or {}
    segments = list(preview.get("suggested_segments_1_indexed") or [])
    pattern_ok = bool(preview.get("pattern_valid", False))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    documents: list[dict[str, Any]] = []

    for i, seg in enumerate(segments, start=1):
        sp1 = int(seg["start_page_1_indexed"])
        ep1 = int(seg["end_page_1_indexed"])
        head = _first_page_text_in_segment(pages, sp1)
        inv = _extract_invoice_number_hint(head)
        vendor = _extract_vendor_hint(head)

        inv_display = inv if inv else f"SEG-{sp1}-{ep1}"
        ctx_fn = {
            "invoiceNumber": inv_display,
            "startPage": sp1,
            "endPage": ep1,
            "currentTime": ts,
            "documentIndex": i,
            "documentType": document_type,
        }
        gen_name = _expand_filename_pattern(file_name_pattern, ctx=ctx_fn)

        conf = 72
        if pattern_ok:
            conf += 10
        if inv:
            conf += min(18, 3 + len(inv))
        if vendor:
            conf += 5
        conf = min(99, max(40, conf))

        status = "AUTO_IDENTIFIED" if conf >= confidence_auto_threshold and inv else "MANUAL_REVIEW_REQUIRED"

        doc: dict[str, Any] = {
            "documentType": document_type,
            "startPage": sp1,
            "endPage": ep1,
            "invoiceNumber": inv,
            "vendorName": vendor,
            "fileNamePattern": file_name_pattern,
            "generatedFileName": gen_name,
            "confidence": conf,
            "status": status,
        }
        documents.append(doc)

    return {
        "sourceFile": source_file,
        "generatedAt": _utc_now_iso_z(),
        "documents": documents,
        "meta": {
            "derivedFrom": "last_ocr_export_payload",
            "patternValid": pattern_ok,
            "segmentCount": len(documents),
        },
    }


def _parse_page_range(s: str) -> tuple[int | None, int | None, str | None]:
    s = (s or "").strip()
    if not s:
        return None, None, None
    if "-" not in s:
        try:
            p = int(s)
            if p < 1:
                return None, None, "Pages are 1-based. Use e.g. 1 or 1-3."
            return p - 1, p - 1, None
        except ValueError:
            return None, None, "Invalid pages format. Use `1-3` or `2`."
    left, right = s.split("-", 1)
    try:
        a = int(left.strip())
        b = int(right.strip())
        if a < 1 or b < 1:
            return None, None, "Pages are 1-based. Use e.g. 1-3."
        if b < a:
            return None, None, "Invalid range: end must be >= start."
        return a - 1, b - 1, None
    except ValueError:
        return None, None, "Invalid pages format. Use `1-3`."


def _get_pdf_page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return int(doc.page_count)
    finally:
        doc.close()


def _apply_page_cap(
    total_pages: int,
    first_page: int | None,
    last_page: int | None,
    max_pages: int,
) -> tuple[int, int, bool]:
    start = first_page if first_page is not None else 0
    end = last_page if last_page is not None else total_pages - 1
    start = max(0, start)
    end = min(total_pages - 1, end)
    capped = False
    if (end - start + 1) > max_pages:
        end = start + max_pages - 1
        capped = True
    return start, end, capped


st.title("Invoice PDF → OCR JSON → split settings")
st.caption(
    "Upload a PDF, run **Tesseract** OCR, and save structured JSON automatically. "
    "Then generate split settings for filenames and review."
)

if "last_ocr_export_payload" not in st.session_state:
    st.session_state["last_ocr_export_payload"] = None
if "last_ocr_export_dir" not in st.session_state:
    st.session_state["last_ocr_export_dir"] = None
if "last_split_settings_payload" not in st.session_state:
    st.session_state["last_split_settings_payload"] = None

upload = st.file_uploader("Upload PDF", type=["pdf"])

with st.sidebar:
    st.header("Tesseract and pages")
    dpi = st.slider("DPI", min_value=100, max_value=300, value=150, step=25)
    page_range = st.text_input("Pages (e.g. `1-5` or empty for all)", value="")
    max_pages_per_run = st.number_input("Max pages per OCR run", min_value=1, max_value=300, value=25, step=1)
    max_image_side = st.slider("Max image side (px)", min_value=1000, max_value=3000, value=1800, step=100)
    tesseract_cmd = st.text_input("Tesseract path (optional)", value=os.environ.get("TESSERACT_CMD", ""))
    tesseract_config = st.text_input("Tesseract extra args", value="--psm 6")
    if tesseract_cmd.strip():
        os.environ["TESSERACT_CMD"] = tesseract_cmd.strip()
    else:
        os.environ.pop("TESSERACT_CMD", None)
    lang = st.text_input("Language", value="eng")

    st.subheader("Where each document starts")
    split_segment_regex = st.text_input(
        "Segment start (regex)",
        value=r"INVOICE\s*NO",
        help="Matched per page in OCR text to build split_preview segments.",
    )
    split_regex_ignore_case = st.checkbox("Ignore case for regex", value=True)

    st.subheader("Split settings")
    split_settings_doc_type = st.text_input("Document type label", value="Invoice")
    split_settings_filename_pattern = st.text_input(
        "Filename pattern",
        value="{invoiceNumber}_{startPage}-{endPage}_{currentTime}.pdf",
        help="Placeholders: {invoiceNumber}, {startPage}, {endPage}, {currentTime}, {documentIndex}, {documentType}",
    )
    split_settings_confidence_threshold = st.slider(
        "Auto-identified if confidence ≥", min_value=50, max_value=95, value=80, step=5
    )

if not upload:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_bytes = upload.getvalue()
pdf_hash = _bytes_sha1(pdf_bytes)
prev_sha = st.session_state.get("_workflow_pdf_sha1")
if prev_sha is not None and prev_sha != pdf_hash:
    st.session_state["last_ocr_export_payload"] = None
    st.session_state["last_ocr_export_dir"] = None
    st.session_state["last_split_settings_payload"] = None
st.session_state["_workflow_pdf_sha1"] = pdf_hash

total_pages = _get_pdf_page_count(pdf_bytes)

first_page, last_page, range_err = _parse_page_range(page_range)
if range_err:
    st.error(range_err)
    st.stop()

engine_id = "tesseract"
engine = get_engine(engine_id)
available, reason = engine.is_available()
if not available:
    st.warning(reason or "Tesseract is not available. Install Tesseract and optional `TESSERACT_CMD`.")

start_page, end_page, was_capped = _apply_page_cap(
    total_pages,
    first_page,
    last_page,
    int(max_pages_per_run),
)

st.subheader("Document")
st.write(
    f"**{upload.name}** — {total_pages} page(s). "
    f"OCR window: **{start_page + 1}–{end_page + 1}** ({end_page - start_page + 1} page(s))."
)
if was_capped:
    st.warning(
        f"The OCR window was limited to **{max_pages_per_run}** pages for stability. "
        f"Adjust **Max pages per OCR run** in the sidebar if you need more."
    )

config: dict[str, Any] = {"tesseract_config": tesseract_config}

st.subheader("Step 1 — Create OCR JSON (Tesseract)")
create_json = st.button(
    "Create JSON",
    type="primary",
    use_container_width=True,
    help="Runs Tesseract on the page window, writes JSON (split_config + split_preview) to the output folder, and enables split settings.",
)

ocr_json_dir = str(_resolved_ocr_json_dir())

if create_json:
    ok, eng_reason = engine.is_available()
    if not ok:
        st.error(eng_reason or "Tesseract is not available.")
        st.stop()

    with st.spinner("Running Tesseract and writing JSON…"):
        t0 = time.perf_counter()
        try:
            images = render_pdf_to_images(pdf_bytes, dpi=dpi, first_page=start_page, last_page=end_page)
            images = [resize_image_max_side(img, max_side=int(max_image_side)) for img in images]
            json_result = engine.extract_from_images(images, lang=lang.strip() or None, config=config)
        except MemoryError:
            st.error("Out of memory during OCR. Try lower DPI, lower max image side, or fewer pages.")
            st.stop()
        except Exception as e:
            st.error(f"OCR failed: {e}")
            st.stop()
        json_elapsed = time.perf_counter() - t0

    split_cfg, split_prev, rx_err = _build_split_config_and_preview(
        [p.text for p in json_result.pages],
        start_page_abs=start_page,
        end_page_abs=end_page,
        range_cap_pages=end_page + 1,
        segment_regex=split_segment_regex,
        regex_ignore_case=split_regex_ignore_case,
    )
    if rx_err:
        st.warning(
            f"Segment regex issue: {rx_err}. JSON is still written; adjust the pattern in the sidebar for valid segments."
        )

    try:
        write_ocr_extract_json(
            ocr_json_dir,
            source_filename=upload.name,
            engine_id=engine.id,
            pdf_sha1_hex=pdf_hash,
            start_page_abs=start_page,
            end_page_abs=end_page,
            total_document_pages=total_pages,
            elapsed_seconds=json_elapsed,
            pages=json_result.pages,
            split_config=split_cfg,
            split_preview=split_prev,
        )
    except OSError as e:
        st.error(f"Could not write JSON file: {e}")
        st.stop()

    dl_payload = build_ocr_export_payload(
        source_filename=upload.name,
        engine_id=engine.id,
        pdf_sha1_hex=pdf_hash,
        start_page_abs=start_page,
        end_page_abs=end_page,
        total_document_pages=total_pages,
        elapsed_seconds=json_elapsed,
        pages=json_result.pages,
        split_config=split_cfg,
        split_preview=split_prev,
    )
    st.session_state["last_ocr_export_payload"] = dl_payload
    st.session_state["last_ocr_export_dir"] = ocr_json_dir
    st.session_state["last_split_settings_payload"] = None
    st.success(f"JSON created in **{json_elapsed:.1f}s** and saved.")

pl_ocr = st.session_state.get("last_ocr_export_payload")
if pl_ocr:
    st.subheader("Download OCR JSON")
    ocr_stem = _safe_filename_component(str(pl_ocr.get("source_file") or upload.name or "document.pdf"))
    ocr_eng = str(pl_ocr.get("engine") or "tesseract")
    ocr_sha = str(pl_ocr.get("pdf_sha1") or pdf_hash)[:12]
    ocr_dl_name = f"{ocr_stem}__{ocr_eng}__{ocr_sha}.json"
    st.download_button(
        "Download OCR JSON",
        data=json.dumps(pl_ocr, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=ocr_dl_name,
        mime="application/json",
        use_container_width=True,
        key="dl_ocr_export_json",
    )

if st.session_state.get("last_ocr_export_payload"):
    st.divider()
    st.subheader("Step 2 — Split settings")
    st.caption(
        "Uses `split_preview` segment ranges and light heuristics on the first page of each segment. "
        "Tune the segment regex in the sidebar if you get no documents."
    )
    gen_cols = st.columns([1, 1])
    with gen_cols[0]:
        gen_split_settings = st.button(
            "Generate split settings",
            use_container_width=True,
            key="btn_gen_split_settings",
        )
    with gen_cols[1]:
        save_split_to_disk = st.button(
            "Save split settings JSON",
            use_container_width=True,
            disabled=not bool(st.session_state.get("last_ocr_export_dir")),
        )

    if gen_split_settings:
        pl = st.session_state["last_ocr_export_payload"]
        if not isinstance(pl, dict):
            st.error("Invalid cached OCR export.")
        else:
            segs = (pl.get("split_preview") or {}).get("suggested_segments_1_indexed") or []
            if not segs:
                st.warning(
                    "No segments in the last OCR export. Adjust **Segment start (regex)** in the sidebar and create JSON again."
                )
                st.session_state["last_split_settings_payload"] = None
            else:
                settings = build_split_settings_payload(
                    pl,
                    document_type=(split_settings_doc_type or "Invoice").strip() or "Invoice",
                    file_name_pattern=split_settings_filename_pattern.strip()
                    or "{invoiceNumber}_{startPage}-{endPage}_{currentTime}.pdf",
                    confidence_auto_threshold=int(split_settings_confidence_threshold),
                )
                st.session_state["last_split_settings_payload"] = settings
                st.success(f"Generated split plan for **{len(settings['documents'])}** document(s).")

    if save_split_to_disk and st.session_state.get("last_split_settings_payload"):
        out_dir = st.session_state.get("last_ocr_export_dir")
        if not out_dir:
            st.warning("Create JSON again to enable saving split settings.")
        else:
            try:
                root = Path(out_dir).expanduser().resolve()
                root.mkdir(parents=True, exist_ok=True)
                pl = st.session_state["last_ocr_export_payload"] or {}
                stem = _safe_filename_component(str(pl.get("source_file") or "document.pdf"))
                spath = root / f"{stem}_split_settings.json"
                spath.write_text(
                    json.dumps(
                        st.session_state["last_split_settings_payload"],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                st.success("Split settings JSON saved.")
            except OSError as e:
                st.error(f"Could not save split settings: {e}")

    if st.session_state.get("last_split_settings_payload"):
        ss = st.session_state["last_split_settings_payload"]
        st.text_area(
            "Split settings preview",
            value=json.dumps(ss, ensure_ascii=False, indent=2),
            height=260,
            key="split_settings_preview_text",
        )
        st.download_button(
            "Download split settings JSON",
            data=json.dumps(ss, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{_safe_filename_component(str((st.session_state.get('last_ocr_export_payload') or {}).get('source_file') or 'document'))}_split_settings.json",
            mime="application/json",
            use_container_width=True,
            key="dl_split_settings_json",
        )

st.caption("Requires a local Tesseract installation (`pytesseract`).")
