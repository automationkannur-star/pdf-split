from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import fitz
import streamlit as st

from ocr_playground.engines import get_engine, list_engines
from ocr_playground.pdf_render import render_pdf_to_images, resize_image_max_side
from ocr_playground.pdf_split import (
    build_ranges_from_starts,
    chunks_to_zip_bytes,
    find_keyword_page_indices,
    split_pdf_by_ranges,
)


st.set_page_config(page_title="PDF OCR Playground", layout="wide")


def _bytes_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _safe_filename_component(name: str) -> str:
    base = name.rsplit(".", 1)[0] if "." in name else name
    base = re.sub(r"[^\w\-]+", "_", base, flags=re.UNICODE).strip("_") or "document"
    return base[:120]


def write_ocr_extract_json(
    output_dir: str,
    *,
    source_filename: str,
    engine_id: str,
    pdf_sha1_hex: str,
    start_page_abs: int,
    end_page_abs: int,
    elapsed_seconds: float,
    pages: list[Any],
) -> str:
    """
    Write OCR pages and metadata to a JSON file under ``output_dir``.
    Returns the absolute path of the written file.
    """
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename_component(source_filename)
    out_name = f"{stem}__{engine_id}__{pdf_sha1_hex[:12]}.json"
    out_path = root / out_name
    payload = {
        "schema_version": 1,
        "source_file": source_filename,
        "engine": engine_id,
        "pdf_sha1": pdf_sha1_hex,
        "page_range_1_indexed": {"start": start_page_abs + 1, "end": end_page_abs + 1},
        "elapsed_seconds": round(elapsed_seconds, 3),
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
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


st.title("PDF OCR Playground")
st.caption("Upload a PDF, switch OCR engines, and compare extracted text.")

with st.sidebar:
    st.subheader("Input")
    upload = st.file_uploader("Upload a PDF", type=["pdf"])

    st.subheader("Rendering")
    dpi = st.slider("DPI", min_value=100, max_value=300, value=150, step=25)
    page_range = st.text_input("Pages (e.g. `1-3` or empty for all)", value="")
    max_pages_per_run = st.number_input("Max pages per run (safety)", min_value=1, max_value=300, value=25, step=1)
    max_image_side = st.slider("Max image side px", min_value=1000, max_value=3000, value=1800, step=100)

    st.subheader("Engine-specific")
    tesseract_cmd = st.text_input("Tesseract binary path (optional)", value=os.environ.get("TESSERACT_CMD", ""))
    tesseract_config = st.text_input("Tesseract config (optional)", value="--psm 6")
    aws_region = st.text_input("AWS region (Textract)", value=os.environ.get("AWS_REGION", "ap-south-1"))
    aws_profile = st.text_input("AWS profile (optional)", value=os.environ.get("AWS_PROFILE", ""))
    aws_access_key_id = st.text_input("AWS access key id (optional)", value=os.environ.get("AWS_ACCESS_KEY_ID", ""))
    aws_secret_access_key = st.text_input(
        "AWS secret access key (optional)",
        value=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        type="password",
    )
    aws_session_token = st.text_input(
        "AWS session token (optional)",
        value=os.environ.get("AWS_SESSION_TOKEN", ""),
        type="password",
    )

    # Apply Tesseract path before checking engine availability so the dropdown
    # status reflects the currently entered path.
    if tesseract_cmd.strip():
        os.environ["TESSERACT_CMD"] = tesseract_cmd.strip()
    else:
        os.environ.pop("TESSERACT_CMD", None)

    if aws_region.strip():
        os.environ["AWS_REGION"] = aws_region.strip()
        os.environ["AWS_DEFAULT_REGION"] = aws_region.strip()
    else:
        os.environ.pop("AWS_REGION", None)
        os.environ.pop("AWS_DEFAULT_REGION", None)

    if aws_profile.strip():
        os.environ["AWS_PROFILE"] = aws_profile.strip()
    else:
        os.environ.pop("AWS_PROFILE", None)

    if aws_access_key_id.strip():
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id.strip()
    else:
        os.environ.pop("AWS_ACCESS_KEY_ID", None)

    if aws_secret_access_key.strip():
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key.strip()
    else:
        os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

    if aws_session_token.strip():
        os.environ["AWS_SESSION_TOKEN"] = aws_session_token.strip()
    else:
        os.environ.pop("AWS_SESSION_TOKEN", None)

    engines = list_engines()
    engine_labels = {
        e.id: (f"{e.name} ✅" if e.available else f"{e.name} ⚠️ (not installed/configured)")
        for e in engines
    }

    st.subheader("OCR Engine")
    engine_id = st.selectbox(
        "Choose engine",
        options=[e.id for e in engines],
        format_func=lambda x: engine_labels.get(x, x),
    )
    lang = st.text_input("Language (engine-specific)", value="eng")

    st.subheader("Keyword splitting")
    split_keyword = st.text_input("Split keyword", value="INVOICE NO")
    split_case_sensitive = st.checkbox("Case sensitive keyword match", value=False)
    split_try_rotations = st.checkbox("Try rotated pages for keyword detection", value=True)

    st.subheader("OCR JSON export")
    save_ocr_json = st.checkbox("Save OCR result as JSON to disk", value=False)
    ocr_json_folder = st.text_input(
        "Output folder for JSON",
        value=os.environ.get("OCR_JSON_OUTPUT_DIR", "ocr_json_output"),
        help="Absolute or relative path. Created if missing. Set OCR_JSON_OUTPUT_DIR env for a default.",
    )


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


def _keyword_in_text(text: str, keyword: str, *, case_sensitive: bool) -> bool:
    if case_sensitive:
        return keyword in (text or "")
    return keyword.lower() in (text or "").lower()


def _get_pdf_page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return int(doc.page_count)
    finally:
        doc.close()


def _extract_pdf_texts(pdf_bytes: bytes, start_page: int, end_page: int) -> list[str]:
    texts: list[str] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for p in range(start_page, end_page + 1):
            page = doc.load_page(p)
            texts.append(page.get_text("text") or "")
    finally:
        doc.close()
    return texts


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


if not upload:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_bytes = upload.getvalue()
pdf_hash = _bytes_sha1(pdf_bytes)
total_pages = _get_pdf_page_count(pdf_bytes)

first_page, last_page, range_err = _parse_page_range(page_range)
if range_err:
    st.error(range_err)
    st.stop()

engine = get_engine(engine_id)  # type: ignore[arg-type]
available, reason = engine.is_available()
if not available:
    st.warning(reason or "Selected engine is not available.")

col_left, col_right = st.columns([0.45, 0.55], gap="large")

with col_left:
    st.subheader("Input summary")
    start_page, end_page, was_capped = _apply_page_cap(
        total_pages,
        first_page,
        last_page,
        int(max_pages_per_run),
    )
    if was_capped:
        st.warning(
            f"Page window was capped to {max_pages_per_run} pages for stability. "
            f"Now processing pages {start_page + 1}-{end_page + 1}."
        )
    st.write(f"Total pages: **{total_pages}**")
    st.write(f"Pages to process: **{start_page + 1}-{end_page + 1}** ({end_page - start_page + 1} pages)")
    st.caption("Images are rendered only when OCR is executed.")
    show_images = st.checkbox("Show page images", value=False)
    if show_images:
        try:
            preview_end = min(end_page, start_page + 2)
            preview_images = render_pdf_to_images(pdf_bytes, dpi=dpi, first_page=start_page, last_page=preview_end)
            preview_images = [resize_image_max_side(img, max_side=int(max_image_side)) for img in preview_images]
            for i, img in enumerate(preview_images, start=start_page + 1):
                st.image(img, caption=f"Page {i}", use_container_width=True)
        except MemoryError:
            st.warning("Could not render preview due to memory pressure. Reduce DPI/max image side.")

with col_right:
    st.subheader("OCR output")

    config: dict[str, Any] = {}
    if engine_id == "tesseract":
        config["tesseract_config"] = tesseract_config
    elif engine_id == "textract":
        config["aws_region"] = aws_region.strip()
        config["aws_profile"] = aws_profile.strip()
        config["aws_access_key_id"] = aws_access_key_id.strip()
        config["aws_secret_access_key"] = aws_secret_access_key.strip()
        config["aws_session_token"] = aws_session_token.strip()

    run = st.button("Run OCR", type="primary", use_container_width=True)
    split_run = st.button("Split PDF by Keyword", use_container_width=True)
    if run:
        ok, reason = engine.is_available()
        if not ok:
            st.error(reason or "Engine not available.")
            st.stop()

        with st.spinner(f"Running {engine.display_name}..."):
            start = time.perf_counter()
            try:
                images = render_pdf_to_images(pdf_bytes, dpi=dpi, first_page=start_page, last_page=end_page)
                images = [resize_image_max_side(img, max_side=int(max_image_side)) for img in images]
                result = engine.extract_from_images(images, lang=lang.strip() or None, config=config)
            except MemoryError:
                st.error(
                    "Out of memory during OCR. Try lower DPI, lower max image side, or smaller page range."
                )
                st.stop()
            except Exception as e:
                st.error(f"OCR failed: {e}")
                st.stop()
            elapsed = time.perf_counter() - start

        st.success(f"Done in {elapsed:.2f}s (sum of pages {result.total_seconds:.2f}s).")

        if save_ocr_json and (ocr_json_folder or "").strip():
            try:
                written = write_ocr_extract_json(
                    (ocr_json_folder or "").strip(),
                    source_filename=upload.name,
                    engine_id=engine.id,
                    pdf_sha1_hex=pdf_hash,
                    start_page_abs=start_page,
                    end_page_abs=end_page,
                    elapsed_seconds=elapsed,
                    pages=result.pages,
                )
                st.info(f"OCR JSON saved to: `{written}`")
            except OSError as e:
                st.warning(f"Could not write OCR JSON to folder: {e}")
        elif save_ocr_json:
            st.warning("Set **Output folder for JSON** to save OCR results to disk.")

        st.download_button(
            "Download extracted text (.txt)",
            data=result.text.encode("utf-8"),
            file_name=f"{upload.name.rsplit('.', 1)[0]}__{engine.id}__{pdf_hash[:8]}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.text_area("Combined text", value=result.text, height=260)
        st.divider()
        for p in result.pages:
            with st.expander(f"Page {p.page_index + 1} ({p.seconds:.2f}s)", expanded=False):
                st.text_area("Text", value=p.text, height=200, key=f"page_{p.page_index}_{engine.id}_{pdf_hash}")

    if split_run:
        keyword = (split_keyword or "").strip()
        if not keyword:
            st.error("Enter a split keyword first (for example: INVOICE NO).")
            st.stop()

        with st.spinner("Reading PDF text (no image rendering)..."):
            page_texts = _extract_pdf_texts(pdf_bytes, start_page, end_page)

        local_hits = find_keyword_page_indices(
            page_texts,
            keyword,
            case_sensitive=split_case_sensitive,
        )

        # Optional OCR fallback for rotated/scanned pages only when needed.
        if split_try_rotations and not local_hits:
            ok, reason = engine.is_available()
            if ok:
                try:
                    with st.spinner("No text hits found. Trying OCR with rotations..."):
                        images = render_pdf_to_images(pdf_bytes, dpi=dpi, first_page=start_page, last_page=end_page)
                        images = [resize_image_max_side(img, max_side=int(max_image_side)) for img in images]
                        ocr_result = engine.extract_from_images(images, lang=lang.strip() or None, config=config)
                        local_hits = find_keyword_page_indices(
                            [p.text for p in ocr_result.pages],
                            keyword,
                            case_sensitive=split_case_sensitive,
                        )
                        if not local_hits:
                            for i, _ in enumerate(ocr_result.pages):
                                for angle in (90, 180, 270):
                                    rotated_img = images[i].rotate(angle, expand=True)
                                    rotated_result = engine.extract_from_images(
                                        [rotated_img],
                                        lang=lang.strip() or None,
                                        config=config,
                                    )
                                    rotated_text = rotated_result.pages[0].text if rotated_result.pages else ""
                                    if _keyword_in_text(rotated_text, keyword, case_sensitive=split_case_sensitive):
                                        local_hits.append(i)
                                        break
                except Exception as e:
                    st.error(f"OCR fallback failed: {e}")
                    st.stop()

                local_hits = sorted(set(local_hits))

        if not local_hits:
            st.warning("No pages matched the keyword. Nothing to split.")
            st.stop()

        # Map local indices to absolute source PDF pages.
        base_page = start_page
        abs_starts = [base_page + i for i in local_hits]
        ranges = build_ranges_from_starts(abs_starts, total_pages=end_page + 1)
        chunks = split_pdf_by_ranges(
            pdf_bytes,
            ranges,
            name_prefix=f"{upload.name.rsplit('.', 1)[0]}__split",
        )
        zip_name, zip_bytes = chunks_to_zip_bytes(
            chunks,
            zip_name_hint=f"{upload.name.rsplit('.', 1)[0]}__split__{pdf_hash[:8]}",
        )

        st.success(
            f"Created {len(chunks)} split PDFs from {len(local_hits)} keyword matches."
        )
        st.write(
            "Detected keyword on pages: "
            + ", ".join(str(base_page + i + 1) for i in local_hits)
        )
        st.download_button(
            "Download split PDFs (.zip)",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            use_container_width=True,
        )

    st.caption(
        "If an engine is marked as not installed, install its extra requirements, then restart the app."
    )

