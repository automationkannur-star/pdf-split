from __future__ import annotations

import hashlib
import os
import time
from typing import Any

import streamlit as st

from ocr_playground.engines import get_engine, list_engines
from ocr_playground.pdf_render import render_pdf_to_images
from ocr_playground.pdf_split import (
    build_ranges_from_starts,
    chunks_to_zip_bytes,
    find_keyword_page_indices,
    split_pdf_by_ranges,
)


st.set_page_config(page_title="PDF OCR Playground", layout="wide")


def _bytes_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


st.title("PDF OCR Playground")
st.caption("Upload a PDF, switch OCR engines, and compare extracted text.")

with st.sidebar:
    st.subheader("Input")
    upload = st.file_uploader("Upload a PDF", type=["pdf"])

    st.subheader("Rendering")
    dpi = st.slider("DPI", min_value=100, max_value=300, value=150, step=25)
    page_range = st.text_input("Pages (e.g. `1-3` or empty for all)", value="")

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


if not upload:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_bytes = upload.getvalue()
pdf_hash = _bytes_sha1(pdf_bytes)

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
    st.subheader("Rendered pages")
    with st.spinner("Rendering PDF pages..."):
        images = render_pdf_to_images(pdf_bytes, dpi=dpi, first_page=first_page, last_page=last_page)
    st.write(f"Pages rendered: **{len(images)}** (DPI={dpi})")
    show_images = st.checkbox("Show page images", value=False)
    if show_images:
        for i, img in enumerate(images, start=1 if first_page is None else first_page + 1):
            st.image(img, caption=f"Page {i}", use_container_width=True)

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
    if run or split_run:
        ok, reason = engine.is_available()
        if not ok:
            st.error(reason or "Engine not available.")
            st.stop()

        with st.spinner(f"Running {engine.display_name}..."):
            start = time.perf_counter()
            result = engine.extract_from_images(images, lang=lang.strip() or None, config=config)
            elapsed = time.perf_counter() - start

        if run:
            st.success(f"Done in {elapsed:.2f}s (sum of pages {result.total_seconds:.2f}s).")

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

            local_hits = find_keyword_page_indices(
                [p.text for p in result.pages],
                keyword,
                case_sensitive=split_case_sensitive,
            )

            # Fallback for rotated/scanned pages: retry only non-hit pages at
            # right-angle rotations and mark hit as soon as keyword is detected.
            if split_try_rotations:
                with st.spinner("Checking rotated pages for missed keyword matches..."):
                    for i, page in enumerate(result.pages):
                        if i in local_hits:
                            continue

                        matched = False
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
                                matched = True
                                break
                        if matched:
                            continue

                local_hits = sorted(set(local_hits))

            if not local_hits:
                st.warning("No pages matched the keyword. Nothing to split.")
                st.stop()

            # Map OCR result page indices to absolute source PDF pages.
            base_page = first_page or 0
            abs_starts = [base_page + i for i in local_hits]
            ranges = build_ranges_from_starts(abs_starts, total_pages=len(images) + base_page)
            chunks = split_pdf_by_ranges(
                pdf_bytes,
                ranges,
                name_prefix=f"{upload.name.rsplit('.', 1)[0]}__{engine.id}",
            )
            zip_name, zip_bytes = chunks_to_zip_bytes(
                chunks,
                zip_name_hint=f"{upload.name.rsplit('.', 1)[0]}__{engine.id}__{pdf_hash[:8]}",
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

