from __future__ import annotations

import io
import os
import time
from typing import Any

from PIL import Image

from ..models import OcrDocumentResult, OcrPageResult
from .base import OcrEngine


class TextractEngine(OcrEngine):
    @property
    def id(self) -> str:
        return "textract"

    @property
    def display_name(self) -> str:
        return "AWS Textract"

    def is_available(self) -> tuple[bool, str | None]:
        try:
            import boto3  # noqa: F401
        except Exception as e:
            return False, f"boto3 not installed. Install `requirements-textract.txt`. Import error: {e}"

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        profile = os.environ.get("AWS_PROFILE")
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY")

        if not region:
            return False, "Set AWS region (AWS_REGION) to use Textract."
        if not profile and not (access_key and secret):
            return False, "Set AWS_PROFILE or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY to use Textract."
        return True, None

    def extract_from_images(
        self,
        images: list[Image.Image],
        *,
        lang: str | None = None,  # Textract auto-detects language/script
        config: dict[str, Any] | None = None,
    ) -> OcrDocumentResult:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

        cfg = config or {}
        region = str(cfg.get("aws_region") or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "")
        profile = str(cfg.get("aws_profile") or os.environ.get("AWS_PROFILE") or "")
        access_key = str(cfg.get("aws_access_key_id") or os.environ.get("AWS_ACCESS_KEY_ID") or "")
        secret_key = str(cfg.get("aws_secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY") or "")
        session_token = str(cfg.get("aws_session_token") or os.environ.get("AWS_SESSION_TOKEN") or "")

        if not region:
            raise RuntimeError("AWS region is required for Textract.")

        session_kwargs: dict[str, Any] = {"region_name": region}
        if profile:
            session_kwargs["profile_name"] = profile
        elif access_key and secret_key:
            session_kwargs["aws_access_key_id"] = access_key
            session_kwargs["aws_secret_access_key"] = secret_key
            if session_token:
                session_kwargs["aws_session_token"] = session_token

        session = boto3.Session(**session_kwargs)
        client = session.client("textract", region_name=region)

        pages: list[OcrPageResult] = []
        for i, img in enumerate(images):
            start = time.perf_counter()
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            try:
                response = client.detect_document_text(Document={"Bytes": image_bytes})
            except NoCredentialsError as e:
                raise RuntimeError(
                    "AWS Textract auth failed: no credentials found. "
                    "Set AWS profile or access key/secret in app settings."
                ) from e
            except ClientError as e:
                err = e.response.get("Error", {}) if hasattr(e, "response") else {}
                code = err.get("Code", "UnknownClientError")
                msg = err.get("Message", str(e))
                raise RuntimeError(
                    f"AWS Textract request failed [{code}]: {msg}. "
                    "Check region, credentials, and textract permissions."
                ) from e
            except BotoCoreError as e:
                raise RuntimeError(
                    f"AWS Textract connection/config error: {e}. "
                    "Verify network access and AWS region."
                ) from e

            lines = [
                block.get("Text", "")
                for block in response.get("Blocks", [])
                if block.get("BlockType") == "LINE" and block.get("Text")
            ]
            text = "\n".join(lines).strip()
            seconds = time.perf_counter() - start
            pages.append(OcrPageResult(page_index=i, text=text, seconds=seconds))

        return OcrDocumentResult(pages=pages)

