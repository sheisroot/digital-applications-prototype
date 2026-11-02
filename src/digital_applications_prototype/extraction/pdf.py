"""Shared PDF loading with physical (1-based) pages and extractor provenance."""

import hashlib
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast

import pymupdf
from pypdf import PdfReader

from .manifest import input_path
from .models import DocumentEntry, PageText

ExtractorName = Literal["pymupdf", "pypdf"]


class PDFTextError(ValueError):
    pass


@dataclass(frozen=True)
class PDFText:
    pages: list[PageText]
    metadata: dict[str, Any]


def load_pdf(
    root: Path,
    document: DocumentEntry,
    *,
    extractor: ExtractorName = "pymupdf",
) -> PDFText:
    """Read/hash the same bytes; pypdf is available only for historical constraints.

    Text is unmodified and never concatenated across pages. Explicit PyMuPDF flags
    retain its ordinary text representation, without OCR, sorting or dehyphenation.
    """
    content = input_path(root, document.path).read_bytes()
    if extractor == "pymupdf":
        flags = pymupdf.TEXTFLAGS_TEXT
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            texts = [
                cast(str, page.get_text("text", sort=False, flags=flags))
                for page in pdf
            ]
        options = {"format": "text", "sort": False, "flags": flags}
    elif extractor == "pypdf":
        reader = PdfReader(BytesIO(content))
        texts = [page.extract_text() or "" for page in reader.pages]
        options = {"extraction_mode": "plain"}
    else:
        raise ValueError(f"Unsupported PDF extractor: {extractor}")
    pages = [
        PageText(page_number=number, text=text)
        for number, text in enumerate(texts, start=1)
    ]
    if not any(page.text.strip() for page in pages):
        raise PDFTextError(
            f"{document.document_id}: no extractable text; inspect for OCR needs"
        )
    empty = [page.page_number for page in pages if not page.text.strip()]
    if empty:
        warnings.warn(
            f"{document.document_id}: pages without text: {empty}; inspect for scans",
            stacklevel=2,
        )
    return PDFText(
        pages,
        {
            "pdf_sha256": hashlib.sha256(content).hexdigest(),
            "extractor": {
                "name": extractor,
                "version": version(extractor),
                "options": options,
            },
        },
    )


def extract_pages(root: Path, document: DocumentEntry) -> list[PageText]:
    """Compatibility entry point using the shared, provenance-aware PyMuPDF loader."""
    return load_pdf(root, document).pages
