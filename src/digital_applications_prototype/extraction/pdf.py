"""Lightweight PDF text extraction, with physical (1-based) page numbers."""

from pathlib import Path
import warnings

from pypdf import PdfReader

from .manifest import input_path
from .models import DocumentEntry, PageText


class PDFTextError(ValueError):
    pass


def extract_pages(root: Path, document: DocumentEntry) -> list[PageText]:
    """Extract input-only PDF text while preserving physical page numbers.

    Warn about empty pages and raise PDFTextError if the entire document has
    no extractable text, leaving OCR decisions to the caller.
    """
    path = input_path(root, document.path)
    with path.open("rb") as source:
        reader = PdfReader(source)
        pages = [PageText(page_number=number, text=page.extract_text() or "")
                 for number, page in enumerate(reader.pages, start=1)]
    if not any(page.text.strip() for page in pages):
        raise PDFTextError(f"{document.document_id}: no extractable text; inspect for OCR needs")
    empty = [page.page_number for page in pages if not page.text.strip()]
    if empty:
        warnings.warn(f"{document.document_id}: pages without text: {empty}; inspect for scans",
                      stacklevel=2)
    return pages
