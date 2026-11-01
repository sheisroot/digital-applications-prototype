"""Deterministic page-boundary chunks with a soft character target, no overlap."""

from .models import DocumentChunk, DocumentEntry, DocumentMetadata, PageText


def chunk_pages(document: DocumentEntry, pages: list[PageText], *,
                max_chars: int = 12000) -> list[DocumentChunk]:
    """Group consecutive pages under a soft character target without splitting pages.

    Oversized pages form their own chunks; chunks containing only whitespace
    are omitted. Page numbering must be consecutive and start at one.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if [page.page_number for page in pages] != list(range(1, len(pages) + 1)):
        raise ValueError("Pages must be consecutive and start at 1")
    metadata = document.model_dump(include=set(DocumentMetadata.model_fields))
    chunks: list[DocumentChunk] = []
    current: list[PageText] = []
    size = 0
    for page in pages:
        added = len(page.text) + (2 if current else 0)
        if current and size + added > max_chars:
            if any(p.text.strip() for p in current):
                chunks.append(DocumentChunk(**metadata, pages=current))
            current, size = [], 0
        current.append(page)
        size += len(page.text) + (2 if len(current) > 1 else 0)
    if current and any(p.text.strip() for p in current):
        chunks.append(DocumentChunk(**metadata, pages=current))
    return chunks
