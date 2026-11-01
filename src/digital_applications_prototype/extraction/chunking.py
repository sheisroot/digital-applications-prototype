"""Deterministic page-boundary chunks with bounded trailing context."""

from .models import DocumentChunk, DocumentEntry, DocumentMetadata, PageText


def chunk_pages(document: DocumentEntry, pages: list[PageText], *,
                max_chars: int = 12000, overlap_chars: int = 600) -> list[DocumentChunk]:
    """Group consecutive pages under a soft character target without splitting pages.

    Oversized pages form their own chunks; chunks containing only whitespace
    are omitted. Prepend at most overlap_chars from the preceding chunk's final
    nonempty page, retaining its page number and avoiding a cut inside a word.
    The overlap is additional to the soft target and never propagates recursively.
    Page numbering must be consecutive and start at one.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be nonnegative")
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
    if overlap_chars:
        for previous, following in zip(chunks, chunks[1:]):
            last = next(page for page in reversed(previous.pages) if page.text.strip())
            start = max(0, len(last.text) - overlap_chars)
            # Preserve an exact source suffix while omitting an incomplete first word.
            if start and not last.text[start - 1].isspace():
                while start < len(last.text) and not last.text[start].isspace():
                    start += 1
            tail = last.text[start:]
            if tail.strip():
                following.overlap = PageText(page_number=last.page_number, text=tail)
    return chunks
