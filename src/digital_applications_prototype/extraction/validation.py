"""Exact grounding after whitespace normalization; no fuzzy quote repair."""

import hashlib
import json

from .models import DocumentChunk, DocumentMetadata, ImplementationObservation, ModelObservation


class GroundingError(ValueError):
    pass


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def validate_observation(item: ModelObservation, chunk: DocumentChunk) -> ImplementationObservation:
    """Validate a claim and attach deterministic document, page, and ID metadata.

    Require nonempty fields, a short label, and an exact quote match after
    whitespace normalization. Validate a supplied source page or infer a unique
    matching page; raise GroundingError rather than repair unsupported output.
    """
    for field in ("observation", "raw_factor_label", "raw_factor_description", "source_quote"):
        if not getattr(item, field).strip():
            raise GroundingError(f"{field} must not be empty")
    label_words = item.raw_factor_label.split()
    if not 1 <= len(label_words) <= 8 or len(item.raw_factor_label) > 100:
        raise GroundingError("raw_factor_label must be a short phrase (1–8 words, at most 100 characters)")
    quote = normalize_whitespace(item.source_quote)
    if quote not in normalize_whitespace(chunk.text):
        raise GroundingError("source_quote is not present in the source chunk")
    matching = [page.page_number for page in chunk.pages
                if quote in normalize_whitespace(page.text)]
    source_page = item.source_page
    if source_page is not None:
        if not chunk.start_page <= source_page <= chunk.end_page:
            raise GroundingError("source_page is outside the chunk range")
        if source_page not in matching:
            raise GroundingError("source_quote is not present on source_page")
    elif len(matching) == 1:
        source_page = matching[0]
    # A quote spanning pages or repeated on several pages retains a null source_page.
    semantic = item.model_dump(mode="json")
    semantic["source_page"] = source_page
    identity = {**semantic, "source_quote": quote,
                "document_id": chunk.document_id, "project_id": chunk.project_id,
                "chunk_start_page": chunk.start_page, "chunk_end_page": chunk.end_page}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    return ImplementationObservation(
        **semantic,
        **chunk.model_dump(include=set(DocumentMetadata.model_fields)),
        observation_id=f"{chunk.document_id}_{digest}",
        chunk_start_page=chunk.start_page, chunk_end_page=chunk.end_page,
    )
