"""Exact grounding after whitespace normalization; no fuzzy quote repair."""

import hashlib
import json
import re
from datetime import date

from .models import (
    DocumentChunk, DocumentMetadata, ImplementationObservation, ModelObservation,
    ObservationValidationResult, ValidationWarning,
)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _date_appears_in_quote(value: date, quote: str) -> bool:
    """Recognize a full ISO or English month-name date without inferring components.

    Permit common PDF separators and abbreviated month names. Ambiguous numeric
    day/month order and partial dates are deliberately not interpreted.
    """
    months = ("January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December")
    name = months[value.month - 1]
    month = rf"(?:{name}|{name[:3]}{'|Sept' if value.month == 9 else ''})\.?"
    day = rf"0?{value.day}" if value.day < 10 else str(value.day)
    numeric_month = rf"0?{value.month}" if value.month < 10 else str(value.month)
    separator = r"[\s,./-]+"
    patterns = (
        rf"{value.year:04d}\s*[-/.]\s*{numeric_month}\s*[-/.]\s*{day}",
        rf"{day}{separator}{month}{separator}{value.year:04d}",
        rf"{month}{separator}{day}{separator}{value.year:04d}",
    )
    return any(re.search(rf"(?<!\w){pattern}(?!\w)", quote, flags=re.IGNORECASE)
               for pattern in patterns)


def observation_signature(
    document_id: str, item: ModelObservation,
) -> tuple[str, str, tuple[tuple[int, str, str], ...]]:
    """Identify a claim by document, type, and order-independent exact evidence.

    Ignore generated wording, summaries, interpreted dates, and chunk boundaries.
    Each span retains its physical page, whitespace-normalized quote, and role.
    """
    spans = tuple(sorted((span.page, normalize_whitespace(span.quote), span.role.value)
                         for span in item.evidence.spans))
    return document_id, item.observation_type.value, spans


def validate_observation(
    item: ModelObservation, chunk: DocumentChunk, *, observation_index: int | None = None,
) -> ObservationValidationResult:
    """Reject ungrounded claims, clear unsupported optional dates, and retain warnings.

    Check all provenance before attempting date repairs. Never mutate the provider
    response, remove evidence, repair a quote, or quotation-validate the summary.
    """
    def warning(code: str, message: str, action: str, span_index: int | None = None) -> ValidationWarning:
        return ValidationWarning(code=code, message=message, action=action,
                                 chunk_start_page=chunk.start_page, chunk_end_page=chunk.end_page,
                                 observation_index=observation_index, evidence_span_index=span_index)

    def reject(code: str, message: str, span_index: int | None = None) -> ObservationValidationResult:
        return ObservationValidationResult(status="rejected", warnings=[
            warning(code, message, "observation_rejected", span_index)])

    for field in ("observation", "factor_description"):
        if not getattr(item, field).strip():
            return reject("observation_rejected", f"{field} must not be empty")
    if not item.evidence.spans:
        return reject("observation_rejected", "At least one evidence span is required")
    for index, span in enumerate(item.evidence.spans, start=1):
        quote = normalize_whitespace(span.quote)
        if not quote:
            return reject("ungrounded_quote", "Evidence quote must not be empty", index)
        pages = [page for page in chunk.source_pages if page.page_number == span.page]
        if not pages:
            return reject("invalid_evidence_page", f"Page {span.page} is not represented in the chunk", index)
        if not any(quote in normalize_whitespace(page.text) for page in pages):
            return reject("ungrounded_quote", f"Quote is not present on page {span.page}", index)

    warnings = []
    spans = []
    for index, span in enumerate(item.evidence.spans, start=1):
        if span.evidence_date is not None and not _date_appears_in_quote(
            span.evidence_date, normalize_whitespace(span.quote)
        ):
            warnings.append(warning(
                "unsupported_evidence_date",
                f"Date {span.evidence_date} is not established in its quote using a supported full-date format",
                "evidence_date_set_to_null", index))
            span = span.model_copy(update={"evidence_date": None})
        spans.append(span)
    repaired = item.model_copy(update={"evidence": item.evidence.model_copy(update={"spans": spans})})
    semantic = repaired.model_dump(mode="json")
    signature = observation_signature(chunk.document_id, item)
    digest = hashlib.sha256(json.dumps(signature, ensure_ascii=False).encode()).hexdigest()[:24]
    observation = ImplementationObservation(
        **semantic,
        **chunk.model_dump(include=set(DocumentMetadata.model_fields)),
        observation_id=f"{chunk.document_id}_{digest}",
        chunk_start_page=chunk.start_page, chunk_end_page=chunk.end_page,
    )
    return ObservationValidationResult(status="repaired" if warnings else "valid",
                                       observation=observation, warnings=warnings)


def deduplicate_observations(
    observations: list[ImplementationObservation],
) -> list[ImplementationObservation]:
    """Keep the first claim with the same document, type, and exact evidence set.

    Span order, harmless whitespace, and generated paraphrases do not affect the
    signature. Different observation types, pages, roles, or quotes remain separate.
    """
    unique = {}
    for item in observations:
        key = observation_signature(item.document_id, item)
        unique.setdefault(key, item)
    return list(unique.values())
