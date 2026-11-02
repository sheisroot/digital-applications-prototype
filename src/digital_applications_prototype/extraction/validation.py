"""Page-constrained exact grounding with conservative PDF spacing equivalences."""

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


VALIDATOR_VERSION = "stage1-quotes-v2"


def normalize_quote_text(text: str) -> str:
    """Keep every non-whitespace character and all ordinary word boundaries.

    Reproduced PDF artifacts include ``on -site``, ``home- grown`` and ``paid .``.
    Only an ASCII alphabetic hyphen's surrounding space and space before punctuation
    immediately following an ASCII letter are optional. Numeric separators/signs, word
    fragments (``repo rted``), case, and punctuation remain significant.
    This representation is never stored as the original quote or source text.
    """
    text = normalize_whitespace(text)
    text = re.sub(r"(?<=[A-Za-z])\s*-\s*(?=[A-Za-z])", "-", text)
    return re.sub(r"(?<=[A-Za-z])\s+([,.;:!?])", r"\1", text)


def quote_matches(quote: str, source: str) -> bool:
    normalized = normalize_quote_text(quote)
    return bool(normalized) and normalized in normalize_quote_text(source)


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
    additional_source: DocumentChunk | None = None,
) -> ObservationValidationResult:
    """Reject ungrounded claims, clear unsupported optional dates, and retain warnings.

    Check all provenance before attempting date repairs. Never mutate the provider
    response, remove evidence, repair a quote, or quotation-validate the summary.
    """
    if additional_source is not None and (
        additional_source.document_id != chunk.document_id
        or [page.page_number for page in additional_source.pages] != [page.page_number for page in chunk.pages]
        or (additional_source.overlap.page_number if additional_source.overlap else None)
        != (chunk.overlap.page_number if chunk.overlap else None)
    ):
        raise ValueError("Additional source must preserve the original document/page/overlap constraints")
    source_pages = [*chunk.source_pages, *(additional_source.source_pages if additional_source else [])]

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
        pages = [page for page in source_pages if page.page_number == span.page]
        if not pages:
            return reject("invalid_evidence_page", f"Page {span.page} is not represented in the chunk", index)
        if not any(quote_matches(span.quote, page.text) for page in pages):
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
    duplicates = observation_duplicate_indexes(observations)
    return [item for index, item in enumerate(observations) if index not in duplicates]


def observation_duplicate_indexes(
    observations: list[ImplementationObservation],
) -> dict[int, int]:
    """Map each duplicate's list index to its first occurrence, without deleting it."""
    first = {}
    duplicates = {}
    for index, item in enumerate(observations):
        key = observation_signature(item.document_id, item)
        if key in first:
            duplicates[index] = first[key]
        else:
            first[key] = index
    return duplicates
