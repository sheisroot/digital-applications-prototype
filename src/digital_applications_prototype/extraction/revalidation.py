"""Offline revalidation of one explicit persisted document/run; never calls a model."""

import argparse
import asyncio
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from digital_applications_prototype.persistence.database import Database
from digital_applications_prototype.persistence.models import (
    ChunkSource,
    ExtractionChunk,
    ExtractionRun,
    Observation,
    ValidationRevision,
)
from digital_applications_prototype.persistence.repository import (
    ExtractionRepository,
    utc_now,
)

from .chunking import chunk_pages
from .manifest import input_path, select_documents
from .models import (
    DocumentChunk,
    DocumentMetadata,
    ModelObservation,
    ObservationValidationResult,
    PageText,
)
from .pdf import ExtractorName, PDFText, load_pdf
from .persistence import reserve_run, write_observations, write_report
from .validation import (
    VALIDATOR_VERSION,
    normalize_quote_text,
    normalize_whitespace,
    observation_duplicate_indexes,
    quote_matches,
    validate_observation,
)

_VALIDATION_CODES = {
    "ungrounded_quote",
    "invalid_evidence_page",
    "observation_rejected",
    "unsupported_evidence_date",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def backup_database(source: Path, destination: Path) -> None:
    """SQLite's online backup includes committed WAL contents; never overwrite a backup."""
    source, destination = source.resolve(strict=True), destination.absolute()
    if source == destination.resolve():
        raise ValueError("Backup must differ from the source database")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb"):
        pass
    try:
        with (
            closing(
                sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)
            ) as original,
            closing(sqlite3.connect(destination)) as backup,
        ):
            original.backup(backup)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


@dataclass
class _Plan:
    fingerprint: str
    states: dict[int, dict[str, Any]]
    warnings: dict[int, list[dict[str, Any]]]
    changes: list[dict[str, Any]]
    summary: dict[str, Any]
    source: dict[str, Any]
    diagnostics: list[dict[str, Any]]


def _columns(row: Any) -> dict[str, Any]:
    return {column.key: getattr(row, column.key) for column in row.__table__.columns}


def _state(row: Observation) -> dict[str, Any]:
    return {
        "validation_status": row.validation_status,
        "provenance_id": row.provenance_id,
        "duplicate_of_observation_id": row.duplicate_of_observation_id,
        "evidence_dates": [
            span.evidence_date.isoformat() if span.evidence_date else None
            for span in row.evidence_spans
        ],
    }


def _counts(
    states: dict[int, dict[str, Any]],
    warnings: dict[int, list[dict[str, Any]]],
    rows: list[Observation],
) -> dict[str, Any]:
    statuses = Counter(state["validation_status"] for state in states.values())
    duplicates = sum(
        state["duplicate_of_observation_id"] is not None for state in states.values()
    )
    rejected = {
        (row.chunk_id, row.observation_index)
        for row in rows
        if states[row.observation_id]["validation_status"] == "rejected"
    }
    reasons = Counter(
        warning["code"]
        for chunk_id, items in warnings.items()
        for warning in items
        if (chunk_id, warning.get("observation_index")) in rejected
        and warning.get("action") == "observation_rejected"
    )
    return {
        "returned": len(states),
        "valid": statuses["valid"],
        "repaired": statuses["repaired"],
        "rejected": statuses["rejected"],
        "duplicates": duplicates,
        "usable": statuses["valid"] + statuses["repaired"] - duplicates,
        "rejection_reasons": dict(sorted(reasons.items())),
    }


async def _sources(
    session: AsyncSession,
    root: Path,
    document_id: str,
    chunks: list[ExtractionChunk],
    *,
    source_pdf: Path | None,
    source_sha256: str | None,
    max_chars: int | None,
    overlap_chars: int | None,
    refresh_pdf_text: bool,
    historical_extractor: ExtractorName,
) -> tuple[
    dict[int, DocumentChunk],
    dict[str, Any],
    list[dict[str, Any]],
    dict[int, DocumentChunk],
]:
    connection = await session.connection()
    has_sources = await connection.run_sync(
        lambda sync: inspect(sync).has_table(ChunkSource.__tablename__)
    )
    stored = (
        list(
            await session.scalars(
                select(ChunkSource)
                .where(ChunkSource.chunk_id.in_([chunk.chunk_id for chunk in chunks]))
                .order_by(ChunkSource.chunk_id)
            )
        )
        if has_sources
        else []
    )
    originals = {
        row.chunk_id: DocumentChunk.model_validate(row.source_json) for row in stored
    }
    source_records = [_columns(row) for row in stored]
    additional_sources: dict[int, DocumentChunk] = {}
    missing = [chunk for chunk in chunks if chunk.chunk_id not in originals]
    retained_metadata = {
        _digest(row.source_json.get("pdf_source")): row.source_json.get("pdf_source")
        for row in stored
    }
    identity: dict[str, Any] = {
        "kind": "retained_chunk_text",
        "chunks": len(originals),
        "retained_pdf_metadata": list(retained_metadata.values()),
    }
    if missing or refresh_pdf_text:
        if source_pdf is None or source_sha256 is None:
            if not missing:
                raise ValueError(
                    "--refresh-pdf-text requires --source-pdf and an expected --source-sha256"
                )
            raise ValueError(
                "Original chunk text/PDF fingerprint was not retained for this run. "
                "Source identity cannot be established automatically: supply --source-pdf "
                "and an independently verified expected --source-sha256. A filename alone is insufficient."
            )
        if missing and (max_chars is None or overlap_chars is None):
            raise ValueError(
                "Legacy source reconstruction requires explicit --max-chars and --overlap-chars"
            )
        document = select_documents(document=document_id)[0]
        pdf = input_path(root, document.path)
        if source_pdf.resolve(strict=True) != pdf:
            raise ValueError(f"--source-pdf must be the selected manifest input: {pdf}")
        source = load_pdf(root, document)
        actual_sha = source.metadata["pdf_sha256"]
        if actual_sha != source_sha256.lower():
            raise ValueError(
                f"Source PDF SHA-256 mismatch: expected {source_sha256}, got {actual_sha}"
            )
        for metadata in retained_metadata.values():
            if metadata and metadata["pdf_sha256"] != actual_sha:
                raise ValueError("Source PDF differs from the retained source SHA-256")
        historical_metadata = None
        if missing:
            assert max_chars is not None and overlap_chars is not None
            historical = load_pdf(root, document, extractor=historical_extractor)
            if historical.metadata["pdf_sha256"] != actual_sha:
                raise ValueError(
                    "PDF changed while reconstructing historical constraints"
                )
            rebuilt = chunk_pages(
                document,
                historical.pages,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
            if len(rebuilt) != len(chunks) or any(
                row.chunk_index != index
                or (row.start_page, row.end_page) != (chunk.start_page, chunk.end_page)
                for index, (row, chunk) in enumerate(zip(chunks, rebuilt), start=1)
            ):
                raise ValueError(
                    "Historical chunk/page mapping differs from the persisted run; "
                    "faithful reconstruction requires retained sources or the original extractor/settings"
                )
            for row, chunk in zip(chunks, rebuilt, strict=True):
                originals.setdefault(row.chunk_id, chunk)
            historical_metadata = historical.metadata["extractor"]
        additional_sources, projection = _project_sources(originals, source)
        identity = {
            "kind": "pdf_reextraction",
            **source.metadata,
            "identity_basis": "caller-supplied expected hash, checked against retained PDF metadata when present",
            "historical_chunk_extractor": historical_metadata,
            "historical_chunk_parameters": {
                "max_chars": max_chars,
                "overlap_chars": overlap_chars,
            }
            if missing
            else None,
            "retained_chunks": len(stored),
            "retained_pdf_metadata": list(retained_metadata.values()),
            "constraints": "recorded pages and original overlap boundaries; no new chunking",
            "matching_policy": "complete span in historical text first, then constrained PyMuPDF text",
            "historical_text": "retained where available; otherwise reconstructed, not original model input",
            "overlap_projection": projection,
        }
    for row in chunks:
        chunk = originals[row.chunk_id]
        if chunk.document_id != document_id or (chunk.start_page, chunk.end_page) != (
            row.start_page,
            row.end_page,
        ):
            raise ValueError(f"Stored source metadata differs for chunk {row.chunk_id}")
    identity["source_text_sha256"] = _digest(
        {
            "historical": {
                key: value.model_dump(mode="json")
                for key, value in sorted(originals.items())
            },
            "additional": {
                key: value.model_dump(mode="json")
                for key, value in sorted(additional_sources.items())
            },
        }
    )
    return originals, identity, source_records, additional_sources


def _project_overlap(old: str, full_old: str, new: str) -> str:
    """Locate an old suffix in new page text, without widening historical access.

    Exact, unique normalized boundary anchors locate the proposed interval. Its
    complete non-whitespace character sequence must equal the original suffix.
    That equality ONLY proves the interval's boundaries: quotes still pass the
    unchanged strict matcher against actual PyMuPDF text, with word spaces intact.
    Refuse ambiguous/reordered/changed intervals rather than expose a whole page.
    """
    if not full_old.endswith(old):
        raise ValueError(
            "Historical overlap is not an exact suffix of its original full page"
        )
    if full_old.strip() == old.strip():
        return new  # The original overlap already exposed the entire physical page.
    old_normalized, new_normalized = (
        normalize_quote_text(old),
        normalize_quote_text(new),
    )
    size = min(48, len(old_normalized))
    if not size:
        raise ValueError("Cannot project an empty historical overlap")
    first, last = old_normalized[:size], old_normalized[-size:]
    if new_normalized.count(first) != 1 or new_normalized.count(last) != 1:
        raise ValueError(
            "Cannot faithfully project overlap: boundary anchors are absent or ambiguous"
        )
    start, end = new_normalized.index(first), new_normalized.index(last) + len(last)
    if end < start + len(first):
        raise ValueError(
            "Cannot faithfully project overlap: boundary anchors are reordered"
        )
    offsets = [index for index, char in enumerate(new) if not char.isspace()]
    start_index = sum(not char.isspace() for char in new_normalized[:start])
    end_index = sum(not char.isspace() for char in new_normalized[:end])
    candidate = new[offsets[start_index] : offsets[end_index - 1] + 1]
    if "".join(old.split()) != "".join(candidate.split()):
        raise ValueError(
            "Cannot faithfully project overlap: interval content differs beyond whitespace"
        )
    return candidate


def _project_sources(
    originals: dict[int, DocumentChunk],
    source: PDFText,
) -> tuple[dict[int, DocumentChunk], list[dict[str, Any]]]:
    pages = {page.page_number: page for page in source.pages}
    original_pages = {
        page.page_number: page.text
        for chunk in originals.values()
        for page in chunk.pages
    }
    projected = {}
    details = []
    for key, chunk in originals.items():
        if any(page.page_number not in pages for page in chunk.source_pages):
            raise ValueError(f"PDF does not contain the recorded pages for chunk {key}")
        overlap = None
        if chunk.overlap is not None:
            number = chunk.overlap.page_number
            if number not in original_pages:
                raise ValueError(
                    f"Missing original full page {number}; overlap cannot be projected"
                )
            try:
                text = _project_overlap(
                    chunk.overlap.text, original_pages[number], pages[number].text
                )
            except ValueError as error:
                raise ValueError(
                    f"Chunk {key}, PDF page {number}: {error}; use retained source text"
                ) from error
            overlap = PageText(page_number=number, text=text)
            details.append(
                {
                    "chunk_id": key,
                    "page": number,
                    "original_chars": len(chunk.overlap.text),
                    "projected_chars": len(text),
                }
            )
        projected[key] = chunk.model_copy(
            update={
                "pages": [pages[page.page_number] for page in chunk.pages],
                "overlap": overlap,
            }
        )
    return projected, details


def _mismatch_details(
    item: ModelObservation,
    chunk: DocumentChunk,
    additional_source: DocumentChunk | None = None,
) -> list[dict[str, Any]]:
    """Optional excerpts only. SequenceMatcher locates diagnostics and NEVER accepts evidence."""
    details = []
    for index, span in enumerate(item.evidence.spans, start=1):
        pages = [
            page
            for source in [chunk, *([additional_source] if additional_source else [])]
            for page in source.source_pages
            if page.page_number == span.page
        ]
        if any(quote_matches(span.quote, page.text) for page in pages):
            continue
        quote = normalize_quote_text(span.quote)
        source = normalize_quote_text(pages[0].text) if pages else ""
        match = SequenceMatcher(
            None, quote, source, autojunk=False
        ).find_longest_match()
        start = max(0, match.b - 60)
        details.append(
            {
                "span_index": index,
                "page": span.page,
                "original_quote": span.quote,
                "whitespace_quote": normalize_whitespace(span.quote),
                "normalized_quote": quote,
                "normalized_source_excerpt": source[start : start + 300],
                "page_in_chunk": bool(pages),
                "matching_other_supplied_pages": [
                    page.page_number
                    for page in chunk.source_pages
                    if page.page_number != span.page
                    and quote_matches(span.quote, page.text)
                ],
            }
        )
    return details


async def _analyze(
    session: AsyncSession,
    root: Path,
    document_id: str,
    run_id: str,
    *,
    diagnostics: bool = False,
    **source_options: Any,
) -> _Plan:
    run = await session.get(ExtractionRun, run_id)
    if run is None or run.document_id != document_id:
        raise ValueError(
            "The selected database/document/run does not identify an existing run"
        )
    chunks = list(
        await session.scalars(
            select(ExtractionChunk)
            .where(ExtractionChunk.run_id == run_id)
            .order_by(ExtractionChunk.chunk_index)
        )
    )
    if run.status is None or any(
        chunk.status in ("pending", "running") for chunk in chunks
    ):
        raise ValueError(
            "Revalidation requires a finalized run with no pending/running chunks"
        )
    rows = list(
        await session.scalars(
            select(Observation)
            .join(ExtractionChunk)
            .where(Observation.run_id == run_id)
            .options(selectinload(Observation.evidence_spans))
            .order_by(ExtractionChunk.chunk_index, Observation.observation_index)
        )
    )
    sources, identity, source_records, additional_sources = await _sources(
        session, root, document_id, chunks, **source_options
    )
    fingerprint = _digest(
        [
            _columns(run),
            [_columns(chunk) for chunk in chunks],
            source_records,
            [
                (_columns(row), [_columns(span) for span in row.evidence_spans])
                for row in rows
            ],
        ]
    )
    before = {row.observation_id: _state(row) for row in rows}
    old_warnings = {chunk.chunk_id: chunk.warnings_json for chunk in chunks}
    warnings = {
        key: [warning for warning in items if warning["code"] not in _VALIDATION_CODES]
        for key, items in old_warnings.items()
    }
    states = {}
    outcomes: dict[int, ObservationValidationResult] = {}
    details = []
    for row in rows:
        chunk = sources[row.chunk_id]
        if any(
            getattr(row, field) != getattr(chunk, field)
            for field in DocumentMetadata.model_fields
        ):
            raise ValueError(
                f"Observation {row.observation_id} has inconsistent document metadata"
            )
        item = ModelObservation.model_validate(row.raw_observation_json)
        if [(span.quote, span.page, span.role) for span in row.evidence_spans] != [
            (span.quote, span.page, span.role.value) for span in item.evidence.spans
        ]:
            raise ValueError(
                f"Persisted evidence differs from raw model output for observation {row.observation_id}"
            )
        outcome = validate_observation(
            item,
            chunk,
            observation_index=row.observation_index,
            additional_source=additional_sources.get(row.chunk_id),
        )
        # Unknown future/manual rejection rules must never be erased by this command.
        other_rejections = [
            warning
            for warning in warnings[row.chunk_id]
            if warning.get("observation_index") == row.observation_index
            and warning.get("action") == "observation_rejected"
        ]
        if other_rejections:
            outcome = ObservationValidationResult(
                status="rejected", warnings=outcome.warnings
            )
        outcomes[row.observation_id] = outcome
        warnings[row.chunk_id].extend(
            warning.model_dump(mode="json") for warning in outcome.warnings
        )
        item_after = outcome.observation or item
        states[row.observation_id] = {
            "validation_status": outcome.status,
            "provenance_id": outcome.observation.observation_id
            if outcome.observation
            else None,
            "duplicate_of_observation_id": None,
            "evidence_dates": [
                span.evidence_date.isoformat() if span.evidence_date else None
                for span in item_after.evidence.spans
            ],
        }
        if diagnostics and outcome.status == "rejected":
            details.append(
                {
                    "observation_id": row.observation_id,
                    "spans": _mismatch_details(
                        item, chunk, additional_sources.get(row.chunk_id)
                    ),
                }
            )
    kept = [row for row in rows if outcomes[row.observation_id].observation is not None]
    observations = [outcomes[row.observation_id].observation for row in kept]
    duplicates = observation_duplicate_indexes(
        [item for item in observations if item is not None]
    )
    for index, first in duplicates.items():
        states[kept[index].observation_id]["duplicate_of_observation_id"] = kept[
            first
        ].observation_id
    changes = [
        {"observation_id": key, "before": before[key], "after": value}
        for key, value in states.items()
        if before[key] != value
    ]
    changes.extend(
        {"chunk_id": key, "warnings_before": old_warnings[key], "warnings_after": value}
        for key, value in warnings.items()
        if old_warnings[key] != value
    )
    recovered = [
        key
        for key in states
        if before[key]["validation_status"] == "rejected"
        and states[key]["validation_status"] != "rejected"
    ]
    newly_rejected = [
        key
        for key in states
        if before[key]["validation_status"] != "rejected"
        and states[key]["validation_status"] == "rejected"
    ]
    return _Plan(
        fingerprint,
        states,
        warnings,
        changes,
        {
            "before": _counts(before, old_warnings, rows),
            "after": _counts(states, warnings, rows),
            "recovered_observation_ids": recovered,
            "newly_rejected_observation_ids": newly_rejected,
        },
        identity,
        details,
    )


async def _export(
    database: Database, root: Path, document_id: str, run_id: str
) -> dict[str, str]:
    """Use the existing writers; distinct content-addressed artifacts survive repeat applies."""
    repository = ExtractionRepository(database.sessions)
    report = await repository.get_run_summary(run_id)
    observations = await repository.get_clean_observations(run_id)
    content_id = _digest(
        [
            report.model_dump(mode="json"),
            [item.model_dump(mode="json") for item in observations],
        ]
    )[:16]
    artifact_id = f"{run_id}__revalidated_{VALIDATOR_VERSION}_{content_id}"
    document = select_documents(document=document_id)[0]
    directory = root.resolve() / "data/derived/extracted" / document.country
    stem = f"{document_id}__{artifact_id}"
    jsonl, report_path = directory / f"{stem}.jsonl", directory / f"{stem}.report.json"
    report.jsonl_file = jsonl.name
    expected = {
        jsonl: "".join(item.model_dump_json() + "\n" for item in observations),
        report_path: report.model_dump_json(indent=2) + "\n",
    }
    for path, content in expected.items():
        if path.is_symlink() or (
            path.exists() and path.read_text(encoding="utf-8") != content
        ):
            raise FileExistsError(
                f"Refusing to replace a changed revalidated artifact: {path}"
            )
    if not any(path.exists() for path in expected):
        artifacts = reserve_run(root, document, datetime.now(UTC), run_id=artifact_id)
        try:
            write_observations(artifacts.jsonl_path, observations)
            write_report(artifacts.report_path, report)
        finally:
            artifacts.release()
    else:
        # Complete an interrupted export while preserving any already published file.
        if not jsonl.exists():
            write_observations(jsonl, observations)
        if not report_path.exists():
            write_report(report_path, report)
    return {"jsonl": str(jsonl), "report": str(report_path)}


async def revalidate(
    database_path: Path,
    document_id: str,
    run_id: str,
    *,
    root: Path | None = None,
    apply: bool = False,
    backup_path: Path | None = None,
    source_pdf: Path | None = None,
    source_sha256: str | None = None,
    max_chars: int | None = None,
    overlap_chars: int | None = None,
    diagnostics: bool = False,
    refresh_pdf_text: bool = False,
    historical_extractor: ExtractorName = "pypdf",
) -> dict[str, Any]:
    """Preview read-only, or back up then atomically update only derived validation state."""
    root = Path.cwd() if root is None else root
    database_path = database_path.resolve(strict=True)
    options: dict[str, Any] = {
        "source_pdf": source_pdf,
        "source_sha256": source_sha256,
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "refresh_pdf_text": refresh_pdf_text,
        "historical_extractor": historical_extractor,
    }
    readonly = Database(database_path, read_only=True)
    try:
        async with readonly.sessions() as session:
            await session.execute(text("BEGIN"))
            plan = await _analyze(
                session, root, document_id, run_id, diagnostics=diagnostics, **options
            )
    finally:
        await readonly.close()
    result = {
        "mode": "apply" if apply else "dry-run",
        "document_id": document_id,
        "run_id": run_id,
        "validator_version": VALIDATOR_VERSION,
        "source": plan.source,
        **plan.summary,
        "validation_changes": [
            {
                "observation_id": change["observation_id"],
                "before": change["before"]["validation_status"],
                "after": change["after"]["validation_status"],
            }
            for change in plan.changes
            if "observation_id" in change
            and change["before"]["validation_status"]
            != change["after"]["validation_status"]
        ],
        "duplicate_link_changes": [
            {
                "observation_id": change["observation_id"],
                "before": change["before"]["duplicate_of_observation_id"],
                "after": change["after"]["duplicate_of_observation_id"],
            }
            for change in plan.changes
            if "observation_id" in change
            and change["before"]["duplicate_of_observation_id"]
            != change["after"]["duplicate_of_observation_id"]
        ],
        "evidence_date_changes": [
            {
                "observation_id": change["observation_id"],
                "before": change["before"]["evidence_dates"],
                "after": change["after"]["evidence_dates"],
            }
            for change in plan.changes
            if "observation_id" in change
            and change["before"]["evidence_dates"] != change["after"]["evidence_dates"]
        ],
    }
    if diagnostics:
        result["diagnostics"] = plan.diagnostics
    if not apply:
        return result
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = backup_path or database_path.with_name(
        f"{database_path.name}.before-revalidation-{stamp}"
    )
    backup_database(database_path, backup_path)
    result["backup"] = str(backup_path)
    database = Database(database_path)
    committed = False
    try:
        # Existing convention: additive create_all, no column migrations.
        await database.initialize()
        async with database.sessions.begin() as session:
            await session.execute(text("BEGIN IMMEDIATE"))
            current = await _analyze(session, root, document_id, run_id, **options)
            if current.fingerprint != plan.fingerprint or current.source != plan.source:
                raise ValueError(
                    "Run/source changed during revalidation; retry with a fresh dry run"
                )
            rows = list(
                await session.scalars(
                    select(Observation)
                    .where(Observation.run_id == run_id)
                    .options(selectinload(Observation.evidence_spans))
                )
            )
            for row in rows:
                state = current.states[row.observation_id]
                row.validation_status = state["validation_status"]
                row.provenance_id = state["provenance_id"]
                row.duplicate_of_observation_id = state["duplicate_of_observation_id"]
                for span, evidence_date in zip(
                    row.evidence_spans, state["evidence_dates"], strict=True
                ):
                    span.evidence_date = (
                        date.fromisoformat(evidence_date) if evidence_date else None
                    )
            chunks = list(
                await session.scalars(
                    select(ExtractionChunk).where(ExtractionChunk.run_id == run_id)
                )
            )
            for chunk in chunks:
                chunk.warnings_json = current.warnings[chunk.chunk_id]
            session.add(
                ValidationRevision(
                    run_id=run_id,
                    validator_version=VALIDATOR_VERSION,
                    revalidated_at=utc_now(),
                    source_json=current.source,
                    summary_json=current.summary,
                    changes_json=current.changes,
                )
            )
        committed = True
        result["exports"] = await _export(database, root, document_id, run_id)
    except (ValueError, OSError, RuntimeError, sqlite3.Error, SQLAlchemyError) as error:
        if committed:
            raise RuntimeError(
                f"Revalidation committed; export failed: {error}. Repeat --apply to retry exports. Backup: {backup_path}"
            ) from error
        raise
    finally:
        await database.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--document", required=True)
    parser.add_argument("--run", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Back up, persist validation decisions, and export",
    )
    mode.add_argument(
        "--dry-run", action="store_true", help="Read-only preview (default)"
    )
    parser.add_argument(
        "--backup", type=Path, help="New SQLite backup path; --apply only"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-pdf", type=Path, help="Legacy run: exact manifest PDF path"
    )
    parser.add_argument(
        "--source-sha256",
        help="Legacy run: independently verified expected PDF SHA-256",
    )
    parser.add_argument(
        "--refresh-pdf-text",
        action="store_true",
        help="Explicitly use PyMuPDF text inside retained historical page/overlap limits",
    )
    parser.add_argument(
        "--historical-extractor",
        choices=["pypdf", "pymupdf"],
        default="pypdf",
        help="Legacy chunk reconstruction only (default: pypdf); the additional PDF representation uses PyMuPDF",
    )
    parser.add_argument(
        "--max-chars", type=int, help="Legacy run: original chunk target"
    )
    parser.add_argument(
        "--overlap-chars", type=int, help="Legacy run: original overlap bound"
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Include unresolved quotes and normalized excerpts",
    )
    args = parser.parse_args(argv)
    if args.backup and not args.apply:
        parser.error("--backup requires --apply")
    try:
        result = asyncio.run(
            revalidate(
                args.database,
                args.document,
                args.run,
                root=args.root,
                apply=args.apply,
                backup_path=args.backup,
                source_pdf=args.source_pdf,
                source_sha256=args.source_sha256,
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
                diagnostics=args.diagnostics,
                refresh_pdf_text=args.refresh_pdf_text,
                historical_extractor=args.historical_extractor,
            )
        )
    except (ValueError, OSError, RuntimeError, sqlite3.Error, SQLAlchemyError) as error:
        print(f"Revalidation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0
