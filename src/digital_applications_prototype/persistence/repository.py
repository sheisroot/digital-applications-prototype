"""Short transactions for Stage 1 runs, chunks, observations, and summaries."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from digital_applications_prototype.extraction.models import (
    ChunkExtractionResult,
    ChunkFailure,
    DocumentChunk,
    ImplementationObservation,
    RunReport,
    RunStatus,
    ValidationWarning,
)
from digital_applications_prototype.extraction.validation import (
    observation_duplicate_indexes,
)

from .models import (
    ChunkSource,
    EvidenceSpanRow,
    ExtractionChunk,
    ExtractionRun,
    Observation,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def derive_run_status(chunk_statuses: Sequence[str]) -> RunStatus:
    """Empty/unusable coverage fails; unfinished coverage with successes is partial."""
    if chunk_statuses and all(status == "completed" for status in chunk_statuses):
        return RunStatus.COMPLETED
    if "completed" in chunk_statuses:
        return RunStatus.PARTIAL
    return RunStatus.FAILED


def _token_total(chunks: list[ExtractionChunk], field: str) -> int | None:
    values = [getattr(chunk, field) for chunk in chunks if chunk.status == "completed"]
    return (
        sum(values) if values and all(value is not None for value in values) else None
    )


def _domain_observation(row: Observation) -> ImplementationObservation:
    """Reconstruct the existing scientific output, retaining its provenance ID."""
    return ImplementationObservation.model_validate(
        {
            "observation_id": row.provenance_id,
            "project_id": row.project_id,
            "country": row.country,
            "document_id": row.document_id,
            "document_type": row.document_type,
            "document_date": row.document_date,
            "chunk_start_page": row.chunk_start_page,
            "chunk_end_page": row.chunk_end_page,
            "observation_type": row.observation_type,
            "valence": row.valence,
            "claim_scope": row.claim_scope,
            "observation": row.observation,
            "factor_description": row.factor_description,
            "implementation_area": row.implementation_area,
            "actors": row.actors_json,
            "condition_or_context": row.condition_or_context_json,
            "action_or_response": row.action_or_response_json,
            "observed_effect": row.observed_effect,
            "evidence_strength": row.evidence_strength,
            "evidence": {
                "summary": row.evidence_summary,
                "spans": [
                    {
                        "quote": span.quote,
                        "page": span.page,
                        "evidence_date": span.evidence_date,
                        "role": span.role,
                    }
                    for span in row.evidence_spans
                ],
            },
        }
    )


class ExtractionRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def create_run(
        self, document_id: str, *, model: str | None = None, max_concurrency: int = 1
    ) -> str:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        started = datetime.now(UTC)
        run_id = f"{started:%Y%m%dT%H%M%SZ}_{uuid4().hex}"
        async with self.sessions.begin() as session:
            session.add(
                ExtractionRun(
                    run_id=run_id,
                    document_id=document_id,
                    started_at=started.isoformat(),
                    model=model,
                    schema_version=RunReport.model_fields["schema_version"].default,
                    max_concurrency=max_concurrency,
                )
            )
        return run_id

    async def set_run_model(self, run_id: str, model: str | None) -> None:
        async with self.sessions.begin() as session:
            await session.execute(
                update(ExtractionRun)
                .where(ExtractionRun.run_id == run_id)
                .values(model=model)
            )

    async def create_chunks(
        self,
        run_id: str,
        chunks: list[DocumentChunk],
        *,
        source_metadata: dict[str, Any] | None = None,
    ) -> list[int]:
        rows = [
            ExtractionChunk(
                run_id=run_id,
                chunk_index=index,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        async with self.sessions.begin() as session:
            session.add_all(rows)
            await session.flush()
            session.add_all(
                [
                    ChunkSource(
                        chunk_id=row.chunk_id,
                        source_json={
                            **chunk.model_dump(mode="json"),
                            "pdf_source": source_metadata,
                        },
                    )
                    for row, chunk in zip(rows, chunks, strict=True)
                ]
            )
            return [row.chunk_id for row in rows]

    async def mark_chunk_running(self, chunk_id: int) -> None:
        async with self.sessions.begin() as session:
            result = await session.execute(
                update(ExtractionChunk)
                .where(
                    ExtractionChunk.chunk_id == chunk_id,
                    ExtractionChunk.status == "pending",
                )
                .values(status="running", started_at=utc_now())
                .returning(ExtractionChunk.chunk_id)
            )
            if result.scalar_one_or_none() is None:
                raise ValueError(f"Chunk {chunk_id} is not pending")

    async def persist_completed_chunk(
        self,
        chunk_id: int,
        chunk: DocumentChunk,
        result: ChunkExtractionResult,
    ) -> None:
        """Commit the complete validated chunk, including rejected output, atomically.

        No remote work happens here. A failed insert rolls back observations,
        evidence, token usage, and the completed transition together.
        """
        if len(result.observation_results) != result.observations_returned:
            raise ValueError(
                "Every returned model observation must have an audit outcome"
            )
        async with self.sessions.begin() as session:
            row = await session.get(ExtractionChunk, chunk_id)
            if row is None or row.status != "running":
                raise ValueError(f"Chunk {chunk_id} is not running")
            for index, audited in enumerate(result.observation_results, start=1):
                outcome = audited.validation
                item = (
                    outcome.observation
                    if outcome.observation is not None
                    else audited.model_observation
                )
                observation = Observation(
                    run_id=row.run_id,
                    chunk_id=chunk_id,
                    observation_index=index,
                    provenance_id=outcome.observation.observation_id
                    if outcome.observation
                    else None,
                    observation_type=item.observation_type.value,
                    valence=item.valence.value,
                    claim_scope=item.claim_scope.value,
                    observation=item.observation,
                    factor_description=item.factor_description,
                    implementation_area=item.implementation_area,
                    actors_json=item.actors,
                    condition_or_context_json=item.condition_or_context,
                    action_or_response_json=item.action_or_response,
                    observed_effect=item.observed_effect,
                    evidence_strength=item.evidence_strength.value,
                    evidence_summary=item.evidence.summary,
                    validation_status=outcome.status,
                    project_id=chunk.project_id,
                    country=chunk.country,
                    document_id=chunk.document_id,
                    document_type=chunk.document_type,
                    document_date=chunk.document_date,
                    chunk_start_page=chunk.start_page,
                    chunk_end_page=chunk.end_page,
                    raw_observation_json=audited.model_observation.model_dump(
                        mode="json"
                    ),
                    evidence_spans=[
                        EvidenceSpanRow(
                            span_index=span_index,
                            quote=span.quote,
                            page=span.page,
                            evidence_date=span.evidence_date,
                            role=span.role.value,
                        )
                        for span_index, span in enumerate(item.evidence.spans, start=1)
                    ],
                )
                session.add(observation)
            row.status = "completed"
            row.finished_at = utc_now()
            row.input_tokens = result.usage.input_tokens if result.usage else None
            row.output_tokens = result.usage.output_tokens if result.usage else None
            row.warnings_json = [
                warning.model_dump(mode="json") for warning in result.warnings
            ]

    async def mark_chunk_failed(self, chunk_id: int, error: Exception) -> None:
        async with self.sessions.begin() as session:
            result = await session.execute(
                update(ExtractionChunk)
                .where(
                    ExtractionChunk.chunk_id == chunk_id,
                    ExtractionChunk.status == "running",
                )
                .values(
                    status="failed",
                    finished_at=utc_now(),
                    error=str(error),
                    error_type=type(error).__name__,
                )
                .returning(ExtractionChunk.chunk_id)
            )
            if result.scalar_one_or_none() is None:
                raise ValueError(f"Chunk {chunk_id} is not running")

    async def mark_duplicates(self, run_id: str) -> int:
        """Reapply the existing first-in-source-order signature rule, including partial runs."""
        async with self.sessions.begin() as session:
            rows = await self._observation_rows(session, run_id, clean=False)
            duplicates = observation_duplicate_indexes(
                [_domain_observation(row) for row in rows]
            )
            for index, row in enumerate(rows):
                first_index = duplicates.get(index)
                row.duplicate_of_observation_id = (
                    rows[first_index].observation_id
                    if first_index is not None
                    else None
                )
            return len(duplicates)

    async def finalize_run(self, run_id: str, *, error: str | None = None) -> None:
        async with self.sessions.begin() as session:
            run = await session.get(ExtractionRun, run_id)
            if run is None:
                raise ValueError(f"Unknown run {run_id}")
            chunks = list(
                await session.scalars(
                    select(ExtractionChunk).where(ExtractionChunk.run_id == run_id)
                )
            )
            run.status = derive_run_status([chunk.status for chunk in chunks]).value
            run.finished_at = utc_now()
            run.input_tokens = _token_total(chunks, "input_tokens")
            run.output_tokens = _token_total(chunks, "output_tokens")
            run.error = error

    async def _observation_rows(
        self, session: AsyncSession, run_id: str, *, clean: bool
    ) -> list[Observation]:
        query = (
            select(Observation)
            .join(ExtractionChunk, Observation.chunk_id == ExtractionChunk.chunk_id)
            .where(
                Observation.run_id == run_id,
                Observation.validation_status.in_(("valid", "repaired")),
            )
            .options(selectinload(Observation.evidence_spans))
            .order_by(ExtractionChunk.chunk_index, Observation.observation_index)
        )
        if clean:
            query = query.where(Observation.duplicate_of_observation_id.is_(None))
        return list(await session.scalars(query))

    async def get_clean_observations(
        self, run_id: str
    ) -> list[ImplementationObservation]:
        """Only valid/repaired nonduplicates; callers can also inspect partial runs."""
        async with self.sessions() as session:
            return [
                _domain_observation(row)
                for row in await self._observation_rows(session, run_id, clean=True)
            ]

    async def get_run_summary(self, run_id: str) -> RunReport:
        async with self.sessions() as session:
            run = await session.get(ExtractionRun, run_id)
            if run is None:
                raise ValueError(f"Unknown run {run_id}")
            chunks = list(
                await session.scalars(
                    select(ExtractionChunk)
                    .where(ExtractionChunk.run_id == run_id)
                    .order_by(ExtractionChunk.chunk_index)
                )
            )
            counts = dict(
                (
                    await session.execute(
                        select(Observation.validation_status, func.count())
                        .where(Observation.run_id == run_id)
                        .group_by(Observation.validation_status)
                    )
                ).all()
            )
            duplicates = await session.scalar(
                select(func.count())
                .select_from(Observation)
                .where(
                    Observation.run_id == run_id,
                    Observation.duplicate_of_observation_id.is_not(None),
                )
            )
            valid, repaired, rejected = (
                counts.get(status, 0) for status in ("valid", "repaired", "rejected")
            )
            return RunReport.model_validate(
                {
                    "run_id": run.run_id,
                    "document_id": run.document_id,
                    "started_at": datetime.fromisoformat(run.started_at),
                    "finished_at": datetime.fromisoformat(run.finished_at)
                    if run.finished_at
                    else None,
                    "status": run.status
                    or derive_run_status([chunk.status for chunk in chunks]),
                    "model": run.model,
                    "schema_version": run.schema_version,
                    "max_concurrency": run.max_concurrency,
                    "chunks_total": len(chunks),
                    "chunks_succeeded": sum(
                        chunk.status == "completed" for chunk in chunks
                    ),
                    "chunks_failed": sum(chunk.status == "failed" for chunk in chunks),
                    "observations_returned": valid + repaired + rejected,
                    "observations_valid": valid,
                    "observations_repaired": repaired,
                    "observations_rejected": rejected,
                    "duplicates_removed": duplicates or 0,
                    "observations_persisted": valid + repaired - (duplicates or 0),
                    "input_tokens": _token_total(chunks, "input_tokens"),
                    "output_tokens": _token_total(chunks, "output_tokens"),
                    "warnings": [
                        ValidationWarning.model_validate(warning)
                        for chunk in chunks
                        for warning in chunk.warnings_json
                    ],
                    "chunk_failures": [
                        ChunkFailure(
                            chunk_index=chunk.chunk_index,
                            chunk_start_page=chunk.start_page,
                            chunk_end_page=chunk.end_page,
                            error_type=chunk.error_type or "Error",
                            message=chunk.error or "",
                        )
                        for chunk in chunks
                        if chunk.status == "failed"
                    ],
                    "error": run.error,
                }
            )
