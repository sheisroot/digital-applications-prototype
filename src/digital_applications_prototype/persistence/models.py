"""SQLAlchemy storage models. Timestamps are ISO 8601 strings with a UTC offset."""

from datetime import date
from typing import Any

from sqlalchemy import JSON, CheckConstraint, Date, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ExtractionRun(Base):
    __tablename__ = "extraction_runs"
    __table_args__ = (
        CheckConstraint("status IS NULL OR status IN ('completed', 'partial', 'failed')"),
        CheckConstraint("max_concurrency > 0"),
    )

    run_id: Mapped[str] = mapped_column(primary_key=True)
    document_id: Mapped[str] = mapped_column(index=True)
    started_at: Mapped[str]
    finished_at: Mapped[str | None]
    # NULL means unfinished; only final outcomes use the three run statuses.
    status: Mapped[str | None]
    model: Mapped[str | None]
    schema_version: Mapped[str]
    max_concurrency: Mapped[int] = mapped_column(default=1)
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    error: Mapped[str | None]


class ExtractionChunk(Base):
    __tablename__ = "extraction_chunks"
    __table_args__ = (
        UniqueConstraint("run_id", "chunk_index"),
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')"),
        Index("ix_extraction_chunks_run_status", "run_id", "status"),
    )

    chunk_id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("extraction_runs.run_id"))
    chunk_index: Mapped[int]
    start_page: Mapped[int]
    end_page: Mapped[int]
    status: Mapped[str] = mapped_column(default="pending")
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    started_at: Mapped[str | None]
    finished_at: Mapped[str | None]
    error: Mapped[str | None]
    error_type: Mapped[str | None]
    warnings_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)


class ChunkSource(Base):
    """Exact model-facing pages, including the bounded overlap, for offline checks."""

    __tablename__ = "extraction_chunk_sources"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("extraction_chunks.chunk_id"), primary_key=True
    )
    source_json: Mapped[dict[str, Any]] = mapped_column(JSON)


class ValidationRevision(Base):
    """Append-only derived-state audit; extraction history remains untouched."""

    __tablename__ = "validation_revisions"

    revalidation_id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("extraction_runs.run_id"), index=True)
    validator_version: Mapped[str]
    revalidated_at: Mapped[str]
    source_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    changes_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON)


class Observation(Base):
    __tablename__ = "implementation_observations"
    __table_args__ = (
        UniqueConstraint("chunk_id", "observation_index"),
        CheckConstraint("validation_status IN ('valid', 'repaired', 'rejected')"),
        Index("ix_observations_clean", "run_id", "validation_status", "duplicate_of_observation_id"),
    )

    # Row identity differs from the domain ID, which intentionally matches duplicates.
    observation_id: Mapped[int] = mapped_column(primary_key=True)
    provenance_id: Mapped[str | None] = mapped_column(index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("extraction_runs.run_id"), index=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("extraction_chunks.chunk_id"), index=True)
    observation_index: Mapped[int]
    observation_type: Mapped[str]
    valence: Mapped[str]
    claim_scope: Mapped[str]
    observation: Mapped[str]
    factor_description: Mapped[str]
    implementation_area: Mapped[str | None]
    actors_json: Mapped[list[str]] = mapped_column(JSON)
    condition_or_context_json: Mapped[list[str]] = mapped_column(JSON)
    action_or_response_json: Mapped[list[str]] = mapped_column(JSON)
    observed_effect: Mapped[str | None]
    evidence_strength: Mapped[str]
    evidence_summary: Mapped[str | None]
    validation_status: Mapped[str]
    duplicate_of_observation_id: Mapped[int | None] = mapped_column(
        ForeignKey("implementation_observations.observation_id"), index=True,
    )
    project_id: Mapped[str]
    country: Mapped[str]
    document_id: Mapped[str] = mapped_column(index=True)
    document_type: Mapped[str]
    document_date: Mapped[date | None] = mapped_column(Date)
    chunk_start_page: Mapped[int]
    chunk_end_page: Mapped[int]
    raw_observation_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    evidence_spans: Mapped[list["EvidenceSpanRow"]] = relationship(
        cascade="all, delete-orphan", order_by="EvidenceSpanRow.span_index", lazy="raise",
    )


class EvidenceSpanRow(Base):
    __tablename__ = "evidence_spans"
    __table_args__ = (UniqueConstraint("observation_id", "span_index"),)

    evidence_span_id: Mapped[int] = mapped_column(primary_key=True)
    observation_id: Mapped[int] = mapped_column(
        ForeignKey("implementation_observations.observation_id"), index=True,
    )
    span_index: Mapped[int]
    quote: Mapped[str]
    page: Mapped[int]
    evidence_date: Mapped[date | None] = mapped_column(Date)
    role: Mapped[str]
