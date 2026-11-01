"""Model-authored semantic fields and application-owned provenance."""

from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from digital_applications_prototype.llm.adapters.base import TokenUsage


class ObservationType(StrEnum):
    CONDITION = "condition"
    ACTION = "action"
    ADAPTATION = "adaptation"
    OUTCOME = "outcome"


class Valence(StrEnum):
    ENABLER = "enabler"
    CONSTRAINT = "constraint"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class ClaimScope(StrEnum):
    ACTUAL = "actual"
    PLANNED = "planned"
    ANTICIPATED = "anticipated"
    RECOMMENDED = "recommended"
    INFERRED = "inferred"


class EvidenceStrength(StrEnum):
    EXPLICIT = "explicit"
    STRONGLY_IMPLIED = "strongly_implied"
    WEAKLY_IMPLIED = "weakly_implied"


class EvidenceRole(StrEnum):
    SUPPORTS = "supports"
    QUALIFIES = "qualifies"
    CONTRADICTS = "contradicts"
    CONTEXT = "context"


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote: str = Field(min_length=1)
    page: int = Field(ge=1)
    evidence_date: date | None = None
    role: EvidenceRole = EvidenceRole.SUPPORTS


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str | None = None
    # Empty typed lists reach observation validation and are rejected independently.
    spans: list[EvidenceSpan]


class DocumentMetadata(BaseModel):
    project_id: str
    country: str = Field(pattern=r"^[a-z]+(?:-[a-z]+)*$")
    document_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    document_type: str
    document_date: date | None


class DocumentEntry(DocumentMetadata):
    path: Path


class PageText(BaseModel):
    page_number: int = Field(ge=1)
    text: str


class DocumentChunk(DocumentMetadata):
    pages: list[PageText] = Field(min_length=1)
    overlap: PageText | None = None

    @property
    def source_pages(self) -> list[PageText]:
        return [self.overlap, *self.pages] if self.overlap is not None else self.pages

    @property
    def start_page(self) -> int:
        return self.source_pages[0].page_number

    @property
    def end_page(self) -> int:
        return self.pages[-1].page_number

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.source_pages)


class ModelObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_type: ObservationType
    valence: Valence
    claim_scope: ClaimScope
    observation: str
    factor_description: str
    implementation_area: str | None = None
    actors: list[str] = Field(default_factory=list)
    condition_or_context: list[str] = Field(default_factory=list)
    action_or_response: list[str] = Field(default_factory=list)
    observed_effect: str | None = None
    evidence_strength: EvidenceStrength
    evidence: EvidenceBundle


class ExtractionResult(BaseModel):
    """Provider response: semantics only, without regenerated metadata."""

    model_config = ConfigDict(extra="forbid")
    observations: list[ModelObservation] = Field(default_factory=list)


class ImplementationObservation(ModelObservation, DocumentMetadata):
    observation_id: str
    chunk_start_page: int = Field(ge=1)
    chunk_end_page: int = Field(ge=1)

    @model_validator(mode="after")
    def require_evidence(self) -> "ImplementationObservation":
        if not self.evidence.spans:
            raise ValueError("Persisted observations require at least one evidence span")
        return self


class ValidationWarning(BaseModel):
    code: str
    message: str
    chunk_start_page: int | None = None
    chunk_end_page: int | None = None
    observation_index: int | None = None
    evidence_span_index: int | None = None
    action: str | None = None


class ObservationValidationResult(BaseModel):
    status: Literal["valid", "repaired", "rejected"]
    observation: ImplementationObservation | None = None
    warnings: list[ValidationWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_outcome(self) -> "ObservationValidationResult":
        """Require a usable observation for kept outcomes and warnings for repairs."""
        if (self.status == "rejected") != (self.observation is None):
            raise ValueError("Only rejected outcomes may have observation=None")
        if self.status == "repaired" and not self.warnings:
            raise ValueError("Repaired outcomes require a warning")
        return self


class AuditedObservation(BaseModel):
    """Keep the original model output alongside the deterministic validation outcome."""

    model_observation: ModelObservation
    validation: ObservationValidationResult


class ChunkExtractionResult(BaseModel):
    observations: list[ImplementationObservation] = Field(default_factory=list)
    observation_results: list[AuditedObservation] = Field(default_factory=list)
    observations_returned: int = 0
    observations_valid: int = 0
    observations_repaired: int = 0
    observations_rejected: int = 0
    warnings: list[ValidationWarning] = Field(default_factory=list)
    usage: TokenUsage | None = None


class RunStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class ChunkFailure(BaseModel):
    chunk_index: int
    chunk_start_page: int
    chunk_end_page: int
    error_type: str
    message: str


class RunReport(BaseModel):
    document_id: str
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    status: RunStatus = RunStatus.FAILED
    schema_version: Literal["stage1-v3"] = "stage1-v3"
    model: str | None = None
    max_concurrency: int = 1
    chunks_total: int = 0
    chunks_succeeded: int = 0
    chunks_failed: int = 0
    observations_returned: int = 0
    observations_valid: int = 0
    observations_repaired: int = 0
    observations_rejected: int = 0
    duplicates_removed: int = 0
    observations_persisted: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    jsonl_file: str | None = None
    warnings: list[ValidationWarning] = Field(default_factory=list)
    chunk_failures: list[ChunkFailure] = Field(default_factory=list)
    error: str | None = None
