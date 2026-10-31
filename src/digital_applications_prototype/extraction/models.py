"""Model-authored semantic fields and application-owned provenance."""

from datetime import date
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


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
    OBSERVED = "observed"
    REPORTED = "reported"
    ANTICIPATED = "anticipated"
    RECOMMENDED = "recommended"
    INFERRED = "inferred"


class EvidenceStrength(StrEnum):
    EXPLICIT = "explicit"
    STRONGLY_IMPLIED = "strongly_implied"
    WEAKLY_IMPLIED = "weakly_implied"


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

    @property
    def start_page(self) -> int:
        return self.pages[0].page_number

    @property
    def end_page(self) -> int:
        return self.pages[-1].page_number

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


class ModelObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_type: ObservationType
    valence: Valence
    claim_scope: ClaimScope
    observation: str
    raw_factor_label: str
    raw_factor_description: str
    implementation_area: str | None = None
    actors: list[str] = Field(default_factory=list)
    condition_or_context: list[str] = Field(default_factory=list)
    action_or_response: list[str] = Field(default_factory=list)
    observed_effect: str | None = None
    evidence_strength: EvidenceStrength
    source_quote: str
    source_page: int | None = Field(default=None, ge=1)


class ExtractionResult(BaseModel):
    """Provider response: semantics only, without regenerated metadata."""

    model_config = ConfigDict(extra="forbid")
    observations: list[ModelObservation] = Field(default_factory=list)


class ImplementationObservation(ModelObservation, DocumentMetadata):
    observation_id: str
    chunk_start_page: int = Field(ge=1)
    chunk_end_page: int = Field(ge=1)
