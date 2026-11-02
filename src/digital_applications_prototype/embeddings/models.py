"""Inspectable embedding inputs, resolved scope, and execution counts."""

from collections import defaultdict
from dataclasses import dataclass, field
from hashlib import sha256

REPRESENTATION = "factor_description"
DEFAULT_BATCH_SIZE = 100


def source_text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddingInput:
    observation_id: int
    source_text: str

    @property
    def source_text_hash(self) -> str:
        return source_text_hash(self.source_text)


@dataclass(frozen=True)
class SelectedRun:
    run_id: str
    document_id: str
    eligible: int
    already_current: int


@dataclass(frozen=True)
class EmbeddingPlan:
    runs: tuple[SelectedRun, ...]
    pending: tuple[EmbeddingInput, ...]
    model: str
    representation: str = REPRESENTATION
    batch_size: int = DEFAULT_BATCH_SIZE
    current_dimensions: tuple[int, ...] = ()

    @property
    def documents(self) -> set[str]:
        return {run.document_id for run in self.runs}

    @property
    def conflicts(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for run in self.runs:
            grouped[run.document_id].append(run.run_id)
        return {document: ids for document, ids in grouped.items() if len(ids) > 1}

    @property
    def eligible(self) -> int:
        return sum(run.eligible for run in self.runs)

    @property
    def already_current(self) -> int:
        return sum(run.already_current for run in self.runs)

    def validate(self, *, allow_multiple_runs_per_document: bool = False) -> None:
        if not self.runs:
            raise ValueError("No extraction runs selected")
        if not self.model.strip():
            raise ValueError("Embedding model must be nonempty")
        if self.representation != REPRESENTATION:
            raise ValueError(
                f"Unsupported embedding representation: {self.representation}"
            )
        if not 1 <= self.batch_size <= 2048:
            raise ValueError("Batch size must be between 1 and 2048")
        if self.conflicts and not allow_multiple_runs_per_document:
            details = "; ".join(
                f"{document}: {', '.join(ids)}"
                for document, ids in self.conflicts.items()
            )
            raise ValueError(
                f"Multiple selected runs for the same document: {details}. "
                "Select one run per document or explicitly use "
                "--allow-multiple-runs-per-document."
            )


@dataclass
class EmbeddingReport:
    submitted: int = 0
    embedded: int = 0
    failed: int = 0
    batches: int = 0
    completed_batches: int = 0
    dimensions: set[int] = field(default_factory=set)
    models: set[str] = field(default_factory=set)
    input_tokens: int = 0
    usage_batches: int = 0
