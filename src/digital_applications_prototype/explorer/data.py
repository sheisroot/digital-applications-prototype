"""Read-only data access, literal search, and single-query cosine ranking.

This module has no dependency on extraction services, settings, or providers.
"""

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

MODEL = "text-embedding-3-large"
SCHEMA_VERSION = 2
FILTER_FIELDS = (
    "country",
    "project_id",
    "document_id",
    "document_type",
    "observation_type",
    "valence",
    "claim_scope",
)


@contextmanager
def connect_readonly(
    path: Path, *, immutable: bool = False
) -> Iterator[sqlite3.Connection]:
    """Never create a missing DB; immutable is only for the frozen artifact."""
    uri = path.resolve().as_uri() + "?mode=ro"
    if immutable:
        uri += "&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    try:
        yield connection
    finally:
        connection.close()


@dataclass(frozen=True)
class Evidence:
    quote: str
    page: int
    role: str
    evidence_date: str | None


@dataclass(frozen=True)
class Observation:
    observation_id: int
    provenance_id: str | None
    run_id: str
    observation: str
    factor_description: str
    observation_type: str
    valence: str
    claim_scope: str
    implementation_area: str | None
    actors: tuple[str, ...]
    context: tuple[str, ...]
    action: tuple[str, ...]
    observed_effect: str | None
    country: str
    project_id: str
    document_id: str
    document_type: str
    document_date: str | None
    source_url: str | None
    evidence: tuple[Evidence, ...]
    factor_label: str | None = None

    @property
    def display_title(self) -> str:
        """Demo-only presentation label; never a semantic retrieval input."""
        return self.factor_label or self.factor_description


@dataclass(frozen=True)
class Corpus:
    observations: tuple[Observation, ...]
    # Matrix row order is explicitly joined and checked against integer IDs.
    vectors: NDArray[np.float64]
    norms: NDArray[np.float64]
    positions: Mapping[int, int]

    def get(self, observation_id: int) -> Observation:
        return self.observations[self.positions[observation_id]]


def load_corpus(path: Path) -> Corpus:
    with connect_readonly(path, immutable=True) as connection:
        if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise ValueError("Unsupported demo snapshot schema")
        spans: dict[int, list[Evidence]] = {}
        for row in connection.execute(
            "SELECT * FROM evidence_spans ORDER BY observation_id, span_index"
        ):
            spans.setdefault(row["observation_id"], []).append(
                Evidence(row["quote"], row["page"], row["role"], row["evidence_date"])
            )
        observations = tuple(
            Observation(
                observation_id=row["observation_id"],
                provenance_id=row["provenance_id"],
                run_id=row["run_id"],
                observation=row["observation"],
                factor_description=row["factor_description"],
                factor_label=row["factor_label"],
                observation_type=row["observation_type"],
                valence=row["valence"],
                claim_scope=row["claim_scope"],
                implementation_area=row["implementation_area"],
                actors=tuple(json.loads(row["actors_json"])),
                context=tuple(json.loads(row["condition_or_context_json"])),
                action=tuple(json.loads(row["action_or_response_json"])),
                observed_effect=row["observed_effect"],
                country=row["country"],
                project_id=row["project_id"],
                document_id=row["document_id"],
                document_type=row["document_type"],
                document_date=row["document_date"],
                source_url=row["source_url"],
                evidence=tuple(spans.get(row["observation_id"], ())),
            )
            for row in connection.execute(
                "SELECT o.*, d.* FROM observations o "
                "JOIN documents d USING (document_id) ORDER BY observation_id"
            )
        )
        embeddings = connection.execute(
            "SELECT * FROM embeddings ORDER BY observation_id"
        ).fetchall()
    if not observations or len(observations) != len(embeddings):
        raise ValueError("Snapshot must have one embedding per observation")
    vectors: list[NDArray[np.float64]] = []
    for observation, embedding in zip(observations, embeddings, strict=True):
        if (
            embedding["observation_id"] != observation.observation_id
            or embedding["source_text"] != observation.factor_description
            or embedding["source_text_hash"]
            != sha256(observation.factor_description.encode("utf-8")).hexdigest()
            or embedding["model"] != MODEL
            or embedding["requested_model"] != MODEL
            or embedding["representation"] != "factor_description"
            or embedding["dimensions_mode"] != "default"
            or not observation.evidence
        ):
            raise ValueError("Snapshot observation/vector/evidence identity mismatch")
        vector = np.frombuffer(embedding["vector"], dtype="<f8")
        if len(vector) != embedding["dimensions"]:
            raise ValueError("Snapshot vector dimension mismatch")
        vectors.append(vector)
    matrix = np.stack(vectors)
    norms = np.linalg.norm(matrix, axis=1)
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(norms).all()
        or (norms <= 0).any()
    ):
        raise ValueError("Snapshot vectors must be finite and nonzero")
    matrix.setflags(write=False)
    norms.setflags(write=False)
    return Corpus(
        observations,
        matrix,
        norms,
        MappingProxyType({o.observation_id: i for i, o in enumerate(observations)}),
    )


def filter_observations(
    observations: Sequence[Observation],
    *,
    keyword: str = "",
    filters: Mapping[str, Sequence[str]] | None = None,
) -> tuple[Observation, ...]:
    """Literal case-insensitive substring search plus intersected metadata filters."""
    selected = filters or {}
    if selected.keys() - set(FILTER_FIELDS):
        raise ValueError("Unsupported metadata filter")
    needle = keyword.strip().casefold()
    return tuple(
        observation
        for observation in observations
        if all(
            not values or getattr(observation, field) in values
            for field, values in selected.items()
        )
        and (
            not needle
            or needle
            in "\n".join(
                (
                    observation.observation,
                    observation.factor_description,
                    observation.implementation_area or "",
                )
            ).casefold()
        )
    )


def related_observations(
    corpus: Corpus,
    observation_id: int,
    *,
    other_documents: bool = False,
    other_countries: bool = False,
) -> tuple[tuple[Observation, float], ...]:
    """Top five across the corpus; browsing filters do not limit this comparison."""
    position = corpus.positions[observation_id]
    seed = corpus.observations[position]
    scores = (corpus.vectors @ corpus.vectors[position]) / (
        corpus.norms * corpus.norms[position]
    )
    eligible = [
        i
        for i, observation in enumerate(corpus.observations)
        if observation.observation_id != observation_id
        and (not other_documents or observation.document_id != seed.document_id)
        and (not other_countries or observation.country != seed.country)
    ]
    ranked = sorted(
        eligible,
        key=lambda i: (-float(scores[i]), corpus.observations[i].observation_id),
    )[:5]
    return tuple(
        (corpus.observations[i], float(np.clip(scores[i], -1.0, 1.0))) for i in ranked
    )
