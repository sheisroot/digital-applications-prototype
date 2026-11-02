"""Explicit offline export into a small purpose-built SQLite demo artifact."""

import json
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from digital_applications_prototype.embeddings.scope import normalize_run_ids

from .data import MODEL, SCHEMA_VERSION, connect_readonly, load_corpus

OBSERVATION_COLUMNS = (
    "observation_id",
    "provenance_id",
    "run_id",
    "document_id",
    "observation",
    "factor_description",
    "observation_type",
    "valence",
    "claim_scope",
    "implementation_area",
    "actors_json",
    "condition_or_context_json",
    "action_or_response_json",
    "observed_effect",
    "validation_status",
)
MAX_SNAPSHOT_BYTES = 50 * 1024 * 1024

SCHEMA = """
CREATE TABLE selected_runs (
    run_id TEXT PRIMARY KEY, document_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status = 'completed')
);
CREATE TABLE documents (
    document_id TEXT PRIMARY KEY, country TEXT NOT NULL, project_id TEXT NOT NULL,
    document_type TEXT NOT NULL, document_date TEXT, source_url TEXT
);
CREATE TABLE observations (
    observation_id INTEGER PRIMARY KEY, provenance_id TEXT,
    run_id TEXT NOT NULL REFERENCES selected_runs(run_id),
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    observation TEXT NOT NULL, factor_description TEXT NOT NULL,
    observation_type TEXT NOT NULL, valence TEXT NOT NULL, claim_scope TEXT NOT NULL,
    implementation_area TEXT, actors_json TEXT NOT NULL,
    condition_or_context_json TEXT NOT NULL, action_or_response_json TEXT NOT NULL,
    observed_effect TEXT,
    validation_status TEXT NOT NULL CHECK (validation_status IN ('valid', 'repaired'))
);
CREATE TABLE evidence_spans (
    evidence_span_id INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL REFERENCES observations(observation_id),
    span_index INTEGER NOT NULL, quote TEXT NOT NULL,
    page INTEGER NOT NULL CHECK (page > 0), role TEXT NOT NULL, evidence_date TEXT,
    UNIQUE (observation_id, span_index)
);
CREATE TABLE embeddings (
    observation_id INTEGER PRIMARY KEY REFERENCES observations(observation_id),
    embedding_id INTEGER NOT NULL UNIQUE, representation TEXT NOT NULL,
    requested_model TEXT NOT NULL, model TEXT NOT NULL, dimensions_mode TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    source_text TEXT NOT NULL, source_text_hash TEXT NOT NULL, created_at TEXT NOT NULL,
    vector BLOB NOT NULL CHECK (length(vector) = dimensions * 8)
);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def source_url_mapping(
    manifest: Sequence[tuple[str, str, str, str]],
) -> dict[str, str]:
    """Use only input entries in the checked-in public source URL manifest."""
    mapping: dict[str, str] = {}
    for split, _country, filename, url in manifest:
        if split != "input":
            continue
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "documents.worldbank.org":
            raise ValueError(f"Unexpected public source URL for {filename}")
        document_id = Path(filename).stem
        if document_id in mapping:
            raise ValueError(f"Duplicate source mapping: {document_id}")
        mapping[document_id] = url
    return mapping


def create_snapshot(
    source: Path,
    destination: Path,
    run_ids: Sequence[str],
    source_urls: Mapping[str, str],
    *,
    expected_documents: int | None = None,
    expected_observations: int | None = None,
) -> dict[str, int | str]:
    """Fail closed on incomplete scope, stale/missing/ambiguous vectors or counts.

    The source connection uses an explicit read transaction (including any WAL).
    The destination must be new. Nothing in the source database is modified.
    """
    selected = normalize_run_ids(run_ids)
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError("Snapshot destination must not already exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".sqlite", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        _export(source, temporary_path, selected, source_urls)
        corpus = load_corpus(temporary_path)
        documents = len({o.document_id for o in corpus.observations})
        count = len(corpus.observations)
        if expected_documents is not None and documents != expected_documents:
            raise ValueError(
                f"Expected {expected_documents} documents, got {documents}"
            )
        if expected_observations is not None and count != expected_observations:
            raise ValueError(
                f"Expected {expected_observations} observations, got {count}"
            )
        size = temporary_path.stat().st_size
        if size >= MAX_SNAPSHOT_BYTES:
            raise ValueError("Snapshot exceeds the 50 MiB repository artifact budget")
        report: dict[str, int | str] = {
            "documents": documents,
            "observations": count,
            "countries": len({o.country for o in corpus.observations}),
            "evidence_spans": sum(len(o.evidence) for o in corpus.observations),
            "embeddings": len(corpus.vectors),
            "dimensions": corpus.vectors.shape[1],
            "model": MODEL,
            "bytes": size,
            "sha256": sha256(temporary_path.read_bytes()).hexdigest(),
        }
        temporary_path.replace(destination)
        return report
    finally:
        temporary_path.unlink(missing_ok=True)


def _export(
    source: Path,
    destination: Path,
    selected: list[str],
    source_urls: Mapping[str, str],
) -> None:
    placeholders = ",".join("?" for _ in selected)
    with connect_readonly(source) as origin, sqlite3.connect(destination) as output:
        origin.execute("BEGIN")
        output.execute("PRAGMA foreign_keys = ON")
        output.executescript(SCHEMA)
        output.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        runs = origin.execute(
            f"SELECT run_id, document_id, status FROM extraction_runs "
            f"WHERE run_id IN ({placeholders}) ORDER BY document_id",
            selected,
        ).fetchall()
        if len(runs) != len(selected):
            raise ValueError("Selected manifest includes unknown extraction runs")
        if len({r["document_id"] for r in runs}) != len(runs):
            raise ValueError("Select exactly one extraction run per document")
        if any(r["status"] != "completed" for r in runs):
            raise ValueError("Demo requires completed selected runs")
        output.executemany("INSERT INTO selected_runs VALUES (?, ?, ?)", runs)
        observations = origin.execute(
            f"SELECT * FROM implementation_observations "
            f"WHERE run_id IN ({placeholders}) "
            "AND validation_status IN ('valid', 'repaired') "
            "AND duplicate_of_observation_id IS NULL ORDER BY observation_id",
            selected,
        ).fetchall()
        usable = [o for o in observations if (o["factor_description"] or "").strip()]
        documents: dict[str, tuple[str | None, ...]] = {}
        for observation in usable:
            document_id = observation["document_id"]
            if document_id != next(
                r["document_id"] for r in runs if r["run_id"] == observation["run_id"]
            ):
                raise ValueError("Observation document differs from its selected run")
            metadata = tuple(
                observation[field]
                for field in (
                    "document_id",
                    "country",
                    "project_id",
                    "document_type",
                    "document_date",
                )
            ) + (source_urls.get(document_id),)
            if document_id in documents and documents[document_id] != metadata:
                raise ValueError("Inconsistent stored document metadata")
            documents[document_id] = metadata
        if set(documents) != {r["document_id"] for r in runs}:
            raise ValueError("Every selected document must have usable observations")
        output.executemany(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)", documents.values()
        )
        columns = ",".join(OBSERVATION_COLUMNS)
        marks = ",".join("?" for _ in OBSERVATION_COLUMNS)
        for observation in usable:
            observation_id = observation["observation_id"]
            output.execute(
                f"INSERT INTO observations ({columns}) VALUES ({marks})",
                tuple(observation[column] for column in OBSERVATION_COLUMNS),
            )
            evidence = origin.execute(
                "SELECT evidence_span_id, observation_id, span_index, quote, page, "
                "role, evidence_date FROM evidence_spans WHERE observation_id = ? "
                "ORDER BY span_index",
                (observation_id,),
            ).fetchall()
            output.executemany(
                "INSERT INTO evidence_spans VALUES (?, ?, ?, ?, ?, ?, ?)", evidence
            )
            text = observation["factor_description"]
            current = origin.execute(
                "SELECT * FROM observation_embeddings WHERE observation_id = ? "
                "AND representation = 'factor_description' AND requested_model = ? "
                "AND model = ? AND dimensions_mode = 'default' "
                "AND source_text_hash = ? AND source_text = ?",
                (observation_id, MODEL, MODEL, sha256(text.encode()).hexdigest(), text),
            ).fetchall()
            if len(current) != 1:
                raise ValueError(
                    f"Expected one current vector for observation {observation_id}"
                )
            embedding = current[0]
            # Float64 retains every value from the persisted JSON without rounding.
            vector = np.asarray(json.loads(embedding["embedding_json"]), dtype="<f8")
            if vector.ndim != 1 or len(vector) != embedding["dimensions"]:
                raise ValueError("Invalid stored vector shape")
            output.execute(
                "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(
                    embedding[field]
                    for field in (
                        "observation_id",
                        "embedding_id",
                        "representation",
                        "requested_model",
                        "model",
                        "dimensions_mode",
                        "dimensions",
                        "source_text",
                        "source_text_hash",
                        "created_at",
                    )
                )
                + (vector.tobytes(),),
            )
        output.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            (
                ("selected_run_ids", json.dumps(selected)),
                ("vector_encoding", "little-endian IEEE 754 float64"),
                ("source_url_manifest", "scripts/download_pilot_corpus.py:MANIFEST"),
            ),
        )
        if output.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Invalid snapshot foreign keys")
        if output.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Invalid SQLite snapshot")
