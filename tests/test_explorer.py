"""Offline snapshot and explorer checks using the real Stage 1 schema."""

import json
import sqlite3
import unittest
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from digital_applications_prototype.embeddings.scope import read_run_ids
from digital_applications_prototype.explorer.data import (
    MODEL,
    connect_readonly,
    filter_observations,
    load_corpus,
    related_observations,
)
from digital_applications_prototype.explorer.snapshot import (
    create_snapshot,
    source_url_mapping,
)
from digital_applications_prototype.persistence.models import (
    Base,
    EvidenceSpanRow,
    ExtractionChunk,
    ExtractionRun,
    Observation,
    ObservationEmbedding,
)
from scripts.download_pilot_corpus import MANIFEST

ROOT = Path(__file__).resolve().parents[1]
URL = "https://documents.worldbank.org/curated/en/example/pdf/example.pdf"


def seed_source(path: Path) -> None:
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session, session.begin():
            for run, document in (
                ("run-a", "A"),
                ("run-b", "B"),
                ("run-c", "C"),
                ("run-d", "D"),
                ("history", "A"),
            ):
                session.add(
                    ExtractionRun(
                        run_id=run,
                        document_id=document,
                        started_at="2026-11-02",
                        status="completed",
                        schema_version="stage1-v3",
                    )
                )
                session.flush()
                session.add(
                    ExtractionChunk(
                        run_id=run,
                        chunk_index=1,
                        start_page=1,
                        end_page=2,
                        status="completed",
                    )
                )
            session.flush()
            chunks = {c.run_id: c.chunk_id for c in session.query(ExtractionChunk)}
            for observation_id, run, document, country, vector in (
                (10, "run-a", "A", "comoros", [10.0, 0.0]),
                (20, "run-a", "A", "comoros", [1.0, 0.0]),
                (30, "run-b", "B", "sierra-leone", [2.0, 0.0]),
                (40, "run-b", "B", "sierra-leone", [100.0, 100.0]),
                (50, "run-c", "C", "mexico", [0.0, 1.0]),
                (60, "run-c", "C", "mexico", [-1.0, 0.0]),
                (70, "run-d", "D", "comoros", [1.0, 2.0]),
                (80, "run-b", "B", "sierra-leone", [1.0, 2.0]),
                (90, "history", "A", "comoros", [1.0, 0.0]),
                (100, "run-a", "A", "comoros", [1.0, 0.0]),
                (110, "run-a", "A", "comoros", [1.0, 0.0]),
                (120, "run-a", "A", "comoros", [1.0, 0.0]),
            ):
                description = f"Capacity mechanism {observation_id}"
                if observation_id == 120:
                    description = " \t\n "
                row = Observation(
                    observation_id=observation_id,
                    # Domain provenance can intentionally repeat, unlike integer identity.
                    provenance_id="shared-domain-id",
                    run_id=run,
                    chunk_id=chunks[run],
                    observation_index=observation_id,
                    observation_type="condition",
                    valence="constraint",
                    claim_scope="planned" if observation_id == 40 else "actual",
                    observation=f"Agency observation {observation_id}",
                    factor_description=description,
                    implementation_area="Delivery",
                    actors_json=["Agency"],
                    condition_or_context_json=["Rural"],
                    action_or_response_json=["Training"],
                    observed_effect="Delay",
                    evidence_strength="explicit",
                    validation_status="rejected" if observation_id == 100 else "valid",
                    duplicate_of_observation_id=10 if observation_id == 110 else None,
                    project_id=f"P{document}",
                    country=country,
                    document_id=document,
                    document_type="ISR" if document == "B" else "PAD",
                    chunk_start_page=1,
                    chunk_end_page=2,
                    raw_observation_json={},
                    evidence_spans=[
                        EvidenceSpanRow(
                            span_index=1,
                            quote=f"Exact source quote {observation_id}",
                            page=2,
                            role="supports",
                        )
                    ],
                )
                if observation_id == 10:
                    row.evidence_spans.append(
                        EvidenceSpanRow(
                            span_index=2,
                            quote="Additional context",
                            page=1,
                            role="context",
                        )
                    )
                session.add(row)
                session.flush()
                session.add(
                    ObservationEmbedding(
                        observation_id=observation_id,
                        representation="factor_description",
                        requested_model=MODEL,
                        model=MODEL,
                        dimensions_mode="default",
                        dimensions=2,
                        source_text=description,
                        source_text_hash=sha256(description.encode()).hexdigest(),
                        embedding_json=vector,
                        created_at="2026-11-02",
                    )
                )
    finally:
        engine.dispose()


class ExplorerTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name) / "stage1.sqlite"
        self.snapshot = Path(temporary.name) / "demo.sqlite"
        self.runs = ["run-a", "run-b", "run-c", "run-d"]
        seed_source(self.source)

    def export(self):
        return create_snapshot(self.source, self.snapshot, self.runs, {"A": URL})

    def test_selected_runs_eligibility_and_lossless_identity_join(self):
        original = self.source.read_bytes()
        report = self.export()
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual((report["observations"], report["documents"]), (8, 4))
        corpus = load_corpus(self.snapshot)
        self.assertEqual(
            [o.observation_id for o in corpus.observations], list(range(10, 90, 10))
        )
        self.assertEqual(
            {o.provenance_id for o in corpus.observations}, {"shared-domain-id"}
        )
        with connect_readonly(self.source) as source:
            for i, observation in enumerate(corpus.observations):
                embedding = source.execute(
                    "SELECT embedding_json FROM observation_embeddings WHERE observation_id = ?",
                    (observation.observation_id,),
                ).fetchone()[0]
                np.testing.assert_array_equal(corpus.vectors[i], json.loads(embedding))
        self.assertEqual(
            [e.role for e in corpus.get(10).evidence], ["supports", "context"]
        )
        self.assertEqual([e.page for e in corpus.get(10).evidence], [2, 1])
        self.assertEqual(corpus.get(10).source_url, URL)
        self.assertIsNone(corpus.get(30).source_url)

    def test_readonly_and_missing_database(self):
        self.export()
        original = self.snapshot.read_bytes()
        for immutable in (False, True):
            with connect_readonly(self.snapshot, immutable=immutable) as connection:
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute("DELETE FROM observations")
                connection.execute("PRAGMA query_only = OFF")
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute("CREATE TABLE changed (id INTEGER)")
        corpus = load_corpus(self.snapshot)
        with self.assertRaises(ValueError):
            corpus.vectors[0, 0] = 5
        with self.assertRaises(ValueError):
            corpus.norms[0] = 5
        self.assertEqual(self.snapshot.read_bytes(), original)
        self.assertFalse(self.snapshot.with_name("demo.sqlite-wal").exists())
        missing = self.snapshot.parent / "missing.sqlite"
        with self.assertRaises(sqlite3.OperationalError), connect_readonly(missing):
            pass
        self.assertFalse(missing.exists())

    def test_unknown_duplicate_unfinished_and_empty_scope_fail(self):
        for runs in ([], ["unknown"], ["run-a", "history"]):
            with self.subTest(runs=runs), self.assertRaises(ValueError):
                create_snapshot(self.source, self.snapshot, runs, {})
            self.assertFalse(self.snapshot.exists())
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "UPDATE extraction_runs SET status = 'partial' WHERE run_id = 'run-a'"
            )
        with self.assertRaisesRegex(ValueError, "completed"):
            self.export()

    def test_stale_missing_or_ambiguous_vectors_fail(self):
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "UPDATE observation_embeddings SET source_text = 'old', source_text_hash = ? WHERE observation_id = 10",
                ("0" * 64,),
            )
        with self.assertRaisesRegex(ValueError, "current vector"):
            self.export()
        self.assertFalse(self.snapshot.exists())
        self.source.unlink()
        seed_source(self.source)
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute(
                "INSERT INTO observation_embeddings (observation_id, representation, requested_model, model, dimensions_mode, dimensions, source_text, source_text_hash, embedding_json, created_at) SELECT observation_id, representation, requested_model, model, dimensions_mode, 3, source_text, source_text_hash, '[1,0,0]', created_at FROM observation_embeddings WHERE observation_id = 10"
            )
        with self.assertRaisesRegex(ValueError, "current vector"):
            self.export()

    def test_bad_evidence_counts_and_existing_destination_fail(self):
        with self.assertRaisesRegex(ValueError, "Expected 15 documents"):
            create_snapshot(
                self.source, self.snapshot, self.runs, {}, expected_documents=15
            )
        self.assertFalse(self.snapshot.exists())
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute("DELETE FROM evidence_spans WHERE observation_id = 10")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.export()
        self.snapshot.write_bytes(b"existing artifact")
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.export()
        self.assertEqual(self.snapshot.read_bytes(), b"existing artifact")

    def test_zero_or_nonfinite_vectors_fail(self):
        for vector in ("[0,0]", "[1,1e999]"):
            with self.subTest(vector=vector):
                with closing(sqlite3.connect(self.source)) as connection, connection:
                    connection.execute(
                        "UPDATE observation_embeddings SET embedding_json = ? WHERE observation_id = 10",
                        (vector,),
                    )
                with self.assertRaisesRegex(ValueError, "finite and nonzero"):
                    self.export()
                self.assertFalse(self.snapshot.exists())

    def test_literal_keyword_search_and_all_metadata_filters(self):
        self.export()
        observations = load_corpus(self.snapshot).observations
        self.assertEqual(len(filter_observations(observations, keyword="CAPACITY")), 8)
        self.assertEqual(len(filter_observations(observations, keyword="agency")), 8)
        self.assertEqual(len(filter_observations(observations, keyword="delivery")), 8)
        for keyword in ("%", "_", "' OR 1=1 --", "missing"):
            self.assertEqual(filter_observations(observations, keyword=keyword), ())
        chosen = filter_observations(
            observations,
            filters={
                "country": ["sierra-leone"],
                "project_id": ["PB"],
                "document_id": ["B"],
                "document_type": ["ISR"],
                "observation_type": ["condition"],
                "valence": ["constraint"],
                "claim_scope": ["planned"],
            },
        )
        self.assertEqual([o.observation_id for o in chosen], [40])
        self.assertEqual(
            len(filter_observations(observations, filters={"country": []})), 8
        )
        with self.assertRaises(ValueError):
            filter_observations(
                observations, filters={"implementation_area": ["Delivery"]}
            )

    def test_cosine_norms_top_five_self_exclusion_and_constraints(self):
        self.export()
        corpus = load_corpus(self.snapshot)
        related = related_observations(corpus, 10)
        self.assertEqual([o.observation_id for o, _ in related], [20, 30, 40, 70, 80])
        self.assertAlmostEqual(related[0][1], 1.0)
        self.assertAlmostEqual(related[2][1], 2**-0.5)
        other_documents = related_observations(corpus, 10, other_documents=True)
        self.assertEqual(
            [o.observation_id for o, _ in other_documents], [30, 40, 70, 80, 50]
        )
        other_countries = related_observations(corpus, 10, other_countries=True)
        self.assertEqual(
            [o.observation_id for o, _ in other_countries], [30, 40, 80, 50, 60]
        )
        self.assertEqual(
            other_countries,
            related_observations(
                corpus, 10, other_documents=True, other_countries=True
            ),
        )
        self.assertEqual(
            related_observations(corpus, 50, other_countries=True)[0][0].observation_id,
            70,
        )


class FrozenArtifactTests(unittest.TestCase):
    def test_frozen_contents_manifest_checksum_and_source_urls(self):
        path = ROOT / "demo/evidence.sqlite"
        report = json.loads((ROOT / "demo/evidence.json").read_text())
        self.assertEqual(path.stat().st_size, report["bytes"])
        self.assertEqual(sha256(path.read_bytes()).hexdigest(), report["sha256"])
        corpus = load_corpus(path)
        self.assertEqual(len(corpus.observations), 1093)
        self.assertEqual(corpus.vectors.shape, (1093, 3072))
        self.assertEqual(len({o.document_id for o in corpus.observations}), 15)
        self.assertEqual(len({o.country for o in corpus.observations}), 3)
        self.assertEqual(sum(len(o.evidence) for o in corpus.observations), 1187)
        self.assertEqual(
            {o.run_id for o in corpus.observations},
            set(read_run_ids(ROOT / "demo/selected_runs.txt")),
        )
        urls = source_url_mapping(MANIFEST)
        for observation in corpus.observations:
            self.assertEqual(observation.source_url, urls[observation.document_id])
            self.assertTrue(observation.provenance_id)
            self.assertTrue(all(e.page > 0 and e.quote for e in observation.evidence))
        with connect_readonly(path) as connection:
            self.assertEqual(
                connection.execute("PRAGMA integrity_check").fetchone()[0], "ok"
            )
            self.assertEqual(
                connection.execute("PRAGMA foreign_key_check").fetchall(), []
            )
            self.assertEqual(
                connection.execute("SELECT count(*) FROM selected_runs").fetchone()[0],
                15,
            )

    def test_source_mapping_excludes_holdouts_and_rejects_bad_links(self):
        mapping = source_url_mapping(MANIFEST)
        self.assertEqual(len(mapping), 15)
        self.assertFalse(any("HOLDOUT" in key for key in mapping))
        self.assertIn("662311592667181272", mapping["KM_PAD_2020"])
        with self.assertRaises(ValueError):
            source_url_mapping([("input", "example", "DOC.pdf", "https://example.com")])


if __name__ == "__main__":
    unittest.main()
