"""Offline integration tests: real SQLite transactions and an indexed fake provider."""

import asyncio
import os
import sqlite3
import unittest
from contextlib import asynccontextmanager, closing, redirect_stderr, redirect_stdout
from dataclasses import replace
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from openai import AsyncOpenAI
from sqlalchemy import MetaData, text, update
from sqlalchemy.exc import IntegrityError
from test_extraction import make_chunk

from digital_applications_prototype.core import config
from digital_applications_prototype.core.llm_runtime import configured_embeddings
from digital_applications_prototype.embeddings import cli
from digital_applications_prototype.embeddings.models import (
    EmbeddingInput,
    source_text_hash,
)
from digital_applications_prototype.embeddings.scope import (
    normalize_run_ids,
    read_run_ids,
)
from digital_applications_prototype.embeddings.service import (
    EmbeddingStageError,
    embed_observations,
)
from digital_applications_prototype.extraction.cli import main
from digital_applications_prototype.llm.adapters.base import TokenUsage
from digital_applications_prototype.llm.adapters.openai_embeddings import (
    OpenAIEmbeddingAdapter,
)
from digital_applications_prototype.llm.embeddings import (
    EmbeddingResponse,
    EmbeddingService,
)
from digital_applications_prototype.persistence.database import (
    Database,
    initialize_database,
)
from digital_applications_prototype.persistence.embeddings import EmbeddingRepository
from digital_applications_prototype.persistence.models import (
    Base,
    EvidenceSpanRow,
    Observation,
    ObservationEmbedding,
)
from digital_applications_prototype.persistence.repository import ExtractionRepository

MODEL = "text-embedding-3-large"


def vector_for(value: str) -> list[float]:
    # Distinct, deterministic values expose ordering mistakes without real calls.
    return [float(int(source_text_hash(value)[:6], 16)), float(len(value)), 0.25]


class FakeEmbeddingAdapter:
    def __init__(self, *, fail_on=None, actual_model=None):
        self.calls = []
        self.fail_on = fail_on
        self.actual_model = actual_model

    async def embed(self, *, texts, model):
        self.calls.append((list(texts), model))
        if len(self.calls) == self.fail_on:
            raise RuntimeError("provider failed: private-test-api-key")
        return EmbeddingResponse(
            vectors=[vector_for(value) for value in texts],
            model=self.actual_model or model,
            usage=TokenUsage(input_tokens=7),
        )


async def seed(database, document="DOC_A", descriptions=("Mechanism A", "Mechanism B")):
    repository = ExtractionRepository(database.sessions)
    run_id = await repository.create_run(document, model="extraction-model")
    chunk_id = (await repository.create_chunks(run_id, [make_chunk()]))[0]
    async with database.sessions.begin() as session:
        rows = []
        for index, description in enumerate(descriptions, 1):
            row = Observation(
                run_id=run_id,
                chunk_id=chunk_id,
                observation_index=index,
                observation_type="condition",
                valence="constraint",
                claim_scope="actual",
                observation=f"Observation {index}",
                factor_description=description,
                actors_json=[],
                condition_or_context_json=[],
                action_or_response_json=[],
                evidence_strength="explicit",
                validation_status="valid",
                project_id="PROJECT",
                country="Example",
                document_id=document,
                document_type="PAD",
                chunk_start_page=1,
                chunk_end_page=1,
                raw_observation_json={"factor_description": description},
                evidence_spans=[
                    EvidenceSpanRow(
                        span_index=1, quote="Grounding quote", page=1, role="supports"
                    )
                ],
            )
            session.add(row)
            rows.append(row)
        await session.flush()
        return run_id, [row.observation_id for row in rows]


def stage1_records(path):
    with closing(sqlite3.connect(path)) as connection:
        return {
            name: connection.execute(f'SELECT * FROM "{name}" ORDER BY 1').fetchall()
            for name in (
                "extraction_runs",
                "extraction_chunks",
                "extraction_chunk_sources",
                "implementation_observations",
                "evidence_spans",
                "validation_revisions",
            )
        }


class ManifestTests(unittest.TestCase):
    def test_manifest_blank_comments_duplicates_and_order(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runs.txt"
            path.write_text(
                "# experiment\n\nrun-b\n  run-a  \nrun-b\n", encoding="utf-8"
            )
            self.assertEqual(read_run_ids(path), ["run-b", "run-a"])
        self.assertEqual(normalize_run_ids(["run-a", "run-a"]), ["run-a"])

    def test_empty_and_malformed_manifests_report_location(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runs.txt"
            for content, expected in (
                ("# only a comment\n", "empty"),
                ("run-a\nrun-b,run-c", "runs.txt:2"),
            ):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected):
                        read_run_ids(path)


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    def adapter(self, indexes=(1, 0), vectors=None, model="returned-model"):
        client = Mock(spec=AsyncOpenAI)
        values = (
            vectors
            if vectors is not None
            else [[float(index), 1.0] for index in indexes]
        )
        client.embeddings = SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    model=model,
                    usage=SimpleNamespace(prompt_tokens=12),
                    data=[
                        SimpleNamespace(index=index, embedding=vector)
                        for index, vector in zip(indexes, values, strict=True)
                    ],
                )
            )
        )
        return client, EmbeddingService(OpenAIEmbeddingAdapter(client))

    async def test_reorders_by_index_uses_full_dimensions_actual_model_and_usage(self):
        client, service = self.adapter()
        result = await service.embed(texts=["first", "second"], model=MODEL)
        self.assertEqual(result.vectors, [[0.0, 1.0], [1.0, 1.0]])
        self.assertEqual(result.model, "returned-model")
        self.assertEqual(result.usage, TokenUsage(input_tokens=12))
        client.embeddings.create.assert_awaited_once_with(
            input=["first", "second"], model=MODEL, encoding_format="float"
        )

    async def test_invalid_indexes_and_vector_shapes_fail(self):
        for indexes, vectors in (
            ((0,), None),
            ((0, 0), None),
            ((-1, 0), None),
            ((0, 2), None),
            ((0, 1), [[], []]),
            ((0, 1), [[1.0], [1.0, 2.0]]),
            ((0, 1), [[float("nan")], [1.0]]),
            ((0, 1), [[1.0], [float("inf")]]),
        ):
            with self.subTest(indexes=indexes, vectors=vectors):
                _, service = self.adapter(indexes, vectors)
                with self.assertRaises(ValueError):
                    await service.embed(texts=["a", "b"], model=MODEL)

    async def test_sdk_failure_propagates_and_no_extra_retries_are_added(self):
        client, service = self.adapter()
        client.embeddings.create.side_effect = RuntimeError("unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            await service.embed(texts=["a"], model=MODEL)
        client.embeddings.create.assert_awaited_once()

    async def test_runtime_uses_shared_key_without_generative_configuration(self):
        with TemporaryDirectory() as directory:
            config._load_environment.cache_clear()
            try:
                with (
                    patch.object(config, "ENV_FILE", Path(directory) / ".env"),
                    patch.dict(
                        os.environ,
                        {"OPENAI_API_KEY": "private-test-api-key"},
                        clear=True,
                    ),
                    patch(
                        "digital_applications_prototype.core.llm_runtime.AsyncOpenAI"
                    ) as factory,
                ):
                    async with configured_embeddings() as provider:
                        self.assertIsInstance(provider, EmbeddingService)
                    factory.assert_called_once_with(api_key="private-test-api-key")
                    factory.return_value.__aexit__.assert_awaited_once()
            finally:
                config._load_environment.cache_clear()


class EmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "stage1.sqlite"
        self.database = await initialize_database(self.path)
        self.addAsyncCleanup(self.database.close)
        self.repository = EmbeddingRepository(self.database.sessions)
        self.run_a, self.ids_a = await seed(self.database)
        self.run_b, self.ids_b = await seed(self.database, "DOC_B", ("Mechanism C",))
        self.adapter = FakeEmbeddingAdapter()
        self.provider = EmbeddingService(self.adapter)

    async def plan(self, **kwargs):
        return await self.repository.plan(run_ids=[self.run_a], model=MODEL, **kwargs)

    async def test_scope_single_multiple_normalized_unknown_and_explicit_all(self):
        one = await self.plan()
        self.assertEqual([item.observation_id for item in one.pending], self.ids_a)
        selected = await self.repository.plan(
            run_ids=[self.run_b, self.run_a, self.run_a], model=MODEL
        )
        self.assertEqual(len(selected.runs), 2)
        self.assertEqual(selected.eligible, 3)
        all_runs = await self.repository.plan(all_runs=True, model=MODEL)
        self.assertEqual(selected, all_runs)
        with self.assertRaisesRegex(ValueError, "Unknown extraction run IDs: absent"):
            await self.repository.plan(run_ids=[self.run_a, "absent"], model=MODEL)
        invalid_scopes: tuple[tuple[list[str] | None, bool], ...] = (
            (None, False),
            ([], False),
            ([self.run_a], True),
        )
        for ids, all_runs in invalid_scopes:
            with (
                self.subTest(ids=ids, all_runs=all_runs),
                self.assertRaises(ValueError),
            ):
                await self.repository.plan(model=MODEL, run_ids=ids, all_runs=all_runs)

    async def test_duplicate_documents_fail_before_provider_and_override_is_explicit(
        self,
    ):
        repeated, _ = await seed(self.database)
        plan = await self.repository.plan(run_ids=[self.run_a, repeated], model=MODEL)
        self.assertEqual(plan.conflicts, {"DOC_A": [run.run_id for run in plan.runs]})
        with self.assertRaisesRegex(ValueError, "DOC_A"):
            await embed_observations(self.repository, self.provider, plan)
        self.assertEqual(self.adapter.calls, [])
        result = await embed_observations(
            self.repository, self.provider, plan, allow_multiple_runs_per_document=True
        )
        self.assertEqual(result.embedded, 4)

    async def test_eligibility_repaired_rejected_duplicate_blank_and_unselected(self):
        run, ids = await seed(
            self.database,
            "ELIGIBILITY",
            ("valid", "repaired", "rejected", "duplicate", " ", "\t\n", ""),
        )
        async with self.database.sessions.begin() as session:
            await session.execute(
                update(Observation)
                .where(Observation.observation_id == ids[1])
                .values(validation_status="repaired")
            )
            await session.execute(
                update(Observation)
                .where(Observation.observation_id == ids[2])
                .values(validation_status="rejected")
            )
            await session.execute(
                update(Observation)
                .where(Observation.observation_id == ids[3])
                .values(duplicate_of_observation_id=ids[0])
            )
        plan = await self.repository.plan(run_ids=[run], model=MODEL)
        self.assertEqual([item.observation_id for item in plan.pending], ids[:2])
        await embed_observations(self.repository, self.provider, plan)
        self.assertEqual(self.adapter.calls, [(["valid", "repaired"], MODEL)])

    async def test_legacy_nullable_factor_description_is_excluded(self):
        # Current Stage 1 disallows NULL. Also support an older/foreign nullable DB.
        metadata = MetaData()
        for table in Base.metadata.sorted_tables:
            table.to_metadata(metadata)
        metadata.tables[Observation.__tablename__].c.factor_description.nullable = True
        legacy = Database(Path(self.temporary.name) / "legacy.sqlite")
        self.addAsyncCleanup(legacy.close)
        async with legacy.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        run, ids = await seed(legacy)
        async with legacy.sessions.begin() as session:
            await session.execute(
                text(
                    "UPDATE implementation_observations SET factor_description=NULL WHERE observation_id=:id"
                ),
                {"id": ids[0]},
            )
        plan = await EmbeddingRepository(legacy.sessions).plan(
            run_ids=[run], model=MODEL
        )
        self.assertEqual([item.observation_id for item in plan.pending], ids[1:])

    async def test_exact_text_hash_roundtrip_foreign_key_and_joined_provenance(self):
        value = "  Unicode mécanisme\nunchanged.  "
        async with self.database.sessions.begin() as session:
            await session.execute(
                update(Observation)
                .where(Observation.observation_id == self.ids_a[0])
                .values(factor_description=value)
            )
        plan = await self.plan()
        report = await embed_observations(self.repository, self.provider, plan)
        self.assertEqual((report.embedded, report.input_tokens), (2, 7))
        self.assertEqual(self.adapter.calls[0][0][0], value)
        rows = await self.repository.get_embeddings([self.run_a], model=MODEL)
        self.assertEqual(len(rows), 2)
        for embedding, observation in rows:
            self.assertEqual(embedding.observation_id, observation.observation_id)
            self.assertEqual(
                embedding.embedding_json, vector_for(embedding.source_text)
            )
            self.assertEqual(
                embedding.source_text_hash,
                sha256(embedding.source_text.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(embedding.dimensions, len(embedding.embedding_json))
            self.assertEqual(embedding.model, MODEL)
            self.assertEqual(embedding.representation, "factor_description")
            self.assertEqual(embedding.dimensions_mode, "default")
            self.assertEqual(observation.run_id, self.run_a)
            self.assertEqual(observation.document_id, "DOC_A")
            self.assertEqual(observation.project_id, "PROJECT")
            self.assertTrue(embedding.created_at)
        with self.assertRaises(IntegrityError):
            await self.repository.save_batch(
                [EmbeddingInput(-1, "missing")],
                EmbeddingResponse([[1.0]], MODEL),
                requested_model=MODEL,
            )

    async def test_idempotence_changed_text_and_history(self):
        await embed_observations(self.repository, self.provider, await self.plan())
        second = await self.plan()
        self.assertEqual((second.already_current, len(second.pending)), (2, 0))
        await embed_observations(self.repository, self.provider, second)
        self.assertEqual(len(self.adapter.calls), 1)
        async with self.database.sessions.begin() as session:
            await session.execute(
                update(Observation)
                .where(Observation.observation_id == self.ids_a[0])
                .values(factor_description="Changed mechanism")
            )
        changed = await self.plan()
        self.assertEqual(
            [item.observation_id for item in changed.pending], self.ids_a[:1]
        )
        await embed_observations(self.repository, self.provider, changed)
        current = await self.repository.get_embeddings([self.run_a], model=MODEL)
        historical = await self.repository.get_embeddings(
            [self.run_a], model=MODEL, current_only=False
        )
        self.assertEqual((len(current), len(historical)), (2, 3))
        self.assertEqual(historical[0][0].source_text, "Mechanism A")
        self.assertEqual((await self.plan()).already_current, 2)

    async def test_actual_returned_model_is_stored_and_requested_alias_is_idempotent(
        self,
    ):
        adapter = FakeEmbeddingAdapter(actual_model="provider-snapshot")
        await embed_observations(
            self.repository, EmbeddingService(adapter), await self.plan()
        )
        rows = await self.repository.get_embeddings(
            [self.run_a], model="provider-snapshot"
        )
        self.assertTrue(all(row.requested_model == MODEL for row, _ in rows))
        self.assertEqual((await self.plan()).already_current, 2)

    async def test_models_representations_and_dimensions_coexist(self):
        plan = await self.plan()
        await embed_observations(self.repository, self.provider, plan)
        small = await self.repository.plan(
            run_ids=[self.run_a], model="text-embedding-3-small"
        )
        await embed_observations(self.repository, self.provider, small)
        await self.repository.save_batch(
            plan.pending,
            EmbeddingResponse([[1.0], [2.0]], MODEL),
            requested_model=MODEL,
            representation="observation",
        )
        await self.repository.save_batch(
            plan.pending,
            EmbeddingResponse([[1.0], [2.0]], MODEL),
            requested_model=MODEL,
            dimensions_mode="explicit:1",
        )
        self.assertEqual(
            len(await self.repository.get_embeddings([self.run_a], model=MODEL)), 2
        )
        self.assertEqual(
            len(
                await self.repository.get_embeddings(
                    [self.run_a], model=MODEL, dimensions_mode=None
                )
            ),
            4,
        )
        self.assertEqual(
            len(
                await self.repository.get_embeddings(
                    [self.run_a],
                    model=MODEL,
                    dimensions_mode="explicit:1",
                    dimensions=1,
                )
            ),
            2,
        )
        self.assertEqual(
            len(
                await self.repository.get_embeddings(
                    [self.run_a],
                    model=MODEL,
                    representation="observation",
                    current_only=False,
                )
            ),
            2,
        )
        self.assertEqual((await self.plan()).already_current, 2)
        with self.assertRaisesRegex(ValueError, "current_only=False"):
            await self.repository.get_embeddings(
                [self.run_a], model=MODEL, representation="observation"
            )

    async def test_alternate_representation_or_reduced_vector_does_not_skip_full_request(
        self,
    ):
        plan = await self.plan()
        response = EmbeddingResponse([[1.0], [2.0]], MODEL)
        await self.repository.save_batch(
            plan.pending, response, requested_model=MODEL, representation="observation"
        )
        await self.repository.save_batch(
            plan.pending, response, requested_model=MODEL, dimensions_mode="explicit:1"
        )
        self.assertEqual(len((await self.plan()).pending), 2)

    async def test_batch_mapping_independent_of_database_insertion_order(self):
        # Reverse logical observation order after insertion; vectors must follow IDs.
        async with self.database.sessions.begin() as session:
            for original, replacement in ((1, 99), (2, 1), (99, 2)):
                await session.execute(
                    update(Observation)
                    .where(
                        Observation.run_id == self.run_a,
                        Observation.observation_index == original,
                    )
                    .values(observation_index=replacement)
                )
        plan = await self.plan()
        self.assertEqual(
            [item.observation_id for item in plan.pending], list(reversed(self.ids_a))
        )
        await embed_observations(self.repository, self.provider, plan)
        rows = await self.repository.get_embeddings([self.run_a], model=MODEL)
        for embedding, observation in rows:
            self.assertEqual(
                embedding.embedding_json, vector_for(observation.factor_description)
            )
            self.assertEqual(
                embedding.source_text_hash,
                source_text_hash(observation.factor_description),
            )

    async def test_later_provider_failure_keeps_commits_and_retry_skips_successes(self):
        plan = await self.repository.plan(all_runs=True, model=MODEL, batch_size=1)
        adapter = FakeEmbeddingAdapter(fail_on=2)
        with self.assertRaises(EmbeddingStageError) as raised:
            await embed_observations(self.repository, EmbeddingService(adapter), plan)
        report = raised.exception.report
        self.assertEqual(
            (
                report.submitted,
                report.embedded,
                report.failed,
                report.completed_batches,
            ),
            (2, 1, 1, 1),
        )
        # An independent connection sees the durable first batch.
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT count(*) FROM observation_embeddings"
                ).fetchone()[0],
                1,
            )
        retry = await self.repository.plan(all_runs=True, model=MODEL, batch_size=100)
        self.assertEqual((retry.already_current, len(retry.pending)), (1, 2))
        await embed_observations(self.repository, self.provider, retry)
        self.assertEqual(len(self.adapter.calls), 1)
        self.assertEqual(len(self.adapter.calls[0][0]), 2)

    async def test_entire_batch_rolls_back_on_database_failure(self):
        async with self.database.engine.begin() as connection:
            await connection.execute(
                text(f"""
                CREATE TRIGGER fail_embedding BEFORE INSERT ON observation_embeddings
                WHEN NEW.observation_id = {self.ids_a[1]}
                BEGIN SELECT RAISE(ABORT, 'test failure'); END
            """)
            )
        with self.assertRaises(EmbeddingStageError):
            await embed_observations(self.repository, self.provider, await self.plan())
        self.assertEqual((await self.plan()).already_current, 0)

    async def test_uniqueness_and_dimension_constraints(self):
        plan = await self.plan()
        response = EmbeddingResponse([[1.0], [2.0]], MODEL)
        self.assertEqual(
            await self.repository.save_batch(
                plan.pending, response, requested_model=MODEL
            ),
            2,
        )
        self.assertEqual(
            await self.repository.save_batch(
                plan.pending, response, requested_model=MODEL
            ),
            0,
        )
        with self.assertRaises(IntegrityError):
            async with self.database.sessions.begin() as session:
                await session.execute(
                    update(ObservationEmbedding).values(dimensions=99)
                )
        for vectors in (
            [[1.0]],
            [[], []],
            [[1.0], [float("nan")]],
            [[1.0], [1.0, 2.0]],
        ):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                await self.repository.save_batch(
                    plan.pending,
                    EmbeddingResponse(vectors, MODEL),
                    requested_model=MODEL,
                )

    async def test_stage1_rows_and_raw_output_unchanged(self):
        before = stage1_records(self.path)
        await embed_observations(self.repository, self.provider, await self.plan())
        self.assertEqual(stage1_records(self.path), before)

    async def test_readonly_planning_before_migration_and_additive_initialization(self):
        async with self.database.engine.begin() as connection:
            await connection.execute(text("DROP TABLE observation_embeddings"))
        before = stage1_records(self.path)
        read_only = Database(self.path, read_only=True)
        try:
            plan = await EmbeddingRepository(read_only.sessions).plan(
                run_ids=[self.run_a], model=MODEL
            )
            self.assertEqual(len(plan.pending), 2)
        finally:
            await read_only.close()
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertIsNone(
                connection.execute(
                    "SELECT name FROM sqlite_master WHERE name='observation_embeddings'"
                ).fetchone()
            )
        await self.repository.initialize()
        await self.repository.initialize()
        self.assertEqual(stage1_records(self.path), before)

    async def test_invalid_execution_options_fail_before_provider(self):
        plan = await self.plan()
        for invalid in (
            replace(plan, batch_size=0),
            replace(plan, batch_size=2049),
            replace(plan, model=""),
            replace(plan, representation="other"),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                await embed_observations(self.repository, self.provider, invalid)
        self.assertEqual(self.adapter.calls, [])


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "stage1.sqlite"

        async def prepare():
            database = await initialize_database(self.path)
            try:
                first, _ = await seed(database)
                second, _ = await seed(database, "DOC_B", ("C",))
                repeated, _ = await seed(database)
                return first, second, repeated
            finally:
                await database.close()

        self.run_a, self.run_b, self.repeated = asyncio.run(prepare())
        self.adapter = FakeEmbeddingAdapter()

        @asynccontextmanager
        async def runtime():
            yield EmbeddingService(self.adapter)

        self.runtime = self.enterContext(
            patch.object(cli, "configured_embeddings", side_effect=runtime)
        )
        self.enterContext(
            patch.object(
                cli.EmbeddingSettings,
                "from_env",
                return_value=config.EmbeddingSettings(),
            )
        )

    def invoke(self, args):
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            result = main(["embed", "--database", str(self.path), *args])
        return result, output.getvalue(), errors.getvalue()

    def test_single_repeated_manifest_and_all_scopes(self):
        manifest = self.root / "runs.txt"
        manifest.write_text(
            f"# selection\n{self.run_a}\n\n{self.run_b}\n{self.run_a}\n",
            encoding="utf-8",
        )
        for args, runs, eligible in (
            (["--run-id", self.run_a], 1, 2),
            (["--run-id", self.run_a, "--run-id", self.run_b], 2, 3),
            (["--run-ids-file", str(manifest)], 2, 3),
            (["--all", "--allow-multiple-runs-per-document"], 3, 5),
        ):
            with self.subTest(args=args):
                result, output, errors = self.invoke([*args, "--dry-run"])
                self.assertEqual(result, 0, errors)
                self.assertIn(f"Selected extraction runs: {runs}", output)
                self.assertIn(f"Eligible observations: {eligible}", output)
                self.assertIn("Representation: factor_description", output)
                self.assertIn("Batch size: 100", output)
        self.runtime.assert_not_called()

    def test_no_scope_or_ambiguous_scope_and_bad_batch_are_rejected(self):
        for args in (
            [],
            ["--all", "--run-id", self.run_a],
            ["--run-id", self.run_a, "--run-ids-file", "runs.txt"],
            ["--all", "--run-ids-file", "runs.txt"],
            ["--all", "--batch-size", "0"],
        ):
            with (
                self.subTest(args=args),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                self.invoke(args)
            self.assertEqual(raised.exception.code, 2)
        self.runtime.assert_not_called()

    def test_unknown_malformed_empty_and_missing_manifests_fail_before_calls(self):
        empty = self.root / "empty.txt"
        empty.write_text("# empty\n", encoding="utf-8")
        for args in (
            ["--run-id", "absent"],
            ["--run-id", "bad run"],
            ["--run-ids-file", str(empty)],
            ["--run-ids-file", str(self.root / "missing.txt")],
        ):
            with self.subTest(args=args):
                result, _, errors = self.invoke(args)
                self.assertEqual(result, 1)
                self.assertIn("failed", errors)
        self.runtime.assert_not_called()

    def test_duplicate_document_reported_in_dry_and_live_before_calls(self):
        before = stage1_records(self.path)
        for extra in ([], ["--dry-run"]):
            with self.subTest(extra=extra):
                result, output, errors = self.invoke(
                    ["--run-id", self.run_a, "--run-id", self.repeated, *extra]
                )
                self.assertEqual(result, 1)
                self.assertIn("CONFLICT DOC_A", output)
                self.assertIn(self.run_a, errors)
                self.assertIn(self.repeated, errors)
        self.assertEqual(stage1_records(self.path), before)
        self.runtime.assert_not_called()

    def test_dryrun_no_database_mutation_or_migration(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("DROP TABLE observation_embeddings")
        before = self.path.read_bytes()
        result, _, errors = self.invoke(["--run-id", self.run_a, "--dry-run"])
        self.assertEqual(result, 0, errors)
        self.assertEqual(self.path.read_bytes(), before)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertIsNone(
                connection.execute(
                    "SELECT name FROM sqlite_master WHERE name='observation_embeddings'"
                ).fetchone()
            )
        self.runtime.assert_not_called()

    def test_manifest_live_run_and_rerun_are_idempotent_and_stage1_unchanged(self):
        manifest = self.root / "runs.txt"
        manifest.write_text(f"{self.run_b}\n{self.run_a}\n", encoding="utf-8")
        before = stage1_records(self.path)
        for attempt in range(2):
            result, output, errors = self.invoke(
                ["--run-ids-file", str(manifest), "--batch-size", "2"]
            )
            self.assertEqual(result, 0, errors)
            if attempt == 0:
                self.assertIn("Embedded successfully: 3", output)
                self.assertIn("Dimensions: 3", output)
                self.assertIn("Input tokens (provider-reported): 14", output)
            else:
                self.assertIn("Already embedded/current: 3", output)
                self.assertIn("Submitted: 0", output)
        self.assertEqual(len(self.adapter.calls), 2)
        self.assertEqual(self.runtime.call_count, 1)
        self.assertEqual(stage1_records(self.path), before)

    def test_model_override_and_failure_summary_do_not_expose_provider_secret(self):
        self.adapter.fail_on = 2
        result, output, errors = self.invoke(
            ["--run-id", self.run_a, "--batch-size", "1", "--model", "override-model"]
        )
        self.assertEqual(result, 1)
        self.assertIn("Embedded successfully: 1", output)
        self.assertIn("Failed: 1", output)
        self.assertIn("RuntimeError", errors)
        self.assertNotIn("private-test-api-key", output + errors)
        self.assertTrue(
            all(model == "override-model" for _, model in self.adapter.calls)
        )
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT model FROM observation_embeddings"
                ).fetchone()[0],
                "override-model",
            )

    def test_missing_database_is_not_created(self):
        self.path = self.root / "missing.sqlite"
        result, _, errors = self.invoke(["--all", "--dry-run"])
        self.assertEqual(result, 1)
        self.assertIn("does not exist", errors)
        self.assertFalse(self.path.exists())
