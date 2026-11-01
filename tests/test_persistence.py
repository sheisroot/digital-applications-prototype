import asyncio
import sqlite3
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from datetime import UTC, date
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from test_extraction import (
    DOCUMENT,
    QUOTE,
    FakeAdapter,
    extractor,
    make_chunk,
    observation,
    span,
    write_pdf,
)

from digital_applications_prototype.extraction.chunking import chunk_pages
from digital_applications_prototype.extraction.cli import run_documents
from digital_applications_prototype.extraction.models import PageText, RunStatus
from digital_applications_prototype.llm.adapters.base import TokenUsage
from digital_applications_prototype.persistence.database import initialize_database
from digital_applications_prototype.persistence.models import (
    EvidenceSpanRow,
    ExtractionChunk,
    ExtractionRun,
    Observation,
)
from digital_applications_prototype.persistence.repository import (
    ExtractionRepository,
    derive_run_status,
)


class StatusTests(unittest.TestCase):
    def test_status_follows_chunk_coverage(self):
        for statuses, expected in (
            (["completed", "completed"], RunStatus.COMPLETED),
            (["completed", "failed"], RunStatus.PARTIAL),
            (["failed", "failed"], RunStatus.FAILED),
            ([], RunStatus.FAILED),
            (["pending", "running"], RunStatus.FAILED),
            (["completed", "running", "pending"], RunStatus.PARTIAL),
        ):
            with self.subTest(statuses=statuses):
                self.assertEqual(derive_run_status(statuses), expected)


class RepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "nested/stage1.sqlite"
        self.database = await initialize_database(self.path)
        self.addAsyncCleanup(self.database.close)
        self.repository = ExtractionRepository(self.database.sessions)
        self.run_id = await self.repository.create_run(DOCUMENT.document_id, model="offline-model")

    async def store(self, items=None, *, chunk=None, usage=None):
        chunk = chunk or make_chunk()
        ids = await self.repository.create_chunks(self.run_id, [chunk])
        result = await extractor(FakeAdapter(items=items, usages=[usage])).extract(chunk)
        await self.repository.mark_chunk_running(ids[0])
        await self.repository.persist_completed_chunk(ids[0], chunk, result)
        return ids[0], result

    async def test_initialization_schema_pragmas_and_pending_chunks(self):
        self.assertTrue(self.path.exists())
        await self.database.initialize()  # Explicit initialization is idempotent.
        async with self.database.engine.connect() as connection:
            names = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
            self.assertEqual(set(names), {"extraction_runs", "extraction_chunks",
                                         "implementation_observations", "evidence_spans"})
            self.assertEqual(await connection.scalar(text("PRAGMA journal_mode")), "wal")
            self.assertEqual(await connection.scalar(text("PRAGMA foreign_keys")), 1)
            self.assertEqual(await connection.scalar(text("PRAGMA synchronous")), 2)
        ids = await self.repository.create_chunks(self.run_id, [make_chunk()])
        async with self.database.sessions() as session:
            run = await session.get(ExtractionRun, self.run_id)
            chunk = await session.get(ExtractionChunk, ids[0])
            self.assertIsNone(run.status)
            self.assertIsNone(run.finished_at)
            self.assertEqual(chunk.status, "pending")
            self.assertIsNone(chunk.started_at)
        with self.assertRaises(IntegrityError):
            await self.repository.create_chunks(self.run_id, [make_chunk()])
        with self.assertRaises(IntegrityError):
            await self.repository.create_chunks("missing-run", [make_chunk()])

    async def test_completed_chunk_round_trip_metadata_order_and_tokens(self):
        quote = "Actual Date: May 13, 2022"
        chunk = make_chunk(QUOTE + "\n" + quote)
        item = observation(
            implementation_area="Service delivery", actors=["Agency", "Providers"],
            condition_or_context=["Rural coverage"], action_or_response=["Expanded access"],
            observed_effect="Delivery constrained", evidence={"summary": "Dated context", "spans": [
                span(quote, evidence_date="2022-05-13", role="context"), span(role="qualifies"),
            ]},
        )
        chunk_id, result = await self.store([item], chunk=chunk, usage=TokenUsage(input_tokens=123, output_tokens=45))
        # Read with an independent connection before finalization: the chunk is already committed.
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT status, input_tokens, output_tokens FROM extraction_chunks").fetchone(),
                             ("completed", 123, 45))
            self.assertEqual(connection.execute("SELECT count(*) FROM evidence_spans").fetchone()[0], 2)
        self.assertEqual(await self.repository.get_clean_observations(self.run_id), result.observations)
        async with self.database.sessions() as session:
            chunk_row = await session.get(ExtractionChunk, chunk_id)
            self.assertLessEqual(chunk_row.started_at, chunk_row.finished_at)
            spans = list(await session.scalars(select(EvidenceSpanRow).order_by(EvidenceSpanRow.span_index)))
            self.assertEqual([s.span_index for s in spans], [1, 2])
            self.assertEqual([s.role for s in spans], ["context", "qualifies"])
            self.assertEqual(spans[0].evidence_date, date(2022, 5, 13))
        await self.repository.finalize_run(self.run_id)
        summary = await self.repository.get_run_summary(self.run_id)
        self.assertEqual((summary.status, summary.input_tokens, summary.output_tokens), (RunStatus.COMPLETED, 123, 45))
        self.assertEqual(summary.model, "offline-model")
        self.assertEqual(summary.schema_version, "stage1-v3")
        self.assertEqual(summary.max_concurrency, 1)
        self.assertEqual(summary.started_at.utcoffset(), UTC.utcoffset(None))
        self.assertGreaterEqual(summary.finished_at, summary.started_at)
        async with self.database.sessions() as session:
            run = await session.get(ExtractionRun, self.run_id)
            self.assertEqual((run.input_tokens, run.output_tokens), (123, 45))

    async def test_failed_evidence_insert_rolls_back_entire_chunk(self):
        # Force a real mid-transaction failure after observation insertion and the first span.
        async with self.database.engine.begin() as connection:
            await connection.execute(text("""
                CREATE TRIGGER reject_second_span BEFORE INSERT ON evidence_spans
                WHEN NEW.span_index = 2 BEGIN SELECT RAISE(ABORT, 'evidence insert failed'); END
            """))
        chunk = make_chunk()
        item = observation(evidence={"spans": [span(), span(role="context")]})
        chunk_id = (await self.repository.create_chunks(self.run_id, [chunk]))[0]
        result = await extractor(FakeAdapter(items=[item], usages=[TokenUsage(input_tokens=10, output_tokens=20)])).extract(chunk)
        await self.repository.mark_chunk_running(chunk_id)
        with self.assertRaises(IntegrityError):
            await self.repository.persist_completed_chunk(chunk_id, chunk, result)
        async with self.database.sessions() as session:
            row = await session.get(ExtractionChunk, chunk_id)
            self.assertEqual(row.status, "running")
            self.assertIsNone(row.finished_at)
            self.assertIsNone(row.input_tokens)
            self.assertIsNone(row.output_tokens)
            self.assertEqual(await session.scalar(select(func.count()).select_from(Observation)), 0)
            self.assertEqual(await session.scalar(select(func.count()).select_from(EvidenceSpanRow)), 0)
        await self.repository.mark_chunk_failed(chunk_id, RuntimeError("storage failed"))
        await self.repository.finalize_run(self.run_id)
        self.assertEqual((await self.repository.get_run_summary(self.run_id)).status, RunStatus.FAILED)

    async def test_validation_outcomes_and_original_repaired_rejected_output_survive(self):
        items = [observation(),
                 observation(observation_type="action", evidence={"spans": [span(evidence_date="2020-10-05")]}),
                 observation(evidence={"spans": [span("Invented evidence")]}),
                 observation(evidence={"spans": []})]
        await self.store(items)
        async with self.database.sessions() as session:
            rows = list(await session.scalars(select(Observation).order_by(Observation.observation_index)))
            self.assertEqual([r.validation_status for r in rows], ["valid", "repaired", "rejected", "rejected"])
            self.assertEqual(rows[1].raw_observation_json["evidence"]["spans"][0]["evidence_date"], "2020-10-05")
            self.assertEqual(rows[2].raw_observation_json["evidence"]["spans"][0]["quote"], "Invented evidence")
            self.assertEqual(rows[3].raw_observation_json["evidence"]["spans"], [])
            self.assertIsNone(await session.scalar(select(EvidenceSpanRow.evidence_date).where(
                EvidenceSpanRow.observation_id == rows[1].observation_id)))
        clean = await self.repository.get_clean_observations(self.run_id)
        self.assertEqual(len(clean), 2)
        self.assertIsNone(clean[1].evidence.spans[0].evidence_date)
        await self.repository.finalize_run(self.run_id)
        summary = await self.repository.get_run_summary(self.run_id)
        self.assertEqual((summary.observations_valid, summary.observations_repaired, summary.observations_rejected), (1, 1, 2))
        self.assertEqual(len(summary.warnings), 3)
        self.assertEqual(summary.warnings[0].action, "evidence_date_set_to_null")

    async def test_overlap_duplicates_retained_and_marking_can_be_rerun(self):
        chunks = chunk_pages(DOCUMENT, [PageText(page_number=1, text=QUOTE),
                                        PageText(page_number=2, text="More context")], max_chars=1)
        ids = await self.repository.create_chunks(self.run_id, chunks)
        for index, (chunk_id, chunk) in enumerate(zip(ids, chunks, strict=True)):
            items = [observation(observation="Paraphrase" if index else QUOTE), observation(observation_type="action")]
            result = await extractor(FakeAdapter(items=items)).extract(chunk)
            await self.repository.mark_chunk_running(chunk_id)
            await self.repository.persist_completed_chunk(chunk_id, chunk, result)
        for _ in range(2):
            self.assertEqual(await self.repository.mark_duplicates(self.run_id), 2)
        async with self.database.sessions() as session:
            rows = list(await session.scalars(select(Observation).order_by(Observation.observation_id)))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[2].duplicate_of_observation_id, rows[0].observation_id)
            self.assertEqual(rows[3].duplicate_of_observation_id, rows[1].observation_id)
            self.assertEqual(rows[0].provenance_id, rows[2].provenance_id)
            self.assertNotEqual(rows[0].observation_id, rows[2].observation_id)
        clean = await self.repository.get_clean_observations(self.run_id)
        self.assertEqual(len(clean), 2)
        self.assertEqual(clean[0].observation, QUOTE)
        await self.repository.finalize_run(self.run_id)
        summary = await self.repository.get_run_summary(self.run_id)
        self.assertEqual((summary.observations_returned, summary.duplicates_removed, summary.observations_persisted), (4, 2, 2))

    async def test_duplicate_chunk_completion_does_not_overwrite_paid_results(self):
        chunk_id, result = await self.store()
        with self.assertRaises(ValueError):
            await self.repository.persist_completed_chunk(chunk_id, make_chunk(), result)
        self.assertEqual(len(await self.repository.get_clean_observations(self.run_id)), 1)


class DurableOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "data/derived/stage1.sqlite"
        write_pdf(self.root, texts=(QUOTE, QUOTE, QUOTE))

    async def test_prior_chunks_are_committed_before_next_call_and_survive_later_failure(self):
        path = self.path
        snapshots = []

        class InspectingAdapter(FakeAdapter):
            async def generate_structured(self, **kwargs):
                with closing(sqlite3.connect(path)) as connection:
                    snapshots.append(connection.execute(
                        "SELECT chunk_index, status FROM extraction_chunks ORDER BY chunk_index").fetchall())
                    self_count = connection.execute("SELECT count(*) FROM implementation_observations").fetchone()[0]
                    if len(self.calls) == 2:
                        if self_count != 2:
                            raise AssertionError("Earlier paid observations were not committed")
                return await super().generate_structured(**kwargs)

        adapter = InspectingAdapter(responses=[
            [observation()], [observation(evidence={"spans": [span(page=2)]})], RuntimeError("provider unavailable"),
        ], usages=[TokenUsage(input_tokens=10, output_tokens=20), TokenUsage(input_tokens=30, output_tokens=40)])
        with redirect_stderr(StringIO()):
            self.assertEqual(await run_documents(self.root, [DOCUMENT], extractor=extractor(adapter),
                                                 max_chars=1, overlap_chars=0, export_files=False,
                                                 max_concurrency=1), 1)
        self.assertEqual(snapshots, [
            [(1, "running"), (2, "pending"), (3, "pending")],
            [(1, "completed"), (2, "running"), (3, "pending")],
            [(1, "completed"), (2, "completed"), (3, "running")],
        ])
        database = await initialize_database(self.path)
        try:
            repository = ExtractionRepository(database.sessions)
            async with database.sessions() as session:
                run = (await session.scalars(select(ExtractionRun))).one()
                self.assertEqual((run.status, run.input_tokens, run.output_tokens), ("partial", 40, 60))
                failed = (await session.scalars(select(ExtractionChunk).where(ExtractionChunk.status == "failed"))).one()
                self.assertEqual(failed.error, "provider unavailable")
                self.assertIsNotNone(failed.finished_at)
            self.assertEqual(len(await repository.get_clean_observations(run.run_id)), 2)
            summary = await repository.get_run_summary(run.run_id)
            self.assertEqual((summary.chunks_total, summary.chunks_succeeded, summary.chunks_failed), (3, 2, 1))
            self.assertEqual(summary.chunk_failures[0].error_type, "RuntimeError")
        finally:
            await database.close()

    async def test_process_interruption_keeps_completed_chunks_and_unfinished_state(self):
        class InterruptedAdapter(FakeAdapter):
            async def generate_structured(self, **kwargs):
                if self.calls:
                    raise asyncio.CancelledError()
                return await super().generate_structured(**kwargs)

        with self.assertRaises(asyncio.CancelledError):
            await run_documents(self.root, [DOCUMENT], extractor=extractor(InterruptedAdapter()),
                                max_chars=1, overlap_chars=0, export_files=False, max_concurrency=1)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT status, finished_at FROM extraction_runs").fetchone(), (None, None))
            self.assertEqual(connection.execute("SELECT status FROM extraction_chunks ORDER BY chunk_index").fetchall(),
                             [("completed",), ("running",), ("pending",)])
            self.assertEqual(connection.execute("SELECT count(*) FROM implementation_observations").fetchone()[0], 1)

    async def test_export_reservation_failure_does_not_prevent_extraction(self):
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()), patch(
            "digital_applications_prototype.extraction.cli.reserve_run", side_effect=OSError("export unavailable")
        ):
            await run_documents(self.root, [DOCUMENT], extractor=extractor())
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT status FROM extraction_runs").fetchone()[0], "completed")
            self.assertEqual(connection.execute("SELECT count(*) FROM implementation_observations").fetchone()[0], 1)

    async def test_no_export_keeps_successful_run_only_in_sqlite_and_prints_summary(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(await run_documents(self.root, [DOCUMENT], extractor=extractor(), export_files=False), 0)
        self.assertFalse((self.root / "data/derived/extracted").exists())
        with closing(sqlite3.connect(self.path)) as connection:
            run_id, status = connection.execute("SELECT run_id, status FROM extraction_runs").fetchone()
            self.assertEqual(status, "completed")
            self.assertEqual(connection.execute("SELECT count(*) FROM evidence_spans").fetchone()[0], 1)
        for expected in (run_id, "status: completed", "1/1 succeeded; 0 failed", "1 returned; 1 clean",
                         "valid: 1; repaired: 0; rejected: 0", "duplicates marked: 0", "tokens:",
                         "offline-model", "stage1-v3", "elapsed:"):
            self.assertIn(expected, output.getvalue())

    async def test_isolated_database_failure_preserves_previous_and_subsequent_chunks(self):
        adapter = FakeAdapter()
        original = ExtractionRepository.persist_completed_chunk
        calls = 0

        async def fail_second(repository, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("database unavailable")
            await original(repository, *args, **kwargs)

        with redirect_stderr(StringIO()) as errors, patch.object(ExtractionRepository, "persist_completed_chunk", fail_second):
            await run_documents(self.root, [DOCUMENT], extractor=extractor(adapter),
                                max_chars=1, export_files=False, max_concurrency=1)
        self.assertEqual(len(adapter.calls), 3)
        self.assertIn("result persistence failed: OSError: database unavailable", errors.getvalue())
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT status FROM extraction_runs").fetchone()[0], "partial")
            self.assertEqual(connection.execute("SELECT status FROM extraction_chunks ORDER BY chunk_index").fetchall(),
                             [("completed",), ("failed",), ("completed",)])
            self.assertEqual(connection.execute("SELECT count(*) FROM implementation_observations").fetchone()[0], 2)


if __name__ == "__main__":
    unittest.main()
