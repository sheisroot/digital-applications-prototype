"""Event-controlled remote fakes against the real temporary SQLite repository."""

import asyncio
import json
import re
import sqlite3
import unittest
from contextlib import asynccontextmanager, closing, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from test_extraction import (
    DOCUMENT,
    QUOTE,
    FakeAdapter,
    extractor,
    observation,
    read_reports,
    run_jsonls,
    span,
    write_pdf,
)

from digital_applications_prototype.extraction.cli import main, run_documents
from digital_applications_prototype.llm.adapters.base import TokenUsage
from digital_applications_prototype.llm.service import LLMService
from digital_applications_prototype.persistence.database import initialize_database
from digital_applications_prototype.persistence.repository import ExtractionRepository


class ControlledAdapter:
    model = "controlled-offline-model"

    def __init__(self, count, *, errors=None, items=None):
        self.started = [asyncio.Event() for _ in range(count)]
        self.release = [asyncio.Event() for _ in range(count)]
        self.errors = errors or {}
        self.items = items or {}
        self.calls = []
        self.finished = []
        self.cancelled = []
        self.active = 0
        self.peak = 0
        self._usage: ContextVar[TokenUsage | None] = ContextVar(
            "test_usage", default=None
        )

    @property
    def last_usage(self):
        return self._usage.get()

    async def generate_structured(self, *, prompt, response_model, model=None):
        index = int(re.findall(r"\[PDF page (\d+)\]", prompt)[-1]) - 1
        self.calls.append(index)
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started[index].set()
        try:
            await self.release[index].wait()
            if index in self.errors:
                raise self.errors[index]
            items = self.items.get(
                index,
                [
                    observation(
                        observation=f"Claim from page {index + 1}",
                        evidence={"spans": [span(page=index + 1)]},
                    )
                ],
            )
            self._usage.set(
                TokenUsage(
                    input_tokens=(index + 1) * 10, output_tokens=(index + 1) * 20
                )
            )
            self.finished.append(index)
            return response_model(observations=items)
        except asyncio.CancelledError:
            self.cancelled.append(index)
            raise
        finally:
            self.active -= 1


class ChunkConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.root = Path(directory)
        self.path = self.root / "data/derived/stage1.sqlite"
        self.output = self.enterContext(redirect_stdout(StringIO()))
        self.errors = self.enterContext(redirect_stderr(StringIO()))
        self.committed = [asyncio.Event() for _ in range(10)]
        self.failed = [asyncio.Event() for _ in range(10)]
        self.commit_order = []
        self.write_owners = []
        original_complete = ExtractionRepository.persist_completed_chunk
        original_failed = ExtractionRepository.mark_chunk_failed

        async def complete(repository, chunk_id, chunk, result):
            self.write_owners.append(asyncio.current_task())
            await original_complete(repository, chunk_id, chunk, result)
            index = chunk.pages[-1].page_number - 1
            self.commit_order.append(index)
            self.committed[index].set()

        async def fail(repository, chunk_id, error):
            self.write_owners.append(asyncio.current_task())
            await original_failed(repository, chunk_id, error)
            with closing(sqlite3.connect(self.path)) as connection:
                index = (
                    connection.execute(
                        "SELECT chunk_index FROM extraction_chunks WHERE chunk_id = ?",
                        (chunk_id,),
                    ).fetchone()[0]
                    - 1
                )
            self.failed[index].set()

        self.enterContext(
            patch.object(ExtractionRepository, "persist_completed_chunk", complete)
        )
        self.enterContext(patch.object(ExtractionRepository, "mark_chunk_failed", fail))

    def start(self, adapter, **kwargs):
        write_pdf(self.root, texts=(QUOTE,) * len(adapter.started))
        options = {"max_chars": 1, "overlap_chars": 0, "export_files": False, **kwargs}
        task = asyncio.create_task(
            run_documents(
                self.root, [DOCUMENT], extractor=extractor(adapter), **options
            )
        )

        async def cleanup():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        self.addAsyncCleanup(cleanup)
        return task

    async def wait(self, *events):
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in events)), 5)

    def snapshot(self):
        # A separate connection proves visibility of committed state while siblings wait.
        with closing(sqlite3.connect(self.path)) as connection:
            return connection.execute(
                "SELECT status, input_tokens, output_tokens FROM extraction_chunks ORDER BY chunk_index"
            ).fetchall()

    async def clean(self):
        database = await initialize_database(self.path)
        try:
            with closing(sqlite3.connect(self.path)) as connection:
                run_id = connection.execute(
                    "SELECT run_id FROM extraction_runs"
                ).fetchone()[0]
            return await ExtractionRepository(database.sessions).get_clean_observations(
                run_id
            )
        finally:
            await database.close()

    async def test_default_bound_three_for_ten_chunks_and_serialized_writes(self):
        adapter = ControlledAdapter(10)
        task = self.start(adapter)
        await self.wait(*adapter.started[:3])
        self.assertEqual(adapter.active, 3)
        self.assertFalse(any(event.is_set() for event in adapter.started[3:]))
        self.assertEqual(
            [row[0] for row in self.snapshot()], ["running"] * 3 + ["pending"] * 7
        )
        # Even while remote calls are blocked, another connection can obtain a write lock.
        with closing(sqlite3.connect(self.path, timeout=0)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.rollback()
        for event in adapter.release:
            event.set()
        self.assertEqual(await asyncio.wait_for(task, 5), 0)
        self.assertEqual(adapter.peak, 3)
        self.assertEqual(adapter.calls, list(range(10)))
        self.assertEqual([row[0] for row in self.snapshot()], ["completed"] * 10)
        self.assertTrue(all(owner is task for owner in self.write_owners))
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT max_concurrency FROM extraction_runs"
                ).fetchone()[0],
                3,
            )

    async def test_override_four_bounds_active_calls(self):
        adapter = ControlledAdapter(6)
        task = self.start(adapter, max_concurrency=4)
        await self.wait(*adapter.started[:4])
        self.assertEqual(adapter.active, 4)
        self.assertFalse(adapter.started[4].is_set())
        for event in adapter.release:
            event.set()
        self.assertEqual(await asyncio.wait_for(task, 5), 0)
        self.assertEqual(adapter.peak, 4)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT max_concurrency FROM extraction_runs"
                ).fetchone()[0],
                4,
            )

    async def test_out_of_order_results_commit_promptly_but_outputs_and_tokens_follow_source(
        self,
    ):
        adapter = ControlledAdapter(3)
        task = self.start(adapter, export_files=True)
        await self.wait(*adapter.started)
        for index in (2, 1, 0):
            adapter.release[index].set()
            await self.wait(self.committed[index])
            self.assertEqual(
                self.snapshot()[index],
                ("completed", (index + 1) * 10, (index + 1) * 20),
            )
        self.assertEqual(await asyncio.wait_for(task, 5), 0)
        self.assertEqual(adapter.finished, [2, 1, 0])
        self.assertEqual(self.commit_order, [2, 1, 0])
        self.assertEqual(
            [item.chunk_end_page for item in await self.clean()], [1, 2, 3]
        )
        lines = run_jsonls(self.root)[0].read_text().splitlines()
        self.assertEqual(
            [json.loads(line)["chunk_end_page"] for line in lines], [1, 2, 3]
        )
        report = read_reports(self.root)[0]
        self.assertEqual((report.input_tokens, report.output_tokens), (60, 120))
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT input_tokens, output_tokens FROM extraction_runs"
                ).fetchone(),
                (60, 120),
            )

    async def test_chunk_failure_does_not_cancel_waiting_siblings_or_lose_durable_results(
        self,
    ):
        adapter = ControlledAdapter(3, errors={1: RuntimeError("provider unavailable")})
        task = self.start(adapter)
        await self.wait(*adapter.started)
        adapter.release[1].set()
        await self.wait(self.failed[1])
        self.assertEqual(adapter.active, 2)
        self.assertEqual(adapter.cancelled, [])
        self.assertFalse(task.done())
        adapter.release[2].set()
        await self.wait(self.committed[2])
        self.assertEqual(self.snapshot()[2][0], "completed")
        adapter.release[0].set()
        self.assertEqual(await asyncio.wait_for(task, 5), 1)
        self.assertEqual(
            [row[0] for row in self.snapshot()], ["completed", "failed", "completed"]
        )
        self.assertEqual([item.chunk_end_page for item in await self.clean()], [1, 3])
        self.assertEqual(adapter.cancelled, [])
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT status, input_tokens, output_tokens FROM extraction_runs"
                ).fetchone(),
                ("partial", 40, 80),
            )
            self.assertEqual(
                connection.execute(
                    "SELECT error_type, error FROM extraction_chunks WHERE chunk_index = 2"
                ).fetchone(),
                ("RuntimeError", "provider unavailable"),
            )

    async def test_persistence_failure_is_failed_and_pending_siblings_still_run(self):
        original = ExtractionRepository.persist_completed_chunk

        async def fail_one(repository, chunk_id, chunk, result):
            if chunk.end_page == 2:
                raise OSError("chunk commit unavailable")
            await original(repository, chunk_id, chunk, result)

        self.enterContext(
            patch.object(ExtractionRepository, "persist_completed_chunk", fail_one)
        )
        adapter = ControlledAdapter(5)
        task = self.start(adapter)
        await self.wait(*adapter.started[:3])
        adapter.release[1].set()
        await self.wait(self.failed[1], adapter.started[3])
        self.assertEqual(self.snapshot()[1], ("failed", None, None))
        for event in adapter.release:
            event.set()
        self.assertEqual(await asyncio.wait_for(task, 5), 1)
        self.assertEqual(adapter.calls, list(range(5)))
        self.assertEqual(
            [item.chunk_end_page for item in await self.clean()], [1, 3, 4, 5]
        )
        with closing(sqlite3.connect(self.path)) as connection:
            status, error, input_tokens = connection.execute(
                "SELECT status, error, input_tokens FROM extraction_runs"
            ).fetchone()
            self.assertEqual((status, input_tokens), ("partial", 130))
            self.assertIn(
                "result persistence failed: OSError: chunk commit unavailable", error
            )
        self.assertIn("result persistence failed", self.errors.getvalue())
        self.assertEqual(adapter.cancelled, [])

    async def test_storage_unavailable_stops_admission_but_drains_active_siblings(self):
        original_complete = ExtractionRepository.persist_completed_chunk
        original_failed = ExtractionRepository.mark_chunk_failed
        failure_attempted = asyncio.Event()

        async def fail_complete(repository, chunk_id, chunk, result):
            if chunk.end_page == 2:
                raise OSError("write unavailable")
            await original_complete(repository, chunk_id, chunk, result)

        async def fail_record(repository, chunk_id, error):
            if chunk_id == 2:
                failure_attempted.set()
                raise OSError("failure recording unavailable")
            await original_failed(repository, chunk_id, error)

        self.enterContext(
            patch.object(ExtractionRepository, "persist_completed_chunk", fail_complete)
        )
        self.enterContext(
            patch.object(ExtractionRepository, "mark_chunk_failed", fail_record)
        )
        adapter = ControlledAdapter(5)
        task = self.start(adapter)
        await self.wait(*adapter.started[:3])
        adapter.release[1].set()
        await self.wait(failure_attempted)
        adapter.release[0].set()
        adapter.release[2].set()
        self.assertEqual(await asyncio.wait_for(task, 5), 1)
        self.assertEqual(adapter.calls, [0, 1, 2])
        self.assertEqual(adapter.cancelled, [])
        self.assertEqual(
            [row[0] for row in self.snapshot()],
            ["completed", "running", "completed", "pending", "pending"],
        )
        self.assertEqual([item.chunk_end_page for item in await self.clean()], [1, 3])
        self.assertIn("failure-state persistence failed", self.errors.getvalue())

    async def test_cancellation_drains_tasks_and_leaves_truthful_unfinished_state(self):
        adapter = ControlledAdapter(6)
        task = self.start(adapter, export_files=True)
        await self.wait(*adapter.started[:3])
        adapter.release[0].set()
        await self.wait(self.committed[0], adapter.started[3])
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(adapter.active, 0)
        self.assertEqual(set(adapter.cancelled), {1, 2, 3})
        self.assertEqual(
            [row[0] for row in self.snapshot()],
            ["completed", "running", "running", "running", "pending", "pending"],
        )
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT status, finished_at FROM extraction_runs"
                ).fetchone(),
                (None, None),
            )
            self.assertEqual(
                connection.execute(
                    "SELECT count(*) FROM implementation_observations"
                ).fetchone()[0],
                1,
            )
        self.assertFalse(
            list((self.root / "data/derived/extracted").rglob("*.reserve"))
        )

    async def test_cancellation_during_persistence_never_claims_completion(self):
        entering_commit = asyncio.Event()

        async def blocked_commit(repository, chunk_id, chunk, result):
            entering_commit.set()
            await asyncio.Event().wait()

        self.enterContext(
            patch.object(
                ExtractionRepository, "persist_completed_chunk", blocked_commit
            )
        )
        adapter = ControlledAdapter(4)
        task = self.start(adapter)
        await self.wait(*adapter.started[:3])
        adapter.release[0].set()
        await self.wait(entering_commit)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(
            [row[0] for row in self.snapshot()], ["running"] * 3 + ["pending"]
        )
        self.assertEqual(await self.clean(), [])
        self.assertEqual(adapter.active, 0)

    async def test_concurrency_one_commits_before_starting_next_call(self):
        adapter = ControlledAdapter(3)
        task = self.start(adapter, max_concurrency=1)
        for index in range(3):
            await self.wait(adapter.started[index])
            self.assertEqual(
                [row[0] for row in self.snapshot()],
                ["completed"] * index + ["running"] + ["pending"] * (2 - index),
            )
            adapter.release[index].set()
        self.assertEqual(await asyncio.wait_for(task, 5), 0)
        self.assertEqual(adapter.peak, 1)
        self.assertEqual(self.commit_order, [0, 1, 2])

    async def test_validation_repairs_and_rejections_remain_observation_level(self):
        items = [
            observation(evidence={"spans": [span(evidence_date="2020-10-05")]}),
            observation(evidence={"spans": [span("invented")]}),
            observation(observation_type="action"),
        ]
        adapter = ControlledAdapter(3, items={0: items})
        task = self.start(adapter, export_files=True)
        await self.wait(*adapter.started)
        for event in reversed(adapter.release):
            event.set()
        self.assertEqual(await asyncio.wait_for(task, 5), 0)
        report = read_reports(self.root)[0]
        self.assertEqual((report.chunks_succeeded, report.chunks_failed), (3, 0))
        self.assertEqual(
            (
                report.observations_valid,
                report.observations_repaired,
                report.observations_rejected,
            ),
            (3, 1, 1),
        )
        self.assertEqual(
            [warning.code for warning in report.warnings],
            ["unsupported_evidence_date", "ungrounded_quote"],
        )
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT validation_status FROM implementation_observations WHERE chunk_id = 1 ORDER BY observation_index"
                ).fetchall(),
                [("repaired",), ("rejected",), ("valid",)],
            )

    async def test_partial_overlap_dedupe_selects_first_source_occurrence_despite_commit_order(
        self,
    ):
        adapter = ControlledAdapter(
            3,
            errors={2: RuntimeError("provider unavailable")},
            items={
                0: [observation(observation="First in source")],
                1: [observation(observation="Later paraphrase")],
            },
        )
        task = self.start(adapter, overlap_chars=600)
        await self.wait(*adapter.started)
        adapter.release[1].set()
        await self.wait(self.committed[1])
        adapter.release[2].set()
        await self.wait(self.failed[2])
        adapter.release[0].set()
        self.assertEqual(await asyncio.wait_for(task, 5), 1)
        self.assertEqual(
            [item.observation for item in await self.clean()], ["First in source"]
        )
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                "SELECT o.observation_id, o.duplicate_of_observation_id FROM implementation_observations o "
                "JOIN extraction_chunks c ON o.chunk_id = c.chunk_id ORDER BY c.chunk_index"
            ).fetchall()
            self.assertEqual(rows[1][1], rows[0][0])
            self.assertIsNone(rows[0][1])
            self.assertGreater(rows[0][0], rows[1][0])

    async def test_invalid_programmatic_concurrency_creates_no_database(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            await run_documents(self.root, [DOCUMENT], max_concurrency=0)
        self.assertFalse(self.path.exists())


class ConcurrencyCLITests(unittest.TestCase):
    def test_cli_default_and_overrides_are_persisted(self):
        for concurrency in (None, 1, 3, 4):
            with (
                self.subTest(concurrency=concurrency),
                TemporaryDirectory() as directory,
                redirect_stdout(StringIO()),
            ):
                adapter = FakeAdapter()

                @asynccontextmanager
                async def runtime(adapter=adapter):
                    yield LLMService(adapter)

                root = Path(directory)
                write_pdf(root)
                args = [
                    "--document",
                    DOCUMENT.document_id,
                    "--root",
                    str(root),
                    "--no-export",
                ]
                if concurrency is not None:
                    args.extend(["--concurrency", str(concurrency)])
                with patch(
                    "digital_applications_prototype.core.llm_runtime.configured_llm",
                    runtime,
                ):
                    self.assertEqual(main(args), 0)
                with closing(
                    sqlite3.connect(root / "data/derived/stage1.sqlite")
                ) as connection:
                    self.assertEqual(
                        connection.execute(
                            "SELECT max_concurrency FROM extraction_runs"
                        ).fetchone()[0],
                        8 if concurrency is None else concurrency,
                    )

    def test_cli_rejects_nonpositive_concurrency(self):
        for concurrency in ("0", "-1"):
            with (
                self.subTest(concurrency=concurrency),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(["--document", DOCUMENT.document_id, "--concurrency", concurrency])
