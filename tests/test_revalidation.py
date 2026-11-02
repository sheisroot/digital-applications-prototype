import asyncio
import hashlib
import json
import sqlite3
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sqlalchemy import select, text
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

from digital_applications_prototype.extraction.cli import main
from digital_applications_prototype.extraction.models import (
    DocumentChunk,
    ModelObservation,
    PageText,
    RunReport,
)
from digital_applications_prototype.extraction.revalidation import (
    backup_database,
    revalidate,
)
from digital_applications_prototype.extraction.validation import (
    normalize_quote_text,
    normalize_whitespace,
    quote_matches,
    validate_observation,
)
from digital_applications_prototype.llm.adapters.base import TokenUsage
from digital_applications_prototype.persistence.database import initialize_database
from digital_applications_prototype.persistence.models import (
    ExtractionChunk,
    Observation,
    ValidationRevision,
)
from digital_applications_prototype.persistence.repository import ExtractionRepository


def old_match(quote, source):
    return bool(normalize_whitespace(quote)) and normalize_whitespace(
        quote
    ) in normalize_whitespace(source)


def records(path):
    with closing(sqlite3.connect(path)) as connection:
        names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        return {
            name: connection.execute(f'SELECT * FROM "{name}" ORDER BY 1').fetchall()
            for name in names
        }


class QuoteMatchingTests(unittest.TestCase):
    def test_saved_run_source_page_fixtures(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures/km_pad_2020_quotes.json").read_text()
        )
        chunk = DocumentChunk(**fixture["document"], pages=fixture["pages"])
        recovered = []
        for case in fixture["cases"]:
            with self.subTest(observation_id=case["observation_id"]):
                item = ModelObservation.model_validate(case["raw_observation"])
                result = validate_observation(item, chunk)
                self.assertEqual(result.status, case["expected_status"])
                if case["previous_status"] == "rejected":
                    self.assertFalse(
                        all(
                            any(
                                old_match(s.quote, p.text)
                                for p in chunk.pages
                                if p.page_number == s.page
                            )
                            for s in item.evidence.spans
                        )
                    )
                    if result.status == "valid":
                        recovered.append(case["observation_id"])
                        assert result.observation is not None
                        self.assertEqual(result.observation.evidence, item.evidence)
        self.assertEqual(
            sorted(recovered), [6, 8, 10, 24, 45, 46, 60, 82, 132, 142, 185]
        )

    def test_conservative_symmetric_spacing_equivalences(self):
        for left, right in [
            (
                "limited on -site inspections conducted.",
                "limited on-site inspections conducted.",
            ),
            ("home-\n grown tools", "home-grown tools"),
            ("e - money issuers", "e-money issuers"),
            ("ATMs , POS", "ATMs, POS"),
            ("paid .", "paid."),
            ("ordinary\n whitespace", "ordinary whitespace"),
        ]:
            self.assertTrue(quote_matches(left, right))
            self.assertTrue(quote_matches(right, left))
            self.assertEqual(normalize_quote_text(left), normalize_quote_text(right))
        self.assertFalse(quote_matches(" \n ", "source"))

    def test_word_boundaries_negation_numbers_and_punctuation_remain_significant(self):
        for fabricated, source in [
            ("Financial data were reliable.", "Financial data were not reliable."),
            ("The rate was 20%.", "The rate was 10%."),
            ("The rate was 10%.", "The rate was 10.5%."),
            ("The amount was 10.", "The amount was -10."),
            ("The value was 1.5.", "The value was 1 . 5."),
            ("2²-loss", "2² - loss"),
            ("2².", "2² ."),
            ("notable", "not able"),
            ("nowhere", "now here"),
            (
                "reported by financial institutions",
                "repo rted by financial institutions",
            ),
            ("power is not available.", "power is not availab le."),
            ("homegrown tools", "home-grown tools"),
            ("AB", "A/B"),
            ("fabricated support", "actual source"),
        ]:
            with self.subTest(quote=fabricated):
                self.assertFalse(quote_matches(fabricated, source))

    def test_page_overlap_and_all_spans_are_required(self):
        chunk = make_chunk("limited on -site inspections conducted.\n" + QUOTE)
        valid = observation(
            evidence={"spans": [span("limited on-site inspections conducted."), span()]}
        )
        self.assertEqual(validate_observation(valid, chunk).status, "valid")
        for spans in [
            [span(), span("invented")],
            [span(QUOTE, page=2)],
            [],
        ]:
            self.assertEqual(
                validate_observation(
                    observation(evidence={"spans": spans}), chunk
                ).status,
                "rejected",
            )
        pages = [
            PageText(page_number=1, text=QUOTE),
            PageText(page_number=2, text="Unrelated page."),
        ]
        chunk = chunk.model_copy(update={"pages": pages})
        self.assertEqual(
            validate_observation(
                observation(evidence={"spans": [span(QUOTE, page=2)]}), chunk
            ).status,
            "rejected",
        )
        # A page number in the overlap grants access only to the actual suffix.
        chunk = chunk.model_copy(
            update={
                "pages": [pages[1]],
                "overlap": PageText(page_number=1, text="delivery."),
            }
        )
        self.assertEqual(validate_observation(observation(), chunk).status, "rejected")


class RevalidationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "stage1.sqlite"
        self.database = await initialize_database(self.path)
        self.addAsyncCleanup(self.database.close)
        self.repository = ExtractionRepository(self.database.sessions)
        self.run_id = await self.repository.create_run(
            DOCUMENT.document_id, model="saved-model"
        )
        quote = "limited on-site inspections conducted."
        chunk = make_chunk("limited on -site inspections conducted.\n" + QUOTE)
        items = [
            observation(evidence={"spans": [span(quote)]}),
            observation(),
            observation(evidence={"spans": [span("Fabricated claim.")]}),
            observation(evidence={"spans": [span(QUOTE, page=2)]}),
            observation(factor_description=" "),
            observation(evidence={"spans": [span(quote), span()]}),
            observation(evidence={"spans": [span(evidence_date="2025-01-01")]}),
            observation(evidence={"spans": []}),
            observation(observation_type="action"),
        ]
        await self.store(
            self.run_id, [chunk, make_chunk(quote)], [items, [items[0]]], old=True
        )
        async with self.database.sessions.begin() as session:
            row = await session.scalar(
                select(Observation).where(
                    Observation.run_id == self.run_id,
                    Observation.observation_index == 9,
                )
            )
            assert row is not None
            row.validation_status = "rejected"
            row.provenance_id = None
            source_chunk = await session.get(ExtractionChunk, row.chunk_id)
            assert source_chunk is not None
            source_chunk.warnings_json = [
                *source_chunk.warnings_json,
                {
                    "code": "manual_rule",
                    "message": "Preserve unrelated validation failure",
                    "action": "observation_rejected",
                    "observation_index": 9,
                    "chunk_start_page": 1,
                    "chunk_end_page": 1,
                    "evidence_span_index": None,
                },
            ]
        self.other_run = await self.repository.create_run(DOCUMENT.document_id)
        await self.store(self.other_run, [make_chunk()], [[observation()]])
        other = make_chunk().model_copy(update={"document_id": "KM_PAD_2020"})
        self.other_document_run = await self.repository.create_run(other.document_id)
        await self.store(self.other_document_run, [other], [[observation()]])

    async def store(self, run, chunks, items, old=False):
        ids = await self.repository.create_chunks(run, chunks)
        for chunk_id, chunk, returned in zip(ids, chunks, items, strict=True):
            with patch(
                "digital_applications_prototype.extraction.validation.quote_matches",
                old_match if old else quote_matches,
            ):
                result = await extractor(
                    FakeAdapter(
                        items=returned,
                        usages=[TokenUsage(input_tokens=123, output_tokens=45)],
                    )
                ).extract(chunk)
            await self.repository.mark_chunk_running(chunk_id)
            await self.repository.persist_completed_chunk(chunk_id, chunk, result)
        await self.repository.mark_duplicates(run)
        await self.repository.finalize_run(run)

    async def invoke(self, **options):
        return await revalidate(
            self.path, DOCUMENT.document_id, self.run_id, root=self.root, **options
        )

    async def test_dry_run_is_read_only_and_shows_proposed_changes(self):
        before = records(self.path)
        files_before = set(self.root.rglob("*"))
        database_bytes = self.path.read_bytes()
        result = await self.invoke()
        self.assertEqual(result["mode"], "dry-run")
        self.assertEqual(result["before"]["rejected"], 7)
        self.assertEqual(result["after"]["rejected"], 5)
        self.assertEqual(len(result["recovered_observation_ids"]), 2)
        self.assertEqual(result["newly_rejected_observation_ids"], [])
        self.assertEqual(
            result["after"]["duplicates"], 2
        )  # Both signatures choose their first occurrence.
        self.assertEqual(result["after"]["rejection_reasons"]["manual_rule"], 1)
        self.assertEqual(before, records(self.path))
        self.assertEqual(database_bytes, self.path.read_bytes())
        self.assertEqual(files_before, set(self.root.rglob("*")))

    async def test_cli_dispatch_defaults_to_dry_run_and_requires_explicit_selection(
        self,
    ):
        output = StringIO()
        with (
            redirect_stdout(output),
            patch(
                "digital_applications_prototype.llm.service.LLMService.generate_structured",
                side_effect=AssertionError("No API calls"),
            ),
        ):
            status = await asyncio.to_thread(
                main,
                [
                    "revalidate",
                    "--database",
                    str(self.path),
                    "--document",
                    DOCUMENT.document_id,
                    "--run",
                    self.run_id,
                    "--root",
                    str(self.root),
                ],
            )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["mode"], "dry-run")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(
                ["revalidate", "--document", DOCUMENT.document_id, "--run", self.run_id]
            )

    async def test_apply_preserves_history_other_runs_and_is_stable_with_calls_blocked(
        self,
    ):
        before = records(self.path)
        backup = self.root / "before.sqlite"
        with (
            patch(
                "digital_applications_prototype.llm.service.LLMService.generate_structured",
                side_effect=AssertionError("No API calls"),
            ),
            patch(
                "digital_applications_prototype.extraction.service.ImplementationObservationExtractor.extract",
                side_effect=AssertionError("No extraction"),
            ),
            patch(
                "digital_applications_prototype.extraction.revalidation.load_pdf",
                side_effect=AssertionError("Retained sources require no PDF"),
            ),
            patch("socket.socket.connect", side_effect=AssertionError("No network")),
        ):
            result = await self.invoke(apply=True, backup_path=backup)
            after = records(self.path)
            export_contents = {
                key: Path(path).read_bytes() for key, path in result["exports"].items()
            }
            repeated = await self.invoke(apply=True)
        self.assertEqual(records(backup), before)
        self.assertEqual(after["extraction_runs"], before["extraction_runs"])
        self.assertEqual(
            after["extraction_chunk_sources"], before["extraction_chunk_sources"]
        )
        for old, new in zip(
            before["extraction_chunks"], after["extraction_chunks"], strict=True
        ):
            self.assertEqual(old[:-1], new[:-1])  # Only derived warnings may change.
        with closing(sqlite3.connect(self.path)) as connection:
            observation_columns = [
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(implementation_observations)"
                )
            ]
            span_columns = [
                row[1]
                for row in connection.execute("PRAGMA table_info(evidence_spans)")
            ]
        derived = {"validation_status", "provenance_id", "duplicate_of_observation_id"}
        for old, new in zip(
            before["implementation_observations"],
            after["implementation_observations"],
            strict=True,
        ):
            self.assertEqual(
                [
                    value
                    for key, value in zip(observation_columns, old)
                    if key not in derived
                ],
                [
                    value
                    for key, value in zip(observation_columns, new)
                    if key not in derived
                ],
            )
            if old[observation_columns.index("run_id")] != self.run_id:
                self.assertEqual(old, new)
        for old, new in zip(
            before["evidence_spans"], after["evidence_spans"], strict=True
        ):
            self.assertEqual(
                [
                    value
                    for key, value in zip(span_columns, old)
                    if key != "evidence_date"
                ],
                [
                    value
                    for key, value in zip(span_columns, new)
                    if key != "evidence_date"
                ],
            )
        self.assertEqual(result["after"], repeated["after"])
        self.assertEqual(result["exports"], repeated["exports"])
        self.assertEqual(
            export_contents,
            {key: Path(path).read_bytes() for key, path in repeated["exports"].items()},
        )
        final = records(self.path)
        for table in after:
            if table != "validation_revisions":
                self.assertEqual(after[table], final[table])
        self.assertEqual(len(final["validation_revisions"]), 2)
        async with self.database.sessions() as session:
            revisions = list(
                await session.scalars(
                    select(ValidationRevision).order_by(
                        ValidationRevision.revalidation_id
                    )
                )
            )
            self.assertTrue(revisions[0].changes_json)
            self.assertEqual(revisions[1].changes_json, [])
            self.assertTrue(revisions[0].revalidated_at)
        report = RunReport.model_validate_json(
            Path(result["exports"]["report"]).read_text()
        )
        summary = await self.repository.get_run_summary(self.run_id)
        clean = await self.repository.get_clean_observations(self.run_id)
        self.assertEqual(report.model_copy(update={"jsonl_file": None}), summary)
        self.assertEqual(
            export_contents["jsonl"].decode(),
            "".join(item.model_dump_json() + "\n" for item in clean),
        )
        self.assertEqual(report.observations_persisted, result["after"]["usable"])
        self.assertEqual((report.input_tokens, report.output_tokens), (246, 90))

    async def test_failed_audit_rolls_back_validation_and_duplicate_changes(self):
        before = records(self.path)
        with (
            patch(
                "digital_applications_prototype.extraction.revalidation.ValidationRevision",
                side_effect=RuntimeError("audit unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "audit unavailable"),
        ):
            await self.invoke(apply=True)
        self.assertEqual(before, records(self.path))

    async def test_export_failure_is_recoverable_without_reextracting(self):
        with (
            patch(
                "digital_applications_prototype.extraction.revalidation.write_report",
                side_effect=OSError("export unavailable"),
            ),
            self.assertRaisesRegex(
                RuntimeError, "Revalidation committed; export failed"
            ),
        ):
            await self.invoke(apply=True)
        result = await self.invoke(apply=True)
        self.assertTrue(Path(result["exports"]["report"]).exists())
        self.assertEqual(result["recovered_observation_ids"], [])

    async def test_wrong_selection_and_unfinished_run_fail_without_changes(self):
        before = records(self.path)
        with self.assertRaisesRegex(ValueError, "does not identify"):
            await revalidate(
                self.path, "KM_PAD_2020", self.run_id, root=self.root, apply=True
            )
        self.assertEqual(before, records(self.path))
        run = await self.repository.create_run(DOCUMENT.document_id)
        with self.assertRaisesRegex(ValueError, "finalized"):
            await revalidate(self.path, DOCUMENT.document_id, run, root=self.root)

    async def test_newly_rejected_observation_is_counted_and_not_exported(self):
        run = await self.repository.create_run(DOCUMENT.document_id)
        chunk = make_chunk()
        item = observation(evidence={"spans": [span("Invented quote.")]})
        ids = await self.repository.create_chunks(run, [chunk])
        with patch(
            "digital_applications_prototype.extraction.validation.quote_matches",
            return_value=True,
        ):
            result = await extractor(FakeAdapter([item])).extract(chunk)
        await self.repository.mark_chunk_running(ids[0])
        await self.repository.persist_completed_chunk(ids[0], chunk, result)
        await self.repository.finalize_run(run)
        result = await revalidate(
            self.path, DOCUMENT.document_id, run, root=self.root, apply=True
        )
        self.assertEqual(len(result["newly_rejected_observation_ids"]), 1)
        self.assertEqual(result["after"]["usable"], 0)
        self.assertEqual(Path(result["exports"]["jsonl"]).read_bytes(), b"")

    async def test_recovered_quote_still_undergoes_evidence_date_repair(self):
        run = await self.repository.create_run(DOCUMENT.document_id)
        quote = "limited on-site inspections conducted."
        chunk = make_chunk("limited on -site inspections conducted.")
        item = observation(
            evidence={"spans": [span(quote, evidence_date="2025-01-01")]}
        )
        await self.store(run, [chunk], [[item]], old=True)
        result = await revalidate(
            self.path, DOCUMENT.document_id, run, root=self.root, apply=True
        )
        self.assertEqual(result["after"]["repaired"], 1)
        self.assertEqual(len(result["evidence_date_changes"]), 1)
        clean = await self.repository.get_clean_observations(run)
        self.assertEqual(clean[0].evidence.spans[0].quote, quote)
        self.assertIsNone(clean[0].evidence.spans[0].evidence_date)

    async def test_legacy_identity_hash_and_chunk_reconstruction_are_required(self):
        path = write_pdf(self.root)
        run = await self.repository.create_run(DOCUMENT.document_id)
        await self.store(run, [make_chunk()], [[observation()]])
        async with self.database.sessions.begin() as session:
            await session.execute(text("DROP TABLE extraction_chunk_sources"))
            await session.execute(text("DROP TABLE validation_revisions"))
        before = records(self.path)

        async def invoke(**options):
            return await revalidate(
                self.path, DOCUMENT.document_id, run, root=self.root, **options
            )

        with self.assertRaisesRegex(
            ValueError, "Source identity cannot be established"
        ):
            await invoke()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            await invoke(
                source_pdf=path,
                source_sha256="0" * 64,
                max_chars=12000,
                overlap_chars=600,
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "explicit --max-chars"):
            await invoke(source_pdf=path, source_sha256=digest)
        self.assertEqual(before, records(self.path))
        result = await invoke(
            source_pdf=path, source_sha256=digest, max_chars=12000, overlap_chars=600
        )
        self.assertEqual(result["after"]["usable"], 1)
        self.assertNotIn("extraction_chunk_sources", records(self.path))
        applied = await invoke(
            apply=True,
            source_pdf=path,
            source_sha256=digest,
            max_chars=12000,
            overlap_chars=600,
        )
        self.assertEqual(records(Path(applied["backup"])), before)
        self.assertIn("validation_revisions", records(self.path))
        # Rebuilding a different number of chunks must fail, even with a matching hash.
        with self.assertRaisesRegex(ValueError, "mapping differs"):
            await revalidate(
                self.path,
                DOCUMENT.document_id,
                self.run_id,
                root=self.root,
                source_pdf=path,
                source_sha256=digest,
                max_chars=12000,
                overlap_chars=600,
            )

    async def test_backup_never_overwrites_and_contains_uncheckpointed_wal(self):
        destination = self.root / "backup.sqlite"
        backup_database(self.path, destination)
        self.assertEqual(records(self.path), records(destination))
        with self.assertRaises(FileExistsError):
            backup_database(self.path, destination)
        with self.assertRaises(ValueError):
            backup_database(self.path, self.path)
