import hashlib
import json
import unittest
from contextlib import redirect_stdout
from importlib.metadata import version
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
from sqlalchemy import delete, select
from test_extraction import (
    DOCUMENT,
    QUOTE,
    FakeAdapter,
    extractor,
    observation,
    span,
    write_pdf,
)
from test_revalidation import records

from digital_applications_prototype.extraction import cli
from digital_applications_prototype.extraction.chunking import chunk_pages
from digital_applications_prototype.extraction.models import (
    DocumentChunk,
    ModelObservation,
    PageText,
)
from digital_applications_prototype.extraction.pdf import PDFText, load_pdf
from digital_applications_prototype.extraction.revalidation import (
    _project_sources,
    revalidate,
)
from digital_applications_prototype.extraction.validation import (
    quote_matches,
    validate_observation,
)
from digital_applications_prototype.persistence.database import initialize_database
from digital_applications_prototype.persistence.models import (
    ChunkSource,
    ValidationRevision,
)
from digital_applications_prototype.persistence.repository import ExtractionRepository


def write_fragment_pdf(root, before, after):
    """Tiny generated fixture: pypdf inserts a word space at this PDF TJ adjustment."""
    path = write_pdf(root)
    writer = PdfWriter(clone_from=PdfReader(BytesIO(path.read_bytes())))
    writer.pages[0].mediabox.upper_right = (1000, 792)
    stream = DecodedStreamObject()
    stream.set_data(
        f"BT /F1 12 Tf 40 700 Td [({before}) -140 ({after})] TJ ET".encode()
    )
    writer.pages[0][NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)
    return path


class PDFSourceTests(unittest.TestCase):
    def test_real_page_regressions_and_generated_extraction_failures(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures/km_pad_2020_quotes.json").read_text()
        )
        old_pages = {p["page_number"]: PageText(**p) for p in fixture["pages"]}
        new_pages = {
            p["page_number"]: PageText(**p) for p in fixture["pymupdf_examples"]
        }
        cases = {c["observation_id"]: c for c in fixture["cases"]}
        for key, before, after, artifact in [
            (
                9,
                "weak quality of data produced and repo",
                "rted by financial institutions",
                "repo rted",
            ),
            (
                88,
                (
                    "Mobile terminals are more cost effective and solar versions can be used in rural areas "
                    "where electric power is not availab"
                ),
                "le.",
                "availab le",
            ),
        ]:
            with self.subTest(observation_id=key), TemporaryDirectory() as directory:
                raw = ModelObservation.model_validate(cases[key]["raw_observation"])
                page = raw.evidence.spans[0].page
                self.assertFalse(
                    quote_matches(raw.evidence.spans[0].quote, old_pages[page].text)
                )
                new_chunk = DocumentChunk(
                    **fixture["document"], pages=[new_pages[page]]
                )
                self.assertEqual(validate_observation(raw, new_chunk).status, "valid")
                root = Path(directory)
                path = write_fragment_pdf(root, before, after)
                old = load_pdf(root, DOCUMENT, extractor="pypdf")
                improved = load_pdf(root, DOCUMENT)
                self.assertIn(artifact, old.pages[0].text)
                self.assertFalse(quote_matches(before + after, old.pages[0].text))
                self.assertTrue(quote_matches(before + after, improved.pages[0].text))
                self.assertEqual(
                    improved.metadata["pdf_sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                self.assertEqual(improved.metadata["extractor"]["name"], "pymupdf")
                self.assertEqual(
                    improved.metadata["extractor"]["version"], version("pymupdf")
                )
                self.assertFalse(improved.metadata["extractor"]["options"]["sort"])

    def test_projection_keeps_recorded_pages_and_exact_overlap_interval(self):
        prefix = "Unique beginning of the permitted historical trailing context. "
        ending = " Unique ending of this same permitted historical context."
        old_tail = prefix + "The data were repo rted." + ending
        full_old = "Outside historical overlap. " + old_tail
        new_tail = prefix + "The data were reported." + ending
        full_new = "Outside historical overlap. " + new_tail
        old_chunks = chunk_pages(
            DOCUMENT,
            [
                PageText(page_number=1, text=full_old),
                PageText(page_number=2, text="Next page."),
            ],
            max_chars=1,
            overlap_chars=len(old_tail) + 1,
        )
        assert old_chunks[1].overlap is not None
        self.assertEqual(old_chunks[1].overlap.text.strip(), old_tail)
        projected, details = _project_sources(
            dict(enumerate(old_chunks)),
            PDFText(
                [
                    PageText(page_number=1, text=full_new),
                    PageText(page_number=2, text="Next page."),
                ],
                {},
            ),
        )
        historical, improved = old_chunks[1], projected[1]
        self.assertEqual(improved.pages[0].page_number, 2)
        assert improved.overlap is not None
        self.assertEqual(improved.overlap.text, new_tail)
        self.assertEqual(details[0]["page"], 1)
        for quote in [
            "Outside historical overlap.",
            "The data were not reported.",
            "The data were 20% reported.",
        ]:
            result = validate_observation(
                observation(evidence={"spans": [span(quote)]}),
                historical,
                additional_source=improved,
            )
            self.assertEqual(result.status, "rejected")
        # Each full span can use either representation, without mixing text inside a span.
        result = validate_observation(
            observation(
                evidence={
                    "spans": [
                        span("The data were repo rted."),
                        span("The data were reported."),
                    ]
                }
            ),
            historical,
            additional_source=improved,
        )
        self.assertEqual(result.status, "valid")
        invalid = improved.model_copy(
            update={"overlap": PageText(page_number=3, text=new_tail)}
        )
        with self.assertRaisesRegex(ValueError, "constraints"):
            validate_observation(observation(), historical, additional_source=invalid)

    def test_unprovable_projection_fails_without_widening_source_access(self):
        old = chunk_pages(
            DOCUMENT,
            [
                PageText(
                    page_number=1, text="outside " + "A unique permitted interval. " * 8
                ),
                PageText(page_number=2, text="next"),
            ],
            max_chars=1,
            overlap_chars=160,
        )
        with self.assertRaisesRegex(ValueError, "anchors are absent or ambiguous"):
            _project_sources(
                dict(enumerate(old)),
                PDFText(
                    [
                        PageText(page_number=1, text=old[0].pages[0].text),
                        PageText(page_number=2, text="next"),
                    ],
                    {},
                ),
            )


class PDFRevalidationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "data/derived/stage1.sqlite"

    async def test_live_and_offline_share_loader_and_retain_metadata_and_history(self):
        path = write_pdf(self.root)
        with (
            redirect_stdout(StringIO()),
            patch(
                "digital_applications_prototype.extraction.cli.load_pdf",
                wraps=load_pdf,
            ) as live_loader,
        ):
            self.assertEqual(
                await cli.run_documents(
                    self.root,
                    [DOCUMENT],
                    extractor=extractor(FakeAdapter()),
                    export_files=False,
                ),
                0,
            )
        live_loader.assert_called_once_with(self.root, DOCUMENT)
        database = await initialize_database(self.path)
        self.addAsyncCleanup(database.close)
        async with database.sessions() as session:
            original = await session.scalar(select(ChunkSource))
            assert original is not None
            metadata = original.source_json["pdf_source"]
            self.assertEqual(metadata["extractor"]["name"], "pymupdf")
            run_id = next(iter(records(self.path)["extraction_runs"]))[0]
        before = records(self.path)
        with self.assertRaisesRegex(ValueError, "--refresh-pdf-text requires"):
            await revalidate(
                self.path,
                DOCUMENT.document_id,
                run_id,
                root=self.root,
                refresh_pdf_text=True,
            )
        options = {
            "root": self.root,
            "source_pdf": path,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        with patch(
            "digital_applications_prototype.extraction.revalidation.load_pdf",
            wraps=load_pdf,
        ) as offline:
            await revalidate(self.path, DOCUMENT.document_id, run_id, **options)
            offline.assert_not_called()  # Historical source text is preferred by default.
            preview = await revalidate(
                self.path,
                DOCUMENT.document_id,
                run_id,
                refresh_pdf_text=True,
                **options,
            )
            offline.assert_called_once_with(self.root, DOCUMENT)
        self.assertEqual(before, records(self.path))
        self.assertEqual(preview["after"]["usable"], 1)
        with (
            patch(
                "digital_applications_prototype.llm.service.LLMService.generate_structured",
                side_effect=AssertionError("No model calls"),
            ),
            patch(
                "socket.socket.connect",
                side_effect=AssertionError("No network"),
            ),
        ):
            first = await revalidate(
                self.path,
                DOCUMENT.document_id,
                run_id,
                apply=True,
                refresh_pdf_text=True,
                **options,
            )
            contents = {
                key: Path(value).read_bytes() for key, value in first["exports"].items()
            }
            second = await revalidate(
                self.path,
                DOCUMENT.document_id,
                run_id,
                apply=True,
                refresh_pdf_text=True,
                **options,
            )
        self.assertEqual(first["after"], second["after"])
        self.assertEqual(first["exports"], second["exports"])
        self.assertEqual(
            contents,
            {key: Path(value).read_bytes() for key, value in second["exports"].items()},
        )
        self.assertEqual(
            before["extraction_chunk_sources"],
            records(self.path)["extraction_chunk_sources"],
        )
        async with database.sessions() as session:
            revisions = list(await session.scalars(select(ValidationRevision)))
            self.assertEqual(
                revisions[-1].source_json["extractor"], metadata["extractor"]
            )
            self.assertEqual(
                revisions[-1].source_json["pdf_sha256"], metadata["pdf_sha256"]
            )
            self.assertEqual(
                revisions[-1].validator_version, second["validator_version"]
            )
            self.assertEqual(revisions[-1].changes_json, [])
        write_pdf(self.root, texts=("Different PDF, same filename.",))
        with self.assertRaisesRegex(ValueError, "retained source SHA-256"):
            await revalidate(
                self.path,
                DOCUMENT.document_id,
                run_id,
                refresh_pdf_text=True,
                root=self.root,
                source_pdf=path,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    async def test_legacy_boundaries_are_preserved_when_new_extractor_would_rechunk(
        self,
    ):
        write_pdf(self.root, texts=(QUOTE, QUOTE))
        historical = load_pdf(self.root, DOCUMENT, extractor="pypdf")
        current = load_pdf(self.root, DOCUMENT)
        limit = sum(len(page.text) for page in historical.pages) + 2
        old_chunks = chunk_pages(DOCUMENT, historical.pages, max_chars=limit)
        self.assertEqual(len(old_chunks), 1)
        self.assertEqual(len(chunk_pages(DOCUMENT, current.pages, max_chars=limit)), 2)
        database = await initialize_database(self.path)
        self.addAsyncCleanup(database.close)
        repository = ExtractionRepository(database.sessions)
        run_id = await repository.create_run(DOCUMENT.document_id)
        chunk_ids = await repository.create_chunks(run_id, old_chunks)
        result = await extractor(FakeAdapter()).extract(old_chunks[0])
        await repository.mark_chunk_running(chunk_ids[0])
        await repository.persist_completed_chunk(chunk_ids[0], old_chunks[0], result)
        await repository.finalize_run(run_id)
        async with database.sessions.begin() as session:
            await session.execute(
                delete(ChunkSource).where(ChunkSource.chunk_id.in_(chunk_ids))
            )
        before = records(self.path)
        preview = await revalidate(
            self.path,
            DOCUMENT.document_id,
            run_id,
            root=self.root,
            source_pdf=self.root / DOCUMENT.path,
            source_sha256=current.metadata["pdf_sha256"],
            max_chars=limit,
            overlap_chars=600,
        )
        self.assertEqual(preview["after"]["usable"], 1)
        self.assertEqual(preview["newly_rejected_observation_ids"], [])
        self.assertEqual(
            preview["source"]["historical_chunk_extractor"]["name"], "pypdf"
        )
        self.assertEqual(before, records(self.path))
