from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from digital_applications_prototype.extraction.chunking import chunk_pages
from digital_applications_prototype.extraction.cli import main, run_documents
from digital_applications_prototype.extraction.manifest import INPUT_DOCUMENTS, input_path, select_documents
from digital_applications_prototype.extraction.models import (
    ExtractionResult, ImplementationObservation, ModelObservation, PageText,
)
from digital_applications_prototype.extraction.pdf import PDFTextError, extract_pages
from digital_applications_prototype.extraction.persistence import output_path, write_observations
from digital_applications_prototype.extraction.service import ImplementationObservationExtractor
from digital_applications_prototype.extraction.validation import GroundingError, validate_observation
from digital_applications_prototype.llm.prompt_renderer import PromptRenderer
from digital_applications_prototype.llm.service import LLMService


DOCUMENT = select_documents(document="KM_ISR07_2023")[0]
QUOTE = "Limited agent coverage constrained rural delivery."


def observation(**changes):
    data = dict(observation_type="condition", valence="constraint", claim_scope="reported",
                observation="Limited agent coverage constrained rural delivery.",
                raw_factor_label="agent coverage", raw_factor_description=(
                    "Geographic availability of agents needed to deliver financial services to intended users."),
                evidence_strength="explicit", source_quote=QUOTE, source_page=1)
    data.update(changes)
    return ModelObservation(**data)


def make_chunk(text=QUOTE):
    return chunk_pages(DOCUMENT, [PageText(page_number=1, text=text)])[0]


def write_pdf(root, document=DOCUMENT, texts=(QUOTE,)):
    """Write a minimal PDF with one text stream per page for offline ingestion tests."""
    path = root / document.path
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        # Small real PDF text fixture without another test dependency.
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 40 700 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as target:
        writer.write(target)
    return path


class FakeAdapter:
    def __init__(self, items=None, error=None):
        self.items = [observation()] if items is None else items
        self.error = error
        self.calls = []

    async def generate_structured(self, *, prompt, response_model, model=None):
        self.calls.append((prompt, response_model, model))
        if self.error:
            raise self.error
        return response_model(observations=self.items)


def extractor(adapter=None):
    return ImplementationObservationExtractor(PromptRenderer(), LLMService(adapter or FakeAdapter()))


class ManifestAndPDFTests(unittest.TestCase):
    def test_explicit_manifest_and_selection(self):
        self.assertEqual(len(INPUT_DOCUMENTS), 15)
        self.assertEqual(len({d.document_id for d in INPUT_DOCUMENTS}), 15)
        self.assertEqual(select_documents(document="KM_ISR07_2023"), [DOCUMENT])
        self.assertEqual(len(select_documents(country="comoros")), 5)
        self.assertEqual(len(select_documents(all_documents=True)), 15)
        self.assertTrue(all(d.path.parts[:2] == ("data", "input") for d in INPUT_DOCUMENTS))
        for options in ({}, {"document": "unknown"}, {"document": DOCUMENT.document_id, "country": "comoros"}):
            with self.assertRaises(ValueError):
                select_documents(**options)

    def test_pdf_preserves_physical_pages(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_pdf(root, texts=(QUOTE, "Second page."))
            pages = extract_pages(root, DOCUMENT)
            self.assertEqual([p.page_number for p in pages], [1, 2])
            self.assertIn(QUOTE, pages[0].text)
            self.assertIn("Second page.", pages[1].text)

    def test_empty_pdf_is_visible_failure(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_pdf(root, texts=("",))
            with self.assertRaisesRegex(PDFTextError, "OCR"):
                extract_pages(root, DOCUMENT)

    def test_holdout_and_traversal_rejected_before_pdf_read(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (Path("data/holdout/comoros/test.pdf"), Path("data/input/../holdout/test.pdf")):
                bad = DOCUMENT.model_copy(update={"path": relative})
                with patch("digital_applications_prototype.extraction.pdf.PdfReader") as reader:
                    with self.assertRaises(ValueError):
                        extract_pages(root, bad)
                    reader.assert_not_called()

    def test_symlink_to_holdout_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / DOCUMENT.path
            source.parent.mkdir(parents=True)
            # A dangling link is sufficient: the target is never opened or inspected.
            source.symlink_to(root / "data/holdout/secret.pdf")
            with self.assertRaises(ValueError):
                input_path(root, DOCUMENT.path)

    def test_symlinked_input_root_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data/input").symlink_to(root / "data/holdout", target_is_directory=True)
            with self.assertRaises(ValueError):
                input_path(root, DOCUMENT.path)


class ChunkAndGroundingTests(unittest.TestCase):
    def test_page_boundary_chunking_soft_limit_and_determinism(self):
        pages = [PageText(page_number=i, text="x" * n) for i, n in [(1, 5), (2, 5), (3, 30)]]
        chunks = chunk_pages(DOCUMENT, pages, max_chars=12)
        self.assertEqual([(c.start_page, c.end_page) for c in chunks], [(1, 2), (3, 3)])
        self.assertEqual(chunks, chunk_pages(DOCUMENT, pages, max_chars=12))
        self.assertEqual(chunks[0].project_id, DOCUMENT.project_id)
        with self.assertRaises(ValueError):
            chunk_pages(DOCUMENT, pages, max_chars=0)

    def test_quote_whitespace_is_allowed_original_retained_and_id_stable(self):
        item = observation(source_quote="Limited agent\ncoverage constrained rural delivery.")
        first = validate_observation(item, make_chunk())
        self.assertEqual(first.source_quote, item.source_quote)
        self.assertEqual(first.observation_id, validate_observation(item, make_chunk()).observation_id)
        self.assertEqual(first.document_date, DOCUMENT.document_date)

    def test_unsupported_quotes_wrong_pages_and_empty_fields_rejected(self):
        for changes in ({"source_quote": "Limited agent coverage improved rural delivery."},
                        {"source_page": 2}, {"observation": " "}, {"raw_factor_description": ""},
                        {"raw_factor_label": "x " * 12}, {"source_quote": " "}):
            with self.subTest(changes=changes), self.assertRaises(GroundingError):
                validate_observation(observation(**changes), make_chunk())

    def test_quote_must_be_on_claimed_page(self):
        chunk = chunk_pages(DOCUMENT, [PageText(page_number=1, text="Other text."),
                                       PageText(page_number=2, text=QUOTE)])[0]
        with self.assertRaisesRegex(GroundingError, "not present on source_page"):
            validate_observation(observation(), chunk)
        result = validate_observation(observation(source_page=None), chunk)
        self.assertEqual(result.source_page, 2)

    def test_cross_page_quote_can_retain_null_page(self):
        chunk = chunk_pages(DOCUMENT, [PageText(page_number=1, text="Limited agent coverage"),
                                       PageText(page_number=2, text="constrained rural delivery.")])[0]
        self.assertIsNone(validate_observation(observation(source_page=None), chunk).source_page)

    def test_schema_rejects_invalid_enum_and_metadata_generation(self):
        with self.assertRaises(ValidationError):
            observation(valence="bad")
        with self.assertRaises(ValidationError):
            observation(project_id="hallucinated")
        first, second = observation(), observation()
        first.actors.append("Agent")
        self.assertEqual(second.actors, [])


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_metadata_and_existing_llm_service(self):
        adapter = FakeAdapter()
        service = extractor(adapter)
        items = await service.extract(make_chunk())
        self.assertIsInstance(service.llm, LLMService)
        self.assertEqual(len(adapter.calls), 1)
        prompt, response_model, _ = adapter.calls[0]
        for value in (DOCUMENT.project_id, DOCUMENT.country, DOCUMENT.document_id,
                      "2023-09-11", "[PDF page 1]", QUOTE):
            self.assertIn(value, prompt)
        self.assertIs(response_model, ExtractionResult)
        self.assertIsInstance(items[0], ImplementationObservation)
        self.assertEqual(items[0].project_id, DOCUMENT.project_id)
        self.assertEqual((items[0].chunk_start_page, items[0].chunk_end_page), (1, 1))

    async def test_empty_results_and_visible_grounding_failure(self):
        self.assertEqual(await extractor(FakeAdapter(items=[])).extract(make_chunk()), [])
        bad = FakeAdapter(items=[observation(source_quote="Invented quote")])
        with self.assertRaisesRegex(GroundingError, "KM_ISR07_2023.*observation 1"):
            await extractor(bad).extract(make_chunk())

    async def test_incremental_execution_failure_force_and_completion(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            root = Path(directory)
            write_pdf(root)
            missing = select_documents(document="MX_ISR05_2014")[0]
            adapter = FakeAdapter()
            status = await run_documents(root, [DOCUMENT, missing], extractor=extractor(adapter))
            self.assertEqual(status, 1)
            path = output_path(root, DOCUMENT)
            original = path.read_text()
            decoded = ImplementationObservation.model_validate_json(original)
            self.assertEqual(decoded.document_id, DOCUMENT.document_id)
            self.assertFalse(output_path(root, missing).exists())
            self.assertEqual(await run_documents(root, [DOCUMENT], extractor=extractor(adapter)), 0)
            self.assertEqual(len(adapter.calls), 1)
            failing = FakeAdapter(error=RuntimeError("offline failure"))
            self.assertEqual(await run_documents(root, [DOCUMENT], extractor=extractor(failing), force=True), 1)
            self.assertEqual(path.read_text(), original)
            self.assertEqual(await run_documents(root, [DOCUMENT], extractor=extractor(adapter), force=True), 0)
            self.assertEqual(len(adapter.calls), 2)
            self.assertFalse(list(path.parent.glob("*.part")))

    async def test_zero_observations_still_completes_document(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            write_pdf(root)
            self.assertEqual(await run_documents(root, [DOCUMENT], extractor=extractor(FakeAdapter(items=[]))), 0)
            self.assertEqual(output_path(root, DOCUMENT).read_text(), "")

    async def test_later_chunk_failure_does_not_write_partial_document(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            root = Path(directory)
            write_pdf(root, texts=(QUOTE, "Unsupported second page."))
            self.assertEqual(await run_documents(root, [DOCUMENT], extractor=extractor(), max_chars=1), 1)
            self.assertFalse(output_path(root, DOCUMENT).exists())


class PersistenceAndCLITests(unittest.TestCase):
    def test_jsonl_round_trip(self):
        with TemporaryDirectory() as directory:
            path = output_path(Path(directory), DOCUMENT)
            item = validate_observation(observation(), make_chunk())
            write_observations(path, [item, item])
            self.assertEqual([ImplementationObservation.model_validate_json(line)
                              for line in path.read_text().splitlines()], [item, item])

    def test_dry_run_needs_no_credentials_no_model_no_writes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_pdf(root)
            output = StringIO()
            with patch.dict("os.environ", {}, clear=True), redirect_stdout(output), \
                 patch("digital_applications_prototype.core.llm_runtime.configured_llm") as runtime, \
                 patch.object(LLMService, "generate_structured") as generate:
                status = main(["--document", DOCUMENT.document_id, "--dry-run", "--root", str(root)])
            self.assertEqual(status, 0)
            runtime.assert_not_called()
            generate.assert_not_called()
            self.assertIn("First prompt:", output.getvalue())
            self.assertNotIn(QUOTE, output.getvalue())
            self.assertFalse((root / "data/derived").exists())

    def test_cli_refuses_ambiguous_selection_and_multi_document_dry_run(self):
        for args in ([], ["--all", "--document", DOCUMENT.document_id], ["--country", "comoros", "--dry-run"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(args)

    def test_completed_cli_run_does_not_require_settings(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            write_observations(output_path(root, DOCUMENT), [])
            with patch("digital_applications_prototype.core.llm_runtime.configured_llm") as runtime:
                self.assertEqual(main(["--document", DOCUMENT.document_id, "--root", str(root)]), 0)
                runtime.assert_not_called()
