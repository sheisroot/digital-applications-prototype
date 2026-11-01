from contextlib import asynccontextmanager, redirect_stdout, redirect_stderr
from datetime import UTC, date, datetime
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from digital_applications_prototype.extraction.chunking import chunk_pages
from digital_applications_prototype.extraction.cli import main, run_documents
from digital_applications_prototype.extraction.manifest import INPUT_DOCUMENTS, input_path, select_documents
from digital_applications_prototype.extraction.models import (
    ClaimScope, EvidenceBundle, EvidenceRole, EvidenceSpan, ExtractionResult,
    ImplementationObservation, ModelObservation, ObservationValidationResult, PageText, RunReport, RunStatus,
)
from digital_applications_prototype.extraction.pdf import PDFTextError, extract_pages
from digital_applications_prototype.extraction.persistence import reserve_run, write_observations, write_report
from digital_applications_prototype.extraction.service import ImplementationObservationExtractor
from digital_applications_prototype.extraction.validation import (
    deduplicate_observations, validate_observation,
)
from digital_applications_prototype.llm.prompt_renderer import PromptRenderer
from digital_applications_prototype.llm.service import LLMService
from digital_applications_prototype.llm.adapters.base import TokenUsage


DOCUMENT = select_documents(document="KM_ISR07_2023")[0]
QUOTE = "Limited agent coverage constrained rural delivery."


def span(quote=QUOTE, page=1, **changes):
    return EvidenceSpan(quote=quote, page=page, **changes)


def observation(**changes):
    data = dict(observation_type="condition", valence="constraint", claim_scope="actual",
                observation="Limited agent coverage constrained rural delivery.",
                factor_description=(
                    "Geographic availability of agents needed to deliver financial services to intended users."),
                evidence_strength="explicit", evidence=EvidenceBundle(spans=[span()]))
    data.update(changes)
    return ModelObservation(**data)


def validated(item, chunk):
    result = validate_observation(item, chunk)
    assert result.status == "valid", result.warnings
    return result.observation


def rejected(item, chunk):
    result = validate_observation(item, chunk)
    assert result.status == "rejected"
    assert result.observation is None
    assert result.warnings
    return result


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
    def __init__(self, items=None, error=None, responses=None, usages=None):
        self.items = [observation()] if items is None else items
        self.error = error
        self.calls = []
        self.model = "offline-model"
        self.last_usage = None
        self.responses = responses
        self.usages = usages

    async def generate_structured(self, *, prompt, response_model, model=None):
        self.calls.append((prompt, response_model, model))
        self.last_usage = None
        if self.error:
            raise self.error
        index = len(self.calls) - 1
        items = self.responses[index] if self.responses is not None else self.items
        if isinstance(items, Exception):
            raise items
        result = response_model.model_validate(items) if isinstance(items, dict) else response_model(observations=items)
        self.last_usage = self.usages[index] if self.usages is not None else None
        return result


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
        chunks = chunk_pages(DOCUMENT, pages, max_chars=12, overlap_chars=0)
        self.assertEqual([(c.start_page, c.end_page) for c in chunks], [(1, 2), (3, 3)])
        self.assertEqual(chunks, chunk_pages(DOCUMENT, pages, max_chars=12, overlap_chars=0))
        self.assertEqual(chunks[0].project_id, DOCUMENT.project_id)
        with self.assertRaises(ValueError):
            chunk_pages(DOCUMENT, pages, max_chars=0)

    def test_overlap_is_a_bounded_verbatim_suffix_with_page_provenance(self):
        pages = [PageText(page_number=1, text="Earlier evidence. " * 50 + "Limited agent coverage"),
                 PageText(page_number=2, text="constrained rural delivery."),
                 PageText(page_number=3, text="Third page.")]
        chunks = chunk_pages(DOCUMENT, pages, max_chars=30, overlap_chars=24)
        self.assertEqual(chunks, chunk_pages(DOCUMENT, pages, max_chars=30, overlap_chars=24))
        self.assertIsNone(chunks[0].overlap)
        overlap = chunks[1].overlap
        self.assertEqual(overlap.page_number, 1)
        self.assertLessEqual(len(overlap.text), 24)
        self.assertTrue(pages[0].text.endswith(overlap.text))
        self.assertNotEqual(overlap.text, pages[0].text)
        self.assertEqual(chunks[1].pages, [pages[1]])
        self.assertEqual((chunks[1].start_page, chunks[1].end_page), (1, 2))
        cross_page = validated(observation(evidence={"spans": [
            span("Limited agent coverage", 1), span("constrained rural delivery.", 2),
        ]}), chunks[1])
        self.assertEqual([s.page for s in cross_page.evidence.spans], [1, 2])
        self.assertEqual(chunks[2].overlap.page_number, 2)
        self.assertNotIn("Limited agent coverage", chunks[2].text)
        rejected(observation(evidence={"spans": [span("Earlier evidence.")]}), chunks[1])

    def test_overlap_page_matching_empty_pages_and_disabled_overlap(self):
        pages = [PageText(page_number=1, text=QUOTE), PageText(page_number=2, text=""),
                 PageText(page_number=3, text="New evidence.")]
        chunks = chunk_pages(DOCUMENT, pages, max_chars=len(QUOTE) + 2)
        self.assertEqual(chunks[1].overlap.page_number, 1)
        self.assertEqual(validated(observation(), chunks[1]).evidence.spans[0].page, 1)
        rejected(observation(evidence={"spans": [span(page=3)]}), chunks[1])
        self.assertTrue(all(c.overlap is None for c in chunk_pages(
            DOCUMENT, pages, max_chars=10, overlap_chars=0)))
        with self.assertRaises(ValueError):
            chunk_pages(DOCUMENT, pages, overlap_chars=-1)

    def test_overlap_does_not_include_a_partial_word(self):
        pages = [PageText(page_number=1, text="Longword ending"),
                 PageText(page_number=2, text="next")]
        chunk = chunk_pages(DOCUMENT, pages, max_chars=15, overlap_chars=9)[1]
        self.assertEqual(chunk.overlap.text, " ending")

    def test_quote_whitespace_is_allowed_original_retained_and_id_stable(self):
        item = observation(evidence={"spans": [span("Limited agent\ncoverage constrained rural delivery.")]})
        first = validated(item, make_chunk())
        self.assertEqual(first.evidence, item.evidence)
        self.assertEqual(first.observation_id, validated(item, make_chunk()).observation_id)
        self.assertEqual(first.document_date, DOCUMENT.document_date)

    def test_unsupported_quotes_wrong_pages_and_empty_fields_rejected(self):
        for changes in ({"evidence": {"spans": [span("Limited agent coverage improved rural delivery.")]}},
                        {"evidence": {"spans": [span(page=2)]}},
                        {"observation": " "}, {"factor_description": ""},
                        {"evidence": {"spans": [span(" ")]}}):
            with self.subTest(changes=changes):
                rejected(observation(**changes), make_chunk())

    def test_quote_must_be_on_claimed_page(self):
        chunk = chunk_pages(DOCUMENT, [PageText(page_number=1, text="Other text."),
                                       PageText(page_number=2, text=QUOTE)])[0]
        failure = rejected(observation(), chunk)
        self.assertIn("not present on page 1", failure.warnings[0].message)
        result = validated(observation(evidence={"spans": [span(page=2)]}), chunk)
        self.assertEqual(result.evidence.spans[0].page, 2)

    def test_cross_page_claim_requires_separate_page_spans(self):
        chunk = chunk_pages(DOCUMENT, [PageText(page_number=1, text="Limited agent coverage"),
                                       PageText(page_number=2, text="constrained rural delivery.")])[0]
        rejected(observation(), chunk)
        item = observation(evidence={"spans": [span("Limited agent coverage", 1),
                                              span("constrained rural delivery.", 2)]})
        self.assertEqual(len(validated(item, chunk).evidence.spans), 2)

    def test_noncontiguous_spans_are_grounded_individually_summary_is_derived(self):
        text = QUOTE + "\nUnrelated material.\nDelivery had not resumed."
        evidence = EvidenceBundle(summary="The passages jointly describe the delivery constraint.",
                                  spans=[span(), span("Delivery had not resumed.", role="qualifies")])
        result = validated(observation(evidence=evidence), make_chunk(text))
        self.assertEqual(result.evidence, evidence)
        bad = evidence.model_copy(update={"spans": [span(), span("Invented second passage.")]})
        failure = rejected(observation(evidence=bad), make_chunk(text))
        self.assertEqual(failure.warnings[0].evidence_span_index, 2)

    def test_page_must_be_represented_in_chunk_not_merely_in_range(self):
        pages = [PageText(page_number=1, text=QUOTE), PageText(page_number=2, text=""),
                 PageText(page_number=3, text="Other evidence.")]
        chunk = chunk_pages(DOCUMENT, pages, max_chars=len(QUOTE) + 2)[1]
        self.assertEqual((chunk.start_page, chunk.end_page), (1, 3))
        failure = rejected(observation(evidence={"spans": [span(page=2)]}), chunk)
        self.assertEqual(failure.warnings[0].code, "invalid_evidence_page")

    def test_schema_rejects_invalid_enum_and_metadata_generation(self):
        with self.assertRaises(ValidationError):
            observation(valence="bad")
        with self.assertRaises(ValidationError):
            observation(project_id="hallucinated")
        first, second = observation(), observation()
        first.actors.append("Agent")
        self.assertEqual(second.actors, [])

    def test_validation_outcome_invariants_and_empty_evidence_guard(self):
        empty = observation(evidence={'spans': []})
        rejected(empty, make_chunk())
        with self.assertRaises(ValidationError):
            ObservationValidationResult(status='valid')
        item = validated(observation(), make_chunk())
        with self.assertRaises(ValidationError):
            ImplementationObservation.model_validate({**item.model_dump(), 'evidence': {'spans': []}})
        with self.assertRaises(ValidationError):
            ObservationValidationResult(status='rejected', observation=item)
        with self.assertRaises(ValidationError):
            ObservationValidationResult(status='repaired', observation=item)

    def test_factor_description_and_evidence_required_legacy_fields_forbidden(self):
        for schema in (ModelObservation, ImplementationObservation):
            for field in ("raw_factor_label", "raw_factor_description",
                          "source_quote", "source_page", "evidence_date"):
                self.assertNotIn(field, schema.model_fields)
            self.assertTrue(schema.model_fields["factor_description"].is_required())
            self.assertTrue(schema.model_fields["evidence"].is_required())
        for required in ("factor_description", "evidence"):
            payload = observation().model_dump()
            del payload[required]
            with self.assertRaises(ValidationError):
                ModelObservation(**payload)
        for old_field in ("raw_factor_label", "raw_factor_description", "source_quote", "source_page", "evidence_date"):
            with self.assertRaises(ValidationError):
                observation(**{old_field: "legacy value"})

    def test_claim_scopes(self):
        self.assertEqual({scope.value for scope in ClaimScope},
                         {"actual", "planned", "anticipated", "recommended", "inferred"})
        for scope in ClaimScope:
            self.assertEqual(observation(claim_scope=scope.value).claim_scope, scope)
        for old_scope in ("observed", "reported"):
            with self.assertRaises(ValidationError):
                observation(claim_scope=old_scope)

    def test_evidence_models_roles_and_dates(self):
        item = observation()
        self.assertIsNone(item.evidence.summary)
        self.assertEqual(len(item.evidence.spans), 1)
        self.assertIsNone(item.evidence.spans[0].evidence_date)
        self.assertEqual(item.evidence.spans[0].role, EvidenceRole.SUPPORTS)
        self.assertEqual({role.value for role in EvidenceRole},
                         {"supports", "qualifies", "contradicts", "context"})
        for role in EvidenceRole:
            self.assertEqual(span(role=role.value).role, role)
        for data in ({"evidence": {"summary": "No citations"}},
                     {"evidence": {"spans": [{"quote": QUOTE}]}},
                     {"evidence": {"spans": [{"quote": QUOTE, "page": None}]}}):
            with self.assertRaises(ValidationError):
                observation(**data)
        for changes in ({"role": "irrelevant"}, {"evidence_date": "2022-02-30"}, {"page": 0}, {"quote": ""}):
            with self.assertRaises(ValidationError):
                EvidenceSpan(**{"quote": QUOTE, "page": 1, **changes})
        self.assertEqual(span(evidence_date="2022-05-13").evidence_date, date(2022, 5, 13))

    def test_table_date_and_undated_prose_keep_separate_temporal_provenance(self):
        table = "Date 01-Jul-2020 05-Oct-2020 05-Oct-2020 30-Jun-2025"
        prose = "The GRM has yet to be established."
        evidence = EvidenceBundle(summary="The table supplies the status date; prose describes the status.",
                                  spans=[span(table, evidence_date="2020-10-05"), span(prose)])
        result = validated(observation(evidence=evidence), make_chunk(table + "\n" + prose))
        self.assertEqual(result.evidence.spans[0].evidence_date, date(2020, 10, 5))
        self.assertIsNone(result.evidence.spans[1].evidence_date)
        self.assertEqual(result.document_date, date(2023, 9, 11))
        unsupported = evidence.model_copy(update={"spans": [span(table), span(prose, evidence_date="2020-10-05")]})
        repaired = validate_observation(observation(evidence=unsupported), make_chunk(table + "\n" + prose))
        self.assertEqual(repaired.status, "repaired")
        self.assertIsNone(repaired.observation.evidence.spans[1].evidence_date)
        self.assertEqual(repaired.warnings[0].evidence_span_index, 2)

    def test_evidence_date_must_appear_in_its_own_exact_quote(self):
        for text in ("Actual Date: May 13, 2022", "Date 13-May-2022", "Date 13 May 2022",
                     "Date May 13 2022", "Date 2022-05-13", "Date 2022 / 05 / 13"):
            with self.subTest(text=text):
                item = observation(evidence={"spans": [span(text, evidence_date="2022-05-13")]})
                self.assertEqual(validated(item, make_chunk(text)).evidence.spans[0].evidence_date,
                                 date(2022, 5, 13))
        for text in (QUOTE, "Date May 2022", "Date May 13, 2023", "Date May 13, 20220"):
            with self.subTest(text=text):
                item = observation(evidence={"spans": [span(text, evidence_date="2022-05-13")]})
                result = validate_observation(item, make_chunk(text), observation_index=18)
                self.assertEqual(result.status, "repaired")
                self.assertIsNone(result.observation.evidence.spans[0].evidence_date)
                self.assertEqual(item.evidence.spans[0].evidence_date, date(2022, 5, 13))
                warning = result.warnings[0]
                self.assertEqual((warning.code, warning.observation_index, warning.evidence_span_index,
                                  warning.action, warning.chunk_start_page, warning.chunk_end_page),
                                 ("unsupported_evidence_date", 18, 1, "evidence_date_set_to_null", 1, 1))

    def test_overlap_deduplication_and_ids_ignore_generated_wording_and_boundaries(self):
        first = validated(observation(), make_chunk())
        chunk = chunk_pages(DOCUMENT, [PageText(page_number=1, text=QUOTE),
                                       PageText(page_number=2, text="More evidence.")], max_chars=len(QUOTE))[1]
        duplicate = validated(observation(observation="A differently phrased claim.",
            factor_description="A different description of the same mechanism.",
            evidence={"summary": "Derived wording also differs.", "spans": [span(QUOTE.replace(" ", "\n"))]}), chunk)
        self.assertNotEqual(first.chunk_end_page, duplicate.chunk_end_page)
        self.assertEqual(first.observation_id, duplicate.observation_id)
        self.assertEqual(deduplicate_observations([first, duplicate]), [first])

    def test_deduplication_preserves_different_types_pages_roles_quotes_and_documents(self):
        chunk = chunk_pages(DOCUMENT, [PageText(page_number=1, text=QUOTE + " Other evidence."),
                                       PageText(page_number=2, text=QUOTE)])[0]
        first = validated(observation(), chunk)
        action = validated(observation(observation_type="action"), chunk)
        other_page = validated(observation(evidence={"spans": [span(page=2)]}), chunk)
        other_role = validated(observation(evidence={"spans": [span(role="qualifies")]}), chunk)
        other_quote = validated(observation(evidence={"spans": [span("Other evidence.")]}), chunk)
        other_document = first.model_copy(update={"document_id": "KM_ISR10_2025"})
        items = [first, action, other_page, other_role, other_quote, other_document]
        self.assertEqual(deduplicate_observations(items), items)
        self.assertEqual(len({i.observation_id for i in items[:-1]}), 5)

    def test_multispan_signature_is_independent_of_span_order(self):
        chunk = make_chunk(QUOTE + " Other evidence.")
        spans = [span(), span("Other evidence.", role="context")]
        first = validated(observation(evidence={"spans": spans}), chunk)
        second = validated(observation(evidence={"spans": list(reversed(spans))}), chunk)
        self.assertEqual(first.observation_id, second.observation_id)
        self.assertEqual(deduplicate_observations([first, second]), [first])
        single_span = validated(observation(), chunk)
        self.assertEqual(deduplicate_observations([first, single_span]), [first, single_span])


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_metadata_and_existing_llm_service(self):
        adapter = FakeAdapter()
        service = extractor(adapter)
        result = await service.extract(make_chunk())
        self.assertIsInstance(service.llm, LLMService)
        self.assertEqual((result.observations_returned, result.observations_valid), (1, 1))
        prompt, response_model, _ = adapter.calls[0]
        for value in (DOCUMENT.project_id, DOCUMENT.country, DOCUMENT.document_id,
                      "2023-09-11", "[PDF page 1]", QUOTE):
            self.assertIn(value, prompt)
        self.assertIs(response_model, ExtractionResult)
        self.assertEqual(result.observations[0].project_id, DOCUMENT.project_id)
        self.assertIsNone(result.observations[0].evidence.spans[0].evidence_date)
        for instruction in ("factor_description", "prerequisites", "tradeoffs", "Do not disproportionately",
                            "independently useful", "Temporal provenance", "evidence_date",
                            "all materially populated fields", "actual:", "planned:", "anticipated:",
                            "evidence.spans", "evidence.summary", "non-contiguous", "THIS call"):
            self.assertIn(instruction, prompt)
        self.assertNotIn("raw_factor_label", prompt)

    async def test_repair_rejection_and_later_observations_are_independent(self):
        repaired = observation(evidence={"spans": [span(evidence_date="2020-10-05")]})
        bad = observation(evidence={"spans": [span("Invented quote")]})
        later = observation(observation_type="action")
        result = await extractor(FakeAdapter(items=[repaired, bad, later])).extract(make_chunk())
        self.assertEqual((result.observations_returned, result.observations_valid,
                          result.observations_repaired, result.observations_rejected), (3, 1, 1, 1))
        self.assertEqual(len(result.observations), 2)
        self.assertEqual(result.observations[-1].observation_type.value, "action")
        self.assertIsNone(result.observations[0].evidence.spans[0].evidence_date)
        self.assertEqual([w.code for w in result.warnings], ["unsupported_evidence_date", "ungrounded_quote"])
        self.assertEqual([w.observation_index for w in result.warnings], [1, 2])

    async def test_empty_results_and_whole_response_failure(self):
        result = await extractor(FakeAdapter(items=[])).extract(make_chunk())
        self.assertEqual(result.observations_returned, 0)
        with self.assertRaises(ValidationError):
            await extractor(FakeAdapter(responses=[{"observations": [{"observation": "malformed"}]}])).extract(make_chunk())

    async def test_empty_evidence_rejects_only_one_observation(self):
        result = await extractor(FakeAdapter(items=[observation(evidence={'spans': []}),
                                                   observation()])).extract(make_chunk())
        self.assertEqual((result.observations_returned, result.observations_rejected,
                          result.observations_valid), (2, 1, 1))
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(result.warnings[0].code, 'observation_rejected')

    async def test_dated_overlap_metadata_and_usage(self):
        text = QUOTE + " Actual Date: May 13, 2022"
        chunk = chunk_pages(DOCUMENT, [PageText(page_number=1, text=text),
                                       PageText(page_number=2, text="New evidence.")], max_chars=50)[1]
        item = observation(evidence={"spans": [span(text, evidence_date="2022-05-13")]})
        adapter = FakeAdapter(items=[item], usages=[TokenUsage(input_tokens=100, output_tokens=20)])
        result = await extractor(adapter).extract(chunk)
        self.assertEqual(result.observations[0].evidence.spans[0].evidence_date, date(2022, 5, 13))
        self.assertEqual(result.usage.input_tokens, 100)
        self.assertIn("[PDF page 1] (trailing overlap)", adapter.calls[0][0])


def read_reports(root):
    reports = [RunReport.model_validate_json(path.read_text())
               for path in (root / 'data/derived/extracted').rglob('*.report.json')]
    return sorted(reports, key=lambda report: report.started_at)


def run_jsonls(root):
    return list((root / 'data/derived/extracted').rglob('*__*.jsonl'))


class RunTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_runs_are_paired_immutable_and_never_skip(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            write_pdf(root)
            legacy = root / 'data/derived/extracted/comoros/KM_ISR07_2023.jsonl'
            legacy.parent.mkdir(parents=True)
            legacy.write_text('old output')
            adapter = FakeAdapter()
            service = extractor(adapter)
            self.assertEqual(await run_documents(root, [DOCUMENT], extractor=service), 0)
            old_paths = {p: p.read_bytes() for p in legacy.parent.iterdir()}
            self.assertEqual(await run_documents(root, [DOCUMENT], extractor=service), 0)
            reports = read_reports(root)
            self.assertEqual(len(reports), 2)
            self.assertEqual(len(adapter.calls), 2)
            self.assertNotEqual(reports[0].run_id, reports[1].run_id)
            for report in reports:
                self.assertEqual(report.status, RunStatus.COMPLETED)
                self.assertEqual(report.model, 'offline-model')
                self.assertEqual(report.schema_version, 'stage1-v3')
                self.assertEqual((report.chunks_total, report.chunks_succeeded, report.chunks_failed), (1, 1, 0))
                self.assertEqual((report.observations_returned, report.observations_valid,
                                  report.observations_persisted), (1, 1, 1))
                self.assertIsNone(report.input_tokens)
                self.assertEqual(report.jsonl_file, f'{DOCUMENT.document_id}__{report.run_id}.jsonl')
                self.assertTrue((legacy.parent / report.jsonl_file).exists())
                self.assertTrue((legacy.parent / f'{DOCUMENT.document_id}__{report.run_id}.report.json').exists())
                self.assertGreaterEqual(report.finished_at, report.started_at)
            for path, original in old_paths.items():
                self.assertEqual(path.read_bytes(), original)
            self.assertFalse(list(legacy.parent.glob('*.part')))
            self.assertFalse(list(legacy.parent.glob('.*.reserve')))

    async def test_warnings_do_not_change_completed_and_counts_reconcile(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            write_pdf(root)
            repaired = observation(evidence={'spans': [span(evidence_date='2020-10-05')]})
            bad = observation(evidence={'spans': [span('Ungrounded text')]})
            action = observation(observation_type='action')
            adapter = FakeAdapter(items=[observation(), repaired, bad, action],
                                  usages=[TokenUsage(input_tokens=40, output_tokens=80)])
            self.assertEqual(await run_documents(root, [DOCUMENT], extractor=extractor(adapter)), 0)
            report = read_reports(root)[0]
            self.assertEqual(report.status, RunStatus.COMPLETED)
            self.assertEqual((report.observations_returned, report.observations_valid,
                              report.observations_repaired, report.observations_rejected), (4, 2, 1, 1))
            self.assertEqual((report.duplicates_removed, report.observations_persisted), (1, 2))
            self.assertEqual((report.input_tokens, report.output_tokens), (40, 80))
            self.assertEqual(len(report.warnings), 2)
            lines = run_jsonls(root)[0].read_text().splitlines()
            records = [ImplementationObservation.model_validate_json(line) for line in lines]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r.evidence.spans[0].quote == QUOTE for r in records))
            self.assertEqual({r.observation_type.value for r in records}, {'condition', 'action'})
            self.assertEqual(report.observations_returned,
                             report.observations_valid + report.observations_repaired + report.observations_rejected)
            self.assertEqual(report.observations_persisted,
                             report.observations_valid + report.observations_repaired - report.duplicates_removed)

    async def test_repaired_date_is_persisted_as_null(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            write_pdf(root)
            item = observation(evidence={'spans': [span(evidence_date='2020-10-05')]})
            await run_documents(root, [DOCUMENT], extractor=extractor(FakeAdapter(items=[item])))
            record = ImplementationObservation.model_validate_json(run_jsonls(root)[0].read_text())
            self.assertIsNone(record.evidence.spans[0].evidence_date)
            warning = read_reports(root)[0].warnings[0]
            self.assertEqual((warning.code, warning.action),
                             ('unsupported_evidence_date', 'evidence_date_set_to_null'))

    async def test_partial_run_continues_after_provider_and_schema_failures(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            root = Path(directory)
            write_pdf(root, texts=(QUOTE, QUOTE, QUOTE))
            adapter = FakeAdapter(responses=[RuntimeError('provider unavailable'),
                                            {'observations': [{'observation': 'invalid schema'}]},
                                            [observation(evidence={'spans': [span(page=3)]})]],
                                  usages=[None, None, TokenUsage(input_tokens=30, output_tokens=60)])
            self.assertEqual(await run_documents(root, [DOCUMENT], extractor=extractor(adapter),
                                                 max_chars=1, overlap_chars=0), 1)
            report = read_reports(root)[0]
            self.assertEqual(report.status, RunStatus.PARTIAL)
            self.assertEqual((report.chunks_total, report.chunks_succeeded, report.chunks_failed), (3, 1, 2))
            self.assertEqual([f.error_type for f in report.chunk_failures], ['RuntimeError', 'ValidationError'])
            self.assertEqual((report.input_tokens, report.output_tokens), (30, 60))
            self.assertIsNone(report.jsonl_file)
            self.assertEqual(report.observations_persisted, 1)
            self.assertFalse(run_jsonls(root))
            self.assertEqual(len(adapter.calls), 3)

    async def test_failed_when_no_chunk_succeeds(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            root = Path(directory)
            write_pdf(root, texts=(QUOTE, QUOTE))
            adapter = FakeAdapter(error=RuntimeError('provider failure'))
            await run_documents(root, [DOCUMENT], extractor=extractor(adapter), max_chars=1)
            report = read_reports(root)[0]
            self.assertEqual((report.status, report.chunks_succeeded, report.chunks_failed), (RunStatus.FAILED, 0, 2))
            self.assertIsNone(report.jsonl_file)
            self.assertIsNone(report.input_tokens)
            self.assertFalse(run_jsonls(root))

    async def test_empty_or_all_rejected_observations_still_cover_document(self):
        for items in ([], [observation(evidence={'spans': [span('Invented')]})]):
            with self.subTest(items=len(items)), TemporaryDirectory() as directory, redirect_stdout(StringIO()):
                root = Path(directory)
                write_pdf(root)
                await run_documents(root, [DOCUMENT], extractor=extractor(FakeAdapter(items=items)))
                report = read_reports(root)[0]
                self.assertEqual(report.status, RunStatus.COMPLETED)
                self.assertEqual(report.observations_persisted, 0)
                self.assertEqual(run_jsonls(root)[0].read_text(), '')

    async def test_successful_call_usage_aggregation_and_unknown_counts(self):
        for usages, expected in (([TokenUsage(input_tokens=10, output_tokens=20),
                                  TokenUsage(input_tokens=30, output_tokens=40)], (40, 60)),
                                 ([TokenUsage(input_tokens=10, output_tokens=20), None], (None, None)),
                                 ([None, TokenUsage(input_tokens=30, output_tokens=40)], (None, None))):
            with self.subTest(expected=expected), TemporaryDirectory() as directory, redirect_stdout(StringIO()):
                root = Path(directory)
                write_pdf(root, texts=(QUOTE, QUOTE))
                adapter = FakeAdapter(items=[], usages=usages)
                await run_documents(root, [DOCUMENT], extractor=extractor(adapter), max_chars=1)
                report = read_reports(root)[0]
                self.assertEqual((report.input_tokens, report.output_tokens), expected)

    async def test_jsonl_failure_preserves_completed_run_and_next_document_continues(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            root = Path(directory)
            missing = select_documents(document='MX_ISR05_2014')[0]
            write_pdf(root)
            await run_documents(root, [missing, DOCUMENT], extractor=extractor())
            self.assertEqual([r.status for r in read_reports(root)], [RunStatus.FAILED, RunStatus.COMPLETED])
            with patch('digital_applications_prototype.extraction.cli.write_observations', side_effect=OSError('disk full')):
                self.assertEqual(await run_documents(root, [DOCUMENT], extractor=extractor()), 1)
            report = read_reports(root)[-1]
            self.assertEqual(report.status, RunStatus.COMPLETED)
            self.assertIsNone(report.jsonl_file)
            self.assertIsNone(report.error)
            self.assertEqual(report.observations_persisted, 1)
            self.assertEqual(len(run_jsonls(root)), 1)

    async def test_report_failure_does_not_relabel_published_extraction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_pdf(root)
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err), patch(
                'digital_applications_prototype.extraction.cli.write_report', side_effect=OSError('report disk error')
            ):
                status = await run_documents(root, [DOCUMENT], extractor=extractor())
            self.assertEqual(status, 1)
            self.assertEqual(len(run_jsonls(root)), 1)
            self.assertFalse(read_reports(root))
            self.assertIn('Completed KM_ISR07_2023', out.getvalue())
            self.assertIn('report: NOT WRITTEN', out.getvalue())
            self.assertIn('extraction status remains completed', err.getvalue())
            self.assertNotIn('Failed KM_ISR07_2023', err.getvalue())

    async def test_display_failure_preserves_completed_report(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()) as err, patch(
            'digital_applications_prototype.extraction.cli._summary', side_effect=RuntimeError('display error')
        ):
            root = Path(directory)
            write_pdf(root)
            self.assertEqual(await run_documents(root, [DOCUMENT], extractor=extractor()), 1)
            self.assertEqual(read_reports(root)[0].status, RunStatus.COMPLETED)
            self.assertEqual(len(run_jsonls(root)), 1)
            self.assertIn('extraction status remains completed', err.getvalue())


class PersistenceAndCLITests(unittest.TestCase):
    def test_jsonl_bundle_round_trip_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = reserve_run(root, DOCUMENT, datetime.now(UTC))
            evidence = EvidenceBundle(summary='Dated context.', spans=[
                span('Actual Date: May 13, 2022', evidence_date='2022-05-13', role='qualifies'), span()])
            chunk = make_chunk('Actual Date: May 13, 2022\n' + QUOTE)
            item = validated(observation(evidence=evidence), chunk)
            write_observations(paths.jsonl_path, [item])
            self.assertEqual(ImplementationObservation.model_validate_json(paths.jsonl_path.read_text()), item)
            original = paths.jsonl_path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_observations(paths.jsonl_path, [])
            self.assertEqual(paths.jsonl_path.read_bytes(), original)
            report = RunReport(document_id=DOCUMENT.document_id, run_id=paths.run_id, started_at=datetime.now(UTC))
            write_report(paths.report_path, report)
            with self.assertRaises(FileExistsError):
                write_report(paths.report_path, report)
            paths.release()

    def test_colliding_run_ids_are_retried_without_overwrite(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            started = datetime(2026, 11, 1, 20, 34, 56, tzinfo=UTC)
            with patch('digital_applications_prototype.extraction.persistence.uuid4', return_value=SimpleNamespace(hex='a' * 32)):
                first = reserve_run(root, DOCUMENT, started)
            write_observations(first.jsonl_path, [])
            first.release()
            with patch('digital_applications_prototype.extraction.persistence.uuid4', side_effect=[
                SimpleNamespace(hex='a' * 32), SimpleNamespace(hex='b' * 32),
                SimpleNamespace(hex='b' * 32), SimpleNamespace(hex='c' * 32),
            ]):
                second = reserve_run(root, DOCUMENT, started)
                third = reserve_run(root, DOCUMENT, started)
            self.assertEqual(len({first.run_id, second.run_id, third.run_id}), 3)
            self.assertEqual(first.jsonl_path.read_text(), '')
            second.release()
            third.release()

    def test_atomic_write_failure_never_exposes_half_file(self):
        with TemporaryDirectory() as directory:
            paths = reserve_run(Path(directory), DOCUMENT, datetime.now(UTC))
            item = validated(observation(), make_chunk())
            def fail_replace(temporary, destination):
                self.assertFalse(destination.exists())
                self.assertEqual(temporary.suffix, '.part')
                self.assertIn(QUOTE, temporary.read_text())
                raise OSError('rename failed')
            with patch.object(Path, 'replace', fail_replace), self.assertRaises(OSError):
                write_observations(paths.jsonl_path, [item])
            self.assertFalse(paths.jsonl_path.exists())
            self.assertFalse(list(paths.jsonl_path.parent.glob('*.part')))
            paths.release()

    def test_dry_run_needs_no_settings_calls_or_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_pdf(root, texts=(QUOTE, 'New evidence.'))
            output = StringIO()
            with patch.dict('os.environ', {}, clear=True), redirect_stdout(output), \
                 patch('digital_applications_prototype.core.llm_runtime.configured_llm') as runtime, \
                 patch.object(LLMService, 'generate_structured') as generate:
                status = main(['--document', DOCUMENT.document_id, '--dry-run', '--root', str(root),
                               '--max-chars', str(len(QUOTE)), '--overlap-chars', '10'])
            self.assertEqual(status, 0)
            runtime.assert_not_called()
            generate.assert_not_called()
            self.assertIn('Chunk 2: pages 1–2; full pages 2–2; overlap', output.getvalue())
            self.assertIn('Prompt 2:', output.getvalue())
            self.assertNotIn(QUOTE, output.getvalue())
            self.assertFalse((root / 'data/derived').exists())

    def test_selected_invocation_is_live_and_reruns_without_a_flag(self):
        adapter = FakeAdapter()
        @asynccontextmanager
        async def fake_runtime():
            yield LLMService(adapter)
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), patch(
            'digital_applications_prototype.core.llm_runtime.configured_llm', fake_runtime
        ):
            root = Path(directory)
            write_pdf(root)
            args = ['--document', DOCUMENT.document_id, '--root', str(root)]
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args), 0)
            self.assertEqual(len(adapter.calls), 2)
            self.assertEqual(len(run_jsonls(root)), 2)
            self.assertEqual(len(read_reports(root)), 2)

    def test_setup_failure_gets_report_without_reading_real_environment(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()), patch(
            'digital_applications_prototype.core.llm_runtime.configured_llm'
        ) as runtime:
            root = Path(directory)
            write_pdf(root)
            runtime.return_value.__aenter__.side_effect = ValueError('missing configured model')
            self.assertEqual(main(['--document', DOCUMENT.document_id, '--root', str(root)]), 1)
            self.assertEqual(read_reports(root)[0].status, RunStatus.FAILED)
            self.assertIn('missing configured model', read_reports(root)[0].error)
            self.assertFalse(run_jsonls(root))

    def test_selection_required_and_removed_force_is_rejected(self):
        for args in ([], ['--all', '--document', DOCUMENT.document_id], ['--country', 'comoros', '--dry-run'],
                     ['--all', '--overlap-chars', '-1'], ['--document', DOCUMENT.document_id, '--force']):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(args)
