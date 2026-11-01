"""Run selected input documents live, or inspect one PDF with --dry-run."""

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path
import sys

from digital_applications_prototype.llm.prompt_renderer import PromptRenderer

from .chunking import chunk_pages
from .manifest import select_documents
from .models import (
    ChunkFailure, DocumentChunk, DocumentEntry, ImplementationObservation, PageText, RunReport, RunStatus,
)
from .persistence import RunArtifacts, reserve_run, write_observations, write_report
from .pdf import extract_pages
from .service import ImplementationObservationExtractor, render_prompt
from .validation import deduplicate_observations


def _load_chunks(root: Path, document: DocumentEntry, max_chars: int,
                 overlap_chars: int) -> tuple[list[PageText], list[DocumentChunk]]:
    """Extract numbered input pages and require at least one nonempty chunk."""
    pages = extract_pages(root, document)
    chunks = chunk_pages(document, pages, max_chars=max_chars, overlap_chars=overlap_chars)
    if not chunks:
        raise ValueError("No nonempty text chunks")
    return pages, chunks


def _inspect_document(root: Path, document: DocumentEntry, max_chars: int, overlap_chars: int) -> None:
    """Render every chunk for inspection without configuring a model or writing artifacts."""
    pages, chunks = _load_chunks(root, document, max_chars, overlap_chars)
    renderer = PromptRenderer()
    print(document.model_dump_json())
    print(f"PDF pages: {len(pages)}; chunks: {len(chunks)}")
    for index, chunk in enumerate(chunks, start=1):
        prompt = render_prompt(renderer, chunk)
        overlap_size = len(chunk.overlap.text) if chunk.overlap is not None else 0
        print(f"Chunk {index}: pages {chunk.start_page}–{chunk.end_page}; "
              f"full pages {chunk.pages[0].page_number}–{chunk.end_page}; overlap {overlap_size} characters")
        print(f"Prompt {index}: {len(prompt):,} characters; "
              f"~{(len(prompt) + 3) // 4:,} tokens (rough characters/4 estimate)")
    print("Dry-run: no model call and no output written")


async def _extract_chunks(extractor: ImplementationObservationExtractor,
                          chunks: list[DocumentChunk], report: RunReport) -> list[ImplementationObservation]:
    """Accumulate usable claims and successful-call usage while recording failed chunks.

    An unavailable usage field makes that aggregate null, rather than presenting
    a partial sum as a complete count. Failed calls are excluded from usage totals.
    """
    report.model = extractor.llm.model
    observations = []
    for index, chunk in enumerate(chunks, start=1):
        try:
            result = await extractor.extract(chunk)
        except Exception as error:
            report.chunks_failed += 1
            report.chunk_failures.append(ChunkFailure(
                chunk_index=index, chunk_start_page=chunk.start_page, chunk_end_page=chunk.end_page,
                error_type=type(error).__name__, message=str(error)))
            continue
        report.chunks_succeeded += 1
        for field in ("observations_returned", "observations_valid", "observations_repaired", "observations_rejected"):
            setattr(report, field, getattr(report, field) + getattr(result, field))
        report.warnings.extend(result.warnings)
        observations.extend(result.observations)
        for field in ("input_tokens", "output_tokens"):
            value = getattr(result.usage, field) if result.usage is not None else None
            previous = getattr(report, field)
            total = value if report.chunks_succeeded == 1 else (
                previous + value if previous is not None and value is not None else None)
            setattr(report, field, total)
    return observations


def _summary(report: RunReport, artifacts: RunArtifacts, report_written: bool) -> str:
    """Describe extraction status separately from report publication and console errors."""
    lines = [f"{report.status.value.capitalize()} {report.document_id}",
             f"  run: {report.run_id}", f"  status: {report.status.value}",
             f"  chunks: {report.chunks_succeeded}/{report.chunks_total}",
             f"  observations: {report.observations_persisted} persisted",
             f"  repaired: {report.observations_repaired}; rejected: {report.observations_rejected}",
             f"  duplicates removed: {report.duplicates_removed}", f"  warnings: {len(report.warnings)}"]
    lines.append(f"  output: {artifacts.jsonl_path}" if report.jsonl_file else "  no downstream-ready JSONL written")
    lines.append(f"  report: {artifacts.report_path}" if report_written else "  report: NOT WRITTEN")
    if report.error:
        lines.append(f"  error: {report.error}")
    return "\n".join(lines)


def _diagnostic(message: str) -> None:
    """Best-effort operational reporting without changing an already committed outcome."""
    try:
        print(message, file=sys.stderr)
    except Exception:
        pass


async def run_documents(
    root: Path, documents: list[DocumentEntry], *,
    extractor: ImplementationObservationExtractor | None = None,
    dry_run: bool = False, max_chars: int = 12000, overlap_chars: int = 600,
) -> int:
    """Create one immutable report per live attempt and JSONL only for full coverage.

    Continue after chunk and document failures. Observation warnings do not change
    coverage status. Report/display errors return nonzero without relabeling a
    successfully published extraction as failed. Dry-run creates no run artifacts.
    """
    if dry_run and len(documents) != 1:
        raise ValueError("Dry-run requires one document; use --document")
    unsuccessful = False
    for document in documents:
        if dry_run:
            try:
                _inspect_document(root, document, max_chars, overlap_chars)
            except Exception as error:
                unsuccessful = True
                _diagnostic(f"Dry-run failed {document.document_id}: {error}")
            continue
        started_at = datetime.now(UTC)
        try:
            artifacts = reserve_run(root, document, started_at)
        except Exception as error:
            unsuccessful = True
            _diagnostic(f"Failed {document.document_id}: cannot reserve run artifacts; no report written: {error}")
            continue
        report = RunReport(document_id=document.document_id, run_id=artifacts.run_id, started_at=started_at,
                           model=extractor.llm.model if extractor is not None else None)
        try:
            _, chunks = _load_chunks(root, document, max_chars, overlap_chars)
            report.chunks_total = len(chunks)
            if extractor is None:
                from digital_applications_prototype.core.llm_runtime import configured_llm

                async with configured_llm() as llm:
                    service = ImplementationObservationExtractor(PromptRenderer(), llm)
                    observations = await _extract_chunks(service, chunks, report)
            else:
                observations = await _extract_chunks(extractor, chunks, report)
            unique = deduplicate_observations(observations)
            report.duplicates_removed = len(observations) - len(unique)
            if report.chunks_succeeded == report.chunks_total:
                write_observations(artifacts.jsonl_path, unique)
                report.jsonl_file = artifacts.jsonl_path.name
                report.observations_persisted = len(unique)
                report.status = RunStatus.COMPLETED
            elif report.chunks_succeeded:
                report.status = RunStatus.PARTIAL
        except Exception as error:
            report.status = RunStatus.FAILED
            report.error = f"{type(error).__name__}: {error}"
        report.finished_at = datetime.now(UTC)
        unsuccessful |= report.status != RunStatus.COMPLETED
        report_written = False
        try:
            write_report(artifacts.report_path, report)
            report_written = True
        except Exception as error:
            unsuccessful = True
            _diagnostic(f"Run {report.run_id} extraction status remains {report.status.value}; "
                        f"report NOT WRITTEN: {error}")
        try:
            print(_summary(report, artifacts, report_written),
                  file=sys.stdout if report.status == RunStatus.COMPLETED else sys.stderr)
        except Exception as error:
            unsuccessful = True
            _diagnostic(f"Run {report.run_id} extraction status remains {report.status.value}; "
                        f"console summary failed: {error}")
        try:
            artifacts.release()
        except OSError as error:
            unsuccessful = True
            _diagnostic(f"Run {report.run_id} reservation cleanup failed: {error}")
    return int(unsuccessful)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--document", help="Exact input document ID, e.g. KM_ISR07_2023")
    selection.add_argument("--country", choices=["mexico", "comoros", "sierra-leone"])
    selection.add_argument("--all", action="store_true", dest="all_documents")
    parser.add_argument("--dry-run", action="store_true", help="Inspect one PDF without calling the model")
    parser.add_argument("--max-chars", type=int, default=12000, help="Soft chunk character target")
    parser.add_argument("--overlap-chars", type=int, default=600, help="Trailing context limit (0 disables)")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current directory)")
    args = parser.parse_args(argv)
    if args.max_chars < 1:
        parser.error("--max-chars must be positive")
    if args.overlap_chars < 0:
        parser.error("--overlap-chars must be nonnegative")
    try:
        documents = select_documents(document=args.document, country=args.country,
                                     all_documents=args.all_documents)
        if args.dry_run and len(documents) != 1:
            parser.error("--dry-run requires a single --document selection")
        return asyncio.run(run_documents(args.root, documents, dry_run=args.dry_run,
                                         max_chars=args.max_chars, overlap_chars=args.overlap_chars))
    except Exception as error:
        _diagnostic(f"Stage 1 failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
