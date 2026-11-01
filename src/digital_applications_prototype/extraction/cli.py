"""Run selected input documents live, or inspect one PDF with --dry-run."""

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path
import sys

from digital_applications_prototype.llm.prompt_renderer import PromptRenderer
from digital_applications_prototype.persistence.database import initialize_database
from digital_applications_prototype.persistence.repository import ExtractionRepository

from .chunking import chunk_pages
from .manifest import select_documents
from .models import (
    DocumentChunk, DocumentEntry, PageText, RunReport, RunStatus,
)
from .persistence import RunArtifacts, reserve_run, write_observations, write_report
from .pdf import extract_pages
from .service import ImplementationObservationExtractor, render_prompt


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
                          chunks: list[DocumentChunk], chunk_ids: list[int],
                          repository: ExtractionRepository, run_id: str) -> None:
    """Persist each result before spending on the next sequential model call."""
    await repository.set_run_model(run_id, extractor.llm.model)
    for chunk_id, chunk in zip(chunk_ids, chunks, strict=True):
        await repository.mark_chunk_running(chunk_id)
        try:
            result = await extractor.extract(chunk)
        except Exception as error:
            await repository.mark_chunk_failed(chunk_id, error)
            continue
        # Database failures abort orchestration rather than spending on more calls.
        await repository.persist_completed_chunk(chunk_id, chunk, result)


def _summary(report: RunReport, artifacts: RunArtifacts | None, report_written: bool) -> str:
    """Describe extraction status separately from report publication and console errors."""
    lines = [f"{report.status.value.capitalize()} {report.document_id}",
             f"  run: {report.run_id}", f"  status: {report.status.value}",
             f"  chunks: {report.chunks_succeeded}/{report.chunks_total} succeeded; {report.chunks_failed} failed",
             f"  observations: {report.observations_returned} returned; {report.observations_persisted} clean",
             f"  valid: {report.observations_valid}; repaired: {report.observations_repaired}; "
             f"rejected: {report.observations_rejected}",
             f"  duplicates marked: {report.duplicates_removed}", f"  warnings: {len(report.warnings)}",
             f"  tokens: input={report.input_tokens}; output={report.output_tokens}",
             f"  model: {report.model}; schema: {report.schema_version}; concurrency: {report.max_concurrency}"]
    if report.finished_at is not None:
        lines.append(f"  elapsed: {(report.finished_at - report.started_at).total_seconds():.2f}s")
    lines.append(f"  output: {artifacts.jsonl_path}" if artifacts and report.jsonl_file
                 else "  no downstream-ready JSONL written")
    lines.append(f"  report: {artifacts.report_path}" if artifacts and report_written else "  report: NOT WRITTEN")
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
    export_files: bool = True,
) -> int:
    """SQLite is canonical; optional exports never determine extraction status."""
    if dry_run and len(documents) != 1:
        raise ValueError("Dry-run requires one document; use --document")
    if dry_run:
        try:
            _inspect_document(root, documents[0], max_chars, overlap_chars)
            return 0
        except Exception as error:
            _diagnostic(f"Dry-run failed {documents[0].document_id}: {error}")
            return 1

    database = await initialize_database(root / "data/derived/stage1.sqlite")
    repository = ExtractionRepository(database.sessions)
    unsuccessful = False
    try:
        for document in documents:
            artifacts = None
            run_id = await repository.create_run(
                document.document_id, model=extractor.llm.model if extractor else None)
            if export_files:
                try:
                    artifacts = reserve_run(root, document, datetime.now(UTC), run_id=run_id)
                except Exception as error:
                    unsuccessful = True
                    _diagnostic(f"Run {run_id}: cannot reserve optional exports: {error}")
            run_error = None
            try:
                _, chunks = _load_chunks(root, document, max_chars, overlap_chars)
                chunk_ids = await repository.create_chunks(run_id, chunks)
                if extractor is None:
                    from digital_applications_prototype.core.llm_runtime import configured_llm

                    async with configured_llm() as llm:
                        service = ImplementationObservationExtractor(PromptRenderer(), llm)
                        await _extract_chunks(service, chunks, chunk_ids, repository, run_id)
                else:
                    await _extract_chunks(extractor, chunks, chunk_ids, repository, run_id)
            except Exception as error:
                run_error = f"{type(error).__name__}: {error}"
            try:
                await repository.mark_duplicates(run_id)
                await repository.finalize_run(run_id, error=run_error)
                report = await repository.get_run_summary(run_id)
            except Exception as error:
                unsuccessful = True
                _diagnostic(f"Run {run_id}: database finalization failed; committed chunks remain stored: {error}")
                if artifacts is not None:
                    artifacts.release()
                continue
            unsuccessful |= report.status != RunStatus.COMPLETED
            report_written = False
            if artifacts is not None:
                if report.status == RunStatus.COMPLETED:
                    try:
                        observations = await repository.get_clean_observations(run_id)
                        write_observations(artifacts.jsonl_path, observations)
                        report.jsonl_file = artifacts.jsonl_path.name
                    except Exception as error:
                        unsuccessful = True
                        _diagnostic(f"Run {run_id} extraction status remains {report.status.value}; "
                                    f"JSONL export failed: {error}")
                try:
                    write_report(artifacts.report_path, report)
                    report_written = True
                except Exception as error:
                    unsuccessful = True
                    _diagnostic(f"Run {run_id} extraction status remains {report.status.value}; "
                                f"report NOT WRITTEN: {error}")
            try:
                print(_summary(report, artifacts, report_written),
                      file=sys.stdout if report.status == RunStatus.COMPLETED else sys.stderr)
            except Exception as error:
                unsuccessful = True
                _diagnostic(f"Run {run_id} extraction status remains {report.status.value}; "
                            f"console summary failed: {error}")
            if artifacts is not None:
                try:
                    artifacts.release()
                except OSError as error:
                    unsuccessful = True
                    _diagnostic(f"Run {run_id} reservation cleanup failed: {error}")
    finally:
        await database.close()
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
    parser.add_argument("--no-export", action="store_true", help="Keep results only in SQLite")
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
                                         max_chars=args.max_chars, overlap_chars=args.overlap_chars,
                                         export_files=not args.no_export))
    except Exception as error:
        _diagnostic(f"Stage 1 failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
