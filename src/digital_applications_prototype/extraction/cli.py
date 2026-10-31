"""Input-only Stage 1 orchestration; dry runs never construct an SDK client."""

import argparse
import asyncio
from pathlib import Path
import sys

from digital_applications_prototype.llm.prompt_renderer import PromptRenderer

from .chunking import chunk_pages
from .manifest import select_documents
from .models import DocumentEntry
from .pdf import extract_pages
from .persistence import output_path, write_observations
from .service import ImplementationObservationExtractor, render_prompt


async def run_documents(
    root: Path, documents: list[DocumentEntry], *,
    extractor: ImplementationObservationExtractor | None = None,
    dry_run: bool = False, force: bool = False, max_chars: int = 12000,
) -> int:
    if dry_run and len(documents) != 1:
        raise ValueError("Dry-run requires one document; use --document")
    failed = 0
    renderer = PromptRenderer()
    for document in documents:
        try:
            destination = output_path(root, document)
            if destination.exists() and not force and not dry_run:
                print(f"Skipping completed {document.document_id}")
                continue
            pages = extract_pages(root, document)
            chunks = chunk_pages(document, pages, max_chars=max_chars)
            if not chunks:
                raise ValueError("No nonempty text chunks")
            if dry_run:
                prompt = render_prompt(renderer, chunks[0])
                print(document.model_dump_json())
                print(f"PDF pages: {len(pages)}; chunks: {len(chunks)}; "
                      f"first chunk pages: {chunks[0].start_page}–{chunks[0].end_page}")
                print(f"First prompt: {len(prompt):,} characters; "
                      f"~{(len(prompt) + 3) // 4:,} tokens (rough characters/4 estimate)")
                print("Dry-run: no model call and no output written")
                continue
            if extractor is None:
                raise ValueError("An extractor is required for model execution")
            observations = []
            for chunk in chunks:
                observations.extend(await extractor.extract(chunk))
            # No overlap: remove exact duplicate records without semantic merging.
            unique = {item.observation_id: item for item in observations}
            write_observations(destination, list(unique.values()))
            print(f"Completed {document.document_id}: {len(unique)} observations -> {destination}")
        except Exception as error:
            failed += 1
            print(f"Failed {document.document_id}: {error}", file=sys.stderr)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--document", help="Exact input document ID, e.g. KM_ISR07_2023")
    selection.add_argument("--country", choices=["mexico", "comoros", "sierra-leone"])
    selection.add_argument("--all", action="store_true", dest="all_documents")
    parser.add_argument("--force", action="store_true", help="Rerun completed documents")
    parser.add_argument("--dry-run", action="store_true", help="Inspect one PDF without calling the model")
    parser.add_argument("--max-chars", type=int, default=12000, help="Soft chunk character target")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current directory)")
    args = parser.parse_args(argv)
    if args.max_chars < 1:
        parser.error("--max-chars must be positive")
    try:
        documents = select_documents(document=args.document, country=args.country,
                                     all_documents=args.all_documents)
        if args.dry_run and len(documents) != 1:
            parser.error("--dry-run requires a single --document selection")
        if args.dry_run:
            return asyncio.run(run_documents(args.root, documents, dry_run=True, max_chars=args.max_chars))
        pending = []
        for document in documents:
            if not args.force and output_path(args.root, document).exists():
                print(f"Skipping completed {document.document_id}")
            else:
                pending.append(document)
        if not pending:
            return 0

        async def execute() -> int:
            from digital_applications_prototype.core.llm_runtime import configured_llm

            async with configured_llm() as llm:
                extractor = ImplementationObservationExtractor(PromptRenderer(), llm)
                return await run_documents(args.root, pending, extractor=extractor,
                                           force=args.force, max_chars=args.max_chars)

        return asyncio.run(execute())
    except Exception as error:
        print(f"Stage 1 failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
