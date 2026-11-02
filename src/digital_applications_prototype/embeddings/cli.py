"""Embed explicitly selected, persisted Stage 1 observations."""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from digital_applications_prototype.core.config import EmbeddingSettings
from digital_applications_prototype.core.llm_runtime import configured_embeddings
from digital_applications_prototype.persistence.database import Database
from digital_applications_prototype.persistence.embeddings import EmbeddingRepository

from .models import DEFAULT_BATCH_SIZE, EmbeddingPlan, EmbeddingReport
from .scope import normalize_run_ids, read_run_ids
from .service import EmbeddingStageError, embed_observations


def print_plan(plan: EmbeddingPlan) -> None:
    print(f"Selected extraction runs: {len(plan.runs)}")
    print(f"Unique documents: {len(plan.documents)}")
    print(f"Eligible observations: {plan.eligible}")
    print(f"Already embedded/current: {plan.already_current}")
    print(f"Require embedding: {len(plan.pending)}")
    print(f"Model: {plan.model}")
    print(f"Representation: {plan.representation}")
    print(f"Batch size: {plan.batch_size}")
    print(f"Documents with multiple selected runs: {len(plan.conflicts)}")
    for run in plan.runs[:50]:
        print(
            f"  {run.run_id}  {run.document_id}  eligible: {run.eligible}  "
            f"current: {run.already_current}"
        )
    if len(plan.runs) > 50:
        print(f"  ... {len(plan.runs) - 50} additional runs")
    for document, ids in plan.conflicts.items():
        print(f"  CONFLICT {document}: {', '.join(ids)}")


def print_report(plan: EmbeddingPlan, report: EmbeddingReport) -> None:
    print(f"Submitted: {report.submitted}")
    print(f"Embedded successfully: {report.embedded}")
    print(f"Failed: {report.failed}")
    print(f"Not submitted: {len(plan.pending) - report.submitted}")
    print(f"Batches: {report.completed_batches} completed / {report.batches} attempted")
    print(f"Returned models: {', '.join(sorted(report.models)) or 'no new responses'}")
    print(
        "Dimensions: "
        + (
            ", ".join(str(value) for value in sorted(report.dimensions))
            or "unavailable"
        )
    )
    if report.usage_batches:
        print(
            f"Input tokens (provider-reported): {report.input_tokens} "
            f"({report.usage_batches} batches with usage)"
        )


async def run_embedding_stage(
    path: Path,
    *,
    run_ids: list[str] | None,
    all_runs: bool,
    model: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
    allow_multiple_runs_per_document: bool = False,
) -> int:
    if not path.is_file():
        raise ValueError(f"Stage 1 database does not exist: {path}")
    database = Database(path, read_only=True)
    try:
        plan = await EmbeddingRepository(database.sessions).plan(
            run_ids=run_ids, all_runs=all_runs, model=model, batch_size=batch_size
        )
    finally:
        await database.close()
    print_plan(plan)
    plan.validate(allow_multiple_runs_per_document=allow_multiple_runs_per_document)
    if dry_run:
        print("Dry run: no API calls or database changes.")
        return 0
    if not plan.pending:
        print_report(plan, EmbeddingReport(dimensions=set(plan.current_dimensions)))
        return 0
    # Credentials/client and writable storage are opened only after validation.
    async with configured_embeddings() as provider:
        database = Database(path)
        try:
            repository = EmbeddingRepository(database.sessions)
            await repository.initialize()
            try:
                report = await embed_observations(
                    repository,
                    provider,
                    plan,
                    allow_multiple_runs_per_document=allow_multiple_runs_per_document,
                )
            except EmbeddingStageError as error:
                print_report(plan, error.report)
                # Provider messages can contain credentials; report only the type.
                print(
                    f"{error} Cause: {type(error.__cause__).__name__}.",
                    file=sys.stderr,
                )
                return 1
        finally:
            await database.close()
    print_report(plan, report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--run-id", action="append", help="Exact run ID; repeat for a set"
    )
    selection.add_argument(
        "--run-ids-file",
        type=Path,
        help="One run ID per line; blank lines and # comment lines allowed",
    )
    selection.add_argument("--all", action="store_true", dest="all_runs")
    parser.add_argument(
        "--dry-run", action="store_true", help="Read-only scope and work preview"
    )
    parser.add_argument("--model", help="Override EMBEDDING_MODEL for this invocation")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--allow-multiple-runs-per-document", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--database",
        type=Path,
        help="SQLite path (default: <root>/data/derived/stage1.sqlite)",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 2048:
        parser.error("--batch-size must be between 1 and 2048")
    if args.model is not None and not args.model.strip():
        parser.error("--model must be nonempty")
    try:
        run_ids = (
            read_run_ids(args.run_ids_file)
            if args.run_ids_file is not None
            else normalize_run_ids(args.run_id)
            if args.run_id is not None
            else None
        )
        return asyncio.run(
            run_embedding_stage(
                args.database or args.root / "data/derived/stage1.sqlite",
                run_ids=run_ids,
                all_runs=args.all_runs,
                model=args.model
                if args.model is not None
                else EmbeddingSettings.from_env().embedding_model,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                allow_multiple_runs_per_document=args.allow_multiple_runs_per_document,
            )
        )
    except (ValueError, OSError, RuntimeError, SQLAlchemyError) as error:
        print(f"Embedding stage failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
