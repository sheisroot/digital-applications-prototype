"""Demo-only presentation labels. Not canonical taxonomy categories.

Run --sample first and inspect its printed pairs before invoking without --sample.
No web lookup, extraction, embedding, or canonical database access is performed.
"""

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from digital_applications_prototype.core.config import Settings
from digital_applications_prototype.explorer.demo_factor_labels import (
    demo_database,
    generate_demo_factor_labels,
    pending_items,
    quality_report,
    refresh_snapshot_report,
    representative_ids,
)
from digital_applications_prototype.llm.adapters.openai import OpenAIAdapter
from digital_applications_prototype.llm.service import LLMService

ROOT = Path(__file__).resolve().parents[1]


async def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "demo/evidence.sqlite")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    opened = False
    try:
        with demo_database(args.snapshot) as connection:
            opened = True
            items = pending_items(connection)
            sample_ids = representative_ids(connection)
            if args.sample:
                items = [item for item in items if item.observation_id in sample_ids]
            if not args.report_only and items:
                settings = Settings.from_env()
                # Disable hidden SDK retries so each audited attempt is one API request.
                async with AsyncOpenAI(
                    api_key=settings.openai_api_key.get_secret_value(),
                    max_retries=0,
                    timeout=120,
                ) as client:
                    await generate_demo_factor_labels(
                        connection,
                        LLMService(OpenAIAdapter(client, model=settings.llm_model)),
                        items,
                        concurrency=args.concurrency,
                    )
            report = quality_report(connection)
            args.snapshot.with_name("demo_factor_labels_report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(json.dumps(report, indent=2))
            if args.sample:
                for row in connection.execute(
                    "SELECT observation_id, factor_label, factor_description, country, "
                    "observation_type, claim_scope FROM observations "
                    "JOIN documents USING (document_id) ORDER BY observation_id"
                ):
                    if row[0] in sample_ids:
                        print(json.dumps(dict(row), ensure_ascii=False))
    finally:
        if opened:
            refresh_snapshot_report(args.snapshot)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
