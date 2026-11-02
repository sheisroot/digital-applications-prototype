"""Offline export; run as `uv run python -m scripts.freeze_evidence_demo`."""

import argparse
import json
from pathlib import Path

from digital_applications_prototype.embeddings.scope import read_run_ids
from digital_applications_prototype.explorer.snapshot import (
    create_snapshot,
    source_url_mapping,
)
from scripts.download_pilot_corpus import MANIFEST

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/derived/stage1.sqlite"
    )
    parser.add_argument("--runs", type=Path, default=ROOT / "demo/selected_runs.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "demo/evidence.sqlite")
    args = parser.parse_args()
    report = create_snapshot(
        args.source,
        args.output,
        read_run_ids(args.runs),
        source_url_mapping(MANIFEST),
        expected_documents=15,
        expected_observations=1093,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
