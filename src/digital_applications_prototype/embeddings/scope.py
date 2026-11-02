"""Explicit run IDs or a small, versionable text manifest; no inferred corpus."""

import re
from collections.abc import Sequence
from pathlib import Path


def normalize_run_ids(run_ids: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for value in run_ids:
        run_id = value.strip()
        # Run IDs are opaque identifiers. Reject delimiters/whitespace rather than
        # assuming a timestamp format that the SQLite schema does not require.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", run_id):
            raise ValueError(f"Malformed extraction run ID: {value!r}")
        if run_id not in normalized:
            normalized.append(run_id)
    if not normalized:
        raise ValueError("The run selection is empty")
    return normalized


def read_run_ids(path: Path) -> list[str]:
    values: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            values.extend(normalize_run_ids([line]))
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return normalize_run_ids(values)
