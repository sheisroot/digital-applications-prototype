"""Immutable run destinations and atomic whole-file publication."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import tempfile
from uuid import uuid4

from .models import DocumentEntry, ImplementationObservation, RunReport


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    jsonl_path: Path
    report_path: Path
    reservation_path: Path

    def release(self) -> None:
        self.reservation_path.unlink(missing_ok=True)


def reserve_run(root: Path, document: DocumentEntry, started_at: datetime) -> RunArtifacts:
    """Reserve a unique timestamp/UUID artifact pair without touching earlier runs.

    An exclusive transient reservation prevents concurrent attempts from claiming
    the same pair. Existing artifacts or reservations cause another ID to be tried.
    The caller releases the reservation after its single final report attempt.
    """
    directory = root.resolve() / "data/derived/extracted" / document.country
    if directory.resolve() != directory:
        raise ValueError("Extraction output must not traverse symlinks")
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(10):
        run_id = f"{timestamp}_{uuid4().hex[:12]}"
        stem = f"{document.document_id}__{run_id}"
        reservation = directory / f".{stem}.reserve"
        try:
            with reservation.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            continue
        jsonl = directory / f"{stem}.jsonl"
        report = directory / f"{stem}.report.json"
        if any(path.exists() or path.is_symlink() for path in (jsonl, report)):
            reservation.unlink()
            continue
        return RunArtifacts(run_id, jsonl, report, reservation)
    raise FileExistsError("Could not reserve a distinct extraction run ID")


def _atomic_write(path: Path, text: str) -> None:
    """Publish a complete file via a temporary file and replacement under a run reservation.

    Refuse existing destinations and symlinks. The run reservation serializes all
    cooperating writers of this artifact pair; no placeholder final file is used.
    """
    if path.resolve() != path or path.exists():
        raise FileExistsError(f"Refusing to replace an existing or symlinked artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".part", delete=False) as target:
            temporary = Path(target.name)
            target.write(text)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Artifact already exists: {path}")
        temporary.replace(path)
        temporary = None  # Publication is complete; no fallible cleanup follows it.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_observations(path: Path, observations: list[ImplementationObservation]) -> None:
    """Atomically publish one completed run's JSONL, including an empty successful result."""
    _atomic_write(path, "".join(item.model_dump_json() + "\n" for item in observations))


def write_report(path: Path, report: RunReport) -> None:
    """Atomically publish the attempt's final report without replacing another report."""
    _atomic_write(path, report.model_dump_json(indent=2) + "\n")
