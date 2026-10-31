"""A complete, atomically replaced JSONL file is the document completion marker."""

from pathlib import Path
import tempfile

from .models import DocumentEntry, ImplementationObservation


def output_path(root: Path, document: DocumentEntry) -> Path:
    path = root.resolve() / "data/derived/extracted" / document.country / f"{document.document_id}.jsonl"
    if path.resolve() != path:
        raise ValueError("Extraction output must not traverse symlinks")
    return path


def write_observations(path: Path, observations: list[ImplementationObservation]) -> None:
    """Call only after every chunk succeeds; even zero observations is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".part", delete=False) as target:
            temporary = Path(target.name)
            for item in observations:
                target.write(item.model_dump_json() + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
