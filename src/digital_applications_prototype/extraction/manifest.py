"""Explicit input-only corpus metadata. No discovery or holdout enumeration."""

from datetime import date
from pathlib import Path

from .models import DocumentEntry


def _entry(
    country: str, project: str, doc: str, kind: str, when: str | None
) -> DocumentEntry:
    return DocumentEntry(
        path=Path("data/input") / country / f"{doc}.pdf",
        project_id=project,
        country=country,
        document_id=doc,
        document_type=kind,
        document_date=date.fromisoformat(when) if when else None,
    )


# Dates are cover dates for PAD/AF papers and archive dates for ISRs.
# Restructuring cover pages do not establish dates; unknown is intentional.
INPUT_DOCUMENTS = (
    _entry("mexico", "P123367", "MX_PAD_2011", "PAD", "2011-10-31"),
    _entry("mexico", "P123367", "MX_ISR05_2014", "ISR", "2014-07-04"),
    _entry("mexico", "P123367", "MX_ISR08_2016", "ISR", "2016-04-04"),
    _entry("mexico", "P123367", "MX_ISR11_2017", "ISR", "2017-07-20"),
    _entry("comoros", "P166193", "KM_PAD_2020", "PAD", "2020-06-17"),
    _entry("comoros", "P166193", "KM_ISR03_2021", "ISR", "2021-11-11"),
    _entry("comoros", "P166193", "KM_ISR07_2023", "ISR", "2023-09-11"),
    _entry("comoros", "P166193", "KM_RESTRUCTURING_2024", "RESTRUCTURING", None),
    _entry("comoros", "P166193", "KM_ISR10_2025", "ISR", "2025-03-31"),
    _entry("sierra-leone", "P143588", "SL_PAD_2014", "PAD", "2014-02-28"),
    _entry("sierra-leone", "P143588", "SL_ISR08_2018", "ISR", "2018-09-06"),
    _entry("sierra-leone", "P143588", "SL_AF2_2019", "AF", "2019-06-10"),
    _entry("sierra-leone", "P143588", "SL_ISR14_2021", "ISR", "2021-12-21"),
    _entry("sierra-leone", "P143588", "SL_RESTRUCTURING_2023", "RESTRUCTURING", None),
    _entry("sierra-leone", "P143588", "SL_ISR17_2023", "ISR", "2023-06-26"),
)


def select_documents(
    *,
    document: str | None = None,
    country: str | None = None,
    all_documents: bool = False,
) -> list[DocumentEntry]:
    """Select manifest entries in order using exactly one selection mode.

    Raise ValueError for ambiguous selections or selections with no matches.
    """
    if sum((document is not None, country is not None, all_documents)) != 1:
        raise ValueError("Select exactly one of --document, --country, or --all")
    selected = [
        entry
        for entry in INPUT_DOCUMENTS
        if all_documents
        or (document is not None and entry.document_id == document)
        or (country is not None and entry.country == country)
    ]
    if not selected:
        raise ValueError("No input documents match the selection")
    return selected


def input_path(root: Path, relative: Path) -> Path:
    """Reject traversal and symlinks escaping the input tree before opening a PDF."""
    root = root.resolve()
    allowed = root / "data" / "input"
    if allowed.resolve() != allowed:
        raise ValueError("data/input must not resolve outside its declared location")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(allowed) or candidate.suffix.lower() != ".pdf":
        raise ValueError(f"Stage 1 only reads PDFs under data/input: {relative}")
    return candidate
