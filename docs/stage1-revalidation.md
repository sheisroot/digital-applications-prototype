# Stage 1 PDF text and offline revalidation

`uv run extract-observations revalidate --database PATH --document ID --run RUN`
defaults to a read-only preview. `--apply` creates a SQLite-safe online backup,
updates derived validation state transactionally, appends a validation audit, and
writes separate JSONL/report artifacts. No model is configured and no LLM/API
calls or new observation extraction occur.

## PDF extraction design and reproduced failures

Live extraction and PDF-backed revalidation share `extraction/pdf.py:load_pdf`.
The default backend is **PyMuPDF**, using `get_text("text", sort=False)` with
explicit `TEXTFLAGS_TEXT` flags. Physical PDF pages remain separate, numbered from
one. There is no OCR, cross-page search, table reconstruction, sorting, or automatic
dehyphenation. The loader hashes the same bytes it extracts and returns untouched
page text plus PDF SHA-256, extractor name/version, and extraction options.

The previous pypdf representation inserted artificial spaces inside words:
observation 9 quoted `reported` where pypdf produced `repo rted`; observation 88
quoted `available` where pypdf produced `availab le`. PyMuPDF extracts the intact
words from the same cited physical pages (14 and 24). The test suite reproduces
both real-page excerpts and the underlying extractor difference with tiny PDFs
generated in memory; no large binary fixture is added.

The shared quote matcher and validator version **`stage1-quotes-v2` are unchanged
by this PDF fix**. Supported normalization remains whitespace collapse, optional
spacing around an ASCII hyphen between ASCII letters, and optional spacing
before `,.;:!?` immediately after an ASCII letter. Every other character and word
boundary remains significant. Each complete span must match one representation
of its own cited page. Spans cannot mix pieces of different representations.
One failing span rejects the observation; evidence-date checks still apply.
There is no arbitrary word joining, fuzzy/semantic matching, or punctuation/number
removal. No extra normalization is added for the deferred cases below.

Dependencies follow the existing bounded ranges and uv lockfile:
`pymupdf>=1.26,<2`, currently locked/tested at **1.28.2**, alongside
`pypdf>=6,<7` (tested at **6.21.4**) for legacy constraint reconstruction and test
PDF generation. Installation succeeded with a native macOS x86-64 wheel on the
project's Python 3.12 environment. PyMuPDF is a native dependency, unlike pypdf;
platforms without a suitable wheel may need a build toolchain. Its
[AGPL/commercial licensing](https://pymupdf.readthedocs.io/en/latest/about.html)
is a material dependency constraint; the repository's MIT license is unchanged.
See the official [installation documentation](https://pymupdf.readthedocs.io/en/latest/installation.html).

## Source retention and historical constraints

New live runs retain the exact chunk pages and bounded overlap in the existing
`extraction_chunk_sources.source_json`. Its `pdf_source` metadata records the
PDF hash, extractor name/version, and options. This uses the existing JSON
facility; no schema migration or observation-schema change is needed. Older
snapshots without extractor metadata remain usable and are reported as unknown,
rather than labeled with the current extractor/version.

Offline revalidation prefers retained original text. Merely providing PDF
arguments does not replace retained sources. To explicitly add the improved
representation, use `--refresh-pdf-text` with the PDF path and expected hash.
Retained PDF fingerprints, when available, must also match. Historical text is
never overwritten.

For legacy runs without snapshots, the existing explicit expected PDF hash and
original `--max-chars`/`--overlap-chars` remain required. The new
`--historical-extractor pypdf` option (the default) reconstructs the old page/chunk
limits using the shared loader's historical backend. Its version/settings are
recorded in the audit. Reconstruction must reproduce the persisted chunk indexes
and ranges; failure requests retained sources or the original extractor/settings.
`--historical-extractor pymupdf` is available for a snapshot-less PyMuPDF run.

PyMuPDF text lengths can change future chunk boundaries. **Newly generated
PyMuPDF chunks are never used to reinterpret an existing run.** Each historical
chunk keeps its recorded full-page numbers. Its overlap remains a bounded window
on the original physical page. Unique exact normalized boundary anchors locate
the corresponding interval in the new page text; the interval must preserve the
entire original non-whitespace character sequence. This is a boundary proof only,
not a quote-matching rule: acceptance still uses the actual source text with word
spaces intact. Ambiguous anchors, reordered or changed interval content, missing
pages, or missing original full-page context fail clearly instead of widening
access to the whole page. A short overlap that originally exposed the whole page
continues to expose that page.

Every span checks the historical representation first, then the constrained
PyMuPDF representation. This matters: replacing pypdf text outright recovered the
25 targets but newly rejected 22 previously valid quotes containing pypdf spacing
or text-order artifacts. Keeping both verified representations preserves those
quotes without changing normalization. Legacy reconstructed text is explicitly
identified as a reconstruction, never as retained original model input.

`validation_revisions` records the PDF fingerprint, current extractor/version and
options, historical extractor/settings when reconstructed, source-text digest,
overlap projections, matching policy, validator version, UTC revalidation time,
before/after counts, and changed derived state. Raw model output, original quotes,
row/evidence IDs, historical source snapshots, extraction timestamps, chunk state,
and token totals remain intact. Duplicate membership uses the existing
first-in-source-order provenance signatures; no semantic deduplication is added.
Repeated apply appends an audit row but produces stable decisions and export
content. Original reports/JSONL are preserved; export failure after commit can be
retried with `--apply`.

## Saved-run verification

Document `KM_PAD_2020`; run
`20261101T223237Z_aaf4b4fdeb6f483cb55ed48ce9e96fe7`.

PDF SHA-256:
`3616a45b8bbd7af682bde14ae9cf55854dba2ab3034b9d7484df22bff76dfb2e`.

The input for this phase was the earlier revalidated database at
`data/derived/revalidation/KM_PAD_2020_quote_validation.sqlite`. A fresh online
backup at `data/derived/revalidation/KM_PAD_2020_pymupdf.sqlite` was revalidated
twice with model and network calls blocked. The user's input database, canonical
`data/derived/stage1.sqlite`, and original outputs were left unchanged.

| Measure | Before | After on copy |
| --- | ---: | ---: |
| Completed chunks | 17 | 17 |
| Returned observations | 196 | 196 |
| Valid | 163 | 188 |
| Repaired | 0 | 0 |
| Rejected | 33 | 8 |
| Provenance duplicates | 3 | 4 |
| Usable observations | 160 | 184 |

**All 25 regression targets recovered**, with no newly rejected observations:
`7, 9, 12, 20, 33, 35, 61, 63, 65, 69, 70, 72, 76, 84, 88, 125, 128, 130,
133, 144, 155, 163, 181, 188, 195`.

Rows 181 and 155 now share a provenance signature; row 155 is the later duplicate
in source chunk order and points to 181. Thus 25 recoveries add **24 usable rows**.
Remaining rejected IDs: **25, 52, 136, 137, 139, 187, 190, 194**. Each retains
`ungrounded_quote` as its implemented-contract failure. They are source-supported
cases with evidence-location/formatting limitations, not established hallucinations.

Input/output tokens remain **75059 / 67453**. Repeat application preserved
validation decisions, duplicate links, canonical membership, and JSONL/report
bytes. SQLite integrity/foreign-key checks and comparisons of every immutable
field in the original extraction tables passed. Refreshed output paths are
recorded in `data/derived/revalidation/KM_PAD_2020_pymupdf.apply.json`, alongside
`KM_PAD_2020_pymupdf.preview.json` and `KM_PAD_2020_pymupdf.repeat.json`.

Historical limitation: this saved run retained neither the original prompt/source
text nor a historical PDF fingerprint or chunk settings. The available PDF's hash
pins the bytes inspected, and pypdf reconstruction reproduces all 17 old chunk
ranges, but the original model input cannot be independently certified. The audit
records this reconstruction explicitly. Extractor versions can change text and
boundaries; use the lockfile and prefer retained snapshots for reproducibility.

## Deferred evidence-location and spacing work

**A. Cross-page evidence:** ID 25 spans physical PDF pages **15–16**, ID 52
spans **17–18**, and ID 194 spans **51–52**. The claims are supported, but no single
cited page contains the whole quote. A later phase should represent page-specific
spans or explicit page ranges while preserving attribution. This change does not
implement unrestricted cross-page searching.

**B. Structured table evidence:** IDs **136, 137, 139**, physical PDF page **48**.
Their quotes combine entries from the appropriate table column while linear text
extraction interleaves neighboring columns. Later work should preserve cell/column
provenance or use separate evidence spans. Flattening away column relationships
or accepting arbitrary noncontiguous text would weaken the evidence contract and
is not implemented.

**C. Additional spacing cases:** ID **187** uses `6 -month` versus `6-month`;
ID **190** uses `audit) .` versus `audit).`. Both are source-supported but outside
the current narrow normalization rules. They remain rejected and require a
separately tested normalization change.

A source-supported claim and an observation satisfying the implemented evidence
contract are distinct outcomes. These deferred cases should be carried into a
later evidence-location phase without labeling all remaining rejections as
fabrication.

## Runnable commands

Run from the repository root. These commands operate on a **new copy**, preserving
both existing databases. The supplied hash identifies the available PDF; it is
not a recovered historical fingerprint.

```sh
stage1_run=20261101T223237Z_aaf4b4fdeb6f483cb55ed48ce9e96fe7
stage1_sha=3616a45b8bbd7af682bde14ae9cf55854dba2ab3034b9d7484df22bff76dfb2e
stage1_input=data/derived/revalidation/KM_PAD_2020_quote_validation.sqlite
stage1_copy=data/derived/revalidation/KM_PAD_2020_pymupdf.review.sqlite
```

SQLite-safe backup/copy; the destination must not already exist:

```sh
uv run python - "$stage1_input" "$stage1_copy" <<'PY'
import sys
from pathlib import Path

from digital_applications_prototype.extraction.revalidation import backup_database

backup_database(Path(sys.argv[1]), Path(sys.argv[2]))
PY
```

Read-only preview:

```sh
uv run extract-observations revalidate \
  --database "$stage1_copy" --document KM_PAD_2020 --run "$stage1_run" \
  --source-pdf data/input/comoros/KM_PAD_2020.pdf --source-sha256 "$stage1_sha" \
  --historical-extractor pypdf --max-chars 12000 --overlap-chars 600
```

Add `--diagnostics` for unresolved quotes and normalized excerpts. Apply on that
copy, with another automatic backup before any changes:

```sh
uv run extract-observations revalidate \
  --database "$stage1_copy" --document KM_PAD_2020 --run "$stage1_run" \
  --source-pdf data/input/comoros/KM_PAD_2020.pdf --source-sha256 "$stage1_sha" \
  --historical-extractor pypdf --max-chars 12000 --overlap-chars 600 \
  --apply > data/derived/revalidation/KM_PAD_2020_pymupdf.review.apply.json
```

`--backup NEW_PATH` optionally chooses the automatic backup filename; existing
backups are never overwritten. Repeat the apply command to verify idempotence;
default backup names are distinct timestamps. For runs with complete retained
snapshots, omit the historical extractor/chunk arguments, and add
`--refresh-pdf-text` only when explicitly requesting the alternative PDF text.

Inspect the existing verified copy's report, export row count, and audit metadata
(or substitute the review apply-summary path produced above):

```sh
uv run python - <<'PY'
import json
import sqlite3
from contextlib import closing
from pathlib import Path

summary = json.loads(
    Path("data/derived/revalidation/KM_PAD_2020_pymupdf.apply.json").read_text()
)
report = json.loads(Path(summary["exports"]["report"]).read_text())
rows = Path(summary["exports"]["jsonl"]).read_text().splitlines()
print(json.dumps(report, indent=2))
print("JSONL rows:", len(rows))
assert len(rows) == report["observations_persisted"] == 184
with closing(sqlite3.connect(
    "file:data/derived/revalidation/KM_PAD_2020_pymupdf.sqlite?mode=ro", uri=True
)) as database:
    audit = database.execute(
        "SELECT validator_version, revalidated_at, source_json "
        "FROM validation_revisions ORDER BY revalidation_id DESC LIMIT 1"
    ).fetchone()
    print("Latest audit:", audit)
PY
```

The CLI prints the exact distinct export paths. Its optional
`--refresh-pdf-text` never updates retained historical source text.

## Validation

The focused tests cover real pypdf/PyMuPDF differences, the shared live/offline
loader, extractor metadata and hash checks, original-first matching, historical
chunk changes, bounded overlap projection and ambiguous-boundary refusal.
Existing tests continue to cover fabrication, negation/numbers, wrong pages,
multiple spans, dry-run immutability, transaction rollback, other-run isolation,
no model/network calls, repeated apply, and report/export agreement.

Commands:

```sh
uv run python -m unittest discover -s tests -v
uv run --with ruff --with pyright ruff check .
uv run --with ruff --with pyright ruff format --check .
uv run --with ruff --with pyright pyright
git diff --check
```

Repository-wide Ruff/format and Pyright have pre-existing findings; results are
compared against the pre-change working-tree snapshot rather than hidden with
ignores. In the sandbox, uv uses `UV_CACHE_DIR=/private/tmp/stage1-uv-cache` and
`--offline` after installation. No LLM/API calls or paid extraction are needed.

Results: **114 tests pass**; the PDF loader, revalidation module, new tests, and
this guide pass targeted Ruff lint/format checks. Full Ruff reports **28 existing
findings** (29 before this change, no introduced findings). Full format checking
still flags **19 existing files** (20 before this change) and the invalid empty
`first_nb.ipynb`. Pyright reports the same **86 existing errors**, with none
introduced. `git diff --check` passes.

Files changed for this phase: `pyproject.toml` and `uv.lock`;
`extraction/pdf.py`, `extraction/cli.py`, `extraction/revalidation.py`,
`extraction/validation.py` and `persistence/repository.py`;
`tests/test_pdf_sources.py`, the existing `tests/test_revalidation.py` and
`tests/fixtures/km_pad_2020_quotes.json`; `README.md`, `docs/stage1.md` and this
guide. Prompts, model-authored observation schemas, chunking policy, relevance
filtering, and semantic deduplication are unchanged.
