# Stage 1 implementation observations

Stage 1 extracts atomic, source-grounded implementation claims from the explicit
15-document input manifest. It reuses the existing Jinja renderer, typed adapter
protocol, `LLMService`, and OpenAI adapter without changing those abstractions.
It does not generate lessons, assign canonical factors, cluster, compare projects,
or access holdout documents.

## Files and responsibilities

```text
src/digital_applications_prototype/
├── core/
│   └── llm_runtime.py                 # Application wiring and SDK client lifetime
├── extraction/
│   ├── __init__.py                    # Feature package
│   ├── models.py                      # Enums, semantic response and provenance schemas
│   ├── manifest.py                    # 15 explicit entries, selection and input path guard
│   ├── pdf.py                         # PDF text as numbered PageText records
│   ├── chunking.py                    # Deterministic page-boundary chunks
│   ├── service.py                     # Prompt → LLMService → validated observations
│   ├── validation.py                  # Grounding, page checks and deterministic IDs
│   ├── revalidation.py                # Offline preview, backup, apply and audit
│   ├── persistence.py                 # Optional atomic JSONL/report exports
│   └── cli.py                         # Selection, dry-run and document orchestration
├── persistence/
│   ├── __init__.py                    # Local persistence package
│   ├── database.py                    # Async SQLite engine, schema, WAL and FK setup
│   ├── models.py                      # Extraction, retained source and validation audit storage
│   └── repository.py                  # Short chunk transactions, queries, finalization
└── llm/prompts/
    └── implementation_observations.jinja2
tests/test_extraction.py              # Offline feature, PDF, CLI and boundary tests
tests/test_persistence.py             # Temporary SQLite durability and audit tests
tests/test_concurrency.py             # Bounded calls, completion order, failure isolation
```

`pyproject.toml` includes `pypdf`, `sqlalchemy[asyncio]`, `aiosqlite`, and the
`extract-observations` command; `uv.lock` records resolved dependencies. SQLAlchemy’s
asyncio extra supplies its required greenlet dependency. The existing `data/` ignore
rule covers the database, WAL/SHM sidecars, and file exports.

## Execution flow

1. Select one manifest entry with `--document`, or explicitly select a country or
   the input corpus. A selection remains required; no selection never means the
   whole corpus. Live is the default; `--dry-run` is the inspection-only mode.
   Initialize SQLite and create a run before reading the document. When file exports
   are enabled, reserve a unique artifact pair independently of extraction success.
   Selection never scans directories. Project IDs and dates are
   explicit: cover dates for PAD/AF papers and archive dates for ISRs. Dates not
   established on restructuring covers are null rather than inferred from filenames.
2. Resolve the PDF path and reject anything outside `data/input/`, including
   symlink escapes. Extract each physical PDF page, retaining 1-based page numbers.
3. Group consecutive pages using a soft character target (`--max-chars`, default
   12000). Full pages are never split, so an oversized page can exceed the target.
   Add bounded trailing context from the previous chunk, as described below.
   Empty pages retain their page numbers; all-empty chunks are omitted. Register all
   intended chunks as pending, then mark each running before its model call.
   Process at most `--concurrency N` chunks concurrently within that document
   (default 3, minimum 1). Document selections are still processed sequentially.
4. Render the extraction template with application metadata and numbered page text.
   The feature calls `LLMService.generate_structured` with `ExtractionResult`.
   Provider wiring lives in `core/llm_runtime.py`; the extraction package imports
   no OpenAI SDK code.
5. The model generates semantic fields and an evidence bundle containing one or
   more exact page-specific spans, each with its own optional evidence date.
   Pydantic validates the whole response. Deterministic validation then examines
   each observation independently: valid claims are kept, unsupported optional dates
   are cleared with warnings, and ungrounded claims are rejected with warnings.
   Subsequent observations are still examined. Call/schema failures are recorded as
   chunk failures, and remaining chunks are attempted without an additional retry layer.
6. Attach manifest metadata, chunk page range, and a SHA-256-derived provenance ID.
   Immediately commit each successful chunk’s completed state, token usage, warnings,
   every model observation and its validation status, and ordered evidence spans in
   one short transaction. Repaired rows use the validated fields; original model
   output is also retained as JSON. Rejected rows retain their original fields,
   including ungrounded spans or an empty evidence list. Chunk-level failures commit
   a failed state and error independently, then extraction continues.
7. Apply deterministic overlap deduplication in source chunk/observation order and
   mark duplicates without deleting them. Finalize status and token totals from
   persisted chunk state. Summaries and clean observations are queried from SQLite.
   By default, completed runs also export clean JSONL to
   `data/derived/extracted/<country>/<document_id>__<run_id>.jsonl`, and all final
   outcomes export a paired report. `--no-export` disables both files. Export
   errors never determine extraction correctness or status.

## Semantic and temporal fields

The enums constrain structure, not domain vocabulary. Factor descriptions,
actors, implementation areas, contexts, responses, and effects remain bottom-up.
Stage 1 produces no factor labels. Required `factor_description` expresses the
implementation mechanism in about 8–25 words for later embeddings and clustering.
It removes incidental names and acronyms while retaining specific causal or
operational meaning; it is not a canonical category, lesson, or recommendation.

`claim_scope` is `actual`, `planned`, `anticipated`, `recommended`, or `inferred`. Actual claims
describe what occurred or was the case as of the relevant evidence date. Planned
claims describe intended, scheduled, proposed, or committed future actions.
Anticipated claims preserve source-authored forecasts or uncertain future effects,
such as potentially higher costs or expected low operating volumes. They are
distinct from an extractor's inference. Recommendations must be explicit in the
source; inference is used sparingly.

Optional `EvidenceSpan.evidence_date` is a calendar date explicitly established in
that span's quotation, such as an indicator measurement date or reporting cutoff.
It remains null when absent or when a full date cannot be established without
guessing. A dated table row and separate undated prose use separate spans, with
the date attached only to the dated row. It never defaults to `document_date`.
Historical measurements must not be described as current merely because they
appear in a newer report.

The prompt seeks meaningful enablers, progress, prerequisites, tradeoffs, adaptations,
and outcomes alongside constraints. It separates substantively useful conditions
from responses and clarifies valence based on supported implementation effects.

## Model-facing schema

The model generates `ExtractionResult`, with this field layout. All model-facing
objects forbid extra fields; application-owned document metadata, chunk ranges,
and observation IDs are attached only after validation.

```python
class EvidenceSpan(BaseModel):
    quote: str = Field(min_length=1)
    page: int = Field(ge=1)
    evidence_date: date | None = None
    role: EvidenceRole = EvidenceRole.SUPPORTS

class EvidenceBundle(BaseModel):
    summary: str | None = None
    spans: list[EvidenceSpan]

class ModelObservation(BaseModel):
    observation_type: ObservationType
    valence: Valence
    claim_scope: ClaimScope
    observation: str
    factor_description: str
    implementation_area: str | None = None
    actors: list[str] = Field(default_factory=list)
    condition_or_context: list[str] = Field(default_factory=list)
    action_or_response: list[str] = Field(default_factory=list)
    observed_effect: str | None = None
    evidence_strength: EvidenceStrength
    evidence: EvidenceBundle

class ExtractionResult(BaseModel):
    observations: list[ModelObservation] = Field(default_factory=list)
```

`EvidenceRole` is `supports` (the default), `qualifies`, `contradicts`, or `context`.
Support is direct evidence; qualification adds a limitation or caveat; contradiction
preserves a source conflict without reconciling it; context adds necessary surrounding
information sparingly. Most spans should simply support the claim.

`evidence.summary` is optional derived prose explaining how the evidence bears on
one coherent claim. Prefer null for a straightforward single quote. Multiple spans
may be non-contiguous but must not become a justification for bundling independently
useful claims. Each call may cite only its supplied chunk, including its overlap.
The extractor performs no additional reasoning pass across chunks or documents.

Illustrative one-span evidence bundle:

```json
{
  "summary": null,
  "spans": [
    {
      "quote": "The first two procurement processes were unsuccessful.",
      "page": 3,
      "evidence_date": null,
      "role": "supports"
    }
  ]
}
```

Illustrative multi-span evidence bundle with a dated table and undated prose:

```json
{
  "summary": "The results table associates the status with the October 2020 reporting date.",
  "spans": [
    {
      "quote": "Date 01-Jul-2020 05-Oct-2020 05-Oct-2020 30-Jun-2025",
      "page": 6,
      "evidence_date": "2020-10-05",
      "role": "supports"
    },
    {
      "quote": "The GRM has yet to be established.",
      "page": 6,
      "evidence_date": null,
      "role": "supports"
    }
  ]
}
```

## Chunk-boundary context

Base chunks retain their page-boundary grouping. Each later chunk prepends at most
600 characters from the previous chunk's final nonempty full page, moving the cut
forward to a word boundary. This exact suffix is stored separately as `overlap`
with its original physical page number; `source_pages` includes it for prompting
and grounding. The prompt marks it as trailing overlap. Very short pages can fit
entirely within the bound; long pages are not repeated in full. Overlap is never
copied recursively from previous overlap.

The suffix is additional to the soft character target and costs at most roughly
150 source tokens per boundary using characters/4. Use `--overlap-chars N` to adjust
the bound or `--overlap-chars 0` to disable it. The chunk's overall page range
includes the overlap page, while dry-run also lists its new full-page range.
This small fixed window helps cross-page passages without guaranteeing that every
long paragraph is complete.

## Grounding and limits

Every span's whole quote must be a substring of its required physical page after
the same conservative spacing normalization on quote and source. In addition to
collapsing whitespace, spaces around an ASCII hyphen between ASCII letters and spaces
before `,.;:!?` immediately following an ASCII letter are optional. Every non-whitespace
character and all other word boundaries remain significant. See
[offline revalidation and reproduced failures](stage1-revalidation.md).
The page must actually be
represented in the supplied chunk; a preceding overlap page exposes only its suffix,
never the rest of that page. A cross-page passage requires separate spans. There is
no fuzzy matching, punctuation replacement, hyphen deletion, word-fragment joining,
or silent quote repair.
Stored quotes retain their original wording, whitespace, and span order. One invalid
span rejects only that observation. Structured warnings identify chunk page ranges,
1-based observation/span indexes, a stable code, a message, and the action taken.
Validation returns `valid`, `repaired`, or `rejected`; only rejected outcomes have
no observation, and repaired outcomes must include warnings. All schema-valid
observations in the response are examined independently.

For a populated span date, an additional deterministic check requires the full date
in that same quote. Recognized forms are year-first numeric dates such as
`2022-05-13` (also slash/dot separators) and English month-name dates such as
`13-May-2022`, `May 13, 2022`, or `13 May 2022`, including abbreviated months and
harmless PDF whitespace. Ambiguous numeric day/month order and partial dates are
not interpreted. An unsupported date is now cleared to null, retaining the grounded
span and observation and recording `unsupported_evidence_date` with action
`evidence_date_set_to_null`. The matcher is unchanged and no repair LLM call is made.
Quotes, pages, and roles are never weakened or removed to retain an ungrounded claim.
Pydantic separately rejects malformed whole responses and invalid calendar dates/roles
as chunk-level failures. A structurally valid empty span list reaches deterministic
validation and rejects only its observation. Missing evidence fields or malformed
span objects are schema failures. The clean `ImplementationObservation` model also enforces
at least one span, so empty evidence cannot enter scientific JSONL. The database
retains rejected raw observations separately for audit, including empty span lists.

The spans must collectively support the claim and its materially populated effects,
responses, and contexts. The prompt requests another span or an empty field when
support is missing. Deterministic checks establish textual/date presence, not
semantic sufficiency, correct table-column association, or atomicity. Those still
need pilot review. The derived summary is persisted without quotation validation.

## Provenance signatures, IDs, and duplicates

After chunk persistence, mark later valid/repaired observations with the same signature:

```text
(document_id, observation_type, sorted((page, normalized_quote, role) for each span))
```

This marks overlap duplicates even when generated observations, factor descriptions,
summaries, span order, or chunk boundaries differ. Different observation types preserve
legitimate same-passage condition/action claims. Different pages, quotes, roles, span
collections, and documents remain distinct. There is no semantic deduplication.

The domain `ImplementationObservation.observation_id` is the document ID followed by
the first 24 hex characters of a
SHA-256 digest of that same signature. Generated wording, interpreted evidence dates,
and chunk ranges no longer affect IDs. Changing source evidence, its role or page,
or observation type changes the ID. The persisted chunk range still records where
the observation was extracted. In SQLite this ID is stored as `provenance_id`;
the integer `observation_id` primary key uniquely identifies each model-produced row,
including duplicate instances and repeated runs. Rejected observations have no
validated provenance ID. Existing pilot files are not migrated.

[PyMuPDF](https://pymupdf.readthedocs.io/en/latest/app1.html) supplies page-level
text for new runs and PDF-backed offline revalidation through one shared loader.
New source snapshots retain the PDF hash, extractor version/options, exact pages,
and overlap. Historical sources remain preferred; alternative PDF text is confined
to recorded page/overlap limits. pypdf remains available for reconstructing legacy
chunk constraints. See [revalidation and deferred evidence limitations](stage1-revalidation.md)
for the dependency/license constraint and historical reproducibility behavior.
Documents with no extractable text fail visibly and request inspection for OCR
needs; empty pages emit warnings. This version does not automatically OCR,
reconstruct tables, match across pages, or infer printed report page numbers.

## SQLite schema and lifecycle

`initialize_database(path)` creates parent directories and tables explicitly; there
are no column migrations or manual SQL setup. Two additive tables retain exact
chunk source text for new runs (`extraction_chunk_sources`) and append offline
validation audits (`validation_revisions`). Existing extraction rows are unchanged.
CLI startup uses
`<root>/data/derived/stage1.sqlite`. The async engine uses aiosqlite, WAL,
foreign keys, a 30-second busy timeout, and FULL synchronous commits. The session
factory uses `expire_on_commit=False`. Model calls run concurrently within one
document and hold no DB session. Each repository method opens a short
session/transaction; all calls to these methods belong to the orchestrator.

| Table | Stored state and constraints |
| --- | --- |
| `extraction_runs` | Timestamp/UUID `run_id` primary key, indexed document ID, UTC start/finish, final status, model, schema version, configured max concurrency, token totals, error |
| `extraction_chunks` | Integer `chunk_id` primary key, run FK, 1-based chunk index and page range, pending/running/completed/failed status, tokens, UTC start/finish, error/type, validation warnings JSON; unique `(run_id, chunk_index)` and index on run/status |
| `implementation_observations` | Integer `observation_id` primary key, domain `provenance_id`, run/chunk FKs, 1-based observation index, every Stage 1 semantic/document/chunk field, actors/context/response JSON arrays, evidence summary, validation status, original model output JSON, self-FK `duplicate_of_observation_id`; unique `(chunk_id, observation_index)` and indexes for clean queries |
| `evidence_spans` | Integer primary key, observation FK, 1-based `span_index`, verbatim quote, physical page, optional evidence date, role; unique `(observation_id, span_index)` and indexed observation FK |

All operational timestamps use ISO 8601 with an explicit `+00:00` UTC offset;
document/evidence dates use SQL date columns. Status constraints prevent invalid
run/chunk/validation vocabularies.

A run’s `status` is NULL until finalized and `finished_at` is also NULL. This keeps
an interrupted attempt distinguishable from a finalized failure without introducing
a fourth final status. All intended chunks are registered in one transaction before
model setup/calls, so setup failures leave inspectable pending chunks. A PDF failure
can leave a run with no chunks.

For each chunk, a short transaction marks it running. Extraction and deterministic
validation then occur outside any database session. The successful chunk transaction
is exactly:

```text
BEGIN
  update chunk: completed, finished_at, tokens, warnings
  insert every observation (valid, repaired, rejected)
  insert each observation’s ordered evidence spans
COMMIT
```

An insert failure rolls back the entire chunk, including the completed transition
and usage. A whole-chunk provider/schema failure instead commits `failed`, its
error/type, and finish time. It does not erase earlier chunk commits. A result-write
failure is logged separately and recorded in the run error; the orchestrator attempts
to mark the chunk failed using a fresh transaction. It continues sibling work if
that succeeds. If failure-state persistence also fails, or a chunk cannot be marked
running, it stops admitting new calls while draining already active tasks and
attempting to persist their outcomes. Uncommitted results never count as completed.
Earlier commits remain stored; unfinished chunks stay running/pending for inspection.

The orchestrator maintains a sliding window of at most `max_concurrency` tasks and
uses `asyncio.wait(..., return_when=FIRST_COMPLETED)` to receive results. A slot is
replenished after its result is persisted. Each task owns only prompt rendering,
remote generation, parsing, deterministic observation validation, and its result.
The orchestrator owns running/terminal transitions, all database sessions, failure
recording, and document finalization. Ordinary task exceptions are caught individually.
There is no task-owned transaction or shared `AsyncSession`. On cancellation,
children are cancelled and awaited before the client/database close; cancellation
propagates without fabricating final chunk or run state.

OpenAI `last_usage` metadata is task-local via `ContextVar`, preserving the existing
typed generation interface. Each chunk carries its own usage to persistence;
final totals still derive from committed chunks. Injected adapters with optional
usage metadata must likewise make it available to the calling task immediately
after generation.

After chunk attempts, `mark_duplicates(run_id)` applies the same exact-evidence
signature rule as before and is safe to rerun, including on partial runs. Rejected
rows do not participate. Finalization derives coverage from all registered chunks:

| Status | Meaning |
| --- | --- |
| `completed` | Every intended chunk produced usable structured output |
| `partial` | At least one chunk completed and coverage is incomplete |
| `failed` | No chunks completed, including preparation/setup failure |

A schema-valid chunk returning zero observations or only rejected observations still
counts as successful coverage, preserving the existing validator’s behavior.
Warnings and operational errors are separate from coverage status. Unknown token
counts remain NULL; a run total is NULL if any completed chunk lacks that count or
there are no completed chunks. Failed call usage is not available through the
existing adapter and is excluded.

Crashes/cancellation leave finalization unfinished but preserve every already
committed chunk. Full resume/retry CLI support is intentionally deferred.

## Queries and summaries

`ExtractionRepository.get_clean_observations(run_id)` reconstructs the existing
Pydantic output with its original provenance IDs, metadata, summaries, and span
order. It works on partial runs as well as completed ones. Its filter is:

```sql
SELECT * FROM implementation_observations
WHERE run_id = :run_id
  AND validation_status IN ('valid', 'repaired')
  AND duplicate_of_observation_id IS NULL;
```

All raw rows, including rejected observations and duplicates, remain queryable.
Evidence can be inspected directly:

```sql
SELECT o.observation_id, o.validation_status, e.span_index,
       e.quote, e.page, e.evidence_date, e.role
FROM implementation_observations AS o
JOIN evidence_spans AS e ON e.observation_id = o.observation_id
WHERE o.run_id = :run_id
ORDER BY o.chunk_id, o.observation_index, e.span_index;
```

`get_run_summary` builds the existing `RunReport` from persisted run/chunk/observation
state. The CLI prints status, run ID, chunk counts, observations returned and clean,
valid/repaired/rejected counts, duplicate count, warnings, tokens, model, schema,
concurrency, and elapsed time. `observations_returned = valid + repaired + rejected`.
The compatibility fields `duplicates_removed` and `observations_persisted` now mean
**duplicates marked** and **clean observations stored**, respectively. The latter is
`valid + repaired - duplicates_marked`, including for partial runs. All returned
observations are stored regardless of that clean count. The configured
`max_concurrency` is included in the run record and report (default 3 for the CLI).
Clean queries, exports, warnings, and failure reports follow source chunk index,
then observation/span index. SQLite row IDs and commit timestamps can reflect
completion order; they do not determine analytical order or duplicate selection.

## Optional file exports

SQLite is canonical. By default each invocation still attempts immutable timestamp/
UUID JSONL/report exports for compatibility. Both use the database run ID:

```text
KM_ISR07_2023__<run_id>.jsonl
KM_ISR07_2023__<run_id>.report.json
```

JSONL is exported only for completed coverage and contains clean observations,
possibly none. Reports can describe completed, partial, or failed attempts. No file
is needed to retain partial results. `--no-export` skips both files and reservations.
Existing files are never overwritten or used to skip extraction. Legacy pilot
files remain untouched and are not imported into SQLite.

File writes retain the existing reservation and atomic `.part`-then-replace helpers.
Failure to reserve/write files or display a summary produces an operational diagnostic
and a nonzero CLI exit, while the persisted extraction status stays based on chunk
coverage. Normally completed runs exit 0; partial/failed runs exit 1. Dry-run needs
no model configuration and creates neither database nor exports.

## Evidence-bundle pilot rerun

Run from the repository root. Dry-run does not need `.env` or credentials and
prints only metadata, page/chunk ranges, overlap sizes, and each prompt's size,
not source text:

```sh
uv run extract-observations --document KM_ISR07_2023 --dry-run
```

The evidence-bundle dry-run found 7 pages and 2 chunks:

| Chunk | Source pages | New full pages | Overlap characters | Prompt characters | Approximate tokens |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | 1–2 | 1–2 | 0 | 18,813 | 4,704 |
| 2 | 2–7 | 3–7 | 596 | 20,748 | 5,187 |

Compared with v2's 16,593 / 18,528 characters (4,149 / 4,632 approximate tokens),
the prompt adds 2,220 characters, about 555 tokens, per call. Chunking is unchanged.

These are rendered-prompt estimates using characters/4, excluding the structured
output schema and provider formatting. They are not exact tokenizer counts.

For the opt-in paid run, fill in `OPENAI_API_KEY` and a structured-output-compatible
`LLM_MODEL` in the ignored project-root `.env` file. Settings load it automatically
once; process environment variables take precedence. A normal invocation is live:

```sh
uv run extract-observations --document KM_ISR07_2023
```

This processes both chunks of that single ISR, persists them immediately in
`data/derived/stage1.sqlite`, and creates a new run. On completed coverage, the
default compatibility exports are:

```text
data/derived/extracted/comoros/KM_ISR07_2023__<run_id>.jsonl
data/derived/extracted/comoros/KM_ISR07_2023__<run_id>.report.json
```

Run the same command again to create another independent attempt. Use
`--max-chars 8000` to adjust the soft chunk target. Existing pilot JSONL files are
not migrated or rewritten; their existence never skips a new invocation.
New observations use evidence bundles; top-level single-quote/page/date fields are removed.
Add `--no-export` to keep the attempt entirely in SQLite.
`--root /path/to/repo` supports execution from another
directory.
`--country comoros` and `--all` are explicit batch options for later use; dry-run
requires a single-document selection. No real model call was run during this
refinement.

For the next PAD pilot, use the default three concurrent chunk calls, or choose a
different bound. These are live commands and are not run by the test suite:

```sh
uv run extract-observations --document KM_PAD_2020 --concurrency 3
uv run extract-observations --document KM_PAD_2020 --concurrency 1
uv run extract-observations --document KM_PAD_2020 --concurrency 4
```

## Offline validation

```sh
uv run python -m unittest discover -s tests -v
```

Tests generate tiny PDF fixtures locally and inject fake adapters into the existing
`LLMService`. They cover metadata and prompt flow, anticipated claims, one/multiple
non-contiguous spans, nullable derived summaries, evidence roles, required physical
pages, per-span quoted dates, page extraction, deterministic chunking, bounded overlap,
cross-page claims, provenance-based IDs, deduplication despite paraphrases and span
reordering, preservation of same-evidence different-type claims, observation repairs
and rejections with later observations retained, completed runs with warnings,
partial/failed reports without JSONL, usage aggregation, immutable reruns and collisions,
atomic failure cleanup, report/display error isolation, JSONL bundle round trips,
holdout/traversal/symlink rejection, explicit selection, credential-free dry-run,
empty successful output, and preservation of earlier runs.

SQLite tests use temporary databases only. They cover idempotent initialization,
required tables and pragmas, uniqueness/FK constraints, full-field round trips,
ordered multi-span evidence, usage, validation/audit states, original unsupported
dates versus repaired NULL dates, rejected empty evidence, retained overlap duplicates,
rerunnable duplicate marking, coverage status derivation, atomic rollback on an
actual evidence-insert failure, independent-connection visibility before the next
model call, two committed chunks surviving a third failure, cancellation preserving
unfinished state, export reservation failures, and isolating a chunk database failure.
Concurrency tests use events to force overlapping calls and reversed completion;
they verify limits of 1/3/4, prompt independent-connection visibility, source-ordered
exports and partial-run dedupe, observation repair/rejection, task-local token usage,
serialized writes, persistence failure isolation, admission stopping when failure
recording also fails, and cancellation cleanup. The entire suite uses fakes and
makes no paid model calls.

The hierarchy remains source document → exact evidence spans → atomic observation →
factor description. Embeddings, clusters, canonical titles, and cross-case lessons
are later stages; this feature builds no provenance graph or synthesis framework.
