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
│   ├── persistence.py                 # Atomic document-level JSONL writes
│   └── cli.py                         # Selection, dry-run and document orchestration
└── llm/prompts/
    └── implementation_observations.jinja2
tests/test_extraction.py              # Offline feature, PDF, CLI and boundary tests
```

`pyproject.toml` adds `pypdf` and the `extract-observations` command; `uv.lock`
records the resolved dependency. The existing `data/` ignore rule is sufficient.

## Execution flow

1. Select one manifest entry with `--document`, or explicitly select a country or
   the input corpus. A selection remains required; no selection never means the
   whole corpus. Live is the default; `--dry-run` is the inspection-only mode.
   Reserve a unique artifact pair for each live document attempt before extraction.
   Selection never scans directories. Project IDs and dates are
   explicit: cover dates for PAD/AF papers and archive dates for ISRs. Dates not
   established on restructuring covers are null rather than inferred from filenames.
2. Resolve the PDF path and reject anything outside `data/input/`, including
   symlink escapes. Extract each physical PDF page, retaining 1-based page numbers.
3. Group consecutive pages using a soft character target (`--max-chars`, default
   12000). Full pages are never split, so an oversized page can exceed the target.
   Add bounded trailing context from the previous chunk, as described below.
   Empty pages retain their page numbers; all-empty chunks are omitted.
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
6. Attach manifest metadata, chunk page range, and a SHA-256-derived observation ID.
   Remove duplicate claims conservatively. Only full-coverage runs atomically publish
   JSONL to `data/derived/extracted/<country>/<document_id>__<run_id>.jsonl`.
   Write a paired final run report for completed, partial, and failed attempts.

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
whitespace-only normalization (`" ".join(text.split())`). The page must actually be
represented in the supplied chunk; a preceding overlap page exposes only its suffix,
never the rest of that page. A cross-page passage requires separate spans. There is
no fuzzy matching, punctuation replacement, dehyphenation, or silent quote repair.
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
span objects are schema failures. Final persisted observation models also enforce
at least one span, so empty evidence can never enter a scientific JSONL.

The spans must collectively support the claim and its materially populated effects,
responses, and contexts. The prompt requests another span or an empty field when
support is missing. Deterministic checks establish textual/date presence, not
semantic sufficiency, correct table-column association, or atomicity. Those still
need pilot review. The derived summary is persisted without quotation validation.

## Provenance signatures, IDs, and duplicates

Before persistence, retain the first observation with the same signature:

```text
(document_id, observation_type, sorted((page, normalized_quote, role) for each span))
```

This removes overlap duplicates even when generated observations, factor descriptions,
summaries, span order, or chunk boundaries differ. Different observation types preserve
legitimate same-passage condition/action claims. Different pages, quotes, roles, span
collections, and documents remain distinct. There is no semantic deduplication.

`observation_id` is now the document ID followed by the first 24 hex characters of a
SHA-256 digest of that same signature. Generated wording, interpreted evidence dates,
and chunk ranges no longer affect IDs. Changing source evidence, its role or page,
or observation type changes the ID. The persisted chunk range still records where
the retained observation was extracted. This deliberately changes IDs from the two
earlier pilot schemas; those files are not migrated.

[pypdf](https://pypdf.readthedocs.io/en/stable/user/extract-text.html) is a lightweight,
pure-Python dependency with page-level text extraction. It does not require an OCR
stack. Documents with no extractable text fail visibly and request inspection for
OCR needs; individual empty pages emit warnings. This version does not automatically
OCR, reconstruct complex tables, or infer printed report page numbers.

## Run completeness and immutable artifacts

Every selected live invocation is a new attempt, regardless of earlier output.
Each document receives a filesystem-safe UTC timestamp plus a random suffix:

```text
KM_ISR07_2023__20261101T203456Z_a1b2c3d4e5f6.jsonl
KM_ISR07_2023__20261101T203456Z_a1b2c3d4e5f6.report.json
```

Both names use the same run ID. An exclusively created transient `.reserve` file
claims the pair until final reporting; a collision with an active reservation or
existing artifact causes another ID to be tried. Writers refuse existing files.
JSONL and report each use a same-directory `.part` file, close, then atomic
replacement to the reserved new destination. No placeholder final JSONL, incremental
append, latest alias, symlink, or overwrite of an earlier run is created.

| Status | Meaning | Normal JSONL |
| --- | --- | --- |
| `completed` | Every intended chunk returned usable structured output and document publication succeeded | Written, possibly empty |
| `partial` | At least one chunk succeeded and at least one chunk failed | Not written |
| `failed` | No chunks succeeded, or document preparation/orchestration/publication failed | Not written |

Repairs/rejections are a separate quality dimension. Any number of warnings can
coexist with `completed`; duplicate removal is a count, not a warning or status.
A schema-valid chunk returning zero observations, or only rejected observations,
still counts as successful coverage. Whole-response/provider/prompt failures are
recorded in `chunk_failures`, not observation warnings. No partial JSONL is published
or saved for resume. Validated results of successful chunks are transient on a partial run.

The CLI attempts a report even after a PDF, setup, or persistence failure. If it
cannot reserve/write artifacts, it says no report was written. If JSONL publication
succeeds but report/display handling fails, extraction remains completed; that
operational problem is separately reported and causes a nonzero exit code.
Report/display errors never retroactively label a published extraction as failed.
Normally, completed runs exit 0; partial/failed runs exit 1. Other selected documents
continue after a failure. Interrupts and process termination are not a durable-run system.

## Run report schema

`RunReport` in `extraction/models.py` is the typed operational artifact, separate
from scientific observations. Its fields are:

```text
document_id: str
run_id: str
started_at: datetime (UTC)
finished_at: datetime | null
status: completed | partial | failed
schema_version: "stage1-v3"
model: str | null
chunks_total: int
chunks_succeeded: int
chunks_failed: int
observations_returned: int
observations_valid: int
observations_repaired: int
observations_rejected: int
duplicates_removed: int
observations_persisted: int
input_tokens: int | null
output_tokens: int | null
jsonl_file: str | null
warnings: list[ValidationWarning]
chunk_failures: list[ChunkFailure]
error: str | null
```

Warnings contain `code`, `message`, nullable `chunk_start_page`, `chunk_end_page`,
`observation_index`, `evidence_span_index`, and `action`. Rejection codes include
`ungrounded_quote`, `invalid_evidence_page`, and `observation_rejected`.
Chunk failures contain `chunk_index`, `chunk_start_page`, `chunk_end_page`,
`error_type`, and `message`. A document-level failure is described in `error`.
Preparation/setup failures may prevent any chunk attempts even if the total is known.

Returned/valid/repaired/rejected counts describe schema-valid successful chunks
before deduplication. `returned = valid + repaired + rejected`; for completed runs,
`persisted = valid + repaired - duplicates_removed`. Partial/failed runs always
record zero persisted and `jsonl_file=null`. Model is the configured adapter model,
or null if configuration never became available.

The adapter exposes optional last-call usage through the existing generic service;
generation still returns its original typed Pydantic response. The sequential feature
reads metadata immediately after a successful call. Token totals aggregate successful
chunks only, excluding failed calls. Each total is null if any successful call lacks
that count, or no calls succeeded; it is never an invented zero or misleading partial sum.
Costs are not calculated. `finished_at` marks extraction/publication completion before
the report itself is serialized. JSONL/report publication failures are best-effort
operational diagnostics, not another run-status vocabulary.

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

This processes both chunks of that single ISR and creates a new timestamped report.
On completed coverage, its result is:

```text
data/derived/extracted/comoros/KM_ISR07_2023__<run_id>.jsonl
data/derived/extracted/comoros/KM_ISR07_2023__<run_id>.report.json
```

Run the same command again to create another independent attempt. Use
`--max-chars 8000` to adjust the soft chunk target. Existing pilot JSONL files are
not migrated or rewritten; their existence never skips a new invocation.
New observations use evidence bundles; top-level single-quote/page/date fields are removed.
`--root /path/to/repo` supports execution from another
directory.
`--country comoros` and `--all` are explicit batch options for later use; dry-run
requires a single-document selection. No real model call was run during this
refinement.

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

The hierarchy remains source document → exact evidence spans → atomic observation →
factor description. Embeddings, clusters, canonical titles, and cross-case lessons
are later stages; this feature builds no provenance graph or synthesis framework.
