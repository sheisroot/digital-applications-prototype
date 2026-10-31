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
   the input corpus. Selection never scans directories. Project IDs and dates are
   explicit: cover dates for PAD/AF papers and archive dates for ISRs. Dates not
   established on restructuring covers are null rather than inferred from filenames.
2. Resolve the PDF path and reject anything outside `data/input/`, including
   symlink escapes. Extract each physical PDF page, retaining 1-based page numbers.
3. Group consecutive pages using a soft character target (`--max-chars`, default
   12000), without overlap. Pages are never split, so an oversized page can exceed
   the target. Empty pages retain their page numbers; all-empty chunks are omitted.
4. Render the extraction template with application metadata and numbered page text.
   The feature calls `LLMService.generate_structured` with `ExtractionResult`.
   Provider wiring lives in `core/llm_runtime.py`; the extraction package imports
   no OpenAI SDK code.
5. The model generates only semantic fields plus a quote and optional source page.
   Pydantic validates enum/field types. Deterministic checks reject blank claims,
   blank descriptions, overlong labels, unsupported quotes, and wrong source pages.
6. Attach manifest metadata, chunk page range, and a SHA-256-derived observation ID.
   Persist each completed document to
   `data/derived/extracted/<country>/<document_id>.jsonl`.

The four enums constrain structure, not domain vocabulary. Labels, descriptions,
actors, implementation areas, contexts, responses, and effects remain bottom-up.
The prompt requests short noun labels (about 2–6 words) and mechanism descriptions
(about 8–25 words). The label validator allows 1–8 words and at most 100 characters;
these bounds catch obvious failures without treating approximate style guidance as
an exact taxonomy rule.

## Grounding and limits

The entire supporting quote must be a substring of the chunk after whitespace-only
normalization (`" ".join(text.split())`). There is no fuzzy matching, punctuation
replacement, dehyphenation, or silent quote repair. Original quote wording remains
in the output. If a source page is supplied, the whole quote must occur on that
exact page. If omitted, a unique matching page is attached deterministically;
cross-page or ambiguous quotes retain null, alongside the chunk page range.
Validation failures identify document, chunk, and observation, and fail that document.

Quote presence establishes textual grounding, not semantic entailment or atomicity.
Those properties still need pilot review. IDs are stable for identical validated
content and chunk provenance; a changed model response or chunk configuration may
produce different IDs. Only identical IDs are deduplicated; no semantic merging is
performed.

[pypdf](https://pypdf.readthedocs.io/en/stable/user/extract-text.html) is a lightweight,
pure-Python dependency with page-level text extraction. It does not require an OCR
stack. Documents with no extractable text fail visibly and request inspection for
OCR needs; individual empty pages emit warnings. This version does not automatically
OCR, reconstruct complex tables, or infer printed report page numbers.

## Completion and reruns

A final JSONL file denotes a completed document, including an empty file when no
observations were found. Only write it after every chunk validates. A temporary
file in the same directory is atomically renamed into place; failures leave no
partial final file. A failed forced rerun preserves the previous completed file.
Other documents continue and retain successful outputs; any failure returns a
nonzero exit code. Reruns skip completed files by default, and `--force` replaces
them. There is no automatic source/prompt/model change detection: use `--force`
after such changes. No chunk-level resume or concurrency framework is introduced.

## First pilot

Run from the repository root. Dry-run does not need `.env` or credentials and
prints only metadata, page/chunk counts, and first-prompt size, not source text:

```sh
uv run extract-observations --document KM_ISR07_2023 --dry-run
```

The local pilot dry-run found 7 pages and 2 chunks. The first chunk covers pages
1–2 and renders 13,756 characters (roughly 3,439 tokens using characters/4).
This token estimate is informational, not a model tokenizer.

For the opt-in paid run, fill in `OPENAI_API_KEY` and a structured-output-compatible
`LLM_MODEL` in the ignored `.env` file, then run:

```sh
uv run --env-file .env extract-observations --document KM_ISR07_2023
```

This command processes both chunks of that single ISR. Its result is:

```text
data/derived/extracted/comoros/KM_ISR07_2023.jsonl
```

Add `--force` to rerun the completed document or `--max-chars 8000` to adjust the
soft chunk target. `--root /path/to/repo` supports execution from another directory.
`--country comoros` and `--all` are explicit batch options for later use; dry-run
requires a single-document selection. No real model call was run during setup.

## Offline validation

```sh
uv run python -m unittest discover -s tests -v
```

Tests generate tiny PDF fixtures locally and inject fake adapters into the existing
`LLMService`. They cover metadata and prompt flow, page extraction, deterministic
chunking and IDs, grounding failures, JSONL round trips, holdout/traversal/symlink
rejection, selection, credential-free dry-run, empty successful output, skipping,
forced rerun, and preservation of completed documents on failure.
