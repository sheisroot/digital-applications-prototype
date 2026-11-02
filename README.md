# digital-applications-prototype

A small Python 3.12+ project using `uv` and a `src/` package layout.
The reusable LLM infrastructure is separate from domain feature code.
[Stage 1 extraction](docs/stage1.md) converts input PDFs into atomic, source-grounded
implementation observations; it does not generate lessons or process holdouts.

```sh
uv sync
uv run python -m unittest discover -s tests -v
```

Tests use local fakes/mocks; they need no credentials and make no model calls.

## Architecture and file responsibilities

```text
src/digital_applications_prototype/
├── __init__.py                 # Existing CLI entry point
├── core/
│   ├── __init__.py             # Configuration package
│   └── config.py               # Explicit environment loading and secret masking
└── llm/
    ├── __init__.py             # Infrastructure package
    ├── prompt_renderer.py      # Jinja Environment, FileSystemLoader, StrictUndefined
    ├── service.py              # Thin provider-independent generation boundary
    ├── adapters/
    │   ├── __init__.py         # Adapter package
    │   ├── base.py             # Generic typed LLMAdapter protocol
    │   └── openai.py           # Async SDK invocation and provider response handling
    ├── prompts/
    │   └── smoke.jinja2        # File-based architecture validation prompt
    └── schemas/
        ├── __init__.py         # Shared schema package
        └── smoke.py            # Trivial SmokeTestResponse Pydantic model
scripts/llm_smoke.py             # Explicit, opt-in live composition example
tests/test_llm.py               # Offline renderer, service, adapter, config checks
```

Feature service → prompt renderer → LLM service → adapter → provider → typed result.
Orchestration will compose feature services later; it does not belong in the renderer
or adapter. Feature-specific prompts and schemas can live alongside those features.
`PromptRenderer(custom_directory)` supports other template directories.

The OpenAI adapter uses `AsyncOpenAI.responses.parse(text_format=Schema)` and
returns `output_parsed`. This lets the SDK derive JSON Schema and validate the
response, without custom JSON parsing. See the official
[Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).
SDK exceptions propagate unchanged; refusals, incomplete responses, and missing
parsed output raise `StructuredOutputError`. The SDK's built-in retry behavior
is retained; no additional retry or logging framework is introduced. Prompts and
credentials are not logged. Only the concrete adapter and application composition
code import the provider SDK.
The adapter also exposes optional last-call token counts through `LLMService`.
The OpenAI adapter isolates these counts per async task. Stage 1 reads them in the
calling task immediately after generation; adapters without usage metadata remain
supported.

## Future feature service example

This illustrative service uses the smoke schema; replace it with a feature-owned
schema and prompt when implementing a real feature:

```python
from digital_applications_prototype.llm.prompt_renderer import PromptRenderer
from digital_applications_prototype.llm.schemas.smoke import SmokeTestResponse
from digital_applications_prototype.llm.service import LLMService


class SummaryService:
    def __init__(self, renderer: PromptRenderer, llm: LLMService):
        self.renderer = renderer
        self.llm = llm

    async def summarize(self, text: str) -> SmokeTestResponse:
        prompt = self.renderer.render("smoke.jinja2", text=text)
        return await self.llm.generate_structured(
            prompt=prompt, response_model=SmokeTestResponse
        )
```

Inject `LLMService(fake_adapter)` in tests. The feature never imports OpenAI.
The application boundary constructs the concrete adapter and owns the async SDK
client lifecycle; `scripts/llm_smoke.py` shows that wiring with `async with`.

## Optional live smoke call

Create a `.env` file in the project root with `OPENAI_API_KEY` and `LLM_MODEL`.
The settings layer loads this file once, when settings are first requested, using
`python-dotenv` with `override=False`. Existing process environment variables take
precedence, including empty values. The path is resolved from the settings module,
independently of the current working directory. A missing `.env` is allowed;
both values must still be nonempty when settings are loaded. Choose an
account-accessible model that supports structured outputs. `.env` is ignored by Git.

The following command makes a paid API call. It is never run by the test suite:

```sh
uv run python scripts/llm_smoke.py
```

## Stage 1 pilot

```sh
# Inspect without calling the model or writing run artifacts
uv run extract-observations --document KM_ISR07_2023 --dry-run

# Normal live extraction using the configured model
uv run extract-observations --document KM_ISR07_2023

# Another live run creates an independent persisted attempt
uv run extract-observations --document KM_ISR07_2023

# SQLite only, without JSONL/report exports
uv run extract-observations --document KM_ISR07_2023 --no-export

# Override the default of 3 concurrent chunk calls (1 runs sequentially)
uv run extract-observations --document KM_PAD_2020 --concurrency 4
```

Live execution is the default for an explicit `--document`, `--country`, or `--all`
selection. SQLite at `data/derived/stage1.sqlite` is the canonical extraction record;
startup creates its directory and schema automatically. Every live document attempt
gets a new UTC timestamp/UUID run ID. Within one document, up to 3 chunk extractions
run concurrently (`--concurrency N`, minimum 1). Documents are processed sequentially.
The orchestrator commits each completed chunk's observations (including rejected
ones), ordered evidence spans, warnings, and token usage promptly. Database writes
are serialized and no model call owns a database session. Later failures or
interruption preserve earlier committed work.

Final status follows chunk coverage: `completed` for all chunks, `partial` for some,
and `failed` for none. Warnings remain separate. Overlap duplicates stay stored with
a pointer to their first occurrence. Clean queries return valid/repaired nonduplicates.
JSONL and `.report.json` remain best-effort compatibility exports; `--no-export`
disables both. Export failures leave database state and extraction status intact.
Previous attempts are never overwritten or used to skip a new run.

See [Stage 1 documentation](docs/stage1.md) for the schema, lifecycle, query examples,
validation rules, and one-document pilot commands.

[Offline revalidation](docs/stage1-revalidation.md) describes the read-only
`extract-observations revalidate` command, PyMuPDF source metadata, apply-on-copy
commands, and the saved `KM_PAD_2020` investigation. Its
[deferred evidence limitations](docs/stage1-revalidation.md#deferred-evidence-location-and-spacing-work)
cover cross-page quotes, structured table evidence, and two narrow spacing cases.
