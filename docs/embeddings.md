# Persisted semantic embeddings

This stage creates durable vectors for persisted Stage 1 `factor_description`s.
It reads SQLite, calls a separate embedding service, and appends embedding rows.
It does not change extraction prompts, observations, validation, duplicate links,
evidence, or Stage 1 reports. No clustering, relevance filtering, or semantic
deduplication is performed.

## Configuration

Add this line to your existing ignored project-root `.env`:

```dotenv
EMBEDDING_MODEL=text-embedding-3-large
```

`.env.example` documents the complete configuration without credentials.
The existing `OPENAI_API_KEY` is reused. `LLM_MODEL` still selects the generative
Stage 1 model; embeddings do not require or use it. `EmbeddingSettings` and
`OpenAISettings` share the existing cached dotenv loader and secret masking with
`Settings`. Process environment variables take precedence. The embedding default
is `text-embedding-3-large`; `--model` overrides it for one invocation.
Dry runs need no API key, and an entirely current selection needs no provider client.

The OpenAI adapter omits the `dimensions` parameter and requests float vectors.
The documented default for `text-embedding-3-large` is 3072 dimensions, but storage
and validation use the actual returned length. See the official
[embedding guide](https://developers.openai.com/api/docs/guides/embeddings) and
[Python embeddings API reference](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create).

## Select an experiment explicitly

There is no corpus table. Select exactly one of these scopes:

```sh
# One persisted run
uv run extract-observations embed --run-id RUN_ID

# Several intentionally selected runs (repeated IDs are normalized)
uv run extract-observations embed --run-id RUN_ID_1 --run-id RUN_ID_2

# A versionable manifest
uv run extract-observations embed --run-ids-file experiments/mini_corpus_v1_runs.txt --dry-run

# All persisted runs, subject to the same duplicate-document check
uv run extract-observations embed --all --dry-run
```

Create the manifest yourself from inspected extraction outputs. It contains one
exact run ID per line; blank lines and full comment lines starting with `#` are
allowed. Whitespace around IDs is ignored. Inline comments and comma-delimited
lists are rejected. An empty selection, malformed identifier, or unknown ID fails
before opening a provider client. No run IDs or countries are hard-coded.

Discover available runs without choosing an experiment automatically:

```sql
SELECT run_id, document_id, status
FROM extraction_runs
ORDER BY document_id, run_id;
```

The default database is `<root>/data/derived/stage1.sqlite`; `--root` defaults to
the current directory. `--database PATH` can point to another existing Stage 1
database. Manifest paths are relative to the current directory. A missing database
fails instead of creating an empty Stage 1 database.

Every selected run is resolved to its document, with eligible/current counts.
Multiple selected runs for one document produce an explicit conflict listing and
a nonzero exit, including in dry runs. Select one intentionally chosen run per
document. `--allow-multiple-runs-per-document` explicitly permits overlaps when an
experiment needs them. This check includes selected runs contributing zero rows.
Partial or unfinished runs are permitted; inspect their status when selecting them.
Timestamps do not determine experiment membership.

## Preview and execute

After creating your manifest, preview the exact selection:

```sh
uv run extract-observations embed --run-ids-file experiments/mini_corpus_v1_runs.txt --dry-run
```

The preview prints selected run/document counts, eligible observations, already
current embeddings, remaining work, model, representation, batch size, duplicate
documents, and per-run counts (up to 50 rows, with conflicts listed separately).
It opens SQLite in read-only mode, makes no API calls, and creates no table or
other database records. This works even before the embedding table exists.

Execute the same manifest when ready to make paid embedding calls:

```sh
uv run extract-observations embed --run-ids-file experiments/mini_corpus_v1_runs.txt
```

Only `valid`/`repaired` observations with no `duplicate_of_observation_id` and a
non-null, nonblank factor description are used. Whitespace-only descriptions are
excluded, including tabs/newlines. The current Stage 1 schema already disallows
NULL descriptions; the query also protects against nullable legacy data.
The persisted description is submitted exactly as-is: no trimming, concatenation,
or other normalization. The representation name is `factor_description`.

Batches run sequentially, defaulting to 100 inputs (`--batch-size N`, 1–2048).
Provider outputs are matched by their explicit response indexes, then validated
for count, finite values, nonempty vectors, and consistent dimensionality within
the batch. The existing async OpenAI client lifecycle and SDK retries are retained.
No model call holds a database session open.

Each completed batch is committed transactionally. If a later call or write fails,
earlier batches remain durable; rerun the same command to retry only missing work.
The command stops on the first failed batch and reports submitted, saved, failed,
and unsubmitted inputs, completed/attempted batches, actual returned models,
dimensions, and provider-reported input tokens when available. Failed counts refer
to the failed batch; inputs in later batches are unsubmitted. Raw provider exception
messages are withheld because authentication errors can contain credentials.

## Storage and currentness

The additive `observation_embeddings` table follows the existing SQLAlchemy
`create_all` convention. Live embedding startup creates only this table and its
indexes if absent. The common database initializer also includes it for new
databases. Dry-run startup never runs schema initialization.

| Field | Meaning |
| --- | --- |
| `embedding_id` | Integer row identity |
| `observation_id` | Foreign key to `implementation_observations.observation_id` |
| `representation` | `factor_description` for this stage |
| `requested_model` | Configured/overridden model requested from the provider |
| `model` | Actual model returned by the provider |
| `dimensions_mode` | `default`; distinguishes future explicitly shortened requests |
| `dimensions` | Actual vector length, also checked against JSON array length by SQLite |
| `source_text` | Exact submitted text, retained for historical reproducibility |
| `source_text_hash` | SHA-256 of that text's UTF-8 bytes |
| `embedding_json` | Portable JSON array of finite floats |
| `created_at` | UTC ISO 8601 timestamp |

Uniqueness covers observation, representation, requested/actual model,
dimension mode, actual dimensions, and source hash. Inserts never replace existing
rows. Normal reruns check observation, representation, requested model, default
dimension mode, and current text hash before any API calls. Full dimensions are
learned from responses rather than hard-coded. Changed text creates a new historical
row; the previous source text and vector remain available. Different models and
representations coexist. There is no `--force` or alternative-representation CLI;
only factor descriptions are generated in this implementation.

Currentness uses the requested model name so a provider's returned snapshot name
does not cause repeated calls. An unchanged alias is treated as current; a provider
silently retargeting an alias cannot be discovered without a new API call. Use an
explicit versioned model identifier when available for such experiments. This is a
single-process research pipeline; simultaneous invocations can both submit the same
missing inputs, although database uniqueness still prevents duplicate rows.

`EmbeddingRepository.get_embeddings(run_ids, model=...)` returns detached pairs of
embedding and observation rows, joining vector/model/representation to run,
document, project, country, and current factor description. The model argument is
the actual stored model. By default the query selects full/default requests;
`dimensions_mode` and `dimensions` can select another vector shape explicitly.
`current_only=True` excludes historical factor-description hashes and newly
ineligible observations. `current_only=False` returns history including exact
submitted text, and is required for future representations whose current-source
matching is not implemented yet. Provenance remains traceable through the existing observation,
chunk, run, and document links, without copying document metadata into embedding
rows.

Feature orchestration lives in `embeddings/`; provider-independent requests live
in `llm/embeddings.py`; only `llm/adapters/openai_embeddings.py` imports the embedding
SDK API. SQLite access lives in `persistence/embeddings.py`. The CLI dispatch matches
the existing `extract-observations revalidate` convention.

## Offline verification

```sh
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run python -m unittest discover -s tests -v
```

Tests use real temporary SQLite databases and fake providers. They cover settings,
scope/manifests, duplicate-document protection, eligibility, vector/index validation,
hash identity, model/representation coexistence, read-only previews, idempotence,
batch durability/rollback/retry, and unchanged Stage 1 records.
