# Evidence Explorer

An offline Streamlit explorer for the selected mini-corpus. Browse and filter
stored metadata, search literal keywords, inspect structured observations and every
supporting quotation, then compare factor-description embeddings. Semantic
similarity is not equivalence or corroboration. Claim scope is displayed explicitly;
planned, anticipated, and recommended claims must be read in that scope.

## Run locally

From the repository root, with Python **3.12** and uv:

```sh
uv sync
uv run streamlit run apps/evidence_explorer/app.py
```

Open the localhost URL printed by Streamlit. **No secrets or environment variables
are required.** The app does not load `.env`, import provider services, perform
extraction, generate embeddings, or contact model APIs. Public PDFs are opened only
when a viewer follows a source link. No local PDFs or external database are needed.

The app's direct runtime dependencies are pinned in
`apps/evidence_explorer/requirements.txt`: Streamlit 1.66.1 and NumPy 2.5.4.
The root uv environment also contains the existing research pipeline dependencies.
The entrypoint adds this checkout's `src/` directory to its import path so Cloud
does not need to install the pipeline package.

Search comes first, followed by Country, Observation type, and Claim scope.
Project, Document, Document type, and Valence remain in the collapsed **Advanced
filters** section. Results use short demo display labels as selectable titles, with the full factor
description underneath and compact country/type/scope metadata. Missing labels
fall back to the full description. The desktop layout gives results
44% of the two-pane content area; narrow screens stack the panes. A borderless
scroll area keeps the results list bounded. Selected cases show the observation,
mechanism, compact scope badges, and supporting evidence together. Structured and
technical details remain collapsed.

## Frozen dataset

| Item | Count |
| --- | ---: |
| Selected completed runs / documents | 15 |
| Usable nonduplicate observations | 1,093 |
| Countries | 3 |
| Supporting evidence spans | 1,187 |
| Current factor-description vectors | 1,093 |
| Dimensions per vector | 3,072 |
| SQLite file size | See `demo/evidence.json` (includes demo label audit) |

`demo/selected_runs.txt` is an exact copy of the intentionally selected local
`experiments/mini_corpus_v1_runs.txt`, rather than an inferred latest-run selection.
`demo/evidence.json` records counts, size, and the snapshot's SHA-256 digest.
`demo/evidence.sqlite` is an ordinary Git artifact, below 50 MiB; Git LFS and
remote artifact downloads are unnecessary.

The snapshot holds only selected runs, document metadata, valid/repaired
nonduplicate observations with nonblank descriptions, their ordered evidence spans,
and exactly one current `text-embedding-3-large` vector per observation. Integer
`observation_id` is retained for joins; `provenance_id` remains a separate field.
Vectors retain the original JSON float values exactly as little-endian float64
blobs. They are never rounded or regenerated. Metadata and text remain verbatim.

Public source URLs come from the checked-in
`scripts/download_pilot_corpus.py:MANIFEST`, excluding holdouts. All 15 documents
currently have links. Unknown document dates remain unknown. Physical PDF pages
are one-based page positions, not printed page labels. A `#page=N` URL fragment is
provided as a convenience; some remote PDF viewers may ignore it, so always use
the displayed page and exact quote.

To reproduce the export from the existing canonical database, write to a **new**
destination; the exporter refuses to overwrite an existing snapshot:

```sh
uv run python -m scripts.freeze_evidence_demo --output /tmp/evidence-rebuilt.sqlite
```

It reads the explicit manifest in a consistent read-only SQLite transaction,
rejects unknown/unfinished/conflicting runs, stale or ambiguous embeddings,
missing evidence, incorrect corpus counts, and artifacts of 50 MiB or larger.
It never initializes or writes the source database. No providers are invoked.
To intentionally publish a different snapshot later, rebuild, inspect its report,
replace the artifact and report together, and update the frozen-content tests.

At runtime SQLite is opened with `mode=ro&immutable=1` and `query_only=ON`, and
closed immediately after loading. The app never points at `data/derived/stage1.sqlite`.
The cached corpus uses immutable records and read-only NumPy arrays. Cosine uses
computed norms and one matrix-vector product per request; no all-pairs matrix is
created. Related results exclude the selected row and return up to five, independently
of browsing filters. The two comparison constraints can be combined.

## Streamlit Community Cloud deployment

The exact **entrypoint is `apps/evidence_explorer/app.py`**.

1. Commit and push the completed branch `demo/evidence-explorer-2026-11-03` to
   `sheisroot/digital-applications-prototype`, including the SQLite artifact,
   manifest, report, application, and backend source files.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) using the GitHub
   account with access to the repository. Click **Create app**, then
   **Yup, I have an app** (or the GitHub deployment option).
3. Set repository to `sheisroot/digital-applications-prototype`, branch to
   `demo/evidence-explorer-2026-11-03`, and main file path to
   `apps/evidence_explorer/app.py`. Enter these manually if absent from suggestions.
4. Choose a memorable available app subdomain, for example
   `implementation-evidence-explorer`. Its assigned `https://….streamlit.app`
   address is the shareable URL; record the actual URL after deployment.
5. Open **Advanced settings**, select **Python 3.12**, leave **Secrets empty**,
   and save. Click **Deploy** and inspect the build logs.
6. Rehearse the flow below on the deployed app, including the source links. Keep
   this branch and app available after the meeting so the URL continues to work.

These settings follow Streamlit's [deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
The requirements file lives beside the entrypoint intentionally: Cloud searches
that directory before the repository root, so it uses the minimal demo requirements
instead of the root pipeline `uv.lock`. See the official
[dependency resolution order](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
The root `.streamlit/config.toml` supplies the theme and disables usage telemetry.

GitHub visibility and app visibility are separate decisions. Public repositories
produce public apps by default. If this repository is private, authorize Streamlit's
GitHub access, then set the deployed app's **Settings → Sharing → Who can view this
app → This app is public and searchable** (or **Share → Make this app public**).
The repository can remain private while the app is public. Check the final URL in
an incognito browser to confirm viewers do not need an invitation. See Streamlit's
[sharing instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).
Public access exposes the demo's displayed public-source content. No repository
visibility or app sharing settings have been changed by this implementation.

## Rehearsal seeds from actual stored-vector rankings

Start with the totals, search **capacity**, and choose **Country → Comoros**.
The recommended seed is **#1157**, on the first results page: select the mechanism
beginning **“Insufficient institutional capacity to operationalize and manage a
specialized payment system…”**. IDs are available in collapsed Technical metadata.
Keep the keyword and country filter while opening related observations; they remain available when you
return to the search selection. Below the supporting evidence, set **Search within
→ Other countries**, then click **Find related observations**. The same control
also offers Full corpus, Other documents, and Other documents and countries,
preserving both restrictions together. Related results use the same label/description/metadata
presentation as browsing results.

| Seed | Grounded evidence | Top cross-country result | Cosine |
| --- | --- | --- | ---: |
| **#1157**, `KM_PAD_2020`, actual: BCC lacked capacity to operationalize and manage ATS+ | Physical PDF p. 21, “for BCC to have support in getting the system live…” | **#1888**, `SL_RESTRUCTURING_2023`, actual: limited payment-stakeholder capacity delayed payments; its own quote is on p. 5 | 0.783 |
| **#1234** (results page 3), `KM_PAD_2020`, actual: low implementation capacity required support | Physical PDF p. 30, “capacity is low” | **#1637**, `SL_PAD_2014`, anticipated: capacity could affect timely implementation; its own quote is on p. 51 | 0.766 |
| **#1280** (results page 4), `KM_PAD_2020`, actual: the World Bank team organized STEP training | Physical PDF p. 56, training quotation | **#1828**, `SL_AF2_2019`, planned: management committed to STEP training; its own quote is on p. 26 | 0.740 |

For the main flow, select the related mechanism **“Insufficient payment personnel
and institutional capacity limited the ability to deliver payments simultaneously
across the country.”** (**#1888**), then show its independent Sierra Leone quote
and physical p. 5 in Supporting evidence. The explicit scope
difference in the second and third alternatives illustrates why similarity alone
cannot establish corroboration. Finish by opening Advanced filters and changing
**Document type → ISR**.
No seed, country, keyword, or related result is hard-coded into the app.

## Verification and limits

```sh
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run python -m unittest discover -s tests -p 'test_explorer*.py' -v
uv run python -m unittest discover -s tests -v
```

Focused tests cover selected-run scope, eligibility, lossless integer-ID vector
joins, stale/ambiguous embeddings, frozen contents and checksums, read-only behavior,
all metadata filters, literal keyword search, norms/ranking/self-exclusion, both
comparison constraints, public URL mapping, and the rehearsed Streamlit interaction
flow. Browser smoke testing also checks source links and filter switching.

Implementation verification on November 2, 2026:

- Ruff check and format check passed; Pyright reported zero errors or warnings.
- All 13 focused explorer tests passed. The full 160-test repository suite passed,
  including the persistence and embedding tests.
- Every snapshot vector was compared against canonical storage by integer
  observation ID and exact source text: all 1,093 matched without numeric changes.
- A clean Python 3.12 environment with only the entrypoint requirements completed
  the rehearsal without OpenAI, SQLAlchemy, secrets, canonical data, or local PDFs.
- The local Streamlit server and headless Chrome completed totals, keyword search,
  Comoros #1157, cross-country similarity, Sierra Leone #1888 and independent
  evidence, source URL inspection, and switching the document-type filter to ISR.

The corpus is small and illustrative, and extracted claims are not independently
adjudicated. Existing quote-location/spacing limitations are documented in
[Stage 1 revalidation](stage1-revalidation.md#deferred-evidence-location-and-spacing-work).
Source availability and PDF-viewer page-fragment behavior depend on the World Bank
host. Cloud deployment and anonymous access must still be verified in the user's
Streamlit account; this task prepares the deployment without publishing it.

## Temporary demo factor labels

**Demo-only presentation labels. Not canonical taxonomy categories.** The nullable
`observations.factor_label` exists only in the frozen demo snapshot (schema v2).
The snapshot exporter creates this empty column; it does not copy labels into or
from Stage 1. Rebuilding the snapshot requires regenerating its presentation labels.
The extraction schema, extraction prompt, canonical database, factor descriptions,
and original embedding values remain unchanged.

The optional generation command uses the existing structured LLM service and the
configured `LLM_MODEL`/`OPENAI_API_KEY`. Only an integer observation ID and that
observation's `factor_description` enter each request. Requests are independent,
with up to eight in flight; no metadata, neighboring descriptions, quotes, or
embeddings enter generation. No web lookup is performed. SDK retries are disabled
so audit attempts correspond to API request attempts, including connection failures.

```sh
uv run python -m scripts.generate_demo_factor_labels --sample
# Inspect all 30 printed pairs for fidelity, specificity, and noun-phrase style.
# Then run the remaining observations:
uv run python -m scripts.generate_demo_factor_labels
# Inspect statistics without invoking the provider:
uv run python -m scripts.generate_demo_factor_labels --report-only
```

The sample is selected offline by round-robin across country/type/scope strata.
Selection metadata never enters model requests. Each successful label is committed
immediately. Rerunning skips populated labels; failures retain null labels and are
retried on the next run. There is deliberately no overwrite mode. The small
`demo_factor_label_audit` table records per-observation attempts, model, prompt
version, description hash, status, error, timestamps, and available token usage.
Interrupted requests remain `started` in the audit and their unlabeled observations
are eligible for retry. The JSON quality report records exact duplicate labels
without merging or normalizing them. Currency cost is not supplied by this API.
`demo/evidence.json` is refreshed to describe the updated frozen artifact.

Browsing and related cards show label, full factor description, then metadata.
A title tooltip explains the demo label. The selected pane retains the exact
observation, full factor description, evidence, metadata, and related results.
Literal search and cosine similarity keep their existing inputs; labels never
enter embeddings, similarity ranking, grouping, or deduplication.

Focused generation and fallback tests:

```sh
uv run python -m unittest discover -s tests -p 'test_demo_factor_labels.py' -v
```

Current generation status and validation: [demo factor-label report](demo-factor-labels-report.md).
