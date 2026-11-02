# Tuesday demo factor-label implementation report

Demo-only presentation labels. Not canonical taxonomy categories.

Implementation, generation, and the requested sample quality review are complete.
The user explicitly approved sending integer IDs and factor descriptions to the
configured OpenAI API. No observations, quotations, metadata, neighboring
descriptions, or embeddings were sent for generation.

## Generation results

| Measure | Result |
| --- | ---: |
| Usable demo observations | 1,093 |
| Populated labels | 1,093 |
| Remaining failures/nulls | 0 |
| Unique exact labels | 1,092 |
| Duplicate exact label groups | 1 (2 observations; 1 excess duplicate) |
| Mean label words | 5.6002 |
| Median label words | 6 |
| Labels longer than eight words | 8 |
| Labels shorter than two words | 0 |
| API request attempts | 1,124 |
| Successful structured responses | 1,094 |
| Failed connection attempts | 30 |
| Provider-reported input tokens | 241,006 |
| Provider-reported output tokens | 167,194 |
| Total reported tokens | 408,200 |

Configured generation model: **`gpt-6-luna`**. Prompt version: `demo-display-v1`.
The 1,124 attempts comprise 30 initial restricted-environment connection failures,
30 successful sample requests after explicit network approval, 1,063 successful
remaining-corpus requests, and one successful targeted regeneration after review.
The earlier network escalation rejection did not execute a request. SDK retries
were disabled. Currency cost is not exposed by the provider response, so no dollar
estimate is made. The failed connections supplied no token usage.

All observations have exactly one current label. The audit retains the 30 failed
attempts and the replaced label's successful attempt. One label was regenerated to
correct a singular/plural ambiguity; this accounts for 1,094 successful responses
and 1,093 current labels. No observation was removed or merged.

The sole exact duplicate is **“Three-stage geographic, community, and proxy-means
targeting”**, independently generated for **#1449** and **#1465**. Both descriptions
explicitly state the same three targeting steps. The labels were left unchanged;
no normalization or deduplication was performed.

## Quality review

Before full generation, all **30** sample pairs were inspected. The sample covered
Mexico (13), Comoros (12), and Sierra Leone (5); action (11), adaptation (9),
condition (6), and outcome (4); actual (10), planned (7), anticipated (9), and
recommended (4). Mean length was 5.47 words; no sample label exceeded eight words.
The prompt was retained after checking specificity, noun-phrase style, fidelity,
and unintended classification or case-detail leakage.

After full generation, **87 systematically selected pairs** were inspected across
all **15 documents**, all three countries, all four observation types, and all four
requested claim scopes, plus targeted spot checks of weaker labels and the exact
duplicate. Selection combined the original stratified sample, a midpoint row from
each document, every 31st observation, and every length outlier. Selection metadata
was used only offline and never entered a generation request.

No new entities, locations, quantities, or causal claims were found in the final
reviewed labels. This is a representative review, not exhaustive semantic
adjudication of all 1,093 labels.

One initial fidelity ambiguity was corrected: **#1124** originally said “Hiring
coordinators meeting experience and integrity requirements,” while its description
specifies “a project coordinator.” That label was deliberately reset and independently
regenerated from the same original description and prompt. Its final label is
**“Hiring Coordinator Meeting Experience and Integrity Requirements”**, restoring
the singular. Both API attempts remain in the audit. No other introduced-information
case was identified in the inspected material.

The output is useful but has some presentation weaknesses, retained explicitly:

- **#868**, “Expanded outreach and new learning initiative,” omits the description's
  anticipated core-team capacity need. It adds no information but is a weaker
  display handle; the complete description remains immediately underneath.
- **#1250**, “Payment-System Configuration Depends on Capacity, Organized Support,
  and Participation,” reads as a sentence rather than the preferred noun phrase.
- **#1764** and **#1769** are also longer, claim-like headings. Their statements
  remain faithful to their descriptions, including the source's quantities and
  association rather than an inferred causal effect.
- Capitalization and hyphenation vary. These have not been standardized into a
  controlled vocabulary.

The eight labels longer than eight whitespace-separated words are **#778, #870,
#1108, #1212, #1250, #1464, #1764, and #1769**. Their complete labels and source
descriptions are available in the frozen snapshot; IDs are also recorded in the
machine-readable [quality report](../demo/demo_factor_labels_report.json).

## Fifteen representative pairs

These pairs span the three countries, four observation types, and actual/planned/
anticipated/recommended scopes. Descriptions are copied exactly from the snapshot.

| ID / country / type / scope | factor_label | factor_description |
| --- | --- | --- |
| #763 / mexico / outcome / actual | Project-financed correspondent and mobile-phone banking approaches | Project-financed studies developed into correspondent and mobile-phone banking approaches for expanding financial access. |
| #764 / mexico / action / actual | External Donor Coordination for Piloting | Coordination with external donors supported piloting of a correspondent-banking initiative. |
| #776 / mexico / action / recommended | Federation-shared technical, administrative, and training support | Affiliation with capable federations gives small entities shared technical, administrative, and training support. |
| #781 / mexico / condition / anticipated | Risk assessment and due diligence | Wholesale lending depends on adequate risk assessment and due diligence to manage exposure and protect depositors. |
| #789 / mexico / adaptation / planned | Monitoring agent services after government-payment transfer stabilization | Monitoring of broader agent services is sequenced after stabilization of the government-payment transfer system. |
| #1120 / comoros / condition / anticipated | Institutional-Capacity Risk for Implementation and Sustainability | Appraisal identified substantial institutional-capacity risk for implementation and sustainability. |
| #1127 / comoros / adaptation / actual | Extended non-performing-loan classification period | Extending the non-performing-loan classification period adjusted loan treatment in response to the economic shock. |
| #1135 / comoros / outcome / anticipated | Agricultural platform payment data for credit scoring | Agricultural platform payment data is expected to provide information usable in agricultural credit scoring. |
| #1161 / comoros / action / recommended | Recommended shared operating-cost arrangement | A shared-cost arrangement was recommended to distribute the payment platform’s operating expenses across its participants and operator. |
| #1236 / comoros / adaptation / planned | Sequencing activities for capacity and task volume | Sequencing project activities to account for limited capacity and the volume of implementation tasks. |
| #1436 / sierra-leone / action / planned | Planned social safety-net systems component | A dedicated project component is planned to develop systems for implementing social safety net interventions. |
| #1487 / sierra-leone / action / recommended | Proposed platform linking transfers and services | A multi-program operational platform was proposed to connect transfers with health and education service delivery. |
| #1704 / sierra-leone / adaptation / actual | Emergency Recovery Households Added to Safety Net | Households temporarily supported during emergency recovery were incorporated into the ongoing safety net. |
| #1860 / sierra-leone / adaptation / anticipated | Official declaration or order as soft trigger | Emergency cash activation may use an official emergency declaration or order as a relatively soft trigger. |
| #1894 / sierra-leone / outcome / actual | Staff Capacity in Digital Data-Collection Tools | Developing staff capacity in digital data-collection tools strengthened monitoring and oversight capability. |

## Files changed

- `src/digital_applications_prototype/explorer/demo_factor_labels.py`: structured
  independent generation, frozen-snapshot migration, identity checks, incremental
  persistence, restart behavior, attempt audit, and quality statistics.
- `scripts/generate_demo_factor_labels.py`: sample/full/report commands using the
  existing LLM service and provider adapter, with hidden SDK retries disabled.
- `src/digital_applications_prototype/explorer/snapshot.py`: nullable demo-only
  `factor_label` in new exports; canonical source columns remain unchanged.
- `src/digital_applications_prototype/explorer/data.py`: schema v2, label readout,
  and presentation fallback.
- `apps/evidence_explorer/app.py`: browsing and related cards show label then full
  description then metadata/optional similarity; tooltip explains the demo title.
  Null labels use the full description as their title. The detail pane retains all
  source-grounded fields.
- `demo/evidence.sqlite`, `demo/evidence.json`: migrated frozen artifact and
  refreshed size/checksum, with all generation attempts retained in its audit.
- `demo/demo_factor_labels_report.json`: current generation statistics.
- `tests/test_demo_factor_labels.py`, `tests/test_explorer_app.py`: focused
  generation, schema, canonical rejection, payload, identity, restart, failure,
  fallback, card hierarchy, and unchanged-vector/ranking checks.
- `docs/evidence-explorer.md`, this report: temporary scope, commands, and status.

## Canonical and semantic boundary verification

`data/derived/stage1.sqlite` remained untouched. Its SHA-256 before and after:

```text
a2ad94ac6cf0d18143b5c5dd3c976a944f482fe97d6b5e86c9019fa3ea101b3c
```

Every original column in every demo observation, and all rows of `embeddings`,
`evidence_spans`, `documents`, `selected_runs`, and `metadata`, matched the captured
pre-change contents exactly. No extraction or embedding generation was invoked.
Cosine retrieval still uses the original factor-description vectors; labels do not
enter similarity, search inputs, grouping, merging, or deduplication. No web lookup
was performed.

## Checks

Commands used the project uv environment and checked-in configuration. A temporary
`UV_CACHE_DIR=/tmp/evidence-demo-uv-cache` was necessary because the default global
uv cache was outside the writable sandbox.

- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed.
- `uv run pyright`: passed, zero errors/warnings.
- `uv run python -m unittest discover -s tests -p 'test_explorer*.py' -v`: passed,
  14 tests, rerun against the final populated snapshot.
- `uv run python -m unittest discover -s tests -v`: passed, 167 tests, including
  all six new focused generation tests.
