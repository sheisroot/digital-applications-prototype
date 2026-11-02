"""Streamlit entrypoint: frozen evidence only, with no pipeline/provider imports."""

import sqlite3
import sys
from collections.abc import Sequence
from datetime import date
from html import escape
from pathlib import Path

import streamlit as st

# Community Cloud installs this directory's minimal requirements, not the pipeline.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from digital_applications_prototype.explorer.data import (
    Corpus,
    Evidence,
    Observation,
    filter_observations,
    load_corpus,
    related_observations,
)

SNAPSHOT = ROOT / "demo/evidence.sqlite"
PAGE_SIZE = 10
SCOPE_HELP = {
    "actual": "Reported as actual in the source.",
    "planned": "A planned activity; not an observed implementation result.",
    "anticipated": "An anticipated condition or effect; not an observed result.",
    "recommended": "A recommendation; not an observed implementation result.",
    "inferred": "An inferred claim; not an explicitly reported implementation result.",
}
# Restrained, accessible foreground/background pairs shared by all card/detail pills.
PILL_COLORS = {
    "condition": ("#455b70", "#eaf0f5"),
    "action": ("#35645e", "#e8f2ee"),
    "adaptation": ("#66527b", "#f0ebf5"),
    "outcome": ("#785f39", "#f5efe3"),
    "actual": ("#3c6347", "#eaf2e9"),
    "planned": ("#455e86", "#eaf0f9"),
    "anticipated": ("#805c35", "#f8efdf"),
    "recommended": ("#73547a", "#f3ebf4"),
}
NEUTRAL_PILL = ("#555d66", "#eef0f2")
DOCUMENT_TYPES = {
    "PAD": "Project Appraisal Document",
    "ISR": "Implementation Status and Results Report",
    "AF": "Additional Financing",
    "RESTRUCTURING": "Restructuring Paper",
}
SORT_OPTIONS = ("Most recent first", "Oldest first", "Document order")
RELATED_SCOPES = (
    "Anywhere in the corpus",
    "Different documents",
    "Different countries",
)


@st.cache_resource(show_spinner="Loading the frozen evidence corpus…")
def cached_corpus() -> Corpus:
    # Immutable tuples and read-only arrays are safe to share across sessions.
    return load_corpus(SNAPSHOT)


def humanize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def open_observation(observation_id: int) -> None:
    st.session_state["selected_id"] = observation_id


def select_browse_observation(observation_id: int) -> None:
    st.session_state["browse_id"] = observation_id
    open_observation(observation_id)


def document_metadata(observation: Observation) -> str:
    parts = [
        DOCUMENT_TYPES.get(
            observation.document_type, humanize(observation.document_type)
        )
    ]
    if observation.document_date:
        parsed = date.fromisoformat(observation.document_date)
        parts.append(f"{parsed:%b} {parsed.day}, {parsed.year}")
    else:
        parts.append("Date unknown")
    return " · ".join(parts)


def document_label(observation: Observation) -> str:
    return (
        f"{humanize(observation.country)} · Project {observation.project_id} · "
        f"{document_metadata(observation)}"
    )


def change_page(offset: int, pages: int) -> None:
    st.session_state["results_page"] = max(
        1, min(pages, int(st.session_state["results_page"]) + offset)
    )


def sort_observations(
    observations: Sequence[Observation], mode: str = "Most recent first"
) -> tuple[Observation, ...]:
    """Presentation order only; never reorder the corpus/vector matrix itself."""

    def key(observation: Observation) -> tuple[bool, int, str, int, int]:
        ordinal = (
            date.fromisoformat(observation.document_date).toordinal()
            if observation.document_date
            else 0
        )
        if mode == "Document order":
            return (
                False,
                0,
                observation.document_id,
                min(s.page for s in observation.evidence),
                observation.observation_id,
            )
        return (
            not bool(ordinal),
            -ordinal if mode == "Most recent first" else ordinal,
            observation.document_id,
            min(s.page for s in observation.evidence),
            observation.observation_id,
        )

    return tuple(sorted(observations, key=key))


def render_pills(observation: Observation) -> None:
    pills = []
    for value, colors in (
        (observation.country, NEUTRAL_PILL),
        (
            observation.observation_type,
            PILL_COLORS.get(observation.observation_type, NEUTRAL_PILL),
        ),
        (
            observation.claim_scope,
            PILL_COLORS.get(observation.claim_scope, NEUTRAL_PILL),
        ),
    ):
        foreground, background = colors
        title = SCOPE_HELP.get(value, "")
        pills.append(
            f'<span class="evidence-pill" style="color:{foreground};background:{background}" '
            f'title="{escape(title, quote=True)}">{escape(humanize(value))}</span>'
        )
    st.html('<div class="evidence-pills">' + "".join(pills) + "</div>")


def render_card(
    observation: Observation,
    *,
    browsing: bool = False,
    selected: bool = False,
    score: float | None = None,
) -> bool:
    """Return a comparison request; only browse cards change the selection."""
    prefix = "browse" if browsing else "related"
    state = "selected" if selected else "idle"
    with st.container(
        key=f"result_card_{prefix}_{state}_{observation.observation_id}", gap="small"
    ):
        label = observation.factor_label or "Unlabelled factor"
        clicked = st.button(
            label if browsing else f"{label} · Compare →",
            key=f"{prefix}_{observation.observation_id}",
            type="tertiary",
            width="stretch",
            wrap=True,
            on_click=select_browse_observation if browsing else None,
            args=(observation.observation_id,) if browsing else (),
        )
        render_pills(observation)
        if score is not None:
            st.caption(f"Similarity {score:.3f}")
    return clicked


def render_evidence_card(observation: Observation, span: Evidence, *, key: str) -> None:
    with st.container(key=key, gap="small"):
        # Text preserves verbatim quotes and keeps source markup inert.
        st.text(span.quote)
        page_label = f"{document_metadata(observation)} · Page {span.page}"
        if span.role != "supports":
            page_label += f" · {humanize(span.role)}"
        st.caption(page_label)
        if observation.source_url:
            st.link_button(
                "View source document", f"{observation.source_url}#page={span.page}"
            )


def render_comparison_factor(observation: Observation, *, side: str) -> None:
    st.subheader("Selected factor" if side == "anchor" else "Related factor")
    st.markdown(f"### {observation.factor_label or 'Unlabelled factor'}")
    st.write(observation.factor_description)
    render_pills(observation)
    st.caption(document_label(observation))
    st.markdown("**Supporting evidence**")
    render_evidence_card(
        observation,
        observation.evidence[0],
        key=f"evidence_card_compare_{side}_{observation.observation_id}",
    )
    remaining = len(observation.evidence) - 1
    if remaining:
        st.caption(f"+ {remaining} more supporting spans")


@st.dialog("Compare implementation evidence", width="large")
def render_comparison(
    anchor: Observation, candidate: Observation, score: float
) -> None:
    # Dialog arguments retain the pair independently of the main selected_id.
    # Native dismissal does not rerun the app or change any browsing state.
    with st.container(key="comparison_workspace"):
        left, right = st.columns(2, gap="large")
        with left:
            render_comparison_factor(anchor, side="anchor")
        with right:
            render_comparison_factor(candidate, side="candidate")
            if st.button(
                "Make this the selected observation",
                key="promote_comparison",
                on_click=open_observation,
                args=(candidate.observation_id,),
            ):
                st.rerun()
        st.caption(f"Semantic similarity: {score:.3f}")
        st.caption(
            "Similarity indicates related factor descriptions; it does not imply "
            "equivalence, corroboration, or causality."
        )


def render_details(observation: Observation) -> None:
    with st.container(key="selected_anchor"):
        st.markdown(f"### {observation.factor_label or 'Unlabelled factor'}")
        with st.container(key="selected_factor"):
            st.write(observation.factor_description)
        render_pills(observation)
        st.caption(document_label(observation))
    supporting, related = st.tabs(["Supporting evidence", "Related observations"])
    with supporting:
        for index, span in enumerate(observation.evidence):
            render_evidence_card(
                observation,
                span,
                key=f"evidence_card_{observation.observation_id}_{index}",
            )
        render_structured_details(observation)
    with related:
        render_related(observation)


def render_structured_details(observation: Observation) -> None:
    with st.expander("Structured details", expanded=False):
        for label, values in (
            (
                "Observation",
                (observation.observation,) if observation.observation else (),
            ),
            ("Actors", observation.actors),
            ("Context", observation.context),
            ("Action / response", observation.action),
            (
                "Observed effect",
                (observation.observed_effect,) if observation.observed_effect else (),
            ),
            (
                "Implementation area",
                (observation.implementation_area,)
                if observation.implementation_area
                else (),
            ),
        ):
            if values:
                st.markdown(f"**{label}**")
                if len(values) > 1:
                    st.html(
                        "<ul>"
                        + "".join(f"<li>{escape(value)}</li>" for value in values)
                        + "</ul>"
                    )
                else:
                    for value in values:
                        st.write(value)


@st.fragment
def render_related(observation: Observation) -> None:
    # Place results above controls once available, including after scope changes.
    results = st.container(key="related_results")
    scope = st.selectbox(
        "Find related evidence in",
        RELATED_SCOPES,
        key="related_scope",
        help="Semantic similarity not equivalence or corroboration.",
    )
    if st.button("Search", key="find_related"):
        st.session_state["related_seed"] = observation.observation_id
    if st.session_state.get("related_seed") != observation.observation_id:
        return
    related = related_observations(
        cached_corpus(),
        observation.observation_id,
        other_documents=scope == "Different documents",
        other_countries=scope == "Different countries",
    )
    with results:
        if not related:
            st.info("No related observations meet these constraints.")
        for candidate, score in related:
            if render_card(candidate, score=score):
                render_comparison(observation, candidate, score)


def render_filters(corpus: Corpus) -> dict[str, list[str]]:
    filters: dict[str, list[str]] = {}
    basic = (
        ("country", "Country"),
        ("observation_type", "Observation type"),
        ("claim_scope", "Claim scope"),
    )
    advanced = (
        ("project_id", "Project"),
        ("document_id", "Document"),
        ("document_type", "Document type"),
        ("valence", "Valence"),
    )
    for column, (field, label) in zip(st.columns(3), basic, strict=True):
        with column:
            filters[field] = st.multiselect(
                label,
                sorted({str(getattr(o, field)) for o in corpus.observations}),
                key=f"filter_{field}",
                format_func=humanize,
            )
    with st.expander("Advanced filters", expanded=False):
        for column, (field, label) in zip(st.columns(4), advanced, strict=True):
            with column:
                filters[field] = st.multiselect(
                    label,
                    sorted({str(getattr(o, field)) for o in corpus.observations}),
                    key=f"filter_{field}",
                    format_func=(
                        {
                            o.document_id: document_label(o)
                            for o in corpus.observations
                        }.__getitem__
                        if field == "document_id"
                        else (lambda value: DOCUMENT_TYPES.get(value, humanize(value)))
                    ),
                )
    return filters


def main() -> None:
    st.set_page_config(page_title="Evidence Explorer", page_icon="🔎", layout="wide")
    # Scope styling to the two result lists; retain native keyboard/button behavior.
    st.html("""
        <style>
        .stMainBlockContainer {max-width: 1480px; padding-top: 2rem;}
        [class*="st-key-result_card_"] {
            position: relative; padding: 0.7rem 0.8rem;
            border: 1px solid #e5e7eb; border-radius: 8px;
        }
        [class*="st-key-result_card_"]:hover {background: #f5f7f9;}
        [class*="st-key-result_card_browse_selected_"] {
            background: #f0f4f7; border-color: #a9bac8;
        }
        [class*="st-key-result_card_"] div {position: static;}
        [class*="st-key-result_card_"] button {
            position: static; display: block; text-align: left;
            padding: 0; height: auto; min-height: 0; border: 0;
        }
        [class*="st-key-result_card_"] button > div {
            justify-content: flex-start; width: 100%;
        }
        [class*="st-key-result_card_"] button::after {
            content: ""; position: absolute; inset: 0; z-index: 1;
            border-radius: 8px;
        }
        [class*="st-key-result_card_"] button:focus-visible::after {
            outline: 2px solid #54718b; outline-offset: 2px;
        }
        [class*="st-key-result_card_"] [data-testid="stMarkdownContainer"] {width: 100%;}
        [class*="st-key-result_card_"] button p {
            text-align: left; font-weight: 600; font-size: 1rem; line-height: 1.4;
        }
        .st-key-results_navigation [data-testid="stHorizontalBlock"] {flex-wrap: nowrap;}
        .st-key-results_navigation [data-testid="stColumn"] {
            min-width: 0; flex: 1 1 0;
        }
        .st-key-results_navigation [data-testid="stColumn"]:nth-child(2) {flex-grow: 2;}
        .st-key-results_page button {display: none;}
        .st-key-selected_anchor {
            background: #f0f4f7; border: 1px solid #d5dfe5;
            padding: 1.25rem; border-radius: 8px;
        }
        .st-key-selected_anchor h3 {
            font-size: 1.4rem; font-weight: 600; line-height: 1.35; padding: 0;
        }
        .st-key-selected_factor p {
            font-size: 1.1rem; font-weight: 400; line-height: 1.55;
        }
        .evidence-pills {display: flex; flex-wrap: wrap; gap: 0.3rem;}
        .evidence-pill {
            display: inline-block; border-radius: 12px; padding: 0.15rem 0.5rem;
            font-size: 0.75rem; line-height: 1.4;
        }
        [class*="st-key-evidence_card_"] {
            background: #fffdf6; color: #34322d; border: 1px solid #e9dfbd;
            padding: 1rem; border-radius: 8px;
        }
        [class*="st-key-evidence_card_"] [data-testid="stText"] {
            font-family: inherit; white-space: pre-wrap; overflow-wrap: anywhere;
        }
        .st-key-comparison_workspace [data-testid="stColumn"] {min-width: 0;}
        .st-key-comparison_workspace {overflow-wrap: anywhere;}
        @media (max-width: 760px) {
            .st-key-comparison_workspace [data-testid="stHorizontalBlock"] {
                flex-direction: column;
            }
            .st-key-comparison_workspace [data-testid="stColumn"] {width: 100%;}
        }
        @media (max-width: 640px) {
            .stMainBlockContainer {padding: 3.5rem 1rem 1rem;}
            .stMainBlockContainer h1 {font-size: 2rem;}
        }
        </style>
    """)
    st.title("Evidence Explorer")
    st.markdown("Explore implementation observations across projects and countries")
    st.caption(
        "Illustrative prototype based on a small corpus of public World Bank documents."
    )
    try:
        corpus = cached_corpus()
    except (OSError, ValueError, sqlite3.Error):
        st.error(
            "The frozen demo dataset could not be loaded. Check the deployment files."
        )
        st.stop()
    st.caption(
        f"{len(corpus.observations):,} observations · "
        f"{len({o.document_id for o in corpus.observations})} documents · "
        f"{len({o.country for o in corpus.observations})} countries"
    )
    keyword = st.text_input(
        "Keyword search",
        placeholder="Search implementation evidence...",
        key="keyword",
        help="Search observation text, factor descriptions, and extraction area labels.",
        label_visibility="collapsed",
    )
    filters = render_filters(corpus)
    browse, detail = st.columns([0.38, 0.62], gap="large")
    with browse:
        sort_mode = st.selectbox(
            "Sort results",
            SORT_OPTIONS,
            key="browse_sort",
            label_visibility="collapsed",
        )
    results = sort_observations(
        filter_observations(corpus.observations, keyword=keyword, filters=filters),
        sort_mode,
    )
    signature = (
        keyword,
        sort_mode,
        tuple((key, tuple(value)) for key, value in filters.items()),
    )
    if st.session_state.get("filter_signature") != signature:
        st.session_state["filter_signature"] = signature
        st.session_state["results_page"] = 1
        st.session_state.pop("browse_id", None)
    if not results:
        st.info("No observations match. Clear the keyword or adjust a metadata filter.")
        return
    with browse:
        pages = (len(results) + PAGE_SIZE - 1) // PAGE_SIZE
        count_column, page_column = st.columns([1, 1], vertical_alignment="center")
        with page_column, st.container(key="results_navigation"):
            previous, page_input, following = st.columns(
                [1, 2, 1], gap="small", vertical_alignment="center"
            )
            current_page = int(st.session_state.get("results_page", 1))
            with previous:
                st.button(
                    "←",
                    key="previous_page",
                    help="Previous page",
                    width="stretch",
                    disabled=current_page <= 1,
                    on_click=change_page,
                    args=(-1, pages),
                )
            with page_input:
                page = int(
                    st.number_input(
                        "Results page",
                        min_value=1,
                        max_value=pages,
                        key="results_page",
                        label_visibility="collapsed",
                        help=f"Choose a page (1–{pages}).",
                    )
                )
            with following:
                st.button(
                    "→",
                    key="next_page",
                    help="Next page",
                    width="stretch",
                    disabled=current_page >= pages,
                    on_click=change_page,
                    args=(1, pages),
                )
        count_column.caption(f"{len(results):,} results · Page {page} of {pages}")
        visible = results[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
        ids = [o.observation_id for o in visible]
        if st.session_state.get("browse_id") not in ids:
            select_browse_observation(ids[0])
        with st.container(height=740, border=False, key="browse_results"):
            for observation in visible:
                render_card(
                    observation,
                    browsing=True,
                    selected=observation.observation_id
                    == st.session_state["selected_id"],
                )
    with detail:
        if st.session_state["selected_id"] != st.session_state["browse_id"]:
            st.button(
                "Back to search selection",
                key="back_to_results",
                type="tertiary",
                on_click=open_observation,
                args=(int(st.session_state["browse_id"]),),
            )
        render_details(corpus.get(int(st.session_state["selected_id"])))


if __name__ == "__main__":
    main()
