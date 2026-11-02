"""Streamlit entrypoint: frozen evidence only, with no pipeline/provider imports."""

import sqlite3
import sys
from pathlib import Path

import streamlit as st

# Community Cloud installs this directory's minimal requirements, not the pipeline.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from digital_applications_prototype.explorer.data import (
    Corpus,
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
SCOPE_COLORS = {
    "actual": "green",
    "planned": "blue",
    "anticipated": "orange",
    "recommended": "violet",
    "inferred": "gray",
}


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


def render_card(
    observation: Observation,
    *,
    browsing: bool = False,
    selected: bool = False,
    score: float | None = None,
) -> None:
    """Use the same mechanism-first presentation for browsing and related results."""
    prefix = "browse" if browsing else "related"
    st.button(
        observation.factor_description,
        key=f"{prefix}_{observation.observation_id}",
        type="secondary" if selected else "tertiary",
        width="stretch",
        wrap=True,
        on_click=select_browse_observation if browsing else open_observation,
        args=(observation.observation_id,),
    )
    metadata = (
        f"{humanize(observation.country)} · "
        f"{humanize(observation.observation_type)} · {humanize(observation.claim_scope)}"
    )
    if score is not None:
        metadata += f" · Similarity {score:.3f}"
    st.caption(metadata)


def render_details(observation: Observation) -> None:
    scope = observation.claim_scope
    st.markdown(
        f":gray-badge[{humanize(observation.observation_type)}] "
        f":{SCOPE_COLORS[scope]}-badge[{humanize(scope)}] "
        f":gray-badge[{humanize(observation.country)}]",
        help=f"Claim scope: {humanize(scope)}. {SCOPE_HELP[scope]}",
    )
    st.subheader("Observation")
    st.write(observation.observation)
    st.markdown("**Factor / mechanism**")
    st.write(observation.factor_description)
    metadata = (
        f"{humanize(observation.country)} · Project {observation.project_id} · "
        f"{observation.document_id} · {observation.document_type}"
    )
    if observation.document_date:
        metadata += f" · {observation.document_date}"
    st.caption(metadata)
    with st.expander("Structured details", expanded=False):
        for label, values in (
            ("Actors", observation.actors),
            ("Context", observation.context),
            ("Action / response", observation.action),
        ):
            if values:
                st.markdown(f"**{label}**")
                for value in values:
                    st.write(value)
        if observation.observed_effect:
            st.markdown("**Observed effect**")
            st.write(observation.observed_effect)
        if observation.implementation_area:
            st.markdown("**Implementation area (source extraction label)**")
            st.write(observation.implementation_area)
        st.caption(f"Valence: {humanize(observation.valence)}")

    st.subheader("Supporting evidence")
    for index, span in enumerate(observation.evidence):
        if index:
            st.divider()
        # Text rendering preserves the exact quote and keeps source markup inert.
        st.text(span.quote)
        page_label = f"Physical PDF page {span.page}"
        if span.role != "supports":
            page_label += f" · {humanize(span.role)}"
        st.caption(page_label)
        if observation.source_url:
            st.link_button(
                "View source document", f"{observation.source_url}#page={span.page}"
            )

    st.divider()
    render_related(observation)
    with st.expander("Technical metadata", expanded=False):
        st.caption(f"Observation ID: {observation.observation_id}")
        st.caption(f"Provenance ID: {observation.provenance_id or 'Not recorded'}")


def render_related(observation: Observation) -> None:
    st.subheader("Related observations")
    st.caption("Semantic similarity, not equivalence or corroboration.")
    scope = st.selectbox(
        "Search within",
        (
            "Full corpus",
            "Other documents",
            "Other countries",
            "Other documents and countries",
        ),
        key="related_scope",
        help="Compares the corpus independently of browsing filters.",
    )
    if st.button("Find related observations", key="find_related"):
        st.session_state["related_seed"] = observation.observation_id
    if st.session_state.get("related_seed") != observation.observation_id:
        return
    related = related_observations(
        cached_corpus(),
        observation.observation_id,
        other_documents=scope in {"Other documents", "Other documents and countries"},
        other_countries=scope in {"Other countries", "Other documents and countries"},
    )
    if not related:
        st.info("No related observations meet these constraints.")
    with st.container(key="related_results"):
        for index, (candidate, score) in enumerate(related):
            if index:
                st.divider()
            render_card(candidate, score=score)


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
                    format_func=humanize if field == "valence" else str,
                )
    return filters


def main() -> None:
    st.set_page_config(page_title="Evidence Explorer", page_icon="🔎", layout="wide")
    # Scope styling to the two result lists; retain native keyboard/button behavior.
    st.html("""
        <style>
        .stMainBlockContainer {max-width: 1480px; padding-top: 2rem;}
        .st-key-browse_results button, .st-key-related_results button {
            justify-content: flex-start; text-align: left; height: auto;
        }
        .st-key-browse_results button p, .st-key-related_results button p {
            text-align: left; font-weight: 600; font-size: 1rem; line-height: 1.5;
        }
        .st-key-browse_results hr, .st-key-related_results hr {margin: 0.5rem 0;}
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
    results = filter_observations(corpus.observations, keyword=keyword, filters=filters)
    signature = (keyword, tuple((key, tuple(value)) for key, value in filters.items()))
    if st.session_state.get("filter_signature") != signature:
        st.session_state["filter_signature"] = signature
        st.session_state["results_page"] = 1
        st.session_state.pop("browse_id", None)
    if not results:
        st.info("No observations match. Clear the keyword or adjust a metadata filter.")
        return
    browse, detail = st.columns([0.44, 0.56], gap="large")
    with browse:
        pages = (len(results) + PAGE_SIZE - 1) // PAGE_SIZE
        count_column, page_column = st.columns([3, 1], vertical_alignment="center")
        with page_column:
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
        count_column.caption(f"{len(results):,} results · Page {page} of {pages}")
        visible = results[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
        ids = [o.observation_id for o in visible]
        if st.session_state.get("browse_id") not in ids:
            select_browse_observation(ids[0])
        with st.container(height=740, border=False, key="browse_results"):
            for index, observation in enumerate(visible):
                if index:
                    st.divider()
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
