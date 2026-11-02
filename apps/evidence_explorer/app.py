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


@st.cache_resource(show_spinner="Loading the frozen evidence corpus…")
def cached_corpus() -> Corpus:
    # Immutable tuples and read-only arrays are safe to share across sessions.
    return load_corpus(SNAPSHOT)


def humanize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def open_observation(observation_id: int) -> None:
    st.session_state["selected_id"] = observation_id


def select_result() -> None:
    open_observation(st.session_state["result_choice"])


def result_label(observation: Observation) -> str:
    text = observation.observation
    preview = text if len(text) <= 125 else text[:122] + "…"
    return (
        f"#{observation.observation_id} · {humanize(observation.country)} · "
        f"{observation.claim_scope.upper()} · {observation.document_id}\n\n{preview}"
    )


def render_details(observation: Observation) -> None:
    st.caption(f"SELECTED OBSERVATION · #{observation.observation_id}")
    scope_text = f"Claim scope: {humanize(observation.claim_scope)}"
    if observation.claim_scope == "actual":
        st.success(scope_text + " · Reported as actual in the source.")
    else:
        st.warning(
            scope_text + " · Do not read this as an observed implementation result."
        )
    st.subheader("Observation")
    st.write(observation.observation)
    st.markdown("**Factor / mechanism**")
    st.write(observation.factor_description)
    st.caption(
        f"Type: {humanize(observation.observation_type)} · "
        f"Valence: {humanize(observation.valence)}"
    )
    st.markdown(
        f"**{humanize(observation.country)}** · Project **{observation.project_id}**"
    )
    st.caption(
        f"{observation.document_id} · {observation.document_type} · "
        f"Document date: {observation.document_date or 'Not recorded'}"
    )
    if observation.source_url:
        st.link_button("View source document", observation.source_url)
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
        st.caption(f"Provenance ID: {observation.provenance_id or 'Not recorded'}")

    evidence_tab, related_tab = st.tabs(["Supporting evidence", "Related observations"])
    with evidence_tab:
        st.caption(
            "Exact extracted quotations · Page numbers count physical PDF pages from 1."
        )
        for span in observation.evidence:
            with st.container(border=True):
                st.markdown(
                    f"**Physical PDF page {span.page}** · {humanize(span.role)}"
                )
                # Text rendering keeps document-authored markup inert.
                st.text(span.quote)
                st.caption(
                    f"{observation.document_id} · {observation.document_type} · "
                    f"{humanize(observation.country)} · {observation.project_id} · "
                    f"Document date: {observation.document_date or 'Not recorded'}"
                )
                if observation.source_url:
                    st.link_button(
                        "View source document",
                        f"{observation.source_url}#page={span.page}",
                    )
    with related_tab:
        render_related(observation)


def render_related(observation: Observation) -> None:
    st.subheader("Find related observations")
    st.caption(
        "Semantic similarity of factor descriptions, not equivalence or corroboration. "
        "Compares the full frozen corpus independently of browsing filters."
    )
    other_documents = st.checkbox("Other documents only", key="other_documents")
    other_countries = st.checkbox("Other countries only", key="other_countries")
    if st.button("Find related observations", key="find_related", type="primary"):
        st.session_state["related_seed"] = observation.observation_id
    if st.session_state.get("related_seed") != observation.observation_id:
        return
    related = related_observations(
        cached_corpus(),
        observation.observation_id,
        other_documents=other_documents,
        other_countries=other_countries,
    )
    if not related:
        st.info("No related observations meet these constraints.")
    for candidate, score in related:
        with st.container(border=True):
            st.caption(
                f"{humanize(candidate.country)} · {candidate.document_id} · "
                f"Scope: {humanize(candidate.claim_scope)} · Similarity {score:.3f}"
            )
            st.write(candidate.observation)
            st.button(
                f"Inspect evidence · #{candidate.observation_id}",
                key=f"related_{candidate.observation_id}",
                on_click=open_observation,
                args=(candidate.observation_id,),
            )


def main() -> None:
    st.set_page_config(page_title="Evidence Explorer", page_icon="🔎", layout="wide")
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
    totals = st.columns(3)
    totals[0].metric("Observations", f"{len(corpus.observations):,}")
    totals[1].metric("Documents", len({o.document_id for o in corpus.observations}))
    totals[2].metric("Countries", len({o.country for o in corpus.observations}))
    with st.sidebar:
        st.header("Filter observations")
        filters: dict[str, list[str]] = {}
        for field, label in (
            ("country", "Country"),
            ("project_id", "Project"),
            ("document_id", "Document"),
            ("document_type", "Document type"),
            ("observation_type", "Observation type"),
            ("valence", "Valence"),
            ("claim_scope", "Claim scope"),
        ):
            options = sorted({str(getattr(o, field)) for o in corpus.observations})
            filters[field] = st.multiselect(
                label,
                options,
                key=f"filter_{field}",
                format_func=humanize
                if field not in {"document_id", "project_id", "document_type"}
                else str,
            )
        st.caption(
            "Filters use stored extraction metadata. Empty selections include all values."
        )
    keyword = st.text_input(
        "Keyword search",
        placeholder="Try capacity, procurement, or payments",
        key="keyword",
    )
    st.caption(
        "Search observation text, factor descriptions, and extraction area labels."
    )
    results = filter_observations(corpus.observations, keyword=keyword, filters=filters)
    signature = (keyword, tuple((key, tuple(value)) for key, value in filters.items()))
    if st.session_state.get("filter_signature") != signature:
        st.session_state["filter_signature"] = signature
        st.session_state["results_page"] = 1
        st.session_state.pop("result_choice", None)
        if results:
            open_observation(results[0].observation_id)
    if not results:
        st.info("No observations match. Clear the keyword or adjust a metadata filter.")
        return
    browse, detail = st.columns([2, 3], gap="large")
    with browse:
        st.subheader(f"{len(results):,} matching observations")
        pages = (len(results) + PAGE_SIZE - 1) // PAGE_SIZE
        page = int(
            st.number_input(
                "Results page", min_value=1, max_value=pages, key="results_page"
            )
        )
        st.caption(
            f"Page {page} of {pages} · Select an observation to inspect its evidence."
        )
        visible = results[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
        ids = [o.observation_id for o in visible]
        if st.session_state.get("result_choice") not in ids:
            st.session_state["result_choice"] = ids[0]
            open_observation(ids[0])
        with st.container(height=570, border=True):
            st.radio(
                "Observations",
                ids,
                format_func=lambda value: result_label(corpus.get(value)),
                key="result_choice",
                on_change=select_result,
                label_visibility="collapsed",
            )
    selected_id = int(st.session_state["selected_id"])
    with detail:
        if selected_id != st.session_state["result_choice"]:
            st.caption(
                "Inspecting a related observation. Your search and filters are preserved."
            )
            st.button(
                "Back to search selection",
                on_click=open_observation,
                args=(int(st.session_state["result_choice"]),),
            )
        render_details(corpus.get(selected_id))


if __name__ == "__main__":
    main()
