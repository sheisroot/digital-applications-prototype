"""Exercise the real Streamlit UI against the committed frozen corpus."""

import runpy
import unittest
from dataclasses import replace
from datetime import date
from hashlib import sha256
from pathlib import Path

from streamlit.testing.v1 import AppTest

from digital_applications_prototype.explorer.data import (
    load_corpus,
    related_observations,
)

ROOT = Path(__file__).resolve().parents[1]


class ExplorerAppTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = ROOT / "demo/evidence.sqlite"
        self.digest = sha256(self.snapshot.read_bytes()).hexdigest()
        self.app = AppTest.from_file(
            str(ROOT / "apps/evidence_explorer/app.py"), default_timeout=20
        ).run()

    def tearDown(self):
        self.assertFalse(list(self.app.exception))
        self.assertEqual(sha256(self.snapshot.read_bytes()).hexdigest(), self.digest)

    def select_seed(self, observation_id=1157):
        app = self.app
        for page in range(1, int(app.number_input(key="results_page").max or 1) + 1):
            app.number_input(key="results_page").set_value(page).run()
            if any(b.key == f"browse_{observation_id}" for b in app.button):
                app.button(key=f"browse_{observation_id}").click().run()
                return
        self.fail("Seed missing from browsing results")

    def test_demo_label_and_null_fallback_card_hierarchy(self):
        app = AppTest.from_string(
            f"""
import runpy
from dataclasses import replace
namespace = runpy.run_path({str(Path("apps/evidence_explorer/app.py").resolve())!r})
corpus = namespace["cached_corpus"]()
labeled = replace(corpus.observations[0], factor_label="Demo mechanism label")
fallback = replace(corpus.observations[1], factor_label=None)
namespace["render_card"](labeled, browsing=True)
namespace["render_card"](fallback, score=0.75)
""",
            default_timeout=20,
        ).run()
        self.assertFalse(list(app.exception))
        corpus = load_corpus(self.snapshot)
        self.assertEqual(app.button[0].label, "Demo mechanism label")
        self.assertEqual(app.button[1].label, "Unlabelled factor · Compare →")
        self.assertFalse(
            any(
                m.value == corpus.observations[0].factor_description
                for m in app.markdown
            )
        )
        self.assertTrue(any("Similarity 0.750" in c.value for c in app.caption))

    def test_rehearsed_capacity_flow_and_independent_evidence(self):
        app = self.app
        self.assertTrue(
            any(
                c.value == "1,093 observations · 15 documents · 3 countries"
                for c in app.caption
            )
        )
        self.assertEqual(len(app.metric), 0)
        app.text_input(key="keyword").set_value("capacity").run()
        app.multiselect(key="filter_country").select("comoros").run()
        self.select_seed()
        self.assertEqual(app.session_state["selected_id"], 1157)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Supporting evidence", "Related observations"],
        )
        self.assertFalse(any(e.label == "Technical metadata" for e in app.expander))
        self.assertFalse(
            next(
                e for e in app.expander if e.label == "Structured details"
            ).proto.expanded
        )
        self.assertIn("getting the system live", app.text[0].value)
        self.assertTrue(any("Jun 17, 2020 · Page 21" in c.value for c in app.caption))
        app.selectbox(key="related_scope").select("Different countries").run()
        app.button(key="find_related").click().run()
        self.assertEqual(
            len([b for b in app.button if b.key and b.key.startswith("related_")]), 5
        )
        app.button(key="related_1888").click().run()
        self.assertEqual(app.session_state["selected_id"], 1157)
        app.button(key="promote_comparison").click().run()
        self.assertEqual(app.session_state["selected_id"], 1888)
        self.assertIn("limited capacity of NaCSA", app.text[0].value)
        self.assertTrue(any("Date unknown · Page 5" in c.value for c in app.caption))
        self.assertTrue(any("Restructuring Paper" in c.value for c in app.caption))
        rendered = " ".join(c.value for c in app.caption)
        self.assertFalse(
            any(
                o.document_id in rendered
                for o in load_corpus(self.snapshot).observations
            )
        )
        self.assertEqual(app.multiselect(key="filter_country").value, ["comoros"])
        self.assertEqual(app.text_input(key="keyword").value, "capacity")
        next(
            b for b in app.button if b.label == "Back to search selection"
        ).click().run()
        self.assertEqual(app.session_state["selected_id"], 1157)
        app.multiselect(key="filter_document_type").select("ISR").run()
        self.assertNotEqual(app.session_state["selected_id"], 1157)
        self.assertTrue(
            any(
                "Implementation Status and Results Report" in c.value
                for c in app.caption
            )
        )

    def test_scope_pagination_and_empty_filter_recovery(self):
        app = self.app
        app.number_input(key="results_page").set_value(100).run()
        app.multiselect(key="filter_claim_scope").select("planned").run()
        self.assertEqual(app.number_input(key="results_page").value, 1)
        self.assertEqual(
            load_corpus(self.snapshot)
            .get(app.session_state["selected_id"])
            .claim_scope,
            "planned",
        )

        app.text_input(key="keyword").set_value("no-such-evidence-00000").run()
        self.assertTrue(any("No observations match" in i.value for i in app.info))
        self.assertFalse(any(b.key and b.key.startswith("browse_") for b in app.button))
        app.text_input(key="keyword").set_value("").run()
        self.assertTrue(any(b.key and b.key.startswith("browse_") for b in app.button))
        self.assertEqual(
            load_corpus(self.snapshot)
            .get(app.session_state["selected_id"])
            .claim_scope,
            "planned",
        )

    def test_comparison_pair_content_and_explicit_promotion(self):
        app = self.app
        corpus = load_corpus(self.snapshot)
        anchor, candidate = corpus.get(1280), corpus.get(1828)
        self.assertNotEqual(anchor.claim_scope, candidate.claim_scope)
        app.text_input(key="keyword").set_value(anchor.factor_description).run()
        app.button(key="browse_1280").click().run()
        app.selectbox(key="related_scope").select("Different countries").run()
        app.button(key="find_related").click().run()
        before = {
            key: app.session_state[key]
            for key in ("selected_id", "browse_id", "results_page", "related_seed")
        }
        related_keys = [b.key for b in app.button if str(b.key).startswith("related_")]
        app.button(key="related_1828").click().run()
        for key, value in before.items():
            self.assertEqual(app.session_state[key], value)
        dialog = app.get("dialog")[0]
        self.assertEqual(dialog.proto.dialog.title, "Compare implementation evidence")
        self.assertEqual(
            [m.value for m in dialog.markdown],
            [
                f"### {anchor.factor_label}",
                anchor.factor_description,
                "**Supporting evidence**",
                f"### {candidate.factor_label}",
                candidate.factor_description,
                "**Supporting evidence**",
            ],
        )
        self.assertEqual(
            [t.value for t in dialog.text],
            [anchor.evidence[0].quote, candidate.evidence[0].quote],
        )
        score = next(
            score
            for observation, score in related_observations(
                corpus, anchor.observation_id, other_countries=True
            )
            if observation.observation_id == candidate.observation_id
        )
        captions = " ".join(c.value for c in dialog.caption)
        self.assertIn(f"Semantic similarity: {score:.3f}", captions)
        self.assertIn("Additional Financing · Jun 10, 2019", captions)
        self.assertIn(
            "does not imply equivalence, corroboration, or causality", captions
        )
        for observation, link in zip(
            (anchor, candidate), dialog.get("link_button"), strict=True
        ):
            self.assertEqual(
                link.proto.url,
                f"{observation.source_url}#page={observation.evidence[0].page}",
            )
        for observation in corpus.observations:
            self.assertNotIn(observation.document_id, captions)
        # A normal rerun after dismissal cannot reopen/promote a comparison.
        app.run()
        self.assertEqual(app.session_state["selected_id"], anchor.observation_id)
        self.assertFalse(app.get("dialog"))
        self.assertEqual(
            [b.key for b in app.button if str(b.key).startswith("related_")],
            related_keys,
        )
        app.button(key="related_1828").click().run()
        app.button(key="promote_comparison").click().run()
        self.assertEqual(app.session_state["selected_id"], candidate.observation_id)
        self.assertEqual(app.session_state["browse_id"], anchor.observation_id)
        self.assertEqual(app.text_input(key="keyword").value, anchor.factor_description)
        self.assertFalse(app.get("dialog"))
        self.assertTrue(
            any(m.value == candidate.factor_description for m in app.markdown)
        )

    def test_all_filters_and_comparison_choices_preserve_scope_and_meaning(self):
        app = self.app
        corpus = load_corpus(self.snapshot)
        seed = corpus.get(1157)
        app.text_input(key="keyword").set_value("capacity").run()
        for field, value in (
            ("country", seed.country),
            ("observation_type", seed.observation_type),
            ("claim_scope", seed.claim_scope),
            ("project_id", seed.project_id),
            ("document_id", seed.document_id),
            ("document_type", seed.document_type),
            ("valence", seed.valence),
        ):
            app.multiselect(key=f"filter_{field}").select(value).run()
        self.select_seed()
        self.assertEqual(app.button(key="browse_1157").label, seed.display_title)
        for scope in (
            "Anywhere in the corpus",
            "Different documents",
            "Different countries",
        ):
            with self.subTest(scope=scope):
                app.selectbox(key="related_scope").select(scope).run()
                app.button(key="find_related").click().run()
                buttons = [
                    b for b in app.button if b.key and b.key.startswith("related_")
                ]
                self.assertEqual(len(buttons), 5)
                for button in buttons:
                    candidate = corpus.get(
                        int(str(button.key).removeprefix("related_"))
                    )
                    self.assertEqual(
                        button.label, f"{candidate.display_title} · Compare →"
                    )
                    self.assertNotEqual(candidate.observation_id, seed.observation_id)
                    if scope == "Different documents":
                        self.assertNotEqual(candidate.document_id, seed.document_id)
                    if scope == "Different countries":
                        self.assertNotEqual(candidate.country, seed.country)

    def test_pagination_arrows_and_related_tooltip(self):
        app = self.app
        first = app.session_state["selected_id"]
        self.assertTrue(app.button(key="previous_page").disabled)
        app.button(key="next_page").click().run()
        self.assertEqual(app.number_input(key="results_page").value, 2)
        self.assertNotEqual(app.session_state["selected_id"], first)
        app.button(key="previous_page").click().run()
        self.assertEqual(app.number_input(key="results_page").value, 1)
        self.assertEqual(app.session_state["selected_id"], first)
        last = int(app.number_input(key="results_page").max or 1)
        app.number_input(key="results_page").set_value(last).run()
        self.assertTrue(app.button(key="next_page").disabled)
        self.assertEqual(app.button(key="find_related").label, "Search")
        self.assertEqual(
            app.selectbox(key="related_scope").help,
            "Semantic similarity not equivalence or corroboration.",
        )
        self.assertFalse(any("Semantic similarity" in c.value for c in app.caption))
        self.assertEqual(len(app.tabs[0].expander), 1)
        self.assertEqual(len(app.tabs[1].expander), 0)
        corpus = load_corpus(self.snapshot)
        selected = corpus.get(app.session_state["selected_id"])
        self.assertTrue(
            any(m.value == f"### {selected.factor_label}" for m in app.markdown)
        )
        country = selected.country.replace("_", " ").replace("-", " ").title()
        self.assertTrue(
            any(
                c.value.startswith(f"{country} · Project {selected.project_id} · ")
                for c in app.caption
            )
        )

    def test_sort_order_unknown_dates_and_document_grouping(self):
        namespace = runpy.run_path(str(ROOT / "apps/evidence_explorer/app.py"))
        sort = namespace["sort_observations"]
        seed = load_corpus(self.snapshot).get(1157)
        # Deliberately unordered input, including tied dates across documents.
        records = (
            replace(seed, observation_id=9, document_id="B", document_date=None),
            replace(
                seed, observation_id=7, document_id="B", document_date="2024-01-01"
            ),
            replace(
                seed,
                observation_id=5,
                document_id="A",
                document_date="2024-01-01",
                evidence=(replace(seed.evidence[0], page=10),),
            ),
            replace(
                seed,
                observation_id=4,
                document_id="A",
                document_date="2024-01-01",
                evidence=(replace(seed.evidence[0], page=2),),
            ),
            replace(
                seed,
                observation_id=3,
                document_id="A",
                document_date="2024-01-01",
                evidence=(replace(seed.evidence[0], page=2),),
            ),
            replace(
                seed, observation_id=1, document_id="C", document_date="2020-01-01"
            ),
        )
        self.assertEqual([o.observation_id for o in sort(records)], [3, 4, 5, 7, 1, 9])
        self.assertEqual(
            [o.observation_id for o in sort(records, "Oldest first")],
            [1, 3, 4, 5, 7, 9],
        )
        self.assertEqual(sort(records), sort(tuple(reversed(records))))
        self.assertEqual(
            [o.document_id for o in sort(records, "Document order")],
            ["A", "A", "A", "B", "B", "C"],
        )
        app = self.app
        corpus = load_corpus(self.snapshot)
        first = corpus.get(app.session_state["selected_id"])
        self.assertEqual(
            date.fromisoformat(first.document_date or "0001-01-01"), date(2025, 3, 31)
        )
        app.number_input(key="results_page").set_value(2).run()
        app.selectbox(key="browse_sort").select("Oldest first").run()
        self.assertEqual(app.number_input(key="results_page").value, 1)

    def test_evidence_spans_are_separate_and_raw_observation_is_collapsed(self):
        app = self.app
        corpus = load_corpus(self.snapshot)
        seed = next(o for o in corpus.observations if len(o.evidence) > 1)
        app.session_state["selected_id"] = seed.observation_id
        app.run()
        structured = next(e for e in app.expander if e.label == "Structured details")
        self.assertTrue(any(m.value == seed.observation for m in structured.markdown))
        self.assertFalse(structured.proto.expanded)
        evidence_tab = app.tabs[0]
        self.assertEqual(
            [t.value for t in evidence_tab.text], [s.quote for s in seed.evidence]
        )
        self.assertEqual(len(evidence_tab.get("link_button")), len(seed.evidence))

    def test_comparison_preview_keeps_multiple_spans_compact(self):
        app = AppTest.from_string(
            f"""
import runpy
namespace = runpy.run_path({str(ROOT / "apps/evidence_explorer/app.py")!r})
corpus = namespace["cached_corpus"]()
observation = next(o for o in corpus.observations if len(o.evidence) > 1)
namespace["render_comparison_factor"](observation, side="anchor")
""",
            default_timeout=20,
        ).run()
        self.assertFalse(list(app.exception))
        observation = next(
            o for o in load_corpus(self.snapshot).observations if len(o.evidence) > 1
        )
        self.assertEqual([t.value for t in app.text], [observation.evidence[0].quote])
        self.assertEqual(len(app.get("link_button")), 1)
        self.assertTrue(
            any(
                c.value == f"+ {len(observation.evidence) - 1} more supporting spans"
                for c in app.caption
            )
        )


if __name__ == "__main__":
    unittest.main()
