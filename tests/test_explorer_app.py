"""Exercise the real Streamlit UI against the committed frozen corpus."""

import unittest
from hashlib import sha256
from pathlib import Path

from streamlit.testing.v1 import AppTest

from digital_applications_prototype.explorer.data import load_corpus

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
        app.button(key="browse_1157").click().run()
        self.assertEqual(app.session_state["selected_id"], 1157)
        self.assertTrue(any(":green-badge[Actual]" in m.value for m in app.markdown))
        self.assertIn("getting the system live", app.text[0].value)
        self.assertTrue(any("Physical PDF page 21" in c.value for c in app.caption))
        app.selectbox(key="related_scope").select("Other documents and countries").run()
        app.button(key="find_related").click().run()
        self.assertEqual(
            len([b for b in app.button if b.key and b.key.startswith("related_")]), 5
        )
        app.button(key="related_1888").click().run()
        self.assertEqual(app.session_state["selected_id"], 1888)
        self.assertIn("limited capacity of NaCSA", app.text[0].value)
        self.assertTrue(any("Physical PDF page 5" in c.value for c in app.caption))
        self.assertTrue(any("SL_RESTRUCTURING_2023" in c.value for c in app.caption))
        self.assertEqual(app.multiselect(key="filter_country").value, ["comoros"])
        self.assertEqual(app.text_input(key="keyword").value, "capacity")
        next(
            b for b in app.button if b.label == "Back to search selection"
        ).click().run()
        self.assertEqual(app.session_state["selected_id"], 1157)
        app.multiselect(key="filter_document_type").select("ISR").run()
        self.assertNotEqual(app.session_state["selected_id"], 1157)
        self.assertTrue(any("ISR" in c.value for c in app.caption))

    def test_scope_pagination_and_empty_filter_recovery(self):
        app = self.app
        app.number_input(key="results_page").set_value(100).run()
        app.multiselect(key="filter_claim_scope").select("planned").run()
        self.assertEqual(app.number_input(key="results_page").value, 1)
        self.assertTrue(any(":blue-badge[Planned]" in m.value for m in app.markdown))
        app.text_input(key="keyword").set_value("no-such-evidence-00000").run()
        self.assertTrue(any("No observations match" in i.value for i in app.info))
        self.assertFalse(any(b.key and b.key.startswith("browse_") for b in app.button))
        app.text_input(key="keyword").set_value("").run()
        self.assertTrue(any(b.key and b.key.startswith("browse_") for b in app.button))
        self.assertTrue(any(":blue-badge[Planned]" in m.value for m in app.markdown))

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
        app.button(key="browse_1157").click().run()
        self.assertEqual(app.button(key="browse_1157").label, seed.factor_description)
        for scope in (
            "Full corpus",
            "Other documents",
            "Other countries",
            "Other documents and countries",
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
                    self.assertEqual(button.label, candidate.factor_description)
                    self.assertNotEqual(candidate.observation_id, seed.observation_id)
                    if scope in {"Other documents", "Other documents and countries"}:
                        self.assertNotEqual(candidate.document_id, seed.document_id)
                    if scope in {"Other countries", "Other documents and countries"}:
                        self.assertNotEqual(candidate.country, seed.country)


if __name__ == "__main__":
    unittest.main()
