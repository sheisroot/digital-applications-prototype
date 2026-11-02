"""Exercise the real Streamlit UI against the committed frozen corpus."""

import unittest
from hashlib import sha256
from pathlib import Path

from streamlit.testing.v1 import AppTest

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
        self.assertEqual([m.value for m in app.metric], ["1,093", "15", "3"])
        app.text_input(key="keyword").set_value("capacity").run()
        app.multiselect(key="filter_country").select("comoros").run()
        app.radio(key="result_choice").set_value(1157).run()
        self.assertEqual(app.session_state["selected_id"], 1157)
        self.assertIn("Claim scope: Actual", app.success[0].value)
        self.assertIn("getting the system live", app.text[0].value)
        self.assertTrue(any("Physical PDF page 21" in m.value for m in app.markdown))
        app.checkbox(key="other_countries").check().run()
        app.checkbox(key="other_documents").check().run()
        app.button(key="find_related").click().run()
        self.assertEqual(
            len([b for b in app.button if b.key and b.key.startswith("related_")]), 5
        )
        app.button(key="related_1888").click().run()
        self.assertEqual(app.session_state["selected_id"], 1888)
        self.assertIn("limited capacity of NaCSA", app.text[0].value)
        self.assertTrue(any("Physical PDF page 5" in m.value for m in app.markdown))
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
        self.assertIn("Claim scope: Planned", app.warning[0].value)
        app.text_input(key="keyword").set_value("no-such-evidence-00000").run()
        self.assertTrue(any("No observations match" in i.value for i in app.info))
        self.assertEqual(len(app.radio), 0)
        app.text_input(key="keyword").set_value("").run()
        self.assertEqual(len(app.radio), 1)
        self.assertIn("Claim scope: Planned", app.warning[0].value)


if __name__ == "__main__":
    unittest.main()
