"""Focused demo presentation boundary and restart/failure checks."""

import json
import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel
from test_explorer import seed_source

from digital_applications_prototype.explorer.data import (
    load_corpus,
    related_observations,
)
from digital_applications_prototype.explorer.demo_factor_labels import (
    DemoLabelInput,
    DemoLabelOutput,
    demo_database,
    generate_demo_factor_labels,
    generation_prompt,
    pending_items,
    quality_report,
)
from digital_applications_prototype.explorer.snapshot import create_snapshot
from digital_applications_prototype.llm.adapters.base import T, TokenUsage
from digital_applications_prototype.llm.service import LLMService


class LabelAdapter:
    model = "fake-demo-model"
    last_usage = TokenUsage(input_tokens=20, output_tokens=10)

    def __init__(self, mode: str = "success") -> None:
        self.prompts: list[str] = []
        self.mode = mode

    async def generate_structured(
        self, *, prompt: str, response_model: type[T], model: str | None = None
    ) -> T:
        self.prompts.append(prompt)
        item = json.loads(prompt.split("Input:\n")[1])
        if self.mode == "failure":
            raise RuntimeError("simulated provider failure")
        result: BaseModel = DemoLabelOutput(
            observation_id=item["observation_id"]
            + (1 if self.mode == "wrong-id" else 0),
            factor_label="Limited monitoring capacity",
        )
        return response_model.model_validate(result.model_dump())


class DemoLabelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name) / "stage1.sqlite"
        self.snapshot = Path(temporary.name) / "demo.sqlite"
        seed_source(self.source)
        self.original = self.source.read_bytes()
        create_snapshot(self.source, self.snapshot, ["run-a", "run-b"], {})

    def tearDown(self) -> None:
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_only_description_and_identity_in_payload(self) -> None:
        item = DemoLabelInput(
            observation_id=10, factor_description="Monitoring capacity"
        )
        payload = json.loads(generation_prompt(item).split("Input:\n")[1])
        self.assertEqual(payload, item.model_dump())
        self.assertEqual(set(payload), {"observation_id", "factor_description"})
        for forbidden in ("country", "project", "quote", "actors", 'observation"'):
            self.assertNotIn(forbidden, json.dumps(payload))

    def test_canonical_database_rejected_without_writes(self) -> None:
        with (
            self.assertRaisesRegex(ValueError, "Only frozen demo"),
            demo_database(self.source),
        ):
            self.fail("Canonical schema must be rejected")
        missing = self.snapshot.parent / "missing.sqlite"
        with self.assertRaises(sqlite3.OperationalError), demo_database(missing):
            pass
        self.assertFalse(missing.exists())

    async def test_snapshot_labels_identity_restart_and_unchanged_similarity(
        self,
    ) -> None:
        before = load_corpus(self.snapshot)
        self.assertTrue(all(o.factor_label is None for o in before.observations))
        with demo_database(self.snapshot) as connection:
            items = pending_items(connection)
            adapter = LabelAdapter()
            service = LLMService(adapter)
            await generate_demo_factor_labels(connection, service, items, concurrency=2)
            self.assertEqual(len(adapter.prompts), 5)
            self.assertEqual(pending_items(connection), [])
            await generate_demo_factor_labels(connection, service, items)
            self.assertEqual(len(adapter.prompts), 5)
            self.assertEqual(quality_report(connection)["requests"], 5)
            hashes = connection.execute(
                "SELECT observation_id, description_hash FROM demo_factor_label_audit"
            ).fetchall()
            self.assertEqual({r[0] for r in hashes}, {10, 20, 30, 40, 80})
        after = load_corpus(self.snapshot)
        for observation in after.observations:
            self.assertEqual(observation.display_title, "Limited monitoring capacity")
        self.assertEqual(
            [
                (o.observation_id, score)
                for o, score in related_observations(before, 10)
            ],
            [(o.observation_id, score) for o, score in related_observations(after, 10)],
        )
        self.assertTrue((before.vectors == after.vectors).all())

    async def test_failure_wrong_identity_null_fallback_and_retry(self) -> None:
        with demo_database(self.snapshot) as connection:
            items = pending_items(connection)
            for mode in ("failure", "wrong-id"):
                await generate_demo_factor_labels(
                    connection, LLMService(LabelAdapter(mode)), items[:1]
                )
            self.assertEqual(len(pending_items(connection)), 5)
            self.assertEqual(
                quality_report(connection)["attempts_by_status"], {"failed": 2}
            )
            original = load_corpus(self.snapshot).get(10)
            self.assertEqual(original.display_title, original.factor_description)
            self.assertEqual(
                replace(original, factor_label="").display_title,
                original.factor_description,
            )
            await generate_demo_factor_labels(
                connection, LLMService(LabelAdapter()), items[:1]
            )
            self.assertEqual(len(pending_items(connection)), 4)
        self.assertEqual(len(load_corpus(self.snapshot).observations), 5)

    async def test_input_identity_text_validation(self) -> None:
        with demo_database(self.snapshot) as connection:
            adapter = LabelAdapter()
            with self.assertRaisesRegex(ValueError, "identity/text"):
                await generate_demo_factor_labels(
                    connection,
                    LLMService(adapter),
                    [DemoLabelInput(observation_id=10, factor_description="Changed")],
                )
            self.assertEqual(adapter.prompts, [])

    def test_migrate_legacy_demo_only(self) -> None:
        with closing(sqlite3.connect(self.snapshot)) as connection, connection:
            connection.execute("ALTER TABLE observations DROP COLUMN factor_label")
            connection.execute("PRAGMA user_version = 1")
        with demo_database(self.snapshot) as connection:
            self.assertEqual(len(pending_items(connection)), 5)
            self.assertTrue(
                all(
                    r[0] is None
                    for r in connection.execute("SELECT factor_label FROM observations")
                )
            )
        self.assertEqual(len(load_corpus(self.snapshot).observations), 5)


if __name__ == "__main__":
    unittest.main()
