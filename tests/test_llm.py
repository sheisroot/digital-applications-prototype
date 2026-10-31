import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from jinja2 import UndefinedError
from pydantic import ValidationError

from digital_applications_prototype.core.config import Settings
from digital_applications_prototype.llm.adapters.openai import OpenAIAdapter, StructuredOutputError
from digital_applications_prototype.llm.prompt_renderer import PromptRenderer
from digital_applications_prototype.llm.schemas.smoke import SmokeTestResponse
from digital_applications_prototype.llm.service import LLMService


class PromptRendererTests(unittest.TestCase):
    def test_renders_supplied_variables(self):
        with TemporaryDirectory() as directory:
            Path(directory, "hello.jinja2").write_text("Hello {{ name }}!", encoding="utf-8")
            self.assertEqual(PromptRenderer(directory).render("hello.jinja2", name="Faye"), "Hello Faye!")

    def test_missing_variables_fail(self):
        with self.assertRaises(UndefinedError):
            PromptRenderer().render("smoke.jinja2")

    def test_bundled_template(self):
        self.assertIn("example", PromptRenderer().render("smoke.jinja2", text="example"))


class FakeAdapter:
    def __init__(self):
        self.calls = []

    async def generate_structured(self, *, prompt, response_model, model=None):
        self.calls.append((prompt, response_model, model))
        return response_model(summary="offline result")


class LLMServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_delegates_and_returns_typed_response(self):
        adapter = FakeAdapter()
        result = await LLMService(adapter).generate_structured(
            prompt="example", response_model=SmokeTestResponse, model="override"
        )
        self.assertIsInstance(result, SmokeTestResponse)
        self.assertEqual(result.summary, "offline result")
        self.assertEqual(adapter.calls, [("example", SmokeTestResponse, "override")])

    async def test_default_model_is_delegated_as_none(self):
        adapter = FakeAdapter()
        await LLMService(adapter).generate_structured(prompt="example", response_model=SmokeTestResponse)
        self.assertIsNone(adapter.calls[0][2])


class OpenAIAdapterTests(unittest.IsolatedAsyncioTestCase):
    def make_adapter(self, **changes):
        values = dict(id="offline", status="completed", output=[], incomplete_details=None,
                      output_parsed=SmokeTestResponse(summary="parsed"))
        values.update(changes)
        self.parse = AsyncMock(return_value=SimpleNamespace(**values))
        return OpenAIAdapter(SimpleNamespace(responses=SimpleNamespace(parse=self.parse)), model="default")

    async def test_sdk_arguments_and_model_selection(self):
        adapter = self.make_adapter()
        for model, expected in [(None, "default"), ("override", "override")]:
            result = await adapter.generate_structured(prompt="hello", response_model=SmokeTestResponse, model=model)
            self.assertIsInstance(result, SmokeTestResponse)
            self.parse.assert_awaited_with(model=expected, input="hello", text_format=SmokeTestResponse)

    async def test_missing_output_raises(self):
        adapter = self.make_adapter(output_parsed=None)
        with self.assertRaisesRegex(StructuredOutputError, "no parsed output"):
            await adapter.generate_structured(prompt="hello", response_model=SmokeTestResponse)

    async def test_incomplete_output_raises(self):
        adapter = self.make_adapter(status="incomplete")
        with self.assertRaisesRegex(StructuredOutputError, "incomplete"):
            await adapter.generate_structured(prompt="hello", response_model=SmokeTestResponse)

    async def test_refusal_raises(self):
        message = SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal="declined")])
        adapter = self.make_adapter(output=[message], output_parsed=None)
        with self.assertRaisesRegex(StructuredOutputError, "refused"):
            await adapter.generate_structured(prompt="hello", response_model=SmokeTestResponse)

    async def test_provider_errors_propagate(self):
        adapter = self.make_adapter()
        error = RuntimeError("provider unavailable")
        self.parse.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            await adapter.generate_structured(prompt="hello", response_model=SmokeTestResponse)
        self.assertIs(raised.exception, error)


class SettingsTests(unittest.TestCase):
    def test_environment_and_secret_redaction(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret", "LLM_MODEL": "test-model"}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.llm_model, "test-model")
        self.assertNotIn("test-secret", repr(settings))

    def test_missing_configuration_fails(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValidationError):
            Settings.from_env()
