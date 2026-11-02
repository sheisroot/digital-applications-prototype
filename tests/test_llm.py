import asyncio
import os
import unittest
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from jinja2 import UndefinedError
from openai import AsyncOpenAI
from pydantic import ValidationError

from digital_applications_prototype.core import config
from digital_applications_prototype.core.config import Settings
from digital_applications_prototype.llm.adapters.base import TokenUsage
from digital_applications_prototype.llm.adapters.openai import (
    OpenAIAdapter,
    StructuredOutputError,
)
from digital_applications_prototype.llm.prompt_renderer import PromptRenderer
from digital_applications_prototype.llm.schemas.smoke import SmokeTestResponse
from digital_applications_prototype.llm.service import LLMService


class PromptRendererTests(unittest.TestCase):
    def test_renders_supplied_variables(self):
        with TemporaryDirectory() as directory:
            Path(directory, "hello.jinja2").write_text(
                "Hello {{ name }}!", encoding="utf-8"
            )
            self.assertEqual(
                PromptRenderer(directory).render("hello.jinja2", name="Faye"),
                "Hello Faye!",
            )

    def test_missing_variables_fail(self):
        with self.assertRaises(UndefinedError):
            PromptRenderer().render("smoke.jinja2")

    def test_bundled_template(self):
        self.assertIn(
            "example", PromptRenderer().render("smoke.jinja2", text="example")
        )


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
        await LLMService(adapter).generate_structured(
            prompt="example", response_model=SmokeTestResponse
        )
        self.assertIsNone(adapter.calls[0][2])

    async def test_optional_model_and_usage_metadata(self):
        adapter = FakeAdapter()
        service = LLMService(adapter)
        self.assertIsNone(service.model)
        self.assertIsNone(service.last_usage)

        class MetadataAdapter(FakeAdapter):
            model = "offline-model"
            last_usage: TokenUsage | None = None

        adapter = MetadataAdapter()
        service = LLMService(adapter)
        adapter.last_usage = TokenUsage(input_tokens=12, output_tokens=34)
        self.assertEqual(service.model, "offline-model")
        usage = service.last_usage
        assert usage is not None
        self.assertEqual(usage.output_tokens, 34)


class OpenAIAdapterTests(unittest.IsolatedAsyncioTestCase):
    def make_adapter(self, **changes):
        values = {
            "id": "offline",
            "status": "completed",
            "output": [],
            "incomplete_details": None,
            "output_parsed": SmokeTestResponse(summary="parsed"),
        }
        values.update(changes)
        self.parse = AsyncMock(return_value=SimpleNamespace(**values))
        client = Mock(spec=AsyncOpenAI)
        client.responses.parse = self.parse
        return OpenAIAdapter(client, model="default")

    async def test_sdk_arguments_and_model_selection(self):
        adapter = self.make_adapter()
        for model, expected in [(None, "default"), ("override", "override")]:
            result = await adapter.generate_structured(
                prompt="hello", response_model=SmokeTestResponse, model=model
            )
            self.assertIsInstance(result, SmokeTestResponse)
            self.parse.assert_awaited_with(
                model=expected, input="hello", text_format=SmokeTestResponse
            )

    async def test_missing_output_raises(self):
        adapter = self.make_adapter(output_parsed=None)
        with self.assertRaisesRegex(StructuredOutputError, "no parsed output"):
            await adapter.generate_structured(
                prompt="hello", response_model=SmokeTestResponse
            )

    async def test_incomplete_output_raises(self):
        adapter = self.make_adapter(status="incomplete")
        with self.assertRaisesRegex(StructuredOutputError, "incomplete"):
            await adapter.generate_structured(
                prompt="hello", response_model=SmokeTestResponse
            )

    async def test_refusal_raises(self):
        message = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="refusal", refusal="declined")],
        )
        adapter = self.make_adapter(output=[message], output_parsed=None)
        with self.assertRaisesRegex(StructuredOutputError, "refused"):
            await adapter.generate_structured(
                prompt="hello", response_model=SmokeTestResponse
            )

    async def test_provider_errors_propagate(self):
        adapter = self.make_adapter()
        error = RuntimeError("provider unavailable")
        self.parse.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            await adapter.generate_structured(
                prompt="hello", response_model=SmokeTestResponse
            )
        self.assertIs(raised.exception, error)

    async def test_provider_usage_is_captured_and_reset_on_failure(self):
        adapter = self.make_adapter(
            usage=SimpleNamespace(input_tokens=100, output_tokens=200)
        )
        await adapter.generate_structured(
            prompt="hello", response_model=SmokeTestResponse
        )
        self.assertEqual(
            adapter.last_usage, TokenUsage(input_tokens=100, output_tokens=200)
        )
        self.parse.side_effect = RuntimeError("unavailable")
        with self.assertRaises(RuntimeError):
            await adapter.generate_structured(
                prompt="hello", response_model=SmokeTestResponse
            )
        self.assertIsNone(adapter.last_usage)

    async def test_missing_provider_usage_stays_unknown(self):
        adapter = self.make_adapter()
        await adapter.generate_structured(
            prompt="hello", response_model=SmokeTestResponse
        )
        self.assertIsNone(adapter.last_usage)

    async def test_concurrent_calls_keep_task_local_usage_including_failure(self):
        adapter = self.make_adapter()
        first_finished = asyncio.Event()
        second_finished = asyncio.Event()

        async def parse(*, input, **kwargs):
            if input == "second":
                await first_finished.wait()
            if input == "failed":
                await second_finished.wait()
                raise RuntimeError("provider unavailable")
            count = 10 if input == "first" else 20
            return SimpleNamespace(
                id=input,
                status="completed",
                output=[],
                output_parsed=SmokeTestResponse(summary=input),
                usage=SimpleNamespace(input_tokens=count, output_tokens=count * 2),
            )

        self.parse.side_effect = parse

        async def call(prompt):
            await adapter.generate_structured(
                prompt=prompt, response_model=SmokeTestResponse
            )
            if prompt == "first":
                first_finished.set()
                await second_finished.wait()
            else:
                second_finished.set()
            # Force an interleaving after generation before reading usage.
            await asyncio.sleep(0)
            return adapter.last_usage

        async def fail():
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                await adapter.generate_structured(
                    prompt="failed", response_model=SmokeTestResponse
                )
            return adapter.last_usage

        usages = await asyncio.wait_for(
            asyncio.gather(call("first"), call("second"), fail()), 3
        )
        self.assertEqual(
            usages,
            [
                TokenUsage(input_tokens=10, output_tokens=20),
                TokenUsage(input_tokens=20, output_tokens=40),
                None,
            ],
        )
        self.assertIsNone(adapter.last_usage)


class SettingsTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.env_file = Path(directory) / ".env"
        self.enterContext(patch.object(config, "ENV_FILE", self.env_file))
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        config._load_environment.cache_clear()
        self.addCleanup(config._load_environment.cache_clear)

    def test_environment_and_secret_redaction(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-secret", "LLM_MODEL": "test-model"},
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.llm_model, "test-model")
        self.assertNotIn("test-secret", repr(settings))

    def test_missing_configuration_fails(self):
        for values in (
            {},
            {"OPENAI_API_KEY": "test-secret"},
            {"LLM_MODEL": "test-model"},
        ):
            with (
                self.subTest(values=list(values)),
                patch.dict(os.environ, values, clear=True),
                self.assertRaises(ValidationError),
            ):
                Settings.from_env()

    def test_loads_local_defaults_independently_of_working_directory(self):
        self.env_file.write_text(
            "OPENAI_API_KEY=file-secret\nLLM_MODEL=file-model\n", encoding="utf-8"
        )
        with TemporaryDirectory() as elsewhere, chdir(elsewhere):
            settings = Settings.from_env()
        self.assertEqual(settings.openai_api_key.get_secret_value(), "file-secret")
        self.assertEqual(settings.llm_model, "file-model")
        self.assertNotIn("file-secret", repr(settings))

    def test_process_environment_takes_precedence_over_local_defaults(self):
        self.env_file.write_text(
            "OPENAI_API_KEY=file-secret\nLLM_MODEL=file-model\n", encoding="utf-8"
        )
        os.environ["OPENAI_API_KEY"] = "process-secret"
        settings = Settings.from_env()
        self.assertEqual(settings.openai_api_key.get_secret_value(), "process-secret")
        self.assertEqual(settings.llm_model, "file-model")
        os.environ["LLM_MODEL"] = "process-model"
        self.assertEqual(Settings.from_env().llm_model, "process-model")

    def test_missing_dotenv_allows_process_only_settings(self):
        self.assertFalse(self.env_file.exists())
        os.environ.update(OPENAI_API_KEY="process-secret", LLM_MODEL="process-model")
        settings = Settings.from_env()
        self.assertEqual(settings.openai_api_key.get_secret_value(), "process-secret")
        self.assertEqual(settings.llm_model, "process-model")

    def test_dotenv_is_loaded_once(self):
        self.env_file.write_text(
            "OPENAI_API_KEY=file-secret\nLLM_MODEL=file-model\n", encoding="utf-8"
        )
        with patch.object(config, "load_dotenv", wraps=config.load_dotenv) as load:
            Settings.from_env()
            Settings.from_env()
        load.assert_called_once_with(dotenv_path=self.env_file, override=False)

    def test_empty_process_value_is_not_replaced_by_dotenv(self):
        self.env_file.write_text(
            "OPENAI_API_KEY=file-secret\nLLM_MODEL=file-model\n", encoding="utf-8"
        )
        os.environ["OPENAI_API_KEY"] = ""
        with self.assertRaises(ValidationError):
            Settings.from_env()
