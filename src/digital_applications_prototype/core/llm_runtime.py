"""Application composition and SDK lifecycle, outside the extraction feature."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .config import Settings
from digital_applications_prototype.llm.adapters.openai import OpenAIAdapter
from digital_applications_prototype.llm.service import LLMService


@asynccontextmanager
async def configured_llm() -> AsyncIterator[LLMService]:
    settings = Settings.from_env()
    async with AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value()) as client:
        yield LLMService(OpenAIAdapter(client, model=settings.llm_model))
