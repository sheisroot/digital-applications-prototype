"""Application composition and SDK lifecycle, outside the extraction feature."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from openai import AsyncOpenAI

from digital_applications_prototype.llm.adapters.openai import OpenAIAdapter
from digital_applications_prototype.llm.adapters.openai_embeddings import (
    OpenAIEmbeddingAdapter,
)
from digital_applications_prototype.llm.embeddings import EmbeddingService
from digital_applications_prototype.llm.service import LLMService

from .config import OpenAISettings, Settings


@asynccontextmanager
async def configured_llm() -> AsyncIterator[LLMService]:
    """Yield an environment-configured service and close its SDK client on exit."""
    settings = Settings.from_env()
    async with AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value()
    ) as client:
        yield LLMService(OpenAIAdapter(client, model=settings.llm_model))


@asynccontextmanager
async def configured_embeddings() -> AsyncIterator[EmbeddingService]:
    """Use the same credential loader and SDK lifecycle without requiring LLM_MODEL."""
    settings = OpenAISettings.from_env()
    async with AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value()
    ) as client:
        yield EmbeddingService(OpenAIEmbeddingAdapter(client))
