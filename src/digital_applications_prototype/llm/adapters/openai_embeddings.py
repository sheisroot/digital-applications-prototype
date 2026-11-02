"""Translate indexed OpenAI embedding results into input-ordered vectors."""

from openai import AsyncOpenAI

from ..embeddings import EmbeddingResponse
from .base import TokenUsage


class OpenAIEmbeddingAdapter:
    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client

    async def embed(self, *, texts: list[str], model: str) -> EmbeddingResponse:
        # Omit dimensions to use the model's full/default output. The SDK retains
        # its built-in retries and the caller owns the async client lifecycle.
        response = await self.client.embeddings.create(
            input=texts, model=model, encoding_format="float"
        )
        indexes = [item.index for item in response.data]
        if sorted(indexes) != list(range(len(texts))):
            raise ValueError(
                "Embedding response has missing, duplicate, or invalid indexes"
            )
        usage = response.usage
        result = EmbeddingResponse(
            vectors=[
                item.embedding
                for item in sorted(response.data, key=lambda item: item.index)
            ],
            model=response.model,
            usage=TokenUsage(input_tokens=usage.prompt_tokens)
            if usage is not None
            else None,
        )
        result.validate(len(texts))
        return result
