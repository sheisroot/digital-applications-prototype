"""Provider-independent batched embedding contract, separate from generation."""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from .adapters.base import TokenUsage


@dataclass(frozen=True)
class EmbeddingResponse:
    """Vectors in input order, with the model actually reported by the provider."""

    vectors: list[list[float]]
    model: str
    usage: TokenUsage | None = None

    def validate(self, input_count: int) -> int:
        if not self.model.strip():
            raise ValueError("Embedding response has no model")
        if not self.vectors or len(self.vectors) != input_count:
            raise ValueError("Embedding response count does not match inputs")
        dimensions = len(self.vectors[0])
        if dimensions < 1 or any(
            len(vector) != dimensions or not all(isfinite(value) for value in vector)
            for vector in self.vectors
        ):
            raise ValueError(
                "Embedding vectors must have consistent, finite dimensions"
            )
        return dimensions


class EmbeddingAdapter(Protocol):
    async def embed(self, *, texts: list[str], model: str) -> EmbeddingResponse: ...


class EmbeddingService:
    def __init__(self, adapter: EmbeddingAdapter) -> None:
        self.adapter = adapter

    async def embed(self, *, texts: list[str], model: str) -> EmbeddingResponse:
        response = await self.adapter.embed(texts=texts, model=model)
        response.validate(len(texts))
        return response
