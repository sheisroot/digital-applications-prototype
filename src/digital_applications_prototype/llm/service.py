"""Small provider-independent boundary for future cross-cutting behavior."""

from .adapters.base import LLMAdapter, T, TokenUsage


class LLMService:
    def __init__(self, adapter: LLMAdapter) -> None:
        self.adapter = adapter

    @property
    def model(self) -> str | None:
        return getattr(self.adapter, "model", None)

    @property
    def last_usage(self) -> TokenUsage | None:
        # Optional metadata; the typed generation contract remains unchanged.
        # Read in the calling task immediately after generation. Concurrent adapters
        # must isolate this metadata per task, as the OpenAI adapter does.
        return getattr(self.adapter, "last_usage", None)

    async def generate_structured(
        self, *, prompt: str, response_model: type[T], model: str | None = None
    ) -> T:
        return await self.adapter.generate_structured(
            prompt=prompt, response_model=response_model, model=model
        )
