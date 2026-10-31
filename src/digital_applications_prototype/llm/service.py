"""Small provider-independent boundary for future cross-cutting behavior."""

from .adapters.base import LLMAdapter, T


class LLMService:
    def __init__(self, adapter: LLMAdapter) -> None:
        self.adapter = adapter

    async def generate_structured(
        self, *, prompt: str, response_model: type[T], model: str | None = None
    ) -> T:
        return await self.adapter.generate_structured(
            prompt=prompt, response_model=response_model, model=model
        )
