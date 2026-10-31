"""The only generation contract feature code needs."""

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMAdapter(Protocol):
    async def generate_structured(
        self, *, prompt: str, response_model: type[T], model: str | None = None
    ) -> T: ...
