"""The only generation contract feature code needs."""

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class TokenUsage(BaseModel):
    """Optional provider-reported counts for one successful generation."""

    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMAdapter(Protocol):
    async def generate_structured(
        self, *, prompt: str, response_model: type[T], model: str | None = None
    ) -> T: ...
