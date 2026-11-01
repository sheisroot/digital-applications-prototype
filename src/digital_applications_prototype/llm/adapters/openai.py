"""Translate the OpenAI Responses API into the generic adapter contract."""

from contextvars import ContextVar

from openai import AsyncOpenAI

from .base import T, TokenUsage


class StructuredOutputError(RuntimeError):
    """The provider completed without the requested typed output."""


class OpenAIAdapter:
    def __init__(self, client: AsyncOpenAI, *, model: str) -> None:
        # The application owns the client lifecycle (use async with AsyncOpenAI).
        self.client = client
        self.model = model
        self._last_usage: ContextVar[TokenUsage | None] = ContextVar(
            "openai_last_usage", default=None
        )

    @property
    def last_usage(self) -> TokenUsage | None:
        """Usage for the latest call in the current task, isolated from sibling calls."""
        return self._last_usage.get()

    async def generate_structured(
        self, *, prompt: str, response_model: type[T], model: str | None = None
    ) -> T:
        """Request typed output using the override model or the adapter default.

        Raise StructuredOutputError for refusals, incomplete responses, or
        missing parsed output; let provider exceptions propagate unchanged.
        """
        self._last_usage.set(None)
        response = await self.client.responses.parse(
            model=model if model is not None else self.model,
            input=prompt,
            text_format=response_model,
        )
        if response.status != "completed":
            raise StructuredOutputError(
                f"OpenAI response {response.id} has status {response.status!r}; "
                f"details: {response.incomplete_details!r}"
            )
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "refusal":
                        raise StructuredOutputError(
                            f"OpenAI refused: {content.refusal}"
                        )
        if response.output_parsed is None:
            raise StructuredOutputError(
                f"OpenAI response {response.id} contained no parsed output"
            )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._last_usage.set(
                TokenUsage(
                    input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
                )
            )
        return response.output_parsed
