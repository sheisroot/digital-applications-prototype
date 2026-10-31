"""Translate the OpenAI Responses API into the generic adapter contract."""

from openai import AsyncOpenAI

from .base import T


class StructuredOutputError(RuntimeError):
    """The provider completed without the requested typed output."""


class OpenAIAdapter:
    def __init__(self, client: AsyncOpenAI, *, model: str) -> None:
        # The application owns the client lifecycle (use async with AsyncOpenAI).
        self.client = client
        self.model = model

    async def generate_structured(
        self, *, prompt: str, response_model: type[T], model: str | None = None
    ) -> T:
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
                        raise StructuredOutputError(f"OpenAI refused: {content.refusal}")
        if response.output_parsed is None:
            raise StructuredOutputError(
                f"OpenAI response {response.id} contained no parsed output"
            )
        return response.output_parsed
