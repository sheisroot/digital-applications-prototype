"""Explicit developer-only smoke call. Running this script uses the paid API."""

import asyncio

from openai import AsyncOpenAI

from digital_applications_prototype.core.config import Settings
from digital_applications_prototype.llm.adapters.openai import OpenAIAdapter
from digital_applications_prototype.llm.prompt_renderer import PromptRenderer
from digital_applications_prototype.llm.schemas.smoke import SmokeTestResponse
from digital_applications_prototype.llm.service import LLMService


async def main() -> None:
    settings = Settings.from_env()
    prompt = PromptRenderer().render(
        "smoke.jinja2", text="This project separates prompts from providers."
    )
    async with AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value()
    ) as client:
        service = LLMService(OpenAIAdapter(client, model=settings.llm_model))
        result = await service.generate_structured(
            prompt=prompt, response_model=SmokeTestResponse
        )
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
