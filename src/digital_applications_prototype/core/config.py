"""Read configuration explicitly at the application boundary."""

import os

from pydantic import BaseModel, Field, SecretStr


class Settings(BaseModel):
    openai_api_key: SecretStr = Field(min_length=1)
    llm_model: str = Field(min_length=1)

    @classmethod
    def from_env(cls) -> "Settings":
        # No import-time environment reads, and no credentials in model reprs.
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            llm_model=os.environ.get("LLM_MODEL", ""),
        )
