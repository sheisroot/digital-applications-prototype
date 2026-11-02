"""Read configuration explicitly at the application boundary."""

import os
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


@cache
def _load_environment() -> None:
    """Load the optional project-root .env once, preserving process variables."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)


class OpenAISettings(BaseModel):
    openai_api_key: SecretStr = Field(min_length=1)

    @classmethod
    def from_env(cls) -> "OpenAISettings":
        _load_environment()
        return cls(openai_api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "")))


class EmbeddingSettings(BaseModel):
    """Model selection is available for offline planning without credentials."""

    embedding_model: str = Field(default="text-embedding-3-large", min_length=1)

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        _load_environment()
        return cls(
            embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
        )


class Settings(OpenAISettings, EmbeddingSettings):
    llm_model: str = Field(min_length=1)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load local defaults once and validate required application settings."""
        # No import-time environment reads, and no credentials in model reprs.
        _load_environment()
        return cls(
            openai_api_key=OpenAISettings.from_env().openai_api_key,
            llm_model=os.environ.get("LLM_MODEL", ""),
            embedding_model=EmbeddingSettings.from_env().embedding_model,
        )
