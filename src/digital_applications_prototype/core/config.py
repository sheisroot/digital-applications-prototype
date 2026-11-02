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


class Settings(BaseModel):
    openai_api_key: SecretStr = Field(min_length=1)
    llm_model: str = Field(min_length=1)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load local defaults once and validate required application settings."""
        # No import-time environment reads, and no credentials in model reprs.
        _load_environment()
        return cls(
            openai_api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "")),
            llm_model=os.environ.get("LLM_MODEL", ""),
        )
