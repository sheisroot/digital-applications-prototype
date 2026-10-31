"""Architecture validation only."""

from pydantic import BaseModel


class SmokeTestResponse(BaseModel):
    summary: str
