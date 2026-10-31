"""Render file-based prompts without domain behavior."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


class PromptRenderer:
    def __init__(self, prompts_dir: str | Path | None = None) -> None:
        directory = prompts_dir if prompts_dir is not None else Path(__file__).parent / "prompts"
        self.environment = Environment(
            loader=FileSystemLoader(directory),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,  # Plain text prompts, not HTML.
        )

    def render(self, template_name: str, **variables: Any) -> str:
        return self.environment.get_template(template_name).render(**variables)
