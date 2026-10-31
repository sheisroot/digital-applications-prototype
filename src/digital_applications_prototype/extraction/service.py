"""Feature semantics and validation, independent of provider SDKs."""

from digital_applications_prototype.llm.prompt_renderer import PromptRenderer
from digital_applications_prototype.llm.service import LLMService

from .models import DocumentChunk, ExtractionResult, ImplementationObservation
from .validation import GroundingError, validate_observation


def render_prompt(renderer: PromptRenderer, chunk: DocumentChunk) -> str:
    return renderer.render(
        "implementation_observations.jinja2",
        project_id=chunk.project_id, country=chunk.country, document_id=chunk.document_id,
        document_type=chunk.document_type,
        document_date=chunk.document_date.isoformat() if chunk.document_date else "unknown",
        start_page=chunk.start_page, end_page=chunk.end_page,
        document_text="\n\n".join(f"[PDF page {p.page_number}]\n{p.text}" for p in chunk.pages),
    )


class ImplementationObservationExtractor:
    def __init__(self, renderer: PromptRenderer, llm: LLMService) -> None:
        self.renderer = renderer
        self.llm = llm

    async def extract(self, chunk: DocumentChunk) -> list[ImplementationObservation]:
        result = await self.llm.generate_structured(
            prompt=render_prompt(self.renderer, chunk), response_model=ExtractionResult
        )
        observations = []
        for index, item in enumerate(result.observations):
            try:
                observations.append(validate_observation(item, chunk))
            except GroundingError as error:
                raise GroundingError(
                    f"{chunk.document_id} pages {chunk.start_page}–{chunk.end_page}, "
                    f"observation {index + 1}: {error}"
                ) from error
        return observations
