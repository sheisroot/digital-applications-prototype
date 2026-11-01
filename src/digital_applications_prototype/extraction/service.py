"""Feature semantics and validation, independent of provider SDKs."""

from digital_applications_prototype.llm.prompt_renderer import PromptRenderer
from digital_applications_prototype.llm.service import LLMService

from .models import ChunkExtractionResult, DocumentChunk, ExtractionResult
from .validation import validate_observation


def render_prompt(renderer: PromptRenderer, chunk: DocumentChunk) -> str:
    """Render chunk metadata and page-marked source text into the extraction prompt."""
    return renderer.render(
        "implementation_observations.jinja2",
        project_id=chunk.project_id, country=chunk.country, document_id=chunk.document_id,
        document_type=chunk.document_type,
        document_date=chunk.document_date.isoformat() if chunk.document_date else "unknown",
        start_page=chunk.start_page, end_page=chunk.end_page,
        document_text="\n\n".join(
            f"[PDF page {p.page_number}]"
            f"{' (trailing overlap)' if p is chunk.overlap else ''}\n{p.text}"
            for p in chunk.source_pages
        ),
    )


class ImplementationObservationExtractor:
    def __init__(self, renderer: PromptRenderer, llm: LLMService) -> None:
        self.renderer = renderer
        self.llm = llm

    async def extract(self, chunk: DocumentChunk) -> ChunkExtractionResult:
        """Validate every observation independently after a schema-valid response.

        Return usable claims, counts, warnings, and optional last-call usage.
        Provider, prompt, and whole-response schema failures propagate as chunk
        failures; observation rejections do not stop subsequent observations.
        """
        result = await self.llm.generate_structured(
            prompt=render_prompt(self.renderer, chunk), response_model=ExtractionResult
        )
        extracted = ChunkExtractionResult(observations_returned=len(result.observations),
                                          usage=self.llm.last_usage)
        for index, item in enumerate(result.observations, start=1):
            validated = validate_observation(item, chunk, observation_index=index)
            field = f"observations_{validated.status}"
            setattr(extracted, field, getattr(extracted, field) + 1)
            extracted.warnings.extend(validated.warnings)
            if validated.observation is not None:
                extracted.observations.append(validated.observation)
        return extracted
