"""Sequential batched embedding, with a durable commit after each successful batch."""

from digital_applications_prototype.llm.embeddings import EmbeddingService
from digital_applications_prototype.persistence.embeddings import EmbeddingRepository

from .models import EmbeddingPlan, EmbeddingReport


class EmbeddingStageError(RuntimeError):
    def __init__(self, report: EmbeddingReport, *, batch_number: int) -> None:
        self.report = report
        super().__init__(
            f"Embedding batch {batch_number} failed; earlier successful batches remain "
            "committed. Rerun the same selection to retry missing embeddings."
        )


async def embed_observations(
    repository: EmbeddingRepository,
    provider: EmbeddingService,
    plan: EmbeddingPlan,
    *,
    allow_multiple_runs_per_document: bool = False,
) -> EmbeddingReport:
    plan.validate(allow_multiple_runs_per_document=allow_multiple_runs_per_document)
    report = EmbeddingReport(dimensions=set(plan.current_dimensions))
    for start in range(0, len(plan.pending), plan.batch_size):
        batch = plan.pending[start : start + plan.batch_size]
        report.batches += 1
        report.submitted += len(batch)
        try:
            response = await provider.embed(
                texts=[item.source_text for item in batch], model=plan.model
            )
            # Usage can be reported even if a subsequent database commit fails.
            if response.usage is not None and response.usage.input_tokens is not None:
                report.input_tokens += response.usage.input_tokens
                report.usage_batches += 1
            saved = await repository.save_batch(
                batch,
                response,
                requested_model=plan.model,
                representation=plan.representation,
            )
        except Exception as error:
            report.failed += len(batch)
            raise EmbeddingStageError(report, batch_number=report.batches) from error
        report.embedded += saved
        report.completed_batches += 1
        report.dimensions.add(len(response.vectors[0]))
        report.models.add(response.model)
    return report
