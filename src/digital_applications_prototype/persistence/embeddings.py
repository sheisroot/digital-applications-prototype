"""Read Stage 1 inputs; write only embedding rows in short batch transactions."""

from collections import Counter
from collections.abc import Sequence

from sqlalchemy import inspect, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from digital_applications_prototype.embeddings.models import (
    DEFAULT_BATCH_SIZE,
    REPRESENTATION,
    EmbeddingInput,
    EmbeddingPlan,
    SelectedRun,
    source_text_hash,
)
from digital_applications_prototype.embeddings.scope import normalize_run_ids
from digital_applications_prototype.llm.embeddings import EmbeddingResponse

from .models import (
    Base,
    ExtractionChunk,
    ExtractionRun,
    Observation,
    ObservationEmbedding,
)
from .repository import utc_now


class EmbeddingRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def initialize(self) -> None:
        """Add only the new table to an existing Stage 1 database, idempotently."""
        async with self.sessions.begin() as session:
            connection = await session.connection()
            await connection.run_sync(
                lambda sync: Base.metadata.create_all(
                    sync,
                    tables=[Base.metadata.tables[ObservationEmbedding.__tablename__]],
                )
            )

    async def plan(
        self,
        *,
        run_ids: Sequence[str] | None = None,
        all_runs: bool = False,
        model: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> EmbeddingPlan:
        """Also works read-only against a database predating the embedding table."""
        if (run_ids is not None) == all_runs:
            raise ValueError("Select either explicit run IDs or all runs")
        ids = normalize_run_ids(run_ids) if run_ids is not None else None
        async with self.sessions() as session:
            query = select(ExtractionRun).order_by(ExtractionRun.run_id)
            if ids is not None:
                query = query.where(ExtractionRun.run_id.in_(ids))
            runs = list(await session.scalars(query))
            if ids is not None:
                missing = sorted(set(ids) - {run.run_id for run in runs})
                if missing:
                    raise ValueError(
                        f"Unknown extraction run IDs: {', '.join(missing)}"
                    )
            if not runs:
                raise ValueError("No extraction runs selected")
            selected = [run.run_id for run in runs]
            observations = list(
                await session.scalars(
                    select(Observation)
                    .join(
                        ExtractionChunk,
                        Observation.chunk_id == ExtractionChunk.chunk_id,
                    )
                    .where(
                        Observation.run_id.in_(selected),
                        Observation.validation_status.in_(("valid", "repaired")),
                        Observation.duplicate_of_observation_id.is_(None),
                        Observation.factor_description.is_not(None),
                    )
                    .order_by(
                        Observation.run_id,
                        ExtractionChunk.chunk_index,
                        Observation.observation_index,
                        Observation.observation_id,
                    )
                )
            )
            observations = [
                row for row in observations if row.factor_description.strip()
            ]
            connection = await session.connection()
            has_embeddings = await connection.run_sync(
                lambda sync: inspect(sync).has_table(ObservationEmbedding.__tablename__)
            )
            current: dict[tuple[int, str], set[int]] = {}
            if has_embeddings:
                stored = await session.execute(
                    select(
                        ObservationEmbedding.observation_id,
                        ObservationEmbedding.source_text_hash,
                        ObservationEmbedding.dimensions,
                    )
                    .join(Observation)
                    .where(
                        Observation.run_id.in_(selected),
                        ObservationEmbedding.requested_model == model,
                        ObservationEmbedding.representation == REPRESENTATION,
                        ObservationEmbedding.dimensions_mode == "default",
                    )
                )
                for observation_id, text_hash, dimensions in stored:
                    current.setdefault((observation_id, text_hash), set()).add(
                        dimensions
                    )
            pending: list[EmbeddingInput] = []
            eligible: Counter[str] = Counter()
            already_current: Counter[str] = Counter()
            current_dimensions: set[int] = set()
            for row in observations:
                eligible[row.run_id] += 1
                item = EmbeddingInput(row.observation_id, row.factor_description)
                dimensions = current.get((item.observation_id, item.source_text_hash))
                if dimensions:
                    already_current[row.run_id] += 1
                    current_dimensions.update(dimensions)
                else:
                    pending.append(item)
            return EmbeddingPlan(
                runs=tuple(
                    SelectedRun(
                        run.run_id,
                        run.document_id,
                        eligible[run.run_id],
                        already_current[run.run_id],
                    )
                    for run in runs
                ),
                pending=tuple(pending),
                model=model,
                batch_size=batch_size,
                current_dimensions=tuple(sorted(current_dimensions)),
            )

    async def save_batch(
        self,
        inputs: Sequence[EmbeddingInput],
        response: EmbeddingResponse,
        *,
        requested_model: str,
        representation: str = REPRESENTATION,
        dimensions_mode: str = "default",
    ) -> int:
        """Validate the whole response, then append atomically; never replace history."""
        dimensions = response.validate(len(inputs))
        inserted = 0
        async with self.sessions.begin() as session:
            for item, vector in zip(inputs, response.vectors, strict=True):
                result = await session.scalar(
                    insert(ObservationEmbedding)
                    .values(
                        observation_id=item.observation_id,
                        representation=representation,
                        requested_model=requested_model,
                        model=response.model,
                        dimensions_mode=dimensions_mode,
                        dimensions=dimensions,
                        source_text=item.source_text,
                        source_text_hash=item.source_text_hash,
                        embedding_json=vector,
                        created_at=utc_now(),
                    )
                    .on_conflict_do_nothing()
                    .returning(ObservationEmbedding.embedding_id)
                )
                inserted += int(result is not None)
        return inserted

    async def get_embeddings(
        self,
        run_ids: Sequence[str],
        *,
        model: str,
        representation: str = REPRESENTATION,
        current_only: bool = True,
        dimensions_mode: str | None = "default",
        dimensions: int | None = None,
    ) -> list[tuple[ObservationEmbedding, Observation]]:
        """Joined vector/provenance records for later analysis; model is the actual model.

        Detached observation rows include run/document/project/country and the current
        factor description. Historical embedding rows retain their exact source text.
        """
        ids = normalize_run_ids(run_ids)
        if current_only and representation != REPRESENTATION:
            raise ValueError(
                "Current-source matching is only implemented for factor_description; "
                "use current_only=False for other representations"
            )
        query = (
            select(ObservationEmbedding, Observation)
            .join(Observation)
            .where(
                Observation.run_id.in_(ids),
                ObservationEmbedding.model == model,
                ObservationEmbedding.representation == representation,
            )
            .order_by(
                Observation.run_id,
                ObservationEmbedding.observation_id,
                ObservationEmbedding.embedding_id,
            )
        )
        if dimensions_mode is not None:
            query = query.where(ObservationEmbedding.dimensions_mode == dimensions_mode)
        if dimensions is not None:
            query = query.where(ObservationEmbedding.dimensions == dimensions)
        async with self.sessions() as session:
            rows = await session.execute(query)
            return [
                (embedding, observation)
                for embedding, observation in rows
                if not current_only
                or (
                    observation.validation_status in ("valid", "repaired")
                    and observation.duplicate_of_observation_id is None
                    and observation.factor_description is not None
                    and bool(observation.factor_description.strip())
                    and embedding.source_text_hash
                    == source_text_hash(observation.factor_description)
                )
            ]
