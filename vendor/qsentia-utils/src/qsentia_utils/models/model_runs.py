from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from qsentia_utils.models.common import Environment, JsonDict, RunStatus


class ModelRunCreate(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(default_factory=uuid4)
    environment: Environment | str
    producer_id: str
    producer_version: str | None = None
    run_type: str
    status: RunStatus | str = RunStatus.STARTED
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    source_trigger: str | None = None
    idempotency_key: str | None = None
    metadata: JsonDict = Field(default_factory=dict)


class ModelRun(ModelRunCreate):
    created_at: datetime
