from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from qsentia_utils.models.common import Environment, JsonDict


class ModelOutputCreate(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    environment: Environment | str
    producer_id: str
    producer_version: str | None = None
    output_type: str = "prediction"
    effective_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: int = 1
    output: JsonDict
    metadata: JsonDict = Field(default_factory=dict)
    output_uri: str | None = None


class ModelOutput(ModelOutputCreate):
    created_at: datetime
