from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def _to_db_record(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=True)


def model_run_record(model: BaseModel) -> dict[str, Any]:
    return _to_db_record(model)


def model_output_record(model: BaseModel) -> dict[str, Any]:
    return _to_db_record(model)
