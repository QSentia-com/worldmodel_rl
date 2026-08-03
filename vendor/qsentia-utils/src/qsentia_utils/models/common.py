from __future__ import annotations

from enum import StrEnum
from typing import Any

JsonDict = dict[str, Any]


class Environment(StrEnum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class RunStatus(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
