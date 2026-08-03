from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import s3fs

from .config import LakeFSRuntimeConfig, bool_env


def output_prefix_from_env() -> str:
    return os.getenv("QSENTIA_OUTPUT_PREFIX", "inference_outputs/world-model-rl").strip("/")


def run_id_from_env() -> str:
    explicit = os.getenv("QSENTIA_RUN_ID", "").strip()
    if explicit:
        return explicit
    batch_job_id = os.getenv("AWS_BATCH_JOB_ID", "").strip()
    if batch_job_id:
        return batch_job_id
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def publish_outputs_to_lakefs(
    artifact_config: LakeFSRuntimeConfig,
    signal_payload: dict[str, Any],
    run_payload: dict[str, Any],
) -> dict[str, str] | None:
    if not bool_env("QSENTIA_PUBLISH_OUTPUTS", True):
        return None
    if artifact_config.skip_download and not artifact_config.endpoint:
        return None

    output_prefix = output_prefix_from_env()
    run_id = run_id_from_env()
    base_uri = f"s3://{artifact_config.repository}/{artifact_config.artifact_ref}/{output_prefix}/{run_id}"
    local_dir = Path(os.getenv("QSENTIA_OUTPUT_DIR", "/app/outputs"))
    local_dir.mkdir(parents=True, exist_ok=True)

    signal_path = local_dir / "latest_signal.json"
    payload_path = local_dir / "run_payload.json"
    signal_path.write_text(json.dumps(signal_payload, indent=2, default=str), encoding="utf-8")
    payload_path.write_text(json.dumps(run_payload, indent=2, default=str), encoding="utf-8")

    filesystem = s3fs.S3FileSystem(**artifact_config.storage_options)
    signal_uri = f"{base_uri}/latest_signal.json"
    payload_uri = f"{base_uri}/run_payload.json"
    filesystem.put(str(signal_path), signal_uri)
    filesystem.put(str(payload_path), payload_uri)

    result = {
        "run_id": run_id,
        "latest_signal_uri": signal_uri,
        "run_payload_uri": payload_uri,
    }

    if bool_env("QSENTIA_COMMIT_OUTPUTS", True):
        response = requests.post(
            f"{artifact_config.endpoint}/api/v1/repositories/{artifact_config.repository}/branches/{artifact_config.artifact_ref}/commits",
            auth=(artifact_config.access_key_id, artifact_config.secret_access_key),
            json={"message": f"publish WORLD_MODEL-RL inference output {run_id}"},
            timeout=30,
        )
        response.raise_for_status()
        result["commit_id"] = response.json().get("id", "")

    return result
