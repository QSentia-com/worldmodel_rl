from __future__ import annotations

import json
import os

from .alpaca_execution import execute_alpaca_trade_intent
from .artifact_manager import download_lakefs_artifacts
from .config import LakeFSRuntimeConfig
from .db_outputs import record_alpaca_trade_orders, record_inference_output
from .live_signal_refresh import assert_live_signal_refresh_ready
from .output_publisher import publish_outputs_to_lakefs
from .signal_inference import run_signal_inference
from .structured_logging import build_signal_summary, emit_event


def main() -> int:
    artifact_config = LakeFSRuntimeConfig.from_env()
    artifact_source = "local_artifact_mount" if artifact_config.skip_download else artifact_config.object_uri
    emit_event(
        "world_model_rl_job_started",
        artifact_source=artifact_source,
        artifact_dir=str(artifact_config.artifact_dir),
        batch_job_id=os.getenv("AWS_BATCH_JOB_ID"),
        batch_job_attempt=os.getenv("AWS_BATCH_JOB_ATTEMPT"),
    )

    downloaded = download_lakefs_artifacts(artifact_config)
    emit_event(
        "world_model_rl_artifacts_ready",
        artifact_source=artifact_source,
        artifact_dir=str(artifact_config.artifact_dir),
        downloaded_artifact_files=len(downloaded),
        skipped_download=artifact_config.skip_download,
    )

    refresh_report = assert_live_signal_refresh_ready(artifact_config)
    if refresh_report:
        emit_event(
            "world_model_rl_live_signal_refresh_verified",
            status=refresh_report.get("status"),
            run_mode=refresh_report.get("run_mode"),
            signal_date=refresh_report.get("signal_date"),
            asof=refresh_report.get("asof"),
            candidate_rows=refresh_report.get("candidate_rows"),
            mapped_option_legs=refresh_report.get("mapped_option_legs"),
        )

    signal_payload = run_signal_inference(artifact_dir=artifact_config.artifact_dir)
    execution_report = execute_alpaca_trade_intent(signal_payload.get("trade"))
    if execution_report:
        signal_payload["alpaca_execution"] = execution_report

    signal_summary = build_signal_summary(signal_payload)
    run_payload = {
        "artifact_source": artifact_source,
        "artifact_dir": str(artifact_config.artifact_dir),
        "downloaded_artifact_files": len(downloaded),
        "inference": signal_payload,
    }
    if refresh_report:
        run_payload["live_signal_refresh"] = refresh_report

    published_outputs = publish_outputs_to_lakefs(artifact_config, signal_payload, run_payload)
    if published_outputs:
        run_payload["published_outputs"] = published_outputs

    db_record = record_inference_output(
        artifact_source=artifact_source,
        artifact_dir=str(artifact_config.artifact_dir),
        downloaded_artifact_files=len(downloaded),
        signal_payload=signal_payload,
        signal_summary=signal_summary,
        published_outputs=published_outputs,
    )
    if db_record:
        run_payload["database_record"] = db_record
        order_record = record_alpaca_trade_orders(db_record=db_record, signal_payload=signal_payload)
        if order_record:
            run_payload["alpaca_trade_order_record"] = order_record

    emit_event("world_model_rl_signal_generated", **signal_summary)
    if published_outputs:
        emit_event("world_model_rl_outputs_published", **published_outputs)
    if db_record:
        emit_event("world_model_rl_database_output_recorded", **db_record)
    if run_payload.get("alpaca_trade_order_record"):
        emit_event("world_model_rl_alpaca_trade_orders_recorded", **run_payload["alpaca_trade_order_record"])
    if execution_report:
        emit_event("world_model_rl_alpaca_execution_completed", **execution_report)

    print("FINAL_SIGNAL " + json.dumps(signal_summary, default=str, separators=(",", ":")), flush=True)
    print("FINAL_PAYLOAD " + json.dumps(run_payload, default=str, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
