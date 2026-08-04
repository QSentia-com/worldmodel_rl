from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import uuid4

from .config import bool_env


def record_inference_output(
    *,
    artifact_source: str,
    artifact_dir: str,
    downloaded_artifact_files: int,
    signal_payload: dict[str, Any],
    signal_summary: dict[str, Any],
    published_outputs: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    if not bool_env("QSENTIA_RECORD_DB_OUTPUTS", True):
        return None

    from psycopg.types.json import Jsonb
    from qsentia_utils.db import model_output_record, model_run_record, open_connection
    from qsentia_utils.models import ModelOutputCreate, ModelRunCreate, RunStatus

    now = datetime.now(timezone.utc)
    signal = signal_payload.get("signal") or {}
    trade_intent = signal_payload.get("trade") if bool_env("QSENTIA_TRADE_INTENT_ENABLED", True) else None
    alpaca_execution = signal_payload.get("alpaca_execution")
    output_payload: dict[str, Any] = {"signal": signal}
    if trade_intent is not None:
        output_payload["trade"] = trade_intent
    if alpaca_execution is not None:
        output_payload["alpaca_execution"] = alpaca_execution

    producer_id = os.getenv("QSENTIA_PRODUCER_ID", "qsentia-world-model-rl")
    producer_version = os.getenv("QSENTIA_PRODUCER_VERSION") or "v12b_exact_same_leg_inverse_plus_small_rl_guardian_v1"
    environment = os.getenv("QSENTIA_ENVIRONMENT", os.getenv("ENVIRONMENT", "dev"))
    batch_job_id = os.getenv("AWS_BATCH_JOB_ID")
    run = ModelRunCreate(
        environment=environment,
        producer_id=producer_id,
        producer_version=producer_version,
        run_type=os.getenv("QSENTIA_RUN_TYPE", "inference"),
        status=RunStatus.SUCCEEDED,
        started_at=_parse_datetime(signal_payload.get("ts")) or now,
        completed_at=now,
        source_trigger=os.getenv("QSENTIA_SOURCE_TRIGGER", "aws_batch"),
        idempotency_key=os.getenv("QSENTIA_RUN_ID") or batch_job_id,
        metadata={
            "artifact_source": artifact_source,
            "artifact_dir": artifact_dir,
            "downloaded_artifact_files": downloaded_artifact_files,
            "batch_job_id": batch_job_id,
            "batch_job_attempt": os.getenv("AWS_BATCH_JOB_ATTEMPT"),
            "output_path": signal_payload.get("output_path"),
        },
    )
    output = ModelOutputCreate(
        run_id=run.id,
        environment=environment,
        producer_id=producer_id,
        producer_version=producer_version,
        output_type=os.getenv("QSENTIA_OUTPUT_TYPE", "trade_signal" if trade_intent is not None else "prediction"),
        effective_at=_parse_datetime(signal.get("asof")) or _parse_datetime(signal_payload.get("ts")) or now,
        schema_version=int(os.getenv("QSENTIA_OUTPUT_SCHEMA_VERSION", "1")),
        output=output_payload,
        metadata={
            "signal_summary": signal_summary,
            "artifact_source": artifact_source,
            "published_outputs": published_outputs or {},
            "trade_intent_enabled": trade_intent is not None,
            "alpaca_execution": alpaca_execution or {},
        },
        output_uri=(published_outputs or {}).get("latest_signal_uri"),
    )

    run_record = model_run_record(run)
    output_record = model_output_record(output)
    run_record["metadata"] = Jsonb(run_record["metadata"])
    output_record["output"] = Jsonb(output_record["output"])
    output_record["metadata"] = Jsonb(output_record["metadata"])
    with open_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO model_runs (
                    id,
                    environment,
                    producer_id,
                    producer_version,
                    run_type,
                    status,
                    started_at,
                    completed_at,
                    source_trigger,
                    idempotency_key,
                    metadata
                ) VALUES (
                    %(id)s,
                    %(environment)s,
                    %(producer_id)s,
                    %(producer_version)s,
                    %(run_type)s,
                    %(status)s,
                    %(started_at)s,
                    %(completed_at)s,
                    %(source_trigger)s,
                    %(idempotency_key)s,
                    %(metadata)s
                )
                ON CONFLICT (idempotency_key) WHERE idempotency_key IS NOT NULL DO UPDATE SET
                    status = EXCLUDED.status,
                    completed_at = EXCLUDED.completed_at,
                    metadata = EXCLUDED.metadata
                RETURNING id
                """,
                run_record,
            )
            persisted_run_id = cursor.fetchone()[0]
            output_record["run_id"] = str(persisted_run_id)
            cursor.execute(
                """
                INSERT INTO model_outputs (
                    id,
                    run_id,
                    environment,
                    producer_id,
                    producer_version,
                    output_type,
                    effective_at,
                    schema_version,
                    output,
                    metadata,
                    output_uri
                ) VALUES (
                    %(id)s,
                    %(run_id)s,
                    %(environment)s,
                    %(producer_id)s,
                    %(producer_version)s,
                    %(output_type)s,
                    %(effective_at)s,
                    %(schema_version)s,
                    %(output)s,
                    %(metadata)s,
                    %(output_uri)s
                )
                """,
                output_record,
            )
        connection.commit()

    return {
        "run_id": str(persisted_run_id),
        "output_id": str(output.id),
        "producer_id": producer_id,
        "producer_version": producer_version or "",
        "output_type": output.output_type,
        "trade_intent": bool(trade_intent and trade_intent.get("orders")),
    }


def record_alpaca_trade_orders(
    *,
    db_record: dict[str, Any],
    signal_payload: dict[str, Any],
) -> dict[str, Any] | None:
    execution = signal_payload.get("alpaca_execution")
    if not isinstance(execution, dict):
        return None

    results = [row for row in execution.get("results", []) if isinstance(row, dict)]
    if not results:
        return {"status": "skipped", "trade_orders_recorded": 0}

    from psycopg.types.json import Jsonb
    from qsentia_utils.db import open_connection

    environment = os.getenv("QSENTIA_ENVIRONMENT", os.getenv("ENVIRONMENT", "dev"))
    producer_id = str(db_record.get("producer_id") or os.getenv("QSENTIA_PRODUCER_ID", "qsentia-world-model-rl"))
    producer_version = str(
        db_record.get("producer_version")
        or os.getenv("QSENTIA_PRODUCER_VERSION")
        or "v12b_exact_same_leg_inverse_plus_small_rl_guardian_v1"
    )
    output_id = str(db_record["output_id"])
    run_id = str(db_record["run_id"])
    requested_at = _parse_datetime(execution.get("ts")) or datetime.now(timezone.utc)

    with open_connection() as connection:
        with connection.cursor() as cursor:
            mapping = _load_active_runtime_mapping(cursor, producer_id, producer_version, environment)
            cursor.execute(
                """
                UPDATE model_outputs
                SET output = jsonb_set(COALESCE(output, '{}'::jsonb), '{alpaca_execution}', %(execution)s, true),
                    metadata = jsonb_set(COALESCE(metadata, '{}'::jsonb), '{alpaca_execution}', %(execution)s, true)
                WHERE id = %(output_id)s::uuid
                """,
                {"execution": Jsonb(execution), "output_id": output_id},
            )

            written = 0
            for index, result in enumerate(results):
                payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
                response = result.get("response") if isinstance(result.get("response"), dict) else {}
                record = _trade_order_record(
                    payload=payload,
                    response=response,
                    result=result,
                    mapping=mapping,
                    output_id=output_id,
                    run_id=run_id,
                    environment=environment,
                    producer_id=producer_id,
                    producer_version=producer_version,
                    requested_at=requested_at,
                    order_index=index,
                )
                cursor.execute(
                    """
                    INSERT INTO trade_orders (
                        id,
                        model_registry_id,
                        model_runtime_mapping_id,
                        model_output_id,
                        run_id,
                        environment,
                        broker,
                        account_id,
                        symbol,
                        side,
                        quantity,
                        order_type,
                        limit_price,
                        status,
                        decision_reason,
                        broker_order_id,
                        idempotency_key,
                        requested_at,
                        submitted_at,
                        completed_at,
                        request_payload,
                        broker_response,
                        metadata
                    )
                    VALUES (
                        %(id)s,
                        %(model_registry_id)s,
                        %(model_runtime_mapping_id)s,
                        %(model_output_id)s,
                        %(run_id)s,
                        %(environment)s,
                        %(broker)s,
                        %(account_id)s,
                        %(symbol)s,
                        %(side)s,
                        %(quantity)s,
                        %(order_type)s,
                        %(limit_price)s,
                        %(status)s,
                        %(decision_reason)s,
                        %(broker_order_id)s,
                        %(idempotency_key)s,
                        %(requested_at)s,
                        %(submitted_at)s,
                        %(completed_at)s,
                        %(request_payload)s,
                        %(broker_response)s,
                        %(metadata)s
                    )
                    ON CONFLICT (idempotency_key) WHERE idempotency_key IS NOT NULL DO UPDATE SET
                        status = EXCLUDED.status,
                        broker_order_id = COALESCE(EXCLUDED.broker_order_id, trade_orders.broker_order_id),
                        broker_response = EXCLUDED.broker_response,
                        metadata = EXCLUDED.metadata,
                        completed_at = EXCLUDED.completed_at
                    """,
                    record,
                )
                written += 1
        connection.commit()

    return {"status": "recorded", "trade_orders_recorded": written}


def _load_active_runtime_mapping(cursor: Any, producer_id: str, producer_version: str, environment: str) -> dict[str, Any]:
    cursor.execute(
        """
        SELECT
            mrm.id,
            mrm.model_registry_id,
            mrm.account_id,
            COALESCE(mrm.metadata->>'broker_provider', mr.metadata->>'broker_provider', 'alpaca') AS broker,
            COALESCE(mrm.metadata->>'broker_instance_id', mr.metadata->>'broker_instance_id') AS broker_instance_id
        FROM model_runtime_mappings mrm
        JOIN model_registry mr
          ON mr.id = mrm.model_registry_id
        WHERE mrm.environment = %(environment)s
          AND mr.model_id = %(producer_id)s
          AND mr.model_version = %(producer_version)s
          AND mrm.active_until IS NULL
        ORDER BY mrm.active_from DESC
        LIMIT 1
        """,
        {"environment": environment, "producer_id": producer_id, "producer_version": producer_version},
    )
    row = cursor.fetchone()
    if not row:
        return {}
    return {
        "id": str(row[0]),
        "model_registry_id": str(row[1]),
        "account_id": row[2],
        "broker": row[3],
        "broker_instance_id": row[4],
    }


def _trade_order_record(
    *,
    payload: dict[str, Any],
    response: dict[str, Any],
    result: dict[str, Any],
    mapping: dict[str, Any],
    output_id: str,
    run_id: str,
    environment: str,
    producer_id: str,
    producer_version: str,
    requested_at: datetime,
    order_index: int,
) -> dict[str, Any]:
    from psycopg.types.json import Jsonb

    quantity = _decimal_or_none(payload.get("qty") or response.get("qty"))
    status = str(response.get("status") or result.get("status") or "unknown")
    submitted_at = _parse_datetime(response.get("submitted_at")) if response else None
    completed_at = _parse_datetime(response.get("filled_at") or response.get("canceled_at") or response.get("expired_at")) if response else None
    client_order_id = str(payload.get("client_order_id") or f"{producer_id}:{run_id}:{order_index}")
    return {
        "id": str(uuid4()),
        "model_registry_id": mapping.get("model_registry_id"),
        "model_runtime_mapping_id": mapping.get("id"),
        "model_output_id": output_id,
        "run_id": run_id,
        "environment": environment,
        "broker": mapping.get("broker") or "alpaca",
        "account_id": mapping.get("account_id") or os.getenv("QSENTIA_BROKER_ACCOUNT_ID") or os.getenv("APCA_ACCOUNT_ID"),
        "symbol": _payload_symbol(payload, response),
        "side": _payload_side(payload, response),
        "quantity": quantity,
        "order_type": str(payload.get("type") or response.get("type") or "market"),
        "limit_price": _decimal_or_none(payload.get("limit_price") or response.get("limit_price")),
        "status": status,
        "decision_reason": "WORLD_MODEL-RL option leg signal",
        "broker_order_id": response.get("id"),
        "idempotency_key": client_order_id,
        "requested_at": requested_at,
        "submitted_at": submitted_at,
        "completed_at": completed_at,
        "request_payload": Jsonb(payload),
        "broker_response": Jsonb(response or result),
        "metadata": Jsonb(
            {
                "producer_id": producer_id,
                "producer_version": producer_version,
                "broker_instance_id": mapping.get("broker_instance_id"),
                "result_status": result.get("status"),
                "legs": payload.get("legs") or [],
            }
        ),
    }


def _payload_symbol(payload: dict[str, Any], response: dict[str, Any]) -> str:
    if payload.get("symbol") or response.get("symbol"):
        return str(payload.get("symbol") or response.get("symbol") or "").upper()
    legs = payload.get("legs") if isinstance(payload.get("legs"), list) else []
    symbols = [str(leg.get("symbol") or "").upper() for leg in legs if isinstance(leg, dict) and leg.get("symbol")]
    return ",".join(symbols)[:255]


def _payload_side(payload: dict[str, Any], response: dict[str, Any]) -> str:
    if payload.get("side") or response.get("side"):
        return str(payload.get("side") or response.get("side") or "").lower()
    legs = payload.get("legs") if isinstance(payload.get("legs"), list) else []
    sides = [str(leg.get("side") or "").lower() for leg in legs if isinstance(leg, dict) and leg.get("side")]
    return "mleg:" + ",".join(sides[:4]) if sides else "mleg"


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
