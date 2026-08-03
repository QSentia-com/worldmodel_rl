from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .artifact_manager import EXPECTED_MODEL_VERSION, EXPECTED_SELECTED_MODEL, validate_artifacts
from .config import SignalRuntimeConfig, bool_env


MODEL_ID = "qsentia-world-model-rl"
PRODUCER_ID = "qsentia-world-model-rl"
PRODUCER_VERSION = EXPECTED_MODEL_VERSION


def run_signal_inference(
    artifact_dir: Path | str | None = None,
    config: SignalRuntimeConfig | None = None,
) -> dict[str, Any]:
    config = config or SignalRuntimeConfig.from_env()
    resolved_artifact_dir = Path(artifact_dir or os.getenv("QSENTIA_ARTIFACT_DIR", "/app/artifacts/current"))
    validate_artifacts(resolved_artifact_dir)

    manifest = _read_json(resolved_artifact_dir / "artifact_manifest.json")
    metadata = _read_json(resolved_artifact_dir / "model_metadata.json")
    live_state = _read_json(resolved_artifact_dir / "live_state.json")
    action_mapping = _read_json(resolved_artifact_dir / "deployment_action_mapping.json")

    now = datetime.now(timezone.utc)
    run_mode = _run_mode()
    signal_date = _signal_date()
    selected_rows = _selected_decisions(resolved_artifact_dir)
    order_map = _order_map(resolved_artifact_dir)
    candidate_rows = _candidate_rows(selected_rows, run_mode, signal_date)
    stale_signal_used = False

    if not candidate_rows and bool_env("QSENTIA_ALLOW_STALE_OPTION_SIGNAL", False):
        candidate_rows = _latest_candidate_rows(selected_rows, run_mode, signal_date)
        stale_signal_used = bool(candidate_rows)

    orders = _orders_from_candidates(
        candidate_rows=candidate_rows,
        order_map=order_map,
        run_mode=run_mode,
        batch_id=os.getenv("AWS_BATCH_JOB_ID") or now.strftime("%Y%m%dT%H%M%SZ"),
    )
    max_orders = int(os.getenv("QSENTIA_MAX_OPTION_ORDERS", "25"))
    orders = orders[:max_orders]

    live_trading_enabled = _bool(live_state.get("live_trading_enabled")) and _bool(action_mapping.get("live_trading_enabled"))
    signal_label = f"{run_mode}:{signal_date}"
    signal = {
        "asof": now.isoformat(),
        "label": signal_label,
        "signal": "option_entry_exit" if orders else "no_current_signal",
        "confidence": _average_confidence(candidate_rows),
        "metadata": {
            "producer_id": os.getenv("QSENTIA_PRODUCER_ID", PRODUCER_ID),
            "producer_version": os.getenv("QSENTIA_PRODUCER_VERSION", PRODUCER_VERSION),
            "model_id": os.getenv("QSENTIA_MODEL_ID", MODEL_ID),
            "display_name": "WORLD_MODEL-RL",
            "selected_model": manifest["selected_model"],
            "model_version": manifest["model_version"],
            "strategy_name": metadata.get("strategy_name"),
            "strategy_mode": metadata.get("strategy_mode"),
            "asset_symbol": metadata.get("asset_symbol"),
            "artifact_live_trading_enabled": live_trading_enabled,
            "order_mapping_required": live_state.get("order_mapping_required"),
            "run_mode": run_mode,
            "signal_date": signal_date,
            "candidate_rows": len(candidate_rows),
            "option_orders": len(orders),
            "stale_signal_used": stale_signal_used,
            "last_backtest_date": live_state.get("last_backtest_date"),
            "deployment_blocker": action_mapping.get("deployment_blocker"),
        },
    }

    payload = {
        "ts": now.isoformat(),
        "signal": signal,
        "trade": {
            "broker": "alpaca",
            "account_id": os.getenv("QSENTIA_BROKER_ACCOUNT_ID") or os.getenv("APCA_ACCOUNT_ID") or os.getenv("ALPACA_ACCOUNT_ID"),
            "broker_instance_id": os.getenv("QSENTIA_BROKER_INSTANCE_ID", "broker-world-model-rl-alpaca"),
            "asset_type": "options_mleg",
            "run_mode": run_mode,
            "artifact_live_trading_enabled": live_trading_enabled,
            "orders": orders,
        },
        "output_path": str(config.output_path),
    }
    _write_signal_output(payload, config)
    return payload


def _run_mode() -> str:
    raw = os.getenv("QSENTIA_RUN_MODE") or os.getenv("WORLD_MODEL_RL_RUN_MODE") or "entry"
    value = raw.strip().lower()
    if value not in {"entry", "exit", "rebalance"}:
        raise RuntimeError(f"Unsupported WORLD_MODEL-RL run mode: {raw}")
    return "entry" if value == "rebalance" else value


def _signal_date() -> str:
    explicit = os.getenv("QSENTIA_SIGNAL_DATE") or os.getenv("QSENTIA_EFFECTIVE_DATE") or os.getenv("QSENTIA_ASOF_DATE")
    if explicit:
        return explicit[:10]
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _selected_decisions(artifact_dir: Path) -> list[dict[str, str]]:
    rows = _read_csv(artifact_dir / "selected_decisions_live.csv")
    return [
        row
        for row in rows
        if row.get("selected_model") == EXPECTED_SELECTED_MODEL and row.get("selected_model_version") == EXPECTED_MODEL_VERSION
    ]


def _order_map(artifact_dir: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(artifact_dir / "live_order_map.csv"):
        key = (str(row.get("ticker") or "").upper(), str(row.get("entry_date") or "")[:10])
        if key[0] and key[1]:
            grouped[key].append(row)
    return grouped


def _candidate_rows(rows: list[dict[str, str]], run_mode: str, signal_date: str) -> list[dict[str, str]]:
    date_column = "exit_date" if run_mode == "exit" else "entry_date"
    return [row for row in rows if str(row.get(date_column) or "")[:10] == signal_date]


def _latest_candidate_rows(rows: list[dict[str, str]], run_mode: str, signal_date: str) -> list[dict[str, str]]:
    date_column = "exit_date" if run_mode == "exit" else "entry_date"
    eligible_dates = sorted({str(row.get(date_column) or "")[:10] for row in rows if str(row.get(date_column) or "")[:10] <= signal_date})
    if not eligible_dates:
        return []
    return [row for row in rows if str(row.get(date_column) or "")[:10] == eligible_dates[-1]]


def _orders_from_candidates(
    *,
    candidate_rows: list[dict[str, str]],
    order_map: dict[tuple[str, str], list[dict[str, str]]],
    run_mode: str,
    batch_id: str,
) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    for index, decision in enumerate(candidate_rows):
        key = (str(decision.get("ticker") or "").upper(), str(decision.get("entry_date") or "")[:10])
        mapped_rows = order_map.get(key, [])
        legs = _legs_from_map_rows(mapped_rows, run_mode)
        if not legs:
            continue
        qty = max(1, int(float(decision.get("chosen_scale") or "1")))
        action = str(decision.get("action") or decision.get("original_model_action") or "")
        orders.append(
            {
                "order_class": "mleg",
                "qty": str(qty),
                "type": os.getenv("QSENTIA_OPTION_ORDER_TYPE", "market"),
                "time_in_force": "day",
                "client_order_id": _client_order_id(batch_id, key[0], run_mode, index),
                "legs": legs,
                "metadata": {
                    "ticker": key[0],
                    "entry_date": key[1],
                    "exit_date": str(decision.get("exit_date") or "")[:10],
                    "event_date": str(decision.get("event_date") or "")[:10],
                    "action": action,
                    "live_execution_action": decision.get("live_execution_action"),
                    "confidence": _float_or_none(decision.get("confidence")),
                    "score": _float_or_none(decision.get("score")),
                    "execution_mode": decision.get("execution_mode"),
                    "source": "world_rl_live_order_map",
                },
            }
        )
    return orders


def _legs_from_map_rows(rows: list[dict[str, str]], run_mode: str) -> list[dict[str, str]]:
    side_column = "exit_order_side" if run_mode == "exit" else "entry_order_side"
    legs: list[dict[str, str]] = []
    for row in rows:
        symbol = _alpaca_option_symbol(row.get("option_ticker"))
        side, intent = _side_and_intent(row.get(side_column))
        if not symbol or not side or not intent:
            continue
        legs.append(
            {
                "symbol": symbol,
                "ratio_qty": str(max(1, int(float(row.get("quantity") or "1")))),
                "side": side,
                "position_intent": intent,
            }
        )
    return legs


def _side_and_intent(raw: str | None) -> tuple[str | None, str | None]:
    value = str(raw or "").strip().upper()
    mapping = {
        "BUY_TO_OPEN": ("buy", "buy_to_open"),
        "SELL_TO_OPEN": ("sell", "sell_to_open"),
        "BUY_TO_CLOSE": ("buy", "buy_to_close"),
        "SELL_TO_CLOSE": ("sell", "sell_to_close"),
    }
    return mapping.get(value, (None, None))


def _alpaca_option_symbol(raw: str | None) -> str:
    symbol = str(raw or "").strip().upper()
    if symbol.startswith("O:"):
        symbol = symbol[2:]
    return symbol


def _client_order_id(batch_id: str, ticker: str, run_mode: str, index: int) -> str:
    digest = hashlib.sha1(f"world-model-rl:{batch_id}:{ticker}:{run_mode}:{index}".encode("utf-8")).hexdigest()[:16]
    return f"world-rl-{digest}"


def _average_confidence(rows: list[dict[str, str]]) -> float | None:
    values = [value for value in (_float_or_none(row.get("confidence")) for row in rows) if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(raw: Any) -> float | None:
    try:
        if raw is None or raw == "":
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _write_signal_output(payload: dict[str, Any], config: SignalRuntimeConfig) -> None:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if not config.append_signal_log:
        return
    log_path = config.output_path.with_name("signal_history.jsonl")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str, separators=(",", ":")) + "\n")
