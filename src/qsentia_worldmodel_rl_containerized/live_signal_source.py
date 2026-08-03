from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import s3fs
except ImportError as exc:  # pragma: no cover
    s3fs = None
    _S3FS_IMPORT_ERROR = exc
else:  # pragma: no cover
    _S3FS_IMPORT_ERROR = None

from .artifact_manager import EXPECTED_MODEL_VERSION, EXPECTED_SELECTED_MODEL
from .config import LakeFSRuntimeConfig


DECISION_COLUMNS = [
    "model",
    "row_id",
    "ticker",
    "sector",
    "entry_date",
    "event_date",
    "exit_date",
    "action",
    "horizon",
    "pred_mean",
    "pred_std",
    "pred_cvar",
    "prob_profit",
    "utility",
    "confidence",
    "score",
    "split",
    "fold_test_year",
    "chosen_scale",
    "param_risk_aversion",
    "param_min_utility",
    "param_min_edge",
    "param_min_confidence",
    "param_top_fraction",
    "param_trade_weight",
    "param_max_day_gross",
    "param_wing_mult",
    "event_date_label",
    "spot_entry",
    "implied_move",
    "call_contract",
    "put_contract",
    "call_strike",
    "put_strike",
    "call_dte",
    "put_dte",
    "straddle_pre",
    "original_model_action",
    "execution_mode",
    "live_execution_action",
    "v12b_inverse_scale",
    "selected_model",
    "selected_model_version",
]

ORDER_MAP_COLUMNS = [
    "ticker",
    "entry_date",
    "original_model_action",
    "original_structure",
    "execution_mode",
    "live_execution_action",
    "leg_role",
    "option_ticker",
    "quantity",
    "entry_order_side",
    "exit_order_side",
]


def maybe_apply_current_signal_source(config: LakeFSRuntimeConfig) -> dict[str, Any] | None:
    """Overlay same-day live WORLD_MODEL-RL signals onto the downloaded artifact.

    The artifact remains the source of model identity and backtest metadata. This only replaces
    the per-run live decision/order files when an explicit fresh source is configured.
    """

    blotter_uri = _first_env(
        "QSENTIA_WORLD_RL_CURRENT_BLOTTER_URI",
        "WORLD_MODEL_RL_CURRENT_BLOTTER_URI",
        "QSENTIA_WORLD_RL_CURRENT_BLOTTER_PATH",
        "WORLD_MODEL_RL_CURRENT_BLOTTER_PATH",
    )
    decisions_uri = _first_env(
        "QSENTIA_WORLD_RL_CURRENT_DECISIONS_URI",
        "WORLD_MODEL_RL_CURRENT_DECISIONS_URI",
        "QSENTIA_WORLD_RL_CURRENT_DECISIONS_PATH",
        "WORLD_MODEL_RL_CURRENT_DECISIONS_PATH",
    )
    order_map_uri = _first_env(
        "QSENTIA_WORLD_RL_CURRENT_ORDER_MAP_URI",
        "WORLD_MODEL_RL_CURRENT_ORDER_MAP_URI",
        "QSENTIA_WORLD_RL_CURRENT_ORDER_MAP_PATH",
        "WORLD_MODEL_RL_CURRENT_ORDER_MAP_PATH",
    )

    if not blotter_uri and not decisions_uri and not order_map_uri:
        return None
    if blotter_uri and (decisions_uri or order_map_uri):
        raise RuntimeError("Configure either a WORLD_MODEL-RL current blotter or decisions/order-map files, not both.")
    if bool(decisions_uri) != bool(order_map_uri):
        raise RuntimeError("WORLD_MODEL-RL current decisions and order-map sources must be provided together.")

    signal_date = _signal_date()
    run_mode = _run_mode()
    if blotter_uri:
        raw_blotter = _read_rows(blotter_uri, config)
        decisions, order_map = _from_blotter(raw_blotter, signal_date=signal_date, run_mode=run_mode)
        source_type = "blotter"
        source_uris = {"blotter_uri": blotter_uri}
    else:
        raw_decisions = _read_rows(str(decisions_uri), config)
        raw_order_map = _read_rows(str(order_map_uri), config)
        decisions = _normalize_decisions(raw_decisions, signal_date=signal_date, run_mode=run_mode)
        order_map = _normalize_order_map(raw_order_map, decisions)
        source_type = "decisions_order_map"
        source_uris = {"decisions_uri": decisions_uri, "order_map_uri": order_map_uri}

    if not decisions:
        raise RuntimeError(f"WORLD_MODEL-RL current {source_type} source had no same-day decision rows for {signal_date}.")
    if not order_map:
        raise RuntimeError(f"WORLD_MODEL-RL current {source_type} source had no option leg rows for {signal_date}.")

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = config.artifact_dir / "selected_decisions_live.csv"
    order_map_path = config.artifact_dir / "live_order_map.csv"
    _write_rows(decisions_path, decisions, DECISION_COLUMNS)
    _write_rows(order_map_path, order_map, ORDER_MAP_COLUMNS)

    report = {
        "status": "applied",
        "source_type": source_type,
        "source_uris": source_uris,
        "signal_date": signal_date,
        "run_mode": run_mode,
        "decision_rows": len(decisions),
        "order_map_rows": len(order_map),
        "selected_model": EXPECTED_SELECTED_MODEL,
        "selected_model_version": EXPECTED_MODEL_VERSION,
        "decisions_path": str(decisions_path),
        "order_map_path": str(order_map_path),
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (config.artifact_dir / "live_signal_source_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return report


def _from_blotter(
    rows: list[dict[str, Any]],
    *,
    signal_date: str,
    run_mode: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    decisions: list[dict[str, str]] = []
    order_rows: list[dict[str, str]] = []
    for index, raw in enumerate(rows):
        ticker = _clean(raw.get("ticker") or raw.get("symbol")).upper()
        action = _clean(raw.get("action") or raw.get("original_model_action")).lower()
        entry_date = _date_string(raw.get("entry_date") or raw.get("date") or raw.get("timestamp") or signal_date)
        exit_date = _date_string(raw.get("exit_date") or raw.get("planned_exit_date") or entry_date)
        event_date = _date_string(raw.get("event_date") or entry_date)
        relevant_date = exit_date if run_mode == "exit" else entry_date
        if not ticker or relevant_date != signal_date:
            continue
        if action not in {"long_vol", "short_vol_defined"}:
            raise RuntimeError(f"WORLD_MODEL-RL current blotter has unsupported action {action!r} for {ticker}.")

        qty = _positive_int(raw.get("chosen_scale") or raw.get("qty") or raw.get("quantity") or 1)
        decision = _decision_row(
            raw,
            row_id=str(raw.get("row_id") or index),
            ticker=ticker,
            entry_date=entry_date,
            event_date=event_date,
            exit_date=exit_date,
            action=action,
            qty=qty,
        )
        decisions.append(decision)
        order_rows.extend(_order_rows_from_blotter_row(raw, decision))
    return decisions, order_rows


def _normalize_decisions(
    rows: list[dict[str, Any]],
    *,
    signal_date: str,
    run_mode: str,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(rows):
        ticker = _clean(raw.get("ticker") or raw.get("symbol")).upper()
        action = _clean(raw.get("action") or raw.get("original_model_action")).lower()
        entry_date = _date_string(raw.get("entry_date") or raw.get("date") or raw.get("timestamp") or signal_date)
        exit_date = _date_string(raw.get("exit_date") or raw.get("planned_exit_date") or entry_date)
        event_date = _date_string(raw.get("event_date") or entry_date)
        relevant_date = exit_date if run_mode == "exit" else entry_date
        if not ticker or relevant_date != signal_date:
            continue
        if action not in {"long_vol", "short_vol_defined"}:
            raise RuntimeError(f"WORLD_MODEL-RL current decisions have unsupported action {action!r} for {ticker}.")
        normalized.append(
            _decision_row(
                raw,
                row_id=str(raw.get("row_id") or index),
                ticker=ticker,
                entry_date=entry_date,
                event_date=event_date,
                exit_date=exit_date,
                action=action,
                qty=_positive_int(raw.get("chosen_scale") or raw.get("qty") or raw.get("quantity") or 1),
            )
        )
    return normalized


def _normalize_order_map(rows: list[dict[str, Any]], decisions: list[dict[str, str]]) -> list[dict[str, str]]:
    keys = {(row["ticker"].upper(), row["entry_date"], row["action"]) for row in decisions}
    normalized: list[dict[str, str]] = []
    for raw in rows:
        ticker = _clean(raw.get("ticker") or raw.get("symbol")).upper()
        entry_date = _date_string(raw.get("entry_date") or raw.get("date"))
        action = _clean(raw.get("original_model_action") or raw.get("action")).lower()
        if (ticker, entry_date, action) not in keys:
            continue
        option_ticker = _normalize_option_symbol(raw.get("option_ticker") or raw.get("option_symbol") or raw.get("contract"))
        if not option_ticker:
            continue
        entry_side = _side(raw.get("entry_order_side") or raw.get("side"))
        exit_side = _side(raw.get("exit_order_side")) or _exit_side(entry_side)
        if not entry_side or not exit_side:
            raise RuntimeError(f"WORLD_MODEL-RL order map row is missing side for {ticker} {entry_date}.")
        normalized.append(
            {
                "ticker": ticker,
                "entry_date": entry_date,
                "original_model_action": action,
                "original_structure": _clean(raw.get("original_structure") or _structure_for_action(action)),
                "execution_mode": "exact_same_leg_inverse",
                "live_execution_action": _live_execution_action(action),
                "leg_role": _clean(raw.get("leg_role") or "option_leg"),
                "option_ticker": option_ticker,
                "quantity": str(_positive_int(raw.get("quantity") or raw.get("qty") or 1)),
                "entry_order_side": entry_side,
                "exit_order_side": exit_side,
            }
        )
    return normalized


def _order_rows_from_blotter_row(raw: dict[str, Any], decision: dict[str, str]) -> list[dict[str, str]]:
    action = decision["action"]
    qty = str(_positive_int(raw.get("quantity") or raw.get("qty") or decision.get("chosen_scale") or 1))
    base = {
        "ticker": decision["ticker"],
        "entry_date": decision["entry_date"],
        "original_model_action": action,
        "original_structure": _structure_for_action(action),
        "execution_mode": "exact_same_leg_inverse",
        "live_execution_action": _live_execution_action(action),
    }
    rows: list[dict[str, str]] = []

    def add(role: str, symbol: Any, entry_side: str) -> None:
        option_ticker = _normalize_option_symbol(symbol)
        if not option_ticker:
            raise RuntimeError(
                f"WORLD_MODEL-RL current blotter row for {decision['ticker']} is missing {role} option ticker."
            )
        row = dict(base)
        row.update(
            {
                "leg_role": role,
                "option_ticker": option_ticker,
                "quantity": qty,
                "entry_order_side": entry_side,
                "exit_order_side": _exit_side(entry_side),
            }
        )
        rows.append(row)

    if action == "long_vol":
        add("original_call", _first(raw, "call_contract", "long_call", "short_call"), "SELL_TO_OPEN")
        add("original_put", _first(raw, "put_contract", "long_put", "short_put"), "SELL_TO_OPEN")
    elif action == "short_vol_defined":
        add("original_short_call", raw.get("short_call"), "BUY_TO_OPEN")
        add("original_short_put", raw.get("short_put"), "BUY_TO_OPEN")
        add("original_long_call_wing", raw.get("long_call_wing"), "SELL_TO_OPEN")
        add("original_long_put_wing", raw.get("long_put_wing"), "SELL_TO_OPEN")
    return rows


def _decision_row(
    raw: dict[str, Any],
    *,
    row_id: str,
    ticker: str,
    entry_date: str,
    event_date: str,
    exit_date: str,
    action: str,
    qty: int,
) -> dict[str, str]:
    row = {column: "" for column in DECISION_COLUMNS}
    row.update({key: _clean(value) for key, value in raw.items() if key in row})
    row.update(
        {
            "model": _clean(raw.get("model") or "live_current_signal"),
            "row_id": row_id,
            "ticker": ticker,
            "entry_date": entry_date,
            "event_date": event_date,
            "exit_date": exit_date,
            "action": action,
            "chosen_scale": str(qty),
            "original_model_action": action,
            "execution_mode": "exact_same_leg_inverse",
            "live_execution_action": _live_execution_action(action),
            "selected_model": EXPECTED_SELECTED_MODEL,
            "selected_model_version": EXPECTED_MODEL_VERSION,
        }
    )
    if not row["confidence"]:
        row["confidence"] = "1"
    if not row["score"]:
        row["score"] = row["confidence"]
    return row


def _read_rows(uri: str, config: LakeFSRuntimeConfig) -> list[dict[str, Any]]:
    text = _read_text(uri, config)
    if uri.lower().endswith(".json"):
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ["rows", "signals", "orders", "blotter", "decisions"]:
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
            return [payload]
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        raise RuntimeError(f"Unsupported WORLD_MODEL-RL JSON source payload in {uri}.")
    return list(csv.DictReader(text.splitlines()))


def _read_text(uri: str, config: LakeFSRuntimeConfig) -> str:
    value = str(uri).strip()
    if value.startswith("lakefs://"):
        value = "s3://" + value.removeprefix("lakefs://").strip("/")
    if value.startswith("s3://"):
        filesystem = _s3_filesystem(config)
        with filesystem.open(value, "r") as handle:
            return handle.read()
    return Path(value).read_text(encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


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
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _date_string(value: Any) -> str:
    raw = _clean(value)
    if not raw:
        return ""
    return raw[:10]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def _positive_int(value: Any) -> int:
    try:
        return max(1, int(float(value)))
    except (TypeError, ValueError):
        return 1


def _normalize_option_symbol(value: Any) -> str:
    symbol = _clean(value).upper()
    if not symbol:
        return ""
    if not symbol.startswith("O:"):
        symbol = f"O:{symbol}"
    return symbol


def _side(value: Any) -> str:
    side = _clean(value).upper()
    if side in {"BUY_TO_OPEN", "SELL_TO_OPEN", "BUY_TO_CLOSE", "SELL_TO_CLOSE"}:
        return side
    if side == "BUY":
        return "BUY_TO_OPEN"
    if side == "SELL":
        return "SELL_TO_OPEN"
    return ""


def _exit_side(entry_side: str) -> str:
    return {
        "BUY_TO_OPEN": "SELL_TO_CLOSE",
        "SELL_TO_OPEN": "BUY_TO_CLOSE",
        "BUY_TO_CLOSE": "SELL_TO_OPEN",
        "SELL_TO_CLOSE": "BUY_TO_OPEN",
    }.get(entry_side, "")


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if _clean(value):
            return value
    return ""


def _structure_for_action(action: str) -> str:
    return "long_straddle" if action == "long_vol" else "short_iron_fly"


def _live_execution_action(action: str) -> str:
    if action == "long_vol":
        return "take_opposite_side_of_same_straddle"
    if action == "short_vol_defined":
        return "take_opposite_side_of_same_iron_fly"
    return "do_not_trade_unknown_action"


def _s3_filesystem(config: LakeFSRuntimeConfig) -> Any:
    if s3fs is None:
        raise RuntimeError("Install s3fs to read WORLD_MODEL-RL current signal sources.") from _S3FS_IMPORT_ERROR
    return s3fs.S3FileSystem(**config.storage_options)
