from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

try:
    import requests
except ImportError as exc:  # pragma: no cover
    requests = None
    _REQUESTS_IMPORT_ERROR = exc
else:  # pragma: no cover
    _REQUESTS_IMPORT_ERROR = None

try:
    import s3fs
except ImportError as exc:  # pragma: no cover
    s3fs = None
    _S3FS_IMPORT_ERROR = exc
else:  # pragma: no cover
    _S3FS_IMPORT_ERROR = None

from .artifact_manager import EXPECTED_MODEL_VERSION, EXPECTED_SELECTED_MODEL
from .config import LakeFSRuntimeConfig, bool_env
from .live_signal_source import DECISION_COLUMNS, ORDER_MAP_COLUMNS, _from_blotter, _run_mode, _signal_date, _write_rows


class LiveDataClient(Protocol):
    def benzinga_earnings(self, start: date, end: date, *, limit: int) -> list[dict[str, Any]]:
        ...

    def stock_daily_bars(self, ticker: str, start: date, end: date) -> list[dict[str, Any]]:
        ...

    def option_contracts(self, ticker: str, start: date, end: date) -> list[dict[str, Any]]:
        ...

    def option_previous_bar(self, option_ticker: str) -> dict[str, Any] | None:
        ...

    def option_snapshot(self, underlying: str, option_ticker: str) -> dict[str, Any] | None:
        ...


@dataclass(frozen=True)
class AutonomousSignalConfig:
    lookahead_days: int
    max_raw_events: int
    max_trades: int
    min_event_importance: float
    min_spot_price: float
    option_min_dte: int
    option_max_dte: int
    default_exit_business_days: int
    option_wing_mult: float
    min_wing_width_pct: float
    short_vol_threshold: float
    min_option_mark: float
    max_contract_qty: int
    current_signal_prefix: str
    publish_current_signals: bool

    @classmethod
    def from_env(cls) -> "AutonomousSignalConfig":
        return cls(
            lookahead_days=_int_env("QSENTIA_WORLD_RL_LIVE_LOOKAHEAD_DAYS", 7),
            max_raw_events=_int_env("QSENTIA_WORLD_RL_MAX_RAW_EVENTS", 300),
            max_trades=_int_env("QSENTIA_WORLD_RL_MAX_CURRENT_TRADES", 1),
            min_event_importance=_float_env("QSENTIA_WORLD_RL_MIN_EVENT_IMPORTANCE", 3.0),
            min_spot_price=_float_env("QSENTIA_WORLD_RL_MIN_SPOT_PRICE", 10.0),
            option_min_dte=_int_env("QSENTIA_WORLD_RL_OPTION_MIN_DTE", 10),
            option_max_dte=_int_env("QSENTIA_WORLD_RL_OPTION_MAX_DTE", 75),
            default_exit_business_days=_int_env("QSENTIA_WORLD_RL_EXIT_BUSINESS_DAYS", 5),
            option_wing_mult=_float_env("QSENTIA_WORLD_RL_OPTION_WING_MULT", 1.10),
            min_wing_width_pct=_float_env("QSENTIA_WORLD_RL_MIN_WING_WIDTH_PCT", 0.06),
            short_vol_threshold=_float_env("QSENTIA_WORLD_RL_SHORT_VOL_THRESHOLD", 1.20),
            min_option_mark=_float_env("QSENTIA_WORLD_RL_MIN_OPTION_MARK", 0.05),
            max_contract_qty=_int_env("QSENTIA_WORLD_RL_MAX_CONTRACT_QTY", 3),
            current_signal_prefix=os.getenv(
                "QSENTIA_WORLD_RL_CURRENT_SIGNAL_PREFIX",
                "inference_outputs/world-model-rl/current-signals",
            ).strip("/"),
            publish_current_signals=bool_env("QSENTIA_WORLD_RL_PUBLISH_CURRENT_SIGNALS", True),
        )


class MassiveLiveDataClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        sleep_seconds: float | None = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY") or "").strip()
        self.base_url = (base_url or os.getenv("MASSIVE_API_BASE_URL") or "https://api.massive.com").rstrip("/")
        self.sleep_seconds = sleep_seconds if sleep_seconds is not None else _float_env(
            "QSENTIA_WORLD_RL_MASSIVE_SLEEP_SECONDS",
            0.15,
        )
        if not self.api_key:
            raise RuntimeError("Missing MASSIVE_API_KEY/POLYGON_API_KEY for WORLD_MODEL-RL autonomous signal generation.")

    def benzinga_earnings(self, start: date, end: date, *, limit: int) -> list[dict[str, Any]]:
        return self._paged(
            "/benzinga/v1/earnings",
            {
                "date.gte": start.isoformat(),
                "date.lte": end.isoformat(),
                "limit": str(limit),
                "sort": "date.asc",
            },
            max_rows=max(1, limit),
        )

    def stock_daily_bars(self, ticker: str, start: date, end: date) -> list[dict[str, Any]]:
        return self._paged(
            f"/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start.isoformat()}/{end.isoformat()}",
            {
                "adjusted": "true",
                "sort": "asc",
                "limit": "5000",
            },
            max_rows=5000,
        )

    def option_contracts(self, ticker: str, start: date, end: date) -> list[dict[str, Any]]:
        row_limit = _int_env("QSENTIA_WORLD_RL_MAX_OPTION_CONTRACT_ROWS", 1000)
        return self._paged(
            "/v3/reference/options/contracts",
            {
                "underlying_ticker": ticker.upper(),
                "expiration_date.gte": start.isoformat(),
                "expiration_date.lte": end.isoformat(),
                "limit": "1000",
                "sort": "expiration_date",
                "order": "asc",
            },
            max_rows=max(1, row_limit),
        )

    def option_previous_bar(self, option_ticker: str) -> dict[str, Any] | None:
        symbol = normalize_occ_symbol(option_ticker, with_prefix=True)
        payload = self._get(f"/v2/aggs/ticker/{symbol}/prev", {"adjusted": "true"})
        rows = _payload_rows(payload)
        return rows[0] if rows else None

    def option_snapshot(self, underlying: str, option_ticker: str) -> dict[str, Any] | None:
        symbol = normalize_occ_symbol(option_ticker, with_prefix=True)
        payload = self._get(f"/v3/snapshot/options/{underlying.upper()}/{symbol}", {})
        results = payload.get("results") if isinstance(payload, dict) else None
        return results if isinstance(results, dict) else None

    def _paged(self, path: str, params: dict[str, str], *, max_rows: int | None = None) -> list[dict[str, Any]]:
        payload = self._get(path, params)
        rows = _payload_rows(payload)
        if max_rows is not None and len(rows) >= max_rows:
            return rows[:max_rows]
        next_url = payload.get("next_url") if isinstance(payload, dict) else None
        while next_url and (max_rows is None or len(rows) < max_rows):
            payload = self._get(str(next_url), {}, full_url=True)
            rows.extend(_payload_rows(payload))
            if max_rows is not None and len(rows) >= max_rows:
                return rows[:max_rows]
            next_url = payload.get("next_url") if isinstance(payload, dict) else None
        return rows

    def _get(self, path_or_url: str, params: dict[str, str], *, full_url: bool = False) -> dict[str, Any]:
        http = _requests()
        query = dict(params)
        query["apiKey"] = self.api_key
        url = path_or_url if full_url else f"{self.base_url}{path_or_url}"
        for attempt in range(3):
            response = http.get(url, params=query, headers={"accept": "application/json", "connection": "close"}, timeout=(10, 60))
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(max(2.0, self.sleep_seconds * 10))
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"Massive request failed {response.status_code} for {path_or_url}: {response.text[:500]}")
            time.sleep(self.sleep_seconds)
            payload = response.json()
            return payload if isinstance(payload, dict) else {"results": payload}
        raise RuntimeError(f"Massive request failed after retries: {path_or_url}")


def maybe_apply_autonomous_current_signal_source(
    config: LakeFSRuntimeConfig,
    *,
    client: LiveDataClient | None = None,
) -> dict[str, Any] | None:
    if not bool_env("QSENTIA_WORLD_RL_AUTONOMOUS_SIGNAL_ENABLED", False):
        return None
    if _explicit_current_source_is_configured():
        return None

    signal_date = _signal_date()
    run_mode = _run_mode()
    autonomous_config = AutonomousSignalConfig.from_env()

    if run_mode == "exit":
        restored = _restore_exit_signal_bundle(config, autonomous_config, signal_date)
        if restored is not None:
            return restored
        return _write_no_current_signal_report(
            config,
            signal_date=signal_date,
            run_mode=run_mode,
            reason="no autonomous WORLD_MODEL-RL entries are due for exit today",
            metadata={"source_type": "autonomous_live_exit_state"},
        )

    data_client = client or MassiveLiveDataClient()
    blotter_rows, generation_summary = build_autonomous_blotter_rows(
        client=data_client,
        signal_date=signal_date,
        settings=autonomous_config,
    )
    if not blotter_rows:
        return _write_no_current_signal_report(
            config,
            signal_date=signal_date,
            run_mode=run_mode,
            reason="autonomous live scorer found no execution-ready current signal",
            metadata={"source_type": "autonomous_live_entry_scorer", **generation_summary},
        )

    decisions, order_map = _from_blotter(blotter_rows, signal_date=signal_date, run_mode=run_mode)
    if not decisions or not order_map:
        return _write_no_current_signal_report(
            config,
            signal_date=signal_date,
            run_mode=run_mode,
            reason="autonomous live scorer produced no artifact-native option legs",
            metadata={"source_type": "autonomous_live_entry_scorer", **generation_summary},
        )

    _write_current_signal_bundle(config.artifact_dir, decisions, order_map)
    report = _write_applied_report(
        config,
        signal_date=signal_date,
        run_mode=run_mode,
        source_type="autonomous_live_entry_scorer",
        decision_rows=len(decisions),
        order_map_rows=len(order_map),
        metadata=generation_summary,
    )
    _publish_current_signal_bundle(config, autonomous_config, decisions, order_map, report)
    return report


def build_autonomous_blotter_rows(
    *,
    client: LiveDataClient,
    signal_date: str,
    settings: AutonomousSignalConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    today = date.fromisoformat(signal_date)
    events = _standardized_events(
        client.benzinga_earnings(today, today + timedelta(days=settings.lookahead_days), limit=settings.max_raw_events),
        min_importance=settings.min_event_importance,
    )
    candidates: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    for event in events:
        try:
            candidate = _candidate_from_event(client, event, today=today, settings=settings)
        except Exception as exc:
            _count(rejected, f"candidate_error:{type(exc).__name__}")
            event["reject_reason"] = str(exc)[:240]
            continue
        if not candidate:
            _count(rejected, "not_execution_ready")
            continue
        candidates.append(candidate)

    candidates = sorted(candidates, key=lambda row: float(row.get("score") or 0.0), reverse=True)
    accepted = candidates[: max(1, settings.max_trades)]
    summary = {
        "raw_event_rows": len(events),
        "execution_ready_candidates": len(candidates),
        "accepted_candidates": len(accepted),
        "rejection_reasons": rejected,
    }
    return accepted, summary


def normalize_occ_symbol(raw: Any, *, with_prefix: bool = False) -> str:
    symbol = str(raw or "").strip().upper().replace(" ", "")
    if symbol.startswith("O:"):
        symbol = symbol[2:]
    return f"O:{symbol}" if with_prefix and symbol else symbol


def _candidate_from_event(
    client: LiveDataClient,
    event: dict[str, Any],
    *,
    today: date,
    settings: AutonomousSignalConfig,
) -> dict[str, Any] | None:
    ticker = str(event["ticker"]).upper()
    event_date = date.fromisoformat(event["event_date"])
    bars = client.stock_daily_bars(ticker, today - timedelta(days=100), today)
    spot = _latest_close(bars)
    if not math.isfinite(spot) or spot < settings.min_spot_price:
        return None

    realized_event_move = _realized_event_move(bars, horizon_days=max(1, (event_date - today).days + 1))
    exp_min = max(today, event_date) + timedelta(days=settings.option_min_dte)
    exp_max = max(today, event_date) + timedelta(days=settings.option_max_dte)
    contracts = _standardized_contracts(client.option_contracts(ticker, exp_min, exp_max))
    structure = _select_option_structure(client, ticker, event_date, spot, contracts, settings, realized_event_move)
    if structure is None:
        return None

    implied_move = float(structure["implied_move"])
    edge = implied_move - realized_event_move
    has_defined_risk_wings = all(structure.get(key) for key in ["long_call_wing", "long_put_wing"])
    if has_defined_risk_wings and edge >= max(0.01, settings.short_vol_threshold * max(realized_event_move, 0.005) - realized_event_move):
        action = "short_vol_defined"
        score = edge
    else:
        action = "long_vol"
        score = max(realized_event_move - implied_move, 0.001)

    confidence = _confidence(score, implied_move, realized_event_move, float(event.get("importance") or 3.0))
    qty = max(1, min(settings.max_contract_qty, int(round(1 + confidence * (settings.max_contract_qty - 1)))))
    exit_date = _add_business_days(today, settings.default_exit_business_days)
    return {
        "model": "autonomous_world_model_rl_live_scorer",
        "ticker": ticker,
        "entry_date": today.isoformat(),
        "event_date": event_date.isoformat(),
        "exit_date": exit_date.isoformat(),
        "action": action,
        "qty": qty,
        "chosen_scale": qty,
        "confidence": round(confidence, 6),
        "score": round(score, 8),
        "pred_mean": round(score, 8),
        "pred_std": round(abs(implied_move - realized_event_move), 8),
        "pred_cvar": round(-max(implied_move, realized_event_move), 8),
        "prob_profit": round(confidence, 6),
        "utility": round(score * confidence, 8),
        "horizon": settings.default_exit_business_days,
        "split": "live",
        "fold_test_year": today.year,
        "original_model_action": action,
        "execution_mode": "exact_same_leg_inverse",
        "live_execution_action": (
            "take_opposite_side_of_same_iron_fly"
            if action == "short_vol_defined"
            else "take_opposite_side_of_same_straddle"
        ),
        "event_date_label": event_date.isoformat(),
        "spot_entry": round(spot, 4),
        **structure,
    }


def _select_option_structure(
    client: LiveDataClient,
    ticker: str,
    event_date: date,
    spot: float,
    contracts: list[dict[str, Any]],
    settings: AutonomousSignalConfig,
    realized_event_move: float,
) -> dict[str, Any] | None:
    if not contracts:
        return None
    for row in contracts:
        row["dte_abs"] = abs((row["expiration_date"] - event_date).days)
        row["strike_dist"] = abs(float(row["strike_price"]) - spot)
    expiration = sorted(contracts, key=lambda row: (row["dte_abs"], row["expiration_date"]))[0]["expiration_date"]
    chain = [row for row in contracts if row["expiration_date"] == expiration]
    calls = sorted([row for row in chain if row["contract_type"] == "call"], key=lambda row: row["strike_dist"])
    puts = sorted([row for row in chain if row["contract_type"] == "put"], key=lambda row: row["strike_dist"])
    if not calls or not puts:
        return None

    call = calls[0]
    put = puts[0]
    call_mark = _option_mark(client, ticker, call["ticker"])
    put_mark = _option_mark(client, ticker, put["ticker"])
    call_price = float(call_mark.get("price", math.nan))
    put_price = float(put_mark.get("price", math.nan))
    if not (_valid_mark(call_price, settings) and _valid_mark(put_price, settings)):
        return None

    straddle = call_price + put_price
    implied_move = max(0.005, min(0.35, straddle / spot))
    wing_width = max(settings.min_wing_width_pct * spot, settings.option_wing_mult * max(implied_move, realized_event_move) * spot)
    call_wings = sorted(
        [
            row
            for row in chain
            if row["contract_type"] == "call" and float(row["strike_price"]) >= float(call["strike_price"]) + wing_width
        ],
        key=lambda row: float(row["strike_price"]),
    )
    put_wings = sorted(
        [
            row
            for row in chain
            if row["contract_type"] == "put" and float(row["strike_price"]) <= float(put["strike_price"]) - wing_width
        ],
        key=lambda row: float(row["strike_price"]),
        reverse=True,
    )

    output: dict[str, Any] = {
        "expiration_date": expiration.isoformat(),
        "call_contract": normalize_occ_symbol(call["ticker"], with_prefix=True),
        "put_contract": normalize_occ_symbol(put["ticker"], with_prefix=True),
        "short_call": normalize_occ_symbol(call["ticker"], with_prefix=True),
        "short_put": normalize_occ_symbol(put["ticker"], with_prefix=True),
        "call_strike": float(call["strike_price"]),
        "put_strike": float(put["strike_price"]),
        "call_dte": (expiration - event_date).days,
        "put_dte": (expiration - event_date).days,
        "straddle_pre": round(straddle, 4),
        "implied_move": round(implied_move, 8),
        "call_price_pre": round(call_price, 4),
        "put_price_pre": round(put_price, 4),
        "call_price_source": call_mark.get("price_source", "massive_previous_bar"),
        "put_price_source": put_mark.get("price_source", "massive_previous_bar"),
    }
    if call_wings and put_wings:
        call_wing = call_wings[0]
        put_wing = put_wings[0]
        call_wing_mark = _option_mark(client, ticker, call_wing["ticker"])
        put_wing_mark = _option_mark(client, ticker, put_wing["ticker"])
        call_wing_price = float(call_wing_mark.get("price", math.nan))
        put_wing_price = float(put_wing_mark.get("price", math.nan))
        credit = call_price + put_price - call_wing_price - put_wing_price
        if _valid_mark(call_wing_price, settings) and _valid_mark(put_wing_price, settings) and credit > 0:
            output.update(
                {
                    "long_call_wing": normalize_occ_symbol(call_wing["ticker"], with_prefix=True),
                    "long_put_wing": normalize_occ_symbol(put_wing["ticker"], with_prefix=True),
                    "call_wing_price_pre": round(call_wing_price, 4),
                    "put_wing_price_pre": round(put_wing_price, 4),
                    "call_wing_price_source": call_wing_mark.get("price_source", "massive_previous_bar"),
                    "put_wing_price_source": put_wing_mark.get("price_source", "massive_previous_bar"),
                }
            )
    return output


def _restore_exit_signal_bundle(
    config: LakeFSRuntimeConfig,
    settings: AutonomousSignalConfig,
    signal_date: str,
) -> dict[str, Any] | None:
    if not config.endpoint or s3fs is None:
        return None
    filesystem = s3fs.S3FileSystem(**config.storage_options)
    prefix = f"s3://{config.repository}/{config.artifact_ref}/{settings.current_signal_prefix}/"
    try:
        decision_files = [
            path
            for path in filesystem.find(prefix)
            if str(path).endswith("/selected_decisions_live.csv")
        ]
    except Exception:
        return None

    decisions: list[dict[str, str]] = []
    order_map: list[dict[str, str]] = []
    entry_keys: set[tuple[str, str, str]] = set()
    for decision_uri in decision_files:
        rows = _read_csv_from_filesystem(filesystem, decision_uri)
        due_rows = [
            row
            for row in rows
            if str(row.get("exit_date") or "")[:10] == signal_date
            and str(row.get("selected_model") or "") == EXPECTED_SELECTED_MODEL
            and str(row.get("selected_model_version") or "") == EXPECTED_MODEL_VERSION
        ]
        if not due_rows:
            continue
        decisions.extend(due_rows)
        for row in due_rows:
            entry_keys.add(
                (
                    str(row.get("ticker") or "").upper(),
                    str(row.get("entry_date") or "")[:10],
                    str(row.get("action") or row.get("original_model_action") or "").lower(),
                )
            )
        map_uri = str(decision_uri).rsplit("/", 1)[0] + "/live_order_map.csv"
        if filesystem.exists(map_uri):
            order_map.extend(
                row
                for row in _read_csv_from_filesystem(filesystem, map_uri)
                if (
                    str(row.get("ticker") or "").upper(),
                    str(row.get("entry_date") or "")[:10],
                    str(row.get("original_model_action") or row.get("action") or "").lower(),
                )
                in entry_keys
            )

    if not decisions or not order_map:
        return None
    _write_current_signal_bundle(config.artifact_dir, decisions, order_map)
    return _write_applied_report(
        config,
        signal_date=signal_date,
        run_mode="exit",
        source_type="autonomous_live_exit_state",
        decision_rows=len(decisions),
        order_map_rows=len(order_map),
        metadata={"restored_entry_bundles": len(decision_files)},
    )


def _standardized_events(rows: list[dict[str, Any]], *, min_importance: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        event_date = _date_value(row.get("date") or row.get("event_date"))
        importance = _float_value(row.get("importance"), default=3.0)
        if not ticker or event_date is None or importance < min_importance:
            continue
        key = (ticker, event_date.isoformat())
        if key in seen:
            continue
        seen.add(key)
        events.append({"ticker": ticker, "event_date": event_date.isoformat(), "importance": importance})
    return events


def _standardized_contracts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for row in rows:
        ticker = normalize_occ_symbol(row.get("ticker") or row.get("symbol"), with_prefix=True)
        expiration_date = _date_value(row.get("expiration_date"))
        contract_type = str(row.get("contract_type") or "").strip().lower()
        strike = _float_value(row.get("strike_price"), default=math.nan)
        if not ticker or expiration_date is None or contract_type not in {"call", "put"} or not math.isfinite(strike):
            continue
        contracts.append(
            {
                "ticker": ticker,
                "expiration_date": expiration_date,
                "contract_type": contract_type,
                "strike_price": strike,
            }
        )
    return contracts


def _write_current_signal_bundle(
    artifact_dir: Path,
    decisions: list[dict[str, str]],
    order_map: list[dict[str, str]],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(artifact_dir / "selected_decisions_live.csv", decisions, DECISION_COLUMNS)
    _write_rows(artifact_dir / "live_order_map.csv", order_map, ORDER_MAP_COLUMNS)


def _write_applied_report(
    config: LakeFSRuntimeConfig,
    *,
    signal_date: str,
    run_mode: str,
    source_type: str,
    decision_rows: int,
    order_map_rows: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "status": "applied",
        "source_type": source_type,
        "signal_date": signal_date,
        "run_mode": run_mode,
        "decision_rows": decision_rows,
        "order_map_rows": order_map_rows,
        "selected_model": EXPECTED_SELECTED_MODEL,
        "selected_model_version": EXPECTED_MODEL_VERSION,
        "accepted_for_execution": True,
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
    }
    (config.artifact_dir / "live_signal_source_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return report


def _write_no_current_signal_report(
    config: LakeFSRuntimeConfig,
    *,
    signal_date: str,
    run_mode: str,
    reason: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "no_current_signal",
        "source_type": metadata.get("source_type", "autonomous_live_scorer"),
        "signal_date": signal_date,
        "run_mode": run_mode,
        "decision_rows": 0,
        "order_map_rows": 0,
        "selected_model": EXPECTED_SELECTED_MODEL,
        "selected_model_version": EXPECTED_MODEL_VERSION,
        "accepted_for_execution": True,
        "reason": reason,
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
    }
    (config.artifact_dir / "live_signal_source_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return report


def _publish_current_signal_bundle(
    config: LakeFSRuntimeConfig,
    settings: AutonomousSignalConfig,
    decisions: list[dict[str, str]],
    order_map: list[dict[str, str]],
    report: dict[str, Any],
) -> None:
    if not settings.publish_current_signals or not config.endpoint or s3fs is None:
        return
    run_id = _run_id()
    run_mode = str(report.get("run_mode") or "entry")
    signal_date = str(report.get("signal_date") or "")
    local_dir = Path(os.getenv("QSENTIA_OUTPUT_DIR", "/app/outputs")) / "current_signal_bundle"
    local_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = local_dir / "selected_decisions_live.csv"
    order_map_path = local_dir / "live_order_map.csv"
    report_path = local_dir / "live_signal_source_report.json"
    _write_rows(decisions_path, decisions, DECISION_COLUMNS)
    _write_rows(order_map_path, order_map, ORDER_MAP_COLUMNS)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    filesystem = s3fs.S3FileSystem(**config.storage_options)
    base_uri = (
        f"s3://{config.repository}/{config.artifact_ref}/{settings.current_signal_prefix}/"
        f"{signal_date}/{run_mode}/{run_id}"
    )
    filesystem.put(str(decisions_path), f"{base_uri}/selected_decisions_live.csv")
    filesystem.put(str(order_map_path), f"{base_uri}/live_order_map.csv")
    filesystem.put(str(report_path), f"{base_uri}/live_signal_source_report.json")


def _explicit_current_source_is_configured() -> bool:
    names = [
        "QSENTIA_WORLD_RL_CURRENT_BLOTTER_URI",
        "WORLD_MODEL_RL_CURRENT_BLOTTER_URI",
        "QSENTIA_WORLD_RL_CURRENT_BLOTTER_PATH",
        "WORLD_MODEL_RL_CURRENT_BLOTTER_PATH",
        "QSENTIA_WORLD_RL_CURRENT_DECISIONS_URI",
        "WORLD_MODEL_RL_CURRENT_DECISIONS_URI",
        "QSENTIA_WORLD_RL_CURRENT_DECISIONS_PATH",
        "WORLD_MODEL_RL_CURRENT_DECISIONS_PATH",
        "QSENTIA_WORLD_RL_CURRENT_ORDER_MAP_URI",
        "WORLD_MODEL_RL_CURRENT_ORDER_MAP_URI",
        "QSENTIA_WORLD_RL_CURRENT_ORDER_MAP_PATH",
        "WORLD_MODEL_RL_CURRENT_ORDER_MAP_PATH",
    ]
    return any(os.getenv(name, "").strip() for name in names)


def _payload_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("results"), list):
        return [row for row in payload["results"] if isinstance(row, dict)]
    for value in payload.values():
        if isinstance(value, list) and (not value or isinstance(value[0], dict)):
            return [row for row in value if isinstance(row, dict)]
    return []


def _latest_close(rows: list[dict[str, Any]]) -> float:
    closes = [_float_value(row.get("c"), default=math.nan) for row in rows]
    finite = [value for value in closes if math.isfinite(value) and value > 0]
    return finite[-1] if finite else math.nan


def _realized_event_move(rows: list[dict[str, Any]], *, horizon_days: int) -> float:
    closes = [_float_value(row.get("c"), default=math.nan) for row in rows]
    finite = [value for value in closes if math.isfinite(value) and value > 0]
    returns = [
        finite[index] / finite[index - 1] - 1.0
        for index in range(1, len(finite))
        if finite[index - 1] > 0
    ]
    tail = returns[-63:] if len(returns) >= 21 else returns
    if len(tail) < 5:
        return 0.05
    mean = sum(tail) / len(tail)
    variance = sum((value - mean) ** 2 for value in tail) / max(1, len(tail) - 1)
    daily_vol = math.sqrt(max(variance, 0.0))
    return max(0.005, min(0.35, daily_vol * math.sqrt(max(1, horizon_days))))


def _option_mark(client: LiveDataClient, underlying: str, option_ticker: str) -> dict[str, Any]:
    snapshotter = getattr(client, "option_snapshot", None)
    if callable(snapshotter):
        try:
            snapshot = snapshotter(underlying, option_ticker) or {}
        except Exception:
            snapshot = {}
        price = _snapshot_price(snapshot)
        if math.isfinite(price) and price > 0:
            return {"price": price, "price_source": "massive_option_snapshot"}

    row = client.option_previous_bar(option_ticker) or {}
    for key in ["c", "vw", "o"]:
        value = _float_value(row.get(key), default=math.nan)
        if math.isfinite(value) and value > 0:
            return {"price": value, "price_source": "massive_previous_bar"}
    return {"price": math.nan, "price_source": "missing"}


def _snapshot_price(snapshot: dict[str, Any]) -> float:
    quote = snapshot.get("last_quote") if isinstance(snapshot.get("last_quote"), dict) else {}
    bid = _float_value(quote.get("bid") or quote.get("bid_price"), default=math.nan)
    ask = _float_value(quote.get("ask") or quote.get("ask_price"), default=math.nan)
    if math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0

    trade = snapshot.get("last_trade") if isinstance(snapshot.get("last_trade"), dict) else {}
    for container in [trade, snapshot.get("day") if isinstance(snapshot.get("day"), dict) else {}, snapshot]:
        for key in ["price", "p", "close", "c", "vw"]:
            value = _float_value(container.get(key), default=math.nan)
            if math.isfinite(value) and value > 0:
                return value
    return math.nan


def _valid_mark(value: float, settings: AutonomousSignalConfig) -> bool:
    return math.isfinite(value) and value >= settings.min_option_mark


def _confidence(score: float, implied_move: float, realized_move: float, importance: float) -> float:
    raw = 0.45 + min(0.30, abs(implied_move - realized_move) * 3.0) + min(0.20, max(0.0, importance - 3.0) * 0.05)
    raw += min(0.05, max(score, 0.0))
    return max(0.05, min(0.99, raw))


def _date_value(raw: Any) -> date | None:
    value = str(raw or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _float_value(raw: Any, *, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _add_business_days(start: date, days: int) -> date:
    current = start
    remaining = max(1, days)
    while remaining:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def _read_csv_from_filesystem(filesystem: Any, uri: str) -> list[dict[str, str]]:
    with filesystem.open(uri, "r") as handle:
        return list(csv.DictReader(handle))


def _run_id() -> str:
    explicit = os.getenv("QSENTIA_RUN_ID", "").strip()
    if explicit:
        return explicit
    batch_id = os.getenv("AWS_BATCH_JOB_ID", "").strip()
    if batch_id:
        return batch_id
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _requests() -> Any:
    if requests is None:
        raise RuntimeError("Install requests to generate WORLD_MODEL-RL live signals.") from _REQUESTS_IMPORT_ERROR
    return requests
