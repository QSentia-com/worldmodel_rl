from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from .alpaca_client import AlpacaRestClient
from .config import bool_env


def execute_alpaca_trade_intent(trade_intent: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(trade_intent, dict):
        return None

    enabled = bool_env("QSENTIA_ALPACA_EXECUTION_ENABLED", False) or bool_env("QSENTIA_EXECUTE_ALPACA_ORDERS", False)
    dry_run = bool_env("DRY_RUN", True) or bool_env("QSENTIA_ALPACA_DRY_RUN", True)
    connectivity_only = bool_env("QSENTIA_CONNECTIVITY_CHECK_ONLY", False)
    respect_artifact_guard = bool_env("QSENTIA_RESPECT_ARTIFACT_LIVE_TRADING_FLAG", True)
    artifact_live_enabled = bool(trade_intent.get("artifact_live_trading_enabled"))
    base_url = os.getenv("APCA_API_BASE_URL") or os.getenv("ALPACA_BASE_URL") or ""
    if not _is_paper_base_url(base_url) and not bool_env("QSENTIA_ALLOW_LIVE_ALPACA_TRADING", False):
        raise RuntimeError("Refusing Alpaca live URL without QSENTIA_ALLOW_LIVE_ALPACA_TRADING=true.")

    client = AlpacaRestClient.from_env(base_url)
    account = client.account()
    clock = client.clock()
    positions = client.positions()
    market_is_open = _clock_is_open(clock)
    orders = trade_intent.get("orders") if isinstance(trade_intent.get("orders"), list) else []

    report: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": "connectivity_checked" if connectivity_only else "started",
        "enabled": enabled,
        "dry_run": dry_run,
        "connectivity_check_only": connectivity_only,
        "artifact_live_trading_enabled": artifact_live_enabled,
        "market_is_open": market_is_open,
        "account": _account_snapshot(account),
        "positions_count": len(positions),
        "results": [],
    }
    if connectivity_only:
        return report

    if respect_artifact_guard and not artifact_live_enabled:
        report["status"] = "skipped_artifact_live_trading_disabled"
        report["results"] = [{"status": "artifact_live_trading_disabled", "payload": payload} for payload in orders]
        return report
    if not enabled:
        report["status"] = "execution_disabled"
        report["results"] = [{"status": "disabled", "payload": payload} for payload in orders]
        return report
    if bool_env("QSENTIA_REQUIRE_MARKET_OPEN", True) and not market_is_open:
        report["status"] = "skipped_market_closed"
        report["results"] = [{"status": "skipped_market_closed", "payload": payload} for payload in orders]
        return report

    for payload in orders:
        if dry_run:
            report["results"].append({"status": "dry_run", "payload": payload})
            continue
        existing = client.get_order_by_client_id(str(payload.get("client_order_id") or ""))
        if existing:
            report["results"].append({"status": "already_exists", "payload": payload, "response": existing})
            continue
        try:
            response = client.submit_order(payload)
        except Exception as exc:
            report["results"].append({"status": "broker_error", "payload": payload, "error": str(exc)})
        else:
            report["results"].append({"status": "submitted", "payload": payload, "response": response})

    if any(row.get("status") == "broker_error" for row in report["results"]):
        report["status"] = "completed_with_errors"
    elif report["results"]:
        report["status"] = "completed"
    else:
        report["status"] = "no_orders"
    return report


def _clock_is_open(clock: dict[str, Any]) -> bool:
    raw = clock.get("is_open", False)
    if isinstance(raw, str):
        return raw.strip().lower() == "true"
    return bool(raw)


def _account_snapshot(account: dict[str, Any]) -> dict[str, Any]:
    keys = ["id", "account_number", "status", "currency", "cash", "buying_power", "equity", "portfolio_value"]
    return {key: account.get(key) for key in keys if key in account}


def _is_paper_base_url(base_url: str) -> bool:
    return "paper-api.alpaca.markets" in str(base_url)
