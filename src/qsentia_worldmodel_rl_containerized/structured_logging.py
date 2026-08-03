from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def emit_event(event: str, **fields: Any) -> None:
    payload = {"event": event, "ts": datetime.now(timezone.utc).isoformat(), **fields}
    print(json.dumps(payload, default=str, separators=(",", ":")), flush=True)


def build_signal_summary(signal_payload: dict[str, Any]) -> dict[str, Any]:
    signal = signal_payload.get("signal") or {}
    metadata = signal.get("metadata") if isinstance(signal.get("metadata"), dict) else {}
    trade = signal_payload.get("trade") if isinstance(signal_payload.get("trade"), dict) else {}
    orders = trade.get("orders") if isinstance(trade.get("orders"), list) else []
    return {
        "producer_id": metadata.get("producer_id", "qsentia-world-model-rl"),
        "producer_version": metadata.get("producer_version", "v12b_exact_same_leg_inverse_plus_small_rl_guardian_v1"),
        "asof": signal.get("asof"),
        "label": signal.get("label"),
        "signal": signal.get("signal"),
        "confidence": signal.get("confidence"),
        "selected_model": metadata.get("selected_model"),
        "artifact_live_trading_enabled": metadata.get("artifact_live_trading_enabled"),
        "run_mode": metadata.get("run_mode"),
        "signal_date": metadata.get("signal_date"),
        "trade_intent_orders": len(orders),
        "candidate_rows": metadata.get("candidate_rows"),
    }
