from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from types import ModuleType
from urllib.parse import quote
from zoneinfo import ZoneInfo

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

from .artifact_manager import download_lakefs_artifacts, validate_artifacts
from .config import LakeFSRuntimeConfig, bool_env
from .live_signal_source import maybe_apply_current_signal_source
from .signal_inference import _candidate_rows, _order_map, _run_mode, _selected_decisions, _signal_date
from .structured_logging import emit_event


ACCEPTED_REFRESH_STATUSES = {"ready", "no_current_signal"}


class MassiveOptionsClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = (api_key or os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY") or "").strip()
        self.base_url = (base_url or os.getenv("MASSIVE_API_BASE_URL") or "https://api.massive.com").rstrip("/")
        if not self.api_key:
            raise RuntimeError("Missing MASSIVE_API_KEY/POLYGON_API_KEY for WORLD_MODEL-RL live option refresh.")

    def validate_option_contract(self, option_ticker: str, *, signal_date: str) -> dict[str, Any]:
        parsed = parse_option_ticker(option_ticker)
        params = {
            "underlying_ticker": parsed["underlying_ticker"],
            "expiration_date": parsed["expiration_date"],
            "contract_type": parsed["contract_type"],
            "strike_price": str(parsed["strike_price"]),
            "as_of": signal_date,
            "expired": "false",
            "limit": "10",
            "apiKey": self.api_key,
        }
        response = _requests().get(
            f"{self.base_url}/v3/reference/options/contracts",
            params=params,
            headers={"accept": "application/json", "connection": "close"},
            timeout=(10, 60),
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Massive option contract lookup failed {response.status_code}: {response.text[:1000]}"
            )
        payload = response.json()
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            results = []
        requested_symbol = normalize_option_ticker(option_ticker)
        matched = [
            row
            for row in results
            if normalize_option_ticker(str(row.get("ticker") or row.get("symbol") or "")) == requested_symbol
        ]
        return {
            "option_ticker": requested_symbol,
            "underlying_ticker": parsed["underlying_ticker"],
            "expiration_date": parsed["expiration_date"],
            "contract_type": parsed["contract_type"],
            "strike_price": parsed["strike_price"],
            "found": bool(matched),
            "result_count": len(results),
        }


def main() -> int:
    artifact_config = LakeFSRuntimeConfig.from_env()
    artifact_source = "local_artifact_mount" if artifact_config.skip_download else artifact_config.object_uri
    emit_event(
        "world_model_rl_live_signal_refresh_started",
        artifact_source=artifact_source,
        artifact_dir=str(artifact_config.artifact_dir),
        batch_job_id=os.getenv("AWS_BATCH_JOB_ID"),
        batch_job_attempt=os.getenv("AWS_BATCH_JOB_ATTEMPT"),
    )

    downloaded = download_lakefs_artifacts(artifact_config)
    current_signal_source = maybe_apply_current_signal_source(artifact_config)
    report = build_live_signal_refresh_report(artifact_config.artifact_dir)
    if current_signal_source:
        report["current_signal_source"] = current_signal_source
    report["artifact_source"] = artifact_source
    report["downloaded_artifact_files"] = len(downloaded)
    published = publish_live_signal_refresh_report(artifact_config, report)
    if published:
        report["published_outputs"] = published

    emit_event("world_model_rl_live_signal_refresh_completed", **_refresh_summary(report))
    print("FINAL_REFRESH " + json.dumps(report, default=str, separators=(",", ":")), flush=True)
    if not report.get("accepted_for_execution"):
        raise RuntimeError(f"WORLD_MODEL-RL live signal refresh failed closed: {report.get('status')}")
    return 0


def build_live_signal_refresh_report(artifact_dir: Path | str) -> dict[str, Any]:
    resolved_artifact_dir = Path(artifact_dir)
    validate_artifacts(resolved_artifact_dir)
    now = datetime.now(timezone.utc)
    run_mode = _run_mode()
    signal_date = _signal_date()
    selected_rows = _selected_decisions(resolved_artifact_dir)
    date_column = "exit_date" if run_mode == "exit" else "entry_date"
    candidate_rows = _candidate_rows(selected_rows, run_mode, signal_date)
    max_entry_date = _max_date(selected_rows, "entry_date")
    max_exit_date = _max_date(selected_rows, "exit_date")
    max_relevant_date = max_exit_date if run_mode == "exit" else max_entry_date
    artifact_current = bool(max_relevant_date and max_relevant_date >= signal_date)

    report: dict[str, Any] = {
        "model_id": os.getenv("QSENTIA_MODEL_ID", "qsentia-world-model-rl"),
        "producer_id": os.getenv("QSENTIA_PRODUCER_ID", "qsentia-world-model-rl"),
        "asof": now.isoformat(),
        "asof_date": datetime.now(ZoneInfo("America/New_York")).date().isoformat(),
        "run_mode": run_mode,
        "signal_date": signal_date,
        "date_column": date_column,
        "selected_rows": len(selected_rows),
        "candidate_rows": len(candidate_rows),
        "selected_decisions_max_entry_date": max_entry_date,
        "selected_decisions_max_exit_date": max_exit_date,
        "massive_api_checked": False,
        "option_contract_checks": [],
        "mapped_option_legs": 0,
        "accepted_for_execution": False,
    }

    if not selected_rows:
        report.update(status="missing_selected_decisions", reason="selected_decisions_live.csv had no valid RL rows")
        return report
    if not artifact_current:
        report.update(
            status="stale_artifact",
            reason=(
                f"latest {date_column} in selected_decisions_live.csv is {max_relevant_date or 'missing'}, "
                f"before signal date {signal_date}"
            ),
        )
        return report
    if not candidate_rows:
        report.update(status="no_current_signal", reason=f"no {run_mode} signal selected for {signal_date}")
        report["accepted_for_execution"] = True
        return report

    order_map = _order_map(resolved_artifact_dir)
    mapped_rows = _mapped_rows_for_candidates(candidate_rows, order_map)
    report["mapped_option_legs"] = len(mapped_rows)
    if not mapped_rows:
        report.update(status="missing_order_map", reason="candidate signal rows have no mapped option legs")
        return report

    if bool_env("QSENTIA_VALIDATE_MASSIVE_OPTION_DATA", True):
        client = MassiveOptionsClient()
        option_tickers = sorted({normalize_option_ticker(str(row.get("option_ticker") or "")) for row in mapped_rows})
        option_tickers = [ticker for ticker in option_tickers if ticker]
        max_checks = int(os.getenv("QSENTIA_MAX_OPTION_DATA_CHECKS", "50"))
        checks = [client.validate_option_contract(ticker, signal_date=signal_date) for ticker in option_tickers[:max_checks]]
        report["massive_api_checked"] = True
        report["option_contract_checks"] = checks
        missing = [row["option_ticker"] for row in checks if not row.get("found")]
        if missing:
            report.update(
                status="option_data_unavailable",
                reason=f"Massive did not return current contract data for {len(missing)} mapped option legs",
                missing_option_tickers=missing[:20],
            )
            return report

    report.update(status="ready", reason=f"{len(candidate_rows)} current {run_mode} signal rows validated")
    report["accepted_for_execution"] = True
    return report


def publish_live_signal_refresh_report(
    artifact_config: LakeFSRuntimeConfig,
    report: dict[str, Any],
) -> dict[str, str] | None:
    if not bool_env("QSENTIA_PUBLISH_OUTPUTS", True):
        return None
    if artifact_config.skip_download and not artifact_config.endpoint:
        return None

    output_prefix = _output_prefix_from_env()
    run_id = _run_id_from_env()
    run_mode = str(report.get("run_mode") or "entry")
    local_dir = Path(os.getenv("QSENTIA_OUTPUT_DIR", "/app/outputs"))
    local_dir.mkdir(parents=True, exist_ok=True)
    report_path = local_dir / "live_signal_refresh.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    filesystem = _s3_filesystem(artifact_config)
    base_uri = f"s3://{artifact_config.repository}/{artifact_config.artifact_ref}/{output_prefix}/live-refresh/{run_mode}"
    latest_uri = f"{base_uri}/latest.json"
    run_uri = f"{base_uri}/{run_id}.json"
    filesystem.put(str(report_path), latest_uri)
    filesystem.put(str(report_path), run_uri)

    result = {"run_id": run_id, "live_signal_refresh_uri": latest_uri, "run_live_signal_refresh_uri": run_uri}
    if bool_env("QSENTIA_COMMIT_OUTPUTS", True):
        response = _requests().post(
            f"{artifact_config.endpoint}/api/v1/repositories/{artifact_config.repository}/branches/{artifact_config.artifact_ref}/commits",
            auth=(artifact_config.access_key_id, artifact_config.secret_access_key),
            json={"message": f"publish WORLD_MODEL-RL live signal refresh {run_id}"},
            timeout=30,
        )
        response.raise_for_status()
        result["commit_id"] = response.json().get("id", "")
    return result


def assert_live_signal_refresh_ready(
    artifact_config: LakeFSRuntimeConfig,
    *,
    run_mode: str | None = None,
    signal_date: str | None = None,
) -> dict[str, Any] | None:
    if not bool_env("QSENTIA_REQUIRE_FRESH_OPTION_DATA", False):
        return None

    expected_run_mode = run_mode or _run_mode()
    expected_signal_date = signal_date or _signal_date()
    report = load_live_signal_refresh_report(artifact_config, expected_run_mode)
    if str(report.get("run_mode") or "") != expected_run_mode:
        raise RuntimeError(
            f"WORLD_MODEL-RL live refresh run_mode mismatch: expected {expected_run_mode}, "
            f"got {report.get('run_mode')}"
        )
    if str(report.get("signal_date") or "")[:10] != expected_signal_date:
        raise RuntimeError(
            f"WORLD_MODEL-RL live refresh signal_date mismatch: expected {expected_signal_date}, "
            f"got {report.get('signal_date')}"
        )

    max_age_minutes = int(os.getenv("QSENTIA_MAX_REFRESH_AGE_MINUTES", "90"))
    age_seconds = (datetime.now(timezone.utc) - _parse_utc(report.get("asof"))).total_seconds()
    if age_seconds > max_age_minutes * 60:
        raise RuntimeError(
            f"WORLD_MODEL-RL live refresh is stale: {age_seconds / 60:.1f} minutes old "
            f"(max {max_age_minutes})"
        )

    status = str(report.get("status") or "")
    accepted_statuses = set(ACCEPTED_REFRESH_STATUSES)
    if not bool_env("QSENTIA_ACCEPT_FRESH_NO_CURRENT_SIGNAL", True):
        accepted_statuses.discard("no_current_signal")
    if status not in accepted_statuses or not report.get("accepted_for_execution"):
        raise RuntimeError(f"WORLD_MODEL-RL live refresh is not execution-ready: {status}")
    return report


def load_live_signal_refresh_report(artifact_config: LakeFSRuntimeConfig, run_mode: str) -> dict[str, Any]:
    local_path = os.getenv("QSENTIA_LIVE_REFRESH_LOCAL_PATH", "").strip()
    if local_path:
        return _read_json(Path(local_path))

    filesystem = _s3_filesystem(artifact_config)
    uri = (
        f"s3://{artifact_config.repository}/{artifact_config.artifact_ref}/"
        f"{_output_prefix_from_env()}/live-refresh/{run_mode}/latest.json"
    )
    if not filesystem.exists(uri):
        raise RuntimeError(f"WORLD_MODEL-RL live refresh report does not exist: {uri}")
    with filesystem.open(uri, "r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"WORLD_MODEL-RL live refresh report is not a JSON object: {uri}")
    return payload


def parse_option_ticker(raw: str) -> dict[str, Any]:
    symbol = normalize_option_ticker(raw)
    compact = symbol[2:] if symbol.startswith("O:") else symbol
    if len(compact) < 15:
        raise RuntimeError(f"Unsupported option ticker format: {raw!r}")
    right_index = max(compact.rfind("C"), compact.rfind("P"))
    if right_index <= 6:
        raise RuntimeError(f"Unsupported option ticker format: {raw!r}")
    underlying = compact[: right_index - 6]
    date_part = compact[right_index - 6 : right_index]
    contract_type = "call" if compact[right_index] == "C" else "put"
    strike_part = compact[right_index + 1 :]
    if not underlying or len(date_part) != 6 or not strike_part.isdigit():
        raise RuntimeError(f"Unsupported option ticker format: {raw!r}")
    expiration_date = f"20{date_part[0:2]}-{date_part[2:4]}-{date_part[4:6]}"
    return {
        "ticker": symbol,
        "underlying_ticker": underlying,
        "expiration_date": expiration_date,
        "contract_type": contract_type,
        "strike_price": int(strike_part) / 1000.0,
    }


def normalize_option_ticker(raw: str) -> str:
    value = str(raw or "").strip().upper()
    if not value:
        return ""
    if not value.startswith("O:"):
        value = f"O:{value}"
    return quote(value, safe=":").replace("%20", "")


def _mapped_rows_for_candidates(
    candidate_rows: list[dict[str, str]],
    order_map: dict[tuple[str, str], list[dict[str, str]]],
) -> list[dict[str, str]]:
    mapped_rows: list[dict[str, str]] = []
    for row in candidate_rows:
        key = (str(row.get("ticker") or "").upper(), str(row.get("entry_date") or "")[:10])
        mapped_rows.extend(order_map.get(key, []))
    return mapped_rows


def _max_date(rows: list[dict[str, str]], column: str) -> str | None:
    values = sorted({str(row.get(column) or "")[:10] for row in rows if str(row.get(column) or "")[:10]})
    return values[-1] if values else None


def _parse_utc(raw: Any) -> datetime:
    value = str(raw or "").strip()
    if not value:
        raise RuntimeError("WORLD_MODEL-RL live refresh report is missing asof.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def _output_prefix_from_env() -> str:
    return os.getenv("QSENTIA_OUTPUT_PREFIX", "inference_outputs/world-model-rl").strip("/")


def _run_id_from_env() -> str:
    explicit = os.getenv("QSENTIA_RUN_ID", "").strip()
    if explicit:
        return explicit
    batch_job_id = os.getenv("AWS_BATCH_JOB_ID", "").strip()
    if batch_job_id:
        return batch_job_id
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _s3_filesystem(artifact_config: LakeFSRuntimeConfig) -> Any:
    if s3fs is None:
        raise RuntimeError("Install s3fs to read or publish WORLD_MODEL-RL live refresh reports.") from _S3FS_IMPORT_ERROR
    return s3fs.S3FileSystem(**artifact_config.storage_options)


def _requests() -> ModuleType:
    if requests is None:
        raise RuntimeError("Install requests to validate option data or publish WORLD_MODEL-RL reports.") from (
            _REQUESTS_IMPORT_ERROR
        )
    return requests


def _refresh_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": report.get("model_id"),
        "run_mode": report.get("run_mode"),
        "signal_date": report.get("signal_date"),
        "status": report.get("status"),
        "accepted_for_execution": report.get("accepted_for_execution"),
        "candidate_rows": report.get("candidate_rows"),
        "mapped_option_legs": report.get("mapped_option_legs"),
        "massive_api_checked": report.get("massive_api_checked"),
        "reason": report.get("reason"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
