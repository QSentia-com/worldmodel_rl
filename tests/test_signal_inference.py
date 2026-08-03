from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qsentia_worldmodel_rl_containerized.config import SignalRuntimeConfig
from qsentia_worldmodel_rl_containerized.config import LakeFSRuntimeConfig
from qsentia_worldmodel_rl_containerized.autonomous_signal_source import maybe_apply_autonomous_current_signal_source
from qsentia_worldmodel_rl_containerized.live_signal_refresh import (
    assert_live_signal_refresh_ready,
    build_live_signal_refresh_report,
)
from qsentia_worldmodel_rl_containerized.live_signal_source import maybe_apply_current_signal_source
from qsentia_worldmodel_rl_containerized.signal_inference import run_signal_inference


class SignalInferenceTests(unittest.TestCase):
    def test_refuses_wrong_artifact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root, selected_model="qsentia-world-model")
            config = SignalRuntimeConfig(
                output_path=root / "latest_signal.json",
                append_signal_log=False,
                use_ppo_policy=False,
                require_ppo_policy=False,
                target_gross_exposure=1.0,
            )
            with self.assertRaisesRegex(RuntimeError, "not the requested WORLD_MODEL-RL artifact"):
                run_signal_inference(root, config)

    def test_current_entry_signal_builds_alpaca_option_mleg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root)
            config = SignalRuntimeConfig(
                output_path=root / "latest_signal.json",
                append_signal_log=False,
                use_ppo_policy=False,
                require_ppo_policy=False,
                target_gross_exposure=1.0,
            )
            with patch.dict("os.environ", {"QSENTIA_SIGNAL_DATE": "2020-08-07", "QSENTIA_RUN_MODE": "entry"}, clear=False):
                payload = run_signal_inference(root, config)
            orders = payload["trade"]["orders"]
            self.assertEqual(payload["signal"]["metadata"]["selected_model"], "v12b_small_rl_guardian")
            self.assertEqual(payload["signal"]["metadata"]["artifact_live_trading_enabled"], False)
            self.assertEqual(len(orders), 1)
            self.assertEqual(orders[0]["order_class"], "mleg")
            self.assertEqual([leg["symbol"] for leg in orders[0]["legs"]], ["NVAX200904C00170000", "NVAX200904P00170000"])
            self.assertEqual([leg["position_intent"] for leg in orders[0]["legs"]], ["sell_to_open", "sell_to_open"])

    def test_no_current_signal_does_not_use_stale_history_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root)
            config = SignalRuntimeConfig(
                output_path=root / "latest_signal.json",
                append_signal_log=False,
                use_ppo_policy=False,
                require_ppo_policy=False,
                target_gross_exposure=1.0,
            )
            with patch.dict("os.environ", {"QSENTIA_SIGNAL_DATE": "2026-08-03", "QSENTIA_RUN_MODE": "entry"}, clear=False):
                payload = run_signal_inference(root, config)
            self.assertEqual(payload["signal"]["signal"], "no_current_signal")
            self.assertEqual(payload["trade"]["orders"], [])

    def test_live_refresh_rejects_historical_artifact_for_current_signal_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root)
            with patch.dict("os.environ", {"QSENTIA_SIGNAL_DATE": "2026-08-03", "QSENTIA_RUN_MODE": "entry"}, clear=False):
                report = build_live_signal_refresh_report(root)
            self.assertEqual(report["status"], "stale_artifact")
            self.assertEqual(report["accepted_for_execution"], False)

    def test_live_refresh_accepts_same_day_signal_when_option_data_validation_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root)
            with patch.dict(
                "os.environ",
                {
                    "QSENTIA_SIGNAL_DATE": "2020-08-07",
                    "QSENTIA_RUN_MODE": "entry",
                    "QSENTIA_VALIDATE_MASSIVE_OPTION_DATA": "false",
                },
                clear=False,
            ):
                report = build_live_signal_refresh_report(root)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["accepted_for_execution"], True)
            self.assertEqual(report["mapped_option_legs"], 2)

    def test_order_job_can_require_local_live_refresh_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "refresh.json"
            report_path.write_text(
                json.dumps(
                    {
                        "asof": "2026-08-03T13:15:00+00:00",
                        "run_mode": "entry",
                        "signal_date": "2026-08-03",
                        "status": "ready",
                        "accepted_for_execution": True,
                    }
                ),
                encoding="utf-8",
            )
            with patch("qsentia_worldmodel_rl_containerized.live_signal_refresh.datetime") as mocked_datetime:
                mocked_datetime.now.return_value = __import__("datetime").datetime(
                    2026, 8, 3, 13, 40, tzinfo=__import__("datetime").timezone.utc
                )
                mocked_datetime.fromisoformat.side_effect = __import__("datetime").datetime.fromisoformat
                with patch.dict(
                    "os.environ",
                    {
                        "QSENTIA_REQUIRE_FRESH_OPTION_DATA": "true",
                        "QSENTIA_LIVE_REFRESH_LOCAL_PATH": str(report_path),
                        "QSENTIA_SIGNAL_DATE": "2026-08-03",
                        "QSENTIA_RUN_MODE": "entry",
                    },
                    clear=False,
                ):
                    accepted = assert_live_signal_refresh_ready(object())  # type: ignore[arg-type]
            self.assertEqual(accepted["status"], "ready")

    def test_current_blotter_overlays_historical_artifact_for_same_day_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root)
            blotter_path = root / "current_blotter.csv"
            blotter_path.write_text(
                "\n".join(
                    [
                        "ticker,entry_date,event_date,exit_date,action,qty,confidence,score,call_contract,put_contract",
                        "AAPL,2026-08-03,2026-08-05,2026-08-07,long_vol,2,0.91,0.44,O:AAPL260814C00200000,O:AAPL260814P00200000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = _lakefs_config(root)
            signal_config = SignalRuntimeConfig(
                output_path=root / "latest_signal.json",
                append_signal_log=False,
                use_ppo_policy=False,
                require_ppo_policy=False,
                target_gross_exposure=1.0,
            )
            with patch.dict(
                "os.environ",
                {
                    "QSENTIA_SIGNAL_DATE": "2026-08-03",
                    "QSENTIA_RUN_MODE": "entry",
                    "QSENTIA_WORLD_RL_CURRENT_BLOTTER_PATH": str(blotter_path),
                    "QSENTIA_VALIDATE_MASSIVE_OPTION_DATA": "false",
                },
                clear=False,
            ):
                applied = maybe_apply_current_signal_source(config)
                report = build_live_signal_refresh_report(root)
                payload = run_signal_inference(root, signal_config)

            self.assertEqual(applied["status"], "applied")
            self.assertEqual(applied["decision_rows"], 1)
            self.assertEqual(applied["order_map_rows"], 2)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["candidate_rows"], 1)
            self.assertEqual(report["mapped_option_legs"], 2)
            self.assertEqual(payload["signal"]["metadata"]["candidate_rows"], 1)
            self.assertEqual(payload["signal"]["metadata"]["artifact_live_trading_enabled"], False)
            self.assertEqual(payload["signal"]["metadata"]["current_signal_execution_enabled"], False)
            self.assertEqual(payload["trade"]["artifact_live_trading_enabled"], False)
            order = payload["trade"]["orders"][0]
            self.assertEqual(order["qty"], "2")
            self.assertEqual([leg["symbol"] for leg in order["legs"]], ["AAPL260814C00200000", "AAPL260814P00200000"])
            self.assertEqual([leg["position_intent"] for leg in order["legs"]], ["sell_to_open", "sell_to_open"])

    def test_current_blotter_can_explicitly_enable_current_signal_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root)
            blotter_path = root / "current_blotter.csv"
            blotter_path.write_text(
                "\n".join(
                    [
                        "ticker,entry_date,event_date,exit_date,action,qty,call_contract,put_contract",
                        "AAPL,2026-08-03,2026-08-05,2026-08-07,long_vol,1,O:AAPL260814C00200000,O:AAPL260814P00200000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = _lakefs_config(root)
            signal_config = SignalRuntimeConfig(
                output_path=root / "latest_signal.json",
                append_signal_log=False,
                use_ppo_policy=False,
                require_ppo_policy=False,
                target_gross_exposure=1.0,
            )
            with patch.dict(
                "os.environ",
                {
                    "QSENTIA_SIGNAL_DATE": "2026-08-03",
                    "QSENTIA_RUN_MODE": "entry",
                    "QSENTIA_WORLD_RL_CURRENT_BLOTTER_PATH": str(blotter_path),
                    "QSENTIA_ALLOW_WORLD_RL_CURRENT_SIGNAL_EXECUTION": "true",
                },
                clear=False,
            ):
                maybe_apply_current_signal_source(config)
                payload = run_signal_inference(root, signal_config)

            self.assertEqual(payload["signal"]["metadata"]["artifact_live_trading_enabled"], False)
            self.assertEqual(payload["signal"]["metadata"]["current_signal_execution_enabled"], True)
            self.assertEqual(payload["trade"]["artifact_live_trading_enabled"], True)

    def test_current_blotter_rejects_non_current_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root)
            blotter_path = root / "stale_blotter.csv"
            blotter_path.write_text(
                "\n".join(
                    [
                        "ticker,entry_date,event_date,exit_date,action,qty,call_contract,put_contract",
                        "AAPL,2026-08-02,2026-08-05,2026-08-07,long_vol,1,O:AAPL260814C00200000,O:AAPL260814P00200000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "QSENTIA_SIGNAL_DATE": "2026-08-03",
                    "QSENTIA_RUN_MODE": "entry",
                    "QSENTIA_WORLD_RL_CURRENT_BLOTTER_PATH": str(blotter_path),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "no same-day decision rows"):
                    maybe_apply_current_signal_source(_lakefs_config(root))

    def test_autonomous_signal_source_generates_same_day_option_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root)
            signal_config = SignalRuntimeConfig(
                output_path=root / "latest_signal.json",
                append_signal_log=False,
                use_ppo_policy=False,
                require_ppo_policy=False,
                target_gross_exposure=1.0,
            )
            with patch.dict(
                "os.environ",
                {
                    "QSENTIA_WORLD_RL_AUTONOMOUS_SIGNAL_ENABLED": "true",
                    "QSENTIA_WORLD_RL_PUBLISH_CURRENT_SIGNALS": "false",
                    "QSENTIA_SIGNAL_DATE": "2026-08-03",
                    "QSENTIA_RUN_MODE": "entry",
                    "QSENTIA_VALIDATE_MASSIVE_OPTION_DATA": "false",
                    "QSENTIA_ALLOW_WORLD_RL_CURRENT_SIGNAL_EXECUTION": "true",
                },
                clear=True,
            ):
                applied = maybe_apply_autonomous_current_signal_source(_lakefs_config(root), client=_FakeLiveDataClient())
                report = build_live_signal_refresh_report(root)
                payload = run_signal_inference(root, signal_config)

            self.assertEqual(applied["status"], "applied")
            self.assertEqual(applied["source_type"], "autonomous_live_entry_scorer")
            self.assertEqual(applied["decision_rows"], 1)
            self.assertEqual(applied["order_map_rows"], 4)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["candidate_rows"], 1)
            self.assertEqual(report["mapped_option_legs"], 4)
            order = payload["trade"]["orders"][0]
            self.assertEqual(order["order_class"], "mleg")
            self.assertEqual(order["metadata"]["source"], "world_rl_live_order_map")
            self.assertEqual(order["metadata"]["action"], "short_vol_defined")
            self.assertEqual(
                [leg["position_intent"] for leg in order["legs"]],
                ["buy_to_open", "buy_to_open", "sell_to_open", "sell_to_open"],
            )

    def test_autonomous_signal_source_accepts_no_current_signal_without_replaying_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifact(root)
            signal_config = SignalRuntimeConfig(
                output_path=root / "latest_signal.json",
                append_signal_log=False,
                use_ppo_policy=False,
                require_ppo_policy=False,
                target_gross_exposure=1.0,
            )
            with patch.dict(
                "os.environ",
                {
                    "QSENTIA_WORLD_RL_AUTONOMOUS_SIGNAL_ENABLED": "true",
                    "QSENTIA_WORLD_RL_PUBLISH_CURRENT_SIGNALS": "false",
                    "QSENTIA_SIGNAL_DATE": "2026-08-03",
                    "QSENTIA_RUN_MODE": "entry",
                },
                clear=True,
            ):
                applied = maybe_apply_autonomous_current_signal_source(_lakefs_config(root), client=_EmptyLiveDataClient())
                report = build_live_signal_refresh_report(root)
                payload = run_signal_inference(root, signal_config)

            self.assertEqual(applied["status"], "no_current_signal")
            self.assertEqual(report["status"], "no_current_signal")
            self.assertEqual(report["accepted_for_execution"], True)
            self.assertEqual(payload["signal"]["signal"], "no_current_signal")
            self.assertEqual(payload["trade"]["orders"], [])


def _write_artifact(root: Path, *, selected_model: str = "v12b_small_rl_guardian") -> None:
    version = "v12b_exact_same_leg_inverse_plus_small_rl_guardian_v1"
    identity = {"selected_model": selected_model, "model_version": version}
    (root / "artifact_manifest.json").write_text(json.dumps({**identity, "files": []}), encoding="utf-8")
    (root / "model_metadata.json").write_text(
        json.dumps(
            {
                **identity,
                "strategy_name": "V12B Exact Same-Leg Inverse + Small RL Guardian",
                "strategy_mode": "RL Guardian Exact Same-Leg Inverse",
                "asset_symbol": "US_EQ_OPTIONS_EVENT_VOL_BASKET",
            }
        ),
        encoding="utf-8",
    )

    (root / "live_state.json").write_text(
        json.dumps({**identity, "live_trading_enabled": False, "order_mapping_required": "exact_same_leg_inverse"}),
        encoding="utf-8",
    )
    (root / "deployment_action_mapping.json").write_text(
        json.dumps({**identity, "live_trading_enabled": False, "deployment_blocker": "broker validation required"}),
        encoding="utf-8",
    )
    (root / "small_rl_guardian_actions.csv").write_text("date,action\n2020-08-07,long_vol\n", encoding="utf-8")
    (root / "small_rl_guardian_summary.csv").write_text("metric,value\nsharpe,4.64\n", encoding="utf-8")
    (root / "selected_decisions_live.csv").write_text(
        "\n".join(
            [
                "selected_model,selected_model_version,ticker,entry_date,event_date,exit_date,action,confidence,score,chosen_scale,execution_mode,live_execution_action",
                "v12b_small_rl_guardian,v12b_exact_same_leg_inverse_plus_small_rl_guardian_v1,NVAX,2020-08-07,2020-08-10,2020-08-11,long_vol,0.98,0.25,1,exact_same_leg_inverse,take_opposite_side_of_same_straddle",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "live_order_map.csv").write_text(
        "\n".join(
            [
                "ticker,entry_date,original_model_action,original_structure,execution_mode,live_execution_action,leg_role,option_ticker,quantity,entry_order_side,exit_order_side",
                "NVAX,2020-08-07,long_vol,long_straddle,exact_same_leg_inverse,take_opposite_side_of_same_straddle,original_call,O:NVAX200904C00170000,1,SELL_TO_OPEN,BUY_TO_CLOSE",
                "NVAX,2020-08-07,long_vol,long_straddle,exact_same_leg_inverse,take_opposite_side_of_same_straddle,original_put,O:NVAX200904P00170000,1,SELL_TO_OPEN,BUY_TO_CLOSE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _lakefs_config(root: Path) -> LakeFSRuntimeConfig:
    return LakeFSRuntimeConfig(
        endpoint="",
        access_key_id="",
        secret_access_key="",
        repository="qsentia-models",
        artifact_ref="main",
        artifact_object="world_rl/test.zip",
        artifact_dir=root,
        clean=False,
        skip_download=True,
    )


class _FakeLiveDataClient:
    def benzinga_earnings(self, start, end, *, limit):
        return [{"ticker": "AAPL", "date": "2026-08-06", "importance": 4}]

    def stock_daily_bars(self, ticker, start, end):
        return [{"c": 100 + (index % 3) * 0.1} for index in range(80)]

    def option_contracts(self, ticker, start, end):
        return [
            {"ticker": "O:AAPL260918C00100000", "expiration_date": "2026-09-18", "contract_type": "call", "strike_price": 100},
            {"ticker": "O:AAPL260918P00100000", "expiration_date": "2026-09-18", "contract_type": "put", "strike_price": 100},
            {"ticker": "O:AAPL260918C00115000", "expiration_date": "2026-09-18", "contract_type": "call", "strike_price": 115},
            {"ticker": "O:AAPL260918P00085000", "expiration_date": "2026-09-18", "contract_type": "put", "strike_price": 85},
        ]

    def option_previous_bar(self, option_ticker):
        prices = {
            "O:AAPL260918C00100000": 6.0,
            "O:AAPL260918P00100000": 6.0,
            "O:AAPL260918C00115000": 1.0,
            "O:AAPL260918P00085000": 1.0,
        }
        return {"c": prices.get(str(option_ticker).upper(), 0.0)}


class _EmptyLiveDataClient(_FakeLiveDataClient):
    def benzinga_earnings(self, start, end, *, limit):
        return []


if __name__ == "__main__":
    unittest.main()
