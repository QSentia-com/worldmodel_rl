from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qsentia_worldmodel_rl_containerized.config import SignalRuntimeConfig
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


if __name__ == "__main__":
    unittest.main()
