from __future__ import annotations

import unittest
from unittest.mock import patch

from qsentia_worldmodel_rl_containerized.alpaca_execution import execute_alpaca_trade_intent


class AlpacaExecutionTests(unittest.TestCase):
    def test_respects_artifact_live_trading_guard(self) -> None:
        trade_intent = {
            "artifact_live_trading_enabled": False,
            "orders": [
                {
                    "order_class": "mleg",
                    "qty": "1",
                    "type": "market",
                    "time_in_force": "day",
                    "client_order_id": "world-rl-test",
                    "legs": [{"symbol": "NVAX200904C00170000", "side": "sell", "position_intent": "sell_to_open"}],
                }
            ],
        }
        with patch("qsentia_worldmodel_rl_containerized.alpaca_execution.AlpacaRestClient") as mocked:
            client = mocked.from_env.return_value
            client.account.return_value = {"id": "acct", "portfolio_value": "1000000"}
            client.clock.return_value = {"is_open": True}
            client.positions.return_value = []
            with patch.dict(
                "os.environ",
                {
                    "APCA_API_BASE_URL": "https://paper-api.alpaca.markets",
                    "APCA_API_KEY_ID": "key",
                    "APCA_API_SECRET_KEY": "secret",
                    "QSENTIA_ALPACA_EXECUTION_ENABLED": "true",
                    "QSENTIA_ALPACA_DRY_RUN": "false",
                    "DRY_RUN": "false",
                },
                clear=False,
            ):
                report = execute_alpaca_trade_intent(trade_intent)
        self.assertEqual(report["status"], "skipped_artifact_live_trading_disabled")
        client.submit_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
