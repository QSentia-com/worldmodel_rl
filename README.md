# WORLD_MODEL-RL

Production runtime for the QSentia WORLD_MODEL-RL model.

This container is intentionally separate from `dreamer-event-vol` / `WORLD_MODEL-NON-RL`. It downloads the RL Guardian artifact from LakeFS, validates that it is exactly `v12b_small_rl_guardian`, builds Alpaca option multi-leg trade intent from the artifact's `live_order_map.csv`, checks Alpaca account connectivity, records the run, and publishes outputs for the QSentia backend.

Live trading requires a fresh option-data refresh before execution. The refresh entrypoint:

```bash
python -m qsentia_worldmodel_rl_containerized.live_signal_refresh
```

downloads the same LakeFS artifact, verifies that `selected_decisions_live.csv` contains same-day entry/exit decisions, validates mapped option contracts against Massive/Polygon reference data, and publishes a `live-refresh/{entry|exit}/latest.json` report. The order job can be configured with `QSENTIA_REQUIRE_FRESH_OPTION_DATA=true`; in that mode it fails closed unless the same-day refresh report exists and is still fresh.

Current production artifact:

`lakefs://qsentia-models/main/world_rl/v12b_small_rl_guardian_full_model_artifact.zip`

Backtest source:

`s3://qsentia-dev-backtest-imports/world_rl/combined/`

Safety defaults:

- The runtime refuses any artifact that is not `v12b_small_rl_guardian` / `v12b_exact_same_leg_inverse_plus_small_rl_guardian_v1`.
- The runtime refuses artifact objects outside the `world_rl/` LakeFS path.
- The artifact currently declares `live_trading_enabled=false`; by default the container respects that and will not submit stale historical option legs.
- `QSENTIA_CONNECTIVITY_CHECK_ONLY=true` checks account, clock, and positions without submitting orders.
- If live trading is deliberately enabled later, the job only uses current-date artifact rows unless `QSENTIA_ALLOW_STALE_OPTION_SIGNAL=true` is explicitly set.
- When `QSENTIA_REQUIRE_FRESH_OPTION_DATA=true`, stale exported decisions are refused before any Alpaca order is submitted.
