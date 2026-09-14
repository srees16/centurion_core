# NSE Engine — long-only core + uncorrelated sleeves

One engine for research, validation, paper and live trading of NSE cash
equities (CNC). It replaces the fast optimizer simulators for validation and
supersedes `services/full_pipeline_backtest.py` (kept, fixed, but no longer
the source of truth).

Scope: NSE equities and NSE-listed metal ETFs only. No BTC, US stocks or options.

## Design rules

- **Point-in-time everything.** A decision dated `t` uses data dated `<= t`.
  `generate_targets(data.until(t), ...)` must equal `generate_targets(data, ...)`
  at `t` (tested).
- **Survivorship-free data.** Prices come from NSE bhavcopy archives (every
  traded security, including later-delisted ones). The universe is chosen
  point-in-time by liquidity, so no index-membership file is needed.
- **Realistic execution.** Decide after close `t`; fill at open `t+1` plus
  impact; stops fill at `min(open, stop)`. Participation is capped at a share
  of median traded value. Per-side statutory costs follow the historical
  schedule. Gross exposure is at most 1 (CNC); idle cash earns a yield.
- **Honest statistics.** Sharpe uses excess returns over the risk-free rate
  and sqrt(252). Every run is recorded (config hash, git commit, data hash,
  daily returns), so PBO and DSR cover every configuration ever evaluated.
- **No in-sample tuning tables.** Signal-group weights are fixed and
  hand-set. Parameter choice happens only inside walk-forward folds.

## Package layout

```
nse_engine/
  config.py            EngineConfig and sub-configs (frozen dataclasses)
  types.py             MarketData, Holding, TargetPortfolio, Trade, BacktestResult
  data/
    validation.py      clean_ohlcv, factors_from_prev_close, adjust_for_factors
    archive.py         BhavcopyArchive: resumable download of NSE archives
    store.py           build_store: normalised parquet tables
    reference.py       ETF list, symbol changes, sector map builder
    panel.py           load_market_data -> MarketData
  costs.py             statutory schedule, impact, participation cap
  universe.py          point-in-time liquidity universe
  signals.py           fast_trend, slow_trend, low_vol forecasts; FDM; combine
  regime.py            NIFTY trend + breadth + India VIX regime
  portfolio.py         core selection, weights, rank-drop exits, trailing stops
  sleeves.py           gold / silver ETF trend sleeves
  allocator.py         core vs metals risk budget, gross <= 1
  metrics.py           performance metrics (excess Sharpe, CAGR, MaxDD, turnover)
  engine.py            generate_targets, run_backtest, run manifests
  validation/
    trials.py          TrialRegistry over run directories
    pbo.py             CSCV probability of backtest overfitting
    dsr.py             deflated Sharpe with clustered effective N
    walk_forward.py    anchored walk-forward re-fitting
    holdout.py         one-shot holdout with lock file
    benchmarks.py      equal-weight hold, naive momentum, NIFTY; benchmark_gate
    diagnostics.py     Aronson detrending, date-aligned alpha/beta, lag test
kite_connect/trading/nse_engine_executor.py   targets -> CNC orders + GTT stops
runners/run_nse_engine.py                      CLI: sync | backtest | validate | holdout
```

## Contracts

### Data layer (`nse_engine.data`)

```python
BhavcopyArchive(root: str | Path, requests_per_second: float = 2.0)
    .sync(start: date, end: date, kinds=("equity", "delivery", "indices", "corpact")) -> dict  # counts; resumable
    .sync_reference() -> dict   # eq_etfseclist.csv, symbolchange.csv, EQUITY_L.csv, ind_nifty500list.csv

build_store(archive_root, store_dir) -> dict          # writes parquet; idempotent
build_sector_map(archive_root, out_path="data/nse_sector_map.json") -> dict[str, str]

load_market_data(store_dir, start, end, *, symbols=None, series=("EQ","BE"),
                 min_median_value_inr=2.5e6, float_dtype="float32",
                 include_symbols=("GOLDBEES","SILVERBEES")) -> MarketData
```

`MarketData` columns are canonical symbols (latest name after renames, linked
through `symbolchange.csv` and ISIN continuity). NSE's bhavcopy PREVCLOSE is
generally *not* adjusted for splits/bonuses, so adjustment factors are taken
in this order: (1) bonus/split/consolidation ratios from NSE's daily
corporate-actions file (`PR{DDMMYY}.zip`, latest revision wins);
(2) `factors_from_prev_close` only where it explains the observed gap and is
not contradicted by other symbols that day; (3) rights (theoretical ex-rights
price) and demergers/capital reductions (ex-date open / prior close);
(4) price-inferred factors for unexplained large gaps, each logged. Prices
are price-return (dividends not adjusted). `index_close` has `NIFTY50`, `NIFTY500` and
`INDIAVIX` where available: VIX from NSE index files, back-filled from yfinance
`^INDIAVIX` before NSE coverage starts.

### Engine (`nse_engine.engine`)

```python
generate_targets(data: MarketData, config: EngineConfig, as_of: pd.Timestamp,
                 holdings: Mapping[str, Holding] | None = None,
                 cache: EngineCache | None = None) -> TargetPortfolio

run_backtest(data: MarketData, config: EngineConfig, *, record: bool = True,
             tag: str = "", lag_days: int = 0) -> BacktestResult
```

`run_backtest` calls `generate_targets` on every decision day, so live and
backtest share identical logic. `EngineCache` holds the causal indicator
panels so that `generate_targets` is not recomputed from scratch each day.
`lag_days` delays execution by N extra sessions (lag sensitivity).

Run directory layout (`config.runs_dir/<run_id>/`):
`config.json`, `manifest.json` (run_id, tag, config_hash, git_commit,
git_dirty, data_hash, start, end, created_at, metrics), `returns.csv`
(date, return), `equity.csv`, `trades.csv`, `weights.parquet`.

### Validation (`nse_engine.validation`)

```python
TrialRegistry(runs_dir).list_trials() -> pd.DataFrame
TrialRegistry(runs_dir).returns_matrix(start=None, end=None, dedupe_config=True) -> pd.DataFrame  # date x run_id
cscv_pbo(returns_matrix: pd.DataFrame, n_splits: int = 16) -> dict  # pbo, logits, n_combinations, ...
deflated_sharpe(returns: pd.Series, trials_matrix: pd.DataFrame | None = None,
                n_trials: float | None = None, rf_annual: float = 0.0) -> dict
run_walk_forward(data, base_config, param_grid: dict[str, list], train_years=4,
                 test_months=12, anchored=True) -> dict  # stitched OOS returns, per-fold choices
run_holdout(data, config, start, end, lock_path="data/nse_engine/holdout.lock",
            force=False) -> dict
run_benchmarks(data, config) -> dict[str, pd.Series]  # daily returns, same costs/fills
benchmark_gate(returns: pd.Series, benchmarks: dict, margin: float = 0.3, rf_annual=...) -> dict
```

### Live (`kite_connect.trading.nse_engine_executor`)

```python
EngineExecutor(kite=None, paper: bool = True, config: EngineConfig | None = None,
               target_fn=generate_targets, data_loader=load_market_data)
    .plan(as_of=None) -> ExecutionPlan    # orders + GTT stop instructions, no side effects
    .execute(plan) -> list[dict]          # CNC orders via order_service; GTT stops
```

Real orders require `CENTURION_PAPER_TRADE=false` and `CENTURION_NSE_ENGINE_LIVE=true`.
