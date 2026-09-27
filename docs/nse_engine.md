# NSE Engine — long-only core + uncorrelated sleeves

One engine for research, validation, paper and live trading of NSE cash
equities (CNC). It replaces the fast optimizer simulators for validation and
supersedes `services/research/full_pipeline_backtest.py` (kept, fixed, but no longer
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
  and sqrt(252); CAGR compounds over calendar years (days / 365.25) in the
  engine, the validation reports and the paper book alike — never sessions
  / 252, which overstates NSE figures by about 0.4 points a year. Every run
  is recorded (config hash, git commit, data hash, daily returns), so PBO
  and DSR cover every configuration ever evaluated.
- **No in-sample tuning tables.** Signal-group weights are fixed and
  hand-set. Parameter choice happens only inside walk-forward folds.
- **Drawdown rule (opt-in).** `nse_engine.drawdown` reads the book's own
  equity, not the market: beyond a drawdown from the episode peak it stops
  new entries and adds (`halt`), then scales core exposure down (`half`),
  then takes the core to zero (`risk_off`); it re-arms on a new
  `rearm_sessions`-session equity high. It is a pure function of the equity
  history, so live replays it from daily snapshots and cannot drift from the
  backtest. Off by default (`DrawdownConfig.enabled`, hash-neutral). Measured
  on the honest baseline (E2, 27 Sep 2026): at 20/30/35% it lifts Calmar
  0.92 → 0.99 and trims MaxDD 24.7% → 23.2% with CAGR unchanged; at 15/25/30%
  it whipsaws through 2015 and lowers Calmar to 0.86.
- **Anchor independence (opt-in).** With the legacy settings a decision
  depends on where the data was loaded from: rebalance, universe-refresh and
  FDM-refresh days were counted from the first loaded row, forecast
  normalisers pooled every loaded date, and pandas EWMs remember every row.
  Measured on the deployed configuration over 2026: loaded from 2011,
  Sharpe 1.04 / +18.0% / 328 trades; loaded from 2021, 0.80 / +15.0% / 312.
  Six fields remove this — `signals.normalizer_window_days` (rolling pooled
  window), `signals.ewm_memory_spans` (EWM kernel truncated at k × span,
  summed lag by lag in a fixed order), `signals.calendar_schedule`,
  `portfolio.calendar_schedule`, `universe.calendar_schedule` (period starts
  by date: 2–5 days → ISO week, 6–21 → month, 22–63 → quarter) and
  `universe.history_window_days` (trailing count instead of count since row
  0). Set them all and `data.load_min_median_value_inr = 0` (the load-time
  liquidity filter is window-dependent; the engine's own universe already
  filters point-in-time), and `EngineConfig.required_warmup_days()` gives the
  rows needed before `start` for the load start not to matter — 2,596 with
  4 × 256 EWM memory, 504-day normalisers and the 504-day FDM, because the
  stages chain. Verified: the same 2026 window loaded from 2011 and from
  2014-07 gives identical daily returns (max |diff| 0.0) and the same 329
  trades; `python -m runners.run_nse_engine anchor-check` runs the test.
  Their legacy values (0 / False) are left out of `config_hash`, so the
  deployed `679cbd0c` keeps its hash and its results (re-verified against the
  recorded holdout to 3e-16).

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
  drawdown.py          drawdown rule from the book's own equity (halt / half / risk_off)
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
(4) price-inferred factors for unexplained large gaps, snapped to exact split/bonus
ratios where possible (ISIN changes flag face-value splits), each logged. Prices are
total-return by default: cash dividends from the corporate-actions file are
back-adjusted (`DataConfig.adjust_dividends`; False gives price-only series).
`index_close` has `NIFTY50`, `NIFTY50_TRI` (derived from NSE's dividend-points index), `NIFTY500` and
`INDIAVIX` where available: VIX from NSE index files, back-filled from yfinance
`^INDIAVIX` before NSE coverage starts. NSE's daily index files start in
February 2012 and also miss NIFTY 50 on a dozen later sessions, so
`load_market_data` fills `NIFTY50` gaps from `<store>/external/nifty50_history.parquet`
when a store has one (gaps only; NSE's values win). The cache is built by
`python -m nse_engine.data.external nifty50 --store <dir>` from Yahoo `^NSEI`
(identical to NSE's closes from 17 Sep 2007), the BSE Sensex scaled to NIFTY
before that (98.4% agreement on the 200-day trend state), and the previous
close on the few special sessions neither covers; every row records its
source. Both stores have it since 27 Sep 2026 (tracker K4, K5).
`MarketData.compute_hash()`, the run manifests' `data_hash`, covers dates,
symbols, closes, traded value and, since K5, the index closes: a change to
an index cache moves the fingerprint, so run `refresh-registry` after it.

### Engine (`nse_engine.engine`)

```python
generate_targets(data: MarketData, config: EngineConfig, as_of: pd.Timestamp,
                 holdings: Mapping[str, Holding] | None = None,
                 cache: EngineCache | None = None, *, equity: float | None = None,
                 stopped_out: Mapping[str, pd.Timestamp] | None = None,
                 drawdown: DrawdownDecision | None = None) -> TargetPortfolio
DrawdownTracker(cfg.drawdown).update(equity) -> DrawdownDecision   # once per session close
replay(equity: pd.Series, cfg.drawdown) -> pd.DataFrame            # the same, over a history

run_backtest(data: MarketData, config: EngineConfig, *, record: bool = True,
             tag: str = "", lag_days: int = 0) -> BacktestResult
```

`run_backtest` calls `generate_targets` on every decision day, so live and
backtest share identical logic. `EngineCache` holds the causal indicator
panels so that `generate_targets` is not recomputed from scratch each day.
`lag_days` delays execution by N extra sessions (lag sensitivity). With
`config.drawdown.enabled` the backtest runs the drawdown rule on its own
equity and passes each day's `DrawdownDecision` to `generate_targets`, which
blocks new names and adds outside `normal`, multiplies the regime scale by
the rule's scale and forces a rebalance on a state change; the per-session
states are kept in `BacktestResult.daily_state` and written to `drawdown.csv`.

Run directory layout (`config.runs_dir/<run_id>/`):
`config.json`, `manifest.json` (run_id, tag, config_hash, git_commit,
git_dirty, data_hash, start, end, created_at, metrics, lag_days, and
`refresh_of` when the run reproduces an earlier one after a store rebuild),
`returns.csv` (date, return), `equity.csv`, `trades.csv`, `weights.parquet`.
`run_backtest(..., manifest_extra={...})` adds fields to the manifest.

### Validation (`nse_engine.validation`)

```python
TrialRegistry(runs_dir).list_trials() -> pd.DataFrame
TrialRegistry(runs_dir).returns_matrix(start=None, end=None, dedupe_config=True) -> pd.DataFrame  # date x run_id
refresh_registry(registry, data, from_hash, window, dry_run=False) -> dict  # re-run same-window configs on new data
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
               target_fn=generate_targets, data_loader=load_market_data,
               equity_history_fn=None, drawdown_rule=None)
    .plan(as_of=None) -> ExecutionPlan    # orders + GTT stop instructions, no side effects
    .drawdown_decision(as_of, equity) -> (DrawdownDecision | None, replay frame | None)
    .live_orders(plan) -> list[dict]      # the exact broker orders: LIMIT, CNC, sells first, tag, variety
    .dry_run_live(plan) -> list[dict]     # the same list as results, nothing sent (dry_run=True routes here)
    .execute(plan) -> list[dict]          # paper: queue next-open orders; live: place_order + GTT reconcile
live_order_outcomes(kite, as_of) -> list[dict]   # complete | partial | rejected | cancelled | open, by tag
cloud_equity_history(cloud=None) -> pd.Series    # the live book's Neon snapshots (drawdown rule default)
    .execute(plan) -> list[dict]          # CNC orders via order_service; GTT stops
```
The deployment file may carry a **risk overlay**: `risk_overlay.drawdown_rule`
(a `DrawdownConfig`, adopted from E2 at halt 20% / half 30% / risk-off 35%,
re-arm on a 60-session high). It is not part of `engine`, so the strategy keeps
its config hash and its recorded trials. Every session `plan()` replays the
rule over the book's equity history (paper: the daily snapshots restored from
Neon; live: an injected `equity_history_fn`) plus today's mark, passes the
decision to `generate_targets`, and records the state on the plan
(`drawdown_state`, `drawdown_scale`, `drawdown_pct`, `drawdown_changed`),
in `paper_sessions`, in the daily email (subject tag and a red alert on every
change) and on the monitor's session card. `promote` carries the overlay over
to the next deployment.

**Live path (L3, 27 Sep 2026).** Real orders need `CENTURION_PAPER_TRADE=false`,
`CENTURION_NSE_ENGINE_LIVE=true`, an approved deployment and a Kite session;
otherwise `execute()` runs the paper path. The live branch is built from
`live_orders(plan)`: sells before buys, LIMIT + CNC at the plan's limit
prices, one idempotent tag per order (`NE<yymmdd><B|S><symbol>`, skipped if
already in the order book), and **variety `amo` whenever the market is
closed** — the engine decides after the close, and a regular order placed
then is refused by the market-hours guard in `order_service.place_order`.
Under the kill switch only reduce-only SELLs go through; a rejection or a
transient failure of one order never stops the others (three retries, then
reported). `reconcile_stop_gtts` then arms one stop GTT per holding, deletes
orphans and reports breached and missing stops. `live_order_outcomes` reads
the next day's order book for the session's tags. All of it was exercised
against a fake Kite on 27 Sep 2026. `EngineExecutor(..., dry_run=True)`
(or `.dry_run_live(plan)`) returns exactly what a live session would send,
without sending it.


Real orders require `CENTURION_PAPER_TRADE=false` and `CENTURION_NSE_ENGINE_LIVE=true`.

### Deployment and paper trading

`config/nse_engine_deployed.json` (loaded by `nse_engine.deployment`) pins the
one configuration that paper/live trades: engine config, `status`
(`placeholder` or `approved`), `source_run_id`, `approved_at`,
`paper_start_date` and `data_anchor_date`. It is written by
`runners/run_nse_engine.py promote`, which checks PBO, deflated Sharpe, the
benchmark gate and the holdout first. Live trading refuses placeholder files.

**Second paper book (tracker D1, from 28 Sep 2026).** `config/nse_engine_candidate.json`
(status `candidate`: paper only, `live_allowed()` refuses it) holds a
configuration on trial under the forward gate - now the K5 walk-forward's
choice `2d64ba4c`, the B1 baseline with `regime.scale_neutral = 0.6`, with the
same drawdown overlay as the deployed book. The daily job runs it right after
the deployed session on the same store, with four settings:
`CENTURION_NSE_DEPLOYMENT` (the candidate file), `CENTURION_PAPER_SCHEMA=candidate`
(every Neon table of the book - positions, snapshots, fills, sessions, weekly
checkpoints, state - lives in that Postgres schema; `PaperCloudSync(schema=)`
qualifies raw SQL and uses `schema_translate_map` for the ORM, never
`search_path`, which Neon's pooler drops between transactions),
`CENTURION_PAPER_DB_PATH` (its own local SQLite) and `CENTURION_PAPER_BOOK_LABEL`
(its emails read `Centurion paper [candidate 2d64ba4c] ...`). It shares only
the paper switch; it never writes the switch row's run status. Its steps
are `continue-on-error`, so a candidate failure only emails. The trade
monitor still shows the deployed book (a book selector is G12).

Paper flow (`EngineExecutor.run_paper_session`, daily after the bhavcopy is
published): GTT-style stop checks at the open → fill yesterday's pending
orders at this session's open with the backtest's impact and statutory costs
→ mark to close → plan from the close (targets scaled by the distribution
shift multiplier for new risk) → queue orders for the next open. See
`docs/nse_engine_validation_plan.md` for gates and monitoring.

