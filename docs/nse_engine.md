# NSE Engine — long-only core + uncorrelated sleeves

One engine for research, validation, paper and live trading of NSE cash
equities (CNC). It replaces the fast optimizer simulators for validation and
supersedes `services/research/full_pipeline_backtest.py` (kept, fixed, but no longer
the source of truth).

Scope: NSE equities and NSE-listed metal ETFs only. No BTC, US stocks or options.

## Design rules

- **Point-in-time everything.** A decision dated `t` uses data dated `<= t`.
  `generate_targets(data.until(t), ...)` must equal `generate_targets(data, ...)`
  at `t` (tested). `until()` cuts every frame on the calendar, the as-printed
  `close_unadj` included (it was dropped until 10 Oct 2026, LN-T8).
- **Survivorship-free data.** Prices come from NSE bhavcopy archives (every
  traded security, including later-delisted ones). The universe is chosen
  point-in-time by liquidity, so no index-membership file is needed.
- **No current lists in history.** Nothing a backtest reads may come from a
  list of today's names (index members, sectors, ETFs) unless it is dated:
  the sector map is kept as dated snapshots (`data/nse_sector_maps/<date>.json`,
  written by `build-store` whenever the NIFTY 500 list changes) and a decision
  uses the latest snapshot dated on or before it, none before the first
  (14 Sep 2026). So the 25% sector cap acts in the paper and live books and
  never in a 2013-2025 backtest (SB2, 8 Oct 2026: before this, a Mac run
  capped only the companies that survived into today's NIFTY 500; Kaggle runs
  never had the map). `tests/test_survivorship.py` guards these rules.
- **Realistic execution.** Decide after close `t`; fill at open `t+1` plus
  impact; stops fill at `min(open, stop)`. Participation is capped at a share
  of median traded value. Per-side statutory costs follow the historical
  schedule; the metal-sleeve ETFs pay ETF STT (none on gold, 0.001% on the
  sell side for silver) rather than the 0.1% a side on shares (cost model 2,
  U25; every run records `cost_model`, and runs are compared only within one
  version). An ETF's open is often a stray first trade, so an ETF fills at
  its open held within 3% of the day's close (cost model 3, D4). Gross
  exposure is at most 1 (CNC). Idle cash earns nothing, as in a Kite account
  (cost model 4, IC1; models 1-3 credited 6% a year that no paper or live
  path earns).
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
- **Sleeve trend confirmation (opt-in, not adopted).**
  `sleeves.trend_confirm_days` > 1 lets a metal sleeve enter or leave its
  trend only after that many closes in a row on the other side of its
  average (the regime gate's hysteresis). Default 1 = the close alone,
  hash-neutral. Tested at 3 (R13, 1 Oct 2026): gold round trips of 20
  sessions or less fell from 22 to 8 and MaxDD by 1.6 points, but Sharpe
  fell 0.04 and CAGR 0.3 points, so it failed its pre-registered rule
  (`docs/nse_engine_validation_plan.md` section 5i).
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
    fo_store.py        build_fo_store: index options / futures from the F&O bhavcopy (O2)
    reference.py       ETF list, symbol changes, sector map builder
    panel.py           load_market_data -> MarketData
  costs.py             statutory schedule, impact, participation cap
  universe.py          point-in-time liquidity universe
  signals.py           fast_trend, slow_trend, low_vol forecasts; FDM; combine
  regime.py            NIFTY trend + breadth + India VIX regime
  drawdown.py          drawdown rule from the book's own equity (halt / half / risk_off)
  portfolio.py         core selection, weights, rank-drop exits, trailing stops
  sleeves.py           gold / silver ETF trend sleeves (+ `sleeves.extra_symbols`, R14)
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
runners/run_nse_engine.py                      CLI: sync | backtest | validate | holdout | scorecard
```

## Contracts

### Data layer (`nse_engine.data`)

```python
BhavcopyArchive(root: str | Path, requests_per_second: float = 2.0)
    .sync(start: date, end: date, kinds=("equity", "delivery", "indices", "corpact")) -> dict  # counts; resumable
                                                  # kinds may add "fo" (F&O bhavcopy, not in the default)
    .sync_reference() -> dict   # eq_etfseclist.csv, symbolchange.csv, EQUITY_L.csv, ind_nifty500list.csv

build_store(archive_root, store_dir) -> dict          # writes parquet; idempotent
build_fo_store(archive_root, store_dir, equity_stores) -> dict  # data/nse_engine/fo_store; own data hash
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
`MarketData.compute_hash()`, the run manifests' `data_hash`, is version 2
since LN-T15 (manifests record `data_hash_version`). It covers dates, every
price frame (close, open, high, low, the as-printed close), volume, traded
value, the ETF set and, since K5, the index closes, so a change to any of
them moves it; run `refresh-registry` after one. Columns are named and
ordered by the name in force on the panel's last date, with the renames
inside the window hashed, so a rename after the window (a 2026 rename and a
2013-25 panel) no longer moves it. `MarketData.trade_names` keeps each
renamed column's names by first date, and the engine breaks forecast ties by
the name in force on the decision date (it used the latest name, so a later
rename could reorder an earlier decision). `corporate_events` and
`dividend_events` carry the adjustments behind the back-adjusted prices for
the paper and live books (LN-T4); neither is hashed.

The calendar is the store's: every day is probed (NSE has held Saturday and
Sunday sessions: Budget days, Muhurat, drills; the Sunday 1 Feb 2026 Budget
session was missing until 10 Oct 2026), and the special sessions in
`nse_engine.nse_calendar` are re-checked on every sync. `build_store` records
dates where more than 20% of EQ prev_close values disagree with the last
close (a missing session) in its manifest; `build-store` fails on a new one.
`nse_engine.nse_calendar` is the one list of holidays and special sessions
the live code uses (the Muhurat session is not traded live).

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
cscv_pbo(returns_matrix: pd.DataFrame, n_splits: int = 16, rf_annual: float = 0.0) -> dict  # pbo, logits, ...
# rf_annual: trials ranked on excess returns, as the deflated Sharpe (every engine caller passes 6.5%, LN-T16)
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
prices on each stock's own NSE tick (from Kite's instruments dump, read once
a day; since April 2025 NSE ticks run from Rs 0.01 below Rs 250 to Rs 5 above
Rs 20,000, so a fixed 5-paise grid put about a third of limit prices off-tick:
sells round down, buys up; without the dump, the next coarser slab's tick
and an alert), Kite's tradingsymbol (`SYM-BE` for a stock in series BE; a
stock Kite does not list is not sent, with an alert), one idempotent tag per
order (`NE<yymmdd><B|S><symbol>`, letters and digits only, so M&M is `MM`;
skipped if already in today's order book at any stage short of rejection,
`AMO REQ RECEIVED` included; if the book cannot be read, nothing is sent),
and **variety `amo` whenever the market is closed** — the engine decides after the close, and a regular order placed
then is refused by the market-hours guard in `order_service.place_order`.
Under the kill switch only reduce-only SELLs go through; a rejection or a
transient failure of one order never stops the others (three retries, then
reported). `reconcile_stop_gtts` then arms one stop GTT per holding, deletes
orphans and reports breached and missing stops. `live_order_outcomes` reads
the next day's order book for the session's tags. All of it was exercised
against a fake Kite on 27 Sep 2026. `EngineExecutor(..., dry_run=True)`
(or `.dry_run_live(plan)`) returns exactly what a live session would send,
without sending it.

`kite_connect/trading/live_session.py` (tracker L5) drives one live session
end to end: previous orders' outcomes into `paper_fills`, the book's
snapshot at the close, plan, orders, stops, session record and a LIVE email,
all in the Neon schema `live`. The book is a ledger of its own capital,
cash and quantities, so holdings outside it are never sold and their stops
are never touched (`reconcile_stop_gtts(quantities=, scope=)`). The ledger
also keeps what the backtest keeps on each `Holding` (LS1): each position's
entry session (`entries`, its first BUY fill) and its last stop (`stops`,
seeded from the stop planned with its BUY order). So live stops ratchet
from the highest close since entry, as in the backtest and paper, and are
never lowered: the plan starts from the higher of the broker's GTT trigger
and the ledger's last stop. A GTT that was not raised (a failed modify, an
edit at Kite) keeps the book's level with an alert. A trigger is rounded up
to the stock's tick, so rounding raises none. When the GTTs cannot be read, or a GTT is gone (it
triggered without a sale, expired or was deleted), the last stop is kept
with an alert, so the engine still exits through it. Each stop keeps the
close it was set against (`stop_basis`), so when the store back-adjusts
prices for a dividend, split or demerger, the stop and the broker's
trigger move with them, as the backtest's adjusted data does. A quantity
that differs at the broker raises an alert. If the data shows no corporate
action for it, the broker's GTT is used, or the stop is recomputed when
there is none. A stop already above the price, so no GTT can be placed,
raises an alert too. Settlement: every NSE share settles T+1 (T+0 is
an optional window for about 500 large caps; the book does not use it),
and since 7 Oct 2024 Zerodha credits 100% of a sale the same day for new
buys, except a sale of T1 holdings (shares bought the session before),
credited the next day. So the planner spends the proceeds of tonight's
sells on tonight's buys, as the backtest does at the open, but leaves out
the shares the book bought at today's open, with a note in the email.

The broker's view decides when the book's own record cannot (tracker
LN-T4..T6). After a missed or failed session, whose fills have left Kite's
one-day order book, the broker's quantities are explained by the orders
still pending, then by stop GTTs that triggered since, and adopted at the
fill session's open within the limit; anything left unexplained holds back
new buys and asks for `live_session --reconcile SYM=QTY@PX`. On a split,
bonus or consolidation (NSE's ratios in the store) the ledger's quantity
moves into the new units once, and while the broker has not yet credited
the new shares the position is valued whole and sells are capped at what
the broker can deliver; rights, demergers and price-only factors move only
the stops. Orders in a stock whose corporate action goes ex at the next open
are held back a night. A dividend is paid to the bank account, not the
book: the ex-date drop is booked as income withdrawn, so it is not a loss
in G4 or the drawdown rule. A renamed stock (the store's change table, or
the same ISIN at the broker under a new name) moves to its new name, and its
stop GTT is re-armed on the new instrument before the old one is deleted. A
merger, delisting or exit offer announced within five sessions sells the
position, with an alert. The paper book applies the same events: renames,
splits and bonuses rebase its lots and pending orders, dividends are paid
in cash, stops move with the prices, and a merged stock that stops trading
is sold at its last close, as the backtest does. The paper book also catches
up: every store session since the last processed one gets its stops and the
open's fills in order (a weekend Budget or Muhurat session, or a missed
run), then one plan is made from the latest close.

Exits are protected against gapped and circuit-bound opens (LN-T14). A
reduce-only SELL sits 500 bp under the close (`EXIT_LIMIT_BAND_BPS`), never
below the next session's lower circuit (tonight's band, from Kite's quote,
applied to tonight's price), or 190 bp under when the quotes cannot be read;
BUYs stay 100 bp over. A stop GTT's limit is 2% under its trigger, floored at
the same circuit. An engine SELL that did not fill (or filled in part) is
sent again the next evening under a new tag, unless that night's plan trades
the stock or wants to keep the shares. Over 2013-25 these choices left the
fewest sells unfilled at the model's CAGR; a stock locked at its lower
circuit cannot be sold at any limit (see the tracker). A stopped-out stock
stays blocked for 5 sessions, as in the backtest (`recent_stops` in the
ledger, LN-T9).

A real session fails closed on its own book (LN-T10): the equity history,
fills and sessions are read strictly (an outage raises instead of reading
as "no history", which would let the drawdown rule read normal); the ledger
and the ladder's state, configuration marker and gate are written in one
transaction; the orders about to be sent are written first, as INTENDED,
and nothing is sent when that write fails; a failed write after sending
raises, and the backup run dedupes by tag. Dry runs and the paper books
keep their "never block a run" behaviour. The nightly dry run is also
Kite's preflight (LN-T12): every order and stop must name an NSE equity
instrument in Kite's dump and sit on its tick, so a dry run counts as clean
only when Kite would accept it all; in a real session the same findings are
advisory, and only a BUY of a stock Kite does not list is held back.

`--dry-run` rehearses it
against the real broker and writes nothing.


Real orders require `CENTURION_PAPER_TRADE=false` and `CENTURION_NSE_ENGINE_LIVE=true`.

### Deployment and paper trading

`config/nse_engine_deployed.json` (loaded by `nse_engine.deployment`) pins the
one configuration that paper/live trades: engine config, `status`
(`placeholder` or `approved`), `source_run_id`, `approved_at`,
`paper_start_date` and `data_anchor_date`. It is written by
`runners/run_nse_engine.py promote`, the forward gate (V3, decision U19): a
paper candidate replaces it only after >= 60 sessions beside the deployed
book, a G4 PASS and a 2017-25 backtest Sharpe within 0.05 of the deployed
config's; PBO, deflated Sharpe, the benchmark gate and the holdout are printed
but no longer gate. Live trading refuses placeholder and candidate files.

**Extra paper books (trackers D1, D5).** Every `config/nse_engine_<book>.json`
other than the deployed file is a paper book (status `candidate`: paper only,
`live_allowed()` refuses it) holding a configuration on trial under the
forward gate, with the same drawdown overlay as the deployed book. Today:
`candidate` = `2d64ba4c`, the B1 baseline with `regime.scale_neutral = 0.6`
(from 28 Sep 2026), and `e4` = `93cf6c4d`, B1 with exit rank 60 and refill
exits (from 5 Oct 2026). The daily job runs them in one loop after the
deployed and live sessions, on the same store; each book's settings come
from its file name: `CENTURION_NSE_DEPLOYMENT` (the file),
`CENTURION_PAPER_SCHEMA=<book>` (every Neon table of the book - positions,
snapshots, fills, sessions, weekly checkpoints, state - lives in that
Postgres schema; `PaperCloudSync(schema=)` qualifies raw SQL and uses
`schema_translate_map` for the ORM, never `search_path`, which Neon's pooler
drops between transactions; the live book's schema is refused),
`CENTURION_PAPER_DB_PATH` (its own local SQLite), its own same-period shift
reference `data/shift_reference_<book>.csv`, and `CENTURION_PAPER_BOOK_LABEL`
(its emails read `Centurion paper [<book> <fingerprint>] ...`). A book is
skipped before its `paper_start_date`. Adding a book is adding its file. The
books share only the paper switch and never write the switch row's run
status; a failing book emails and the next one still runs. The trade
monitor shows any of them (G12): `GET /api/v1/screener/monitor/books` lists
the books from the config files, and the monitor endpoints take `?book=<book>`
(the deployed book without it; an unknown book is 404).

**Books register and the weekly comparison (tracker V4).** The operating
reference (books, the three checks, candidate selection, promotion, go-live
checklist) is `docs/paper_trade_strategy.md`.
`docs/books_register.csv` holds one row per book: its configuration and
one-line `description` (a key of the book file), the backtest scores of the
registry's like-for-like run (same window, newest cost model), the Sharpe
of that run's 2017–25 returns (`bt_sharpe_2017_25`, the forward gate's check
3; not a walk-forward, which re-fits the family and is in the scorecards),
PBO / DSR from its validation, and, once the Saturday job has filled them,
its paper scores and
forward-gate status, with a one-line summary. `python -m nse_engine.books
register` rebuilds the backtest columns from the run registry (research
machine only; commit the file). Every Saturday `tools/books_report.py` reads
each book's Neon schema and emails one report: every book over its whole
record (return, alpha against NIFTY 50, Sharpe, MaxDD, G4); each trial
against the deployed book over their common sessions (difference in points,
the t of the daily differences, tracking error); the three forward-gate
checks per trial as PASS / FAIL / PENDING; and, for a trial that has cleared
all three, its promotion review (the same tables, the backtest comparison
and the rationale) with READY FOR YOUR REVIEW in the subject. The register
is attached. Nothing promotes on its own: `promote` stays a hand-run
command, and `python -m nse_engine.books review --book <book>` prints the
review from the register alone.

Adding a book: (1) record its 2013–25 backtest in the registry on the
current cost model (Kaggle, `cloud.kaggle_local`) and `validate` it; (2)
write `config/nse_engine_<book>.json` with `status: candidate`, the
`source_run_id`, a `paper_start_date`, a one-line `description` and `notes`;
(3) `python -m nse_engine.books register`, and commit both files. The daily
job picks the book up at its start date, the Saturday report includes it,
and after 60 sessions `promote --check` shows the gate. The trial budget
(U27) still applies: a new book is a pre-registered configuration.

Paper flow (`EngineExecutor.run_paper_session`, daily after the bhavcopy is
published): GTT-style stop checks at the open → fill yesterday's pending
orders at this session's open with the backtest's impact and statutory costs
→ mark to close → plan from the close (targets scaled by the distribution
shift multiplier for new risk) → queue orders for the next open. See
`docs/nse_engine_validation_plan.md` for gates and monitoring.

## Scorecard (tracker SC1)

`python -m runners.run_nse_engine scorecard --book all` writes one report per
book (`docs/scorecards/<as-of>_<book>.md`, JSON under
`data/nse_engine/scorecard/`) from the book's latest recorded run on the
current cost model: return and risk (Sharpe, Sortino, information ratio vs
NIFTY 50 TRI, Calmar, MaxDD, volatility, CVaR, skew, kurtosis, beta),
attribution (style factors built point in time from the store, the alpha
left after them, alpha decay by horizon and by year), trading (turnover,
hit rate, win/loss, profit factor, P&L per round trip, modelled impact by
participation), capacity (the capital at which impact eats half the gross
edge), robustness (walk-forward OOS, deflated Sharpe and PBO from the
registry, one-setting neighbours, NIFTY-trend and VIX regimes), correlation
with the other books, the options sleeves and the metal ETFs, and the paper
book's G4 gate when `CENTURION_DATABASE_URL` is set (`--paper-schema` for a
second book).  Pass rules are section 1's targets of the tracker, fixed before
the data is read; everything else is reported.  `nse_engine/scorecard.py`.
