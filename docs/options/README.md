# Options toolkit

Prices, analyses and trades NSE options (NFO) through Kite Connect, built
from Zerodha Varsity Module 5 (options theory) and Module 6 (option
strategies). Paper by default; real orders only with `--live` and a typed
confirmation.

- Concepts and formulas, with every worked example as a test: [CONCEPTS.md](CONCEPTS.md)
- Strategies, generalization formulas and the selector: [STRATEGIES.md](STRATEGIES.md)

## What is where

All code is in `kite_connect/options/`.

| Module | Does |
|---|---|
| `theory.py` | Payoffs, moneyness, Black-Scholes price and Greeks, implied volatility, put-call parity, volatility, SD ranges, the ch. 18 rules |
| `strategies.py` | `Leg`, `Strategy` and the 12 Module 6 strategies: P&L, max profit / loss, breakevens, payoff table and chart, net Greeks; max pain, PCR |
| `selector.py` | Market view + volatility view + days to expiry → ranked strategies with the PDF rule behind each score, and strike guidance |
| `fno_costs.py` | F&O charges by date (brokerage, STT incl. exercise, exchange, SEBI, stamp, GST) and slippage |
| `options_config.py` | Every rate, threshold and hard limit (`OptionsConfig`) |
| `broker.py` | Kite session from the stored token, throttled quotes, basket margins, autoslice orders, the order log |
| `instruments.py` | Underlying + expiry + strike + CE/PE → contract, lot size and tick from Kite's live dump |
| `live_chain.py` | Option chain from live quotes with our own IV and Greeks, ATM IV, max pain, PCR |
| `iv_history.py` | 30-day ATM IV per session from the F&O store, IV rank / percentile, the IV level for the selector, the realised-vs-implied record (OD1) |
| `positioning.py` | FII / DII / client positioning from NSE's participant-wise open interest (OD2) |
| `pretrade.py` | The pre-trade report and the hard-limit check |
| `basket_executor.py` | Places a basket: buys first, every slice filled before the next leg, stops and reports on failure |
| `position_monitor.py` | Ledger of filled baskets; live P&L, Greeks, breakeven / stop-loss / max-loss alerts |
| `cli.py` | The commands below |
| `backtest.py`, `sleeves.py` | Research: options sleeves backtested on the NSE F&O archive (tracker O2; validation plan § 5q) |
| `signal_futures.py` | Research: PyPatel's put-call ratio, TRIN, VIX and breakout strategies on NIFTY futures (tracker O3; validation plan § 5r: none passed) |
| `fo_anomalies.py` | Research: awesome-systematic-trading's option-expiry week, volatility risk premium and overnight strategies on NIFTY (tracker O4; validation plan § 5s) |

The older files in the same folder (`option_chain.py`, `options_executor.py`,
`options_monitor.py`, the three `*_strategy.py`) serve `/api/v1/options` and
the retired scheduler jobs. The toolkit neither uses nor changes them.

## Setup

1. Python 3.11+ with the core requirements and the Kite client:
   ```bash
   pip install -r requirements-core.txt "kiteconnect==5.1.0"
   ```
   `kiteconnect` also brings `cryptography`, which decrypts the stored
   token. Tests and `select` need neither.
2. Environment, set in your shell for the session or in a local `.env` that
   git ignores. Never commit them or put them in code. The toolkit does not
   load `.env` itself: `set -a; source .env; set +a`.

   | Variable | Needed for |
   |---|---|
   | `ZERODHA_API_KEY` | every Kite call |
   | `CENTURION_DATABASE_URL` | reading the day's stored token (Neon) |
   | `CENTURION_KITE_TOKEN_KEY` | decrypting that token |
   | `CENTURION_KITE_PROXY` | live orders: the SSH tunnel to the static-IP VM, e.g. `socks5h://127.0.0.1:1080` |
   | `CENTURION_KITE_STATIC_IP` | live orders: refused unless the egress IP matches |

3. Optional overrides: a TOML file with the tables `[market]`,
   `[charges]`, `[slippage]`, `[limits]`, `[selector]`, given before the
   command (`cli --config file.toml report ...`) or as
   `CENTURION_OPTIONS_CONFIG`. Example:
   ```toml
   [limits]
   max_loss_per_trade_inr = 15000
   max_lots_per_leg = 4
   [market]
   risk_free_rate = 0.0548        # 91-day T-bill; update with its date
   risk_free_as_of = "2026-10-07"
   ```

## The Kite login

No script ever logs in: Kite Connect's terms forbid a stored password and
TOTP (decision U23).

1. On NSE trading days, while `CENTURION_LIVE_MODE` is `dry_run` or `live`,
   an email at **09:00 IST** carries the Kite login link (again at 17:30 if
   nobody has logged in). The *Open Kite Login* button on the Fly Kite page
   leads to the same login.
2. The link opens Zerodha's own login page; you log in there.
3. Zerodha redirects to the backend (`/ind-stocks/auth/callback`), which
   exchanges the one-time request token for the day's access token and
   stores it in Neon, encrypted. The tab closes itself after 3 seconds
   where the browser allows it.
4. The token is valid until **06:00 IST** the next day. Every command here
   reads it; without a login today they stop with "no Kite token for today".

Market data works from any IP. Orders must leave from the IP registered on
developers.kite.trade (mandatory since April 2026), so `--live` runs only
through `CENTURION_KITE_PROXY` with the egress check.

## Commands

```bash
# Live chain: prices, bid/ask, OI, our IV and Greeks; ATM IV, max pain, PCR
python -m kite_connect.options.cli chain --underlying NIFTY
python -m kite_connect.options.cli chain --underlying BANKNIFTY --expiry 2026-10-27 --strikes 10

# Market context: today's IV against its history, and positioning (offline; --live for IV from quotes)
python -m kite_connect.options.cli context --underlying NIFTY
python -m kite_connect.options.cli context --underlying RELIANCE --live

# Bring the local data up to date: archive, equity store (with the registry check), F&O store, IV histories
python -m kite_connect.options.cli refresh

# Strategy selector, offline; --underlying reads the IV level from history instead of --iv-level
python -m kite_connect.options.cli select --view moderate_bull --dte 6 --underlying NIFTY
python -m kite_connect.options.cli select --view moderate_bull --dte 6
python -m kite_connect.options.cli select --view neutral_big_move --dte 20 --iv-level low --vol-view rising --cost-sensitive

# Pre-trade report of a basket (legs: "BUY|SELL [ratio] CE|PE strike", comma-separated)
python -m kite_connect.options.cli report --underlying NIFTY --legs "BUY CE 25000, SELL CE 25150" --lots 2
python -m kite_connect.options.cli report --underlying NIFTY --legs "SELL CE 24900, BUY 2 CE 25100"

# Trade: paper by default; --dry-run logs the orders and sends nothing; --live sends real orders
python -m kite_connect.options.cli trade --underlying NIFTY --legs "BUY PE 25000, SELL PE 24850"
python -m kite_connect.options.cli trade --underlying NIFTY --legs "BUY PE 25000, SELL PE 24850" --dry-run
python -m kite_connect.options.cli trade --underlying NIFTY --legs "BUY PE 25000, SELL PE 24850" --live

# Open positions: P&L, Greeks, alerts
python -m kite_connect.options.cli monitor

# End to end in paper mode on the nearest expiry
python -m kite_connect.options.cli demo --underlying NIFTY --view moderate_bull
```

`--expiry` defaults to the nearest expiry at least a day away. Exit codes of
`report` and `trade`: `0` done; `1` a basket stopped part-way, or live not
confirmed; `2` refused by a hard limit. `demo` exits `1` unless its paper
basket filled in full.

**From GitHub, without local secrets:** Actions → *Options paper demo* →
Run workflow (underlying, view, lots). It runs `demo` in paper mode with
the repository's secrets and keeps the report, the paper fill, the monitor
output and the order log as the `options-paper-demo` artifact. Run it after
the day's login.

## Market context (OD1, OD2)

`chain`, `demo`, `context` and `select --underlying` print what history says
about today, and `demo` and `select --underlying` hand the IV level to the
selector instead of assuming "normal":

```
IV NIFTY 30-day 13.3% (eod 2026-10-07): 1-year rank 22, percentile 68; level normal (21-session realised cone: mean 12.3%, +/-1 SD 7.6% to 17.0%)
At IV percentile 60-80 (879 sessions since 2007-01-09): the next 30 days' realised volatility came in below IV 72% of the time, by a median +2.9 vol points (...)
Positioning 2026-10-07 (NSE participant-wise open interest; context, not a signal):
  FII index futures: 9% long (+0.6 pts on the day), 1-year percentile 17
  FII index calls: net -400k contracts (-53k on the day), 1-year percentile 1; near the year's extreme
  Client index futures: 84% long (+0.9 pts on the day), 1-year percentile 96; near the year's extreme
```

- **30-day IV**: each session's ATM call and put IVs of the two expiries
  around 30 days (at least 7 days away), interpolated in total variance, as
  India VIX does; the same construction from live quotes. NIFTY's tracks
  India VIX with correlation 0.92 (2015-26, 1.2 vol points median gap; VIX
  also prices the out-of-the-money skew). History: NIFTY from 2006,
  BANKNIFTY from 2012, stocks from 2013 (OD3).
- **IV level**: the selector's own rule (M5 ch. 20, M6 ch. 4) against the
  21-session realised-volatility cone over two years, annualised over
  trading days (sqrt 252), because implied volatility accrues only on them.
- **Realised vs implied**: on past sessions in today's IV-percentile bucket,
  how often the next 30 days' realised volatility came in below the IV.
  Overlapping windows: a description of the past, not a test.
- **Positioning**: FII index and stock futures (% long), FII index calls and
  puts (net contracts), client index futures, each with its 1-year
  percentile. It is context for your view; the selector never scores it.

The data is local (`data/nse_engine/`): run `refresh` before trading on a
new day. It rebuilds the local equity store through `run_nse_engine
build-store`, which checks the trial registry: when it prints STORE
FINGERPRINT CHANGED, refresh the registry (on Kaggle, where its runs live)
before recording any backtest.  Options are stored under the symbol they
traded as, so after a rename or demerger the IV history starts at the new
symbol (TATAMOTORS became TMPV on 24 Oct 2025). In CI (the *Options paper demo* workflow) there is no store, so the
context lines say "unavailable" and the selector keeps "normal".

## The pre-trade report

Printed before anything is sent, and shown again at the live confirmation:

- each leg: side, quantity (ratio × lots × lot size), contract, reference
  price (ask for a buy, bid for a sell, else last price), LIMIT price
  (reference moved 2% against the order, on the tick), IV;
- net premium paid or received, max profit and max loss in rupees;
- breakevens, each with its distance from spot in SDs to expiry;
- net Greeks: delta in units of the underlying, theta in ₹ a day, vega in
  ₹ a volatility point;
- the expected range by expiry at 1, 2 and 3 SD from the ATM IV;
- Kite's basket margin (with and without hedge benefit) and Kite's charges,
  beside the toolkit's cost model (charges and slippage);
- the hard-limit check.

## Safety

1. **Paper by default.** `--live` is the only way to send a real order.
2. **Typed confirmation.** `--live` prints the report and needs
   `PLACE <n> ORDERS` typed at a terminal. CI has no terminal, so a workflow
   can never send a live order.
3. **Hard limits**, refused in every mode (`[limits]`): max loss per trade
   ₹25,000, at most 10 lots a leg, underlyings NIFTY and BANKNIFTY. An
   unlimited loss (a naked short) is always refused.
4. **Order of legs.** BUY legs first: on entry they are the hedges, on exit
   they close the shorts.
5. **Fill check.** Orders are DAY LIMIT with autoslice: Kite splits a
   quantity above the freeze limit into up to 10 orders. Every slice of a
   leg must fill before the next leg starts. Slices still open after 30
   seconds are cancelled.
6. **Failure.** The first leg that does not fill in full stops the basket.
   The report states the partial position and flags any short without a
   long of the same type, so a naked short is never left unreported.
7. **Throttles.** Quotes at one request a second, orders at 8 a second
   (Kite allows 10).
8. **Audit.** Every order request and response, real or simulated, is
   appended to `data/options/orders.jsonl`.

## Monitoring

`monitor` marks every open position in the ledger to live quotes and reports
P&L and Greeks, with alerts when:

- spot crosses a breakeven from the profit side, or comes within 0.5% of
  one while still on it;
- spot passes the volatility stop-loss of M5 ch. 18 (entry spot × (1 ∓ daily
  σ × √days held), against the position's delta, σ from the entry ATM IV);
- the loss reaches 80% of the basket's maximum loss.

It alerts; it never trades.

## Files it writes

| Path | Holds |
|---|---|
| `data/options/orders.jsonl` | every order request and response (all modes) |
| `data/options/positions.json` | filled baskets: legs, fill prices, entry spot and IV |
| `data/options/demo/` | the demo's report and its own ledger |
| `data/nse_engine/fo_store/iv/` | the IV histories (`<SYMBOL>_v1.parquet`), refreshed on use |

`data/` is not tracked by git.

## Tests

```bash
python -m pytest tests/test_options_theory.py tests/test_options_strategies.py \
                 tests/test_options_costs.py tests/test_options_kite.py -q
```

Every Varsity worked example is a test (IDs `M5.x.y`, `M6.x.y` in the two
docs). The Kite tests replace every Kite call with a fake: no network, no
token. CI installs only `requirements-core.txt`; the toolkit imports
without `kiteconnect`.

## Market facts are read, not assumed

Lot sizes, expiry days, strike steps, ticks, margins and charges change
(the PDFs date from 2015–16). The toolkit reads lot sizes, expiries,
strikes and ticks from Kite's instruments dump on every run, and margins
and charges from Kite's basket-margin API. The freeze limit is left to
Kite's autoslice. The dated charge schedules in `fno_costs` are for
backtests and the offline estimate; keep `risk_free_rate` and its date
current.
