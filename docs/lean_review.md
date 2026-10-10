# QuantConnect Lean compared with Centurion (LN1, 10 Oct 2026)

Tracker LN1. Lean commit 80e7843 (6 Oct 2026) and its Zerodha brokerage plugin were compared with Centurion in eight areas: data, execution realism, framework, statistics, live brokerage, options, indicators and reuse. Each improvement candidate was verified against both codebases by a separate reviewer. A coverage pass then added missed areas, and a final pass ranked what survived. In all, 48 candidates were verified: 9 confirmed, 34 partly confirmed with a corrected proposal, and 5 refuted.

**Status since the review was written (10 Oct):**

- Done: cost model 4 (IC1).
- Done: kill threshold 0.239 (IC1b).
- Done: live entry dates, never-lowered and rescaled live stops (LS1).
- Done: unsettled-share sale proceeds kept out of the buy budget (ST1).
- Open with you: DDPI and AMO funding (ST2).
- Each ranked item is tracked as LN-T<number> in docs/ACTION_TRACKER.txt (P0 and P1 in the queue, P2 and P3 in the backlog); its status lives there, not here.

## Summary

Verdict on Lean: use it for patterns only. Do not adopt it as an engine, library, CLI, cloud service or broker plugin. Centurion is already ahead on data integrity, costs, validation and Kite plumbing (see centurion_ahead).

The review's real value is in the seams between the validated backtest and the paper/live books. Several of those gaps would hurt from the first Rs 6 lakh session in mid December:

- **Off-tick prices.** Since 2025, NSE ticks are price-tiered. About a third of tonight's limit prices and about a quarter of GTT prices are rounded to a fixed Rs 0.05, which is off-tick, so Kite would reject them.
- **No depository sell authorisation check.** Nothing confirms DDPI, so every AMO exit and every GTT stop could be refused.
- **Duplicate orders.** An AMO whose response is lost is re-sent, because the retry check does not recognise 'AMO REQ RECEIVED'. A re-run when the order book cannot be read re-sends the whole plan.
- **Corporate actions.** Splits and bonuses are not applied to paper lots, stops or the live ledger. In 2013-25 this happened about 3 times a year, each a fake gap stop costing about 2% of equity. The GOLDBEES 1:100 split would show as a -34.7% day. A 60-session G4 window has about a 54% chance of containing one.
- **Series moves and renames.** These break the broker-to-ledger mapping. Today HFCL, MTARTECH and STLTECH trade only as SYM-BE; HEG became HEGAM.
- **Missed sessions.** A missed login permanently loses the previous night's fills. The result is a doubled position and an orphaned lot with no stop.

All of these are live-robustness or data-integrity fixes. None changes backtests and none spends the remaining trial.

**Items that change recorded results.** Each is a data re-baseline with no trial spent:
- the point-in-time name tie-break, which ships with a rename-invariant, widened data hash;
- the missing Sunday 1 Feb 2026 Budget session and the Sunday-probe calendar fix;
- the options label-to-contract settlement map.

Run these, and the switch to excess-basis PBO, as one Kaggle refresh before the one remaining configuration is evaluated, so that trial is judged on the corrected baseline.

**Already done or refuted:**
- Idle-cash yield removed as cost model 4 (IC1). It is still uncommitted, and IC1b, the kill threshold 0.247 -> 0.239, is still pending.
- Live entry dates (LS1).
- The overlay chain.
- The G4 reference yield fix, which IC1 covers.
- An authorised_quantity preflight.

**Time-critical:** the DDPI check, the calendar and paper catch-up before the 8 Nov Muhurat session, and every P0 item before the first real session.

## Can Lean's libraries or APIs be used?

Patterns only, plus a few small pieces of logic re-implemented in Python: Lean's ApplySplit arithmetic, its daily-bar LimitFill rule for a reporting reference, and the OTM-mirror IV rule. None of the following routes works:

- **Engine or library.** Lean is .NET 10 hosting CPython 3.11 through QuantConnect's pythonnet fork, with pandas 2.3.3 and numpy 1.26.4 pinned. Centurion runs pandas 3.0.6 and numpy 2.4.6. Loading Lean's DataFrame support monkey-patches pandas indexing for the whole process. Kaggle now runs Python 3.13 with no .NET and no internet, and the paper job deletes /usr/share/dotnet.
- **lean CLI.** It needs a paid QuantConnect organisation plus Docker with a 14 GB image. That does not fit a 14 GB Actions SSD, a Kaggle kernel or an unprivileged HF Space.
- **Cloud API.** QuantConnect has no NSE history, live trading needs a paid tier, and the Kite token would be handed to a third party.
- **Data.** Lean's India reference data is stale: incomplete holidays, no weekend or Muhurat sessions, NIFTY lot 75, last-Thursday expiries wrong in 18 of 237 months. It has no NSE equity factor files or map files and no point-in-time universe.
- **Zerodha plugin.** It is weaker than Centurion's Kite path: DAY SL orders with no GTT or AMO, no proxy, a static token, verbatim tradingsymbols, an unused tick_size, and blind 5x HTTP retries. At start-up it posts machine identifiers to a QuantConnect licence endpoint and exits without a paid module.
- **Trial budget.** Swapping engines would re-baseline every registry trial and consume the last one.

What is worth taking is Lean's design patterns:
- one dump-backed symbol and tick source;
- central price rounding;
- split and dividend events applied to holdings and open orders;
- a single status model for orders;
- refusing to trade on an uncertain broker state;
- regression algorithms with expected statistics in CI;
- a point-in-time mapped symbol;
- a pre-submit check that returns a reason.

## Ranked plan

| # | Priority | Item | Effort | Changes backtests | Spends the trial |
|---|---|---|---|---|---|
| 1 | P0 | Round every live limit, unwind and GTT price to the instrument's NSE tick from Kite's dump | small (about 1 day) | no | no |
| 2 | P0 | Prove the account can sell: DDPI confirmed, one supervised AMO sell and one triggered GTT sell recorded, enforced as a third go-live check | small (code about half a day; one supervised evening plus the next session; about Rs 40 in fees) | no | no |
| 3 | P0 | Make order placement idempotent for AMOs and fail closed when the order book cannot be read | small | no | no |
| 4 | P0 | Apply splits and bonuses (and dividends) to paper lots, pending orders, stops and the live ledger, using the store's own factors | medium | no | no |
| 5 | P0 | One EquityInstruments resolver: send BE-series tradingsymbols, map holdings, orders and GTTs back to engine symbols, and carry positions across renames and announced mergers | medium | no | no |
| 6 | P0 | Reconcile the live ledger after a missed or failed session and block buys until it balances | medium | no | no |
| 7 | P1 | Probe every Sunday, backfill the missing 1 Feb 2026 Budget session, make paper catch up across sessions, and keep one shared NSE calendar | medium (deadline: before 8 Nov 2026) | yes | no |
| 8 | P1 | Make MarketData.until() keep close_unadj, so trial books plan on the as-printed price filter they were validated with | small | no | no |
| 9 | P1 | Give the live book the backtest's 5-session stop cooldown across sessions | small (about 25 lines) | no | no |
| 10 | P1 | Fail closed when the live ledger, the intended orders or the risk history cannot be saved or read | small | no | no |
| 11 | P1 | Settle whether Kite credits pending sell AMOs, cap buys at the broker's free cash, and read rejections the same night | small | no | no |
| 12 | P1 | Make the nightly dry run check every order and stop against Kite's instrument dump, so a dry run is clean only with zero WOULD_REJECT | small | no | no |
| 13 | P1 | Nightly expected-statistics canary for the three books on Actions, after fixing the Actions store's missing NIFTY back-fill | small (about 1 minute added to the nightly job) | no | no |
| 14 | P1 | Live exit and stop price protection: GTT buffer 2% with a circuit floor, wider circuit-clamped exit sells, and carry-forward of unfilled sells | medium | no | no |
| 15 | P1 | Break forecast ties by the name in force on the decision date, and ship a rename-invariant, widened data hash in the same re-baseline | small code, plus one Kaggle job | yes | no |
| 16 | P1 | Rank CSCV PBO trials on excess returns, consistent with DSR and walk-forward selection | small | no | no |
| 17 | P2 | Report rejected stop and exit sells the same evening, with Kite's reason and a fix hint | small | no | no |
| 18 | P2 | Scorecard regime split: label each day with the regime known at the previous close | small | no | no |
| 19 | P2 | Stop a single missing NIFTY or sleeve print from blanking the 200-day means; fix the archive rule and add a nightly presence alert | small | no | no |
| 20 | P2 | One label-to-contract settlement map used by every options backtest, plus a guard against using the settle column as a price | small, plus one Kaggle options job | yes | no |
| 21 | P2 | Model the orders live actually sends: observational limit and GTT outcomes in paper now, a hash-neutral live-rule reference, and DailyLimitFill only inside X1 | medium | no | no |
| 22 | P3 | Remove hash-seed dependence so reruns are bit-identical | small (two lines) | no | no |
| 23 | P3 | Capacity: name the assets that bind first and value-weight the over-cap share; fix the 'cut by the 5% cap' mislabel | small | no | no |
| 24 | P3 | Report the Sharpe standard error, PSR against 1.2 and MinTRL beside the pass rules | small | no | no |
| 25 | P3 | Drawdown periods (depth, duration, recovery) and the book's behaviour across NIFTY TRI's own drawdown episodes | small | no | no |
| 26 | P3 | Stamp the runtime in every run manifest and warn when a trial set mixes OS or architecture | small | no | no |
| 27 | P3 | Options toolkit: give each ITM leg its OTM mirror's IV (Lean's smoothing rule), so monitor Greeks and the volatility stop are never lost | small (steps 1-2); moderate with the forward | no | no |
| 28 | P3 | Document the missing 2006-09 corporate actions and dividends in store_ext2006; optionally import the ~17 missed events | small (caveat); medium (import) | yes | no |
| 29 | P3 | After go-live, consider sweeping idle cash into a growth liquid ETF with hysteresis (would be cost model 5) | medium | yes | no |

### 1. Round every live limit, unwind and GTT price to the instrument's NSE tick from Kite's dump (P0, `tick-size-from-instrument-dump`)

Category: live_robustness. Reuse from Lean: adopt_pattern.

**Why.** Since 2025-Q2, NSE equity ticks follow price bands. Today about 140 of the top 300 have a tick above 0.05.

Fixed 0.05 rounding leaves these prices off-tick:
- 93-96 of roughly 300 BUY limits and the same number of SELL limits;
- 62-78 GTT prices;
- in the H2-2025 replay, about 34% of all engine orders.

Kite rejects off-tick prices, and order_service treats that rejection as final. A GTT with an off-tick price can fail at the moment it triggers. Dry runs never reach the exchange, so D3's five clean runs cannot catch this.

Lean contributes only the pattern of central rounding. Its own India equity tick defaults to 0.01, which is also wrong.

**Action.**

Add one module, kite_connect/trading/ticks.py (shared with the identity resolver). Each session it loads kite.instruments('NSE') once, cached by date, filtered to EQ rows, into {exact tradingsymbol, including -BE: tick_size}. Thread the tick through every price:
- nse_engine_executor._tick: BUY rounds up, SELL rounds down (executor ~:656/:660).
- live_session.unwind_orders (~:357).
- gtt_stops.round_to_tick, stop_limit_price, _payload and reconcile_stop_gtts. Round the trigger UP to the tick, so the never-lowered LS1 stop holds. Round the limit DOWN, floored at one tick.
- The 'unchanged' check (gtt_stops ~:278) and live_session's stop-lowering tolerance (~:247) use a per-symbol tick/2.
- reconcile modifies any existing GTT whose trigger or limit is off the current tick, since NSE re-tiers quarterly.

When the dump is unavailable or lacks a symbol, fall back to the NSE price-slab tick one slab coarser than the close's slab and raise an alert. Never fall back to 0.05.

Correct the '5-paise tick' statements in docs/nse_engine.md:274, nse_engine_validation_plan.md:1583 and ACTION_TRACKER.txt:447-448.

**Test plan.**

Temporary tests, deleted after verification; then the core-only fast suite.

1. Discriminating units:
   - 1355.0 at tick 0.1: SELL 1341.4 (not 1341.45), BUY 1368.6.
   - 7005 at tick 0.5: 6934.5 / 7075.5.
   - 11450 at tick 1: 11335 / 11565.
   - 123075 at tick 5: 121840 / 124310.
   - GTT trigger 123072 rounds up to 123075.
2. Fake Kite that raises InputException on off-tick prices, fed a top-300 plan plus unwind and reconcile.
   - Before the fix: about 93 order rejections and about 78 GTT rejections.
   - After the fix: 0.
3. Dump fetch raises: an alert is raised, and every price lands on the coarser-slab grid (2,694/2,694 store symbols on-tick).
4. One live_session --dry-run against the real dump reports 0 off-tick prices.
5. Re-run the store grid check (tickverify/grid.py, gtt.py) each quarter.

### 2. Prove the account can sell: DDPI confirmed, one supervised AMO sell and one triggered GTT sell recorded, enforced as a third go-live check (P0, `sell-path-ddpi-readiness`)

Category: live_robustness. Reuse from Lean: adopt_pattern.

**Why.** Without DDPI or POA, every CNC sell from demat needs a same-day CDSL TPIN authorisation; Kite returns HTTP 428 'needs authorisation at depository'. Nobody is present to give it, either for the evening AMO exits or for GTTs that trigger in market hours.

Today:
- readiness() checks only paper G4 and 5 dry runs;
- dry runs never send a sell;
- DDPI appears only in the MTF due diligence.

The first stop-out or rebalance exit after the Rs 6 lakh rung would therefore be the first sell the system has ever attempted. Lean has nothing comparable: CanSubmitOrder is static, and only its reason-returning shape is reused.

**Action.**

1. Now. Check Console > Account > Segments/DDPI on the primary Zerodha ID. If DDPI is absent, submit it online.
2. Before about 12 Dec, run one supervised session using 2 shares of a cheap liquid name kept outside the engine ledger:
   - an API AMO LIMIT SELL CNC sent through order_service and the static-IP tunnel, with a non-NE tag;
   - a single-leg SELL CNC GTT with its trigger near LTP.
   Record order_history, get_gtt and order_result the same day. In the same session, settle three other open broker questions:
   - a buy AMO larger than free cash but smaller than free cash plus the pending sale, cancelled before 09:00;
   - a far-from-market AMO carrying a '-' in its tag;
   - profile().meta.demat_consent and the key names of one raw holdings row.
3. Code. Add a SELL_PATH_KEY attestation and a subcommand that verifies the recorded order IDs against Kite. Add a third tuple to capital_ladder.readiness() that requires all of:
   - ddpi_confirmed_on;
   - a matching user ID;
   - the AMO sell COMPLETE;
   - the GTT triggered with its order COMPLETE.
   The first-real gate in live_session uses it. CENTURION_GO_LIVE_OVERRIDE is unchanged.
4. Add the item to D3. Defer the trading_lock change for MU2 accounts until registration.

**Test plan.**

Unit tests on readiness():
- These must fail: attestation missing, wrong user ID, AMO REJECTED, GTT triggered but its order REJECTED, GTT still active.
- This must pass: everything COMPLETE.

In run_live_session for the first real session:
- without the attestation it raises 'go-live refused: sell path';
- with the override it runs, and the email says OVERRIDDEN.

The real proof is the funded test: the AMO sell reaches COMPLETE at the next open and the GTT order reaches COMPLETE. Record both responses in ACTION_TRACKER.

### 3. Make order placement idempotent for AMOs and fail closed when the order book cannot be read (P0, `idempotent-amo-placement`)

Category: live_robustness. Reuse from Lean: adopt_pattern.

**Why.** Every engine order is an AMO, which sits in 'AMO REQ RECEIVED'. That status is not in the retry whitelist (OPEN, COMPLETE, TRIGGER PENDING), and calls go through an SSH tunnel with a 7 s timeout.

Fake-Kite runs:
- a lost response gives 2 or 3 live AMOs on one tag;
- a same-evening re-run with one failed order-book read sends the whole plan twice.

Five of today's top 300 (M&M, GVT&D, BAJAJ-AUTO, NAM-INDIA, M&MFIN) get non-alphanumeric tags.

Lean is weaker here: it sends no tags and retries blindly. Its IsOpen/IsClosed status model is the pattern being adopted.

**Action.**

1. Add one status module:
   - failed = status starts with REJECTED or CANCELLED;
   - terminal = failed or COMPLETE;
   - placed = not failed.
   Use placed() in place_order's retry check (order_service ~:225) and in _execute_live's dedupe. Keep broker.FINAL_STATUSES as the terminal set.
2. place_order. On a retry where kite.orders() raises, return UNKNOWN, do not re-send, and email it as UNKNOWN.
3. _execute_live. Read kite.orders() directly, with 2-3 attempts. If it is still unreadable, place nothing and raise a NOT_SENT alert.
4. Tags. Apply kite_tag(s) = re.sub('[^A-Za-z0-9]', '', s)[:20] inside order_tag, so the dedupe still matches, and also in place_order.
5. Options BasketExecutor._live. On OrderError, look the order up by tag + tradingsymbol + side, because all legs share one tag. Track or cancel any slices found. If the book read fails, report UNKNOWN and stop the basket.

**Test plan.**

Use scratchpad/idem_check/repro.py, with DB, email and sleep patched out.

Before the fix, orders per tag:
- AMO with ReadTimeout: 2.
- AMO with NetworkException: 2.
- Two lost responses: 3.
- Regular OPEN: 1.
- Regular OPEN where the retry-time read fails: 2.
- VALIDATION PENDING: 2.
- PUT ORDER REQ RECEIVED: 2.
- A re-run with one failed book read: the 3-order plan becomes 6 orders.

After the fix:
- exactly 1 order per tag in every case;
- UNKNOWN returned when the retry-time read fails;
- NOT_SENT with 0 orders when the book stays unreadable;
- order_tag('2026-10-09', 'BUY', 'M&M') == 'NE261009BMM';
- all 3,143 symbols of 2025-26 give tags matching ^[A-Za-z0-9]{1,20}$ with no collisions;
- basket: a fake broker that books two slices and then raises leaves both slices tracked, never NOT_FILLED.

The supervised '-' tag check (sell-path item) records whether Kite rejects such a tag, and the exact status strings.

### 4. Apply splits and bonuses (and dividends) to paper lots, pending orders, stops and the live ledger, using the store's own factors (P0, `corporate-actions-in-books`)

Category: live_robustness. Reuse from Lean: port_logic.

**Why.** The backtest works in back-adjusted units, but the books hold as-printed quantities and stops.

Deployed run 679cbd0c, 2013-25:
- about 42 split, bonus or rights events on held names (about 3.2 a year);
- 39 of 41 opened below the stored 6×ATR stop, so the paper book records a fake gap stop and a cooldown;
- the fake loss is a median of about 2% of equity, with a maximum of 5.9% (RAJTV);
- GOLDBEES 1:100 at 35% weight is a fake -34.7% day, which trips the 20% drawdown halt.

Effect on G4:
- a 60-session window has about a 54% chance of containing such an event;
- the deterministic shortfall breaks the -3 bp/day PASS limit in about 35% of windows.

Effect on live:
- min(ledger, broker) marks the old quantity at the post-split price, a phantom loss that feeds the drawdown overlay, the ladder and G4;
- the bonus shares have no stop and sit outside the book;
- the engine then re-buys the apparently underweight name.

Dividends add a 0.25 bp/day bias.

The arithmetic comes from Lean's ApplySplit (SecurityPortfolioManager.cs:819-862) and DefaultBrokerageModel.ApplySplit (153-165).

**Action.**

1. Factor source.
   - Return a per-cell source from adjustment_multipliers. Keep the factor frame and the dividends as non-hashed MarketData attributes, so data_hash is unchanged.
   - share_factors(session) uses only source 1 (ca_ratio) and source 6 (ca_ratio_from_prices). Source 4 (inferred) applies only with an alert. Rights, demerger, prev_close and unexplained factors raise an alert and change no quantity.
   - f_total = the ratio close/close_unadj since the last processed session, used for stops.
   - ETFs are included.
2. Paper. In run_paper_session's once-per-session branch, before the gap stops:
   - lots: quantity = floor(q/f), entry × f, stop × f_total, and cash in lieu at the as-printed close;
   - pending orders: quantity / f, and ref and stop × f;
   - persist the applied (symbol, ex_date) keys in paper state and Neon, so a re-run never applies them twice;
   - credit q × D on dividend ex-dates as fill rows with source 'dividend'.
3. Live, before scoped_book:
   - scale ledger positions and stops once per (symbol, ex_date), recorded in ledger['ca_applied'];
   - for 5 sessions, broker < adjusted ledger means 'awaiting credit': value at the ledger quantity, and cap sell and GTT quantities at the broker's sellable quantity;
   - alert when broker > ledger (silent today);
   - when the GTT is missing, use the re-based stop as the ratchet floor, and skip the never-lowered stops_unknown branch on the ex-date;
   - defer new AMOs in a symbol whose ex-date is the next session;
   - book live dividends as income plus an equal withdrawal flow, as Lean also skips live dividend cash.
4. Never build on services/market_data/corporate_actions.py. It has no ex-date filter, parses 'Split 10 to 2' inverted, and re-applies on every cycle. Fix or retire it separately.
5. Safety net: a close_unadj/close jump below 0.9 on a held name with no store event blocks orders on that name and raises an alert.

**Test plan.**

Use a /tmp SQLite paper book (CENTURION_PAPER_DB_PATH set, DATABASE_URL unset). Temporary tests; afterwards run the core-only fast suite.

1. Paper replays:
   - BSE, 22/23/26 May 2025. Before: GTT_SL_GAP at 2,358 (-63%). After: quantity ×3, stop 1,901.59, no stop event, ex-date return within about 1 bp of the back-adjusted panel.
   - RELIANCE 2017-09-07 (f 0.5) and NATCOPHARM 2015-11-26 (f 0.2).
   - GOLDBEES 2019-12-19 (f 0.01): drawdown state stays NORMAL.
   - GODFRYPHLP 2025-09-16 with a pending rank-exit sell: it sells 3× the old quantity.
   - BAJFINANCE 2025-06-16 with a product factor of 0.1.
   - Running the same session twice applies each event once.
   - A source-2 factor raises an alert only.
2. Live, with a fake Kite. Ledger BSE 44, stop 5,704.78, GTT cancelled; the broker shows 44 on the ex-date and 132 two sessions later. Expect:
   - ledger 132 and stop 1,901.59;
   - no fake equity drop;
   - GTT sized at the sellable quantity;
   - no BUY on a rebalance day;
   - no double scaling on a re-run;
   - an alert if the broker still shows 44 at ex-date + 6.
3. Coverage: events_for over 2013-25 reproduces all 42 held factors.
4. G4 replay of the 2016 window from 16 Sep: daily gap WATCH (7.3 bp/day) before, roughly 0 attributable to events after.
5. Dividend: PTC Rs 5.50 on 2026-10-07 is credited as q × D.

### 5. One EquityInstruments resolver: send BE-series tradingsymbols, map holdings, orders and GTTs back to engine symbols, and carry positions across renames and announced mergers (P0, `instrument-identity-both-directions`)

Category: live_robustness. Reuse from Lean: adopt_pattern.

**Why.** Today's top 300 includes HFCL, MTARTECH, STLTECH and DIACABS as BE-only names. None has a bare instrument, so orders and GTTs on them are rejected. The deployed backtest made 86-89 trades on BE-only days, 8 of them stops.

On a series move (HFCL moved on 3 Sep 2026 while held in the 2026 holdout):
- scoped_book drops the position;
- mark_book removes it from equity, which feeds the drawdown overlay, the ladder and G4;
- reconcile deletes the old GTT as an orphan and places none;
- a GTT sale of SYM-BE is filtered out, so the cash is never credited.

On a rename (8 held in 2013-25; HEG to HEGAM on 22 Sep 2026):
- paper skips the exit for lack of a price and keeps the position at its entry price forever;
- live leaves the shares outside the book with no stop, and may buy a second full position.

The ISIN was kept in 493 of 496 renames. Lean maps by verbatim tradingsymbol and has the same gap; only its one-dump-backed-map pattern is reused.

**Action.**

Use the same dump module as the tick item, filtered to segment NSE and type EQ.

1. Forward, engine symbol to broker.
   - Keep last_series per canonical on MarketData before panel.py:819 drops it.
   - to_broker = the bare symbol for EQ, SYM-BE for BE.
   - The only fallback is bare to SYM-BE, with an alert. Never wildcard other suffixes (-RE rights, -SG/-N0 bonds, -D1).
   - An unresolved symbol blocks the order and raises an alert.
   - Use to_broker in live_orders, unwind, the GTT payload and the reconcile keys.
2. Inbound, broker to engine symbol (to_canonical), in this order:
   - holdings by ISIN against the canonical's ISIN set, which also handles face-value splits;
   - engine order outcomes via placed[tag]['symbol'];
   - order book, positions and GTTs via the inverse map, then an exact series-code strip (BE, BZ, BL, SM, ST, IL, T0), then resolve_symbols.
   Apply this in kite_book, get_held_quantities, list_stop_gtts/_normalise_gtt, the reconcile ltp keys, live_order_outcomes and external_sells.
3. Ledger.
   - Store tradingsymbol and isin per position.
   - When either changes, raise 'series/symbol changed', re-key positions, entries and stops once, and re-arm the GTT on the new instrument before deleting the old one.
   - Refuse a BUY whose ISIN is already held under another key.
4. Paper. Re-key PaperTrader positions, pending orders and stops when a held symbol resolves to a new canonical.
5. Mergers. An AMALGAMATION/MERGER purpose (not DEMERGER) in the Bc file flags the holding.
   - Paper sells at the last close, exactly as engine.py:611-625 does.
   - Live sends a warning email and plans a SELL by the last session before the record date.
   - Exit-offer delistings stay a manual alert, as does a holding that resolves to BZ.

**Test plan.**

Fixture dump: HFCL-BE only, RELIANCE EQ, and decoys MOTHERSON/MOTHERSON-D1 and ABC-RL.

1. Forward:
   - live_orders emits 'HFCL-BE';
   - ABC is blocked, not mapped to ABC-RL;
   - BAJAJ-AUTO, NAM-INDIA and UPL-RE are left untouched.
2. Series move: ledger HFCL 54, broker HFCL-BE 54 with the same ISIN, no GTT. Expect:
   - the position is kept and marked;
   - a GTT is placed on HFCL-BE and nothing is deleted as an orphan;
   - an alert names the series change;
   - an external GTT sale of HFCL-BE is booked to HFCL.
3. Renames:
   - ledger HEG with broker HEGAM is re-keyed, with no BUY of HEGAM;
   - paper ZOMATO seeded on 2025-03-20 and run at 2025-04-09 gives one ETERNAL position, no no_price skip and no duplicate buy (repeat for LTIM to LTM and HEG to HEGAM);
   - INDOSOLAR to WAAREEINDO goes through the change-table fallback.
4. Mergers:
   - paper RANBAXY's exit price equals the backtest's 'delisted' trade;
   - HDFC gets a warning from 2023-07-03 and a SELL by 2023-07-12.
5. Replay over the deployed run's 1,920 holding spans: all 25 series moves, 13 BE-only spans and 8 renames map back, with 0 false matches across the 255 hyphenated symbols.
6. Next dry run: 302/302 resolve, and a read-only log of tradingsymbol, isin and token for held BE names settles how Kite shows them in holdings.

### 6. Reconcile the live ledger after a missed or failed session and block buys until it balances (P0, `missed-session-reconcile`)

Category: live_robustness. Reuse from Lean: adopt_pattern.

**Why.** Kite's order book lasts one day, and the live book learns its fills only from it. A missed manual login is an expected event (U23).

Replay with the real code:
- lost fills leave ledger cash undebited;
- the next plan re-sends the buy, doubling the position with the account's personal cash;
- the first lot stays outside the book with no GTT;
- lost sells and GTT sales keep the symbol in the ledger and never credit the proceeds.

Deployed run: sells happen on 46% of sessions (p95 is 18.7% of equity). The expected permanent phantom loss is 3.05% of equity per missed login, and it feeds the drawdown halt, the ladder and G4.

This follows Lean's pattern of refusing to trade unless the broker state loads (BrokerageSetupHandler.cs:310-318), with a daily cash sync that logs deltas above 2%.

**Action.**

1. Trigger. Set reconcile_needed when any of these holds:
   - the stored decision date is older than the previous store session;
   - a PLACED tag is missing from the order book;
   - outcomes are unknown.
   Carry the unresolved orders in state as live_orders_pending instead of overwriting them (live_session ~:564).
2. For ledger symbols plus pending symbols, compute broker minus ledger quantity and explain each delta in this order:
   - a. Pending BUY tag with a broker excess: adopt it. Price it at the holding's average price for a new symbol, otherwise at the fill-session open capped at the limit. Charge costs, set the entry date and seed the stop from the placed stop price.
   - b. Pending SELL tag with a broker shortfall: adopt it at the fill-session open, floored at the limit.
   - c. A shortfall with triggered-GTT evidence (list_stop_gtts active_only=False): book an external sale at the GTT limit or trigger, otherwise the day's low, and add a cooldown at that date.
   - d. Anything unexplained (a manual sale, a symbol change, a bonus excess, a cash gap above 2%): send exits only that night via the kill-switch path, and email RECONCILE NEEDED.
3. Add 'live_session --reconcile SYM=QTY@PX' to clear an unexplained item by hand.
4. Optional: persist verified postbacks (keep the highest filled quantity per order_id), but only after confirming that Kite posts back AMO fills on days with no login.

**Test plan.**

Extend scratchpad/msr/replay.py, which drives the real run_live_session with a fake Kite and a fake book. Scenario: on D the engine places SELL Y and BUY X; session S is skipped (and Z's GTT fires); at S1 the order book is empty.

Before the fix:
- fills are empty and the ledger is unchanged;
- BUY X 200 is re-sent;
- the GTT scope lacks X;
- recorded equity is 97k too low.

After the fix:
- ledger equals broker for H, X, Y and Z;
- cash is within statutory costs;
- the snapshot is within 0.5% of true equity;
- no BUY X is planned;
- a GTT covers X:200;
- Z is on cooldown dated S.

Variants:
- an unexplained excess blocks all BUYs and raises the alert;
- if orders() raises on S, the orders are carried forward and reconciled on S1;
- a symbol change books no sale, blocks buys and alerts;
- a normal two-session run is identical to today's code.

### 7. Probe every Sunday, backfill the missing 1 Feb 2026 Budget session, make paper catch up across sessions, and keep one shared NSE calendar (P1, `nse-calendar-special-sessions`)

Category: data_integrity. Reuse from Lean: adopt_pattern.

**Why.** The store is missing the Sunday 1 Feb 2026 Budget session: on 2 Feb, 2,223 of 2,404 EQ prev_close values mismatch. It sits inside the holdout and every 2026 run.

On the deployed holdout:
- the 20 names held had an implied 1 Feb move of -6.6% (weight-averaged);
- 5 stop exits filled at the 2 Feb open instead of being tested on the 1 Feb bar.

The 8 Nov Muhurat session falls inside the G4 window and will be missed the same way.

The two live holiday sets already disagree (2026-01-15) and have no 2027 dates.

Lean's India calendar has no weekend sessions at all, so it offers only the pattern of one shared dated calendar; Centurion is already ahead here.

**Action.**

1. session_dates probes every Sunday, as it already does Saturdays. A 404 is recorded in missing.json: about 52 requests a year, about 1,040 once for history.
2. Turn the mass prev_close warning at panel.py:540-548 into a build_store manifest entry and a failing check, with 2006-06-24 allow-listed.
3. Backfill 2026-02-01 locally, in the Actions cache (a one-off --start/--end run) and in the Kaggle dataset. Rebuild the 2026 store year, run refresh-registry --dry-run (user rule), and record the holdout delta as a data-fix re-baseline.
4. Before 8 Nov, run_paper_session must catch up session by session from engine_last_session: gap stops, fills and intraday stops for each session, then plan from the latest close. Otherwise, run a manual workflow_dispatch on Sunday night. Fix cloud_paper_runner's bdate_range missed-session count.
5. Create one nse_engine/nse_calendar.py with holidays and special sessions (kind, plus the NSE circular reference), imported by both carver_pipeline and api/market. Add the 2027 list before go-live. State explicitly whether Muhurat counts as a live session.

Do NOT just add 2026-11-08 to the Sunday list: the Monday run would then cancel Friday's orders as stale.

**Test plan.**

1. Store checks:
   - dates where more than 20% of EQ symbols have a prev_close mismatch fall from 2 to 1 (2006-06-26, allow-listed);
   - calendar.parquet contains 2026-02-01, and the 2 Feb mismatch count is about 0;
   - session_dates includes 2026-02-01 and 2026-11-08.
2. Re-run the deployed config over 2026-01-01..09-11. Report the change in CAGR, Sharpe and MaxDD, and how the 5 stop exits move from the 2 Feb open to the 1 Feb bar. Count affected records with refresh-registry --dry-run.
3. Paper fault injection with Fri, Sun and Mon sessions: Friday's orders fill at the Sunday open instead of being cancelled as stale, and engine_last_session reaches Monday.
4. A mocked unlisted Sunday file is downloaded.
5. Calendar consistency test:
   - every 2025+ weekday that is not a store session is a holiday;
   - carver_pipeline and api/market resolve the same set;
   - a 2027 entry exists.
6. Watch the 8 Nov Muhurat bhavcopy land in the nightly sync.

### 8. Make MarketData.until() keep close_unadj, so trial books plan on the as-printed price filter they were validated with (P1, `until-keeps-close-unadj`)

Category: data_integrity. Reuse from Lean: adopt_pattern.

**Why.** Paper and live plan through data.until(). Today until() drops close_unadj, so the candidate, e4 and the one remaining trial (plan 5b requires price_filter_unadjusted=true) build their universe history on back-adjusted prices while their registered runs use as-printed prices.

Measured:
- 12.5 names differ per session over 2013-25;
- that flows through the expanding normalisers into today's forecasts;
- e4's 2026 tracking error is 1.39%/yr (17% of G4's 8% budget), with different trades from 23 Jan.

The deployed book is unaffected. conftest.truncate keeps the field, which hides the bug from tests.

This matches Lean's pattern of carrying both prices on one object (CoarseFundamental).

**Action.**

1. In nse_engine/types.py:83-98, build until() from a dataclasses.fields loop: dates becomes dates[:n], every DataFrame field becomes iloc[:n], and everything else passes through.
2. Add a regression test to the existing tests/test_universe_lookahead.py:
   - every DataFrame field set on the panel is still set after until(t);
   - with price_filter_unadjusted=True, the universe mask under until() equals the full-panel mask.
3. Correct docs/nse_engine_validation_plan.md:195-196 and docs/nse_engine.md:12-14.
4. Log in the tracker the date the candidate and e4 paper books change behaviour.
5. Drop the proposed nightly parity-check subcommand.

**Test plan.**

Re-run scratchpad/until_verify/measure.py and measure2.py.

Before the fix, candidate and e4:
- 2,709 of 3,220 universe sessions differ;
- forecast difference 0.0252;
- a name swap on 6 Oct;
- e4 2026 tracking error 1.39%/yr.

After the fix, all three books:
- 0/3,220 and 0/188 universe sessions differ;
- forecast difference 0;
- 0 of 15 target sessions differ;
- tracking error 0.00% and identical trade lists.

The new unit test fails on the current until() and passes after. Spot-check a registered run: its returns are unchanged, since run_backtest never calls until(). Next nightly: candidate and e4 plans equal the same-day reference targets.

### 9. Give the live book the backtest's 5-session stop cooldown across sessions (P1, `live-stop-cooldown-parity`)

Category: live_robustness. Reuse from Lean: adopt_pattern.

**Why.** The backtest and paper books block a stopped name for 5 sessions. Live blocks it only on the stop session itself.

With 5-session rebalancing, a rebalance falls inside the window after roughly 4 of every 5 stops. Over 2013-25 that allows about 20 (deployed) and 17 (e4) re-buys the validated backtest never makes, costing 0.67-0.90%/yr of unregistered tracking error before the Rs 6 lakh rung.

This follows Lean's pattern of one book state shared by backtest and live.

**Action.**

1. Persist ledger['recent_stops'] as {symbol: ISO date}, written in apply_fills:
   - external SELL fills (GTT or manual) are dated by their session;
   - FILLED engine SELLs with reason exit:stop are dated by their decision date.
2. Keep 'reason' in LIVE_ORDERS_KEY and in the fill notes.
3. Prune entries older than 30 calendar days. generate_targets already does the session arithmetic.
4. Pass the persisted map as stopped_out, replacing cooldown = {} at live_session ~:405/409/460.
5. When the broker holds less than the ledger with no recorded sale (a missed session), put the symbol on cooldown from the current session.

The entry-date half of the original candidate is already done (LS1).

**Test plan.**

1. Extend the LS1 whole-history replay. The live-built stopped_out must equal the backtest's recent_stops (engine.py ~:672) on all ~3,219 nights.
2. Re-run scratchpad/cooldown_ab/cooldown_ab.py with the live map. Early re-entries go from 20 (deployed) and 17 (e4) to 0, and tracking error goes from 0.67%/0.90% a year to 0.00%.
3. Fake Kite over five sessions:
   - a GTT sale of A on S blocks A on S..S+4 and frees it at S+5;
   - an exit:stop for B is dated S, not S+1;
   - a missed-session shortfall puts B2 on cooldown;
   - the state survives a sync_state round trip.

### 10. Fail closed when the live ledger, the intended orders or the risk history cannot be saved or read (P1, `live-ledger-fail-closed`)

Category: live_robustness. Reuse from Lean: adopt_pattern.

**Why.** Every sync_* call swallows exceptions, and live_session ignores what they return. If the step-6 write fails and no later cron that evening succeeds, the next session looks up the wrong decision date. That night's fills are then lost, which leaves an unprotected orphan lot and a second buy of the same name.

Two more failure modes:
- A ladder move followed by failed ledger writes leaves the rung and the ledger capital out of step.
- A read blip returns an empty history, so the drawdown rule reads 'normal' and a halted book can buy again.

The live write path has never run in production. This follows Lean's pattern of refusing to trade when book state is uncertain.

**Action.**

Live only; paper books and dry runs keep 'never block a run'.

1. Strict-write helper. Wrap sync_fills and sync_snapshot with it.
2. One write-ahead sync_state before ex.execute, holding:
   - the ledger;
   - the ladder state (have _ladder_step return lstate instead of writing it at ~:677);
   - the config marker and the gate;
   - LIVE_ORDERS_KEY with decision date S and status INTENDED.
   If it fails, raise: nothing is sent, the failure email fires, and the backup cron retries.
3. Check the post-execute sync_state. If it fails, raise after sending; the re-run then dedupes by tag.
4. Strict reads on the live path for read_snapshots, read_fills and read_sessions, so an empty history can no longer make the drawdown rule read 'normal'.
5. Step 2 alerts on INTENDED entries that are missing from the order book.

**Test plan.**

Temporary fake-book tests. Each case should behave as follows:
- Write-ahead fails: RuntimeError, execute is never called, the exit code is non-zero.
- Post-execute write fails: it raises after one placement; a re-run gives DUPLICATE, and the place count stays 1.
- No re-run after that failure: S+1 books X and does not buy it again. This fails on today's code.
- read_snapshots raises over a history 25% below its peak: the run raises instead of planning buys.
- A ladder GO move followed by a failed write-ahead: after a re-run, capital equals the rung.

tests/test_paper_book.py passes unchanged. In production, grep the Actions logs for 'live book write failed'.

### 11. Settle whether Kite credits pending sell AMOs, cap buys at the broker's free cash, and read rejections the same night (P1, `amo-funding-and-same-night-rejections`)

Category: live_robustness. Reuse from Lean: adopt_pattern.

**Why.** The model and paper books fund buys with that same open's sale proceeds. Live sends everything as evening AMOs with SELL_PROCEEDS_CREDIT=1.0.

In the deployed run, 342 of 591 buy sessions (58%) and 24% of buy value relied on same-session proceeds.

The dry runs cannot show this, because they start from a fresh book with no sells. An RMS rejection is also only visible 24 hours later.

Lean's buying-power model gives no credit to pending sells; that is the pattern here.

**Action.**

1. Settle the fact once: whether a pending sell AMO funds a buy AMO, and whether funds are checked when the AMO is placed or when it is released at 09:00. Use a Zerodha ticket, or the supervised AMO pair from the sell-path item. Record the answer in the tracker.
2. About 5-10 s after the placement loop, re-read the order book with live_order_outcomes. Alert on REJECTED engine tags, with their status_message, that same night.
3. If Kite gives no credit for pending sells:
   - Set the buy budget to min(ledger cash + credit × net sells, margins live_balance), and cut buys cleanly with a note.
   - Keep a cash buffer outside the ledger, sized from measured reliance on same-session sale proceeds: p90 is 13% of equity and p95 is 29% (Rs 0.8-1.75 lakh at Rs 6 lakh).
   - Alert when ledger cash goes negative.
4. Use SELL_CREDIT=0 only as a fallback: unfunded entries then wait up to 5 sessions (about 0.7 pt/yr).
5. Use two-phase morning buys only if no buffer is possible. They need a login before 09:08 and a reliable scheduler.

**Test plan.**

Fake Kite with live_balance 10k that rejects BUYs once open buy value exceeds it.

Cases:
- Today's code: the order shows PLACED, and the rejection appears only the next day.
- Fixed planner: buys satisfy sum(limit × qty × 1.0025) ≤ live_balance, with a note recording the cut.
- The post-placement read surfaces a REJECTED order in the same session's results.
- With live_balance 200k (a buffer outside the ledger), the full entry is accepted and the ledger cash used stays within ledger cash + sale proceeds.

Re-run /tmp/amo_check/m.py on the deployed and candidate configs to size the buffer. After go-live, track weekly: cut or rejected buys, and planned versus placed buy value.

### 12. Make the nightly dry run check every order and stop against Kite's instrument dump, so a dry run is clean only with zero WOULD_REJECT (P1, `dry-run-broker-preflight`)

Category: live_robustness. Reuse from Lean: adopt_pattern.

**Why.** The 5 clean dry runs that gate go-live never check an order against the broker's instruments. A replay of the deployed config's May-Dec 2025 orders finds 142 of 327 limit prices off-tick, and BE targets go out under the bare symbol.

The detector makes the dry runs unclean until the tick and identity fixes land, and it catches regressions such as NSE's quarterly re-tiering.

The pattern is Lean's CanSubmitOrder, which refuses with a reason, although Lean checks none of these things.

**Action.**

In dry_run_live, fetch the dump once per run. Then check:
- (a) Each tradingsymbol must be an NSE EQ instrument in the dump. If not, mark it WOULD_REJECT, with a hint when SYM-BE exists.
- (b) Each price must be a multiple of its tick (compare in paise). Apply this to orders, each StopInstruction trigger and each stop_limit_price. A failing stop sets the gtt row to success=False.
- (c) If the dump fetch fails, raise an alert. The run is then unclean, but it does not raise.
- (d) A static guard alerts if ORDER_LIMIT_BAND_BPS is 200 or more without the circuit clamp.

In _execute_live, run the same check as advisory only. Never skip a SELL or a GTT on its result; at most skip a BUY that fails it.

Do not add a quote()-based band check to the dry run: the evening band is stale. Do not add a ledger check either, because a dry run has no ledger.

**Test plan.**

Fake dump: RELIANCE (tick 0.10), HFCL-BE only, MRF (tick 5.0), INFY (tick 0.10).

Expect:
- WOULD_REJECT for RELIANCE at 1414.05, for HFCL (with the hint) and for MRF at 140001.0.
- WOULD_PLACE for INFY at 1500.10.
- A RELIANCE stop trigger of 1300.05 gives gtt success=False.
- A dry-run run_live_session shows an 'orders refused' alert and clean=False.
- If instruments() raises: an alert and no exception.
- In live, a failing SELL is still sent.

Replay of the May-Dec 2025 orders: 142 of 327 flagged before the tick fix, 0 after it.

The first nightly dry-run email lists WOULD_REJECTs. D3's clean-run count should advance only once it lists none.

### 13. Nightly expected-statistics canary for the three books on Actions, after fixing the Actions store's missing NIFTY back-fill (P1, `book-regression-canary-and-actions-store-parity`)

Category: validation. Reuse from Lean: adopt_pattern.

**Why.** Nothing catches a code, dependency or runner change that alters a book while its hashes stay the same. IC1 did exactly that today.

Paper and live run on ubuntu with Python 3.11, while every trial ran on Kaggle or the Mac with Python 3.13; the two have never been compared. The shift reference is produced by the same code as the paper book, so the two always move together.

The review also found that the Actions store lacks the NIFTY back-fill. Its 2013-25 data hash differs (simulated: 47 NIFTY sessions missing), although 2026 decisions are unaffected.

The pattern is Lean's regression algorithms with ExpectedStatistics.

**Action.**

1. Store parity. Put external/nifty50_history.parquet (3,376 bytes) on the runner. Either commit a frozen copy and copy it into place in 'Build NSE store', or run nse_engine.data.external nifty50 there.
2. Canary subcommand, run after the store build and before the live install:
   - load 2012-01-02..2025-12-31 once;
   - run the three engine configs with record=False;
   - compare against config/canary_expected.json: config_hash, data_hash, cost_model, n_trades, the sha256 of the sorted trade list, Sharpe/CAGR/MaxDD to 1e-9, and final equity to Rs 0.01;
   - print the runtime provenance.
   Today's expected values (cost model 4, data hash ba7c098240b4c9ec):
   - 679cbd0c: 6,087 trades, Sharpe 1.1580110002, Rs 7,347,660.79;
   - 2d64ba4c: 6,408 trades, Sharpe 1.1317745944;
   - 93cf6c4d: 5,713 trades, Sharpe 1.1613415497.
3. Two tiers. The registry tier is for information. The Actions tier is recorded by the first workflow_dispatch and is the one that raises alarms.
4. Outcomes:
   - Data hash differs: email; never run refresh-registry on Actions.
   - Same hashes but different outputs: email and set the step output drift=true. Whether drift forces a live --dry-run is the user's call, through a repo variable (a dry run also withholds exits).
5. Any deliberate change to outputs updates the JSON in the same commit, with a tracker entry. Extend test_config_identity.py to check the JSON's hashes.

**Test plan.**

Local fault injection:
- Patch engine.IDLE_CASH_YIELD_ANNUAL to 0.06. 679cbd0c goes back to about 6,151 trades, and the canary reports drift with drift=true.
- Hide read_index_history. The canary reports 'data differs' with hash c5a71e81, does not call refresh-registry and does not block live.
- A malformed expected file fails loudly.

On Actions:
- Run one workflow_dispatch in dry-run mode before and after the NIFTY fix. The hash should differ from ba7c098240b4c9ec before and equal it after; then compare outputs with the registry tier to 1e-9.
- Two consecutive nights should give identical outputs.

### 14. Live exit and stop price protection: GTT buffer 2% with a circuit floor, wider circuit-clamped exit sells, and carry-forward of unfilled sells (P1, `live-exit-price-protection`)

Category: execution_realism. Reuse from Lean: adopt_pattern.

**Why.** The backtest and paper books fill exits and gap stops at the open. Live uses a LIMIT 1% under the close and a GTT LIMIT 1% under the trigger.

Gap stops, 2013-25:
- 10 of 41 never fill at 1%;
- 7 of those opened locked at the lower circuit, which no buffer can fix;
- the other 3 (DELTACORP, TITAN, BERGEPAINT) fill at 2%, which also keeps most of the gain from fills above a gapped open.

De-risking sells:
- On 24 Aug 2015, all 19 sells (about 42% of equity) would go unfilled at 100 bp.
- Non-rebalance days keep drifted weights, so a missed sell can wait up to 5 sessions.

Widening limits without a clamp would put them outside the price band of 2% and 5% band names, and the exchange rejects those outright. So the clamp is a precondition for any wider band.

Lean's stop-market gap fill is the behaviour the backtest assumes.

**Action.**

1. GTT stops.
   - Change the default buffer from 1% to 2% in code (gtt_stops.py:38).
   - New limit = max(round_down(trigger × 0.98), round_up(next lower circuit)).
   - Take the band from the evening kite.quote as (previous close − lower circuit) / previous close. Cap it at 10% for F&O names, whose band resets daily.
   - Apply the floor only when the trigger is above the next lower circuit.
   - The existing limit comparison re-modifies the GTT every night.
   - If the quote fails, use today's rule and raise an alert.
2. Reduce-only SELL AMOs (engine exits and unwind_orders).
   - Add EXIT_LIMIT_BAND_BPS, default 500, clamped to the next lower circuit plus one tick.
   - Fall back to 1.9% when the quote fails.
   - BUYs stay at 100 bp.
3. Carry-forward. If yesterday's engine SELL is CANCELLED or partly filled and the ledger still holds the stock (and it is not a fresh target), re-send the residual next evening at the new band, under a new tag.
4. Show limit prices and clamps in the dry-run and live emails.
5. Before fixing the defaults, re-run both replays below. Stay at 2% for the GTT and do not use 3% for sells.

**Test plan.**

1. Replays. First reproduce run 20261009T185910593890Z_679cbd0c exactly: CAGR 22.97%, Sharpe 1.158, 6,087 trades.
   - scratchpad/gtt_buffer_check.py: misses that are not lower-circuit locks go from 3 to 0 at a 2% buffer, and net versus the model stays at least +0.7% of equity.
   - scratchpad/exitband_verify/replay2.py, clamped, at 1/2/3/5%: net at least 0, MaxDD within 0.1 pt of the model, and no limit below a lower circuit. Repeat on store_ext2006.
   - scratchpad/lmtv/crash.py: missed sells on 24 Aug 2015 fall from 19 to 6 or fewer.
2. Fake Kite that rejects any order below lower_circuit_limit:
   - SELL limit = close × 0.95 for F&O names; clamped for 2% and 5% band names;
   - BUY limit = close × 1.01;
   - quote failure falls back to 1.9%;
   - GTT limit = max(trigger × 0.98, lower circuit + tick), and a band move triggers modify_gtt;
   - carry-forward re-sends the residual, but not when the stock was sold outside the engine or is a fresh target.
3. After go-live: compare live_order_outcomes misses with the predicted outcome from the next bhavcopy.

### 15. Break forecast ties by the name in force on the decision date, and ship a rename-invariant, widened data hash in the same re-baseline (P1, `pit-name-tiebreak-and-fingerprint`)

Category: data_integrity. Reuse from Lean: adopt_pattern.

**Why.** Forecasts are capped at 20, and ties are broken alphabetically by the LATEST name:
- on 26% of sessions more than 20 universe names sit at the cap;
- on 169 of 644 rebalance days the rank-20/21 boundary falls inside a tie;
- using point-in-time names changes the top 20 on 34 rebalance days.

Renames in 2026, about 5 a month (ETERNAL, VIYASH), already change 2013-25 decisions: a pure relabel changes 26 of 61 registry configs. Under cost model 4:
- deployed Sharpe 1.158 becomes 1.163;
- candidate Sharpe 1.151 becomes 1.180, enough to sway the candidate-vs-deployed call.

The G4 reference re-runs on the latest store, so a rename shows up there as false tracking error. Live already ranks by today's names.

The pattern is Lean's GetMappedSymbol(date).

**Action.**

1. In panel.py, after canonicalise, build trade_names as {canonical: [(first_date, as_traded_name)]} for the 352 renamed columns.
2. Add MarketData.trade_names and include it in compute_hash when non-empty.
3. In engine.py ~:239, sort by (-forecast, name in force that day, latest-name rank), using segments forward-filled in EngineCache.
4. In the same release:
   - key the hash on each column's name as of the panel's last date, plus the in-window change-table rows;
   - widen the hash to opens, highs/lows, close_unadj, volume and the ETF set;
   - record a hash version in the manifest.
   Ship the hash change only after the relabel gate passes (0 of 61 configs differ).
5. Run one Kaggle refresh-registry over the ~61 same-window configs (RR1 pattern). Report each config's Sharpe change, the Spearman rank correlation, and PBO/DSR before and after.

Ship this as a data re-baseline, not a config flag: a flag changes the config hash and would spend the trial. Adopt it on principle, not on its result.

**Test plan.**

1. Synthetic unit test: two capped names tie, and one is renamed after as_of. The output must not change, wherever the rename date falls.
2. Append a synthetic rename dated 2026-12-01 (ETERNAL to ZZETERNAL). After the fix the 2013-25 returns are identical (max diff 0); before it they differ.
3. Relabel gate (rename_probe3.py): 26 of 61 configs differ today; after the fix, 0 of 61.
4. Hash checks:
   - the full change table and the table cut at 2025-12-31 give the same hash (today ba7c0982 vs 046ef77e);
   - changing one open, high or close_unadj changes the hash.
5. Live parity from 2026-09-16 to today: generate_targets is unchanged and matches the recorded paper orders.
6. The next rebuild that includes a real rename reports 'unchanged' for every config.

### 16. Rank CSCV PBO trials on excess returns, consistent with DSR and walk-forward selection (P1, `pbo-excess-basis`)

Category: validation. Reuse from Lean: adopt_pattern.

**Why.** PBO is Centurion's only Sharpe-family statistic computed on raw returns, while selection, DSR and the plan's gates all use excess Sharpe.

Under cost model 4:
- raw basis: PBO 60.4% ('reject'), probability of OOS loss 0%;
- excess basis: PBO 37.2% ('caution'), probability of OOS loss 0.86%.

The D2 haircut mixes the two bases. On the excess basis the deployed book's expected OOS Sharpe is 0.78, below the plan's 0.8 line. This changes how the one remaining trial will be judged.

The pattern is Lean putting every Sharpe-family statistic on one rf basis.

**Action.**

1. Add rf_annual and periods_per_year to cscv_pbo, and subtract the daily risk-free rate before the block sums.
2. Return rf_annual and return_basis in the result.
3. Pass each call site the rf it already uses for its DSR:
   - run_nse_engine.py :200 and :268;
   - scorecard.py :448, with rf added to the cache key at :442;
   - diagnostics.py :223;
   - options backtest.py :403 (BOOK_RF).
4. Pass an explicit 0.0 at backtest.py:351 and signal_futures.py:418, whose returns are already in excess of cash.
5. Recompute validation.json for the three books and update the plan and tracker figures. Print the legacy raw PBO beside the new one for one cycle.
6. Flag U27: the <30% target fails on both bases.

**Test plan.**

1. Identity: cscv_pbo(m, rf=0.065) == cscv_pbo(m - 0.065/252) on every key. The default rf=0 reproduces today's output bit for bit (0.6037296037296037).
2. A constructed A/B volatility case flips which trial is best in-sample.
3. validate on 20261009T185910593890Z_679cbd0c gives pbo 0.3715, prob_oos_loss 0.0086, verdict caution, and a haircut of x0.80 (expected OOS Sharpe 0.78). On the cost model 3 run it gives 0.4863.
4. The options-family PBO (22.2% over 10) reproduces exactly.
5. refresh-registry --dry-run shows before == after.

### 17. Report rejected stop and exit sells the same evening, with Kite's reason and a fix hint (P2, `rejected-sell-classifier`)

Category: live_robustness. Reuse from Lean: adopt_pattern.

**Why.** Today a rejected GTT sell shows up only as a generic 'gone' line. Kite's reason arrives a session later, through the engine's own exit, which fails the same way when DDPI is missing.

In the deployed backtest, 8 of 214 stops fell on the same day as a trim, so the GTT would have asked for more shares than were held.

This is the durable complement to the one-off DDPI attestation, since DDPI can lapse. The pattern is Lean's: every rejection becomes a message carrying the broker's own text.

**Action.**

Add failed_stop_sells:
- Call list_stop_gtts(active_only=False) for ledger symbols with status triggered, rejected or disabled.
- Follow each order_result.order_id into today's order book.
- A failure is any of: REJECTED; CANCELLED with filled quantity below the order quantity; order_result 'failed'.
- Alert only when the broker still holds the symbol after today's fills. This suppresses orphan GTTs that fired after the AMO exit had already filled.

The alert is one line per symbol: 'STOP SELL NOT DONE: SYM xQTY trigger T - Kite <status>: <raw message>', plus a hint:
- authoris/edis/tpin/ddpi/poa: authorise via CDSL TPIN and enable DDPI.
- holding/quantity/insufficient: GTT quantity exceeds the holding.
- circuit/band/price: price outside the band.

This line replaces the generic LS1 'triggered without a sale' line for that symbol. Append the same hint to the notes on the engine's own rejected orders. Also alert on untagged REJECTED CNC sells of ledger symbols.

**Test plan.**

Fake Kite cases:
- (a) Authorisation-text rejection: exactly one alert, with the symbol, quantity, raw text and 'DDPI', and no generic line.
- (b) order_result 'failed' with 'Insufficient holdings': quantity hint.
- (c) A CANCELLED limit with 0 filled: alert.
- (d) A COMPLETE GTT sell: external_sells output identical to today's.
- (e) A rejected GTT on a name whose AMO exit filled (broker holds 0): no alert.
- (f) No triggered GTTs: report['alerts'] identical before and after.

One read-only dry run after login confirms that get_gtts returns triggered GTTs with orders[0].result.order_result.

### 18. Scorecard regime split: label each day with the regime known at the previous close (P2, `regime-split-ex-ante`)

Category: validation. Reuse from Lean: adopt_pattern.

**Why.** Each day's return is currently labelled with that same day's close. Down days that flip the regime are therefore pulled into 'bear' and 'VIX elevated'.

The published tables say the book loses about 22% a year in bear phases. Measured ex ante, it is roughly flat. That reverses the case for a bear-regime hedge (O5, U28).

Lean's crisis tables condition only on dates fixed in advance; that is the pattern followed here.

**Action.**

In scorecard.regime_stability (~:501-526), shift both the trend label and the VIX label by one session on the index calendar, then reindex to the returns. Update the docstring and headings.

Drop the same-day split, or keep it only as a table explicitly labelled 'contemporaneous'.

Re-render the 2026-10-10 scorecards and their JSON, and correct the regime sentence of the SC1 entry at ACTION_TRACKER.txt:554.

**Test plan.**

Synthetic unit test, added to the existing tests/test_metrics.py: a crash on day t must not change day t's label, and the new function must equal the old one applied to ic.shift(1).

Then run scorecard --book all for deployed. Expected changes:
- bear: -23.7% / Sharpe -2.07 becomes +6.2% / -0.02;
- bull: 2.51 becomes 1.79;
- VIX elevated: Sharpe -1.64 becomes 0.90.

Shares of days must stay within 0.1 pt, and every other section must be byte-identical.

### 19. Stop a single missing NIFTY or sleeve print from blanking the 200-day means; fix the archive rule and add a nightly presence alert (P2, `nightly-index-and-sleeve-presence`)

Category: data_integrity. Reuse from Lean: adopt_pattern.

**Why.** A rolling(200, min_periods=200) mean goes blank for 200 sessions after one missing print. Two effects:
- NIFTY: the candidate book can never reach risk_on (it stays at 0.6 exposure), and the deployed and e4 books can only go risk-off through VIX at 35 or above.
- Sleeve: one missing GOLDBEES print sells the whole sleeve, which then stays out.

The archive rule that created the 12 NIFTY gaps of 2013-16 is still in place, and the back-fill cache ends on 2016-06-20.

There is no impact today. The pattern is Lean's sample-based SMA.

**Action.**

1. Archive (archive.py:356-358, 372-376). Do not record an indices 404 as a holiday when the equity file for that day exists. Retry it within the 14-day lookback and warn.
2. regime.py:101-102. Compute the NIFTY trend mean over the last 200 available closes and carry the trend over a day with no print. An alternative: fill interior NIFTY50 gaps of 5 sessions or fewer after the cache back-fill, and log it.
3. sleeves.py:52-61. Compute the sleeve mean over valid closes and carry in_trend over a day with no print.
4. Leave breadth unchanged.
5. EngineExecutor.plan. Check that NIFTY50 has a close for as_of, has no NaN in the trailing 200 sessions, and that every enabled sleeve has a close. On failure, add a note and send an email, but still plan.

**Test plan.**

1. Re-run 679cbd0c, 2d64ba4c and 93cf6c4d unrecorded: identical trades and metrics, and an unchanged data hash.
2. One NaN in NIFTY on day t: only day t is affected, and its trend equals day t-1's.
3. One NaN in GOLDBEES: no EXIT_SLEEVE_TREND, and in_trend is restored at t+1.
4. MON100 documents its 290 changed sessions.
5. Mocked fetch returning 200 for equity and 404 for indices on a date more than 3 days old: the date is not recorded as missing.
6. Fault injection on a /tmp store copy: an alert note, the email mock called, the plan still runs, and no sleeve sell.

### 20. One label-to-contract settlement map used by every options backtest, plus a guard against using the settle column as a price (P2, `options-canonical-settlement`)

Category: options. Reuse from Lean: adopt_pattern.

**Why.** NSE relabels contracts in the bhavcopy. As a result:
- O2 skips 2-3 cycles per sleeve and records a phantom -26.3% day for A2 on 30 Jun 2023;
- X3 and X4 carry stale marks;
- Y2 settles June 2023 a day early.

This is a prerequisite for any O5 weekly round. The pattern is Lean's single settlement source per contract; Lean's own India expiry rule is wrong.

**Action.**

1. Extend settlement_sessions in place:
   - follow a relabel when a same-month label starts the very next session (latest-settling candidate, applied recursively);
   - drop contracts still trading at the data end.
2. Wire the map into three places:
   - backtest._Market: re-key rows to the settlement session, then take the month's highest re-keyed label;
   - signal_futures.run_strategy: roll on the settlement session;
   - fo_anomalies.monthly_expiries.
3. Settle-column guard: an untraded row on its contract's settlement session is valued at intrinsic against the exact index close, never the settle column (which has equalled the index level since 2020).
4. Correct validation_plan:1293 and the 'round 2 unaffected' line.
5. Re-run O2 A1/A2/B, O3 X3/X4 and O4 Y2 on Kaggle; confirm X1, X2 and Y1 reproduce. Record it as a data re-baseline: same hashes, N unchanged, following the 9 Oct precedent.

**Test plan.**

1. Label mappings:
   - 2023-06-29 → 06-28, 2014-04-24 → 04-23, 2008-11-27 → 11-28, 2014-02-27 → 02-26, 2018-03-29 → 03-28, 2023-03-30 → 03-29, 2025-09-25 → 09-30;
   - 2026-03-26 and 2026-03-31 → 03-30;
   - 2026-10-13 absent;
   - all other labels unchanged, and no duplicate rows.
2. Re-run expectations:
   - A2: Sharpe ≈ -0.183; 30 Jun 2023 return = 0; 28 Jun ≈ -3.49%.
   - A1: Sharpe ≈ 0.828, DSR ≈ 0.892.
   - B ≈ 0.259; X3 ≈ 0.494; X4 ≈ 0.144.
   - Y2's June 2023 settles at 18,972.10.
   - Every gate-1 verdict unchanged.

### 21. Model the orders live actually sends: observational limit and GTT outcomes in paper now, a hash-neutral live-rule reference, and DailyLimitFill only inside X1 (P2, `live-order-rule-reference`)

Category: execution_realism. Reuse from Lean: port_logic.

**Why.** Live sends ±1% LIMIT AMOs and GTT stop-limits, while the model and paper fill at the open. Over 2013-25, about 10% of buys and 7% of sells would not fill at the open, and 1.5-2% would not fill that day.

The order rules alone, with no slippage, put G4's daily gap at FAIL in 74 rolling windows, and any FAIL steps the ladder down.

G4's cost ratio is biased lenient: a later limit fill is booked as negative impact, and a cancelled order costs nothing.

The ported logic is Lean's daily-bar LimitFill.

**Action.**

1. Phase 1, October, bit-identical. Paper PENDING rows carry the order kind and limit, and stops carry their trigger and stop_limit_price. Fills stay as they are today, but each order also records would_fill_live (open, at_limit or unfilled) from the session's OHLC.
2. Phase 2, a hash-neutral reference.
   - Add run_backtest(fill_rule='open'|'live'), kept outside the config hash. 'live' with record=True is refused.
   - Add pure functions:
     - LimitFill, ported from Lean EquityFillModel.cs:430-468: strict penetration, with as-printed limits.
     - Centurion's own GTT rule: open between limit and trigger fills at the open; open below the limit fills at the limit if high > limit, otherwise the order carries to the next day; an intraday trigger fills at the trigger.
   - Band and buffer are read from the live constants.
   - Show the live-rule reference's tracking error and gap beside G4. Keep the pre-registered G4 reference unless the user changes it before D3's first session.
   - X1 and SC1b measure impact against this reference and report the opportunity cost of unfilled orders, split into ETFs and shares.
3. Phase 3, inside X1 after 30 or more real fills. DailyLimitFill becomes cost model 5, with a Kaggle re-baseline. Treat any upward move with suspicion.

**Test plan.**

1. fill_rule='open' is byte-identical to the current code: 6,087 trades, Sharpe 1.1580, config hash unchanged.
2. record=True with fill_rule='live' raises.
3. Synthetic bars:
   - a gap above the band, with and without a later touch of the limit;
   - strict open == limit;
   - the mirrored cases for sells;
   - each GTT variant;
   - an ex-date sell left unfilled.
4. The deployed replay lands near scratchpad/lrv/replay_rules.py: 6,024 trades, Sharpe 1.1475, tracking error 1.30%/yr, daily-gap FAIL in 74 and WATCH in 58 of 3,161 windows.
5. Phase 1: replaying the last 10 paper sessions gives identical fills, with would_fill_live populated. The classifier reproduces BUY 89.6/8.8/1.57% and SELL 92.6/5.5/1.88%.
6. Mocked fills_from_outcomes fixtures for an open fill, a later limit fill and a cancel.

### 22. Remove hash-seed dependence so reruns are bit-identical (P3, `deterministic-set-order`)

Category: validation. Reuse from Lean: adopt_pattern.

**Why.** Today trades.csv row order and the last bits of equity vary with the per-process hash seed. That hides real changes behind row-order noise, and the canary has to sort trades to compare them.

The fix has no economic effect. It is cheap hygiene.

**Action.**

Change two loops to iterate in sorted order:
- portfolio.py:221: sorted(set(target) | set(current))
- engine.py:561 (_projected_holdings): sorted(...)

Keep the 1e-9 and 1e-12 tolerances, and do not use PYTHONHASHSEED. Optionally reword 'bit-identical' in kaggle_research.md:79 to 'within one job'.

**Test plan.**

Run the probe script under PYTHONHASHSEED 1, 2 and 3, at lag 0 and lag 1. The pass criteria:
- the equity bytes and the raw trades.csv are identical across seeds;
- the sorted trade set is unchanged (6,087 trades at lag 0, 6,271 at lag 1);
- CAGR is bit-identical;
- refresh-registry still reports 'identical' at 1e-9.

### 23. Capacity: name the assets that bind first and value-weight the over-cap share; fix the 'cut by the 5% cap' mislabel (P3, `capacity-binding-and-cap-label`)

Category: validation. Reuse from Lean: adopt_pattern.

**Why.** The scorecard says the 5% cap 'cuts' 5.3% of fills. In fact it cut 1; the other 322 were buys scaled down for cash.

The headline also hides that sleeve rebalances bind first, at about Rs 6.4 crore, against a ladder ceiling of Rs 30 lakh.

This follows Lean's LowestCapacityAsset idea.

**Action.**

In scorecard.capacity:
- add cap_capital = max_participation × ADV / weight, plus sleeve/core labels;
- report first_binding (the 5 lowest), the first core asset to bind, value-weighted quantiles, and a value-weighted share beside each rung;
- print one decimal, so 0.42% no longer shows as 0%;
- keep 'impact eats half the edge' as the headline, with one sentence underneath on the first binding assets.

Split the trading line into two counts:
- cut by the participation cap: 1 of 6,087;
- scaled down for lack of cash: 322.

**Test plan.**

Deployed run 20261009T185910593890Z_679cbd0c should give:
- SILVERBEES binds first, at Rs 6.36 cr;
- GOLDBEES at Rs 7.39 cr;
- ACUTAAS is the first core asset to bind, at Rs 42.09 cr;
- value share over the cap: 2.94% at Rs 10 cr, 52.0% at Rs 293 cr.

Also run a synthetic two-fill check. All other sections must stay unchanged.

### 24. Report the Sharpe standard error, PSR against 1.2 and MinTRL beside the pass rules (P3, `psr-and-sharpe-se-reporting`)

Category: validation. Reuse from Lean: adopt_pattern.

**Why.** Both '1.2' Sharpe rules are being decided inside the estimate's noise (deployed 1.16 ± 0.29). Showing PSR makes that visible ahead of the one remaining trial.

Lean's PSR is the reference, but Lean has no MinTRL.

**Action.**

1. In overfitting() and walk_forward_oos(), report:
   - the standard error of the annual Sharpe;
   - PSR against 1.2, the pass-rule target;
   - MinTRL against 0, in sessions and in years.
2. Add the paper Sharpe's standard error to paper_section().
3. These are reported only; they never decide a pass.
4. Leave the trade ratios unnormalised and keep the 'kurtosis' key name. A comment at metrics.py:117 is enough.

**Test plan.**

1. Deployed: PSR(1.2) 0.442, standard error 0.290, MinTRL 546 sessions, and PSR(0) equal to validation.json.
2. Candidate: PSR(1.2) 0.407. E4: 0.447.
3. Walk-forward: r12a4 0.531, r12b4 0.262.
4. On a synthetic iid series, the standard error equals sqrt((1+SR²/2)/(T−1)).

### 25. Drawdown periods (depth, duration, recovery) and the book's behaviour across NIFTY TRI's own drawdown episodes (P3, `drawdown-periods-tri-episodes`)

Category: validation. Reuse from Lean: port_logic.

**Why.** No code computes drawdown length today. The hand-written tracker figure has already drifted (726 vs 748 days).

A rebound-capture view also matters for decision U22.

The drawdown logic is ported from Lean's DrawdownCollection, with its bugs fixed: an open final drawdown is kept, and episodes are not merged by peak value.

**Action.**

1. Add metrics.drawdown_periods:
   - key each episode by a counter of new highs, not by the peak value;
   - count a return to equal the peak as a recovery;
   - keep the drawdown still open at the end.
2. Show in scorecard.return_risk:
   - longest time under water;
   - share of days under water;
   - age of the open drawdown;
   - top 5 periods.
3. Add a descriptive table of NIFTY 50 TRI's top-N drawdowns, with the book's fall leg and recovery leg for each. Hand-picked crisis windows are not used.
4. Refresh the stale '726 days' figure in ACTION_TRACKER.
5. Do not add any of this to G4 or the ladder.

**Test plan.**

1. Golden curve [100, 110, 99, 104, 110, 121, 108.9, 115]: expect two periods, the second still open. Two equal peaks must stay separate.
2. Deployed cost-model-4 run:
   - longest under water 28 Aug 2018 to 14 Sep 2020, 748 days, -23.9%;
   - 86.3% of days under water;
   - fall over the 2020 TRI episode -10.2% against -38.3%.
3. The cost-model-3 run gives 710 days.

### 26. Stamp the runtime in every run manifest and warn when a trial set mixes OS or architecture (P3, `run-provenance-in-registry`)

Category: validation. Reuse from Lean: adopt_pattern.

**Why.** Mac and Kaggle runs of the same configuration and window diverge (Sharpe 0.697 vs 0.706, for example). The docs forbid mixing platforms, but no code can check it.

This matters if the one remaining trial runs on a different platform from the 61 runs it is judged against.

The pattern is Lean's single labelled runtime.

**Action.**

1. Write manifest['runtime'] from both record_run and record_result. It carries the provenance() fields plus scipy, system, machine, and origin (local, kaggle, actions or hf_space). It is not added to config.json.
2. list_trials derives an environment class from origin, system and machine.
3. returns_matrix gets environment=None and warns when:
   - the selected trials span more than one class;
   - dedupe kept a run from a different class than the duplicate it replaced.
4. Backfill the history in a sidecar index (from kaggle_out) without rewriting manifests.
5. Correct kaggle_research.md:86-88.
6. Bundle with IC2.

**Test plan.**

1. A toy run's manifest carries the runtime stamp, and config_hash is unchanged.
2. Origin detection works through environment variables.
3. A mixed set warns, and can be filtered.
4. The classifier reproduces today's registry:
   - 5,558 manifests: 2,925 local and 2,633 Kaggle, with 0 unmapped;
   - 297 mixed keys, all under cost model 1.
5. Cost-model-4 validate output is unchanged and raises no warning.

### 27. Options toolkit: give each ITM leg its OTM mirror's IV (Lean's smoothing rule), so monitor Greeks and the volatility stop are never lost (P3, `forward-iv-otm-mirror`)

Category: options. Reuse from Lean: port_logic.

**Why.** With spot pricing and q = 0, 23% of puts 2-4% ITM and 43% of puts 4-7% ITM get no IV. The monitor then drops all Greeks and skips the volatility stop.

The toolkit is paper/manual only, so this is low urgency. It is a prerequisite for any O5 options round.

**Action.**

1. In build_chain, add iv_strike: the OTM side's IV for both options at a strike, falling back to the mirror when a contract's own IV fails to solve. Add an iv_source column. pretrade reads iv_strike.
2. position_monitor:
   - quote each leg's mirror;
   - solve in this order: own IV, mirror IV, then entry ATM IV, flagging the result as approximate;
   - never set Greeks to None while spot and a price exist.
3. Optional, live chain only: a per-expiry forward taken from the future, or from put-call parity at the strike with the smallest |C-P|, then priced Black-76 with spot=F and dividend=r.
4. Leave iv_history and IV_VERSION unchanged.

**Test plan.**

1. Fake-Kite weekly chain priced on F = S·e^((r+1.5%)T). Use a bull put spread whose short put is 3% ITM.
   - Today: greeks are None and no volatility-stop alert is raised.
   - After the fix: greeks are present and the stop is evaluated.
2. The Black-76 path recovers σ within 1e-6.
3. The CONCEPTS M5.21.x tests pass unchanged.
4. Store coverage: the share of legs more than 2% ITM with no IV falls to about 3%.

### 28. Document the missing 2006-09 corporate actions and dividends in store_ext2006; optionally import the ~17 missed events (P3, `pre2010-corporate-action-caveat`)

Category: data_integrity. Reuse from Lean: centurion_only (Lean contributes no India factor data).

**Why.** Only the 2007-25 crash-window runs (R4, R11, the 5i gate) are affected, and none flips. The point is to document the bias honestly; it affects neither live trading nor the 2013-25 window.

**Action.**

1. Now: add a dated caveat to plan section 5e and the K4 entry.
   - store_ext2006 has no Bc corporate-action or dividend records before 2010-01-04.
   - About 17 non-circuit bonus or demerger gaps stay unadjusted (ONGC 2006-10-27, JPASSOCIAT 2009-12-17, DABUR, CROMPGREAV, RELIANCE 2006-01-18).
   - 2008 drawdowns read about 1 pt too deep.
   - Same-data A/B comparisons are unaffected.
2. Optional, only after confirming a source that covers 2006-09 including delisted names:
   - import the events in Bc schema into corpact/2006-2009 (or a hand-verified CSV of the ~17);
   - rebuild store_ext2006;
   - run refresh-registry --dry-run.
3. Do not lower the 35% inference threshold.

**Test plan.**

After an import:
- ONGC on 2006-10-27 and JPASSOCIAT on 2009-12-17 show adjusted moves within ±5%;
- non-circuit ratio gaps in 2006-09 fall from 19 to about 2;
- counts at the 0.8 and 0.909 circuit levels barely change;
- dividends reach about 1,000 rows a year;
- re-running R4, R11 and the four 2007-25 runs leaves every verdict unchanged.

### 29. After go-live, consider sweeping idle cash into a growth liquid ETF with hysteresis (would be cost model 5) (P3, `idle-cash-liquid-etf-sweep`)

Category: other. Reuse from Lean: centurion_only (Lean credits nothing on idle cash).

**Why.** IC1 removed the backtest's 6% idle-cash credit, so no consistency gap remains.

The sweep is a modest treasury gain: about +0.4 pt CAGR pre-tax, about +0.3 pt after slab-rate tax. A naive daily sweep without hysteresis earns only about +0.15-0.34 pt.

It adds new failure modes, so it must not land before the mid-December go-live.

**Action.**

Post go-live only, and after the cost-model-4 re-score.

1. Instrument: a growth-type liquid ETF (LIQUIDCASE, CASHIETF or LIQUIDADD), not LIQUIDBEES.
2. Ledger: keep the holding in a separate 'sweep' field, and have scoped_book count it as cash.
3. Hysteresis:
   - sweep in only when idle cash exceeds the sweep plus a 2% buffer by 2-5% of the book;
   - redeem only what tonight's buys need, as the first SELL.
4. Backtest, paper and G4: model the same sweep identically in engine, benchmarks and paper as cost model 5, using a dated overnight/TREPS yield net of TER.

**Test plan.**

1. Replay cost model 5 against cost model 4 on 679cbd0c, E4 and the candidate: CAGR change within ±0.1 pt of the estimate, sweep costs 15-20 bp/yr.
2. Candidate paper rehearsal: daily P&L equals quantity × the ETF's close-to-close change.
3. Fault injection: a rejected sweep sell scales buys down to real cash, with an alert.

## Not adopted

- **Run Lean as the backtest engine (lean CLI + Docker), as a second reference engine, or inside Python through pythonnet/QuantBook.** Lean needs .NET 10 and a CPython 3.11 build pinned to pandas 2.3.3 and numpy 1.26.4. Centurion runs pandas 3.0.6 and numpy 2.4.6, and Lean's PandasMapper monkey-patches pandas indexing for the whole process.

The Docker image is 14 GB, and the CLI requires a paid QuantConnect organisation. None of this runs on Kaggle (Python 3.13, no .NET, no internet), on the HF Space, or within the Actions disk budget.

Lean also has no NSE history or point-in-time universe. Swapping engines would re-baseline every trial and spend the one remaining.
- **Backtest or trade through QuantConnect's cloud REST API.** QuantConnect has no India equity history for backtesting, and live trading needs a paid tier. The Kite token would go to a third party, and results would sit outside Centurion's trial registry.
- **Use Lean's Zerodha brokerage plugin for live execution.** It is weaker than Centurion's Kite path:
- DAY SL/SL-M orders only, with no GTT and no AMO;
- no proxy, a static token and a default MIS product;
- tradingsymbols used verbatim, and tick_size parsed but never used;
- blind 5x HTTP retries and no tags.

It also posts machine identifiers to a QuantConnect licence endpoint and exits unless a paid module licence is held.
- **Lean's ZerodhaFeeModel / IndiaFeeModel.** It uses intraday (MIS) rates on a static schedule for every product, with no DP charge and no dates. That understates CNC round-trip costs by 60-77% (10.7 bp against 27.5 bp on Rs 30,000) and F&O costs by about 85%. Centurion's cost model is dated and specific to each product.
- **Lean's slippage models (Null, VolumeShare, MarketImpact) and volatility-scaled impact.** Null is zero. VolumeShare comes to about 0.04 bp on Centurion's fills. MarketImpact uses a 75 ms horizon with random draws, about 93 bp at the median order.

Adding a sigma×sqrt term to Centurion's model without real fill evidence would be a strategy change that spends the trial; revisit it inside X1.
- **Lean's settlement and buying-power models, MOO/MOC orders and trailing-stop orders.** Zerodha lets executed CNC sale proceeds fund buys the same day, which Lean's T+1 cash model blocks. Its default margin model applies 5x leverage.

An MOC timeline would change the validated fill timing, which counts as a trial. Trailing stops do not exist for CNC on Kite; the nightly GTT modification already trails the stop.
- **Lean's stock risk models, including a drift-aware sector cap.** Centurion's 6×ATR never-lowered GTT stop, its graded drawdown rule with re-arm, its regime gate and its volatility target already cover or beat these models.

A buffer-aware sector cap would be a new strategy rule spending the trial. Measured equity never exceeded 25% in a sector.
- **Lean's indicator definitions (SMA-seeded EMA, Wilder ATR, population std) or a Python TA library.** These are not bugs in Centurion. Switching would change every forecast and stop (Wilder ATR differs by a median 5.9%) and spend the trial.

TA libraries add dependencies and work one instrument at a time. Centurion's pandas code is already causal, NaN-preserving and independent of the anchor date.
- **Lean's India market hours, symbol properties, expiry functions, QuantLib option engine and F&O fee handling.** They are out of date or wrong for India:
- holidays are incomplete and there are no weekend or Muhurat sessions;
- the NIFTY lot is listed as 75;
- the last-Thursday expiry rule is wrong in 18 of 237 months;
- the QuantLib model adds an extra settlement day and uses a US calendar and rates;
- exercise is treated as fee-free.

Centurion is ahead on every one of these.
- **Lean's Sharpe, Sortino, PSR, VaR and IR definitions; grid/Euler optimisation; Lean's 2%-of-bar capacity number.** Lean compounds only the Sharpe numerator (1.41 against the correct 1.21). Its Sortino is not a downside deviation, its PSR uses excess kurtosis where raw kurtosis belongs, and its VaR is normal.

In-sample search maximises selection bias, which a one-trial budget cannot afford. The capacity heuristic has no impact model.
- **TargetOverlay chain (CompositeRiskManagementModel pattern).** Refuted. It would hold only one member, the shift multiplier. That multiplier is computed from the very paper book G4 judges, so a backtest can only import it. Moving it into the reference would change the G4 reference pre-registered on 28 Sep.

Separately, the user may want to note how multiplier-scaled sessions count in the drift check, since today a 'drifting' state can sustain itself.
- **Point the G4 reference at a 0% idle-cash yield / cost model 4 for idle cash.** Refuted: already done as IC1 (cost model 4, IDLE_CASH_YIELD_ANNUAL = 0), which also re-baselined the registry. It is still uncommitted, so nightly Actions runs still credit 6% until the user commits and pushes it. IC1b (kill threshold 0.239) is still pending.
- **Give live holdings their entry date for the ATR stop.** Refuted: LS1 already did this today. The ledger keeps entries and stops, scoped_book passes the entry date, and a replay matches the backtest on 3,219 of 3,219 nights. At most, add a P3 warning in holdings_from_mapping.
- **Preflight check comparing planned sells with Kite's authorised_quantity.** Refuted. It could never fire:
- dry runs start from an empty ledger, so there is nothing to compare;
- it would be switched off on attested accounts;
- it cannot detect a lapsed DDPI.

Its purpose is covered by the sell-path readiness attestation and the rejected-sell classifier.
- **Model exits of holdings that move to BZ or another non-EQ/BE series in the backtest.** There are only 2 episodes in 14 years (KESORAMIND 2015, JETAIRWAYS 2019), and the deployed book held neither. A live alert from the identity resolver is enough.
- **Lean's permanent FirstTicker+FirstDate key for the forecast tie-break; hand-picked crisis windows; 'min of the two' capacity headline; trade ratios normalised by equity at open; renaming metrics['kurtosis'].** - FirstTicker would change live ranking and depends on the store's first row.
- Hand-picked windows swing results and were chosen after the results were seen.
- The Rs 6.4 crore sleeve binding is not a strategy ceiling.
- Normalising by equity at open inflates winners (+0.23), whereas neutral denominators match INR to within 0.09.
- Renaming the kurtosis key would break 5,559 manifests.
- **Nightly parity-check subcommand; bumping IV_VERSION for a forward-based iv_history; reusing services/market_data/corporate_actions.py.** - The parity check would only re-test what the until() unit test already covers.
- Moving iv_history to a forward shifts IVs by about -0.07 vol points against a ±4.7-point cone, which is not worth a version bump.
- The legacy corporate-action helper has no ex-date filter, parses 'Split 10 to 2' inverted, and re-applies on every cycle.

## Where Centurion is already ahead

- Survivorship-free NSE store whose calendar comes from the data itself. It includes 21 weekend sessions and every Muhurat session from 2012 to 2025 (store.py:387-400; archive.py:76-80, 194-205). Lean's Equity-india entry has no weekend sessions, no early closes and incomplete holidays.
- Corporate actions built from NSE's own daily Bc files: ratio parsing, rights TERP, demerger and capital-reduction factors, ex-date revisions and slip correction, plausibility rejection, and ISIN-aware gap inference (reference.py:383-547; panel.py:250-652). Lean has no NSE factor files.
- Renames linked from two sources, symbolchange.csv and ISIN continuity, with a valid_from guard (reference.py:123-244). All 369 EQ renames from 2013 to 2025 are linked, including ones NSE's own file omits (PHILIPCARB->PCBL, BURGERKING->RBA).
- As-printed price filter (close_unadj) kept alongside total-return prices, plus a delisting exit that pays realistic sell costs (engine.py:611-625). Lean fills a delisting at zero fee.
- Cost model 3 with dated, instrument-specific rates: STT schedules, ETF exemptions, stamp, exchange, SEBI, GST and the DP charge (costs.py:34-62, 115-172). Also a deterministic square-root impact model and a 5% participation cap (costs.py:175-200). F&O charges are dated, include exercise STT, and are reconciled against Kite (fno_costs.py:71-179).
- Fills at the next open, guarded against stale prices: fillable_open checks ETF opens, symbols with no open are skipped, and same-day sale proceeds fund buys, which matches Zerodha CNC (engine.py:452-545; costs.py:249-258).
- Overfitting control Lean lacks entirely: a trial registry with data hash, cost model and refresh, deflated Sharpe on the raw trial count, CSCV PBO, an anchored walk-forward stitched across platforms, pre-registered pass rules, and D2 haircuts (validation/trials.py, dsr.py, pbo.py, walk_forward.py, cloud/wf_stitch.py; scorecard.py:76-78).
- Correct statistics. Sharpe is the excess mean over its sd, Sortino uses the target semideviation, PSR uses non-excess kurtosis, alpha and beta are HAC-corrected, and CVaR is historical (metrics.py:105-112; dsr.py:344; diagnostics.py:77-117; scorecard.py:162-189). Lean overstates the deployed Sharpe as 1.41 against the correct 1.21.
- Forward parity gate G4. Each book is compared with a backtest over the same period under pre-registered limits, and the result drives the capital ladder (paper_gate.py; deployment.py:120-133; capital_ladder.py). Lean only charts live results against a backtest from another period.
- One pure target function shared by backtest, paper and live (engine.py:198-416), with a drawdown state machine replayed from equity, so a restart cannot lose risk state (drawdown.py; executor:517-547). Lean's risk models keep peaks in memory and re-seed them on restart.
- Kite plumbing Lean's plugin lacks: AMO variety, tags with a dedupe check against the order book, persistent GTT stops reconciled every night, a ledger scoped to the book's symbols in a shared account, T1-aware quantities, live_balance cash, an exits-only kill switch with a circuit breaker, a static-IP proxy, a token window check, dry runs and readiness gating (order_service.py; gtt_stops.py; live_session.py; nse_engine_executor.py).
- Independence from the data's start date is proven: finite-memory EWMs and a pinned anchor, with anchor-check at max |diff| 0. The same results reproduce to 15 significant digits on the Mac and on Kaggle (signals.py:61-114; deployment.py:91-95).
- Options toolkit. Contracts are resolved from Kite's instrument dump (tick, lot, expiry), and historical expiries come from settlement_sessions (instruments.py:51-98; signal_futures.py:161-170). Time to expiry is measured to 15:30 IST, and an IV that cannot be solved returns None rather than 0 (live_chain.py:30-62; theory.py:139-195).

## Recommended tests

1. This week, no code: check DDPI in Console (Account > Segments/DDPI) for the primary Zerodha ID. If it is absent, submit DDPI now. Also open a Zerodha support ticket asking two things: whether a pending sell AMO funds a buy AMO (and whether funds are checked at placement or at release), and how GTTs behave on bonus or split ex-dates and on a move from EQ to BE.
2. Record read-only baselines to use as before/after references for every fix:
- off-tick limit and GTT counts on the current Kite dump: about 96/96/62 of 302, and 34% in the H2-2025 order replay;
- BE resolution: 302/302;
- held corporate-action events in 2013-25: 42, with the G4-window shortfall above 3 bp/day in 35% of windows;
- until() differences: 2,709/3,220 sessions, e4 tracking error 1.39%/yr;
- cooldown_ab early re-entries: 20 deployed, 17 e4;
- idem_check repro counts;
- msr replay.
3. Temporary fake-Kite fault-injection tests for each P0 and P1 item, in this order: tick, idempotency, corporate actions, identity, missed-session, fail-closed, cooldown, funding, preflight, exit protection. Run them in myenv and in the core-only venv, and delete them once verified, per the house rules.
4. Before 8 Nov: backfill the 2026-02-01 store session, rebuild 2026, run refresh-registry --dry-run, and measure the deployed holdout delta (2026-01-01..09-11, the 5 stop exits). Then run the Fri/Sun/Mon paper catch-up fault-injection test, and confirm the 8 Nov Muhurat bhavcopy lands in the nightly sync.
5. Paper checks: after the until() fix, check that the candidate and e4 nightly plans equal the same-day reference targets. After the corporate-action fix, replay BSE 2025-05-23, RELIANCE 2017-09-07 and GOLDBEES 2019-12-19 through run_paper_session on a /tmp SQLite, plus the 16 Sep 2016 G4-window replay; the event-attributable daily gap should be about 0.
6. Actions: run one workflow_dispatch dry run with the canary step before and after adding the NIFTY cache to the runner's store (expect the hash to go from different to ba7c098240b4c9ec). Then confirm two consecutive nights give identical canary outputs.
7. Local replays to choose the exit defaults before setting them: gtt_buffer_check.py (expect 2%), exitband_verify/replay2.py with the circuit clamp at 1/2/3/5% on both stores, and lmtv/crash.py for 24 Aug 2015.
8. Supervised funded broker test before about 12 Dec, one evening plus the next session:
- an AMO SELL of 1 share;
- a triggered single-leg GTT SELL;
- a buy AMO funded only by the pending sale, cancelled before 09:00;
- a far-from-market AMO with a '-' in its tag.
Record the status strings, profile().meta.demat_consent and the raw holdings key names, then store the sell-path attestation.
9. One Kaggle data re-baseline job, run detached, which does not spend the trial. It combines the point-in-time tie-break, the rename-invariant widened hash, the corrected 2026 store and PBO on the excess basis, over the ~61 same-window configs plus the r12a4/r12b4 walk-forwards. Report per-config Sharpe deltas, Spearman rank stability, PBO/DSR before and after, and the relabel gate at 0/61. Run it before evaluating the one remaining configuration.
10. One Kaggle options job: O2 A1/A2/B, O3 X1-X4 and O4 Y1/Y2 with the canonical settlement map and the settle-column guard. Expect A2's 30 Jun 2023 day to be 0, A1's DSR about 0.892, and every gate-1 verdict unchanged.
11. Nightly live dry runs once the tick, identity and preflight fixes ship: each email should show zero WOULD_REJECT and display limit prices. Only dry runs like that should count toward D3's five clean runs.
12. Regenerate the scorecards for the reporting items (ex-ante regime labels, drawdown periods, capacity binding and cap label, PSR/MinTRL) and diff them: only the new or corrected sections should change.
13. After go-live (mid December):
- nightly, compare live_order_outcomes with the outcome predicted from the next bhavcopy;
- weekly, count cut or rejected buys and rejected-sell alerts, and track G4 with the order-rule share shown alongside;
- after 30 or more real fills, run X1 on Kaggle with DailyLimitFill as cost model 5.
