# Scorecard: candidate (2d64ba4c), 2013-01-01 to 2025-12-31

Run 20261010T031406532878Z_2d64ba4c, data dab49d3cbd323c16, cost model 4, risk-free 6.5%, as of 2026-10-10T11:50 UTC. The recorded run is the registry's configuration; the paper and live books add the deployment's drawdown overlay (E2).

## Pass rules (fixed before the data was read)

| Rule | Target | Value | Verdict |
|---|---|---|---|
| net Sharpe | > 1.2 | 1.16 | FAIL |
| max drawdown | >= -0.3 | -23.6% | PASS |
| Calmar | >= 1.0 | 0.93 | FAIL |
| deflated Sharpe | >= 0.95 | 0.96 | PASS |
| walk-forward OOS Sharpe | >= 1.2 | 1.24 | PASS |

## Return and risk

| Metric | Value |
|---|---|
| CAGR | 21.9% |
| Sharpe (excess) | 1.16 |
| Sortino | 1.61 |
| Information ratio vs NIFTY 50 TRI | 0.43 |
| Active return vs NIFTY 50 TRI | 6.6% |
| Tracking error vs NIFTY 50 TRI | 15.3% |
| Calmar | 0.93 |
| Max drawdown | -23.6% |
| Volatility | 12.5% |
| CVaR 95% (daily) | -1.90% |
| CVaR 99% (daily) | -3.12% |
| Worst day | -6.6% |
| Worst month | -9.7% |
| Skew | -0.87 |
| Kurtosis (excess) | 5.61 |
| Beta to NIFTY 50 TRI | 0.35 |
| Beta on NIFTY down days | 0.37 |
| Market alpha (annual, t) | 11.7% (t 3.4) |

**Drawdowns** (151 periods): the longest under water ran 2018-01-15 to 2020-07-27, 924 days, -17.8% deep; 87.1% of days under water; none open at the end.

| Peak | Trough | Recovered | Depth | Days |
|---|---|---|---|---|
| 2015-04-08 | 2016-01-29 | 2017-04-20 | -23.6% | 743 |
| 2018-01-15 | 2019-08-01 | 2020-07-27 | -17.8% | 924 |
| 2022-04-11 | 2023-03-13 | 2023-06-13 | -11.4% | 428 |
| 2013-01-03 | 2013-04-05 | 2013-12-19 | -10.9% | 350 |
| 2024-09-27 | 2024-11-14 | 2025-05-22 | -10.1% | 237 |

**Across NIFTY 50 TRI's deepest drawdowns** (descriptive: the fall is peak to trough, the recovery trough to the TRI's recovery or the end)

| TRI peak | Trough | Recovered | TRI fall | Book fall | TRI recovery | Book recovery |
|---|---|---|---|---|---|---|
| 2020-01-14 | 2020-03-23 | 2020-11-06 | -38.3% | -6.5% | 62.4% | 33.1% |
| 2015-03-03 | 2016-02-25 | 2016-08-31 | -21.4% | -16.5% | 27.5% | 17.3% |
| 2021-10-18 | 2022-06-17 | 2022-11-11 | -16.4% | -5.7% | 20.7% | -2.2% |
| 2024-09-26 | 2025-03-04 | 2025-10-23 | -15.4% | -3.6% | 18.3% | 20.4% |
| 2018-08-28 | 2018-10-26 | 2019-04-02 | -14.4% | -7.0% | 17.2% | -6.5% |

## Attribution: style factors and the alpha left

| Factor | Beta | t (HAC) | Return explained per year |
|---|---|---|---|
| size | 0.04 | 1.2 | 0.1% |
| momentum | 0.31 | 13.5 | 3.7% |
| low_vol | -0.24 | -10.2 | -1.5% |
| market | 0.26 | 5.7 | 2.0% |
| **alpha** | 10.7% per year | 3.3 | 72% of the excess return |

R² 0.38 over 3198 days. Factors are built from the store, point in time: size is a liquidity proxy (traded value), value is not available: the store holds no fundamentals (book value, earnings).

## Alpha decay

Rank IC of the combined forecast against forward returns (universe members, every 21 sessions):

| Horizon (sessions) | 5 | 21 | 63 | 126 | 252 |
|---|---|---|---|---|---|
| mean IC | 0.034 | 0.049 | 0.076 | 0.090 | 0.092 |
| t-stat | 2.6 | 3.2 | 5.0 | 6.2 | 6.5 |

| Year | Alpha vs NIFTY 50 TRI | t | Beta |
|---|---|---|---|
| 2013 | -5.3% | -0.6 | 0.12 |
| 2014 | 38.2% | 2.6 | 0.70 |
| 2015 | -3.9% | -0.3 | 0.73 |
| 2016 | -2.6% | -0.2 | 0.29 |
| 2017 | 13.6% | 1.2 | 0.97 |
| 2018 | -17.0% | -2.3 | 0.21 |
| 2019 | -4.1% | -0.6 | 0.22 |
| 2020 | 21.3% | 1.6 | 0.12 |
| 2021 | 51.0% | 3.1 | 0.68 |
| 2022 | -11.4% | -1.2 | 0.25 |
| 2023 | 15.2% | 1.9 | 0.59 |
| 2024 | 11.2% | 1.0 | 0.70 |
| 2025 | 22.0% | 2.6 | 0.28 |

Alpha trend 1.1% per year (p 0.50); first half mean 3.8%, second half 15.0%.

| Holding period | Round trips | Mean P&L | Hit rate |
|---|---|---|---|
| <=21d | 747 | Rs -2,051 | 22% |
| 22-63d | 762 | Rs -442 | 38% |
| 64-126d | 302 | Rs 10,040 | 83% |
| 127-252d | 87 | Rs 23,918 | 93% |
| >252d | 8 | Rs 1.0 L | 100% |

## Trading

| Metric | Value |
|---|---|
| Turnover (one-way, per year) | 6.7x |
| Cost drag (modelled, per year) | 2.7% = impact 1.3% + statutory 1.4% |
| Round trips | 1906 (median holding 29 days) |
| Hit rate | 41.7% |
| Win/loss ratio (avg win / avg loss) | 2.65 |
| Profit factor | 1.89 |
| Average P&L per round trip | Rs 2,124 (13 bp of equity) |
| Largest win / loss | Rs 2.5 L / Rs -46,298 |
| Participation (order / median traded value) | median 0.01%, p95 0.07%; 1 of 6,466 fills cut by the 5% cap, 249 buys scaled down for cash |
| Modelled impact | 9.8 bp of traded value |

## Capacity

Fills of 2023-12-21 to 2025-12-30 (1027) re-sized to each capital; gross edge 16.7% per year (net excess CAGR plus modelled impact).

| Capital | Impact drag per year | Edge left | Fills over the 5% cap | Traded value over it |
|---|---|---|---|---|
| Rs 6.0 L | 0.43% | 16.3% | 0.0% | 0.0% |
| Rs 12.0 L | 0.48% | 16.2% | 0.0% | 0.0% |
| Rs 21.0 L | 0.54% | 16.2% | 0.0% | 0.0% |
| Rs 30.0 L | 0.58% | 16.1% | 0.0% | 0.0% |
| Rs 50.0 L | 0.66% | 16.0% | 0.0% | 0.0% |
| Rs 1.00 cr | 0.80% | 15.9% | 0.0% | 0.0% |
| Rs 2.00 cr | 1.01% | 15.7% | 0.0% | 0.0% |
| Rs 5.00 cr | 1.41% | 15.3% | 0.0% | 0.0% |
| Rs 10.00 cr | 1.86% | 14.8% | 0.4% | 3.0% |

**Impact eats half the edge at Rs 270.81 cr**, where 25.3% of fills (47.4% of traded value) would exceed the participation cap. The engine would cap fills over the participation limit rather than pay the impact shown; the capped share says how much of the book would then go untraded.

The cap binds first on SILVERBEES (sleeve) at Rs 6.48 cr, GOLDBEES (sleeve) at Rs 6.97 cr, HSCL at Rs 37.77 cr, ACUTAAS at Rs 42.16 cr, ASTERDM at Rs 44.52 cr; the first core stock is HSCL at Rs 37.77 cr; 5%, 25%, 50% of traded value is over it at Rs 15.48 cr, Rs 117.69 cr, Rs 299.84 cr.

## Robustness

**Walk-forward OOS** (2017-01-02 to 2025-12-31, 9 folds over 32 grid points of the configuration family, base bd79bf28; data/nse_engine/wf_oos_returns_r12a5.csv): Sharpe 1.24, CAGR 23.1%, MaxDD -22.1%, Calmar 1.04; mean IS 0.95 vs OOS 1.01 per fold (OOS/IS 1.30), 3 negative OOS years; Sharpe standard error 0.35, PSR(1.2) 0.543, MinTRL vs 0 479 sessions (1.9 years); reported, not a pass rule.

**Overfitting** over 61 recorded configurations (data dab49d3cbd323c16, cost model 4): deflated Sharpe 0.963 (annual Sharpe 1.16 vs the expected best of 0.64 from 61 trials; clustered 1.000), PBO 29.2%, probability of an OOS loss 0.7%; Sharpe standard error 0.29, PSR(1.2) 0.442, MinTRL vs 0 547 sessions (2.2 years); reported, not a pass rule.

**Parameter sensitivity**: 5 recorded one-setting neighbours (Sharpe change -0.04 to 0.03); fragile settings (|change| > 0.10): none.

| Setting | From | To | Sharpe | Change | CAGR change |
|---|---|---|---|---|---|
| portfolio.no_trade_buffer | 0.25 | 0.5 | 1.12 | -0.04 | -1.0% |
| sleeves.trend_confirm_days | 1 | 3 | 1.12 | -0.04 | -0.2% |
| regime.scale_neutral | 0.6 | 1.0 | 1.14 | -0.02 | 0.8% |
| universe.price_filter_unadjusted | True | False | 1.15 | -0.01 | -0.1% |
| portfolio.exit_rank | 40 | 60 | 1.19 | 0.03 | 0.7% |

**Regime stability by NIFTY 50 trend** (the regime known at the previous close)

| Regime | Days | Annual return | Sharpe | Up days | Worst day |
|---|---|---|---|---|---|
| bear | 19% | 7.0% | 0.04 | 53% | -6.6% |
| bull | 67% | 28.6% | 1.74 | 60% | -6.2% |
| sideways | 15% | 3.4% | -0.26 | 54% | -4.1% |

**Regime stability by India VIX** (the regime known at the previous close)

| Regime | Days | Annual return | Sharpe | Up days | Worst day |
|---|---|---|---|---|---|
| calm | 94% | 21.2% | 1.18 | 58% | -6.6% |
| elevated | 6% | 16.2% | 0.79 | 58% | -3.1% |

## Correlation of daily returns

| Series | Daily | Monthly | Days |
|---|---|---|---|
| book:deployed | 0.97 | 0.97 | 3220 |
| book:e4 | 0.95 | 0.95 | 3220 |
| options:O2-A1 SD call spread, 15 days | -0.01 | -0.11 | 3218 |
| options:O2-A2 SD call spread, 4 days | -0.04 | -0.07 | 3218 |
| options:O2-B max-pain call spread | -0.07 | -0.06 | 3218 |
| options:O3-X1 PCR bands (PyPatel) | -0.05 | -0.19 | 3218 |
| options:O3-X2 TRIN bands (PyPatel) | -0.03 | 0.01 | 3218 |
| options:O3-X3 India VIX >= 22 (PyPatel) | 0.15 | 0.22 | 3218 |
| options:O3-X4 55-day breakout (PyPatel) | 0.19 | 0.19 | 3218 |
| options:O4-Y1 option-expiry week (awesome-systematic-trading) | 0.19 | 0.34 | 3218 |
| options:O4-Y2 volatility risk premium (awesome-systematic-trading) | 0.39 | 0.44 | 3218 |
| options:O4-Y3 overnight with sentiment (awesome-systematic-trading) | 0.14 | 0.23 | 3208 |
| etf:GOLDBEES | 0.26 | 0.08 | 3220 |
| etf:SILVERBEES | 0.48 | 0.36 | 967 |
| index:NIFTY50_TRI | 0.45 | 0.49 | 3220 |

## Live: implementation shortfall, slippage, tracking error (paper book)

Pending: CENTURION_DATABASE_URL is not set here: the paper record lives in Neon; run with it set (or on GitHub Actions) once paper trading completes. Real fills to date (tracker, 8 Oct 2026): 2, at -8 bp and +10 bp against the model.
