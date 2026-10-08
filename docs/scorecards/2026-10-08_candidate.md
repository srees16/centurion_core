# Scorecard: candidate (2d64ba4c), 2013-01-01 to 2025-12-31

Run 20261008T075322161185Z_2d64ba4c, data ba7c098240b4c9ec, cost model 3, risk-free 6.5%, as of 2026-10-08T18:01 UTC. The recorded run is the registry's configuration; the paper and live books add the deployment's drawdown overlay (E2).

## Pass rules (fixed before the data was read)

| Rule | Target | Value | Verdict |
|---|---|---|---|
| net Sharpe | > 1.2 | 1.21 | PASS |
| max drawdown | >= -0.3 | -23.0% | PASS |
| Calmar | >= 1.0 | 0.98 | FAIL |
| deflated Sharpe | >= 0.95 | 0.99 | PASS |
| walk-forward OOS Sharpe | >= 1.2 | 1.15 | FAIL |

## Return and risk

| Metric | Value |
|---|---|
| CAGR | 22.6% |
| Sharpe (excess) | 1.21 |
| Sortino | 1.69 |
| Information ratio vs NIFTY 50 TRI | 0.47 |
| Active return vs NIFTY 50 TRI | 7.2% |
| Tracking error vs NIFTY 50 TRI | 15.2% |
| Calmar | 0.98 |
| Max drawdown | -23.0% |
| Volatility | 12.3% |
| CVaR 95% (daily) | -1.87% |
| CVaR 99% (daily) | -3.09% |
| Worst day | -6.6% |
| Worst month | -9.6% |
| Skew | -0.88 |
| Kurtosis (excess) | 5.74 |
| Beta to NIFTY 50 TRI | 0.35 |
| Beta on NIFTY down days | 0.37 |
| Market alpha (annual, t) | 12.3% (t 3.6) |

## Attribution: style factors and the alpha left

| Factor | Beta | t (HAC) | Return explained per year |
|---|---|---|---|
| size | 0.04 | 1.1 | 0.1% |
| momentum | 0.31 | 13.7 | 3.6% |
| low_vol | -0.24 | -10.3 | -1.5% |
| market | 0.26 | 5.8 | 2.0% |
| **alpha** | 11.2% per year | 3.5 | 73% of the excess return |

R² 0.38 over 3198 days. Factors are built from the store, point in time: size is a liquidity proxy (traded value), value is not available: the store holds no fundamentals (book value, earnings).

## Alpha decay

Rank IC of the combined forecast against forward returns (universe members, every 21 sessions):

| Horizon (sessions) | 5 | 21 | 63 | 126 | 252 |
|---|---|---|---|---|---|
| mean IC | 0.034 | 0.049 | 0.076 | 0.090 | 0.092 |
| t-stat | 2.6 | 3.2 | 5.0 | 6.2 | 6.5 |

| Year | Alpha vs NIFTY 50 TRI | t | Beta |
|---|---|---|---|
| 2013 | -3.0% | -0.4 | 0.12 |
| 2014 | 38.4% | 2.6 | 0.69 |
| 2015 | -3.5% | -0.3 | 0.72 |
| 2016 | -1.3% | -0.1 | 0.30 |
| 2017 | 13.5% | 1.2 | 0.96 |
| 2018 | -15.8% | -2.1 | 0.21 |
| 2019 | -2.9% | -0.4 | 0.22 |
| 2020 | 21.8% | 1.7 | 0.13 |
| 2021 | 49.1% | 3.1 | 0.68 |
| 2022 | -10.4% | -1.1 | 0.25 |
| 2023 | 15.4% | 1.9 | 0.59 |
| 2024 | 11.5% | 1.0 | 0.70 |
| 2025 | 22.9% | 2.7 | 0.28 |

Alpha trend 1.0% per year (p 0.51); first half mean 4.7%, second half 15.3%.

| Holding period | Round trips | Mean P&L | Hit rate |
|---|---|---|---|
| <=21d | 756 | Rs -2,184 | 22% |
| 22-63d | 758 | Rs -475 | 37% |
| 64-126d | 301 | Rs 10,571 | 83% |
| 127-252d | 88 | Rs 24,444 | 93% |
| >252d | 8 | Rs 1.0 L | 100% |

## Trading

| Metric | Value |
|---|---|
| Turnover (one-way, per year) | 6.7x |
| Cost drag (modelled, per year) | 2.7% = impact 1.3% + statutory 1.4% |
| Round trips | 1911 (median holding 29 days) |
| Hit rate | 41.3% |
| Win/loss ratio (avg win / avg loss) | 2.64 |
| Profit factor | 1.86 |
| Average P&L per round trip | Rs 2,166 (13 bp of equity) |
| Largest win / loss | Rs 2.3 L / Rs -49,303 |
| Participation (order / median traded value) | median 0.01%, p95 0.07%, 3.7% of fills cut by the 5% cap |
| Modelled impact | 9.9 bp of traded value |

## Capacity

Fills of 2023-12-21 to 2025-12-30 (1018) re-sized to each capital; gross edge 17.4% per year (net excess CAGR plus modelled impact).

| Capital | Impact drag per year | Edge left | Fills over the 5% cap |
|---|---|---|---|
| Rs 6.0 L | 0.44% | 17.0% | 0% |
| Rs 12.0 L | 0.49% | 16.9% | 0% |
| Rs 21.0 L | 0.54% | 16.9% | 0% |
| Rs 30.0 L | 0.58% | 16.8% | 0% |
| Rs 50.0 L | 0.66% | 16.7% | 0% |
| Rs 1.00 cr | 0.81% | 16.6% | 0% |
| Rs 2.00 cr | 1.01% | 16.4% | 0% |
| Rs 5.00 cr | 1.41% | 16.0% | 0% |
| Rs 10.00 cr | 1.87% | 15.5% | 0% |

**Impact eats half the edge at Rs 291.20 cr**, where 28% of fills would exceed the participation cap. The engine would cap fills over the participation limit rather than pay the impact shown; the capped share says how much of the book would then go untraded.

## Robustness

**Walk-forward OOS** (2017-01-02 to 2025-12-31, 9 folds over 32 grid points of the configuration family, base bd79bf28; data/nse_engine/wf_oos_returns_r12a.csv): Sharpe 1.15, CAGR 20.9%, MaxDD -20.9%, Calmar 1.00; mean IS 1.02 vs OOS 0.99 per fold (OOS/IS 1.13), 3 negative OOS years.

**Overfitting** over 61 recorded configurations (data ba7c098240b4c9ec, cost model 3): deflated Sharpe 0.989 (annual Sharpe 1.21 vs the expected best of 0.54 from 61 trials; clustered 1.000), PBO 57.7%, probability of an OOS loss 0.0%.

**Parameter sensitivity**: 5 recorded one-setting neighbours (Sharpe change -0.04 to 0.03); fragile settings (|change| > 0.10): none.

| Setting | From | To | Sharpe | Change | CAGR change |
|---|---|---|---|---|---|
| regime.scale_neutral | 0.6 | 1.0 | 1.17 | -0.04 | 0.6% |
| sleeves.trend_confirm_days | 1 | 3 | 1.17 | -0.04 | -0.2% |
| portfolio.no_trade_buffer | 0.25 | 0.5 | 1.20 | -0.01 | -0.6% |
| universe.price_filter_unadjusted | True | False | 1.24 | 0.03 | 0.5% |
| portfolio.exit_rank | 40 | 60 | 1.25 | 0.03 | 0.9% |

**Regime stability by NIFTY 50 trend**

| Regime | Days | Annual return | Sharpe | Up days | Worst day |
|---|---|---|---|---|---|
| bear | 19% | -13.7% | -1.71 | 52% | -6.6% |
| bull | 67% | 36.6% | 2.43 | 61% | -5.5% |
| sideways | 15% | -2.9% | -0.77 | 56% | -6.2% |

**Regime stability by India VIX**

| Regime | Days | Annual return | Sharpe | Up days | Worst day |
|---|---|---|---|---|---|
| calm | 94% | 24.0% | 1.45 | 59% | -5.5% |
| elevated | 6% | -19.4% | -1.66 | 53% | -6.6% |

## Correlation of daily returns

| Series | Daily | Monthly | Days |
|---|---|---|---|
| book:deployed | 0.97 | 0.97 | 3220 |
| book:e4 | 0.95 | 0.95 | 3220 |
| options:O2-A1 SD call spread, 15 days | -0.01 | -0.12 | 3218 |
| options:O2-A2 SD call spread, 4 days | -0.03 | -0.07 | 3218 |
| options:O2-B max-pain call spread | -0.07 | -0.04 | 3218 |
| etf:GOLDBEES | 0.26 | 0.08 | 3220 |
| etf:SILVERBEES | 0.48 | 0.36 | 967 |
| index:NIFTY50_TRI | 0.45 | 0.48 | 3220 |

## Live: implementation shortfall, slippage, tracking error (paper book)

Pending: CENTURION_DATABASE_URL is not set here: the paper record lives in Neon; run with it set (or on GitHub Actions) once paper trading completes. Real fills to date (tracker, 8 Oct 2026): 2, at -8 bp and +10 bp against the model.
