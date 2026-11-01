# Centurion measured audit (from its own records, 31 Oct 2026; all data ends 2025, 2026 holdout unread)

Basis: deployed family walk-forward 2008-25 (wf_stitched_r4b), live-like = idle cash at the liquid rate less the D2
haircut. D2 is ALREADY inside every live-like figure, so do not apply an implementation haircut to 13.0% / 0.60 again.

## Result ladder
- Backtest 2013-25 in-sample: 23.1% CAGR, Sharpe 1.16, MaxDD -23.9% (settings chosen on this window).
- Walk-forward live-like 2013-25: 16.9%, 0.88, -20.9%.
- Walk-forward live-like 2008-25: 13.0%, 0.60, -29.5%, vol 11%.
- NIFTY total return 2008-25: 9.8%, 0.25, -59.4% (before Feb 2012: price plus an assumed 1.3% yield).
- Liquid rate 6.2%. After tax about 10.5% (20.8% on yearly gains, losses carried).
- Bootstrap 5-year CAGR: p10 4.3, median 13.2, p90 23.2; 2008-12 actual 3.7% a year.

## Regimes (ex-post, NIFTY +/-20% turning points; range = non-bear with 6-month centred move under 7.5%)
| Regime | Share of days | NIFTY a year | Book a year | Capture |
| Bull | 49% | 42.0% | 30.1% | 0.73 |
| Range | 34% | 13.6% | 7.0% | 0.51 |
| Bear | 17% | -51.3% | -15.4% | 0.23 |
Bear episodes, NIFTY vs book: 2008 H1 -39/-24; 2008 Aug-Oct -45/-3; 2010-11 -28/-1; 2015-16 -23/-11; 2020 -38/-6.
75% of log return is earned in causal "bull" state (NIFTY above a rising 200-day mean), 19% range, 6% bear.
A causal 200-day state does not separate market returns in India: NIFTY 5.2% a year in it, 19.6% range, 16.9% bear,
because crashes start above the mean and rebounds start below it.

## Alpha and exposure
Beta 0.17, corr 0.32, alpha 5.7% a year (t 2.0). By period: 2008-12 -2.3% (t -0.4); 2013-19 +4.4% (t 1.0);
2020-25 +11.2% (t 2.6). Up capture 0.31, down capture 0.21. Mean invested 70%: stocks 41%, gold and silver 29%, idle 30%.

## Attribution, points a year (walk-forward, simple sum)
| Period | Stocks | Metals | Costs and timing | Total |
| 2008-25 | 10.7 | 4.7 | -2.2 | 13.3 |
| 2008-12 | -1.1 | 6.2 | -2.2 | 2.9 |
| 2013-19 | 14.7 | 1.1 | -2.4 | 13.3 |
| 2020-25 | 15.8 | 7.7 | -1.8 | 21.8 |
Correlation of stock and metal P&L -0.01. Metals earn 35% of the return on 29% of capital, and were picked knowing
gold and silver survived and rose: a selection bias.

## Signal quality (forecast rank IC against the universe, monthly)
5-day IC 0.037; 21-day 0.051 (ICIR 0.85 a year, 64% positive, top-bottom quintile +1.17%); 63-day 0.079 (ICIR 1.38).
21-day IC by period: 2008-12 0.060; 2013-19 0.070; 2020-25 0.021. The signal is slower than the rebalance and has decayed.
Next-21-session excess after each decision: buys +1.50% (hit 52%); rank exits +0.97%; rebalance sells +1.39%; regime
exits +0.25%; stops -1.24%. Only stops clearly remove losers: most sold names keep beating the universe, so much of
the turnover (420 trades a year, about 2.2 points of cost) buys almost nothing. Median 20 names held.

## Sleeves and other books (2008-25 live-like unless stated)
Bonds 11.5% / 0.59 / -21.5%. MCX 12.6% / 0.63 / -20.7%. A1 13.2% / 0.62. All sleeves combined (estimate, 50% core,
25% gilts, 25% MCX, options overlay) 11.2% / 0.61 / -14.5%: drawdown halved, Sharpe unchanged. E4 fixed settings
17.8% / 0.80 / -28.4% (backtest; no walk-forward before 2017). Levered book on real stock futures 2008-25 13.9% / 0.45 /
-40.3% against 13.2% / 0.62 unlevered, 4.2 points a year of futures costs, lowest NAV 63% of start, 56% without the
drawdown rule; plain 1.75x on NIFTY's -59.9%: -104.7% held, -81.6% reset daily.

## Statistics and paper
DSR 0.939 at N 61 after autocorrelation widening (gate 0.95). PBO 29.2% on excess returns, 51.2% raw. Paper: 30
sessions, -0.71% against NIFTY -0.93%. 2026 holdout (opened once, 15 Sep): 26.9% annualised, Sharpe 1.04.

## Survivorship (SB3)
1,038 universe names, 291 gone, kept in the data; survivors-only run +0.10 CAGR point, so the core is clean. Fixed:
for-cause exits, live BZ exits, legacy Yahoo pipeline labelled biased. Open: instruments picked knowing they survived.

## Passive yardstick (one look, not a trial; 2008-25, monthly rebalanced, recorded series)
| Series | CAGR | Sharpe | Vol | MaxDD | Calmar |
| Centurion deployed, live-like | 13.0% | 0.60 | 11.0% | -29.5% | 0.44 |
| Gold (GOLDBEES) alone | 13.8% | 0.52 | 15.1% | -27.3% | 0.51 |
| Passive 60 NIFTY / 40 gold | 12.4% | 0.48 | 13.1% | -39.2% | 0.32 |
| Passive 50 NIFTY / 25 gold / 25 gilts | 10.9% | 0.43 | 10.6% | -32.2% | 0.34 |
| Centurion 75 + gold 12.5 + gilts 12.5 | 12.6% | 0.65 | 9.2% | -20.7% | 0.61 |
| NIFTY total return | 9.8% | 0.25 | 20.4% | -59.4% | 0.17 |
| Gilts proxy | 6.5% | 0.01 | 5.3% | -12.8% | 0.51 |
| MCX-B stream | 9.7% | 0.25 | 18.8% | -41.1% | 0.24 |
Centurion's edge over a passive mix is risk (about +0.12 to +0.17 Sharpe), not much return. Gold's 2008-25 run flatters every figure that holds it.
By period, CAGR, Centurion / 50-25-25: 2008-12 3.7 / 9.0; 2013-19 13.6 / 9.3; 2020-25 21.5 / 15.0.

## Crash correlations (weekly; crash = NIFTY 15%+ below its peak, 22% of days)
Calm: book-NIFTY 0.52, book-gold 0.21, book-gilt 0.07, book-MCX 0.10, NIFTY-gold -0.10.
Crash: book-NIFTY 0.19, book-gold 0.55, book-gilt 0.04, book-MCX 0.03, NIFTY-gold 0.03, NIFTY-MCX -0.18.
NIFTY's 30 worst weeks, mean weekly return: NIFTY -7.06%, book -0.99%, gold +0.70%, gilt -0.25%, MCX +1.32%.
In crashes the book becomes largely a gold position. Equity, gold, rates and MCX trend stayed separate here.

## Live-safety checks
Nightly GTT reconcile with missing_stop reporting exists; orders use a registered static IP (Oracle tunnel); engine orders are LIMIT at most 10 a second not breached by design but not enforced by a limiter; nothing sets market_protection, relevant only to manual MARKET orders (to verify with Zerodha).
