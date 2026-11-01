# Centurion regime audit, 1 Nov 2026 (computed from the walk-forward book, live-like, 2008-25)

Source: `bond_stream.walk_forward_base()` through `diversifier_gate._live_book`, NIFTY 50 price index
(not total return, about 1.3 points a year below TRI; TRI starts 2012). 4,457 sessions. Holdout 2026 untouched.

## Headline
Live-like 2008-25: 13.3% CAGR, Sharpe 0.60 (excess of 6.5% rf), MaxDD -29.5%, invested 70% on average, daily beta to NIFTY 0.17.
Rolling 3-year: CAGR min 1.7%, median 17.2%, max 28.9%; median rolling 3-year Sharpe 0.83 (2012-25). Longest time under water 621 sessions.
Years: 3 of 18 above 35%, 8 above 17%, 4 negative. Worst 2008 -26.4%. Other losses: 2016 -4.4%, 2018 -8.3%, 2022 -9.2%.

## By causal state at the previous close (bull: NIFTY above 200-day and up over 63 days; bear: below and down; range: the rest)

| State | Days | Share | Book a year | NIFTY a year | Book vol | NIFTY vol | Beta | Invested |
|---|---|---|---|---|---|---|---|---|
| Bull | 2,715 | 61% | 19.1% | 11.5% | 11.3% | 15.7% | 0.35 | 78% |
| Bear | 1,073 | 24% | 4.7% | 10.2% | 10.7% | 29.7% | 0.04 | 49% |
| Range | 669 | 15% | 2.0% | 5.1% | 9.8% | 19.0% | 0.20 | 70% |

By period (book / NIFTY a year, beta, invested):
- 2008-12: bull 5.2 / 8.1, 0.21, 74%; bear 4.0 / -2.4, -0.03, 40%; range 1.5 / 2.5, 0.06, 54%.
- 2013-19: bull 19.5 / 9.0, 0.44, 75%; bear 2.1 / 29.4, 0.27, 54%; range 0.5 / -2.4, 0.39, 70%.
- 2020-25: bull 26.5 / 16.3, 0.53, 84%; bear 9.9 / 7.1, 0.08, 58%; range 3.9 / 15.1, 0.33, 85%.

Monthly capture: up months 54%, down months 15% (122 up, 94 down).

## By ex-post leg (NIFTY reversal of 20% down, 25% up)

| Leg | Dates | Book | NIFTY | Invested |
|---|---|---|---|---|
| Bear | Jan-Oct 2008 | -28.9% | -59.8% | 34% |
| Bull | Oct 2008-Nov 2010 | +52.6% | +143.9% | 62% |
| Bear | Nov 2010-Dec 2011 | -0.7% | -27.6% | 54% |
| Bull | Dec 2011-Mar 2015 | +109.3% | +95.0% | 68% |
| Bear | Mar 2015-Feb 2016 | -11.2% | -22.2% | 62% |
| Bull | Feb 2016-Jan 2020 | +45.3% | +76.1% | 75% |
| Bear | Jan-Mar 2020 | -6.3% | -38.3% | 68% |
| Bull | Mar 2020-Dec 2025 | +231.6% | +198.8% | 80% |

## Reading (mine, for the writer to test against the research)
- Defence is strong: every bear leg lost about half or less of NIFTY's fall, and much less in three of four.
- The weakness is the early bull: the 2008-10 rebound caught 37% of NIFTY's rise, and 2016-20 about 60%.
  Causal "bear" states carry NIFTY's strongest returns in 2013-19 (+29%), when the book sat at 54% invested with beta 0.27: rebounds.
- Range markets earn almost nothing: +2.0%, under the 6.5% rf, on 15% of days.
- The book already beats NIFTY in sustained bull states, +19.1% against +11.5%, but only at 78% invested and beta 0.35.
- Golden target arithmetic, ignoring volatility drag: 35% CAGR over a 6.5% rf is 28.5 points of excess. At 11% vol that needs Sharpe 2.6; at 20% vol the target's 1.4; at 25% vol 1.1, with a far worse tail.

## State of play the plan must respect
- Failed since the 31 Oct report: GT2B slower core; GT2C idle capital into NIFTYBEES (13.04/0.600 to 12.63/0.525); GT2E metals cap;
  S151A sector momentum; DYN1 rotation among book, gold and gilt; HAN1, REG1, KOH1, YZ1, VOL1, VOL1d.
- Leads: `confirm_days` 1 (Sharpe 0.768 against 0.704, lower in 2008-12); 252-day volatility ranks stocks better (forecast only, failed its t test).
- Queued with pre-registration: V3 forward gate, GT7 PEAD, MV1 Minervini bundle, INT1 intraday, GT2D quality veto and post-crash rebound rule, January re-baseline, A1 option overlay on paper, gilts and MCX on paper.
- Passive equal NIFTY/gold/gilt has Sharpe 0.59 before costs, against the walk-forward book's 0.57 on the same basis.
- Control coverage: paper has run rebalance, rank exit, stops and a sleeve exit, never a regime exit or the drawdown rule.
- Owner's rules: golden target stays the aspiration, first milestone Sharpe 0.8 and 17-18%; no structural leverage; no RL in the core; intraday only on evidence; MaxDD reported, never a gate; live needs backtest, walk-forward, OOS, 60 paper sessions, 5 dry runs; bounded tail 10-15% per overlay cycle; automated orders India only, US advisory and on hold.
