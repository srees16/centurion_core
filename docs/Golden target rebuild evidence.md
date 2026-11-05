# Build Centurion for 0.8, Not 1.4

No evidence supports Centurion reaching Sharpe 1.4 and 35% CAGR as a sustained result; the honest aim is to go live safely on a book that really earns about 0.54 today and to find out, slowly, whether 0.8 over a full cycle is real. The recorded live-like record for 2008-25 is 13.0% CAGR and Sharpe 0.60, but that figure credits a free sweep on idle cash. With a realistic sweep it is 12.4% / 0.54, and as paper and live actually run today it is 10.9% / 0.43. On 2013-25 the book already sits at the first milestone (16.9% / 0.88). The whole shortfall sits in 2008-12 (Sharpe -0.24), where NSE's own momentum index also shows zero alpha. The research in hand offers no proven new uncorrelated stream: my estimate for the whole new list is 0 to +0.20 Sharpe, with overlapping upper ends. Residual momentum, ranked first in one note, has failed three separate looks, including the 2008-12 test it was meant to pass, and the protocol lists more price signals in the core under "Never". The core has no trials left before January, and six claims wait on a budget with almost no PBO headroom. So the plan puts controls, fills and the idle-cash fix first, spends three core slots in January, and treats everything else as later. Nothing here can promise that live will match the backtest, or that the book cannot lose: twelve months of live data cannot separate 0.6 from 1.0, and only process evidence can be checked in year one.

**Basis.** Centurion figures come from the [1 Nov regime audit](research_notes/Golden%20target%20rebuild%20evidence/centurion_regime_audit.md), `docs/MEMORY_TRACKER.txt` (cited by item ID) and `docs/nse_engine_validation_plan.md`. "Live-like" means walk-forward 2008-25 with D2's haircut. Sharpe is excess of 6.5%. NIFTY in the state tables is the price index, about 1.3 points a year below total return. Arithmetic marked "my calc" uses CAGR ≈ 6.5% + Sharpe × vol − vol²/2. Anything marked "estimate" is judgement, not measurement. No backtest was run for this report; I read recorded results and code only.

## Systems that clear 1.4 and 35% run a different business

Only one record clearly exceeds the bar, and it is not a precedent. Medallion reported a mean of 39.2% net over 1988-2018 ([Cornell excerpt](https://www.scribd.com/document/470291821/SSRN-id3504766-pdf)), unaudited, and Guo and Liu argue its compounded pre-fee return was "probably under 35%" ([arXiv](https://arxiv.org/abs/2405.10917)). It lives on 50.75% trade accuracy ([Institutional Investor](https://institutionalinvestor.com/article/b1k2fymby99nj0/Famed-Medallion-Fund-Stretches-Explanation-to-the-Limit-Professor-Claims)) across thousands of bets, cheap leverage and capped capacity. **At that accuracy Sharpe 1.4 needs about 8,700 independent bets a year; Centurion makes 340 trades a year (tracker GT2B) across about 20 correlated names (my calc, symmetric payoffs).** Renaissance's own public funds made +8.5% and +3.2% in the year Medallion made +76% (same source).

Positional records sit far lower. Berkshire's 0.76-0.79 is the best long-run Sharpe among 30-year histories, on about 1.6x cheap leverage ([AQR](https://www.aqr.com/Insights/Research/Journal-Article/Buffetts-Alpha)). Man AHL Diversified shows 9.86% at 0.44 over 30 years ([factsheet](https://swissfunddata.ch/sfdpub/docs/fsm-2320_00_01-20260828-en.pdf)). Higher CAGRs came with ruinous drawdowns: Mulvaney about 18% at -61% ([AutumnGold](https://www.autumngold.com/Advisor/cta_profile.php?id=341)). Carver puts the practical ceiling at 1.0 however good the backtest ([notes](https://threadreaderapp.com/thread/1234306875379683328.html)). India offers single years, not persistence. Nifty200 Momentum 30 returned 35% or more in 8 of 21 financial years and still compounded 18.8%, mostly back-filled and before costs ([HDFC MF deck](https://files.hdfcfund.com/s3fs-public/Others/2026-07/HDFC%20NIFTY200%20Momentum%2030%20Index%20Fund_PPT%20%28July%202026%29.pdf)). Capitalmind's trailing five years went from 30.2% in June 2024 ([Capitalmind](https://capitalmind.in/blog/capitalmind-pms-surpasses-2000-crore-in-aum-demonstrating-exceptional-market-performance)) to 7.1% in October 2026 ([Capitalmind](https://www.capitalmind.in/portfolio-management-service/momentum)). Wright's "38% plus" 10-year figure is a backtest before costs with one month live ([Wright Research](https://wrightresearch.in/blog/complete-guide-factor-investing-wright-factor-fund-pms)), which is what a 35% claim usually looks like.

Backtests do not bridge the gap. Bank alternative-beta products fell from a median Sharpe of 1.20 to 0.31 live, and only 18 of 215 matched their backtest ([CXO Advisory](https://www.cxoadvisory.com/big-ideas/live-performance-of-alternative-beta-products/)). McLean and Pontiff find returns 26% lower out of sample and 58% lower after publication ([paper](https://Www.Gwern.net/doc/economics/2016-mclean.pdf)). Harvey and Liu find the multiple-testing haircut almost always above 50% for Sharpes below 0.4, and at most 25% above 1.0 ([paper](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF)).

Centurion's own arithmetic closes the question. Holding NIFTY with perfect foresight of every bear gives about 25%, and a true Sharpe of 0.6 tops out near 24.5% at any leverage (GT2 review). The book reached 35% in 3 of 18 years, and its best rolling three-year CAGR is 28.9%. Target vol is about 22% (my calc: 34.9% at Sharpe 1.4), and even a true 1.4 at 22% vol has about a 43% chance of a -30% drawdown in 18 years (Gaussian, so a floor).

**Why the arithmetic rules out the target.** Long-run compound growth is roughly CAGR ≈ rf + Sharpe × vol − vol²/2. Leverage raises vol; the middle term grows in step with it, but the drag term (vol²/2) grows with its square, so past some point more risk lowers growth. Growth peaks at vol = Sharpe, where CAGR = rf + Sharpe²/2:

| True Sharpe | Peak CAGR (at vol = Sharpe) | At today's ~11% vol, no leverage |
|---|---|---|
| 0.6 | 6.5% + 0.36/2 = **24.5%** (needs 60% vol) | 6.5% + 6.6% − 0.6% ≈ 12.5% |
| 0.8 | 6.5% + 0.64/2 = 38.5% (needs 80% vol) | 6.5% + 8.8% − 0.6% ≈ **14.7%** |
| 1.4 | 6.5% + 1.96/2 = 104.5% | 34.9% needs ~22% vol |

So at Sharpe 0.6 no amount of leverage reaches 35%. At 0.8 the ceiling sits above 35% only at about 80% vol, which means structural leverage, and that is ruled out. Unlevered, CAGR is Sharpe × the book's own vol plus the risk-free rate: mid-teens, not 35%. The target needs a better Sharpe, not more leverage (my calc; Gaussian, continuous rebalancing, costs of leverage ignored, so these are upper bounds).

| Golden leg | Target | Live-like 2008-25 | 2013-25 | Reading |
|---|---|---|---|---|
| Sharpe | ≥ 1.4 | 0.60 (0.54 realistic sweep) | 0.88 | Out of reach as a sustained result |
| CAGR | ≥ 35% | 13.0% | 16.9% | Out of reach without leverage the owner has ruled out |
| Calmar | ≥ 1 | 0.44 | 0.81 | Reachable in some windows; not across 2008 on evidence so far |
| PBO | ≤ 30% | 29.2% on excess returns (N 61) | n/a | Passes with no headroom; 51.2% raw |
| DSR (reported) | ≥ 0.95 | 0.42 walk-forward at N 61; 0.13 at 1,000; 0.05 at 7,000 | 0.94-0.96 backtest | Effective N is unknown, and it decides the answer (GT2A) |

My judgement, using the earlier prior width of 0.3 on true Sharpe: about a 25% chance that true full-cycle Sharpe is 0.8 or more if the mean is 0.60, about 19% if it is 0.54, and under 1% for 1.4. Those are opinions about a distribution, not measurements.

The milestone has a hidden tension worth stating. Sharpe 0.8 and 17-18% needs about 15% vol. At today's 11% vol, Sharpe 0.8 gives 14.7% (my calc). The book runs 11% at 70% invested, so 15% vol implies roughly 95% invested if vol scales linearly (estimate). With no structural leverage and a gate that stays, **expect about 14-15% CAGR even if 0.8 is reached on full-cycle data**, unless deployment rises in risk-on states.

## Centurion defends well, lags rebounds and earns little in ranges

### Realistic returns and exposure by state

Three bases describe one book, and the notes appear to conflict only because they mix them. `diversifier_gate._live_book` adds idle cash at the liquid rate and subtracts 3 points a year on invested capital. The tracker's 13.2% / 0.62 (LN-T28b) has neither. D2 alone takes it to about 11% / 0.43, and crediting a free sweep takes it back to 13.0% / 0.60. So plan 5at's "D2 is inside" and the survival note's "D2 means 0.43" are both correct.

| Basis, 2008-25 walk-forward (LN-T29d) | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| D2 plus a cost-free sweep (the recorded base) | 13.0% | 0.60 | -29.5% |
| D2 plus a realistic LIQUIDCASE sweep at 3 bp | 12.4% | 0.54 | -29.9% |
| D2 and no sweep, how paper and live run today | 10.9% | 0.43 | -33.0% |

The audit's 13.3% headline sits 0.3 points above the tracker's 13.04%. I did not reconcile that gap; the state tables below use the audit, and comparisons use the tracker.

| Causal state at prior close | Share of days | Book a year | NIFTY a year | Book vol / NIFTY vol | Beta | Invested |
|---|---|---|---|---|---|---|
| Bull (above 200-day, up over 63 days) | 61% | 19.1% | 11.5% | 11.3% / 15.7% | 0.35 | 78% |
| Bear (below and down) | 24% | 4.7% | 10.2% | 10.7% / 29.7% | 0.04 | 49% |
| Range (the rest) | 15% | 2.0% | 5.1% | 9.8% / 19.0% | 0.20 | 70% |

This state definition sorts risk well: NIFTY vol nearly doubles from bull to bear. It does not sort return, because the bear state holds rebounds. Book and NIFTY by era tell that story. In causal bear, 2013-19, NIFTY made +29.4% a year while the book made 2.1% at 54% invested. In bull, 2008-12, the book made 5.2% against NIFTY's 8.1%. In bull, 2020-25, it made 26.5% against 16.3%.

Ex-post legs confirm it. The four bear legs lost 48%, 3%, 50% and 16% of NIFTY's fall (my calc from the audit), so the audit's "much less in three of four" is two of four, with the other two near half. The rise capture was 37% in 2008-10 and 60% in 2016-20, but 115% in 2011-15 and 116% in 2020-25. **Participation failed in two of four bull legs, the rebound after 2008 and the 2016-20 run, and worked in the other two.** Range earns 2.0%, under the 6.5% risk-free rate. GT2A's ex-post range split (a third of days) shows why: costs of -3.0 points, the worst of any state, 8.1 regime flips a year against 3.9 in bulls, and 27 regime sells a year against 6.

### Alpha and signal quality

**Stock-only alpha against NIFTY plus gold, net of costs and D2, is +6.0% a year in 2013-25 (t 2.03), +3.6% in 2008-25 (t 1.38) and -5.8% in 2008-12 (t -1.24); the whole book against the same hurdle is +3.2% (t 1.13) (GT2A).** Equal NIFTY, gold and gilt shows Sharpe 0.59 before costs against the walk-forward book's 0.57 on the same basis (REG1), so across the full cycle the book has not clearly beaten a passive mix.

Signal quality rises with horizon: rank IC 0.037 at 5 days, 0.051 at 21, 0.079 at 63. GT2A settled the earlier puzzle about low recent IC and high recent alpha. In 2020-25 the whole universe beat NIFTY plus gold by 11.3%, so spreads were narrow while alpha was high. The top 20 less the universe earned 19.7 points in 2013-19 and 8.3 in 2020-25. Ranks 21-40 match the top 20 (+13.7% against +13.2% gross). Over 63 sessions sold names keep beating the universe: rank exits +2.57% (t 4.6), rebalance sells +3.09% (t 3.2), against buys +3.80%. Regime exits sit at +0.88% (t 0.6) and stops at -1.07% (t -0.6), so stops are not shown to remove losers.

In 2008-12 ranks gave no alpha, and NSE's index shows the same shape. Nifty200 Momentum 30 compounded -1.15% a year against -1.08% for its parent over 2008-12, on a calculation from the whitepaper's yearly excess figures ([NSE](https://www.niftyindices.com/docs/default-source/indices/nifty200-momentum-30/momentum-strategy-whitepaper_2026.pdf)). The mechanism is known: after declines past winners can have betas below 0.5 while loser betas exceed 3, and the March-May 2009 loser decile made +163% against +8% for winners ([Daniel-Moskowitz](https://www.nber.org/system/files/working_papers/w20439/w20439.pdf)). A rank tweak cannot fix a market-wide effect easily. To lift 2008-25 Sharpe to 0.8, 2008-12 would need about +0.48 against -0.24 (my calc, simple time weighting, which reproduces 0.56 against 0.60).

### Risk management

The gate cuts risk; it does not pick days. NIFTY earned 7.6% a year with the gate in and 27.0% with it out, with rotation p 0.94, while worst weeks improved from -7.1% to -2.3% (MIT1). Removing the regime exit gives 14.9% / 0.64 / -40.1% against 15.6% / 0.70 / -28.3%, in sample (GT2A). Metals gave return, not crash defence: with metals out, the book is 10.8% / 0.46 and NIFTY's 30 worst weeks barely change, -1.12% against -0.99%. That leaves about a third of return on assets chosen knowing they survived, a hindsight risk GT2E kept knowingly after the 35% cap failed. Paper has exercised rebalance, rank exit, stops and a sleeve exit, but never a regime exit or the drawdown rule. The real weakness in risk management is therefore unproven code paths, not design.

## The tracker settles the residual momentum conflict against a retest

`regime_adaptive_exposure.md` ranks residual momentum first; `new_india_alpha_sources.md` says it was rejected. Both are partly right, and the note ranking it first is the one that fails.

| Date | What was tested | Result |
|---|---|---|
| 17 Sep (plan Stage F item 5) | Incremental rank IC on the deployed universe, 2013-25, pass t ≥ 2.5 | Passed. The information is real |
| 17 Sep, same | Added as a signal group at fixed 0.2 and 1/3 weights, rule +0.10 Sharpe | Failed. Sharpe -0.04 / -0.09 (my mapping from the plan's listing order), CAGR lower; no walk-forward |
| 13 Oct (R6) | R21A rules as signal groups; not run | Dropped by owner. It cites dilution of -0.00 to -0.25 across residual momentum, 52-week high, low beta, delivery, low vol and R21A |
| 31 Oct (GT2D look) | Group at a third, live-like 2008-25, 20 / 30 / 40 names | 0.685 / 0.658 / 0.658 against trend-only 0.697 / 0.750 / 0.768. 2008-12: 0.025 / -0.121 / -0.069 against 0.037 / -0.017 / 0.037 |
| 1 Nov (S151 L1) | Book's form, market factor, no intercept | 15.37% / 0.698 against 15.48% / 0.697; 2008-12 -0.010 against 0.037; lower at 2013-19, higher at 2020-25 |

52-week high passed the same screen and cost -0.00 / -0.12 Sharpe. In GT2D it read 0.717 at 20 names but lowered 30 and 40 names, and it hurt 2008-12 most of all (-0.197 at 20 names). GT4 is a separate result: a value and quality sleeve that failed at correlation 0.56-0.63. R6 was a dropped run, not a test, so the note's citation of it needs that qualifier.

**What was not tested:** residual momentum replacing the rank outright. Every test blended it at 0.2 or 1/3, which is the regime note's distinction. Centurion's form follows Blitz (36-month regression, 11-month window, skip-month, scaled by standard deviation) but on the equal-weight market only, not three factors (`signals.residual_momentum_raw`).

**Is a retest justified? No, not in January.** The note's case is mechanism: it should fix 2008-12. Every available run shows 2008-12 lower with it, at each width and form. The supporting literature is largely long-short and has no Indian long-only test ([Hanauer-Windmüller via Quantpedia](https://quantpedia.com/three-methods-to-fix-momentum-crashes/); [Blitz et al](https://econpapers.repec.org/RePEc:eee:empfin:v:18:y:2011:i:3:p:506-521)), and NSE's index suggests the 2008-12 gap is market structure. The protocol also lists "more price signals inside the core" under Never, so a retest needs an amendment. If the owner wants the question closed, the cheap route is a one-look sole-rank read on fixed settings, spending no trial, with a pre-stated rule: at least +0.05 Sharpe on 2008-25 and not lower in any of the three eras, at twice costs as well. Only a clearance could then compete for a slot. I would not spend even that before the January verdicts.

**Slower rebalancing, ranked second, also falls.** A monthly rebalance added nothing in GT2A. R7's buffer gave -5% turnover and flat CAGR. GT2B's slower core failed, at 13.19% / 0.593 against 13.04% / 0.600, with 2008-12 at -0.32 against -0.24. Rank 4, volatility scaling, runs into the same list: book-level targeting is under Never, after all 9 folds chose it off on 17 Sep.

**Core trial budget: zero before January, and the docs are stale.** GT2B spent "the core's last configuration before January" on 31 Oct. The tracker's queue header (lines 815-819), the protocol's table and the alpha note's "1 core trial left" all predate that. Six claims now wait after V3: MV1, GT7, GT9, GT11 and GT2D's two slots (quality veto, rebound rule). The sector-momentum claim is gone (S151A failed). No document caps January in numbers; the real cap is PBO ≤ 30%, at 29.2% with the family "at the PBO bar".

**Two listed "new" items are already queued, both in GT11:** the ASM/GSM exclusion and the promoter pledge exclusion (tracker line 1228). They are one core configuration, not new sources. I read "two" this way because they are the only rows in the alpha table named in the queue. Three overlaps sit nearby. Slower rebalancing is GT9. The half-step is the natural occupant of GT2D's rebound slot. Revision breadth overlaps MV1's earnings screen and GT7. Separately, index-inclusion longs and earnings drift appear in plan 5at's "Not doing", yet GT7 stays queued, an inconsistency for the owner to settle.

GT7 has a second problem. As written it is a fourth forecast group at 1/3 weight, which is exactly what the GT2D look says not to add ("no signal group"), and Indian evidence shows no long-side drift ([Gupta and Dhusia](https://journals.vutbr.cz/index.php/trends/article/view/541)). Plan 5aa allows dated clarifications until a run, so it can be reshaped as a veto or withdrawn.

## Every gap, ranked by value-add

Sharpe and point figures are estimates unless marked measured. They overlap and do not add. Values are measured on the recorded 0.60 base unless stated.

| Value-add | Gap | Evidence | Size | Action |
|---|---|---|---|---|
| **High** | No sweep on idle cash in paper or live | LN-T29d, measured | +1.5 points, Sharpe 0.43 to 0.54 on the live path; closes a gap, adds nothing to 0.60 | Settle Q5, cost model 5 |
| **High** | Regime exit and drawdown rule never run on the order path | Audit; 20% drawdown in the next 29 sessions is about 0% at Sharpe 0.43 | Protects the book; no return | Replay through a stub broker |
| **High** | No session order and notional cap; reconcile alerts but does not block | Knight: 4 million orders for 212 customer orders in 45 minutes, 97 warning emails unactioned ([SEC](https://www.sec.gov/news/press-release/2013-222)); GT2H built only 10 a second | Bounds a bug's loss | Cap, and a break blocks next-day buys |
| **High** | Entry fills under the changed open are unmeasured | Limit buys may miss gap-up names; relayed rules below | Unknown; could eat the live edge | Fill-rate-by-gap audit |
| **High** | Slow implementation leak invisible | CUSUM catches -0.5 sd in about 39 sessions, but a 0.1 sd shift alarms 3% in 60; a 30 bp gap sd hides about 19% a year at 0.25 sd (note's calc) | Protects the live excess | Cumulative gap band at 60, 120, 250 sessions |
| **High** | Decisions made pre-tax against stale hurdles | At a 30% slab, 12.4% nets 8.5-9.8%; a 12% passive holder nets 11.0%. `paper_trade_strategy.md` §7 still says Sharpe ~1.0 | Changes what "worth running" means | After-tax line, fix the docs |
| **Medium** | Deployment in risk-on | 78% invested in bull; ranks 21-40 match the top 20; E1 refill +0.07 Sharpe (p 0.08) in sample, never walk-forwarded alone | +0.5 to +1.5 points, 0 to +0.05 Sharpe. Upper bound +1.3 points from 78% to 90% (my calc) | January core slot |
| **Medium** | Rebound lag in risk_off | 2013-19 bear state NIFTY +29.4% at 54% invested; R11 failed; GT2C failed | 0 to +1 point, Sharpe -0.02 to +0.05; a bound, since 25% of book in up to 24% of days | Half-step overlay, one trial |
| **Medium** | Non-price information | Revision breadth: 8%+ excess CAGR gross, 257% turnover, to 2016 ([S&P](https://www.indexologyblog.com/2018/02/21/do-earnings-revisions-matter-in-india/), snippet) | +0.05 to +0.15 if point-in-time data exists; 0 if not | Answer the data question first |
| **Medium** | Retail-quota tenders | Acceptance 14-21% in TCS 2022 ([5paisa](https://test.5paisa.com/blog/tcs-buyback-offer-opportunity-for-retail-investors)); value peaks at Rs 3-30 lakh | 0 to +0.10, only inside that capital range | Hand table, no code |
| **Low** | Exclusion filters (GT11) | GSM -6 to -7% CAR on 128 observations ([NSE-NYU draft](https://archives.nseindia.com/research/content/nse_nyu/NidhiAggarwal_Rules_vs_discretion_Impact_of_trading_rules_on_market.pdf)); mostly micro-caps | 0 to +0.05; protective | Hit-rate count before spending a slot |
| **Low** | Option overlay A1G | Measured +0.015 Sharpe, 13.04% to 13.23%, in sample; capped at 10% | About +0.2 points | Finish paper |
| **Low** | MCX trend, gilts | MCX-B Sharpe 0.25, gilts 0.01; book + 25% gilts 11.7% / 0.61 | About +0.03 by arithmetic | Stay on paper |
| **Low** | Range-state churn | Costs -3.0 points on a third of days, and 49% of gate switches reverse | +0.3 to +0.7 points; confirm_days 1 is t 0.6-0.9 noise and goes the other way | Later, one hysteresis rule |
| **Low** | Index deletion, demergers, promoter buys, rating downgrades, IPO day 29-33 | In-sample, gross, or post-2020 only | 0 to +0.08 each | Event studies, later |
| **Unknown** | Intraday (INT1) | N 0; Rs 50,000 approved; nothing in this research supports a stream | Unknown | Stays gated |
| **None** | Residual momentum, 52-week high, low beta as groups | Table above | Negative | Closed |
| **None** | Sector momentum | S151A: 0.764 was survivorship, 0.702 on a fuller map | Zero | Closed |
| **None** | Idle capital into NIFTY, metals cap, slower core | GT2C 0.525, GT2E 0.581, GT2B 0.593 against 0.600 | Negative | Closed |
| **None** | Leverage, REITs, single-stock shorts | L7 13.9% / 0.45 against 13.2% / 0.62; GT14; U28 | Negative or closed | Closed |
| **None** | Turnover cuts alone (GT9) | GT2A, R7, GT2B | About zero | Withdraw the slot |
| **None** | HMM, jump model | S&P HMM 8.5% / 0.54 against buy-and-hold 10.2% / 0.48, with HMM below market on DAX and Nikkei at a 5-day delay ([Shu-Yu-Mulvey](https://arxiv.org/pdf/2402.05272)); about 5 Indian bears | 0-2 points ceiling, unattainable to prove | Do not build |
| **None** | Concentration, high beta, sector rotation | Sector edge 2.3% with perfect foresight, gross, not significant with realistic error ([CXO](https://www.cxoadvisory.com/?p=2458)); Sharpe stops rising past about 25% of the universe ([Institutional Investor](https://www.institutionalinvestor.com/article/b1f4qr0pyxy45k/Research-Affiliates-Says-This-Is-the-Best-Way-to-Do-Factor-Investing)) | Negative | Reject |

Three earlier "highest" or "best" labels are withdrawn by results since 31 Oct: turnover as a lever, idle capital at +2 to +3 points, and REITs as the best new stream.

## New alpha is thin, and timeframes add robustness, not breadth

To move 0.60 to 0.80 with one stream takes an own Sharpe of 0.53 at correlation 0, 0.69 at 0.3 or 0.76 at 0.5 (formula from the alpha note, checked). A stream at 0.3 and correlation 0.3 reaches 0.61. **GT4's surprise of 0.56-0.63 against a hoped 0.4-0.5 suggests planning high on any equity-side correlation.** Without a short leg, event trades carry market risk while capital waits.

| Source | Status | Who pays | Correlation normal / crash | Sharpe add | Gate problem |
|---|---|---|---|---|---|
| Analyst revision breadth, veto form | New, data-blocked | Slow updating | 0.4-0.6 / 0.4-0.6 (est) | +0.05-0.15 (est) | No free point-in-time data found |
| Retail-quota tenders | New | Reserved quota, cannot scale | 0.1-0.3 / 0.3-0.6 (est) | 0-0.10 (est) | Semi-manual; buyback tax changed April 2026 |
| Index deletion rebound | New; inclusion longs are in "Not doing" | Date-bound funds | 0.4-0.6 / 0.6+ (est) | 0-0.08 (est) | Under 20 events a year (est); auction may shift flows |
| Composite demergers | New | Forced selling, not evidenced | 0.3-0.5 / 0.6 (est) | 0-0.03 (est) | Return sits at announcement |
| Rating downgrade, IPO day 29-33 filters | New | Holders forced | Filter | 0-0.05 with GT11 (est) | Rarely hits a liquid book |
| Promoter buying | New | Insiders, outsiders earn nothing pre-2015 ([NITK](https://idr.nitk.ac.in/items/2e0597cb-33a1-4f4d-bfd2-6970ede2914a/full)) | 0.2-0.4 / 0.5 (est) | 0-0.05 (est) | Feed starts Oct 2020 |
| ASM/GSM, pledge | **Queued: GT11** | Rule-driven | Filter | 0-0.05 (est) | GSM lists start 2017 |
| MCX trend | On paper | Trend premium | 0.10 / 0.03 measured | About +0.03 | Passed nothing as a family; last slot held |
| Gilts | On paper | Duration | 0.07 / 0.04 measured | Risk reducer | Bonds closed 3 of 3 |
| A1G monthly options | On paper | Vol premium | About 0.3-0.5 / 0.7+ (est) | +0.015 measured | 10% cycle cap |

The honest count is that equity-side ideas are filters and tilts, and true separation stays outside equities in gold, MCX trend and gilts, which earn too little. **My estimate for all new rows together is 0 to +0.20 on Sharpe, not their sum, and it would not by itself reach 0.8.** Rejected as sources: tax-loss March, SIP and month-start flows, dividends, QIP, holdco discount, MTF data, circuit-band changes and participant OI. Bulk deals (BLK1) and option-implied stock selection (IVX1) are closed.

Multi-timeframe diversification favours slower signals and ensembles, not faster trading. Across 84 futures, median Sharpe moved only from 0.55 at a 5-day span to 0.33 at two years, far from the square-root law ([Sepp and Lucic](https://arxiv.org/html/2607.19497v1)). Carver's fast crossover has a law-implied 1.68 and an actual 0.07 ([qoppac](https://qoppac.blogspot.com/2020/04/how-fast-should-we-trade.html)). Long-only TSMOM's best lookback is about 9 months and very imprecise, needing about 250 years of monthly data to be 80% sure of beating buy-and-hold ([CXO](https://www.cxoadvisory.com/?p=34113)). GT2B adds the local result: slower spans paid in 2020-25 (1.13 to 1.40) and cost in 2013-19 (0.64 to 0.45), so span choice is era choice. The one transferable structure is asymmetric: in Goulding et al the fast signal gets 0% weight in corrections and 58% in rebounds, gross and US ([CXO](https://www.cxoadvisory.com/momentum-investing/handling-reversals-in-price-trend-direction/)). Fast signals help re-entry and hurt exit. That is the design logic of the half-step, nothing more.

## The rethink: controls first, three core slots, then events

The fundamental change is to stop using backtests as the search engine. The record shows why. GT2A's exit rank 60 read 17.4% / 0.82 in sample and vanished in walk-forward. Sector momentum's 0.764 was survivorship. With two correlated 18-year books, the standard error of a Sharpe difference is about 0.075-0.105, so **a true +0.10 improvement passes a t ≥ 2 test only about 16-26% of the time (my calc)**, and a true +0.05 about 7-10%. Most January trials will therefore fail even when the idea is sound, and any that pass deserve suspicion. Trials should be few, chosen on prior evidence, and judged live.

The pipeline has five stages. Signals: a free look with a read rule, then one pre-registered configuration. Backtest and walk-forward: 18 folds, 2008-25, live-like, on the realistic sweep basis once cost model 5 exists, reporting the passive hurdle. OOS: 2026 stays sealed, once per family. Paper: 60 sessions judged on tracking. Live: 5 dry runs, then the Rs 3 lakh canary, with rungs advancing on fills.

### Do now, no trial needed (2 Nov to 14 Dec)

| When | Step | Pass rule | Stop |
|---|---|---|---|
| By 6 Nov | Send the Zerodha ticket: margin at 09:00, Q4, Q5, Q6, market_protection, token expiry, BZ, ETF AMO time | Written answers | No reply by 30 Nov: model the worst case, no same-morning funding |
| By 13 Nov | Confirm the relayed items in NSE circulars (below) | Each marked confirmed or changed | Unconfirmed items stay "assumed adverse" |
| By 20 Nov | Fix stale figures: core budget in tracker and protocol, `paper_trade_strategy.md` §7, D2 wording, the three-basis table | One stated base everywhere | None |
| By 20 Nov | Fill audit on sessions since 7 Sep: promote LN-T21b; fill rate by open-gap tercile | Top-gap tercile within 10 points of the rest (proposed) | Fail: AMO entry re-think before any rung |
| By 27 Nov | Session cap (about 30 orders, 60% of book a session, proposed), reconcile break blocks buys, intended-against-sent check, secondary static IP on another provider, kill-switch semantics, release-day freeze | Each proven on a stub broker | Any open: dry runs do not count |
| By 11 Dec | Replay game day: 2008, 2020, locked circuit, gap-through-stop, broker down, runaway loop through the live code | Regime exit and drawdown rule fire, no duplicates, no reject storm | A failure blocks RA1's go-ahead |
| By 11 Dec | Pre-register cumulative gap bands, monthly universe rank IC (Tier B), after-tax line | Written before the first live session | None |
| By 11 Dec | Counts only: ASM/GSM hits on past holdings; a vendor quote on point-in-time consensus; chartered accountant view | Numbers in hand | Hits under 1% of buys (proposed): GT11 withdrawn, no slot spent |

These spend no family budget. Temporary tests are deleted after verification, as the standing rule requires.

### January budget (after RA1 on 14 Dec and V3 about 24 Dec to early January)

RA1 is first. ACCEPT with 5 clean dry runs opens the canary. WATCH is decided at 90 sessions, about 25 Jan. **REJECT stops all live work and the live config is not re-tuned.** E4 looks unlikely to clear V3, since its family reads 1.11 against 1.24 in walk-forward, outside U19's 0.05 band (my reading). Each slot below runs against whichever config V3 leaves.

| When | Step | Family, budget | Pre-registered pass | Stop |
|---|---|---|---|---|
| Week of 4 Jan | Re-baseline: cost model 5, LN-T30 locked circuits, OPS1A, for-cause exits; 2008-25 only, 2026 unread | None; new base, not a trial | Reproduces fold hashes before changes; changes itemised | Mismatch: fix before any trial |
| Week of 11 Jan | **J1 Refill exits alone**, walk-forward (only if V3 does not settle it) | Core, slot 1 of 3 | GT2B's rule: higher Sharpe than base, `bootstrap_p` ≤ 0.10, higher in 2008-12 | Fail: U20 dropped, flag removed. PBO over 30%: core closes |
| Week of 18 Jan | **J2 GT11 bundle** (ASM/GSM, pledge, plus downgrade if data is clean), only if the count clears | Core, slot 2 of 3 | Sharpe not lower on 2008-25 and in 2017-25, by dated addendum; DSR at N, PBO ≤ 30% | Fail: no code. PBO breach: stop |
| Week of 25 Jan | **J3 MV1**, as fixed in 5aa | Core, slot 3 of 3 | 5aa's gates: DSR ≥ 0.95 at N, beat reference, PBO ≤ 30%, t ≥ 2 | Fail: no code; O9 stays shut |
| By 5 Feb | **J4 Half-step**: risk_off and NIFTY 21-session return above 0, hold NIFTYBEES at 25%, weekly | GT2C's overlay family, 2 of 3 (not core, arithmetic on the recorded walk-forward) | Sharpe and Calmar not lower on 2008-25 and in 2 of 3 eras; overlay cycle loss within 10-15% | Fail: no code, and no third try |

J1 needs one change of record: U20 says drop refill if E4 fails, but E4 bundles exit rank 60, which failed in GT2B, so E4's fate says little about refill alone. J4 also departs from the tracker, which books the rebound rule as a GT2D core slot; as arithmetic on a recorded walk-forward it matches GT2C's own family, and it still counts as a trial there. The half-step's size comes from the cap, not data: 25% × a -45% leg is -11.25% of book (note's calc). Its prior is modest. R11 failed because it bought the April-May and August-September 2008 bear rallies, and GT2C showed NIFTY failing to beat the liquid rate in the states it held. It runs after the re-baseline because it competes with the sweep. A pass in any slot then goes to 60 paper sessions, so nothing reaches the live book before about April 2027 at the earliest, and any passer counts as a promotion trial.

Withdrawn from January: GT9 (evidence above), GT7 as written, and residual momentum. The quality veto waits because filings start in 2012, so it cannot speak to 2008-12, which is the failing era.

### Later, in order (February to December 2027)

| When | Step | Family | Pass | Stop |
|---|---|---|---|---|
| Late Jan to Feb | A1G, MCX, gilts: finish 60 sessions and dry runs | Own, already tested | G4 tracking and own gates | Fail: stays paper |
| Feb to Mar | Index deletion, NSE leg | Event study, N 1 | Net t ≥ 3 after removing momentum exposure | Fail: no code, MSCI leg never opens |
| Mar to Apr | Composite demergers | Event study, N 1 | Same | Same |
| Mar to May | Tender table by hand, at least 30 events, then a sleeve only if it clears | Event study, then sleeve family, 1 of 3 | Mean net per event at or above 2% after funding (the note's assumption), with a per-event size cap | Under 2%, or book past Rs 30 lakh with no fit: close |
| Once data exists | Revision breadth as a veto | One new core slot | Same as MV1; never a rank group | No point-in-time source: never run |
| After V3, per plan 5av | INT1A-E | Intraday, up to 3 | §5.1 gate, 60 paper, start at 5% or less | None pass: family closes, no code |
| After January verdicts | RF1 window change | n/a | Owner's decision below | n/a |
| 2027 onward | Canary 60 sessions, then rungs 6 / 12 / 21 / 30 lakh at 20 each; full size about July 2027 at the earliest | n/a | Costs, vol and turnover 0.7-1.3x, event minimums; no profit read | CUSUM steps down a rung |

## Going live on Kite: safe from ruin, honest about what stays unknown

### Blow-up: what must be true

The likely killers of an unlevered long-only book at Rs 3-30 lakh are an order bug, a stop that was never live, a limit nobody enforced, a dependency failure, or leverage added after a good year, not a low Sharpe. The arithmetic bounds the market side: a -13% no-exit day costs at most 13% unlevered, and only gross above about 1.15x breaks the 15% limit; on real days the book made -1.25% on 23 March 2020 and -7.59% on 4 June 2024 (GT2H). Days to exit at 5% of ADV peaked at 1.74 at Rs 30 lakh, so no ADV cap was built, which updates the note's open item.

"Cannot blow up" would still be false. Vakrangee fell through eight straight lower circuits with no buyers ([Business Standard](https://business-standard.com/amp/article/markets/vakrangee-locked-lower-circuit-for-eight-straight-day-falls-34-in-8-days-118040400398_1.html)). Karvy pledged clients' securities for its own borrowing ([Lok Sabha reply](https://eparlib.nic.in/bitstream/123456789/991819/1/AU2112.pdf)). NSE's investor protection ceiling is reported at Rs 35 lakh per investor ([Moneylife](https://www.moneylife.in/article/nse-increases-investor-claim-amount-to-rs35-lakh-from-rs25-lakh-per-claim/74911.html)). A GTT "does not assure execution" and may be cancelled on an ex-date ([Zerodha](https://zerodha.com/tos/gtt)). Zerodha's Console kill switch makes triggered GTTs reject ([Zerodha](https://support.zerodha.com/category/trading-and-markets/charts-and-orders/gtt/articles/why-was-my-sell-gtt-order-rejected)), so it must not be used with open positions. Centurion's own flag, which refuses buys and leaves stops running, is the right design. Static IP changes are limited to one a week, so a dead Oracle VM can stop API orders unless a secondary IP is already registered ([Zerodha](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/static-ip)). The notes did not audit the code on these items and neither did I, so "verify" means read the code.

Stress should test exit failure, not just price. The replay set needs history (2008, the 2018-20 grind, March 2020, 4 June 2024), correlated gaps through stops, locked names, band cuts mid-exit, gold-specific shocks (a -6% duty day, and an ETF premium collapse like January 2026's fall of nearly 10% against MCX's 4%, per [Business Today](https://www.businesstoday.in/personal-finance/investment/story/gold-silver-tumble-up-to-4-on-mcx-after-record-rally-etfs-slide-as-profit-booking-intensifies-513667-2026-01-30)), and infrastructure loss at the open. Losses are recorded, not gated, under the owner's MaxDD rule. The pass condition is that nothing behaves in an unplanned way.

### Market-structure items are relayed; AMO LIMIT entries may need a re-think

**To confirm against NSE primary circulars; none was read.** The note's NSE fetches all timed out, and its sources are relays. The tracker's own CAS1 and ST2 entries are also not NSE text.

| Item, as relayed | Source | Effect if confirmed |
|---|---|---|
| Pre-open rewrite from 7 Sep 2026: limit-only after 09:05, random close before 09:10, market orders have priority, algo market orders only in the first five minutes | Sister-exchange circular ([MSE](https://www.msei.in/SX-Content/Circulars/2026/August/Circular-19531.pdf)) | Limit AMOs may miss, skewed to gap-up names |
| Closing auction from 7 Dec 2026: 15:30-15:40 for all stocks; a second date, 8 Feb 2027, for limit carry | Zerodha bulletin ([link](https://zerodha.com/marketintel/bulletin/461085/changes-to-closing-auction-session-cas-from-december-7-2026)), [Bajaj](https://www.bajajbroking.in/share-market-news/sebi-cas-rules-2026-timings-vwap-and-f-and-o-changes-explained) | Closes change mechanism at go-live; unfilled DAY limits may fill at the close |
| ETF AMOs sent at 09:15, stock AMOs at 09:00 | Zerodha support ([link](https://support.zerodha.com/category/trading-and-markets/charts-and-orders/order/articles/what-is-amo-and-when-can-we-place-it)), one observation in ST2; support was wrong once before | A LIQUIDCASE or gold sell cannot fund that morning's stock buys |

If these hold, **the backtest's open-or-nothing fill no longer describes what an AMO LIMIT does**, which is the most direct route to live failing the backtest. Unfilled entries would cluster in names that gapped up, the ones the book most wanted. A DAY limit can also fill later in the session, and from February perhaps in the auction, unlike the model. The options are to keep LIMIT with a price rule set from measured fill rates, to use a protected MARKET AMO in the first five minutes (which depends on the unresolved market_protection answer, where a forum reply says a zero value is rejected, per [Kite forum](https://kite.trade/forum/discussion/comment/52171)), or to add a 09:20 second pass, which needs that day's login. Choosing among them is an execution change, not a signal trial, but it must be recorded, shadowed in G4 and reflected in the backtest's fill model before the canary. Do not pick one until the fill audit reports. The sweep's funding also depends on the ETF timing, so LN-T29 and cost model 5 hang on the same answer.

### More aggressive in uptrends, more defensive in downtrends

Aggression should mean deployment of stock selection, not beta. GT2C closed index exposure, leverage is closed, and concentration and high-beta tilts add tail without net return. The lever left is filling slots in risk-on from a rank list where 21-40 earns the same as the top 20, tested as J1 and in E4 on paper. No exposure rule improves net return for an unlevered long-only book on audited money: PTLC's 200-day rule returned 8.98% against the S&P's 14.00% over 11 years ([Pacer](https://www.paceretfs.com/media/ptlc.pdf)), one unfavourable draw with 2-3 bears, and real-time volatility management scored 0.42 against 0.46 ([Cederburg et al](https://www.lehigh.edu/~xuy219/research/COWY.pdf)).

Defence should stay as it is. Beta is 0.04 in the bear state, the regime exit earns its place, and the 20/30/35 rule is ruin insurance that fires late. Neither insurance rule beats buy-and-hold on mean ([Annaert et al](https://econpapers.repec.org/RePEc:eee:jbfina:v:33:y:2009:i:2:p:272-280)). Adding defence would worsen the book's one real weakness, rebound lag. Participation is worth more: an upper bound of 2.5 points from higher bull capture against 0.7 from lower bear capture (note's calc, ex-post regimes). The half-step is the only addition, and it is bounded.

### What can and cannot be verified

| Claim | Evidence available | When |
|---|---|---|
| Live book equals tested book | Position match, rejects, GTT coverage | From day 1 |
| Costs match the model | About 81 trades in 60 sessions gives SE near 7 bp at 60 bp per-trade sd (note's calc) | 60 sessions |
| Same risk shape | Vol and turnover 0.7-1.3x | 60 sessions |
| Signal still ranks | Monthly universe rank IC: t 1 to 2.3, supports but does not prove | 12 months |
| Sharpe above zero | At true 0.6, 17.2 years at 5% / 80%; 0.8, 9.7; 1.0, 6.2; 1.4, 3.2 (IID, a best case) | Years |
| Beats a passive 0.5 | True 0.8 needs about 40 years at correlation 0.8 | Effectively never |

Pre-commit to the fan now. At a true Sharpe of 0.6, a single year has a 27% chance of being negative, 14% of being below -0.5 and 35% of being above 1.0, and a +2.0 year moves a sensible posterior only from about 0.40 to 0.57. At Sharpe 0.6 and an assumed 15% vol, a 10% drawdown within 12 months has a 57% chance and 20% has about 5%. These are IID figures, so reality is worse. The longest time under water was 621 sessions. Kill on broken mechanism, scale on process, and review P&L at 36 months against the fan. Paper at 60 sessions proves behaviour and nothing about edge. RA1 stands as fixed, though it tests against the old Sharpe 1.16, not the honest base, so its ACCEPT will carry little information about 0.6.

## Decisions only the owner can make

| # | Decision | My recommendation |
|---|---|---|
| 1 | Which basis defines the milestone: full-cycle 2008-25, 2013-25 (the book is already there) or rolling three-year median (17.2%, 0.83, also there)? | Full-cycle on the realistic sweep, with others reported |
| 2 | Which three of six January claims run; withdraw GT9; reshape or withdraw GT7, given 5at's "Not doing"; keep refill alive past U20 | J1, J2, MV1; withdraw GT9; GT7 as veto or out |
| 3 | RF1 turns 2026 into test data and seals 2027 in January, but 5at and your rule keep 2026 out, and GT7 and MV1 are set to move to 2013-26 | Run January on 2008-25, and change the window only after the last verdict |
| 4 | Can short-history streams pass a shorter gate at lower size? GSM from 2017, PIT from 2020, filings from 2012 cannot meet 2008-25 | Yes for filters and capped event sleeves, with the gap stated, under a recorded exception like U38 |
| 5 | Waive tracker MIT1's "no exposure may be added" on the gate, for J4 only | Allow once, capped at 25% |
| 6 | Close GT8 and U36. The tracker still accepts 1.75x while your rule says no structural leverage | Close it, and reopen only by a new decision |
| 7 | Keep t ≥ 2 and DSR ≥ 0.95 for all, or use a "not lower" tier for risk and implementation changes? The base itself fails DSR at 0.42 | Strict for return claims, "not lower" for risk rules and fills |
| 8 | If the book cannot clear the after-tax passive hurdle, is the product a passive mix plus overlay? Which tax treatment? Buy consensus data? | Take a chartered accountant's view before the first full year; buy data only once a quote is weighed against +0.05-0.15 |

**Owner's decisions, 4 Nov 2026.**
- **Golden target, 5 Nov:** Sharpe ≥ 1.4 and CAGR ≥ 35% (and `paper_trade_strategy.md` §7's Sharpe ≥ 1.0, CAGR ≥ 20%) stay as aspirations the book aims to reach in the near future, not as gates; decisions are taken on the live-like basis above.
- **1, decided; revised 5 Nov:** the milestone is 2008-25 on the live-like as-run basis (no sweep, D2's haircut, buys funded only by free cash): 10.9% / 0.41 today (plan 5bt), with the 2013-25 and rolling three-year figures reported alongside. The sweep basis it first named no longer exists: both sweep designs failed (FND1, FND2) and LN-T29 is closed.
- **2, decided 5 Nov:** January runs J1 refill exits, J2 GT11 (only if its hit count clears 1% of buys by 11 Dec) and J3 MV1 as core slots, highest prior first, and J4 the half-step in the overlay family. GT9 and GT7 are withdrawn; the quality veto waits; J4 covers the rebound rule.
- **3, decided 5 Nov:** 2026 stays sealed. Every January trial runs on 2008-25; RF1 moves the window to 2013-26 only after the last January verdict.
- **4, decided:** yes. Filters and capped event sleeves fed by short-history streams may pass a shorter gate at a smaller size, with the gap stated, under a recorded exception like U38.
- **5, decided:** yes. MIT1's "no exposure may be added" rule is waived once, for J4 only, capped at 25%.
- **6, decided:** GT8 and U36 are closed. They reopen only by a new decision.
- **7, decided:** strict gates (t ≥ 2, DSR ≥ 0.95) for return claims; a "not lower" tier for risk rules and fills.
- **8, decided 5 Nov:** build the sweep (LN-T29) first; take a chartered accountant's view before the first full year; go live at the canary regardless; scale past it only if the as-run book beats the passive NIFTY and gold mix after tax on 2008-25 (revised 5 Nov: without the sweep, which is closed; plan 5bu's after-tax line is the test). If it does not, the product becomes a passive NIFTY, gold and gilt core held long-term, with Centurion a capped satellite of about 25% until it earns more.

## Conclusion

The research changes the question from "what gets Centurion to 1.4" to "which number is it really running at, and how would we know". Three figures look like one, 0.60, 0.54 and 0.43, and the distance between them is an operations problem with a Zerodha timing answer in it, not a research problem. That single fix is worth more on the live path than the whole list of new alpha, and it needs no trial. The evidence also inverts where the risk lies: the signal is real in 2013-25 and absent in 2008-12 for reasons NSE's index shares, so the regime weakness is probably a property of the market, while unexercised exits and an unmeasured open auction are properties of the system, and those can be fixed.

The sharpest constraint is statistical. With a detection threshold near +0.15 to +0.21 Sharpe and honest effects near +0.05, Centurion can no longer learn much from its own 18 years, however carefully the trials are registered. The golden target stays useful as a direction, but the next real information comes from small live capital and fills. If the 0.8 is real, it will show first as boring process passes, and only years later as a Sharpe.
