# Regime-adaptive exposure, leverage and blow-up prevention for a long-biased swing book

As of 31 Oct 2026. Context: Centurion (India, cash book, 6-ATR trailing stop, regime risk-off, 20/30/35% drawdown rule, 8% per-name cap, kill switch, live-like Sharpe 0.60; L7 levered 13.9% / 0.45 / MaxDD -40.3% vs unlevered 13.2% / 0.62, 2008-25).

**Evidence grades (the four the brief asked for, plus two for non-performance facts).**
`[AUDITED LIVE]` real-money record or index of live managers. `[ACAD-OOS]` academic with an out-of-sample test (still simulated money). `[BACKTEST]` backtest or in-sample only, including academic work with no holdout. `[MARKETING]` issuer or sponsor claim. `[RECORD]` regulator, court or official post-mortem (fact of what happened). `[PRESS]` secondary reporting.

**Effect tags.** `Sharpe+` raises risk-adjusted return. `DD-` cuts drawdown or tail without a proven Sharpe gain. `Sharpe0` no reliable effect.

**Retrieval caveat.** Most academic items were read from abstracts or secondary summaries, because PDFs were unreadable or returned 403 here. Where a figure comes from one secondary source, it says so. All arithmetic labelled "my calc" was done by hand under stated Gaussian assumptions: nothing was simulated and no Centurion code or data was read.

**Cross-cut verdict.**

| Tool | Sharpe | Drawdown | Best grade found |
|---|---|---|---|
| Trend overlay on an equity index | Small + at best, lower mean return | Clear - | ACAD-OOS, decays over time |
| Multi-asset trend as a stream | + but live 0.30, not 0.8-1.3 | + as crisis diversifier | AUDITED LIVE index |
| Vol targeting, equities | + in risk assets only, disputed OOS | - tail | ACAD-OOS, contested |
| Vol scaling of momentum | + large in backtest | - crash | Academic; no audited record found |
| HMM / regime models | Small + | - vol about 41% | ACAD-OOS, simulated only |
| Uptrend-only leverage | + in US backtest | Worse than unlevered timing | BACKTEST |
| Drawdown stops | Depends on momentum in returns | Mixed, can cut at the low | Academic theory + US data |
| Rolling index puts | Negative, about -8% a year drag | - but divesting did better | ACAD-OOS / marketing split |
| 1/N or simple weights | Beats optimised on estimation error | n/a | ACAD-OOS, strong |

## Time-series momentum and trend following

### Takeaway
Trend is real as a diversifying stream but far weaker than headline numbers: academic gross Sharpe of about 1.0-1.3 and simulated net 0.77 compare with 0.30 for the live SG Trend Index through Feb 2024, before a further drawdown of about 20%. The Huang et al critique weakens the asset-by-asset claim, not the diversified portfolio's record.

### Cited Findings
- `[BACKTEST]` Moskowitz-Ooi-Pedersen 2012: significant time-series momentum in all 58 liquid futures; 12-month profits positive for every contract; persistence 1-12 months, partial reversal after — [SSRN](https://papers.ssrn.com/abstract=2089463); [EconPapers](https://econpapers.repec.org/RePEc:eee:jfinec:v:104:y:2012:i:2:p:228-250).
- `[BACKTEST]` The headline portfolio Sharpe is quoted inconsistently: 1.31 from Quantpedia (derived from a t-stat) vs about 1.0 from a blog; the primary table was not retrieved — [Quantpedia](https://quantpedia.com/strategies/time-series-momentum-effect/); contradicted by [Menon Lab](https://themenonlab.blog/blog/time-series-momentum-hedge-fund-strategy-explained).
- `[BACKTEST]` Babu et al "Trends Everywhere": 1.17 gross Sharpe for the 58 traditional assets at 12 months, equal-weighted — [Scribd copy](https://www.scribd.com/document/568229115/AQR-Trends-Everywhere).
- `[BACKTEST]` Hurst-Ooi-Pedersen, Jan 1880-Dec 2013, 67 markets: gross 14.9%, net of 2/20 11.2%, vol 9.7%, net Sharpe 0.77 — [paper](https://oxfordstrat.com/coasdfASD32/uploads/2016/03/A-Century-of-Evidence-on-Trend-Following-Investing.pdf). Swedroe reports 0.76, and positive net returns in every decade, lowest 4.1% in the decade from 1919 — [ETF.com](https://www.etf.com/sections/index-investor-corner/swedroe-why-financial-trends-persist).
- `[BACKTEST]` A CXO summary of a related intrinsic-momentum study extended to 2016 reports average net Sharpe of 0.4; different construction, not directly comparable — [CXO](https://www.cxoadvisory.com/momentum-investing/trend-following-with-intrinsic-momentum-over-the-very-long-run).
- `[ACAD-OOS]` Huang, Li, Wang, Zhou (JFE 2020), 55 futures, 1985-2015: asset-by-asset regressions show little evidence in or out of sample; pooled t-stat fails bootstrap critical values; pooled slope biased upward by differing means and vol scaling — [SMU](https://ink.library.smu.edu.sg/lkcsb_research/6550). Returns are gross — [CXO](https://www.cxoadvisory.com/momentum-investing/intrinsic-time-series-momentum-does-not-really-exist/). Contradicted in part by Alpha Architect's replication, which reads the pooled evidence as fairly robust — [Alpha Architect](https://alphaarchitect.com/are-trend-following-and-time-series-momentum-research-results-robust/).
- `[ACAD-OOS]` Baltas-Kosowski: better vol estimation (Yang-Zhang) and a continuous signal cut turnover by more than a third without significant performance loss; TSMOM underperformed post-2008 on higher co-movement, addressed with a correlation-aware leverage term — [SSRN](https://papers.ssrn.com/abstract=2140091); [CME copy](https://www.cmegroup.com/education/files/demystifiing-time-series-momentum-strategies.pdf); [QuantConnect note](https://www.quantconnect.com/research/15272/improved-momentum-strategy-on-commodities-futures/).
- `[AUDITED LIVE]` SG Trend Index Sharpe 0.30 and MaxDD -20.7% from late 1999 to Feb 2024, as reported by Invesco — [Invesco](https://invesco.com/content/dam/invesco/emea/en/pdf/RRE_2024_Q2_NavigatingMomentum.pdf).
- `[AUDITED LIVE]` SG Trend fell 20.4% May 2024-May 2025, second-largest drawdown since 2000; 16 drawdowns over 10% since 2000; 2008 +20.9% vs MSCI ACWI -42.2%; 2022 +27.4% vs -18.4%; same window ACWI +18.3%; causes given: abrupt reversals, range-bound prices, lack of persistent trends — [Cambridge Associates](https://www.cambridgeassociates.com/insight/does-trend-followings-recent-struggle-signal-that-the-strategy-is-structurally-broken/). Virtus calls -18.7% over 12 months to Apr 2025 the index's worst ever — [AdvisorAnalyst](https://advisoranalyst.com/?p=167192), a ranking conflict driven by windows. The H1 2025 drop of 10.48% erased 2024's +2.6% — [Alternatives Watch](https://www.alternativeswatch.com/2025/07/01/trend-following-hedge-funds-loss-h1-societe-generale-index/).

### Inferences
- The gross-academic to live gap (1.0-1.3 to 0.30) is a 70-77% reduction, close to the 73% median backtest-to-live Sharpe decline found across bank alternative-beta products (see Leverage section). Samples and fee loads differ, so this is an order-of-magnitude comparison.
- For Centurion, budget a multi-asset trend stream at Sharpe 0.3-0.5 net, valued for positive crisis payoff, not as a Sharpe engine. Indian equity trend alone cannot replicate the 2008/2022 pattern, because that payoff came from being able to short and from many asset classes.
- Fix the turnover cheaply: continuous signals rather than binary, per Baltas-Kosowski. This is the most transferable finding for a swing book.

### Gaps
- Kim-Tse-Wald (vol scaling explains TSMOM alpha) was not retrieved; the search returned nothing and the budget ran out.
- The primary MOP Table 3 Sharpe, and a clean decade-by-decade decay series, were not retrieved. Hurst PDF was unreadable, so the crisis-table count is not reported.
- No study of time-series momentum on Indian assets was found.

## Trend overlays on equity indices

### Takeaway
Overlays mainly cut drawdown. Mean return is usually lower, Sharpe rises modestly in backtests, the advantage deteriorates over time and is overstated by data mining. Antonacci's GEM regressed toward the median out of sample, and no audited live record was found for it.

### Cited Findings
- `[ACAD-OOS]` Faber's 10-month SMA is applied to five asset classes and tested on four other markets to address data mining — [paper copy](https://www.scribd.com/doc/116881446/SSRN-id962461).
- `[BACKTEST]` Rebalance date matters: faster response captures more of the move but with more whipsaw and cost — [Newfound](https://blog.thinknewfound.com/2013/08/the-luck-of-the-rebalance-timing/). The base rule shows substantial whipsaw; ±5% bands are the tested fix — [Systematic Investor](https://systematicinvestor.github.io/strategy/Strategy-TAA-BANDS).
- `[ACAD-OOS]` Zakamulin: reported timing performance carries considerable data-mining bias and ignores frictions; out-of-sample tests find it "highly overstated" — [abstract](https://fitforthem.unipa.it/rep:6533b82ffe1ef96bd1295a6c).
- `[ACAD-OOS]` Zakamulin long-run: mean returns below buy-and-hold, Sharpe generally higher, and odds of beating the market rise with horizon — [CXO](https://www.cxoadvisory.com/technical-trading/market-timing-with-moving-averages-over-the-very-long-run). Performance deteriorates over time; only MOM beats the market in over 50% of 5-year windows; support comes mainly from 1860-1942 — [CXO](https://www.cxoadvisory.com/?p=26981). The 0.25% one-way friction used may be too low — [CXO](https://www.cxoadvisory.com/technical-trading/long-run-moving-average-horse-race-for-timing-the-u-s-stock-market).
- `[BACKTEST]` `DD-` S&P 500 spent 69% of recession periods below its 200-day average vs 19% of expansions, so the average works more as downside protection — [the7circles](https://the7circles.uk/?p=36245).
- `[BACKTEST]` Antonacci GEM was tested 1974-2011, then extended to 1950-2018 giving two holdout windows — [Hedge Fund Alpha](https://hedgefundalpha.com/strategies/global-equity-momentum-a-craftsmans-perspective/). ReSolve tested 324 lookback combinations and argues for ensembles; a summary of that work reports the original spec regressed to the median out of sample, labelled simulated — same source (figures not in the retrieved excerpt).
- `[ACAD-OOS]` Clare et al (2016): trend following improved risk-adjusted performance vs buy-and-hold and beat risk parity; risk parity adds little within asset classes — [EconPapers](https://econpapers.repec.org/RePEc:eee:beexfi:v:9:y:2016:i:c:p:63-80); [working paper](https://ideas.repec.org/p/yor/yorken/12-25.html). A later CXO note reports gross Sharpe 1.06 and MaxDD -4.9% for a five-asset combination; gross and secondary — [CXO](https://cxoadvisory.com/?p=21860).
- `[ACAD-OOS]` Asness-Ilmanen-Maloney: strong in-sample vs "weak and disappointing" out-of-sample value timing; adding momentum helps; test range 50-150% equity; value-add marginal; advice is to "sin a little" — [AQR PDF](https://www.Aqr.Com/-/media/AQR/Documents/Insights/White-Papers/Market-Timing-Sin-a-Little.pdf); [Newfound commentary](https://advisoranalyst.com/?p=87835).
- Note: Asness-Frazzini-Pedersen (2012) is not a trend-overlay paper. It is "Leverage Aversion and Risk Parity", covered under Portfolio Construction — [CFA Institute](https://rpc.cfainstitute.org/research/financial-analysts-journal/2012/leverage-aversion-and-risk-parity).

### Inferences
- A binary 0/100% overlay is the wrong shape. The "sin a little" range and the whipsaw evidence argue for a bounded tilt, for example 60-120%, with hysteresis bands.
- Centurion already holds three correlated exits (ATR stop, regime off, drawdown rule). Adding a fourth index overlay likely adds whipsaw, not protection. Test the marginal value of each by removal.
- A range-bound market is where these rules bleed: the SG Trend's 2024-25 cause list is exactly this. Centurion should report the regime filter's cost by market state, not only its total effect.

### Gaps
- Faber's trades per year, percent invested and post-publication numbers were not retrievable (PDF unreadable here). No authoritative whipsaw count was found.
- No retrieved study measures SMA overlays on the Nifty. Both India searches failed or returned junk, so whipsaw frequency in India is unknown; Centurion's own 2008-25 data is the better source.
- Allocate Smartly and Portfolio Visualizer records for GEM were not accessed.

## Volatility targeting

### Takeaway
Claims that survived: vol scaling reduces tail outcomes across asset classes and raises Sharpe in equities and credit, not in bonds, currencies or commodities. Claims that did not survive: Moreira-Muir's broad alpha as a tradable, real-time edge. Momentum-specific scaling is the strongest result, with two independent papers finding Sharpe roughly doubles, but it is unaudited and comes from long-short momentum.

### Cited Findings
- `[BACKTEST]` `Sharpe+` Moreira-Muir (NBER 2016, JF 2017): cutting risk when volatility is high gives large alphas, higher factor Sharpe ratios and sizable mean-variance utility gains across market, value, momentum, profitability, ROE, investment and currency carry, because volatility moves without a proportional change in expected return. The abstract gives no figures — [NBER](https://www.nber.org/papers/w22208).
- `[ACAD-OOS]` `Sharpe0` Cederburg et al (JFE 2020), 103 equity strategies: managed versions have higher gross Sharpe in 53 cases vs 50, and only 8 of 53 wins are significant; spanning alphas are positive for 77 of 103 but need full-sample foresight; real-time versions generally earn lower CER and Sharpe than unmanaged — [paper](https://www.lehigh.edu/~xuy219/research/COWY.pdf). Secondary: lower for 72 of 103; returns are gross — [CXO](https://www.cxoadvisory.com/?p=32455). Failure attributed to structural instability in the regressions — [Arizona](https://experts.arizona.edu/en/publications/on-the-performance-of-volatility-managed-portfolios/).
- `[ACAD-OOS]` Liu-Tang-Zhou (JPM 2019): the stock-market application has look-ahead bias; once corrected, MaxDD was 68-93% in almost all cases and it beat the market only in the financial crisis — [WashU](https://profiles.wustl.edu/en/publications/volatility-managed-portfolio-does-it-really-work/).
- `[ACAD-OOS]` A later paper confirms the look-ahead in Moreira-Muir and in Harvey et al, then proposes an implementable version needing often over 200% turnover a year and large time-varying leverage — [CORE](https://core.ac.uk/download/328853217.pdf).
- `[BACKTEST]` `Sharpe+ DD-` Harvey et al (JPM 2018): Sharpe rises for risk assets (equity, credit), linked to the leverage effect; negligible for bonds, currencies, commodities; extreme returns fall in all asset classes; helps 60/40 and risk parity — [abstract copy](https://sf.cufe.edu.cn/info/1198/5688.htm); [Duke](https://scholars.duke.edu/publication/1370354). US equity vol-of-vol falls from 4.6% to 1.8% — [Quantpedia](https://quantpedia.com/?p=830). This paper is also named in the look-ahead critique above.
- `[BACKTEST]` `Sharpe+ DD-` Barroso-Santa-Clara (JFE 2015): managing momentum risk virtually eliminates crashes and nearly doubles Sharpe — [UCP](https://ciencia.ucp.pt/en/publications/momentum-has-its-moments/); 0.53 to 0.97, 1927-2011, single secondary — [ETF.com](https://www.etf.com/node/94292.md). Sharpe about 1.7 after safe months vs 0.28 after risky months — [CFA Institute blog](https://blogs.cfainstitute.org/blog/2018/01/17/timing-the-market-momentum-and-beyond).
- `[ACAD-OOS]` `Sharpe+ DD-` Daniel-Moskowitz (NBER 2014): momentum crashes are partly predictable, clustering in panic states after market declines with high volatility and coinciding with rebounds; a dynamic strategy forecasting momentum's mean and variance about doubles alpha and Sharpe vs static momentum, robust across periods, international equities and other asset classes. This is cross-market robustness, not a live test — [NBER](https://www.nber.org/papers/w20439).

### Inferences
- Daniel-Moskowitz's crash comes partly from the rebound of past losers, which a long-only book does not short. Centurion's version of the risk is different: after a panic, a momentum screen holds yesterday's defensives and lags the rebound. That points to testing re-entry and rank rules after panic states, rather than importing the crash result.
- Centurion is long-biased Indian equity with momentum characteristics, the case where vol scaling has the best claim. Use it to reduce exposure when vol spikes, not to lever up in quiet markets, since the look-ahead and instability critiques hit the upside leg hardest.
- Use a fixed ex-ante target and estimate only on data available at the date. The cost of the look-ahead error is the difference between a 56% and a 68-93% drawdown in the critique above.
- The Barroso safe-vs-risky Sharpe gap (1.7 vs 0.28) is a long-short result, so a long-only book should expect less. Test it in the WF pipeline rather than importing the number.

### Gaps
- Exact alphas and Sharpe figures for Moreira-Muir and Daniel-Moskowitz are not in their abstracts; the full papers were not read.
- Bongaerts et al "Conditional Volatility Targeting" was not retrieved. From memory only, not verified: it argues for scaling down in high vol without scaling up in low vol. Do not cite until checked.
- Cederburg's 72-of-103 figure remains secondary; the PDF was unreadable here.
- No India vol-targeting study was found.

## Regime models: HMM, Markov-switching, breadth, VIX, credit

### Takeaway
Out-of-sample academic tests show regime models lower volatility and add small excess return, while forecast theory says small misclassification destroys the advantage. All evidence is simulated, none audited. Simple two-state models with a holding delay are the defensible form; complex ones are overfit risk.

### Cited Findings
- `[ACAD-OOS]` `DD- small Sharpe+` Bulla et al (US, Japan, Germany indices): profitable out of sample after costs, volatility down about 41% on average, excess return 18.5-201.6 bps a year — [MPRA](https://mpra.ub.uni-muenchen.de/21154/1/MPRA_paper_21154.pdf).
- `[ACAD-OOS]` Nystrup et al: adaptively estimated two-state Gaussian HMM beat a rebalancing benchmark after costs, realistic delay and no foresight — [DTU](https://www2.imm.dtu.dk/pubdb/pubs/6808-full.html). Later work moved from fixed decision rules to model predictive control — [EconPapers](https://econpapers.repec.org/RePEc:taf:quantf:v:18:y:2018:i:1:p:83-95).
- `[ACAD-OOS]` Kritzman-Page-Turkington (FAJ 2012): most Markov-switching studies only fit in sample; a simple model on turbulence, inflation and growth outperformed static allocation out of sample, especially for those avoiding large losses; arbitrary thresholds give false signals — [CFA Institute](https://rpc.cfainstitute.org/research/financial-analysts-journal/2012/regime-shifts-implications-for-dynamic-strategies-corrected). Independent thesis replication reports lower vol and better risk-adjusted returns — [UCP](https://repositorio.ucp.pt/bitstreams/0c68578b-5c05-421d-9778-8006d9744279/download).
- `[ACAD-OOS]` Shu-Yu-Mulvey (2024): statistical jump models beat HMM-guided and buy-and-hold over 1990-2023 with costs and a one-day delay; preprint — [arXiv](https://arxiv.org/pdf/2402.05272v1.pdf).
- `[ACAD-OOS]` Counter-evidence: Dacco-Satchell (1999) show a small misclassification of the future regime is enough to lose any advantage from knowing the right model — [IAOR](https://iaorifors.com/paper/30956). Bessec-Bouabdallah confirm by Monte Carlo that misclassification is the main error source — [working paper](https://econwpa.ub.uni-muenchen.de/econ-wp/em/papers/0503/0503018.pdf).

### Inferences
- A regime model is justified by volatility reduction, not by alpha. Expect 0-2% a year of excess at best, with the upper end less likely after Indian costs.
- Centurion's regime filter should be validated by false-signal cost, not by hit rate alone. Dacco-Satchell implies the deciding metric is how expensive a wrong call is.
- The Kritzman caution on thresholds cuts against VIX-above-X style rules. If India VIX is used, prefer a persistent state over a level trigger.

### Gaps
- Ang-Timmermann was not retrieved.
- Breadth, VIX and credit filter evidence was **not** retrieved (search budget exhausted); no finding is reported, which is different from "no evidence exists". India also lacks a long liquid credit-spread series on my understanding, unverified.
- No audited live regime-switched fund record was found.

## Leverage in uptrends and how much is safe

### Takeaway
Conditional leverage raises CAGR in US backtests, but no audited record supports it, and L7's own result (0.7pp more CAGR for 27% less Sharpe) looks like a book at its net-of-cost growth peak. Defensible up-regime ceilings are about 1.0-1.25x at Sharpe 0.6, 1.25-1.5x at 0.9, and 1.35-1.7x at 1.2.

### Cited Findings
- `[BACKTEST]` Gayed-Bilello: 2x S&P when above the 200-day SMA, T-bills below, Sharpe 0.51 vs 0.30 buy-and-hold, Oct 1928-Oct 2015, 1% leverage cost, switching costs ignored — [CXO](https://www.cxoadvisory.com/volatility-effects/leveraging-the-u-s-stock-market-based-on-sma-rules/). Unlevered 100-day 9.5%, 200-day 8.8%, 10-day 3.9% annualised, so fast rules lose — [Proactive Advisor](https://proactiveadvisormagazine.com/?p=9672).
- `[BACKTEST]` Daily-reset leveraged products do not multiply long-run returns by the leverage ratio — [FPA](https://www.financialplanningassociation.org/sites/default/files/2021-08/MAY08%20Leveraged%20ETFs%20A%20Risky%20Double%20That%20Doesnt%20Multiply%20by%20Two.pdf).
- Theory: Kelly leverage f* = r/s², Kelly risk target = expected Sharpe. Never exceed full Kelly; risk appetite is a separate cap — [Carver 2018](https://qoppac.blogspot.com/2018/06).
- Carver's example (SR 0.5, 10% vol, 10 years monthly): median leverage 5; 75% confident of not over-gearing gives SR about 0.29 and leverage 2.9; half Kelly is about the 22nd percentile, which varies with Sharpe — [Carver 2018](https://qoppac.blogspot.com/2018/06).
- Sharpe standard error about sqrt((1+0.5·SR²)/N); 10 years daily at SR 0.5 gives about 0.32, 95% interval about -0.1 to 1.1. At the 10th percentile leverage falls to about 0.5x, and his own 25% target equals half of Kelly at SR 0.5 — [Carver 2024](https://qoppac.blogspot.com/2024/12).
- `[BACKTEST]` Carver's book: absolute maximum Sharpe to expect is 1.0 whatever the backtest; run negative-skew strategies at half the risk — [quoted thread](https://threadreaderapp.com/thread/1234306875379683328.html).
- Half Kelly gives up no more than 25% of growth and limits overbetting risk; Thorp argues the reason is a tendency to overestimate edge — [Downey on Thorp](https://matthewdowney.github.io/uncertainty-kelly-criterion-optimal-bet-size). "Good and bad properties" paper: care is crucial in using Kelly — [EconPapers](https://econpapers.repec.org/bookchap/wsiwschap/9789811285530_5f0015.htm). Simulated chance of losing half: about 1% at half Kelly vs 8.4% at full Kelly over 700 bets at a 14% edge — [escholarship](https://escholarship.org/content/qt5mr5k8qj/qt5mr5k8qj.pdf), snippet attribution not fully confirmed. Full Kelly's chance of eventually halving is about 50% in the idealised model — [LuxAlgo](https://www.luxalgo.com/library/concept/kelly-criterion/).
- Merton share k = μ/(γσ²); assumes one risky asset, normal returns, CRRA utility — [Capital Spectator](https://www.capitalspectator.com/?p=21198). A drawdown constraint changes the answer: the optimal fraction falls as wealth falls below its peak — [Tom Owens](https://tomowens.substack.com/p/the-missing-billionaires).
- `[ACAD-OOS]` Suhonen et al: median Sharpe 1.20 in backtest vs 0.31 live across 215 bank alternative-beta strategies (73% decline); only 8.4% matched their backtest; complexity worsens the haircut by over 30% — [Alpha Architect](https://alphaarchitect.com/?p=36892); [summary](https://alphaarchitect.com/looking-at-alternatives-avoid-complexity-and-magical-backtests/).

### Inferences
All below are my calc, Gaussian, continuous, by hand. "Cash" is assumed; substitute Centurion's real vol and cash rate.

**L7 decomposition.** Sharpe and CAGR imply vol once a cash rate is assumed (CAGR ≈ arithmetic - σ²/2).

| Cash rate | Unlevered vol | L7 vol | Implied scale | Drag per unit of added exposure | Growth-peak leverage net of drag |
|---|---|---|---|---|---|
| 5% | 15% | 29.5% | 2.0x | 5.2% | 1.8x |
| 6% | 13% | 24% | 1.85x | 4.8% | 1.9x |
| 7% | 11% | 19.7% | 1.8x | 4.2% | 2.2x |

- L7 therefore looks **at or near full Kelly on its net edge**: all extra risk, almost no extra growth. That matches the observed +0.7pp. The marginal Sharpe of the added risk is about 0.25 (2.7pp of excess for 11pp of vol) against 0.62 for the base.
- Frictionless scaling would give about 18% CAGR. The roughly 4-5% a year per unit of added exposure is the number to explain: futures basis and roll, stock-futures slippage, or leverage being on at the wrong time. Cutting drag to 1.5% would restore Sharpe to about 0.57 and CAGR to about 16.7%. **The right next step is diagnosing that drag, not trying more leverage rules.**

**Defensible leverage at 13% base vol.** SE taken as 0.24 (18 years daily, IID, so a floor given fat tails and autocorrelation). L = (Sharpe × vol - drag) / vol², then halved, floored at 1.0x (no leverage).

| Live-like Sharpe | 25th pct | 10th pct | Half-Kelly L, 1.5% drag (25th / 10th) | Half-Kelly L, 4.8% drag (25th / 10th) | Defensible up-regime ceiling |
|---|---|---|---|---|---|
| 0.6 | 0.44 | 0.29 | 1.25 / 1.0 | 1.0 / 1.0 | **1.0-1.25x** |
| 0.9 | 0.74 | 0.59 | 2.4 / 1.8 | 1.4 / 1.0 | **1.25-1.5x** |
| 1.2 | 1.04 | 0.89 | 3.6 / 3.0 | 2.6 / 2.0 | **1.35-1.7x** |

- The final column is tighter than Kelly because the tail budget binds first. With the 10-15% per-cycle overlay loss cap, a -42% cycle (ACWI 2008) with no exit allows only 1.24-1.36x; exiting halfway through allows 1.48-1.71x; a -13% one-day shock allows 1.77-2.15x. Sharpe 1.2 should be treated as at most 1.0 until live, per Carver.
- These are ceilings for risk-on only, and for an edge confirmed live. At today's 0.60 the honest answer is **no structural leverage** until drag is under about 2%.

**Expected years to a first -35% drawdown** (standard CUSUM-type formula τ = (s²/2m²)(e^k - 1 - k), k = 2mb/s², b = 0.431, cash 6%):

| True Sharpe | 13% vol | 20% vol | 26% vol |
|---|---|---|---|
| 0.6 | about 370 | 21 | 7 |
| 0.3 | about 99 | 11 | 4.5 |

- Read as a ranking, not a forecast; Gaussian numbers ignore gaps and vol clustering. The 13% column is implausibly long for that reason. The levered check is more reasonable: at L7's implied 13.0% log growth and 24% vol, the expected time to -40% is about 12 years, consistent with one such event in 18.
- Going from 13% to 26% vol shrinks the wait by a factor of about 50 at Sharpe 0.6. Being wrong on Sharpe (0.6 to 0.3) costs a further 35-50%. This is why Sharpe uncertainty and leverage must be assessed together.
- Fractional Kelly chance of ever falling to x of current wealth is x^(2/c-1): halving at c=1 is 50%, c=0.5 is 12.5%, c=0.25 is 0.8%. That holds from one starting level; over repeated peaks, any drawdown eventually happens.
- Trend-conditional leverage is logical because conditional Sharpe is higher in calm uptrends, but its failure mode is an overnight gap, when the signal cannot act. A US 1928-2015 result ignoring switching cost at 1% financing does not transfer to Indian futures after the STT rise.

### Gaps
- Primary Gayed-Bilello tables (levered CAGR, MaxDD, turnover) were not retrieved.
- Haghani-Dewey's coin-flip experiment, Mauboussin's "Size Matters" and Return Stacked's record were not retrieved. Return Stacked remains ungraded, and presumably marketing until checked.
- No audited long record of trend-conditional leverage was found.
- L7's true vol, its leverage path, and whether it was regime-conditional are unknown here: centurion_core was out of scope. Replace the table's assumed vols with actuals.

## Drawdown control

### Takeaway
Drawdown rules are theoretically optimal only with continuous trading. They work when returns have momentum or regime persistence, and lose money under random walks. They can amplify crowded unwinds, as in August 2007, so they suit solvency protection more than performance.

### Cited Findings
- Theory: Grossman-Zhou (1993) scale risky exposure to the surplus above a fixed fraction of peak wealth, similar to CPPI — [IDEAS](https://ideas.repec.org/a/bla/mathfi/v3y1993i3p241-276.html). Klass-Nowicki show the long-run growth optimality does not carry to discrete time — [Lund](https://lup.lub.lu.se/record/223851).
- `[ACAD-OOS]` Kaminski-Lo: stop-loss rules always lower expected return under a random walk, and can add value under momentum or regime switching; in US monthly equity data 1950-2004 they added 50-100 bps a month during stop-out periods — [MIT](https://dspace.mit.edu/handle/1721.1/114876); [working paper](https://econpapers.repec.org/RePEc:hhs:sifrwp:0063).
- Theory: Bailey-Lopez de Prado triple penance: recovery takes about 3 times as long as reaching the drawdown, whatever the Sharpe — [escholarship](https://escholarship.org/uc/item/758637dr). Positive serial correlation can shorten it to as low as 1.6 — [CAIA](https://caia.org/node/4579). Ignoring serial correlation underestimates downside by up to 70% — [IDEAS](https://ideas.repec.org/a/rsk/journ4/2435335.html). Sharpe-based stop-outs may fire over three times as many skilled PMs as firms intend — same source.
- `[RECORD-style academic]` Khandani-Lo, August 2007: long/short, 130/30 and long-only funds de-levering exacerbated losses, and stop-loss rules played a role; the unwind ran 7-9 August with a significant reversal on the 10th; authors call this speculative — [MIT](https://web.mit.edu/~alo/www/Papers/august07.html); [NBER](https://nber.org/system/files/working_papers/w14465/w14465.pdf).
- `[PRESS]` Indian analogue: investor wealth was reported up about ₹59 lakh crore from the 4 June 2024 level by 18 July 2024 — [multibagg](https://www.multibagg.ai/market-pulse/articles/nifty-june-4-crash-cmvmepzt2000e2vpiq95gaw4r), low quality and single source.

### Inferences
- From -35% the book needs +54% to recover, about 3.5 years at 13.2%. From -40.3% it needs +67.5%, about 4 years at 13.9% (my calc). With exposure cut, recovery takes longer, which is the true cost of drawdown rules.
- Centurion's 20/30/35% ladder is reasonable as ruin insurance, but it fires after the regime filter and the ATR stops, so it overlaps them. Measure the ladder's marginal value by removing it, and define a re-entry rule before it is needed. Cutting without a re-entry rule is how you sell the low and miss the 10 August style reversal.
- Scale down in steps tied to the signal rather than to a hard flat. This fits Kaminski-Lo: rule value depends on persistence, which is uncertain.
- Aligned with the project's standing view that MaxDD is recorded, not a gate: use drawdown rules for survival, and judge them on Sharpe, Calmar and OOS results.

### Gaps
- Chekhlov-Uryasev-Zabarankin CDaR was not retrieved.
- No audited evidence was found on drawdown rules in live funds, and no Indian study.

## Crash protection and tail hedging

### Takeaway
Rolling puts cost about 8% a year in a major study, and simple divesting beat them on drawdown. Trend is the cheaper indirect hedge, but it had its own 20% drawdown in 2024-25. Universa's result is real as an unaudited investor claim but not transferable. For Centurion, cutting equity is the benchmark any hedge must beat.

### Cited Findings
- `[ACAD-OOS]` Harvey, Hoyle, Rattray, Sargaison, Taylor, Van Hemert (2019), 1985-2018, eight S&P drawdowns over 15%: puts protect well but cost about 8% a year; credit protection is cheaper but still loses; gold worked but near-zero long-run return with long underwater spells; bonds were negative in the worst quintiles before 2000; trend worked in recessions and drawdowns, with 3-month momentum best; 10% in a hedge lifted drawdown-period returns from -44.3% to -36.8% annualised — [the7circles summary](https://the7circles.uk/?p=28982); [Scribd](https://www.scribd.com/document/491534322/jpm-2019-45-5-007). The summary is secondary and the primary tables were not retrieved.
- `[ACAD-OOS]` Ilmanen, Thapar, Tummala, Villalon (2021) compare out-of-the-money puts and trend; they call the "puts cost more, hedge better" view only partly right and favour trend — [AQR](https://www.aqr.com/Insights/Research/White-Papers/Tail-Risk-Hedging-Contrasting-Put-and-Trend-Strategies). AQR's position is that protracted drawdowns hurt long-term investors more than short sharp shocks — [Advisor Perspectives](https://www.advisorperspectives.com/articles/2020/05/06/an-aqr-warning-and-a-3-612-return-fire-up-the-black-swan-debate).
- `[MARKETING]` Universa: 3,612% in March 2020; a 96.7% S&P / 3.3% fund mix came out of March unscathed and beat the S&P since 2008; Taleb dismissed AQR's comparison and argued it did not test Universa's hedged portfolio — same [Advisor Perspectives](https://www.advisorperspectives.com/articles/2020/05/06/an-aqr-warning-and-a-3-612-return-fire-up-the-black-swan-debate). Reported Q1 figure of 4,144% differs. The private fund has no public audit that I found.
- `[BACKTEST]` Independent CBOE-data backtest of a Universa-like structure gives 11.0% vs Universa's claimed 11.5% and SPX 7.9%, 2006-2020 — [Advisor Perspectives](https://api.advisorperspectives.com/articles/2020/07/13/empirical-support-for-tail-risk-strategies).
- `[PRESS]` CalPERS' exit from tail hedges before March 2020 was described as a roughly $1 billion miss — [Naked Capitalism](https://www.nakedcapitalism.com/2020/05/universa-debunks-calpers-defense-of-1-billion-tail-hedge-miss-insider-accounts-expose-additional-misrepresentations.html).
- `[ACAD-OOS]` Israelov, Pathetic Protection (JAI 2019): CBOE 5% put index and Monte Carlo; divested portfolios (fixed equity plus cash) beat protected ones on peak-to-trough drawdown 97% of the time at very short horizons and 100% at longer ones; a 40/60 equity/cash mix matched the protected portfolio's returns; puts fail when the market rises and then crashes near expiry — [Alpha Architect](https://alphaarchitect.com/pathetic-protection/); [AQR](https://www.aqr.com/insights/research/white-papers/pathetic-protection-the-elusive-benefits-of-protective-puts?aqrPDF=1). Secondary figures; the fetch was refused.
- `[MARKETING]` CBOE-commissioned study: PUT, BXM and CLL about 30% lower vol than the S&P over 25 years; CLL buys a 3-month 95% put and sells a 1-month 110% call — [Mondo Visione](https://mondovisione.com/media-and-resources/news/new-study-compares-25-year-performance-of-options-strategy-benchmarks-to-traditi-2012214/).
- Managed futures as crisis alpha: see SG Trend 2008 and 2022 vs the 2024-25 loss in the first section, `[AUDITED LIVE]`.

### Inferences
- A permanent put program at anything near 8% a year would exceed Centurion's entire excess return. It is not affordable as a static overlay.
- Ordering for Centurion: first reduce exposure (Israelov), second an uncorrelated trend or bond stream, third a small convex sleeve on explicit events only. Each option overlay stays subject to the 10-15% per-cycle loss cap.
- If leverage is ever used, buying defined protection only on the levered portion is the clean design: the premium is charged against the leverage decision, and must be added to the L7-style drag.
- Taleb-style portfolios depend on extreme convexity that retail Indian option liquidity may not give. Test, do not assume.

### Gaps
- AQR's and Harvey's primary figures were not retrieved; the AQR PDF was unreadable. Annual cost of PPUT, CLL and put-spread collars was not found.
- Israelov-Klein on collars was not retrieved.
- No India-specific cost was found for rolling Nifty puts or put-spread collars. Compute this from Centurion's own options store.

## Portfolio construction for many streams

### Takeaway
Estimation error is the deciding factor: simple weights hold up best out of sample and no optimiser reliably beats 1/N. HRP has no strong independent proof over equal-risk weighting. For a limited number of streams, use equal risk or modest shrinkage, and treat uncorrelated stream count as the main lever.

### Cited Findings
- `[ACAD-OOS]` DeMiguel-Garlappi-Uppal (RFS 2009): 14 models on seven datasets, none consistently better than 1/N on Sharpe, CER or turnover; estimation window needed is around 3,000 months, and about 6,000 months for 50 assets (asset count on the first figure unconfirmed) — [IDEAS](https://ideas.repec.org/a/oup/rfinst/v22y2009i5p1915-1953.html). Working paper examples run from 50 years to over 1,000 years — [CEPR](https://econpapers.repec.org/RePEc:cpr:ceprdp:5142).
- `[BACKTEST]` Asness-Frazzini-Pedersen: leverage aversion means safer assets should earn higher risk-adjusted returns; risk parity overweights them — [CFA Institute](https://rpc.cfainstitute.org/research/financial-analysts-journal/2012/leverage-aversion-and-risk-parity). 1926-2010 stocks Sharpe 0.35 vs bonds 0.47 — [ThinkAdvisor](https://thinkadvisor.com/2012/08/23/how-to-get-high-performance-with-low-risk-from-new-strategy). Contradicted by Bianchi-Goldberg: result depends on frictions and contains look-ahead — [escholarship](https://escholarship.org/content/qt23t2s950/qt23t2s950.pdf).
- `[BACKTEST]` HRP: no rigorous net-of-cost test against equal weight was found. Monte Carlo work finds HRP close to, or below, naive risk parity out of sample — [Marti, part 3](https://gmarti.gitlab.io/qfin/2019/12/04/hierarchical-risk-parity-part-3.html); [part 2](https://gmarti.gitlab.io/qfin/2018/10/15/hierarchical-risk-parity-part-2.html). A CBS thesis finds lower out-of-sample variance than minimum variance, at higher transaction cost — [CBS](https://research-api.cbs.dk/ws/portalfiles/portal/76452070/1332322_Master_Thesis_Hierarchical_Risk_Parity.pdf). Vendor claims of beating 1/N cite no study, `[MARKETING]` — [MetricGate](https://metricgate.com/docs/hierarchical-risk-parity/).
- `[ACAD-OOS]` Clare et al: risk parity adds little within asset classes — [working paper](https://ideas.repec.org/p/yor/yorken/12-25.html).

### Inferences
- Centurion has far less than 3,000 months of any stream, so optimised weights are not credible. Default to equal risk within groups with a correlation shrink toward a fixed value. This resembles Carver's handcrafting, though that was not retrieved here.
- The Sharpe arithmetic favours stream count more than weighting: at the same stream Sharpe, adding streams at pairwise correlation 0.3 caps portfolio Sharpe at about 1.8x the single-stream value (1/sqrt(0.3)). Correlation near 0.8 caps it at about 1.1x (my calc). Measure live correlation in stress, not average correlation.
- Weights must be stress-tested on 2008, 2020 and 2024: the Baltas-Kosowski co-movement result implies diversification shrinks when needed most.

### Gaps
- Lopez de Prado's original HRP paper, Carver's handcrafting and Ledoit-Wolf shrinkage were not retrieved (budget). Peer-reviewed net-of-cost HRP evidence remains open.

## Blow-up case studies

### Takeaway
Nearly every failure combines leverage or short convexity with an exit that was not there when needed. The losses come from size relative to liquidity, not from wrong direction. Centurion's stops and drawdown rule address slow losses, while its largest gaps are exposure measured on gross notional and liquidity sizing.

### Cited Findings
- `[RECORD]` LTCM: balance-sheet leverage above 25:1 at the start of 1998; heavy repo funding — [Fed testimony](https://www.federalreserve.gov/boarddocs/testimony/1998/19981216.htm). It lost almost 90% of capital Jan-Sep 1998, and counterparties' weak risk management let leverage grow — [GAO](https://webarchive.library.unt.edu/web/20081126185007/http://www.gao.gov/archive/2000/gg00003.pdf).
- `[RECORD]` Amaranth: Senate report (25 June 2007) found hundreds of thousands of gas contracts held, dominating the market until its September 2006 collapse; a substantial share moved from NYMEX to ICE, out of NYMEX's view — [Senate report](https://www.hsgac.senate.gov/imo/media/doc/REPORTExcessiveSpeculationintheNaturalGasMarket.pdf?attempt=2); [InvestmentNews](https://www.investmentnews.com/alternatives/amaranth-blurred-holdings-says-senate/9843). EDHEC disputes emphasis — [EDHEC](https://www.premiacap.com/publications/EDHEC_Working_Paper_Review_US_Senate_Report_Amaranth.pdf).
- `[ACAD/RECORD]` Quant Quake: see the Drawdown section. Refined account: deleveraging through July, withdrawal of market-making capital from 8 August, two unwinds on 1 and 6 August in financials and in value/earnings momentum — [MIT](https://web.mit.edu/~alo/www/Papers/august07b.html).
- `[PRESS]` Volmageddon: VIX about doubled, 17 to 37, on 5 Feb 2018 — [Bloomberg](https://www.bloomberg.com/news/articles/2019-02-06/the-day-the-vix-doubled-tales-of-volmageddon). Short-vol ETPs had to buy VIX futures after the close to rebalance, losing over 90%, with XIV about 96% — [Convextrade](https://convextrade.com/history/2018-volmageddon), low quality. XIV had an acceleration clause at ≤20% of prior value; last trade 20 Feb 2018 — [Investing.com](https://m.investing.com/news/stock-market-news/credit-suisse-to-redeem-popular-exchangetraded-volatility-notes-1203505?ampMode=1). Academic treatment: [FAJ 2021](https://rpc.cfainstitute.org/research/financial-analysts-journal/2021/volmageddon-failure-short-volatility-products), not read.
- `[RECORD via PRESS]` Archegos: Credit Suisse's 165-page Paul Weiss review cited "fundamental failure of management and controls", ignored potential-exposure breaches, and a swap margin cut to 7.5% in 2019 — [The TRADE](https://www.thetradenews.com/?p=79792); [Caixin](https://www.caixinglobal.com/2021-07-30/incompetence-caused-archegos-bust-that-wiped-billions-from-chinese-edtech-firms-credit-suisse-probe-reveals-101748967.html). Loss about $5.5bn, Nomura about $2.9bn — [Wikipedia](https://www.wikipedia.com/wiki/Archegos_Capital). Exposure conflict: $36bn vs $50-100bn, unresolved.
- `[PRESS]` Melvin: down about 53% in January 2021 on GameStop and over a dozen other bets; $2.75bn from Citadel and Point72; started the year near $12.5bn — [Bloomberg](https://www.bloombergquint.com/markets/melvin-lost-53-in-january-hurt-by-gamestop-other-bets-dj); [Benzinga](https://benzinga.com/news/21/01/19312927/gamestop-short-seller-melvin-gets-2-75b-funding-from-citadel-point72).
- `[RECORD via PRESS]` Karen Bruton: short strangles on index options; SEC complaint 31 May 2016 alleged hidden losses over $50m rolled forward; settled about $1.5m; neither admitted nor denied. Sources say it was concealment, not a confirmed wipe-out — [InvestorMint](https://investormint.com/investing/how-karen-the-supertrader-blew-up); [OptionsTradingIQ](https://optionstradingiq.com/karen-the-supertrader/); [Slope of Hope](https://slopeofhope.com/?p=189853). All secondary.
- `[RECORD]` Karvy: SEBI ex parte order 22 Nov 2019 after NSE found misuse of client securities — [Business Standard](https://www.business-standard.com/amp/article/markets/sebi-bans-karvy-stock-broking-for-rs-2-000-crore-client-defaults-119112300172_1.html). NSE settled about Rs 2,300 crore for 235,000 investors — [Moneylife](https://www.moneylife.in/article/nse-expels-karvy-stock-broking-declares-it-as-defaulter/62168.html). SEBI fined NSE Rs 2 crore and BSE Rs 3 crore in April 2022 for laxity — [Moneylife](https://www.moneylife.in/article/bse-nse-penalised-by-sebi-over-laxity-in-karvy-stock-broking-case/66898.html).
- `[PRESS]` June 2024: Nifty -5.93% to 21,884.5, worst since March 2020; intraday over -8%; low 21,281.45 — [Moneylife](https://moneylife.in/article/nifty-sensex-crash-on-election-surprise-tuesday-closing-report/74316.html); [IIFL](https://www.indiainfoline.com/news/markets/indian-benchmark-indices-plunge-over-5-on-vote-counting-day). India VIX peak is reported variously as 28.05 to 31.71, and options swung wildly — [Business Standard](https://www.business-standard.com/amp/markets/news/lok-sabha-election-results-trends-send-options-market-in-a-tizzy-124060401523_1.html).
- `[RECORD]` Retail options: 93% of over 1 crore individual F&O traders lost in FY22-FY24, average about ₹2 lakh, aggregate above ₹1.8 lakh crore — [SEBI](https://www.sebi.gov.in/media-and-notifications/press-releases/sep-2024/updated-sebi-study-reveals-93-of-individual-traders-incurred-losses-in-equity-fando-between-fy22-and-fy24-aggregate-losses-exceed-1-8-lakh-crores-over-three-years_86906.html). FY25: 91% lost, net about ₹1.06 lakh crore, up 41% — [Business Standard](https://www.business-standard.com/amp/markets/news/net-losses-of-traders-in-fo-widens-in-fy25-sebi-study-125070701221_1.html).
- `[RECORD]` 2020 circuit breakers: see India section.
- `[RECORD]` Jane Street: SEBI interim order 3 July 2025 against four entities, Rs 4,843.57 crore of alleged illegal gains, BANKNIFTY expiry-day manipulation; about Rs 735 crore on 17 Jan 2024 — [Tribune](https://www.tribuneindia.com/news/business/sebi-imposes-highest-ever-penalty-of-rs-4843-57-crore-on-jane-street-group-for-index-manipulation). The amount was deposited in July 2025 and the ban lifted — [Kotak Neo](https://www.kotakneo.com/news/market-news/sebi-bars-jane-street/). Interim only: SAT reserved an order on document disclosure, reported 21 Oct 2026, and I found no final merits ruling — [NewsBytes](https://www.newsbytesapp.com/news/business/sat-reserves-order-in-jane-street-document-disclosure-case/story). Allegations are unproven.

### Inferences

| Case | Mechanism | Guardrail that would have stopped it | Centurion status |
|---|---|---|---|
| LTCM | Leverage rose as equity shrank; repo funding | Gross cap measured on current equity, forced to de-lever as NAV falls | Needed before any leverage |
| Amaranth | Position larger than market could absorb | Size by days-to-exit, not capital | 8% cap is capital-based; add ADV cap |
| Quant Quake | Crowded unwind plus others' stops | Slower, stepwise DD rule; crowding check | DD rule exists, re-entry rule to confirm |
| XIV | Short vol with path-dependent rebalance, termination clause | Ban on products with acceleration clauses or forced rebalance | No such exposure; keep banned |
| Archegos | Concentration hidden by swaps on thin margin | Cap per name on notional, including futures delta | 8% must apply to notional/NAV |
| Melvin | Concentrated shorts in a squeeze | Long-only; cap any stock-futures shorts | Compliant today |
| Bruton | Short strangles; losses rolled forward | Daily mark-to-market; no rolling to avoid a loss; defined risk | Keep for any option sleeve |
| Karvy | Broker misused client securities | Own demat, verify depository holdings, second broker | Check |
| June 2024 | Binary event gap, vol spike | Cut levered overlay before scheduled events | Add event calendar |
| Retail option sellers | Negative skew, margin | No unhedged short options | Keep; aligns with per-cycle cap |
| 2020 circuits | No exit during halt | Size for -13% with no exit | Add stress test |
| Jane Street | Alleged expiry-day index moves | Avoid expiry-settled exposure near close | Applies to options only |

- Common thread: leverage or negative carry plus a market that stops giving you the exit. A long-only unlevered book with an 8% cap has almost none of these failure modes. **Adding L7 leverage reintroduces nearly all of them, for +0.7pp.**
- The kill switch only helps if it can trade. Most cases above were exit failures, so it must not be counted as tail protection.

### Gaps
- Amaranth's about $6bn and LTCM's $4.6bn are widely quoted but neither appeared in retrieved sources, so they are not asserted here.
- Primary Credit Suisse report, SEC Bruton complaint and SEBI orders were not read directly.
- I found no quantified, dated case of an Indian retail option seller wiped out, and no source quantifying option-seller losses on 4 June 2024.

## Risk of ruin specifics for India

### Takeaway
For a long book, the Indian ruin path is not one bad stock. It is a market-wide gap with halts, stops that cannot execute, and MTM on futures paid in cash while stock positions are locked or margined. Peak-margin and futures rules make leverage far less forgiving than a backtest implies, and 2026 STT and algo rules worsen the economics.

### Cited Findings
- `[RECORD]` Market-wide breakers trigger at 10%, 15% and 20% of index movement, reset daily — [NSE](https://www.nseindia.com/static/products-services/equity-market-circuit-breakers). 10% before 1pm halts 45 minutes; 20% halts for the rest of the day; 15% tiers differ between sources, unconfirmed. Reopening follows a 15-minute pre-open call auction — [Outlook Business](https://www.outlookbusiness.com/amp/story/equity/what-is-index-based-marketwide-circuit-breaker-and-how-it-is-applied-4492).
- `[PRESS]` 13 March 2020: 10% fall, 45-minute halt; first in about 12 years — [Business Standard](https://www.business-standard.com/amp/article/pti-stories/bourses-halt-trading-for-first-time-in-12-years-120031300868_1.html). 23 March 2020: 45-minute halt, Nifty -12.98% to 7,610.25, Sensex -13.15% — [Business Standard](https://www.business-standard.com/amp/article/news-cm/sensex-nifty-crash-10-hit-lower-circuit-120032300280_1.html); [Business Today](https://www.businesstoday.in/markets/stocks/story/sbe-nse-sensex-worst-day-in-history-crashes-252759-2020-03-23).
- `[RECORD]` Stock bands: fixed 2/5/10/20%; Aug 2023 counts 1,543 at 20%, 87 at 10%, 420 at 5%, 75 at 2% — [5paisa](https://5paisa.com/news/changes-in-price-bands-effective-from-august-07-2023). F&O stocks have a dynamic band from ±10% that flexes with a 15-minute cooling period — [Zerodha](https://zerodha.com/z-connect/business-updates/explainer-on-new-price-band-rules-for-stocks-trading-in-the-fo-segment). Flex rules changed 18 Nov 2024, with the opposite band moving too — [Paytm Money](https://www.paytmmoney.com/blog/?p=5135). SEBI proposed uniform dynamic bands across exchanges in April 2026; adoption unconfirmed — [5paisa](https://www.5paisa.com/news/sebi-to-enforce-dynamic-price-bands-for-all-fo-stocks-equally-on-exchanges).
- `[PRESS]` 848 stocks were locked at lower circuit on 4 June 2024, per headline; article body not read — [Business Standard](https://www.business-standard.com/amp/markets/news/848-stocks-locked-in-lower-circuit-on-june-4-post-ls-polls-here-what-it-means-124060400745_1.html). If no buyers bid at the band the stock cannot trade; multi-day locks trap holders, with DHFL and IL&FS subsidiaries as examples — [multibagg](https://www.multibagg.ai/market-pulse/articles/lower-circuit-stocks-india-reasons-cmreq8yel000dnp0jocsb95km), low quality.
- `[RECORD]` GTT execution, from Zerodha's own terms: trigger conditions are stored and monitored by Zerodha, not the exchange; a gap past the trigger sends the order at your limit with "no guaranteed execution"; an unfilled limit is cancelled at session end, and a GTT triggers only once, so it must be placed again; orders whose limit falls outside circuit limits may be rejected; a trailing-stop trigger failing on dynamic circuit limits does not reset; Zerodha bears no liability for non-execution — [Zerodha terms](https://zerodha.com/tos/gtt).
- `[RECORD]` Market GTTs exist: they use Market Price Protection, and any unexecuted portion beyond the protection range becomes a resting limit at the boundary, with slippage on you. Trailing uses LTP only and can fail on data delays, gaps, low volume or glitches. Minimum trigger distance is 0.25% for stocks above Rs 50; trigger cut-offs are 3:15pm for F&O stocks, 3:30pm for non-F&O, 3:40pm for F&O contracts; maximum 500 pending; 365-day life; sell GTTs need TPIN authorisation unless POA or DDPI is on file; NSE call-auction-list scrips are excluded — [Zerodha terms](https://zerodha.com/tos/gtt).
- `[RECORD]` GTT works for CNC, NRML and MTF, and is cancelled for corporate actions with over 2% impact or a stock category change — [Zerodha support](https://support.zerodha.com/category/trading-and-markets/kite-features/gtt/articles/gtt-market-orders). This supersedes third-party claims that GTT is CNC-only, limit-only and has no trailing stop — [Chittorgarh](https://www.chittorgarh.com/keyword/gtt-order/1351/), outdated.
- `[RECORD]` Peak margin: 100% from 1 Sep 2021; at least four random intraday snapshots — [Angel One](https://www.angelone.in/knowledge-center/mtf/what-is-peak-margin); [5paisa](https://www.5paisa.com/stock-market-guide/generic/peak-margin). Applied to 100% of Span + Exposure for derivatives — [Moneylife](https://moneylife.in/article/explainer-how-sebis-new-peak-margin-norms-impact-traders-from-1st-september/64993.html). Penalty slabs of 0.5-5% conflict across sources. Brokers pass penalties to clients — [InCred](https://www.incredmoney.com/knowledge-center/?p=5803).
- `[RECORD]` NSE outage 24 Feb 2021: trading halted at 11:40 for about four hours; NSE cited its online risk management system — [Money9](https://money9.com/news/equities/nse-says-unavailability-of-online-risk-management-system-led-to-shutdown-11308.html). SEBI found faulty design and insufficient peak capacity — [Moneylife](https://www.moneylife.in/article/nse-trading-halt-sebi-probe-reveals-faulty-design-and-low-capacity-at-the-exchange-and-ncl/66739.html). NSE settled for Rs 72.64 crore — [Outlook Business](https://www.outlookbusiness.com/amp/story/news/nse-arm-ncl-settle-2021-s-trading-glitch-case-with-sebi-pay-rs-72-cr-news-296631). A client was charged a short-margin penalty after a late margin-file spike — [Zerodha](https://zerodha.com/z-connect/general/24th-feb-2021-when-nse-was-down-for-almost-the-entire-trading-day).
- `[RECORD]` Broker risk: daily upstreaming of client funds from 1 July 2023 — [Business Standard](https://www.business-standard.com/amp/markets/news/sebi-makes-rule-for-upstreaming-of-client-funds-by-brokers-to-clear-corps-123060801106_1.html). NSE investor-protection cap is ₹35 lakh per investor per defaulter on Zerodha's page — [Zerodha](https://support.zerodha.com/category/trading-and-markets/trading-faqs/general/articles/what-happens-to-funds-and-holdings-if-stockbroker-defaults), vs ₹25 lakh in 2023 — [Zee Business](https://www.zeebiz.com/market-news/news-zee-business-exclusive-sebi-exploring-unlimited-cover-for-clients-in-case-of-broker-default-karvy-238672). Verify with NSE.
- `[RECORD]` Costs: Budget 2026 raised futures STT from 0.02% to 0.05% and option premium from 0.1% to 0.15% — [Business Today](https://www.businesstoday.in/markets/stocks/story/stt-hike-why-futures-options-caught-centres-attention-this-budget-514014-2026-02-01), from 1 April 2026, exercise 0.125% to 0.15% — [5paisa](https://www.5paisa.com/news/stt-hike-on-fo-to-take-effect-from-april-1-amid-rising-options-activity).
- `[PRESS]` SEBI's retail algo framework became mandatory 1 April 2026, with static IP and exchange Algo-ID — [Fyers](https://fyers.in/notice-board/new-sebi-framework-for-retail-algo-trading-from-april-01-2026/); [Sahi](https://www.sahi.com/blogs/sebi-algo-trading-rules-2026-what-every-retail-trader-must-know-before-april). Sources conflict on whether a self-built personal Kite Connect bot needs extra registration — [Rana](https://jayadevrana.com/zerodha-automated-trading-policy/).

### Inferences
My calc unless cited; figures depend on the assumptions shown.
- **Index gap test.** A 23 March 2020 repeat, -13% with beta near 1 and no exit, costs 13% unlevered, 16% at 1.25x, 19.5% at 1.5x and 26% at 2x, all before stops. This is the one-day cost of the book's leverage, and the 20% rung of the drawdown ladder could trip in a single day at 1.5x.
- **Futures MTM is the ruin mechanism.** If stock holdings carry the book and futures add the leverage, futures losses settle in cash daily while equities may be locked, and margin can rise at the same moment (24 Feb 2021 example). Assume a futures margin of m percent; a -13% day consumes 13/m of it, so near-total at 13-15% (m is a placeholder, not sourced). **Hold liquid collateral worth at least 3x posted margin** so that no forced stock sale funds an MTM call.
- **Roll cost.** Monthly-rolled futures pay STT at 0.05% each roll, about 0.6% a year of notional vs 0.24% before, assuming sell-side only and 12 rolls. That is about 0.36pp a year of new drag on levered notional, before basis and slippage.
- **Single-name lock is survivable.** At 8% of NAV, three consecutive 20% lower circuits cost 3.9% of NAV, and five 5% circuits 1.8%. If leverage raises the name to 12%, the three-circuit case becomes 5.9%. Market-wide lock matters more; 848 stocks at once is the actual threat.
- **6-ATR stop.** It is a slow-loss control, and the gap beyond it is unmodelled risk. Backtests should apply stops at next available tradable price, including band-locked days, not at the stop level.
- **A GTT is a one-shot, broker-held stop, and this is the most actionable finding here.** On a gap through a limit it fires, may not fill, is cancelled at close, and leaves the position with no stop the next day. A daily reconciliation must confirm every open position still has a live stop, covering triggered-unfilled, rejected, corporate-action and category-change cases. Prefer market-with-protection for exits, and treat a resting order at the protection boundary as not yet filled. Whether Centurion does this today was outside scope to check.
- **Stops on lower-circuit days.** A limit outside the band may be rejected, so a stop computed from ATR can fail exactly when the stock is locked. The re-placement logic must clamp the limit inside the day's band.
- **Sell-authorisation failure.** A sell GTT on holdings can fail without TPIN, POA or DDPI, a silent failure mode for an automated book; check DDPI or POA is in place.
- **Operational single points.** GTT triggers sit at Zerodha, so a broker outage disables stops as well as manual exits. Static-IP whitelisting means a lost IP may block orders. An exchange outage (four hours in 2021) stops all exits. Hold a documented manual exit path and a second broker for hedging the same index, on a separate network.
- **Broker concentration.** Keep securities in own demat and verify at depository regularly. Compare the ₹25-35 lakh cover with actual book size; the cap does not cover a large book.
- **Index futures over stock futures** for any leverage: they avoid single-name physical settlement and ban-period risk, on my understanding (unverified, see Gaps).

### Gaps
- Market Price Protection's actual range was not retrieved (separate Zerodha article), so how far a protected market exit can fall short on a gap is unknown.
- Stock-futures physical settlement, MWPL ban period rules, Nifty futures margin levels, the 50% cash-equivalent collateral rule and pledge haircuts were **not verified**, because the search budget ran out. Confirm them before any futures design.
- Broker-outage history (Zerodha March 2020 and similar) was not retrieved.
- SEBI's delta-based OI, MWPL and intraday limit changes, and the Oct 2024 index-derivative measures, were not retrieved.
- Adani 2023 and Vakrangee lower-circuit episodes were not retrieved.
- The exact 15% circuit tier timings, peak-margin penalty slabs and current IPF cap need primary NSE or SEBI confirmation.
- Whether a self-built Kite Connect bot is compliant under the April 2026 rules needs Zerodha's or SEBI's text, not blogs.
