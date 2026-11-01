# Regime-adaptive exposure without structural leverage: what raises participation and cuts loss

As of 1 Nov 2026. Builds on `research_notes/Centurion golden target feasibility/regime_leverage_ruin.md` (not repeated here) and uses `centurion_audit.md` for Centurion's own numbers.

**Evidence grades** (same as the earlier note). `[AUDITED LIVE]` real money or index of live managers. `[ACAD-OOS]` academic with an out-of-sample test, still simulated money. `[BACKTEST]` backtest or in-sample, including back-filled indices. `[MARKETING]` issuer or sponsor claim. `[BLOG]` informal, unreviewed.

**Retrieval.** Read in primary text: Cederburg et al, DeMiguel et al, Daniel-Moskowitz (NBER), Shu-Yu-Mulvey, Shu-Mulvey, Etienne et al, NSE April 2026 momentum whitepaper, PTLC fact sheet of 30 Sep 2026. Everything else comes from abstracts or secondary summaries and says so where it matters. Alpha Architect, the Bongaerts full text and the Barroso-Detzel draft would not load. "My calc" is hand arithmetic under stated assumptions. **"Independent regimes" counts are my own rough counts, not the papers'.** No Centurion code or data was run.

## Evidence-ranked answer: what raises participation and cuts loss

### Takeaway
No overlay in this literature raises net return for an unlevered long-only book on audited money. Exposure rules reliably cut loss and usually cost return; the longest real-money 200-day record lags its index by 5.0 points a year. The better-supported levers are on selection and cost (residual momentum, slower rebalancing), and on the exposure side the one defensible test is a small, bounded half-step in risk_off, which is my construction and untested anywhere.

### Cited Findings

| Rank | Lever | Up / down effect | Best grade | Key number (net unless stated) | Independent regimes (my count) | India evidence | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Residual (idiosyncratic) momentum rank | Better rebounds / milder crashes | ACAD-OOS, 48 countries | Sharpe about double in US; highest break-even cost of 3 fixes; absolute numbers not retrieved | US about 2 big crashes (1930s, 2009); international 1-2 | None found | **Test first as a selection rule** |
| 2 | Slower rank rebalancing | Neutral / neutral, saves cost | BACKTEST index, India | Quarterly Nifty200 Mom 30: 17.12% against semi-annual 19.19%, beat it 6 of 20 years | 5 Indian bears | Direct, back-filled | **Test; outside this brief's exposure scope** |
| 3 | Bounded half-step in risk_off (fast-slow disagreement) | Better rebound / small extra loss | ACAD-OOS, gross, US | Static slow-fast blend 0.51 against buy-and-hold 0.38, gross; dynamic adds 0.01 | US 1969-2018, about 7-8 bears | None; Centurion's own state split supports the motive | **Single exposure rule to pre-register** |
| 4 | Down-only conditional vol scaling | Neutral / better tail | ACAD (two-sided version), long-short momentum strongest | Momentum Sharpe more than doubles, MaxDD -54.1% to -20.1%; market gains far less | About 3 global crises | None found | Low priority: overlaps India VIX in gate |
| 5 | Single 200-day overlay | Worse / better | AUDITED LIVE | PTLC 8.98% against S&P 14.00%, 11 years | 2-3 US bears | Own data: state does not separate returns | Already in the gate; do not add more |
| 6 | Jump model or HMM | Worse or flat / better | ACAD-OOS | S&P jump 11.2%, 0.68 against 10.2%, 0.48; HMM 8.5%, 0.54 | About 4 shared bears | No credible test found | Do not build; keep simple rules |
| 7 | Drawdown rule, CPPI | Worse / better | ACAD simulation | Neither dominates buy-and-hold; buy-and-hold has higher mean | Bootstrap | None found | Keep as ruin insurance only |
| 8 | Concentration, high-beta tilt, sector rotation | Unclear / worse | ACAD-OOS against | Sector edge 2.3% with perfect foresight, about 1% with a 2-month timing error, gross | Multiple cycles | None found | **Reject** |
| 9 | Kelly-style sizing | Not informative for a cash book | Theory | Full Kelly 5.5x on Centurion's numbers (my calc) | n/a | n/a | Reject as a rule |

Source links for each row are in the sections below.

Centurion's own frame, all from [centurion_audit.md](centurion_core/docs/research_notes/Centurion%20golden%20target%20feasibility/centurion_audit.md):
- Capture by ex-post regime: bull 0.73 (book 30.1% a year, NIFTY 42.0%, 49% of days), range 0.51, bear 0.23.
- Mean invested 70%: stocks 41%, metals 29%, idle 30%. Beta 0.17, up capture 0.31.
- A causal 200-day state does not separate market returns in India: NIFTY 5.2% a year in "bull", 19.6% range, 16.9% bear, because crashes start above the mean and rebounds below it.
- Stock P&L by period: 2008-12 -1.1 points a year, 2013-19 +14.7, 2020-25 +15.8.
- 420 trades a year cost about 2.2 points.

### Inferences
- **Where the money is (my calc, simple sum, upper bound, ex-post regimes).** Lifting bull capture from 0.73 to 0.85 is worth 0.12 × 42% × 0.49 = 2.5 points a year. Cutting bear capture from 0.23 to 0.15 is worth 0.08 × 51.3% × 0.17 = 0.7. Participation is worth more than further defence, and defence is already strong.
- **Detectability (my calc, Gaussian).** Over 18 years the standard error of a Sharpe difference between two correlated books is about sqrt(2(1-ρ)/18): 0.075 at ρ 0.95, 0.105 at ρ 0.90. So confirm_days 1's +0.064 is t 0.6-0.9, **indistinguishable from noise**, and worse in 2008-12. Proving a change at t 2 needs about +0.15 to +0.21. Most effects below are smaller than that, so choose on prior evidence plus one pre-registered trial, not on the point estimate.
- **Single exposure rule to pre-register, "risk_off half-step".** Reasoning: GT2C failed because idle money sits in risk_off; NIFTY earned 16.9% a year in the causal bear state, because rebounds start there; past winners have beta below 0.5 after declines, so ranks cannot capture the rebound. Hence use the index, not ranks, and only in risk_off:
  - Condition: gate is risk_off, and NIFTY total return over 21 sessions is above 0. Fast window taken from Goulding et al's 1-month signal, not fitted.
  - Action: hold NIFTYBEES at a fixed 25% of book; otherwise liquid. Check once a week at the close, trade next session. No other change.
  - Size comes from the bounded-tail cap, not data: 25% × a -45% leg (2008 Aug-Oct) = -11.25% of book with no exit, inside 10-15%.
  - Pass: walk-forward live-like, D2 costs, 2026 holdout untouched. Sharpe and Calmar not lower in 2008-25, and not lower in at least 2 of 2008-12, 2013-19, 2020-25. Per-cycle overlay loss within the cap. Counts as one trial in DSR. Fail means code deleted and tracker record kept.
  - **Be candid: R11 (faster re-entry) failed and this is a close cousin.** It differs in being partial, index-based and reversible, but the prior is modest. No paper tests this exact rule.
- **Second, separate trial:** replace the rank with 12-1 residual momentum. It targets the 2008-12 weakness directly, where stock P&L was negative.

### Gaps
- No study was found of any of these overlays on a long-only Indian stock book, net of costs, out of sample.
- Centurion's actual share of days in risk_off, and R11's exact design, were not read, so the half-step's expected gain is not estimated.

## Volatility-managed exposure after Moreira-Muir and Cederburg

### Takeaway
The market-level claim did not survive: real-time managed market Sharpe was 0.42 against 0.46. The exception that holds up is momentum, in long-short factor form. Evidence on long-only books is thin and points to little gain. I found no Indian study.

### Cited Findings
- `[ACAD-OOS]` Real-time combination of managed and unmanaged market: Sharpe 0.42 against 0.46, CER also lower. Across 103 strategies, CER is lower in 72 cases. Scaled momentum, profitability and BAB still give large utility gains out of sample; the other six base factors show little evidence — [Cederburg et al, JFE 2020](https://www.lehigh.edu/~xuy219/research/COWY.pdf).
- `[ACAD-OOS]` After transaction costs, vol management of factors other than the market generally gives zero abnormal return and significantly reduces Sharpe; the managed market works only when sentiment is high — [Barroso-Detzel, JFE 2021](https://econpapers.repec.org/RePEc:eee:jfinec:v:140:y:2021:i:3:p:744-767). A 2019 draft's abnormal returns are reported as negligible in long-only portfolios, from a search snippet only; the PDF would not load — [draft](https://sbfc.sydney.edu.au/2019/papers/P057_Named.pdf).
- `[ACAD-OOS]` Counter-evidence: a conditional multifactor portfolio with weights shifting on volatility reaches net out-of-sample Sharpe 1.062, 13% above unconditional and significant at 1%, January 1977 to December 2020. Gross gain beats extra costs — [DeMiguel et al, JF 2024](https://lbsresearch.london.edu/id/eprint/3716/1/The%20Journal%20of%20Finance%20-%202024%20-%20DeMIGUEL%20-%20A%20Multifactor%20Perspective%20on%20Volatility%E2%80%90Managed%20Portfolios.pdf). This is a long-short factor combination, so it does not rescue a long-only book.
- `[ACAD]` Conditional targeting changes exposure only in extreme vol states. Sharpe improves in all markets tested against 8 of 10 for the conventional form; momentum Sharpe more than doubles, MaxDD -54.1% to -20.1%; market MaxDD falls 6.6% on average; turnover and leverage are low — [Bongaerts et al, FAJ 2020, CFA summary](https://rpc.cfainstitute.org/research/financial-analysts-journal/2020/conditional-volatility-targeting). The published rule also raises exposure in low vol, so a cash book cannot run it as tested.
- `[ACAD-OOS]` Emerging markets: a 2025 master's thesis across 153 factors in developed and emerging markets reports broad gains and weaker out-of-sample results; its net-of-cost method is unverified — [UCP](https://ciencia.ucp.pt/en/studentTheses/volatility-managed-portfolios/).
- `[BACKTEST]` India: 450 BSE stocks, a risk-managed momentum rule forecasts severe losses and roughly doubles adjusted Sharpe, no figures or period in the abstract — [Singh et al, FIIB 2022](https://ideas.repec.org/a/sae/fbbsrw/v11y2022i3p347-358.html). Time-series momentum on BSE data 1996-2020 is reported at about 2.5 times the standard version's return, search summary only — [JEFAS record](https://libmast.utm.my/Record/doaj.art-ce4c907115b3479c9c7150a76efa91a7).

### Inferences
- For Centurion the gate already uses India VIX, so a book-level vol rule mostly duplicates it. Its marginal value should be tested by adding it, once, and expected to be small.
- Book vol is 11% with beta 0.17. Vol scaling suits a fully invested book; this one already de-risks. The strongest published result, doubling momentum Sharpe, is a long-short effect.
- If tested, use a down-only variant on the stock sleeve and report it as a different rule from the published one.

### Gaps
- No Indian or emerging-market vol-managed test with explicit costs was found.
- Bongaerts' thresholds were not retrieved, so a faithful specification cannot be written yet.

## Time-series momentum and trend overlay on an equity index

### Takeaway
Horizon choice is close to unknowable, and speed is not the fix: the best evidence favours blending slow and fast signals over changing a single window. The audited record of a 200-day rule is poor in a bull-heavy period. Credible Indian evidence is absent, and Centurion's own split says the 200-day state does not sort returns.

### Cited Findings
- `[AUDITED LIVE]` PTLC, 200-day S&P rule with 5-day confirmation and a 50/50 middle state, at 30 Sep 2026 NAV: since June 2015 8.98% against S&P 14.00%; 10-year 11.01% against 15.33%; 5-year 10.60% against 13.79%; 1-year 9.76% against 15.74%; expenses 0.60% — [Pacer fact sheet](https://www.paceretfs.com/media/ptlc.pdf). Rule detail — [ETF Trends](https://www.etftrends.com/modern-etf-approach-for-trend-following-strategy/).
- `[MARKETING]` Lag from the sponsor's own example: the signal fell after -11.14% in 2000 and -6.98% in 2007. The re-entry dates were 21 Mar 2003 after a 9 Oct 2002 low, and 29 May 2009 after a 9 Mar 2009 low — [Pacer Perspective](https://PacerETFS.com/media/pacer_perspective_nov_2017.pdf).
- `[ACAD-OOS]` Long-only US TSMOM, 1857-2018: best lookback about 9 months, very imprecise; gross monthly Sharpe about 30% above buy-and-hold; under 60% chance of beating it over 5-10 years; about 250 years of monthly data needed for 80% — [CXO on Giner-Zakamulin](https://www.cxoadvisory.com/?p=34113); low test power confirmed in the abstract — [UiA](https://uia.brage.unit.no/uia-xmlui/handle/11250/3031165).
- `[ACAD-OOS]` Zakamulin's robustness test: 300 weighting schemes, 15 lookbacks from 4 to 18 months, 30 sub-periods, January 1860 to December 2014; he assumes no optimal lookback exists — [CXO on Zakamulin](https://cxoadvisory.com/?p=27203).
- `[ACAD-OOS]` Goulding-Harvey-Mazzoleni. Slow 12-month, fast 1-month, US equities: equal blend gross Sharpe 0.51 and MaxDD about -34%, buy-and-hold 0.38 in 1969-2018. Dynamic weights fitted on 1926-68 reach 0.52 out of sample, return over MaxDD 0.23 against 0.17. States: Bull 48.3%, Correction 24.5%, Bear 16.7%, Rebound 10.5%. Fast weight 0% in Correction, 58% in Rebound. Holds in developed markets except Japan. Gross, no turnover comparison — [CXO summary](https://www.cxoadvisory.com/momentum-investing/handling-reversals-in-price-trend-direction/). A second summary gives a 2-month fast signal, unresolved — [paper listing](https://benny.aeaweb.org/conference/2021/preliminary/paper/492Ds6fk).
- `[ACAD-OOS]` TSMOM and moving-average crossovers are essentially the same filter — [Levine-Pedersen, FAJ 2016](https://rpc.cfainstitute.org/research/financial-analysts-journal/2016/which-trend-is-your-friend).
- `[BACKTEST]` Multi-asset CTA replication 2013-25: the 6-month layer is largely redundant once short and long are held, and a "barbell" does better; simple-weight Trend Sharpe 0.79, dynamic 0.86, heavily optimised 0.72. Vendor-authored preprint — [Etienne et al, arXiv 2025](https://arxiv.org/html/2510.23150v2).
- `[BLOG]` India: no academic 200-day Nifty study found. A 50/200 cross on Nifty is reported to trail a SIP over 30 years, but its count of 238 fires looks odd for that signal, and I would not rely on it — [TFfinance](https://finance-platform-fawn-ten.vercel.app/projects/the-200-dma-golden-cross-on-nifty-confirmed-too-late-to-matter). A 6-month equity-or-debt rule, April 1998 to July 2018, on fund proxies, was riskier than rebalancing with inconsistent reward — [freefincal](https://freefincal.com/nifty-momentum/).
- Local evidence: NIFTY 5.2% a year in Centurion's causal 200-day bull state, 19.6% range, 16.9% bear — [centurion_audit.md](centurion_core/docs/research_notes/Centurion%20golden%20target%20feasibility/centurion_audit.md).

### Inferences
- PTLC's 5.0-point lag is the honest price of a 200-day rule when bears are short: 2015-26 is 2-3 bears, so it is one draw, unfavourable by construction. It also held a 50% middle state, so a middle state alone does not fix participation.
- The window debate is closed for practical purposes. Moving 200 days to anything else is noise-fitting, matching GT2B's split result. Ensembles earn their place by reducing wrong-call cost, not by raising return.
- The asymmetry in the fast weights, none in Corrections and 58% in Rebounds, is the one transferable structure. Fast signals are useful for re-entry, harmful for exit. It maps directly to 49% of gate switches reversing within 21 sessions and supports leaving exit logic alone.
- Goulding's blends have betas near zero, implying long-short timing, so the long-only version is an extrapolation.

### Gaps
- Faber's post-publication numbers and Babu et al's overlay-specific figures remain unretrieved; both were also gaps in the earlier note.
- Signal counts of 160 for 1-day against 38 for 5-day confirmation since 1999 appeared in a search summary, but I could not relocate the page, so treat as unverified.
- No net-of-cost number was found for Goulding's blends.

## Momentum crash protection for a long-only book in rebounds

### Takeaway
The crash in the literature is a short-leg event, so crash fixes do not port to long-only. The long-only problem is lag: winners are low beta at the turn. NSE's own index shows the same zero alpha over 2008-12 that Centurion had, which suggests a market-structure effect, not a Centurion bug. Residual momentum is the best-supported fix, with no Indian test.

### Cited Findings
- `[BACKTEST]` March to May 2009: past-loser decile +163%, past winners +8%. After market declines loser betas can exceed 3 and winner betas fall below 0.5. The bear-market beta asymmetry is driven by losers. From March 2009 to March 2013 winners underperformed the market — [Daniel-Moskowitz NBER](https://www.nber.org/system/files/working_papers/w20439/w20439.pdf).
- `[BACKTEST]` Nifty200 Momentum 30 excess over Nifty 200, parsed from Exhibit 7: 2008 -3.95, 2009 -19.47, 2010 +4.13, 2011 +10.00, 2012 +5.17; 2020 +3.13. Back-filled before the 2020 launch; 2021-25 live: +24.93, -10.36, +17.05, +6.58, -14.13. The other two momentum indices lagged by 25.92 and 29.70 points in 2009 — [NSE whitepaper, April 2026](https://www.niftyindices.com/docs/default-source/indices/nifty200-momentum-30/momentum-strategy-whitepaper_2026.pdf). I validated the column mapping against the quarterly table's active returns; the PDF's table order is scrambled. NSE's since-inception Nifty 500 figure of 18.85% against Nifty 200's 15.08% looks wrong, since the 10- and 15-year figures nearly match.
- `[BACKTEST]` MaxDD on 27 Oct 2008: Nifty200 Mom 30 -67.7% against -63.7%, Nifty500 Mom 50 -70.2% against -63.7%. Stock weights already use volatility-adjusted 6- and 12-month scores — same source.
- `[ACAD-OOS]` US 1930-2017 and 48 countries 1991-2017: idiosyncratic momentum, constant and dynamic vol scaling all reduce crashes and raise Sharpe; US Sharpe and t-statistics roughly double; internationally, idiosyncratic momentum's Sharpe and t-statistic gains are more than twice those of vol scaling; it ranks best in the multiple-model comparison and on global break-even cost; vol-scaling alpha is distinct from it — [Quantpedia on Hanauer-Windmüller](https://quantpedia.com/three-methods-to-fix-momentum-crashes/); [CAIA](https://caia.org/node/5817), which notes after-cost performance was not directly measured.
- `[ACAD-OOS]` Residual momentum cuts time-varying factor exposure and about doubles risk-adjusted profit — [Blitz et al, JEF 2011](https://econpapers.repec.org/RePEc:eee:empfin:v:18:y:2011:i:3:p:506-521). Robust in other universes and periods — [Huij-Lansdorp 2017](https://repub.eur.nl/pub/100801). Milder crashes through Nov 2022 to Feb 2023 in MSCI Developed, long-short, no numbers in the text — [Robeco](https://www.robeco.com/en-ch/insights/2023/02/quant-chart-taming-momentum-crashes).
- `[BACKTEST]` Stops on momentum (Han-Zhou-Zhu): the 2014 abstract's claim of a 10% stop cutting the worst monthly loss from about -49.8% to -11.3% came through a search summary and its page did not load, so check before citing. Most of the gain comes from the loser side, and the winner side's turnover rises from 39% to 49%, gross, with data-snooping concerns — [CXO](https://www.cxoadvisory.com/technical-trading/stop-losses-to-avoid-stock-momentum-crashes/). Long-only, 1927-2013, plain momentum beats the stop versions on CAGR and Sharpe; stops cut drawdown — [Alpha Architect](https://alphaarchitect.com/taming-the-momentum-roller-coaster-fact-or-fiction/), search excerpt only.
- `[BLOG]` A high-to-price momentum variant on Indian stocks: "jury is still out", no numbers in the post — [StockViz](https://stockviz.substack.com/p/momentum-without-the-crash).

### Inferences
- **2008-12 compounded (my calc from the table above): Nifty200 Mom 30 0.944 against parent 0.947, or -1.15% against -1.08% a year. Alpha zero over five years, with the loss in 2008-09, mostly 2009.** The shortfall was only earned back by 2012. That matches Centurion's 2008-12 result, so a rank tweak may lift it only partly. 2021-25: 19.0% against 16.2% a year.
- Daniel-Moskowitz dynamic scaling should not be imported: it shuts off the short-loser risk Centurion does not hold. Scaling down during panic would worsen long-only lag in the rebound.
- Residual momentum is logically aimed at the right mechanism, since it removes the low-beta tilt winners acquire after a decline. Its evidence is mostly long-short, so expect less.
- The evidence on 2020 differs from 2009: NSE's +3.13 suggests the crash was too short for ranks to turn defensive. That is two rebound episodes, a sample of two.

### Gaps
- No long-only residual-momentum study in India or emerging markets was found, and no leg-level split for 2009 or 2020 in India.
- Absolute Sharpe and break-even figures from Hanauer-Windmüller were not retrieved. Blitz's window length should be confirmed before fixing a specification.

## Conviction concentration, beta tilts, sector rotation and factor timing

### Takeaway
Reject all of these as uptrend-return levers. Evidence that concentration or high beta adds net return in confirmed uptrends is weak or points the wrong way, sector rotation fails out of sample, and regime factor timing has one promising preprint against a century-scale negative.

### Cited Findings
- `[BACKTEST]` Multi-factor portfolios: Sharpe rises with concentration only until about 25% of the universe, then stops, with higher trading costs — [Institutional Investor on Research Affiliates](https://www.institutionalinvestor.com/article/b1f4qr0pyxy45k/Research-Affiliates-Says-This-Is-the-Best-Way-to-Do-Factor-Investing).
- `[ACAD]` US 1929-95, momentum earned 0.93% a month after up markets and -0.37% after down — [Cooper et al, JF 2004](https://typeset.io/pdf/market-states-and-momentum-4wtc7g4rqv.pdf). Shanghai, Shenzhen and Hong Kong 1994-2010 show no such link, though momentum lost money in all three in the GFC, especially in months when conditions improved — [Cheema-Nartea](https://researchcommons.waikato.ac.nz/bitstreams/20849f32-e124-4a23-8d14-832cbc4b28a8/download). The earlier India note reports momentum strongest after up markets.
- `[BACKTEST]` Hedegaard: BAB's return rises after strong markets. US monthly gross -0.43% in the lowest past-market quintile, 1.64% in the highest; similar at 6 and 12 months. In-sample, gross, and timing did not beat plain BAB (Sharpe 0.55-0.71 against 0.99) — [CXO](https://www.cxoadvisory.com/volatility-effects/betting-against-beta-plus-market-momentum/).
- `[ACAD-OOS]` Sector rotation by cycle, 1948-2007: 2.3% a year with perfect foresight, and realistic timing errors and frictions remove significance — [CXO on Stangl et al](https://www.cxoadvisory.com/?p=2458). About 1.9% and 1% with one- and two-month timing errors, gross; simply going to cash in recession beats it at one-fifth of the costs, from the working paper via search summary — [Scribd copy](https://es.scribd.com/document/54155014/Sector-Rotation-Over-Business-Cycles). Predictability is no different from chance once assumptions relax — [Molchanov-Stangl](https://mro.massey.ac.nz/handle/10179/69590).
- `[ACAD-OOS]` A century of factor data: predictability modest, likely fails to overcome frictions — [Ilmanen et al](https://papers.ssrn.com/abstract=3400998). Value-based factor timing shows no robust gain — [AQR](https://www.aqr.com/Insights/Research/Journal-Article/Contrarian-Factor-Timing-is-Deceptively-Difficult-Supplement?aqrPDF=1).
- `[ACAD-OOS]` Counter-evidence: long-only US factor indices, sparse jump model, Black-Litterman, 2007-24, 5 bp costs: information ratio 0.05 to about 0.4 against the market; on par with benchmarks until about 2012, then ahead. Preprint — [Shu-Mulvey, arXiv 2024](https://arxiv.org/pdf/2410.14841).

### Inferences
- Up-market momentum gives a reason to keep full rank exposure in uptrends; it does not support concentrating further. Centurion's 20 names from a universe of about 1,000 are roughly 2%, far past the 25% point. It is already concentrated.
- High-beta tilts in uptrends buy the leverage-substitute crowd's trade, and Hedegaard's result points the wrong way, though BAB is long-short so the read-across is indirect. Beta also rises fastest into a crash, breaching the bounded-tail intent without gross leverage. That is structural leverage by another name.
- A 7-index test with a tuned jump model is too complex to pre-register with 18 years and 5 bears. Shu-Mulvey's flat period to 2012 is a warning, not a reassurance, for 2008-12.

### Gaps
- No direct study varying momentum holdings from about 10 to 100 names net of costs was found; this is an open question. Nothing Indian on concentration by regime or sector rotation by regime was found.
- Hong-Sraer, Ehsani-Linnainmaa and Gupta-Kelly factor momentum were not retrieved. "Factor Timing: Keep It Simple" is paywalled.

## Regime detection: HMM and change-point versus simple rules

### Takeaway
The HMM is the published loser even in the paper promoting its replacement. Jump models look better but depend on one research line, three correlated indices and about four bears. Breadth, the gate's other leg, has essentially no out-of-sample support. No credible Indian test exists.

### Cited Findings
- `[ACAD-OOS]` 1990-2023, 10 bp one-way, one-day delay, buy-and-hold / HMM / jump: S&P CAGR 10.2 / 8.5 / 11.2, Sharpe 0.48 / 0.54 / 0.68, MaxDD -55.2 / -28.9 / -26.6, turnover 0 / 141% / 44%. DAX CAGR 6.8 / 6.4 / 8.6, Sharpe 0.30 / 0.35 / 0.44, turnover 246% / 170% for HMM and jump. Nikkei 0.8 / 2.5 / 4.7, Sharpe 0.12 / 0.19 / 0.31. Time in market 68-73% HMM, 75-84% jump. The HMM's CAGR is described as on par with the index. **On DAX and Nikkei the HMM underperforms the market with a 5-day delay**, while the jump model's Sharpe stays better than or comparable to the index at a two-week delay — [Shu-Yu-Mulvey](https://arxiv.org/pdf/2402.05272).
- The same paper notes HMM sensitivity to mis-estimation and mis-specification from limited samples and high persistence — same source.
- `[BACKTEST]` Breadth: forward 12-month S&P returns averaged about 10.4% with about 86% positive when 70%+ of stocks were above their 200-day, no holdout — [Nationwide](https://www.nationwide.com/financial-professionals/blog/markets-economy/articles/how-sustainable-is-the-current-bull-market-run). Moving-average breadth and forward returns were uncorrelated in MSCI country data — [Ned Davis via IBKR](https://ibkrcampus.com/wp-content/uploads/2023/01/2021-WB-4065-Ned-Davis-Research-October-25-2022-v2.pdf).
- `[BACKTEST]` India VIX: asymmetric link to Nifty; rule moving to large-caps as VIX rises and mid-caps as it falls; out-of-sample results not seen — [Chandra-Thenmozhi, Vikalpa 2015, via search](https://doi.org/10.1007/s40622-014-0070-0). India HMM: three regimes, walk-forward, results unclear in the excerpt — [IAENG](https://www.iaeng.org/IJAM/issues_v55/issue_10/IJAM_55_10_18.pdf).

### Inferences
- Turnover is the discriminator: 141-290% for the HMM against 44-170% for the jump model. Persistence, not intelligence, is what pays. Centurion's 49% reversal rate is the same disease in a simple rule. The analogous cheap fix is a penalty on switching, such as hysteresis, rather than a new model.
- Note the Sharpe gain partly comes from time in safe assets, at 80% invested. The market evidence is not an alpha claim.
- With 5 Indian bears, an HMM's parameters would be fitted to a few events, and the expected-gain range in the earlier note (0-2% a year) remains the right ceiling. Do not build one.

### Gaps
- No published head-to-head of HMM or change-point against a moving-average rule, with costs, was found.
- No peer-reviewed out-of-sample breadth test was found, in any market. This gap is the most relevant to Centurion's gate, and the breadth leg should be tested by removal.
- Change-point methods specifically were not retrieved.

## Drawdown control: CPPI, Grossman-Zhou and de-risking

### Takeaway
Neither insurance nor buy-and-hold dominates. Rules cut tails and cost mean, so they are insurance, not return. Centurion's rule rarely fires because the gate already holds drawdown near -30%, so a tighter one would mainly lock in losses at the low.

### Cited Findings
- `[ACAD-OOS]` Block bootstrap of stop-loss, synthetic put and CPPI: buy-and-hold has higher mean excess return yet neither stochastically dominates the other; a higher CPPI multiple adds upside and harms protection; daily rebalancing dominates despite cost — [Annaert et al, JBF 2009](https://econpapers.repec.org/RePEc:eee:jbfina:v:33:y:2009:i:2:p:272-280).
- `[ACAD]` Rej-Seager-Bouchaud give distributions of last-drawdown depth and length, as a test of when to lower an assumed Sharpe, not as a de-risking rule — [arXiv 1707.01457](https://arxiv.org/abs/1707.01457v2).
- Theory: 2026 preprint on drawdown-modulated control generalises the cushion rule to multi-asset discrete time; no empirical test — [Hsieh](https://arxiv.org/pdf/2609.23272).
- Local: levered book's lowest NAV 63% of start, 56% without the drawdown rule; unlevered MaxDD -29.5% — [centurion_audit.md](centurion_core/docs/research_notes/Centurion%20golden%20target%20feasibility/centurion_audit.md).

### Inferences
- With a -29.5% MaxDD, the 20% rung is the only one that matters. Since it fires late in moves the gate has already cut, its marginal value is likely about zero when unlevered, and real only for a levered book. Keep it for survival and judge it by removal, per the earlier note.
- CPPI is a poor fit: its exposure falls to zero as the cushion goes, which is a forced version of the rebound lag Centurion already has (logic, not an empirical result).
- Use Rej-style tests as a live monitor on Sharpe belief, not as an order generator.

### Gaps
- No Indian drawdown-rule study, and no live-fund evidence, was found.
- A claim that Grossman-Zhou under proportional costs gives negative expected returns near the limit appeared in a search summary without a clear source, so it is not asserted.

## Sizing: fractional Kelly and what a cash-only book can do

### Takeaway
Kelly is uninformative here because it never binds below 1.0x. The binding limits are the bounded-tail rule and no leverage, so sizing has no route to more return beyond deploying existing idle capital.

### Cited Findings
- Local inputs: Sharpe 0.60, vol 11%, mean invested 70%, stocks 41% — [centurion_audit.md](centurion_core/docs/research_notes/Centurion%20golden%20target%20feasibility/centurion_audit.md). Kelly mechanics and the 10th-percentile Sharpe of 0.29 are in the earlier note — [regime_leverage_ruin.md](centurion_core/docs/research_notes/Centurion%20golden%20target%20feasibility/regime_leverage_ruin.md).

### Inferences
- My calc: f* = Sharpe / vol. Full Kelly 5.5x, half Kelly 2.7x. At Sharpe 0.29: 2.6x and 1.3x. So even a heavily shrunk Kelly exceeds 1.0x, and the rule says only that being under 100% needs a reason.
- The reason exists: conditional edge is believed low in risk_off. But with 5 bear episodes, a state-conditional Sharpe is not estimable, so state-based Kelly would be noise-fitting.
- A cash book's real sizing levers are three: how much sits idle in risk_off (the half-step), the split between sleeves, and equal-risk weighting within sleeves. All are already covered by the earlier note or this one's main rule.

### Gaps
- No new Kelly literature was searched, as the earlier note covers it. Audited cash-only Kelly evidence remains absent.
