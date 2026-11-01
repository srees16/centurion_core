# India uncorrelated return sources at swing or positional horizon (as of 31 Oct 2026)

Evidence grades used on every return claim:
- **A** audited live: fund NAV, exchange-calculated index, or regulator data. **A-** live but unaudited or self-reported through an aggregator.
- **B** academic with out-of-sample testing.
- **C** backtest only, including in-sample academic work and practitioner tests.
- **D** marketing, anonymous, or unverifiable.
- **R** regulatory or tax fact, not a return claim.

Research limits, so the reader can weigh these notes:
- The web-search budget ran out partway through, so later topics rest on fewer sources.
- SSRN and ResearchGate returned 403, so Pillai, Agarwal and the delta-neutral study are read at abstract or snippet level only.
- PDF factsheets could not be parsed, so several fund and index figures are search-snippet readings.
- Nothing under centurion_core was read and no code was run.
- Arithmetic in the Inferences sections is hand calculation.

## Ranked list: how many real streams exist, and can diversification lift Sharpe from 0.6 towards 1.4?

### Takeaway
There are about 4 to 5 independent risk sources, not 10 or more: equity momentum, gold and commodity trend, rates, short volatility (which merges with equity in a crash), and small idiosyncratic events. Diversification alone probably reaches about 0.8 to 1.0 on realistic inputs. No source found supports 1.4, which needs either mean pairwise correlation under about 0.13 with 8 or more streams at net Sharpe 0.5, or a much stronger core.

### Cited Findings
- A long-only momentum index product shows Sharpe 0.44 at beta 1.00, portfolio data as of 31 Aug 2026 (A-, snippet read, PDF not parsed) — [ICICI Prudential Nifty 200 Momentum 30 ETF factsheet](https://www.icicipruamc.com/blob/knowledgecentre/factsheet-schemes/Passive%20Schemes/ETFs%20Schemes/ICICI%20Prudential%20Nifty%20200%20Momentum%2030%20ETF.pdf)
- Passive Nifty 50 TRI, 10Y G-Sec and GoldBees mix, 7 Nov 2016 to 31 Dec 2025: 50/25/25 gave 13.2% CAGR at 8.8% vol, against 100% equity at 14.4% and 16.3% (C, summarises third-party backtests, method unverified) — [Multibagg](https://www.multibagg.ai/market-pulse/articles/nifty-gold-allocation-backtests-india-cmoxatc420007qm0jcqu7c76s)
- Left-tail correlations exceed right-tail correlations across asset classes, 1970 to 2017, though "all correlations go to 1" is called an exaggeration (academic, FAJ 2018, global not India) — [T. Rowe Price brief](https://www.troweprice.com/content/dam/ide/articles/pdfs/2018/q4/when-diversification-fails.pdf); [Alpha Architect](https://alphaarchitect.com/when-diversification-fails/)

### Inferences
Ranked table. Sharpe, correlation and capacity figures are **my estimates** unless a section below cites them. Rank reflects contribution to book Sharpe after evidence quality, not stand-alone Sharpe.

| # | Stream | Net Sharpe est. | Corr to long momentum, normal then crash | Capacity at ₹0.5-3 cr | Capital or margin | Tail | Rule or cost change | Best evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | Gold in INR (ETF, MCX futures, optional trend filter) | 0.4-0.6, after a hot decade | -0.05 to -0.10, then negative | High | ETF unlevered; MCX margin unsourced | Long drawdowns; Oct 2026 correction | SGB secondary tax exemption gone Apr 2026; CTT unchanged | C, A- |
| 2 | REITs and InvITs | 0.4-0.6 forward (0.75 trailing 5Y) | 0.15-0.34, crash unknown | Medium | Unlevered | Rate and credit shock; shares risk with gilts | 90:10 index launched Jul 2026 | A |
| 3 | MCX multi-commodity trend, ex gold | 0.3-0.5, thin breadth | About 0 (0.04 on a gold-only test), crash unknown | Medium, liquidity by contract unsourced | Futures margin, unsourced | Gap moves, whipsaw | CTT 0.01% now one-fifth of equity futures STT | B globally, none on MCX |
| 4 | G-Secs and gilt funds | 0.1-0.35 | About 0, crash unsourced | High | Unlevered | Tightening cycle now | Repo raised to 5.50% Oct 2026 | A |
| 5 | Defined-risk monthly NIFTY VRP (put spreads, condors) | 0 to 0.4 | 0.3-0.5, then 0.7+ | High relative to book | Margin 10-25% of notional, unsourced at current levels | Bounded by wings | ELM, lots, STT +50% | C, conflicting |
| 6 | Buyback, open-offer, delisting, IPO events | Unmeasured, maybe 0.5+ on deployed capital | Low, rises in crash | Very low, retail quota based | Cash | Acceptance-ratio miss, failed offer | Buyback tax back to capital gains Apr 2026 | D, anecdote |
| 7 | Stock-futures market-neutral or pairs | 0 to 0.5 | 0.2-0.5 as run by live funds | Medium | Margin plus basis | Squeeze, crowding | Futures STT +150% | C, conflicting; A- on beta |
| 8 | International equity (GIFT, LRS) | 0.3-0.5 | Unsourced | Limited by LRS | Unlevered | Same global risk in crash | Domestic ETF premiums up to 111.8% | Unsourced on correlation |
| 9 | Arbitrage funds | Not a stream, cash substitute | About 0 | High | Unlevered | Basis compression | Equity tax treatment, STT | A- |
| 10 | Calendar and flow effects | Overlay only, under 0.2 alone | Same as equity | n/a | n/a | n/a | Expiry day moved | C, weak |
| 11 | USD-INR futures | Excluded | n/a | n/a | n/a | n/a | Rule tightened 10 Oct 2026; possible purpose restriction | R |
| 12 | Naked strangles, dispersion, unhedged put-write | Excluded | High in crash | n/a | n/a | Breaches 10-15% cap or unevidenced | n/a | C, negative |

Sharpe arithmetic (rf assumed 6%):
- Uncorrelated streams add in squares: SR² = sum of each SR². A 1.4 target needs 1.96. Base 0.6 gives 0.36, so 1.60 must come from elsewhere: four uncorrelated streams at 0.63 each, or eight at about 0.45.
- With N equal streams at Sharpe s and correlation ρ, SR = s × √(N / (1 + (N-1)ρ)). For s = 0.5 and target 1.4: ρ = 0 needs 8 streams, ρ = 0.05 needs 13, ρ = 0.10 needs 33, and ρ above 0.128 cannot reach it at any N. For s = 0.7: ρ = 0 needs 4, ρ = 0.10 needs 6, ρ = 0.20 needs 16.
- Using a realistic set (0.6 core, 0.5 gold, 0.4 REIT/InvIT, and four streams at 0.3) with optimal weights and uniform correlation, book Sharpe is about 1.06 at ρ = 0, 0.95 at 0.05, 0.87 at 0.10, 0.76 at 0.20, 0.71 at 0.30. This assumes the inputs are known exactly, so real results land lower.
- Book Sharpe 0.6 at 11.5-13.2% CAGR implies vol near 9-12%. Sharpe 1.4 at that vol implies about 19-23% CAGR, or the same 12-13% CAGR at only about 4-5% vol.

Lead worth testing in-house: the simple 50/25/25 passive mix works out to about 0.82 Sharpe against 0.52 for pure equity. That is above the 0.6 sleeve books, but on a window where INR gold compounded 18.1% over 10 years, so it says more about gold in 2016-25 than about the method. Testing it over 2008-25 is cheap and would reveal whether sleeves are adding anything over a passive blend.

### Gaps
- No India-accessible strategy with audited multi-year Sharpe of 1 or more turned up. That is absence in a limited search, not proof of absence.
- Correlation of each stream with a long-only momentum index specifically is not published. All figures use Nifty or are estimates.
- Multi-asset fund Sharpe as a live benchmark was not retrieved.

## Why have Centurion's sleeves cut risk rather than raised Sharpe, and what could raise it?

### Takeaway
This is the expected result when sleeves have low own Sharpe and low risk weight. A stream lifts book Sharpe only if its Sharpe exceeds ρ × book Sharpe, and the lift is quadratic, so a 0.3 sleeve at zero correlation moves 0.60 only to 0.67. The sources most likely to help are ones with their own positive excess return and a non-equity driver: gold and commodity trend, then real-asset yield.

### Cited Findings
- 10-year G-Sec near 6.87% on 21 Aug 2026 (R-type market data) — [Trading Economics](https://tradingeconomics.com/india/government-bond-yield/news/577154); repo 5.50% after the 7 Oct 2026 hike — [5paisa](https://www.5paisa.com/index.php/blog/rbi-mpc-meeting-schedule)
- 10-year constant-maturity gilt funds showed roughly 5.9-6.0% 5-year CAGR at 6 Sep 2026 (A-) — [Arthgyaan](https://arthgyaan.com/funds/categories/gilt-with-10-year-constant-duration-debt-funds)
- Net of all costs every short-vol variant tested on Nifty 2015-25 was negative, best about -0.9% a year (C) — [Pillai, SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6876580)
- Long-short AIF category averages moved about 0.4 of BSE 500 in Jan and Mar 2025 (A-) — [PMS Bazaar Jan 2025](https://pmsbazaar.com/Blogs/Jan-2025-AIF-Performance-Market-Pressures-Weigh-on-Returns-Long-Short-Funds-Outshine); [PMS Bazaar Mar 2025](https://pmsbazaar.com/Blogs/AIFs-Delivered-a-Strong-Comeback-in-March-2025-as-Markets-Rebound)

### Inferences
- **Thin carry.** A gilt sleeve earning roughly cash has Sharpe near zero, and top arbitrage funds out-earned 10Y gilt funds over the last 5 years. So it blends the book toward cash: CAGR and vol fall together, Sharpe barely moves. That matches the observed 11.5-13.2% at about 0.6.
- **Capital weight is not risk weight.** A 10-25% capital sleeve at a third of equity vol carries a small share of book risk, so it cannot move book Sharpe much without leverage.
- **Overlays share the same driver.** NIFTY spread overlays are short volatility with equity delta, so ρ is not near zero and rises in sell-offs, which fails the ρ × SR test when it matters.
- **Costs rose.** STT on options is up 50% and on futures 150% since April 2026, which hits exactly the rolled sleeves.
- **No cheap leverage to rescale.** Raising the risk weight of low-vol sleeves needs margin, and any cash-collateral requirement limits stacking (unverified, see the option section).
- What could genuinely raise Sharpe: higher-breadth commodity trend if MCX liquidity permits; gold held at a real risk weight; InvIT yield; and improving the core alpha, because the core's own Sharpe enters squared and is the largest single term.

### Gaps
- Actual sleeve weights, sleeve Sharpes and in-crash correlations inside Centurion were not available and centurion_core was out of scope. The diagnosis above is structural and should be checked against those numbers.

## NIFTY and BANKNIFTY option selling and the variance risk premium: size, Sharpe, tails, and post-SEBI results

### Takeaway
The premium exists but is small and hard to capture: the best-cost study finds every systematic short-vol variant net negative over 2015-25, with tail loss rather than costs as the main cause. Costs and contract rules have worsened since, and premium compressed to record lows in late 2025. Treat it as a bounded, modest sleeve, not a Sharpe engine.

### Cited Findings
**Size of the premium**
- Consistently positive, significant VRP on daily NIFTY and India VIX data, April 2021 to July 2025, tested via short VIX futures, synthetic variance swaps and delta-hedged straddles; profits in calm regimes, concentrated losses in vol shocks (C, listed on a LinkedIn profile, publisher unverified) — [Arsh Hora profile](https://in.linkedin.com/in/arshhora)
- 1-minute option data August 2022 to March 2026: costs leave 72.4% of gross edge, median net edge about 1.13 vol points; the author makes no claim of exploitability or out-of-sample validity (C, descriptive) — [Agarwal, SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6530119)
- Vega-weighted order flow in the most expensive Nifty options drives VRP changes (academic, in-sample) — [Chakrabarti and Kumar 2017](https://sites.uwasa.fi/mfjuva/?p=2597)

**Sharpe and strategy returns**
- January 2015 to April 2025, net of all costs: all strategies negative; best put-write at 0.94 moneyness about -0.9% a year; straddle variants about -44% to -45% a year; tail risk primary, costs secondary; concludes VRP not capturable by retail under current microstructure (C, working paper, abstract only, sizing unknown) — [Pillai, SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6876580)
- Monthly Nifty short strangle 2007-2019, 141 months: more successful and profitable than long strangle at every leg, third leg best; reports points, not Sharpe (C, journal) — [Bangur, Journal of Investment Strategies](https://www.risk.net/ja/node/7722171)
- Short straddles and strangles on Nifty 50 and constituents, 2019-2024: positive mean returns, but transaction costs omitted and sample short (C) — [ResearchGate listing](https://www.researchgate.net/publication/404728908_Profitability_of_Delta-Neutral_Option_Strategies_on_Nifty_50_and_Constituent_Stocks_Evidence_from_2019-2024)
- Garg and Vipul (2015) found Nifty short vol significantly profitable before costs, diminished after (C, cited secondhand) — [Somaiya USDINR study](https://svu-naac.somaiya.edu/C3/DVV/3.4.4/NAAC+Publication+Drive/192.pdf)
- Frictions materially reduce OTM put-writing profit, more so when written aggressively (US, academic) — [Santa-Clara and Saretto 2009](https://personal.utdallas.edu/~axs125732/OptionsJFM2009.pdf)
- Marketing to discount: a course claims Sharpe 3.73 with 2.12% drawdown (D) — [Udemy](https://www.udemy.com/course/options-selling-strategy-for-monthly-income/)
- The headline arXiv VRP result (Sharpe 4.3-5.8) is S&P 500 0DTE, so it is neither India nor swing — [arXiv 2608.24786](https://arxiv.org/html/2608.24786)

**Tail events**
- 4 June 2024: Nifty closed down 5.93% to 21,884.5, its sharpest since March 2020 — [Multibagg](https://www.multibagg.ai/market-pulse/articles/nifty-june-4-crash-cmvmepzt2000e2vpiq95gaw4r); [Angel One](https://www.angelone.in/news/share-market/election-day-chaos-indian-stock-market-crashes-to-record-lows)
- India VIX close that day is reported inconsistently at 25.89, 28.05 and 31.71 — [5paisa](https://5paisa.com/news/election-results-2024-sensex-nifty-50-cracks-investors-loose)
- Late 2025: India VIX at record lows, 3-month realised vol below every other major market, Nifty inside a 1.5% range for over 150 sessions, derivatives turnover in its first annual fall since 2017; vol-selling returns shrink — [investingLive, 21 Dec 2025](https://investinglive.com/stock-market-update/indias-stock-market-enters-deep-freeze-record-low-volatility-as-options-boom-hits-a-wall-20251221)

**Rule and cost timeline** (grade R)
- Oct 2024 package, effective 20 Nov 2024: minimum index contract about ₹15 lakh, up from ₹5 lakh; one weekly expiry per exchange; extra 2% ELM on short options on expiry day — [Business Standard](https://www.business-standard.com/amp/markets/news/sebi-announces-six-key-changes-to-curb-speculation-in-derivatives-trading-124100101316_1.html); [Business Today](https://www.businesstoday.in/markets/story/sebis-new-fo-rules-kick-off-today-heres-what-are-these-changes-454352-2024-11-20)
- Upfront premium collection and removal of expiry-day calendar-spread margin benefit: 10 Feb 2025 per [Business Standard](https://www.business-standard.com/amp/markets/news/two-key-changes-f-o-norms-will-now-take-effect-from-february-10-125010901188_1.html); another source says 1 Feb — [FII and DII Data](https://fiianddiidata.com/sebi-new-fo-rules-2026/)
- Delta-based FutEq OI, index options limits of ₹1,500 crore net end-of-day and ₹10,000 crore gross, index futures ₹500 crore or 15% of OI — [Business Standard, May 2025](https://www.business-standard.com/markets/news/sebi-fo-revamp-new-oi-metric-position-limits-july-december-rollout-125052901798_1.html)
- From 1 Sep 2025: Nifty weeklies expire Tuesday, Sensex Thursday, other indices monthly only — [FII and DII Data](https://fiianddiidata.com/sebi-new-fo-rules-2026/); BankNifty monthly expiry falls on the last Tuesday — [Business Today](https://businesstoday.in/markets/stocks/story/nse-bse-fo-expiry-days-what-has-changed-for-stock-investors-history-more-480781-2025-06-18)
- Jan 2026 contracts: Nifty lot 65 from 75, BankNifty 30 from 35 — [FII and DII Data](https://fiianddiidata.com/sebi-new-fo-rules-2026/)
- From 1 Apr 2026: options premium STT 0.15% from 0.10%, exercised 0.15% from 0.125%, futures 0.05% from 0.02% — [ICICI Direct, citing NSE/FATAX/73524](https://www.icicidirect.com/futures-and-options/articles/stt-changes-in-budget-2026-what-f-o-traders-need-to-know). Before Oct 2024 the options rate was 0.0625% — [Algoshastra](https://algoshastra.in/learn/honest-backtest-india)
- From 1 Apr 2026: retail API orders must use a registered app and whitelisted static IP (secondary, unverified against SEBI) — [FII and DII Data](https://fiianddiidata.com/sebi-new-fo-rules-2026/)
- Closing auction live 3 Aug 2026 for F&O stocks — [Angel One](https://www.angelone.in/news/market-updates/sebi-closing-auction-session-from-august-3-2026-new-timings-and-rules-for-f-o-stocks)
- From 7 Dec 2026: regular trading to 3:30 PM, auction 3:30-3:40 PM, F&O final settlement on VWAP of 3:00-3:30 PM, expiring index contracts stop at 3:30 PM, other F&O trade to 3:45 PM — [Zerodha Z-Connect, 16 Oct 2026](https://zerodha.com/z-connect/?p=461083)
- Feb 2026: SEBI chair said no further derivatives measures "for now" and no return of BankNifty weeklies; a Sept 2025 promise of a weekly-expiry consultation had not been seen delivered in anything found — [Informist](https://informistmedia.com/MoneyWire/44239/F-O-Market-Not-contemplating-any-measures-for-F-O-mkt-for-now-says-SEBI-Pandey); [Outlook Money](https://www.outlookmoney.com/amp/story/news/sebi-to-float-consultation-paper-on-endin-futures-and-options-weekly-expiry-soon-tuhin-kanta-pandey)

**Costs and participants**
- Hedged 4-leg strangle, 1 lot: about ₹292 round trip, 0.71% of ₹40,950 net premium; Nifty futures round trip about ₹899 at 23,150 × 65, of which STT 84% (author's estimate, June 2026) — [TradingView post](https://www.tradingview.com/chart/NIFTY1!/q817Binb-What-a-NIFTY-trade-really-costs-after-the-STT-hike/)
- SEBI: about 91% of individual traders lost money in FY25, net loss ₹1,05,603 crore against ₹74,812 crore in FY24; quarterly unique traders fell from 61.4 lakh to 42.7 lakh (A, regulator) — [Business Standard](https://www.business-standard.com/amp/markets/news/net-losses-of-traders-in-fo-widens-in-fy25-sebi-study-125070701221_1.html); [The Week](https://www.theweek.in/wire-updates/business/2025/07/07/dcm92-biz-sebi-derivatives-study.html)

### Inferences
- Pillai's -44% straddle figure implies aggressive sizing or margin assumptions, so its sign matters more than its size. The put-write near zero is the more useful result: after tails, the premium was about paid for by losses.
- Rough net edge of 1.13 vol points on, say, 12-15 vol implies only a small premium. Against a -6% day, it takes many months to earn back, which is the Sharpe problem in one line.
- Fewer retail traders and record-low vol both suggest less rich premium than the 2021-24 samples, so post-2024 Sharpe should be assumed lower than those windows. No study isolates this cleanly yet.
- Most SEBI changes hit weekly and expiry-day trading. Monthly positions closed before expiry avoid the 2% ELM and lost calendar benefit, so a monthly swing design is much less affected. The STT rise is what bites: 12 monthly hedged cycles cost about ₹3,500 per lot a year, about 0.23% of notional before slippage.
- One lot is now about ₹15 lakh notional, so a ₹50 lakh book carries 30% notional per lot and cannot size finely.
- A 30% gap on one naked lot loses about ₹4.5 lakh, or 9% of a ₹50 lakh book. Only one naked lot fits the 10% cap, which is why defined-risk structures are the only ones compatible with the bounded-tail rule.
- Position-limit changes at ₹1,500 crore are irrelevant at this capital size.
- BankNifty has no weekly and a concentrated sector driver, so it adds little over monthly Nifty.

### Gaps
- March 2020 loss figures for systematic Indian short-vol strategies were not found.
- No published post-2024 Sharpe for monthly Indian strangles or condors exists in what I found. AlgoTest, Stockmock, Sensibull and Opstra public results were not retrieved.
- An SSRN-level study isolating before and after the Nov 2024 rules at monthly tenor was not found.
- The "contract value later raised to ₹20-30 lakh" claim in one search result conflicts with 23,150 × 65 = ₹15 lakh and is unverified.
- The BankNifty lot history conflicts across sources (15 to 30 in Nov 2024, against 35 to 30 in Jan 2026). Both may be true with interim changes, not confirmed.

## Hedged and defined-risk structures, dispersion, put-writing indices, and tail-hedging cost

### Takeaway
US put-writing shows a real but equity-linked Sharpe of about 0.65, gross of costs, with heavy negative skew. It is a lower-vol equity substitute, not an uncorrelated stream. I found no NSE put-write index, no quantified Indian dispersion result, and no Nifty tail-hedge cost data.

### Cited Findings
- PUT index over 32+ years: Sharpe 0.65 vs 0.49 for S&P 500 (A-, exchange index history, gross of costs, Cboe sponsored) — [Cboe summary](https://www.cboe.com/insights/posts/white-paper-shows-volatility-risk-premium-facilitated-higher-risk-adjusted-returns-for-put-index/)
- Mid-1986 to 2015: PUT 10.13% at 10.16% vol, Sharpe 0.67; S&P 500 9.85% at 15.26%, Sharpe 0.47 — [Indexology](https://www.indexologyblog.com/2016/02/18/paper-by-professor-bondarenko-has-intriguing-new-analysis-of-put-and-wput-indexes/)
- Feb 2006 to Dec 2015: PUT 6.6% against S&P TR 7.1%, gross Sharpe 0.52 vs 0.46; PUT skew -2.09 vs -0.79; high turnover not costed — [CXO Advisory](https://www.cxoadvisory.com/equity-options/performance-of-cboe-putwrite-indexes)
- A PUT-tracking fund launched 2007 showed 0.99 correlation to the index, so the US strategy is live-implementable (A-) — [Plansponsor](https://www.plansponsor.com/new-study-focuses-on-benchmark-for-collateralized-putwrite-strategy/)
- PUT beat BXM by about 1.1% a year, 1986-2015, attributed to about four hours of expiry-day return timing — [Quantpedia on Israelov](https://vvv.quantpedia.com/es/?p=723)
- Hedged vs naked margin gap was only about 10% in one Zerodha calculator example (₹1,19,029 vs ₹1,33,669, old Nifty level, D) — [TradingQnA](https://tradingqna.com/t/capital-requirement-differences-between-iron-condor-and-short-strangle/74018)
- Half of margin must be cash or equivalent, per a practitioner post; Zerodha's own policy text was not surfaced (D) — [TradingView post](https://es.tradingview.com/chart/NIFTY/2tjtkZAv-Firefighting-a-Short-Iron-Condor-Strategy-on-Nifty50-on-Expiry-D); liquid funds can be pledged for margin — [Zerodha support](https://support.zerodha.com/category/console/portfolio/pledging/articles/margin-on-liquid-funds)
- Protective puts held long term on S&P 500 gave poor value; cutting exposure gave better downside protection (academic, reported secondhand) — [Benzinga on Israelov 2019](https://benzinga.com/z/44501856)
- 60/40 with 5% OTM puts about 7.0% vs 9.0% unhedged over 8 years, about 2% annual drag (secondhand, unconfirmed) — [Advisorpedia](https://advisorpedia.com/markets/the-cost-of-tail-risk-protection)
- Dispersion is usually profitable when correlation is low and loses in stress; index writers earn more VRP than single-stock writers (US and global, academic via Quantpedia) — [Quantpedia](https://quantpedia.com/strategies/dispersion-trading/)
- An IIM Bangalore study tested naive dispersion on CNX Bank for one year; results not visible — [IIMB repository](https://repository.iimb.ac.in/handle/2074/18212); QuantInsti student projects exist, not validated — [QuantInsti](https://blog.quantinsti.com/dispersion-trading-on-nse-stocks/)

### Inferences
- Put-write Sharpe 0.65 at about two-thirds of index vol means it mostly keeps equity beta and removes the upper tail, so its correlation with long momentum will be high. It would fail the ρ × SR test as a diversifier.
- Dispersion is short correlation, which spikes in exactly the crash this exercise wants to hedge. Excluded on structure, as well as on lack of evidence.
- Buying wings is effectively a partial tail hedge, so condor net Sharpe already embeds hedge cost. Layering a separate tail hedge likely cuts Sharpe further. Exposure reduction looks better, in line with Israelov.

### Gaps
- No NSE Indices put-write, buy-write or option-strategy index or whitepaper was found. It may exist but was not reachable.
- No Nifty skew or annual tail-hedge cost data was found.
- Current exchange margins and the cash-share rule need a live Kite margin check.

## Cash-futures arbitrage and basis trades

### Takeaway
Top direct-plan arbitrage funds earned about 6.6-6.7% in the year to 30 Oct 2026, about 7.0-7.4% over 3 years and about 6.8-6.9% over 5, with very low vol. They are a cash substitute with a tax edge rather than a diversifying return stream, and they effectively set the hurdle for every low-risk sleeve. No evidence shows retail self-execution beats them after costs.

### Cited Findings
- Direct plans at 30 Oct 2026: Edelweiss 6.66% 1Y, 7.37% 3Y, 6.92% 5Y; Nippon 6.71%, 7.28%, 6.83%; Axis 6.68%, 7.31%, 6.84%; SBI 6.59%, 7.24%, 6.83%; Bajaj 6.61%, 6.98%. Category average 1Y about 6%, which spans all plans and is lower than these leaders (A, NAV-based, via aggregator; top funds, so survivorship applies) — [INDmoney](https://www.indmoney.com/mutual-funds/best-arbitrage-funds)
- Earlier: arbitrage funds about 7-8% over one year against about 6-7% for liquid funds, as of 11 Sept 2025 (A-, aggregator) — [Policybazaar](https://www.policybazaar.com/mutual-funds/arbitrage-funds)
- Liquid funds have generally outpaced arbitrage pre-tax, so tax decides the outcome (A-) — [FundsIndia](https://fundsindia.com/blog/mf-research/fundsindia-strategies-liquid-or-arbitrage-funds-for-short-term-money/15233)
- Equity-taxed: 20% short-term under 12 months, 12.5% long-term above ₹1.25 lakh; usually 0.25% exit load within 30 days — [5paisa](https://www.5paisa.com/stock-market-guide/mutual-funds/arbitrage-vs-liquid-funds)
- Standard deviation 0.39-0.41%, 36 monthly points, Dec 2025 to Apr 2026 (A-, snippet, basis unclear) — [Baroda BNP Paribas Arbitrage factsheet](https://www.barodabnpparibasmf.in/assets/download_documents/Baroda_BNP_Paribas_Arbitrage_Fund_1_10746.pdf); another fund shows 1.30% since 2015 — [Sharpely](https://sharpely.in/mutual-funds/axis-arbitrage-fund/27587/performance)

### Inferences
- Futures STT at 0.05% adds about 0.72% a year of statutory cost for a monthly-rolled futures leg, using the 0.06% round trip above. Self-run basis trades now need a wider spread than funds, which also net internally. Retail self-execution looks unattractive.
- Best use is as margin collateral or a parking sleeve, if pledge terms allow. A near-zero-vol stream adds no meaningful Sharpe once measured against cash.
- 1Y returns near 6.7% are down from 7-8% a year earlier, though the April 2026 STT effect cannot be separated from lower rates and basis. Spread over repo of about 1.4 points (repo was 5.25% until 7 Oct) is close to the 10Y term spread, for almost no duration risk.
- Use about 6.5-6.9% as the practical rf hurdle for Indian sleeves, rather than repo. Sharpe measured against that hurdle will be lower than against repo.

### Gaps
- One Whalesbook article had "[Simulated cite]" markers, so its figures were discarded.
- The effect of the April 2026 STT rise on fund returns is not isolated anywhere I found.
- Nifty 50 Arbitrage Index long-run return and vol: only a 2010 factsheet surfaced, unreadable.
- Basis levels during stress and retail delivery-leg cost were not sourced.

## Stock futures long-short, market-neutral equity, pairs and statistical arbitrage on NSE

### Takeaway
Live Indian long-short funds behave like about 0.4 beta equity, with average returns near long-only, not as market-neutral alpha. Published pairs evidence conflicts: up to 34% a year best-case on 2011-17 stock futures, against Sharpe 0.09 in a 2016-25 walk-forward on large caps. A short leg probably adds crash protection more than Sharpe.

### Cited Findings
- FY2025 averages: long-short AIFs 8.68% (about 30 funds), long-only 8.7% (70+), Nifty 50 TRI 6.65%, BSE 500 TRI 5.96%; long-short top 31.79%, bottom quartile 0.8%; basis mixed between net and gross (A-) — [PMS Bazaar FY2025](https://pmsbazaar.com/Blogs/From-Highs-to-Headwinds-How-AIFs-Performed-in-FY2025)
- Jan 2025: long-short -1.45% vs BSE 500 TRI -3.43% — [PMS Bazaar](https://pmsbazaar.com/Blogs/Jan-2025-AIF-Performance-Market-Pressures-Weigh-on-Returns-Long-Short-Funds-Outshine); Feb 2025: long-short about -3.09% vs long-only -8.70% — [PMS Bazaar](https://pmsbazaar.com/Blogs/Feb-2025-AIF-Performance-Defensive-Long-Short-Strategies-Soften-Market-Blow); Mar 2025: +2.98% vs BSE 500 TRI +7.32% — [PMS Bazaar](https://pmsbazaar.com/Blogs/AIFs-Delivered-a-Strong-Comeback-in-March-2025-as-Markets-Rebound)
- SIFs: all seven with 6+ months beat benchmark by 3-10 points to 23 Jun 2026, fund returns -5.5% to +8.0% (A, AMFI via Value Research; short record); hedging and portfolio construction credited over stock selection — [Business Today](https://www.businesstoday.in/amp/personal-finance/investment/story/sifs-off-to-a-strong-start-as-every-eligible-fund-beats-its-benchmark-jioblackrock-joins-the-race-540105-2026-07-01)
- Later data: equity long-short SIF average about -1.8% year to date; one Quant fund +29.23% over six months; datasets not directly comparable (A-) — [5paisa](https://www.5paisa.com/index.php/blog/sif-performance-data-shows-wide-gap-across-long-short-strategies)
- SIFs cap unhedged derivative shorts at 25% of net assets (R) — [Finnovate](https://www.finnovate.in/learn/blog/sif-hybrid-long-short-funds-india)
- Pairs on NSE stock futures 2011-17: "up to 34%" annualised after costs, rolling formation and trading, FF3 adjusted (C, journal, best-case wording) — [Aggarwal and Aggarwal 2021](https://ideas.repec.org/a/kap/apfinm/v28y2021i1d10.1007_s10690-020-09317-1.html)
- The same dissertation found spot and spot-futures stat-arb unprofitable, and commodity pairs up to 59% a year after costs (C) — [PAU dissertation](https://krishikosh.egranth.ac.in/items/79c40268-bdee-41b3-8d31-c9998a994d25/full)
- Cointegrated pairs in 25 NSE large caps, walk-forward 2016 to June 2025, 3 pairs, 5 bps a side: Sharpe 0.089, 0.30% a year, -34.31% drawdown (C, student project, tiny breadth) — [QuantInsti](https://blog.quantinsti.com/cointegrated-pairs-trading-indian-equity-market-epat-project/)
- Momentum churn: over half the large-cap index turned over at a recent rebalance; no long-short test published (C) — [StockViz](https://stockviz.substack.com/p/if-you-cant-beat-em?open=false)
- Momentum beat the index in India with higher vol and deep drawdowns (C, long-only) — [Capitalmind](https://www.capitalmind.in/insights/momentum-investing-basics-india)
- BSE stock-level time-series momentum long-short, 1996-2020 (academic, results not seen) — [arXiv 2302.13245](https://www.arxiv.org/pdf/2302.13245)

### Inferences
- Beta estimate: -1.45/-3.43 = 0.42 and 2.98/7.32 = 0.41. Two points only, but consistent, and Feb 2025 fits a defensive pattern. "Long-short" as sold in India is mostly reduced-beta equity.
- Same FY2025 average as long-only with lower beta does suggest some alpha or better crash behaviour. One year is not evidence.
- Short-leg cost is about 0.72% a year of notional at index-lot statutory rates, plus wider stock spreads and any negative carry. Futures STT at 0.05% hits this harder than it hits options, and F&O eligibility limits the short universe to larger names, where momentum shorts may be weaker. The 34% figure predates the higher STT regime.
- A short book could add most as a crash hedge sized off drawdown risk, so judge it on correlation change rather than stand-alone Sharpe.

### Gaps
- No peer-reviewed long-short momentum study on Indian stock futures, with the short leg's contribution separated, was found. This is the most valuable in-house test.
- F&O universe count, current basis, rollover cost and stock-futures spreads were not sourced.
- A snippet attributing "pairs loss-making after 2002" to Gatev et al. (2006) looks misattributed, as a 2006 paper cannot cover that period. It is probably Do and Faff (2012). Not used.

## MCX commodity trend and carry, and currency futures on NSE and BSE

### Takeaway
Global trend evidence is the strongest academic support in this report, but I found no peer-reviewed MCX study, and MCX has far fewer liquid markets than the 58 instruments tested. Gold has the only India-specific evidence, and it is thin. USD-INR is effectively out, as RBI tightened the threshold on 10 Oct 2026 and the purpose restriction is unresolved.

### Cited Findings
- Significant time-series momentum in each of 58 liquid futures, equity, currency, commodity and bond (B, peer-reviewed, JFE 2012) — [Moskowitz, Ooi, Pedersen](https://papers.ssrn.com/abstract=2089463)
- Diversified portfolio Sharpe about 1.0, 1985-2009 (secondary blog) — [Menon Lab](https://themenonlab.blog/blog/time-series-momentum-hedge-fund-strategy-explained); Quantpedia lists 1.31 at 15.7% vol, sample and cost not stated — [Quantpedia](https://quantpedia.com/strategies/time-series-momentum-effect/)
- Long-only hourly Supertrend on Nifty and GoldM, about 2015-25: strategy correlation 0.04; 50:50 max drawdown 6.6% vs 18.5% Nifty and 11.8% GoldM; returns 187% vs buy-hold 214% Nifty and 161% GoldM; no costs, no leverage (C, practitioner) — [TradingQnA](https://tradingqna.com/t/trend-following-in-india-part-3-nifty-gold-strategy/188767)
- CTT 0.01% on sell-side non-agri futures, 0.05% on options premium; Budget 2026 did not raise it (R) — [Fyers](https://fyers.in/blog/what-is-commodity-transaction-tax/); [Business Today](https://www.businesstoday.in/opinion/columns/story/stt-hike-without-ctt-reform-a-missed-chance-for-true-market-parity-515403-2026-02-10)
- MCX energy options turnover up 44% month on month in Sep 2026; options are 75% of revenue, bullion 40% and energy 60% of transaction revenue; gold in a correction, unsized — [Business Today, Jefferies note, 26 Oct 2026](https://www.businesstoday.in/markets/stocks/story/gold-price-selloff-jefferies-calls-mcx-shares-a-hedge-against-crude-volatility-heres-why-563543-2026-10-26)
- Broker MCX guides describe trend-following but give no data (D) — [Upstox](https://upstox.com/learning-center/commodity/mcx-trading-strategies-complete-commodity-market-guide/article-1790/)
- RBI Circular 25 (RBI/2026-27/291), 10 Oct 2026, effective immediately: INR-pair positions without establishing underlying exposure cut from USD 100 million to USD 5 million notional (R, secondary) — [TaxGuru](https://taxguru.in/?p=1088521)
- Under the 2024 framework, users may be required to hold contracted exposure, with sources differing on whether retail can trade without proof below the limit; non-INR pairs carry no purpose restriction — [HDFC Sky](https://hdfcsky.com/blogs/derivatives/rbi-circular-on-currency-derivatives); [Motilal Oswal](https://www.motilaloswal.com/learning-centre/2024/4/impact-of-rbi-circular-on-currency-trading-for-retail-investors-and-market-dynamics); [Business Standard](https://www.business-standard.com/amp/economy/news/rbi-defers-implementing-currency-derivatives-regulations-till-may-3-124040401233_1.html)

### Inferences
- Trend Sharpe scales with breadth. A 58-market Sharpe near 1.0 does not transfer to 5-8 MCX contracts that are largely dollar-denominated commodities. My net estimate is 0.3-0.5, and the gold leg should not be counted twice with the gold allocation.
- CTT at one-fifth of equity futures STT is a real relative gain since April 2026, in favour of commodities over equity futures.
- The hourly Supertrend test breaches the no-intraday rule in spirit and has no costs, so its value is only in the 0.04 correlation.
- USD 5 million exceeds this book's scale many times over, so size is not the constraint. The constraint is whether a purely speculative rupee position is permitted at all. Treat USD-INR as unavailable until confirmed.
- The rupee's weakness is captured more simply through INR gold.

### Gaps
- No MCX-specific trend or carry Sharpe, roll cost, or backwardation data. This is a priority in-house test, and Centurion's own MCX sleeve is likely the best evidence available.
- Current MCX margins, lot sizes and contract liquidity by tenor were not sourced.
- Net recent-decade trend performance (Hurst, Ooi, Pedersen; SG Trend or CTA indices) was not retrieved, so the post-2010 decay question is open.
- A search result noted Huang et al. (2018) arguing trend returns partly reflect static bets, but the page carrying it was not pinned down, so it is listed here rather than cited.
- Whether Kite currently allows USD-INR, and current currency turnover, were not confirmed.

## Indian government bonds, T-bills, gilt funds, IRFs, SGBs and gold as a diversifier

### Takeaway
Gold is the strongest diversifier found, near zero correlation and negative in crises, but after an exceptional run. Gilts mainly earn near cash and face a fresh tightening cycle, so they reduce risk without adding Sharpe, which matches Centurion's result. Interest-rate futures could not be shown to be liquid.

### Cited Findings
- RBI raised repo 25 bps to 5.50% on 7 Oct 2026, unanimous, stance calibrated tightening; SDF 5.25%, MSF 5.75%; next MPC 2-4 Dec 2026 (R) — [5paisa](https://www.5paisa.com/index.php/blog/rbi-mpc-meeting-schedule); [CorpLaw Updates](https://www.corplawupdates.in/updates/rbi-repo-rate-hike-5-50-percent-october-2026)
- 10-year G-Sec about 6.87% on 21 Aug 2026, supply-driven, Brent about $87 with Hormuz disruptions — [Trading Economics](https://tradingeconomics.com/india/government-bond-yield/news/577154); about 7.1% around April 2026, off multi-year highs — [Trading Economics](https://tradingeconomics.com/india/government-bond-yield/news/539296)
- 10-year constant-maturity gilt funds: 1-year category 2.19% at 7 Oct 2026 — [Arthgyaan](https://arthgyaan.com/fund/compare/sbi-constant-maturity-10-year-gilt-fund-direct-growth-vs-uti-gilt-fund-with-10-year-constant-duration-direct-growth/120137-vs-150410); another site, undated, shows 4.01% — [Sharpely](https://sharpely.in/mutual-funds/category/debt-10-year-constant-maturity-gilt)
- 3-year CAGR about 7.1-7.8% and 5-year about 5.9-6.0% at 6 Sep 2026 (A-) — [Arthgyaan](https://arthgyaan.com/funds/categories/gilt-with-10-year-constant-duration-debt-funds)
- IRFs began June 2003, 10-year cash-settled contract December 2013, and NSE daily volume averaged ₹2,562 crore in 2015; nothing recent surfaced — [Business Standard](https://www.business-standard.com/article/pti-stories/irf-daily-turnover-swells-nearly-3-times-to-rs-2-562-cr-on-nse-115082500696_1.html)
- SGB capital-gains exemption limited from 1 Apr 2026 to original subscribers holding to maturity; secondary buyers lose it, 12.5% long-term (R) — [NISM](https://www.nism.ac.in/blog/how-budget-2026-changes-sovereign-gold-bond-sgb-taxation); [Value Research](https://www.valueresearchonline.com/stories/227647/budget-2026-changes-sgb-tax-rules-for-secondary-buyers-now)
- Gold vs Nifty 2005-2023, DCC-GARCH: negative correlation in 2008-09, 2020 and 2022, significant safe-haven behaviour, no coefficients in the excerpt (academic, in-sample) — [Bhunia and Das 2025](https://asianonlinejournals.com/index.php/Economy/article/view/7726)
- Gold-equity correlation about -0.045 (25-year daily), -0.102 (from 2008), -0.099 (GoldBees monthly); gold INR CAGR 38.4% 3Y, 28.4% 5Y, 18.1% 10Y, 13.5% 15Y, 15.1% 20Y; Nifty 50 TRI 10.0%, 10.0%, 12.5%, 10.7%, 11.2%; rupee contributed nearly 3 points a year (C, social-media backtest summaries) — [Multibagg](https://www.multibagg.ai/market-pulse/articles/nifty-gold-allocation-backtests-india-cmoxatc420007qm0jcqu7c76s)
- Gold-Nifty correlation -0.066 with lower vol in an 80/20 mix, and lower return (C) — [DSIJ](https://insights.dsij.in/dsijarticledetail/can-gold-give-your-portfolio-an-extra-shine-44928)

### Inferences
- Term spread 10Y minus repo is about 1.4 points, or about 1.6 against SDF, which is modest carry. With RBI newly tightening and supply heavy, duration timing is unfavourable now.
- Over 5 years, 10Y constant-maturity gilt funds at about 5.9-6.0% lagged top arbitrage funds at about 6.8-6.9%, and over 1 year 2.19% lagged about 6.7% (dates differ: 6 Sep and 7 Oct against 30 Oct 2026). Measured against a cash-like hurdle, recent gilt Sharpe is negative. Over 3 years gilts led at 7.1-7.8% vs 7.0-7.4%, so this is window-dependent, but it fits a sleeve that cuts vol without adding Sharpe.
- Gilts as a crash diversifier still make sense, and the Sharpe contribution comes from rate cuts in crises. That is episodic rather than steady. Expect 0.1-0.35.
- Gold's 3-year 38.4% will not persist, and the recent correction is a reminder. Using the 13.5-15% 15-20 year range and vol of perhaps 15% gives Sharpe about 0.5-0.6 in-sample. Assume less going forward, with INR weakness a structural positive.
- SGBs are no longer a secondary-market free lunch from FY27, so gold ETF or MCX is cleaner. No new-issue status was confirmed.

### Gaps
- End-October 2026 10-year yield and T-bill levels were not found (the tracker PDF was unreadable).
- Bond-equity correlation in India, including March 2020 behaviour, was not found.
- Gilt ETF liquidity, spreads and AUM; IRF liquidity 2025-26 were not found. Do not rely on IRFs until checked on NSE statistics.
- Gold vol in INR and March 2020 gold behaviour (the sources state correlation only).

## International diversification for a resident Indian: LRS, GIFT City, and Nasdaq and S&P ETFs in India

### Takeaway
Domestic global ETFs are structurally broken as a clean exposure by premiums tied to the capped USD 7 billion limit. GIFT City and LRS work but are outside Kite automation as far as I can tell, and carry TCS friction. I found no India-specific correlation-benefit data.

### Cited Findings
- Industry-wide USD 7 billion overseas cap with USD 1 billion ETF sublimit prevents unit creation, so market makers cannot close premiums (R, secondary, read from a search summary; one source's USD 1 billion overall figure looks wrong; confirm against SEBI) — [Finowings](https://www.finowings.com/Trending/why-global-etfs-in-india-are-trading-at-high-premiums)
- MON100 traded at a 22% premium on 31 Dec 2024 (A-) — [Value Research](https://www.valueresearchonline.com/stories/224548/nasdaq-etf-time-to-invest/); nearly 20% at ₹289.77 vs NAV ₹241.48, undated — [Angel One](https://www.angelone.in/news/mutual-funds/indian-investors-face-20-premium-on-us-etfs-amid-surging-demand-and-limited-liquidity-here-s-why); about 9% in a later report, source page not pinned
- Up to 111.8% premium on Motilal Oswal Nasdaq Q50, 59.5% on Mirae FANG+, 58.9% on S&P 500 Top 50 (A-, dated snapshots) — [DSIJ](https://insights.dsij.in/dsijarticledetail/6-global-etfs-in-india-trade-at-premiums-of-up-to-1118-to-nav-60143)
- NSE IX Global Access launched Feb 2026 for US stocks and ETFs under LRS, USD 250,000 per financial year; no retail derivatives — [DeshGujarat](https://deshgujarat.com/2026/02/26/indian-retail-investors-can-now-trade-us-stocks-via-nse-ix-at-gift-city/)
- GIFT funds of funds at USD 5,000 minimum for Parag Parikh; UDRs cover about 50 stocks; tax same across routes, 24-month 12.5% long-term, slab short-term; 20% TCS above ₹10 lakh a year (all single-source, verify) — [Arthgyaan](https://arthgyaan.com/blog/parag-parikh-gift-city-sp500-nasdaq100-fof.html); [Finnovate](https://www.finnovate.in/learn/blog/gift-city-vs-lrs-global-investing-india)

### Inferences
- Buying at a 10-20% premium is a large hidden cost if the premium closes, and it adds a stream of domestic liquidity risk unrelated to the underlying. Avoid premium ETFs, or treat the premium as its own trade.
- At a few crore, USD 250,000 per person per year is usable but slow to build, and TCS is an upfront cash drag.
- Global equities share the global crash factor with Indian equities, so I expect the benefit is mostly normal-time. Not a tail diversifier, and it is a further equity bet rather than a new source.
- Note the US module is on hold in Centurion, so this stream cannot be worked on until restarted.

### Gaps
- No correlation evidence (Nifty or momentum vs S&P or Nasdaq in INR) was retrieved.
- Current premium levels and whether the USD 7 billion cap has moved were not confirmed. Value Research's 2026 pieces returned 403.
- Whether GIFT routes are available through Zerodha or any API, and whether Kite could automate them, was not confirmed.

## Special situations: open offers, delisting, buybacks, merger arbitrage, demergers, rights, IPO flipping, REIT and InvIT yield, dividend capture

### Takeaway
REITs and InvITs are the one sizeable stream here, with live index data of about 12.7% 5-year CAGR, about 9% vol and 0.15-0.34 equity correlation. Event trades are mostly small, unmeasured, and capacity-limited by retail quotas. I found no systematic return study on open-offer, delisting or merger arbitrage in India.

### Cited Findings
**REITs and InvITs**
- Nifty REITs & InvITs 90:10 index, launched July 2026: 18.29% 1-year TR and 12.72% 5-year at 30 Jun 2026, 1-year Nifty correlation 0.15, vol about 10.5% (A, read via search, PDF timed out) — [NSE factsheet](https://www.niftyindices.com/Factsheet/Factsheet_Nifty_REITs_and_InvITs_9010.pdf)
- Sept 2026: 7.41% 1-year, 12.55% 5-year, vol 8.75%, beta 0.23 (A-) — [indexPe](https://indexpe.in/nifty-reits-invits-90-10)
- Dividend yield 5.11%, 13.07% CAGR since 1 Apr 2021 inception, confirming 18.29% and 12.72% (A-, article dated 27 Jul 2026) — [JM Financial](https://www.jmfinancialservices.in/market-news-and-insights/1713371); the older index factsheet showed 1.84% yield and 5-year correlation 0.34 at Mar 2025, which conflicts on yield — [NSE factsheet](https://www.niftyindices.com/Factsheet/Factsheet_REITs_InvITs.pdf)
- 5-year SD 8.89% vs Nifty 50 13.82%, as of 31 Jul 2026 (A-) — [Asianet Newsable](https://newsable.asianetnews.com/business/india-s-reits-invits-aum-to-grow-2-3-times-by-2030-ionic-wealth-articleshow-ktgldua)

**Buybacks**
- Oct 2024 regime moved tax to shareholders at slab on proceeds, ending the old tax-free arbitrage — [Business Standard](https://www.business-standard.com/amp/markets/news/tax-tweak-may-tilt-infosys-buyback-in-favour-of-retail-mutual-funds-125102301131_1.html); [DSIJ](https://insights.dsij.in/dsijarticledetail/buyback-to-tender-or-not-to-tender-52962)
- From 1 Apr 2026 buybacks are taxed as capital gains on the gain only, 12.5% long-term, 20% short-term, no TDS; promoters pay effective 22-30% (R; one April 2026 source treats as enacted) — [Trendlyne](https://trendlyne.com/posts/5514385/buybacks-after-the-tax-reset-what-has-changed)
- 2025: 15 companies, ₹19,749 crore, mostly Infosys at ₹18,000 crore — [Trendlyne](https://trendlyne.com/posts/5514385/buybacks-after-the-tax-reset-what-has-changed)
- TCS 2022 retail acceptance 14-21%, against full acceptance in 2017 and 2018; 15% retail reservation, holdings up to ₹2 lakh — [5paisa](https://test.5paisa.com/blog/tcs-buyback-offer-opportunity-for-retail-investors)
- Highly oversubscribed buybacks accept about 30%, undersubscribed 100% (tracker) — [IPO Pulse](https://ipopulse.talkytools.com/buybacks)
- Headline "38% risk-free" and "16.5%" returns ignore execution risk (D) — [Money9](https://www.money9.com/news/markets/earn-up-to-38-risk-free-returns-by-participating-in-this-buyback-offer-69835.html/amp); a 2016 case earned 8.7% in two months (A-, single case) — [Capitalmind](https://premium.capitalmind.in/2016/07/infratel-buyback-offer-yields-retail-shareholders-neat-8-7-two-months/)

**Open offers and delisting**
- Arbitrageurs bet on acceptance; United Spirits expected about 70%, got 55%; failed offers see fast price falls (press case) — [Business Standard](https://www.business-standard.com/article/pf/why-open-offers-aren-t-attractive-for-many-investors-114101300002_1.html)
- Significant negative post-announcement returns, 2007-10, crisis window (C, working paper) — [Nangia, Agrawal, Reddy](https://ideas.repec.org/p/pra/mprapa/60432.html)
- Fixed-price delisting notified Sept 2024: frequently traded shares, at least 15% above floor, alongside reverse book building (R) — [Morgan Lewis](https://www.morganlewis.com/pubs/2024/07/this-isnt-hotel-california-or-indias-new-fixed-offer-price-mechanism-for-taking-public-companies-private); [Business Standard](https://www.business-standard.com/amp/markets/news/sebi-amends-rules-introduces-fixed-price-process-for-voluntary-delisting-124092600853_1.html)

**IPO flipping**
- SEBI, 144 IPOs, April 2021 to Dec 2023: 54% of non-anchor shares sold within a week; individuals sold 67.6% by value when returns exceeded 20%, against 23.3% when negative; mutual funds about 3.3% (A, regulator) — [Mondo Visione](https://mondovisione.com/news/sebi-study-shows-54-of-ipo-shares-allotted-to-investors-excluding-anchor-inves-202493); [Angel One](https://www.angelone.in/news/ipos/sebi-study-reveals-investor-behavior-in-ipo)
- An "8.41% average" flipping return from a partner piece has no primary source, so it was not used (D) — [Tribune partner content](https://www.tribuneindia.com/partner-exclusives/investor-behaviour-analysis-2015-2025-ipo-flippers-portfolio-managers-and-long-term-unlisted-investors)

### Inferences
- REIT/InvIT trailing Sharpe works out to about 0.75 at rf 6%, against perhaps 0.3 for Nifty on the same 5 years. That window saw rate cuts, so forward I assume 0.4-0.6. It shares rate risk with gilts, so the two are not fully separate streams, and with a tightening RBI the near term is weaker.
- Buyback economics improved in April 2026 versus the 2024-26 regime, because tax now applies only to gain. Quota-based acceptance still falls as arbitrage crowds in, and the ₹2 lakh retail cap makes this a few lakh per event per account. Excluding Infosys, 2025 had only about ₹1,750 crore of buybacks across 14 companies. It cannot move a ₹1-3 crore book.
- Event streams are hard to automate through Kite, as tenders run through a separate corporate-action workflow, and they are semi-manual by nature.
- IPO flips are pro-cyclical. Allotment is a lottery with thin size, and they vanish after sell-offs, so they cannot be counted as uncorrelated.

### Gaps
- No systematic Indian study on merger arbitrage, demerger spin-offs, rights arbitrage or dividend capture was found. Any claims would be speculation, so none are made.
- 2024-26 buyback spreads and acceptance after the April 2026 tax change are not available.
- Mainboard average listing gain by year and individual Embassy or IndiGrid yields were not retrieved.

## Calendar and flow effects: month-end, expiry, FII positioning, Budget, election and RBI event drift

### Takeaway
The turn-of-month effect is the only one with academic support, and it is a timing tilt on equity rather than a separate stream. FII positioning evidence is anecdotal, and Centurion has already found pre-announcement drift fails. None of these adds an independent return source.

### Cited Findings
- Turn-of-month significant on Nifty 50, Sensex and small-cap indices, April 2007 to Feb 2023, stronger in small caps; DIIs active at month-end; effect size and costs not in abstract (C, in-sample) — [Satish and Bheemanagouda 2023](https://rfb.ase.ro/Vol15-2023/dec2a.asp)
- High mean returns around month start, Nifty 1994-2005 (C) — [Mangala and Sharma 2007](https://ideas.repec.org/a/sae/padigm/v11y2007i2p9-15.html)
- Bank Nifty futures April 2013 to June 2019: 74 expiries, high return, low vol after weekly options began (C) — [IJEFI 2020](https://doaj.org/article/670623ed8fd148dca38ae9f134115cb3)
- FII long-short ratio at or below 0.15 precedes short-covering rallies, on just four post-March-2020 instances (D/C, broker note) — [Business Standard, FII shorts](https://www.business-standard.com/amp/markets/news/fiis-hold-10-short-bets-for-every-long-trade-in-index-futures-125080700077_1.html); [Geojit](https://insights.geojit.com/?p=22550)
- FII/DII flows explaining 23% of daily moves, with a 0.31 correlation to next-day returns (D, personal backtest) — [DEV Community](https://dev.to/shaktitiwari/fii-vs-dii-who-actually-moves-nifty-data-driven-analysis-2024-2026-13a0)

### Inferences
- Month-end effects could trim entry timing for momentum rebalances, but cannot diversify anything, as they are long-equity by construction.
- Expiry effects from before the 2025 Tuesday move and the December 2026 VWAP settlement should be treated as stale, because the structural change breaks pre-2025 samples.
- FII ratio extremes may have some value as a regime flag, but four events cannot be tested honestly. A check over the full history is cheap if participant OI is already in the store.

### Gaps
- No rigorous Indian study on Budget, election or RBI event drift was found.
- No peer-reviewed participant-OI predictive study was found.

## Realistic correlation among these streams in a crash

### Takeaway
Correlations rise but not all to 1. Evidence says tail correlations exceed normal correlations across return-seeking assets, and all short-vol, carry and equity-linked streams should be expected to converge. In India the sources that stay separate are gold, and probably gilts on central-bank easing. Plan crash ρ near 0.7+ for short vol, about 0.5 for long-short, and near zero or negative for gold.

### Cited Findings
- Tail correlation is higher in sell-offs than rallies across styles, sizes, regions and alternatives, monthly data Jan 1970 to June 2017; full-sample correlations mislead; "all go to 1" is an oversimplification; suggested tools include tail hedging, defensive momentum and dynamic risk methods — [T. Rowe Price](https://www.troweprice.com/content/dam/ide/articles/pdfs/2018/q4/when-diversification-fails.pdf); [FAJ abstract](https://ideas.repec.org/a/taf/ufajxx/v74y2018i3p19-32.html)
- Factor strategies diversify better, partly since they can short — [Alpha Architect](https://alphaarchitect.com/when-diversification-fails/)
- Gold stayed negatively correlated with Nifty in 2008-09, 2020 and 2022 — [Bhunia and Das 2025](https://asianonlinejournals.com/index.php/Economy/article/view/7726)
- Short-vol VRP profit in calm regimes, concentrated losses in shocks — [Arsh Hora profile](https://in.linkedin.com/in/arshhora); tail risk primary in Pillai — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6876580)
- Dispersion loses in stress as correlation rises — [Quantpedia](https://quantpedia.com/strategies/dispersion-trading/)
- Feb 2025: long-short AIFs about -3.09% vs long-only -8.70% — [PMS Bazaar](https://pmsbazaar.com/Blogs/Feb-2025-AIF-Performance-Defensive-Long-Short-Strategies-Soften-Market-Blow)

### Inferences
Crash correlation to long momentum, my planning values:

| Stream | Normal ρ | Crash ρ | Basis |
|---|---|---|---|
| Defined-risk VRP | 0.3-0.5 | 0.7+ | Short vol plus delta; loss capped by wings |
| Long-short equity | 0.2-0.5 | 0.4-0.6 | About 0.4 beta in live funds |
| MCX commodities ex gold | 0-0.1 | 0.2-0.5 | Demand shock moves crude and copper with equities |
| REIT/InvIT | 0.15-0.34 | 0.4-0.6 | Liquidity sell-offs hit yield assets |
| Gilts | About 0 | -0.1 to 0.3 | Depends on whether it is a growth or inflation shock |
| Gold | -0.05 to -0.10 | -0.3 to 0.1 | Safe haven, though liquidation can hit early |
| Events | Low | 0.3-0.6 | Offers fail, spreads widen |

- A stress-weighted book therefore has fewer effective streams than its normal-time matrix suggests: about 2-3 (equity-like, gold, rates). The bounded-tail cap handles the overlay, but does not change the fact that losses cluster.
- Use the crash matrix, not full-sample ρ, when sizing. With ρ around 0.2 the book Sharpe estimate above falls to about 0.76, which is why 1.4 is not credible on diversification alone.

### Gaps
- No Indian multi-asset crash correlation matrix was found, covering Jan 2008, Mar 2020, Jun 2024 or the Feb 2025 sell-off, so the table is estimated. Centurion's own stores can compute it directly, and this is the single most decision-useful next step.
- Numeric left-tail correlations from Page and Panariello were not extractable: the PDF could not be parsed and Alpha Architect returned 403.
- NSE factsheets for Nifty 50 and Nifty200 Momentum 30 timed out, as did the 75:25 Momentum plus G-Sec index factsheet. That index would be a direct live test of what a bond sleeve does to momentum Sharpe, so it is worth reading by hand.
