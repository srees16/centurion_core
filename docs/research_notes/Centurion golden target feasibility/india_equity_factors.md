# India equity factors and anomalies: evidence-graded map

**Grades.** [LIVE] live index or audited record. [ACAD-OOS] academic with out-of-sample test. [ACAD-IS] academic, in-sample only. [BT] backtest only, including back-filled index history and vendor or retail backtests. [MKT] marketing or affiliated source. Flags: *(abstract)* means only an abstract or search snippet was read; *(back-filled)* means the figure includes pre-launch history; *(survivorship)* means survivorship bias is present or unverified.

**Coverage limits of this pass (read first).** The web-search budget ran out partway through, so several questions were never searched: buybacks, FII/DII flows, F&O ban, ASM/GSM, circuit limits, SME premia, sector rotation, short-side rules, SLB, impact costs. No search-limit workaround was used. They are listed as gaps, not as findings of "no evidence". After that point only direct page fetches were used: pages surfaced by earlier searches, plus one first-party page, Zerodha's charges page, for statutory rates. niftyindices.com timed out on every attempt, SSRN returned 403, and several NSE and AMC PDFs came back as unreadable binary, so no NSE whitepaper or factsheet was read directly. NSE figures below come from third-party tabulations of NSE data or from search snippets.

## What have NSE factor indices delivered, and how far apart are back-filled and live figures?

### Takeaway
On 2005-2026 TRI data, Nifty200 Momentum 30 (18.2% CAGR, Sharpe 0.62, MaxDD -67.7%) has a lower Sharpe than low-vol (0.66-0.71) and quality (0.62-0.64), and momentum + quality blends have the best risk-adjusted record (Sharpe 0.75-0.76). Most of that window is back-filled. The only live-period read available suggests momentum's excess return over its parent fell sharply after the August 2020 launch.

### Cited Findings

**Comparable table: 1 Apr 2005 to 16 Oct 2026, TRI, weekly returns, 6% risk-free, daily data from niftyindices.com** — [RohitVarre/nifty-factor-analytics](https://github.com/RohitVarre/nifty-factor-analytics). Grade: [BT] + [LIVE] mix *(back-filled)*: third-party computation on NSE series, gross of costs, pre-launch years are NSE back-fill. Calmar column is derived here (CAGR / |MaxDD|).

| Index | CAGR | Vol | Sharpe | MaxDD | Calmar | 5Y rolling median / worst |
|---|---|---|---|---|---|---|
| Nifty200 Momentum 30 | 18.20% | 21.16% | 0.62 | -67.69% | 0.27 | 19.58% / -1.98% |
| Nifty500 Momentum 50 | 21.25% | 22.68% | 0.71 | -70.24% | 0.30 | 21.59% / -4.38% |
| Nifty Midcap150 Momentum 50 | 22.14% | 22.04% | 0.75 | -72.52% | 0.31 | 23.97% / 1.16% |
| Nifty Alpha 50 | 18.77% | 25.10% | 0.58 | -78.88% | 0.24 | 19.84% / -6.92% |
| Nifty200 Alpha 30 | 17.42% | 23.30% | 0.56 | -75.34% | 0.23 | 18.38% / -8.09% |
| Nifty100 Low Volatility 30 | 16.47% | 16.12% | 0.66 | -48.65% | 0.34 | 15.84% / 2.19% |
| Nifty Low Volatility 50 | 16.40% | 15.97% | 0.66 | -51.25% | 0.32 | 15.97% / 2.83% |
| Nifty500 Low Volatility 50 | 16.96% | 15.58% | 0.71 | -48.26% | 0.35 | 15.74% / 2.81% |
| Nifty200 Quality 30 | 16.77% | 17.43% | 0.64 | -55.57% | 0.30 | 16.69% / 1.33% |
| Nifty500 Quality 50 | 16.49% | 17.91% | 0.62 | -53.60% | 0.31 | 16.56% / -0.64% |
| Nifty Midcap150 Quality 50 | 17.34% | 18.54% | 0.64 | -58.51% | 0.30 | 17.81% / 2.63% |
| Nifty Smallcap250 Quality 50 | 18.00% | 21.91% | 0.60 | -70.26% | 0.26 | 18.89% / -1.51% |
| Nifty500 Value 50 | 16.24% | 27.01% | 0.47 | -66.06% | 0.25 | 14.87% / -8.13% |
| Nifty200 Value 30 | 16.03% | 26.93% | 0.47 | -63.09% | 0.25 | 13.22% / -8.42% |
| Nifty Quality Low-Volatility 30 | 15.51% | 15.35% | 0.63 | -47.44% | 0.33 | 15.98% / 0.40% |
| Nifty Alpha Low-Volatility 30 | 17.62% | 16.82% | 0.70 | -57.31% | 0.31 | 17.96% / 4.19% |
| Nifty Alpha Quality Low-Vol 30 | 16.98% | 16.10% | 0.69 | -49.00% | 0.35 | 17.68% / 3.10% |
| Nifty Alpha Quality Value Low-Vol 30 | 16.84% | 16.29% | 0.68 | -53.73% | 0.31 | 17.24% / 3.13% |
| Nifty500 Multifactor MQVLv 50 | 19.03% | 17.45% | 0.75 | -53.93% | 0.35 | 18.94% / 1.71% |
| Nifty Total Market Momentum Quality 50 | 20.38% | 19.05% | 0.76 | -59.04% | 0.35 | 22.26% / 5.05% |
| Nifty500 Multicap Momentum Quality 50 | 20.08% | 18.99% | 0.75 | -60.06% | 0.33 | 22.13% / 4.35% |
| Nifty MidSmallcap400 Momentum Quality 100 | 21.36% | 21.47% | 0.74 | -68.04% | 0.31 | 21.45% / 1.54% |
| Nifty Smallcap250 Momentum Quality 100 | 21.31% | 24.32% | 0.68 | -72.43% | 0.29 | 21.92% / 0.73% |
| Nifty Midcap 150 | 16.69% | 22.33% | 0.54 | -72.89% | 0.23 | 16.87% / -2.34% |
| Nifty Smallcap 250 | 15.56% | 24.63% | 0.47 | -75.56% | 0.21 | 14.07% / -6.17% |
| Nifty Microcap 250 | 17.50% | 27.38% | 0.51 | -82.04% | 0.21 | 15.51% / -11.93% |
| Nifty Next 50 | 14.97% | 23.09% | 0.47 | -72.13% | 0.21 | 14.42% / 0.26% |
| Nifty 500 | 13.71% | 19.81% | 0.45 | -63.71% | 0.22 | 13.38% / -1.40% |
| Nifty 50 | 13.26% | 19.53% | 0.44 | -59.50% | 0.22 | 12.97% / -1.03% |
| Nifty High Beta 50 | 6.88% | 33.32% | 0.19 | -78.72% | 0.09 | 4.29% / -15.90% |

**Back-filled versus live, Nifty200 Momentum 30**
- [BT] Launched 25 Aug 2020 with history computed from 2005; holds 30 Nifty 200 stocks, weight is the lower of 5% or free-float cap × normalised momentum score; rebalanced semi-annually in June and December — [Curious Investing Insights](https://curiousinvestinginsights.substack.com/p/momentum-or-mayhem-unpacking-the) (personal commentary citing NSE, MOAMC, HDFC AMC).
- [BT] *(back-filled, abstract)* NSE launch whitepaper reported since-inception CAGR 18.6% versus 12.7% for Nifty 200, and outperformance 98.3% of the time on a rolling basis (rolling window not seen) — [NSE whitepaper, Sep 2020](https://www.niftyindices.com/docs/default-source/indices/nifty200-momentum-30-index/nifty200_momentum_30_index_whitepaper_sep_20.pdf?sfvrsn=3eb99734_4), search snippet only.
- [BT]+[LIVE] *(abstract)* NSE's April 2026 whitepaper gives 19.19% annualised from April 2005 to 27 Feb 2026, versus 15.08% for a parent index; the snippet cuts off before naming it — [NSE whitepaper, Apr 2026](https://www.niftyindices.com/docs/default-source/indices/nifty200-momentum-30/momentum-strategy-whitepaper_2026.pdf), search snippet only.
- [LIVE] Five-year return to 15 Mar 2025, almost entirely post-launch: Momentum 30 154%, Nifty 50 125%, Nifty 200 139%, Nifty 500 147%; TRI or price basis not stated; last six months of that window -27% versus Nifty 50 -12% and Nifty 200 -15% — [Curious Investing Insights](https://curiousinvestinginsights.substack.com/p/momentum-or-mayhem-unpacking-the).
- [LIVE] Price-return read to July 2026: 1-year -2.77%, 6-month -2.88%, July -1.59% versus Nifty 50 +1.09% — [Wright Research](https://www.wrightresearch.in/blog/how-did-the-nifty200-momentum-30-perform-in-july-2026/) (vendor of momentum products).
- [LIVE] 2024-25 drawdown: 27 Sep 2024 to 7 Apr 2025, -31.79% over 192 days, unrecovered at Aug 2025 — [Wright Research](https://www.wrightresearch.in/blog/momentum-strategies-underperforming-2025-data-insights).
- [MKT] Beat Nifty 200 TRI in 13 of the last 17 years — [Motilal Oswal deck, as quoted in search result](https://www.motilaloswalmf.com/CMS/assets/uploads/Documents/aa841-presentation-motilal-oswal-nifty-200-momentum-30-etf-index-fund-rd-june-2025.pdf). Beat Nifty 200 in 15 of 20 calendar years, and in 100% of 7- and 10-year rolling windows, 16 Sep 2005 to 30 Sep 2024 — [Curious Investing Insights](https://curiousinvestinginsights.substack.com/p/momentum-or-mayhem-unpacking-the), mostly *(back-filled)*.

**Cost-aware reconstruction**
- [BT] Top-200 by cap, PE>0, top 30 by 12-month momentum, equal weight, annual December rebalance, Dec 2006 to Dec 2025 (methodology section says Jun 2025), delisted names included, 0.11% cost + 0.05% slippage, ₹50 lakh, net of STCG/LTCG: Base Momentum 30 14.60% CAGR (14.01% elsewhere on the same page), vol 23.02%, MaxDD -70.61%, recovery 67 months, 92% positive 3-year periods, worst 3-year -13.9% — [BacktestIndia](https://backtestindia.com/blog/momentum-factor-india-liquidity-premium-scaled-turnover). Not peer reviewed, author runs the platform, does not reconcile to the actual index.

### Inferences
- **Sharpe-adjusted, the core is not the best single sleeve.** Nifty200 Momentum 30 sits below Nifty100 Low Vol 30 on Sharpe (0.62 v 0.66) and Calmar (0.27 v 0.34), for a 1.7pp CAGR edge. All these figures are gross and mostly back-filled.
- **Implied live excess looks far below the back-filled excess.** Over the five years to March 2025, 154% v 139% is about 20.5% v 19.1% a year, or roughly 1.4pp, against 5.9pp in the launch whitepaper (18.6 v 12.7). The window starts near the Covid low and ends in a momentum trough, so 1.4pp is a pessimistic read, but one that needs replacing with the NSE TRI series.
- **Back-solving since-inception figures gives a wide range.** Using 18.6% to about Aug 2020, the implied live CAGR is roughly 20.8% if 19.19% to Feb 2026 is right and 17.2% if 18.20% to Oct 2026 is right (hand calculation). The two sources imply about -6.7% between Feb and Oct 2026, which I could not confirm. Treat this as a bound, not a measurement.
- **Momentum strengthens down-cap on index data** (Sharpe 0.62 large/mid, 0.71 Nifty 500, 0.75 midcap), which is where impact is highest. Index figures carry no costs, so this gradient overstates what is capturable.
- Value's volatility (about 27%) is well above the market's (19.8%) and its Sharpe (0.47) barely beats Nifty 500's (0.45), so value earns its return as risk rather than as an edge in this window.
- The table's order is by Sortino, and Nifty High Dividend Yield 15's 19.51% CAGR with 24.78% vol looks unusual; check that series before using it.

### Gaps
- No NSE whitepaper or factsheet was read directly, so index-level turnover, official volatility and official return/risk figures are missing for every index.
- Launch dates for all indices except Nifty200 Momentum 30 were not found, so no live-only return can be split out for them.
- Nifty50 Value 20 is not in the table, and I found no figures for it. One author reports its data starts only on 1 Jan 2009 and runs a separate window from then for a fair comparison; the figures sit in chart images and could not be read — [Personal Finance Plan](https://personalfinanceplan.in/comparison-nifty-next-50-factor-indices-alpha-momentum/). If that is right, its index history cannot cover 2008, which matters for the 2008-in-every-store rule.
- The snippet from NSE's 2026 whitepaper showing 21.33% (2024) and -4.57% (2025) beside 6.58% and -14.13% was flagged as hard to read, and the row looks inconsistent with a Nifty 200 comparator. Do not use it until checked against the PDF.
- Survivorship does not apply to the index series itself, but the back-fill does carry look-ahead risk from rule selection. How much was not measurable here.

## What does academic evidence show on momentum, quality, low volatility, value, size, liquidity and turnover, and is there publication decay?

### Takeaway
Momentum has the strongest academic support in India (about 21.9% a year on a long-short paper basis to the mid-2010s), and long-only momentum, low-vol and quality alpha is reported to survive implementation costs. No India-specific publication-decay study was found, and the low-turnover effect is the most interesting unverified lead.

### Cited Findings
- [ACAD-IS] *(abstract)* IIMA four-factor model: average annual momentum return 21.9%, market excess return 11.5%; the HML and SMB figures are garbled on the page (15.3% and "nearly 0%" are both shown, assignment unclear). The abstract gives data as Oct 1993 to Dec 2013 and returns as Jan 1994 to Dec 2014 — [IIMA publication page](https://www.iima.ac.in/publication/four-factor-model-indian-equities-market).
- [ACAD-IS] Method: CMIE Prowess data, wider firm coverage than earlier Indian work, illiquid firms excluded so portfolios are investable, firm-size-distribution cut-off, correction for vanished firms. The revised paper reflects firms added retrospectively to Prowess — [IIMA publication page](https://www.iima.ac.in/publication/four-factor-model-indian-equities-market); library start given as January 1993 — [Varma blog](https://www.jrvarma.in/blog/Y2013/fama.french.html). Monthly updates were promised — [search summary of IIMA PDF](https://iima.ac.in/sites/default/files/rnpfiles/1970004922013-09-05.pdf).
- [ACAD-IS] *(abstract)* Raju & Teli (2022), top 200 stocks, monthly rebalanced, equal-weight, long-only: momentum, low-vol and quality deliver alpha that survives implementation costs; no factor is a consistent winner; low-vol and quality have fairly low turnover. Sample not seen — [SSRN 4000418](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4000418).
- [ACAD-IS] *(abstract)* Raju & Teli (2022), Conservative Formula in India: 100 stocks on low-vol, payout yield and momentum beat S&P BSE 100 by 12.6% a year compound. It tests a formula published on other markets, so it is a geographic out-of-sample test, but cost treatment and sample were not seen — [SSRN 4163613](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4163613).
- [ACAD-IS] *(abstract)* Raju (2023): lower-risk portfolios show higher risk-adjusted returns than high-risk portfolios — [SSRN 4398656](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4398656).
- [ACAD-OOS] *(abstract)* Raju (2025): predictive regressions and out-of-sample tests show limited factor-timing ability; only momentum at the 12-month horizon is significant — [SSRN 5513940](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5513940).
- [ACAD-IS] *(abstract)* Raju (2025), 2005-2025: factor returns stay consistent after universe expansion — [SSRN 5757103](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5757103).
- [ACAD-IS] Aggarwal & Gupta (2017): long side only, 16 formation/holding combinations of 3, 6, 9 and 12 months; "the pure momentum strategy can be profitably exploited" after costs and exchange restrictions, but "Indian market does not seem to recognise or reward medium or long-term momentum". Sample and numbers not visible — [JCAR](https://www.publishingindia.com/archive/jcar/can-small-investors-gain-from-momentum-trading-in-india-an-empirical-investigation). This conflicts with the IIMA 12-month-style momentum premium and with the 12-month significance in Raju (2025).
- [ACAD-IS] *(abstract)* Maheshwari & Dhankar (2018): momentum and long-term overreaction effects are present, consistent with Jegadeesh-Titman and De Bondt-Thaler — [IIMK Society & Management Review](https://econpapers.repec.org/RePEc:sae:iimkoz:v:7:y:2018:i:2:p:154-170).
- [BT] Low-turnover premium, Nifty 500 universe, top 50 per factor split at median Scaled Turnover (daily turnover / market cap), equal weight, annual, Dec 2006 to Dec 2025, delisted included, net of costs and tax. Low v high: Quality 15.45% v 8.16% (Sharpe 0.49 v 0.22, MaxDD -65.04% v -72.02%); Momentum 15.04% v 11.89% (0.63 v 0.43, -75.73% v -77.70%); Value 13.74% v 12.23% (0.51 v 0.36, -54.03% v -64.35%); Low Vol 14.11% v 14.03% (0.87 v 0.85, -42.80% v -45.79%); benchmark 10.41%, 0.51, -55.12% — [BacktestIndia](https://backtestindia.com/blog/factor-scaled-turnover-nifty500-universal-liquidity-premium). The page labels the 10.41% benchmark Nifty 500, and its sister article labels the same figures Nifty 50 — [BacktestIndia momentum](https://backtestindia.com/blog/momentum-factor-india-liquidity-premium-scaled-turnover). Flat slippage understates costs in the low-turnover bucket, as the page concedes.
- [BT] Top-30 momentum split by Scaled Turnover, Dec 2006 to Jun/Dec 2025: low 19.43% net (21.07% gross), high 8.51% net (9.02% gross); rolling 10-year low-minus-high spread positive in over 90% of windows — [BacktestIndia momentum](https://backtestindia.com/blog/momentum-factor-india-liquidity-premium-scaled-turnover); summarised as "a liquidity illusion" — [Freefincal](https://freefincal.com/is-indian-momentum-investing-just-a-liquidity-illusion/).
- [ACAD-OOS] *(abstract)* Non-India context: Baltussen, Swinkels and van Vliet find limited out-of-sample decay across 24 global factor premiums — [SSRN 3325720](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3325720).

### Inferences
- The academic and index records agree on the sign for momentum, low-vol and quality. They do not establish net magnitude, because most figures are gross, abstract-only or back-filled.
- The Scaled Turnover spread could be a real attention or liquidity premium, or a cost artefact from flat slippage. A low-turnover filter on the momentum sleeve is worth testing with a size-dependent impact model before believing it.
- Data-library caution: retrospective additions to Prowess mean the IIMA series is not strictly point-in-time.

### Gaps
- Updated IIMA library statistics to 2026 were not obtained: `web.iima.ac.in` did not resolve and the working-paper PDF was unreadable. HML and SMB premia remain unverified.
- No Sehgal or other direct Jegadeesh-Titman replication with numbers was read.
- No India-specific publication-decay or capacity study was found, but only one search was run.
- McLean-Pontiff's decay figures (about 26% lower out-of-sample, 58% lower post-publication) came back only in a search summary with no page attached; the candidate page returned 429, so they are unconfirmed here and should be cited from the original paper.
- "Shades of Momentum" and "Comparing Equity Factor Libraries" exist in Raju's SSRN listing; neither abstract was retrieved ([author page](https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=3354364), [7518980](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7518980)).
- An NSE research paper, [RP_2_Feb2012](https://nsearchives.nseindia.com/research/content/RP_2_Feb2012.pdf), surfaced under momentum and costs but timed out; title and content unknown.

## Post-earnings drift, surprises, guidance and analyst revisions: what survives costs at a swing horizon?

### Takeaway
The small body of evidence found mostly supports Centurion's null: no reliable long-side drift, with some sign of a negative drift after bad news. The only contrary result found is a low-rigour student project.

### Cited Findings
- Centurion's own test: no earnings premium on NSE, -0.26%, t -3.12, 16,635 events (from the brief; not independently verified here).
- [ACAD-IS] Gupta & Dhusia (2021), 100 NSE firms, 2014-2018, 1,130 observations: negative association of abnormal returns with earnings surprise; "The Indian stock market tends to be efficient with respect to earnings announcements"; responses may vary with liquidity; no consistent superior return from value-glamour — [Trends Economics and Management](https://journals.vutbr.cz/index.php/trends/article/view/541). Small sample, abstract gives no drift size.
- [ACAD-IS] *(abstract)* Nottingham dissertation, 665 events, S&P BSE 500 firms, 2007-2012: relatively efficient on good news, negative drift for over a week after bad news — [record](https://cendekia.unisza.edu.my/neuaxis-e/Record/nottingham-26569/Description). Dissertation-level evidence.
- [BT] *(abstract)* IIM Bangalore student project: significant drift well into the third month after a surprise — [IIMB repository](https://repository.iimb.ac.in/handle/2074/20501). Low rigour; conflicts with the two above.
- [ACAD-IS] *(abstract)* US context: transaction costs significantly reduce PEAD profits and may explain its existence as well as its persistence (Ng, Rusticus and Verdi, 2008) — [LBS summary](https://www.london.edu/faculty-and-research/academic-research/i/implications-of-transaction-costs-for-the-post-earnings-announcement-drift).

### Inferences
- A long-only earnings-drift sleeve has no support strong enough to justify re-testing it against Centurion's null.
- If any leg exists, it is the bad-news short, which needs stock futures and so is limited to the F&O universe. That is a filter (do not hold through a bad surprise) more than a stream.
- A negative mean with t -3.12 on 16,635 events is itself a finding worth stratifying by liquidity, given Gupta & Dhusia's heterogeneity note.

### Gaps
- No India evidence was found on guidance, analyst revisions or SUE deciles after costs. Only one PEAD search was run before the budget ran out.

## Index rebalance and inclusion effects: magnitude, predictability, decay

### Takeaway
Inclusion effects are short-lived and reverse within about 60 days in the better-sampled study. No study found supports a positional long, and the post-exclusion rebound of about 4-7% over days 60-240 is the only positional lead.

### Cited Findings
- [ACAD-IS] *(abstract)* Chhatwani (2020), Nifty 50, 2000-2016: inclusions earn abnormal returns that reverse within 60 days; exclusions fall only for the first ten days, then earn about 4% to 7% abnormal between days 60 and 240 — [JCAR](https://publishingindia.com/archive/jcar/price-and-volume-effects-of-nifty-50-index-reorganization).
- [ACAD-IS] *(abstract)* Joshipura & Janakiramanan (2015): no significant announcement return on inclusion, positive effective-day return that does not persist, negative exclusion effect, volume spike on effective day attributed to index funds and ETFs; supports price pressure, not liquidity — [Afro-Asian J. Finance & Accounting](https://ideas.repec.org/a/ids/afasfa/v5y2015i1p21-36.html). No magnitudes in the abstract.
- [ACAD-IS] *(abstract)* Parthasarathy (2010), Nifty additions 1999-2010: significant positive permanent abnormal returns, rejects price pressure, argues additions convey information — [Asian J. Finance & Accounting](https://ideas.repec.org/a/mth/ajfa88/v2y2010i2p55-80.html). This conflicts with the two later studies.
- [ACAD-IS] *(abstract)* An earlier Nifty addition study finds positive abnormal returns on and after inclusion, attributed to index-fund demand — [JMR](https://i-scholar.in/index.php/jmr/article/view/89089).
- [MKT] 19 Nifty 50 swaps, Mar 2017 to Sep 2022, raw one-year returns from replacement date, no benchmark: added 11 up, 8 down; removed 6 up, 12 down; Bharti Infratel merged — [Angel One](https://angelone.in/blog/analysis-of-nifty50-rebalancing-since-2017).

### Inferences
- From the Angel One table, by hand: added mean +6.7%, median +5.9%; removed mean +1.7%, median -4.7%, and the removed mean rests on Vedanta's +165.2%. With n=19 and no benchmark this does not support a trade.
- The contradiction between Parthasarathy and later work is consistent with decay as index investing grew, but no study found tests decay directly.
- The reversal pattern argues against buying additions at or after the effective date.

### Gaps
- Nothing was found on Next 50, MSCI India or FTSE events, and magnitudes in basis points are missing for all studies.
- A 2006-2025 rebalance deck surfaced but is AI-generated, so it was not used.

## FII/DII flows, delivery %, promoter buying, insider trades, pledges, buybacks, block and bulk deals, F&O ban, ASM/GSM, circuit limits

### Takeaway
Only insider trades and pledges have usable evidence here, and neither is proven tradable after disclosure lag and costs. Most of the list was not searched before the budget ran out.

### Cited Findings
- Centurion: bulk deals leave nothing the morning after (from the brief).
- [ACAD-IS] *(abstract)* Kumar, NIT Karnataka thesis (2018), BSE insider disclosures Apr 2007 to Mar 2015, Carhart and event study: insider purchases earn positive and sales negative abnormal returns, while outsiders do not earn significant abnormal returns — [NITK repository](https://idr.nitk.ac.in/items/2e0597cb-33a1-4f4d-bfd2-6970ede2914a/full). Predates the 2015 PIT Regulations; not promoter-specific.
- [ACAD-IS] Gupta, Patel and Sane (2026), market-model event study, -10 to +10 days, NSE firms, 2009-2023, 176 SEBI final orders and 42 SAT decisions: no significant reaction to any enforcement type; the SAT sample has low power — [Leap Blog](https://blog.theleapjournal.org/2026/05/market-reaction-to-insider-trading.html).
- [ACAD-IS] Sonti & Thirumalai, 2,567 NSE pledge announcements after the Jan 2009 disclosure rule: mean CAR -0.53% [-1,+1] and -2.35% [-10,+10] around the pledge date, only 56% negative; just -0.05% to -0.58% around the exchange disclosure, attributed to leakage. Disclosure can lag 7 + 7 days — [article repost](https://npbang.blogspot.com/2016/08/share-pledges-and-market-reactions.html).
- [ACAD-IS] *(abstract)* Chauhan, Mishra and Spahr (2021), 1,452 firms: heavy pledging raises volatility and lowers firm value — [Financial Management record](https://digitalcommons.memphis.edu/facpubs/11664).
- [ACAD-IS] *(abstract)* Kalia (2024), 257 S&P BSE 500 firms, 2011-2020, panel fixed effects and IV-2SLS: pledging has a significantly positive link to future crash risk and a significantly negative link to future financial performance — [J. Advances in Management Research](https://econpapers.repec.org/RePEc:eme:jamrpp:jamr-01-2023-0003).
- [ACAD-IS] *(abstract)* Delivery %: a 2017 Granger study of 13 NSE banking stocks exists, but the result was not visible — [AMiner](https://www.aminer.org/pub/61c754915244ab9dcb0da8bc/study-on-relationship-between-delivery-quantity-to-total-quantity-traded-and-stock). A Capitalmind ranking of January 2016 delivery data is anecdotal — [Capitalmind](https://premium.capitalmind.in/2016/02/delivery-shockers-the-january-2016-edition/).

### Inferences
- The tradable part of a pledge event is the small disclosure-day reaction, so pledges are better as a portfolio filter (avoid high or rising pledges, given crash risk) than as a short signal.
- Kumar's "outsiders earn nothing" result fits Centurion's bulk-deal null: the disclosed version of an informed trade tends to be priced by the time an outsider can act.
- SEBI enforcement news is not a usable signal on this evidence.

### Gaps
- **Not searched (budget exhausted):** buybacks, FII/DII flows, F&O ban list, ASM/GSM lists, circuit limits, block deals.
- No study isolates promoter buying under the 2015 PIT Regulations.
- No delivery-% study with a usable result or post-cost figures was found.

## Small-cap, micro-cap and SME premia: returns, liquidity, and what survives impact and circuit-limit trapping

### Takeaway
Size alone pays little per unit of risk: Smallcap 250 Sharpe 0.47 and Microcap 250 Sharpe 0.51 against Nifty 500's 0.45, with drawdowns of -76% and -82%. Selection inside small caps helps on paper, but no net-of-impact or circuit-trapping evidence was found.

### Cited Findings
- [BT]+[LIVE] *(back-filled)* 1 Apr 2005 to 16 Oct 2026, TRI, gross: Nifty Smallcap 250 15.56%, vol 24.63%, Sharpe 0.47, MaxDD -75.56%, worst 5Y -6.17%; Nifty Microcap 250 17.50%, 27.38%, 0.51, -82.04%, worst 5Y -11.93%; Nifty Smallcap250 Momentum Quality 100 21.31%, 24.32%, 0.68, -72.43%, worst 5Y +0.73%; Smallcap250 Quality 50 18.00%, 21.91%, 0.60, -70.26% — [nifty-factor-analytics](https://github.com/RohitVarre/nifty-factor-analytics).
- [ACAD-IS] *(abstract)* IIMA SMB premium could not be read reliably; one of the figures displayed is "nearly 0%" — [IIMA](https://www.iima.ac.in/publication/four-factor-model-indian-equities-market).

### Inferences
- Gross index data gives roughly 2-4pp a year for size against Nifty 500. Against that, drawdown is 12-18pp deeper, and index figures carry no impact or circuit cost, so the size premium alone does not look worth adding.
- A Smallcap Momentum Quality sleeve shows a large gross figure, but its cost-adjusted value is unknown.

### Gaps
- **Not searched (budget exhausted):** SME platform returns, circuit-limit trapping, small-cap impact costs, and microcap survivorship. Nothing here measures what survives net of these.

## Sector rotation and relative strength in India

### Takeaway
No evidence was gathered: the budget ran out before this question was searched.

### Cited Findings
- None.

### Inferences
- None drawn. Absence here reflects no search, not absence of evidence.

### Gaps
- All of it: sector momentum returns, rebalance frequency, costs and regime behaviour.

## Fundamentals-based selection (Piotroski, Greenblatt, quality-growth, earnings momentum) with point-in-time data

### Takeaway
F-score and Magic Formula look strong in Indian studies, but none read here confirms point-in-time data, and the numbers are in-sample, so they are leads to test rather than proven streams. Simple ROE-quality without a liquidity filter looks weak.

### Cited Findings
- [ACAD-IS] Singh & Kaur (2015), high book-to-market Indian stocks: top F-score beat the value portfolio by 18.402% a year market-adjusted; each F-score point adds about 4.93% to one-year market-adjusted return. Sample not stated — [IJLM](https://www.emerald.com/insight/content/doi/10.1108/IJLMA-09-2014-0055/full/html).
- [ACAD-IS] *(abstract)* IIM Calcutta (2009): F-score 7-9 portfolios far above market, 0-3 very poor; authors caution on direct application — [IIMC repository](https://ir.iimcal.ac.in:8443/jspui/handle/123456789/4128?mode=full).
- [MKT] *(abstract)* 5Paisa Capital working paper (2025), Nifty 100 constituents, 2007-2024: even the unmodified Magic Formula outperforms substantially; broker-affiliated, PDF unreadable, so PIT, survivorship, cost and magnitude are unverified — [MPRA 126237](https://mpra.ub.uni-muenchen.de/126237/1/MPRA_paper_126237.pdf).
- [BT] ROE-only Quality, top 50 of Nifty 500, net: low Scaled Turnover 15.45%, high Scaled Turnover 8.16%, recovery 14 v 45 months; Value composite (PE, PB, yield) 13.74% v 12.23% — [BacktestIndia](https://backtestindia.com/blog/factor-scaled-turnover-nifty500-universal-liquidity-premium).
- [ACAD-IS] Value-glamour offered no consistent superior return, 100 NSE firms, 2014-2018 — [Gupta & Dhusia](https://journals.vutbr.cz/index.php/trends/article/view/541).

### Inferences
- An 18.4% a year top-F-score spread is too large to take at face value; the small top-score group and unknown data timing likely inflate it.
- F-score looks most useful as a filter inside a value or quality sleeve, tested on filing-date-stamped data.

### Gaps
- No study read confirms point-in-time fundamentals. Prowess itself has had retrospective additions ([IIMA](https://www.iima.ac.in/publication/four-factor-model-indian-equities-market)).
- Nothing found for quality-growth or earnings momentum in India. No study tests F-score and Magic Formula together.

## Short-side anomalies and whether they can be captured

### Takeaway
Rules and liquidity were not searched, so capture is unresolved. What is known is that long-only alpha is reported to survive, so the short leg is not required for momentum, low-vol or quality.

### Cited Findings
- [ACAD-IS] *(abstract)* Long-only momentum, low-vol and quality alpha survives implementation costs, top 200 stocks — [Raju & Teli 2022](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4000418).
- [ACAD-IS] Momentum is profitable after costs when only the long side is implemented — [Aggarwal & Gupta 2017](https://www.publishingindia.com/archive/jcar/can-small-investors-gain-from-momentum-trading-in-india-an-empirical-investigation).
- [BT]+[LIVE] *(back-filled)* Nifty High Beta 50: 6.88% CAGR, Sharpe 0.19, MaxDD -78.72%, 2005-2026, TRI — [nifty-factor-analytics](https://github.com/RohitVarre/nifty-factor-analytics).
- [ACAD-IS] *(abstract)* Negative drift after bad earnings news, about a week — [Nottingham](https://cendekia.unisza.edu.my/neuaxis-e/Record/nottingham-26569/Description).
- [BT] *(survivorship)* Monthly long-short on 77 current Nifty 500 names, Jan 2015 to Dec 2024, momentum 50% + reversal 30% + earnings-drift proxy 20%, shorts via stock futures, about 0.35% all-in round trip, 541 trades a year, about 270% turnover: net -7.15% a year, Sharpe -3.271, MaxDD -48.15%, against Nifty 50 +15.39% — [IMMR-Hybrid](https://anirudh-vadrevu.github.io/Quant-Journals/IMMR-Hybrid.html). Single-author write-up; gross return is estimated rather than measured, and the page's cost arithmetic does not reconcile, so it is a warning rather than a measurement.

### Inferences
- High beta is the cleanest candidate to exclude from the long book. Whether shorting it through futures pays after roll and carry is unknown.
- Statutory cost is not the obstacle on the short side: a futures round trip costs about 0.06% against about 0.22% for delivery (see Costs). Basis, roll slippage and the limit to F&O names are the open issues.

### Gaps
- **Not searched (budget exhausted):** SEBI cash-short rules, SLB liquidity, current F&O universe size, futures basis.

## Regime behaviour: momentum, quality, low-vol and value in bull, bear and sideways markets, and the momentum-crash question

### Takeaway
Momentum crashes in crises on Indian academic evidence and is strongest after up markets, but in the Covid fall it held up better than the index. Low-vol is the steadiest on drawdown. A full factor-by-episode table could not be built from the sources available.

### Cited Findings
- [ACAD-IS] *(abstract)* Maheshwari & Dhankar (2017): momentum returns high before the 2008 crisis, negative during it, recovering after; strategies "crash during the financial crises periods" — [Vision](https://ideas.repec.org/a/sae/vision/v21y2017i1p1-12.html).
- [ACAD-IS] *(abstract)* Maheshwari & Dhankar (2018): strong momentum profits after an up market, long-term contrarian stronger after a down market, and market-state risk models do not explain the profits — [IIMK Society & Management Review](https://econpapers.repec.org/RePEc:sae:iimkoz:v:7:y:2018:i:2:p:154-170).
- [ACAD-IS] *(abstract)* Ali & Joshi (2022), large caps: five momentum strategies still beat the benchmark in a later period of domestic slowdown, so crash evidence is mixed — [IARAS, search snippet; PDF later returned 404](https://www.iaras.org/filedownloads/ijems/2022/007-0002(2022).pdf).
- [ACAD-OOS] *(abstract)* Factor timing has limited ability; only 12-month momentum is significant — [Raju 2025](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5513940).
- Nifty200 Momentum 30 episodes [BT] for 2008 *(back-filled)*, [LIVE] from 2020: 2008 -67.9% v Nifty 50 -64.6%; 2020 -34.2% v -39.6%; 2022-23 -14.8% v -13.2% — [Curious Investing Insights](https://curiousinvestinginsights.substack.com/p/momentum-or-mayhem-unpacking-the), source of figures not named.

| Episode | Drawdown | Days to trough | Days to recover |
|---|---|---|---|
| 2013 Jul-Aug | -14.05% | 44 | 41 |
| 2018 Aug-Oct | -16.76% | 58 | 441 |
| 2010-12 | -26.34% | 405 | 325 |
| 2020 Covid | -34.21% | 33 | 178 |
| 2024-25 | -31.79% | 192 | Not recovered at Aug 2025 |

Table: Momentum 200 / Nifty200 Momentum 30, top-10 average 141 days to trough and 191 to recover — [Wright Research](https://www.wrightresearch.in/blog/momentum-strategies-underperforming-2025-data-insights). Grade: vendor; pre-Aug 2020 is *(back-filled)*.

- [BT]+[LIVE] Worst drawdowns 2005-2026 *(back-filled)*: low-vol -48% to -51%, quality -54% to -56%, value -63% to -66%, momentum -68% to -70%, Alpha 50 -79% — [nifty-factor-analytics](https://github.com/RohitVarre/nifty-factor-analytics).
- [MKT] Alpha Low Vol contains drawdowns better and lags in rallies relative to Momentum 30 (opinion) — [PrimeInvestor](https://primeinvestor.in/reports/is-uti-nifty-momentum-30-index-fund-nfo-worth-it/).

### Inferences
- India's recent momentum damage differs from the Daniel-Moskowitz pattern of crashes in rebounds. In 2024-25 momentum fell with the market, and by more; in 2020 it fell less. That is two episodes, not a test.
- Recovery is the under-reported cost: 2018 was shallow but took 441 days, and 2024-25 was unrecovered at August 2025, with the index still down over the year to July 2026 on a price basis.
- Low-vol is the natural regime complement because its drawdown is about 20pp shallower, but no calendar-year correlation or episode data was obtained to prove the offset.

### Gaps
- No factor-by-episode returns for 2011, 2015-16, 2018-19, 2022 or 2024-26 for quality, low-vol or value.
- No sideways-market classification with factor returns.
- No Daniel-Moskowitz-style rebound test on Indian data was found.

## Costs: STT, stamp duty, impact and slippage, and net returns by rebalance frequency

### Takeaway
Statutory delivery cost is about 0.22% per round trip plus a per-sale DP charge, so each 100% of annual turnover costs about 0.24% before slippage on a ₹1 lakh position. No controlled India study of weekly versus monthly versus quarterly rebalancing was found, and impact cost, the larger unknown in mid and small caps, was not reached before the search budget ran out.

### Cited Findings

**Statutory and broker charges, Zerodha, read 31 Oct 2026; page gives no effective date for these equity rates** — [Zerodha charges](https://zerodha.com/charges). Grade: first-party broker schedule.

| Charge | Equity delivery | Equity futures | Equity options |
|---|---|---|---|
| Brokerage | ₹0 | 0.03% or ₹20 per order, whichever is lower | ₹20 per order |
| STT/CTT | 0.1% buy and sell | 0.05% sell side | 0.15% sell side on premium; 0.15% of intrinsic value if bought and exercised |
| NSE transaction charge | 0.00307% | 0.00183% | 0.03553% on premium |
| SEBI charges | ₹10 per crore | ₹10 per crore | ₹10 per crore |
| GST | 18% on brokerage + SEBI + transaction charges | Same | Same |
| Stamp duty | 0.015% buy side | 0.002% buy side | 0.003% buy side |
| DP charge | ₹15.34 per scrip on sale | Not listed | Not listed |

- [LIVE] Capitalmind says it rebalances its momentum smallcase weekly, estimates about 1% of gains for brokerage, STT and DP, and reports live data only since Jan 2019 — [Capitalmind portfolio](https://premium.capitalmind.in/momentum-portfolio/), [rebalance guide](https://premium.capitalmind.in/2020/10/best-practices-smallcase-rebalance/). Self-reported manager data, not audited here.
- [BT] Cost assumption in the Indian net backtests: 0.11% transaction + 0.05% flat slippage; STCG 20%, LTCG 12.5% — [BacktestIndia](https://backtestindia.com/blog/momentum-factor-india-liquidity-premium-scaled-turnover).
- [BT] *(survivorship)* Retail test, today's large caps, 10 years to Jun 2026, 5 bps per leg: top 3 of 8 on 6-month lookback 15.7% CAGR, MaxDD 39.3%; top 1 of 3 on 60-day lookback 7.6%, MaxDD 46.1%, permutation p = 0.711, against Nifty 50 at about 11.3%. It states but does not test that rebalancing faster than monthly mostly adds cost and tax — [Thuztra](https://thuztra.pages.dev/research/momentum-investing-india-honest-backtest).
- [BT] *(survivorship)* "India full stack" cost of about 0.35% per round trip, including impact and stock-futures financing; at about 270% turnover the write-up reports net -7.15% a year — [IMMR-Hybrid](https://anirudh-vadrevu.github.io/Quant-Journals/IMMR-Hybrid.html). Its cost arithmetic does not reconcile with its own gap figures.

### Inferences
- **Delivery round trip, by hand from the table:** STT 0.20% + stamp 0.015% + transaction 0.0061% + SEBI 0.0002% + GST 0.0011% = about 0.22%. DP adds 0.015% on a ₹1 lakh sale and 0.06% on ₹25,000, so small lots at high churn pay noticeably more.
- **Futures round trip, ₹10 lakh notional:** STT ₹500 + transaction ₹37 + brokerage ₹40 + stamp ₹20 + GST ₹14 + SEBI ₹2 = about ₹613, or 0.06%. A monthly roll would cost about 0.7% a year before basis and slippage.
- **Frequency drag, statutory only, ₹1 lakh lots, per year:** 200% turnover 0.5%; 500% 1.2%; 1,000% 2.4%. Slippage comes on top: each 0.10% of one-way slippage adds another 0.2% per 100% of turnover. The turnover levels are illustrative, not estimates for any strategy.
- BacktestIndia's 0.11% looks like one side of this 0.22% round trip, so its statutory assumption is sound if applied per side; the page does not say. Its 0.05% flat slippage is the weak part.
- Momentum's semi-annual index rebalance and the annual BacktestIndia rebalance avoid most frequency cost, while weekly systems carry far more; the frequency trade-off remains unmeasured on Indian data.
- Flat slippage is the main weakness in the Indian net backtests read here. The low-turnover and small-cap results are the most exposed to it.

### Gaps
- **Not searched (budget exhausted):** NSE impact cost by index and any mid- and small-cap slippage studies. No sourced impact figure exists in these notes.
- No gross-to-net comparison by rebalance frequency.
- Zerodha's page gives no effective date, and the budget or circular history behind the F&O STT rates was not checked.
- No audited live record was obtained. Capitalmind says Adaptive Momentum has been live since 2019, but the page read shows no return figures and gives inconsistent AUM (₹1,550 crore and ₹1,750 crore) — [Capitalmind PMS](https://www.capitalmind.in/blog/best-pms-capitalmind-pms-strategies-performance).

## Evidence-graded map: what to add to a momentum core

### Takeaway
Add low-vol and quality first, ideally as a blended sleeve; test a low-turnover tilt inside momentum; use high beta and pledge as filters. Do not spend time on PEAD, index inclusion longs, bulk deals or size alone. This ranking rests largely on gross, back-filled index data and needs Centurion's own net test.

### Cited Findings
All figures are from the sections above; 2005-2026 TRI gross figures are from [nifty-factor-analytics](https://github.com/RohitVarre/nifty-factor-analytics).

| Source | Verdict | Grade | CAGR / Sharpe / MaxDD | Turnover, cost | Capacity at ₹ few cr | Bull / bear / range |
|---|---|---|---|---|---|---|
| Momentum (core) | Real; live excess looks reduced | ACAD-IS, BT, some LIVE | 18.2% / 0.62 / -67.7% | Semi-annual index; high relative to others | Fine in large/mid | Strong after up markets; crashes in crises; range unknown |
| Low volatility | Real, best diversifier candidate | ACAD-IS, BT | 16.5% / 0.66 / -48.7% | Low (Raju & Teli); no turnover-bucket gap | Fine | Lags rallies, shallower falls; range unknown |
| Quality | Real, needs liquidity screen | ACAD-IS, BT | 16.8% / 0.64 / -55.6% | Low (Raju & Teli); high-turnover half weak | Fine | Unknown by episode |
| Momentum + Quality blend | Best on paper | BT (back-filled) | 20.4% / 0.76 / -59.0% | Not found | Check small-cap share | Worst 5Y +5.05% |
| Multi-factor MQVLv | Good on paper | BT (back-filled) | 19.0% / 0.75 / -53.9% | Not found | Fine | Worst 5Y +1.71% |
| Value | Weak risk-adjusted | BT, ACAD-IS mixed | 16.2% / 0.47 / -66.1% | Not found | Fine | Worst 5Y -8.13% |
| Size, microcap | Little per unit of risk | BT (gross) | 17.5% / 0.51 / -82.0% | Unknown | Impact, circuits unmeasured | Worst 5Y -11.93% |
| High beta (avoid) | Filter | BT | 6.9% / 0.19 / -78.7% | n/a | n/a | Worst 5Y -15.90% |
| F-score, Magic Formula | Lead, unproven | ACAD-IS, MKT | Not comparable | Unknown | Fine | Unknown |
| Low Scaled Turnover tilt | Lead, cost-suspect | BT | 19.4% net / n.r. / -66.4% | Annual; flat slippage | Lower-liquidity names | Spread positive in 90%+ of 10Y windows |
| Pledge | Filter | ACAD-IS | -2.35% CAR, mostly before disclosure | Low | Fine | Crash-risk link |
| Insider buying | Unproven post-2015 | ACAD-IS | Outsiders earn nothing | Unknown | Fine | Unknown |
| PEAD | No long edge | ACAD-IS + Centurion | Null | n/a | n/a | n/a |
| Index inclusion | Reverses | ACAD-IS, conflicted | Reverses in 60 days | Event | Fine | Unknown |
| Bulk deals | Nothing | Centurion | Null | n/a | n/a | n/a |
| Sector rotation, buybacks, flows, ASM/GSM, F&O ban, SME | Not researched | n/a | n/a | n/a | n/a | n/a |

### Inferences
- Candidate order to test in Centurion: (1) low-vol or quality-low-vol sleeve; (2) momentum + quality combination at signal level; (3) momentum with low-Scaled-Turnover filter under size-dependent impact; (4) high-beta and pledge exclusions; (5) F-score filter on PIT data.
- On the user's preferred metrics, Calmar improves from 0.27 for the momentum index to 0.33-0.35 for low-vol, MQ and MQVLv, on gross, partly back-filled data.
- This map is incomplete. Ranking the unsearched items as "not worth adding" would not be justified.

### Gaps
- Net-of-cost Sharpe for each sleeve on a common basis does not exist in the sources found; Centurion's own walk-forward is the only way to get it.
- Correlation of factor sleeves with the momentum core was not found.
