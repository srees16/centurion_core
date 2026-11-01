# Audited Live Track Records of Systematic and Quant Investors in India (reality check on the golden target)

As of 31 Oct 2026. Golden target: Sharpe >= 1.4, CAGR >= 35%, Calmar >= 1, pre-tax, swing or positional.

**Evidence grades used on every figure:**
- **[REG]** regulator document or regulator study (SEBI), or a press or blog report of one (marked "via")
- **[MF]** SEBI-regulated mutual fund factsheet disclosure
- **[CRS]** CRISIL AIF benchmark report, as issued to a fund manager
- **[IDX]** index provider data (NSE Indices), usually relayed by an AMC
- **[MGR]** manager-reported
- **[AGG]** aggregator (PMS Bazaar, PMS AIF World, APMI data relayed by a third party)
- **[PRS]** press or blog
- **Broker** a broker's own support page, staff forum reply or blog (retail algo section only)

**No figure below is graded "audited".** I found no audited statement for any manager. Capitalmind's own page says its performance data "is not verified by SEBI". APMI's own per-manager page returned only a navigation menu, so SEBI-format data reached me only through aggregators.

**Research limits:** the web search budget ran out partway through, so several named funds were not searched at all. These are listed under Gaps rather than guessed. Style labels in the master table are my classification unless the source says so. Numbers marked "my calc" are derived here from cited figures and are not published by the source.

## Audited or regulator-reported returns of Indian quant and systematic managers (PMS, AIF, MF)

### Takeaway
Live quant records in India are short and modest. The longest-running quant PMS found (Capitalmind Adaptive Momentum, since Mar 2019) shows about 17% since inception, and its trailing 5-year return fell from 30.2% in June 2024 to about 7% in October 2026. The longest discretionary PMS records (2002-03 starts) run about 19-20% and cover every crash, and no strategy found clears 30% over 10 years.

### Cited Findings

**Master table, net basis as stated, TWRR where the source says so**

| Strategy (style) | Grade | Start | Window end | SI CAGR vs benchmark | Trailing returns | Max DD / Sharpe | Skips | Basis | Source |
|---|---|---|---|---|---|---|---|---|---|
| Capitalmind Adaptive Momentum (rule-based, long-only, cash switch) | MGR | 5 Mar 2019 | 29 Oct 2026 | not stated | 1Y 4%, 2Y -5.2%, 3Y 8.3%, 4Y 9.5%, 5Y 7.1%, 6Y 15.8%, 7Y 17.6% (Nifty 50 TRI: -11%, -2.1%, 7.6%, 7.8%, 6.6%, 13.2%, 11.2%) | not disclosed | 2008, 2011, 2018 | net of fees and costs; "not verified by SEBI" | [Capitalmind](https://www.capitalmind.in/portfolio-management-service/momentum) |
| Same | AGG | Mar 2019 | 30 Jun 2026 | 17.31% vs Nifty 50 TRI 12.50% | 1Y -3.88%, 3Y 10.76%, 5Y 8.61% | none | same | not stated | [PMS Bazaar](https://my.pmsbazaar.com/AMC/Capitalmind-Financial-Services); SI via [PMS AIF World](https://www.pmsaifworld.com/portfolio/wizemarkets-capitalmind-momentum-fund) (17.14% on the later 30 Sep 2026 view) |
| Capitalmind Surge India (style not captured) | AGG | Nov 2017 | 30 Jun 2026 | not provided | 1Y -5.50%, 3Y 19.48%, 5Y 17.32% | none | 2008, 2011 | not stated | [PMS Bazaar](https://my.pmsbazaar.com/AMC/Capitalmind-Financial-Services) |
| Capitalmind All Weather Equity | AGG | Jul 2022 | 30 Jun 2026 | not provided | 1Y 0.85%, 3Y 15.11% | none | 2008, 2011, 2018, most of 2022 | not stated | [PMS Bazaar](https://my.pmsbazaar.com/AMC/Capitalmind-Financial-Services) |
| Capitalmind Altitude (formerly SmartCore, quant) | AGG | Jan 2025 | 30 Jun 2026 | not provided | 1Y 4.07% | none | all but 2024-26 | not stated | [PMS Bazaar](https://my.pmsbazaar.com/AMC/Capitalmind-Financial-Services) |
| Qode All Weather | AGG | Nov 2024 | 30 Sep 2026 | not provided | 1Y 18.55% | none | all but 2024-26 | TWRR, net basis unstated | [PMS Bazaar](https://pmsbazaar.com/AMC/Qode-Advisors-LLP/PMS) |
| Qode Tactical | AGG | Nov 2024 | 30 Sep 2026 | not provided | 1Y 12.55% | none | all but 2024-26 | TWRR | [PMS Bazaar](https://pmsbazaar.com/AMC/Qode-Advisors-LLP/PMS) |
| Qode Growth (small cap) | AGG | Mar 2025 | 30 Sep 2026 | not provided | 1Y 32.65% | none | all but 2026 | TWRR | [PMS Bazaar](https://pmsbazaar.com/AMC/Qode-Advisors-LLP/PMS) |
| Motilal Oswal Value (discretionary) | AGG | 18 Feb 2003 | 30 Sep 2026 | 18.72% vs BSE 500 TRI 16.75% | 1Y -1.38%, 3Y 15.69%, 5Y 13.16% (benchmark -2.33%, 9.20%, 8.90%) | 1Y SD 22.62% | none | not stated | [PMS AIF World](https://www.pmsaifworld.com/portfolio/motilal-oswal-value-pms/) |
| Motilal Oswal Value Migration (same inception date) | MGR | 18 Feb 2003 | 28 Feb 2026 | 13.44% vs BSE 500 TRI 16.09% | not captured | none | none | conflicts with row above; page later returned HTTP 502, so seen only via search summary | [Motilal Oswal](https://pms.motilaloswalmf.com/pageDetail/value) |
| Alchemy High Growth (discretionary) | AGG | May 2002 | Feb 2025 | 19.9% vs BSE 500 TRI 17.0% | Rs 1 cr to Rs 60.5 cr; beat benchmark 15 of 23 calendar years; "+275 bps net annualised alpha" | none | none | basis unstated | [PMS Bazaar](https://pmsbazaar.com/News/Alchemy-Capital-Management/Alchemy_High_Growth_PMS_Strategy_colon_23_Years_Rs_1_Crore_to_Rs_60_Crore_Quiet_Compounder_Journey) |
| Buoyant Opportunities (discretionary multi-cap) | AGG | 1 Jun 2016 | 30 Sep 2026 | 19.98% vs BSE 500 TRI 13.34% | 1Y 5.03%, 3Y 13.50%, 5Y 15.64% (benchmark -2.33%, 9.20%, 8.91%) | -5.8% vs -11.8% in Oct 2024-Mar 2025; no max DD | 2008, 2011 | not stated | [PMS AIF World](https://www.pmsaifworld.com/portfolio/buoyant-capital-opportunities-multi-cap-fund/), [Sharescart](https://www.sharescart.com/pms/buoyant-capital-private-limited/buoyant-opportunities-pms/) |
| Aequitas India Opportunities (discretionary) | AGG | not stated | 30 Sep 2026 | 31.40% SI | 10Y 26.43%, 5Y 30.33% | none | 2008, 2011 in the 10Y window | "not independently verified"; 5Y conflicts across the page's tables: 49.21%, 46.55%, 29.55%, 30.33% | [AltportFunds citing APMI](https://www.altportfunds.com/evaluating-the-best-pms-in-india-2025-rankings-fees-performance-comparisons/) |
| Sameeksha India Equity | AGG | not stated | 30 Sep 2026 | 20.15% | 10Y 19.92%, 5Y 14.78% | none | 2008, 2011 in the 10Y window | same page | [AltportFunds](https://www.altportfunds.com/evaluating-the-best-pms-in-india-2025-rankings-fees-performance-comparisons/) |
| ValueQuest Platinum | AGG | not stated | 30 Sep 2026 | 18.36% | 10Y 17.61%, 5Y 15.16% | none | 2008, 2011 in the 10Y window | same page | [AltportFunds](https://www.altportfunds.com/evaluating-the-best-pms-in-india-2025-rankings-fees-performance-comparisons/) |
| 2Point2 Long Term Value | AGG | not stated | 30 Sep 2026 | 18.10% | 10Y 17.80%, 5Y 14.10% | none | 2008, 2011 in the 10Y window | same page | [AltportFunds](https://www.altportfunds.com/evaluating-the-best-pms-in-india-2025-rankings-fees-performance-comparisons/) |
| Stallion Core | AGG | Oct 2018 | 30 Sep 2026 | 28.31% | 5Y 23.79% | none | 2008, 2011, most of 2018 | same page | [AltportFunds](https://www.altportfunds.com/evaluating-the-best-pms-in-india-2025-rankings-fees-performance-comparisons/) |
| ICICI Prudential PMS Contra | PRS (citing PMS Bazaar) | not stated | Sep 2025 | not stated | 1Y -3%, 5Y 28%; AUM Rs 12,110 cr | none | 2008, 2011, 2018 in the 5Y window | not stated | [NiftyTrader](https://www.niftytrader.in/content/pms-returns-turn-negative-as-market-skew-hits-icici-prudential/amp) |
| ASK India Entrepreneurs | PRS (citing PMS Bazaar) | not stated | Sep 2025 | not stated | 1Y -9%, 5Y 14%; AUM Rs 10,484 cr | none | same | not stated | [NiftyTrader](https://www.niftytrader.in/content/pms-returns-turn-negative-as-market-skew-hits-icici-prudential/amp) |
| ICICI Prudential Nifty 200 Momentum 30 ETF (rule-based, investable tracker) | MF | 4 Aug 2022 | 30 Sep 2026 | 12.62% vs Nifty 200 Momentum 30 TRI 13.08% vs Nifty 50 TRI 7.76% | 1Y -1.06%, 3Y 8.55% (TRI -0.47%, 9.19%; Nifty 50 TRI -7.09%, 6.05%) | none | 2008, 2011, 2018, most of 2022 | scheme returns, loads excluded | [ICICI Prudential factsheet](https://www.icicipruamc.com/blob/knowledgecentre/factsheet-schemes/Passive%20Schemes/ETFs%20Schemes/ICICI%20Prudential%20Nifty%20200%20Momentum%2030%20ETF.pdf) |
| DSP India Enhanced Equity Fund (Cat III AIF) | CRS | not stated | 30 Sep 2024 | not stated | 1Y 28.62%, 3Y 13.64%, 5Y 11.90%, 7Y 12.12% (Cat III index 31.56%, 14.63%, 18.86%, 13.85%) | none | 2008, 2011 in the 7Y window | post-expense, pre-carry, pre-tax | [CRISIL report](https://www.dspim.com/media/pages/documents/31db646381-1757771758/benchmark-report_dsp-india-enhanced-equity-fund_cat-iii_sep-2024.pdf) |

**Point-in-time decay of a headline number**
- Capitalmind's press release of 25 Jun 2024 reported Adaptive Momentum at 30.2% annual over five years, Surge India at 24.2% (NIFTY 50 16.1%, CNX 500 18.6%), and both above 60% over the past year [MGR, basis unstated] — [Capitalmind PR](https://capitalmind.in/blog/capitalmind-pms-surpasses-2000-crore-in-aum-demonstrating-exceptional-market-performance).
- Capitalmind's launch blog for its first AIF cited 28.8% CAGR since inception against 17.6% for Nifty 50 TRI [MGR, seen via search summary, date not captured] — [Capitalmind blog](https://capitalmind.in/blog/capitalmind-launches-its-first-aif-capitalmind-select-india-one).
- Against those, the same strategy now shows 7.1% over 5 years net [MGR] — [Capitalmind](https://www.capitalmind.in/portfolio-management-service/momentum).

**Industry context**
- FY25 PMS Bazaar universe of over 370 strategies: equity PMS averaged 6.81% in FY25 after 46.28% in FY24, with Nifty 50 TRI 6.65% and BSE 500 TRI 5.96% [AGG] — [PMS Bazaar](https://pmsbazaar.com/Blogs/PMS-Strategies-Face-Reality-Check-in-FY25-A-Year-of-Tested-Patience). The page does not mention closures.
- Quant and factor strategies were 40-50% of new PMS launches since 2024, and quant PMS launches were "almost non-existent" just over a year earlier. New schemes (Qode All Weather and Tactical, Capitalmind SmartCore, ArthAlpha Quant) held about Rs 160 cr combined and returned 10-17% over six months, against BSE 500 TRI 7.19% [PRS citing PMS AIF World, Oct 2025] — [5paisa](https://www.5paisa.com/news/quant-pms-funds-surge-as-data-driven-investing-gains-favour-among-hnis).
- Buoyant Opportunities took the highest PMS net inflows in Apr-May 2026, Rs 1,340 cr [PRS] — [Cafemutual](https://cafemutual.com/news/cafe-alt/38278-pms-inflows-fy27-buoyant-capital-leads-delhi-tops-state-wise-flows).

**Quant Mutual Fund and the June 2024 probe**
- SEBI ran search-and-seizure at the Mumbai head office and Hyderabad over suspected front-running [PRS] — [Moneylife](https://moneylife.in/article/sebi-investigates-quant-mutual-fund-for-suspected-frontrunning-amid-rapid-growth-moneycontrol/74473.html).
- The fund said it had received inquiries and would cooperate [PRS] — [Moneylife](https://moneylife.in/article/quant-mutual-fund-finally-comes-clean-on-inquiries-from-sebi-mf-industry-is-watching-with-apprehension/74663.html).
- Net inflows of Rs 1,905 cr in the week of 17-21 Jun 2024 reversed to Rs 2,794 cr of outflows the following week [PRS] — [Angel One](https://www.angelone.in/news/mutual-funds/quant-mutual-fund-sees-positive-inflows-outflows-due-front-running-allegation).
- A Zee Business Hindi headline says Sandeep Tandon and Sumana Paruchuri approached SEBI to settle the case. The article itself was paywalled (HTTP 402), so only the headline is confirmed and its date is unknown [PRS] — [Zee Business](https://zeebiz.com/hindi/stock-markets/quant-mutual-funds-sandeep-tandon-approaches-sebi-to-settle-front-running-case-197758).
- Quant Flexi Cap 10Y 19.54% as of April 2026 [PRS, broker list] — [Angel One](https://www.angelone.in/news/mutual-funds/best-performing-equity-mutual-funds-for-april-2026-based-on-5-yr-cagr).

**Other**
- Capitalmind Flexi Cap MF: launched 4 Aug 2025, 3.98% simple annualised to 31 Jan 2026 against Nifty 500 TRI 2.57% [PRS] — [Whalesbook](https://www.whalesbook.com/news/English/mutual-funds/Capitalmind-Fund-Traction-Soars-But-Returns-Lag-Amid-Volatility/69871d29b81b2b81dcfc2ae5).
- Abakkus All Cap Approach inception 29 Oct 2020 [broker deck, seen via search summary] — [HDFC Securities](https://hdfcsec.com/hsl.docs/Monthly-Product-Deck-November25-202512171008470413106.pdf). Return columns were not readable.

### Inferences
- The two longest PMS records (2002-03 starts) beat the BSE 500 TRI by only 2-3 pts a year, and that benchmark itself ran 16-17% because the window opens near a market low. So 19-20% over 23 years is mostly beta plus a small premium.
- All high 5-year CAGRs in the tables are post-COVID windows starting 2020-21. They skip 2008, 2011 and 2018.
- Adaptive Momentum's five-year fall from 30.2% to 7.1% shows how much a 5-year number depends on its window. The June 2024 window began before the 2020 rebound and ended three months before the momentum high; the October 2026 window sits inside the momentum drawdown.
- The ICICI ETF ran 0.46 pts a year behind its index live (my calc), which is the investable cost of the rule-based premium before tax.
- The two Motilal Oswal rows disagree by over 5 pts. Until resolved, use neither as a firm number. Both beat or trail a mid-teens benchmark, so neither changes the conclusion.

### Gaps
- No data found, because the search budget ran out: Wright Research, Samco, 360 ONE PMS long-run record, Avendus, Edelweiss Alternative, Nuvama PMS, Tata Quant, DSP Quant, Motilal Oswal Quant, NJ smart-beta funds, Kotak Quant, Sundaram, Abakkus returns, ValueQuest S.C.A.L.E.
- "QRC" in the brief: the only QRC I found is PMS AIF World's "Quality, Risk, Consistency" rating, not a manager — [PMS AIF World](https://www.pmsaifworld.com/quality-risk-consistency-qrc-5p-analysis-motilal-oswal-value-strategy).
- No max drawdown or Sharpe for any named PMS from a reliable source. Aggregator Sharpe tables were garbled in extraction and I did not attribute them.
- SEBI's own PMS and APRN pages were not reachable. No final outcome of the Quant MF probe was found.
- Survivorship: no count of closed or merged PMS strategies was found.

## Category III long-short AIFs: returns, volatility, drawdowns, Sharpe

### Takeaway
The CRISIL Cat III index (all sub-strategies, pre-carry, pre-tax) compounded about 12% over CY15-CY23, and the top-quartile threshold over 7 years to Sep 2024 was only 14.48% across just 15 schemes. Long-short averages run roughly 9-22% a year on AMC-supplied numbers with mixed fee bases. I found no published Sharpe; annual data imply about 0.5 for the whole of Cat III (my calc).

### Cited Findings

**CRISIL AIF Index, Category III (INR), post-expense, pre-carry, pre-tax, values at 30 Sep 2024, 326 schemes, covering long-only, long-short, hybrid and other Cat III [CRS]** — [CRISIL report for DSP](https://www.dspim.com/media/pages/documents/31db646381-1757771758/benchmark-report_dsp-india-enhanced-equity-fund_cat-iii_sep-2024.pdf)

| Horizon | Index return | Schemes with history | Top quartile threshold | Median | Bottom quartile threshold |
|---|---|---|---|---|---|
| 1Y | 31.56% | 208 | 49.15% | 36.07% | 24.18% |
| 2Y | 23.28% | 147 | 35.53% | 28.64% | 15.76% |
| 3Y | 14.63% | 93 | 21.45% | 15.44% | 8.99% |
| 5Y | 18.86% | 51 | 25.26% | 17.50% | 10.95% |
| 7Y | 13.85% | 15 | 14.48% | 9.73% | 7.39% |

- CRISIL Cat III calendar-year returns in INR: CY15 7.75%, CY16 8.82%, CY17 25.01%, CY18 -5.60%, CY19 6.54%, CY20 18.06%, CY21 30.99%, CY22 -1.03%, CY23 23.01%, CY24 to Sep 21.76% [CRS]. The PDF text scrambled the row order. I assigned this row as the index because its USD row differs by the rupee's fall in 2018 and 2022 — [CRISIL report](https://www.dspim.com/media/pages/documents/31db646381-1757771758/benchmark-report_dsp-india-enhanced-equity-fund_cat-iii_sep-2024.pdf).
- Average rolling returns for that index: 3Y 12.22%, 5Y 11.05%, 7Y 12.10%. Worst 3Y rolling -2.97% at 31 Mar 2020. The report says historical rolling values are revised as coverage changes [CRS] — [CRISIL report](https://www.dspim.com/media/pages/documents/31db646381-1757771758/benchmark-report_dsp-india-enhanced-equity-fund_cat-iii_sep-2024.pdf).

**PMS Bazaar monthly tracker of long-short and long-only funds**

| Period | Long-short avg (n) | Long-only avg (n) | Nifty 50 TRI | BSE 500 TRI | Source |
|---|---|---|---|---|---|
| CY2023 | 19.21% | 37.92% | 21.30% | 26.55% | [PMS Bazaar](https://pmsbazaar.com/Blogs/AIFs-in-2024-Decoding-Resilience-Returns-and-Art-of-Alpha-Creation) |
| CY2024 | 13.72% (29) | 26.18% (68) | 10.09% | 15.81% | [PMS Bazaar](https://pmsbazaar.com/Blogs/AIFs-in-2024-Decoding-Resilience-Returns-and-Art-of-Alpha-Creation) |
| FY2024 | 21.68% | about 50% | n/a | n/a | [PMS Bazaar](https://pmsbazaar.com/Blogs/From-Highs-to-Headwinds-How-AIFs-Performed-in-FY2025) |
| FY2025 | 8.68% (30-odd) | 8.7% (70+) | 6.65% | 5.96% | [PMS Bazaar](https://pmsbazaar.com/Blogs/From-Highs-to-Headwinds-How-AIFs-Performed-in-FY2025) |

- All rows are [AGG]. "All the Returns mentioned in this blog are provided by the respective asset management companies" and may differ by reporting basis — [PMS Bazaar](https://pmsbazaar.com/Blogs/AIFs-in-2024-Decoding-Resilience-Returns-and-Art-of-Alpha-Creation). Footnote bases include post-expenses and tax, post-expenses pre-tax, gross, and pre-performance-fee — [PMS Bazaar](https://pmsbazaar.com/Blogs/From-Highs-to-Headwinds-How-AIFs-Performed-in-FY2025).
- FY25 long-short dispersion: top quartile 16.26%, bottom quartile 0.8% [AGG] — [PMS Bazaar](https://pmsbazaar.com/Blogs/From-Highs-to-Headwinds-How-AIFs-Performed-in-FY2025).
- FY25 named long-short results: Swyom India Alpha 31.79%; ASK Absolute Return and InQuant Debt Plus above 14%; True Beacon 1, Neo Treasury Plus and Nuvama EDGE 13-14% [AGG] — [PMS Bazaar](https://pmsbazaar.com/Blogs/From-Highs-to-Headwinds-How-AIFs-Performed-in-FY2025).
- FY25 long-only top: A9 Finsight Finavenue 46.66%, Negen Undiscovered Value 33.61%, 360 ONE High Growth Companies 23.48%, 360 ONE High Conviction Series 1 20.86% [AGG] — [PMS Bazaar](https://pmsbazaar.com/Blogs/From-Highs-to-Headwinds-How-AIFs-Performed-in-FY2025).
- 2025 monthly long-short averages: Jan -1.45% — [PMS Bazaar](https://pmsbazaar.com/Blogs/Jan-2025-AIF-Performance-Market-Pressures-Weigh-on-Returns-Long-Short-Funds-Outshine); Feb -3.09% — [PMS Bazaar](https://pmsbazaar.com/Blogs/Feb-2025-AIF-Performance-Defensive-Long-Short-Strategies-Soften-Market-Blow); Mar 2.98% (35 schemes) — [PMS Bazaar](https://pmsbazaar.com/Blogs/AIFs-Delivered-a-Strong-Comeback-in-March-2025-as-Markets-Rebound); Sep 0.94% — [PMS Bazaar](https://pmsbazaar.com/Blogs/Long-Short-AIFs-Outperform-Again-Even-as-Markets-Rebound-in-September).
- Helios India Long Short runs 30-90% net and 75-175% gross exposure [AGG] — [PMS Bazaar](https://pmsbazaar.com/Blogs/AIFs-in-2024-Decoding-Resilience-Returns-and-Art-of-Alpha-Creation).

### Inferences
- FY25 was a tie between long-short and long-only, 8.68% against 8.7%. In the CY2023-24 bull, long-short lagged long-only by 12-19 pts. Reported long-short returns did not behave like uncorrelated alpha at category level on these figures.
- With a 15-pt top-to-bottom-quartile spread in FY25 and mixed fee bases, category averages are soft. A top-quartile long-short fund at about 16% is far below 35%.
- CRISIL Cat III over CY15-CY23: CAGR about 12.0%, annual standard deviation about 12.4%, Sharpe about 0.5 at a 6.5% risk-free rate (my calc from nine annual points, pre-carry and pre-tax, so a live net Sharpe is lower). Worst calendar year -5.6%.
- The scheme count falls from 208 at 1Y to 15 at 7Y. Most Cat III funds have no record through a full cycle, and those 15 are the ones still reporting. A 35% return over 7 years would be more than double the top-quartile threshold of 14.48%.
- The 5Y top quartile of 25.26% to Sep 2024 starts in Sep 2019 and skips 2018. It falls to 14.48% at 7Y, a window that includes 2018 and is the better guide, though the 7Y sample is only 15 schemes.

### Gaps
- No published Sharpe, volatility or drawdown for long-short specifically. A Geojit note on long-short Sharpe could not be opened (host unreachable).
- The CRISIL report gives no long-short sub-index. Eureka Hedge and PMS AIF World category data not obtained.
- The CRISIL data end at Sep 2024, before the 2024-26 drawdown. CY2025 full-year and 2026 YTD long-short averages, and Avendus and Edelweiss Alternative records, not found.

## Long-horizon benchmarks and Buffett-style Indian records

### Takeaway
The best long-run rule-based reference, Nifty200 Momentum 30 TRI, shows 18.8% since Apr 2005, but only about six of those 21 years are live, and its live 5-year edge over Nifty 200 TRI is 0.8 pts against 4.9 pts since base. Broad Indian equity has returned low-to-mid teens, and the top long-lived active funds 13-15% over 20 years.

### Cited Findings

**Trailing CAGR at 30 Jun 2026, before costs [IDX via AMC]** — [HDFC MF deck, Jul 2026](https://files.hdfcfund.com/s3fs-public/Others/2026-07/HDFC%20NIFTY200%20Momentum%2030%20Index%20Fund_PPT%20%28July%202026%29.pdf)

| Horizon | Nifty200 Momentum 30 TRI | Nifty 200 TRI | Nifty 50 TRI |
|---|---|---|---|
| 1Y | -2.5% | -2.2% | -5.4% |
| 3Y | 14.1% | 12.1% | 8.8% |
| 5Y | 12.6% | 11.8% | 10.0% |
| 7Y | 16.6% | 13.6% | 11.9% |
| 10Y | 17.8% | 13.5% | 12.5% |
| 15Y | 18.1% | 12.3% | 11.4% |
| Since 1 Apr 2005 | 18.8% | 13.9% | 13.6% |

**Financial-year returns, Momentum 30 TRI / Nifty 200 TRI / Nifty 50 TRI [IDX via AMC]** — [HDFC MF deck](https://files.hdfcfund.com/s3fs-public/Others/2026-07/HDFC%20NIFTY200%20Momentum%2030%20Index%20Fund_PPT%20%28July%202026%29.pdf)
- FY06 79.7/63.7/67.4; FY07 4.7/11.0/14.3; FY08 40.9/23.5/25.1; FY09 -34.9/-38.5/-35.4; FY10 62.9/86.4/75.3; FY11 14.1/9.5/12.4; FY12 -0.3/-7.9/-8.2
- FY13 16.4/7.6/8.7; FY14 21.4/19.3/19.5; FY15 60.2/33.1/28.2; FY16 -4.4/-6.9/-7.8; FY17 40.2/24.0/20.2; FY18 21.6/12.4/11.8; FY19 11.6/11.9/16.4
- FY20 -16.4/-26.1/-25.0; FY21 62.4/74.6/72.5; FY22 37.7/21.4/20.3; FY23 -9.2/-1.1/0.6; FY24 70.0/38.3/30.1; FY25 -7.6/6.4/6.7; FY26 -3.4/-2.8/-4.0
- FY27 to 30 Jun 2026: 14.4/10.9/7.4. Column order inferred from scrambled text; chain-linking all 22 momentum figures reproduces 18.8%, which confirms the 14.4.
- Momentum beat Nifty 200 TRI in 14 of 21 financial years and Nifty 50 TRI in 15 of 21.

**Rolling returns, daily roll, Apr 2005 to Jun 2026 [IDX via AMC]** — [HDFC MF deck](https://files.hdfcfund.com/s3fs-public/Others/2026-07/HDFC%20NIFTY200%20Momentum%2030%20Index%20Fund_PPT%20%28July%202026%29.pdf)

| Window | Momentum avg / SD / ratio | Nifty 200 TRI avg / SD / ratio | Nifty 50 TRI avg / SD / ratio |
|---|---|---|---|
| 1Y | 22.3% / 28.6% / 0.78 | 16.4% / 24.8% / 0.66 | 15.9% / 22.8% / 0.70 |
| 3Y | 18.2% / 8.7% / 2.10 | 12.8% / 7.1% / 1.79 | 12.5% / 6.5% / 1.91 |
| 5Y | 18.6% / 6.3% / 2.94 | 12.9% / 5.1% / 2.52 | 12.6% / 4.7% / 2.68 |
| 10Y | 19.1% / 2.9% / 6.57 | 12.6% / 2.6% / 4.88 | 12.2% / 2.4% / 5.09 |

Ratio is average rolling return divided by SD of rolling returns, with no risk-free deduction, so it is not a Sharpe ratio.

**Other benchmarks**
- Methodology: top 30 of Nifty 200 F&O stocks by volatility-adjusted 6 and 12-month momentum, rebalanced each Jun and Dec, stock cap the lower of 5% or 5x free-float weight [IDX via AMC] — [HDFC MF deck](https://files.hdfcfund.com/s3fs-public/Others/2026-07/HDFC%20NIFTY200%20Momentum%2030%20Index%20Fund_PPT%20%28July%202026%29.pdf).
- An independent review states that as a new index "its entire history is based on backtested data", as of its 2020 launch, and that the momentum investor suffers more in bearish phases [PRS] — [Freefincal](https://freefincal.com/nifty200-momentum-30-index-a-new-strategy-index-from-the-nse/).
- NSE Indices white paper: 19.19% against 15.08% for Nifty 200, Apr 2005 to 27 Feb 2026; index launched 25 Aug 2020, base 1 Apr 2005, so pre-launch history is back-tested [PRS citing NSE] — [Dealplexus](https://www.dealplexus.com/blog/momentum-funds-india).
- 20-year CAGRs: Nifty 50 TRI about 11.4% (about 8.7x), Nifty Midcap 150 TRI 14.6%, Nifty Smallcap 250 TRI 12.7%, gold in INR 14.6%; as-of date not stated [PRS citing FundsIndia, May 2026] — [Business Today](https://www.businesstoday.in/mutual-funds/story/can-indian-equities-and-mutual-funds-still-create-long-term-wealth-for-investors-531157-2026-05-13).
- Nippon India Growth Mid Cap, HDFC Flexi Cap, Franklin India Flexi Cap and Aditya Birla SL Flexi Cap: 13-15% annualised over 20 years [PRS] — [Business Today](https://www.businesstoday.in/mutual-funds/story/can-indian-equities-and-mutual-funds-still-create-long-term-wealth-for-investors-531157-2026-05-13).
- Nippon India Small Cap 10Y 21.13% and Quant Flexi Cap 10Y 19.54%, April 2026 [PRS] — [Angel One](https://www.angelone.in/news/mutual-funds/best-performing-equity-mutual-funds-for-april-2026-based-on-5-yr-cagr). A different date gives 21.89% for Nippon — [Jainam](https://www.jainam.in/blog/highest-return-mutual-funds-in-india/).
- SPIVA India (mid-2025 per press): 73% of large-cap funds and 82% of mid and small-cap funds underperformed over 10 years; 90% of large-cap over 5 years [PRS citing S&P] — [Angel One](https://www.angelone.in/news/mutual-funds/most-indian-equity-funds-underperform-s-p-indices-over-10-years).
- Past five years to Jun 2026: 8.1% compound in rupees, under 5% in USD [PRS citing Ambit] — [Business Today](https://www.businesstoday.in/latest/economy/story/heavy-selling-by-fiis-may-not-reverse-soon-despite-ceasefire-in-west-asia-experts-at-ambit-institutional-equities-539203-2026-06-25).

### Inferences
- Start date matters a lot: Nifty 50 TRI is 13.6% from Apr 2005 but about 11.4% over 20 years ending in spring 2026. A one-year shift in start moves the CAGR by about 2 pts.
- Top long-lived funds at 13-15% sit near the Midcap 150 TRI's 14.6%. These are the survivors, so the true average is lower.
- Momentum's 4.9-pt long-run edge is the largest rule-based premium found, but it has shrunk as the window moves into the live period: 5.8 pts at 15Y, 4.3 pts at 10Y, 3.0 at 7Y, 0.8 at 5Y (my calc). Part of that is the current drawdown, so it does not prove the premium has gone, but the live evidence is far thinner than the headline.
- Chain-linking the FY table (my calc): Apr 2005 to Mar 2020 about 17.3%; Apr 2021 to Jun 2026 about 15.9%. The second window starts seven months after launch, and the deck's own 5Y figure from Jun 2021 is 12.6%. So the live range is about 13-16% depending on start date.
- Momentum returned 35% or more in 8 of 21 financial years yet compounded only 18.8%. Single-year precedent for 35% is common. Sustained precedent is not.
- A rough Sharpe from the 1Y rolling figures at a 7% risk-free rate is about 0.5 for momentum and about 0.4 for Nifty 200 and Nifty 50 TRI (my calc). This uses rolling annual dispersion, not daily volatility.
- Momentum's max drawdown is at least 34.9%, the FY09 loss, so its Calmar since 2005 cannot exceed about 0.54 (my calc). Point-to-point FY data understate the true peak-to-trough.
- The two long-run figures cannot be reconciled: 19.19% and 15.08% to Feb 2026 against 18.8% and 13.9% to Jun 2026. The BSE 500 was about flat from March to June 2026, so timing does not explain 1.2 pts on Nifty 200. Use the HDFC deck, read directly, over the blog relay of the white paper.

### Gaps
- Nifty 500 TRI and Nifty Midcap 150 and Smallcap 250 TRI since-base figures, and a Momentum 30 since-launch (25 Aug 2020) figure, not obtained. niftyindices.com timed out.
- Momentum 30 published max drawdown, volatility and Sharpe not obtained. A blog relay put a historical drawdown near 68% on one measure, but I could not confirm it, so it is excluded from the tables.
- Rakesh Jhunjhunwala, Raamdeo Agrawal, Porinju Veliyath, Vijay Kedia, Mukul Agrawal: no verifiable-holdings work done before the search budget ran out. No figure is given here rather than one from memory.
- Top-decile 10, 15 and 20-year MF distribution not obtained; only named-fund press figures.

## SEBI studies of individual F&O traders, and Jane Street

### Takeaway
About 9 in 10 individual F&O traders lose every year, and the loss has been running at Rs 75,000-1,12,000 crore a year. Gross profits go to proprietary firms and FPIs, about 96-99% of it to algo entities.

### Cited Findings
| Study | Period | Share losing | Total net loss | Avg loss | Grade | Source |
|---|---|---|---|---|---|---|
| Jan 2023 | FY22 | 89% (90% of active) | n/a | Rs 1.1 lakh (Rs 1.25 lakh active) | REG via press | [Business Standard](https://www.business-standard.com/amp/article/markets/sebi-study-suggests-89-retail-traders-in-equity-f-o-suffered-losses-123012501466_1.html), [Moneylife](https://www.moneylife.in/article/90-percentage-individual-traders-in-equity-fo-segment-incurred-losses-sebi-research/69634.html) |
| Sep 2024 | FY22-FY24 | 92.8% | Rs 1.81 lakh cr, 1.13 cr traders | about Rs 2 lakh | REG via TaxGuru | [TaxGuru](https://taxguru.in/sebi/sebi-study-shows-93-percent-individual-fo-traders-incur-losses.html) |
| Sep 2024 | FY24 | 91.1% | about Rs 75,000 cr | Rs 1.20 lakh | REG via TaxGuru | [TaxGuru](https://taxguru.in/sebi/sebi-study-shows-93-percent-individual-fo-traders-incur-losses.html) |
| Jul 2025 | FY25 | 91% | Rs 1,05,603 cr, revised to Rs 1,11,788 cr | about Rs 1.1 lakh | REG via press | [Business Standard](https://www.business-standard.com/markets/news/net-losses-of-traders-in-fo-widens-in-fy25-sebi-study-125070701221_1.html), [Finnovate](https://www.finnovate.in/learn/blog/sebi-fno-trader-losses-fy26-individual-derivatives) |
| Aug 2026 | FY26 | 87.7% (82.1% before costs) | Rs 91,685 cr | Rs 1.17 lakh; losers Rs 1.47 lakh; winners +Rs 1.22 lakh | REG via press | [PolicyEdge](https://www.policyedge.in/p/nearly-9-in-10-individual-fo-traders-lost-money-as-participation-fell-18-sebi-studies), [Finnovate](https://www.finnovate.in/learn/blog/sebi-fno-trader-losses-fy26-individual-derivatives) |

- Only 7.2% of individuals made a profit over FY22-FY24, and only about 1% earned more than Rs 1 lakh after costs [REG via] — [TaxGuru](https://taxguru.in/sebi/sebi-study-shows-93-percent-individual-fo-traders-incur-losses.html).
- Top 3.5% of loss-makers (about 4 lakh) averaged Rs 28 lakh and were 55.3% of losses [REG via] — [TaxGuru](https://taxguru.in/sebi/sebi-study-shows-93-percent-individual-fo-traders-incur-losses.html).
- Of traders active in every year FY22-FY26, only 0.5% were profitable every year [REG via] — [PolicyEdge](https://www.policyedge.in/p/nearly-9-in-10-individual-fo-traders-lost-money-as-participation-fell-18-sebi-studies).
- Over 75% of those losing two straight years kept trading; only 8.3% of those profited in year three [REG via] — [TaxGuru](https://taxguru.in/sebi/sebi-study-shows-93-percent-individual-fo-traders-incur-losses.html).
- FY26 individual transaction costs about Rs 24,859 cr, of which STT Rs 6,645 cr. Costs turned about 4.4 lakh pre-cost winners into losers. Losing predominant option sellers averaged about Rs 51.7 lakh. Options were 92% of aggregate individual losses [REG via] — [PolicyEdge](https://www.policyedge.in/p/nearly-9-in-10-individual-fo-traders-lost-money-as-participation-fell-18-sebi-studies).
- Active individual traders fell 18%, 106.2 lakh to 87.5 lakh, FY25 to FY26 [REG via] — [PolicyEdge](https://www.policyedge.in/p/nearly-9-in-10-individual-fo-traders-lost-money-as-participation-fell-18-sebi-studies).

**Who earns it, gross before costs**
| Year | Proprietary | FPI | Others | Individuals | Algo share of winners' profit |
|---|---|---|---|---|---|
| FY24 | +Rs 33,000 cr | +Rs 28,000 cr | n/a | individuals and others -Rs 61,000 cr | 96% prop, 97% FPI — [TaxGuru](https://taxguru.in/sebi/sebi-study-shows-93-percent-individual-fo-traders-incur-losses.html) |
| FY26 | +Rs 44,000 cr | +Rs 14,000 cr | corporates +8,000; MFs +3,000; partnerships and LLPs +3,000 | -Rs 72,000 cr | about 99% — [Finnovate](https://www.finnovate.in/learn/blog/sebi-fno-trader-losses-fy26-individual-derivatives) |

- FY24 algo users: 306 of 376 FPIs and 347 of 626 prop traders; about 13% of individuals, which SEBI says is overstated [REG via] — [TaxGuru](https://taxguru.in/sebi/sebi-study-shows-93-percent-individual-fo-traders-incur-losses.html).

**Jane Street (interim, ex parte, disputed)**
- SEBI's interim order of 3 Jul 2025 barred the group and impounded Rs 4,843.57 cr, deposited in escrow on 14 Jul 2025. It covers 18 expiry days between Jan 2023 and Mar 2025 [REG via academic blog] — [Oxford Business Law Blog](https://blogs.law.ox.ac.uk/oblb/blog-post/2025/07/jane-street-and-expiry-day-trap-unpacking-sebis-crackdown-algorithmic).
- Law-firm summary of the order, 1 Jan 2023 to 31 May 2025: index options +Rs 43,289 cr, stock futures -Rs 7,208 cr, index futures -Rs 191 cr, cash -Rs 288 cr, stated net +Rs 36,502 cr [REG via law firm] — [Legal 500](https://www.legal500.com/intelligence/india/corporate-commercial-law/sebi-update-interim-order-against-jane-street-group-for-alleged-index-manipulation). Those components sum to Rs 35,602 cr, Rs 900 cr short of the stated net, and I could not resolve this.
- A press breakdown for Jan 2023 to Mar 2025 gives net Rs 36,671 cr with Rs 44,358 cr in index options, which does sum [REG via press, search summary] — [Moneylife](https://moneylife.in/article/jane-street-accused-of-massively-rigging-indias-derivatives-market-sebi-orders-ban-escrow-of-rs4843-crore-illegal-gains/77585.html). Use "about Rs 36,500 cr net, about Rs 43,000-44,000 cr in index options".
- The Rs 4,843.57 cr covers 18 days: 15 Bank Nifty and 3 Nifty expiries. One day, 17 Jan 2024, made Rs 734.93 cr. The entities are JSI Investments, JSI2 Investments, Jane Street Singapore and Jane Street Asia Trading, treated as one group. The bar lapses once the amount is complied with [REG via law firm] — [Legal 500](https://www.legal500.com/intelligence/india/corporate-commercial-law/sebi-update-interim-order-against-jane-street-group-for-alleged-index-manipulation).
- Pattern alleged: buy Bank Nifty stocks and futures 9:15-11:46 while building short index option positions, then reverse from 11:49 to close; its trades often exceeded 20% of market-wide traded value in several scrips [REG via] — [Oxford Business Law Blog](https://blogs.law.ox.ac.uk/oblb/blog-post/2025/07/jane-street-and-expiry-day-trap-unpacking-sebis-crackdown-algorithmic).
- Status: SAT reserved its order on document disclosure on 21 Oct 2026. That is not a ruling on the merits, and Jane Street disputes the findings [PRS] — [Newsbytes](https://www.newsbytesapp.com/news/business/sat-reserves-order-in-jane-street-document-disclosure-case/story), [Lapaas Voice](https://lapaasvoice.com/sat-reserves-order-in-jane-street-case-as-firm-seeks-more-sebi-documents/).

### Inferences
- The FY26 gross table nets to zero: 44+14+8+3+3 = +72 against individuals' -72. So the individual loss is close to the transfer to these groups before costs.
- One group's index-options profit of about Rs 43,000-44,000 cr over about 29 months is comparable to the entire prop segment's FY26 annual gross profit. On SEBI's numbers, F&O profit is highly concentrated, algo-driven and intraday or expiry-day. That edge is not available to a swing system and sits outside the golden target's scope.
- The impounded amount is about 13% of the stated net profit (my calc). SEBI's manipulation allegation covers 18 days, not the whole profit, so most of the group's profit is outside the alleged conduct.
- FY26's fall in the loss share to 87.7% follows the Nov 2024 curbs, but average loss per trader rose. Curbs reduced participation more than they improved outcomes.

### Gaps
- SEBI's study PDFs were not read directly, only through relays. FY25 prop and FPI gross figures not captured.
- SEBI's order itself was not read, and no final order exists; the case remains unadjudicated.

## Retail algo: verified results, and what a self-built Kite system must comply with

### Takeaway
I found no independently verified live return for any retail algo platform. A self-built Kite swing system must now use a registered static IP, stay at or below 10 orders per second, set market protection on market orders, and log out API sessions daily. It needs no exchange algo registration below 10 orders per second.

### Cited Findings

**Results**
- No independent live-returns data was found for Streak, Tradetron, AlgoTest, Quantiply or uTrade. What surfaced was vendor marketing, for example [AlgoTest](https://algotest.in/blog/streak-vs-tradetron) [MGR].
- Over 100 brokers reportedly received SEBI warnings in Oct 2024 for allowing APIs of TradeTron, alleged to offer assured returns [PRS] — [Business Standard](https://www.business-standard.com/amp/markets/news/sebi-scrutinises-brokers-linked-to-algo-trading-with-guaranteed-returns-124100901218_1.html). Seen via search summary; the page returned 403.
- PaRRVA, the verifier of past returns: CARE Ratings as agency and NSE as data centre, circular of 29 Apr 2026, operational 4 May 2026. It covers IAs, RAs and algo providers [REG via press] — [Taxmann](https://www.taxmann.com/post/blog/sebi-launches-parrva-for-verified-performance-in-ads/). Verification applies prospectively only from the date of opting in [REG via press, search summary] — [Outlook Money](https://www.outlookmoney.com/news/sebi-operationalises-parrva-framework-sets-deadline-for-advisers-to-verify-data-on-past-performance).

**Timeline**
- SEBI circular SEBI/HO/MIRSD/MIRSD-PoD/P/CIR/2025/132 of 30 Sep 2025, read directly: the 4 Feb 2025 circular was due 1 Aug 2025, moved to 1 Oct 2025 by circular of 29 Jul 2025. Exchanges issued operational modalities on 22 Jul 2025. Broker milestones: apply by 31 Oct 2025, register by 30 Nov 2025, mock session by 3 Jan 2026. Laggards were barred from onboarding new retail API clients from 5 Jan 2026. From 1 Apr 2026 the framework applies to all stock brokers [REG] — [SEBI](https://www.sebi.gov.in/sebi_data/attachdocs/sep-2025/1759232056254.pdf).
- NSE implementation standards were issued 5 May 2025 (NSE/INVG/67858) [REG via search summary] — [Outlook Business](https://www.outlookbusiness.com/markets/sebi-extends-deadline-to-implement-retail-algo-trading-by-april-2026). I found no extension after 1 Apr 2026, and Zerodha's page gives 1 Apr 2026 as effective.

**Compliance list for a self-built Kite system**
| Rule | Detail | Grade | Source |
|---|---|---|---|
| Static IP | Required for API order placement from 1 Apr 2026, regardless of order frequency. Unregistered-IP orders are rejected | Broker | [Zerodha](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/static-ip), [Kite forum](https://kite.trade/forum/discussion/comment/52171) |
| IP count and change | Primary required, secondary optional; one modification per calendar week | Broker | [Zerodha](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/static-ip) |
| Read-only calls | WebSocket, order book and positions work from any IP | Broker | [Zerodha](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/static-ip) |
| Scope of IP | Applies across all apps on one Kite Connect account. Family sharing: spouse, dependent children, dependent parents, with declaration | Broker | [Kite forum](https://kite.trade/forum/discussion/comment/52171) |
| Network | IPv4 or IPv6 must match the real outgoing IP. Localhost is not valid. Backup connection advised | Broker | [Kite forum](https://kite.trade/forum/discussion/comment/52171), [Zerodha](https://zerodha.com/z-connect/general/a-comprehensive-overview-of-nses-circular-on-the-new-retail-algo-trading-framework) |
| Order rate | 10 orders per second per Kite client ID, shared across apps; excess gets HTTP 429. NSE wording is per exchange or segment | Broker, REG via | [Kite forum](https://kite.trade/forum/discussion/comment/52171), [Groww](https://groww.in/blog/nse-issues-new-compliance-norms-for-retail-algo-trading) |
| Registration | None at or below 10 OPS. Above it, register each algo with each exchange through the broker; logic changes need re-approval | REG via | [Groww](https://groww.in/blog/nse-issues-new-compliance-norms-for-retail-algo-trading), [Finseclaw](https://www.finseclaw.com/article/nse-releases-implementation-standards-for-retail-participation-in-algorithmic-trading) |
| Tagging | Below threshold, orders are tagged algo with a generic exchange ID | REG via | [Groww](https://groww.in/blog/nse-issues-new-compliance-norms-for-retail-algo-trading) |
| API keys | Unregistered algos run through one designated key | REG via | [Groww](https://groww.in/blog/nse-issues-new-compliance-norms-for-retail-algo-trading), [Zerodha](https://zerodha.com/z-connect/general/a-comprehensive-overview-of-nses-circular-on-the-new-retail-algo-trading-framework) |
| Daily logout | All API sessions must be logged out daily | REG via | [Groww](https://groww.in/blog/nse-issues-new-compliance-norms-for-retail-algo-trading) |
| Market orders | Non-zero market protection required on MARKET and SL-M; "0" is rejected; -1 uses Zerodha's automatic limits; also needed when modifying to market | Broker | [Kite forum](https://kite.trade/forum/discussion/comment/52171) |
| Slicing | Cap at 10 slices | Broker | [Kite forum](https://kite.trade/forum/discussion/comment/52171) |
| Daily order cap | 3,000 orders a day is an internal Zerodha limit, not an exchange rule (staff, May 2025) | Broker | [Zerodha](https://zerodha.com/z-connect/general/a-comprehensive-overview-of-nses-circular-on-the-new-retail-algo-trading-framework) |

**Conflicts to note**
- Angel One says market and IOC orders are banned for algos, while Zerodha accepts market orders with protection. This may be broker-specific — [Angel One](https://www.angelone.in/news/market-updates/what-s-changing-in-angel-one-s-smartapi-access-from-april-1-2026).
- A FlyTradr post says retail algos must be hosted in India and is self-flagged as unverified — [dev.to](https://dev.to/reyazk08/sebis-new-algo-trading-rules-for-retail-traders-what-changed-in-april-2026-4k04). Zerodha staff said the IP need not be from India, in a May 2025 comment — [Zerodha](https://zerodha.com/z-connect/general/a-comprehensive-overview-of-nses-circular-on-the-new-retail-algo-trading-framework).

**Data and paper-trading note**
- A Kite forum reply of 9 Oct (year not shown; user not labelled as staff) says Kite data may be used for personal analysis, while permanent storage and paper trading on live data are restricted. The sandbox is unavailable and back-office APIs are not provided, so reconciliation should use Console reports — [Kite forum](https://kite.trade/forum/discussion/comment/53060/).

### Inferences
- A swing or positional book is far below 10 OPS, so registration is not the issue. The practical items are a static IP on the order path, market protection or limit orders, and daily logout.
- PaRRVA starts in 2026 and is prospective, so no verified historical retail algo record can exist yet.
- The forum note on paper trading on live Kite data bears directly on Centurion's Kite paper run and should be checked with Zerodha directly.

### Gaps
- The primary text of the 4 Feb 2025 SEBI circular and the NSE standards was not read; those rules come through broker and law-firm relays. Only the 30 Sep 2025 timeline circular was read directly. Daily-logout mechanics on Kite, and whether Kite's token expiry satisfies them, are unconfirmed.
- The SEBI order on TradeTron dated 25 Mar 2026 (Order/JS/YK/2025-26/32267) surfaced only in a secondary source and is unverified.

## Who has compounded above 30% for 10+ years in India, verifiably

### Takeaway
No one, in the sources reached. The highest 10-year PMS figure is Aequitas at 26.43%, and it has inconsistent data. No record above 30% over 10+ years was verifiable.

### Cited Findings
- In a table sourced to APMI data at 30 Sep 2026, no strategy reaches 30% over 10 years. Top: Aequitas 26.43%, Buoyant 19.97%, Sameeksha 19.92%, Solidarity Prudence 18.15%, Valentis 18.10%, 2Point2 17.80% [AGG, "not independently verified"] — [AltportFunds](https://www.altportfunds.com/evaluating-the-best-pms-in-india-2025-rankings-fees-performance-comparisons/).
- 30%+ since-inception claims: Aequitas 31.40%, Counter Cyclical Diversified Long Term Value 44.54%, Ameya 40.33%, BB Money Plant 42.33%. The last two have no stated inception or 5-year figure [AGG] — [AltportFunds](https://www.altportfunds.com/evaluating-the-best-pms-in-india-2025-rankings-fees-performance-comparisons/).
- 5-year CAGRs of 31-77% at 31 Jan 2026 for 15 strategies, with max drawdowns of -25% to -34%, include Alpha Portfolio at 77.21% on AUM of Rs 9.35 cr. Varanium strategies show 69-71% over 5 years on AUM under Rs 11 cr [AGG] — [AltportFunds](https://www.altportfunds.com/evaluating-the-best-pms-in-india-2025-rankings-fees-performance-comparisons/).
- Stallion Core: 28.31% since Oct 2018 [AGG] — [AltportFunds](https://www.altportfunds.com/evaluating-the-best-pms-in-india-2025-rankings-fees-performance-comparisons/).

### Inferences
- High figures cluster in very small AUM, short post-2020 windows, or both. They are not comparable to a 10+ year record through 2018 and 2022.
- The 30%+ records in this set are discretionary and concentrated, not rule-based. How they were earned (concentration, private markets, promoter access) could not be evidenced from the sources reached.
- Survivorship cannot be measured from a table of currently listed strategies, and closed strategies are absent by construction.

### Gaps
- Jhunjhunwala, Agrawal, Veliyath, Kedia, Mukul Agrawal, SageOne and others were not searched before the budget ran out.
- No closure or merger counts for PMS strategies were found.

## The 2024-26 Indian market regime

### Takeaway
Two legs: Sep 2024 to Mar 2025, then a West Asia war shock from 28 Feb 2026 with record FII selling. Momentum peaked in Sep 2024 and was still about 19% below peak two years later.

### Cited Findings
- Sep 2024 peaks: Nifty 50 26,277.35, Nifty Midcap 100 60,925.95, Nifty Smallcap 100 19,716.20. By 21 Jan 2025 they were down 11.88%, 12.82% and 12.90% [PRS, broker blog] — [IIFL](https://www.indiainfoline.com/blog/how-sharp-has-this-stock-market-correction-been).
- By 4 Mar 2025, Nifty 50, Midcap and Smallcap were down about 16%, 21% and 24% from peak [PRS, broker note] — [Motilal Oswal](https://www.motilaloswal.com/site/rreports/html/638773649960028976/index.htm).
- 2026 leg: from a January high of 26,373.85 the Nifty fell about 13.4% to the 22,450 zone by 30 Mar 2026. West Asia conflict began 28 Feb 2026. March FII outflow was a record Rs 1.14 lakh cr, beating the prior record of Rs 94,017 cr, while DIIs bought about Rs 1.28 lakh cr. Brent reached $115.53 and India VIX went from 14.20 to 28.81 [PRS] — [Republic](https://www.republicworld.com/business/nifty-underperforms-global-peers-with-2800-point-quarterly-slide-as-west-asia-war-triggers-record-fii-exodus).
- 29 Sep 2026: six-month low 22,569 against the 6 Apr low of 22,542; Nifty down 13.62% YTD; FII YTD selling above Rs 2 lakh cr; crude above $107 [PRS] — [Business Today](https://businesstoday.in/markets/stocks/story/nifty-trading-below-23000-mark-for-second-day-more-correction-likely-558454-2026-09-29).
- FPIs sold $18.9bn in all of 2025 and $29.8bn in 2026 to 24 Jun, per NSDL [PRS] — [Business Today](https://www.businesstoday.in/latest/economy/story/heavy-selling-by-fiis-may-not-reverse-soon-despite-ceasefire-in-west-asia-experts-at-ambit-institutional-equities-539203-2026-06-25).
- Centurion's data, supplied in the brief: Nifty 50 closed 23,053 on 30 Oct 2026, 12.4% below the 2 Jan 2026 close of 26,329. The arithmetic checks.
- Nifty200 Momentum 30: all-time high 38,115.25 on 27 Sep 2024; 30,718.25 on 4 Sep 2026, about 19% below peak [PRS citing CEIC] — [Dealplexus](https://www.dealplexus.com/blog/momentum-funds-india).
- Momentum TRI FY24 +70.0%, FY25 -7.6%, FY26 -3.4%, FY27 to Jun 30 +14.4%; Nifty 200 TRI FY24 +38.3%, FY25 +6.4%, FY26 -2.8%, then +10.9% [IDX via AMC] — [HDFC MF deck](https://files.hdfcfund.com/s3fs-public/Others/2026-07/HDFC%20NIFTY200%20Momentum%2030%20Index%20Fund_PPT%20%28July%202026%29.pdf).
- Momentum TRI 1Y to 30 Sep 2026 -0.47%, against Nifty 50 TRI -7.09% [MF] — [ICICI Prudential factsheet](https://www.icicipruamc.com/blob/knowledgecentre/factsheet-schemes/Passive%20Schemes/ETFs%20Schemes/ICICI%20Prudential%20Nifty%20200%20Momentum%2030%20ETF.pdf).
- Index sector mix swung from about 48% financials in Dec 2025 to about 21% at Jun 2026 after 22 replacements [IDX via AMC] — [HDFC MF](https://www.hdfcfund.com/learn/macros-markets-more/index-insider/nifty200-momentum-30-index-june-2026-portfolio-review-capitalizing-ongoing-market-trends).
- ICICI Prudential momentum index fund: 1Y 5.63%, 6M 0.09%, 3Y absolute 41.34% to 1 Sep 2026 [PRS citing Paytm Money] — [Dealplexus](https://www.dealplexus.com/blog/momentum-funds-india).
- June 2026 press called it "one of their deepest drawdowns in over two decades", with no percentage [PRS] — [Whalesbook](https://www.whalesbook.com/news/English/stock-investment-ideas/Momentum-Investing-Hits-a-Rough-Patch-What-Investors-Should-Know/6a28980647791627fe2f6e18).
- Quant PMS in this regime: Capitalmind Adaptive Momentum 2Y -5.2% a year to 29 Oct 2026 [MGR] — [Capitalmind](https://www.capitalmind.in/portfolio-management-service/momentum).
- BSE 500 TRI monthly in 2026: Jan -3.34%, Feb 0.45%, Mar -11.37%, Apr 10.38%, May -0.17%, Jun 1.73%, Jul 2.19%, Aug -0.09%, Sep -5.57% [AGG] — [Sharescart](https://www.sharescart.com/pms/buoyant-capital-private-limited/buoyant-opportunities-pms/).

### Inferences
- Give-back (my calc from the FY table): FY25 and FY26 together were -10.7% for momentum and +3.4% for Nifty 200 TRI. From end-Mar 2024 to 30 Jun 2026, momentum TRI rose about 2.1%, Nifty 200 TRI about 14.7% and Nifty 50 TRI about 10.0%.
- In relative terms momentum led Nifty 200 TRI by about 23% in FY24 and then lagged by about 11% over the next 27 months, so it gave back roughly 55-60% of that lead. In absolute terms the +70% was mostly kept on a TRI basis at June 2026.
- An investor who joined at the Sep 2024 peak was still about 19% down on price two years later. This is the first long drawdown of the factor since the index went live in Aug 2020 that the FY table shows as lagging the market in consecutive years (FY25 and, narrowly, FY26 against Nifty 200 TRI).
- DIIs bought more than FIIs sold in March 2026, Rs 1.28 lakh cr against Rs 1.14 lakh cr.

### Gaps
- Momentum 30 trough date and level after Sep 2024, the true peak-to-trough, and calendar-year figures. The HDFC deck gives financial years only.
- Nifty Smallcap 250 and Midcap 150 peak-to-trough for 2024-26. Only Motilal's unnamed small-cap index (-24% at Mar 2025) and Smallcap 100 to Jan 2025 were found.
- Quant PMS returns specifically across Sep 2024 to Mar 2025 other than the Capitalmind figures.

## Highest verifiable rule-based Sharpe and CAGR, and the golden-target precedent

### Takeaway
No verified live precedent for the golden target was found in India. The best long-run rule-based CAGR is 18.8% (mostly back-tested, before costs), about half of 35%, with a Sharpe near 0.5 and a Calmar of at most about 0.54. No published live Sharpe near 1.4 was found for any Indian strategy.

### Cited Findings

| Measure | Highest found | Evidence | Golden target | Source |
|---|---|---|---|---|
| Long-run rule-based CAGR | 18.8%, Nifty200 Momentum 30 TRI, Apr 2005-Jun 2026 | IDX via AMC; back-tested before Aug 2020; before costs | 35% | [HDFC MF deck](https://files.hdfcfund.com/s3fs-public/Others/2026-07/HDFC%20NIFTY200%20Momentum%2030%20Index%20Fund_PPT%20%28July%202026%29.pdf) |
| Live net rule-based CAGR, 5+ years | about 17%, Capitalmind Adaptive Momentum, since Mar 2019 (7Y 17.6%, 5Y 7.1%) | MGR and AGG; "not verified by SEBI" | 35% | [Capitalmind](https://www.capitalmind.in/portfolio-management-service/momentum), [PMS AIF World](https://www.pmsaifworld.com/portfolio/wizemarkets-capitalmind-momentum-fund) |
| Live investable rule-based tracker | 12.62%, ICICI Momentum 30 ETF, Aug 2022-Sep 2026 | MF | 35% | [ICICI Prudential factsheet](https://www.icicipruamc.com/blob/knowledgecentre/factsheet-schemes/Passive%20Schemes/ETFs%20Schemes/ICICI%20Prudential%20Nifty%20200%20Momentum%2030%20ETF.pdf) |
| 10-year PMS, any style | 26.43%, Aequitas, conflicting data | AGG, "not independently verified" | 35% | [AltportFunds](https://www.altportfunds.com/evaluating-the-best-pms-in-india-2025-rankings-fees-performance-comparisons/) |
| 7-year Cat III top-quartile threshold | 14.48% at Sep 2024, pre-carry and pre-tax | CRS | 35% | [CRISIL report](https://www.dspim.com/media/pages/documents/31db646381-1757771758/benchmark-report_dsp-india-enhanced-equity-fund_cat-iii_sep-2024.pdf) |
| Sharpe, any live strategy | none published in the sources reached | gap | 1.4 | n/a |

### Inferences
- **Sharpe:** about 0.5 for momentum and about 0.5 for the CRISIL Cat III index (my calcs, rough). The golden target asks for nearly three times that, sustained.
- **CAGR:** 35% sits about 2.6 SD above momentum's average 5-year rolling return (18.6%, SD 6.3%) and about 5.5 SD above its average 10-year rolling return (19.1%, SD 2.9%) (my calc). Rolling windows overlap, so these are distances and not probabilities.
- **Calmar:** momentum's is at most about 0.54. Calmar >= 1 at 35% CAGR allows a maximum drawdown of 35%, which that index's FY09 alone reached at 34.9%.
- **Volatility implied:** Sharpe 1.4 at 35% CAGR implies annual volatility of about 20-22% at a 6.5-7% risk-free rate (my calc). So the target needs index-like volatility with roughly double the best long-run rule-based return.
- **Scope of the claim:** "no precedent" means none found in sources reachable here, not proven absence. Private prop books do not publish returns. Those with very high reported rupee profits (prop, FPI, Jane Street) work intraday or at expiry, so they are not precedents for a swing or positional book.
- The golden target is an extreme outlier against every live Indian reference found. If a Centurion backtest shows it, the Indian record suggests treating that as a result to disprove, rather than as an expected live outcome.

### Gaps
- No published live Sharpe for any Indian rule-based strategy was found, and the sources for named PMS Sharpe were garbled.
- Not searched, for lack of budget: the Buffett-style investors, SageOne and other long-record PMS, and quant mutual funds. A 30%+ 10-year record from these names could change the "highest 10-year" row, though not the rule-based rows. Follow-up searches would need a new session.
