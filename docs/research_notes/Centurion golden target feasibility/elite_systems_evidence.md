# Elite systematic trading: what the record supports for Sharpe >= 1.4, CAGR >= 35%, Calmar >= 1 at swing or positional horizon

**Bottom line.** I found no documented precedent for the joint target (Sharpe >= 1.4, CAGR >= 35%, Calmar >= 1, unlevered or modestly levered, no intraday, sustained over 10+ years). Medallion is the only record that clears the numbers, and it does so with short holds, reported double-digit leverage, a capacity cap and institutional financing. Its compounded figure is also disputed. The long-run live Sharpes I could source for positional systems sit around 0.4 to 0.8.

**Evidence grades used in this file**
- **[AUDITED-LIVE]**: real money in a regulated fund or listed company with audited accounts. I read factsheet or NAV data, not the audit reports themselves.
- **[REPORTED-LIVE]**: real money, but the figure comes from press, investors or third-party trackers, with no public audit. I added this fifth grade because the four requested grades would otherwise overstate or understate these figures.
- **[ACADEMIC-OOS]**: peer-reviewed or working-paper result using out-of-sample or live data.
- **[BACKTEST]**: simulated only.
- **[MARKETING]**: produced by a manager, distributor or adviser with a commercial interest.
- **[OPINION]**: practitioner judgement.
- **[MATH]**: my own hand arithmetic. No code was run.

**Research limits.** The web search budget ran out partway through, and several PDFs could not be read as text. Claims I could not source are in Gaps and marked as unverified recollection. Do not use them as findings.

**Baseline conventions for [MATH].** rf = 6.5% is my assumption for an INR cash rate. It is not a sourced figure. CAGR is approximated as g = rf + S x vol - vol^2 / 2.

---

## What is documented about Medallion, Jane Street, Two Sigma, DE Shaw and Citadel Wellington, and which figures are audited?

### Takeaway
No Medallion figure is publicly audited. The canonical numbers (66% gross, about 39% net) trace to Zuckerman's book, and a later paper argues the compounded gross return was probably under 35%. Its edge is built on short holds, many small bets and cheap leverage, and sibling funds at the same firm look ordinary. The other elite firms have only press-reported returns or firm-level accounts, so I found no sourced Sharpe for any of them.

### Cited Findings

**Medallion returns**
- [REPORTED-LIVE] 1988-2018: mean gross annual return 66.1% and net 39.2%, with standard deviation 31.7% gross and 20.3% net; gross CAGR 63.3%; no negative gross year. — [Cornell paper, search excerpt](https://www.scribd.com/document/470291821/SSRN-id3504766-pdf)
- [REPORTED-LIVE] Sources conflict on whether 39.2% is an arithmetic mean or a compound return. The excerpt above reads as a mean, while one summary calls it the compound return after fees. — [The7Circles summary](https://the7circles.uk/?p=35029)
- [REPORTED-LIVE] Cornell's performance data came from Zuckerman's book, plus results of two other Renaissance funds and comments from academics and practitioners. The article does not say the figures were audited. — [Institutional Investor](https://institutionalinvestor.com/article/b1k2fymby99nj0/Famed-Medallion-Fund-Stretches-Explanation-to-the-Limit-Professor-Claims)
- [REPORTED-LIVE] $100 at the start of 1988 grew to $398.7M gross by end-2018, against $1,910 for the CRSP index and $331,288 for perfect monthly market timing. — [Institutional Investor](https://institutionalinvestor.com/article/b1k2fymby99nj0/Famed-Medallion-Fund-Stretches-Explanation-to-the-Limit-Professor-Claims)
- [REPORTED-LIVE] Net annual returns were above 29% in every year from 2010 to 2018, and 2018 was +76%. — [Institutional Investor](https://institutionalinvestor.com/article/b1k2fymby99nj0/Famed-Medallion-Fund-Stretches-Explanation-to-the-Limit-Professor-Claims)
- [REPORTED-LIVE] The last net losing year was 1989 (-3.2%). Market beta was about -1.0, with SMB and HML loadings not significant. — [The7Circles summary of Cornell](https://the7circles.uk/?p=35029)

**The compounding dispute**
- [ACADEMIC-OOS, working paper] Guo and Liu argue that compounding yearly returns directly is wrong. Estimating from fund sizes and trading profits gives 31.8% compounded before fees. Using manager wealth as a proxy gives 25.6% over the 33 years to 2020. They conclude the compounded pre-fee return is "probably under 35%". — [arXiv 2405.10917](https://arxiv.org/abs/2405.10917)
- A search summary of the same paper gave a modified Sharpe of 1.40. The abstract page does not contain that figure and I could not read the PDF, so it stays unconfirmed. — [arXiv PDF](https://arxiv.org/pdf/2405.10917v2)

**Sharpe**
- [REPORTED-LIVE] Zuckerman's book gives a Sharpe of 2.5 for the "most recent five-year period" and says it "even hit 7.5" in 2004. — [Book highlights on Goodreads](https://www.goodreads.com/notes/46025922-the-man-who-solved-the-market/88044824-alok-kejriwal?page=5)
- [REPORTED-LIVE] Secondary summaries add about 2.0 for most of the 1990s and about 6.0 in early 2003. These came from search results and I did not trace each one to a page.
- [REPORTED-LIVE] A secondary summary of Cornell's figures reports a Sharpe "of more than 2". — [The7Circles](https://the7circles.uk/?p=35029)

**Mechanics**
- [REPORTED-LIVE] Trade accuracy, as attributed to Robert Mercer, is 50.75%. — [Institutional Investor](https://institutionalinvestor.com/article/b1k2fymby99nj0/Famed-Medallion-Fund-Stretches-Explanation-to-the-Limit-Professor-Claims)
- [REPORTED-LIVE] Zuckerman describes leverage as "ample and cheap" without giving a ratio in the sources I read. — [The7Circles](https://the7circles.uk/?p=35029)
- [REPORTED-LIVE] The book explains the short-horizon preference statistically: "A one-day trading signal can incorporate data points for every trading day of the year" while "a one-year signal depends on just one annual data point". "Simons's scientists were more confident about short-term signals". — [Book highlights on Goodreads](https://www.goodreads.com/notes/46025922-the-man-who-solved-the-market/88044824-alok-kejriwal?page=5)
- [REPORTED-LIVE] Size limit: "The size limit meant Medallion sometimes identified more market aberrations and phenomena than it could put to use". The fund sits at about $10bn, kept roughly constant by annual distributions. — [Goodreads](https://www.goodreads.com/notes/46025922-the-man-who-solved-the-market/88044824-alok-kejriwal?page=5); [Institutional Investor](https://institutionalinvestor.com/article/b1k2fymby99nj0/Famed-Medallion-Fund-Stretches-Explanation-to-the-Limit-Professor-Claims)
- [REPORTED-LIVE] Closed to outside capital since 1993 per Institutional Investor; another source says 1993 or 2003. — [Institutional Investor](https://institutionalinvestor.com/article/b1k2fymby99nj0/Famed-Medallion-Fund-Stretches-Explanation-to-the-Limit-Professor-Claims); [The7Circles](https://the7circles.uk/?p=35029)
- [REPORTED-LIVE] Fees at one stage were "20 percent of their gains and 5 percent of the money". — [Goodreads](https://www.goodreads.com/notes/46025922-the-man-who-solved-the-market/88044824-alok-kejriwal?page=5)
- [REPORTED-LIVE, unsourced estimates on a secondary page] "an average holding period of one to two days", "an estimated 150,000+ trades daily", a cap of "roughly $10 billion", and fees of "5% of assets and 44% of investment profits", with a 2002 timeline entry of 36% "(later 44%)". The page labels the trade count and holding period as estimates and names no source for them. — [Faster Than Normal](https://fasterthannormal.co/businesses/renaissance-technologies)
- [REPORTED-LIVE] A block quote with no explicit speaker says "moments to months is their holding period" and "On average, I would argue a day or two". The post is built around a Zuckerman conversation, but the attribution is not clear. — [Investorama](https://investorama.substack.com/p/the-investing-goat-a-conversation)
- [REPORTED-LIVE] Renaissance's own statement to the Senate subcommittee gave an average holding period of 455 days for the Deutsche Bank basket options, 2000-2009. That is the holding period of the tax wrapper, not of the underlying trades. — [PSI statement](https://www.hsgac.senate.gov/wp-content/uploads/imo/media/doc/STMT%20-%20Renaissance%20(July%2022%202014).pdf)

**The natural experiment: Renaissance's public funds**
- [REPORTED-LIVE] 2018: Medallion +76%, RIEF +8.5%, Diversified Alpha +3.2%. — [Institutional Investor](https://institutionalinvestor.com/article/b1k2fymby99nj0/Famed-Medallion-Fund-Stretches-Explanation-to-the-Limit-Professor-Claims)
- [REPORTED-LIVE] 2020 to about 17 April: Medallion about +24% (14 April, WSJ via II), against -7% to -9% for RIEF, RIDA and RIDGE. — [Institutional Investor](https://www.institutionalinvestor.com/article/b1l98yt4p0bvr4/The-Famed-Medallion-Fund-Is-Crushing-It-Other-RenTech-Funds-Not-So-Much)
- [REPORTED-LIVE] RIEF: -19.4% in 2020, +20.5% in 2021, +22.5% to October 2024, +34% in 2011. RIDA: -31.6% in 2020. RIEF's design is long-biased, beta 0.4 or lower, with expected average leverage of about 2.5 to 1. — [Institutional Investor](https://www.institutionalinvestor.com/article/2e0uykr3vn5booz0smrcw/hedge-funds/renaissances-2024-rebirth)
- [MARKETING, target only] RIEF aimed for 400-600 bps above the S&P 500, gross, over rolling 3- to 5-year periods. — [Institutional Investor](https://www.institutionalinvestor.com/article/b1l98yt4p0bvr4/The-Famed-Medallion-Fund-Is-Crushing-It-Other-RenTech-Funds-Not-So-Much)
- [REPORTED-LIVE] An investor letter from around end-2010 gave RIEF +4.55% total since its 2005 inception, at about 60% of S&P 500 volatility, against -5.34% for the index. — [Institutional Investor](https://www.institutionalinvestor.com/article/b150xvq1p8r7rh/simons-medallion-up-more-than-30-percent-in-2010)

**Other elite firms**
- [REPORTED-LIVE] Citadel Wellington: about 19.1% to 19.2% annualised net since 1990, from an investor via the FT and from people familiar with the firm. No source says it is audited and none gives a Sharpe. — [BusinessDay/FT](https://businessday.ng/?p=258813); [NBC Chicago](https://www.nbcchicago.com/news/business/money-report/billionaire-ken-griffins-hedge-funds-at-citadel-are-all-in-the-green-for-2025/3780143/)
- [REPORTED-LIVE] Citadel's main funds lost 55% in 2008, on leveraged convertible-bond exposure, then reportedly returned 62% the next year. Sources are secondary. — [Phoenix Learning](https://phoenixlearning.substack.com/p/citadels-rise-to-immortality); [IBJ](https://ibj.com/?p=6316)
- [REPORTED-LIVE] DE Shaw Composite: about 12.9% annualised net since 2001, about 18.5% in 2025 (Reuters syndication). — [Tiger Brokers/Reuters](https://www.itiger.com/news/2600609750)
- [REPORTED-LIVE] Millennium: about 14% average annual return since inception. The article is undated. — [Motley Fool](https://www.fool.com/investing/how-to-invest/famous-investors/millennium-management)
- [REPORTED-LIVE] Two Sigma: Absolute Return Enhanced +12% and Spectrum +8.6% in 2023. No since-inception figure found. — [Hedgeweek](https://hedgeweek.com/two-sigma-posts-solid-gains-despite-year-of-drama)
- [AUDITED firm accounts, via prospectus reporting; not a fund] Jane Street: about $20.5bn net trading revenue in 2024 and total assets of $140.2bn at end-2024. — [FI Desk](https://www.fi-desk.com/jane-street-issues-us1-35-bn-bond-as-market-makers-bulk-up-balance-sheets/)
- [REPORTED-LIVE] Jane Street Q2 2025: $10.1bn net trading revenue and $6.9bn net profit (one source). — [Wikipedia](https://en.wikipedia.org/wiki/Jane_Street_Capital)

### Inferences
- [MATH] Cornell's net mean and standard deviation imply a Sharpe of roughly 1.75 ((39.2 - 3.5) / 20.3), assuming about 3.5% average cash. That cash rate is my assumption, not a sourced figure.
- [MATH] At p = 0.5075 with symmetric payoffs, per-bet Sharpe is about 0.015. Sharpe 1.4 then needs about 8,700 independent bets a year and Sharpe 2.0 about 17,800. This is a structurally high-turnover edge that no swing book can copy.
- If Guo and Liu are right, the 63% compounded gross figure was never earned on a growing capital base, because profits were paid out each year. It also shows the edge did not scale: Medallion's percentage return applies to a fixed, capped base.
- RIEF, RIDA and RIDGE are the closest natural experiment for scaling and slowing a Medallion-class research team. They suggest the same people, once running large capital in a long-biased design, produce ordinary equity-like results with deep drawdowns. The sources do not document RIEF's holding period, so the link to horizon is inference. See Gaps.
- Wellington shows how the best multi-decade numbers can still carry a very deep loss: about 19% against -55% is a full-history Calmar of about 0.35.

### Gaps
- Leverage of 12.5x on average and up to 20x came from a secondary summary in search results. I could not tie it to a specific page, and the page I checked says only that leverage "has been substantial". Treat 12.5x as an unverified estimate, not a disclosed figure.
- Holding period is consistently reported as a day or two, but only through secondary or unattributed text. The book reportedly gives a range of one or two days to one or two weeks. I could not read the original.
- 26 to 39 million trades a year, "many held for a few seconds", came from a blog reprint of Senate material in search results. I could not confirm it from the PSI report itself.
- The 5-and-44 fee structure now has one secondary source (above) but no primary one.
- Cornell's own Sharpe and alpha t-statistics are unconfirmed, because the full paper was not readable.
- Direct commentary from Simons, Berlekamp and Patterson's book was not retrieved. Any quote from them other than the above should be treated as unverified.
- Sharpe ratios for Wellington, DE Shaw, Two Sigma and Millennium: I found none from a reliable source.

---

## What Sharpe, CAGR and drawdown have long-run systematic strategies delivered live?

### Takeaway
Multi-decade live Sharpes cluster at about 0.4 to 0.8, even for the famous names. The best sourced 30+ year figure is Berkshire at 0.76 to 0.79, with about 1.6x leverage. Higher CAGRs in this group came with high volatility and drawdowns near -60%, and none of them reached 35% over the long run.

### Cited Findings

| Strategy | Period | Return p.a. | Vol | Sharpe | Max DD | Grade | Source |
|---|---|---|---|---|---|---|---|
| Berkshire Hathaway | 30+ yrs | not on page read | n/a | 0.76 (earlier version), 0.79 (2018 version) | n/a | ACADEMIC-OOS on audited listed-company returns | [AQR](https://www.aqr.com/Insights/Research/Journal-Article/Buffetts-Alpha); [SSRN](https://papers.ssrn.com/abstract=3197185) |
| SG Trend Index | Dec 1999 to Jun 2026 | n/a | n/a | 0.47 | n/a | REPORTED-LIVE index of real programmes, via an academic paper | [Sepp and Lucic](https://arxiv.org/html/2607.19497v1) |
| SG CTA Index | Jan 2000 to Aug 2023 | n/a | n/a | 0.56 (S&P 500 TR: 0.51) | n/a | REPORTED-LIVE, quoted by a CTA manager | [CFM](https://www.cfm.com/is-trend-really-having-a-bad-year/) |
| CTA indices (unnamed) | 2000 to Dec 2021 | 4.5% | 11% | n/a | -18% | MARKETING | [RCM via HFA](https://hedgefundalpha.com/strategies/a-closer-look-at-the-average-historical-return-for-ctas/) |
| Man AHL Diversified | Mar 1996 to Aug 2026 | 9.86% | 16.01% | 0.44 | n/a | AUDITED-LIVE fund factsheet (figures read via a search excerpt) | [Factsheet](https://swissfunddata.ch/sfdpub/docs/fsm-2320_00_01-20260828-en.pdf) |
| Dunn WMA | Nov 1984 onward | about 12.2-12.4% | 31.5% | 0.49-0.54 | -60.26% | REPORTED-LIVE tracker; "Compiled By: N/A" | [AutumnGold](https://autumngold.com/sweet/cta_profile.php?id=393); [FundPeak](https://ctadatabase.fundpeak.com/Program.aspx?id=890) |
| Mulvaney | May 1999 onward | about 18.2-18.5% (reports conflict) | n/a | 0.53-0.84 (reports conflict) | -60.87% | REPORTED-LIVE tracker | [AutumnGold](https://www.autumngold.com/Advisor/cta_profile.php?id=341) |
| Bridgewater Pure Alpha | Dec 1991 onward | 11.4-12%; Pure Alpha II 13.6% | 12% and 18% targets | not verified | -13% in 2020 (secondary) | REPORTED-LIVE | [Bloomberg](https://www.bloomberg.com/news/articles/2022-07-05/bridgewater-flagship-fund-gains-32-through-first-half-of-2022); [II](https://www.institutionalinvestor.com/article/b1cl6h4y6fd8xh/Bridgewater-Posts-a-Stellar-2018); [CNBC](https://www.cnbc.com/id/101565460?view=story) |
| AQR Style Premia (QSPIX) | Oct 2013 onward | about 6.72% avg (as of about Sep 2025) | n/a | n/a | -41.37% | AUDITED-LIVE mutual fund, via data sites | [PortfoliosLab](https://portfolioslab.com/tools/stock-comparison/QSPIX/AQMIX) |
| AQR Managed Futures (AQMIX) | Jan 2010 onward | not found | n/a | n/a | -26.52% | AUDITED-LIVE mutual fund, via data sites | [PortfoliosLab](https://portfolioslab.com/tools/stock-comparison/QSPIX/AQMIX) |
| HFR macro funds, median | about 15 yrs | n/a | n/a | about 0.4, systematic and discretionary alike | n/a | MARKETING (manager study of HFR data) | [Two Sigma](https://www.twosigma.com/articles/systematic-and-discretionary-macro-better-together) |
| Bank alternative-beta indices, median | live periods | n/a | n/a | 0.31 live vs 1.20 backtest | n/a | ACADEMIC-OOS | [CXO on Suhonen et al](https://www.cxoadvisory.com/big-ideas/live-performance-of-alternative-beta-products/) |
| Multi-strategy index | 5 yrs to about 2025 | 11.9% | n/a | 2.9 | n/a | MARKETING, index aggregate | [Foresight](https://www.yieldreport.com.au/wp-content/uploads/2025/06/F_Foresight-Alternative-Investment-Review_May25_14.06.25_18062025.pdf) |
| HFT firms, E-mini | short sample | n/a | n/a | median 4.3 | n/a | ACADEMIC-OOS on regulator data | [Baron et al, CFTC draft](https://www.cftc.gov/idc/groups/public/@economicanalysis/documents/file/oce_riskandreturn0414.pdf) |
| HFT firms, Swedish equities | short sample | n/a | n/a | mean 4.16; median 1.61 (my reading of the slide columns); p10 -1.47, p90 11.14 | n/a | ACADEMIC-OOS | [Slides](https://sambuz.com/doc/risk-and-return-in-high-frequency-trading-ppt-presentation-964307) |

**Further detail**
- [ACADEMIC-OOS] Berkshire's realised Sharpe is the highest among stocks and mutual funds with more than 30 years of history, and its alpha becomes insignificant after controlling for Betting-Against-Beta and Quality-Minus-Junk. — [SSRN](https://papers.ssrn.com/abstract=3197185)
- [ACADEMIC-OOS] Leverage is about 1.6-to-1, partly financed by insurance float at a low rate. I found no source for the 1.7 figure in the brief. — [AQR](https://www.aqr.com/Insights/Research/Journal-Article/Buffetts-Alpha)
- [REPORTED-LIVE] Best calendar year on record: SG Trend about +27.35% and SG CTA about +20.13%, both 2022, since January 2000. — [Alpha Week](https://alpha-week.com/2022-cta-index-performance-review); [Hedgeweek](https://www.hedgeweek.com/trend-followers-turn-leaders-ctas-deliver-record-returns-2022/)
- [BACKTEST] Sepp and Lucic's three trend replicas, Dec 1999 to Jun 2026 net of costs and 2/20 fees, show Sharpes of 0.47, 0.50 and 0.55, statistically indistinguishable from SG Trend at 0.47. — [arXiv 2607.19497](https://arxiv.org/html/2607.19497v1)
- [BACKTEST] AQR's simulated time-series momentum, 1880-2013, shows a net Sharpe of 0.77. The snippet's unlabelled figures were 14.9%, 11.2% and 9.7%. By my reading those are gross return, net return and volatility. — [TrendFollowing.com copy](https://www.TrendFollowing.com/whitepaper/Century_Evidence_Trend_Following.pdf)
- [BACKTEST] The 2017 update, 1880-2016 across 67 assets, reports average per-asset net Sharpe of 0.4 under 2/20 fees. Costs of rolling futures are not included. — [CXO Advisory](https://cxoadvisory.com/?p=29992)
- [REPORTED-LIVE] QSPIX calendar years: 2017 +12.12%, 2018 -12.35%, 2019 -8.22%, 2020 -21.96%, 2021 +24.93%, 2022 +30.76%, 2023 +12.46%, 2024 +21.48%, 2025 +14.82%. — [Stock data aggregator via search](https://stockanalysis.com/quote/mutf/QSPIX); [YCharts](https://ycharts.com/mutual_funds/M:QSPIX)
- [REPORTED-LIVE] AQMIX 2018 -8.88%, 2019 +1.92%, 2020 -0.43%. — [PortfoliosLab](https://portfolioslab.com/tools/stock-comparison/QSPIX/AQMIX)
- [MARKETING] Man's AHL Diversified Programme was reported at 20.1% annualised since 1996, in an article dated January 2009 with no end date given. — [Hedgeweek](https://hedgeweek.com/news/man-announces-launch-managed-futures-product)
- [REPORTED-LIVE] Mulvaney returned about 3.3% annualised over 2009-2019, then 45.7% annualised over the four years to January 2024, with two 45% drawdowns (2011 and 2018). Figures are unattributed in the post. — [Price Action Lab](https://priceactionlab.substack.com/p/the-amazing-performance-of-mulvaney)
- [ACADEMIC-OOS] Man vs Machine, 9,000+ HFR funds 1996-2014: systematic macro earned 4.9% annualised net alpha against 1.6% for discretionary. Equity hedge was about equal at 1.1% and 1.2%. — [CXO summary](https://cxoadvisory.com/?p=29357); [Paper](https://papers.ssrn.com/abstract=2880641)
- [OPINION] Carver: "Not even hedge funds can sustain an SR of more than 1 over a period of several years". — [The7Circles summary of Systematic Trading](https://the7circles.uk/?p=23464)

### Inferences
- Live Sharpe by type, synthesised from the table: single-country long-only about 0.2 (Carver's expectation, not measured), macro and CTA funds 0.4 to 0.6, the best 30-year compounder 0.76 to 0.79, HFT firms well above 1.5.
- [MATH] Target Sharpe 1.4 is about 3x SG Trend and Man AHL, and about 1.8x Berkshire.
- [MATH] Bridgewater Pure Alpha at 11.4% and 12% volatility implies roughly 0.7, assuming 2.5% to 3% average cash. That cash range is my assumption.
- [MATH] The SG Trend Index's best year in 26+ years (+27.35%) is below the target's annual average. A diversified trend book of ordinary risk would not get there.
- [MATH] QSPIX compounds to about +20.7% a year over 2021-2025, about -37% over 2018-2020, and about 6.1% a year over 2018-2025. A 3-year window can look elite while the full cycle is ordinary.
- [MATH] If the 20.1% and 9.86% figures describe comparable exposure, Man AHL earned only about 3% a year from 2009 to August 2026. The two may be different vehicles or fee loads, so treat this as a direction, not a number.
- Backtest to live for trend: simulated net Sharpe of about 0.77 in AQR's long history against about 0.47 for the live SG Trend Index since 2000. The periods differ, so this is not a clean decay estimate.
- HFT Sharpes depend on queue position, rebates and infrastructure and are not transferable. The Swedish p10 of -1.47 and p90 of 11.14 also show that dispersion is large even there.

### Gaps
- Winton, Millburn and BTOP50: search budget ran out before these were covered.
- AQR's headline live net figures since inception for AQMIX were not found. QSPIX's -41.37% drawdown date is not stated.
- SG's official factsheet stats for return, vol and drawdown were unreadable.
- Dunn and Mulvaney Sharpes use varying rf assumptions. AutumnGold's uses 1%, which flatters them compared with actual cash rates.
- The Man AHL row was read from a search excerpt of the factsheet. The PDF itself could not be read, so net-of-fee status and the exact share class are unconfirmed.
- No live percentile distribution of Sharpe across pods or across all CTAs was found.
- Index and fund survivorship bias is not quantified here. Published track records come from funds that survived, so the live figures above are likely upper bounds on what a typical entrant earned.

---

## What does the Fundamental Law imply, how many uncorrelated streams are needed, and how does Sharpe scale with holding period?

### Takeaway
Reaching 1.4 from 0.5-0.6 streams takes only 6 to 8 streams at zero correlation. It becomes impossible once average correlation exceeds about 0.18 (at 0.6) or 0.13 (at 0.5). Breadth is the only lever that adds Sharpe, but horizon does not behave as the square-root law implies, since IC falls as speed rises.

### Cited Findings
- [ACADEMIC-OOS, theory plus practice] Under the extended Fundamental Law, an unconstrained portfolio has TC = 1.0, and managers in practice operate at TC of 0.3 to 0.8. — [Clarke, de Silva, Thorley, FAJ 2002](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/portfolio-constraints-and-the-fundamental-law-of-active-management)
- [ACADEMIC-OOS] At TC = 0.3, only 9% of performance variation comes from signal success and 91% is constraint-induced noise. — [same paper, via search summary](https://ideas.repec.org/a/taf/ufajxx/v58y2002i5p48-66.html)
- [ACADEMIC-OOS] Raising tracking error in a long-only portfolio typically lowers TC, because the long-only constraint binds for more securities. — [same paper](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/portfolio-constraints-and-the-fundamental-law-of-active-management)
- [BACKTEST, Carver] Moving-average crossover rules across instruments: MAC 64/256 has a pre-cost SR of about 0.28. MAC 2/8 has a law-implied pre-cost SR of 1.68 but an actual 0.07. — [qoppac, "How fast should we trade"](https://qoppac.blogspot.com/2020/04/how-fast-should-we-trade.html)
- [OPINION, Carver] His law-of-active-management illustration for Buffett: IR about 0.7 at a 5-year hold, just over 1.0 at 2 years, over 10 at 1 week. It is a purely illustrative extrapolation, and the point is that the law's scaling does not hold in practice. — [qoppac](https://qoppac.blogspot.com/2020/04/how-fast-should-we-trade.html)
- [OPINION, Carver] "Momentum just doesn't work so well at shorter timeframes", and trade "as slow as possible" subject to diversification. Rule: "DO NOT SPEND MORE THAN ONE THIRD OF YOUR EXPECTED PRE-COST RETURNS ON COSTS". — [qoppac](https://qoppac.blogspot.com/2020/04/how-fast-should-we-trade.html)
- [BACKTEST, in-sample] Across 84 futures, the median autocorrelation-channel Sharpe falls only from 0.55 at a 5-day span to 0.33 at a 2-year span. Lag-1 autocorrelation of vol-normalised returns falls from about 0.04 in the 1990s to about 0.01 after 2010. Short-memory predictability is not exploitable net of frictions at observed autocorrelations. — [Sepp and Lucic 2026](https://arxiv.org/html/2607.19497v1)
- [ACADEMIC-OOS] Extrapolating Sharpe by the square root of time is unreliable when returns are autocorrelated. Half of hedge funds change performance decile when ranked on triennial rather than monthly Sharpes. — [Kinlaw, Kritzman, Turkington via search](https://www.cxoadvisory.com/?p=25773)
- [ACADEMIC, no costs] One moving-average rule study found Sharpes for daily, weekly and monthly trading overlap, with daily costlier. — [arXiv 1907.00212](https://arxiv.org/pdf/1907.00212)
- [OPINION, weak] Anonymous practitioner posts claim higher pre-cost Sharpe for daily over monthly holds, and at least 0.2 Sharpe gain from daily versus monthly rebalance on a 40+ market trend system. Treat as anecdote. — [Reddit snapshot](https://reddit.sentinel-team.org/posts/1ptvpgg/snapshots/2025-12-24T06%3A00%3A44.310294Z)
- [OPINION, Chan] A strategy profitable in almost every month typically has Sharpe above 2, and one profitable almost every day usually exceeds 3. Trading every five minutes can show Sharpe near 3 before costs. — [Excerpt of Quantitative Trading, unofficial copy](https://libcat.ru/knigi/priklyucheniya/unrecognised/562956-9-ernest-p-chan-quantitative-trading.html)
- [ACADEMIC-OOS] Median HFT firm Sharpe was 4.3 in one sample. — [Baron et al, CFTC draft](https://www.cftc.gov/idc/groups/public/@economicanalysis/documents/file/oce_riskandreturn0414.pdf)

### Inferences
- [MATH] Portfolio Sharpe for N equal-Sharpe streams at average pairwise correlation rho: S_p = S x sqrt(N / (1 + (N-1) rho)). The ceiling as N grows is S / sqrt(rho). N needed for target T is N = k(1 - rho) / (1 - k rho), with k = (T/S)^2.

| Avg correlation rho | N for 1.4 from S = 0.6 | N for 1.4 from S = 0.5 | Ceiling at S = 0.6 | Ceiling at S = 0.5 |
|---|---|---|---|---|
| 0.00 | 6 | 8 | none | none |
| 0.05 | 8 | 13 | 2.68 | 2.24 |
| 0.10 | 11 | 33 | 1.90 | 1.58 |
| 0.15 | 26 | unreachable | 1.55 | 1.29 |
| 0.20 | unreachable | unreachable | 1.34 | 1.12 |
| 0.30 | unreachable | unreachable | 1.10 | 0.91 |

- [MATH] Hard limit: 1.4 is unreachable once rho exceeds 0.184 at S = 0.6 or 0.128 at S = 0.5.
- [MATH] If each stream's 0.6 is a backtest figure and live decays by half to 0.3, then k = 21.8: 22 streams at rho = 0, and unreachable above rho of about 0.046.
- [MATH] Symmetric bets give S = (2p - 1) / (2 sqrt(p(1-p))) x sqrt(n). For S = 1.4, the hit rate needed is about 68.7% at 12 independent bets a year, 59.5% at 52, 56.9% at 100, 54.4% at 250, and 51.4% at 2,500. Correlated positions held together count as fewer independent bets.
- [MATH] Long-only cost: a long-only book at TC 0.5 needs twice the IC of an unconstrained book for the same IR. On the law's terms, that is equivalent to needing 4x the breadth.
- Correlations are not stable. A portfolio built on low calm-period correlations will have higher correlation when it matters. Stream count should be judged on stress-period correlations, not full-sample averages.
- Empirically, slowing down does not cost as much Sharpe as the square-root law implies, and speeding up does not earn as much. Sepp and Lucic's Sharpe changes about 1.7x across a 100x span where the law would imply 10x. That supports positional trading as a sensible horizon, while removing any argument that a slightly faster swing cadence alone will close a 2.3x Sharpe gap.

### Gaps
- Grinold-Kahn original text not retrieved; the law is stated here from its standard form.
- Lopez de Prado's precision and frequency treatment was not retrieved. The hit-rate arithmetic above is my own derivation of the same symmetric-bet result.
- "Harris" on Sharpe and holding period, and Aldridge's Sharpe-by-frequency table: not retrieved.
- A clean live, net-of-cost dataset of Sharpe by holding period: none found. The evidence is backtest and anecdote, plus the HFT papers.
- Ding and Martin's redux of the law and other empirical work on realised IC by horizon: not retrieved.

---

## What do Carver, Clenow, Chan and Simons say a realistic retail or small-fund Sharpe is?

### Takeaway
Carver is the only one with sourced numbers that map onto this question: about 0.4 for one instrument, up to about 0.8 for a multi-asset dynamic system, 1.0 as an absolute ceiling, and a backtested SR of 2+ treated as overfitting. Nothing retrieved from any of the four supports a retail Sharpe of 1.4.

### Cited Findings
- [OPINION, Carver, Systematic Trading via notes] Expected SR ladder: stocks baseline 0.15; single-country portfolio 0.20; global equities 0.25; multiple asset classes up to 0.40; dynamic trading single instrument 0.40; dynamic with multiple asset classes up to 0.80. — [The7Circles summary](https://the7circles.uk/?p=23464)
- [OPINION, Carver] "The absolute max. Sharpe we should expect is 1.0, regardless of how good the backtest is" (p. 146). — [Reading notes thread](https://threadreaderapp.com/thread/1234306875379683328.html)
- [OPINION, Carver] "My back-tested Sharpe ratio is 1.0, which gives me a realistic achievable SR of 0.75" (p. 146). — [Reading notes thread](https://threadreaderapp.com/thread/1234306875379683328.html)
- [OPINION, Carver] Per-instrument subsystem SR is "unlikely to be higher than 0.40 on average" (p. 187). — [Reading notes thread](https://threadreaderapp.com/thread/1234306875379683328.html)
- [OPINION, Carver] A backtested SR of 2 or more is a sign of over-fitting. — [The7Circles summary](https://the7circles.uk/?p=23464)
- [OPINION, Carver] Use half-Kelly: "That would be a 75% volatility target at full Kelly, which I then halved to get 37%" (p. 146). On an absolute cap: "you should never run a higher risk target than this" for 50% annualised (p. 148). He ran 25% himself at the time of writing. — [Reading notes thread](https://threadreaderapp.com/thread/1234306875379683328.html)
- [OPINION, Carver] Costs: "It's foodhardy [sic] to pay more than a third of your expected Sharpe ratio in trading costs" (p. 187). Fast systems are too expensive for private investors, and medium speed (a few days to several months) is preferred. — [Reading notes thread](https://threadreaderapp.com/thread/1234306875379683328.html); [The7Circles](https://the7circles.uk/?p=23464)
- [BACKTEST, Carver] In 2015 his futures system backtested at 0.88 net and about 0.94 gross, and he said he did not expect to realise this live, partly because the sample included a large one-off repricing gain. — [qoppac via search](https://qoppac.blogspot.com.es/2015/03/simulating-my-futures-system.html)
- [OPINION, Carver] A forward planning assumption of SR about 0.24 per instrument, "roughly in line with the slower MAC rules". — [qoppac](https://qoppac.blogspot.com/2020/04/how-fast-should-we-trade.html)
- [BACKTEST, Clenow] A book-style reproduction of Stocks on the Move, 1999-2014, shows about 12.2-12.3% annual return with about -23.5% to -24% drawdown. Table labels were hard to read. — [The Chartist](https://www.thechartist.com.au/?p=5829)
- [BACKTEST, community] Independent versions of Clenow's method range from 5.9% CAGR at Sharpe 0.84 to about 14.9% at Sharpe about 1.1, and one at 7.3% with -38% drawdown and Sharpe 0.33. — [QuantConnect examples](https://www.quantconnect.com/forum/discussion/16578)
- [OPINION, Chan] A blog summary says Sharpe above 1.0 is needed for standalone use. Unconfirmed in Chan's text. — [Algomatrix summary](https://algomatrix.blogspot.com/2025/10/book-summary-quantitative-trading-by.html)

### Inferences
- Centurion's current 0.60 is already inside Carver's expected band for a diversified system. It is not an obvious sign of underperformance.
- A move to 0.8 to 1.0 is the top of what Carver would call plausible. That still requires genuine multi-asset diversification, and 1.4 is above his stated ceiling.
- Carver's rules were written mainly for liquid futures with cheap shorting. An Indian cash-equity book has higher frictions and a long bias, which argues for the lower end of his range, not the upper. This is my inference.
- Clenow's cash-equity momentum is the closest public analogue to a swing or positional equity system, and its backtests show CAGRs in the low teens, well short of 35%.

### Gaps
- Advanced Futures Trading Strategies per-strategy Sharpes and the 102-instrument tables: not retrieved.
- Leveraged Trading: only the 0.24 figure via his blog.
- Carver's own audited or published live Sharpe: not retrieved, although he is known to publish results.
- Following the Trend core model results: not retrieved.
- Simons's own statements on realistic Sharpes: not retrieved. Nothing here should be attributed to him beyond book quotes.
- Chan's own fund results: not retrieved.

---

## How large is post-publication and backtest-to-live decay?

### Takeaway
The range is roughly 50% to 75%. Published anomaly returns fall about 26% out of sample and about 58% after publication, and Sharpes roughly halve. Commercial backtests lose a median 73% of Sharpe going live. A backtested Sharpe of 1.4 is therefore not a 1.4 expectation, and applied to a backtest, the target implies a level practitioners call overfitted.

### Cited Findings
- [ACADEMIC-OOS] McLean and Pontiff (JF 2016), 97 predictors: returns about 26% lower out of sample and about 58% lower post-publication, implying about 32% from publication-informed trading. Decay is larger for predictors with higher in-sample returns. — [Paper copy](https://Www.Gwern.net/doc/economics/2016-mclean.pdf)
- [ACADEMIC-OOS] Falck, Rej and Thesmar, 72 strategies: Sharpes fall by roughly half after publication. Publication year explains 30% of the variance in decay, with decay rising about 5 ppt per year of newer publication. Overfitting proxies add 15%, and arbitrage variables add little. — [arXiv 2105.01380](https://arxiv.org/pdf/2105.01380)
- [ACADEMIC-OOS] Published version summary: out-of-sample anomalies deliver about 50% of in-sample performance. — [Quantitative Finance 2022](https://econpapers.repec.org/RePEc:taf:quantf:v:22:y:2022:i:11:p:1955-1969)
- [ACADEMIC-OOS] Hou, Xue and Zhang: of 447 anomalies, 286 (64%) are insignificant at the 5% level, and 380 (85%) fail at a t cutoff of 3. — [NBER w23394](https://www.nber.org/papers/w23394.pdf)
- [ACADEMIC] Harvey, Liu and Zhu: a newly discovered factor should clear "a t-ratio greater than 3.0", and "most claimed research findings in financial economics are likely false". — [NBER w20592](https://www.nber.org/papers/w20592)
- [ACADEMIC-OOS, counter-evidence] Jensen, Kelly and Pedersen reach a different conclusion using a Bayesian model: most factors replicate, group into 13 themes, and "work out-of-sample in a new large data set covering 93 countries". The abstract page gives no numeric replication rate. Replication of a gross factor premium is a weaker claim than a tradable live net-of-cost Sharpe. — [NBER w28432](https://www.nber.org/papers/w28432)
- [ACADEMIC-OOS, live data] Suhonen, Lennkh and Perez, about 215 bank strategy indices across 17 banks: median Sharpe 1.20 backtest against 0.31 live, a 73% fall. 65 of 215 had negative live Sharpe and 18 of 215 matched their backtest. Decay was about 30% stronger for the most complex strategies. — [CXO Advisory](https://www.cxoadvisory.com/big-ideas/live-performance-of-alternative-beta-products/); [Alpha Architect](https://alphaarchitect.com/looking-at-alternatives-avoid-complexity-and-magical-backtests/)
- [ACADEMIC-OOS] Wiecki et al, 888 Quantopian algorithms with 6+ months out of sample: backtest Sharpe had almost no predictive value (R-squared below 0.025), and more backtesting meant a larger gap. Non-linear classifiers reached 0.17 on hold-out data. — [hedgefundalpha summary](https://hedgefundalpha.com/strategies/comparing-backtest-sample-performance-large-cohort-trading-algorithms/)
- [ACADEMIC, methods] Arnott, Harvey and Markowitz set out a protocol: a prior hypothesis should guide tests, and the only truly fresh samples are live trading data. — [SSRN](https://papers.ssrn.com/abstract=3275654)

### Inferences
- [MATH] To expect live Sharpe of 1.4 under these haircuts, the honest backtest would need 2.8 (50% haircut), 3.3 (58%), or 5.2 (73%). Carver treats 2+ as overfitting. The two readings together make a backtest route to 1.4 nearly self-defeating.
- Centurion's 0.60 is described as live-like, after walk-forward and a haircut. It should not be haircut a second time for overfitting. Further decay from crowding and post-publication arbitrage can still apply to any stream built from published anomalies.
- Newer publications decay more. Streams built from recent public literature should get the larger haircut.
- [MATH] Under iid returns, the standard error of Sharpe is about sqrt((1 + S^2/2) / T), per Lo (2002), a formula recalled and not re-fetched. S = 0.6 over 18 years has SE 0.26, so a 95% interval of about 0.1 to 1.1. S = 1.4 over 3 years has SE about 0.81. 60 days of paper trading gives an SE above 2, so it can only catch operational failures and cannot validate edge.

### Gaps
- Jacobs and Muller: findings unverified. The SSRN page returned 403, and the only trace found was a passing search-snippet citation, in a Hou, Qiao and Zhang paper on China, saying international anomaly patterns differ from the US. My recollection that post-publication decline appears mainly in the US, with little reliable decline internationally, is unverified. It matters directly for India, because it would mean published anomalies may decay less there, so it is the top item to check.
- Harvey and Liu's "Backtesting" Sharpe haircuts: not retrieved. My recollection is that the haircut is non-linear and not a flat 50%, unverified.
- Bailey et al's minimum backtest length and PBO numbers: not retrieved (page returned 403).
- India-specific post-publication decay: not covered.

---

## What does leverage do to a Sharpe 0.6-0.9 stream? Can it produce 35% CAGR, and with what drawdown and risk of ruin?

### Takeaway
A Sharpe 0.6 stream cannot reach 35% at any leverage. Growth peaks at about 24.5% at full Kelly and 60% volatility, where a -50% drawdown is routine. A Sharpe 0.9 stream needs about 41% volatility, which implies a -50% drawdown about every 10 years under Gaussian assumptions that understate tails. Hitting 35% with rolling 3-year Calmar >= 1 takes a true Sharpe of about 1.2 or better. Kite's margin financing (about 14.6% a year) would make cash-equity leverage worse than none for a Sharpe 0.6 stream.

### Cited Findings
- [OPINION, Carver] Sharpe estimates are optimistic, so use half-Kelly at most. "If you get your estimate of Sharpe ratio wrong and bet more than optimal, you have a large chance of losing your shirt" (p. 145). — [Reading notes thread](https://threadreaderapp.com/thread/1234306875379683328.html)
- [OPINION, Carver] Negative-skew strategies should run at half the risk of a benign system (p. 146). — [Reading notes thread](https://threadreaderapp.com/thread/1234306875379683328.html)
- [REPORTED-LIVE] Wellington's -55% in 2008, from leveraged convertible exposure, shows high-grade leverage failing in a crisis. — [Phoenix Learning](https://phoenixlearning.substack.com/p/citadels-rise-to-immortality)
- [ACADEMIC-OOS] Berkshire's best-in-class record used only about 1.6x leverage, on cheap float. — [AQR](https://www.aqr.com/Insights/Research/Journal-Article/Buffetts-Alpha)
- [Broker tariff] Zerodha MTF interest is "0.04% per day (₹40 per lakh) on the funded amount", applied from T+1 until the stock is sold. Debit balances pay "18% a year or 0.05% per day". MTF brokerage is 0.3% or Rs. 20 per order, whichever is lower, plus a pledge charge of ₹15 + GST per request per ISIN. — [Zerodha charges](https://zerodha.com/charges)

### Inferences
All figures in this section are [MATH], hand-computed under a frictionless Gaussian model unless they state a spread.

**CAGR by Sharpe and vol, rf = 6.5% (g = rf + S x vol - vol^2 / 2)**

| Sharpe | 10% vol | 15% vol | 23% vol | 30% vol |
|---|---|---|---|---|
| 0.6 | 12.0% | 14.4% | 17.7% | 20.0% |
| 0.9 | 15.0% | 18.9% | 24.6% | 29.0% |
| 1.2 | 18.0% | 23.4% | 31.5% | 38.0% |
| 1.4 | 20.0% | 26.4% | 36.1% | 44.0% |

- If the project's Sharpe is computed with rf = 0 and not as excess over cash, subtract 6.5 points from every cell before comparing. That definition must be confirmed, because it moves the target materially. Raw 1.4 at 23% vol is only about 1.12 in excess terms.
- Volatility drag is 0.5, 1.1, 2.6 and 4.5 points at the four vols.

**Vol needed for 35% CAGR (vol = S - sqrt(S^2 - 0.57))**

| Sharpe | Vol needed | Kelly growth limit (rf + S^2/2) | Half-Kelly growth (rf + 3/8 S^2) |
|---|---|---|---|
| 0.6 | impossible | 24.5% at 60% vol | 20.0% at 30% vol |
| 0.755 | 75.5% (exactly full Kelly, the minimum possible) | 35.0% | 27.9% at 38% vol |
| 0.9 | 41.0% | 47.0% at 90% vol | 36.9% at 45% vol |
| 1.2 | 26.7% | 78.5% at 120% vol | 60.5% at 60% vol |
| 1.4 | 22.1% | 104.5% at 140% vol | 80.0% at 70% vol |

**Time to a given peak-to-trough loss.** Reflected Brownian motion, where b = -ln(1 - D), theta = 2g / vol^2, and expected time = (e^(theta b) - 1 - theta b) / (g theta). The 18-year probability uses an exponential approximation.

| Case | D | Expected years to first hit | P(within 18 yrs) |
|---|---|---|---|
| S 0.6, 30% vol, g 20% | -30% | 2.6 | about 100% |
| S 0.6, 30% vol, g 20% | -50% | 19.9 | about 60% |
| S 0.6, 60% vol (full Kelly), g 24.5% | -50% | 1.9 | about 100% |
| S 0.6, 60% vol (full Kelly), g 24.5% | -80% | 17.2 | about 65% |
| S 0.9, 41% vol, g 35% | -30% | 1.3 | about 100% |
| S 0.9, 41% vol, g 35% | -50% | 9.6 | about 85% |
| S 0.9, 41% vol, g 35% | -60% | 27.9 | about 48% |
| S 1.2, 26.7% vol, g 35% | -30% | 8.3 | about 89% |
| S 1.2, 26.7% vol, g 35% | -50% | 261 | about 7% |
| S 1.4, 22.1% vol, g 35% | -30% | 31.9 | about 43% |
| S 1.4, 22.1% vol, g 35% | -35% | 93.8 | about 17% |

- These are lower bounds on real risk, since they assume normal returns and no gap risk.
- Estimation error is the main danger. Run 41% vol believing S = 0.9 when the true S is 0.6, and g falls to about 22.7% with the drawdowns above or worse.
- Financing: the tables assume borrowing at rf. Each 1x of borrowed capital at spread s over rf subtracts s from CAGR. Zerodha's MTF rate of 0.04% a day is about 14.6% a year, simple, by my multiplication, which is about 8 points over the assumed rf. Example: S = 0.6 at 12% vol earns about 13%. Levering 2.5x to 30% vol gives 20.0% before spread. With 1.5x borrowed at the 8-point MTF spread, that falls to about 8%, worse than unlevered. At an assumed 2-point spread (futures-like), it is 17.0%.
- Medallion's "ample and cheap" leverage has no retail equivalent. MTF is priced so that it destroys a Sharpe 0.6 stream, so only exchange-embedded leverage (index futures and similar) could plausibly be used.
- Calmar is not the binding constraint. Using the Magdon-Ismail expected drawdown formula (recalled, not re-fetched), a 3-year window at S = 1.4 and about 22% vol gives expected MaxDD near 21% and Calmar near 1.6. At S = 0.9 and 18% vol, it gives about 19% and Calmar near 1.1. At S = 0.6 and 12% vol, about 13% and near 1.0. Real fat tails will lower these.
- Sharpe is the binding constraint. The CAGR target then follows from vol: 1.4 at 22% gives 35%.
- Consistency check for the baseline. 13% CAGR with Sharpe 0.60 implies about 12% vol if Sharpe is excess of 6.5% rf, or about 28% vol if rf = 0. At 12% vol, a -29.5% drawdown is very unlikely under Gaussian assumptions (expected wait about 230 years), which suggests either a higher real vol, a different Sharpe definition, or fat tails such as 2008. This should be confirmed from the project's own metrics, which I did not read.
- The project's bounded-tail rule, capping any overlay's per-cycle loss at 10-15% of the book, argues against any structure sized to reach 30%+ vol through leverage.

### Gaps
- The cost of carry embedded in Nifty futures and how it behaves in stress: not retrieved.
- Magdon-Ismail and Lo formulas were not re-fetched.

---

## Which live swing or positional, non-intraday systems have audited Sharpe above 1.2 over 10+ years, and at what capacity?

### Takeaway
None found. No audited positional or swing system showed Sharpe above 1.2 over 10+ years in anything I could retrieve, and none met the joint target. This is absence of evidence under an incomplete search, because many closed funds do not publish, but the sourced ceiling for long-run positional live records is about 0.8. The only sourced 30% multi-decade CAGR outside Medallion is Druckenmiller's Duquesne: discretionary macro, unaudited publicly, no Sharpe given. The target is not supported as a sustained expectation under these constraints.

### Cited Findings
- [ACADEMIC-OOS] Best documented long-run compounder in stocks and funds: Berkshire at 0.76-0.79, about 1.6x leverage, very high capacity. — [SSRN](https://papers.ssrn.com/abstract=3197185); [AQR](https://www.aqr.com/Insights/Research/Journal-Article/Buffetts-Alpha)
- [REPORTED-LIVE] Long-run trend programmes: Dunn about 12% at -60%; Mulvaney about 18% at -61%, AUM $428M through September 2026. — [AutumnGold Dunn](https://autumngold.com/sweet/cta_profile.php?id=393); [AutumnGold Mulvaney](https://www.autumngold.com/Advisor/cta_profile.php?id=341)
- [AUDITED-LIVE] Man AHL Diversified: 0.44 over 30 years. — [Factsheet](https://swissfunddata.ch/sfdpub/docs/fsm-2320_00_01-20260828-en.pdf)
- [MARKETING] Multi-strategy index Sharpe of 2.9 over 5 years at 11.9% return. This is an index of levered pod platforms, not a positional retail analogue, and it covers only five years. — [Foresight](https://www.yieldreport.com.au/wp-content/uploads/2025/06/F_Foresight-Alternative-Investment-Review_May25_14.06.25_18062025.pdf)
- [REPORTED-LIVE] Duquesne Capital: "an average annual return of 30 percent without any money-losing year", over 30 years from its 1981 founding, with over $12bn at closure in August 2010. The measurement period and citation are loose, and it was discretionary macro, not systematic. No Sharpe, volatility or leverage figure is given. — [Wikipedia, citing Bloomberg and Forbes](https://en.wikipedia.org/wiki/Stanley_Druckenmiller)
- [REPORTED-LIVE] Multi-year stretches can reach the target range: Mulvaney 45.7% annualised for four years, but about 3.3% a year in the preceding decade. — [Price Action Lab](https://priceactionlab.substack.com/p/the-amazing-performance-of-mulvaney)

### Inferences

**Which levers a swing or positional retail system can use**

| Lever | Share of elite performance | Available to Centurion? | Note |
|---|---|---|---|
| Very short holding period | Very high for Medallion and HFT | No | No-intraday rule, and execution infrastructure is absent |
| Market making, queue position, rebates | Very high for Jane Street and HFT | No | Different business |
| Cheap institutional leverage | High for Medallion, moderate for Berkshire via float | Partly | Futures only; MTF at about 14.6% a year destroys the edge |
| Number of independent bets | The main mechanical driver | Partly | Limited by market factor correlation; rho must stay below about 0.13-0.18 |
| Multiple uncorrelated asset classes | High for CTAs and macro funds | Partly | India offers fewer liquid uncorrelated instruments; bonds, gold and options are the candidates |
| Long-short construction (TC) | High | Partly | Stock shorting is restricted; TC likely 0.3-0.8 for constrained books |
| Capacity advantage | Negative for large funds, positive for small | Yes | The one true structural edge at tens of lakhs to a few crore |
| Slower alpha tolerated by cost | Moderate | Yes | Carver's speed limit is friendly to positional horizons |
| Data and research scale | High for Medallion | Weak | Small team; no alternative data |

**Verdict and realistic ranges**
- Sharpe 1.4 at 22% vol is the required combination. On the sourced evidence, that is above Carver's 1.0 ceiling, about 1.8x Berkshire, and about 3x SG Trend and Man AHL.
- Modest leverage cannot close the gap, because it scales return and risk equally. Only a higher Sharpe does.
- Medallion is not a precedent for the constraints. It differs on horizon, leverage, financing, data and execution, and its compounded figure is disputed.
- [MATH] Excellent live outcomes for a diversified system would be about S = 0.9 at 18% vol, giving about 21% CAGR, or S = 1.0 at 20% vol, giving about 24.5%. These sit near the top of what the evidence supports, and getting there requires real new uncorrelated streams.
- Rolling 3-year figures can exceed the target in good regimes, as QSPIX 2021-25 and Mulvaney 2020-23 show. Using the median across rolling windows mitigates this, but a median still centres on the true Sharpe, so it does not lower the bar.
- [MATH] If 35% is kept as a goal, the honest path is S of at least 1.2 at 27% vol. That implies a roughly 89% chance of a -30% drawdown within 18 years even under Gaussian assumptions.

### Gaps
- Candidates I could not verify, listed from recollection only. Do not cite these without checking: Soros's Quantum Fund (about 30% over decades, discretionary), Julian Robertson's Tiger (discretionary), Ed Thorp's Princeton Newport (one retrieved source gives 17.7% gross and 14.1% net over its first decade, [RCM](https://www.rcmalternatives.com/?p=20498), which conflicts with the higher figures commonly quoted), Richard Dennis's Turtles and JWH (high early CAGR, deep drawdowns, unaudited).
- Those names would matter because the closest non-Medallion matches to high CAGR at positional horizon appear to be discretionary, not systematic, as Duquesne is. That pattern is only partly verified here.
- Madoff (a fabricated high-Sharpe positional record) and LJM (option-selling blow-up) are the obvious warnings about high reported Sharpe at positional horizon. Neither was verified in this run.
- AutumnGold's and IASG's ranked CTA databases were not readable, so a screen for 10+ year programmes with Sharpe above 1.0 was not possible. The claim of "none found" is limited accordingly.
- Indian PMS, AIF or fund precedents were not searched. Other researchers may cover them.
- Closed proprietary funds (PDT, TGS and others) may have higher Sharpes but publish nothing, so they cannot count as documented precedent.
