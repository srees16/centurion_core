# New India alpha sources for a long-only swing book (as of 1 Nov 2026)

**How far to trust these notes**
- Grades: **LIVE** audited or regulator data. **ACAD-OOS** academic with out-of-sample test. **ACAD-IS** academic, in-sample. **BT** backtest or broker study. **MKT** marketing or unmethodical claim. **R** rule or tax fact. Flag *(snippet)* means I read only a search summary or abstract.
- Full text read: the NSE-NYU GSM draft, the SIP timing preprint, the TaxGuru summary of SEBI's anchor study, one Wright Research page, one 5paisa article, the abstract-level page on gross profitability.
- Blocked: Business Perspectives, Indexology and Business Standard returned 403. The SSRN paper by Raju on the 52-week high, the NSE ASM whitepaper and several NSE PDFs timed out or could not be fetched. Numbers from those are snippet-level.
- No net-of-cost Indian event-trade study turned up on any topic. Every "net" figure below is either mine and labelled an estimate, or absent.
- All Sharpe, correlation and capacity figures in the ranked table are **my estimates**, not measurements.
- Centurion's tracker (`centurion_core/docs/MEMORY_TRACKER.txt`) was grepped to avoid repeats. No code or data was run.

**Two additions to the brief's "already tested" list, found in the tracker:**
- Residual momentum, 52-week high and low beta were rejected: they lowered book Sharpe on 17 Sep, and a 20-name book dilutes the trend forecast by Sharpe -0.00 to -0.25 (tracker lines 3326-3333, 4233-4234).
- GT4 value and quality sleeve failed, with correlation to the book of 0.56-0.63 (tracker lines 3485-3512). So the Indian papers on 52-week high and gross profitability are not new sources here: [Raju 2023, Oct 2004 to Aug 2023](https://www.scribd.com/document/694152012/SSRN-id4587697) *(snippet, no alpha seen)*, and a gross profitability plus 52-week-high double sort, Oct 2002 to Sept 2022, 0.32% a month equal-weight long-short, Fama-French alpha 1.08%, no costs shown — [IJEFM](https://ijefm.co.in/v8i2/52.php).

## Ranked candidates: what is genuinely new, and what could each add?

### Takeaway
Nothing found is a proven, uncorrelated, net-positive stream. The list holds one new information signal (analyst revision breadth, blocked by point-in-time data), one small-capital niche (retail-quota tenders) and a handful of avoidance filters. Together my estimate is 0 to +0.20 Sharpe, which is not enough on its own for the 0.8 milestone. The rows below do not add linearly: their upper ends overlap and are unlikely to arrive together.

### Cited Findings
Evidence for each row is cited in the sections below. The arithmetic uses the formula from the earlier notes: optimal-weight book Sharpe = √(SR² + (s − ρSR)² / (1 − ρ²)), with SR = 0.60.

### Inferences
**All Sharpe, correlation and capacity entries are my estimates.** "Sharpe add" is the likely change to book Sharpe after capacity limits, not stand-alone Sharpe. Own-Sharpe is omitted because no source supports one.

| # | Candidate | Who pays, why it persists | Best Indian evidence | Corr. to book, normal / crash (estimate) | Data and PIT quality | Capacity at Rs 30L | Failure mode | Sharpe add (estimate) |
|---|---|---|---|---|---|---|---|---|
| 1 | Analyst revision breadth (EPS diffusion) as veto or tie-break | Slow analyst and investor updating; downgrades ignored in bull markets | BT: 8%+ excess CAGR, 257% turnover, gross, to 2016 | 0.4-0.6 / 0.4-0.6; overlaps price momentum | Poor: no free PIT consensus found | Fine | Data buy-in, turnover, dilution of trend signal, decay since 2016 | +0.05 to +0.15 if PIT data exists; 0 otherwise |
| 2 | Retail-quota tenders: buyback, open offer, delisting | Institutions cannot use the reserved quota; small-lot premium persists because it cannot scale | ACAD-IS on announcement only; no arbitrage study; acceptance 14-21% in TCS 2022 | 0.1-0.3 / 0.3-0.6 | Good: exchange filings dated | Rs 2L per event; small book is the edge | Low acceptance, residual falls, semi-manual, few events | 0 to +0.10 |
| 3 | Index events: pre-announcement additions, post-deletion rebound (NIFTY 50, Next 50, MSCI) | Tracker funds are price-insensitive and date-bound | ACAD-IS: deletions +4-7% days 60-240; MKT: 8-15% pre-MSCI | 0.4-0.6 / 0.6+; winners cross thresholds, so overlaps momentum | Good for NSE circulars; medium for MSCI candidates | Fine | Crowded by broker forecasts; closing auction changes flow timing | 0 to +0.08 |
| 4 | Avoidance filters: ASM/GSM entry, rating downgrades, pledge, recent IPO day 29-33 | Holders forced or discouraged; rules restrict trading | ACAD-IS: GSM -6 to -7% relative CAR (n=128); SEBI: -3.5% mean in high-sell IPOs | n/a, a filter | Good from circulars and filings; short history | Fine | Rarely hits a liquid book; filter trials cost core budget | 0 to +0.05 combined |
| 5 | Promoter and insider purchases (system-driven PIT feed) | Insiders know more; bought with own cash | ACAD-IS pre-2015: insiders earn, outsiders do not | 0.2-0.4 / 0.5 | Good from Oct 2020 only, so cannot reach 2008 | Fine, small-cap skew | Outsider edge absent; collateral and creeping-acquisition noise | 0 to +0.05 |
| 6 | Composite-scheme demergers | Forced selling by index and mandate holders (not evidenced in India) | ACAD-IS: +8.27% vs -1.06% for pure, 41 days around announcement | 0.3-0.5 / 0.6 | Good: corporate actions already in store | Fine | Few events; return sits at announcement | 0 to +0.03 |
| 7 | F&O ban and F&O exit | Short constraint at a mechanical threshold | None on bans; option delistings negative, 2001-12 | Unknown | Good: daily list | Fine | Rule changed in 2025, so the sample breaks | 0, test only if cheap |
| 8 | ETF discount at gold or silver entry | Thin arbitrage | ACAD-IS: 2-4 day closure vs persistence, conflict; no gold study | n/a, execution rule | Needs iNAV history | Fine | Tiny | 0 to +0.01 |

**Rejected as sources:** tax-loss March, SIP and month-start flows, dividends, QIP, holdco discount, MTF data, circuit-band changes, participant OI. Reasons are in the sections below.

**What the milestone needs (arithmetic):**
- To lift 0.60 to 0.80 with one stream, it needs own Sharpe of 0.53 at ρ 0, 0.69 at ρ 0.3, or 0.76 at ρ 0.5.
- A stream at 0.3 and ρ 0.5 adds nothing. A stream at 0.3 and ρ 0.3 reaches only 0.61. At 0.4 and ρ 0.1 it reaches 0.69, but only at the optimal weight, which the capacity limits above prevent.

**Test order, cheapest and most informative first:**
1. Item 4 inside the existing GT11 (it is already queued, and the ASM question below sharpens it).
2. Item 3, NSE leg only, as a sleeve-family event study.
3. Item 6, using demerger records already in the store.
4. Item 2, as a hand-collected event table before any code.
5. Item 1 only after the data question below is answered.

## Which Indian flows and calendar mechanics are structural?

### Takeaway
Index flows and tender quotas are structural. Surveillance moves are mechanical but not rule-predictable. Most other items (circuit bands, MTF, SIP, March) have no usable evidence. Rule changes in 2024-26 break samples for F&O bans and buybacks, and possibly for index rebalance days.

### Cited Findings
**Index reconstitution**
- ACAD-IS *(snippet)*: NIFTY 50 2000-2016, inclusion gains reverse within 60 days; exclusions fall for ten days, then earn about 4-7% abnormal over days 60-240 — [Chhatwani 2020](https://publishingindia.com/archive/jcar/price-and-volume-effects-of-nifty-50-index-reorganization).
- ACAD-IS *(snippet)*: positive effective-day return that does not persist, with a volume spike attributed to index funds — [Joshipura and Janakiramanan 2015](https://ideas.repec.org/a/ids/afasfa/v5y2015i1p21-36.html).
- Next 50 cut-off dates are 31 January and 31 July — [NSE factsheet](https://niftyindices.com/Factsheet/ind_next50.pdf). Changes normally take effect at the end of March and September with about four weeks' notice, per secondary sources that partly conflict — [Wikipedia](https://en.wikipedia.org/wiki/NIFTY_Next_50).
- Flow sizes: Next 50 rebalance about Rs 2,500 crore in Sept 2022 — [Business Standard](https://www.business-standard.com/amp/article/markets/nifty-next-50-reshuffle-brings-dozen-stocks-into-sharp-focus-including-ael-122091100392_1.html). Five probable Next 50 additions about $221 million in Sept 2025 (Nuvama estimate) — [Business Standard](https://www.business-standard.com/amp/markets/news/nse-nifty-rejig-indigo-max-healthcare-set-for-nifty-entry-announcement-due-post-market-125082200518_1.html).
- MSCI May 2026 estimates: $491 million Federal Bank, $373 million MCX, $308 million National Aluminium, $209 million Indian Bank (Nuvama) — [Outlook Money](https://www.outlookmoney.com/invest/four-indian-stocks-set-to-enter-msci-global-standard-index-on-may-29-how-big-can-passive-inflows-be).
- MKT: JM Financial claims 8-15% cumulative excess returns in the 20 trading days before MSCI announcements for high-conviction candidates, method unpublished. For August 2026 it projected $3.4 billion in and $159 million out, announced 12 Aug, effective 31 Aug — [Tribune](https://www.tribuneindia.com/news/business/msci-august-review-may-trigger-usd-3-2-billion-passive-inflows-into-indian-equities-jm-financial/).
- Candidate lists conflict between sources for the same review — [Fuzz, unreliable](https://askfuzz.ai/discover/news/connect-the-dots/msci-india-rebalancing-2-3-billion-inflows-loom) against the JM list above.
- I found no academic MSCI India or FTSE India event study, and nothing on FTSE mechanics.

**F&O ban, entry and exit**
- R: from a May 2025 circular, MWPL is the lower of 15% of free float or 65 times average cash volume, updated quarterly, with ban at 95% on delta-adjusted FutEq OI. Dates are stated inconsistently across sources — [Business Today](https://www.businesstoday.in/amp/markets/story/sebi-equity-fo-rules-futeq-method-mwpl-risk-management-478321-2025-05-29); [Outlook Business](https://www.outlookbusiness.com/markets/sebi-shakes-up-derivatives-game-with-new-rules-for-measuring-open-interest-mwpl).
- Banned stocks stay tradable in cash, and the trigger is a position threshold, not news — [IIFL](https://www.indiainfoline.com/news/business/april-15-nse-bans-nine-stocks-in-fo-segment).
- ACAD-IS *(snippet)*: 237 option listings and relistings and 140 delistings on NSE, 2001-2012: positive for listings, negative for delistings — [Global Finance Journal 2018](https://ideas.repec.org/a/eee/glofin/v35y2018icp157-169.html).
- BT, small sample, gross: F&O entrants bought 23 Mar 2017 gained over 18% in a week against 1% for Nifty — [Bloomberg Quint](https://www.bloombergquint.com/markets/2017/04/20/the-sweet-16-new-fando-entrants). Of the Dec 2021 and Jan 2022 entrants, two were up, eight down and one flat six months later — [DSIJ](https://insights.dsij.in/dsijarticledetail/the-fo-effect-with-45-stocks-set-to-be-included-in-new-additions-drive-stock-prices-a-look-into-previous-inclusion-performances-id015).
- I found no event study on ban-list returns and no SEBI paper on F&O exit.

**ASM and GSM**
- ACAD-IS (full draft read): GSM entry, March 2017 to March 2019, 72 treated matched to 65 controls, 128 observations. Treatment coefficient on CAR is -0.058 to -0.068, significant at 5%, in both the (-4,+5) and (-9,+10) windows around the effective date. There is no significant gap for five days after announcement, then divergence. The effective date falls about ten days after announcement. Transactions and turnover fall significantly. Only 35% of firms met the three Stage 0 criteria jointly, and average Stage 0 stay was 92 days. The draft calls itself preliminary and counts 110 securities, against 121 in a later NSE version — [Aggarwal, Bhatia, Zaveri draft](https://archives.nseindia.com/research/content/nse_nyu/NidhiAggarwal_Rules_vs_discretion_Impact_of_trading_rules_on_market.pdf); [later version](https://nsearchives.nseindia.com/s3fs-public/inline-files/Rules_vs_discretion_in_market_surveillance_Aggarwal_Bhatia_Zaveri_0.pdf).
- Stage 0 targets net worth up to Rs 10 crore and net fixed assets up to Rs 25 crore, alongside P/E conditions (same draft). Zaveri is reported as saying the negative effect lasts after restrictions lift *(snippet)* — [Business Standard](https://www.business-standard.com/amp/article/markets/how-sebi-s-warning-system-curbs-gamestop-like-gains-in-indian-stock-markets-121021000097_1.html).
- ASM criteria include high-low variation, close-to-close variation, price-band hits and client concentration *(snippet)* — [Mondaq](https://www.mondaq.com/india/commoditiesderivativesstock-exchanges/729832/asm-framework-a-value-added-check-to-curb-volatility-in-stock-market). One broker notes 100% margin blocking — [Zerodha support](https://support.zerodha.com/category/trading-and-markets/trading-faqs/articles/what-is-asm).
- An NSE paper on the earlier STASM measure reports CAR falling after inclusion *(snippet)*; the PDF timed out, so no number is available — [NSE](https://nsearchives.nseindia.com/s3fs-public/inline-files/Effectiveness%20of%20additional%20surveillance%20measures_WhitePaper.pdf).

**Circuit bands**
- Band changes are frequent and two-way (560 BSE names in one batch; 216 NSE names in Sept 2013; 20% to 10% moves in 2023) — [5paisa](https://5paisa.com/blog/circuit-filter-limits-modified-for-560-stocks-effective-07-oct); [Business Standard](https://www.business-standard.com/article/markets/nse-revises-price-band-on-over-200-stocks-from-today-113090600134_1.html). I found no return study.

**Promoter and event mechanics**
- IPO anchors (SEBI staff study via TaxGuru, 242 mainboard IPOs, Apr 2022 to Oct 2025): when anchors sold 2.5% or less, trading days 29-33 averaged -0.4%; when they sold over 10%, -3.5% mean and about -6% median. Day 90 shows little move. Cumulative anchor sales are 3.5% by day 30, 18.5% by day 90, 50.7% by day 365. FPIs exited about 60%, mutual funds about 38%. The source warns this is not proof of causation — [TaxGuru](https://taxguru.in/?p=1072787). Sources conflict on whether lock-in runs from listing or from allotment — [Rupeezy](https://rupeezy.in/blog/ipo-lock-in-period).
- Pledge: without controls, reaction is negative; with controls, positive on average. Revocations show weak negative reaction, contrary to expectation — [Bang, Sonti, Thirumalai at ISB](https://www.isb.edu/faculty-and-research/research-directory/market-reaction-to-share-pledges-evidence-from-india). Revocations are partly mechanical after rallies — [Business Standard](https://www.business-standard.com/article/markets/pledge-revoking-gains-momentum-107-of-bse-500-companies-reduce-holdings-117032300469_1.html).
- Demergers: 221 BSE spin-offs 2003-2020, positive parent reaction peaking at day +1 *(snippet)* — [Cogent 2022](https://ideas.repec.org/a/taf/oaefxx/v10y2022i1p2109277.html). 109 pure and 51 composite spin-offs 2010-2023: CAAR -1.06% and +8.27% over 41 days *(snippet)* — [Managerial Finance 2024](https://www.emerald.com/insight/content/doi/10.1108/MF-05-2024-0398/full/html). Long-run conclusions conflict — [Nirma](https://repository.nirmauni.ac.in/jspui/handle/123456789/8105?mode=full).
- QIP: only press evidence. 10 of 13 placements in 2009 below offer, -12% average ex-Unitech — [Business Standard, citing Crisil](https://www.business-standard.com/amp/article/markets/three-fourths-of-recent-qips-trading-in-the-red-109071500044_1.html); one in three lagged in 2014 — [Business Standard](https://www.business-standard.com/amp/article/markets/one-out-of-three-qips-underperform-the-market-114111400404_1.html). No multi-year study found.

**Flows and calendar**
- DII flows have marginal effect on returns, while returns affect flows (VAR) — [Saxena and Sikdar 2024](https://ideas.repec.org/a/ids/gbusec/v31y2024i2p230-258.html).
- Expiry-day vs first-day SIP, Nifty 2003-2024: gap 0.03-0.07 points over 10-20 years, "economically negligible", no costs modelled. The abstract's 0.5-2.5% does not match its own tables at 3 and 5 years — [arXiv 2507.04859](https://arxiv.org/html/2507.04859v2).
- March effect: one Sensex regression fits tax-loss selling — [MPRA 24185](https://mpra.ub.uni-muenchen.de/24185/1/MPRA_paper_24185.pdf); a Nifty 2007-2012 test finds no month effect — [Publishing India](https://publishingindia.com/storage/PDFBrochures/2312.pdf). Which paper carries which result was matched from a search summary, so verify before citing.
- MTF: NSE holds about 97% of financing, with the average book at Rs 1.16 lakh crore in Jan 2026 — [CareEdge via Textile Value Chain](https://textilevaluechain.in/mtf-growth-moderates-while-adto-rises-in-january-2026-careedge). March 2026 readings conflict (Rs 1.06 trillion vs 1.13 trillion) — [Business Standard](https://www.business-standard.com/markets/news/mtf-books-decline-march-volatility-leverage-trading-india-126041400292_1.html); [Whalesbook](https://www.whalesbook.com/news/English/bankingfinance/India-MTF-Book-Shrinks-Amid-Geopolitical-Fears-Regulatory-Shifts/69d77b07e1f61dfbf51239f8). No predictive study found.
- Dividends: announcement evidence conflicts — [Chatterjee and Dutta 2017](https://econpapers.repec.org/RePEc:eee:riibaf:v:39:y:2017:i:pa:p:327-337); no net-of-tax ex-date capture study found.

**Tracker results that already answer parts of this:** the FII fragility index had nil out-of-sample R² (tracker line 1420), pre-announcement drift failed (GT13, line 3781) and option-expiry week scored -0.16 Sharpe (O4, line 4238).

### Inferences
- **ASM may matter more than GSM for this book.** GSM catches micro-caps that a liquid book rarely holds. ASM's triggers are price-behaviour measures, which momentum winners can trip, though I found no count of how often. GT11 should measure hit rate on historical holdings first.
- **GSM's lead time is the useful detail.** The gap opens around the effective date, about ten days after a public announcement, so a next-day exit could avoid part of it. That needs testing on a liquid sample. The 72 firms are mostly tiny, and two-thirds of entries failed the stated rules, so entry is not predictable from fundamentals.
- **Index effects look like short-horizon pressure, with reversal.** So trade before the announcement or avoid the event. The deletion rebound is the only positional lead, and its breadth is low: my estimate is under 20 NIFTY 50 and Next 50 events a year.
- **Momentum overlap is the main risk for additions.** Stocks reach index thresholds by rising, so the core may already hold them. Test the return after removing momentum, not the raw figure.
- **Closing auction may change this.** Passive funds gain a new venue for rebalance flows from August 2026, per the earlier notes. Samples before that may overstate the day's pressure, which is inference only.
- **Bans:** the 2025 formula changes which names trip, so pre-October 2025 data and later data are different regimes. Long-only use would be only an entry filter.

### Gaps
- No Indian study on circuit-band changes, ban-list returns, IPO lock-in for promoter shares, tender arbitrage, stock-level MTF or FTSE events.
- Whether NSE publishes stock-level MTF daily was not confirmed.
- The JM 8-15% figure cannot be checked without its data.
- ASM effect sizes need the NSE paper read by hand.

## Which have peer-reviewed or credible evidence after costs and 2024-26 rule changes?

### Takeaway
None has a published net-of-cost result after the 2024-26 changes. Peer-reviewed work stops at announcement-window returns, which an outsider usually cannot capture. Buyback rules and tax changed twice in 2025-26, so old buyback samples do not carry forward.

### Cited Findings
- Buybacks: risk-adjusted returns exist before announcement; "negligible trading opportunities" after; authors point to possible insider trading *(snippet)* — [Pandey, Bhama, Mohapatra 2020](https://businessperspectives.org/journals/investment-management-and-financial-innovations/issue-351/trading-strategy-using-share-buybacks-evidence-from-india).
- 239 Indian announcements, day-zero abnormal return 2.07% *(snippet)* — Dayanandan et al. 2020, cited in the [buyback search result set](https://businessperspectives.org/index.php/journals/investment-management-and-financial-innovations/issue-421/open-repurchase-announcements-and-abnormal-returns-of-indian-firms-an-industry-wise-analysis). I did not read the paper, and this link is the citing page rather than the paper.
- R: the open-market route ended 1 Apr 2025 and returned 1 Aug 2026, capped below 15% of capital and reserves. Notification date conflicts (1 or 6 July). Promoters cannot trade during open-market programmes — [5paisa](https://www.5paisa.com/news/sebi-revives-open-market-buybacks-through-exchanges-from-august-1); [Mondaq](https://www.mondaq.com/sebis-2026-buy-back-reforms-what-listed-companies-and-promoters-need-to-know/1834328). One commentary expects tenders to remain dominant — [Basis Point](https://www.basispointinsight.com/Story/sebi-reopens-buyback-route--but-markets-may-barely-use-it_ec0673e7019c.html).
- R: buyback tax moved in Oct 2024 to the shareholder at slab on proceeds, then from 1 Apr 2026 to capital gains on gain only — [Trendlyne](https://trendlyne.com/posts/5514385/buybacks-after-the-tax-reset-what-has-changed).
- Acceptance: TCS 2022 retail 14-21% against full acceptance in 2017 and 2018, 15% retail reservation, holdings up to Rs 2 lakh — [5paisa](https://test.5paisa.com/blog/tcs-buyback-offer-opportunity-for-retail-investors). 2025 had 15 buybacks worth Rs 19,749 crore, mostly Infosys — [Trendlyne](https://trendlyne.com/posts/5514385/buybacks-after-the-tax-reset-what-has-changed).
- Open offers: 96 offers 2011-2017, substantial positive returns at announcement *(snippet)* — [Inderscience 2021](https://www.inderscience.com/offers.php?id=113165). 187 offers 2007-10 — [MPRA 60432](https://mpra.ub.uni-muenchen.de/60432/1/MPRA_paper_60432.pdf); the earlier notes report negative post-announcement returns from this paper.
- One case: Shanthi Gears, 74% acceptance, with the residual exit price estimated from the pre-announcement price — [CFA Society India](https://cfasocietyindia.org/using-corporate-actions-to-create-value-investing-opportunities-delhi/).
- Statutory delivery cost is about 0.22% a round trip, as worked out in the earlier notes from [Zerodha's schedule](https://zerodha.com/charges).

### Inferences
- **Tender maths, illustrative only.** With premium 15%, acceptance 20% and a 5% fall on the unaccepted 80%, the result is 0.2×15% − 0.8×5% = −1% before the 0.22% cost. At 50% acceptance it is +5%. Acceptance forecasting is the whole trade, and it worsened as arbitrageurs crowded in.
- **Size in the book, my estimate.** Twelve events at Rs 2 lakh and an assumed 2% net make Rs 48,000, or 1.6% of Rs 30 lakh, before about 0.5% of funding cost (assuming five weeks per event, so about 7.7% of the book tied up, at the 6.5% excess return). The result is about +0.1 Sharpe as an upper figure, and about zero if net per event is 0.5%. The 2% is an assumption, not a sourced number.
- **Reverses an earlier note.** The earlier notes dismissed this because it "cannot move a Rs 1-3 crore book". At Rs 30 lakh it can, and at Rs 3 lakh one event is two-thirds of the book, so it needs a size cap.
- The pre-announcement finding is not tradable without predicting announcements.
- Any buyback sample before April 2026 sits under a different tax regime, so history shows acceptance patterns, not current net returns.

### Gaps
- No published distribution of Indian retail acceptance ratios or post-tender price falls. A hand-built table from filings is the first step.
- The retail reservation terms under the 2026 buyback amendment were not confirmed.
- Whether Kite can tender automatically was not confirmed, so assume manual.
- No Indian study on merger-swap spreads or rights entitlements was found; only anecdotes such as the Utkarsh case — [Unlisted Zone](https://unlistedzone.com/the-utkarsh-merger-why-the-obvious-arbitrage-doesnt-exist).

## What published work exists on revisions, insider disclosures, shareholding changes and credit ratings?

### Takeaway
Revisions have the strongest sized result, and it is an S&P study ending 2016. Insider work says insiders earn and outsiders do not, on pre-2015 data. Downgrades carry information and upgrades mostly do not. I found no rigorous test of shareholding-pattern changes.

### Cited Findings
**Revisions**
- BT, hypothetical, gross *(snippet)*: S&P India BMI names with at least three estimates, data to Dec 2016. Equal-weight top quintile earned over 8% (EPS diffusion) and 4.5% (EPS change) excess CAGR. Turnover 257% and 263%. In down markets the top diffusion quintile averaged +1.44% a month excess against -1.30% for the bottom. Size bias was not a driver — [S&P Indexology](https://www.indexologyblog.com/2018/02/21/do-earnings-revisions-matter-in-india/).
- India, Korea and Taiwan showed the most significant effect in Asia; Japan did not *(snippet)* — [S&P Indexology](https://www.indexologyblog.com/2018/02/08/earnings-revision-strategies-in-asia/).
- Broker commentary only since then: revision breadth fell to about -70% in 2018 — [Business Standard](https://www.business-standard.com/amp/article/markets/india-tops-emerging-markets-in-negative-earnings-revision-morgan-stanley-118090900644_1.html).

**Insider and SAST**
- ACAD-IS: BSE firms Apr 2007 to Mar 2015. Purchase portfolios earn positive and sale portfolios negative abnormal return, and outsiders earn nothing significant — [NITK thesis](https://idr.nitk.ac.in/items/2e0597cb-33a1-4f4d-bfd2-6970ede2914a/full).
- ACAD-IS *(snippet)*: substantial shareholders show the highest abnormal price and return, promoters the highest abnormal volume, with no link between abnormal return and volume — [Nanda and Barai](https://ideas.repec.org/a/eme/ajarpp/ajar-09-2020-0089.html).
- R: from 1 Oct 2020 depositories send promoter, promoter-group and director trades to exchanges daily — [AMSS](https://www.amsshardul.com/insight/automation-of-continual-disclosures-under-regulation-72-of-sebi-pit-regulations-2015/). One filing shows the Rs 10 lakh quarterly threshold and the 2% SAST threshold — [NSE filing](https://nsearchives.nseindia.com/corporate/RADIOWALLA_14032026124741_SHAREPURCHASE_DISCLOSURE__1_.pdf).
- Enforcement orders produce no reaction — [Leap Blog](https://blog.theleapjournal.org/2026/05/market-reaction-to-insider-trading.html).

**Shareholding pattern**
- Only commentary and screeners — [Trendlyne help](https://help.trendlyne.com/support/solutions/articles/84000346032-what-are-shareholding-screeners-). No rigorous Indian return test found.

**Credit ratings**
- ACAD-IS *(snippet, abstract garbled)*: top NSE firms 2009-2019. Downgrades bring significant negative abnormal returns, stronger for domestic agencies; upgrades are mostly insignificant — [Dawar, Bhatia, Bindal 2023](https://ideas.repec.org/a/ids/afasfa/v13y2023i1p41-53.html).
- Adverse reaction was smaller after the crisis, and the market values CRISIL over ICRA and CARE — [Krishnan, IIMB](https://repository.iimb.ac.in/bitstream/123456789/12774/1/DIS_IIMB_FPM_P18_05_Abstract.pdf).
- Prices fell almost 5% on average in the next 10 days after a downgrade (blog repost of a column) — [Bang](https://npbang.blogspot.com/2012/01).
- For banks, downgrades showed no negative wealth effect (paper link matched from search results) — [Kushwah and Vigg 2023](https://econpapers.repec.org/RePEc:sae:busper:v:11:y:2023:i:2:p:190-209).

**Queued items, does evidence change?**
- GT7 PEAD via SUE: no new Indian evidence. The revision result is a reason to prefer a breadth measure to a surprise measure if consensus data can be had.
- MV1: revision breadth fits the earnings screen better than extra price rules, which agrees with the tracker's dilution warning.
- INT1: nothing found.

### Inferences
- **Revision breadth is the one real information lead.** It is non-price information, which is what the 13 Oct review said the core lacks. Its down-market spread is unusual for a momentum-linked signal, if it survives after 2016.
- **The 8% is not 8% net.** At 257% turnover, statutory cost alone is about 0.6% a year (0.24% per 100%), plus slippage. The result is gross, hypothetical and a decade stale. My planning haircut is to half or less. A veto form would also use far less turnover.
- **It must not be added as another ranking group.** The tracker shows extra groups dilute a 20-name book. A veto on negative breadth, or a tie-break, is the form to test.
- **Downgrade filter:** the likely hit is rare for a liquid momentum book, since strong-momentum names are seldom being downgraded. That is why its Sharpe add is small.
- **PIT purchases:** a disclosure lag plus "outsiders earn nothing" makes this a weak prior. Clustering, size relative to market cap and open-market-only filters are untested in India.

### Gaps
- **Blocking question:** is point-in-time consensus available to a retail developer? I found no free source. Vendors were not priced. Without it, a rebuilt history would carry look-ahead.
- No post-2016 Indian revision study was found, and no study under the 2015 PIT Regulations.
- The revision result is hypothetical and the actual page returned 403, so its numbers are snippet-level.
- The PIT feed starts 1 Oct 2020, so a 2008-25 backtest is impossible.

## Which are truly uncorrelated with a long-only book, given no overnight cash shorts?

### Takeaway
None is proven. Only tender events plausibly have low correlation, and they carry market exposure while capital waits. Without a short leg, event streams stay long-equity risk, which fits the earlier finding that stress leaves about 2-3 real streams.

### Cited Findings
- The earlier notes (`india_uncorrelated_sources.md`, crash table) planned event streams at low normal correlation and 0.3-0.6 in a crash, as an estimate.
- Failed offers see fast price falls; United Spirits expected about 70% acceptance and got 55% — [Business Standard](https://www.business-standard.com/article/pf/why-open-offers-aren-t-attractive-for-many-investors-114101300002_1.html).
- Single-stock futures shorts are closed for this project (U28, tracker line 792).
- One index-futures lot is about Rs 15 lakh, per the earlier notes' arithmetic of 23,150 × 65 — [TradingView post](https://www.tradingview.com/chart/NIFTY1!/q817Binb-What-a-NIFTY-trade-really-costs-after-the-STT-hike/).
- GT4's fundamental sleeve came in at 0.56-0.63 against 0.4-0.5 hoped for (tracker lines 3502-3503).
- Holdco discounts are 30-90% in India against 10-25% globally, per broker research — [HDFC Securities](https://www.hdfcsec.com/hsl.docs/Holdcos%20for%20portfolio%20diversification%20-%20HSIE-202112021525595803421.pdf). They rarely close without restructuring (forum opinion) — [ValuePickr](https://forum.valuepickr.com/t/risk-free-arbitrage-opportunities-take-them-or-leave-them/86).
- ETF pricing evidence conflicts: gaps close in 2-4 days on one study — [Gnanaganga record](https://gnanaganga.inflibnet.ac.in:8443/jspui/handle/123456789/14506); persistent on another — [Asia University](https://journal.asia.edu.tw/index.php/ADS/article/view/97). CPSE ETF discount gave no arbitrage — [JCAR](https://publishingindia.com/archive/jcar/an-investigation-of-pricing-efficiency-of-cpse-etf-in-india).

### Inferences
- A Nifty hedge for a Rs 2 lakh tender is impossible at Rs 15 lakh a lot, and half the book at Rs 30 lakh. So tender positions are unhedged stock for several weeks.
- Merger-spread trades need a short leg in the acquirer, so they are out unless both names have futures and the U28 closure is revisited. I would not reopen it.
- GT4's correlation surprise suggests planning high for anything equity-side. My ρ figures may be low.
- Honest count: new equity-side ideas are filters and tilts. The true diversifiers remain outside equities (gold, MCX trend, gilts), as the earlier notes found.

### Gaps
- Crash correlation of event streams is unmeasured anywhere; all figures are estimates.
- No gold ETF premium study was read, and the Nargunam and Anuradha 2017 result was not seen.

## What do Indian quant PMS and AIF managers disclose about return sources?

### Takeaway
Very little. Managers name factor families and report headline returns; I found no factor attribution. Nothing suggests a hidden source beyond momentum, quality, low-vol, regime switching and allocation to gold or bonds.

### Cited Findings
- Wright Factor Fund: "10 factor groups" including value, momentum, growth and quality, shifting among equity factors, bonds and gold, monthly rebalance. A backtest of "38% plus" 10-year CAGR before costs, one month live, about Rs 10 crore, no live returns given, unverified by SEBI — [Wright Research](https://wrightresearch.in/blog/complete-guide-factor-investing-wright-factor-fund-pms).
- PL AQUA labels its figures as back-tested results of the in-house model, not actual returns *(snippet)* — [PL brochure](https://www.plindia.com/wp-content/uploads/2026/03/PL-AQUA-PMS-Brochure-FEB-2026.pdf).
- Quant PMS strategies returned 10-17% over six months against BSE 500 TRI 7.19% and Nifty 50 TRI 5.53%, Oct 2025. There are no per-fund figures. Drivers cited are "momentum indicators" and analytics. Quant models were 40-50% of new launches, with new funds near Rs 160 crore combined — [5paisa](https://www.5paisa.com/news/quant-pms-funds-surge-as-data-driven-investing-gains-favour-among-hnis).
- CFA Society India asks SEBI to encourage a standard attribution summary for PMS and AIF strategies, which implies none exists today — [CFA India Insights](https://cfasocietyindia.org/wp-content/uploads/2026/03/India-Insights-April-2026.pdf); factor attribution can show that outperformance is a Size or Momentum bet — [July newsletter](https://cfasocietyindia.org/wp-content/uploads/2026/08/CFA_India_Insights_July_Newsletter.pdf).
- A manager brief argues true alpha needs factor diversification, regime awareness and India-specific alternative data, with no figures — [PMS Bazaar Vol 72](https://resources.pmsbazaar.com/fund-manager-brief/2025/PMSBazaar-Fund-Manager-Brief-Vol-72.pdf).
- Long-short AIFs moved about 0.4 of BSE 500 in Jan and Mar 2025, per the earlier notes — [PMS Bazaar](https://pmsbazaar.com/Blogs/Jan-2025-AIF-Performance-Market-Pressures-Weigh-on-Returns-Long-Short-Funds-Outshine).

### Inferences
- Six months of 10-17% is one regime, not evidence of a source. The set is also self-selected.
- A 38% backtest with one month live is a warning about the golden target's 35% CAGR, not support for it.

### Gaps
- Capitalmind, Abakkus and ArthAlpha attribution was not found. Their factsheets need reading directly.
- No audited multi-year quant PMS Sharpe was found.
- GitHub was not searched for Indian event-study code.

## Decisions these findings raise for the owner

### Takeaway
Three rule conflicts decide whether any of this can proceed.

### Cited Findings
- Live gates require backtest 2008-25, walk-forward, OOS, 60 days' paper and 5 dry runs (project memory, streams and live gates).
- 13 Oct tracker note: 1 core trial left before January, reserved for GT7 and MV1 after V3 (tracker lines 3329-3333). This may be out of date.
- GSM began Feb 2017, per the GSM draft, and the PIT feed Oct 2020, per the AMSS note. The anchor study starts Apr 2022.

### Inferences
- **Short-history streams cannot pass a 2008-25 gate.** Decide whether event streams get a shorter gate with lower size, or stay out.
- **Filters are core changes.** They spend core trials, so run them as one pre-registered bundle under GT11.
- **Small-book niches shrink as capital grows.** Tender value peaks between about Rs 3 lakh and Rs 30 lakh, by my estimate, and fades beyond that. It is worth building only if the ladder stays in that range for years.

### Gaps
- Whether a sleeve with hand-placed tenders fits the automated order path is open.
