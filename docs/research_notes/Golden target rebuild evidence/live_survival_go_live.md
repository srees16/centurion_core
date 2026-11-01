# What makes systematic retail or small-fund trading systems fail or survive at go-live (Centurion, India rules to 1 Nov 2026)

Evidence grades: **[PRIMARY]** regulator, exchange or company filing, text read directly. **[BROKER]** Zerodha's own docs, terms or staff, read directly. **[REPORTED]** press, law-firm or blog relay of a rule or event. **[EXCERPT]** seen only in a search summary, page not opened. **[ACAD]** paper. **[PROJECT]** Centurion's own docs, read for this note. **[CALC]** my calculation, run in Python this session. **[OPINION]** my judgement.

Limits on this note: nsearchives.nseindia.com timed out on every fetch, so **no NSE circular was read directly**; NSE rules come through SEBI text, a sister exchange's circular (MSE) or relays, and are marked. Sources behind 403 pages were not opened. The Finance Act 2026 and Income-tax Act 2025 texts were not read; tax items are relays. The earlier notes' findings are built on, not repeated.

## What must be true so the system does not blow up

### Takeaway
A small unlevered long-only book is unlikely to die from a low Sharpe in year one. It is far more likely to die from an order bug, a stop that was never live, a limit nobody enforced, a broker or login dependency, or leverage added after a good run. Thirteen conditions follow; the four least covered today are a session order and notional cap far below 10 per second, replay tests of the regime exit and drawdown rule, a pre-registered secondary static IP, and an after-tax hurdle.

### Cited Findings
- Knight had no pre-submission check comparing orders leaving the router with orders entered, and 97 automated emails flagged the error before the open without action [PRIMARY] — [SEC release 2013-222](https://www.sec.gov/news/press-release/2013-222).
- Credit Suisse's limits fired for months: potential exposure exceeded 10x its limit by April 2020 and 25x by September 2020, with no deadline set after the September 2020 risk meeting [PRIMARY] — [Paul Weiss report, SEC exhibit](https://www.sec.gov/Archives/edgar/data/1159510/000137036821000064/a210729-ex992.htm).
- LTCM's models, built on recent prices, underestimated shock size and cross-market correlation, and it was diversified by market but not by strategy [PRIMARY] — [President's Working Group, April 1999](https://www.cftc.gov/sites/default/files/tm/tmhedgefundreport.htm).
- Deployed paper after 31 sessions shows regime exit 0, drawdown rule 0 [PROJECT] — `docs/MEMORY_TRACKER.txt`.

### Inferences
[OPINION, status from the project docs read here, not a code audit]

| # | Must be true | Failure it stops | Status |
|---|---|---|---|
| 1 | A hard session cap on order count and rupee notional, set near expected activity. 340 trades a year is about 1.35 a session, so a cap of, say, 30 orders and 60% of book turnover a session binds long before 10 per second does | Knight | 10 per second is a broker ceiling, not a safety control. A bug at 10 per second can send the whole universe in under a minute. Verify a tighter cap exists |
| 2 | Intended orders reconciled to broker orders before release, and broker holdings, GTTs and cash reconciled to the ledger every evening | Knight, silent lost stops | Reconciliation alerts nightly and does not block GO. Make a break block the next session's buys |
| 3 | Every open position has a live GTT, or a logged reason it has none, checked daily | Corporate-action cancels, one-shot triggers, fired-but-unfilled | Verify |
| 4 | Stops are treated as a partial tool: gaps, circuits and halts modelled as open-or-worse or no fill | Adani, Vakrangee, 2020 | Earlier note's point; see stress section |
| 5 | Kill switch semantics written down: what it stops, what stays live, and that Zerodha's segment switch is not used in a crash | Zerodha's switch makes triggered GTTs reject | Centurion's flag refuses new buys, which is the right design. Verify it also cancels unreleased AMOs |
| 6 | A limit breach is a rule that executes, not an email to discuss | Archegos, Knight | Step-down is automatic, kill is "your decision". Keep it, but set a response deadline |
| 7 | No leverage until the gated 12+ months of live tracking, and that gate is not relaxed after a good year | LTCM, Amaranth, Archegos | GT8 rule exists. Note the rule's 12 months cannot prove a Sharpe (see power section) |
| 8 | Liquidity sized for exit, not entry: position as a share of average daily volume, tighter for 5% and 2% band names | Vakrangee, Archegos | Verify a days-to-exit cap at Rs 30 lakh |
| 9 | Daily ASM, GSM and price-band screen before entry and at each hold | Adani, Vakrangee | Verify |
| 10 | Regime exit and drawdown rule exercised end to end on replayed history through the real order path | Paths never run in paper | Open. Waiting will not do it: chance of a 20% drawdown in the next 29 sessions is about 0% |
| 11 | No single point of failure in the order path: primary and secondary static IP registered now, login fallback, manual sell runbook | Broker, VM or GitHub failure | One Oracle VM. IP change is limited to one a week, so a second IP must already be registered |
| 12 | Software freeze rule: no deploy on a release day, one change at a time, rollback tested, unused code paths removed from the live route | Knight's dormant function | The project's canary and G4 count help. Add a freeze |
| 13 | Decision metrics are after tax and after fixed costs, against a passive holder | Running a book that loses to passive for years | Open, see tax section |

Also fix before go-live: `docs/paper_trade_strategy.md` section 7 still shows a first rung of Rs 6 lakh, funding of Rs 6 lakh, and "Realistic expectation after the haircut: Sharpe ~1.0, CAGR ~20%" with "Hold the live book to Sharpe >= 1.0". The tracker (GT2I) has the Rs 3 lakh rung, and the honest base is about 11-12.4% and 0.43-0.54. A Sharpe 1.0 hold-to line would be missed in about 66% of years even by a true Sharpe of 0.6 [CALC]. It should go, or be labelled aspiration.

### Gaps
- No published study was found ranking the causes of small systematic book failure. The list above is case-derived.
- I did not audit the code for items 1-3, 5, 8, 9 and 12. The status column reflects documents only.

## Operational blow-ups: what failed and which control would have caught it

### Takeaway
Global cases share three patterns: concentration or leverage meeting a liquidity squeeze, models calibrated to calm data, and alarms that fired without action. Indian cases add a local failure set: circuit-locked exits, surveillance rules that change mid-event, shares held through a broker, and outages at the open. A single liquid stop covers almost none of them.

### Cited Findings
**Global**

| Case | What failed | Figures | Source |
|---|---|---|---|
| Knight, 1 Aug 2012 [PRIMARY] | A dormant function reactivated by a bad deploy. The router could not recognise fills | About 4 million orders for 212 customer orders in 45 minutes; 397 million shares; over $460 million lost; $12 million penalty | [SEC](https://www.sec.gov/news/press-release/2013-222) |
| Knight, controls [PRIMARY] | No order comparison check; capital thresholds could not stop orders; the receiving account was not linked to exposure controls; weak deployment testing; thin incident procedures; prior incidents not root-caused | | [SEC](https://www.sec.gov/news/press-release/2013-222) |
| LTCM, 1998 [PRIMARY] | Over 25:1 balance-sheet leverage; returned $2.7 billion at end-1997 without cutting positions, so capital fell 36%; spreads widened together after 17 Aug 1998 | August loss $1.8 billion; capital $2.3 billion; about $3.6 billion consortium | [PWG report](https://www.cftc.gov/sites/default/files/tm/tmhedgefundreport.htm) |
| Archegos, 2021 [PRIMARY] | Static margin, swap margin cut to 7.5% and averaging 5.9%; top five longs 175% of NAV; several positions needing 2.5-10 days of volume; one $3.3 billion position over 8% of float | Credit Suisse lost about $5.5 billion | [Paul Weiss](https://www.sec.gov/Archives/edgar/data/1159510/000137036821000064/a210729-ex992.htm) |
| Archegos, inaction [PRIMARY] | The right to call margin on three days' notice was never used; the report calls the protections "illusory". Estimated liquidation time had risen from "a few days" to two weeks to a month | | [Paul Weiss](https://www.sec.gov/Archives/edgar/data/1159510/000137036821000064/a210729-ex992.htm) |
| Amaranth, 2006 [REPORTED] | Lost $6.4 billion, held about 40% of winter 2006-07 natural gas futures at one point | Senate report summary | [NGI](https://www.naturalgasintel.com/senate-panel-seeks-to-curb-speculation-in-natural-gas-markets/) |
| Amaranth, speed [REPORTED] | "Roughly $5 billion in a week"; assets from $9 billion to about $4.5 billion. Exit from the fund itself traded at 10-40 cents on the dollar | | [WSJ copy](https://kellogg.northwestern.edu/faculty/mcdonald/ftp/common/articles/wsj_092006_amaranth2.html) |
| Amaranth, gaming limits [EXCERPT] | Exceeded NYMEX and CFTC levels, then moved positions to ICE when told to reduce | Senator's statement | [Coleman statement](https://webarchive.library.unt.edu/eot2008/20080920001326/http://hsgac.senate.gov/public/_files/OPENINGColemanJuly907.pdf) |
| Aug 2007 quant quake [ACAD, EXCERPT] | Rapid unwind of market-neutral books spread by portfolio overlap; worst on 7-9 Aug; press cited -5% to -30% for the month; simulated strategy rebounded by most of the loss. The authors call the conclusions speculative | | [Khandani and Lo, NBER](https://www.nber.org/papers/w14465), [Lo site](https://web.mit.edu/Alo/www/Papers/august07.html) |

**India**

| Case | What failed | Figures and grade | Source |
|---|---|---|---|
| Jane Street, 2025 | Interim, disputed, not a final finding. Alleged index manipulation on 18 expiry days | Earlier note has the detail. SAT reserved its order on document disclosure on 21 Oct 2026; I found nothing dated November [REPORTED] | [Newsbytes](https://www.newsbytesapp.com/news/business/sat-reserves-order-in-jane-street-document-disclosure-case/story) |
| Jane Street, aftermath | Volume fell at once: about 21% at the next NSE expiry; measures differ, and one Nifty read matched the year's average [REPORTED] | | [Angel One](https://www.angelone.in/news/market-updates/nse-derivatives-turnover-slumps-21-post-jane-street-ban-underscoring-market-concentration), [Business Today](https://www.businesstoday.in/markets/stocks/story/options-trading-india-falls-jane-street-ban-sebi-crackdown-484189-2025-07-11) |
| Karvy, 2019 | Broker pledged client shares for its own borrowing | About Rs 2,300 crore of over 95,000 clients' securities [PRIMARY, parliamentary reply seen as EXCERPT] | [Lok Sabha reply](https://eparlib.nic.in/bitstream/123456789/991819/1/AU2112.pdf) |
| Karvy, outcome | Nearly 90% got securities back after SEBI action [EXCERPT]. Expelled by NSE and BSE Nov 2020, registration cancelled May 2023 [REPORTED] | | [Report](https://jpmc.iiit.ac.in/static/news_pages/92.html), [Moneylife](https://www.moneylife.in/article/karvy-stock-broking-pledged-client-securities-diverted-funds-to-group-companies-sebi-cancels-ksbls-registration/70963.html) |
| IL&FS, 2018 | Default on a Rs 1,000 crore SIDBI loan; panic moved to NBFC and housing-finance shares and debt funds. DHFL fell 42% on 21 Sep 2018; another source says as much as 60% in September [EXCERPT, conflict] | Nifty six-month low 11 Oct | [Quartz](https://qz.com/india/1860466), [Scroll](https://scroll.in/article/926444/scroll-explainer-what-happened-to-finance-firm-dhfl-and-what-that-means-for-the-indian-economy), [Kuvera](https://kuvera.in/blog/one-year-since-ilfs-meltdown-and-the-latest-pmc-bank-crisis/) |
| 4 June 2024 | Nifty -5.93% to 21,884.5, worst day since March 2020. BSE SmallCap -6.79%, MidCap -8.07%. VIX close conflicts, 25.89 vs 31.71 peak [EXCERPT] | No lower-circuit detail found | [Angel One](https://www.angelone.in/news/share-market/election-day-chaos-indian-stock-market-crashes-to-record-lows), [IIFL](https://www.indiainfoline.com/news/markets/indian-benchmark-indices-plunge-over-5-on-vote-counting-day) |
| Adani, Jan-Feb 2023 | Lower circuits on consecutive days, then exchanges cut bands from 20% to 10% and then 5%, ASM margins applied, two names already in F&O ban [REPORTED]. Five of nine group names hit lower circuits on 6 Feb [EXCERPT] | | [Business Standard](https://bsppd.business-standard.com/amp/article/markets/stock-exchanges-bse-nse-lower-circuit-limits-on-adani-group-stocks-123013001356_1.html), [Outlook Business](https://www.outlookbusiness.com/amp/story/news/nse-lowers-price-band-on-adani-green-and-adani-transmission-after-hindenburg-issue-report-news-260040), [Business Standard](https://bsppd.business-standard.com/amp/article/markets/three-adani-group-stocks-put-in-additional-surveillance-measure-list-123020201675_1.html) |
| Vakrangee, 2018 | Eight straight lower circuits, -34% from Rs 286, no buyers, 12.34 million shares pending sale, after ASM was applied. Earlier 20% lower circuits with no buyers [REPORTED] | | [Business Standard](https://business-standard.com/amp/article/markets/vakrangee-locked-lower-circuit-for-eight-straight-day-falls-34-in-8-days-118040400398_1.html), [Business Standard](https://business-standard.com/article/markets/vakrangee-tanks-20-for-second-day-investors-lose-rs-226-billion-in-4-days-118020100182_1.html) |
| NSE outage, 24 Feb 2021 | Halted about 11:40, shut until about 15:30, with close extended to 17:00; BSE traded normally. SEBI's notice alleged NSE was not ready to move to its DR site [REPORTED] | NSE and NCL settled for Rs 72.64 crore, no admission | [Angel One](https://www.angelone.in/blog/all-you-need-to-know-about-the-nse-trading-outage), [Moneylife](https://moneylife.in/article/nse-and-ncl-pay-rs7264-crore-to-settle-24-february-2021-trading-halt-issue-with-sebi/71153.html), [Outlook Business](https://www.outlookbusiness.com/news/nse-arm-ncl-settle-2021-s-trading-glitch-case-with-sebi-pay-rs-72-cr-news-296631) |
| Zerodha, 3 Feb 2026 | Forced logouts, failed logins, stale prices, wrong portfolios, stuck orders on web and mobile, on a day Nifty rose up to 4.7%. Cause unconfirmed. API and GTT effects not reported [REPORTED] | | [NiftyTrader](https://www.niftytrader.in/markets/?p=20681) |
| Zerodha, pattern | Four outages in four months in 2024; 6 Oct 2025; 10 Sep 2026 with a BSE issue [REPORTED] | | [Moneylife](https://moneylife.in/article/zerodhas-kite-app-goes-down-for-4th-time-in-4-straight-months/73240.html), [IANS](https://ianslive.in/zerodha-faces-outage-users-report-losses-amid-glitches--20251006133931), [Free Press Journal](https://www.freepressjournal.in/business/bse-zerodha-glitches-disrupt-fo-orders-traders-flag-price-update-and-execution-issues) |
| Closing auction, day one, 3 Aug 2026 | Nifty jumped about 200 points on one-sided orders, and Goldman measured median auction volumes at 1.7% against 16.3% in the prior month [REPORTED] | Volumes later grew: Rs 39,718 crore on 31 Aug | [Devdiscourse](https://www.devdiscourse.com/article/3959300-niftys-sharp-closing-move-on-monday-reflects-lack-of-buyers-and-sellers-at-right-time-not-flaw-in-new-closing-auction-syst), [Tribune](https://www.tribuneindia.com/news/business/closing-auction-session-sees-muted-participation-on-day-1-volumes-likely-to-pick-up-goldman-sachs/), [newKerala](https://www.newkerala.com/news/a/nse-closing-auction-sees-rs-39718-crore-turnover-488.htm) |

**Broker terms and protection**
- GTT: "does not assure execution"; no liability for non-execution or lost opportunity; Zerodha may cancel a GTT on an ex-date with a day's notice; triggered orders outside circuit limits are not placed or may be rejected [BROKER] — [GTT terms](https://zerodha.com/tos/gtt).
- A triggered GTT can now place a market order with price protection. The unfilled balance rests as a limit at the range edge [BROKER] — [GTT terms](https://zerodha.com/tos/gtt). This corrects the earlier note, which saw no market-order fallback.
- Zerodha's Kill Switch: the segment turns off in about 5 minutes, open orders and positions should be closed first, re-enable only after 12 hours, and a GTT that triggers while disabled is rejected [BROKER] — [Kill switch](https://support.zerodha.com/category/console/profile/killswitch/articles/what-is-the-kill-switch), [GTT rejections](https://support.zerodha.com/category/trading-and-markets/charts-and-orders/gtt/articles/why-was-my-sell-gtt-order-rejected).
- Kite Backup lets you exit via WhatsApp but replaces only the front end, so it fails if backend servers fail [EXCERPT] — [Zerodha](https://zerodha.com/z-connect/updates/introducing-kite-backup-a-whatsapp-based-emergency-mode), [support](https://support.zerodha.com/category/trading-and-markets/general-kite/others-kite/articles/kite-backup).
- NSE investor protection ceiling is Rs 35 lakh per investor per defaulter, raised from Rs 25 lakh in 2024; above that a civil suit is the route [REPORTED, BROKER excerpt] — [Moneylife](https://www.moneylife.in/article/nse-increases-investor-claim-amount-to-rs35-lakh-from-rs25-lakh-per-claim/74911.html), [Zerodha](https://support.zerodha.com/category/trading-and-markets/trading-faqs/general/articles/what-happens-to-funds-and-holdings-if-stockbroker-defaults).
- Client clear credit balances must be upstreamed to clearing corporations at end of day, only as cash, FD lien or overnight-fund pledge [REPORTED] — [Business Standard](https://www.business-standard.com/amp/markets/news/sebi-makes-rule-for-upstreaming-of-client-funds-by-brokers-to-clear-corps-123060801106_1.html).

### Inferences
[OPINION, mapping each case to a Centurion control]

| Case | Lesson for an unlevered Rs 3-30 lakh book | Control |
|---|---|---|
| Knight | Rate limits do not bound loss. Dormant code on the live path is a hazard | Session order and notional caps; intended-vs-sent check; alert that blocks instead of emailing; freeze on release days |
| LTCM | Taking capital out without cutting risk raises leverage. Recent-data models miss joint moves. Many streams can be one bet | Stress on 2008, 2018-20 and 2020, not recent years; measure streams' joint behaviour in stress. Withdrawing tax money shrinks the book: resize rather than run lean |
| Archegos | Exposure as days of volume is the right metric. A breached limit that is discussed is a limit that does not exist | Position cap in days of volume; automatic action with a deadline |
| Amaranth | Fast loss needs concentration plus forced selling. Moving a position to dodge a limit is a warning | Sector and single-name caps; no leverage. Any manual F&O (MO2) is where this risk re-enters |
| Aug 2007 | Crowding hurts without a news cause, and selling at the low locks in a loss that reversed | Do not kill on a sharp factor loss alone. Check overlap with Nifty200 Momentum 30 and its Jun/Dec rebalances |
| Jane Street | A profit pool can vanish overnight by regulatory action. Not a template for Centurion: order size and horizon differ | Treat any expiry-day or index-reversal idea as high regulatory risk; any F&O stream needs a "rule removed" scenario |
| Karvy | Shares in your own demat were still pledged by the broker | Monthly CDSL statement vs Kite vs ledger; keep idle cash low or in LIQUIDCASE units; keep the book under Rs 35 lakh or accept the excess at broker risk |
| IL&FS | Credit and sector contagion spreads faster than quarterly data. AAA was no protection | Sector cap incl. NBFC and housing finance; gilts, not credit, in any debt sleeve |
| 4 June 2024 | A scheduled event can move 6-8% in a day. Pre-open limits may not fill and stops gap | Event calendar review, not an automatic trade: one rule, pre-registered |
| Adani | Bands narrow while you are trying to exit, so liquidity falls in steps | Daily band and ASM screen; treat a band cut as an exit-now signal for review |
| Vakrangee | A locked name has no exit at any price for days. The stop does nothing | Size by band and ADV; book-level assumption of no exit |
| NSE 2021 | Exchange outage is survivable because BSE stayed open | Test selling on BSE; stops are exchange-specific |
| Zerodha outages | Broker failure happens on big days and has been reported as a recurring event | Runbook, Kite Backup registered, phone route tested, no action needed for a one-day miss |
| CAS day one | New auctions are thin at first | Do not trade new market structure until it settles; treat closes after 3 Aug 2026 as a new regime in data checks |

- Karvy's route was client shares pledged by the broker. The client-funds rules have tightened since (upstreaming), but I did not research current pledge safeguards, so the residual risk is unquantified. Monthly reconciliation is cheap either way.
- Zerodha's kill switch is the wrong tool in a crash. It blocks the exits. Centurion's flag, which refuses new buys and leaves stops running, is correct. State in the runbook that the Console switch is not to be used while positions are open.

### Gaps
- Exact Amaranth sale terms not retrieved. The Senate report itself was not read.
- Knight's order text not read; the deployment detail comes from the press release only.
- Share of Indian stocks locked at lower circuit on 23 Mar 2020 and 4 Jun 2024: nothing found.
- No Zerodha post-mortem for 3 Feb 2026, and nothing on whether GTTs fired or API orders worked during any outage. This matters most and remains unknown.
- Whether SEBI's final Jane Street findings will change retail rules: no final order exists.

## What the SEBI retail algo framework and 2026 rules require

### Takeaway
A self-coded book below 10 orders per second needs no algo registration, but must use one API key, a whitelisted static IP, OAuth with 2FA and a daily manual login, and its orders carry a generic exchange algo tag applied through the broker. The 2026 changes that actually touch Centurion are market-structure ones: the pre-open rewrite from 7 Sep 2026 and the closing-auction revision from 7 Dec 2026. Delivery STT, stamp duty, capital gains rates, weekly expiry and T+1 did not change.

### Cited Findings
**SEBI circular of 4 Feb 2025, SEBI/HO/MIRSD/MIRSD-PoD/P/CIR/2025/0000013, read directly [PRIMARY]** — [SEBI](https://www.sebi.gov.in/sebi_data/attachdocs/feb-2025/1738665456458.pdf)
- Brokers are principal; algo providers are their agents.
- All algo orders through API extended to providers carry an exchange-issued unique identifier.
- Algos developed by retail investors themselves are registered through the broker "only if they cross the specified order per second threshold". A registered algo may be used for family (self, spouse, dependent children, dependent parents) "but not for other investors".
- Brokers shall not permit open APIs, must give access only through a unique vendor-client API key and whitelisted static IP, use OAuth only, and authenticate through two-factor authentication.
- Exchanges keep a kill switch for orders from an algo ID, defined as an emergency function expected to halt trading on pre-defined conditions.
- Black-box providers must register as Research Analysts. White-box logic is disclosed and replicable.
- Brokers are solely responsible for algo grievances and for monitoring APIs.

**Timeline**
- Applicable from 1 Apr 2026 for all brokers, by circular of 30 Sep 2025 [PRIMARY, read for the earlier note] — [SEBI](https://www.sebi.gov.in/sebi_data/attachdocs/sep-2025/1759232056254.pdf). No later algo change turned up in searches to Oct 2026 [EXCERPT].

**NSE implementation standards of 5 May 2025 (circular NSE/INVG/67858 not read; relayed)**
- 10 orders per second per exchange or segment; exchanges may change it after market notice; below it no registration, orders tagged algo with a generic ID; multiple keys allowed but unregistered algos must use one designated key; one static IP per client, family excepted; IP updatable weekly; all API sessions logged out daily [REPORTED] — [Groww](https://groww.in/blog/nse-issues-new-compliance-norms-for-retail-algo-trading).
- One broker reads the limit as a rolling second including placements, modifications and cancellations [EXCERPT] — [search result relaying Fyers guidance](https://fyers.in/blog/sebi-algo-trading-rules-and-regulations-in-india/). Not confirmed for Kite.

**Zerodha and Kite Connect**

| Item | Rule | Grade, source |
|---|---|---|
| Static IP | Mandatory for API order placement from 1 Apr 2026; WebSocket and read-only calls from any IP | [BROKER] [Zerodha](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/static-ip) |
| IP count and change | One primary, one optional secondary. "Exchange rules allow just one modification per calendar week" | [BROKER] [Zerodha](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/static-ip) |
| Location | Need not be in India; cloud such as AWS Mumbai suggested | [BROKER, staff May 2025] [Zerodha](https://zerodha.com/z-connect/general/a-comprehensive-overview-of-nses-circular-on-the-new-retail-algo-trading-framework) |
| Order limits | 10 per second, 400 per minute, 5,000 per day per user or key; 25 modifications per order | [BROKER] [Kite docs](https://kite.trade/docs/connect/v3/exceptions/) |
| Conflict | Staff in May 2025 said 3,000 a day, an internal limit. The docs page is the likely current figure | [BROKER] [Zerodha](https://zerodha.com/z-connect/general/a-comprehensive-overview-of-nses-circular-on-the-new-retail-algo-trading-framework) |
| Over-limit | HTTP 429 per client ID, shared across apps; above 10 per second requires exchange registration | [BROKER] [Forum](https://kite.trade/forum/discussion/comment/52171) |
| Market orders | Non-zero market protection required on MARKET and SL-M; limit and SL orders do not need it | [BROKER] [Forum](https://kite.trade/forum/discussion/comment/52171) |
| Login | "users must log in manually at least once each day… we do not recommend automating the Kite Connect login flow" (Nivas, 27 Jul, year not shown; thread IDs suggest 2026) | [BROKER] [Forum](https://kite.trade/forum/discussion/16130/feature-request-api-based-totp-login-for-kite-connect-no-browser-redirect) |
| AMO window | Equity AMOs 16:00 to 08:58; market and limit sent at 09:00; IOC, SL, DQ at 09:15; **ETFs at 09:15** | [BROKER] [Zerodha](https://support.zerodha.com/category/trading-and-markets/charts-and-orders/order/articles/what-is-amo-and-when-can-we-place-it) |
| AMO circuit | AMO priced outside today's band is refused until NSE updates limits at 17:00 | [BROKER, excerpt] [Zerodha](https://support.zerodha.com/category/trading-and-markets/charts-and-orders/order/articles/unable-to-place-the-amo-for-the-price-outside-of-today-s-circuit-limits) |
| Plan cost | Connect Rs 500 a month, Personal free | [BROKER] [Charges](https://zerodha.com/charges/) |

**Technical glitch framework, SEBI circular HO/38/44/12(1)2026-MIRSD-TPD1 of 9 Jan 2026, read directly [PRIMARY]** — [SEBI](https://www.sebi.gov.in/sebi_data/attachdocs/jan-2026/1767957683485.pdf)
- Applies to brokers with over 10,000 clients offering internet trading. A glitch is a malfunction of five contiguous minutes or more in login, order placement or risk functions.
- Brokers inform clients and exchanges within two hours; preliminary report T+1; root cause in 14 working days; exchanges publish incidents.
- **Excluded from the definition: disruption at cloud or other global technology providers, and technical issues at exchanges.** No disincentive applies where one of web or mobile fails and the other works.
- It cites the SEBI (Stock Brokers) Regulations 2026, confirming they have replaced the 1992 regulations.

**Market structure**
- Pre-open from 7 Sep 2026 [PRIMARY, sister exchange MSE; NSE's own circulars not reachable]: entry for market and limit 09:00-09:05; limit only 09:05 to a random close in the last two minutes before 09:10; matching to 09:12; buffer to 09:15. SL, IOC and DQ barred. "Algorithmic market orders shall be allowed only during the initial five minutes." Pre-open trades cannot be cancelled. **Market orders have execution priority over limit orders** — [MSE circular 19531, 10 Aug 2026](https://www.msei.in/SX-Content/Circulars/2026/August/Circular-19531.pdf).
- Closing auction: SEBI circular of 16 Jan 2026 began it on 3 Aug 2026 for F&O stocks only at 15:15-15:35, ±3% of a reference price; non-F&O stocks keep the 15:00-15:30 VWAP [BROKER staff, REPORTED] — [Zerodha bulletin](https://zerodha.com/marketintel/bulletin/439989/introduction-of-closing-auction-session-cas-in-the-equity-cash-segment-and-certain-modifications-in-the-pre-open-auction-session), [TradingQnA](https://tradingqna.com/t/sebi-introduces-closing-auction-session-cas-for-f-o-stocks-and-revises-pre-open-session/190558).
- **Revised from 7 Dec 2026** by SEBI's October 2026 circular: continuous trading runs to 15:30 for all stocks; auction 15:30-15:40, market and limit 15:31-15:36, then limit only, random close 15:38-15:40; stock and index F&O to 15:45, expiring index contracts stop at 15:30; final F&O settlement on the 15:00-15:30 VWAP [BROKER] — [Zerodha bulletin, 16 Oct 2026](https://zerodha.com/marketintel/bulletin/461085/changes-to-closing-auction-session-cas-from-december-7-2026). A second date, 8 Feb 2027, covers pending limit and iceberg orders [EXCERPT] — [Bajaj Broking](https://www.bajajbroking.in/share-market-news/sebi-cas-rules-2026-timings-vwap-and-f-and-o-changes-explained). The tracker's CAS1 entry agrees.
- Price bands: 2%, 5% or 10%, no fixed band for derivative stocks with a 10% operating range, 20% for the rest [REPORTED via NSE excerpts] — [NSE](https://www.nseindia.com/products-services/equity-market-price-bands), [NSE April 2026 mirror](https://rhnvrm.github.io/stock-market-circulars/circulars/nse/2026/nse-2026-04-07-b7f93c9c96e0244b-changes-in-price-bands-from-april-08-2026). That April circular listed 23 names at 2%, 367 at 5%, 170 at 10% and 2,386 at 20%.
- GSM trade-for-trade stage: band 5% or lower, 50% surveillance deposit from buyers [BROKER, excerpt] — [Zerodha](https://support.zerodha.com/category/trading-and-markets/trading-faqs/trading-categories-and-groups/articles/what-does-gsm-mean).

**Taxes, charges and F&O**
- Finance Act 2026 assented 30 Mar 2026. STT from 1 Apr 2026: futures 0.02% to 0.05%, options premium 0.10% to 0.15%, options exercise 0.125% to 0.15% [REPORTED, NSE circular FATAX73524 seen as excerpt] — [NSE mirror](https://rhnvrm.github.io/stock-market-circulars/circulars/nse/2026/nse-2026-03-31-2f6e7cf0d5f77d5f-changes-in-rates-of-securities-transaction-tax), [Finnovate](https://www.finnovate.in/learn/blog/budget-2026-stt-hike-fno-trades-explained).
- The I-T Department reportedly said other STT rates are unchanged [EXCERPT] — [Moneylife](https://www.moneylife.in/article/stt-hike-applies-only-to-options-and-futures-it-dept-clarifies/79539.html). Delivery STT is 0.1% both sides, NSE transaction 0.00307%, stamp 0.015% on buy, DP Rs 15.34 per scrip when sold, regardless of quantity, SEBI Rs 10 per crore, GST 18% [BROKER] — [Zerodha charges](https://zerodha.com/charges/).
- Weekly expiry: no change; SEBI's chair said in Feb 2026 no derivatives measures were contemplated. Nifty weekly Tuesday, Sensex Thursday [REPORTED] — [Informist](https://informistmedia.com/MoneyWire/44239/F-O-Market-Not-contemplating-any-measures-for-F-O-mkt-for-now-says-SEBI-Pandey), [Share.Market](https://www.share.market/buzz/insights/weekly-expiry-days-in-indian-fo-markets/).
- T+0: mandatory QSB implementation was extended without a fixed date around 30 Oct 2025; I found no later date. T+1 remains the working cycle for this book [REPORTED, one report differs on length] — [5paisa](https://www.5paisa.com/news/sebi-indefinitely-extends-deadline-for-optional-t0-settlement-cycle-implementation), [Business Standard](https://www.business-standard.com/markets/news/sebi-extends-timeline-for-qsbs-to-implement-optional-t-0-settlement-cycle-125103001214_1.html).
- From 1 Oct 2026 brokers must show SEBI investor awareness messages, mandatory from 1 Nov 2026 [EXCERPT] — [SEBI](https://www.sebi.gov.in/sebi_data/attachdocs/oct-2026/1790852162769.pdf). A claimed F&O circular SEBI/HO/MRD/TPD/CIR/2026/034 appeared only in a low-quality market blog: unverified.

### Inferences
- **Confirmed as applying to Centurion:** static IP, one key, OAuth and 2FA, broker RMS, generic tagging. Registration does not apply at 340 trades a year. The rule is not a burden at this size; dependency is the issue.
- **Not to be run for others:** the family limit in the circular, plus the tracker's own note on sharing the IP, means any plan beyond family needs a separate look at IA, RA and algo-provider status.
- **Manual login is right.** U23's emailed link matches Zerodha's stated position. Do not automate TOTP to remove the chore; that is the one shortcut with direct staff guidance against it. A missed login should be a planned, harmless event: no new orders, stops still at Zerodha.
- **Secondary IP now.** With one change a week, a dead VM on a Tuesday can leave the book without API orders. Register a second static IP on an independent provider today. Oracle's Always Free terms allow stopping idle instances, with 20% thresholds reported [EXCERPT] — [Oracle docs](https://docs.oracle.com/en-us/iaas/Content/FreeTier/resourceref.htm), [blog](https://oracommit.blogspot.com/2026/02/understanding-oci-always-free-compute.html). A box used only for a tunnel looks idle by those metrics, so check its CPU, memory and network or move it to pay-as-you-go. Standard GitHub-hosted runners have no stable IP, so the tunnel is required, and its host is now production infrastructure [EXCERPT] — [GitHub docs mirror](https://huggingface.co/AbdulElahGwaith/github-docs-arabic-enhanced/blob/main/content/actions/reference/runners/github-hosted-runners.md).
- **ETF timing answers tracker Q5.** On Zerodha's page ETFs go at 09:15 while stock AMOs go at 09:00. So a stock buy AMO cannot be funded by a LIQUIDCASE or gold ETF sell AMO the same morning. Worth confirming on the open ticket, since support's phone answer was already wrong once.
- **Pre-open fill risk is new.** With market orders first, a limit buy can be partly or wholly unfilled at the equilibrium, and the unfilled residual then rests in the day. Unfilled entries will skew toward names that gap up, the adverse-selection case. The backtest's fill assumption should be checked against live fill rate split by open gap. The tracker's 14 Oct 09:09 fill fits the new random close, though its note quotes the old 09:08 timing.
- **Limit price and band:** AMOs placed between 16:00 and 17:00 cannot price beyond today's band, so place after 17:00 if the limit may sit outside it.
- **Data:** opens since 7 Sep 2026 and closes in F&O names since 3 Aug 2026 come from a different mechanism than the 2008-25 data, and closes change again on 7 Dec. Signals built on close or open behaviour have a small structural break exactly at go-live. Tracker CAS1 covers the same-day G4 comparison; the effect on signal quality remains unmeasured.
- **GTT cut-off:** the terms still say 15:15 for F&O stocks and 15:30 for others. That matches the current CAS; from 7 Dec it should move to 15:30. A stop in an F&O stock cannot fire 15:15-15:30 until then. Check when the terms change.
- **Glitch rule gap:** an outage blamed on Cloudflare is outside the framework. Do not count on an RCA, a disincentive or any compensation. A complaint route exists, but no payout was found.
- **F&O STT:** the rise hits MO2's manual F&O and any stock-future leverage, not the core. If L7's 7-point shortfall on real stock futures predates 1 Apr 2026, it should be refreshed with the new cost.

### Gaps
- NSE's 5 May 2025 standards, its FAQ, its pre-open circulars and the October 2026 SEBI CAS circular not read directly. A search summary named NSE/CMTR/74969 (1 Jul 2026) and NCL/CMPT/76060 (31 Aug 2026) as the pre-open references, but I could not tie those numbers to a page I opened. Whether modifications and cancels count toward 10 per second on Kite is unconfirmed.
- Whether daily logout is met by Kite's token expiry at 06:00 is unconfirmed.
- Whether GTT placement through the API is IP-validated: no staff statement found. Treat it as yes.
- What happens to an unmatched AMO limit in the 09:12-09:15 transition under the new rules: not stated in sources read. Support's older line is that it carries to the normal market.
- Whether ETFs join the pre-open: Kotak's list omits them, and Zerodha's 09:15 send suggests not, unconfirmed.
- No November 2026 regulatory item surfaced; searches found nothing after late October.

## How many months of live returns tell skill from luck

### Takeaway
In 12 months no Sharpe test can separate 0.6, 0.8 or 1.0 from zero: even a +2.0 year moves the odds of 0.8 over zero by only about 3.6 to 1. So scale on process evidence and kill on broken mechanism, and treat Sharpe as a multi-year question. The golden 1.4 is the only level that a live record could confirm against zero within about 2-3 years, and matching a passive hurdle takes far longer.

### Cited Findings
- Minimum track record length: the track record needed to reject that the Sharpe is below a threshold at a chosen confidence, in a skew and kurtosis-adjusted form [ACAD, EXCERPT] — [Bailey and López de Prado](https://www.risk.net/journal-risk/2223785/sharpe-ratio-efficient-frontier), [formula relay](https://portfoliooptimizer.io/blog/the-probabilistic-sharpe-ratio-bias-adjustment-confidence-intervals-hypothesis-testing-and-minimum-track-record-length/), [R implementation](https://rdrr.io/cran/PerformanceAnalytics/man/MinTrackRecord.html).
- Triple penance rule: time under water is about three times the time taken to accumulate the quantile loss. Firms giving higher-Sharpe managers looser stop-outs has a theoretical basis. Ignoring serial correlation can understate downside by up to 70% [ACAD, EXCERPT] — [Bailey and López de Prado, Journal of Risk](https://ideas.repec.org/a/rsk/journ4/2435335.html).
- Practitioner kill rules found were blog or forum level only: pause beyond 2x the 95th-percentile backtest drawdown, resampled drawdown bands. No derivation given, and they are not industry standards [OPINION, low authority] — [AIFinHub](https://aifinhub.pages.dev/articles/backtest-to-paper-to-live-playbook/).
- Earlier note carries Philips CUSUM, Rej drawdown bounds and Carver on equity-curve filters — [overfitting_live_decay.md](centurion_core/docs/research_notes/Centurion%20golden%20target%20feasibility/overfitting_live_decay.md).
- Project CUSUM on the paired daily gap: k 0.25, h 12 sd, 0.6% false alarms in 60 sessions, 4.6% in 250, -0.5 sd gap in a median 38 sessions [PROJECT] — tracker GT2I. My simulation reproduces these: 0.6%, 3.9%, 39 sessions.

### Inferences
All [CALC] unless marked. Returns assumed IID, so these are best cases. Positive autocorrelation lengthens them. Higher moments in the MinTRL column are monthly skew -0.5, kurtosis 6, an assumption.

**Time to distinguish from zero**

| True Sharpe | MinTRL 95%, months: normal / fat tails | Fixed test, 5% false, 80% power, years | Fixed test, 20% / 20%, years | SPRT 5% / 20%, expected years if true / if zero | SPRT 20% / 20%, years |
|---|---|---|---|---|---|
| 0.43 (D2) | 178 / 191 | 33.4 | 15.3 | 20.6 / 14.6 | 9.0 |
| 0.6 | 93 / 102 | 17.2 | 7.9 | 10.6 / 7.5 | 4.6 |
| 0.8 | 53 / 61 | 9.7 | 4.4 | 6.0 / 4.2 | 2.6 |
| 1.0 | 35 / 42 | 6.2 | 2.8 | 3.8 / 2.7 | 1.7 |
| 1.4 | 19 / 24 | 3.2 | 1.4 | 1.9 / 1.4 | 0.8 |

MinTRL assumes the observed Sharpe equals the true one, so it carries about 50% power. That is why it is shorter than the fixed test.

**Against a passive hurdle** (assumed passive Sharpe 0.5, an assumption to replace with the NIFTY-and-gold figure; Jobson-Korkie variance)

| True Sharpe | Years, correlation 0.8, 5% / 80% | Years, correlation 0.8, 20% / 20% | Years, correlation 0.5, 5% / 80% |
|---|---|---|---|
| 0.6 | 317 | 145 | 761 |
| 0.8 | 40 | 19 | 92 |
| 1.0 | 17 | 8 | 37 |
| 1.4 | 8 | 3.7 | 15 |

MinTRL against a 0.5 hurdle, read as a level rather than a paired test: 0.8 needs 371 months, 1.0 needs 136, 1.4 needs 44.

**What a 12-month live Sharpe will look like** (standard error about 1)

| True Sharpe | P(year < 0) | P(year < -0.5) | P(year > 1.0) |
|---|---|---|---|
| 0 | 50% | 31% | 16% |
| 0.43 | 33% | 18% | 28% |
| 0.6 | 27% | 14% | 35% |
| 0.8 | 21% | 10% | 42% |
| 1.0 | 16% | 7% | 50% |
| 1.4 | 8% | 3% | 66% |

A -0.5 year is only about 2.3 times likelier at zero skill than at Sharpe 0.6, so it is not a kill signal. At 36 months the same test is 19% vs 3%, a ratio near 7, which starts to mean something.

**Bayesian update.** Prior [OPINION]: 35% no edge (Sharpe 0), 50% modest (0.5), 15% good (1.0). Posterior mean Sharpe:

| Observed | After 1 year | After 3 years | After 5 years |
|---|---|---|---|
| -1.0 | 0.26 (no-edge 54%) | 0.09 (82%) | 0.03 (94%) |
| 0 | 0.35 (40%) | 0.28 (48%) | 0.23 (56%) |
| +0.5 | 0.41 (33%) | 0.42 (29%) | 0.43 (24%) |
| +1.0 | 0.46 (26%) | 0.56 (14%) | 0.64 (6%) |
| +2.0 | 0.57 (16%) | 0.82 (2%) | 0.94 (0%) |

A great year moves the estimate from 0.40 to 0.57, not to 2.0. The rule-book's "12+ months of live tracking" before leverage therefore does not establish the haircut Sharpe of 0.7 that half-Kelly assumes.

**CUSUM on monthly returns**, calibrated to one false alarm in 84 months, as in Philips:

| Book's good Sharpe | Mean months to detect fall to 0 | Fall to half | False alarm within 12 months | Caught within 12 / 24 months if edge is gone |
|---|---|---|---|---|
| 0.6 | 37 | 52 | 5% | 12% / 41% |
| 0.8 | 29 | 45 | 6% | 20% / 53% |
| 1.0 | 24 | 41 | 6% | 27% / 62% |
| 1.4 | 18 | 33 | 8% | 43% / 79% |

**The configured gap CUSUM (k 0.25, h 12) by size of shift**

| Daily gap shift, sd | Median sessions to alarm | Alarm within 60 / 120 / 250 |
|---|---|---|
| 0.1 | 761 | 3% / 9% / 19% |
| 0.2 | 212 | 10% / 28% / 57% |
| 0.25 | 132 | 18% / 45% / 78% |
| 0.3 | 94 | 28% / 64% / 92% |
| 0.5 | 39 | 82% / 99% / 100% |
| 1.0 | 16 | 100% |

| Daily gap sd | Note | 0.25 sd as annual drag | 0.5 sd as annual drag | SE of mean gap, 60 sessions, annualised |
|---|---|---|---|---|
| 15 bp | TE about 2.4%/yr | 9.4% | 18.9% | 4.9% |
| 30 bp | TE about 4.8%/yr | 18.9% | 37.8% | 9.8% |
| 50 bp | TE 8%/yr, G4's limit | 31.5% | 63.0% | 16.3% |

This is the most useful result here. The CUSUM is a good break detector but a poor leak detector. A steady 5-10% a year leak, which would consume the whole live excess return, sits under k unless gap sd is very small. Add a slower check: cumulative mean gap with a confidence band, read at 60, 120 and 250 sessions. Report the actual gap sd, since everything depends on it.

**Drawdown odds, fat-tailed IID, volatility 15%** (assumption; real clustering makes these higher)

| True Sharpe | Horizon | DD ≥ 10% | ≥ 15% | ≥ 20% | ≥ 25% | ≥ 30% | Losing period |
|---|---|---|---|---|---|---|---|
| 0.43 | next 29 sessions | 2.8% | 0.1% | 0% | 0% | 0% | 39% |
| 0.43 | 60 sessions | 12.5% | 1.4% | 0.1% | 0% | 0% | 34% |
| 0.43 | 12 months | 62% | 24% | 7.3% | 1.8% | 0.3% | 20% |
| 0.43 | 36 months | 96% | 63% | 30% | 12% | 4.2% | 6.5% |
| 0.6 | 12 months | 57% | 19% | 5.4% | 1.2% | 0.2% | 15% |
| 1.0 | 12 months | 46% | 13% | 3.0% | 0.5% | 0.1% | 7.5% |

At 18% volatility the 12-month chance of a 20% drawdown rises to 17% at Sharpe 0.43. So the 20/30/35 rule will almost certainly not be seen in paper, is not unlikely within three live years, and must be proved by replay, not by waiting.

**How to decide scale and kill** [OPINION]

| Layer | Question | Evidence and when it can answer | Action |
|---|---|---|---|
| Mechanics | Is the live book the book we tested? | Position match, fills, reject rate, GTT coverage: from day 1 | Breach blocks new buys |
| Implementation | What is the shortfall? | Slippage: about 81 trades in 60 sessions gives SE near 7 bp at 60 bp per-trade sd. Mean gap band at 60, 120, 250 sessions | Scale only inside the band |
| Risk shape | Same risk as tested? | Volatility, beta, turnover, holding period: 60 sessions; volatility persists (Wiecki via earlier note) | 0.7-1.3x already in GO |
| Signal, Tier B | Does the signal still rank stocks? | Monthly rank IC over the whole universe, including unheld names. IC 0.03 over 500 names for 12 months: t up to 2.3 if independent, about 1.0 if effective breadth is a fifth | Watch, no auto action |
| P&L | Is there an edge? | Sharpe vs shadow forecast fan: 3+ years | Review at 36 months against the pre-registered fan |
| Hurdle | Worth doing vs passive? | After-tax comparison, multi-year | Own decision, not a test |

- Tier B is the only real statistical gain available in year one, because it uses names the book never holds. Effective breadth is well below 500 in practice, so it supports but does not prove. If added, pre-register it, since it is a new measured quantity under the protocol.
- A 20-session rung is an operations check. At 1.35 trades a session it gives about 27 trades and slippage SE about 12 bp. Six months to full capital is defensible only because GO has no profit read. It would be sturdier if each rung required events, not just sessions: a minimum count of sells, at least one real GTT trigger, and one AMO partial or unfilled case read correctly.
- Triple penance from the formula as I recall it, unverified: at Sharpe 0.6, the 5% quantile loss builds over about 1.9 years with about 5.6 further years to recover. This agrees with Rej's 5.9 years. Pre-commit now that a 3-year flat period at Sharpe 0.6 is expected behaviour.
- The kill logic in `paper_trade_strategy.md` (drawdown beyond 1.5x backtest MaxDD and NIFTY's) sits beyond the drawdown rule's cash trigger at 35% for a -29.5% MaxDD. So it rarely adds anything for the unlevered book, and that's fine because MaxDD is not a gate.

### Gaps
- No primary institutional document on how quant funds or pods set stop-outs or scale-ups was retrieved. Publication of this seems rare.
- Bailey-López de Prado full texts not read. Formulas come through implementation docs, cross-checked with the expected structure.
- Actual Centurion paired-gap sd, autocorrelation, skew and kurtosis were not read. Replace the assumed volatility and moments with the real series.
- Passive hurdle Sharpe is assumed at 0.5.

## Backtest-to-live decay: updates and what PBO and DSR miss

### Takeaway
Three gaps in the earlier note are now closed: McLean-Pontiff's published 26% and 58% are confirmed, the Harvey-Liu nonlinearity is confirmed from the paper, and Evans's incubation result is added. D2's "0.85×" is in effect a 30% Sharpe haircut because it acts on CAGR, which sits inside the literature band; the wording should be fixed. PBO and DSR are silent on fills, regime tailwind, market-structure change and tail events.

### Cited Findings
- McLean and Pontiff (JF 2016), abstract read directly: portfolio returns are "26% lower out-of-sample and 58% lower post-publication". The out-of-sample decline is an upper bound on data mining, and 32% is attributed to publication-informed trading. Declines are greater for predictors with higher in-sample returns, and returns are higher in illiquid, high idiosyncratic-risk stocks [ACAD] — [paper](https://Www.Gwern.net/doc/economics/2016-mclean.pdf).
- Harvey and Liu (JPM 2015), read directly: across methods and test counts, the haircut is almost always above 50% when annual Sharpe is below 0.4, and at most 25% above 1.0. "The 50% rule of thumb" is too lenient below 0.4 and too harsh above 1.0. They advocate BHY. With 50 trials the Bonferroni haircut was almost 50% for E/P (Sharpe 0.43) but 7.9% for BAB (0.78) [ACAD] — [Harvey and Liu](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF). The exhibit grids are images and were not extracted.
- Evans (JF 2010): incubated funds outperform by 3.5% risk-adjusted, and it disappears after incubation; families pick the best to open; alpha overstated by 0.84% equal-weighted; age filters remove the bias [ACAD, EXCERPT] — [JF listing](https://ideas.repec.org/a/bla/jfinan/v65y2010i4p1581-1611.html), [Darden](https://ideas.darden.virginia.edu/watch-out-for-fairy-tale-returns).
- Suhonen, Falck, Wiecki, Arnott, CPCV and walk-forward: see the earlier note; not repeated.
- D2 in the protocol reads "0.85× for the core". The table shows book 13.2% / 0.62, live estimate about 11% / 0.43 [PROJECT] — `docs/research_protocol.md`, `docs/golden_target_review.md`.

### Inferences
- [CALC] 13.2% × 0.85 = 11.2%, while Sharpe falls 0.62 to 0.43, or 31%. So D2 is 0.85× on CAGR, equal to about 30% on excess return at a 6.5% risk-free rate. The protocol's wording says Sharpe. The number is reasonable, since 0.62 is already walk-forward, and sits in the earlier note's 20-35% band for a further haircut. If anyone applies 0.85× to Sharpe, they get 0.53 and overstate.
- A live Sharpe near 0.43 sits at the edge of Harvey-Liu's "more than 50%" zone. That is a warning: marginal Sharpes are the ones most often false. It supports the earlier note's 0.4-0.5 central case and does not support adding to the core family while it is near its PBO limit.
- McLean-Pontiff's illiquid-stock point cuts against small-cap alpha at scale: returns concentrated where exit is hardest.
- Evans applies by analogy. Paper books chosen after seeing paper results are incubation. If a book is promoted because its paper ran well, its paper record is not out of sample. G4's rule that only the deployed configuration's record counts is the right guard.

**What PBO and DSR do not see** [OPINION]

| Blind spot | Why | Other test |
|---|---|---|
| Fill and cost model errors | Both take the return series as given | Canary fills, gap band |
| Pre-launch regime tailwind | Common to all trials | Sub-period, peer benchmark, launch discount (Liu, earlier note) |
| Market-structure breaks | Sample predates them. New pre-open and CAS are absent in 2008-25 | Post-change data audit |
| Tail and gap events | Sharpe-based; stops assumed to work | Stress section |
| Trial count honesty | N is an input; implicit trials are invisible | Registry plus a range of N |
| Selection among configurations on live evidence | Registry may not count live promotions | Count every promotion as a trial |
| Junk dilution of PBO | Bad trials can lower it | Fix the trial set first (earlier note) |
| Operational failure | Outside the statistics entirely | Must-be-true list |

- CPCV's evidence remains synthetic and walk-forward remains the honest simulation of how the book trades, per the earlier note. For this go-live, neither adds as much as fixing the fill model.
- "Live-like" should remain before tax. Report a separate after-tax line so the hurdle comparison is on the same footing.

### Gaps
- Harvey-Liu's numeric grid for Sharpe 0.4-1.0 not extracted, so no exact haircut for 0.62 at N 61.
- No India-specific backtest-to-live decay study found. Same as the earlier note.
- No paper found that measures PBO or DSR against later live results for retail systems.

## Stress design for a long-only NSE book with gold

### Takeaway
Design stress around failures of exit, not just price: a price-only replay of 2008 or 2020 assumes stops and orders work, and in the worst Indian events they do not. Run four groups: history, correlated gap, locked exit, and infrastructure failure at the open. Under the owner's rule, losses are recorded and not used as a gate; what must pass is behaviour.

### Cited Findings
- Sensex fell 61.5% from January 2008 to March 2009 [EXCERPT] — [Upstox](https://upstox.com/news/market-news/latest-updates/from-covid-to-tariffs-market-crashes-that-shook-india/webstory-179104). I found no confirmation of the Nifty's Jan-Oct 2008 peak-to-trough. Sources also conflict on the 21-22 Jan 2008 halt date [EXCERPT] — [Wikipedia](https://en.wikipedia.org/wiki/Stock_market_crashes_in_India), [ETV Bharat](https://etvbharat.com/english/business/markets/sensex-nifty-today-important-things-to-know/na20200313203224492). Small-cap drawdown of 77% in 2008 is one blog only [EXCERPT] — [MarketCalls](https://www.marketcalls.in/?p=52121).
- 23 Mar 2020: 10% circuit, a 45-minute halt, Nifty closed -12.98% at 7,610.25; about 38% from the January high on closes, 39.6% to the 7,511 intraday low [REPORTED] — [Business Standard](https://www.business-standard.com/amp/article/markets/sensex-hits-10-lower-circuit-trading-halted-for-45-minutes-120032300237_1.html), [Stable Investor](https://stableinvestor.com/2021/03/biggest-nifty-1-day-falls.html), [Angel One](https://angelone.in/blog/nifty-rallies-200-covid-fall-past-4-years).
- Market-wide breaker tiers are 10%, 15% and 20%, followed by a pre-open call auction on restart. Afternoon cut-offs differ across sources, so read SEBI's text before coding them [REPORTED] — [Business Standard](https://www.business-standard.com/amp/article/markets/us-crash-what-is-a-circuit-breaker-in-a-stock-market-how-does-it-work-120030901196_1.html).
- Small-cap index fell 40% from January 2018 to August 2019 [REPORTED] — [Value Research](https://www.valueresearchonline.com/stories/226591/dark-side-small-cap-investing/). The book's own long drawdown was 2018-20, 748 days, a slow grind in calm volatility [PROJECT] — `docs/golden_target_review.md`.
- Gold, budget gap: customs duty cut on 23 Jul 2024 took MCX gold down about 5-6% that day [REPORTED] — [Indian Express via PressReader mirror](https://db.westchesterlibraries.org:2090/india/the-indian-express/20240803/282265260705402), [Upstox](https://upstox.com/news/business-news/economy/gold-prices-drop-by-5000-per-10-gram-post-budget-bring-cheers-to-retail-investors/article-105045).
- Gold, ETF gap: on 30 Jan 2026 Nippon Gold BeES fell nearly 10% while MCX fell up to 4%; on 2 Feb gold ETFs fell around 8% more [REPORTED, excerpt] — [Business Today](https://www.businesstoday.in/personal-finance/investment/story/gold-silver-tumble-up-to-4-on-mcx-after-record-rally-etfs-slide-as-profit-booking-intensifies-513667-2026-01-30), [5paisa](https://www.5paisa.com/news/gold-silver-etfs-slide-up-to-20-as-mcx-prices-extend-sharp-fall). ETFs had traded at premiums that then vanished [REPORTED] — [Upstox](https://upstox.com/news/market-news/commodities/gold-silver-et-fs-fall-on-diwali-as-global-prices-ease-after-record-rally-check-key-details/article-183372), [BasuNivesh, blog](https://www.basunivesh.com/gold-and-silver-etf-nav-trap-why-you-just-lost-24-in-1-day/).
- Gold as diversifier: WGC finds mostly negative correlation with Indian equities since 2006, an industry source; a DCC-GARCH study finds the same in 2008-09, 2020 and 2022 [REPORTED, ACAD] — [WGC](https://www.gold.org/goldhub/research/why-gold-2026-anchor-indian-portfolios), [Asian Online Journals](https://asianonlinejournals.com/index.php/Economy/article/view/7726).
- Liquidity yardstick: AMFI stress tests, small-cap funds 22-60 days to sell 50% in March 2024, and SBI Small Cap 59 days in May 2026, at fund scale [REPORTED] — [Value Research](https://www.valueresearchonline.com/stories/54588/stress-test-results-of-mid-cap-and-small-cap-funds/), [Bajaj Broking](https://www.bajajbroking.in/share-market-news/midcap-and-smallcap-fund-stress-test-may-2026-data).
- Tracker GT2I: Rs 3 lakh holds about 17 names, 85% invested [PROJECT].

### Inferences
[OPINION; arithmetic is CALC, using about 5% per name]

| # | Scenario | Parameters | Book-level read | Behaviour that must pass |
|---|---|---|---|---|
| H1 | 2008 replay | Sensex -61.5% over 14 months; Jan circuit days | Record | Regime exit and 20/30/35 fire on the real path, orders valid |
| H2 | 2018-20 grind | Small caps -40%, 748 days, no vol spike | Record | No tinkering; owner's review pre-written; costs of churn visible |
| H3 | Mar 2020 | -13% day, 45-minute halt, recovery rally | Record, plus the missed rebound | Stops queued during a halt treated as no fill; restart logic safe |
| H4 | 4 Jun 2024 | All names -6 to -8% at once | About -6 to -8% | Mass GTT trigger: no duplicates, no reject storm, evening reconcile clean |
| H5 | Sep 2018 sector | 3 names of one sector, -40% | -6% | Sector cap holds |
| H6 | Aug 2007 style | Factor unwind with no index move, 3-5 days | Record vs Nifty200 Momentum 30 overlap | No kill on factor loss alone |
| G1 | Correlated gap down at open | All 17 names gap -10%, stops skip | -8.5% then whatever follows | Limit-stops refused or unfilled are detected the same day; manual sell list issued |
| L1 | Locked lower circuit | 3 names locked 8 sessions at 5% (-34%) | -5.1%, no exit | Reconcile flags stuck positions; drawdown rule's sells for those names fail safely, not in a loop |
| L2 | Band cut mid-exit | Band 20% to 10% to 5%, ASM on | Extra days | Limit prices recomputed to new band; a modify count stays under 25 per order |
| L3 | GSM trade-for-trade | Holding moves stage | Liquidity falls | Buy blocked; exit path logged |
| A1 | Gold duty cut | Gold -6% on Budget day | Gold weight × -6% | Sells use limit near iNAV |
| A2 | Gold ETF premium collapse | ETF -10%, then -8%, MCX -4% | 20% weight: about -3.6% | Measure ETF vs iNAV, not only MCX |
| A3 | Both fall | Gold and equity down together for a month | Record | Model correlation breaking; do not assume hedge |
| B1 | Zerodha down 09:00-13:00 | AMOs not released; stuck orders; market +4.7% | Missed entries | No double orders on recovery. Idempotent release. Sequence checked against broker order book first |
| B2 | Zerodha down in a -6% day | No sells | Full loss until recovery | Kite Backup and phone route rehearsed; no panic market sell on return |
| B3 | NSE down, BSE up | As Feb 2021 | | Sell route on BSE tested; GTT state re-read when NSE returns, including an extended close |
| B4 | Login missed | No token | No new orders | Book holds; GTTs persist; next-day catch-up not doubled |
| B5 | VM dead, IP change blocked | Oracle stopped | No API orders | Secondary IP path takes over without code change |
| B6 | GitHub cron late or skipped | Cache refusal, known incident | Evening run late | AMO window to 08:58 absorbs it; hard stop at a set time |
| B7 | Data wrong | Stale or missing bhavcopy, bad corporate action | Bad orders | Sanity gate blocks release; order count cap binds |
| B8 | Runaway order bug | Loop sends orders | Bounded | Session cap trips; alert; broker order book cancels |
| R1 | Rule change overnight | STT, band, session times, API limit | Cost shift | Cost model and calendar update process, and nothing hard-coded silently |

- **Replay how:** feed recorded days from 2008, 2018 and 2020 into the live code in dry-run, against a stub broker that returns rejections, partials, circuit-locked names and timeouts. This is what the 5 dry runs do not currently prove, since they cover token, tunnel and order building on calm days. One or two "game days" a quarter is plenty.
- **Second broker:** a second account cannot sell Zerodha holdings, because the shares sit in that demat. It helps only for a hedge, which needs F&O approval and size the canary lacks. At Rs 3-12 lakh, accept broker risk; from Rs 21 lakh a NIFTY lot passes MO2, so a pre-agreed hedge route is an option. It is not worth building first.
- **Gold has its own India risks:** duty changes, ETF premium, and a 09:15 send. It needs its own stress rows, not a generic "safe haven" assumption.
- Stress results are recorded, per the rule that MaxDD is never a gate. The pass condition is that nothing behaves in an unplanned way.

### Gaps
- No data on bid-side depth at lower circuit for Indian small caps, or on GTT behaviour in a market-wide halt.
- Gold's March 2020 path in INR, and the Nifty 2008 peak-to-trough, were not confirmed.
- The July 2024 ETF same-day fall was not found. Only MCX figures are cited.
- Actual sector and ADV profile of the deployed book not read, so scenario weights are illustrative.

## After-tax CAGR for an individual running 340 trades a year, FY2026-27

### Takeaway
Tax turns the golden 35% into about 27-27.7% under capital gains or 24.1% as business income at the 30% slab, and the honest 11-12.4% into about 8-9.8%. Because a passive holder defers tax and pays 12.5% at exit, Centurion needs about 14-15% net of costs under capital gains, or about 16% as business income, just to match a 12% passive return after tax. There is no safe harbour for capital-gains treatment on holdings under 12 months, so the business-income case has to be planned for with a chartered accountant.

### Cited Findings
- FY2026-27 rates unchanged by Budget 2026. Listed equity STCG 20%, LTCG 12.5% above Rs 1.25 lakh, for holdings over 12 months [REPORTED, not statute] — [Angel One](https://www.angelone.in/news/taxation/union-budget-2026-ltcg-tax-unchanged-at-12-5-stt-on-commodity-futures-hiked), [Finnovate](https://www.finnovate.in/learn/blog/budget-2026-stt-hike-fno-trades-explained).
- New-regime slabs: nil to Rs 4 lakh, then 5%, 10%, 15%, 20%, 25% in Rs 4 lakh steps, 30% above Rs 24 lakh. 4% cess. Rebate up to Rs 60,000 to Rs 12 lakh income, not available on special-rate gains. Surcharge on these gains capped at 15% [REPORTED, aggregators] — [BankBazaar](https://bankbazaar.com/tax/income-tax-slabs.html), [Business Today](https://www.businesstoday.in/personal-finance/tax/story/income-tax-slabs-fy-2026-27-will-rates-change-from-april-1-under-new-tax-rules-details-523303-2026-04-01).
- Income-tax Act 2025 applies from 1 Apr 2026. 44AB became section 63, with Rs 1 crore and Rs 10 crore thresholds carried over [REPORTED] — [TaxGuru excerpt](https://taxguru.in/?p=1056974), [Taxscan](https://www.taxscan.in/top-stories/section63-new-act-vs-section44ab-old-income-tax-act-low-profit-declarations-now-mandate-audit-1445774), [Zerodha Varsity](https://zerodha.com/varsity/?p=2148).
- **CBDT Circular 6/2016, 29 Feb 2016:** (a) a taxpayer may treat listed shares as stock-in-trade, business income "irrespective of the period of holding"; (b) for shares held over 12 months, a capital-gains stance shall not be disputed, and that stance binds later years; (c) "all other cases" are decided under Instruction 1827 and Circular 4 of 2007 [REPORTED, law-firm relay of the text] — [Legal 500](https://www.legal500.com/intelligence/india/tax/clarification-by-cbdt-on-taxability-of-income-arising-out-of-transfer-of-shares). Circular 4/2007 recognises separate investment and trading portfolios, with holding period among the tests [EXCERPT] — [TMI](https://www.taxmanagementindia.com/web/tmi_blog_details.asp?id=429054). So the undisputed capital-gains path covers only holdings above 12 months, not swing trades.
- Gold ETFs: gains within 12 months at slab, over 12 months 12.5% [REPORTED, blogs, one outlier says 10%] — [Zerodha Fund House](https://www.zerodhafundhouse.com/blog/understanding-etf-taxation-in-india-a-simplified-guide-in-2025/), [IIFL](https://www.iifl.com/index.php/blogs/gold-loan/gold-etf-tax-in-india-2026-post-finance-act-2024-rules).
- SGB exemption now applies only to original subscribers held to maturity, so secondary purchases lose it [EXCERPT] — [Patron Accounting](https://www.patronaccounting.com/blog/itr-capital-gains-notifications-2026-what-changed).
- Buybacks moved back to capital gains from 1 Apr 2026 [REPORTED] — [SCC Online](https://www.scconline.com/blog/?p=390469).
- STT is deductible for a trader's business income but not for capital gains; the new Act may renumber the provision [EXCERPT] — [TMI commentary](https://www.taxtmi.com/tmi_blog_details?id=443545), [TMI manual](https://www.taxmanagementindia.com/visitor/detail_manual.asp?ID=1123), [Court Kutchehry](https://courtkutchehry.com/blog/capital-gains-taxation-brokerage-stt-investment-deductions).
- Project's plan assumes "Effective rate for algo trading: 20% STCG" [PROJECT] — `docs/business_plan.md`.

### Inferences
All [CALC], new regime, all gains short term and realised each year, cess included. Inputs are net of trading costs. The business column is therefore right as shown, since costs including STT are deductible. **The capital-gains column flatters by about 0.35-0.7 points a year:** STT is not deductible there, so the taxed gain exceeds the net profit by the STT paid (1.7-3.4% of book at 8.5-17x turnover, taxed at 20.8%).

**Other income at or above Rs 24 lakh (marginal 30%): 20.8% vs 31.2%**

| Pre-tax CAGR, net of trading costs | Capital gains | Business income |
|---|---|---|
| 11% (D2 live estimate) | 8.7% | 7.6% |
| 12.4% (with LIQUIDCASE) | 9.8% | 8.5% |
| 13% | 10.3% | 8.9% |
| 17.5% (first milestone) | 13.9% | 12.0% |
| 23% | 18.2% | 15.8% |
| 35% (golden) | 27.7% | 24.1% |

A 15-year simulation with volatility 15% and loss carry-forward gives the same retention, about 80% and 69%, so loss years do not change the picture.

**No other income, Rs 30 lakh book**

| Pre-tax | Gain | CG tax | After-tax | Business tax | After-tax |
|---|---|---|---|---|---|
| 11-13% | Rs 3.3-3.9 lakh | 0 | 11-13% | 0 | 11-13% |
| 17.5% | Rs 5.25 lakh | Rs 26,000 | 16.6% | Rs 6,500 | 17.3% |
| 23% | Rs 6.9 lakh | Rs 60,320 | 21.0% | Rs 15,080 | 22.5% |
| 35% | Rs 10.5 lakh | Rs 1,35,200 | 30.5% | Rs 46,800 | 33.4% |

This assumes unused basic exemption can absorb STCG, from memory and not verified. It also assumes no rebate on the gains. Business treatment is better at low income and worse once other income fills the slabs. The owner's other income decides which case applies.

**Passive hurdle after tax** (buy and hold, 13% at exit, Rs 1.25 lakh exemption ignored)

| Passive pre-tax | After tax over 10 years | Centurion pre-tax needed, CG | Needed, business |
|---|---|---|---|
| 10% | 9.1% | 11.5% | 13.2% |
| 12% | 11.0% | 13.9% | 16.0% |
| 14% | 12.9% | 16.3% | 18.7% |

Add about 0.45-0.9 points to the capital-gains column for non-deductible STT.

At the 30% slab, D2's 11-12.4% after tax falls short of a 12% passive holder's 11.0%. This uses the live estimate, not a forecast, and the mix of passive return is itself uncertain. It does change the question for the 17-18% milestone: 17.5% before tax is about 13.2-13.9% after, about 2-3 points over that passive case under capital gains, and about 1 point under business income.

**Trading costs by rung, before slippage and impact** (5% per name; round trip 0.2225% from Zerodha's published rates; 340 read both as orders and as round trips, since the source count is ambiguous)

| Book | Variable, 170 / 340 round trips | DP | Rs 500 a month API | Total, 170 / 340 |
|---|---|---|---|---|
| Rs 3 lakh | 1.89% / 3.78% | 0.87% / 1.74% | 2.00% | 4.8% / 7.5% |
| Rs 6 lakh | 1.89% / 3.78% | 0.43% / 0.87% | 1.00% | 3.3% / 5.7% |
| Rs 12 lakh | 1.89% / 3.78% | 0.22% / 0.43% | 0.50% | 2.6% / 4.7% |
| Rs 21 lakh | 1.89% / 3.78% | 0.12% / 0.25% | 0.29% | 2.3% / 4.3% |
| Rs 30 lakh | 1.89% / 3.78% | 0.09% / 0.17% | 0.20% | 2.2% / 4.2% |

- The fixed drag at the canary is about 2.9-3.7 points against 0.3-0.4 at Rs 30 lakh. Canary P&L will read about 2.6-3.4 points low against Rs 30 lakh for reasons unrelated to skill. G4 should compare costs by component, and the fixed part should be stated when reading canary returns. Check whether the Personal (free) plan meets the need, since it would remove 2 points at Rs 3 lakh; plan contents were not verified.
- Annual sale value at Rs 30 lakh is about Rs 2.6-5.1 crore, under the Rs 10 crore line if that threshold applies, and if delivery turnover is taken as sale value, which is unverified.

**Practical tax plan** [OPINION, needs a chartered accountant]
1. Get a written view before the first full year. At 340 short-hold trades, with automation, business treatment is the risk case; the Circular's undisputed path only covers holdings above 12 months.
2. Choose one treatment and keep it. Keep separate records if any long holdings are to be an investment portfolio.
3. Model business income as the planning case when other income is high. If taxed as business, claim STT, charges, VM, Kite plan and data. A loss year may be useful against other non-salary income, subject to advice.
4. Hold gold over 12 months where the strategy allows: 13% against up to 31.2%. Rebalancing that trades gold within a year costs more than the same equity trade.
5. Avoid buying SGBs in the secondary market for the redemption exemption.
6. Reserve tax in cash. Withdrawals shrink the book each year, so the backtest's compounding overstates live capital growth unless it pays tax.
7. Add an after-tax line next to live-like metrics, and replace "Effective rate 20%" in the business plan with both cases.

### Gaps
- The Finance Act 2026, Income-tax Act 2025 and CBDT circular texts were not read directly. New section numbers for STCG and LTCG were not confirmed.
- Whether delivery sale value is the audit turnover: not found. Whether section 63's presumptive and low-profit limbs bind a delivery trader: sources conflict.
- Circular 4/2007's full list of factors not read, including whether frequency is named.
- Advance-tax timing and interest on capital gains, and loss set-off and carry-forward rules under the new Act, not verified. The simulation assumes the familiar carry-forward.
- Treatment of the Rs 500 plan, VM and data as capital-gains expenses: assumed not deductible.
- Whether unused basic exemption can absorb STCG under the new regime: from memory.
- The meaning of "340 trades" (orders or round trips) in the tracker was not resolved; both are shown.
