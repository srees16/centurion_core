# Leverage due diligence — India (tracker L4)

27 September 2026. Question from decision U5: may the IND book use leverage in
Indian markets, and on what terms? Sources were read on this date; broker
terms change, so re-check them before any funded order.

## Verdict

| Question | Answer |
|---|---|
| Is it permitted? | **Yes**, through the Margin Trading Facility (MTF), for Group I stocks, with the funded shares pledged to the broker. |
| Can the engine operate it? | **Yes, after five account checks** (below). Kite Connect accepts product `MTF`; pledging is automatic once DDPI is active, so no per-order OTP blocks automation. |
| Is it worth it? | **No, not at 14.6% a year.** Each 0.1× of leverage adds 0.5–0.75 CAGR points for about 3 points of MaxDD and −0.05 Sharpe. The deployed book needs about 1.5× to reach 25% CAGR in backtest, at MaxDD −39%. |

**Recommendation: NO-GO for now.** The only backtest combination that reaches
25% CAGR with MaxDD under 30% is the unconfirmed E1 variant at 1.1×. After the
usual live haircut of about 2–3 points, that lands back below 25%. Revisit
only if E1 is confirmed forward (decision U20), and then run E3 with the
constraints in section 6.

## 1. What the rules allow

- **MTF is the only route.** Collateral margin from pledged holdings cannot pay
  for delivery (CNC) purchases; it covers intraday, futures and option
  writing only. A cash-equity book can therefore be levered only through MTF,
  or through derivatives (see section 5).
- **Eligible securities are SEBI's Group I:** mean impact cost at most 1% and
  traded on at least 80% of days over the previous 18 months. About 2,000
  securities qualify. NSE tightened MTF eligibility in July 2024, excluding
  1,010 stocks. The criteria were not verified here because the source was
  paywalled.
- **Funded shares must be pledged** in the depository to the broker's
  "Client Securities under Margin Funding Account" (SEBI circular of 25 Feb
  2020). The client keeps beneficial ownership, dividends and corporate
  actions.

## 2. Zerodha's terms

| Item | Terms |
|---|---|
| Interest | 0.04% a day on the funded amount (₹40 per lakh), from T+1, calendar days including weekends and holidays: 14.6% a year |
| Brokerage | 0.3% or ₹20 per executed order, whichever is lower. Delivery (CNC) stays free |
| Pledge | ₹15 + GST per ISIN per pledge and per unpledge request |
| Leverage | Up to 5×, set per stock. Margin is VaR + 3×ELM for F&O stocks and VaR + 5×ELM for others |
| Eligible stocks | More than 1,300. The list at zerodha.com/mtf-approved-securities is dynamic and could not be fetched |
| Limits | ₹10 crore per stock for Nifty 500 names, ₹6 crore for others, ₹30 crore per account. Not binding at ₹30 lakh |
| Holding period | No maximum |
| Pledging | Automatic on purchase for accounts with DDPI (Zerodha, March 2025). No CDSL OTP per purchase. DDPI is mandatory for MTF |
| Collateral | MTF-pledged shares give no collateral margin |
| Mark-to-market | Collected daily on losing positions |
| Forced square-off | When losses exceed 80% of the funded amount, even with free cash in the account; ₹50 + GST per order. A margin shortfall not met within T+5 days lets Zerodha liquidate pledged shares |
| Stops | GTT orders on MTF holdings are supported since June 2025 (Kite web) |
| Conversion | MTF holdings can be converted to CNC if cash covers the funded value |

## 3. Can the engine drive it?

- **Orders:** the Kite Connect v3 order API lists `product = MTF`. The
  installed Python client (kiteconnect 5.1.0) has no `PRODUCT_MTF` constant;
  Zerodha said in January 2025 that it would be added. The client passes the
  product string through unchanged, so `product="MTF"` should work. **Untested.**
- **After-market orders:** the engine decides after the close and places
  orders with variety `amo` (L3). No source says whether MTF orders can be
  placed as AMO. **Untested.**
- **Stops:** GTT on MTF holdings works on Kite web. The API's GTT payload
  passes the product through, so `MTF` should be accepted. **Untested.**
- **Pledging:** automatic with DDPI, so nothing to automate.
- **Margin calls:** the engine has no margin-call handling. A levered book
  must keep cash, or plan sells, to meet mark-to-market calls. Otherwise
  Zerodha square-offs will sell names the engine still wants to hold.

## 4. Eligibility of the book's names

As of the last store session (23 Sep 2026) the deployed engine targets 20
names: ANANDRATHI, DIVISLAB, LAURUSLABS, KTKBANK, SONACOMS, SAILIFE, GLAND,
MCX, TFCILTD, SANSERA, REDINGTON, PAYTM, SYRMA, PAISALO, APARINDS, STLTECH,
ATHERENERG, DIACABS, WELCORP, CUPID.

All of them trade ₹31–657 crore a day by median, and each traded on at least
91% of sessions over the past 18 months. That fits Group I, but eligibility
and the per-stock margin must be checked name by name on Zerodha's list.
Non-eligible names would have to be bought with cash, which lowers the
achievable leverage.

## 5. Economics

Recorded daily returns, 2013–2025, scaled for leverage L. The cost is 0.04%
per calendar day on the funded part, plus 0.4% a year for MTF brokerage and
pledge fees. Margin calls are not modelled, and the figures are backtest,
before any live haircut.

| Book | L | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| Deployed (B1 baseline) | 1.0 | 22.75% | 1.14 | −24.7% | 0.92 |
| | 1.2 | 23.85% | 1.03 | −30.9% | 0.77 |
| | 1.5 | 25.33% | 0.92 | −39.3% | 0.64 |
| E1 refill variant | 1.0 | 24.68% | 1.21 | −25.5% | 0.97 |
| | 1.1 | 25.43% | 1.15 | −28.6% | 0.89 |
| | 1.2 | 26.15% | 1.11 | −32.0% | 0.82 |

Funding at 14.6% is far above the 6.5% risk-free rate. So each unit of
leverage earns only the book's return minus 14.6%, while adding the book's
full volatility. That is why Sharpe falls with every step.

- **NIFTY futures overlay:** closed in A1. It adds market beta (0.40), not
  edge, and 0.25× already takes MaxDD to −31%.
- **Stock futures:** they cover only the F&O list, and each lot is worth ₹5–15
  lakh. That is too lumpy for a 20-name book at ₹30 lakh.

## 6. If E3 is ever run: what it must model

1. Funding of 0.04% per calendar day on the funded amount, from T+1.
2. ₹20 brokerage per MTF order, and ₹17.70 per pledge and per unpledge per
   stock per day.
3. Funding for eligible names only, each at its own margin (VaR + 3×ELM or
   VaR + 5×ELM). Proxy: 25–40% if the list is unavailable.
4. Daily mark-to-market calls. Forced square-off when losses exceed 80% of
   the funded amount, and liquidation after an unmet shortfall of T+5 days.
   Replay 2018–20 with these rules.
5. No collateral margin from MTF-pledged shares.

The proposed E3 rule, Sharpe within 0.05 of the unlevered book, allows at
most about 1.1× at these terms.

## 7. Account checks before any funded order

1. DDPI is active: Console, then Account, then Segments.
2. MTF is enabled on the account, and the pledge is automatic: place one
   small MTF buy on Kite web and confirm no OTP was asked.
3. The engine's names are on the approved list, with their margins:
   zerodha.com/mtf-approved-securities.
4. An API test with one small order, `product="MTF"`, variety `amo`, then a
   GTT stop on it, then cancel. Record the responses.
5. Tax, with a chartered accountant. Interest on MTF is not deductible
   against capital gains, but is deductible if the activity is assessed as
   business income. At about 500 trades a year and 7× turnover, business
   income is plausible, and that changes the rate applied and what is
   deductible.

## Sources (read 27 Sep 2026)

- SEBI, securities eligible for margin trading (Group I): https://www.sebi.gov.in/sebi_data/commondocs/cirsmd152004_h.html
- NSE, FAQs on the Margin Trading Facility: https://www.nseindia.com/static/trade/members-faqs-margin-trading-facility
- Zerodha, MTF terms and conditions: https://zerodha.com/tos/mtf/
- Zerodha, MTF FAQs: https://support.zerodha.com/category/trading-and-markets/margins/margin-trading-facility/articles/margin-trading-facility-mtf-faqs
- Zerodha, MTF launch (19 Dec 2024): https://zerodha.com/z-connect/featured/introducing-margin-trade-funding-mtf-on-kite
- Zerodha, MTF updates: GTT, limits, conversion (3 Jun 2025): https://zerodha.com/z-connect/featured/mtf-updates
- Zerodha, charges: https://zerodha.com/charges/
- Zerodha on X, MTF auto-pledge without OTP (Mar 2025): https://x.com/zerodhaonline/status/1904791723579965520
- Zerodha, what collateral margin can be used for: https://support.zerodha.com/category/console/portfolio/pledging/articles/will-zerodha-give-me-margin-on-the-shares-i-hold-and-what-can-i-use-my-collateral-margin-for
- Kite Connect v3, orders (product `MTF`): https://kite.trade/docs/connect/v3/orders/
- Kite Connect forum, MTF in pykiteconnect: https://kite.trade/forum/discussion/14723/mtf-orders-using-python-api
- Business Standard, NSE excludes 1,010 stocks from MTF (July 2024): https://www.business-standard.com/markets/news/nse-tightens-margin-funding-rules-excludes-1010-stocks-including-paytm-124071600623_1.html
- Tax: Zerodha Varsity, taxation for traders: https://zerodha.com/varsity/chapter/taxation-for-traders/ ; interest on margin funding against capital gains: https://jainanuragassociates.com/knowledge-center/tax-treatment-of-interest-paid-on-margin-funding-while-calculating-short-term-capital-gain
