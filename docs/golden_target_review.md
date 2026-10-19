# Golden target review: research, audit and plan (GT, 10 Oct 2026)

The owner's golden target: **Sharpe ≥ 1.2, CAGR ≥ 35%, MaxDD ≤ 35%, Calmar ≥ 1,
PBO ≤ 30%**, held live. This review asks what kind of system meets it, how far
the Ind stocks book is from it, and what would close the gap. It skips intraday
and US auto-trading; Centurion is swing and positional on NSE through Kite.

Inputs:
- three web research memos (10 Oct 2026): system design and overfitting
  science; Indian factor evidence; uncorrelated Indian alpha sources;
- an audit of the engine and its 2026-10-10 scorecards (cost model 4, data
  hash v2);
- scoping on the deployed book's daily returns (run
  `20261010T032441064151Z_679cbd0c`).

The scoping is arithmetic on recorded returns, not a trial: nothing was
recorded, and the trial count does not change. Items are tracked as GT<n> in
`docs/MEMORY_TRACKER.txt`; their status lives there.

**Decision (GT0, 10 Oct 2026, the owner's call): the golden target is a
stretch goal, not the plan.** It is restated as Sharpe ≥ 1.4, CAGR ≥ 35%,
MaxDD ≤ 35%, Calmar ≥ 1 and PBO ≤ 30%, all pre-tax. The tracker's TARGET stays
the plan, and the strategy is researched and refined continuously toward the
stretch (GT12, under the protocol in section 7).

The legs fit together at Sharpe ≈ 1.4 and σ ≈ 23% only if drawdowns are held
to about 1.5σ. At this book's 1.8σ, the 35% drawdown cap limits σ to about
19%, and 35% CAGR then needs Sharpe ≈ 1.55. Section 1.2 gives the arithmetic.

## 0. Verdict

**Update, 17 Oct 2026: the honest base (your rule).** It is 2008–25,
walk-forward, cost model 4 and real instruments. The rest of this verdict
uses 2013–25 backtests and reads high.

| 2008–25, walk-forward | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| The book (LN-T28b) | 13.2% | 0.62 | −32.5% | 0.41 |
| Live estimate (D2's haircut) | about 11% | 0.43 | −31.5% | 0.35 |
| + a 25% gilt sleeve (BD1's B3) | 11.7% | 0.61 | −22.9% | 0.51 |
| 1.75× modelled leverage, the rule on | 19.4% | 0.71 | −44.6% | 0.43 |
| 1.75× modelled + a 25% gilt sleeve (a look) | 16.6% | 0.71 | −32.6% | 0.51 |

- **Real stock futures run about 7 points below the model** (L7,
  LN-T28b). They can lever only the book's weaker half, its F&O names (19.5%
  a year against 38.0% for the rest, 2017–25).
- **Bonds are the drawdown lever, not a return stream (BD1).** Gilts earned
  about what cash earned over 2008–25. A 25% sleeve cuts MaxDD by about 10
  points, at every leverage, with Sharpe unchanged. That brings the MaxDD
  leg within reach of a levered book.
- **MCX trend (GT5-B) is the second drawdown lever, provisionally (DG2,
  your decision 19 Oct: drawdown levers judged at their own weight).**
  - At 25% it cuts the live-like walk-forward MaxDD from −29.5% to −20.7%,
    Sharpe +0.03. The gain holds in 84% of bootstrap resamples.
  - It made +18% in 2026 while NIFTY lost 13%.
  - Its 2026 was seen, so paper is the test (DL1).
  - Gilts and MCX-B together (25% each) reach −14.5%, but fail by 0.01 of
    Sharpe on the in-sample backtest book. So far they run as separate
    paper books.
- **The diversifier gate (19 Oct) found no reliable Sharpe lever yet.** A1
  adds +0.16 of Sharpe but loses in crashes. X2 and Y3 are not significant.
- **The gap is Sharpe, as below, and larger than it read.** The honest
  base is about 0.6, not 1.16. Uncorrelated streams add in quadrature, so
  reaching 1.4 needs about 1.26 more: roughly 6–7 independent streams of
  Sharpe 0.5 each. Every stream tested so far failed (section 6, stream
  register).

- **The golden target is a Sharpe problem, not a CAGR problem.**
  - CAGR ≈ rf + Sharpe·σ − σ²/2, and this book's MaxDD runs about 1.8σ.
  - So CAGR ≥ 35% with MaxDD ≤ 35% needs σ ≤ ~19% and a net Sharpe of about
    1.55. Drawdown control that holds MaxDD near 1.5σ lowers the bar to
    about 1.35.
  - The book has a backtest Sharpe of 1.16 and a live estimate of about 1.0.
- **Nothing inside the current engine closes the gap:**
  - futures leverage at 1.75× reaches 35% CAGR at −42% MaxDD;
  - volatility targeting makes this book worse;
  - MTF leverage lowers Sharpe;
  - more tuning raises PBO, which sits at 29.2% against the 30% bar.
- **The route has three parts, in this order:**
  1. Genuinely different return streams, built mostly from data Centurion
     does not have yet. Point-in-time fundamentals would open value and
     quality. MCX futures would open commodity trend.
  2. A cash engine for idle cash, worth about 0.3–0.4 point of CAGR after
     tax: the book is only 12.8% idle on average (GT3, section 6.1).
  3. Futures leverage last. It is capital-gated: a NIFTY or stock-futures lot
     is about Rs 15 lakh of notional, so it is too coarse below about
     Rs 1 crore. It is also evidence-gated: it needs 12+ months of live
     tracking.
- **The odds:**
  - A combined backtest Sharpe of about 1.2–1.3 and a live Sharpe of about
    1.0–1.1 in 2027–28 are plausible. The research memos put the realistic
    gain from new sleeves at +0.05 to +0.3 Sharpe. Section 1.1 shows that
    holding factor-index ETFs beside the book lowers its Sharpe.
  - CAGR ≥ 35% with MaxDD ≤ 35% sustained live needs a live Sharpe of about
    1.4, and the book past Rs 1 crore for futures leverage. That is
    top-decile for any monthly-horizon systematic book.
  - No Indian long-only systematic record shows 35% over 10+ years. The best
    are 17–23% (factor indices, momentum with a regime filter). 35% shows up
    only in 3–5-year windows that start from market lows.
  - **Keep 35% as a stretch goal tied to leverage, not as the plan.** A
    target the evidence supports: Sharpe ≥ 1.2 (backtest; about 1.0 live),
    CAGR 22–26% pre-tax, MaxDD ≤ 30%, PBO ≤ 30% (decision GT0).
- **The target is pre-tax.** Nearly all of this book's gains are short-term,
  taxed at 20% STCG. The deployed book's 23.6% (252-day convention) is about
  18.5% after tax. A 35% pre-tax CAGR is about 28% after tax.

## 1. What the target implies

### 1.1 Scoping on the deployed book's daily returns

2013–25, cost model 4, 252-day convention. Financing at rf plus a spread on
the borrowed part. (L5's figures for the same leverage, 26.9 / 30.7 / 34.5%,
are on the calendar-year convention, B2; both are 2013–25 scoping looks, not
the honest base above.)

| Variant | CAGR | Sharpe | Vol | MaxDD | Calmar |
|---|---|---|---|---|---|
| Deployed, 1.0× | 23.6% | 1.16 | 13.4% | −23.9% | 0.99 |
| Futures leverage 1.25× (rf + 1%) | 27.5% | 1.15 | 16.7% | −30.2% | 0.91 |
| Futures leverage 1.5× | 31.4% | 1.14 | 20.1% | −36.6% | 0.86 |
| Futures leverage 1.75× | 35.3% | 1.13 | 23.4% | −42.6% | 0.83 |
| MTF 1.25× (14.6% funding) | 25.3% | 1.04 | 16.7% | −32.0% | 0.79 |
| MTF 1.5× | 26.8% | 0.96 | 20.1% | −40.0% | 0.67 |
| Vol target 16%, cap 1.5× (60-day vol) | 24.7% | 1.07 | 15.7% | −35.9% | 0.69 |
| Vol target 20%, cap 2.0× | 29.6% | 1.07 | 20.0% | −45.6% | 0.65 |

What the scoping shows:
- **Volatility targeting hurts this book.** Its long drawdown (2018–20, 748
  days) was a slow grind in calm volatility, so a vol target levered up into
  it. This matches Cederburg et al. (JFE 2020): real-time volatility
  management lost in 72 of 103 strategies, and helped momentum factors
  specifically. It is not a book-level fix here.
- **A second stream:** an independent stream with Sharpe 0.8 and the book's
  tails, at equal risk, lifts the combined Sharpe to about 1.4. Sharpe 1.0
  lifts it to about 1.53. The combined MaxDD still runs near 2σ, so
  drawdown control has to come from the sleeves' timing, not only from
  diversification.
- **Factor-index sleeves do not qualify.** These are fixed capital blends with
  NSE factor indices: price indices without dividends, back-calculated before
  launch, Jul 2016–2025. Over that window the book's Sharpe is 1.22.

  | Index | Own Sharpe | Daily / monthly correlation with the book | Book + 20% / + 35%: Sharpe, MaxDD |
  |---|---|---|---|
  | NIFTY 50 | 0.44 | 0.47 / 0.56 | 1.15, −25.2% / 1.05, −27.0% |
  | NIFTY100 Low Volatility 30 | 0.50 | 0.47 / 0.54 | 1.17, −24.3% / 1.10, −25.4% |
  | NIFTY Alpha Low-Volatility 30 | 0.50 | 0.42 / 0.50 | 1.18, −24.2% / 1.09, −25.1% |
  | NIFTY200 Quality 30 | 0.39 | 0.37 / 0.43 | 1.16, −24.1% / 1.05, −25.0% |
  | NIFTY Quality Low-Volatility 30 | 0.35 | 0.42 / 0.47 | 1.15, −24.1% / 1.05, −25.1% |

  Every blend lowers Sharpe. Long-only factor portfolios share the market's
  beta, so they correlate 0.4–0.5 with the book. The −0.4 to −0.6
  value–momentum correlation in the literature is a long-short result. An
  equity sleeve helps only if it is selective (Sharpe ≥ 0.8 on its own) or
  hedged with NIFTY futures, and a hedge is capital-gated like leverage.

### 1.2 Sharpe and volatility against CAGR and drawdown

Research memo; Gaussian, 15 years, rf 6.5%. Each cell: CAGR / expected MaxDD
(Magdon-Ismail) / simulated 95th-percentile MaxDD.

| Sharpe | σ 15% | σ 20% | σ 25% |
|---|---|---|---|
| 1.2 | 26% / 21% / 28% | 33% / 28% / 38% | 40% / 35% / 47% |
| 1.4 | 30% / 20% / 27% | 38% / 26% / 35% | 47% / 33% / 43% |
| 1.6 | 34% / 18% / 25% | 44% / 25% / 33% | 54% / 31% / 40% |

Real drawdowns are 30–50% deeper than Gaussian. This book drew down 1.78σ
against a Gaussian median of about 1.3σ. The cell that meets every target
with margin is **Sharpe ≈ 1.4–1.6 at σ ≈ 20%**.

## 2. What the research says

### 2.1 Systems that sustain Sharpe ≥ 1.2 at monthly horizons

- **They are many-sleeve and mostly long-short, not one long-only sleeve.**
  - N equal-risk streams with Sharpe s and average correlation ρ give
    s·√(N/(1+(N−1)ρ)), capped at s/√ρ.
  - The three books (ρ = 0.96) add nothing to each other.
  - Four sleeves at Sharpe 0.8 give 1.40 at ρ = 0.1 and 1.16 at ρ = 0.3.
- **A new sleeve's correlation matters more than its Sharpe.**
  - Adding a 0.6 sleeve to a 1.0 core gives 1.05 at ρ = +0.3, 1.17 at
    ρ = 0, and 1.37 at ρ = −0.3.
  - Value and momentum correlate −0.4 to −0.6. A 50/50 global
    stock-selection combination ran a gross Sharpe of 0.71–0.77, and
    across asset classes 1.42–1.59 (Asness, Moskowitz, Pedersen 2013).
- **Long-only constraints cost most of the transfer coefficient.**
  IR = IC·√breadth·TC, and long-only, turnover-limited books run TC ≈ 0.3
  (Clarke, de Silva, Thorley 2002).
- **Trend following is the classic crisis diversifier.**
  - 1880–2013: net Sharpe 0.77, correlation 0 to stocks. 2000–13: 0.62
    (Hurst, Ooi, Pedersen 2017).
  - The live SG Trend Index since 2000 is about 0.3–0.56: a 30–60%
    haircut.
- **Momentum crashes in rebounds.**
  - The IIMA Indian momentum factor fell −53% from Dec 2008 to Sep 2009
    (Daniel & Moskowitz 2016).
  - Scaling momentum by its own volatility roughly doubles its Sharpe
    (Barroso & Santa-Clara 2015). That applies to a momentum sleeve, not to
    a book that already has a regime gate (section 1.1).
- **Drawdown control.** Grossman–Zhou cushion rules (risk in proportion to
  equity above a floor) bound drawdowns but cost growth and can leave the
  book stuck in cash. Centurion's 20/30/35 rule is a coarse version.
- **Costs.**
  - Partial trading toward the target plus wide exit bands adds 26–36% to
    net Sharpe (Gârleanu & Pedersen 2013; Novy-Marx & Velikov 2016).
  - Rebalance-date luck alone is worth 100–350 bp a year in concentrated
    momentum books; staggered tranches remove most of it (Hoffstein, Faber,
    Braun 2020).

### 2.2 Live benchmarks: what is realistic

| Book | Live result |
|---|---|
| SG CTA / Trend indices since 2000 | Sharpe 0.3–0.56 |
| AQR Style Premia (live "value and momentum everywhere") | About 7.6% a year since 2013 (Sharpe ≈ 0.5), including −37% over 2018–20 |
| Renaissance RIEF (monthly horizon) | About market-like (Medallion's Sharpe is short-horizon and capacity-capped) |
| Multi-strategy pods (Citadel) | About 19% net, from hundreds of market-neutral books, high gross leverage and hard stops |
| India: NIFTY200 Momentum 30 | 18.8% CAGR, Apr 2005–Jun 2026; −67.9% in 2008 |
| India: NIFTY200 Quality 30 | 18.0% against 14.5% for its parent; −50.8% against −56.6% in 2008 |
| India: IIMA momentum factor (WML), 1993–2025 | About 10.7–11.0% a year, vol 17% |
| India: IIMA value factor (HML) | 6.9% a year, MaxDD −64%; −52% over 2018–20 |

A live net Sharpe of 0.7–1.0 is good. A sustained 1.2 is top-decile. A 35%
CAGR over 10+ years would rank among the best long-only records anywhere.

### 2.3 Overfitting science

- **The best of 61 zero-skill trials** is expected to show a Sharpe of about
  0.6 over 15 years. The book's 1.16 clears pure noise by about 0.5
  (Bailey, Borwein, López de Prado, Zhu 2014).
- **Deflated Sharpe 0.964** says P(true Sharpe > 0). P(true Sharpe > 1.2)
  is **0.45** (LN-T24).
- **Decay.**
  - Anomaly returns fall 26% out of sample and 58% after publication
    (McLean & Pontiff 2016).
  - Factor premia run about 30% lower out of sample over a century
    (Ilmanen et al.).
  - On Quantopian, backtest Sharpe explained less than 2.5% of
    out-of-sample results, and the more an algorithm was backtested, the
    wider the gap (Wiecki et al. 2016).
- **Hurdles and protocol.**
  - New factors need t > 3 (Harvey, Liu, Zhu 2016).
  - Write the hypothesis first, log every trial, freeze data choices, and
    treat live trading as the only true out-of-sample (Arnott, Harvey,
    Markowitz 2019).
- **Live length needed (95%, monthly data).**
  - Telling Sharpe 1.2 from 0 takes about 2.1 years.
  - Telling 1.2 from 0.8 takes about 18 years.
  - So **judge the live book by tracking error against its shadow backtest,
    not by its Sharpe.** That is G4, already in place.

### 2.4 Indian factor evidence

IIMA library, gross; computed in the factor memo.

| Series | 2013–25: CAGR / vol / Sharpe / MaxDD |
|---|---|
| Market excess | 8.5% / 15.5% / 0.61 / −34% |
| Value (HML) | 8.1% / 17% / 0.54 / −50% |
| Momentum (WML) | 12.9% / 15.8% / 0.85 / −33% |
| WML, volatility-scaled (126 days, cap 2×; the memo's own test) | 14.5% / 14.8% / 1.00 / −28% |
| Winners only, held in bull months, cash otherwise | 21.6% CAGR / 15.4% / 0.96 / −27% |

- **The book matches the closest academic analogue.** "Winners in bull
  months, cash otherwise" is the book's profile, and the book beats it: 1.16
  against 0.96. WML earned about +21% a year in bull months and −11% in bear
  months over 2013–25. Being flat in bears is normal; don't chase it.
- **Low volatility and quality carry strong premia in India.**
  - Lowest-volatility decile +11.4% a year against +1.3% for the highest
    (2001–15).
  - QMJ (quality minus junk) alpha above 10% a year, with the shallowest
    drawdowns.
  - The book's −0.26 low-vol loading works against its Sharpe. As section
    1.1 shows, the fix is a selective sleeve, not an index ETF.
- **Live decay is large.**
  - NIFTY200 Momentum 30 kept about 40% of its backtest excess after launch.
  - Nine NSE factor indices showed no significant alpha after launch
    (Lalwani 2023).
  - DSP Quant: 9.85% a year since 2019 against 13.4% for its benchmark.
  - Plan live excess at 40–60% of backtest excess for a new sleeve.
- **Momentum crashes in India:** −53% (Dec 2008–Sep 2009), −37% (2013–14),
  −28% (2020), and Momentum 30 about −30% from Sep 2024 to Mar 2025.
  Volatility scaling softened April–May 2009 from −18% / −25% to −7% / −9%
  (memo's test).
- **Exclusion filters with evidence:** promoter pledge above 20% (crash
  risk), and NSE's ASM/GSM surveillance lists. These are filters, not alpha.
- **Benchmarks to track the live book against:** NIFTY200 Momentum 30,
  Midcap150 Momentum 50 and Alpha Low-Volatility 30, with Sep 2024–Mar 2025
  as a stress test.

### 2.5 What transfers from Renaissance and Jane Street

- **Transfers:**
  - research hygiene: log every trial, deflate for them, keep the holdout
    sealed;
  - breadth: many weak, uncorrelated signals;
  - obsessive cost measurement;
  - an independent risk function with hard stops.
- **Does not transfer:**
  - the headline Sharpe, which comes from short-horizon, latency-bound,
    capacity-capped spread capture (RIEF, Renaissance's monthly-horizon
    fund, earns about the market);
  - cheap gross leverage on market-neutral books;
  - proprietary data and execution moats.
- **The edge available here** is diversification, risk control and cost
  control on public data. Assume McLean–Pontiff decay.

## 3. Audit: the Ind stocks book

### 3.1 What it is

- **Universe:** the 300 most liquid NSE stocks (median value ≥ Rs 1 crore,
  price ≥ Rs 20, 252 days of history), refreshed every 21 sessions.
- **Signal:** two price-trend groups only (EWMAC 8/32 and 16/64; 64/256),
  volatility-normalised and capped, combined with an FDM.
- **Portfolio:**
  - 20 names (max 30), exit rank 40, max weight 8%, sector cap 25%;
  - rebalance every 5 sessions with a 25% no-trade buffer;
  - 6×ATR stops with a 5-session cooldown.
- **Risk:**
  - regime gate on NIFTY's 200-day mean and breadth (risk-off scale 0)
    and India VIX at 25 / 35;
  - gold and silver ETF trend sleeve at 30–40% of risk;
  - drawdown rule 20/30/35 (E2).
- **Execution:** after-market limit orders, GTT stops, CNC delivery only.
- **Variants:** E4 (exit rank 60) and the candidate (neutral scale 0.6) are
  the same engine; their daily correlation with the deployed book is
  0.96–0.97.

### 3.2 Returns by regime and year

| Regime (known at the previous close) | Days | Annual return | Sharpe |
|---|---|---|---|
| NIFTY bull (above its 200-day mean and rising) | 67% | 29.9% | 1.79 |
| NIFTY bear | 19% | 6.3% | −0.02 |
| Sideways | 15% | 6.3% | −0.02 |
| India VIX calm | 94% | 22.3% | 1.18 |
| India VIX elevated | 6% | 19.1% | 0.88 |

- **All the excess return comes from uptrends.** Bear and sideways phases are
  roughly flat: no loss, no gain. Under cost model 4 the idle cash earns
  nothing.
- **Concentration.**
  - Financial-year returns range from −16% to +86%. Four of 13 years are
    negative.
  - FY2015 (+86%), FY2021 (+67%) and FY2022 (+59%) carry the CAGR.
  - 86% of days are below the previous peak. The longest time under water
    is 748 days (2018–20).
- **Crash years.** On 2007–25 data, a 2008-type crash costs about 33% even
  with the drawdown rule (R4). The regime gate took 309 days after the 2008
  low to restore exposure.

### 3.3 Alpha and signal quality

- **Factor attribution.**
  - Market beta 0.30–0.40 (0.42 on down days), momentum +0.31 (t 12),
    low volatility −0.26 (t −10), so a tilt toward volatile stocks.
  - Size is not significant. R² is 0.40.
  - **Alpha after factors: 11.4% a year (t 3.4), 72% of the excess
    return.** The book is not just repackaged market or momentum
    exposure.
- **Signal quality.**
  - Rank IC rises with horizon: 0.034 at 5 sessions, 0.049 at 21, 0.092
    at 252. It is a slow signal.
  - Alpha shows no decay: +0.5 point a year, p = 0.74.
  - Hit rate 43%, win/loss 2.4, profit factor 1.82: a classic trend
    profile.
- **Price-signal saturation.** Residual momentum, 52-week-high proximity,
  low beta, delivery, low volatility and the R21A rules all carry real
  information. Inside a 20-name book they dilute the trend forecast:
  Sharpe changes of −0.00 to −0.25 (plan section 6, item 5).
  **More price signals in the core is a dead end.**
- **Parameter fragility.**
  - The deployed configuration sits on a local peak: 14 one-setting
    neighbours change Sharpe by −0.39 to 0.00, and rebalance frequency,
    stop distance and group weights are fragile.
  - The candidate's neighbours move it only ±0.04, a more robust region.
    V3, the forward gate, chooses between them.

### 3.4 Risk management

- **In place:**
  - the regime gate;
  - ATR stops;
  - sector and weight caps;
  - the drawdown rule;
  - a metal sleeve uncorrelated with the core (−0.02);
  - a kill switch, G4 tracking gates and a dead man's switch.
- **Gaps:**
  - **A fat left tail.** Skew −0.86, excess kurtosis 4.9, CVaR 99% −3.4% a
    day, worst day −6.7%. Sharpe understates the risk.
  - **MaxDD ≈ 1.8σ,** with drawdowns in calm volatility, so volatility
    sizing cannot fix them.
  - **No sleeve earns in bear or sideways regimes** (a third of all days),
    and the cash sits idle.
  - **No momentum-crash guard** for violent rebounds like 2009 or June
    2020. The worst month was −11.3%.
  - **One risk budget for one engine;** no portfolio-level allocation
    across independent sleeves.

### 3.5 Robustness

- **Overfitting statistics:** deflated Sharpe 0.964 at N = 61 (passes 0.95
  narrowly); PBO 29.2% (at the 30% bar); PSR(1.2) 0.45. It would take 2.1
  years of live data to show the Sharpe is above 0.
- **Walk-forward OOS, 2017–25:** 1.24 for the deployed family and 1.11 for
  E4's family. Over 2008–25: 0.87.
- **Trial budget:** one configuration left before January. The core
  family's PBO leaves no room for more search.

### 3.6 Costs and tax

- **Modelled cost drag:** 2.8% a year (1.4% impact, 1.4% statutory) at a
  turnover of 6.9× a year and a median hold of 29 days.
- **Execution is unmeasured:** 2 real fills so far. X1 needs about 30.
- **Tax:** at 20% STCG on nearly all gains, about 5 points of CAGR go to
  tax. A lower-turnover variant (E4: 5.8×, median hold 40 days) and any
  sleeve that holds past 12 months (LTCG 12.5%) are worth more after tax
  than before.
- **Futures and MCX profits** are business income, taxed at slab rate
  (about 31–36%). F&O losses can offset STCG; capital losses cannot offset
  F&O profits.

### 3.7 Gap register

Value-add is the expected effect on the golden target: H high, M medium, L low.

| # | Gap | Category | Value-add | Evidence | Tracked as |
|---|---|---|---|---|---|
| 1 | One return engine: three books at 0.96 correlation; the metal sleeve is the only diversifier | Architecture | H | Correlations; section 2.1 arithmetic | GT4–GT7 |
| 2 | No fundamentals or earnings dates in the store, so value, quality and earnings drift cannot be tested | Data | H | Store contents; value correlates −0.4 to −0.6 with momentum | GT1 |
| 3 | Bear and sideways regimes earn nothing, and idle cash earns 0% under cost model 4 | Regime / return | L–M (+0.3–0.4 pt CAGR after tax, measured) | Section 3.2; IC1; section 6.1 | GT3 → LN-T29 (after IC2) |
| 4 | Leverage: MTF dropped (L11, your call 17 Oct); futures are cheaper but lumpy (lots about Rs 15 lakh) and cover only the book's weaker half (L7, 3 of 3 FAIL) | Leverage | H, capital-gated | Section 1.1; plans 5ae–5af | GT8, L9 |
| 5 | No multi-asset diversifier beyond gold and silver ETFs; MCX commodity trend untested | Architecture | M | Section 2.1; Indian alpha memo | GT2, GT5 |
| 6 | Price-signal saturation in the core | Signal | Avoids waste | Plan 6, item 5 | Not doing |
| 7 | The deployed configuration sits on a fragile peak | Robustness | M | Neighbours −0.39 to 0.00 | V3 (existing) |
| 8 | Fat left tail; MaxDD ≈ 1.8σ; no momentum-crash guard | Risk | M | Skew, kurtosis, CVaR; 2009 and June 2020 | GT6 (portfolio cushion), GT9 |
| 9 | Slow re-entry after crashes (309 days in 2008) | Regime | M | R4; R11 failed | Covered by sleeves (value earns in rebounds) |
| 10 | Sharpe is too noisy to judge live; PSR(1.2) 0.45 | Statistics | M (governance) | LN-T24 | GT10 protocol; G4 |
| 11 | Core trial budget spent; PBO at the bar | Governance | H (protects the target's PBO leg) | Plan section 5u | GT10 (family budgets) |
| 12 | Execution cost unmeasured | Costs | M | 2 real fills | X1 (existing) |
| 13 | After-tax drag about 5 pts; turnover 6.9× | Tax / costs | M | Section 3.6 | GT9 (tranches, bands), V3 |
| 14 | Rebalance-date luck in a 20-name book | Costs / robustness | L–M | Hoffstein et al. 2020 | GT9 |
| 16 | No exclusion of names on NSE's ASM/GSM lists or with promoter pledge above 20% | Risk | L–M | Factor memo, section 2.4 | GT11 |
| 17 | The live book is not benchmarked against NSE factor indices | Statistics | L | Factor memo | GT11 |
| 15 | Capacity | — | None needed | Impact eats half the edge at about Rs 297 crore | — |

## 4. The rethink

Stop asking the trend engine for more; it is good at one thing. Build a
**portfolio of independent return streams under one risk budget**, add cash
yield, and lever last.

```
                 ┌──────────────── risk budget (fixed, pre-registered weights) ───────────────┐
 Core trend (as live) │ Value / quality sleeve │ Commodity trend (MCX) │ Metals ETF sleeve (as live)
                 └──────────────┬─────────────────────────────────────────────────────────────┘
                                │ portfolio drawdown cushion (Grossman-Zhou style floor)
                                │ cash engine: idle cash in a growth liquid ETF (GT3, LN-T29)
                                │ leverage: futures only, capital >= ~Rs 1 crore, after 12+ months live
```

- **Each sleeve has its own economic hypothesis, its own trial family and
  registry, and at most three pre-registered configurations.** The combined
  book is judged by a deflated Sharpe at the full trial count across
  families. This is how the options family (U32) is run already.
- **Sleeve weights are fixed in advance** (equal risk), never fitted.
- **The core is not re-tuned.** V3's forward gate picks the deployed book or
  the more robust candidate. Momentum-crash protection comes from the value
  sleeve's timing, not from a new core rule.
- **Volatility targeting** is allowed only inside a pure momentum sleeve,
  never on the whole book.

## 5. Plan of action

| Phase | When | What | Gate to the next phase |
|---|---|---|---|
| 0. Build, no trials | Oct–Dec 2026 | Go live with the core as planned (D3). GT1: point-in-time fundamentals and results dates. GT2: MCX bhavcopy and continuous roll series. GT3: cash-engine decision (after tax). GT10: research protocol written. Scoping looks (one look each, nothing recorded) allowed | Data stores built and fingerprinted; protocol and pre-registrations written |
| 1. Sleeve trials | Jan–Mar 2027 (GT4 and GT5 ran early, 12 Oct, on cost model 4: both FAIL; BD1 bonds 17 Oct: FAIL) | Core family, one configuration each after V3: GT7 earnings-drift overlay, GT9 tranches and exit bands, GT11 exclusion filters. New streams per the register (section 6). Each on 2008–25 where data allows, with a walk-forward (your rule 4) | Per sleeve: net Sharpe ≥ 0.5; combined Sharpe ≥ core + 0.10 at the fixed weight; monthly correlation with the core reported, not a gate (owner, 11 Oct); family PBO ≤ 30%; t ≥ 3 for a new signal |
| 2. Combination | Once a sleeve passes (GT6 closed 12 Oct with GT5: none to combine) | GT6: the combined book at fixed equal-risk weights with a portfolio drawdown cushion; walk-forward; combined deflated Sharpe at total N | Combined backtest Sharpe ≥ 1.30, MaxDD ≤ 1.6σ, PBO ≤ 30% |
| 3. Paper, then live | Apr 2027 → | Sleeves paper-traded beside the live core (≥ 60 sessions, G4-style tracking per sleeve), then ladder in at small weight | Tracking within G4 limits; costs ≤ 1.5× model |
| 4. Leverage | Capital ≥ about Rs 1 crore and ≥ 12 months of live combined tracking | GT8: futures leverage up to 1.75× (your call, 17 Oct: MaxDD about −40% accepted), at most half-Kelly on a haircut Sharpe of 0.7, with a 25–30% gilt sleeve as the drawdown lever (BD1; your call, 17 Oct). The sleeve runs your five steps first (2026 holdout, 60 paper sessions, 5 dry runs) | Live tracking within limits; drawdown inside the simulated 95th percentile |

**Scorecards to expect at each step** (backtest, pre-tax; live after about a
30% haircut):

| Step | Sharpe | CAGR | MaxDD | Meets |
|---|---|---|---|---|
| Today (core, cost model 4) | 1.16 (live about 1.0) | 23% (live about 20%) | −24% | MaxDD, PBO |
| + cash engine (GT3, measured) | about 1.19 | about 23.5% (+0.4 pt pre-tax) | −24% | A small gain: the book is only 12.8% idle |
| + 1–2 sleeves passing phase 1 | 1.25–1.35 | 24–27% | −20% to −25% | Sharpe; Calmar about 1 |
| + futures leverage 1.5× (≥ Rs 1 crore) | 1.25–1.35 | 29–33% | −32% to −36% | All but CAGR 35%; that needs a combined Sharpe ≥ 1.4 |
| **Honest base, 17 Oct: 2008–25 walk-forward** | 0.62 (live about 0.43) | 13.2% (live about 11%) | −32.5% | None |
| + 25% gilt sleeve (BD1, B3) | 0.61 | 11.7% | −22.9% | MaxDD |
| 1.75× modelled + 25% gilts (a look; real contracts about 7 points lower) | 0.71 | 16.6% | −32.6% | MaxDD |

## 6. New alpha sources and timeframes

Ranked by expected value at Rs 6–30 lakh.

| # | Source | Expected net Sharpe | Correlation with the core | Capital | Data | Verdict |
|---|---|---|---|---|---|---|
| 1 | Cash engine: idle cash in a growth-type liquid ETF on the exchange (section 6.1) | About rf; +0.3–0.4 pt CAGR after tax (12.8% average idle) | About 0 | Any | ETF prices in the store | **Do, after go-live (GT3 → LN-T29)** |
| 2 | Selective value / quality sleeve (point-in-time fundamentals), unhedged or beta-hedged with NIFTY futures | Must reach ≥ 0.8 on its own to help (section 1.1); value earns in the rebounds momentum misses | About 0.4–0.5 long-only; low or negative when hedged | Rs 3 lakh; about Rs 15 lakh more to hedge | **No:** NSE XBRL results to build (GT1) | **Test (GT4)**: the best equity-side diversifier. Factor-index ETFs fail (section 1.1) |
| 3 | MCX commodity trend (silver micro, crude / natural gas mini, base-metal minis) | 0.3–0.5 | −0.2 to +0.1; earns in crises | Rs 15–20 lakh total | Partial: MCX bhavcopy, rolls to build (GT2) | **Test (GT5)** once capital allows |
| 4 | Post-earnings drift overlay on the core ranking | +0.1–0.2 to the core | 0.4–0.6 | 0 | Needs results dates (GT1) | Test after GT1 (GT7); core family, so it uses the core budget |
| 5 | NIFTY index deletions (anti-momentum; +4–7% abnormal over days 60–240) | 0.2–0.4 | 0.2–0.4 | Rs 1–2 lakh | In hand | Marginal: 5–15 events a year |
| 6 | Buyback tenders | +0.5–1 pt a year | About 0.3 | Rs 2 lakh per event | In hand | Marginal: tendering is in Console, outside the API |
| 7 | FII participant-OI overlay on exposure | ≤ +0.2 | Overlay | 0 | `participant_oi.parquet` | Marginal: the same family as the rejected PCR / VIX timing |
| 8 | Listed REITs / InvITs (owner's ask, 11 Oct 2026): 6–14% distributions | 0.47 measured (2017–25) | 0.16 daily / 0.26 monthly | Rs 1–5 lakh | Raw bhavcopies, series RR / IV | **Closed (GT14, 11 Oct)**: Sharpe 0.47 alone (REITs 0.31, InvITs 0.67, with imputed payouts). Every blend lowers CAGR 2–4 pts, so it fails gate 2. Fell 25% in the Covid crash vs the book's 14.5% |
| — | Option premium selling (NIFTY put-writes, condors) | Literature: negative net, 2015–25. Our tests (O2–O8, 2013–25): Sharpe up to 0.82, but every one failed deflated Sharpe and gate 2 | Long beta in the tails | Rs 14.5 lakh a lot | In hand | **Not viable** (tested O2–O8; naked selling on the never-list) |
| — | Pairs / sector-neutral stock futures | About 0.1 | About 0 | Rs 30–50 lakh+ | In hand | Not viable |
| — | USDINR, G-sec futures | — | — | — | — | Closed: RBI exposure rule; no liquidity |
| — | Short-term (weekly) reversal | Tested in R10 (rank-IC screen, 2013–25) | — | — | — | Failed the screen (17 Sep) |
| — | F&O ban list, dividend capture, delisting, demergers, IPO drift | ≤ 0.2 or no evidence (desk) | — | — | — | Not viable |
| — | MON100 sleeve | Tested (R14, 2013–25) | — | — | — | FAIL (1 Oct) |
| — | Short-momentum futures, NIFTY trend, MTF | Looks, not trials (U28, R5; L4 and L11) | — | — | — | Closed; MTF dropped by your call (17 Oct) |

**Stream register (17 Oct 2026, your rule 3).** Every India stream, with
the kind of evidence behind its verdict:
- **test:** a recorded trial, with runs in a registry;
- **look:** numbers computed but not a trial;
- **desk:** reasoning or literature only;
- **bar:** a rule or the broker forbids it.

"2008" says whether the 2008 crash was inside a window that was run. Only a
test closes a stream for good. The others say what would reopen them.

| Stream | IDs | Evidence | Window; 2008 | N | Verdict (date) | Reopens if |
|---|---|---|---|---|---|---|
| Core trend engine | core | in the core | 2013–25; WF 2017–25 and 2008–25; yes | 61 | Live / paper. WF 2008–25 13.2% / 0.62 / −32.5% | V3 picks the live config |
| Gold / silver ETF trend sleeve | core, R13 | in the core | inside every core run; yes | — | In the core. Correlation −0.02. R13 variant FAIL (1 Oct) | — |
| **Government bonds (gilts)** | **BD1, 5ag** | **test** | **2008–25, backtest and WF; yes** | **3** | **FAIL (17 Oct).** Sharpe 0.01 alone. The 25% sleeve cuts MaxDD ~10 pts, Sharpe flat | Measured debt-ETF costs plus a low-churn rule (your call) |
| MCX commodity trend | GT5, 5y; DG1, 5ah; DG2, 5ai | test | 2013–25; 2008–25 under the diversifier gate; yes | 2 | GT5-B: PASS as a drawdown lever at its own 25% (DG2, 19 Oct; provisional): walk-forward MaxDD −29.5% → −20.7%, Sharpe +0.03. GT5-A: FAIL. Both FAIL at equal risk (DG1) | Paper (DL1) is the test |
| Value / quality sleeve | GT4, 5z | test | 2013–25; no (filings from 2012) | 2 | FAIL (12 Oct). B 16.8% / 0.69, gate 2 fails | Beta-hedged variant once capital covers a NIFTY lot |
| Options rounds O2–O8 | O2–O8 | test | 2013–25; 2007–25 reported; mostly yes | 17 | FAIL / CLOSED (4–10 Oct). Sharpe −0.54 to 0.82 | OX bridge, then OX6 re-tests |
| Options near misses re-tested (A1, X2, Y3) | DG1, 5ah | test (diversifier gate) | 2008–25 (Y3 from 2012), live-like; yes for A1 and X2 | 0 new | FAIL (19 Oct). A1 +0.16 Sharpe (p 0.03) but MaxDD 5 points worse (lost 7.8% in 2008); X2 +0.19 (p 0.20); Y3 +0.22 (p 0.15) | A1 with a crash guard, as a new pre-registered config; the options family's 2026 stays sealed |
| Options execution bridge | OX1–OX6 | desk | — | 0 | OPEN, P1 | In progress |
| Minervini SEPA sleeve | O9 | test | 2013–25; no | 1 | CLOSED (11 Oct). Sharpe −2.24 | Rules go to the core as MV1 |
| Pre-announcement drift (RBI, FOMC) | GT13 | test (event study) | 2013–25; no | 2 | FAIL (10 Oct). t 1.16 / 0.91 | New data via event_drift.py |
| Crash-recovery staged buys | CR1, CR1b | test (event study) | 1929–2022, 13 indices; yes | 6 | FAIL (11 Oct) | Closed for good |
| Leverage from real stock futures | L7, L7-C | test | 2017–25; 2026 holdout; no (no contracts before 2013) | 3 | FAIL (17 Oct). 26–30% | Family closed; L9 paper book runs |
| Synthetic leverage on the whole book | L5, L6, L7a | look | 2017–25 and 2008–25; yes | 0 | NO-GO. 1.75x 35.9% (2017–25) is a model | GT8 route |
| MTF leverage | L4, L11 | look | 2013–25, 2008–25; yes | 0 | DROPPED (17 Oct, your call) | — |
| MON100 sleeve | R14 | test | 2013–25; no | 1 | FAIL (1 Oct) | — |
| Short momentum in stock futures | R15, U28 | look | 2013–25; no | 0 | CLOSED. −11%/yr | Book past Rs 2 crore, market-neutral |
| NIFTY trend on idle cash | R5 | look | 2013–25; no | 0 | CLOSED. At most +0.5 pt | — |
| NIFTY futures overlay (beta) | L2 | look | 2013–25; no | 0 | CLOSED. Adds beta, not edge | — |
| REITs / InvITs | GT14 | look | 2017–25; no | 0 | CLOSED. Sharpe 0.47 alone | Only if the MaxDD leg binds |
| Factor-index ETFs beside the book | §1.1 | look | 2016–25; no | 0 | NOT VIABLE. Lowers Sharpe | A selective or hedged sleeve |
| Price / calendar anomalies | R10 | test (rank-IC screen) | 2013–25; no | 6 | FAIL (17 Sep) | — |
| Book-level volatility targeting | R3 | test (WF) | 2017–25; no | 3 | CLOSED: every fold chose off | Only inside a momentum sleeve |
| Cash engine: liquid ETF sweep | GT3, LN-T29 | look | 2013–25; no | 0 | "Do, after go-live". BD1's baseline: −0.4 pt with equity impact, +1 pt with spread-only costs | Measured liquid-ETF costs |
| Earnings drift; Minervini on the core | GT7, MV1 | desk (pre-registered) | — | 0 | PENDING after V3 | — |
| Index deletions; buybacks; FII OI | §6 rows 5–7 | desk | — | 0 | Marginal | — |
| Pairs / sector-neutral futures | — | desk | — | 0 | NOT VIABLE (Sharpe ~0.1) | — |
| Option premium selling, naked | — | desk (and O2–O4) | — | — | NOT VIABLE; never-list | — |
| Intraday options | OX | bar (swing only) | — | 0 | CLOSED | — |
| USDINR and G-sec futures | — | bar (RBI exposure rule; no liquidity) | — | 0 | CLOSED | A rule change |
| Weekly reversal, ban list, dividend capture, IPO drift | — | desk | — | 0 | NOT VIABLE | — |
| US stocks | US5, US6 | test | 2013–25; 2007–25 reported | 3 | FAIL (17 Oct); on hold | Your restart |

**What the register shows:**
- Of the streams closed by a test, only options, crash recovery, MCX and
  now bonds covered 2008. Value, SEPA, MON100, the anomalies and stock
  futures did not, because their data starts later.
- Forex is closed by RBI's exposure rule, not by a test.

**Timeframes, each with its own holding period:**
- weekly: the core rebalance;
- monthly: the value / quality rebalance;
- quarterly: results drift;
- multi-week to multi-month: commodity trend;
- 12+ months: any LTCG-eligible holding inside the value sleeve.

The point is different holding periods with different drivers, not the same
signal at several speeds. Several trend speeds are already inside the core's
EWMAC.

### 6.1 GT3: the cash engine, measured (11 Oct 2026)

The deployed book's recorded daily weights (run
`20261010T032441064151Z_679cbd0c`, 2013–25) put its **average idle cash at
12.8%** (median 5.6%).
- Only 8.8% of days are half or more in cash.
- Those spells are short: 34 of them, median 3 sessions, the longest 45.
- **Why:** the regime gate's risk-off weight goes to the gold and silver
  sleeve (`risk_off_to_metals`), not to cash.
- **Upper bound:** if every idle rupee earned 6.5%, the book gains 0.83
  point a year pre-tax, 0.57 after slab tax (31.2%) and 0.66 if taxed as
  equity (20.8%).
- **With hysteresis:** the lean review (item 29) put it at about +0.4 point
  pre-tax and +0.3 after tax.
- The research memo's "+2–3 points" assumed 40% idle cash, which this book
  never holds.

| Vehicle | After-tax edge on this book | Fit with nightly AMO orders | Verdict |
|---|---|---|---|
| **Growth-type liquid ETF on the exchange** (LIQUIDCASE, CASHIETF, LIQUIDADD) | +0.3–0.4 pt (slab tax) | Uses the existing CNC order path, like GOLDBEES and SILVERBEES. A sell settles T+1. | **Chosen** |
| LIQUIDBEES | Same yield | Same path, but it pays daily dividends as units, a tax and ledger mess | Rejected (lean review) |
| Arbitrage fund (equity-taxed) | About +0.09 pt more than the ETF at most (0.66 against 0.57 upper bound) | Kite Connect can place fund orders (`place_mf_order`), but they deal at NAV after cut-off, are allotted a day later, and redeem in 1–3 days. That is too slow for spells of a median 3 sessions. Many schemes also charge an exit load on short holdings, and the 2026 futures STT rise trims their yield. | Rejected |

**Decision.** Park idle cash in a growth-type liquid ETF, as LN-T29 already
specifies:
- **Hysteresis:** sweep in only when idle cash exceeds the sweep plus a 2%
  buffer, by 2–5% of the book.
- **Redeem** only what tonight's buys need, as the first sell of the night.
- **Ledger:** a separate sweep field, counted as cash.

**Timing.** After go-live, in this order:
1. IC2 (cost-model stamping), since the sweep is cost model 5;
2. the ST2 / LN-T11 answer on whether a same-night sell can fund that
   night's buys. If it cannot, the sweep sits above LN-T11's cash buffer.

No code changes now.

## 7. Robustness protocol for go-live (GT10)

Now a standalone document: `docs/research_protocol.md` (GT10, 11 Oct 2026),
which supersedes the list below.

1. **Hypothesis first.** Each sleeve states why the return exists (a risk
   premium or a behavioural or structural cause) before any data is read.
2. **Budgets.**
   - At most 3 pre-registered configurations per sleeve family, each
     recorded in that family's registry.
   - Sleeve weights fixed in advance.
   - The core family stays closed except for V3.
3. **One environment.** Every family runs on Kaggle; LN-T26 warns if the
   platforms mix.
4. **Hurdles.**
   - A new signal needs t ≥ 3.
   - The combined book needs a deflated Sharpe ≥ 0.95 at the total trial
     count across families, and PBO ≤ 30% per family (aim for 10–20%).
5. **Holdouts.**
   - 2026 stays sealed (`holdout.lock`) and is opened once per family.
   - Walk-forward OOS 2017–25 for every passing sleeve.
6. **Haircuts.** Plan live as 0.6–0.7× the backtest Sharpe for a new sleeve,
   and 0.85× for the core (D2).
7. **Judge live by tracking, not Sharpe.** Use G4 per sleeve (tracking error
   and cost ratio against a shadow backtest). Set drawdown review triggers
   at the simulated 95th percentile.
8. **Leverage rule.** Futures only; at most half-Kelly on a haircut Sharpe of
   0.7; only after phase 3 tracking holds.
9. **Never:**
   - book-level volatility targeting (scoped: it hurts);
   - naked option selling outside a defined-risk structure;
   - more price signals in the core;
   - re-tuning a failed sleeve.

   MTF is not a never rule. It was dropped by your call on 17 Oct after a
   re-look (L11): its 14.6% interest lowers Sharpe. Defined-risk option
   selling stays testable (OX).

## 8. Unverified items

The alpha-sources memo cites these from secondary sources only. Check them
before acting:
- STT on futures 0.02% → 0.05% and on option sales 0.1% → 0.15% of premium
  from 1 Apr 2026 (Budget 2026). Two memos agree, and the factor memo says
  delivery STT and the STCG / LTCG rates were unchanged, but both read
  secondary sources. This bears on IC2 and cost model 5, and on the options
  family's cost model (`fno_costs`).
- The repo rate reportedly at 5.50% from 7 Oct 2026.
- The 8–10% pledge haircut on liquid ETFs.
- ~~The commodity transaction tax rate after Budget 2026.~~ Checked 11 Oct
  2026 on Zerodha's charges page: CTT is still 0.01% of sell notional on
  non-agricultural futures. The same page shows equity futures STT at 0.05%
  on sells, which confirms the futures part of the first item. Encoded in
  `nse_engine/mcx_costs.py` (GT2).

## Sources

System design and overfitting:
- Asness, Moskowitz, Pedersen, "Value and Momentum Everywhere", JF 2013.
- Moskowitz, Ooi, Pedersen, "Time Series Momentum", JFE 2012.
- Hurst, Ooi, Pedersen, "A Century of Evidence on Trend-Following", 2017.
- Clarke, de Silva, Thorley, FAJ 2002.
- Moreira & Muir, JF 2017.
- Harvey et al., "The Impact of Volatility Targeting", 2018.
- Barroso & Santa-Clara, JFE 2015.
- Daniel & Moskowitz, "Momentum Crashes", JFE 2016.
- Cederburg et al., JFE 2020.
- Grossman & Zhou, Math. Finance 1993.
- Magdon-Ismail et al., J. Appl. Prob. 2004.
- Frazzini & Pedersen, "Betting Against Beta", JFE 2014.
- Bailey, Borwein, López de Prado, Zhu, Notices AMS 2014 and J. Comp. Finance 2017.
- Bailey & López de Prado, "The Deflated Sharpe Ratio", 2014.
- Harvey, Liu, Zhu, RFS 2016.
- Harvey & Liu, "Backtesting", JPM 2015.
- McLean & Pontiff, JF 2016.
- Wiecki et al. (Quantopian), 2016.
- Arnott, Harvey, Markowitz, 2019.
- Gârleanu & Pedersen, JF 2013.
- Novy-Marx & Velikov, RFS 2016.
- Frazzini, Israel, Moskowitz, 2018.
- Hoffstein, Faber, Braun, 2020.
- CFM, Invesco and Zacks pages for live CTA and QSPIX records.

India:
- IIMA Indian Fama-French-Momentum data library (Agarwalla, Jacob, Varma).
- NSE NIFTY200 Quality 30 white paper.
- HDFC and ICICI factor fund documents.
- Sehgal & Bijoy 2015 (earnings drift).
- Chhatwani (NIFTY reorganisation).
- Acharya, Anshuman, Kumar (FII flows).
- Pillai, SSRN 2026 (NIFTY short volatility, 2015–25).
- QuantInsti EPAT (NSE pairs, 2016–25).
- SEBI F&O study FY25.
- Zerodha support pages (MTF, collateral).
- Business Standard (USDINR volumes after the RBI rule).
- MarketCalls (participant OI).
