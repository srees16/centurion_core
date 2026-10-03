# CONCEPTS: Module 5, Options Theory for Professional Trading

Source: Zerodha Varsity Module 5 (23 chapters, about 2015). Every formula the
toolkit implements comes from here. Every worked number listed is a pytest
case unless it is marked **excluded**.

All numbers below were recomputed before writing this file. Where the PDF's
arithmetic is wrong, the corrected value is the test value and the PDF value
is listed in [§ PDF errors](#pdf-errors). Facts about today's market that have
changed since the PDF are in [§ Out of date](#out-of-date-for-todays-indian-market).

Test IDs: `M5.<chapter>.<n>`, in `tests/test_options_theory.py`. Code:
`theory.py` (this file's formulas), `fno_costs.py` (charges and slippage) and
`options_config.py` (every rate and threshold), all in `kite_connect/options/`.

---

## Conventions the toolkit fixes

| Item | Convention | Why |
|---|---|---|
| Time to expiry in Black-Scholes | calendar days / 365 | Reproduces ch. 21 and Module 6 ch. 7 exactly (§ 21) |
| Theta | per calendar day: annual theta / 365 | Matches the PDF's calculator output |
| Vega | per 1 volatility point (σ + 0.01) | Matches ch. 19 ("vega 0.15 per % change") and ch. 21 |
| Rho | per 1 percentage point of rate | Matches ch. 21 |
| Daily ↔ annual volatility | σ_annual = σ_daily × √N, **N configurable, default 365** | Ch. 16 and ch. 18 use 365; ch. 17 uses 252. Tests pass N explicitly |
| Standard deviation | `ddof` configurable | Ch. 15 uses the population SD (÷N); ch. 16 uses Excel `STDEV`, the sample SD (÷N−1); the ch. 20 cone matches the sample SD |
| Daily return | log return ln(Pₜ / Pₜ₋₁) | Ch. 16 |
| P&L sign | + = money in, − = money out; per unit of underlying, ₹ = × lot size | All chapters |
| Rate input | continuously compounded, decimal (0.074769 = 7.4769 %) | Ch. 21 |
| Dividend | continuous yield q in the pricer; ch. 21 passes 0 | Ch. 21 lists dividend as an input |

---

## Ch. 1 to 2: Call option basics and jargon

- A **call** gives the buyer the right, not the obligation, to buy at the strike
  on expiry; the seller is obliged. The buyer pays the **premium**.
- Strike, underlying (spot), exercise, expiry, premium and settlement are defined.
- European exercise: settlement is on expiry only (ch. 2, ch. 4 § 4.6).

| ID | Worked example | Expected |
|---|---|---|
| M5.1.1 | Land deal as a call: strike 500,000, premium 100,000; value at expiry 1,000,000 | +400,000 |
| M5.1.2 | Same; value 300,000 | −100,000 (walk away) |
| M5.1.3 | Same; value 500,000 | −100,000 |
| M5.1.4 | Stock call: strike 75, premium 5; expiry 85 / 65 / 75 | +5 / −5 / −5 |
| M5.2.1 | JP Associates 25 CE @ 1.35, lot 8000: premium outlay | ₹10,800 |
| M5.2.2 | Same, expiry spot 32: cash differential / net profit / return | ₹56,000 / ₹45,200 / 418.5 % (PDF "419 %") |

## Ch. 3: Buying a call

- Intrinsic value at expiry: **IV_call = max(0, S − K)**
- **P&L_long_call = max(0, S − K) − premium**
- **Breakeven_long_call = K + premium**
- Limited risk (the premium paid), unlimited upside.

Bajaj Auto 2050 CE @ 6.35 (spot 2026.9, 15 days):

| ID | Spot at expiry | P&L |
|---|---|---|
| M5.3.1 | 1990, 2000, 2010, 2020, 2030, 2040, 2050 | −6.35 each |
| M5.3.2 | 2060 / 2070 / 2080 / 2090 / 2100 | +3.65 / +13.65 / +23.65 / +33.65 / +43.65 |
| M5.3.3 | 2051 … 2059, step 1 | −5.35, −4.35, −3.35, −2.35, −1.35, −0.35, +0.65, +1.65, +2.65 |
| M5.3.4 | Formula checks 2023 / 2072 / 2055 | −6.35 / +15.65 / −1.35 |
| M5.3.5 | Breakeven | 2056.35, P&L 0 there |

## Ch. 4: Selling a call

- **P&L_short_call = premium − max(0, S − K)**; the mirror of the long call.
- **Breakdown (seller's breakeven) = K + premium**
- Limited profit (the premium), unlimited risk; margin is blocked (see § Out of date).

| ID | Worked example | Expected |
|---|---|---|
| M5.4.1 | Short 2050 CE @ 6.35, spot 1990 … 2050 | +6.35 each |
| M5.4.2 | Spot 2060 / 2070 / 2080 / 2090 / 2100 | −3.65 / −13.65 / −23.65 / −33.65 / −43.65 |
| M5.4.3 | Spot 2023 / 2072 / 2055 | +6.35 / **−15.65** (PDF prints −15.56) / +1.35 |
| M5.4.4 | Spot 2050 … 2059, step 1 | +6.35, 5.35, 4.35, 3.35, 2.35, 1.35, 0.35, −0.65, −1.65, −2.65 |
| M5.4.5 | Breakdown | 2056.35 |
| M5.4.6 | Buyer P&L + seller P&L at any spot | 0 (zero-sum symmetry) |

**Excluded:** the 2015 SPAN and exposure margin screenshots (₹31,762 futures,
₹36,706 short call). Margins come from the Kite margins API at runtime.

## Ch. 5: Buying a put

- **IV_put = max(0, K − S)**
- **P&L_long_put = max(0, K − S) − premium**
- **Breakeven_long_put = K − premium**

Bank Nifty 18400 PE @ 315 (spot 18417):

| ID | Spot at expiry | IV | P&L |
|---|---|---|---|
| M5.5.1 | 16195 / 16510 / 16825 / 17140 / 17455 / 17770 / 18085 | 2205 / 1890 / 1575 / 1260 / 945 / 630 / 315 | +1890 / +1575 / +1260 / +945 / +630 / +315 / 0 |
| M5.5.2 | 18400 / 18715 / 19030 / 19345 / 19660 | 0 | −315 each |
| M5.5.3 | Formula checks 16510 / 19660 | | +1575 / −315 |
| M5.5.4 | Breakeven | 18085 | 0 |
| M5.5.5 | Spot 17000 on the expiry date | | +1085 |

## Ch. 6: Selling a put

- **P&L_short_put = premium − max(0, K − S)**
- **Breakdown_short_put = K − premium** (the ch. 6 takeaway says "premium paid"; it means premium received)

| ID | Worked example | Expected |
|---|---|---|
| M5.6.1 | Short 18400 PE @ 315 over the ch. 5 grid | Exact negatives of M5.5.1 and M5.5.2 |
| M5.6.2 | Spot 16510 / 19660 | −1575 / +315 |
| M5.6.3 | Breakdown | 18085 |

## Ch. 7: Summary of calls and puts

| View | Position | Also called | Premium |
|---|---|---|---|
| Bullish | Buy call | Long call | Pay |
| Flat or bullish | Sell put | Short put | Receive |
| Flat or bearish | Sell call | Short call | Receive |
| Bearish | Buy put | Long put | Pay |

- A buy is "long" only when it opens a position; one that closes a short is a square-off. The same applies to sells.
- The expiry formulae apply only when the position is held to expiry; before expiry, P&L is the change in premium.

| ID | Worked example | Expected |
|---|---|---|
| M5.7.1 | BHEL 230 CE, lot 1000, 2-point capture | ₹2,000 |
| M5.7.2 | IDEA 190 CE, lot 2000, 2-point capture | ₹4,000 |
| M5.7.3 | The four-way view table above | Encoded as the selector's base map |

## Ch. 8: Moneyness

- **ITM**: intrinsic value > 0. **OTM**: intrinsic value = 0. **ATM**: the listed strike nearest to spot.
- Calls: strikes below ATM are ITM and strikes above are OTM. Puts are the reverse.
- "Deep ITM" and "Deep OTM" are not defined numerically in the PDF. The toolkit uses configurable strike-step thresholds (default: 3 or more strikes from ATM), flagged as an assumption.
- Intrinsic value is never negative; that floor is what keeps the buyer's loss capped at the premium (the 920 CE @ 15 example).

| ID | Worked example | Expected |
|---|---|---|
| M5.8.1 | Nifty 8070, 8050 CE: intrinsic value | 20 |
| M5.8.2 | Long call 280 @ spot 310; long put 1040 @ 980; long call 920 @ 918; long put 80 @ 88 | 30; 60; 0; 0 |
| M5.8.3 | Nifty 8060, strikes 7100…8700 step 50: ATM | 8050 |
| M5.8.4 | Calls, spot 8060: 7100 / 7500 / 8100 / 8300 | ITM (960) / ITM (560) / OTM / OTM |
| M5.8.5 | Puts, spot 8202 (PDF computes with 8200): ATM; 7500 / 8000 / 8300 / 8500 | 8200; OTM / OTM / ITM (98) / ITM (298) |
| M5.8.6 | Ashok Leyland spot 68.7, strike step 2.5: ATM | 67.5 |
| M5.8.7 | 920 CE @ 15, spot 918 at expiry: buyer's loss | 15, not 17 |

## Ch. 9 to 11: Delta

- **Δ = ∂premium / ∂spot**. A call is in [0, 1], a put in [−1, 0]. From Black-Scholes, Δ_call = e^(−qT)·N(d₁) and Δ_put = Δ_call − e^(−qT).
- First-order estimate: **new premium ≈ premium + Δ × ΔS**
- Approximate delta bands for a given moneyness (a heuristic, used by the selector only):

  | Moneyness | Call Δ | Put Δ |
  |---|---|---|
  | Deep ITM | 0.8 to 1 | −0.8 to −1 |
  | Slightly ITM | 0.6 to 1 | −0.6 to −1 |
  | ATM | 0.45 to 0.55 | −0.45 to −0.55 |
  | Slightly OTM | 0.3 to 0.45 | −0.3 to −0.45 |
  | Deep OTM | 0 to 0.3 | 0 to −0.3 |

- Delta changes with spot (the S-curve). Varsity names the stages predevelopment, take-off and acceleration, and stabilisation.
- **Deltas are additive** within one underlying: position Δ = Σ (sign × lots × Δ), where long = +1 and short = −1. A futures contract has Δ = 1 and Γ = 0.
- Delta neutral: Σ Δ = 0.
- **Delta as probability of expiring ITM** (ch. 11) is an approximation. The risk-neutral probability is N(d₂) for a call and N(−d₂) for a put. The toolkit reports both and labels the delta figure "approx".

| ID | Worked example | Expected |
|---|---|---|
| M5.9.1 | 8250 CE @ 133, Δ 0.55, Nifty 8288 → 8310 | +12.1, so 145.1 |
| M5.9.2 | Same, 8288 → 8200 | −48.4, so 84.6 |
| M5.9.3 | 100-point move, Δ 0.05 vs Δ 0.2 | +5 vs +20 |
| M5.9.4 | 8300 PE @ 128, Δ −0.55, Nifty 8268 → 8310 | −23.1, so 104.9 |
| M5.9.5 | Same, 8268 → 8230 | +20.9, so 148.9 |
| M5.10.1 | 8700 CE @ 12, Δ 0.05, +100 | 17 (+41.67 %) |
| M5.10.2 | 8500 CE @ 20, Δ 0.25, +100 | 45 (+125 %) |
| M5.10.3 | 8400 CE @ 60, Δ 0.5, +100 | 110 (+83.3 %) |
| M5.10.4 | 8300 CE @ 105, Δ 0.8 / 8200 CE @ 210, Δ 1.0, +100 | 185 (+76.19 %) / 310 (+47.62 %) |
| M5.10.5 | Futures 8409 and 8000 CE @ 450, Δ 1, spot +30 | 8439 and 480 |
| M5.10.6 | Bajaj Auto 2210, +30: Δ 0.05 @ 3, 0.3 @ 7, 0.5 @ 12, 0.7 @ 22, 1 @ 75 | 4.5, 16, 27, 43, 105 (+50 %, +128.57 %, +125 %, +95.45 %, +40 %) |
| M5.11.1 | Long 8000 CE 0.7 + 8120 CE 0.5 + 8300 CE 0.05; Nifty +50 | Δ +1.25; +62.5 |
| M5.11.2 | Add long 8300 PE (−1.0) | Δ +0.25; +12.5 |
| M5.11.3 | With 2 lots of 8300 PE | Δ −0.75; −37.5 |
| M5.11.4 | Long 8100 CE (0.5) + long 8100 PE (−0.5) | Δ 0 (delta neutral) |
| M5.11.5 | Short 8100 CE + long 8100 PE | Δ −1.0 |
| M5.11.6 | 5 long deep-ITM calls; 5 short deep-ITM puts | +5; +5 |
| M5.11.7 | Δ 0.3 → probability; Δ 0.1 → probability | 30 %; 10 % (labelled approx) |

**Excluded:** the 2009 election-day story (₹2 lakh to ₹28 lakh, a narrative with no option inputs).

## Ch. 12 to 13: Gamma

- **Γ = ∂Δ / ∂spot = e^(−qT)·φ(d₁) / (S σ √T)**; the same for calls and puts, and always positive.
- Long options are long gamma; short options are short gamma.
- The PDF's step update, implemented as `delta_gamma_step` and labelled first-order:
  premium ← premium + Δ·ΔS (using the **old** Δ), then Δ ← Δ + Γ·ΔS, with Γ held constant.
- Gamma peaks at ATM and rises sharply for ATM options near expiry; ITM and OTM gamma tends to 0 (ch. 20).
- Risk control: position delta drifts with gamma, so a lot-count limit understates the risk of short options.

| ID | Worked example | Expected |
|---|---|---|
| M5.13.1 | 8400 CE @ 26, Δ 0.3, Γ 0.0025; Nifty 8326 → 8396 | premium 47, Δ 0.475 |
| M5.13.2 | then 8396 → 8466 | premium 80.25, Δ 0.65 |
| M5.13.3 | then 8466 → 8416 | premium 47.75, Δ 0.525 |
| M5.13.4 | ATM put Δ −0.5, Γ 0.004; spot +10 / −10 | Δ −0.46 / −0.54 |
| M5.13.5 | Short 10 lots 8400 CE, Δ 0.5, Γ 0.005; spot +70 | position Δ 5 → 8.5 lots-equivalent |
| M5.13.6 | Futures gamma | 0 |

**Excluded:** the car velocity and acceleration example in ch. 12 (physics, with no option inputs).

## Ch. 14: Theta

- **Premium = intrinsic value + time value**; time value = premium − intrinsic value.
- Theta is the premium lost per day with everything else fixed. Longs pay theta (Θ < 0); shorts collect it.
- Decay accelerates near expiry.

| ID | Worked example | Expected |
|---|---|---|
| M5.14.1 | Nifty 8423: 8350 CE / 8450 CE / 8400 PE / 8450 PE intrinsic value | 73 / 0 / 0 / 27 |
| M5.14.2 | 8600 CE @ 99.4, spot 8531: time value | 99.4 |
| M5.14.3 | Next day @ 87.9, spot 8537.9: time value; drop | 87.9; 11.5 |
| M5.14.4 | 8450 CE @ 160, spot 8514.5: IV / TV | 64.5 / 95.5 |
| M5.14.5 | IDEA 190 CE @ 0.30, spot 179.6, 1 day: TV | 0.30 |
| M5.14.6 | 2.75 with Θ −0.05, one day | 2.70 |
| M5.14.7 | Short @ 54, Θ 0.75, 3 days | 51.75; seller profit 2.25 |

## Ch. 15 to 17: Volatility, standard deviation, normal distribution

- Mean, variance and SD of a series. Volatility is the SD of returns.
- **Simple range** (ch. 15): S × (1 ± σ_period).
- **Log-normal range** (ch. 17): S × exp(μ_period ± k·σ_period), k = 1, 2, 3.
- **Linear range** (ch. 18): S × (1 + μ_period ± k·σ_period).
- Scaling: μ_period = μ_daily × n, σ_period = σ_daily × √n.
- Coverage: 1 SD ≈ 68.27 %, 2 SD ≈ 95.45 %, 3 SD ≈ 99.73 %. The PDF says 68/95/99.7 and, in a takeaway, 99.5.
- The toolkit exposes all three range methods. The PDF uses each in a different chapter, so each test calls the method its chapter used.

| ID | Worked example | Expected |
|---|---|---|
| M5.15.1 | Billy runs [20, 23, 21, 24, 19, 23]: sum / mean / population var / SD | 130 / 21.67 / 3.22 / 1.79 |
| M5.15.2 | Mike runs [45, 13, 18, 12, 26, 19]: sum / mean / population SD | 133 / 22.17 / 11.19 (PDF 11.18) |
| M5.15.3 | Billy and Mike range, mean ± 1 SD (PDF uses mean 21.6) | 19.81 to 23.39; 10.98 to 33.34 |
| M5.15.4 | Nifty 8547, σ 16.5 %, 1 year, simple method | 7136.7 to 9957.3 (PDF 7136 / 9957) |
| M5.15.5 | TCS 2585, σ 27 % | 1887.05 to 3282.95 (PDF 1887 / 3282) |
| M5.16.1 | Wipro closes 558.75 … 542.95 (14 values): log returns | 2.15, 1.04, −4.58, 1.08, −1.14, −1.16, −1.56, 2.33, 0.16, 0.34, 0.23, −0.84, −0.93 % |
| M5.16.2 | Daily 1.47 % → annual, N = 365 | 28.08 % |
| M5.16.3 | Annual 25.5 % → daily, N = 365 | 1.33 % (PDF 1.34 %, from 25.52 %) |
| M5.17.1 | Nifty 8337, annual μ 9.66 %, σ 16.61 % (N = 252): 1 SD, log-normal | 7777 to 10841 (PDF prints the exponent as 26.66 %; it is 26.27 %) |
| M5.17.2 | 2 SD | 6587 to 12800 |
| M5.17.3 | 30 days: μ 1.15 %, σ 5.73 %: 1 SD / 2 SD | 7963 to 8930 / 7520 to 9457 |
| M5.17.4 | Daily σ 1.046 % → annual (N = 252) / 30-day | 16.60 % / 5.73 % |

**Excluded:** the Galton board (illustration only) and the six return-histogram charts (no data).

## Ch. 18: Volatility applications

**SD-based strike selection for writing:**

1. σ_n = σ_daily·√n and μ_n = μ_daily·n, where n = days to expiry.
2. Range: S(1 + μ_n ± k·σ_n), linear method.
3. Write calls **above** the upper bound and puts **below** the lower bound.

The PDF's own trading rules become selector defaults:

- writes only calls ("panic spreads faster than greed")
- 1 SD when 3 to 4 days are left; 2 SD when writing earlier
- never writes with more than 15 days to expiry
- skips event days
- exits when the written strike turns ATM

**Volatility stop-loss:** σ_n = σ_daily·√(holding days); SL = entry × (1 − σ_n) for a long, or entry × (1 + σ_n) for a short; the stop goes beyond that level.

| ID | Worked example | Expected |
|---|---|---|
| M5.18.1 | Nifty 8462, 16 days, daily σ 0.89 %, μ 0.04 % | 8214.9 to 8817.4; PDF ≈ 8214 to 8818 (it uses σ 0.8917 %); tolerance ±1 |
| M5.18.2 | Writing candidates above 8818 from the chain: 8850 @ 7.45, 8900 @ 4.85 | Selected strikes ≥ 8850 |
| M5.18.3 | 7.45 × lot 25 / margin ₹12,000 | ₹186.25 / 1.55 % (2015 lot and margin, for the arithmetic only) |
| M5.18.4 | Capital split ₹5,00,000: 35 / 40 / 25 %, then 35 % of the 25 % | 1,75,000 / 2,00,000 / 1,25,000 / 43,750 (8.75 %) |
| M5.18.5 | Airtel entry 395, daily σ 1.8 %, 5 days | σ₅ 4.02 %; SL level 379.1 |
| M5.18.6 | Airtel with target 417: fixed SL 385 / volatility SL 375 | reward 22; RRR 2.2 / 1.1 (PDF prints 3.2 and 1.6, see errors) |

## Ch. 19 to 20: Vega and Greek interactions

- **Vega** is the premium change per 1 volatility point and is positive for long calls and puts. Volatility types: historical, forecast (GARCH), implied, realised. India VIX is the 30-day implied volatility of Nifty options.
- Vega is larger with more time to expiry; buy options when volatility is expected to rise and sell when it is expected to fall (ch. 22).
- **Volatility smile:** implied volatility is lowest near ATM.
- **Volatility cone:** realised volatility over rolling windows (10, 20, 30, 45, 60, 90 days) → max, mean ± 1 and ± 2 SD (sample SD), min. Current implied volatility plotted on it shows rich (above +2 SD) or cheap (below −2 SD) options.
- With low IV, deep ITM and OTM deltas flatten toward 1 and 0; with high IV, far OTM options keep a non-zero delta.

| ID | Worked example | Expected |
|---|---|---|
| M5.19.1 | Vega 0.15, IV +1 point | premium +0.15 |
| M5.20.1 | Nifty 10-day realised-vol windows [41, 38, 33, 28, 28, 41, 26, 22, 56, 19, 13, 34, 17, 41, 21] % | max 56, mean 30.5 (31), +1 SD 42.1, +2 SD 53.7, −1 SD 19.0, −2 SD 7.4, min 13 (PDF 54 / 42 / 31 / 19 / 7) |
| M5.20.2 | 6800 PE, Nifty 7794.05, IV 41.45 %, 13 calendar days, r 7–7.5 % | ≈ 8.6 against a market price of 8.3 (loose tolerance ±0.5) |
| M5.20.3 | Directional checks: vega(30 d) > vega(15 d) > vega(5 d); gamma at ATM rises as expiry nears; far OTM delta rises with IV | Property tests |

**Excluded** (chart readings or missing inputs):

- vega graph values (97 to 190, 67 to 100, 38 to 56)
- the 24 Aug 2015 crash narrative
- the 7800/8000 bull call spread costing 72 at 20 % IV and 82 at 35 % (the PDF gives no days to expiry or rate)

## Ch. 21: Black-Scholes calculator and put-call parity

d₁ = [ln(S/K) + (r − q + σ²/2)T] / (σ√T),  d₂ = d₁ − σ√T

C = S e^(−qT) N(d₁) − K e^(−rT) N(d₂),  P = K e^(−rT) N(−d₂) − S e^(−qT) N(−d₁)

Inputs: spot (or futures, for futures-based options), strike, rate (91-day
T-bill; from config, never hard-coded), IV, dividend, days to expiry.
Outputs: price, Δ, Γ, Θ, V (and ρ).

**Implied volatility** is the σ that solves BS(σ) = market premium. The solver
brackets σ (Brent's method), with Newton-vega as an optional speed-up. It
rejects premiums below the no-arbitrage bound (intrinsic value, discounted) and
returns `None` instead of an invented number.

**Put-call parity** (European, same strike and expiry; it holds for **any**
strike, not only ATM as the PDF states):

**C − P = S e^(−qT) − K e^(−rT)**

At expiry this reads "put + spot = strike + call".

| ID | Worked example | Expected (tolerance) |
|---|---|---|
| M5.21.1 | ICICI 280, spot 272.7, 1 day, IV 43.55 %, r 7.4769 %, q 0 | call 0.39, put 7.63, Δ 0.127 / −0.873, Γ 0.0336, vega 0.030, Θ −0.656 / −0.598, ρ 0.001 / −0.007 (±0.001 to 0.01) |
| M5.21.2 | IV solver round trip: price at 43.55 %, then solve | 43.55 % ± 0.01 |
| M5.21.3 | Infosys 1200: A = 1200 PE + share, B = 1200 CE + ₹1200 cash; expiry 1100 / 1350 | both 1200 / both 1350 |
| M5.21.4 | Parity on M5.21.1 output | C − P = S − K e^(−rT) to 1e-9 |
| M6.7.x | Module 6 ch. 7 calculator screenshots (two strikes, full Greeks) | see STRATEGIES.md § ch. 7 |

## Ch. 22 to 23: Strike selection for buyers, and case studies

Strike guide for **buying** naked options (calls and puts alike), keyed by when
the trade starts and when the target is expected:

| Trade starts | Target hit within | Best strike |
|---|---|---|
| 1st half of series | 5 days | Far OTM (2 strikes from ATM) |
| 1st half | 15 days | ATM or slightly OTM (1 strike) |
| 1st half | 25 days | Slightly ITM |
| 1st half | expiry day | ITM |
| 2nd half | same day | Far OTM (2 to 3 strikes) |
| 2nd half | 5 days | Slightly OTM (1 strike) |
| 2nd half | 10 days | Slightly ITM or ATM |
| 2nd half | expiry day | ITM |

Assumes a ~30-day monthly series; see the weekly-expiry flag below.

| ID | Worked example | Expected |
|---|---|---|
| M5.22.1 | The table above | Selector lookup returns these buckets |
| M5.23.1 | CEAT 1220 PE bought 45.75, sold 52, lot 500 | +6.25 per unit (PDF says 7), ₹3,125 |
| M5.23.2 | Nifty short 7800 CE 203 + PE 176 = 379; exit 191 + 178 | +10 per unit |
| M5.23.3 | Infosys short 1140 CE 48 + PE 47 = 95; exit 55 + 20 | +20 per unit |
| M5.23.4 | Infosys long 1100 CE 18.9 → 41.5 | +22.6 per unit (+119.6 %) |

---

## PDF errors

The test value is the corrected one.

| Where | PDF says | Correct | Effect |
|---|---|---|---|
| Ch. 4, @2072 short call | −15.56 | −15.65 | M5.4.3 |
| Ch. 7, BHEL | "over 350 %", "180 %" | +256 %, +80 % (the PDF quotes ratios as % changes) | narrative only |
| Ch. 9, negative-delta illustration | Δ −0.2 × 88 down = −17.6 | the sign is muddled; the point (call delta cannot be negative) stands | not tested |
| Ch. 10, slightly OTM row | 129 % | 128.57 % | M5.10.6 |
| Ch. 15, Billy mean | 21.6 | 21.67 | M5.15.3 uses the PDF's 21.6 to reproduce its range |
| Ch. 17, annual 1 SD exponent | 26.66 % | 26.27 % (9.66 + 16.61); the PDF's 10841 is right | M5.17.1 |
| Ch. 17, daily mean | 0.04 % | ≈ 0.0383 % (the PDF's annual 9.66 % and 30-day 1.15 % imply it); 0.04 % is rounded | tests use the PDF's 9.66 % and 1.15 % |
| Ch. 17, takeaway | 3 SD = 99.5 % | 99.73 % | — |
| Ch. 18, 16-day SD | 3.567 % from daily 0.89 % | 0.89 × 4 = 3.56 %; the PDF's figure implies 0.8917 % | M5.18.1 tolerance ±1 |
| Ch. 18, Airtel reward | 417 − 385 = 32 | 417 − 395 = 22 (measured from entry, not from the stop) | RRR 3.2 → 2.2 and 1.6 → 1.1 |
| Ch. 18, 10-day note | 1.6·√10 | 1.8·√10 | — |
| Ch. 20, 6800 PE | "trading at 5.5" and "8.3" | 8.3 (the screenshot) | M5.20.2 |
| Ch. 21, calculator | rate 7.4567 in the screenshot vs 7.4769 in the text | negligible (< 0.001 on all outputs) | M5.21.1 uses 7.4769 |
| Ch. 21, parity | requires ATM options | parity holds for any common strike | M5.21.4 |
| Ch. 23, CEAT | "₹7 profit" | 6.25 | M5.23.1 |
| Ch. 2 vs ch. 7 | "5 factors" vs "4 forces" | Δ, Γ, Θ, V (plus ρ) | — |

---

## Out of date for today's Indian market

Checked on 3 Oct 2026. The toolkit reads every item marked *runtime* from Kite
when it runs and never from code.

| PDF says (2015) | Today | Toolkit handling |
|---|---|---|
| F&O expire on the **last Thursday**; three monthly expiries | Since **1 Sep 2025**, NSE F&O expire on **Tuesday**: NIFTY weekly each Tuesday, monthly contracts on the last Tuesday ([Zerodha bulletin](https://zerodha.com/marketintel/bulletin/417370/revision-in-expiry-day-of-index-and-stock-derivatives-contracts), [Business Standard](https://www.business-standard.com/amp/markets/news/nse-bids-adieu-to-thursday-expiry-as-dates-swap-come-into-effect-explained-125082800635_1.html)). Since Nov 2024 NSE has **one** weekly index option (NIFTY); BANKNIFTY, FINNIFTY and MIDCPNIFTY are monthly only ([Angel One](https://www.angelone.in/news/market-updates/nifty-weekly-expiry-today-see-what-changed-on-sept-1-2025)). Holidays shift the day. | *runtime*: expiries from the `NFO` instruments dump |
| Lot sizes: Nifty 25, Bajaj Auto 125, JP 8000, and so on | Revised under SEBI's minimum contract value. NIFTY went 75 → **65** and BANKNIFTY 35 → **30** from the Jan 2026 series ([Business Standard](https://www.business-standard.com/markets/capital-market-news/nse-announces-reduction-in-derivative-lot-size-for-four-key-indices-125100701081_1.html), [HDFC Sky](https://hdfcsky.com/news/nse-revises-market-lot-sizes-for-major-index-derivatives-effective-january-2026)); revised twice a year | *runtime*: `lot_size` from the instruments dump |
| "All options are cash settled" (ch. 1, 2) | **Stock** F&O have been **physically settled** since Oct 2019; index options are still cash settled. ITM stock options held to expiry mean delivery and expiry-week delivery margins | executor warns for stock options held into expiry week; the report shows settlement type |
| Short-option margin ≈ futures margin; SPAN + exposure screenshots | Peak-margin rules, upfront premium, hedge benefit for spreads, extra ELM on expiry day for short index options, and no calendar-spread benefit on expiry day (SEBI Nov 2024) | *runtime*: `/margins/basket` |
| STT on exercised ITM options as the "STT trap"; Zerodha "1.95 points per lot" | STT from **1 Apr 2026**: options **0.15 %** of sell-side premium (and 0.15 % on exercise), futures **0.05 %** ([ICICI Direct](https://www.icicidirect.com/futures-and-options/articles/stt-changes-in-budget-2026-what-f-o-traders-need-to-know), [Upstox](https://upstox.com/news/personal-finance/tax/explained-how-the-stt-hike-on-equity-futures-and-options-affects-traders-and-investors/article-189260/)). Exchange charges went flat ("true to label") in Oct 2024. Brokerage is Zerodha's current schedule. | date-aware schedules in `ChargesConfig` (STT, exchange, stamp; brokerage ₹20 per executed order), used by backtests and the pre-trade report; Kite's basket-margin charges ([Kite margins docs](https://kite.trade/docs/connect/v3/margins/)) are shown beside them before a live trade (decision 3) |
| 91-day T-bill 7.4769 % (Sep 2015) | 5.52 % (RBI auction, 30 Sep 2026) | `MarketConfig.risk_free_rate` with its date; update it from the weekly auction |
| Freeze quantity | Not in the PDF. **Kite does not expose it**: no API field and not in the instruments dump; a too-large order is rejected with the limit in the error ([Kite forum, Apr 2024](https://kite.trade/forum/discussion/13944/getting-max-quantity-limit)). Kite now offers `autoslice=true` on `place_order`, up to **10 slices**, one order per slice ([Kite orders docs](https://kite.trade/docs/connect/v3/orders/)) | `autoslice=true` (decision 2); each slice is an order and pays brokerage |
| API trading from any machine | **Static IP mandatory** for every Kite Connect order since **1 Apr 2026**; orders from an unregistered IP are rejected. **10 orders per second** limit (HTTP 429 beyond it); more needs exchange registration. Data endpoints work from any IP ([Zerodha Substack](https://inthemoneybyzerodha.substack.com/p/sebi-algo-trading-changes-april-2026), [Kite forum](https://kite.trade/forum/discussion/15912/preparing-to-comply-with-sebis-retail-algo-rules-static-ip-ratelimits-order-types)) | live orders only from the registered static IP; client-side throttle below 10/s; 429 handled |
| Market orders | `market_protection` (1 to 100 %, or −1 for automatic) now converts MARKET and SL-M orders to protected limits | the toolkit uses LIMIT orders only, as specified |
| NSE website option chain and quote pages; Zerodha Pi | Both obsolete | Kite quotes only |
| "1st half / 2nd half of a ~30-day series" strike guidance | NIFTY now has **weekly** series (≤ 7 days), so a "1st half" barely exists on weeklies | selector keys on days to expiry and days to target, not series halves (open decision 5) |
| Strike interval (Nifty 100 in 2015 tables) | Varies by underlying and price | *runtime*: from the chain |
| American stock options (until ~2011) | All European; still true | European pricer only |
| Market hours 9:55 (2009 story) | 9:15 to 15:30 | not used |

---

## Decisions (step 1 review, 3 Oct 2026)

1. **Location.** The code is in `kite_connect/options/`, beside the existing
   `option_chain.py`, `options_executor.py` and the three strategy files,
   which stay untouched under the keep rule. New modules use names that do
   not collide with them and import nothing from them. Tests are in
   `tests/test_options_*.py`, so CI runs them.
2. **Freeze quantity.** Kite `autoslice=true`. The executor waits for
   **every** slice of a leg to fill before it places the next leg. If any
   slice fails, it stops and reports.
3. **Charges and slippage.** Every cost figure includes F&O charges and
   slippage (`fno_costs.py`).
   - Charges: brokerage per executed order, so per slice; STT on the sell
     side and on exercise of ITM longs (the pre-Sep 2019 full-value base
     included); exchange charges; SEBI fee; stamp duty on buys; GST.
   - Rates are date-aware, so backtests use the rates of their day.
   - Slippage: max(1 tick, 0.5 % of premium) for index options and 2 % for
     stock options. These are placeholders until paper fills measure the
     real spread.
4. **Login.** No automated login (U23). The toolkit reuses the token the
   daily email-link login stores (`daily_login.kite_from_stored_token()`).
   Live orders go through the static-IP proxy, as the nightly session's
   do. Tests and backtests need no Kite session at all.
5. **Weekly expiries.** The selector treats more than 15 days to expiry as
   the PDF's "1st half" and 15 days or fewer as the "2nd half", and also
   takes days to target. On a weekly NIFTY expiry only the 2nd-half rules
   apply.
6. **Annualisation.** N = 365 for volatility conversion and calendar
   days / 365 for T in Black-Scholes. Ch. 17's tests pass N = 252
   explicitly.
7. **Static IP.** Live orders must come from the IP registered on
   developers.kite.trade. `--live` runs only through
   `CENTURION_KITE_PROXY`, checked against `CENTURION_KITE_STATIC_IP`, and
   Kite's IP rejection is shown verbatim.
