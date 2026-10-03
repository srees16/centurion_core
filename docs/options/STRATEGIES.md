# STRATEGIES: Module 6, Option Strategies (chapters 2 to 13)

Source: Zerodha Varsity Module 6 (about 2016). Payoffs use the formulas in
CONCEPTS.md (ch. 3 to 6). Every table below was recomputed from the legs and
matches the PDF unless it is listed in [§ PDF errors](#pdf-errors).
Out-of-date market facts are in CONCEPTS.md § Out of date, plus the Module 6
items [here](#out-of-date-module-6-specific).

Test IDs: `M6.<chapter>.<n>`, in `tests/test_options_strategies.py`. Code:
`strategies.py` and `selector.py` in `kite_connect/options/`. P&L is per unit
of underlying; ₹ = × lot size × lots.

---

## Design

- `Leg(option_type CE|PE, side BUY|SELL, strike, premium, ratio=1, expiry, iv=None)`
- `Strategy(legs)` computes everything **from the legs**:
  - `payoff(S) = Σ sign × ratio × (intrinsic(S) − premium)`
  - net debit/credit = Σ premiums paid − received
  - max profit and max loss from the piecewise-linear payoff (the kinks are at the strikes); a non-zero end slope means unlimited (`inf`), and below spot 0 the payoff stays bounded
  - `max_loss_region()` / `max_profit_region()`: where the extreme occurs (a strike, or a flat interval)
  - breakevens are the roots between kinks
  - `payoff_table(lo, hi, step)`, `payoff_chart()` (matplotlib, lazy import)
  - `net_greeks()`, the sum of each leg's Black-Scholes Greeks, signed by side (`with_ivs()` sets a per-leg IV; a leg can also carry given Greeks, as the PDF's delta examples do)
- Each strategy class also exposes `generalization()`, the PDF's closed-form
  formulas. Tests assert **formula == leg-derived value** for every worked example.
- Formulas carry their validity conditions (`conditions()`), for example the net credit for ratio spreads; `check_generalization()` refuses to compare when one fails.

---

## Ch. 2: Bull Call Spread (moderately bullish, net debit)

**Legs:** buy 1 lower-strike CE (classically ATM) + sell 1 higher-strike CE (OTM); same underlying, expiry and quantity.

| Generalization | Formula |
|---|---|
| Spread | K_high − K_low |
| Net debit | P_low − P_high |
| Max loss | net debit (spot ≤ K_low) |
| Max profit | spread − net debit (spot ≥ K_high) |
| Breakeven | K_low + net debit |

| ID | Example | Expected |
|---|---|---|
| M6.2.1 | Nifty 7846: buy 7800 CE @ 79, sell 7900 CE @ 25 | debit 54, max loss 54, max profit 46, BE 7854 |
| M6.2.2 | Payoff at 7000…7800 / 7900…8500 | −54 / +46 |
| M6.2.3 | Set 1 (spot 7883): 7700 @ 296 / 7800 @ 227 | debit 69, max profit 31, BE 7769 |
| M6.2.4 | Set 2: 7800 @ 227 / 7900 @ 167 | debit 60, max profit 40, BE 7860 |
| M6.2.5 | Set 3: 7900 @ 167 / 8000 @ 116 | debit 51, max profit 49, BE 7951 |
| M6.2.6 | Formula == legs for M6.2.1 and M6.2.3 to M6.2.5 | equal |

**"Moderate"** (ch. 2): an index move under 5 %; a low-volatility stock under 5 %; a volatile stock 5 to 8 %. Thresholds go in config.

**Strike selection** (spot 8000, 300-point spread, target +3.75 % to 8300):

| Trade starts | Target hit in | Spread |
|---|---|---|
| 1st half | 5 days | Far OTM: 8600 / 8900 |
| 1st half | 15 days | Slightly OTM: 8200 / 8500 |
| 1st half | 25 days | ATM: 8000 / 8300 (OTM loses) |
| 1st half | expiry | ATM: 8000 / 8300 (far OTM loses more) |
| 2nd half | 1 day | Far OTM: 8600 / 8900 |
| 2nd half | 5 days | Far OTM (lower profit, theta) |
| 2nd half | 10 days | Slightly OTM (1 strike from ATM) |
| 2nd half | expiry | ATM (far OTM loses) |

**Volatility** (M5 ch. 20): cheaper at low IV. The 7800/8000 example costs 72 at 20 % IV and 82 at 35 % IV, so prefer low IV.

## Ch. 3: Bull Put Spread (moderately bullish, net credit)

**Legs:** buy 1 lower-strike PE (OTM) + sell 1 higher-strike PE (ITM).
**When:** the market has fallen, put premiums are swollen, IV is high, there is ample time to expiry, and the view is a moderate recovery. Prefer it over a bull call spread when the puts are richer.

| Generalization | Formula |
|---|---|
| Net credit | P_high − P_low |
| Max profit | net credit (spot ≥ K_high) |
| Max loss | spread − net credit (spot ≤ K_low) |
| Breakeven | K_high − net credit |

| ID | Example | Expected |
|---|---|---|
| M6.3.1 | Nifty 7805: buy 7700 PE @ 72, sell 7900 PE @ 163 | credit 91, max profit 91, max loss 109, BE 7809 |
| M6.3.2 | Payoff 7000…7700 / 7800 / 7900…8500 | −109 / −9 / +91 |
| M6.3.3 | Spot 7612: 7500 @ 62 / 7700 @ 137 | credit 75, max loss 125, BE 7625 |
| M6.3.4 | 7400 @ 40 / 7800 @ 198 | credit 158, max loss 242, BE 7642 |
| M6.3.5 | 7500 @ 62 / 7800 @ 198 | credit 136, max loss 164, BE 7664 |

Wider spread means more profit and a higher breakeven. Use a larger spread only on high conviction.

## Ch. 4: Call Ratio Back Spread (outright bullish, net credit, 2:1)

**Legs:** sell 1 ITM CE (K_low) + buy 2 OTM CE (K_high), in the ratio 1:2 (2:4, …). Execute only for a **net credit** (ch. 9 states the rule; ch. 4 says "usually").

| Generalization (2:1, net credit) | Formula |
|---|---|
| Spread | K_high − K_low |
| Net credit | P_low − 2·P_high |
| Payoff when spot ≤ K_low | net credit |
| Max loss | spread − net credit, **at K_high** |
| Lower breakeven | K_low + net credit |
| Upper breakeven | K_high + max loss |
| Upside | unlimited |

| ID | Example | Expected |
|---|---|---|
| M6.4.1 | Nifty 7743: sell 1× 7600 CE @ 201, buy 2× 7800 CE @ 78 | credit 45, max loss 155 at 7800, BEs 7645 / 7955 |
| M6.4.2 | Payoff 7000…7600 / 7645 / 7700 / 7800 / 7900 / 7955 / 8000 / 8100 / 8500 | +45 / 0 / −55 / −155 / −55 / 0 / +45 / +145 / +545 |

**Strike selection** (spot 8000, 300-point spread, target +6.25 % to 8500):

- **1st half, any target horizon:** sell 7800 (slightly ITM), buy 2× 8100 (slightly OTM).
- **2nd half, any horizon (same day to expiry):** sell 7600 (deep ITM), buy 2× 7900 (slightly ITM).
- Far OTM long strikes lose even when the direction is right.

**Volatility:**

| Days to expiry | Effect of IV rising 15 % → 30 % on the strategy |
|---|---|
| 30 | helps: −67 → +43 |
| 15 | helps less: −77 → −47 |
| about 5 | **hurts** |

Avoid at the start of a series when IV is already high (more than about 2× normal).

## Ch. 5: Bear Call Ladder (outright bullish despite the name; net credit; 1:1:1)

**Legs:** sell 1 ITM CE (K₁) + buy 1 ATM CE (K₂) + buy 1 OTM CE (K₃).

| Generalization | Formula |
|---|---|
| Net credit | P₁ − P₂ − P₃ |
| Spread | K₂ − K₁ (the PDF writes "ITM and ITM"; it means ITM and ATM) |
| Max loss | spread − net credit, flat **between K₂ and K₃** |
| Payoff when spot ≤ K₁ | net credit |
| Lower breakeven | K₁ + net credit |
| Upper breakeven | K₂ + K₃ − K₁ − net credit |
| Upside | unlimited |

| ID | Example | Expected |
|---|---|---|
| M6.5.1 | Nifty 7790: sell 7600 CE @ 247, buy 7800 CE @ 117, buy 7900 CE @ 70 | credit 60, max loss 140 on [7800, 7900], BEs 7660 / 8040 |
| M6.5.2 | Payoff 7000…7600 / 7660 / 7700 / 7800 / 7900 / 8000 / 8040 / 8100 / 8300 / 8700 | +60 / 0 / −40 / −140 / −140 / −40 / 0 / +60 / +260 / +660 |

**When:** only when convinced of a large move up; the PDF's own use is stocks around quarterly results. Volatility behaves as in ch. 4.

## Ch. 6: Synthetic Long, and its arbitrage

**Legs:** buy 1 ATM CE + sell 1 ATM PE, same strike K.

| Generalization | Formula |
|---|---|
| Net premium | P_call − P_put (debit if positive) |
| Breakeven | K + net debit |
| Payoff | S − K − net debit (linear, Δ ≈ 1: mimics long futures) |

| ID | Example | Expected |
|---|---|---|
| M6.6.1 | Nifty 7389: buy 7400 CE @ 107, sell 7400 PE @ 80 | debit 27, BE 7427 |
| M6.6.2 | Payoff 6700 / 7200 / 7400 / 7427 / 7600 / 8400 | −727 / −227 / −27 / 0 / +173 / +973 |
| M6.6.3 | Symmetry: BE ± 200 (7627 / 7227) | +200 / −200 |

**Arbitrage check:** long call + short put + short futures F at the same strike K.
P&L at **every** expiry = **(F − K) − (C − P)**, a constant.

| ID | Example | Expected |
|---|---|---|
| M6.6.4 | 21 Jan 2016: spot 7304.80, fut 7316 (short), 7300 CE @ 79.5 (buy, ask), 7300 PE @ 73.85 (sell, bid) | +10.35 at every expiry 6700…8400 |

The toolkit adds what the PDF leaves out:

- **cost of carry.** Parity is C − P = (F − K)·e^(−rT), so it reports the residual after financing.
- **charges** (both option legs, the futures leg, and STT on the ITM exercise at expiry), per unit. They come from `fno_costs.py`, or from Kite's basket charges before a live trade.
- It flags an opportunity only if the residual exceeds charges plus a buffer (config). It uses executable bid and ask, not LTP.

The excluded fish-market example (buy 100, sell 150, transport 20) has no option inputs.

## Ch. 7: Bear Put Spread (moderately bearish, net debit)

**Legs:** buy 1 higher-strike PE (ITM) + sell 1 lower-strike PE (OTM).

| Generalization | Formula |
|---|---|
| Net debit | P_high − P_low |
| Breakeven | K_high − net debit |
| Max profit | spread − net debit (spot ≤ K_low) |
| Max loss | net debit (spot ≥ K_high) |

| ID | Example | Expected |
|---|---|---|
| M6.7.1 | Nifty 7485: buy 7600 PE @ 165, sell 7400 PE @ 73 | debit 92, BE 7508, max profit 108, max loss 92 |
| M6.7.2 | Payoff 6600…7400 / 7500 / 7508 / 7600…8100 | +108 / +8 / 0 / −92 |
| M6.7.3 | Black-Scholes, spot 7485, IV 18 %, r 7.25 %, q 0, expiry 25 Feb 2016, **16 days** (the PDF omits the valuation date; 16 days reproduces every printed figure): 7600 strike | call 73.52, put 164.41, Δ 0.382 / −0.618, Θ −3.913 / −2.408, ρ 1.220 / −2.101, Γ 0.0014, vega 5.974 |
| M6.7.4 | Same, 7400 strike | call 174.23, put 65.75, Δ 0.658 / −0.342, Θ −4.181 / −2.716, ρ 2.082 / −1.152, Γ 0.0013, vega 5.757 |
| M6.7.5 | Net delta: long 7600 PE + short 7400 PE | −0.618 + 0.342 = **−0.276** |

**Strike selection** (4 % fall, 5000 → 4800):

| Trade starts | Target in | Higher (buy) | Lower (sell) |
|---|---|---|---|
| 1st half | 5 days | Far OTM | Far OTM |
| 1st half | 15 days | ATM | Slightly OTM |
| 1st half | 25 days | ATM | OTM |
| 1st half | expiry | ATM | OTM |
| 2nd half | same day | OTM | OTM |
| 2nd half | 5 days | ITM or OTM | OTM |
| 2nd half | 10 days | ITM or OTM | OTM |
| 2nd half | expiry | ITM or OTM | OTM |

**Volatility:** cost barely moves with IV at 30 days, moderately at 15 days, a lot at 5 days. Take it in the 2nd half only when IV is expected to rise.

## Ch. 8: Bear Call Spread (moderately bearish, net credit)

**Legs:** buy 1 higher-strike CE (OTM) + sell 1 lower-strike CE (ITM).
**When:** the market has rallied so calls are rich, and the view is a moderate fall. Preferred over a bear put spread for the credit.

| Generalization | Formula |
|---|---|
| Net credit | P_low − P_high |
| Breakeven | K_low + net credit |
| Max profit | net credit (spot ≤ K_low) |
| Max loss | spread − net credit (spot ≥ K_high) |

| ID | Example | Expected |
|---|---|---|
| M6.8.1 | Nifty 7222: buy 7400 CE @ 38, sell 7100 CE @ 136 | credit 98, BE 7198, max profit 98, max loss 202 |
| M6.8.2 | Payoff 6600…7100 / 7198 / 7202 / 7302 / 7402…8102 | +98 / 0 / −4 / −104 / −202 |
| M6.8.3 | Net delta: 7400 CE +0.32, short 7100 CE −0.89 | −0.57 (PDF inputs; no BS parameters given) |

**Strike selection** (4 % fall):

| Trade starts | Target in | Higher (buy) | Lower (sell) |
|---|---|---|---|
| 1st half | 5 days | Far OTM | ATM + 2 strikes |
| 1st half | 15 days | Far OTM | ATM + 2 strikes |
| 1st half | 25 days | OTM | ATM + 1 strike |
| 1st half | expiry | OTM | ATM |
| 2nd half | same day | Far OTM | Far OTM |
| 2nd half | 5 days | Far OTM | Slightly OTM |
| 2nd half | 10 days | Slightly OTM | ATM |
| 2nd half | expiry | OTM | ATM or ITM |

The PDF labels the 2nd-half rows 5 / 15 / 25 days / expiry, copied from the 1st-half table. The graphs show same day / 5 / 10 days / expiry, and the toolkit follows the graphs.

**Volatility:** as in ch. 7; the PDF advises taking it when IV is expected to rise.

## Ch. 9: Put Ratio Back Spread (bearish, net credit, 2:1)

**Legs:** sell 1 ITM PE (K_high) + buy 2 OTM PE (K_low). Execute only for a net credit.

| Generalization (2:1, net credit) | Formula |
|---|---|
| Spread | K_high − K_low |
| Net credit | P_high − 2·P_low |
| Payoff when spot ≥ K_high | net credit |
| Max loss | spread − net credit, **at K_low** |
| Upper breakeven | K_low + max loss (= K_high − net credit) |
| Lower breakeven | K_low − max loss |
| Downside | unlimited down to spot 0 |

| ID | Example | Expected |
|---|---|---|
| M6.9.1 | Nifty 7506: sell 1× 7500 PE @ 134, buy 2× 7200 PE @ 46 | credit 42, max loss 258 at 7200, BEs 6942 / 7458 |
| M6.9.2 | Payoff 6500 / 6800 / 6900 / 6942 / 7000 / 7100 / 7200 / 7300 / 7400 / 7458 / 7500…8000 | +442 / +142 / +42 / 0 / −58 / −158 / −258 / −158 / −58 / 0 / +42 |
| M6.9.3 | Net delta: short 7500 PE (−0.55 → +0.55) + 2× long 7200 PE (−0.29) | −0.03 |

**Strikes:** the classic ITM + OTM pair at any time to expiry.

**Volatility:**

| Days to expiry | Effect of IV rising |
|---|---|
| 30 | helps: −57 → +10 |
| 15 | helps less: −77 → −47 |
| near expiry | little effect |

## Ch. 10: Long Straddle (neutral, expects a big move, net debit)

**Legs:** buy 1 ATM CE + buy 1 ATM PE, same strike K.

| Generalization | Formula |
|---|---|
| Net debit | P_call + P_put |
| Max loss | net debit, at K |
| Breakevens | K ± net debit |
| Profit | unlimited either way |
| Net delta | ≈ 0 (delta neutral) |

| ID | Example | Expected |
|---|---|---|
| M6.10.1 | Nifty 7579: buy 7600 CE @ 77 + 7600 PE @ 88 | debit 165, BEs 7435 / 7765, max loss 165 at 7600 |
| M6.10.2 | Payoff 6500 / 7200 / 7435 / 7500 / 7600 / 7700 / 7765 / 7800 / 8000 / 8700 | +935 / +235 / 0 / −65 / −165 / −65 / 0 / +35 / +235 / +935 |
| M6.10.3 | Breakeven move | 165 / 7600 = 2.17 % each way |

**Profitable only if all hold** (selector checks):

1. IV is low at entry.
2. IV rises while the position is held.
3. There is a large move.
4. The move comes quickly, well before expiry.
5. It is set around an event whose outcome **differs** from the consensus.

If the event merely matches expectations, the IV crush breaks the straddle.

## Ch. 11: Short Straddle (neutral, expects a range, net credit)

**Legs:** sell 1 ATM CE + sell 1 ATM PE. This mirrors ch. 10: max profit = net credit at K, breakevens K ± credit, unlimited loss.

| ID | Example | Expected |
|---|---|---|
| M6.11.1 | Nifty 7589: sell 7600 CE @ 77 + 7600 PE @ 88 | credit 165, BEs 7435 / 7765, max profit 165 |
| M6.11.2 | Payoff over the M6.10.2 grid | exact negatives |
| M6.11.3 | Infosys event straddle: sell 1140 CE @ 48 + PE @ 47 = 95; close 55 + 20 = 75 | +20 per unit |

**When:** IV is high at entry and expected to fall. Before an event, IV inflates premiums; exit just after the announcement.
Delta neutral at entry only; it drifts as spot moves (gamma), which the monitor reports.

## Ch. 12: Long and Short Strangle

The chapter is titled "Straddle"; the content is the strangle.

**Long strangle legs:** buy 1 OTM PE (K_p) + buy 1 OTM CE (K_c), strikes roughly equidistant from ATM, same ratio.

| Generalization (long) | Formula |
|---|---|
| Net debit | P_put + P_call |
| Max loss | net debit, anywhere in [K_p, K_c] |
| Upper breakeven | K_c + net debit |
| Lower breakeven | K_p − net debit |
| Profit | unlimited either way |

| ID | Example | Expected |
|---|---|---|
| M6.12.1 | Nifty 7921: buy 7700 PE @ 28 + 8100 CE @ 32 | debit 60, BEs 7640 / 8160, max loss 60 on [7700, 8100] |
| M6.12.2 | Payoff 7000 / 7500 / 7600 / 7640 / 7700…8100 / 8160 / 8200 / 8300 / 8800 | +640 / +140 / +40 / 0 / −60 / 0 / +40 / +140 / +640 |
| M6.12.3 | Short strangle, same strikes | credit 60, +60 on [7700, 8100], BEs 7640 / 8160, exact negatives elsewhere |
| M6.12.4 | Comparison straddle at 5900: CE @ 66 + PE @ 57 | debit 123; BEs **6023 / 5777** using the strike (the PDF uses spot 5921 and gets 6044 / 5798) |
| M6.12.5 | Net delta: 7700 PE −0.3 + 8100 CE +0.3 | 0 |

**Short strangle when:** a range-bound stock (double or triple tops and bottoms); write strikes outside the range and watch for a breakout. The Greeks behave as for straddles.

## Ch. 13: Max Pain and Put-Call Ratio

**Max pain:** for each strike X taken as the expiry price, the writers' loss is

L(X) = Σ_{K<X} (X − K)·callOI_K + Σ_{K>X} (K − X)·putOI_K

Max pain is argmin L(X).

| ID | Example | Expected |
|---|---|---|
| M6.13.1 | 3 strikes: 7700 (call OI 1,823,400 / put OI 5,783,025), 7800 (3,448,575 / 4,864,125), 7900 (5,367,450 / 2,559,375) | L = 998,287,500 / **438,277,500** / **709,537,500** (the PDF prints 7,095,375,000); max pain 7800 |
| M6.13.2 | 10 May 2016 table, strikes 7000…8600 (OI in the PDF) | every cumulative call, put and total value matches the PDF; min 4,341,862,500 at **7800** |
| M6.13.3 | PCR on the same table | Σ put OI 37,016,925 / Σ call OI 42,874,200 = **0.8634** |

**PCR** = Σ put OI / Σ call OI, read as a **contrarian** signal:

| PCR | Reading |
|---|---|
| > 1.3 | Extreme bearishness, so expect a rise |
| < 0.5 | Extreme bullishness, so expect a fall |
| 0.5 to 1 | Normal |
| 1 to 1.3 | **Undefined** in the PDF; treated as normal |

The thresholds go in config, and the PDF says to calibrate them on 1 to 2 years of history per underlying.

**The author's modified max pain:**

1. Compute it at 15 days to expiry.
2. Add a 5 % buffer: max pain 7800 → about 8200 strike (7800 × 1.05 = 8190).
3. Expect expiry inside [max pain, max pain + 5 %].
4. Write calls beyond the band; avoid writing puts; hold to expiry; don't average down.

| ID | Example | Expected |
|---|---|---|
| M6.13.4 | Max pain 7800, buffer 5 %, strike step 100 | band 7800 to 8190 → write calls ≥ 8200 |

The PDF says the 5 % buffer is "about 1.5 to 2 SD, so about 34 %". Those don't match: 34 % is roughly the two-sided tail beyond 1 SD. The toolkit reports the actual SD distance of the buffer from current volatility instead.

---

## Selector

The selector ranks strategies, using only the PDF's rules.

**Inputs:**

- market view: strong bull, moderate bull, moderate bear, strong bear, neutral range-bound, or neutral big move
- volatility view: rising, falling or flat
- days to expiry and days to target
- optional current IV against the realised-vol cone (M5 ch. 20)

| View | Candidates (PDF chapter) | Ranking tie-breaks from the PDF |
|---|---|---|
| Moderate bull | Bull call spread (2), bull put spread (3) | Rich puts or high IV after a fall → bull put (credit); otherwise bull call. Strikes from the ch. 2 table |
| Strong bull | Call ratio back spread (4), bear call ladder (5) | Net credit required. Rising IV with long DTE helps; rising IV near expiry hurts. Ladder needs a large move |
| Moderate bear | Bear put spread (7), bear call spread (8) | Rich calls after a rally → bear call (credit). Ch. 7/8 tables; prefer when IV is expected to rise in the 2nd half |
| Strong bear | Put ratio back spread (9) | Net credit; ITM/OTM pair |
| Neutral, big move | Long straddle (10), long strangle (12) | Low IV now and rising IV expected; event with a non-consensus outcome; strangle when cost matters |
| Neutral, range | Short straddle (11), short strangle (12) | High IV now and falling expected; range-bound underlying for strangles; never short ATM near expiry (M5 ch. 13, gamma) |
| Futures-like | Synthetic long (6) | Arbitrage check when parity residual > carry + charges |

**Series half** = days to expiry > 15 ("1st half") else "2nd half" (CONCEPTS.md open decision 5).

The selector returns ranked candidates with the PDF rule cited for each score change, and strike guidance in the PDF's moneyness labels. `strike_for()` turns a label into a listed strike (offsets from ATM in `SelectorConfig.strike_offsets`). It never auto-trades.

---

## PDF errors

| Where | PDF says | Correct |
|---|---|---|
| Ch. 2, set tables | "Lower Strike (ATM, Long) 7700/7900" | ITM / ATM / OTM labels mixed; the numbers are right |
| Ch. 3, scenario 1 | "Premium Paid – Intrinsic Value = 100 − 72" | the label is reversed (IV − premium); the number 28 is right |
| Ch. 5 | spread = "difference between the ITM and ITM options" | ITM and ATM (K₂ − K₁) |
| Ch. 8 | 2nd-half table rows 5 / 15 / 25 / expiry | same day / 5 / 10 / expiry, per the graphs |
| Ch. 9 | "3 options bought for every 2 sold" | 2:1 means 4 for 2 |
| Ch. 9 | volatility chart labelled "Sell 4900 CE / Buy 5100 CE" | the chart is reused from the call version; the text describes puts |
| Ch. 11, takeaways | "max profit = net premium paid … long straddle" | net premium **received** … **short** straddle |
| Ch. 12 | title "The Long & Short Straddle" | Strangle |
| Ch. 12 | "Nifty is trading at 5921" (strangle example) | 7921 |
| Ch. 12 | straddle breakevens 5921 ± 123 = 6044 / 5798 (from spot) | from the strike, as in ch. 10: 5900 ± 123 = 6023 / 5777 |
| Ch. 12, takeaway | long strangle loss "limited to premium received" | premium **paid** |
| Ch. 13 | 7800 PE loss at 7700 = "4,864,125,000" | 486,412,500 (the total 998,287,500 is right) |
| Ch. 13 | 7700 CE loss at 7900 = "3,646,800,000"; total "7,095,375,000" | 364,680,000; 709,537,500 (10×; max pain still 7800) |
| Ch. 13 | 5 % buffer "≈ 1.5 to 2 SD → 34 %" | inconsistent; computed instead |
| Ch. 6 | arbitrage residual treated as risk-free profit | ignores cost of carry and current charges (see § ch. 6) |

## Out of date (Module 6 specific)

Read with CONCEPTS.md § Out of date: expiry day, lot sizes, settlement, margins, STT, static IP, freeze quantity, weekly series.

- **Ch. 6 arbitrage economics.** The "Zerodha breakeven 4 to 5 points" and the old STT trap are both 2016 figures. From 1 Apr 2026, STT is 0.15 % on the options sell side and on exercise, and 0.05 % on futures. A 10-point Nifty arbitrage must be re-costed through Kite's charges before it counts.
- **Strike-selection tables** assume a ~30-day monthly series with a 300-point spread at Nifty 8000. The toolkit maps them to strike steps from ATM in the live chain (ATM + n strikes) and to days to expiry, not to absolute 2016 levels.
- **Ch. 13 thresholds** (PCR 1.3 / 0.5, 5 % buffer, "15 days to expiry") come from the 2016 monthly-only market. They are config defaults and are flagged for re-calibration on weekly NIFTY data.

## Not built

Ch. 1 lists these but chapters 2 to 13 never cover them:

- Call butterfly, long and short butterfly
- Synthetic call, synthetic put (ch. 6 covers the synthetic **long** only)
- Straps, strip
- Bull put ladder
- Long and short iron condor
- Box
- Volatility arbitrage with dynamic delta hedging
