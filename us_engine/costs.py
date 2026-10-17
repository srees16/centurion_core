"""What a US book costs an India-resident investor beyond its trades, and what tax takes (US tracker US3).

The per-order costs are the engine's (``nse_engine.costs``): the broker's commission
(``CostConfig.commission_rate`` and ``commission_cap``, set from ``BROKERS`` by ``us_engine.backtest``),
the SEC and FINRA fees on sells, and spread plus impact.  This module adds what a backtest in dollars
cannot see, for a book run in USD and judged in INR (US-Y2):

- Moving money: the FX markup over the reference rate each way, GST on the conversion (CGST rule
  32(2)(b)), the broker's plan fee, and TCS on LRS remittances above the yearly threshold, which is
  credited against tax later, so it costs only the wait.
- Tax: the US withholds 25% of dividends (India-US treaty, W-8BEN), credited in India.  India taxes the
  gain on each lot in rupees, at today's rules: long-term after 24 months at 12.5%, short-term at the
  slab rate, plus 4% cess, with India's set-off and 8-year carry-forward.  Taxes and plan fees are paid
  from savings in India; the view charges them India's 6.5% for the years until the end.

Rates are dated (``ACCOUNT_MODEL_VERSION``; sources beside each, checked 17 Oct 2026; UG12 for the rules).
The brokers publish FX markups only as ranges, so the midpoint is used.  Rupee amounts for tax use the
rate on the last day of the month before each sale, purchase or dividend (Income-tax Rules 2026, rule 206,
which names the SBI TT buying rate; the store's USD/INR stands in for it).  No rule fixes how a resident
converts cost; the same convention is used.

Not modelled: US estate tax on US-situs assets above $60,000 (US-listed shares and ETFs; no India-US
treaty), which Irish-domiciled UCITS ETFs avoid; and the 180-day rule (LRS para A.17) under which idle
cash and proceeds abroad must be reinvested or brought home.

    python -m us_engine.costs report        # round 1's books and SPY, after costs and tax, in INR
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: 1 = rates checked 17 Oct 2026 (tracker US3).
ACCOUNT_MODEL_VERSION = 1


@dataclass(frozen=True)
class Broker:
    """One broker's costs: commission per order, plan fee, and FX markups over the reference rate."""

    name: str
    commission_rate: float      # of each order's value, GST included
    commission_cap: float       # USD per order; 0 = no cap
    plan_fee_inr: float         # a year
    fx_in: float                # INR -> USD, markup over the reference rate
    fx_out: float               # USD -> INR
    gst_on_withdrawal: bool     # whether the conversion back is an Indian bank's (GST due)
    source: str


#: Pages checked 17 Oct 2026.  Vested: vestedfinance.com/in/pricing, /legal/fee-schedule and support articles
#: (deposit "1.5% over IBR" in its own comparison table, marked illustrative; INR withdrawals "1-2% inclusive"
#: per a staff forum post, no GST since converted abroad).  INDmoney: indmoney.com/pricing?type=us-stocks
#: (Global Access, effective 1 Aug 2026: 0.25% + GST, at most $25 + GST; IFSCA fee 0.005%) and
#: /us-stocks/fund-transfers (bank markup 0.5-1.2%; withdrawal spread not published).
BROKERS: Dict[str, Broker] = {
    "vested_basic": Broker("Vested Basic", 0.0025, 35.0, 0.0, 0.015, 0.015, False,
                           "0.25% a trade (max $35); deposit ~1.5% over IBR; INR withdrawal 1-2%"),
    "vested_premium": Broker("Vested Premium", 0.0015, 35.0, 4500.0, 0.015, 0.015, False,
                             "0.15% a trade (max $35); Rs 4,500 a year; FX as Basic"),
    "indmoney": Broker("INDmoney", 0.0025 * 1.18 + 0.00005, 25.0 * 1.18, 0.0, 0.0085, 0.0085, True,
                       "0.25% + 18% GST a trade (max $29.50) + 0.005% IFSCA fee; bank FX 0.5-1.2%"),
}
#: The cheapest route for a book that trades more than ~$47,000 a year (Premium's fee over its saving).
DEFAULT_BROKER = "vested_premium"

#: GST on currency conversion: 18% of the value set by CGST rule 32(2)(b).
GST_RATE = 0.18
#: LRS TCS (Income-tax Act 2025 s.394(1), table item 7, formerly s.206C(1G); unchanged by Finance Act 2026):
#: nothing on the first Rs 10 lakh a tax year across all LRS remittances, 20% above it for investment;
#: credited as tax paid (s.390(5)) and refunded at assessment.
LRS_TCS_THRESHOLD_INR = 1_000_000.0
LRS_TCS_RATE = 0.20
#: US withholding on dividends for an India resident with a W-8BEN (treaty article 10(2)(b); 30% without
#: one); India credits it up to its own tax on the dividend (s.159, rule 76; Form No. 44).
US_DIVIDEND_WITHHOLDING = 0.25
#: Foreign shares and ETFs (no STT, so no Rs 1.25 lakh exemption): long-term after 24 months at 12.5%
#: without indexation (Income-tax Act 2025 ss.2(101), 197(1)(b); since 23 Jul 2024), short-term and
#: dividends at the slab rate; health and education cess 4%.
LTCG_MONTHS = 24
LTCG_RATE = 0.125
CESS = 0.04
#: The marginal slab rate before cess (surcharge left out: set your own).
DEFAULT_SLAB = 0.30
CARRY_FORWARD_YEARS = 8
#: India's risk-free convention (the books' Sharpe), charged on money paid out before the end.
INDIA_RF = 0.065


def gst_on_conversion(inr: float) -> float:
    """GST on converting ``inr`` rupees of currency (rule 32(2)(b): 1% up to Rs 1 lakh with a Rs 250 floor,
    then Rs 1,000 + 0.5% to Rs 10 lakh, then Rs 5,500 + 0.1%, the value capped at Rs 60,000)."""
    a = abs(float(inr))
    if a <= 0:
        return 0.0
    if a <= 1e5:
        value = max(0.01 * a, 250.0)
    elif a <= 1e6:
        value = 1000.0 + 0.005 * (a - 1e5)
    else:
        value = min(5500.0 + 0.001 * (a - 1e6), 60000.0)
    return GST_RATE * value


def tcs(remit_inr: float, earlier_in_year_inr: float = 0.0) -> float:
    """TCS on an LRS remittance of ``remit_inr``, given what was already remitted this financial year."""
    before = max(float(earlier_in_year_inr), 0.0)
    above = max(before + float(remit_inr), LRS_TCS_THRESHOLD_INR) - max(before, LRS_TCS_THRESHOLD_INR)
    return LRS_TCS_RATE * above


def tax_rate_on(rate: pd.Series, day) -> float:
    """USD/INR for tax on ``day``: the rate on the last day of the month before it (rule 206)."""
    return float(rate.asof(pd.Timestamp(day).replace(day=1) - pd.Timedelta(days=1)))


def financial_year(day: pd.Timestamp) -> int:
    """India's financial year, by the calendar year it starts in (April to March)."""
    day = pd.Timestamp(day)
    return day.year if day.month >= 4 else day.year - 1


def _fy_end(fy: int) -> pd.Timestamp:
    return pd.Timestamp(f"{fy + 1}-03-31")


def realised_lots(trades: pd.DataFrame, fx: pd.Series, close_out: Optional[Dict[str, float]] = None,
                  close_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Each sale matched first-in, first-out against the buys of its symbol, in rupees.

    ``trades``: the engine's ``trades.csv`` (date, symbol, side, quantity, value_inr = dollars, cost_inr =
    dollar costs).  Buy costs join the cost, sell costs come off the proceeds.  ``close_out`` (symbol ->
    dollar value still held on ``close_date``) sells what is left at the end, so its gain is taxed too.
    One row per matched piece: sell date, buy date, cost and proceeds in INR, and whether it is long-term
    (held over 24 months)."""
    trades = trades.copy()
    if close_out:
        held: Dict[str, float] = defaultdict(float)
        for t in trades.itertuples(index=False):
            held[t.symbol] += float(t.quantity) if t.side == "BUY" else -float(t.quantity)
        sells = [{"date": pd.Timestamp(close_date), "symbol": sym, "side": "SELL", "quantity": held[sym],
                  "value_inr": value, "cost_inr": 0.0}
                 for sym, value in close_out.items() if held.get(sym, 0.0) > 1e-9 and value > 0]
        trades = pd.concat([trades, pd.DataFrame(sells)], ignore_index=True)
    rate = fx.sort_index().ffill()
    at = lambda d: tax_rate_on(rate, d)
    open_lots: Dict[str, deque] = defaultdict(deque)
    rows: List[Dict] = []
    for t in trades.sort_values("date", kind="stable").itertuples(index=False):
        q = float(t.quantity)
        if q <= 0:
            continue
        if t.side == "BUY":
            open_lots[t.symbol].append([pd.Timestamp(t.date), q, (float(t.value_inr) + float(t.cost_inr)) / q])
            continue
        unit = (float(t.value_inr) - float(t.cost_inr)) / q
        left = q
        while left > 1e-9 and open_lots[t.symbol]:
            lot = open_lots[t.symbol][0]
            take = min(left, lot[1])
            rows.append({"date": pd.Timestamp(t.date), "bought": lot[0], "symbol": t.symbol,
                         "cost_inr": take * lot[2] * at(lot[0]), "proceeds_inr": take * unit * at(t.date),
                         "long": pd.Timestamp(t.date) > lot[0] + pd.DateOffset(months=LTCG_MONTHS)})
            lot[1] -= take
            left -= take
            if lot[1] <= 1e-9:
                open_lots[t.symbol].popleft()
    return pd.DataFrame(rows, columns=["date", "bought", "symbol", "cost_inr", "proceeds_inr", "long"])


def capital_gains_tax(lots: pd.DataFrame, slab: float = DEFAULT_SLAB) -> pd.Series:
    """Tax on realised gains by financial year (INR), with India's set-off: short-term losses against any
    gain, long-term losses against long-term gains only, the rest carried forward 8 years."""
    if lots.empty:
        return pd.Series(dtype="float64")
    g = lots.assign(gain=lots["proceeds_inr"] - lots["cost_inr"], fy=lots["date"].map(financial_year))
    carry: List[List] = []                 # [fy, kind, amount] of losses still usable
    out = {}
    for fy in range(int(g["fy"].min()), int(g["fy"].max()) + 1):
        year = g[g["fy"] == fy]
        st = year.loc[~year["long"], "gain"].sum()
        lt = year.loc[year["long"], "gain"].sum()
        if st < 0:                          # this year's short-term loss: against long-term gains first
            use = min(-st, max(lt, 0.0))
            lt -= use
            st += use
        carry = [c for c in carry if fy - c[0] <= CARRY_FORWARD_YEARS and c[2] > 0]
        for c in carry:                     # old losses, oldest first
            if c[1] == "st":
                for kind in ("st", "lt"):
                    gain = st if kind == "st" else lt
                    use = min(c[2], max(gain, 0.0))
                    c[2] -= use
                    st, lt = (st - use, lt) if kind == "st" else (st, lt - use)
            else:
                use = min(c[2], max(lt, 0.0))
                c[2] -= use
                lt -= use
        if st < 0:
            carry.append([fy, "st", -st])
            st = 0.0
        if lt < 0:
            carry.append([fy, "lt", -lt])
            lt = 0.0
        out[fy] = (st * slab + lt * LTCG_RATE) * (1.0 + CESS)
    return pd.Series(out, dtype="float64")


def with_commission(returns: pd.Series, equity: pd.Series, trades: pd.DataFrame, broker: Broker) -> pd.Series:
    """Daily returns of a run made without a commission, less ``broker``'s on each recorded order."""
    t = trades.assign(date=pd.to_datetime(trades["date"]))
    fee = t["value_inr"].abs() * broker.commission_rate
    if broker.commission_cap > 0:
        fee = fee.clip(upper=broker.commission_cap)
    daily = fee.groupby(t["date"]).sum().reindex(returns.index).fillna(0.0)
    prev = equity.shift(1).reindex(returns.index).bfill()
    return returns - daily / prev


def inr_view(returns_usd: pd.Series, fx: pd.Series, broker: Broker, capital_usd: float = 10_000.0,
             lots: Optional[pd.DataFrame] = None, dividend_yield: Optional[pd.Series] = None,
             slab: float = DEFAULT_SLAB, other_lrs_inr: float = 0.0) -> Dict[str, float]:
    """A dollar book judged in rupees, before and after the account's costs and Indian tax.

    One deposit of ``capital_usd`` at the start and a full withdrawal at the end; ``returns_usd`` already
    net of the engine's per-order costs.  ``lots`` (``realised_lots`` scaled to this capital) are taxed by
    financial year; whatever is still held at the end is sold then (``lots`` must include that sale).
    ``dividend_yield`` (daily, gross) is for books whose returns count dividends reinvested whole: the
    US keeps 25% of each, and India adds its slab rate above that.  Money paid out from India before the
    end (plan fees, Indian tax, TCS until credited) is charged ``INDIA_RF`` to the end."""
    r = returns_usd.dropna()
    rate = fx.sort_index().ffill()
    at = lambda d: float(rate.asof(pd.Timestamp(d)))
    start, end = r.index[0], r.index[-1]
    years = (end - start).days / 365.25
    grow = lambda d: (1.0 + INDIA_RF) ** max((end - pd.Timestamp(d)).days / 365.25, 0.0)

    remit = capital_usd * at(start) * (1.0 + broker.fx_in)
    outlay = remit + gst_on_conversion(remit)
    paid = tcs(remit, other_lrs_inr) * (grow(start) - grow(pd.Timestamp(f"{financial_year(start) + 1}-07-31")))
    r_net = r - US_DIVIDEND_WITHHOLDING * dividend_yield.reindex(r.index).fillna(0.0) if dividend_yield is not None else r
    value_usd = capital_usd * float((1.0 + r_net).prod())
    gross_inr = capital_usd * float((1.0 + r).prod()) * at(end)
    proceeds = value_usd * at(end) * (1.0 - broker.fx_out)
    proceeds -= gst_on_conversion(proceeds) if broker.gst_on_withdrawal else 0.0
    fees = sum(broker.plan_fee_inr * grow(d) for d in pd.date_range(start, end, freq="12MS"))
    tax = 0.0
    if lots is not None and not lots.empty:
        tax += sum(v * grow(_fy_end(fy)) for fy, v in capital_gains_tax(lots, slab).items())
    if dividend_yield is not None:
        eq = capital_usd * (1.0 + r_net).cumprod().shift(1).fillna(capital_usd)
        tax_fx = pd.Series([tax_rate_on(rate, d) for d in r.index], index=r.index)
        div_inr = dividend_yield.reindex(r.index).fillna(0.0) * eq * tax_fx
        top_up = max(slab * (1.0 + CESS) - US_DIVIDEND_WITHHOLDING, 0.0)
        for fy, v in div_inr.groupby(div_inr.index.map(financial_year)).sum().items():
            tax += top_up * v * grow(_fy_end(fy))
    cagr = lambda final, first: (final / first) ** (1.0 / years) - 1.0
    return {"inr_cagr_before_costs": cagr(gross_inr, capital_usd * at(start)),
            "inr_cagr_after_costs": cagr(proceeds - fees - paid, outlay),
            "inr_cagr_after_tax": cagr(proceeds - fees - paid - tax, outlay),
            "fx_and_gst_inr": (outlay - capital_usd * at(start)) + (value_usd * at(end) - proceeds),
            "plan_fees_inr": fees, "tax_inr": tax, "years": years}


def report(runs_dir: Path, slab: float = DEFAULT_SLAB) -> Dict[str, Dict]:
    """Round 1's books and SPY held throughout, in INR at each broker: before costs, after them, after tax."""
    from us_engine.backtest import CAPITAL_USD, PANEL_DIR, _latest_runs, fund_returns

    fx = pd.read_parquet(PANEL_DIR / "market.parquet")["USDINR"]
    out: Dict[str, Dict] = {}
    for key, run in _latest_runs(Path(runs_dir), "us5-").items():
        r = pd.read_csv(run / "returns.csv", index_col=0, parse_dates=True)["return"].iloc[1:]
        eq = pd.read_csv(run / "equity.csv", index_col=0, parse_dates=True)["equity"]
        trades = pd.read_csv(run / "trades.csv", parse_dates=["date"])
        out[key] = {}
        for name, b in BROKERS.items():
            rr = with_commission(r, eq, trades, b)
            scale = CAPITAL_USD / float(eq.iloc[0])
            fee = trades["value_inr"].abs() * b.commission_rate
            fee = fee.clip(upper=b.commission_cap) if b.commission_cap > 0 else fee
            t = trades.assign(quantity=trades["quantity"] * scale, value_inr=trades["value_inr"] * scale,
                              cost_inr=(trades["cost_inr"] + fee) * scale)      # the commission is a cost of each lot
            last = pd.read_parquet(run / "weights.parquet").iloc[-1]
            held = {sym: float(w) * float(eq.iloc[-1]) * scale for sym, w in last.items() if w > 0}
            lots = realised_lots(t, fx, close_out=held, close_date=r.index[-1])
            out[key][name] = inr_view(rr, fx, b, CAPITAL_USD, lots=lots, slab=slab)
    any_run = next(iter(_latest_runs(Path(runs_dir), "us5-").values()))
    idx = pd.read_csv(any_run / "returns.csv", index_col=0, parse_dates=True).index[1:]
    spy = fund_returns("SPY", idx)
    spx = pd.read_parquet(PANEL_DIR / "market.parquet")["SPX"].reindex(idx)
    price = spx.pct_change().fillna(0.0)
    yearly = ((1 + spy).groupby(idx.year).prod() / (1 + price).groupby(idx.year).prod() - 1.0).clip(lower=0.0)
    div = pd.Series(0.0, index=idx)                 # SPY's yield, paid quarterly at each quarter's last session
    for q_end in idx.to_series().groupby([idx.year, idx.quarter]).max():
        div[q_end] = yearly[q_end.year] / 4.0
    out["SPY"] = {}
    for name, b in BROKERS.items():
        rr = spy.copy()
        rr.iloc[0] -= b.commission_rate                 # one buy at the start, one sale at the end
        rr.iloc[-1] -= b.commission_rate
        eq = CAPITAL_USD * (1 + rr - US_DIVIDEND_WITHHOLDING * div).cumprod()
        # dividends net of US tax are reinvested: each is a purchase, so it joins the cost
        tax_fx = pd.Series([tax_rate_on(fx, d) for d in idx], index=idx)
        reinvested = ((1 - US_DIVIDEND_WITHHOLDING) * div * eq.shift(1).fillna(CAPITAL_USD) * tax_fx).sum()
        lots = pd.DataFrame([{"date": idx[-1], "bought": idx[0], "symbol": "SPY",
                              "cost_inr": CAPITAL_USD * tax_rate_on(fx, idx[0]) + float(reinvested),
                              "proceeds_inr": float(eq.iloc[-1]) * tax_rate_on(fx, idx[-1]), "long": True}])
        out["SPY"][name] = inr_view(rr, fx, b, CAPITAL_USD, lots=lots, dividend_yield=div, slab=slab)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    from us_engine.backtest import RUNS_DIR
    from nse_engine.deployment import REPO_ROOT

    ap = argparse.ArgumentParser(prog="python -m us_engine.costs", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report", help="round 1's books and SPY after costs and tax, in INR")
    p.add_argument("--runs-dir", default=str(REPO_ROOT / RUNS_DIR))
    p.add_argument("--slab", type=float, default=DEFAULT_SLAB, help="your marginal slab rate before cess")
    args = ap.parse_args(argv)
    print(json.dumps(report(Path(args.runs_dir), args.slab), indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
