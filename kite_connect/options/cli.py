"""
Options toolkit command line: chain, strategy selection, pre-trade report,
execution and monitoring on NSE options through Kite Connect.

    python -m kite_connect.options.cli chain --underlying NIFTY
    python -m kite_connect.options.cli context --underlying NIFTY          # IV history + positioning (OD1, OD2)
    python -m kite_connect.options.cli select --view moderate_bull --dte 6 --underlying NIFTY
    python -m kite_connect.options.cli refresh                             # bring the local data up to date
    python -m kite_connect.options.cli report --underlying NIFTY --legs "BUY CE 25000, SELL CE 25150"
    python -m kite_connect.options.cli trade  --underlying NIFTY --legs "BUY CE 25000, SELL CE 25150"   # paper
    python -m kite_connect.options.cli trade  ... --dry-run      # log the orders, send nothing
    python -m kite_connect.options.cli trade  ... --live         # real orders, typed confirmation
    python -m kite_connect.options.cli monitor
    python -m kite_connect.options.cli demo --underlying NIFTY --view moderate_bull

Everything except ``select``, ``context`` and ``refresh`` needs today's Kite
token (the daily email-link login, U23).  ``context``, and ``select`` with
``--underlying``, read the local F&O store: today's IV against its history
sets the selector's IV level instead of a typed guess.  Safety, in order:

1. Paper is the default; ``--live`` is the only way to send real orders.
2. ``--live`` prints the pre-trade report and needs ``PLACE <n> ORDERS``
   typed at a terminal (never in CI), and the registered static IP.
3. A basket that breaches a hard limit (``LimitsConfig``: max loss per
   trade, max lots per leg, allowed underlyings) is refused in every mode.
4. Every order request and response is logged to ``data/options/orders.jsonl``.
"""

from __future__ import annotations

import argparse
import logging
import math
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from kite_connect.options.basket_executor import BasketExecutor, ExecutionReport
from kite_connect.options.broker import Broker, connect
from kite_connect.options.instruments import InstrumentResolver
from kite_connect.options.iv_history import iv_context, iv_history, live_iv30
from kite_connect.options.live_chain import IST, chain_summary, days_to_expiry, fetch_chain
from kite_connect.options.options_config import OptionsConfig, SelectorConfig, load_config
from kite_connect.options.position_monitor import PositionLedger, monitor
from kite_connect.options.positioning import positioning
from kite_connect.options.pretrade import LegSpec, PreTradeReport, build_report, order_legs, parse_legs
from kite_connect.options.selector import VIEWS, Candidate, MarketContext, select, strike_for
from kite_connect.options.theory import BUY, CALL, PUT, SELL

logger = logging.getLogger(__name__)

CHAIN_STRIKES_EACH_SIDE = 15
DEMO_DIR = Path("data/options/demo")


# ── building blocks ─────────────────────────────────────────────

def _expiry(resolver: InstrumentResolver, underlying: str, raw: Optional[str], today: date) -> date:
    """``--expiry`` when given, else the nearest expiry at least a day away."""
    return date.fromisoformat(raw) if raw else resolver.nearest_expiry(underlying, today, min_days=1)


def prepare(broker: Broker, resolver: InstrumentResolver, underlying: str, expiry: date, spec: Sequence[LegSpec],
            lots: int, cfg: OptionsConfig, now: datetime) -> PreTradeReport:
    """The pre-trade report of a leg spec from live quotes, the chain's IVs and Kite's basket margin."""
    chain, spot = fetch_chain(broker, resolver, underlying, expiry, now, cfg.market.risk_free_rate,
                              CHAIN_STRIKES_EACH_SIDE)
    contracts = [resolver.resolve(underlying, expiry, k, t) for _, t, k, _ in spec]
    quotes = broker.quotes([c.quote_key for c in contracts])
    ivs = dict(zip(chain["tradingsymbol"], chain["iv_strike"]))       # an ITM leg takes its OTM mirror's IV
    legs = order_legs(spec, resolver, underlying, expiry, quotes, lots, cfg.limits.limit_slippage_cap, ivs)
    try:
        margins = broker.basket_margins([l.kite_order() for l in legs])
    except Exception as exc:                              # noqa: BLE001 - reported, the model's charges remain
        logger.warning("Kite basket margin unavailable: %s", exc)
        margins = None
    return build_report(legs, spot, days_to_expiry(expiry, now), chain_summary(chain, spot)["atm_iv"], lots,
                        now.date(), cfg, margins)


def market_context(underlying: str, cfg: OptionsConfig, broker: Optional[Broker] = None,
                   resolver: Optional[InstrumentResolver] = None,
                   now: Optional[datetime] = None) -> Tuple[List[str], Optional[str]]:
    """Today's IV against its history and the market's positioning (OD1, OD2), and the IV level
    for the selector (None when unavailable).  Live IV with a broker, else the store's last session.
    Never fails the caller: missing data becomes a line saying so."""
    from nse_engine.data.fo_store import load_participant_oi

    lines: List[str] = []
    level = None
    try:
        live = live_iv30(broker, resolver, underlying, now, cfg) if broker is not None else None
        ctx = iv_context(underlying, cfg, live)
        lines += ctx.lines()
        level = ctx.level
    except Exception as exc:                              # noqa: BLE001 - context only
        lines.append(f"IV context unavailable for {underlying}: {exc} (run the refresh command)")
    try:
        pos = positioning(load_participant_oi(cfg.data.fo_store), cfg.data.lookback_sessions)
        lines += pos.lines() if pos else ["Positioning unavailable: no participant data in the store"]
    except Exception as exc:                              # noqa: BLE001 - context only
        lines.append(f"Positioning unavailable: {exc}")
    return lines, level


def confirm_live(report: PreTradeReport, stdin=sys.stdin, ask=input) -> bool:
    """The typed confirmation for real orders: a terminal, and ``PLACE <n> ORDERS`` exactly."""
    if not stdin.isatty():
        print("live orders need a terminal for the typed confirmation; refused")
        return False
    phrase = f"PLACE {len(report.legs)} ORDERS"
    print(report.text())
    return ask(f"\nType '{phrase}' to send these real orders: ").strip() == phrase


def run_trade(broker: Broker, report: PreTradeReport, mode: str, cfg: OptionsConfig, now: datetime,
              ledger: PositionLedger) -> Optional[ExecutionReport]:
    """Execute a report's basket in ``mode`` and record what filled; None when the hard limits refuse it."""
    if not report.ok:
        print("REFUSED by the hard limits: " + "; ".join(report.violations))
        return None
    execution = BasketExecutor(broker, mode, cfg.limits).execute(report.legs, tag=f"opt{now:%m%d%H%M}")
    if mode != "dry_run":
        ledger.record(report, execution, now)
    return execution


def _step(strikes: List[float], strike: float, n: int) -> float:
    i = strikes.index(strike)
    return strikes[min(max(i + n, 0), len(strikes) - 1)]


def vertical_spec(candidate: Candidate, spot: float, strikes: List[float], cfg: SelectorConfig) -> List[LegSpec]:
    """A two-leg spread from the selector's strike guidance (M6 ch. 2, 3, 7, 8)."""
    s, n = candidate.strikes, cfg.spread_strikes

    def pick(role: str, option_type: str) -> float:
        return strike_for(s[role], option_type, spot, strikes, cfg)

    if candidate.strategy == "Bull Call Spread":
        lo = pick("buy CE", CALL)
        return [(BUY, CALL, lo, 1), (SELL, CALL, _step(strikes, lo, n), 1)]
    if candidate.strategy == "Bull Put Spread":
        hi, lo = pick("sell PE", PUT), pick("buy PE", PUT)
        return [(BUY, PUT, lo if lo < hi else _step(strikes, hi, -n), 1), (SELL, PUT, hi, 1)]
    if candidate.strategy == "Bear Put Spread":
        hi, lo = pick("buy PE", PUT), pick("sell PE", PUT)
        return [(BUY, PUT, hi, 1), (SELL, PUT, lo if lo < hi else _step(strikes, hi, -n), 1)]
    if candidate.strategy == "Bear Call Spread":
        hi, lo = pick("buy CE", CALL), pick("sell CE", CALL)
        return [(BUY, CALL, hi if hi > lo else _step(strikes, lo, n), 1), (SELL, CALL, lo, 1)]
    raise ValueError(f"{candidate.strategy} is not a two-leg spread")


# ── commands ────────────────────────────────────────────────────

def cmd_chain(args, cfg: OptionsConfig) -> int:
    now = datetime.now(IST)
    broker = connect()
    resolver = InstrumentResolver(broker.instruments())
    expiry = _expiry(resolver, args.underlying, args.expiry, now.date())
    chain, spot = fetch_chain(broker, resolver, args.underlying, expiry, now, cfg.market.risk_free_rate, args.strikes)
    s = chain_summary(chain, spot)
    print(f"{args.underlying} {expiry} ({days_to_expiry(expiry, now):.1f} days): spot {spot:,.2f}, ATM {s['atm_strike']:g}, "
          f"ATM IV {s['atm_iv']:.1%}, max pain {s['max_pain']:g}, PCR {s['pcr']:.2f}")
    print("\n".join(market_context(args.underlying.upper(), cfg, broker, resolver, now)[0]))
    print(chain.round(4).to_string(index=False))
    return 0


def cmd_context(args, cfg: OptionsConfig) -> int:
    """Today's IV read against its history, and positioning (offline unless ``--live``)."""
    broker = resolver = now = None
    if args.live:
        now = datetime.now(IST)
        broker = connect()
        resolver = InstrumentResolver(broker.instruments())
    lines, level = market_context(args.underlying.upper(), cfg, broker, resolver, now)
    print("\n".join(lines))
    return 0 if level else 1


def cmd_refresh(args, cfg: OptionsConfig) -> int:
    """Bring the local data up to date: the archive's last ``--days``, the equity store (which runs
    the trial registry's dry-run check), the F&O store, then the IV histories."""
    start = (date.today() - timedelta(days=args.days)).isoformat()
    steps = [["-m", "nse_engine.data.archive", "--start", start, "--no-reference",
              "--kinds", "equity,delivery,indices,corpact,fo,participant"],
             ["-m", "runners.run_nse_engine", "build-store"],
             ["-m", "nse_engine.data.fo_store", "--store", cfg.data.fo_store]]
    for step in steps:
        print("$ python " + " ".join(step), flush=True)
        subprocess.run([sys.executable, *step], check=True)
    for symbol in args.symbols:
        h = iv_history(symbol, cfg)
        print(f"IV history {symbol}: {len(h)} sessions to {str(h['date'].iloc[-1])[:10]}")
    return 0


def cmd_select(args, cfg: OptionsConfig) -> int:
    iv_level = args.iv_level
    if args.underlying:
        lines, level = market_context(args.underlying.upper(), cfg)
        print("\n".join(lines) + "\n")
        iv_level = iv_level or level
    ctx = MarketContext(view=args.view, days_to_expiry=args.dte, vol_view=args.vol_view, days_to_target=args.target_days,
                        iv_level=iv_level or "normal", rich_side=args.rich_side, event=args.event,
                        event_vs_consensus=args.event_vs_consensus, range_bound=args.range_bound,
                        cost_sensitive=args.cost_sensitive)
    for i, c in enumerate(select(ctx, cfg.selector), 1):
        print(f"{i}. {c.strategy} (M6 ch. {c.chapter}) score {c.score:.2f}  strikes {c.strikes}")
        for line in c.reasons + [f"note: {w}" for w in c.warnings]:
            print(f"     {line}")
    return 0


def cmd_report_or_trade(args, cfg: OptionsConfig, trade: bool) -> int:
    now = datetime.now(IST)
    mode = "live" if getattr(args, "live", False) else "dry_run" if getattr(args, "dry_run", False) else "paper"
    broker = connect(for_orders=(mode == "live"))
    resolver = InstrumentResolver(broker.instruments())
    expiry = _expiry(resolver, args.underlying, args.expiry, now.date())
    report = prepare(broker, resolver, args.underlying.upper(), expiry, parse_legs(args.legs), args.lots, cfg, now)
    if not trade:
        print(report.text())
        return 0 if report.ok else 2
    if mode == "live" and not confirm_live(report):
        print("not confirmed: nothing sent")
        return 1
    if mode != "live":
        print(report.text())
    execution = run_trade(broker, report, mode, cfg, now, PositionLedger())
    if execution is None:
        return 2
    print(execution.text())
    return 0 if execution.completed or mode == "dry_run" else 1


def cmd_monitor(args, cfg: OptionsConfig) -> int:
    statuses = monitor(connect(), PositionLedger(), datetime.now(IST), cfg.market.risk_free_rate)
    print("\n".join(s.text() for s in statuses) or "no open positions")
    return 0


def cmd_demo(args, cfg: OptionsConfig) -> int:
    """End to end in paper mode: nearest expiry, chain, selector, pre-trade report, paper fill, monitor."""
    now = datetime.now(IST)
    out: List[str] = []

    def say(text: str) -> None:
        print(text)
        out.append(text)

    broker = connect()
    resolver = InstrumentResolver(broker.instruments())
    underlying = args.underlying.upper()
    expiry = resolver.nearest_expiry(underlying, now.date(), min_days=1)
    chain, spot = fetch_chain(broker, resolver, underlying, expiry, now, cfg.market.risk_free_rate,
                              CHAIN_STRIKES_EACH_SIDE)
    s = chain_summary(chain, spot)
    dte = days_to_expiry(expiry, now)
    say(f"1. CHAIN {underlying} {expiry} ({dte:.1f} days): spot {spot:,.2f}, ATM {s['atm_strike']:g}, "
        f"ATM IV {s['atm_iv']:.1%}, max pain {s['max_pain']:g}, PCR {s['pcr']:.2f}, {len(chain)} contracts quoted")
    context, level = market_context(underlying, cfg, broker, resolver, now)
    say("\n".join(context))
    ranked = select(MarketContext(view=args.view, days_to_expiry=max(int(math.floor(dte)), 1),
                                  iv_level=level or "normal"), cfg.selector)
    spreads = [c for c in ranked if c.strategy in ("Bull Call Spread", "Bull Put Spread", "Bear Put Spread",
                                                    "Bear Call Spread")]
    if not spreads:
        raise SystemExit(f"the demo trades two-leg spreads: view {args.view!r} gives none")
    pick = spreads[0]
    spec = vertical_spec(pick, spot, sorted(chain["strike"].unique()), cfg.selector)
    say(f"2. SELECTOR ({args.view}): {pick.strategy} (M6 ch. {pick.chapter}), guidance {pick.strikes}; legs "
        + ", ".join(f"{side} {t} {k:g}" for side, t, k, _ in spec))
    report = prepare(broker, resolver, underlying, expiry, spec, args.lots, cfg, now)
    say("3. " + report.text())
    ledger = PositionLedger(DEMO_DIR / "positions.json")
    execution = run_trade(broker, report, "paper", cfg, now, ledger)
    if execution is not None:
        say("4. " + execution.text())
        say("5. MONITOR\n" + "\n".join(st.text() for st in monitor(broker, ledger, datetime.now(IST),
                                                                       cfg.market.risk_free_rate)))
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    path = DEMO_DIR / f"demo_{now:%Y%m%d_%H%M}.txt"
    path.write_text("\n\n".join(out) + "\n")
    print(f"\nwritten: {path}")
    return 0 if execution is not None and execution.completed else 1


# ── entry point ─────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Options toolkit on Kite Connect (paper by default)")
    p.add_argument("--config", help="TOML overrides of OptionsConfig (also $CENTURION_OPTIONS_CONFIG)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chain", help="live option chain with IV, Greeks, max pain and PCR")
    c.add_argument("--underlying", default="NIFTY")
    c.add_argument("--expiry", help="YYYY-MM-DD (default: nearest a day or more away)")
    c.add_argument("--strikes", type=int, default=CHAIN_STRIKES_EACH_SIDE, help="strikes each side of ATM")

    x = sub.add_parser("context", help="today's IV against its history, and FII / DII positioning (OD1, OD2)")
    x.add_argument("--underlying", default="NIFTY")
    x.add_argument("--live", action="store_true", help="today's IV from live Kite quotes (default: the store's last session)")

    r = sub.add_parser("refresh", help="bring the archive, the equity and F&O stores and the IV histories up to date")
    r.add_argument("--days", type=int, default=14, help="archive days to (re)check")
    r.add_argument("--symbols", nargs="*", default=["NIFTY", "BANKNIFTY"], help="IV histories to update")

    s = sub.add_parser("select", help="rank Module 6 strategies for a view (no Kite needed)")
    s.add_argument("--view", required=True, choices=VIEWS)
    s.add_argument("--dte", type=int, required=True, help="days to expiry")
    s.add_argument("--vol-view", default="flat", choices=("rising", "falling", "flat"))
    s.add_argument("--target-days", type=int)
    s.add_argument("--underlying", help="read today's IV level from its history (the store) instead of --iv-level")
    s.add_argument("--iv-level", choices=("low", "normal", "high", "very_high"),
                   help="your own IV level; default: from --underlying's history, else normal")
    s.add_argument("--rich-side", choices=("puts", "calls"))
    s.add_argument("--event", action="store_true")
    s.add_argument("--event-vs-consensus", choices=("differs", "matches"))
    s.add_argument("--range-bound", action="store_true")
    s.add_argument("--cost-sensitive", action="store_true")

    for name, text in (("report", "pre-trade report of a basket"), ("trade", "report, then execute (paper by default)")):
        t = sub.add_parser(name, help=text)
        t.add_argument("--underlying", default="NIFTY")
        t.add_argument("--expiry", help="YYYY-MM-DD (default: nearest a day or more away)")
        t.add_argument("--legs", required=True, help='e.g. "BUY CE 25000, SELL CE 25150" or "SELL CE 24900, BUY 2 CE 25100"')
        t.add_argument("--lots", type=int, default=1)
        if name == "trade":
            mode = t.add_mutually_exclusive_group()
            mode.add_argument("--dry-run", action="store_true", help="log the orders, send nothing")
            mode.add_argument("--live", action="store_true", help="real orders, after the typed confirmation")

    sub.add_parser("monitor", help="open positions: P&L, Greeks, breakeven and stop-loss alerts")

    d = sub.add_parser("demo", help="end to end in paper mode on the nearest expiry")
    d.add_argument("--underlying", default="NIFTY")
    d.add_argument("--view", default="moderate_bull", choices=("moderate_bull", "moderate_bear"))
    d.add_argument("--lots", type=int, default=1)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.cmd == "chain":
        return cmd_chain(args, cfg)
    if args.cmd == "select":
        return cmd_select(args, cfg)
    if args.cmd == "context":
        return cmd_context(args, cfg)
    if args.cmd == "refresh":
        return cmd_refresh(args, cfg)
    if args.cmd in ("report", "trade"):
        return cmd_report_or_trade(args, cfg, trade=args.cmd == "trade")
    if args.cmd == "monitor":
        return cmd_monitor(args, cfg)
    return cmd_demo(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
