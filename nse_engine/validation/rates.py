"""The dated rates the cost models apply, where each comes from, and what is published but not applied yet
(tracker PL-15).

The rates stay where they are used: ``nse_engine.costs`` (shares and ETFs), ``ChargesConfig`` (F&O) and
``McxChargesConfig`` (MCX).  This module sets no rate.  It records:

- ``EQUITY``: every dated rate of the share model with its source, which a test holds the model's own
  functions to, on each change date, the day before it and today;
- ``FNO_SOURCES`` and ``MCX_SOURCES``: the source of every start date in those models' schedules, so a rate
  cannot be added without saying where it comes from;
- ``PENDING``: published rates the models do not apply yet.  Each names an open lesson of the ledger, whose
  review date fails the suite if the rate is still waiting then;
- ``CHECKED``: when the published schedule was last read, and ``check_due`` for the next reading: after each
  Budget (1 Feb) and each 1 April and 1 October, when Finance Act and exchange rates usually change.

    python -m nse_engine.validation.rates      # the tables and what is pending
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

#: The published schedule was last read on this day, against ``CHECKED_AGAINST``.
CHECKED = "2026-11-14"
CHECKED_AGAINST = "zerodha.com/charges (equity delivery, F&O futures and options, DP charge)"
#: A reading is due by these (month, day)s: after the Budget, and after the 1 April and 1 October changes.
REVIEW_DAYS = ((2, 15), (4, 15), (10, 15))
#: The first date of the share model's data; a row from here is the model's rate for every earlier date too.
START = "2006-01-01"

#: (charge, in force from, rate, source).  Rates are a share of the traded value; ``dp_sell`` is rupees.
EQUITY: Tuple[Tuple[str, str, float, str], ...] = (
    ("stt_delivery", START, 0.001, "the model's rate before 1 Jun 2006"),
    ("stt_delivery", "2006-06-01", 0.00125, "Finance Act 2006"),
    ("stt_delivery", "2012-07-01", 0.001, "Finance Act 2012"),
    ("stamp_buy", START, 0.0001, "state stamp duties, approximated by one rate"),
    ("stamp_buy", "2020-07-01", 0.00015, "the uniform stamp duty in force from 1 Jul 2020: 0.015% on delivery buys"),
    ("exchange", START, 0.0000325, "NSE's cash-market charge to 30 Sep 2024, its slabs approximated by one rate"),
    ("exchange", "2024-10-01", 0.0000297, "NSE's flat charge after SEBI's circular of 1 Jul 2024"),
    ("sebi", START, 0.000001, "SEBI's turnover fee, Rs 10 a crore; one rate for every date"),
    ("indirect_tax", START, 0.15, "service tax, approximated at its last rate"),
    ("indirect_tax", "2017-07-01", 0.18, "GST"),
    ("dp_sell", START, 15.93, "Zerodha's depository charge per stock sold, Rs 13.5 + GST; one figure for every date"),
)

#: The source of every start date in ``ChargesConfig``'s schedules (``kite_connect.options.options_config``).
FNO_SOURCES: Dict[str, str] = {
    "2004-10-01": "Finance (No. 2) Act 2004: STT begins.  Exchange and stamp rates: one rate for the years "
                  "before their own change",
    "2005-06-01": "Finance Act 2005",
    "2006-06-01": "Finance Act 2006",
    "2008-06-01": "Finance Act 2008: STT on exercised options",
    "2013-06-01": "Finance Act 2013",
    "2016-06-01": "Finance Act 2016",
    "2023-04-01": "Finance Act 2023",
    "2024-10-01": "Finance (No. 2) Act 2024 (STT); NSE's flat charge after SEBI's circular of 1 Jul 2024 (exchange)",
    "2026-04-01": "Finance Act 2026",
}
#: The same for ``McxChargesConfig`` (``nse_engine.mcx_costs``).
MCX_SOURCES: Dict[str, str] = {
    "2003-01-01": "the start of the data: one rate for the years before each change",
    "2013-07-01": "Finance Act 2013: CTT on non-agricultural commodity futures",
    "2024-10-01": "MCX's flat charge after SEBI's circular of 1 Jul 2024",
}

_NSE_2026 = ("NSE circular NSE/FA/73061 of 27 Feb 2026, as relayed (its own text unread): from 1 Mar 2026 the "
             "charge took in the investor protection fund levy, which the model had left out before it too")
#: Published rates the models do not apply yet.  Found on 14 Nov 2026, inside the RA1 freeze on cost
#: figures: they go in with the next cost-model version, on the owner's decision.
PENDING: Tuple[Dict[str, Any], ...] = (
    {"model": "equity", "charge": "exchange", "published": 0.0000307, "found": "2026-11-14", "source": _NSE_2026,
     "lesson": "L-RATES-2026"},
    {"model": "fno", "charge": "exchange_futures", "published": 0.0000183, "found": "2026-11-14",
     "source": _NSE_2026, "lesson": "L-RATES-2026"},
    {"model": "fno", "charge": "exchange_option", "published": 0.0003553, "found": "2026-11-14",
     "source": _NSE_2026, "lesson": "L-RATES-2026"},
    {"model": "equity", "charge": "dp_sell", "published": 15.34, "found": "2026-11-14",
     "source": "zerodha.com/charges: Rs 3.5 CDSL + Rs 9.5 Zerodha + GST a stock sold", "lesson": "L-RATES-2026"},
)


def model_rate(model: str, charge: str, day) -> float:
    """The rate the cost model itself applies on ``day``: ``model`` "equity", "fno" or "mcx"."""
    if model == "equity":
        from nse_engine import costs
        from nse_engine.config import CostConfig

        return {"stt_delivery": costs.stt_rate, "stamp_buy": costs.stamp_duty_rate, "exchange": costs.exchange_rate,
                "sebi": lambda d: costs.SEBI_FEE_RATE, "indirect_tax": costs.indirect_tax_rate,
                "dp_sell": lambda d: CostConfig().dp_charge_inr}[charge](day)
    from kite_connect.options.fno_costs import rate_on

    if model == "fno":
        from kite_connect.options.options_config import ChargesConfig as Config
    elif model == "mcx":
        from nse_engine.mcx_costs import McxChargesConfig as Config
    else:
        raise ValueError(f"model must be equity, fno or mcx, got {model!r}")
    return rate_on(getattr(Config(), charge), day)


def schedules(model: str) -> Dict[str, Tuple[Tuple[str, float], ...]]:
    """The dated schedules of the F&O or MCX model: {field: ((start, rate), ...)}."""
    from dataclasses import fields

    if model == "fno":
        from kite_connect.options.options_config import ChargesConfig as Config
    else:
        from nse_engine.mcx_costs import McxChargesConfig as Config
    cfg = Config()
    return {f.name: getattr(cfg, f.name) for f in fields(cfg) if isinstance(getattr(cfg, f.name), tuple)}


def check_due(checked: str = CHECKED) -> date:
    """The day by which the published schedule is read again: the first review day after ``checked``."""
    last = date.fromisoformat(checked)
    return min(d for d in (date(y, m, day) for y in (last.year, last.year + 1) for m, day in REVIEW_DAYS) if d > last)


def status(today: Optional[date] = None) -> str:
    """One line for the preflight ("!" when a reading is overdue)."""
    today = today or date.today()
    due = check_due()
    waiting = f"{len(PENDING)} published rate(s) not applied yet ({', '.join(sorted({p['lesson'] for p in PENDING}))})"
    line = f"read against the published schedule on {date.fromisoformat(CHECKED):%d %b %Y}; " \
           f"{waiting if PENDING else 'none pending'}; next reading by {due:%d %b %Y}"
    return f"! {line}: overdue" if today > due else line


def main() -> int:
    today = date.today()
    rows: List[str] = [f"{c:13} from {since}  {rate:<10g} {source}" for c, since, rate, source in EQUITY]
    print("Shares and ETFs (nse_engine.costs):\n  " + "\n  ".join(rows))
    for model, sources in (("fno", FNO_SOURCES), ("mcx", MCX_SOURCES)):
        print(f"{model.upper()} schedules:")
        for name, sched in schedules(model).items():
            print("  " + name + ": " + "; ".join(f"{s} {r:g}" for s, r in sched))
        print("  sources: " + "; ".join(f"{d} {s}" for d, s in sources.items()))
    print("Published, not applied yet:")
    for p in PENDING:
        print(f"  {p['model']}.{p['charge']}: the model applies {model_rate(p['model'], p['charge'], today):g}, "
              f"published {p['published']:g} ({p['source']})")
    print("Cost rates: " + status(today))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
