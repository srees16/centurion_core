"""One reading of Kite order statuses and tags for every placement path (tracker LN-T3).

An order is *failed* when its status starts with REJECTED or CANCELLED (so
"CANCELLED AMO" counts), *terminal* when failed or COMPLETE, and *placed*
otherwise: OPEN, TRIGGER PENDING and every interim state, including the
"AMO REQ RECEIVED" every after-market order sits in until the open.  A retry
or a re-run must treat a placed order as already sent, whatever its stage.
Kite tags are alphanumeric, at most 20 characters, so symbols such as M&M or
BAJAJ-AUTO are stripped to their letters and digits.  ``rejection_hint`` turns
Kite's rejection text into what to do about it (tracker AL4).
"""

from __future__ import annotations

import re

TAG_MAX = 20


def order_failed(status) -> bool:
    s = str(status or "").strip().upper()
    return s.startswith("REJECTED") or s.startswith("CANCELLED")


def order_terminal(status) -> bool:
    return order_failed(status) or str(status or "").strip().upper() == "COMPLETE"


def order_placed(status) -> bool:
    """A booked order at any stage short of rejection or cancellation."""
    return bool(str(status or "").strip()) and not order_failed(status)


def kite_tag(tag: str) -> str:
    """``tag`` as Kite accepts it: letters and digits only, at most 20 characters."""
    return re.sub(r"[^A-Za-z0-9]", "", str(tag or ""))[:TAG_MAX]


#: Kite's rejection text -> what to do (tracker AL4, LN-T17); the first match wins.
SELL_HINTS = ((("authoris", "authoriz", "edis", "tpin", "ddpi", "poa"),
               "authorise the sale with CDSL TPIN, and enable DDPI (Console > Account) so stop and exit sells go "
               "through unattended"),
              (("holding", "quantity", "insufficient"),
               "the sell asked for more shares than are held (a stop on a position trimmed the same day?)"),
              (("circuit", "band", "price"), "the price was outside the day's band (a circuit)"))
BUY_HINTS = ((("margin", "fund", "insufficient", "balance"),
              "not enough cash at the 09:00 check: an after-market buy is checked before the day's sells execute "
              "(tracker ST2)"),
             (("circuit", "band", "price"), "the price was outside the day's band (a circuit)"))


def rejection_hint(message, side) -> str:
    """What to do about Kite's rejection ``message`` of a ``side`` order; "" when it is not recognised."""
    text = str(message or "").lower()
    for words, hint in (SELL_HINTS if str(side or "").upper() == "SELL" else BUY_HINTS):
        if any(w in text for w in words):
            return hint
    return ""
