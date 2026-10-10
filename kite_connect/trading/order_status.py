"""One reading of Kite order statuses and tags for every placement path (tracker LN-T3).

An order is *failed* when its status starts with REJECTED or CANCELLED (so
"CANCELLED AMO" counts), *terminal* when failed or COMPLETE, and *placed*
otherwise: OPEN, TRIGGER PENDING and every interim state, including the
"AMO REQ RECEIVED" every after-market order sits in until the open.  A retry
or a re-run must treat a placed order as already sent, whatever its stage.
Kite tags are alphanumeric, at most 20 characters, so symbols such as M&M or
BAJAJ-AUTO are stripped to their letters and digits.
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
