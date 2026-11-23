"""Time zones."""

from __future__ import annotations

from datetime import timedelta, timezone

#: India Standard Time: NSE's clock, and the one every owner-facing date is written in.
IST = timezone(timedelta(hours=5, minutes=30))
