"""Counts only in the public Actions logs (the owner's call, 26 Oct 2026).

This repository is public, and so are its GitHub Actions logs.  A log line holding book detail (symbols,
quantities, prices, amounts) is logged with ``extra=book_detail(public)``: on Actions the root handlers print
``public``, its counts, instead (nothing when it is empty), as the live session prints counts only.  Run locally,
every line prints in full.  The detail is in the emails and Neon.

    logger.info("PAPER BUY %s x %d", sym, qty, extra=book_detail())                  # dropped on Actions
    logger.info("Daily snapshot: equity=%.0f open=%d", eq, n, extra=book_detail(f"Daily snapshot: open={n}"))
"""

from __future__ import annotations

import logging
import os
from typing import Dict


def book_detail(public: str = "") -> Dict[str, str]:
    """``extra=`` for a log line holding book detail: ``public`` is what Actions prints instead ('' drops it)."""
    return {"public_log": public}


class PublicLogFilter(logging.Filter):
    """Prints a ``book_detail`` line as its counts, or drops it; other lines pass unchanged."""

    def filter(self, record: logging.LogRecord) -> bool:
        public = getattr(record, "public_log", None)
        if public is None:
            return True
        if not public:
            return False
        record.msg, record.args = public, ()
        return True


def install_public_log_filter() -> bool:
    """On Actions, the root handlers print ``book_detail`` lines as counts; True when installed."""
    if os.getenv("GITHUB_ACTIONS") != "true":
        return False
    for handler in logging.getLogger().handlers:
        handler.addFilter(PublicLogFilter())
    return True
