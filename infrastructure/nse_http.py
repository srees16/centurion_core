"""Shared browser-like session for nseindia.com.

NSE rejects plain HTTP clients: every request needs a browser User-Agent and a
cookie obtained by visiting the homepage first. Four modules used to carry
their own copy of these headers and their own session bootstrap; they all call
in here now.

`requests` is imported lazily so that importing this module stays cheap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import requests

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

#: Headers accepted by both the archive (CSV/ZIP) and the api/ (JSON) endpoints.
NSE_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

_SESSION: Optional["requests.Session"] = None


def new_session(accept: Optional[str] = None) -> "requests.Session":
    """Return a fresh session with NSE cookies pre-loaded.

    `accept` overrides the Accept header for endpoints that want a specific
    content type. Cookie pre-fetch failures are logged, not raised: the caller
    still gets a usable session and its own request will report the error.
    """
    import requests

    sess = requests.Session()
    sess.headers.update(NSE_HEADERS)
    if accept:
        sess.headers["Accept"] = accept
    try:
        sess.get("https://www.nseindia.com", timeout=10)
    except Exception as exc:
        logger.debug("NSE session cookie pre-fetch failed: %s", exc)
    return sess


def get_session(reset: bool = False) -> "requests.Session":
    """Return the process-wide NSE session, rebuilding it on `reset` (e.g. after a 403)."""
    global _SESSION
    if _SESSION is None or reset:
        _SESSION = new_session()
    return _SESSION
