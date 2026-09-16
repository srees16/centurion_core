"""Fetch historical NIFTY500 constituent lists from the Wayback Machine.

Every distinct archived version (CDX ``collapse=digest``) of the official
``ind_nifty500list.csv`` is downloaded and labelled with its CAPTURE date.
A list captured on date D is known on D, so using it from D onward is
point-in-time safe (it may be stale, never look-ahead).

Output: data/nifty500_wayback_raw.json
    {"YYYY-MM-DD": [symbols...], ..., "_meta": {...}}

Run from anywhere:
    python scripts/fetch_wayback_nifty500.py
Then build the PIT file with scripts/build_pit_json.py.
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "nifty500_wayback_raw.json"

HEADERS = {"User-Agent": "centurion-research/1.0 (PIT universe backfill; polite crawler)"}
CDX_URL = "https://web.archive.org/cdx/search/cdx"
SOURCE_URLS = [
    "niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "nseindia.com/content/indices/ind_nifty500list.csv",
    "www1.nseindia.com/content/indices/ind_nifty500list.csv",
    "archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
]
REQUEST_PAUSE_SECS = 3.0
MAX_RETRIES = 4


def _get(url: str, params: Optional[dict] = None, timeout: int = 60) -> Optional[requests.Response]:
    """GET with polite pauses and exponential backoff on 429/5xx/offline pages."""
    delay = REQUEST_PAUSE_SECS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            offline = "Temporarily Offline" in r.text[:2000] if r.status_code == 200 else False
            if r.status_code == 200 and not offline:
                return r
            logger.warning("GET %s -> %s%s (attempt %d)", url, r.status_code,
                           " offline page" if offline else "", attempt)
            if r.status_code == 404:
                return None
        except requests.RequestException as exc:
            logger.warning("GET %s failed: %s (attempt %d)", url, exc, attempt)
        time.sleep(delay)
        delay *= 2
    return None


def list_captures(source: str) -> List[Tuple[str, str]]:
    """Return [(timestamp, original_url)] of distinct 200-status captures."""
    r = _get(CDX_URL, params={
        "url": source, "filter": "statuscode:200", "collapse": "digest",
        "fl": "timestamp,original",
    })
    time.sleep(REQUEST_PAUSE_SECS)
    if r is None:
        return []
    out = []
    for line in r.text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            out.append((parts[0], parts[1]))
    return out


def parse_symbols(text: str) -> List[str]:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    header = [h.strip().lower() for h in rows[0]]
    sym_col = next((i for i, h in enumerate(header) if "symbol" in h), None)
    if sym_col is None:
        return []
    syms = []
    for row in rows[1:]:
        if len(row) > sym_col:
            s = row[sym_col].strip()
            if s and s.lower() != "symbol":
                syms.append(s)
    return syms


def main() -> Dict[str, List[str]]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    captures: Dict[str, str] = {}  # timestamp -> original
    for source in SOURCE_URLS:
        caps = list_captures(source)
        logger.info("CDX %s: %d distinct captures", source, len(caps))
        for ts, orig in caps:
            captures.setdefault(ts, orig)

    results: Dict[str, List[str]] = {}
    fetched_from: Dict[str, str] = {}
    failures: List[str] = []
    for ts, orig in sorted(captures.items()):
        url = f"https://web.archive.org/web/{ts}id_/{orig}"
        r = _get(url)
        time.sleep(REQUEST_PAUSE_SECS)
        syms = parse_symbols(r.text) if r is not None else []
        label = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        # Sanity: a NIFTY500 list has ~500 names; skip error pages / partial files
        if len(syms) < 450:
            logger.warning("Capture %s (%s): %d symbols — skipped", ts, orig, len(syms))
            failures.append(ts)
            continue
        if label in results and len(results[label]) >= len(syms):
            continue
        results[label] = sorted(set(syms))
        fetched_from[label] = url
        logger.info("Capture %s: %d symbols", label, len(syms))

    payload: Dict = dict(sorted(results.items()))
    payload["_meta"] = {
        "source": "Wayback Machine captures of NSE/niftyindices ind_nifty500list.csv",
        "labels": "capture date (YYYY-MM-DD)",
        "urls": fetched_from,
        "failed_captures": failures,
        "fetched_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved {len(results)} snapshots to {OUT_PATH}")
    for k, v in results.items():
        print(f"  {k}: {len(v)} symbols")
    return results


if __name__ == "__main__":
    main()
