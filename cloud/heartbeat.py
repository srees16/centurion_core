"""Heartbeats for long unattended runs.

A heartbeat URL is any endpoint that reads a request as "still alive":
healthchecks.io, an UptimeRobot heartbeat monitor, Better Stack, or anything
self-hosted. The healthchecks.io sub-path convention is used — ``/start`` when
work begins, the bare URL for progress, ``/fail`` on an exception — and
services that do not know those paths simply see a ping.

Set ``CENTURION_HEARTBEAT_URL`` to switch it on. With no URL the heartbeat
still writes its state file, which is what ``cloud.kaggle_local status`` and a
resumed session read.

Nothing here raises: a monitoring failure must never kill the job it watches.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ENV_URL = "CENTURION_HEARTBEAT_URL"
DEFAULT_STATE_PATH = "heartbeat.json"
_TIMEOUT = 10


class Heartbeat:
    """Ping a monitoring URL and keep a state file describing the run.

    ``min_interval`` throttles progress pings so a fast loop cannot hammer the
    monitor; ``start``, ``fail`` and ``done`` are never throttled.
    """

    def __init__(self, url: Optional[str] = None, *, name: str = "",
                 state_path: Optional[str | Path] = None, min_interval: float = 30.0) -> None:
        self.url = (url or os.getenv(ENV_URL) or "").rstrip("/")
        self.name = name
        self.state_path = Path(state_path) if state_path else Path(DEFAULT_STATE_PATH)
        self.min_interval = min_interval
        self.started_at = time.time()
        self._last_ping = 0.0
        self.state: Dict[str, Any] = {"name": name, "status": "created", "message": ""}

    # ── lifecycle ────────────────────────────────────────────────

    def start(self, message: str = "", **extra: Any) -> None:
        self.started_at = time.time()
        self._update("running", message, extra)
        self._post("/start", message, force=True)

    def progress(self, message: str = "", **extra: Any) -> None:
        """Say the run is alive and how far it has got (throttled)."""
        self._update("running", message, extra)
        self._post("", message, force=False)

    def done(self, message: str = "", **extra: Any) -> None:
        self._update("completed", message, extra)
        self._post("", message, force=True)

    def paused(self, message: str = "", **extra: Any) -> None:
        """Out of time but healthy — the next session resumes from the state file."""
        self._update("paused", message, extra)
        self._post("", message, force=True)

    def fail(self, message: str = "", **extra: Any) -> None:
        self._update("failed", message, extra)
        self._post("/fail", message, force=True)

    # ── internals ────────────────────────────────────────────────

    def _update(self, status: str, message: str, extra: Dict[str, Any]) -> None:
        self.state.update(status=status, message=message,
                          elapsed_s=round(time.time() - self.started_at, 1),
                          updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.state.update(extra)
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.state, indent=2, default=str))
        except OSError as exc:
            logger.debug("heartbeat state write failed: %s", exc)

    def _post(self, path: str, message: str, *, force: bool) -> None:
        if not self.url:
            return
        now = time.time()
        if not force and now - self._last_ping < self.min_interval:
            return
        self._last_ping = now
        try:
            import requests

            requests.post(f"{self.url}{path}", data=(message or "")[:10000], timeout=_TIMEOUT)
        except Exception as exc:                      # noqa: BLE001 - monitoring is best-effort
            logger.debug("heartbeat ping failed: %s", exc)


def read_state(state_path: str | Path) -> Dict[str, Any]:
    """Return a heartbeat state file, or ``{}`` if it is missing or unreadable."""
    try:
        return json.loads(Path(state_path).read_text())
    except (OSError, ValueError):
        return {}
