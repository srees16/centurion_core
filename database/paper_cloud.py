"""
Paper Trading Cloud Sync — Dual-write to Neon PostgreSQL.

After each local SQLite write (position, snapshot, signal, weekly checkpoint),
the PaperTrader calls into this module to persist the same data to Neon.

The Paper Dashboard UI reads from Neon first (cloud), falling back to
local SQLite when the cloud DB is unavailable.

All writes are best-effort: a Neon failure never blocks the trading loop.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

_cloud: Optional["PaperCloudSync"] = None


def _to_utc(value):
    try:
        ts = pd.Timestamp(value)
        if ts.tz is None:
            ts = ts.tz_localize("Asia/Kolkata")
        return ts.tz_convert("UTC")
    except Exception:
        return None


def _epoch_from_state(state) -> Optional[pd.Timestamp]:
    """UTC timestamp of the current book's epoch, or None if no book was started."""
    ep = (state or {}).get("epoch")
    return _to_utc(ep) if ep else None


_NSE_OPEN = pd.Timedelta(hours=9, minutes=15)


def _book_start_from_state(state) -> Optional[pd.Timestamp]:
    """UTC instant from which position rows belong to the current book.

    Not the epoch itself: engine fills are stamped at the session open, so a
    book started during a trading day (epoch 10:46 IST) fills its first orders
    at that day's 09:15 IST open, before the epoch. Filtering positions on the
    epoch hid all 21 fills of the 17 Sep 2026 book from the trade monitor and
    would have dropped them on the next restore. ``start_new_book`` records
    ``book_start``; a book started before that key existed begins at the NSE
    open of its epoch's day, or at the epoch if that is earlier.
    """
    state = state or {}
    if state.get("book_start"):
        return _to_utc(state["book_start"])
    ep = _epoch_from_state(state)
    if ep is None:
        return None
    day_open = (ep.tz_convert("Asia/Kolkata").normalize() + _NSE_OPEN).tz_convert("UTC")
    return min(ep, day_open)


def _records(df) -> List[dict]:
    if df is None or getattr(df, "empty", True):
        return []
    out = []
    for rec in df.to_dict("records"):
        out.append({k: (None if (isinstance(v, float) and v != v) else v) for k, v in rec.items()})
    return out


#: Columns added to paper_sessions after the table first existed (create_all
#: never alters a table, so a live book needs ALTER TABLE).  name -> DDL type.
SESSION_COLUMNS_ADDED = {"drawdown_state": "VARCHAR(12) DEFAULT 'normal'", "drawdown_pct": "FLOAT DEFAULT 0"}


#: paper_cloud_state key holding the distribution-shift state (JSON).  The
#: paper runner's disk is discarded after every GitHub Actions run, so the
#: state must live here for the next session's plan to see yesterday's verdict.
SHIFT_STATE_KEY = "distribution_shift_state"


#: Env var naming the Postgres schema of a second paper book (tracker D1).
#: Unset = the deployed book in the default schema.
ENV_SCHEMA = "CENTURION_PAPER_SCHEMA"
_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")


def paper_schema_from_env() -> Optional[str]:
    """The paper book's schema from ``CENTURION_PAPER_SCHEMA`` (None = default schema)."""
    raw = (os.environ.get(ENV_SCHEMA) or "").strip()
    if not raw:
        return None
    if not _SCHEMA_RE.match(raw):
        raise ValueError(f"{ENV_SCHEMA}={raw!r} is not a plain lower-case identifier")
    return raw


def add_missing_columns(engine, table: str, columns: Dict[str, str], schema: Optional[str] = None) -> List[str]:
    """``ALTER TABLE table ADD COLUMN`` for each column not present; returns the ones added.

    Idempotent and dialect-neutral (Postgres on Neon, SQLite in tests).
    """
    from sqlalchemy import inspect, text

    try:
        present = {c["name"] for c in inspect(engine).get_columns(table, schema=schema)}
    except Exception as exc:                          # noqa: BLE001 - table may not exist yet
        logger.debug("add_missing_columns: cannot inspect %s: %s", table, exc)
        return []
    qualified = f'"{schema}".{table}' if schema else table
    added: List[str] = []
    with engine.begin() as conn:
        for name, ddl in columns.items():
            if name in present:
                continue
            conn.execute(text(f"ALTER TABLE {qualified} ADD COLUMN {name} {ddl}"))
            added.append(name)
    if added:
        logger.info("%s: added column(s) %s", table, ", ".join(added))
    return added


def restore_paper_state(cloud) -> Optional[dict]:
    """Rebuild the paper book from the cloud store.

    Returns ``{"cash", "initial_capital", "positions", "closed_positions",
    "snapshots"}`` or ``None`` when the cloud holds no paper history (first
    run).  Cash comes from ``paper_cloud_state``; if missing it falls back
    to the latest daily snapshot's cash.  ``cloud`` is any object with the
    PaperCloudSync read API (``read_state``, ``read_positions``,
    ``read_snapshots``), so tests can pass a fake store.
    """
    if cloud is None:
        return None
    # read_state raises on DB errors: let it propagate (caller must not start
    # a fresh book over an unreadable one).
    state = cloud.read_state() if hasattr(cloud, "read_state") else {}
    positions = _records(cloud.read_positions())
    snapshots = sorted(_records(cloud.read_snapshots()), key=lambda s: str(s.get("date")))
    # Weekly checkpoints too: without them a fresh runner sees an empty table,
    # numbers every Saturday "Week 1" and measures the week over the whole book.
    weekly = sorted(_records(cloud.read_weekly()) if hasattr(cloud, "read_weekly") else [],
                    key=lambda w: int(w.get("week_number") or 0))

    def _is_open(p):
        v = p.get("is_open")
        return v in (True, 1, "1", "true", "True", "t")

    open_pos = [p for p in positions if _is_open(p)]
    closed_pos = [p for p in positions if not _is_open(p)]

    cash = state.get("cash")
    cash = float(cash) if cash not in (None, "") else None
    if cash is None:
        # No persisted book yet.  Rows written by the pre-restore runner (each
        # run started from a fresh ₹1L) are not a consistent book: ignore them.
        if positions or snapshots:
            logger.warning(
                "Cloud restore: no persisted cash state — ignoring %d legacy position rows "
                "and %d snapshots; this run starts a new paper book", len(positions), len(snapshots),
            )
        return None
    initial = state.get("initial_capital")
    initial = float(initial) if initial not in (None, "") else None

    epoch = state.get("epoch")
    if epoch:
        ep = _to_utc(epoch)
        start = _book_start_from_state(state)
        if ep is not None and start is not None:
            before = len(open_pos) + len(closed_pos)
            open_pos = [p for p in open_pos if (_to_utc(p.get("opened_at")) or start) >= start]
            closed_pos = [p for p in closed_pos if (_to_utc(p.get("opened_at")) or start) >= start]
            snapshots = [s for s in snapshots if str(s.get("date")) >= ep.date().isoformat()]
            dropped = before - len(open_pos) - len(closed_pos)
            if dropped:
                logger.warning("Cloud restore: ignored %d position rows from before this book (start %s, epoch %s)",
                               dropped, start.isoformat(), epoch)
    return {
        "cash": cash,
        "initial_capital": initial,
        "positions": open_pos,
        "closed_positions": closed_pos,
        "snapshots": snapshots,
        "weekly": weekly,
        "shift_state": state.get(SHIFT_STATE_KEY) or None,
    }


def get_paper_cloud() -> Optional["PaperCloudSync"]:
    """Return the singleton PaperCloudSync (or None if DB not configured).

    Performs a lightweight health check on the cached singleton to detect
    stale Neon connections (auto-suspend after idle timeout).
    """
    global _cloud
    if _cloud is not None:
        # Health check: verify the underlying engine is still usable
        try:
            with _cloud._session() as session:
                session.execute(text("SELECT 1"))
        except Exception:
            logger.info("Cloud sync connection stale — reinitialising")
            _cloud = None
    if _cloud is not None:
        return _cloud
    try:
        from database.connection import get_db_manager
        mgr = get_db_manager()
        if mgr is None:
            return None
        _cloud = PaperCloudSync(mgr, schema=paper_schema_from_env())
        _cloud.ensure_tables()
        return _cloud
    except Exception as exc:
        logger.debug("Paper cloud sync unavailable: %s", exc)
        return None


class PaperCloudSync:
    """Best-effort sync of paper trading data to Neon PostgreSQL.

    ``schema`` puts the whole book - positions, snapshots, fills, sessions,
    weekly checkpoints and the key/value state - in its own Postgres schema,
    so a second paper book (tracker D1: the candidate beside the deployed
    book) can never touch the first.  Table names are qualified in the SQL
    text itself (raw SQL) or by SQLAlchemy's ``schema_translate_map`` (ORM),
    never through ``search_path``, which Neon's transaction-mode pooler does
    not keep between transactions.
    """

    schema: Optional[str] = None          # class defaults: objects built without __init__ (tests) stay valid
    _factory = None

    def __init__(self, db_manager, schema: Optional[str] = None):
        if schema is not None and not _SCHEMA_RE.match(schema):
            raise ValueError(f"paper book schema {schema!r} is not a plain lower-case identifier")
        self._db = db_manager
        self.schema = schema
        self._factory = None

    def _t(self, table: str) -> str:
        """``table`` qualified with this book's schema, for raw SQL."""
        return f'"{self.schema}".{table}' if self.schema else table

    def _translated_engine(self):
        return self._db.engine.execution_options(schema_translate_map={None: self.schema})

    @contextmanager
    def _session(self):
        """A session whose ORM statements land in this book's schema."""
        if self.schema is None:
            with self._db.get_session() as session:
                yield session
            return
        if self._factory is None:
            from sqlalchemy.orm import sessionmaker
            self._factory = sessionmaker(bind=self._translated_engine(), expire_on_commit=False)
        session = self._factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Table creation ─────────────────────────────────────────

    def ensure_tables(self):
        """Create paper trading tables if they don't exist (idempotent)."""
        self._ensure_schema()
        self._ensure_state_table()
        try:
            from database.models import Base
            engine = self._db.engine if self.schema is None else self._translated_engine()
            Base.metadata.create_all(
                engine,
                tables=[
                    Base.metadata.tables["paper_positions"],
                    Base.metadata.tables["paper_daily_snapshots"],
                    Base.metadata.tables["paper_signal_log"],
                    Base.metadata.tables["paper_weekly_checkpoints"],
                    Base.metadata.tables["paper_fills"],
                    Base.metadata.tables["paper_sessions"],
                ],
            )
            add_missing_columns(self._db.engine, "paper_sessions", SESSION_COLUMNS_ADDED, schema=self.schema)
            logger.info("Paper trading cloud tables ensured%s.", f" in schema {self.schema}" if self.schema else "")
        except Exception as exc:
            logger.warning("Could not create paper cloud tables: %s", exc)

    def _ensure_schema(self) -> None:
        """``CREATE SCHEMA IF NOT EXISTS`` for a second book (Postgres only)."""
        if self.schema is None or self._db.engine.dialect.name != "postgresql":
            return
        try:
            with self._db.engine.begin() as conn:
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'))
        except Exception as exc:
            logger.warning("Could not create schema %s: %s", self.schema, exc)

    # ── Write methods (called by PaperTrader) ──────────────────

    def sync_position(self, pos_data: dict) -> bool:
        """Upsert a paper position row."""
        try:
            from database.models import PaperPositionRecord
            with self._session() as session:
                # Try to find existing by symbol + opened_at
                existing = session.query(PaperPositionRecord).filter_by(
                    symbol=pos_data["symbol"],
                    opened_at=pos_data["opened_at"],
                ).first()
                if existing:
                    for k, v in pos_data.items():
                        if k != "id" and hasattr(existing, k):
                            setattr(existing, k, v)
                else:
                    row = PaperPositionRecord(
                        symbol=pos_data["symbol"],
                        side=pos_data["side"],
                        quantity=pos_data["quantity"],
                        entry_price=pos_data["entry_price"],
                        stop_loss=pos_data["stop_loss"],
                        target_price=pos_data["target_price"],
                        opened_at=pos_data["opened_at"],
                        closed_at=pos_data.get("closed_at", ""),
                        exit_price=pos_data.get("exit_price", 0),
                        exit_reason=pos_data.get("exit_reason", ""),
                        pnl=pos_data.get("pnl", 0),
                        pnl_pct=pos_data.get("pnl_pct", 0),
                        is_open=pos_data.get("is_open", True),
                    )
                    session.add(row)
                session.commit()
            return True
        except Exception as exc:
            logger.warning("Cloud sync position failed: %s", exc)
            return False

    def sync_snapshot(self, snap: dict) -> bool:
        """Upsert a daily snapshot row."""
        try:
            from database.models import PaperDailySnapshotRecord
            with self._session() as session:
                existing = session.query(PaperDailySnapshotRecord).filter_by(
                    date=snap["date"],
                ).first()
                if existing:
                    for k, v in snap.items():
                        if hasattr(existing, k):
                            setattr(existing, k, v)
                else:
                    session.add(PaperDailySnapshotRecord(**snap))
                session.commit()
            return True
        except Exception as exc:
            logger.warning("Cloud sync snapshot failed: %s", exc)
            return False

    def sync_signals(self, date_str: str, signals: List[dict]) -> bool:
        """Batch-insert signal log rows (delete-then-insert for the date)."""
        if not signals:
            return True
        try:
            from database.models import PaperSignalLogRecord
            with self._session() as session:
                session.query(PaperSignalLogRecord).filter_by(date=date_str).delete()
                for sig in signals:
                    session.add(PaperSignalLogRecord(
                        date=date_str,
                        symbol=sig.get("symbol", ""),
                        forecast=sig.get("forecast", 0),
                        combined_forecast=sig.get("combined_forecast", 0),
                        action=sig.get("action", ""),
                        entry_price=sig.get("entry_price", 0),
                        stop_loss=sig.get("stop_loss", 0),
                        target_price=sig.get("target_price", 0),
                        quantity=sig.get("quantity", 0),
                        pipeline_sources=sig.get("pipeline_sources", ""),
                        was_traded=bool(sig.get("was_traded")),
                    ))
                session.commit()
            return True
        except Exception as exc:
            logger.warning("Cloud sync signals failed: %s", exc)
            return False

    def sync_session(self, row: dict) -> bool:
        """Upsert one session's activity record (keyed by session date)."""
        try:
            from database.models import PaperSessionRecord
            with self._session() as session:
                existing = session.query(PaperSessionRecord).filter_by(
                    session_date=row["session_date"]).first()
                if existing:
                    for k, v in row.items():
                        if hasattr(existing, k):
                            setattr(existing, k, v)
                else:
                    session.add(PaperSessionRecord(**{k: v for k, v in row.items()
                                                      if hasattr(PaperSessionRecord, k)}))
                session.commit()
            return True
        except Exception as exc:                          # noqa: BLE001 - never block a run
            logger.warning("Cloud sync session failed: %s", exc)
            return False

    def read_sessions(self, since_epoch: bool = True) -> pd.DataFrame:
        """Session activity of the current book."""
        df = self._read(f"SELECT * FROM {self._t('paper_sessions')} ORDER BY session_date")
        return self._since_epoch(df, "session_date", since_epoch)

    def sync_fills(self, fills: List[dict]) -> bool:
        """Insert execution events, skipping ones already stored.

        Keyed by (order_id, symbol, occurred_at) so a re-run of the same
        session cannot double-count a fill.
        """
        if not fills:
            return True
        try:
            from database.models import PaperFillRecord
            with self._session() as session:
                for f in fills:
                    exists = session.query(PaperFillRecord).filter_by(
                        order_id=str(f.get("order_id") or ""),
                        symbol=f.get("symbol", ""),
                        occurred_at=str(f.get("occurred_at") or ""),
                    ).first()
                    if exists:
                        continue
                    session.add(PaperFillRecord(**{k: v for k, v in f.items()
                                                   if hasattr(PaperFillRecord, k)}))
                session.commit()
            return True
        except Exception as exc:                          # noqa: BLE001 - never block a run
            logger.warning("Cloud sync fills failed: %s", exc)
            return False

    def read_fills(self, since_epoch: bool = True) -> pd.DataFrame:
        """Execution events of the current book (all books with ``since_epoch=False``)."""
        df = self._read(f"SELECT * FROM {self._t('paper_fills')} ORDER BY occurred_at")
        return self._since_epoch(df, "occurred_at", since_epoch)

    def sync_weekly(self, ckpt: dict) -> bool:
        """Upsert a weekly checkpoint row."""
        try:
            from database.models import PaperWeeklyCheckpointRecord
            with self._session() as session:
                existing = session.query(PaperWeeklyCheckpointRecord).filter_by(
                    week_number=ckpt["week_number"],
                ).first()
                if existing:
                    for k, v in ckpt.items():
                        if hasattr(existing, k):
                            setattr(existing, k, v)
                else:
                    session.add(PaperWeeklyCheckpointRecord(**ckpt))
                session.commit()
            return True
        except Exception as exc:
            logger.warning("Cloud sync weekly failed: %s", exc)
            return False

    def sync_stop_loss(self, symbol: str, opened_at: str, new_sl: float) -> bool:
        """Update trailing stop-loss on an open position."""
        try:
            from database.models import PaperPositionRecord
            with self._session() as session:
                pos = session.query(PaperPositionRecord).filter_by(
                    symbol=symbol, opened_at=opened_at, is_open=True,
                ).first()
                if pos:
                    pos.stop_loss = new_sl
                    session.commit()
            return True
        except Exception as exc:
            logger.warning("Cloud sync SL update failed: %s", exc)
            return False

    # ── Key/value state (cash, initial capital) ────────────────

    def _ensure_state_table(self) -> None:
        try:
            with self._session() as session:
                session.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self._t("paper_cloud_state")} (
                        key        VARCHAR(64) PRIMARY KEY,
                        value      TEXT NOT NULL,
                        updated_at VARCHAR(40)
                    )
                """))
                session.commit()
        except Exception as exc:
            logger.warning("Could not create paper_cloud_state: %s", exc)

    def sync_state(self, values: Dict[str, object]) -> bool:
        """Upsert key/value state (e.g. ``{"cash": 98000.0, "initial_capital": 100000}``)."""
        from datetime import datetime, timezone
        try:
            now = datetime.now(timezone.utc).isoformat()
            with self._session() as session:
                has_epoch = session.execute(text(
                    f"SELECT 1 FROM {self._t('paper_cloud_state')} WHERE key = 'epoch'")).fetchone()
                if not has_epoch and "epoch" not in values:
                    # First write of a persistent book: positions before this are legacy
                    values = {**values, "epoch": now}
                for key, value in values.items():
                    session.execute(text(
                        f"DELETE FROM {self._t('paper_cloud_state')} WHERE key = :key"), {"key": key})
                    session.execute(text(
                        f"INSERT INTO {self._t('paper_cloud_state')} (key, value, updated_at) "
                        "VALUES (:key, :value, :ts)"),
                        {"key": key, "value": str(value), "ts": now})
                session.commit()
            return True
        except Exception as exc:
            logger.warning("Cloud sync state failed: %s", exc)
            return False

    def read_state(self) -> Dict[str, str]:
        """Key/value book state.  RAISES on DB errors (unlike ``_read``) so a
        transient outage is never mistaken for "no book yet" — that would
        overwrite the persisted cash with the initial capital."""
        with self._session() as session:
            rows = session.execute(text(f"SELECT key, value FROM {self._t('paper_cloud_state')}")).fetchall()
        return {str(k): str(v) for k, v in rows}

    def read_open_positions(self, since_epoch: bool = True) -> pd.DataFrame:
        """Open paper positions (stops and entry dates included)."""
        df = self._read(
            f"SELECT * FROM {self._t('paper_positions')} WHERE is_open = TRUE ORDER BY opened_at"
        )
        return self._since_epoch(df, "opened_at", since_epoch)

    # ── Book identity ──────────────────────────────────────────
    #
    # Rows are never deleted.  A book is the rows written since its ``epoch``
    # (``paper_cloud_state``); everything before that is the previous book's
    # history and is filtered out of the readers by default.

    def epoch(self) -> Optional[pd.Timestamp]:
        try:
            return _epoch_from_state(self.read_state())
        except Exception as exc:                          # noqa: BLE001 - reading only
            logger.debug("epoch lookup failed: %s", exc)
            return None

    def book_owner(self) -> str:
        """``"nse_engine"`` while the GitHub Actions engine job owns the book, else ``""``."""
        try:
            return str(self.read_state().get("book_owner") or "")
        except Exception as exc:                          # noqa: BLE001 - reading only
            logger.debug("book_owner lookup failed: %s", exc)
            return ""

    def book_writer(self) -> str:
        """Which runner last wrote this book (``github_actions``, ``hf_scheduler``, ...)."""
        try:
            return str(self.read_state().get("book_writer") or "")
        except Exception as exc:                          # noqa: BLE001 - reading only
            logger.debug("book_writer lookup failed: %s", exc)
            return ""

    def start_new_book(self, initial_capital: float, owner: str = "nse_engine",
                       force: bool = False) -> Dict[str, object]:
        """Open a fresh paper book at ``initial_capital`` without deleting history.

        Writes a new ``epoch`` plus cash, initial capital, owner, and clears the
        engine's pending orders and last-session marker.  Older rows stay in the
        tables and stop being part of the current book.  Refuses while the
        current book holds open positions unless ``force`` is set.
        """
        state = self.read_state()
        current = restore_paper_state(self) or {}
        open_now = current.get("positions") or []
        if open_now and not force:
            raise RuntimeError(f"{len(open_now)} open positions in the current book — "
                               "close them first or pass force=True")
        now_ts = pd.Timestamp(datetime.now(timezone.utc))
        now = now_ts.isoformat()
        # First instant this book's rows can carry: fills are stamped at a session open,
        # which can precede ``now`` by a few hours (today's 09:15 IST), but must come
        # after every row the previous books wrote.
        day_open = (now_ts.tz_convert("Asia/Kolkata").normalize() + _NSE_OPEN).tz_convert("UTC")
        book_start = min(now_ts, day_open)
        try:
            old = self.read_positions(since_epoch=False)
            stamps = [t for col in ("opened_at", "closed_at") if old is not None and col in old.columns
                      for t in old[col].map(lambda v: _to_utc(v) if v else None) if t is not None and t <= now_ts]
            if stamps:
                book_start = max(book_start, max(stamps) + pd.Timedelta(microseconds=1))
        except Exception as exc:                          # noqa: BLE001 - keep the conservative start
            logger.warning("book_start: could not read earlier rows (%s); using the epoch", exc)
            book_start = now_ts
        values = {
            "epoch": now,
            "book_start": book_start.isoformat(),
            "cash": float(initial_capital),
            "initial_capital": float(initial_capital),
            "engine_pending_orders": "[]",
            "engine_last_session": "",
            "book_owner": owner,
            "previous_epoch": state.get("epoch") or "",
        }
        if not self.sync_state(values):
            raise RuntimeError("could not write the new book state to Neon")
        logger.info("New paper book: epoch=%s capital=%.0f owner=%s (previous epoch %s, %d open positions forced)",
                    now, initial_capital, owner, values["previous_epoch"] or "none", len(open_now))
        return values

    def _since_epoch(self, df: pd.DataFrame, column: str, enabled: bool) -> pd.DataFrame:
        """Rows of the current book only: ``column`` at or after the epoch."""
        if not enabled or df is None or df.empty or column not in df.columns:
            return df
        try:
            state = self.read_state()
        except Exception as exc:                          # noqa: BLE001 - reading only
            logger.debug("epoch lookup failed: %s", exc)
            return df
        ep = _epoch_from_state(state)
        if ep is None:
            return df
        if column in ("date", "week_start", "session_date"):
            cutoff = ep.date().isoformat()
            keep = df[column].astype(str) >= cutoff
        else:
            start = _book_start_from_state(state)
            keep = df[column].map(lambda v: (_to_utc(v) or start) >= start)
        return df[keep].reset_index(drop=True)

    # ── Read methods (called by Paper Dashboard UI) ────────────

    def read_snapshots(self, since_epoch: bool = True) -> pd.DataFrame:
        """Daily snapshots of the current book (all books with ``since_epoch=False``)."""
        df = self._read(f"SELECT * FROM {self._t('paper_daily_snapshots')} ORDER BY date")
        return self._since_epoch(df, "date", since_epoch)

    def read_signals(self, since_epoch: bool = True) -> pd.DataFrame:
        """Signal log rows of the current book."""
        df = self._read(f"SELECT * FROM {self._t('paper_signal_log')} ORDER BY date DESC, symbol")
        return self._since_epoch(df, "date", since_epoch)

    def read_positions(self, since_epoch: bool = True) -> pd.DataFrame:
        """Open and closed positions of the current book."""
        df = self._read(f"SELECT * FROM {self._t('paper_positions')} ORDER BY opened_at DESC")
        return self._since_epoch(df, "opened_at", since_epoch)

    def read_weekly(self, since_epoch: bool = True) -> pd.DataFrame:
        """Weekly checkpoints of the current book."""
        df = self._read(f"SELECT * FROM {self._t('paper_weekly_checkpoints')} ORDER BY week_number")
        return self._since_epoch(df, "week_start", since_epoch)

    def _read(self, sql: str) -> pd.DataFrame:
        try:
            with self._session() as session:
                result = session.execute(text(sql))
                rows = result.fetchall()
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame(rows, columns=result.keys())
        except Exception as exc:
            logger.warning("Cloud read failed: %s", exc)
            return pd.DataFrame()
