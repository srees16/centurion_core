"""
Weekly CSV export of every paper book's record (India tracker JR2).

Each book's record lives only in Neon: its fills, positions, daily snapshots, sessions, signals, weekly
checkpoints and state (the futures, gilt and MCX ledgers among it).  This copies the paper tables of every book's
schema to ``data/nse_engine/book_exports/<book>/<table>.csv.gz``, which the private Kaggle backup carries
(``cloud.backup``'s state piece).  Read-only.  The weekly refresh on this Mac runs it before its backup push
(``runners.refresh_data``); GitHub Actions cannot, because this repository's logs and artifacts are public.

- Books: every ``config/nse_engine_*.json`` (the deployed book in the default schema), the live book, the US
  advisory journal and the ETF order-book snapshots (BD2).  A schema that does not exist yet is listed as such.
- Tables: the seven paper tables, those present.  ``paper_cloud_state`` is copied key by key, except any key
  naming a token, secret, password or API key (the stored Kite login is one), and the registry of users' Zerodha
  accounts (``kite_accounts``, in the live book's state): a user who deletes their account (MU2) must not live on
  in a backup.

    python -m tools.export_books            # needs CENTURION_DATABASE_URL (.env on this Mac)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

OUT = Path("data/nse_engine/book_exports")
TABLES = ("paper_positions", "paper_daily_snapshots", "paper_signal_log", "paper_weekly_checkpoints",
          "paper_fills", "paper_sessions", "paper_cloud_state")
SECRET_KEY = re.compile(r"token|secret|passw|api_key|kite_accounts", re.IGNORECASE)   # accounts.REGISTRY_KEY
#: Schemas that are not config books: the live book's default, the US advisory journal, BD2's snapshots.
EXTRA = {"live": "live", "us_advisory": "us_advisory", "etf_depth": "etf_depth"}


def schemas() -> Dict[str, str]:
    """Book name -> Neon schema."""
    from nse_engine.books import discover_books

    out = {b.name: (b.schema or "public") for b in discover_books()}
    for name, schema in EXTRA.items():
        out.setdefault(name, schema)
    return out


def export(out: Path = OUT) -> Dict:
    from sqlalchemy import inspect, text

    from database.connection import get_db_manager

    engine = get_db_manager().engine
    insp = inspect(engine)
    known = set(insp.get_schema_names())
    manifest: Dict = {"exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "books": {}}
    for book, schema in schemas().items():
        if schema not in known:
            manifest["books"][book] = {"schema": schema, "tables": {}, "note": "no such schema yet"}
            continue
        present = set(insp.get_table_names(schema=schema))
        folder = out / book
        folder.mkdir(parents=True, exist_ok=True)
        counts: Dict[str, int] = {}
        for table in TABLES:
            if table not in present:
                continue
            with engine.connect() as conn:
                df = pd.read_sql(text(f'SELECT * FROM "{schema}".{table}'), conn)
            if table == "paper_cloud_state":
                df = df[~df["key"].astype(str).str.contains(SECRET_KEY)]
            df.to_csv(folder / f"{table}.csv.gz", index=False, compression="gzip")
            counts[table] = int(len(df))
        manifest["books"][book] = {"schema": schema, "tables": counts}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.export_books", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    import os

    if not (os.environ.get("CENTURION_DATABASE_URL") or os.environ.get("DATABASE_URL")):
        print("CENTURION_DATABASE_URL is not set: add the Neon connection string to .env (the same value as the "
              "GitHub secret), then re-run", file=sys.stderr)
        return 2
    print(json.dumps(export(Path(args.out)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
