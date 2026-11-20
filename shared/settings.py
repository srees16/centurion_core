"""Settings read from the environment."""

from __future__ import annotations

import os

#: The GitHub repository, for a run where ``GITHUB_REPOSITORY`` is not set (on this Mac).
REPOSITORY = "srees16/centurion_core"


def database_url() -> str:
    """The Neon connection string: ``CENTURION_DATABASE_URL``, else ``DATABASE_URL``, else empty.  An empty
    ``CENTURION_DATABASE_URL`` falls through to ``DATABASE_URL``."""
    return os.environ.get("CENTURION_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
