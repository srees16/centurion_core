"""Authentication helpers shared by the API and the trading services.

The Streamlit login (``authenticator.py``) was removed with the rest of the
Streamlit UI; the API authenticates in ``api/auth.py``. What remains here is
the cross-process shared-session token used by both.
"""

from .shared_session import (  # noqa: F401
    SHARED_COOKIE_MAX_AGE,
    SHARED_COOKIE_NAME,
    create_shared_token,
    verify_shared_token,
)

__all__ = [
    "SHARED_COOKIE_MAX_AGE",
    "SHARED_COOKIE_NAME",
    "create_shared_token",
    "verify_shared_token",
]
