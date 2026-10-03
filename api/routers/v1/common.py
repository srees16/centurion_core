"""Shared by the /api/v1 route modules (moved from v1_gateway.py, tracker H4)."""

import logging
import math

# The routes still log as the gateway did before the split.
logger = logging.getLogger("api.routers.v1_gateway")

# No Kite login is a client state, not a server fault: 409, not 503. A 5xx is
# reported to Sentry by the FastAPI integration and retried by the frontend's
# API client, so every poll of a page left open after the token expired became
# three requests and three Sentry errors.
KITE_SESSION_INACTIVE = "Kite session not active"


def _sanitize_floats(obj):
    """Replace inf/nan floats with None so JSON serialization doesn't fail."""
    if isinstance(obj, float):
        return None if math.isinf(obj) or math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    return obj
