"""
FastAPI Application Factory for Centurion Capital Trading Platform.

Creates and configures the root FastAPI app with:
    - All module routers (US stocks, Indian stocks, RAG, Crypto)
    - CORS middleware
    - Lifespan for startup/shutdown hooks
    - Exception handlers
    - Auth-gated /docs, /redoc, /openapi.json
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Ensure project root is on sys.path so all internal imports resolve
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api.auth import (
    LOGIN_PAGE_HTML,
    SESSION_COOKIE,
    LoginThrottled,
    authenticate_user_async,
    create_session_token,
    revoke_session_token,
    session_from_request,
)
from auth.shared_session import (
    SHARED_COOKIE_MAX_AGE,
    SHARED_COOKIE_NAME,
    create_shared_token,
)

# ---------------------------------------------------------------------------
# Sentry — error tracking + performance tracing
# ---------------------------------------------------------------------------

# Noise patterns from yfinance that should NOT create Sentry events.
_SENTRY_DROP_PATTERNS = (
    "Failed download",
    "possibly delisted",
    "No data found",
    "Invalid Crumb",
)


def _sentry_before_send(event, hint):
    """Drop noisy yfinance / ticker-not-found events from Sentry; never send query strings (``?token=``)."""
    message = (event.get("logentry") or {}).get("message", "")
    if not message:
        message = event.get("message", "")
    for pattern in _SENTRY_DROP_PATTERNS:
        if pattern in message:
            return None  # drop the event
    (event.get("request") or {}).pop("query_string", None)
    return event


def _init_sentry() -> None:
    """Initialise Sentry SDK if a DSN is configured."""
    dsn = os.getenv("SENTRY_DSN", "")
    if not dsn:
        return
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_sdk.init(
            dsn=dsn,
            environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.2")),
            send_default_pii=False,
            include_local_variables=False,     # frames hold Kite tokens and secrets
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                StarletteIntegration(transaction_style="endpoint"),
                LoggingIntegration(
                    level=logging.INFO,        # capture breadcrumbs from INFO+
                    event_level=logging.ERROR,  # send events for ERROR+
                ),
            ],
            before_send=_sentry_before_send,
        )

        # Suppress yfinance & peewee loggers from creating Sentry events.
        # yfinance logs "1 Failed download" / "possibly delisted" at ERROR
        # level internally — these are expected for Indian tickers and
        # should not pollute Sentry.
        for noisy_logger in ("yfinance", "peewee"):
            logging.getLogger(noisy_logger).setLevel(logging.CRITICAL)

        logging.getLogger(__name__).info("Sentry initialised (env=%s)",
                                         os.getenv("SENTRY_ENVIRONMENT"))
    except ImportError:
        logging.getLogger(__name__).debug("sentry-sdk not installed — skipping")
    except Exception as exc:
        logging.getLogger(__name__).warning("Sentry init failed: %s", exc)

_init_sentry()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auth: every request needs a signed-in session (trackers S1, S2)
# ---------------------------------------------------------------------------

_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
#: Reached without a session: the logins and logout, self-service sign-up, activation and
#: password reset (MU2, rate-limited in api.routers.v1.auth), Kite's login callback, the
#: terms the holder accepts there (a signed ticket from that page, MU1) and the
#: order postback (which carries its own checksum), the health check the uptime
#: monitor pings, and the docs pages (which redirect to their own login).
_PUBLIC = frozenset({("POST", "/api/v1/auth/login"), ("POST", "/auth/login"), ("POST", "/api/v1/auth/logout"),
                     ("POST", "/stream/postback"), ("GET", "/ind-stocks/auth/callback"),
                     ("POST", "/ind-stocks/auth/consent"),
                     *(("POST", f"/api/v1/auth/{p}") for p in ("signup", "activate", "resend-activation",
                                                                "forgot-password", "reset-password")),
                     ("GET", "/"), ("HEAD", "/"), ("GET", "/health"), ("HEAD", "/health"), ("GET", "/favicon.ico"),
                     ("GET", "/auth/login"), ("GET", "/auth/logout"),
                     ("GET", "/docs"), ("GET", "/redoc"), ("GET", "/openapi.json")})
#: Writes that trade or change a broker account: the admin role only.
_ADMIN_WRITE_PREFIXES = ("/api/v1/kite/", "/api/v1/screener/execute", "/api/v1/drivewealth/",
                         "/ind-stocks/orders", "/ind-stocks/auth")
#: Never reached by a signed-up user (role "user", MU2): the operator's own broker accounts
#: (the Kite and DriveWealth sessions: holdings, positions, orders, P&L), the trade monitor's
#: paper-validation and daily-detail data, and the G4 audit that rewrites the deployed
#: parameters.  The user's own Zerodha account is the exception (_USER_OWN_PREFIXES), which
#: its router scopes to them.
_USER_DENIED_PREFIXES = (
    "/api/v1/kite/", "/api/v1/drivewealth/", "/api/v1/screener/execute", "/ind-stocks/auth",
    "/ind-stocks/orders", "/ind-stocks/positions", "/ind-stocks/holdings", "/ind-stocks/pipeline/walk-forward",
    *(f"/api/v1/screener/monitor/{p}" for p in ("paper-dashboard", "daily-snapshots", "sessions", "signal-log",
                                                 "weekly-checkpoints", "daily-detail")))
_USER_OWN_PREFIXES = ("/api/v1/kite/accounts",)
#: Shared state a signed-up user reads but does not change: the operator's price alerts.
_USER_DENIED_WRITES = ("/stream/alerts",)
#: Work that costs minutes of CPU or an LLM call, limited per signed-in user (SEC2): at most
#: HEAVY_LIMIT requests in HEAVY_WINDOW_S seconds, so a stolen session cannot run up the bill.
_HEAVY = frozenset({("GET", "/api/v1/rag/query")} | {("POST", p) for p in (
    "/api/v1/analysis/run", "/api/v1/backtest/run", "/api/v1/verdict/run", "/api/v1/screener/run",
    "/api/v1/fml/run", "/api/v1/tts/run", "/api/v1/aronson/run", "/api/v1/ehlers/run", "/api/v1/vince/run",
    "/api/v1/rl-bot/train", "/api/v1/rl-bot/evaluate", "/api/v1/options/overlay/scan", "/rag/query",
    "/rag/evaluate", "/rag/ingest", "/rag/ingest/directory", "/rag/reingest", "/api/v1/rag/upload",
    "/us-stocks/analysis", "/us-stocks/backtest", "/us-stocks/carver/pipeline", "/us-stocks/news",
    "/us-stocks/sentiment", "/us-stocks/decision", "/crypto/backtest", "/portfolio/backtest", "/r22/backtest",
    "/ind-stocks/pipeline/full", "/ind-stocks/pipeline/walk-forward", "/ind-stocks/pipeline/screen",
    "/ind-stocks/penfold/calibrate")})
HEAVY_LIMIT, HEAVY_WINDOW_S = 60, 600
_heavy_calls: dict = {}


def _heavy_allowed(user: str) -> bool:
    """Record one heavy request for ``user``; False when the window is already full."""
    import time

    now = time.monotonic()
    recent = [t for t in _heavy_calls.get(user, []) if now - t < HEAVY_WINDOW_S]
    if len(recent) >= HEAVY_LIMIT:
        _heavy_calls[user] = recent
        return False
    _heavy_calls[user] = recent + [now]
    return True


#: CORS origins when CENTURION_ALLOWED_ORIGINS is unset: the production frontend
#: and local development (never "*", which with credentials reflects any origin).
_DEFAULT_ORIGINS = ["https://centurion-core-fe.vercel.app", "http://localhost:3000"]


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown hooks)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise heavyweight singletons at startup; tear down on shutdown.
    """
    logger.info("Centurion API starting up")

    # ── Database startup diagnostics ────────────────────────
    db_url_set = bool(os.getenv("CENTURION_DATABASE_URL") or os.getenv("DATABASE_URL"))
    if not db_url_set:
        logger.warning(
            "CENTURION_DATABASE_URL is NOT set — database will be unavailable. "
            "Set it in HF Spaces Settings → Repository secrets."
        )

    try:
        from api.dependencies import get_db_service
        db = get_db_service()
        if db:
            logger.info("Database connection OK (Neon)")
        else:
            logger.warning(
                "Database not available — DB-dependent endpoints will 503. "
                "Ensure CENTURION_DATABASE_URL is set in environment/secrets."
            )
    except Exception as exc:
        logger.warning("Database init skipped: %s", exc)

    logger.info("Centurion API ready")
    yield
    logger.info("Centurion API shutting down ...")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Build and return the FastAPI application."""

    # Disable built-in docs routes — we serve our own auth-gated versions
    app = FastAPI(
        title="Centurion Capital LLC API",
        description=(
            "RESTful API for the Centurion Capital algorithmic trading "
            "platform — US stocks analysis, Indian stocks (Zerodha Kite), "
            "RAG pipeline, and crypto mean-reversion strategies."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # --- Signed-in session for every request (S1, S2); added before CORS so a 401 still carries CORS headers ---
    @app.middleware("http")
    async def require_session(request: Request, call_next):
        if request.method == "OPTIONS" or (request.method, request.url.path) in _PUBLIC:
            return await call_next(request)
        session = session_from_request(request)
        if session is None:
            return JSONResponse(status_code=401, content={"detail": "sign in to do that"})
        path, role = request.url.path, session.get("r")
        own = role == "user" and path.startswith(_USER_OWN_PREFIXES)
        if role == "user" and not own and (path.startswith(_USER_DENIED_PREFIXES) or (
                request.method in _WRITE_METHODS and path.startswith(_USER_DENIED_WRITES))):
            return JSONResponse(status_code=403, content={"detail": "not available to your account"})
        if (request.method in _WRITE_METHODS and path.startswith(_ADMIN_WRITE_PREFIXES)
                and role != "admin" and not own):
            return JSONResponse(status_code=403, content={"detail": "only an admin can trade or change broker accounts"})
        if (request.method, request.url.path) in _HEAVY and not _heavy_allowed(str(session.get("u", ""))):
            return JSONResponse(status_code=429, content={
                "detail": f"Too many heavy requests (at most {HEAVY_LIMIT} in {HEAVY_WINDOW_S // 60} minutes): "
                          "try again shortly"})
        return await call_next(request)

    # --- CORS ---
    # Allowed origins from env (comma-separated), else the production frontend and local dev
    _raw_origins = os.getenv("CENTURION_ALLOWED_ORIGINS", "")
    _cors_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()] if _raw_origins else _DEFAULT_ORIGINS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routers ---
    from api.routers.health import router as health_router
    from api.routers.us_stocks import router as us_stocks_router
    from api.routers.ind_stocks import router as ind_stocks_router
    from api.routers.rag import router as rag_router
    from api.routers.crypto import router as crypto_router
    from api.routers.streaming import router as streaming_router
    from api.routers.pipeline import router as pipeline_router
    from api.routers.v1_gateway import router as v1_gateway_router
    from api.routers.portfolio import router as portfolio_router
    from api.routers.r22 import router as r22_router

    app.include_router(health_router)
    app.include_router(us_stocks_router)
    app.include_router(ind_stocks_router)
    app.include_router(rag_router)
    app.include_router(crypto_router)
    app.include_router(streaming_router)
    app.include_router(pipeline_router)
    app.include_router(v1_gateway_router)
    app.include_router(portfolio_router)
    app.include_router(r22_router)

    # ------------------------------------------------------------------
    # Authentication endpoints
    # ------------------------------------------------------------------

    @app.get("/auth/login", include_in_schema=False)
    async def login_page(request: Request):
        """Serve the login form. If already authenticated, redirect to docs."""
        if session_from_request(request):
            return RedirectResponse(url="/docs", status_code=302)
        return HTMLResponse(LOGIN_PAGE_HTML)

    @app.post("/auth/login", include_in_schema=False)
    async def login(
        username: str = Form(...),
        password: str = Form(...),
    ):
        """Validate credentials, set a session cookie, redirect to docs."""
        try:
            ok, display_name, role = await authenticate_user_async(username, password)
        except LoginThrottled:
            return JSONResponse(status_code=429, content={"success": False,
                                                          "detail": "Too many failed sign-ins: try again later"})
        if not ok:
            return JSONResponse(
                status_code=401,
                content={"success": False, "detail": "Invalid username or password"},
            )
        token = create_session_token(username, role)
        shared_token = create_shared_token(username, role)
        response = RedirectResponse(url="/docs", status_code=302)
        response.set_cookie(
            key=SESSION_COOKIE,
            value=token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=28800,
        )
        # Shared SSO cookie (auth/shared_session.py)
        response.set_cookie(
            key=SHARED_COOKIE_NAME,
            value=shared_token,
            httponly=True,
            secure=True,
            samesite="lax",
            path="/",
            max_age=SHARED_COOKIE_MAX_AGE,
        )
        logger.info("API docs login: user=%s role=%s", username, role)
        return response

    @app.get("/auth/logout", include_in_schema=False)
    async def logout(request: Request):
        """Revoke the docs session, clear its cookies and redirect to the login page."""
        token = request.cookies.get(SESSION_COOKIE)
        if token:
            revoke_session_token(token)
        response = RedirectResponse(url="/auth/login", status_code=302)
        response.delete_cookie(SESSION_COOKIE)
        response.delete_cookie(SHARED_COOKIE_NAME, path="/")
        return response

    # ------------------------------------------------------------------
    # Auth-gated OpenAPI / Swagger / ReDoc routes
    # ------------------------------------------------------------------

    @app.get("/openapi.json", include_in_schema=False)
    async def openapi_json(request: Request):
        if not session_from_request(request):
            return RedirectResponse(url="/auth/login", status_code=302)
        return JSONResponse(app.openapi())

    @app.get("/docs", include_in_schema=False)
    async def docs(request: Request):
        if not session_from_request(request):
            return RedirectResponse(url="/auth/login", status_code=302)
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=app.title + " — Swagger UI",
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc(request: Request):
        if not session_from_request(request):
            return RedirectResponse(url="/auth/login", status_code=302)
        return get_redoc_html(
            openapi_url="/openapi.json",
            title=app.title + " — ReDoc",
        )


    @app.exception_handler(StarletteHTTPException)
    async def http_errors(request: Request, exc: StarletteHTTPException):
        """An HTTPException's own message, except a 500's: that is an exception's text (paths,
        hosts, connection details), so it goes to the log and Sentry, not to the caller."""
        if exc.status_code == 500:
            logger.error("500 on %s %s: %s", request.method, request.url.path, exc.detail)
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})
        return await http_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(                               # the cause is in the log and Sentry, not the response
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": "Internal server error",
            },
        )

    _FAVICON_PATH = Path(__file__).resolve().parent.parent / "ui" / "assets" / "centurion_logo.png"

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        if _FAVICON_PATH.is_file():
            return FileResponse(_FAVICON_PATH, media_type="image/png")
        return JSONResponse(status_code=204, content=None)


    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "message": "Centurion Capital LLC API",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
        }

    return app


# Allow `uvicorn api.main:app` to work directly
app = create_app()
