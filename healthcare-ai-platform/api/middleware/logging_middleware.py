# stdlib
import logging
import os
import time
import uuid
from pathlib import Path

# third-party
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# ─── Set up dual logging (console + file) ─────────────────────────────────────
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# Console handler
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)

# File handler
_file_handler = logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8")
_file_handler.setFormatter(_formatter)

# Root logger configuration
_root_logger = logging.getLogger()
if not _root_logger.handlers:
    _root_logger.addHandler(_console_handler)
    _root_logger.addHandler(_file_handler)

_request_logger = logging.getLogger("api.requests")
_request_logger.setLevel(logging.INFO)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that logs every HTTP request and response.

    Logs include: method, path, request_id (UUID4), timestamp,
    response status code, and request duration in milliseconds.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Intercept the request, log it, call the next handler, and log the response.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP response from the downstream handler.
        """
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # Attach request_id to request state so routes can reference it
        request.state.request_id = request_id

        _request_logger.info(
            "→ REQUEST  | id=%s | method=%s | path=%s | client=%s",
            request_id,
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _request_logger.error(
                "✗ ERROR    | id=%s | path=%s | error=%s | duration=%.2fms",
                request_id,
                request.url.path,
                str(exc),
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _request_logger.info(
            "← RESPONSE | id=%s | status=%d | duration=%.2fms",
            request_id,
            response.status_code,
            elapsed_ms,
        )

        # Inject request ID into response headers for tracing
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed_ms:.2f}ms"
        return response
