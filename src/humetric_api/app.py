from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from humetric_core import Err
from litestar import Litestar
from litestar.config.cors import CORSConfig
from litestar.exceptions import HTTPException

from humetric_api.deps import init_state, shutdown_state
from humetric_api.routes import history, query


@asynccontextmanager
async def _lifespan(_app: Litestar) -> AsyncIterator[None]:
    r = init_state()
    if isinstance(r, Err):
        # Fail fast: a misconfigured pipeline should crash the worker, not
        # silently start an API that 500s on every request.
        msg = f"humetric-api failed to initialize: {r.error!r}"
        raise RuntimeError(msg)
    try:
        yield
    finally:
        shutdown_state()


def build_app() -> Litestar:
    cors_origins = os.environ.get(
        "HUMETRIC_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    cors = CORSConfig(
        allow_origins=[o.strip() for o in cors_origins if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )
    return Litestar(
        route_handlers=[query, history],
        cors_config=cors,
        lifespan=[_lifespan],
        exception_handlers={HTTPException: _http_exception_handler},
    )


def _http_exception_handler(_request, exc):  # type: ignore[no-untyped-def]
    """Wire the `extra` dict from `unwrap_or_problem` into the JSON body."""
    from litestar.response import Response

    body: dict[str, object] = {}
    extra = getattr(exc, "extra", None)
    if isinstance(extra, dict):
        body.update(extra)
    if "error" not in body:
        body["error"] = "http_error"
    if "detail" not in body:
        body["detail"] = str(getattr(exc, "detail", "")) or "error"
    return Response(content=body, status_code=getattr(exc, "status_code", 500))


# Module-level instance for `litestar --app humetric_api.app:app run ...`
app = build_app()
