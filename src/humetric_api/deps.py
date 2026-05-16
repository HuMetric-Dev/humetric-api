from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import psycopg
from humetric_core import Err, Ok, Result
from humetric_embed import TextEncoder
from humetric_orchestrator import LLMBackend
from humetric_retrieval import DenseBranch, SearchEngine, build_bm25, build_engine, open_bm25
from humetric_store import StoreError, VectorIndex, load_vector_index, open_db

from humetric_api._runtime import DataPaths, resolve_paths, select_backend
from humetric_api.errors import (
    ApiError,
    BackendWrapped,
    EmbedWrapped,
    IndexMissing,
    RetrievalWrapped,
    StoreWrapped,
)


@dataclass(slots=True)
class AppState:
    """Process-wide singletons created at lifespan start, reused per request."""

    paths: DataPaths
    conn: psycopg.Connection
    encoder: TextEncoder
    engine: SearchEngine
    backend: LLMBackend


_state: AppState | None = None


def get_state() -> AppState:
    if _state is None:
        msg = "AppState not initialized; call init_state() in app lifespan."
        raise RuntimeError(msg)
    return _state


def _read_dsn() -> Result[str, ApiError]:
    dsn = os.environ.get("HUMETRIC_DB_URL")
    if not dsn:
        return Err(
            BackendWrapped(
                detail="HUMETRIC_DB_URL is unset; export a Postgres DSN to start the API."
            )
        )
    return Ok(dsn)


def init_state() -> Result[AppState, ApiError]:
    """Build the SearchEngine + encoder + DB connection once at startup."""
    global _state

    data_dir = os.environ.get("HUMETRIC_DATA_DIR", "./data")
    paths = resolve_paths(data_dir)

    dsn_r = _read_dsn()
    if isinstance(dsn_r, Err):
        return dsn_r

    conn_r = open_pg(dsn_r.value)
    if isinstance(conn_r, Err):
        return Err(StoreWrapped(cause=conn_r.error))
    conn = conn_r.value

    encoder_loaded = TextEncoder(model=os.environ.get("HUMETRIC_ENCODER", "bge-small")).load()
    if isinstance(encoder_loaded, Err):
        return Err(EmbedWrapped(cause=encoder_loaded.error))
    enc = encoder_loaded.value

    idx_r = load_vector_index(conn, "text")
    if isinstance(idx_r, Err):
        return Err(StoreWrapped(cause=idx_r.error))
    text_index: VectorIndex = idx_r.value
    if text_index.size == 0:
        return Err(
            IndexMissing(
                path="persons.vec_text",
                hint="run `humetric build-index` to populate text vectors.",
            )
        )
    dense = DenseBranch(encoder=enc, index=text_index)

    bm25_r = open_bm25(paths.bm25_index) if paths.bm25_index.is_dir() else build_bm25(conn)
    if isinstance(bm25_r, Err):
        return Err(RetrievalWrapped(cause=bm25_r.error))

    engine_r = build_engine(conn, bm25=bm25_r.value, text_branch=dense)
    if isinstance(engine_r, Err):
        return Err(RetrievalWrapped(cause=engine_r.error))
    engine = engine_r.value

    try:
        backend = select_backend(
            name=os.environ.get("HUMETRIC_BACKEND", "fake"),
            base_url=os.environ.get("HUMETRIC_BASE_URL"),
            model=os.environ.get("HUMETRIC_LLM_MODEL"),
        )
    except ValueError as e:
        return Err(BackendWrapped(detail=str(e)))

    _state = AppState(
        paths=paths,
        conn=conn,
        encoder=enc,
        engine=engine,
        backend=backend,
    )
    return Ok(_state)


def open_pg(dsn: str) -> Result[psycopg.Connection, StoreError]:
    """Open and migrate the PG store. Thin re-export so callers (incl. tests)
    can swap in a custom DSN without going through the env var path."""
    return open_db(dsn)


def shutdown_state() -> None:
    global _state
    if _state is not None:
        with contextlib.suppress(psycopg.Error):
            _state.conn.close()
        _state = None
