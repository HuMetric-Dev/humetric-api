from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import psycopg
from humetric_core import Err, Ok, Result
from humetric_embed import TextEncoder
from humetric_orchestrator import LLMBackend
from humetric_retrieval import (
    DenseBranch,
    SearchEngine,
    TypeBranches,
    build_bm25,
    build_engine,
    open_bm25,
)
from humetric_store import StoreError, load_vector_index, open_db

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
    # open_db opens autocommit=False so the idempotent DDL migration runs in
    # one transaction. The api holds a singleton connection across requests,
    # so we flip to autocommit afterwards — otherwise any read-only request
    # leaves the connection "idle in transaction", holding row/share locks
    # and blocking the next migrating connection (e.g. a `build-index` run)
    # behind a relation lock. open_db's post-DDL register_vector() opens
    # an implicit txn that's never committed, so commit before the flip
    # (psycopg refuses to change autocommit while INTRANS).
    conn.commit()
    conn.autocommit = True

    encoder_loaded = TextEncoder(model=os.environ.get("HUMETRIC_ENCODER", "bge-small")).load()
    if isinstance(encoder_loaded, Err):
        return Err(EmbedWrapped(cause=encoder_loaded.error))
    enc = encoder_loaded.value

    persons_b_r = _build_type_branches(conn, enc, paths.bm25_index, table="persons")
    if isinstance(persons_b_r, Err):
        return persons_b_r
    persons_b = persons_b_r.value

    orgs_bm25_path = paths.bm25_index.parent / f"{paths.bm25_index.name}_orgs"
    orgs_b_r = _build_type_branches(conn, enc, orgs_bm25_path, table="organizations")
    if isinstance(orgs_b_r, Err):
        return orgs_b_r
    orgs_b = orgs_b_r.value

    if persons_b is None and orgs_b is None:
        return Err(
            IndexMissing(
                path="persons.vec_text / organizations.vec_text",
                hint="run `humetric build-index` to populate text vectors for at least one table.",
            )
        )

    engine_r = build_engine(conn, persons=persons_b, organizations=orgs_b)
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


def _build_type_branches(
    conn: psycopg.Connection,
    enc: TextEncoder,
    bm25_path,  # Path; left untyped here to avoid an extra import
    *,
    table: str,
) -> Result[TypeBranches | None, ApiError]:
    """Build a TypeBranches for one entity table, or return None when there are
    no indexed text vectors for that table (so the engine can skip the type)."""
    idx_r = load_vector_index(conn, "text", table=table)
    if isinstance(idx_r, Err):
        return Err(StoreWrapped(cause=idx_r.error))
    text_index = idx_r.value
    if text_index.size == 0:
        return Ok[TypeBranches | None](None)

    dense = DenseBranch(encoder=enc, index=text_index)

    bm25_r = (
        open_bm25(bm25_path) if bm25_path.is_dir() else build_bm25(conn, table=table)  # type: ignore[arg-type]
    )
    if isinstance(bm25_r, Err):
        # An empty corpus (CorpusEmpty) is fine — skip this type.
        return Ok[TypeBranches | None](None)

    return Ok[TypeBranches | None](TypeBranches(bm25=bm25_r.value, text_branch=dense))


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
