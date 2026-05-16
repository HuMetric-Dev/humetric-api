from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from humetric_core import Err, Result
from humetric_orchestrator import (
    AnthropicBackend,
    BackendCallFailed,
    BackendMisconfigured,
    BackendUnavailable,
    FakeBackend,
    FeedWriteFailed,
    HistoryReadFailed,
    HistoryWriteFailed,
    LLMBackend,
    OpenAIBackend,
    ParseRejected,
)
from humetric_retrieval import (
    CorpusEmpty,
    FilterFailed,
)
from humetric_retrieval import (
    EmbedWrapped as RetrievalEmbedWrapped,
)
from humetric_retrieval import (
    StoreWrapped as RetrievalStoreWrapped,
)
from humetric_store import NotFound
from litestar.exceptions import HTTPException

from humetric_api.dtos import ErrorBody
from humetric_api.errors import (
    ApiError,
    BackendWrapped,
    EmbedWrapped,
    IndexMissing,
    OrchestratorWrapped,
    RetrievalWrapped,
    StoreWrapped,
)


@dataclass(frozen=True, slots=True)
class DataPaths:
    """On-disk artifacts the API reads (BM25 + history). The canonical store
    is Postgres; its DSN comes from HUMETRIC_DB_URL, not from here.
    """

    root: Path

    @property
    def bm25_index(self) -> Path:
        return self.root / "bm25.idx"

    @property
    def history(self) -> Path:
        return self.root / "history" / "queries.jsonl"


def resolve_paths(data_dir: str | Path) -> DataPaths:
    root = Path(data_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return DataPaths(root=root)


def select_backend(name: str, base_url: str | None, model: str | None) -> LLMBackend:
    if name == "fake":
        return FakeBackend()
    if name == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if model:
            return AnthropicBackend(api_key=api_key, model=model)
        return AnthropicBackend(api_key=api_key)
    if name in ("openai", "vllm"):
        api_key = os.environ.get("OPENAI_API_KEY", "dummy")
        if model:
            return OpenAIBackend(base_url=base_url, api_key=api_key, model=model)
        return OpenAIBackend(base_url=base_url, api_key=api_key)
    msg = f"unknown backend: {name}"
    raise ValueError(msg)


def unwrap_or_problem[T](r: Result[T, ApiError]) -> T:
    """Lift Result[T, ApiError] into either the value or a Litestar HTTPException.

    Mirrors `humetric_cli._runtime.unwrap_or_die` — the HTTP layer is the
    second (and only other) place in the codebase where Errs stop being values.
    """
    if isinstance(r, Err):
        raise _to_http(r.error)
    return r.value


def _to_http(err: ApiError) -> HTTPException:
    status, code = _classify(err)
    detail = _describe(err)
    body = ErrorBody(error=code, detail=detail)
    return HTTPException(
        status_code=status,
        detail=detail,
        extra={"error": body.error, "detail": body.detail},
    )


def _classify(err: ApiError) -> tuple[int, str]:
    """Map an ApiError variant to (HTTP status, machine-readable code)."""
    if isinstance(err, OrchestratorWrapped):
        cause = err.cause
        if isinstance(cause, ParseRejected):
            return 400, "parse_failed"
        if isinstance(cause, BackendMisconfigured | BackendUnavailable):
            return 502, "llm_unavailable"
        if isinstance(cause, BackendCallFailed | FeedWriteFailed):
            return 502, "llm_failed"
        if isinstance(cause, HistoryReadFailed | HistoryWriteFailed):
            return 503, "history_failed"
        return 502, "orchestrator_failed"
    if isinstance(err, RetrievalWrapped):
        cause = err.cause
        if isinstance(cause, CorpusEmpty):
            return 503, "corpus_empty"
        if isinstance(cause, FilterFailed):
            return 500, "filter_failed"
        if isinstance(cause, RetrievalStoreWrapped | RetrievalEmbedWrapped):
            return 503, "backend_unavailable"
        return 500, "retrieval_failed"
    if isinstance(err, StoreWrapped):
        cause = err.cause
        if isinstance(cause, NotFound):
            return 404, "not_found"
        return 503, "store_failed"
    if isinstance(err, EmbedWrapped):
        return 503, "embed_failed"
    if isinstance(err, BackendWrapped):
        return 502, "backend_init_failed"
    if isinstance(err, IndexMissing):
        return 503, "index_missing"
    return 500, "unknown"


def _describe(err: ApiError) -> str:
    if isinstance(err, IndexMissing):
        return f"{err.path}: {err.hint}"
    if isinstance(err, BackendWrapped):
        return err.detail
    return repr(err)
