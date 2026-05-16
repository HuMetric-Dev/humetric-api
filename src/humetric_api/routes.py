from __future__ import annotations

import json
import time
from pathlib import Path

from humetric_core import Err, Ok, ParsedQuery, Result
from humetric_orchestrator import append_history, parse_query, write_feed
from humetric_retrieval import Candidate
from humetric_store import get_person
from litestar import get, post

from humetric_api._runtime import unwrap_or_problem
from humetric_api.deps import get_state
from humetric_api.dtos import (
    HistoryItem,
    HistoryResponse,
    ParsedQueryDTO,
    PersonResult,
    QueryRequest,
    QueryResponse,
)
from humetric_api.errors import (
    EmbedWrapped,
    OrchestratorWrapped,
    RetrievalWrapped,
    StoreWrapped,
)


@post("/api/query", sync_to_thread=False, status_code=200)
def query(data: QueryRequest) -> QueryResponse:
    state = get_state()
    text = data.text.strip()

    parsed_r = parse_query(state.backend, text)
    parsed: ParsedQuery = unwrap_or_problem(_lift_orch(parsed_r))

    enc_r = state.encoder.encode_one(parsed.free_text)
    text_vec = unwrap_or_problem(_lift_embed(enc_r))

    hist_r = append_history(state.paths.history, parsed, text_vec)
    unwrap_or_problem(_lift_orch(hist_r))

    search_r = state.engine.search(parsed, k=10)
    cands: list[Candidate] = unwrap_or_problem(_lift_retr(search_r))

    pairs: list[tuple[str, str]] = []
    results: list[PersonResult] = []
    for i, c in enumerate(cands):
        p_r = get_person(state.conn, c.person_id)
        p = unwrap_or_problem(_lift_store(p_r))
        pairs.append((p.id, p.text_blob()))
        results.append(
            PersonResult(
                rank=i + 1,
                person_id=p.id,
                name=p.name,
                headline=p.headline,
                location=p.location,
                follower_count=p.follower_count,
                last_active_days_ago=p.last_active_days_ago,
                source=p.source,
                raw_url=p.raw_url,
                skills=tuple(s.normalized for s in p.skills),
                score=float(c.score),
                explanation="",
            )
        )

    feed_r = write_feed(state.backend, parsed.free_text, pairs)
    explanations = unwrap_or_problem(_lift_orch(feed_r))
    expl_by_pid = {e.person_id: e.text for e in explanations}
    filled = tuple(
        PersonResult(
            rank=r.rank,
            person_id=r.person_id,
            name=r.name,
            headline=r.headline,
            location=r.location,
            follower_count=r.follower_count,
            last_active_days_ago=r.last_active_days_ago,
            source=r.source,
            raw_url=r.raw_url,
            skills=r.skills,
            score=r.score,
            explanation=expl_by_pid.get(r.person_id, ""),
        )
        for r in results
    )

    return QueryResponse(
        ts=time.time(),
        parsed=_parsed_to_dto(parsed),
        results=filled,
    )


@get("/api/history", sync_to_thread=False)
def history(limit: int = 20) -> HistoryResponse:
    state = get_state()
    items = _read_recent_history(state.paths.history, limit=max(1, min(limit, 100)))
    return HistoryResponse(items=tuple(items))


# --- helpers ---

# Each `_lift_*` converts a `Result[T, ComponentError]` into `Result[T, ApiError]`
# by wrapping the Err side in the appropriate ApiError variant. The Ok side
# flows through unchanged. The route handler then calls `unwrap_or_problem`,
# which is the single place this layer turns Errs into HTTP responses.


def _lift_orch[T](r):  # type: ignore[no-untyped-def]
    return r if isinstance(r, Ok) else Err(OrchestratorWrapped(cause=r.error))


def _lift_retr[T](r):  # type: ignore[no-untyped-def]
    return r if isinstance(r, Ok) else Err(RetrievalWrapped(cause=r.error))


def _lift_store[T](r):  # type: ignore[no-untyped-def]
    return r if isinstance(r, Ok) else Err(StoreWrapped(cause=r.error))


def _lift_embed[T](r):  # type: ignore[no-untyped-def]
    return r if isinstance(r, Ok) else Err(EmbedWrapped(cause=r.error))


def _parsed_to_dto(p: ParsedQuery) -> ParsedQueryDTO:
    return ParsedQueryDTO(
        free_text=p.free_text,
        must_skills=tuple(p.must_skills),
        nice_skills=tuple(p.nice_skills),
        location=p.location,
        min_followers=p.min_followers,
        min_years_experience=p.min_years_experience,
    )


def _read_recent_history(path: Path, *, limit: int) -> list[HistoryItem]:
    """Lightweight history reader — only the fields the UI needs.

    Avoids loading the embedding (a ~384-float blob per row). Skips malformed
    rows silently; this endpoint is best-effort for the sidebar.
    """
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[HistoryItem] = []
    for line in reversed(lines[-(limit * 4) :]):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
            ts = float(rec["ts"])
            free_text = str(rec["parsed"]["free_text"])
        except (ValueError, KeyError, TypeError):
            continue
        out.append(HistoryItem(ts=ts, free_text=free_text))
        if len(out) >= limit:
            break
    return out


__all__ = ["history", "query"]
# Re-export so import-time linting doesn't drop `Result` (used in type hints above).
_ = Result
