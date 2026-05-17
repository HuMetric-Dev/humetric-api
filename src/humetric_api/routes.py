from __future__ import annotations

import time

from humetric_core import EntityType, Err, Ok, ParsedQuery, Result, User
from humetric_orchestrator import append_history, parse_query, read_history, write_feed
from humetric_retrieval import Candidate
from humetric_store import get_organization, get_person
from litestar import get, post

from humetric_api._runtime import unwrap_or_problem
from humetric_api.deps import get_state
from humetric_api.dtos import (
    HistoryItem,
    HistoryResponse,
    OrgResult,
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
def query(data: QueryRequest, user: User) -> QueryResponse:
    state = get_state()
    text = data.text.strip()

    parsed_r = parse_query(state.backend, text)
    parsed: ParsedQuery = unwrap_or_problem(_lift_orch(parsed_r))

    enc_r = state.encoder.encode_one(parsed.free_text)
    text_vec = unwrap_or_problem(_lift_embed(enc_r))

    hist_r = append_history(state.conn, user.id, parsed, text_vec)
    unwrap_or_problem(_lift_orch(hist_r))

    et_override = _coerce_entity_types(data.entity_types)
    search_r = state.engine.search(parsed, k=10, entity_types=et_override)
    cands: list[Candidate] = unwrap_or_problem(_lift_retr(search_r))

    triples: list[tuple[str, EntityType, str]] = []
    person_results: list[PersonResult] = []
    org_results: list[OrgResult] = []
    person_rank = 0
    org_rank = 0
    for c in cands:
        if c.entity_type == "organization":
            o_r = get_organization(state.conn, c.entity_id)
            o = unwrap_or_problem(_lift_store(o_r))
            org_rank += 1
            triples.append((o.id, "organization", o.text_blob()))
            org_results.append(
                OrgResult(
                    rank=org_rank,
                    org_id=o.id,
                    name=o.name,
                    headline=o.headline,
                    location=o.location,
                    source=o.source,
                    raw_url=o.raw_url,
                    org_kind=o.org_kind,
                    score=float(c.score),
                    explanation="",
                )
            )
        else:
            p_r = get_person(state.conn, c.entity_id)
            p = unwrap_or_problem(_lift_store(p_r))
            person_rank += 1
            triples.append((p.id, "person", p.text_blob()))
            person_results.append(
                PersonResult(
                    rank=person_rank,
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

    feed_r = write_feed(state.backend, parsed.free_text, triples)
    explanations = unwrap_or_problem(_lift_orch(feed_r))
    expl_by_eid = {e.entity_id: e.text for e in explanations}
    persons_filled = tuple(
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
            explanation=expl_by_eid.get(r.person_id, ""),
        )
        for r in person_results
    )
    orgs_filled = tuple(
        OrgResult(
            rank=r.rank,
            org_id=r.org_id,
            name=r.name,
            headline=r.headline,
            location=r.location,
            source=r.source,
            raw_url=r.raw_url,
            org_kind=r.org_kind,
            score=r.score,
            explanation=expl_by_eid.get(r.org_id, ""),
        )
        for r in org_results
    )

    return QueryResponse(
        ts=time.time(),
        parsed=_parsed_to_dto(parsed),
        results=persons_filled,
        organizations=orgs_filled,
    )


@get("/api/history", sync_to_thread=False)
def history(user: User, limit: int = 20) -> HistoryResponse:
    state = get_state()
    capped = max(1, min(limit, 100))
    r = read_history(state.conn, user.id, capped)
    entries = unwrap_or_problem(_lift_orch(r))
    items = tuple(HistoryItem(ts=e.ts, free_text=e.parsed.free_text) for e in entries)
    return HistoryResponse(items=items)


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


def _coerce_entity_types(raw: tuple[str, ...] | None) -> tuple[EntityType, ...] | None:
    """Translate client-provided entity_types into the EntityType literal type,
    silently dropping unknown values. None means "let parse_query decide"."""
    if raw is None:
        return None
    out: list[EntityType] = []
    for v in raw:
        s = v.strip().lower()
        if s in ("person", "people"):
            out.append("person")
        elif s in ("organization", "org", "company", "companies"):
            out.append("organization")
    return tuple(out) if out else None


def _parsed_to_dto(p: ParsedQuery) -> ParsedQueryDTO:
    return ParsedQueryDTO(
        free_text=p.free_text,
        must_skills=tuple(p.must_skills),
        nice_skills=tuple(p.nice_skills),
        location=p.location,
        min_followers=p.min_followers,
        min_years_experience=p.min_years_experience,
    )


__all__ = ["history", "query"]
# Re-export so import-time linting doesn't drop `Result` (used in type hints above).
_ = Result
