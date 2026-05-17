from __future__ import annotations

import msgspec


class QueryRequest(msgspec.Struct, frozen=True):
    text: str
    # Override which entity tables to retrieve. None lets the LLM parser
    # decide via ParsedQuery.target_entity_types.
    entity_types: tuple[str, ...] | None = None


class ParsedQueryDTO(msgspec.Struct, frozen=True):
    free_text: str
    must_skills: tuple[str, ...] = ()
    nice_skills: tuple[str, ...] = ()
    location: str | None = None
    min_followers: int | None = None
    min_years_experience: int | None = None


class PersonResult(msgspec.Struct, frozen=True):
    rank: int
    person_id: str
    name: str
    headline: str
    location: str
    follower_count: int
    last_active_days_ago: int | None
    source: str
    raw_url: str
    skills: tuple[str, ...]
    score: float
    explanation: str


class OrgResult(msgspec.Struct, frozen=True):
    rank: int
    org_id: str
    name: str
    headline: str
    location: str
    source: str
    raw_url: str
    org_kind: str
    score: float
    explanation: str


class QueryResponse(msgspec.Struct, frozen=True):
    ts: float
    parsed: ParsedQueryDTO
    # `results` is kept as the person block for frontend back-compat.
    results: tuple[PersonResult, ...] = ()
    organizations: tuple[OrgResult, ...] = ()


class HistoryItem(msgspec.Struct, frozen=True):
    ts: float
    free_text: str


class HistoryResponse(msgspec.Struct, frozen=True):
    items: tuple[HistoryItem, ...]


class ErrorBody(msgspec.Struct, frozen=True):
    error: str
    detail: str
