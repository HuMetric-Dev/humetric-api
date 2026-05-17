from __future__ import annotations

import os
from typing import cast

from humetric_auth import (
    SESSION_TTL_S,
    AuthError,
    AutoLinked,
    ClaimPending,
    LoginRequest,
    RegisterRequest,
    claim_person,
    login,
    register,
    revoke_session,
)
from humetric_auth import StoreWrapped as AuthStoreWrapped
from humetric_core import Err, Ok, Person, Result, Source, User
from humetric_store import ConstraintViolated, upsert_person
from litestar import Request, Response, get, post
from litestar.datastructures import Cookie

from humetric_api._runtime import unwrap_or_problem
from humetric_api.auth_deps import SESSION_COOKIE
from humetric_api.auth_dtos import (
    ClaimRequestDTO,
    LoginRequestDTO,
    RegisterRequestDTO,
    RegisterResponseDTO,
    UserDTO,
)
from humetric_api.deps import get_state
from humetric_api.dtos import PersonResult
from humetric_api.errors import ApiError, AuthWrapped, ClaimRequestInvalid, StoreWrapped

# --- helpers ---------------------------------------------------------------


def _user_to_dto(user: User) -> UserDTO:
    return UserDTO(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        person_id=user.person_id,
        claim_state="linked" if user.person_id else "pending",
        created_at=user.created_at,
    )


def _person_to_candidate(p: Person, rank: int) -> PersonResult:
    return PersonResult(
        rank=rank,
        person_id=p.id,
        name=p.name,
        headline=p.headline,
        location=p.location,
        follower_count=p.follower_count,
        last_active_days_ago=p.last_active_days_ago,
        source=p.source,
        raw_url=p.raw_url,
        skills=tuple(s.normalized for s in p.skills),
        score=0.0,
        explanation="",
    )


def _session_cookie(raw_token: str) -> Cookie:
    secure = os.environ.get("HUMETRIC_ENV", "dev") != "dev"
    return Cookie(
        key=SESSION_COOKIE,
        value=raw_token,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
        max_age=SESSION_TTL_S,
    )


def _clear_cookie() -> Cookie:
    return Cookie(key=SESSION_COOKIE, value="", path="/", max_age=0)


def _request_meta(request: Request) -> tuple[str, str]:
    ua = request.headers.get("user-agent", "")[:512]
    ip = request.client.host if request.client else ""
    return ua, ip


def _lift_auth[T](r: Result[T, AuthError]) -> Result[T, ApiError]:
    if isinstance(r, Ok):
        return r
    return Err(AuthWrapped(cause=r.error))


# --- routes ----------------------------------------------------------------


@post("/auth/register", sync_to_thread=False, status_code=201)
def register_route(data: RegisterRequestDTO, request: Request) -> Response[RegisterResponseDTO]:
    state = get_state()
    ua, ip = _request_meta(request)
    r = register(
        state.conn,
        RegisterRequest(
            email=data.email,
            password=data.password,
            display_name=data.display_name,
            github_username=data.github_username,
            linkedin_url=data.linkedin_url,
            user_agent=ua,
            ip=ip,
        ),
    )
    outcome = unwrap_or_problem(_lift_auth(r))

    if isinstance(outcome, AutoLinked):
        body = RegisterResponseDTO(
            user=_user_to_dto(outcome.user),
            claim_state="linked",
            candidates=(),
        )
        return Response(
            content=body, status_code=201, cookies=[_session_cookie(outcome.minted.raw_token)]
        )

    pending = cast(ClaimPending, outcome)
    body = RegisterResponseDTO(
        user=_user_to_dto(pending.user),
        claim_state="pending",
        candidates=tuple(_person_to_candidate(p, i + 1) for i, p in enumerate(pending.candidates)),
    )
    return Response(
        content=body, status_code=201, cookies=[_session_cookie(pending.minted.raw_token)]
    )


@post("/auth/login", sync_to_thread=False, status_code=200)
def login_route(data: LoginRequestDTO, request: Request) -> Response[UserDTO]:
    state = get_state()
    ua, ip = _request_meta(request)
    r = login(
        state.conn,
        LoginRequest(email=data.email, password=data.password, user_agent=ua, ip=ip),
    )
    outcome = unwrap_or_problem(_lift_auth(r))
    return Response(
        content=_user_to_dto(outcome.user),
        status_code=200,
        cookies=[_session_cookie(outcome.minted.raw_token)],
    )


@post("/auth/logout", sync_to_thread=False, status_code=204)
def logout_route(request: Request) -> Response[None]:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        revoke_session(get_state().conn, token)  # idempotent; ignore Err
    return Response(content=None, status_code=204, cookies=[_clear_cookie()])


@get("/auth/me", sync_to_thread=False)
def me_route(user: User) -> UserDTO:
    return _user_to_dto(user)


@post("/auth/claim-person", sync_to_thread=False, status_code=200)
def claim_route(data: ClaimRequestDTO, user: User) -> UserDTO:
    # Caller must specify exactly one of: known person_id, or create_new.
    if data.person_id is None and not data.create_new:
        unwrap_or_problem(
            Err(ClaimRequestInvalid(detail="must supply person_id or create_new=true"))
        )
    if data.person_id is not None and data.create_new:
        unwrap_or_problem(
            Err(ClaimRequestInvalid(detail="person_id and create_new are mutually exclusive"))
        )

    state = get_state()
    target_person_id = data.person_id
    if data.create_new:
        if not data.new_person_name:
            unwrap_or_problem(
                Err(ClaimRequestInvalid(detail="create_new requires new_person_name"))
            )
        custom_id = f"p:custom:{user.id.split(':', 1)[1]}"
        person = Person(
            id=custom_id,
            source=cast(Source, "custom"),
            name=data.new_person_name or "",
            headline=data.new_person_headline or "",
        )
        up = upsert_person(state.conn, person)
        if isinstance(up, Err):
            unwrap_or_problem(Err(StoreWrapped(cause=up.error)))
        target_person_id = custom_id

    assert target_person_id is not None  # narrowed by checks above
    claim_r = claim_person(state.conn, user.id, target_person_id)
    if isinstance(claim_r, Err):
        # Translate the common race-condition StoreWrapped(ConstraintViolated)
        # into a 409 so the frontend can re-fetch candidates.
        if isinstance(claim_r.error, AuthStoreWrapped):
            inner = claim_r.error.cause
            if isinstance(inner, ConstraintViolated):
                unwrap_or_problem(Err(AuthWrapped(cause=claim_r.error)))
        unwrap_or_problem(_lift_auth(claim_r))
    return _user_to_dto(cast(Ok, claim_r).value)


__all__ = ["claim_route", "login_route", "logout_route", "me_route", "register_route"]
