from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import psycopg
import pytest
from humetric_core import Ok, Person
from humetric_orchestrator import FakeBackend
from humetric_retrieval import SearchEngine, TypeBranches
from humetric_retrieval.bm25 import build_bm25
from humetric_store import upsert_person
from litestar import Litestar
from litestar.di import Provide
from litestar.testing import TestClient

from humetric_api import deps
from humetric_api._runtime import DataPaths
from humetric_api.auth_deps import SESSION_COOKIE, provide_current_user, require_user
from humetric_api.auth_routes import (
    claim_route,
    login_route,
    logout_route,
    me_route,
    register_route,
)


class _FakeEncoder:
    dim = 1024

    def encode_one(self, _text: str):
        v = np.zeros(1024, dtype=np.float32)
        v[0] = 1.0
        return Ok(v)


@pytest.fixture
def fixture_state(tmp_path: Path, pg_conn: psycopg.Connection) -> Iterator[deps.AppState]:
    # The /api/query route isn't exercised here, but init still needs a
    # populated engine/bm25 because deps.AppState requires non-None fields.
    upsert_person(pg_conn, Person(id="gh:seed", source="github", name="Seed Person")).unwrap()
    bm25_r = build_bm25(pg_conn, table="persons")
    assert isinstance(bm25_r, Ok), bm25_r
    engine = SearchEngine(conn=pg_conn, persons=TypeBranches(bm25=bm25_r.value))

    deps._state = deps.AppState(  # type: ignore[arg-type]
        paths=DataPaths(root=tmp_path),
        conn=pg_conn,
        encoder=_FakeEncoder(),  # type: ignore[arg-type]
        engine=engine,
        backend=FakeBackend(),
    )
    yield deps._state
    deps._state = None


def _client() -> TestClient:
    return TestClient(
        Litestar(
            route_handlers=[
                register_route,
                login_route,
                logout_route,
                me_route,
                claim_route,
            ],
            dependencies={
                "current_user": Provide(provide_current_user),
                "user": Provide(require_user),
            },
        )
    )


def test_register_returns_201_with_cookie_and_pending_claim(
    fixture_state: deps.AppState,
) -> None:
    with _client() as c:
        r = c.post(
            "/auth/register",
            json={
                "email": "alice@example.com",
                "password": "hunter2hunter2",
                "display_name": "Alice",
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["claim_state"] == "pending"
        assert body["candidates"] == []
        assert body["user"]["email"] == "alice@example.com"
        assert SESSION_COOKIE in r.cookies


def test_register_then_login_then_me(fixture_state: deps.AppState) -> None:
    with _client() as c:
        c.post(
            "/auth/register",
            json={
                "email": "bob@example.com",
                "password": "hunter2hunter2",
                "display_name": "Bob",
            },
        ).raise_for_status()
        # Drop the registration cookie and log in fresh.
        c.cookies.clear()
        r = c.post(
            "/auth/login",
            json={"email": "bob@example.com", "password": "hunter2hunter2"},
        )
        assert r.status_code == 200, r.text
        assert SESSION_COOKIE in r.cookies

        me = c.get("/auth/me")
        assert me.status_code == 200
        assert me.json()["email"] == "bob@example.com"


def test_me_without_cookie_is_401(fixture_state: deps.AppState) -> None:
    with _client() as c:
        r = c.get("/auth/me")
        assert r.status_code == 401


def test_logout_clears_session(fixture_state: deps.AppState) -> None:
    with _client() as c:
        c.post(
            "/auth/register",
            json={
                "email": "carol@example.com",
                "password": "hunter2hunter2",
                "display_name": "Carol",
            },
        ).raise_for_status()
        r = c.post("/auth/logout")
        assert r.status_code == 204
        # subsequent /me should 401 even with the (now-cleared) cookie
        me = c.get("/auth/me")
        assert me.status_code == 401


def test_claim_person(fixture_state: deps.AppState) -> None:
    upsert_person(
        fixture_state.conn,
        Person(id="p:gh:to-claim", source="github", name="Claimable"),
    ).unwrap()
    with _client() as c:
        c.post(
            "/auth/register",
            json={
                "email": "dave@example.com",
                "password": "hunter2hunter2",
                "display_name": "Dave",
            },
        ).raise_for_status()
        r = c.post("/auth/claim-person", json={"person_id": "p:gh:to-claim"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["person_id"] == "p:gh:to-claim"
        assert body["claim_state"] == "linked"


def test_claim_request_invalid_when_both_specified(fixture_state: deps.AppState) -> None:
    with _client() as c:
        c.post(
            "/auth/register",
            json={
                "email": "eve@example.com",
                "password": "hunter2hunter2",
                "display_name": "Eve",
            },
        ).raise_for_status()
        r = c.post(
            "/auth/claim-person",
            json={"person_id": "p:gh:x", "create_new": True, "new_person_name": "E"},
        )
        assert r.status_code == 400


def test_duplicate_registration_returns_409(fixture_state: deps.AppState) -> None:
    with _client() as c:
        c.post(
            "/auth/register",
            json={
                "email": "frank@example.com",
                "password": "hunter2hunter2",
                "display_name": "F",
            },
        ).raise_for_status()
        c.cookies.clear()
        r = c.post(
            "/auth/register",
            json={
                "email": "frank@example.com",
                "password": "hunter2hunter2",
                "display_name": "F",
            },
        )
        assert r.status_code == 409
