from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import psycopg
import pytest
from humetric_core import Ok, Person, Skill
from humetric_orchestrator import FakeBackend
from humetric_retrieval import SearchEngine
from humetric_retrieval.bm25 import build_bm25
from humetric_store import upsert_person
from litestar import Litestar
from litestar.testing import TestClient

from humetric_api import deps
from humetric_api._runtime import DataPaths
from humetric_api.routes import history, query


class _FakeEncoder:
    """Duck-types `TextEncoder` for the routes we touch (encode_one only)."""

    dim = 4

    def encode_one(self, _text: str):
        return Ok(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))


@pytest.fixture
def fixture_state(tmp_path: Path, pg_conn: psycopg.Connection) -> Iterator[deps.AppState]:
    upsert_person(
        pg_conn,
        Person(
            id="gh:ada",
            source="github",
            name="Ada Lovelace",
            headline="Distributed systems engineer",
            about="ships rust raft implementations",
            skills=(Skill(name="rust", normalized="rust"),),
            raw_url="https://github.com/ada",
            follower_count=2400,
        ),
    ).unwrap()
    upsert_person(
        pg_conn,
        Person(
            id="gh:linus",
            source="github",
            name="Linus Torvalds",
            headline="Kernel maintainer",
            skills=(Skill(name="c", normalized="c"),),
            raw_url="https://github.com/torvalds",
            follower_count=184000,
        ),
    ).unwrap()
    for i, (name, headline, skill) in enumerate(
        [
            ("Grace Hopper", "compiler pioneer", "cobol"),
            ("Edsger Dijkstra", "structured programming", "algol"),
            ("Donald Knuth", "TAOCP author", "metafont"),
            ("Margaret Hamilton", "Apollo flight software", "assembly"),
            ("Barbara Liskov", "abstract data types", "clu"),
            ("Bjarne Stroustrup", "C++ inventor", "cpp"),
            ("Guido van Rossum", "scripting language", "abc"),
        ]
    ):
        upsert_person(
            pg_conn,
            Person(
                id=f"gh:noise{i}",
                source="github",
                name=name,
                headline=headline,
                skills=(Skill(name=skill, normalized=skill),),
                raw_url=f"https://github.com/noise{i}",
            ),
        ).unwrap()

    bm25_r = build_bm25(pg_conn)
    assert isinstance(bm25_r, Ok), bm25_r
    engine = SearchEngine(conn=pg_conn, bm25=bm25_r.value)

    paths = DataPaths(root=tmp_path)
    paths.history.parent.mkdir(parents=True, exist_ok=True)

    deps._state = deps.AppState(  # type: ignore[arg-type]
        paths=paths,
        conn=pg_conn,
        encoder=_FakeEncoder(),  # type: ignore[arg-type]
        engine=engine,
        backend=FakeBackend(),
    )
    yield deps._state
    deps._state = None


def _client() -> TestClient:
    return TestClient(Litestar(route_handlers=[query, history]))


def test_query_returns_ranked_results(fixture_state: deps.AppState) -> None:
    with _client() as c:
        r = c.post("/api/query", json={"text": "rust engineer"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["parsed"]["free_text"] == "rust engineer"
        assert "rust" in body["parsed"]["must_skills"]
        assert isinstance(body["results"], list)
        assert len(body["results"]) >= 1
        first = body["results"][0]
        assert first["rank"] == 1
        assert first["person_id"] == "gh:ada"
        assert first["name"] == "Ada Lovelace"
        assert "rust" in first["skills"]
        assert first["explanation"], "explanation should be populated by FakeBackend"


def test_history_reflects_recent_query(fixture_state: deps.AppState) -> None:
    with _client() as c:
        c.post("/api/query", json={"text": "rust engineer"}).raise_for_status()
        c.post("/api/query", json={"text": "kafka python"}).raise_for_status()
        r = c.get("/api/history")
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 2
        assert items[0]["free_text"] == "kafka python"
        assert items[1]["free_text"] == "rust engineer"


def test_history_empty_when_no_queries(fixture_state: deps.AppState) -> None:
    with _client() as c:
        r = c.get("/api/history")
        assert r.status_code == 200
        assert r.json() == {"items": []}
