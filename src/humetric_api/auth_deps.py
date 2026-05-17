from __future__ import annotations

from humetric_auth import resolve_session
from humetric_core import Err, Ok, User
from humetric_store import get_user_by_id
from litestar import Request
from litestar.exceptions import HTTPException

from humetric_api.deps import get_state

SESSION_COOKIE = "humetric_session"


async def provide_current_user(request: Request) -> User | None:
    """Best-effort current-user resolver. Returns None when no cookie is
    present or the session is invalid/expired. The handler decides whether
    that's a 401 (via `user: User`) or a no-op (via `current_user: User | None`)."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    state = get_state()
    sess_r = resolve_session(state.conn, token)
    if isinstance(sess_r, Err):
        return None
    user_r = get_user_by_id(state.conn, sess_r.value.user_id)
    if isinstance(user_r, Ok):
        return user_r.value
    return None


async def require_user(current_user: User | None) -> User:
    if current_user is None:
        raise HTTPException(
            status_code=401,
            detail="not authenticated",
            extra={"error": "unauthenticated", "detail": "session cookie missing or expired"},
        )
    return current_user
