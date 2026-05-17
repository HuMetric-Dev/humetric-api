from __future__ import annotations

from typing import Literal

import msgspec

from humetric_api.dtos import PersonResult


class RegisterRequestDTO(msgspec.Struct, frozen=True):
    email: str
    password: str
    display_name: str
    github_username: str | None = None
    linkedin_url: str | None = None


class LoginRequestDTO(msgspec.Struct, frozen=True):
    email: str
    password: str


class ClaimRequestDTO(msgspec.Struct, frozen=True):
    """User picks a candidate Person to claim. Frontend sends `person_id` for
    a known candidate, or `create_new=true` with `new_person` fields if none
    of the candidates fit. Exactly one path must be specified — the route
    rejects requests with both or neither."""

    person_id: str | None = None
    create_new: bool = False
    new_person_name: str | None = None
    new_person_headline: str | None = None


class UserDTO(msgspec.Struct, frozen=True):
    id: str
    email: str
    display_name: str
    person_id: str | None
    claim_state: Literal["linked", "pending"]
    created_at: float


class RegisterResponseDTO(msgspec.Struct, frozen=True):
    """When `claim_state == "linked"`, `candidates` will be empty and the
    user is ready to query immediately. When `claim_state == "pending"`,
    the frontend should prompt the user with `candidates` and POST the
    chosen one back via /auth/claim-person."""

    user: UserDTO
    claim_state: Literal["linked", "pending"]
    candidates: tuple[PersonResult, ...]
