from __future__ import annotations

from dataclasses import dataclass

from humetric_auth import AuthError
from humetric_core import HumetricError
from humetric_embed import EmbedError
from humetric_orchestrator import OrchestratorError
from humetric_retrieval import RetrievalError
from humetric_store import StoreError


@dataclass(frozen=True, slots=True)
class OrchestratorWrapped(HumetricError):
    cause: OrchestratorError


@dataclass(frozen=True, slots=True)
class RetrievalWrapped(HumetricError):
    cause: RetrievalError


@dataclass(frozen=True, slots=True)
class StoreWrapped(HumetricError):
    cause: StoreError


@dataclass(frozen=True, slots=True)
class EmbedWrapped(HumetricError):
    cause: EmbedError


@dataclass(frozen=True, slots=True)
class AuthWrapped(HumetricError):
    cause: AuthError


@dataclass(frozen=True, slots=True)
class BackendWrapped(HumetricError):
    """Raised at startup when the configured LLM backend cannot be loaded."""

    detail: str


@dataclass(frozen=True, slots=True)
class IndexMissing(HumetricError):
    path: str
    hint: str


@dataclass(frozen=True, slots=True)
class NotAuthenticated(HumetricError):
    """Route required a session cookie; none was presented or it resolved
    to an expired/missing session."""


@dataclass(frozen=True, slots=True)
class ClaimRequestInvalid(HumetricError):
    detail: str


type ApiError = (
    OrchestratorWrapped
    | RetrievalWrapped
    | StoreWrapped
    | EmbedWrapped
    | AuthWrapped
    | BackendWrapped
    | IndexMissing
    | NotAuthenticated
    | ClaimRequestInvalid
)
