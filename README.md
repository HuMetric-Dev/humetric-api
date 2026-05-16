# humetric-api

Litestar HTTP layer that wraps the Humetric query pipeline as JSON endpoints,
so a browser (the `humetric-web` SvelteKit SPA) can drive it.

The package is a transport adapter only — it owns no business logic. Every
endpoint composes existing functions from `humetric-orchestrator`,
`humetric-retrieval`, and `humetric-store`, and lifts their `Result[T, E]`
returns into HTTP responses.

## Endpoints

```
POST /api/query          { text } → { parsed, results[], ts }
GET  /api/history?limit  → { items[] }
```

## Dev

```bash
uv sync --extra dev
uv run litestar --app humetric_api.app:app run --reload --port 8000
```

The pipeline reads from the same `./data/` directory the CLI uses
(`HUMETRIC_DATA_DIR` env var to override). Build the CLI's data first:

```bash
cd ../humetric-cli
uv run humetric ingest github --seeds kennethreitz --users 500
uv run humetric build-index
```

Then point the API at it:

```bash
HUMETRIC_DATA_DIR=../humetric-cli/data \
  uv run litestar --app humetric_api.app:app run --port 8000
```

## Conventions

This package follows the project-wide rules:

- Every fallible function returns `Result[T, E]`.
- Errors are wrapped at the package boundary (`ApiError` enum in `errors.py`).
- The HTTP route handler is the only place that converts `Err` → response
  status, via `_runtime.unwrap_or_problem()`.
