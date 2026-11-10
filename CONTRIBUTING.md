# Contributing

## Setup

Backend (Python >= 3.10):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set your LLM provider + API key
```

Frontend:

```bash
cd web
npm install
```

Run both together:

```bash
./dev.sh
```

## Tests

```bash
pytest                                          # backend, 45 tests
cd web && npx tsc --noEmit && npm run build     # frontend typecheck + build
```

Both run in CI on every push and pull request.

## Architecture

See the [README](README.md#architecture) for the request-flow diagram and
the [Configuration](README.md#configuration) table for what each setting
does. If you're touching Cross-CoVe or retrieval behavior specifically,
[`docs/RESEARCH.md`](docs/RESEARCH.md) documents the ablation methodology
used to validate changes there.

## Pull requests

- Keep PRs scoped to one change — separate refactors from behavior changes.
- Add or update a test for anything you fix or change.
- Fill in the PR template's "why" — the diff shows what changed, the PR
  description should explain the reason.
