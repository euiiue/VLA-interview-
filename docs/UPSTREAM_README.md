<p align="center">
  <img src="web/public/logo.png" width="120" alt="docs-to-quiz logo" />
</p>

<h1 align="center">docs-to-quiz</h1>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="Next.js 14" src="https://img.shields.io/badge/Next.js-14-000000?logo=next.js&logoColor=white" />
  <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-green" />
  <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen" />
</p>

> Turn your documents into quiz questions — every question backed by
> evidence from your files, verified by multiple LLMs. Not hallucinated.

📖 **[Full documentation](https://docs2quiz.github.io)** — installation, usage guides, configuration reference (EN/ID)

Upload PDFs, DOCX, or notes → get grounded multiple-choice questions with
cited evidence spans. Runs fully offline with Ollama, or with Gemini, Groq,
Cerebras, and any OpenAI-compatible provider.

<p align="center">
  <img src="docs/screenshots/demo.gif" width="720" alt="docs-to-quiz demo: picking a knowledge base, entering a keyword, streaming generation, and the finished question with cited evidence" />
</p>

## Features

- **Evidence-grounded** — every question stem cites a span from your
  uploaded documents; ungrounded questions are rejected before they ship.
- **Cross-CoVe verification** — 3 independently configured reviewer LLMs
  vote on whether the answer actually follows from the evidence, majority
  wins. See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the methodology.
- **BM25 + optional dense rerank** — fast lexical retrieval by default,
  switch on `hybrid_rerank` for semantic reranking when it's worth the
  latency.
- **Multi-provider** — Gemini, Groq, Cerebras, any OpenAI-compatible
  endpoint, or fully local via Ollama.
- **Multi knowledge base** — upload and manage several independent
  document sets, generate against whichever one you need.
- **Live streaming** — questions stream over WebSocket as they're
  generated and verified, not after a full batch completes.
- **Export** — DOCX and PDF export of finished question sets.

## Architecture

```
 upload (PDF/DOCX/TXT/MD)
        │
        ▼
 chunk + embed  ──────────────►  knowledge base (per kb_id)
        │                              │
        │                              ▼
        │                     BM25 index (+ optional
        │                      dense rerank / RRF fusion)
        │                              │
        ▼                              ▼
      keyword       ───────►  retrieve evidence
                                        │
                                        ▼
                                 LLM generates MCQ
                                        │
                                        ▼
                         verification gate: schema, evidence
                         span match, topic relevance
                                        │
                                        ▼
                    Cross-CoVe: 3 reviewer LLMs vote
                                        │
                              ┌─────────┴─────────┐
                              ▼                    ▼
                         OK → stream to        rejected → retry
                         client over WS         or fall back
```

## Quickstart

Backend (Python >= 3.10):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # set your LLM provider + API key
uvicorn app.main:app   # API on :8000
```

Frontend:

```bash
cd web
npm install
npm run dev            # UI on :3000
```

Or run both together from the repo root:

```bash
./dev.sh
```

Or with Docker:

```bash
cp .env.example .env   # set your LLM provider + API key
docker compose up --build -d
```

## Configuration

All settings live in `.env` (see [`.env.example`](.env.example) for the
full annotated list — 70+ tunables covering retrieval, LLM providers,
Cross-CoVe reviewers, and knowledge-base ingestion). The ones you'll
actually touch:

| Variable                                | Purpose                                                                 |
| --------------------------------------- | ----------------------------------------------------------------------- |
| `LLM_PROVIDER`                          | `gemini_sdk` \| `openai_compatible` \| `groq` \| `cerebras` \| `ollama` |
| `LLM_API_KEY`, `LLM_MODEL`              | Credentials + model for the main generator                              |
| `RETRIEVAL_MODE`                        | `bm25_only` (default) or `hybrid_rerank` for dense reranking            |
| `KNOWLEDGE_DIR`                         | Where uploaded knowledge bases are stored                               |
| `EMBED_MODEL`                           | Embedding model for chunk/dense retrieval                               |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`           | Document chunking at ingest time                                        |
| `COVE_ENABLED`, `COVE_VOTE_COUNT`       | Turn Cross-CoVe verification on/off, vote count                         |
| `REVIEWER_1/2/3_PROVIDER/API_KEY/MODEL` | The 3 independent reviewers Cross-CoVe votes across                     |
| `OK_ONLY_MODE`                          | Only ever emit verified (`OK`) questions to the client                  |

## API

Interactive docs at `http://localhost:8000/docs` once the backend is
running. Key routes:

| Route                                                                                  | Purpose                                        |
| -------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `POST /knowledge`, `GET /knowledge`, `PATCH /knowledge/{id}`, `DELETE /knowledge/{id}` | Manage knowledge bases                         |
| `POST /knowledge/{id}/files`, `DELETE /knowledge/{id}/files/{filename}`                | Add/remove documents                           |
| `POST /generate`                                                                       | Synchronous batch generation                   |
| `WS /ws/generate`                                                                      | Streaming generation (what the UI uses)        |
| `GET /eval/rouge`, `/eval/nli`, `/eval/semantic`                                       | Offline quality metrics over the question bank |

## Design decisions

This is built for **self-hosting by a single user** — there is no
authentication. Do not expose it directly to the public internet; put it
behind your own auth/reverse proxy if you need remote access.

## Tests

```bash
pip install pytest
pytest   # 45 tests, backend only
cd web && npx tsc --noEmit && npm run build   # frontend typecheck + build
```

## Research

The Cross-CoVe verification step was validated with an ablation study —
methodology, scripts, and how to read the metrics are in
[`docs/RESEARCH.md`](docs/RESEARCH.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT
