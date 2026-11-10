# VLA Practice Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for task-by-task implementation. Independent content and exercise assets may be prepared in parallel under dispatching-parallel-agents.

**Goal:** Deliver a local browser application for VLA MCQ, cloze, source review and isolated PyTorch exercises.
**Architecture:** Extend the cloned docs-to-quiz repository with a lightweight `practice` FastAPI service, reusing upstream parsers, chunking, BM25 and LLM client. Replace its landing page with a React practice UI, served as a static export by the same loopback service. SQLite owns durable data; Docker is the only code executor.
**Tech Stack:** Python, FastAPI, SQLite, React/Next, Monaco, CPU PyTorch, Docker.
**Spec:** docs/spec.md (approved in chat).

## Global Constraints
- Preserve 84 source QAs, source numbers, diagrams and citations; do not fabricate unknown project facts.
- 20 code exercises use deterministic fixtures and include reference implementations and negative checks.
- No API key required for included practice. No direct execution fallback when Docker is unavailable.
- Bind 127.0.0.1. No publishing, shared-branch push, or robot connection.
- Preserve upstream MIT license and pin upstream commit in provenance.

## Review Focus
- Partial/invalid import must not mutate the bank; duplicate imports must be visible.
- Shuffled options and edited question versions must not corrupt historical results.
- Invalid/negative cloze responses must not pass by substring matching.
- User code timeout/output overflow must terminate and remove its container.
- Reload must recover drafts/results; unavailable AI/container must remain distinguishable from incorrect answers.

### Task 1 Data and grading
Files practice/models.py, store.py, grading.py, importers.py; practice_tests/test_bank.py.
- [ ] Write failing tests for stable option IDs, exact normalized aliases, numeric units/tolerance, invalid imports and versioned attempt history.
- [ ] Run red, implement validation and SQLite transactions, run green.
- [ ] Import 84 source QAs; include independently curated objective questions and source-backed cloze without inventing biography.

### Task 2 Isolated coding
Files practice/runner.py, harness.py, exercises/catalog.py, Dockerfile.runner; practice_tests/test_runner.py.
- [ ] Write failing tests for unavailable engine, resource flags, valid reference, incorrect candidate, infinite loop, output overflow and no network.
- [ ] Implement container-only runner with unique temporary directory, read-only mounts, limits, timeout and cleanup.
- [ ] Verify all 20 references and representative incorrect implementations using real CPU PyTorch.

### Task 3 Import and generation API
Files practice/main.py, generation.py; practice_tests/test_api.py.
- [ ] Write failing tests for JSON/Markdown preview and confirmation, duplicate detection, export round trip, origin/host restrictions, draft and attempt persistence.
- [ ] Reuse upstream ingestion and LLM generation modules; add evidence/schema checks and mandatory review before publication.
- [ ] Implement source, filtering, grading, favorite, code, backup and environment APIs. Run green.

### Task 4 Browser UI
Files web/app/page.tsx, globals.css, layout.tsx, web/app/lib/practice.ts.
- [ ] Build Chinese responsive dashboard, practice, code, import/generate, history and source view.
- [ ] Include Monaco editing, formula/Markdown rendering, image zoom, stable option shuffling and visible error/loading states.
- [ ] Typecheck/build; browser test correct/incorrect choice, cloze, persistence, import, code execution and mobile layout.

### Task 5 Delivery
Files README.md, start.sh, requirements-practice.lock, THIRD_PARTY.md, test report.
- [ ] Run full relevant backend and upstream tests, build, reference tests and browser QA.
- [ ] Review implementation against spec, fix important findings, rerun affected tests.
- [ ] Commit local branch; package source and report local launch URL plus any verification limits.
