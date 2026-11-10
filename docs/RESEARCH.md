# Cross-CoVe ablation research

This app's core claim — that generated questions are actually grounded in
the source documents — rests on **Cross-CoVe**: a Chain-of-Verification
step where 3 independently configured reviewer LLMs each vote on whether a
generated answer is derivable from the retrieved evidence. A question only
ships if a majority of reviewers agree.

The scripts in this directory answer a narrower question: does that voting
step actually catch bad questions, and at what cost? They were written as
part of an ablation study and are kept in the repo (translated to English,
otherwise unchanged) because the methodology is reusable by anyone tuning
`REVIEWER_*` / `COVE_*` settings for their own provider mix.

## What "ablation" means here

An ablation study removes one component of a system and measures the
difference. Here, the component is Cross-CoVe verification. Each script
below runs the same generation pipeline with the verification step forced
on or off (or deliberately sabotaged) and compares the outputs.

## Scripts

### `app/scripts/analyze_cove_ablation.py`

Generates the same set of questions twice against a real knowledge base —
once with Cross-CoVe enabled, once with it disabled — and reports how many
questions Cross-CoVe rejected that would otherwise have reached the
question bank.

```bash
python -m app.scripts.analyze_cove_ablation --kb-id <your-kb-id>
```

`TEST_CASES` in the script defines the topic/competency pairs to generate
for; edit them to match content actually present in your knowledge base
before running.

### `app/scripts/cove_detection_benchmark.py`

Feeds the reviewer pipeline questions with **intentionally wrong answers**
(swapped options, contradicted-by-evidence claims) and measures what
fraction Cross-CoVe correctly flags. This is the detection-rate number:
if Cross-CoVe can't catch answers that are wrong on purpose, it won't catch
the subtler mistakes an LLM makes on its own.

```bash
python -m app.scripts.cove_detection_benchmark
```

### `app/eval/ablation.py`

A second, retrieval-focused ablation: compares `bm25_only` vs
`hybrid_rerank` retrieval modes on JSON-valid rate, evidence-support rate,
reviewer-agreement rate, and runtime. Useful for deciding whether the dense
reranker (`RETRIEVAL_MODE=hybrid_rerank`) is worth its latency cost for a
given knowledge base.

```bash
python -m app.eval.ablation --input path/to/test-set.jsonl --output results.csv
```

### `app/scripts/evaluate_quality_metrics.py`

Computes offline quality metrics (ROUGE, NLI entailment, semantic
similarity) over the question bank, independent of Cross-CoVe. Same metrics
backing the `/eval/rouge`, `/eval/nli`, `/eval/semantic` API routes, runnable
in bulk from the CLI.

## Reading the metrics

- **Filtering rate** — share of generated questions Cross-CoVe rejected.
  Higher isn't automatically better: it also reflects how strict your
  reviewer models are and how much retrieval noise they're seeing.
- **Detection rate** — share of *known-wrong* answers Cross-CoVe caught.
  This is the metric that actually validates the verification step; treat
  filtering rate as a side effect, not the goal.
- **Evidence-support rate** — share of questions whose evidence spans were
  found verbatim (or near-verbatim) in the source documents, independent of
  the reviewer LLMs' opinions.
- **ROUGE / NLI / semantic** — standard offline text-quality proxies
  comparing generated stem+answer against the retrieved evidence; see
  `app/eval/` for the exact implementations.

None of these scripts are required to run the app — they're for anyone
tuning retrieval or reviewer configuration and wanting to measure the
effect before shipping a change.
