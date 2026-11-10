"""Presentation metadata for the /settings dashboard.

The field list is derived from `Settings.model_fields` rather than hardcoded,
so a field added to config.py shows up on the dashboard automatically (in the
"Other" group until it gets an entry in FIELD_META).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.core.config import Settings, env_overridden, settings

# Fields that exist in Settings but are read by nothing. They are documented in
# .env.example, which makes them look functional -- keeping them off the
# dashboard is the point. Verified by sweeping app/ (incl. app/scripts/),
# tests/ and web/ for each name.
EXCLUDED_FIELDS = {
    "environment",  # zero references anywhere
    "LLM_JSON_MODE",  # zero references; llm_client.py has no json-mode path
    "COVE_VOTE_COUNT",  # zero references; gate.py:99 derives quorum from reviewer count
    "CHUNK_SIZE",  # shadowed by the hardcoded constant at chunker.py:11
    "CHUNK_OVERLAP",  # shadowed by the hardcoded constant at chunker.py:12
}

GROUP_ORDER = [
    "General",
    "Retrieval",
    "Knowledge Base",
    "LLM Provider",
    "Generation",
    "Evidence",
    "Verification",
    "Reviewer 1",
    "Reviewer 2",
    "Reviewer 3",
    "Question Bank",
    "Evaluation",
    "Other",
]

_PROVIDERS = ("gemini_sdk", "openai_compatible", "groq", "cerebras", "ollama")
_REVIEWER_PROVIDERS = ("",) + _PROVIDERS


@dataclass(frozen=True)
class FieldMeta:
    group: str
    label: str
    help: str = ""
    kind: str = ""  # blank => inferred from the annotation
    choices: Tuple[str, ...] = ()
    restart_required: bool = False
    caution: str = ""


FIELD_META: Dict[str, FieldMeta] = {
    # General
    "project_name": FieldMeta(
        "General",
        "Project name",
        "Shown as the FastAPI title. Baked in at import, so a restart is required.",
        restart_required=True,
    ),
    "CORS_ALLOW_ORIGINS": FieldMeta(
        "General",
        "CORS allowed origins",
        "Comma-separated. Baked into the middleware at import, so a restart is required.",
        restart_required=True,
    ),
    # Retrieval
    "RETRIEVAL_MODE": FieldMeta(
        "Retrieval",
        "Retrieval mode",
        "bm25_only skips the dense rerank stage entirely.",
        kind="enum",
        choices=("bm25_only", "hybrid_rerank"),
    ),
    "BM25_CANDIDATES_TOP_N": FieldMeta(
        "Retrieval", "BM25 candidates", "How many BM25 hits feed the rerank stage."
    ),
    "RERANK_TOP_N": FieldMeta(
        "Retrieval", "Rerank top N", "Candidates kept after the dense rerank."
    ),
    "EVIDENCE_TOP_K": FieldMeta(
        "Retrieval", "Evidence top K", "Documents sliced into the prompt as evidence."
    ),
    "EVIDENCE_TOP_K_EXPANDED": FieldMeta(
        "Retrieval",
        "Evidence top K (expanded)",
        "Widened slice used on retry when evidence was judged insufficient.",
    ),
    "FUSION_ENABLED": FieldMeta(
        "Retrieval", "Enable RRF fusion", "Fuse BM25 and dense rankings via reciprocal rank fusion."
    ),
    "RRF_K": FieldMeta("Retrieval", "RRF k", "Rank constant in the RRF formula."),
    # Knowledge Base
    "KNOWLEDGE_DIR": FieldMeta(
        "Knowledge Base",
        "Knowledge directory",
        "Where ingested knowledge bases are stored.",
        caution="Repointing this hides every knowledge base stored under the old path.",
    ),
    "EMBED_MODEL": FieldMeta(
        "Knowledge Base",
        "Embedding model",
        "Used for dense rerank. Changing it reloads the model and requires re-ingesting "
        "knowledge bases for their stored embeddings to match.",
    ),
    # LLM Provider
    "LLM_PROVIDER": FieldMeta(
        "LLM Provider", "Provider", kind="enum", choices=_PROVIDERS
    ),
    "LLM_API_KEY": FieldMeta("LLM Provider", "API key"),
    "LLM_API_BASE": FieldMeta("LLM Provider", "API base URL"),
    "LLM_MODEL": FieldMeta("LLM Provider", "Model"),
    "GROQ_API_KEY": FieldMeta("LLM Provider", "Groq API key", "Used when the provider is groq."),
    "CEREBRAS_API_KEY": FieldMeta(
        "LLM Provider", "Cerebras API key", "Used when the provider is cerebras."
    ),
    "LLM_FALLBACKS": FieldMeta(
        "LLM Provider",
        "Fallback chain",
        "JSON array of alternate targets tried in order when the primary fails.",
        kind="json",
    ),
    "CEREBRAS_TIMEOUT_SEC": FieldMeta("LLM Provider", "Cerebras timeout (s)"),
    "CEREBRAS_MAX_RETRIES": FieldMeta("LLM Provider", "Cerebras max retries"),
    # Generation
    "LLM_TEMPERATURE": FieldMeta("Generation", "Temperature"),
    "LLM_TOP_P": FieldMeta("Generation", "Top P"),
    "LLM_MAX_COMPLETION_TOKENS": FieldMeta("Generation", "Max completion tokens"),
    "LLM_TIMEOUT_SEC": FieldMeta("Generation", "Request timeout (s)"),
    "LLM_MAX_ATTEMPTS": FieldMeta(
        "Generation", "Max attempts", "Retries of the retrieve → prompt → verify loop per question."
    ),
    "LLM_CANDIDATES": FieldMeta(
        "Generation", "Candidates per attempt", "Questions generated per attempt before picking one."
    ),
    "LLM_DEDUP_ATTEMPTS": FieldMeta(
        "Generation", "Dedup attempts", "Extra regenerations allowed when a stem duplicates an earlier one."
    ),
    "LLM_REASONING_EFFORT": FieldMeta(
        "Generation", "Reasoning effort", "Passed through verbatim; leave blank to omit."
    ),
    "LLM_STOP": FieldMeta("Generation", "Stop sequence"),
    "LLM_STREAM": FieldMeta("Generation", "Stream responses"),
    "OK_ONLY_MODE": FieldMeta(
        "Generation",
        "OK-only mode",
        "Emit only questions that pass verification, retrying until the budgets below run out.",
    ),
    "OK_ONLY_MAX_ROUNDS": FieldMeta("Generation", "OK-only max rounds"),
    "OK_ONLY_MAX_SECONDS": FieldMeta("Generation", "OK-only max seconds"),
    # Evidence
    "EVIDENCE_FIRST": FieldMeta(
        "Evidence", "Evidence-first trimming", "Trim each document to its leading sentences before prompting."
    ),
    "EVIDENCE_FIRST_MAX_SENTENCES": FieldMeta("Evidence", "Max sentences per span"),
    "EVIDENCE_FIRST_MAX_CHARS": FieldMeta("Evidence", "Max chars per span"),
    "EVIDENCE_MAX_CHARS_PER_DOC": FieldMeta("Evidence", "Max chars per document"),
    "EVIDENCE_MAX_TOTAL_CHARS": FieldMeta("Evidence", "Max chars total"),
    # Verification
    "REVIEWER_ENABLED": FieldMeta(
        "Verification", "Enable reviewer", "Single-reviewer path. Ignored when Cross-CoVe is on."
    ),
    "COVE_ENABLED": FieldMeta(
        "Verification", "Enable Cross-CoVe", "Run the three reviewers below in parallel and take a majority vote."
    ),
    "REVIEWER_PROVIDER": FieldMeta(
        "Verification", "Provider", "Blank inherits the main LLM provider.",
        kind="enum", choices=_REVIEWER_PROVIDERS,
    ),
    "REVIEWER_API_KEY": FieldMeta("Verification", "API key"),
    "REVIEWER_API_BASE": FieldMeta("Verification", "API base URL"),
    "REVIEWER_MODEL": FieldMeta("Verification", "Model"),
    "REVIEWER_TIMEOUT_SEC": FieldMeta("Verification", "Timeout (s)"),
    "REVIEWER_FALLBACKS": FieldMeta("Verification", "Fallback chain", kind="json"),
    "REVIEWER_MAX_COMPLETION_TOKENS": FieldMeta("Verification", "Max completion tokens"),
    # Question Bank
    "QUESTION_BANK_PATH": FieldMeta("Question Bank", "Bank path", "JSONL file backing the question bank."),
    "QUESTION_BANK_FALLBACK": FieldMeta(
        "Question Bank", "Use as fallback", "Serve a banked question when generation fails."
    ),
    "QUESTION_BANK_WRITE_OK": FieldMeta(
        "Question Bank", "Write OK questions", "Append newly generated OK questions to the bank."
    ),
    "QUESTION_BANK_ALLOW_ANY_TOPIC": FieldMeta(
        "Question Bank", "Allow any topic", "Let fallback pull banked questions from unrelated topics."
    ),
    # Evaluation
    "NLI_MODEL": FieldMeta("Evaluation", "NLI model", "Entailment scoring for /eval/nli."),
    "SEMANTIC_MODEL": FieldMeta("Evaluation", "Semantic model", "Embedding similarity for /eval/semantic."),
}

# Reviewer 1/2/3 follow one shape
for _n in (1, 2, 3):
    _group = f"Reviewer {_n}"
    FIELD_META[f"REVIEWER_{_n}_PROVIDER"] = FieldMeta(
        _group, "Provider", kind="enum", choices=_REVIEWER_PROVIDERS
    )
    FIELD_META[f"REVIEWER_{_n}_API_KEY"] = FieldMeta(_group, "API key")
    FIELD_META[f"REVIEWER_{_n}_API_BASE"] = FieldMeta(_group, "API base URL")
    FIELD_META[f"REVIEWER_{_n}_MODEL"] = FieldMeta(_group, "Model")
    FIELD_META[f"REVIEWER_{_n}_FALLBACKS"] = FieldMeta(_group, "Fallback chain", kind="json")


def infer_kind(name: str, annotation: Any) -> str:
    """Control type for a field with no explicit `kind` in FIELD_META."""
    if annotation is bool:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if name.endswith("_API_KEY"):
        return "secret"
    if name.endswith("_FALLBACKS"):
        return "json"
    return "text"


def editable_fields() -> List[str]:
    return [n for n in Settings.model_fields if n not in EXCLUDED_FIELDS]


def describe_settings() -> Dict[str, Any]:
    """Current config as the dashboard consumes it: value + how to render it."""
    fields: List[Dict[str, Any]] = []

    for name in editable_fields():
        model_field = Settings.model_fields[name]
        meta = FIELD_META.get(name)
        if meta is None:
            # Field added to config.py but not yet annotated here.
            meta = FieldMeta("Other", name)
        value = getattr(settings, name)
        fields.append(
            {
                "name": name,
                "value": value,
                "group": meta.group,
                "label": meta.label,
                "help": meta.help,
                "kind": meta.kind or infer_kind(name, model_field.annotation),
                "choices": list(meta.choices),
                "restart_required": meta.restart_required,
                "caution": meta.caution,
                "env_override": env_overridden(name),
                "is_default": value == model_field.default,
            }
        )

    present = {f["group"] for f in fields}
    groups = [g for g in GROUP_ORDER if g in present]
    groups += sorted(present - set(groups))

    # Group order comes from GROUP_ORDER; within a group, keep config.py order.
    rank = {g: i for i, g in enumerate(groups)}
    fields.sort(key=lambda f: rank[f["group"]])
    return {"groups": groups, "fields": fields}
