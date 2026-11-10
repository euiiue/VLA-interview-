"""Validate → persist → reload → invalidate, for dashboard config edits.

Nothing is written until a candidate Settings validates, because assignment
onto the settings singleton is unvalidated (Settings sets no
validate_assignment) and a bad value would stick and fail much later.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Set

from pydantic import ValidationError

from app.core.config import Settings, env_file_path, reload_settings, settings
from app.core.env_writer import EnvWriteError, serialize_value, update_env_file
from app.core.settings_schema import EXCLUDED_FIELDS, FIELD_META

LOGGER = logging.getLogger(__name__)

# Serialises concurrent saves; a generate may be running against these caches.
_WRITE_LOCK = threading.Lock()

# Keys this process has pushed into os.environ to out-rank a shell export.
# Tracked so reload_from_disk can drop them and let the file win again.
_OWNED_ENV_KEYS: Set[str] = set()

# Snapshotted in LLMClient.__init__, so the cached client has to be dropped.
_LLM_KEYS = {
    "LLM_PROVIDER", "LLM_API_KEY", "LLM_API_BASE", "LLM_MODEL", "LLM_TIMEOUT_SEC",
    "LLM_TEMPERATURE", "LLM_TOP_P", "LLM_MAX_COMPLETION_TOKENS", "LLM_FALLBACKS",
    "LLM_REASONING_EFFORT", "LLM_STOP", "LLM_STREAM",
    "GROQ_API_KEY", "CEREBRAS_API_KEY",
}

# Snapshotted in Retriever.__init__, so cached retrievers have to be dropped.
_RETRIEVAL_KEYS = {
    "RETRIEVAL_MODE", "BM25_CANDIDATES_TOP_N", "RERANK_TOP_N",
    "EVIDENCE_TOP_K", "FUSION_ENABLED", "RRF_K",
}


class SettingsUpdateError(Exception):
    """Bad request payload. `code` is the snake_case detail the API returns."""

    def __init__(self, code: str, status: int = 400, errors: Dict[str, str] | None = None):
        super().__init__(code)
        self.code = code
        self.status = status
        self.errors = errors or {}


def _validate(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Reject unknown/excluded keys and type-check the whole candidate config."""
    unknown = [k for k in patch if k not in Settings.model_fields]
    if unknown:
        raise SettingsUpdateError("unknown_setting", 400, {k: "unknown setting" for k in unknown})

    excluded = [k for k in patch if k in EXCLUDED_FIELDS]
    if excluded:
        raise SettingsUpdateError(
            "unknown_setting", 400, {k: "setting is not read by the application" for k in excluded}
        )

    errors: Dict[str, str] = {}

    for key, value in patch.items():
        meta = FIELD_META.get(key)
        if meta and meta.choices and str(value) not in meta.choices:
            errors[key] = f"must be one of: {', '.join(c or '(blank)' for c in meta.choices)}"
        if key.endswith("_FALLBACKS") and isinstance(value, str) and value.strip():
            try:
                json.loads(value)
            except json.JSONDecodeError as exc:
                errors[key] = f"invalid JSON: {exc.msg}"

    if errors:
        raise SettingsUpdateError("invalid_settings", 422, errors)

    candidate = {**settings.model_dump(), **patch}
    try:
        validated = Settings.model_validate(candidate)
    except ValidationError as exc:
        for err in exc.errors():
            field = str(err["loc"][0]) if err["loc"] else "_"
            errors[field] = err["msg"]
        raise SettingsUpdateError("invalid_settings", 422, errors) from exc

    # Coerced values ("0.7" -> 0.7) so .env stores canonical forms.
    coerced = {key: getattr(validated, key) for key in patch}

    # Checked here, rather than only in the writer, so the message can be
    # attached to the field that caused it.
    for key, value in coerced.items():
        if not isinstance(value, str):
            continue
        if "\n" in value or "\r" in value:
            errors[key] = "must not contain a line break"
        elif "${" in value:
            errors[key] = (
                "must not contain ${...}: the .env reader expands it on load, "
                "so the value could not be read back as written"
            )

    if errors:
        raise SettingsUpdateError("invalid_settings", 422, errors)

    return coerced


def apply_runtime_settings(changed: Set[str]) -> None:
    """Drop caches and rebind import-time constants that snapshotted settings.

    Reviewers are deliberately absent: _get_reviewer_client and
    _get_cross_cove_reviewer rebuild from settings on every run, so REVIEWER_*
    and COVE_* changes take effect with no invalidation.
    """
    from app.services import _question_generator, pipeline

    if changed & _LLM_KEYS:
        pipeline._LLM_CLIENT = None
    if "EMBED_MODEL" in changed:
        pipeline._EMBEDDER = None
    if changed & _RETRIEVAL_KEYS or "EMBED_MODEL" in changed:
        pipeline._RETRIEVERS.clear()
        pipeline._RETRIEVER_ERRORS.clear()

    pipeline._MAX_DEDUP_ATTEMPTS = max(settings.LLM_DEDUP_ATTEMPTS, 1)
    _question_generator._MAX_ATTEMPTS = max(settings.LLM_MAX_ATTEMPTS, 1)
    _question_generator._CANDIDATES_PER_ATTEMPT = max(settings.LLM_CANDIDATES, 1)


def apply_settings_update(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Persist `patch` to .env and apply it to the running process."""
    if not patch:
        return {"updated": [], "restart_required": [], "warnings": []}

    coerced = _validate(patch)

    with _WRITE_LOCK:
        path = env_file_path()
        try:
            update_env_file(path, coerced)
        except EnvWriteError as exc:
            raise SettingsUpdateError("invalid_value", 400, {"_": str(exc)}) from exc
        
        for key, value in coerced.items():
            os.environ[key] = serialize_value(value)
            _OWNED_ENV_KEYS.add(key)

        reload_settings(path)
        changed = set(coerced)
        apply_runtime_settings(changed)

    restart_required = sorted(
        key for key in changed if key in FIELD_META and FIELD_META[key].restart_required
    )
    warnings = [
        f"{key}: {FIELD_META[key].caution}"
        for key in sorted(changed)
        if key in FIELD_META and FIELD_META[key].caution
    ]

    LOGGER.info("Settings updated via dashboard: %s", ", ".join(sorted(changed)))
    return {
        "updated": sorted(changed),
        "restart_required": restart_required,
        "warnings": warnings,
        "env_file": str(path),
    }


def reload_from_disk() -> List[str]:
    """Re-read .env without writing, for recovery after a manual edit.

    Keys this process previously wrote into os.environ are cleared first, so
    the file on disk is what actually wins.
    """
    with _WRITE_LOCK:
        before = settings.model_dump()
        for key in _OWNED_ENV_KEYS:
            os.environ.pop(key, None)
        _OWNED_ENV_KEYS.clear()

        reload_settings(env_file_path())
        after = settings.model_dump()
        changed = {key for key in after if before[key] != after[key]}
        apply_runtime_settings(changed)
    return sorted(changed)
