"""Surgical .env editing.

The .env in this repo is hand-maintained: it carries section comments and
commented-out key blocks kept as a rotation log, and it defines LLM_FALLBACKS
and REVIEWER_FALLBACKS twice each. dotenv resolves duplicates last-wins, so a
rewriter that regenerates the file -- or that dedupes keys -- would silently
change which value is live. This module therefore edits individual lines and
leaves every other byte alone.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

MANAGED_HEADER = "# --- Managed by Settings UI ---"

# KEY=..., allowing leading whitespace and an optional `export ` prefix.
_ASSIGNMENT = re.compile(r"^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z0-9_]*)(\s*=)(.*)$")


class EnvWriteError(ValueError):
    """Raised for a value that cannot be represented on a single .env line."""


def serialize_value(value: Any) -> str:
    """Render a Python value the way this .env already writes that type."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def quote_value(value: str) -> str:
    """Quote only when the value would otherwise parse back differently."""
    if value == "":
        return ""
    if "\n" in value or "\r" in value:
        raise EnvWriteError("value_contains_newline")
    if "${" in value:
        raise EnvWriteError("value_contains_variable_expansion")

    needs_quotes = (
        value != value.strip()
        or " " in value
        or "#" in value
        or '"' in value
        or "'" in value
    )
    if not needs_quotes:
        return value

    # Single quotes match how the existing JSON fallback blobs are written
    if "'" not in value:
        return f"'{value}'"
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format_line(prefix: str, key: str, sep: str, value: str) -> str:
    return f"{prefix}{key}{sep}{quote_value(value)}"


def update_env_file(path: Path, updates: Dict[str, Any]) -> List[str]:
    """Apply `updates` to the .env at `path`, preserving everything else.

    Existing keys are rewritten in place at their *last* uncommented
    occurrence, which is the one dotenv actually reads. Earlier duplicates and
    all comment lines are left byte-identical. Unknown keys are appended under
    a managed footer. Returns the keys that were written.
    """
    if not updates:
        return []

    rendered = {key: serialize_value(value) for key, value in updates.items()}
    for key, value in rendered.items():
        # Check every value up front so the error names the offending key
        try:
            quote_value(value)
        except EnvWriteError as exc:
            raise EnvWriteError(f"{key}: {exc}") from exc

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = existing.splitlines()

    # Last uncommented assignment per key -- the live one under last-wins.
    last_index: Dict[str, int] = {}
    for i, line in enumerate(lines):
        match = _ASSIGNMENT.match(line)
        if match and not line.lstrip().startswith("#"):
            last_index[match.group(2)] = i

    written: List[str] = []
    appended: List[str] = []

    for key, value in rendered.items():
        index = last_index.get(key)
        if index is None:
            appended.append(f"{key}={quote_value(value)}")
        else:
            match = _ASSIGNMENT.match(lines[index])
            assert match is not None  # matched during the scan above
            lines[index] = _format_line(match.group(1), key, match.group(3), value)
        written.append(key)

    if appended:
        if lines and lines[-1].strip():
            lines.append("")
        if MANAGED_HEADER not in lines:
            lines.append(MANAGED_HEADER)
        lines.extend(appended)

    content = "\n".join(lines)
    if content and not content.endswith("\n"):
        content += "\n"

    _atomic_write(path, content)
    return written


def _atomic_write(path: Path, content: str) -> None:
    """Back up, then swap the new content in via a rename.

    This file holds live API keys, so a half-written .env is not an acceptable
    failure mode and a bad save has to stay recoverable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = None
    if path.exists():
        mode = path.stat().st_mode
        shutil.copy2(path, path.with_name(path.name + ".bak"))

    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
