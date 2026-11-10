from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.env_writer import EnvWriteError, quote_value, serialize_value, update_env_file

pytestmark = pytest.mark.usefixtures("isolated_env")

# Mirrors the shape of the real .env
SAMPLE = """\
# App environment
environment=local

# LLM config
LLM_FALLBACKS='[{"model":"a"}]'
LLM_TEMPERATURE=0.2
# LLM_API_KEY=old-rotated-key
LLM_API_KEY=live-key

# Reviewer
REVIEWER_FALLBACKS='[{"model":"first"}]'
REVIEWER_FALLBACKS=''
"""


@pytest.fixture
def env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


def test_preserves_comments_blank_lines_and_order(env_file):
    update_env_file(env_file, {"LLM_TEMPERATURE": 0.7})
    lines = env_file.read_text(encoding="utf-8").splitlines()

    assert lines[0] == "# App environment"
    assert lines[2] == ""
    assert lines[3] == "# LLM config"
    # The commented-out rotation note survives untouched.
    assert "# LLM_API_KEY=old-rotated-key" in lines
    assert lines[5] == "LLM_TEMPERATURE=0.7"


def test_only_the_targeted_line_changes(env_file):
    before = env_file.read_text(encoding="utf-8").splitlines()
    update_env_file(env_file, {"LLM_TEMPERATURE": 0.7})
    after = env_file.read_text(encoding="utf-8").splitlines()

    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert differing == [5]


def test_duplicate_key_updates_last_occurrence_only(env_file):
    # dotenv resolves duplicates last-wins, so the last line is the live one;
    # rewriting the first would silently change which value the app reads.
    update_env_file(env_file, {"REVIEWER_FALLBACKS": '[{"model":"new"}]'})
    lines = env_file.read_text(encoding="utf-8").splitlines()

    assert lines[10] == """REVIEWER_FALLBACKS='[{"model":"first"}]'"""
    assert lines[11] == """REVIEWER_FALLBACKS='[{"model":"new"}]'"""


def test_commented_assignment_is_not_treated_as_the_key(env_file):
    update_env_file(env_file, {"LLM_API_KEY": "rotated"})
    text = env_file.read_text(encoding="utf-8")

    assert "# LLM_API_KEY=old-rotated-key" in text
    assert "\nLLM_API_KEY=rotated\n" in text


def test_absent_key_appended_under_managed_footer(env_file):
    update_env_file(env_file, {"OK_ONLY_MAX_ROUNDS": 5})
    lines = env_file.read_text(encoding="utf-8").splitlines()

    assert "# --- Managed by Settings UI ---" in lines
    assert lines[-1] == "OK_ONLY_MAX_ROUNDS=5"


def test_managed_footer_written_only_once(env_file):
    update_env_file(env_file, {"OK_ONLY_MAX_ROUNDS": 5})
    update_env_file(env_file, {"RRF_K": 42})
    lines = env_file.read_text(encoding="utf-8").splitlines()

    assert lines.count("# --- Managed by Settings UI ---") == 1
    # The second write updates in place rather than appending a duplicate.
    update_env_file(env_file, {"RRF_K": 43})
    assert env_file.read_text(encoding="utf-8").count("RRF_K=") == 1


def test_json_value_round_trips_through_settings(env_file):
    blob = '[{"provider":"groq","model":"x","api_key":"k"}]'
    update_env_file(env_file, {"LLM_FALLBACKS": blob})

    # Read back the way the app does, without importing python-dotenv (which
    # is only a transitive dependency here).
    loaded = Settings(_env_file=str(env_file), _env_file_encoding="utf-8")
    assert loaded.LLM_FALLBACKS == blob


def test_newline_value_is_rejected_before_touching_the_file(env_file):
    before = env_file.read_text(encoding="utf-8")
    with pytest.raises(EnvWriteError):
        update_env_file(env_file, {"LLM_MODEL": "a\nb"})
    assert env_file.read_text(encoding="utf-8") == before


def test_variable_expansion_is_rejected(env_file):
    before = env_file.read_text(encoding="utf-8")
    with pytest.raises(EnvWriteError):
        update_env_file(env_file, {"LLM_MODEL": "${HOME}"})
    assert env_file.read_text(encoding="utf-8") == before


def test_rejection_names_the_offending_key(env_file):
    before = env_file.read_text(encoding="utf-8")

    with pytest.raises(EnvWriteError, match="LLM_STOP"):
        update_env_file(
            env_file,
            {"LLM_TEMPERATURE": 0.5, "LLM_STOP": "bad\nvalue", "LLM_MODEL": "fine"},
        )

    # None of the keys in a rejected batch may be written.
    assert env_file.read_text(encoding="utf-8") == before


def test_backup_written_and_no_temp_file_left(env_file):
    update_env_file(env_file, {"LLM_TEMPERATURE": 0.7})

    assert (env_file.parent / ".env.bak").read_text(encoding="utf-8") == SAMPLE
    assert not (env_file.parent / ".env.tmp").exists()


def test_creates_file_when_missing(tmp_path):
    path = tmp_path / ".env"
    update_env_file(path, {"RRF_K": 10})
    assert path.read_text(encoding="utf-8").endswith("RRF_K=10\n")


def test_serialize_bool_matches_file_style():
    assert serialize_value(True) == "true"
    assert serialize_value(False) == "false"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("plain", "plain"),
        ("", ""),
        ("has space", "'has space'"),
        ("has#hash", "'has#hash'"),
        ('{"a":1}', '\'{"a":1}\''),
        (" padded ", "' padded '"),
    ],
)
def test_quote_value(value, expected):
    assert quote_value(value) == expected


def test_quoted_values_round_trip(tmp_path):
    path = tmp_path / ".env"
    values = {
        "LLM_MODEL": "model with space",
        "LLM_STOP": "###",
        "LLM_FALLBACKS": '[{"a":"b"}]',
        "LLM_API_KEY": "",
    }
    update_env_file(path, values)

    loaded = Settings(_env_file=str(path), _env_file_encoding="utf-8")
    for key, expected in values.items():
        assert getattr(loaded, key) == expected
