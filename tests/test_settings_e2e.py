"""End-to-end tests for the settings dashboard.

These exercise the whole path a real edit takes: HTTP request → validation →
.env write → reload → what a *freshly started process* would then read. The
last part matters most, since the dashboard's promise is that a save both
applies now and survives a restart.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, settings
from app.main import app
from app.services import settings_service

pytestmark = pytest.mark.usefixtures("isolated_env")

SAMPLE = """\
# App environment
environment=local

# LLM config
LLM_PROVIDER=openai_compatible
LLM_MODEL=cx/gpt-5.6-sol
# LLM_API_KEY=old-rotated-key
LLM_API_KEY=live-secret-key
LLM_TEMPERATURE=0.2
LLM_FALLBACKS='[{"provider":"groq","model":"a","api_key":"k1"}]'

# Reviewer
REVIEWER_FALLBACKS='[{"model":"first"}]'
REVIEWER_FALLBACKS=''
REVIEWER_API_KEY=reviewer-secret
"""


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(SAMPLE, encoding="utf-8")
    monkeypatch.setattr(settings_service, "env_file_path", lambda: path)
    monkeypatch.setattr(settings_service, "_OWNED_ENV_KEYS", set())
    return path


@pytest.fixture
def client(env_file):
    return TestClient(app)


@pytest.fixture(autouse=True)
def _restore_settings():
    before = settings.model_dump()
    yield
    settings.__dict__.update(before)


@pytest.fixture(autouse=True)
def _restore_environ():
    """apply_settings_update writes os.environ directly, which monkeypatch does
    not track. Without this, those writes leak into later tests."""
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


def restart(path) -> Settings:
    """What a freshly started server process would read from this file.

    The settings-shaped variables must be stripped first: os.environ outranks
    the dotenv file in pydantic-settings, and apply_settings_update writes
    there too, so leaving them set would make this read back the value we just
    put in the environment instead of the one on disk -- testing nothing.
    """
    saved = {}
    for name in Settings.model_fields:
        for key in (name, name.upper()):
            if key in os.environ:
                saved[key] = os.environ.pop(key)
    try:
        return Settings(_env_file=str(path), _env_file_encoding="utf-8")
    finally:
        os.environ.update(saved)


def test_multi_key_save_survives_a_restart(client, env_file):
    payload = {
        "LLM_MODEL": "gpt-4o",
        "LLM_TEMPERATURE": 0.85,
        "EVIDENCE_TOP_K": 15,
        "OK_ONLY_MODE": False,
        "RETRIEVAL_MODE": "bm25_only",
    }
    resp = client.patch("/settings", json={"values": payload})
    assert resp.status_code == 200
    assert set(resp.json()["updated"]) == set(payload)

    fresh = restart(env_file)
    assert fresh.LLM_MODEL == "gpt-4o"
    assert fresh.LLM_TEMPERATURE == 0.85
    assert fresh.EVIDENCE_TOP_K == 15
    assert fresh.OK_ONLY_MODE is False
    assert fresh.RETRIEVAL_MODE == "bm25_only"


def test_saving_one_key_leaves_other_secrets_intact(client, env_file):
    client.patch("/settings", json={"values": {"LLM_TEMPERATURE": 0.5}})

    fresh = restart(env_file)
    assert fresh.LLM_API_KEY == "live-secret-key"
    assert fresh.REVIEWER_API_KEY == "reviewer-secret"
    assert fresh.LLM_FALLBACKS == '[{"provider":"groq","model":"a","api_key":"k1"}]'
    assert "# LLM_API_KEY=old-rotated-key" in env_file.read_text(encoding="utf-8")


def test_repeated_saves_do_not_grow_the_file(client, env_file):
    baseline = len(env_file.read_text(encoding="utf-8").splitlines())

    for temperature in (0.3, 0.4, 0.5):
        client.patch("/settings", json={"values": {"LLM_TEMPERATURE": temperature}})

    after = env_file.read_text(encoding="utf-8").splitlines()
    assert len(after) == baseline
    assert sum(1 for line in after if line.startswith("LLM_TEMPERATURE=")) == 1
    assert restart(env_file).LLM_TEMPERATURE == 0.5


def test_duplicate_key_stays_a_duplicate_and_keeps_last_wins(client, env_file):
    client.patch("/settings", json={"values": {"REVIEWER_FALLBACKS": '[{"model":"new"}]'}})

    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert sum(1 for line in lines if line.startswith("REVIEWER_FALLBACKS=")) == 2
    # dotenv reads the last one; that is the one that must carry the new value.
    assert restart(env_file).REVIEWER_FALLBACKS == '[{"model":"new"}]'

@pytest.mark.parametrize(
    "value",
    [
        "model#with-hash",
        "model with spaces",
        'model"with-double-quote',
        "model'with-single-quote",
        "both\"and'quotes",
        "trailing-space ",
        " leading-space",
        "sk-key$with$dollars",
        "key=with=equals",
        "unicode-ăîșț-模型",
        "a" * 500,
        "back\\slash",
    ],
)
def test_awkward_values_round_trip_through_a_restart(client, env_file, value):
    resp = client.patch("/settings", json={"values": {"LLM_MODEL": value}})
    assert resp.status_code == 200, resp.json()

    assert restart(env_file).LLM_MODEL == value


@pytest.mark.parametrize("value", ["${NOT_SET}", "prefix-${HOME}-suffix", "${A}"])
def test_variable_expansion_is_rejected_not_silently_corrupted(client, env_file, value):
    """python-dotenv expands ${VAR} on read under every quoting style, and no
    escape round trips. Storing such a value would read back as something the
    user never entered, so it must be refused instead."""
    before = env_file.read_text(encoding="utf-8")

    resp = client.patch("/settings", json={"values": {"LLM_MODEL": value}})

    assert resp.status_code == 422
    assert "LLM_MODEL" in resp.json()["detail"]["errors"]
    assert env_file.read_text(encoding="utf-8") == before
    assert restart(env_file).LLM_MODEL == "cx/gpt-5.6-sol"


def test_clearing_a_value_persists_as_empty(client, env_file):
    client.patch("/settings", json={"values": {"LLM_API_KEY": ""}})
    assert restart(env_file).LLM_API_KEY == ""


def test_fallback_chain_edited_as_json_round_trips(client, env_file):
    chain = [
        {"provider": "openai_compatible", "model": "m1", "api_key": "k1", "api_base": "https://a/v1"},
        {"provider": "groq", "model": "m2", "groq_api_key": "k2"},
    ]
    resp = client.patch("/settings", json={"values": {"LLM_FALLBACKS": json.dumps(chain)}})
    assert resp.status_code == 200

    from app.generation.llm_client import _parse_fallbacks

    # Verify through the parser that actually consumes this value.
    assert _parse_fallbacks(restart(env_file).LLM_FALLBACKS) == chain


def test_one_bad_value_rejects_the_whole_patch(client, env_file):
    before = env_file.read_text(encoding="utf-8")

    resp = client.patch(
        "/settings",
        json={"values": {"LLM_MODEL": "good-value", "EVIDENCE_TOP_K": "not-a-number"}},
    )

    assert resp.status_code == 422
    assert "EVIDENCE_TOP_K" in resp.json()["detail"]["errors"]
    # The valid key in the same request must not have been written either.
    assert env_file.read_text(encoding="utf-8") == before
    assert restart(env_file).LLM_MODEL == "cx/gpt-5.6-sol"


def test_concurrent_saves_do_not_corrupt_the_file(client, env_file):
    def save(i: int):
        return client.patch("/settings", json={"values": {f"RRF_K": 10 + i}}).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(save, range(8)))

    assert set(codes) == {200}
    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert sum(1 for line in lines if line.startswith("RRF_K=")) == 1
    assert 10 <= restart(env_file).RRF_K <= 17
    assert not (env_file.parent / ".env.tmp").exists()


def test_save_wins_over_a_shell_export_and_persists(client, env_file, monkeypatch):
    """The oh-my-zsh `dotenv` plugin exports .env into the shell on cd.

    os.environ outranks the dotenv file, so a save has to update both or it
    is a silent no-op for the running process.
    """
    monkeypatch.setenv("LLM_MODEL", "stale-from-shell")

    resp = client.patch("/settings", json={"values": {"LLM_MODEL": "chosen-in-ui"}})
    assert resp.status_code == 200

    assert settings.LLM_MODEL == "chosen-in-ui"  # running process
    assert os.environ["LLM_MODEL"] == "chosen-in-ui"  # its environment
    assert restart(env_file).LLM_MODEL == "chosen-in-ui"  # next process


def test_reload_from_disk_drops_values_this_process_wrote(client, env_file):
    client.patch("/settings", json={"values": {"RRF_K": 11}})
    assert os.environ["RRF_K"] == "11"

    # Someone edits the file by hand behind the server's back.
    env_file.write_text(SAMPLE + "RRF_K=99\n", encoding="utf-8")
    resp = client.post("/settings/reload")

    assert resp.status_code == 200
    assert "RRF_K" in resp.json()["changed"]
    assert settings.RRF_K == 99
    assert "RRF_K" not in os.environ


def test_reload_from_disk_cannot_override_a_foreign_env_var(client, env_file, monkeypatch):
    """A documented limitation, pinned so it cannot regress silently.

    If something outside this process exported a variable, os.environ still
    outranks the file, so 'reload from disk' genuinely cannot pick up a manual
    edit for that key. The endpoint must report that honestly rather than
    claim a change it did not make.
    """
    monkeypatch.setenv("RRF_K", "7")
    settings_service.reload_settings(env_file)
    assert settings.RRF_K == 7

    env_file.write_text(SAMPLE + "RRF_K=99\n", encoding="utf-8")
    resp = client.post("/settings/reload")

    assert resp.status_code == 200
    assert resp.json()["changed"] == []
    assert settings.RRF_K == 7


def test_reviewer_changes_need_no_cache_invalidation(client):
    """Reviewers are rebuilt from settings on every run, so a change to them
    must be visible immediately without touching any cache."""
    resp = client.patch(
        "/settings",
        json={"values": {"REVIEWER_MODEL": "new-reviewer", "COVE_ENABLED": False}},
    )

    assert resp.status_code == 200
    assert settings.REVIEWER_MODEL == "new-reviewer"
    assert settings.COVE_ENABLED is False
