from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, settings
from app.core.settings_schema import EXCLUDED_FIELDS, FIELD_META, describe_settings
from app.main import app
from app.services import _question_generator, pipeline, settings_service

pytestmark = pytest.mark.usefixtures("isolated_env")

SAMPLE = """\
# LLM config
LLM_FALLBACKS='[{"model":"a"}]'
LLM_TEMPERATURE=0.2
LLM_MAX_ATTEMPTS=4
EVIDENCE_TOP_K=10
"""


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """Point the settings service at a throwaway .env, never the real one."""
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
    """Undo in-place mutation of the process-wide settings singleton."""
    before = settings.model_dump()
    yield
    settings.__dict__.update(before)


def test_get_settings_exposes_every_field_except_the_denylist(client):
    resp = client.get("/settings")
    assert resp.status_code == 200

    names = {f["name"] for f in resp.json()["fields"]}
    assert names == set(Settings.model_fields) - EXCLUDED_FIELDS


def test_denylist_has_not_rotted():
    assert EXCLUDED_FIELDS <= set(Settings.model_fields)


def test_every_exposed_field_is_annotated():
    unannotated = [f["name"] for f in describe_settings()["fields"] if f["group"] == "Other"]
    assert unannotated == []


def test_secret_and_enum_kinds_are_inferred(client):
    fields = {f["name"]: f for f in client.get("/settings").json()["fields"]}

    assert fields["LLM_API_KEY"]["kind"] == "secret"
    assert fields["LLM_FALLBACKS"]["kind"] == "json"
    assert fields["FUSION_ENABLED"]["kind"] == "bool"
    assert fields["RETRIEVAL_MODE"]["choices"] == ["bm25_only", "hybrid_rerank"]


def test_restart_required_flagged_for_import_time_fields(client):
    fields = {f["name"]: f for f in client.get("/settings").json()["fields"]}

    assert fields["project_name"]["restart_required"] is True
    assert fields["CORS_ALLOW_ORIGINS"]["restart_required"] is True
    assert fields["LLM_TEMPERATURE"]["restart_required"] is False


def test_unknown_key_rejected_and_file_untouched(client, env_file):
    before = env_file.read_text(encoding="utf-8")
    resp = client.patch("/settings", json={"values": {"NOT_A_SETTING": 1}})

    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "unknown_setting"
    assert env_file.read_text(encoding="utf-8") == before


def test_excluded_field_rejected_and_file_untouched(client, env_file):
    before = env_file.read_text(encoding="utf-8")
    resp = client.patch("/settings", json={"values": {"CHUNK_SIZE": 500}})

    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "unknown_setting"
    assert env_file.read_text(encoding="utf-8") == before


def test_bad_type_rejected_and_file_untouched(client, env_file):
    before = env_file.read_text(encoding="utf-8")
    resp = client.patch("/settings", json={"values": {"EVIDENCE_TOP_K": "abc"}})

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "invalid_settings"
    assert "EVIDENCE_TOP_K" in detail["errors"]
    assert env_file.read_text(encoding="utf-8") == before


def test_invalid_enum_rejected(client, env_file):
    before = env_file.read_text(encoding="utf-8")
    resp = client.patch("/settings", json={"values": {"RETRIEVAL_MODE": "nonsense"}})

    assert resp.status_code == 422
    assert "RETRIEVAL_MODE" in resp.json()["detail"]["errors"]
    assert env_file.read_text(encoding="utf-8") == before


def test_invalid_json_fallbacks_rejected(client, env_file):
    before = env_file.read_text(encoding="utf-8")
    resp = client.patch("/settings", json={"values": {"LLM_FALLBACKS": "{not json"}})

    assert resp.status_code == 422
    assert "LLM_FALLBACKS" in resp.json()["detail"]["errors"]
    assert env_file.read_text(encoding="utf-8") == before


def test_valid_patch_writes_file_and_updates_singleton(client, env_file):
    resp = client.patch("/settings", json={"values": {"LLM_TEMPERATURE": 0.75}})

    assert resp.status_code == 200
    assert resp.json()["updated"] == ["LLM_TEMPERATURE"]
    assert "LLM_TEMPERATURE=0.75" in env_file.read_text(encoding="utf-8")
    assert settings.LLM_TEMPERATURE == 0.75


def test_patch_survives_a_shell_exported_override(client, env_file, monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")

    resp = client.patch("/settings", json={"values": {"LLM_TEMPERATURE": 0.9}})

    assert resp.status_code == 200
    assert settings.LLM_TEMPERATURE == 0.9


def test_llm_client_cache_dropped(client, monkeypatch):
    monkeypatch.setattr(pipeline, "_LLM_CLIENT", object())
    client.patch("/settings", json={"values": {"LLM_MODEL": "some-model"}})
    assert pipeline._LLM_CLIENT is None


def test_retriever_cache_cleared_on_retrieval_change(client, monkeypatch):
    monkeypatch.setitem(pipeline._RETRIEVERS, "kb1", object())
    monkeypatch.setitem(pipeline._RETRIEVER_ERRORS, "kb2", "boom")

    client.patch("/settings", json={"values": {"EVIDENCE_TOP_K": 7}})

    assert pipeline._RETRIEVERS == {}
    assert pipeline._RETRIEVER_ERRORS == {}


def test_import_time_constants_are_rebound(client, monkeypatch):
    monkeypatch.setattr(_question_generator, "_MAX_ATTEMPTS", 4)

    client.patch("/settings", json={"values": {"LLM_MAX_ATTEMPTS": 9}})

    assert _question_generator._MAX_ATTEMPTS == 9


def test_restart_required_reported_back(client):
    resp = client.patch("/settings", json={"values": {"CORS_ALLOW_ORIGINS": "http://x"}})

    assert resp.status_code == 200
    assert resp.json()["restart_required"] == ["CORS_ALLOW_ORIGINS"]


def test_caution_surfaced_as_warning(client, tmp_path):
    resp = client.patch("/settings", json={"values": {"KNOWLEDGE_DIR": str(tmp_path / "kb")}})

    assert resp.status_code == 200
    assert any("KNOWLEDGE_DIR" in w for w in resp.json()["warnings"])
    assert FIELD_META["KNOWLEDGE_DIR"].caution


def test_empty_patch_is_a_noop(client, env_file):
    before = env_file.read_text(encoding="utf-8")
    resp = client.patch("/settings", json={"values": {}})

    assert resp.status_code == 200
    assert resp.json()["updated"] == []
    assert env_file.read_text(encoding="utf-8") == before


def test_reload_endpoint_picks_up_a_manual_edit(client, env_file):
    client.patch("/settings", json={"values": {"RRF_K": 11}})
    assert settings.RRF_K == 11

    env_file.write_text(SAMPLE + "RRF_K=99\n", encoding="utf-8")
    resp = client.post("/settings/reload")

    assert resp.status_code == 200
    assert "RRF_K" in resp.json()["changed"]
    assert settings.RRF_K == 99
