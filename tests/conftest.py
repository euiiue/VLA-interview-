from __future__ import annotations

import pytest

from app.core.config import Settings


@pytest.fixture
def isolated_env(monkeypatch):
    """Make an .env file the only source of config for the duration of a test.

    pydantic-settings ranks os.environ above the dotenv file, and a shell can
    easily have the whole .env exported into it (the oh-my-zsh `dotenv` plugin
    does this on cd). Without stripping those, a test that writes a temp .env
    would silently read the developer's real values instead.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    yield
