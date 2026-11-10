import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = "docs-to-quiz"
    environment: str = "local"
    CORS_ALLOW_ORIGINS: str = "http://localhost:3000"
    RETRIEVAL_MODE: str = "bm25_only"
    BM25_CANDIDATES_TOP_N: int = 200
    RERANK_TOP_N: int = 50
    EVIDENCE_TOP_K: int = 10
    EVIDENCE_TOP_K_EXPANDED: int = 20
    FUSION_ENABLED: bool = False
    RRF_K: int = 60
    LLM_PROVIDER: str = "gemini_sdk"
    LLM_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    CEREBRAS_API_KEY: str = ""
    LLM_API_BASE: str = "https://generativelanguage.googleapis.com/v1beta"
    LLM_MODEL: str = "gemini-2.5-flash-lite"
    LLM_TIMEOUT_SEC: float = 60.0
    LLM_TEMPERATURE: float = 0.2
    LLM_TOP_P: float = 1.0
    LLM_MAX_COMPLETION_TOKENS: int = 2048
    LLM_FALLBACKS: str = ""
    LLM_JSON_MODE: bool = True
    LLM_MAX_ATTEMPTS: int = 4
    LLM_CANDIDATES: int = 1
    LLM_DEDUP_ATTEMPTS: int = 2
    LLM_REASONING_EFFORT: str = ""
    LLM_STOP: str = ""
    LLM_STREAM: bool = False
    CEREBRAS_TIMEOUT_SEC: float = 30.0
    CEREBRAS_MAX_RETRIES: int = 1
    OK_ONLY_MODE: bool = True
    OK_ONLY_MAX_ROUNDS: int = 3
    OK_ONLY_MAX_SECONDS: float = 90.0
    EVIDENCE_FIRST: bool = True
    EVIDENCE_FIRST_MAX_SENTENCES: int = 2
    EVIDENCE_FIRST_MAX_CHARS: int = 600
    EVIDENCE_MAX_CHARS_PER_DOC: int = 800
    EVIDENCE_MAX_TOTAL_CHARS: int = 4000
    QUESTION_BANK_PATH: str = "app/logging/question_bank.jsonl"
    QUESTION_BANK_FALLBACK: bool = False
    QUESTION_BANK_WRITE_OK: bool = False
    QUESTION_BANK_ALLOW_ANY_TOPIC: bool = False

    # Knowledge base ingestion (domain-agnostic, local-only)
    KNOWLEDGE_DIR: str = "data/knowledge"
    EMBED_MODEL: str = "intfloat/multilingual-e5-small"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    REVIEWER_ENABLED: bool = True
    REVIEWER_PROVIDER: str = ""
    REVIEWER_API_KEY: str = ""
    REVIEWER_API_BASE: str = ""
    REVIEWER_MODEL: str = ""
    REVIEWER_TIMEOUT_SEC: float = 60.0
    REVIEWER_FALLBACKS: str = ""
    REVIEWER_MAX_COMPLETION_TOKENS: int = 256
    
    # Cross-CoVe: 3 different reviewers for majority voting
    COVE_ENABLED: bool = True
    COVE_VOTE_COUNT: int = 3
    
    # Reviewer 1 config
    REVIEWER_1_PROVIDER: str = ""
    REVIEWER_1_API_KEY: str = ""
    REVIEWER_1_API_BASE: str = ""
    REVIEWER_1_MODEL: str = ""
    REVIEWER_1_FALLBACKS: str = ""
    
    # Reviewer 2 config
    REVIEWER_2_PROVIDER: str = ""
    REVIEWER_2_API_KEY: str = ""
    REVIEWER_2_API_BASE: str = ""
    REVIEWER_2_MODEL: str = ""
    REVIEWER_2_FALLBACKS: str = ""
    
    # Reviewer 3 config
    REVIEWER_3_PROVIDER: str = ""
    REVIEWER_3_API_KEY: str = ""
    REVIEWER_3_API_BASE: str = ""
    REVIEWER_3_MODEL: str = ""
    REVIEWER_3_FALLBACKS: str = ""
    NLI_MODEL: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    SEMANTIC_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


settings = Settings()


def env_file_path() -> Path:
    """Absolute path of the .env pydantic-settings reads.

    Resolved the same way pydantic does (relative to the process CWD) so the
    settings writer and the settings reader can never disagree about which
    file is authoritative.
    """
    return Path(Settings.model_config["env_file"]).resolve()


def reload_settings(env_file: Path | None = None) -> Settings:
    """Re-read the env file into the existing `settings` singleton, in place.

    Every module does `from app.core.config import settings` and holds that
    object, so a rebind would be invisible to them; the field values have to
    be copied onto the instance they already have.

    `env_file` is passed explicitly by callers that just wrote a file, so the
    reader is provably reading what the writer wrote instead of relying on
    both resolving the same CWD-relative path.
    """
    path = env_file or env_file_path()
    fresh = Settings(_env_file=str(path))
    settings.__dict__.update(fresh.__dict__)
    return settings


def env_overridden(name: str) -> bool:
    """True when a real environment variable shadows this field.

    pydantic-settings ranks os.environ above the dotenv file, so for these
    fields writing .env has no effect until the process environment changes
    (typical under Docker). Callers surface this rather than silently
    pretending the write took.
    """
    return name in os.environ or name.upper() in os.environ
