from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VoltAgent"
    debug: bool = False
    database_url: str = "sqlite:///./data/voltagent.db"
    llm_provider: str = Field("openai_compatible", validation_alias="LLM_PROVIDER")
    llm_model: str = Field("", validation_alias="LLM_MODEL")
    llm_api_key: str | None = Field(None, validation_alias="LLM_API_KEY")
    llm_base_url: str | None = Field(None, validation_alias="LLM_BASE_URL")
    gemini_api_key: str | None = Field(None, validation_alias="GEMINI_API_KEY")
    embedding_provider: str = Field("local", validation_alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        validation_alias="EMBEDDING_MODEL",
    )
    dense_index_path: str = Field(
        "data/indexes/kuphaldt-arctic-embed-l-v2",
        validation_alias="DENSE_INDEX_PATH",
    )
    reranker_enabled: bool = Field(False, validation_alias="RERANKER_ENABLED")
    reranker_model: str = Field(
        "BAAI/bge-reranker-v2-m3",
        validation_alias="RERANKER_MODEL",
    )
    reranker_candidate_k: int = Field(30, validation_alias="RERANKER_CANDIDATE_K")
    reranker_batch_size: int = Field(16, validation_alias="RERANKER_BATCH_SIZE")
    rag_top_k: int = Field(5, validation_alias="RAG_TOP_K")
    chunk_size: int = Field(800, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(120, validation_alias="CHUNK_OVERLAP")
    decision_provider: str = Field("llm", validation_alias="DECISION_PROVIDER")
    jev_api_key: str | None = Field(None, validation_alias="JEV_API_KEY")
    yolo_model_path: str | None = Field(None, validation_alias="YOLO_MODEL_PATH")
    yolo_model_repo: str = Field(
        "boscochanam/circuit-component-detector",
        validation_alias="YOLO_MODEL_REPO",
    )
    yolo_model_filename: str = Field(
        "yolo26m_obb_16class_aug.pt",
        validation_alias="YOLO_MODEL_FILENAME",
    )
    yolo_model_runtime: str = Field("auto", validation_alias="YOLO_MODEL_RUNTIME")
    yolo_device: str = Field("auto", validation_alias="YOLO_DEVICE")
    upload_dir: str = "data/uploads"
    vector_db_path: str = "data/vectors.sqlite3"
    max_upload_bytes: int = 10 * 1024 * 1024

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    @property
    def sqlite_path(self) -> str:
        value = self.database_url.removeprefix("sqlite:///")
        Path(value).parent.mkdir(parents=True, exist_ok=True)
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
