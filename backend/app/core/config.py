import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import BaseModel

_backend_dir = Path(__file__).parent.parent.parent
# ENV_FILE can be set to switch between configurations, e.g.:
#   ENV_FILE=.env.cloud uv run langgraph dev
_env_file = os.getenv("ENV_FILE", str(_backend_dir / ".env.local"))


class QdrantConfig(BaseModel):
    url: str
    api_key: str
    vector_size: int
    collection: str
    distance: str
    output_dir: str

class CeleryConfig(BaseModel):
    broker_url: str
    result_backend: str

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_file,
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
        case_sensitive=False
    )
    # Logging configuration
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    EMBEDDING_MODEL_NAME: str

    SUPERVISOR_MODEL_NAME: str
    PF_AGENT_MODEL_NAME: str
    PF_FILTER_MODEL_NAME: str
    QA_AGENT_MODEL_NAME: str
    QA_EVALUATION_MODEL_NAME: str
    QA_EVALUATOR_MODEL_NAME: str
    QA_BASELINE_MODEL_NAME: str

    COHERE_API_KEY: str
    

    PDF_DOWNLOAD_DIR: str

    QDRANT_URL: str
    QDRANT_API_KEY: str = ""
    QDRANT_VECTOR_SIZE: int
    QDRANT_COLLECTION: str
    QDRANT_DISTANCE: str

    S2_API_KEY: str

    REDIS_URL: str

    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    GROBID_SERVER_URL: str

    CLERK_JWKS_URL: str = ""  # only required by the LangGraph backend, not the Celery worker
    DISABLE_AUTH: bool = False
    
    @property
    def qdrant_config(self) -> QdrantConfig:
        return QdrantConfig(
            url=self.QDRANT_URL,
            api_key=self.QDRANT_API_KEY,
            vector_size=self.QDRANT_VECTOR_SIZE,
            collection=self.QDRANT_COLLECTION,
            distance=self.QDRANT_DISTANCE,
            output_dir=self.PDF_DOWNLOAD_DIR,
        )

    @property
    def celery_config(self) -> CeleryConfig:
        return CeleryConfig(
            broker_url=self.CELERY_BROKER_URL,
            result_backend=self.CELERY_RESULT_BACKEND,
        )

settings = Settings()
