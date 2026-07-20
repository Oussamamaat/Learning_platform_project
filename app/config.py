from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "IBLOG AI Assistant"
    app_version: str = "0.1.0"
    database_url: str = "postgresql://assistant:changeme@localhost:5432/iblog_assistant"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
