"""Environment variable loading and constants."""
import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Slack
    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_app_token: str = ""

    # LLM
    openai_api_key: str = ""

    # Weaviate
    weaviate_url: str = "http://localhost:8080"

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "northstar-slack-bot"

    # App config
    database_path: Path = Path("data/synthetic_startup.sqlite")
    log_level: str = "INFO"
    max_retry_count: int = 2
    sql_row_limit: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# Constants
APP_NAME = "Northstar Signal"
APP_VERSION = "1.0.0"
