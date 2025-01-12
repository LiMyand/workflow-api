from pydantic_settings import BaseSettings
from typing import Dict, Optional


class Settings(BaseSettings):
    PROJECT_NAME: str = "Workflow LLM System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    OPENAI_API_KEY: str
    OPENAI_API_BASE: Optional[str] = None

    AVAILABLE_MODELS: Dict[str, Dict] = {
        "gpt-4": {
            "provider": "openai",
            "max_tokens": 8192,
            "temperature_range": (0.0, 2.0),
        },
        "gpt-4o-mini": {
            "provider": "openai",
            "max_tokens": 8192,
            "temperature_range": (0.0, 2.0),
        },
        "gpt-4-turbo": {
            "provider": "openai",
            "max_tokens": 4096,
            "temperature_range": (0.0, 2.0),
        },
        "gpt-3.5-turbo": {
            "provider": "openai",
            "max_tokens": 4096,
            "temperature_range": (0.0, 2.0),
        },
    }

    class Config:
        env_file = ".env"


settings = Settings()
