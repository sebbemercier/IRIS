# Copyright 2026 The OpenSLM Project
from pydantic_settings import BaseSettings, SettingsConfigDict

class IrisSettings(BaseSettings):
    # AI Settings
    TOKENIZER_PATH: str = "models/ecommerce_tokenizer.model"
    BASE_MODEL_ID: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Analytics Settings
    ALERT_THRESHOLD_DAYS: int = 3
    ANALYTICS_DB_URL: str = "sqlite:///./analytics.db"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = IrisSettings()
