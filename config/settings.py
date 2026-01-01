# text2sql_mvp/config/settings.py
"""
Configuration settings for Text-to-SQL MVP.
Uses pydantic-settings for environment variable management.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseMode(str, Enum):
    """Database mode selection."""
    DUCKDB = "duckdb"
    BIGQUERY = "bigquery"


class LogLevel(str, Enum):
    """Logging level options."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class GeminiSettings(BaseSettings):
    """Gemini API configuration."""
    
    model_config = SettingsConfigDict(env_prefix="GEMINI_")
    
    api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    model: str = Field(default="gemini-2.0-flash-exp")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=4096, ge=1, le=8192)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=100)


class BigQuerySettings(BaseSettings):
    """BigQuery configuration."""
    
    model_config = SettingsConfigDict(env_prefix="BIGQUERY_")
    
    project: str = Field(default="")
    dataset: str = Field(default="text2sql_finance")
    location: str = Field(default="US")
    credentials_path: Optional[str] = Field(
        default=None, 
        alias="GOOGLE_APPLICATION_CREDENTIALS"
    )


class DuckDBSettings(BaseSettings):
    """DuckDB configuration for local testing."""
    
    model_config = SettingsConfigDict(env_prefix="DUCKDB_")
    
    path: str = Field(default=":memory:")


class PerformanceSettings(BaseSettings):
    """Performance and rate limiting configuration."""
    
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    request_timeout: int = Field(default=60, ge=10, le=300)
    query_timeout: int = Field(default=30, ge=5, le=120)
    max_rows_returned: int = Field(default=1000, ge=1, le=10000)
    dry_run_default: bool = Field(default=False)


class CacheSettings(BaseSettings):
    """Caching configuration."""
    
    enabled: bool = Field(default=True, alias="ENABLE_CACHE")
    ttl: int = Field(default=3600, alias="CACHE_TTL")


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Google API Key (needed at top level for Gemini)
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    
    # Database mode
    database_mode: DatabaseMode = Field(default=DatabaseMode.DUCKDB)
    
    # Default domain
    default_domain: str = Field(default="finance")
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: str = Field(default="json")
    
    # Nested settings
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    bigquery: BigQuerySettings = Field(default_factory=BigQuerySettings)
    duckdb: DuckDBSettings = Field(default_factory=DuckDBSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    
    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def domains_dir(self) -> Path:
        """Path to domains configuration directory."""
        return self.project_root / "data" / "domains"
    
    @property
    def config_domains_dir(self) -> Path:
        """Path to domain YAML configurations."""
        return self.project_root / "config" / "domains"
    
    def get_domain_path(self, domain: str) -> Path:
        """Get path to a specific domain's data directory."""
        return self.domains_dir / domain
    
    @field_validator("database_mode", mode="before")
    @classmethod
    def validate_database_mode(cls, v: str) -> DatabaseMode:
        """Validate and convert database mode."""
        if isinstance(v, DatabaseMode):
            return v
        return DatabaseMode(v.lower())


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()


# Convenience function for accessing settings
def get_gemini_settings() -> GeminiSettings:
    """Get Gemini-specific settings."""
    settings = get_settings()
    # Override API key from top-level if not set in nested
    if not settings.gemini.api_key and settings.google_api_key:
        settings.gemini.api_key = settings.google_api_key
    return settings.gemini


def get_database_settings() -> DuckDBSettings | BigQuerySettings:
    """Get database-specific settings based on mode."""
    settings = get_settings()
    if settings.database_mode == DatabaseMode.DUCKDB:
        return settings.duckdb
    return settings.bigquery
