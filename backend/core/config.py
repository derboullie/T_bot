"""Configuration management using Pydantic settings."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # API Credentials
    polygon_api_key: str = Field(..., description="Polygon.io API key")
    alpaca_api_key: str = Field(..., description="Alpaca API key")
    alpaca_secret_key: str = Field(..., description="Alpaca secret key")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca base URL (paper or live)",
    )

    # Database
    database_url: str = Field(
        default="sqlite:///./tradingbot.db", description="Database connection string"
    )

    # Trading Parameters
    max_position_size: float = Field(default=10000.0, description="Maximum position size in USD")
    max_daily_loss: float = Field(default=1000.0, description="Maximum daily loss in USD")
    max_positions: int = Field(default=10, description="Maximum concurrent positions")
    risk_per_trade: float = Field(default=1.0, description="Risk per trade as % of portfolio")

    # Resource Management
    cpu_limit_percent: int = Field(default=85, ge=1, le=100, description="CPU usage limit")
    worker_threads: int = Field(default=4, ge=1, le=32, description="Number of worker threads")
    data_refresh_interval: int = Field(
        default=1, ge=1, description="Data refresh interval in seconds"
    )

    # Security
    jwt_secret_key: str = Field(
        default="change-me-in-production", description="JWT secret key"
    )
    jwt_expiration_minutes: int = Field(default=60, description="JWT expiration in minutes")
    api_access_token: Optional[str] = Field(default=None, description="API access token")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_file: str = Field(default="logs/tradingbot.log", description="Log file path")

    # Backtesting
    backtest_days: int = Field(default=365, description="Default backtesting period in days")
    backtest_initial_capital: float = Field(
        default=100000.0, description="Initial capital for backtesting"
    )

    # Machine Learning
    ml_training_enabled: bool = Field(default=False, description="Enable ML training")
    ml_model_path: str = Field(default="models/", description="ML model storage path")
    ml_training_lookback_days: int = Field(
        default=730, description="Training data lookback in days"
    )
    ml_feature_update_hours: int = Field(
        default=24, description="Feature update interval in hours"
    )

    # Frontend
    frontend_url: str = Field(default="http://localhost:5173", description="Frontend URL")
    backend_url: str = Field(default="http://localhost:8000", description="Backend API URL")

    # Monitoring
    prometheus_enabled: bool = Field(default=False, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, description="Prometheus port")

    # Application
    app_name: str = Field(default="HFT Trading Bot", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings
    """
    return Settings()


# Export settings instance
settings = get_settings()
