"""
Configuration settings for Voice Style Replication application.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "Voice Style Replication"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API
    API_BASE_URL: str = "http://localhost:8000"
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str = "sqlite:///./voice_cloning.db"
    
    # File Storage
    UPLOAD_DIR: str = "uploads"
    RESULTS_DIR: str = "results"
    MODELS_DIR: str = "models"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # Audio Processing
    SUPPORTED_AUDIO_FORMATS: List[str] = [".mp3", ".wav", ".flac", ".m4a"]
    SUPPORTED_VIDEO_FORMATS: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    DEFAULT_SAMPLE_RATE: int = 22050
    
    # Text Processing
    MAX_TEXT_LENGTH: int = 1000
    SUPPORTED_LANGUAGES: List[str] = [
        "english", "spanish", "french", "german", "italian", "portuguese"
    ]
    
    # Voice Analysis
    MIN_AUDIO_DURATION: float = 5.0  # seconds
    MAX_AUDIO_DURATION: float = 300.0  # 5 minutes
    VOICE_QUALITY_THRESHOLD: float = 0.7
    
    # Synthesis
    DEFAULT_SYNTHESIS_QUALITY: str = "high"
    MAX_SYNTHESIS_QUEUE_SIZE: int = 100
    SYNTHESIS_TIMEOUT: int = 600  # 10 minutes
    
    # Performance
    MAX_CONCURRENT_TASKS: int = 10
    TASK_TIMEOUT: int = 600  # 10 minutes
    CLEANUP_INTERVAL: int = 3600  # 1 hour
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    LOG_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # Redis (for Celery)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    SESSION_TIMEOUT: int = 3600  # 1 hour
    
    # Monitoring
    ENABLE_PERFORMANCE_MONITORING: bool = True
    METRICS_RETENTION_DAYS: int = 30
    
    # Error Handling
    ERROR_RETRY_ATTEMPTS: int = 3
    ERROR_RETRY_DELAY: int = 1  # seconds
    CIRCUIT_BREAKER_THRESHOLD: int = 5
    CIRCUIT_BREAKER_TIMEOUT: int = 60  # seconds
    
    # Legacy settings for compatibility
    MAX_FILE_SIZE_MB: int = 100
    PROCESSING_TIMEOUT_SECONDS: int = 300
    TORTOISE_MODEL_PATH: str = "./models/tortoise"
    RVC_MODEL_PATH: str = "./models/rvc"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULTS_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
os.makedirs(settings.LOG_DIR, exist_ok=True)