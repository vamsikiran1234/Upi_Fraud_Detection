"""
Configuration settings for the UPI Fraud Detection API
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    app_name: str = "UPI Fraud Detection API"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database Settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "fraud_detection"
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    
    # Redis Settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Kafka Settings
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_transactions: str = "raw-transactions"
    kafka_topic_preprocessed: str = "preprocessed"
    kafka_topic_alerts: str = "alerts"
    kafka_topic_labels: str = "labels"
    
    # Model Settings
    model_path: str = "models/"
    ensemble_weights: dict = {
        "xgboost": 0.4,
        "lstm": 0.3,
        "autoencoder": 0.2,
        "gnn": 0.1
    }
    
    # Feature Store Settings
    feature_store_type: str = "redis_postgres"  # or "feast"
    feature_cache_ttl: int = 3600  # 1 hour
    
    # Security Settings
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Monitoring Settings
    prometheus_port: int = 9090
    log_level: str = "INFO"
    
    # Decision Engine Settings
    risk_threshold_high: float = 0.8
    risk_threshold_medium: float = 0.5
    risk_threshold_low: float = 0.2
    
    # Rate Limiting
    rate_limit_per_minute: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = False
