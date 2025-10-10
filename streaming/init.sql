-- Initialize database for streaming pipeline
CREATE TABLE IF NOT EXISTS transaction_logs (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    risk_score DECIMAL(5,4) NOT NULL,
    fraud_probability DECIMAL(5,4) NOT NULL,
    decision VARCHAR(20) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    processing_time_ms DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_features (
    id SERIAL PRIMARY KEY,
    user_hash VARCHAR(64) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    UNIQUE(user_hash, feature_name)
);

CREATE TABLE IF NOT EXISTS device_features (
    id SERIAL PRIMARY KEY,
    device_id VARCHAR(255) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    UNIQUE(device_id, feature_name)
);

CREATE TABLE IF NOT EXISTS streaming_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tags JSONB
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_transaction_logs_created_at ON transaction_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_user_features_hash ON user_features(user_hash);
CREATE INDEX IF NOT EXISTS idx_user_features_expires ON user_features(expires_at);
CREATE INDEX IF NOT EXISTS idx_device_features_device_id ON device_features(device_id);
CREATE INDEX IF NOT EXISTS idx_streaming_metrics_name_time ON streaming_metrics(metric_name, timestamp);
