-- Initialize database schema for UPI Fraud Detection System

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- User features table
CREATE TABLE IF NOT EXISTS user_features (
    user_id VARCHAR(32) PRIMARY KEY,
    features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Device features table
CREATE TABLE IF NOT EXISTS device_features (
    device_id VARCHAR(64) PRIMARY KEY,
    features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Location features table
CREATE TABLE IF NOT EXISTS location_features (
    location_hash VARCHAR(64) PRIMARY KEY,
    features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transaction logs table
CREATE TABLE IF NOT EXISTS transaction_logs (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) UNIQUE,
    user_id VARCHAR(32),
    amount DECIMAL(15,2),
    merchant_id VARCHAR(64),
    merchant_category VARCHAR(50),
    device_id VARCHAR(64),
    ip_address INET,
    location_lat DECIMAL(10,8),
    location_lon DECIMAL(11,8),
    risk_score FLOAT,
    fraud_probability FLOAT,
    decision VARCHAR(20),
    confidence FLOAT,
    processing_time_ms FLOAT,
    model_versions JSONB,
    explanation JSONB,
    alerts JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Case management table
CREATE TABLE IF NOT EXISTS fraud_cases (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(32) UNIQUE DEFAULT uuid_generate_v4()::text,
    transaction_id VARCHAR(64),
    status VARCHAR(20) DEFAULT 'open',
    priority VARCHAR(10) DEFAULT 'medium',
    assigned_to VARCHAR(100),
    description TEXT,
    resolution TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Model performance metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    metric_name VARCHAR(50),
    metric_value FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature drift detection table
CREATE TABLE IF NOT EXISTS feature_drift (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100),
    drift_score FLOAT,
    p_value FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_features_updated_at ON user_features(updated_at);
CREATE INDEX IF NOT EXISTS idx_device_features_updated_at ON device_features(updated_at);
CREATE INDEX IF NOT EXISTS idx_transaction_logs_created_at ON transaction_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_transaction_logs_user_id ON transaction_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_transaction_logs_decision ON transaction_logs(decision);
CREATE INDEX IF NOT EXISTS idx_transaction_logs_risk_score ON transaction_logs(risk_score);
CREATE INDEX IF NOT EXISTS idx_fraud_cases_status ON fraud_cases(status);
CREATE INDEX IF NOT EXISTS idx_fraud_cases_created_at ON fraud_cases(created_at);
CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp ON model_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_feature_drift_timestamp ON feature_drift(timestamp);

-- Create GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_user_features_features_gin ON user_features USING GIN (features);
CREATE INDEX IF NOT EXISTS idx_device_features_features_gin ON device_features USING GIN (features);
CREATE INDEX IF NOT EXISTS idx_location_features_features_gin ON location_features USING GIN (features);
CREATE INDEX IF NOT EXISTS idx_transaction_logs_explanation_gin ON transaction_logs USING GIN (explanation);

-- Create materialized view for daily fraud statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_fraud_stats AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_transactions,
    COUNT(CASE WHEN decision = 'BLOCK' THEN 1 END) as blocked_transactions,
    COUNT(CASE WHEN decision = 'CHALLENGE' THEN 1 END) as challenged_transactions,
    COUNT(CASE WHEN decision = 'ALLOW' THEN 1 END) as allowed_transactions,
    AVG(risk_score) as avg_risk_score,
    AVG(processing_time_ms) as avg_processing_time_ms,
    COUNT(CASE WHEN risk_score > 0.8 THEN 1 END) as high_risk_transactions
FROM transaction_logs
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Create function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_daily_fraud_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW daily_fraud_stats;
END;
$$ LANGUAGE plpgsql;

-- Create function to get user transaction history
CREATE OR REPLACE FUNCTION get_user_transaction_history(
    p_user_id VARCHAR(32),
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    transaction_id VARCHAR(64),
    amount DECIMAL(15,2),
    merchant_category VARCHAR(50),
    risk_score FLOAT,
    decision VARCHAR(20),
    created_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tl.transaction_id,
        tl.amount,
        tl.merchant_category,
        tl.risk_score,
        tl.decision,
        tl.created_at
    FROM transaction_logs tl
    WHERE tl.user_id = p_user_id
    AND tl.created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days
    ORDER BY tl.created_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Create function to calculate fraud rate by merchant
CREATE OR REPLACE FUNCTION get_merchant_fraud_rate(
    p_merchant_id VARCHAR(64),
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    merchant_id VARCHAR(64),
    total_transactions BIGINT,
    fraud_transactions BIGINT,
    fraud_rate FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tl.merchant_id,
        COUNT(*) as total_transactions,
        COUNT(CASE WHEN tl.decision IN ('BLOCK', 'CHALLENGE') THEN 1 END) as fraud_transactions,
        ROUND(
            COUNT(CASE WHEN tl.decision IN ('BLOCK', 'CHALLENGE') THEN 1 END)::FLOAT / 
            NULLIF(COUNT(*), 0) * 100, 2
        ) as fraud_rate
    FROM transaction_logs tl
    WHERE tl.merchant_id = p_merchant_id
    AND tl.created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days
    GROUP BY tl.merchant_id;
END;
$$ LANGUAGE plpgsql;

-- Insert sample data for testing
INSERT INTO user_features (user_id, features) VALUES 
('user123', '{"user_velocity": 2.5, "avg_amount": 1500.0, "max_amount": 5000.0, "risk_score": 0.3}'),
('user456', '{"user_velocity": 8.2, "avg_amount": 25000.0, "max_amount": 100000.0, "risk_score": 0.7}')
ON CONFLICT (user_id) DO NOTHING;

INSERT INTO device_features (device_id, features) VALUES 
('device123', '{"device_risk_score": 0.2, "device_age_days": 365, "os_version": "Android 13", "is_mobile": true}'),
('device456', '{"device_risk_score": 0.8, "device_age_days": 1, "os_version": "iOS 16", "is_mobile": true}')
ON CONFLICT (device_id) DO NOTHING;

-- Create a scheduled job to refresh materialized view (requires pg_cron extension)
-- SELECT cron.schedule('refresh-daily-stats', '0 1 * * *', 'SELECT refresh_daily_fraud_stats();');
