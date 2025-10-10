"""
UPI Fraud Detection - FastAPI Inference Service
Main API endpoint for real-time fraud detection with explainability
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import joblib
import shap
from loguru import logger

from .models.ensemble import FraudDetectionEnsemble
from .models.explainability import ExplainabilityEngine
from .feature_store import FeatureStore
from .decision_engine import DecisionEngine
from .config import Settings

# Initialize FastAPI app
app = FastAPI(
    title="UPI Fraud Detection API",
    description="Real-time fraud detection with explainable AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
settings = Settings()
feature_store = None
ensemble_model = None
explainability_engine = None
decision_engine = None

# Pydantic models
class TransactionRequest(BaseModel):
    """Transaction data for fraud detection"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    upi_id: str = Field(..., description="UPI ID (will be hashed for privacy)")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(..., description="Merchant category code")
    device_id: str = Field(..., description="Device fingerprint")
    ip_address: str = Field(..., description="IP address")
    location: Dict[str, float] = Field(..., description="GPS coordinates")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    payment_method: str = Field(default="UPI", description="Payment method")
    session_id: Optional[str] = Field(None, description="User session ID")
    user_agent: Optional[str] = Field(None, description="Browser/app user agent")
    sms_content: Optional[str] = Field(None, description="SMS content for analysis")
    merchant_notes: Optional[str] = Field(None, description="Merchant notes")

class FraudResponse(BaseModel):
    """Fraud detection response"""
    transaction_id: str
    risk_score: float = Field(..., ge=0, le=1, description="Risk score (0-1)")
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability")
    decision: str = Field(..., description="Decision: ALLOW, CHALLENGE, BLOCK")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    explanation: Dict[str, Any] = Field(..., description="SHAP explanation")
    model_versions: Dict[str, str] = Field(..., description="Model versions used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    features_used: List[str] = Field(..., description="Features used for prediction")
    alerts: List[str] = Field(default=[], description="Generated alerts")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    models_loaded: bool
    feature_store_connected: bool
    redis_connected: bool
    postgres_connected: bool

def hash_upi_id(upi_id: str) -> str:
    """Hash UPI ID for privacy protection"""
    return hashlib.sha256(upi_id.encode()).hexdigest()[:16]

def get_feature_store():
    """Dependency to get feature store instance"""
    return feature_store

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global feature_store, ensemble_model, explainability_engine, decision_engine
    
    logger.info("Starting UPI Fraud Detection API...")
    
    try:
        # Initialize feature store
        feature_store = FeatureStore(settings)
        await feature_store.initialize()
        logger.info("Feature store initialized")
        
        # Initialize ensemble model
        ensemble_model = FraudDetectionEnsemble(settings)
        await ensemble_model.load_models()
        logger.info("Ensemble model loaded")
        
        # Initialize explainability engine
        explainability_engine = ExplainabilityEngine(settings)
        logger.info("Explainability engine initialized")
        
        # Initialize decision engine
        decision_engine = DecisionEngine(settings)
        logger.info("Decision engine initialized")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check feature store connection
        fs_connected = feature_store and await feature_store.is_healthy()
        
        # Check Redis connection
        redis_connected = False
        try:
            r = redis.Redis(host=settings.redis_host, port=settings.redis_port, db=0)
            redis_connected = r.ping()
        except:
            pass
        
        # Check PostgreSQL connection
        postgres_connected = False
        try:
            conn = psycopg2.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                database=settings.postgres_db,
                user=settings.postgres_user,
                password=settings.postgres_password
            )
            postgres_connected = True
            conn.close()
        except:
            pass
        
        return HealthResponse(
            status="healthy" if all([fs_connected, redis_connected, postgres_connected]) else "degraded",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            models_loaded=ensemble_model is not None,
            feature_store_connected=fs_connected,
            redis_connected=redis_connected,
            postgres_connected=postgres_connected
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            models_loaded=False,
            feature_store_connected=False,
            redis_connected=False,
            postgres_connected=False
        )

@app.post("/predict", response_model=FraudResponse)
async def predict_fraud(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks,
    fs: FeatureStore = Depends(get_feature_store)
):
    """Predict fraud for a transaction"""
    start_time = time.time()
    
    try:
        # Hash UPI ID for privacy
        hashed_upi_id = hash_upi_id(transaction.upi_id)
        
        # Extract features
        features = await extract_features(transaction, fs)
        
        # Get ensemble prediction
        prediction = await ensemble_model.predict(features)
        
        # Generate explanation
        explanation = await explainability_engine.explain(features, prediction)
        
        # Make decision
        decision_result = decision_engine.decide(prediction, features, explanation)
        
        # Log transaction for monitoring
        background_tasks.add_task(
            log_transaction,
            transaction.transaction_id,
            prediction,
            decision_result,
            time.time() - start_time
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return FraudResponse(
            transaction_id=transaction.transaction_id,
            risk_score=prediction['risk_score'],
            fraud_probability=prediction['fraud_probability'],
            decision=decision_result['decision'],
            confidence=prediction['confidence'],
            explanation=explanation,
            model_versions=prediction['model_versions'],
            processing_time_ms=processing_time,
            features_used=features['feature_names'],
            alerts=decision_result['alerts']
        )
        
    except Exception as e:
        logger.error(f"Prediction failed for transaction {transaction.transaction_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

async def extract_features(transaction: TransactionRequest, fs: FeatureStore) -> Dict[str, Any]:
    """Extract features for fraud detection"""
    # Basic transaction features
    features = {
        'amount': transaction.amount,
        'hour': transaction.timestamp.hour,
        'day_of_week': transaction.timestamp.weekday(),
        'is_weekend': transaction.timestamp.weekday() >= 5,
        'merchant_category': transaction.merchant_category,
        'payment_method': transaction.payment_method,
    }
    
    # Get historical features from feature store
    historical_features = await fs.get_user_features(
        hash_upi_id(transaction.upi_id),
        transaction.timestamp
    )
    features.update(historical_features)
    
    # Get device features
    device_features = await fs.get_device_features(
        transaction.device_id,
        transaction.timestamp
    )
    features.update(device_features)
    
    # Get location features
    location_features = await fs.get_location_features(
        transaction.location,
        transaction.timestamp
    )
    features.update(location_features)
    
    return {
        'features': features,
        'feature_names': list(features.keys())
    }

async def log_transaction(transaction_id: str, prediction: Dict, decision: Dict, processing_time: float):
    """Log transaction for monitoring and analysis"""
    try:
        # Log to PostgreSQL for analytics
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password
        )
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO transaction_logs 
                (transaction_id, risk_score, fraud_probability, decision, 
                 confidence, processing_time_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                transaction_id,
                prediction['risk_score'],
                prediction['fraud_probability'],
                decision['decision'],
                prediction['confidence'],
                processing_time * 1000,
                datetime.utcnow()
            ))
            conn.commit()
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to log transaction {transaction_id}: {e}")

@app.get("/models/status")
async def get_model_status():
    """Get status of all models in the ensemble"""
    if not ensemble_model:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return await ensemble_model.get_status()

@app.post("/models/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    background_tasks.add_task(ensemble_model.retrain)
    return {"message": "Retraining triggered"}

@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics"""
    # This would integrate with Prometheus metrics
    return {
        "total_predictions": 0,  # Would be populated from metrics
        "fraud_detected": 0,
        "false_positives": 0,
        "average_latency_ms": 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
