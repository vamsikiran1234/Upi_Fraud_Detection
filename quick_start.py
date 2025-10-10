#!/usr/bin/env python3
"""
Quick Start UPI Fraud Detection System
Simplified version that works immediately without Docker
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib
import json
import time
from typing import Dict, List, Any, Optional
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="UPI Fraud Detection API",
    description="Quick Start - Real-time fraud detection",
    version="1.0.0"
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
model = None
scaler = None
is_ready = False

# Pydantic models
class TransactionRequest(BaseModel):
    transaction_id: str
    upi_id: str
    amount: float
    merchant_id: str
    merchant_category: str
    device_id: str
    ip_address: str
    location: Dict[str, float]
    timestamp: str
    payment_method: str = "UPI"
    session_id: Optional[str] = None
    user_agent: Optional[str] = None
    sms_content: Optional[str] = None
    merchant_notes: Optional[str] = None

class FraudResponse(BaseModel):
    transaction_id: str
    risk_score: float
    fraud_probability: float
    decision: str
    confidence: float
    explanation: Dict[str, Any]
    processing_time_ms: float
    features_used: List[str]
    alerts: List[str]

def hash_upi_id(upi_id: str) -> str:
    """Hash UPI ID for privacy"""
    return hashlib.sha256(upi_id.encode()).hexdigest()[:16]

def create_demo_model():
    """Create a demo model for testing"""
    global model, scaler
    
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    X = np.random.randn(n_samples, 10)
    X[:, 0] = np.random.uniform(100, 50000, n_samples)  # amount
    X[:, 1] = np.random.randint(0, 24, n_samples)  # hour
    X[:, 2] = np.random.randint(0, 7, n_samples)  # day_of_week
    X[:, 3] = np.random.randint(0, 2, n_samples)  # is_weekend
    X[:, 4] = np.random.randint(0, 10, n_samples)  # merchant_category
    X[:, 5] = np.random.uniform(0, 1, n_samples)  # user_velocity
    X[:, 6] = np.random.uniform(0, 1, n_samples)  # device_risk_score
    X[:, 7] = np.random.uniform(0, 1, n_samples)  # location_risk_score
    X[:, 8] = np.random.uniform(0, 24, n_samples)  # time_since_last_tx
    X[:, 9] = np.random.uniform(0, 1, n_samples)  # amount_vs_avg
    
    # Create labels (fraud = 1, normal = 0)
    y = np.zeros(n_samples)
    # High amount transactions are more likely to be fraud
    y[X[:, 0] > 20000] = 1
    # Night time transactions are more likely to be fraud
    y[X[:, 1] < 6] = 1
    # High risk merchants are more likely to be fraud
    y[X[:, 4] > 7] = 1
    # Add some noise
    y[np.random.random(n_samples) < 0.1] = 1
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    print("✅ Demo model trained successfully")

def extract_features(transaction: TransactionRequest) -> np.ndarray:
    """Extract features from transaction"""
    # Basic features
    amount = transaction.amount
    hour = datetime.fromisoformat(transaction.timestamp.replace('Z', '+00:00')).hour
    day_of_week = datetime.fromisoformat(transaction.timestamp.replace('Z', '+00:00')).weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Merchant category encoding
    merchant_categories = {
        'ecommerce': 0, 'food': 1, 'transport': 2, 'entertainment': 3,
        'utilities': 4, 'healthcare': 5, 'education': 6, 'finance': 7,
        'crypto': 8, 'gambling': 9, 'adult': 10
    }
    merchant_category = merchant_categories.get(transaction.merchant_category, 0)
    
    # Simulate additional features
    user_velocity = np.random.uniform(0, 1)  # Would come from feature store
    device_risk_score = np.random.uniform(0, 1)  # Would come from feature store
    location_risk_score = np.random.uniform(0, 1)  # Would come from feature store
    time_since_last_tx = np.random.uniform(0, 24)  # Would come from feature store
    amount_vs_avg = amount / 5000  # Would come from feature store
    
    features = np.array([
        amount, hour, day_of_week, is_weekend, merchant_category,
        user_velocity, device_risk_score, location_risk_score,
        time_since_last_tx, amount_vs_avg
    ])
    
    return features.reshape(1, -1)

def generate_explanation(features: np.ndarray, prediction: float) -> Dict[str, Any]:
    """Generate explanation for prediction"""
    feature_names = [
        'amount', 'hour', 'day_of_week', 'is_weekend', 'merchant_category',
        'user_velocity', 'device_risk_score', 'location_risk_score',
        'time_since_last_tx', 'amount_vs_avg'
    ]
    
    # Simple explanation based on feature values
    explanations = []
    risk_factors = []
    
    if features[0, 0] > 20000:  # High amount
        explanations.append(f"High transaction amount (₹{features[0, 0]:,.2f}) increases fraud risk")
        risk_factors.append({
            'feature': 'amount',
            'impact': 0.3,
            'direction': 'increases',
            'severity': 'high'
        })
    
    if features[0, 1] < 6 or features[0, 1] > 22:  # Night time
        explanations.append("Transaction during night hours increases fraud risk")
        risk_factors.append({
            'feature': 'hour',
            'impact': 0.2,
            'direction': 'increases',
            'severity': 'medium'
        })
    
    if features[0, 4] > 7:  # High risk merchant
        explanations.append(f"High-risk merchant category increases fraud risk")
        risk_factors.append({
            'feature': 'merchant_category',
            'impact': 0.25,
            'direction': 'increases',
            'severity': 'high'
        })
    
    if features[0, 6] > 0.7:  # High device risk
        explanations.append("Suspicious device characteristics increase fraud risk")
        risk_factors.append({
            'feature': 'device_risk_score',
            'impact': 0.2,
            'direction': 'increases',
            'severity': 'medium'
        })
    
    if not explanations:
        explanations.append("Transaction appears normal with low fraud risk")
    
    return {
        'human_readable': '. '.join(explanations) + '.',
        'risk_factors': risk_factors,
        'feature_importance': [
            {'feature': name, 'importance': abs(np.random.random())}
            for name in feature_names
        ]
    }

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global is_ready
    print("🚀 Starting UPI Fraud Detection API...")
    create_demo_model()
    is_ready = True
    print("✅ System ready!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "UPI Fraud Detection API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if is_ready else "starting",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "model_loaded": is_ready
    }

@app.post("/predict", response_model=FraudResponse)
async def predict_fraud(transaction: TransactionRequest):
    """Predict fraud for a transaction"""
    if not is_ready:
        raise HTTPException(status_code=503, detail="System not ready")
    
    start_time = time.time()
    
    try:
        # Extract features
        features = extract_features(transaction)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        fraud_probability = model.predict_proba(features_scaled)[0][1]
        risk_score = fraud_probability
        
        # Make decision
        if risk_score > 0.8:
            decision = "BLOCK"
        elif risk_score > 0.5:
            decision = "CHALLENGE"
        else:
            decision = "ALLOW"
        
        # Generate explanation
        explanation = generate_explanation(features, risk_score)
        
        # Generate alerts
        alerts = []
        if risk_score > 0.8:
            alerts.append("High fraud risk detected")
        if transaction.amount > 50000:
            alerts.append("High amount transaction")
        if features[0, 1] < 6 or features[0, 1] > 22:
            alerts.append("Night time transaction")
        
        processing_time = (time.time() - start_time) * 1000
        
        return FraudResponse(
            transaction_id=transaction.transaction_id,
            risk_score=float(risk_score),
            fraud_probability=float(fraud_probability),
            decision=decision,
            confidence=float(min(0.95, max(0.5, 1.0 - abs(risk_score - 0.5) * 2))),
            explanation=explanation,
            processing_time_ms=processing_time,
            features_used=[
                'amount', 'hour', 'day_of_week', 'is_weekend', 'merchant_category',
                'user_velocity', 'device_risk_score', 'location_risk_score',
                'time_since_last_tx', 'amount_vs_avg'
            ],
            alerts=alerts
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models/status")
async def get_model_status():
    """Get model status"""
    return {
        "loaded": is_ready,
        "model_type": "RandomForest",
        "version": "1.0.0",
        "features": 10
    }

@app.get("/metrics")
async def get_metrics():
    """Get basic metrics"""
    return {
        "total_predictions": 0,
        "fraud_detected": 0,
        "false_positives": 0,
        "average_latency_ms": 0
    }

if __name__ == "__main__":
    print("🚀 Starting UPI Fraud Detection System...")
    print("📊 API will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
