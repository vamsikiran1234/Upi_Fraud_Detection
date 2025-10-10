#!/usr/bin/env python3
"""
Simple Backend API for UPI Fraud Detection Frontend
This is a lightweight version that works with the frontend without complex dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import time
from datetime import datetime
from typing import List, Dict, Any
import json

# Initialize FastAPI app
app = FastAPI(
    title="UPI Fraud Detection API",
    description="Simple backend API for fraud detection frontend",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TransactionRequest(BaseModel):
    transaction_id: str
    upi_id: str = None
    merchant_id: str = None
    amount: float = None
    hour: int = None
    device_risk_score: float = None
    location_risk_score: float = None
    user_behavior_score: float = None

class TransactionResponse(BaseModel):
    transaction_id: str
    risk_score: float
    risk_level: str
    factors: List[Dict[str, Any]]
    recommendation: str
    model_confidence: str
    processing_time_ms: int

class DashboardMetrics(BaseModel):
    transaction_volume: int
    fraud_rate: str
    model_accuracy: str
    response_time: str
    active_transactions: int
    fraud_detected: int

class TransactionItem(BaseModel):
    id: str
    amount: int
    merchant: str
    location: str
    status: str
    timestamp: str

# Mock data generators
def generate_risk_factors(risk_level: str) -> List[Dict[str, Any]]:
    """Generate mock risk factors based on risk level"""
    factors = {
        "low": [
            {"name": "Amount", "score": random.uniform(0.1, 0.3), "status": "safe"},
            {"name": "Location", "score": random.uniform(0.1, 0.3), "status": "safe"},
            {"name": "Merchant", "score": random.uniform(0.1, 0.3), "status": "safe"},
            {"name": "Time", "score": random.uniform(0.1, 0.3), "status": "safe"}
        ],
        "medium": [
            {"name": "Amount", "score": random.uniform(0.4, 0.6), "status": "medium"},
            {"name": "Location", "score": random.uniform(0.4, 0.6), "status": "medium"},
            {"name": "Merchant", "score": random.uniform(0.4, 0.6), "status": "medium"},
            {"name": "Time", "score": random.uniform(0.4, 0.6), "status": "medium"}
        ],
        "high": [
            {"name": "Amount", "score": random.uniform(0.7, 0.9), "status": "high"},
            {"name": "Location", "score": random.uniform(0.7, 0.9), "status": "high"},
            {"name": "Merchant", "score": random.uniform(0.7, 0.9), "status": "high"},
            {"name": "Time", "score": random.uniform(0.7, 0.9), "status": "high"}
        ]
    }
    return factors.get(risk_level, factors["low"])

def get_recommendation(risk_level: str) -> str:
    """Get recommendation based on risk level"""
    recommendations = {
        "low": "Transaction appears safe. Proceed with normal processing.",
        "medium": "Transaction shows some risk indicators. Consider additional verification.",
        "high": "High risk transaction detected. Recommend blocking or manual review."
    }
    return recommendations.get(risk_level, recommendations["low"])

def generate_mock_transactions(count: int = 10) -> List[TransactionItem]:
    """Generate mock transaction data"""
    merchants = ["Amazon", "Flipkart", "Swiggy", "Zomato", "Uber", "Ola", "Paytm", "PhonePe", "Google Pay", "BHIM"]
    locations = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
    statuses = ["safe", "risky", "fraud"]
    
    transactions = []
    for i in range(count):
        amount = random.randint(100, 100000)
        status = random.choices(statuses, weights=[70, 25, 5])[0]  # 70% safe, 25% risky, 5% fraud
        merchant = random.choice(merchants)
        location = random.choice(locations)
        
        transactions.append(TransactionItem(
            id=f"TXN{1000000 + i}",
            amount=amount,
            merchant=merchant,
            location=location,
            status=status,
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))
    
    return transactions

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "UPI Fraud Detection API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "frontend": "http://localhost:3000"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/status")
async def get_models_status():
    """Get ML models status"""
    return {
        "model_type": "Ensemble (XGBoost + LightGBM + Random Forest)",
        "training_status": "Completed",
        "feature_count": 20,
        "accuracy": random.uniform(0.92, 0.98)
    }

@app.post("/predict")
async def predict(transaction: TransactionRequest):
    """Predict fraud for a transaction"""
    start_time = time.time()
    
    # Simulate processing time
    time.sleep(random.uniform(0.01, 0.05))
    
    # Convert Pydantic model to dict for processing
    transaction_dict = transaction.dict()
    
    # Calculate risk score based on transaction data
    risk_score = 0.0
    
    # Amount-based risk
    amount = transaction_dict.get("amount", 0)
    if amount > 50000:
        risk_score += 0.3
    elif amount > 20000:
        risk_score += 0.1
    
    # Time-based risk
    hour = transaction_dict.get("hour", 12)
    if hour < 6 or hour > 22:  # Late night/early morning
        risk_score += 0.2
    
    # Device risk
    device_risk = transaction_dict.get("device_risk_score", 0.0)
    risk_score += device_risk * 0.2
    
    # Location risk
    location_risk = transaction_dict.get("location_risk_score", 0.0)
    risk_score += location_risk * 0.2
    
    # User behavior
    user_behavior = transaction_dict.get("user_behavior_score", 0.5)
    risk_score += (1 - user_behavior) * 0.2
    
    # Add some randomness
    risk_score += random.uniform(-0.05, 0.05)
    risk_score = max(0.0, min(1.0, risk_score))
    
    # Determine decision
    if risk_score >= 0.7:
        decision = "BLOCK"
        alerts = [
            "Unusual transaction amount",
            "Suspicious location detected",
            "Device risk score high"
        ]
        explanation = "High risk transaction detected due to unusual amount, location, and device risk."
    elif risk_score >= 0.4:
        decision = "CHALLENGE"
        alerts = []
        explanation = "Medium risk transaction. Additional verification recommended."
    else:
        decision = "ALLOW"
        alerts = []
        explanation = "Low risk transaction. Proceed normally."
    
    # Generate response
    processing_time = int((time.time() - start_time) * 1000)
    
    return {
        "risk_score": risk_score,
        "decision": decision,
        "confidence": random.uniform(0.85, 0.98),
        "processing_time_ms": processing_time,
        "alerts": alerts if "alerts" in locals() else [],
        "explanation": explanation if "explanation" in locals() else ""
    }

@app.get("/api/dashboard/metrics")
async def get_dashboard_metrics():
    """Get dashboard metrics"""
    return {
        "transaction_volume": random.randint(5000, 15000),
        "fraud_rate": f"{random.uniform(0.01, 0.05):.3f}%",
        "model_accuracy": f"{random.uniform(95, 99):.1f}%",
        "response_time": f"{random.randint(30, 80)}ms",
        "active_transactions": random.randint(50, 200),
        "fraud_detected": random.randint(0, 15)
    }

@app.get("/api/transactions")
async def get_transactions(limit: int = 10):
    """Get recent transactions"""
    return generate_mock_transactions(limit)

@app.get("/api/models/status")
async def get_models_status():
    """Get ML models status"""
    return {
        "models": [
            {
                "name": "XGBoost",
                "status": "active",
                "accuracy": f"{random.uniform(92, 96):.1f}%",
                "last_trained": "2024-01-15T10:30:00Z"
            },
            {
                "name": "LightGBM", 
                "status": "active",
                "accuracy": f"{random.uniform(90, 95):.1f}%",
                "last_trained": "2024-01-15T10:25:00Z"
            },
            {
                "name": "Random Forest",
                "status": "active", 
                "accuracy": f"{random.uniform(88, 93):.1f}%",
                "last_trained": "2024-01-15T10:20:00Z"
            },
            {
                "name": "Isolation Forest",
                "status": "active",
                "accuracy": f"{random.uniform(85, 90):.1f}%", 
                "last_trained": "2024-01-15T10:15:00Z"
            }
        ]
    }

@app.get("/api/alerts")
async def get_alerts():
    """Get security alerts"""
    return {
        "alerts": [
            {
                "id": 1,
                "title": "High Risk Transaction Detected",
                "description": f"Transaction TXN{random.randint(1000000, 9999999)} flagged as high risk due to unusual amount and location.",
                "level": "high",
                "timestamp": "2 minutes ago"
            },
            {
                "id": 2,
                "title": "Model Performance Degradation",
                "description": "XGBoost model accuracy dropped below 95% threshold.",
                "level": "medium", 
                "timestamp": "15 minutes ago"
            },
            {
                "id": 3,
                "title": "New Threat Intelligence Update",
                "description": f"Updated threat intelligence feed with {random.randint(20, 50)} new high-risk IP addresses.",
                "level": "low",
                "timestamp": "1 hour ago"
            }
        ]
    }

@app.get("/api/analytics/federated-learning")
async def get_federated_learning_stats():
    """Get federated learning statistics"""
    return {
        "participating_banks": random.randint(3, 8),
        "global_model_accuracy": f"{random.uniform(94, 98):.1f}%",
        "privacy_level": "High",
        "last_update": datetime.now().isoformat()
    }

@app.get("/api/analytics/blockchain")
async def get_blockchain_stats():
    """Get blockchain audit trail statistics"""
    return {
        "blocks_mined": random.randint(1000, 2000),
        "audit_records": random.randint(40000, 60000),
        "integrity_score": "100%",
        "last_block": datetime.now().isoformat()
    }

@app.get("/api/analytics/threat-intelligence")
async def get_threat_intelligence():
    """Get threat intelligence statistics"""
    return {
        "high_risk_ips": random.randint(20, 50),
        "medium_risk_ips": random.randint(100, 200),
        "low_risk_ips": random.randint(1000, 2000),
        "last_update": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting UPI Fraud Detection Backend API...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🌐 Frontend: http://localhost:3000")
    print("🔗 API Base URL: http://localhost:8000")
    print("\n" + "="*60)
    print("✅ Backend API is ready!")
    print("="*60)
    
    uvicorn.run(
        "simple_backend_api:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )
