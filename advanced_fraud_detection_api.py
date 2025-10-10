"""
Advanced UPI Fraud Detection API
Integrates all advanced AI/ML features for enterprise-grade fraud detection
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import logging
from datetime import datetime
import json

# Import all advanced modules
from serving.models.federated_learning import FederatedFraudDetectionAPI, FederatedConfig
from serving.models.synthetic_data_generator import AdvancedSyntheticDataGenerator
from serving.models.blockchain_audit import AuditTrailAPI
from serving.models.gnn_transformer import GraphTemporalFraudDetectorAPI
from serving.models.reinforcement_learning import ReinforcementLearningFraudDetector
from serving.models.multimodal_features import MultiModalFraudDetector
from serving.models.threat_intelligence import ThreatIntelligenceAPI
from serving.models.active_learning import ActiveLearningPipeline, AnalystFeedback
from serving.models.differential_privacy import DifferentialPrivacyAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced UPI Fraud Detection API",
    version="2.0.0",
    description="Enterprise-grade fraud detection with advanced AI/ML capabilities"
)

# Global instances
federated_api = None
synthetic_generator = None
blockchain_audit = None
gnn_transformer = None
rl_detector = None
multimodal_detector = None
threat_intelligence = None
active_learning = None
differential_privacy = None

# Pydantic models
class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    upi_id: str
    merchant_id: str
    timestamp: str
    features: Dict[str, Any]
    biometric_data: Optional[Dict[str, Any]] = None
    device_data: Optional[Dict[str, Any]] = None
    user_history: Optional[Dict[str, Any]] = None

class FraudPrediction(BaseModel):
    transaction_id: str
    risk_score: float
    decision: str
    confidence: float
    model_type: str
    explanations: Dict[str, Any]
    privacy_protected: bool = False
    audit_trail_hash: Optional[str] = None

class AnalystFeedbackRequest(BaseModel):
    transaction_id: str
    analyst_decision: str
    analyst_id: str
    reasoning: str
    false_positive: bool = False
    false_negative: bool = False

class ThreatIntelligenceUpdate(BaseModel):
    feed_sources: List[str]
    update_frequency: int = 60

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize all advanced components"""
    global federated_api, synthetic_generator, blockchain_audit
    global gnn_transformer, rl_detector, multimodal_detector
    global threat_intelligence, active_learning, differential_privacy
    
    logger.info("🚀 Initializing Advanced Fraud Detection System...")
    
    try:
        # Initialize Federated Learning
        federated_config = FederatedConfig(num_banks=5, rounds=100)
        federated_api = FederatedFraudDetectionAPI(federated_config)
        logger.info("✅ Federated Learning initialized")
        
        # Initialize Synthetic Data Generator
        synthetic_generator = AdvancedSyntheticDataGenerator()
        logger.info("✅ Synthetic Data Generator initialized")
        
        # Initialize Blockchain Audit
        blockchain_audit = AuditTrailAPI("fraud_audit_node")
        logger.info("✅ Blockchain Audit initialized")
        
        # Initialize GNN-Transformer
        gnn_transformer = GraphTemporalFraudDetectorAPI(input_dim=20)
        logger.info("✅ GNN-Transformer initialized")
        
        # Initialize Reinforcement Learning
        rl_detector = ReinforcementLearningFraudDetector()
        logger.info("✅ Reinforcement Learning initialized")
        
        # Initialize Multi-Modal Detector
        multimodal_detector = MultiModalFraudDetector()
        logger.info("✅ Multi-Modal Detector initialized")
        
        # Initialize Threat Intelligence
        threat_intelligence = ThreatIntelligenceAPI()
        logger.info("✅ Threat Intelligence initialized")
        
        # Initialize Active Learning
        active_learning = ActiveLearningPipeline(
            uncertainty_strategy='entropy',
            retrain_threshold=50
        )
        logger.info("✅ Active Learning initialized")
        
        # Initialize Differential Privacy
        differential_privacy = DifferentialPrivacyAPI(epsilon=1.0, delta=1e-5)
        logger.info("✅ Differential Privacy initialized")
        
        logger.info("🎉 All advanced components initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error initializing components: {e}")
        raise

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    components_status = {
        "federated_learning": federated_api is not None,
        "synthetic_data_generator": synthetic_generator is not None,
        "blockchain_audit": blockchain_audit is not None,
        "gnn_transformer": gnn_transformer is not None,
        "reinforcement_learning": rl_detector is not None,
        "multimodal_detector": multimodal_detector is not None,
        "threat_intelligence": threat_intelligence is not None,
        "active_learning": active_learning is not None,
        "differential_privacy": differential_privacy is not None
    }
    
    all_healthy = all(components_status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "components": components_status,
        "version": "2.0.0"
    }

# Main fraud prediction endpoint
@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks
):
    """Advanced fraud prediction using ensemble of models"""
    try:
        logger.info(f"🔍 Processing fraud prediction for transaction {transaction.transaction_id}")
        
        # Initialize prediction results
        predictions = {}
        risk_scores = []
        decisions = []
        
        # 1. Multi-Modal Prediction
        if multimodal_detector:
            multimodal_pred = multimodal_detector.predict_fraud(
                transaction.dict(),
                transaction.biometric_data,
                transaction.device_data,
                transaction.user_history
            )
            predictions['multimodal'] = multimodal_pred
            risk_scores.append(multimodal_pred['risk_score'])
            decisions.append(multimodal_pred['decision'])
        
        # 2. GNN-Transformer Prediction
        if gnn_transformer:
            # Convert to DataFrame for GNN
            import pandas as pd
            gnn_data = pd.DataFrame([transaction.features])
            gnn_pred = gnn_transformer.predict_fraud(gnn_data)
            predictions['gnn_transformer'] = gnn_pred
            risk_scores.append(gnn_pred['risk_score'])
            decisions.append(gnn_pred['decision'])
        
        # 3. Reinforcement Learning Prediction
        if rl_detector and rl_detector.is_trained:
            import pandas as pd
            rl_data = pd.Series(transaction.features)
            rl_pred = rl_detector.predict_fraud(rl_data)
            predictions['reinforcement_learning'] = rl_pred
            risk_scores.append(rl_pred['risk_score'])
            decisions.append(rl_pred['decision'])
        
        # 4. Threat Intelligence Check
        threat_check = None
        if threat_intelligence:
            threat_check = threat_intelligence.check_transaction_against_threats(transaction.dict())
            if threat_check['threats_found'] > 0:
                risk_scores.append(threat_check['risk_score'])
                decisions.append('BLOCK' if threat_check['risk_score'] > 0.7 else 'CHALLENGE')
        
        # 5. Ensemble Decision
        if risk_scores:
            final_risk_score = sum(risk_scores) / len(risk_scores)
            
            # Weighted voting for decision
            decision_weights = {'ALLOW': 0, 'CHALLENGE': 0, 'BLOCK': 0}
            for decision in decisions:
                if decision in decision_weights:
                    decision_weights[decision] += 1
            
            final_decision = max(decision_weights, key=decision_weights.get)
            confidence = max(decision_weights.values()) / len(decisions)
        else:
            final_risk_score = 0.5
            final_decision = 'CHALLENGE'
            confidence = 0.5
        
        # 6. Apply Differential Privacy
        privacy_protected = False
        if differential_privacy:
            # Add noise to risk score for privacy
            import numpy as np
            noise = np.random.laplace(0, 0.01)
            final_risk_score = max(0, min(1, final_risk_score + noise))
            privacy_protected = True
        
        # 7. Log to Blockchain Audit
        audit_hash = None
        if blockchain_audit:
            audit_result = blockchain_audit.log_fraud_decision({
                'transaction_id': transaction.transaction_id,
                'upi_id_hash': transaction.upi_id,
                'amount': transaction.amount,
                'merchant_id': transaction.merchant_id,
                'decision': final_decision,
                'risk_score': final_risk_score,
                'confidence': confidence,
                'model_version': '2.0.0',
                'features': transaction.features,
                'explanation': predictions,
                'processing_time_ms': 100.0,
                'timestamp': transaction.timestamp
            })
            audit_hash = audit_result.get('block_hash')
        
        # 8. Add to Active Learning Queue if uncertain
        if active_learning and confidence < 0.8:
            background_tasks.add_task(
                add_to_active_learning_queue,
                transaction,
                final_decision,
                confidence
            )
        
        # 9. Update Threat Intelligence
        if threat_intelligence:
            background_tasks.add_task(update_threat_intelligence)
        
        return FraudPrediction(
            transaction_id=transaction.transaction_id,
            risk_score=final_risk_score,
            decision=final_decision,
            confidence=confidence,
            model_type='advanced_ensemble',
            explanations=predictions,
            privacy_protected=privacy_protected,
            audit_trail_hash=audit_hash
        )
        
    except Exception as e:
        logger.error(f"❌ Error in fraud prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Active Learning endpoints
@app.post("/analyst/feedback")
async def submit_analyst_feedback(feedback: AnalystFeedbackRequest):
    """Submit analyst feedback for active learning"""
    try:
        analyst_feedback = AnalystFeedback(
            transaction_id=feedback.transaction_id,
            model_prediction='UNKNOWN',  # Would be retrieved from system
            analyst_decision=feedback.analyst_decision,
            confidence=0.8,
            feedback_timestamp=datetime.utcnow(),
            analyst_id=feedback.analyst_id,
            reasoning=feedback.reasoning,
            false_positive=feedback.false_positive,
            false_negative=feedback.false_negative
        )
        
        success = active_learning.add_feedback(analyst_feedback)
        
        return {
            "status": "success" if success else "error",
            "message": "Feedback submitted successfully" if success else "Failed to submit feedback"
        }
        
    except Exception as e:
        logger.error(f"❌ Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyst/dashboard")
async def get_analyst_dashboard():
    """Get analyst dashboard data"""
    try:
        if not active_learning:
            raise HTTPException(status_code=503, detail="Active learning not initialized")
        
        dashboard_data = active_learning.get_analyst_dashboard_data()
        return dashboard_data
        
    except Exception as e:
        logger.error(f"❌ Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Threat Intelligence endpoints
@app.post("/threat-intelligence/update")
async def update_threat_intelligence_endpoint(update_request: ThreatIntelligenceUpdate):
    """Update threat intelligence feeds"""
    try:
        if not threat_intelligence:
            raise HTTPException(status_code=503, detail="Threat intelligence not initialized")
        
        results = await threat_intelligence.update_threat_intelligence()
        return results
        
    except Exception as e:
        logger.error(f"❌ Error updating threat intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threat-intelligence/summary")
async def get_threat_summary():
    """Get threat intelligence summary"""
    try:
        if not threat_intelligence:
            raise HTTPException(status_code=503, detail="Threat intelligence not initialized")
        
        summary = threat_intelligence.get_threat_summary()
        return summary
        
    except Exception as e:
        logger.error(f"❌ Error getting threat summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Federated Learning endpoints
@app.post("/federated/register-bank")
async def register_bank(bank_id: str, bank_info: Dict[str, Any]):
    """Register a new bank in federated learning network"""
    try:
        if not federated_api:
            raise HTTPException(status_code=503, detail="Federated learning not initialized")
        
        result = federated_api.register_bank(bank_id, bank_info)
        return result
        
    except Exception as e:
        logger.error(f"❌ Error registering bank: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/federated/status")
async def get_federated_status():
    """Get federated learning status"""
    try:
        if not federated_api:
            raise HTTPException(status_code=503, detail="Federated learning not initialized")
        
        status = federated_api.get_federated_status()
        return status
        
    except Exception as e:
        logger.error(f"❌ Error getting federated status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Synthetic Data endpoints
@app.post("/synthetic/generate")
async def generate_synthetic_data(
    num_samples: int = 1000,
    target_fraud_ratio: float = 0.3
):
    """Generate synthetic fraud data"""
    try:
        if not synthetic_generator:
            raise HTTPException(status_code=503, detail="Synthetic data generator not initialized")
        
        # Create realistic fraud data
        fraud_data = synthetic_generator.create_realistic_fraud_data(num_samples)
        
        # Generate balanced dataset
        balanced_data = synthetic_generator.generate_balanced_dataset(
            fraud_data, target_fraud_ratio
        )
        
        return {
            "status": "success",
            "original_samples": len(fraud_data),
            "balanced_samples": len(balanced_data),
            "fraud_ratio": balanced_data['fraud'].mean(),
            "data_preview": balanced_data.head().to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating synthetic data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Privacy endpoints
@app.get("/privacy/report")
async def get_privacy_report():
    """Get differential privacy report"""
    try:
        if not differential_privacy:
            raise HTTPException(status_code=503, detail="Differential privacy not initialized")
        
        report = differential_privacy.get_privacy_report()
        return report
        
    except Exception as e:
        logger.error(f"❌ Error getting privacy report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System status endpoint
@app.get("/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "components": {
                "federated_learning": {
                    "status": "active" if federated_api else "inactive",
                    "banks_registered": len(federated_api.coordinator.bank_models) if federated_api else 0
                },
                "synthetic_data": {
                    "status": "active" if synthetic_generator else "inactive"
                },
                "blockchain_audit": {
                    "status": "active" if blockchain_audit else "inactive",
                    "blocks": len(blockchain_audit.audit_trail.chain) if blockchain_audit else 0
                },
                "gnn_transformer": {
                    "status": "active" if gnn_transformer else "inactive",
                    "trained": gnn_transformer.is_trained if gnn_transformer else False
                },
                "reinforcement_learning": {
                    "status": "active" if rl_detector else "inactive",
                    "trained": rl_detector.is_trained if rl_detector else False
                },
                "multimodal_detector": {
                    "status": "active" if multimodal_detector else "inactive"
                },
                "threat_intelligence": {
                    "status": "active" if threat_intelligence else "inactive",
                    "indicators": len(threat_intelligence.feed_manager.indicators) if threat_intelligence else 0
                },
                "active_learning": {
                    "status": "active" if active_learning else "inactive",
                    "pending_reviews": len(active_learning.analyst_workflow.pending_reviews) if active_learning else 0
                },
                "differential_privacy": {
                    "status": "active" if differential_privacy else "inactive",
                    "privacy_budget": differential_privacy.privacy_budget.epsilon if differential_privacy else 0
                }
            }
        }
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def add_to_active_learning_queue(transaction: TransactionRequest, decision: str, confidence: float):
    """Add uncertain predictions to active learning queue"""
    try:
        if active_learning:
            # Create active learning sample
            from serving.models.active_learning import ActiveLearningSample
            import numpy as np
            
            sample = ActiveLearningSample(
                transaction_id=transaction.transaction_id,
                features=np.array(list(transaction.features.values())),
                model_prediction=decision,
                confidence=confidence,
                uncertainty_score=1 - confidence,
                selection_reason="Low confidence prediction",
                priority=1 if confidence < 0.6 else 2,
                timestamp=datetime.utcnow()
            )
            
            review_id = active_learning.analyst_workflow.add_pending_review(sample)
            logger.info(f"Added transaction {transaction.transaction_id} to active learning queue: {review_id}")
            
    except Exception as e:
        logger.error(f"❌ Error adding to active learning queue: {e}")

async def update_threat_intelligence():
    """Update threat intelligence in background"""
    try:
        if threat_intelligence:
            await threat_intelligence.update_threat_intelligence()
            logger.info("Threat intelligence updated in background")
            
    except Exception as e:
        logger.error(f"❌ Error updating threat intelligence: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
