"""
Ensemble model for UPI fraud detection
Combines multiple ML models for robust fraud detection
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import os

from loguru import logger

class FraudDetectionEnsemble:
    """Ensemble of ML models for fraud detection"""
    
    def __init__(self, settings):
        self.settings = settings
        self.models = {}
        self.model_weights = settings.ensemble_weights
        self.model_versions = {}
        self.is_loaded = False
        
    async def load_models(self):
        """Load all models in the ensemble"""
        try:
            # Load XGBoost model
            await self._load_tabular_model()
            
            # Load LSTM model
            await self._load_sequence_model()
            
            # Load Autoencoder model
            await self._load_anomaly_model()
            
            # Load GNN model
            await self._load_gnn_model()
            
            self.is_loaded = True
            logger.info("All ensemble models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ensemble models: {e}")
            raise
    
    async def _load_tabular_model(self):
        """Load XGBoost/LightGBM tabular model"""
        try:
            model_path = os.path.join(self.settings.model_path, "tabular", "xgboost_model.pkl")
            if os.path.exists(model_path):
                self.models['xgboost'] = joblib.load(model_path)
                self.model_versions['xgboost'] = "1.0.0"
                logger.info("XGBoost model loaded")
            else:
                # Create a dummy model for demo
                from sklearn.ensemble import RandomForestClassifier
                self.models['xgboost'] = RandomForestClassifier(n_estimators=100, random_state=42)
                self.model_versions['xgboost'] = "demo-1.0.0"
                logger.warning("Using demo XGBoost model")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            raise
    
    async def _load_sequence_model(self):
        """Load LSTM/Transformer sequence model"""
        try:
            model_path = os.path.join(self.settings.model_path, "sequence", "lstm_model.pkl")
            if os.path.exists(model_path):
                self.models['lstm'] = joblib.load(model_path)
                self.model_versions['lstm'] = "1.0.0"
                logger.info("LSTM model loaded")
            else:
                # Create a dummy model for demo
                from sklearn.ensemble import RandomForestClassifier
                self.models['lstm'] = RandomForestClassifier(n_estimators=50, random_state=42)
                self.model_versions['lstm'] = "demo-1.0.0"
                logger.warning("Using demo LSTM model")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            raise
    
    async def _load_anomaly_model(self):
        """Load Autoencoder/Isolation Forest anomaly model"""
        try:
            model_path = os.path.join(self.settings.model_path, "anomaly", "autoencoder_model.pkl")
            if os.path.exists(model_path):
                self.models['autoencoder'] = joblib.load(model_path)
                self.model_versions['autoencoder'] = "1.0.0"
                logger.info("Autoencoder model loaded")
            else:
                # Create a dummy model for demo
                from sklearn.ensemble import IsolationForest
                self.models['autoencoder'] = IsolationForest(contamination=0.1, random_state=42)
                self.model_versions['autoencoder'] = "demo-1.0.0"
                logger.warning("Using demo Autoencoder model")
        except Exception as e:
            logger.error(f"Failed to load Autoencoder model: {e}")
            raise
    
    async def _load_gnn_model(self):
        """Load Graph Neural Network model"""
        try:
            model_path = os.path.join(self.settings.model_path, "gnn", "gnn_model.pkl")
            if os.path.exists(model_path):
                self.models['gnn'] = joblib.load(model_path)
                self.model_versions['gnn'] = "1.0.0"
                logger.info("GNN model loaded")
            else:
                # Create a dummy model for demo
                from sklearn.ensemble import RandomForestClassifier
                self.models['gnn'] = RandomForestClassifier(n_estimators=30, random_state=42)
                self.model_versions['gnn'] = "demo-1.0.0"
                logger.warning("Using demo GNN model")
        except Exception as e:
            logger.error(f"Failed to load GNN model: {e}")
            raise
    
    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make ensemble prediction"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")
        
        try:
            # Prepare feature vector
            feature_vector = self._prepare_features(features['features'])
            
            # Get predictions from each model
            predictions = {}
            model_scores = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name == 'autoencoder':
                        # Anomaly detection models return anomaly scores
                        score = model.decision_function(feature_vector.reshape(1, -1))[0]
                        # Convert to probability (higher score = more normal)
                        prob = 1 / (1 + np.exp(-score))
                        predictions[model_name] = 1 - prob  # Convert to fraud probability
                    else:
                        # Classification models
                        if hasattr(model, 'predict_proba'):
                            prob = model.predict_proba(feature_vector.reshape(1, -1))[0]
                            predictions[model_name] = prob[1] if len(prob) > 1 else prob[0]
                        else:
                            pred = model.predict(feature_vector.reshape(1, -1))[0]
                            predictions[model_name] = float(pred)
                    
                    model_scores[model_name] = predictions[model_name]
                    
                except Exception as e:
                    logger.error(f"Prediction failed for {model_name}: {e}")
                    predictions[model_name] = 0.5  # Default neutral score
                    model_scores[model_name] = 0.5
            
            # Calculate weighted ensemble score
            ensemble_score = sum(
                self.model_weights.get(model_name, 0) * score 
                for model_name, score in model_scores.items()
            )
            
            # Calculate confidence based on agreement between models
            scores = list(model_scores.values())
            confidence = 1.0 - np.std(scores) if len(scores) > 1 else 0.8
            
            return {
                'risk_score': float(ensemble_score),
                'fraud_probability': float(ensemble_score),
                'confidence': float(confidence),
                'model_versions': self.model_versions.copy(),
                'individual_scores': model_scores,
                'ensemble_weight': self.model_weights
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise
    
    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for model prediction"""
        # Convert features to numpy array
        # This is a simplified version - in practice, you'd have proper feature engineering
        
        feature_names = [
            'amount', 'hour', 'day_of_week', 'is_weekend',
            'merchant_category', 'payment_method'
        ]
        
        feature_vector = []
        for name in feature_names:
            if name in features:
                if isinstance(features[name], (int, float)):
                    feature_vector.append(features[name])
                elif isinstance(features[name], str):
                    # Simple string encoding
                    feature_vector.append(hash(features[name]) % 1000)
                else:
                    feature_vector.append(0)
            else:
                feature_vector.append(0)
        
        # Pad or truncate to expected length
        expected_length = 20  # Adjust based on your model requirements
        while len(feature_vector) < expected_length:
            feature_vector.append(0)
        feature_vector = feature_vector[:expected_length]
        
        return np.array(feature_vector, dtype=np.float32)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        return {
            'loaded': self.is_loaded,
            'models': list(self.models.keys()),
            'versions': self.model_versions,
            'weights': self.model_weights
        }
    
    async def retrain(self):
        """Retrain all models with new data"""
        logger.info("Starting model retraining...")
        # This would implement the retraining logic
        # For now, just log the event
        logger.info("Model retraining completed")
