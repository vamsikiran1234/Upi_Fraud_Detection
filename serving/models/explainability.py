"""
Explainability engine for fraud detection models
Provides SHAP explanations and human-readable insights
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Optional
import logging
from loguru import logger

class ExplainabilityEngine:
    """Engine for generating model explanations"""
    
    def __init__(self, settings):
        self.settings = settings
        self.explainers = {}
        self.feature_names = [
            'amount', 'hour', 'day_of_week', 'is_weekend',
            'merchant_category', 'payment_method', 'user_velocity',
            'device_risk_score', 'location_risk_score', 'time_since_last_tx',
            'amount_vs_avg', 'merchant_frequency', 'session_duration',
            'ip_reputation', 'device_age', 'location_consistency',
            'payment_pattern', 'merchant_risk', 'time_pattern',
            'amount_pattern'
        ]
    
    async def explain(self, features: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for a prediction"""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_features(features['features'])
            
            # Generate SHAP explanation
            shap_explanation = await self._generate_shap_explanation(
                feature_vector, prediction
            )
            
            # Generate human-readable explanation
            human_explanation = await self._generate_human_explanation(
                features['features'], prediction, shap_explanation
            )
            
            # Generate risk factors
            risk_factors = await self._identify_risk_factors(
                features['features'], shap_explanation
            )
            
            return {
                'shap_values': shap_explanation,
                'human_readable': human_explanation,
                'risk_factors': risk_factors,
                'feature_importance': self._get_feature_importance(shap_explanation),
                'confidence_explanation': self._explain_confidence(prediction)
            }
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return self._get_default_explanation()
    
    async def _generate_shap_explanation(self, feature_vector: np.ndarray, prediction: Dict) -> Dict[str, Any]:
        """Generate SHAP explanation"""
        try:
            # For demo purposes, create mock SHAP values
            # In production, you'd use actual SHAP explainers for each model
            
            n_features = len(feature_vector)
            shap_values = np.random.normal(0, 0.1, n_features)
            
            # Adjust SHAP values based on feature values
            for i, value in enumerate(feature_vector):
                if i == 0 and value > 10000:  # High amount
                    shap_values[i] += 0.3
                elif i == 1 and value in [22, 23, 0, 1, 2]:  # Night time
                    shap_values[i] += 0.2
                elif i == 4 and value in ['gambling', 'adult', 'crypto']:  # High-risk merchant
                    shap_values[i] += 0.4
            
            return {
                'values': shap_values.tolist(),
                'base_value': 0.5,
                'data': feature_vector.tolist(),
                'feature_names': self.feature_names[:n_features]
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'values': [0] * len(feature_vector), 'base_value': 0.5}
    
    async def _generate_human_explanation(self, features: Dict, prediction: Dict, shap_explanation: Dict) -> str:
        """Generate human-readable explanation"""
        try:
            risk_score = prediction['risk_score']
            explanations = []
            
            # Amount-based explanation
            amount = features.get('amount', 0)
            if amount > 50000:
                explanations.append(f"High transaction amount (₹{amount:,.2f}) increases fraud risk")
            elif amount < 100:
                explanations.append(f"Low transaction amount (₹{amount:,.2f}) reduces fraud risk")
            
            # Time-based explanation
            hour = features.get('hour', 12)
            if hour in [22, 23, 0, 1, 2, 3, 4, 5]:
                explanations.append("Transaction during night hours increases fraud risk")
            elif 9 <= hour <= 17:
                explanations.append("Transaction during business hours reduces fraud risk")
            
            # Merchant category explanation
            merchant_category = features.get('merchant_category', '')
            if merchant_category in ['gambling', 'adult', 'crypto']:
                explanations.append(f"High-risk merchant category ({merchant_category}) increases fraud risk")
            elif merchant_category in ['grocery', 'pharmacy', 'utilities']:
                explanations.append(f"Low-risk merchant category ({merchant_category}) reduces fraud risk")
            
            # User behavior explanation
            user_velocity = features.get('user_velocity', 0)
            if user_velocity > 10:
                explanations.append(f"High transaction frequency ({user_velocity} transactions) increases fraud risk")
            
            # Device risk explanation
            device_risk = features.get('device_risk_score', 0)
            if device_risk > 0.7:
                explanations.append("Suspicious device characteristics increase fraud risk")
            elif device_risk < 0.3:
                explanations.append("Trusted device characteristics reduce fraud risk")
            
            # Overall risk assessment
            if risk_score > 0.8:
                explanations.append("Overall high fraud risk based on multiple risk factors")
            elif risk_score < 0.3:
                explanations.append("Overall low fraud risk based on transaction characteristics")
            else:
                explanations.append("Moderate fraud risk requiring additional verification")
            
            return ". ".join(explanations) + "."
            
        except Exception as e:
            logger.error(f"Human explanation failed: {e}")
            return "Unable to generate explanation due to technical error."
    
    async def _identify_risk_factors(self, features: Dict, shap_explanation: Dict) -> List[Dict[str, Any]]:
        """Identify key risk factors"""
        try:
            risk_factors = []
            shap_values = shap_explanation.get('values', [])
            feature_names = shap_explanation.get('feature_names', [])
            
            # Sort features by SHAP value (impact on prediction)
            feature_impacts = list(zip(feature_names, shap_values))
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feature_name, impact in feature_impacts[:5]:  # Top 5 factors
                if abs(impact) > 0.1:  # Significant impact
                    risk_factors.append({
                        'feature': feature_name,
                        'impact': float(impact),
                        'direction': 'increases' if impact > 0 else 'reduces',
                        'severity': 'high' if abs(impact) > 0.3 else 'medium' if abs(impact) > 0.2 else 'low'
                    })
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Risk factor identification failed: {e}")
            return []
    
    def _get_feature_importance(self, shap_explanation: Dict) -> List[Dict[str, Any]]:
        """Get feature importance ranking"""
        try:
            shap_values = shap_explanation.get('values', [])
            feature_names = shap_explanation.get('feature_names', [])
            
            importance = []
            for name, value in zip(feature_names, shap_values):
                importance.append({
                    'feature': name,
                    'importance': abs(value),
                    'impact': value
                })
            
            # Sort by importance
            importance.sort(key=lambda x: x['importance'], reverse=True)
            return importance
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return []
    
    def _explain_confidence(self, prediction: Dict) -> str:
        """Explain model confidence"""
        try:
            confidence = prediction.get('confidence', 0.5)
            
            if confidence > 0.8:
                return "High confidence due to clear patterns in transaction data"
            elif confidence > 0.6:
                return "Moderate confidence with some uncertainty in the prediction"
            else:
                return "Low confidence due to conflicting signals or insufficient data"
                
        except Exception as e:
            logger.error(f"Confidence explanation failed: {e}")
            return "Unable to assess confidence level"
    
    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for explanation"""
        # Convert features to numpy array
        feature_vector = []
        for name in self.feature_names:
            if name in features:
                if isinstance(features[name], (int, float)):
                    feature_vector.append(features[name])
                elif isinstance(features[name], str):
                    feature_vector.append(hash(features[name]) % 1000)
                else:
                    feature_vector.append(0)
            else:
                feature_vector.append(0)
        
        return np.array(feature_vector, dtype=np.float32)
    
    def _get_default_explanation(self) -> Dict[str, Any]:
        """Get default explanation when generation fails"""
        return {
            'shap_values': {'values': [0], 'base_value': 0.5},
            'human_readable': "Unable to generate explanation due to technical error.",
            'risk_factors': [],
            'feature_importance': [],
            'confidence_explanation': "Unable to assess confidence level"
        }
