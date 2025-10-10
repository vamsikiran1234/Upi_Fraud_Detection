"""
Decision Engine for UPI Fraud Detection
Combines ML predictions with business rules to make fraud decisions
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from loguru import logger

class DecisionEngine:
    """Decision engine for fraud detection"""
    
    def __init__(self, settings):
        self.settings = settings
        self.rules = self._load_business_rules()
        self.policy_weights = {
            'ml_score': 0.7,
            'business_rules': 0.2,
            'risk_factors': 0.1
        }
    
    def _load_business_rules(self) -> List[Dict[str, Any]]:
        """Load business rules for fraud detection"""
        return [
            {
                'name': 'high_amount_rule',
                'condition': lambda features: features.get('amount', 0) > 100000,
                'action': 'CHALLENGE',
                'weight': 0.3,
                'description': 'High amount transaction requires additional verification'
            },
            {
                'name': 'night_time_rule',
                'condition': lambda features: features.get('hour', 12) in [22, 23, 0, 1, 2, 3, 4, 5],
                'action': 'CHALLENGE',
                'weight': 0.2,
                'description': 'Night time transaction requires verification'
            },
            {
                'name': 'high_velocity_rule',
                'condition': lambda features: features.get('user_velocity', 0) > 20,
                'action': 'BLOCK',
                'weight': 0.4,
                'description': 'High transaction velocity indicates potential fraud'
            },
            {
                'name': 'new_device_rule',
                'condition': lambda features: features.get('device_age_days', 365) < 1,
                'action': 'CHALLENGE',
                'weight': 0.3,
                'description': 'New device requires verification'
            },
            {
                'name': 'high_risk_merchant_rule',
                'condition': lambda features: features.get('merchant_category', '') in ['gambling', 'adult', 'crypto'],
                'action': 'CHALLENGE',
                'weight': 0.4,
                'description': 'High risk merchant category requires verification'
            },
            {
                'name': 'location_anomaly_rule',
                'condition': lambda features: features.get('location_consistency', 1.0) < 0.3,
                'action': 'CHALLENGE',
                'weight': 0.3,
                'description': 'Unusual location requires verification'
            },
            {
                'name': 'device_anomaly_rule',
                'condition': lambda features: features.get('device_consistency', 1.0) < 0.3,
                'action': 'CHALLENGE',
                'weight': 0.3,
                'description': 'Unusual device requires verification'
            },
            {
                'name': 'suspicious_ip_rule',
                'condition': lambda features: features.get('ip_reputation', 0.5) < 0.2,
                'action': 'BLOCK',
                'weight': 0.5,
                'description': 'Suspicious IP address blocked'
            },
            {
                'name': 'low_confidence_rule',
                'condition': lambda features: features.get('confidence', 1.0) < 0.3,
                'action': 'CHALLENGE',
                'weight': 0.2,
                'description': 'Low confidence prediction requires verification'
            },
            {
                'name': 'amount_pattern_anomaly_rule',
                'condition': lambda features: features.get('amount_pattern_score', 1.0) < 0.2,
                'action': 'CHALLENGE',
                'weight': 0.3,
                'description': 'Unusual amount pattern requires verification'
            }
        ]
    
    async def decide(self, prediction: Dict[str, Any], features: Dict[str, Any], explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Make fraud decision based on ML prediction and business rules"""
        try:
            # Extract key values
            ml_score = prediction.get('risk_score', 0.5)
            confidence = prediction.get('confidence', 0.5)
            risk_factors = explanation.get('risk_factors', [])
            
            # Apply business rules
            rule_results = await self._apply_business_rules(features)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                ml_score, rule_results, risk_factors, confidence
            )
            
            # Make decision
            decision = self._make_decision(composite_score, rule_results, ml_score)
            
            # Generate alerts
            alerts = self._generate_alerts(features, rule_results, risk_factors)
            
            # Calculate decision confidence
            decision_confidence = self._calculate_decision_confidence(
                ml_score, rule_results, confidence
            )
            
            return {
                'decision': decision,
                'composite_score': composite_score,
                'ml_score': ml_score,
                'rule_score': rule_results['score'],
                'confidence': decision_confidence,
                'alerts': alerts,
                'rule_violations': rule_results['violations'],
                'risk_factors': risk_factors,
                'reasoning': self._generate_reasoning(decision, rule_results, risk_factors)
            }
            
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            return self._get_default_decision()
    
    async def _apply_business_rules(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply business rules to features"""
        violations = []
        total_weight = 0.0
        weighted_score = 0.0
        
        for rule in self.rules:
            try:
                if rule['condition'](features):
                    violations.append({
                        'rule_name': rule['name'],
                        'description': rule['description'],
                        'weight': rule['weight']
                    })
                    weighted_score += rule['weight']
                total_weight += rule['weight']
            except Exception as e:
                logger.error(f"Rule {rule['name']} failed: {e}")
        
        # Normalize score
        rule_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return {
            'score': rule_score,
            'violations': violations,
            'total_rules': len(self.rules),
            'violated_rules': len(violations)
        }
    
    def _calculate_composite_score(self, ml_score: float, rule_results: Dict, risk_factors: List, confidence: float) -> float:
        """Calculate composite fraud score"""
        try:
            # ML score component
            ml_component = ml_score * self.policy_weights['ml_score']
            
            # Business rules component
            rule_component = rule_results['score'] * self.policy_weights['business_rules']
            
            # Risk factors component
            risk_component = 0.0
            if risk_factors:
                high_risk_factors = [rf for rf in risk_factors if rf.get('severity') == 'high']
                medium_risk_factors = [rf for rf in risk_factors if rf.get('severity') == 'medium']
                
                risk_component = (
                    len(high_risk_factors) * 0.3 +
                    len(medium_risk_factors) * 0.1
                ) / max(len(risk_factors), 1)
            
            risk_component *= self.policy_weights['risk_factors']
            
            # Confidence adjustment
            confidence_adjustment = (1.0 - confidence) * 0.1
            
            composite_score = ml_component + rule_component + risk_component + confidence_adjustment
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, composite_score))
            
        except Exception as e:
            logger.error(f"Composite score calculation failed: {e}")
            return ml_score
    
    def _make_decision(self, composite_score: float, rule_results: Dict, ml_score: float) -> str:
        """Make final fraud decision"""
        try:
            # Check for blocking conditions
            blocking_rules = [v for v in rule_results['violations'] if v['weight'] >= 0.4]
            if blocking_rules:
                return 'BLOCK'
            
            # Check composite score thresholds
            if composite_score >= self.settings.risk_threshold_high:
                return 'BLOCK'
            elif composite_score >= self.settings.risk_threshold_medium:
                return 'CHALLENGE'
            elif composite_score >= self.settings.risk_threshold_low:
                return 'CHALLENGE'
            else:
                return 'ALLOW'
                
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            return 'CHALLENGE'  # Default to challenge on error
    
    def _generate_alerts(self, features: Dict[str, Any], rule_results: Dict, risk_factors: List) -> List[str]:
        """Generate alerts based on risk factors"""
        alerts = []
        
        try:
            # High amount alert
            if features.get('amount', 0) > 50000:
                alerts.append(f"High amount transaction: ₹{features['amount']:,.2f}")
            
            # High velocity alert
            if features.get('user_velocity', 0) > 10:
                alerts.append(f"High transaction velocity: {features['user_velocity']} transactions")
            
            # New device alert
            if features.get('device_age_days', 365) < 1:
                alerts.append("New device detected")
            
            # High risk merchant alert
            if features.get('merchant_category', '') in ['gambling', 'adult', 'crypto']:
                alerts.append(f"High risk merchant: {features['merchant_category']}")
            
            # Location anomaly alert
            if features.get('location_consistency', 1.0) < 0.3:
                alerts.append("Unusual location detected")
            
            # Device anomaly alert
            if features.get('device_consistency', 1.0) < 0.3:
                alerts.append("Unusual device detected")
            
            # Suspicious IP alert
            if features.get('ip_reputation', 0.5) < 0.2:
                alerts.append("Suspicious IP address")
            
            # Low confidence alert
            if features.get('confidence', 1.0) < 0.3:
                alerts.append("Low confidence prediction")
            
            # Risk factor alerts
            for risk_factor in risk_factors:
                if risk_factor.get('severity') == 'high':
                    alerts.append(f"High risk factor: {risk_factor['feature']}")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            return ["System error in alert generation"]
    
    def _calculate_decision_confidence(self, ml_score: float, rule_results: Dict, confidence: float) -> float:
        """Calculate confidence in the decision"""
        try:
            # Base confidence from ML model
            base_confidence = confidence
            
            # Rule agreement factor
            rule_agreement = 1.0 - (rule_results['violated_rules'] / rule_results['total_rules'])
            
            # ML score confidence (scores near 0.5 are less confident)
            ml_confidence = 1.0 - abs(ml_score - 0.5) * 2
            
            # Combine confidences
            decision_confidence = (
                base_confidence * 0.4 +
                rule_agreement * 0.3 +
                ml_confidence * 0.3
            )
            
            return max(0.0, min(1.0, decision_confidence))
            
        except Exception as e:
            logger.error(f"Decision confidence calculation failed: {e}")
            return confidence
    
    def _generate_reasoning(self, decision: str, rule_results: Dict, risk_factors: List) -> str:
        """Generate human-readable reasoning for the decision"""
        try:
            reasoning_parts = []
            
            # Decision explanation
            if decision == 'BLOCK':
                reasoning_parts.append("Transaction blocked due to high fraud risk")
            elif decision == 'CHALLENGE':
                reasoning_parts.append("Transaction requires additional verification")
            else:
                reasoning_parts.append("Transaction approved with low fraud risk")
            
            # Rule violations
            if rule_results['violations']:
                violation_descriptions = [v['description'] for v in rule_results['violations']]
                reasoning_parts.append(f"Triggered rules: {', '.join(violation_descriptions)}")
            
            # Risk factors
            if risk_factors:
                high_risk_factors = [rf['feature'] for rf in risk_factors if rf.get('severity') == 'high']
                if high_risk_factors:
                    reasoning_parts.append(f"High risk factors: {', '.join(high_risk_factors)}")
            
            return ". ".join(reasoning_parts) + "."
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return f"Decision: {decision} (reasoning unavailable due to technical error)"
    
    def _get_default_decision(self) -> Dict[str, Any]:
        """Get default decision when processing fails"""
        return {
            'decision': 'CHALLENGE',
            'composite_score': 0.5,
            'ml_score': 0.5,
            'rule_score': 0.0,
            'confidence': 0.5,
            'alerts': ['System error in decision processing'],
            'rule_violations': [],
            'risk_factors': [],
            'reasoning': 'Default challenge due to system error'
        }
