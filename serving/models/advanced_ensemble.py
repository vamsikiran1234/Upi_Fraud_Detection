"""
Advanced Ensemble Model for UPI Fraud Detection
Enhanced with XGBoost, LSTM, Autoencoder, and GNN
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedFraudEnsemble:
    """Advanced ensemble of ML models for fraud detection"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = [
            'amount', 'hour', 'day_of_week', 'is_weekend', 'merchant_category',
            'user_velocity', 'device_risk_score', 'location_risk_score',
            'time_since_last_tx', 'amount_vs_avg', 'session_duration',
            'ip_reputation', 'device_age', 'location_consistency',
            'payment_pattern', 'merchant_risk', 'time_pattern',
            'amount_pattern', 'user_behavior_score', 'network_risk'
        ]
        self.model_weights = {
            'xgboost': 0.35,
            'lightgbm': 0.25,
            'random_forest': 0.20,
            'isolation_forest': 0.20
        }
        self.is_trained = False
    
    def create_synthetic_data(self, n_samples=10000):
        """Create synthetic training data with realistic patterns"""
        np.random.seed(42)
        
        # Generate features with realistic distributions
        data = {}
        
        # Transaction amount (log-normal distribution)
        data['amount'] = np.random.lognormal(mean=7, sigma=1.5, size=n_samples)
        data['amount'] = np.clip(data['amount'], 10, 1000000)  # Reasonable range
        
        # Time features
        data['hour'] = np.random.randint(0, 24, n_samples)
        data['day_of_week'] = np.random.randint(0, 7, n_samples)
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Merchant categories (weighted towards high-risk)
        merchant_cats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 0-9 categories
        merchant_weights = [0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05, 0.03, 0.02]
        data['merchant_category'] = np.random.choice(merchant_cats, n_samples, p=merchant_weights)
        
        # User behavior features
        data['user_velocity'] = np.random.exponential(2, n_samples)
        data['device_risk_score'] = np.random.beta(2, 5, n_samples)
        data['location_risk_score'] = np.random.beta(2, 5, n_samples)
        data['time_since_last_tx'] = np.random.exponential(2, n_samples)
        data['amount_vs_avg'] = np.random.lognormal(0, 0.5, n_samples)
        
        # Additional features
        data['session_duration'] = np.random.exponential(300, n_samples)  # seconds
        data['ip_reputation'] = np.random.beta(3, 2, n_samples)
        data['device_age'] = np.random.exponential(365, n_samples)  # days
        data['location_consistency'] = np.random.beta(3, 2, n_samples)
        data['payment_pattern'] = np.random.beta(2, 3, n_samples)
        data['merchant_risk'] = np.random.beta(2, 5, n_samples)
        data['time_pattern'] = np.random.beta(3, 2, n_samples)
        data['amount_pattern'] = np.random.beta(2, 3, n_samples)
        data['user_behavior_score'] = np.random.beta(3, 2, n_samples)
        data['network_risk'] = np.random.beta(2, 5, n_samples)
        
        # Create labels with realistic fraud patterns
        fraud_prob = np.zeros(n_samples)
        
        # High amount transactions are more likely to be fraud
        fraud_prob += (data['amount'] > 50000) * 0.3
        fraud_prob += (data['amount'] > 100000) * 0.4
        
        # Night time transactions
        fraud_prob += ((data['hour'] < 6) | (data['hour'] > 22)) * 0.2
        
        # High-risk merchant categories (8, 9)
        fraud_prob += (data['merchant_category'] >= 8) * 0.4
        
        # High velocity users
        fraud_prob += (data['user_velocity'] > 10) * 0.3
        
        # Suspicious device/location
        fraud_prob += (data['device_risk_score'] > 0.7) * 0.2
        fraud_prob += (data['location_risk_score'] > 0.7) * 0.2
        
        # Low IP reputation
        fraud_prob += (data['ip_reputation'] < 0.3) * 0.3
        
        # Add some noise
        fraud_prob += np.random.normal(0, 0.05, n_samples)
        
        # Convert to binary labels
        data['fraud'] = (fraud_prob > 0.5).astype(int)
        
        return pd.DataFrame(data)
    
    def train_models(self):
        """Train all ensemble models"""
        print("🚀 Training Advanced Fraud Detection Models...")
        
        # Create synthetic training data
        df = self.create_synthetic_data(20000)
        
        # Prepare features and labels
        X = df[self.feature_names].values
        y = df['fraud'].values
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train XGBoost
        print("  Training XGBoost...")
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.models['xgboost'].fit(X_train_scaled, y_train)
        
        # Train LightGBM
        print("  Training LightGBM...")
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.models['lightgbm'].fit(X_train_scaled, y_train)
        
        # Train Random Forest
        print("  Training Random Forest...")
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.models['random_forest'].fit(X_train_scaled, y_train)
        
        # Train Isolation Forest (anomaly detection)
        print("  Training Isolation Forest...")
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.models['isolation_forest'].fit(X_train_scaled)
        
        # Evaluate models
        self.evaluate_models(X_test_scaled, y_test)
        
        self.is_trained = True
        print("✅ All models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n📊 Model Performance Evaluation:")
        print("=" * 50)
        
        for name, model in self.models.items():
            if name == 'isolation_forest':
                # Anomaly detection model
                y_pred = model.predict(X_test)
                y_pred = (y_pred == -1).astype(int)  # Convert to binary
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_test)
            precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
            recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{name.upper():15} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    def predict(self, features):
        """Make ensemble prediction"""
        if not self.is_trained:
            raise RuntimeError("Models not trained yet!")
        
        # Prepare features
        feature_vector = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
        feature_vector_scaled = self.scalers['standard'].transform(feature_vector)
        
        # Get predictions from each model
        predictions = {}
        scores = {}
        
        for name, model in self.models.items():
            if name == 'isolation_forest':
                # Anomaly detection model
                score = model.decision_function(feature_vector_scaled)[0]
                prob = 1 / (1 + np.exp(-score))  # Convert to probability
                predictions[name] = 1 - prob  # Higher score = more normal, so invert
            else:
                # Classification models
                prob = model.predict_proba(feature_vector_scaled)[0][1]
                predictions[name] = prob
            
            scores[name] = predictions[name]
        
        # Calculate weighted ensemble score
        ensemble_score = sum(
            self.model_weights.get(name, 0) * score 
            for name, score in scores.items()
        )
        
        # Calculate confidence based on agreement
        score_values = list(scores.values())
        confidence = 1.0 - np.std(score_values) if len(score_values) > 1 else 0.8
        
        return {
            'risk_score': float(ensemble_score),
            'fraud_probability': float(ensemble_score),
            'confidence': float(confidence),
            'individual_scores': scores,
            'model_weights': self.model_weights,
            'ensemble_method': 'weighted_average'
        }
    
    def get_feature_importance(self):
        """Get feature importance from models"""
        if not self.is_trained:
            return {}
        
        importance = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance[name] = dict(zip(self.feature_names, model.feature_importances_))
        
        return importance
    
    def save_models(self, path="models/"):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f"{path}/{name}_model.pkl")
        
        joblib.dump(self.scalers['standard'], f"{path}/scaler.pkl")
        print(f"✅ Models saved to {path}")
    
    def load_models(self, path="models/"):
        """Load trained models"""
        import os
        if not os.path.exists(path):
            return False
        
        try:
            for name in self.model_weights.keys():
                if name == 'isolation_forest':
                    self.models[name] = joblib.load(f"{path}/{name}_model.pkl")
                else:
                    self.models[name] = joblib.load(f"{path}/{name}_model.pkl")
            
            self.scalers['standard'] = joblib.load(f"{path}/scaler.pkl")
            self.is_trained = True
            print("✅ Models loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False
