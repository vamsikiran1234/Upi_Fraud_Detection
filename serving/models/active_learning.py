"""
Analyst-in-the-Loop + Active Learning Pipeline
Enables continuous model improvement through human feedback and uncertainty sampling
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
from abc import ABC, abstractmethod

@dataclass
class AnalystFeedback:
    """Analyst feedback on model predictions"""
    transaction_id: str
    model_prediction: str
    analyst_decision: str
    confidence: float
    feedback_timestamp: datetime
    analyst_id: str
    reasoning: str
    false_positive: bool = False
    false_negative: bool = False
    uncertainty_score: float = 0.0

@dataclass
class ActiveLearningSample:
    """Sample selected for active learning"""
    transaction_id: str
    features: np.ndarray
    model_prediction: str
    confidence: float
    uncertainty_score: float
    selection_reason: str
    priority: int  # 1=high, 2=medium, 3=low
    timestamp: datetime

class UncertaintySampler:
    """Sample selection strategies for active learning"""
    
    def __init__(self, strategy: str = 'entropy'):
        self.strategy = strategy
    
    def select_samples(self, 
                      predictions: np.ndarray, 
                      probabilities: np.ndarray,
                      n_samples: int = 100) -> List[int]:
        """Select samples for annotation based on uncertainty"""
        
        if self.strategy == 'entropy':
            return self._entropy_sampling(probabilities, n_samples)
        elif self.strategy == 'margin':
            return self._margin_sampling(probabilities, n_samples)
        elif self.strategy == 'least_confidence':
            return self._least_confidence_sampling(probabilities, n_samples)
        elif self.strategy == 'query_by_committee':
            return self._query_by_committee_sampling(predictions, probabilities, n_samples)
        else:
            return self._random_sampling(len(predictions), n_samples)
    
    def _entropy_sampling(self, probabilities: np.ndarray, n_samples: int) -> List[int]:
        """Select samples with highest entropy (uncertainty)"""
        # Calculate entropy for each sample
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        
        # Select top n_samples with highest entropy
        selected_indices = np.argsort(entropy)[-n_samples:][::-1]
        return selected_indices.tolist()
    
    def _margin_sampling(self, probabilities: np.ndarray, n_samples: int) -> List[int]:
        """Select samples with smallest margin between top 2 predictions"""
        # Sort probabilities for each sample
        sorted_probs = np.sort(probabilities, axis=1)
        
        # Calculate margin (difference between top 2 predictions)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        # Select samples with smallest margins
        selected_indices = np.argsort(margins)[:n_samples]
        return selected_indices.tolist()
    
    def _least_confidence_sampling(self, probabilities: np.ndarray, n_samples: int) -> List[int]:
        """Select samples with lowest confidence in prediction"""
        # Get maximum probability for each sample
        max_probs = np.max(probabilities, axis=1)
        
        # Select samples with lowest confidence
        selected_indices = np.argsort(max_probs)[:n_samples]
        return selected_indices.tolist()
    
    def _query_by_committee_sampling(self, 
                                   predictions: np.ndarray, 
                                   probabilities: np.ndarray, 
                                   n_samples: int) -> List[int]:
        """Select samples where committee of models disagree most"""
        # Calculate disagreement (variance in predictions)
        disagreement = np.var(probabilities, axis=1)
        
        # Select samples with highest disagreement
        selected_indices = np.argsort(disagreement)[-n_samples:][::-1]
        return selected_indices.tolist()
    
    def _random_sampling(self, total_samples: int, n_samples: int) -> List[int]:
        """Random sampling baseline"""
        return np.random.choice(total_samples, min(n_samples, total_samples), replace=False).tolist()

class AnalystWorkflow:
    """Manage analyst workflow and feedback collection"""
    
    def __init__(self):
        self.pending_reviews = []
        self.completed_feedback = []
        self.analyst_performance = {}
    
    def add_pending_review(self, sample: ActiveLearningSample) -> str:
        """Add sample to pending review queue"""
        review_id = f"REVIEW_{len(self.pending_reviews)}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        review_item = {
            'review_id': review_id,
            'sample': sample,
            'assigned_analyst': None,
            'status': 'pending',
            'created_at': datetime.utcnow(),
            'priority': sample.priority
        }
        
        self.pending_reviews.append(review_item)
        
        # Sort by priority
        self.pending_reviews.sort(key=lambda x: x['priority'])
        
        return review_id
    
    def assign_review(self, review_id: str, analyst_id: str) -> bool:
        """Assign review to analyst"""
        for review in self.pending_reviews:
            if review['review_id'] == review_id:
                review['assigned_analyst'] = analyst_id
                review['status'] = 'assigned'
                review['assigned_at'] = datetime.utcnow()
                return True
        return False
    
    def submit_feedback(self, review_id: str, feedback: AnalystFeedback) -> bool:
        """Submit analyst feedback"""
        for i, review in enumerate(self.pending_reviews):
            if review['review_id'] == review_id:
                # Move to completed feedback
                review['status'] = 'completed'
                review['feedback'] = feedback
                review['completed_at'] = datetime.utcnow()
                
                # Move to completed list
                self.completed_feedback.append(review)
                del self.pending_reviews[i]
                
                # Update analyst performance
                self._update_analyst_performance(feedback)
                
                return True
        return False
    
    def _update_analyst_performance(self, feedback: AnalystFeedback):
        """Update analyst performance metrics"""
        analyst_id = feedback.analyst_id
        
        if analyst_id not in self.analyst_performance:
            self.analyst_performance[analyst_id] = {
                'total_reviews': 0,
                'correct_predictions': 0,
                'false_positives_caught': 0,
                'false_negatives_caught': 0,
                'average_response_time': 0,
                'last_activity': None
            }
        
        perf = self.analyst_performance[analyst_id]
        perf['total_reviews'] += 1
        perf['last_activity'] = datetime.utcnow()
        
        if feedback.false_positive:
            perf['false_positives_caught'] += 1
        if feedback.false_negative:
            perf['false_negatives_caught'] += 1
        
        # Calculate accuracy
        if feedback.model_prediction == feedback.analyst_decision:
            perf['correct_predictions'] += 1
    
    def get_analyst_performance(self, analyst_id: str = None) -> Dict[str, Any]:
        """Get analyst performance metrics"""
        if analyst_id:
            return self.analyst_performance.get(analyst_id, {})
        else:
            return self.analyst_performance
    
    def get_pending_reviews(self, analyst_id: str = None) -> List[Dict[str, Any]]:
        """Get pending reviews for analyst"""
        if analyst_id:
            return [r for r in self.pending_reviews if r['assigned_analyst'] == analyst_id]
        else:
            return self.pending_reviews

class ModelRetrainer:
    """Handle model retraining with new feedback data"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.training_history = []
        self.feature_importance_history = []
    
    def retrain_model(self, 
                     X_train: np.ndarray, 
                     y_train: np.ndarray,
                     X_val: np.ndarray = None,
                     y_val: np.ndarray = None,
                     feedback_data: List[AnalystFeedback] = None) -> Dict[str, Any]:
        """Retrain model with new data and feedback"""
        
        print(f"🔄 Retraining {self.model_type} model...")
        
        # Prepare training data
        if feedback_data:
            # Incorporate feedback into training data
            X_feedback, y_feedback = self._prepare_feedback_data(feedback_data)
            X_train = np.vstack([X_train, X_feedback])
            y_train = np.hstack([y_train, y_feedback])
        
        # Train model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_train, y_train)
        
        # Evaluate model
        training_metrics = self._evaluate_model(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            validation_metrics = self._evaluate_model(X_val, y_val)
        else:
            validation_metrics = None
        
        # Store training history
        training_record = {
            'timestamp': datetime.utcnow(),
            'training_samples': len(X_train),
            'feedback_samples': len(feedback_data) if feedback_data else 0,
            'training_metrics': training_metrics,
            'validation_metrics': validation_metrics,
            'feature_importance': self._get_feature_importance()
        }
        
        self.training_history.append(training_record)
        self.feature_importance_history.append(training_record['feature_importance'])
        
        return training_record
    
    def _prepare_feedback_data(self, feedback_data: List[AnalystFeedback]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feedback data for training"""
        # This would extract features from feedback data
        # For demo, we'll create synthetic features
        n_samples = len(feedback_data)
        n_features = 20  # Match your feature dimension
        
        X_feedback = np.random.randn(n_samples, n_features)
        y_feedback = np.array([1 if f.analyst_decision == 'BLOCK' else 0 for f in feedback_data])
        
        return X_feedback, y_feedback
    
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        if self.model is None:
            return {}
        
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y)
        precision = np.sum((y_pred == 1) & (y == 1)) / max(1, np.sum(y_pred == 1))
        recall = np.sum((y_pred == 1) & (y == 1)) / max(1, np.sum(y == 1))
        f1_score = 2 * (precision * recall) / max(1e-8, precision + recall)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc': self._calculate_auc(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0
        }
    
    def _calculate_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate AUC score"""
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(y_true, y_scores)
        except:
            return 0.0
    
    def _get_feature_importance(self) -> np.ndarray:
        """Get feature importance from model"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return np.array([])
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is not None:
            joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        self.model = joblib.load(filepath)

class ActiveLearningPipeline:
    """Main active learning pipeline"""
    
    def __init__(self, 
                 uncertainty_strategy: str = 'entropy',
                 retrain_threshold: int = 100,
                 model_type: str = 'random_forest'):
        
        self.uncertainty_sampler = UncertaintySampler(uncertainty_strategy)
        self.analyst_workflow = AnalystWorkflow()
        self.model_retrainer = ModelRetrainer(model_type)
        self.retrain_threshold = retrain_threshold
        
        # Active learning state
        self.pool_data = None
        self.labeled_data = None
        self.unlabeled_data = None
        self.model_predictions = None
        self.uncertainty_scores = None
        
        # Performance tracking
        self.learning_curve = []
        self.sample_efficiency = []
    
    def initialize_pool(self, X: np.ndarray, y: np.ndarray = None, initial_size: int = 100):
        """Initialize the active learning pool"""
        print("🎯 Initializing Active Learning Pool...")
        
        if y is not None:
            # Supervised initialization
            X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
                X, y, train_size=initial_size, random_state=42, stratify=y
            )
            
            self.labeled_data = {
                'X': X_labeled,
                'y': y_labeled,
                'indices': np.arange(initial_size)
            }
            
            self.unlabeled_data = {
                'X': X_unlabeled,
                'y': y_unlabeled,
                'indices': np.arange(initial_size, len(X))
            }
        else:
            # Unsupervised initialization
            initial_indices = np.random.choice(len(X), initial_size, replace=False)
            
            self.labeled_data = {
                'X': X[initial_indices],
                'y': None,
                'indices': initial_indices
            }
            
            self.unlabeled_data = {
                'X': X[np.setdiff1d(np.arange(len(X)), initial_indices)],
                'y': None,
                'indices': np.setdiff1d(np.arange(len(X)), initial_indices)
            }
        
        self.pool_data = X
        
        print(f"   Labeled samples: {len(self.labeled_data['X'])}")
        print(f"   Unlabeled samples: {len(self.unlabeled_data['X'])}")
    
    def select_samples_for_annotation(self, n_samples: int = 50) -> List[ActiveLearningSample]:
        """Select samples for analyst annotation"""
        if self.model_retrainer.model is None:
            print("⚠️ Model not trained yet. Training initial model...")
            self._train_initial_model()
        
        # Get predictions on unlabeled data
        predictions = self.model_retrainer.model.predict(self.unlabeled_data['X'])
        probabilities = self.model_retrainer.model.predict_proba(self.unlabeled_data['X'])
        
        # Calculate uncertainty scores
        uncertainty_scores = self._calculate_uncertainty(probabilities)
        
        # Select samples using uncertainty sampling
        selected_indices = self.uncertainty_sampler.select_samples(
            predictions, probabilities, n_samples
        )
        
        # Create active learning samples
        samples = []
        for i, idx in enumerate(selected_indices):
            sample = ActiveLearningSample(
                transaction_id=f"TXN_{self.unlabeled_data['indices'][idx]:06d}",
                features=self.unlabeled_data['X'][idx],
                model_prediction=self._map_prediction(predictions[idx]),
                confidence=np.max(probabilities[idx]),
                uncertainty_score=uncertainty_scores[idx],
                selection_reason=f"Uncertainty sampling - {self.uncertainty_sampler.strategy}",
                priority=self._calculate_priority(uncertainty_scores[idx], predictions[idx]),
                timestamp=datetime.utcnow()
            )
            samples.append(sample)
        
        return samples
    
    def _train_initial_model(self):
        """Train initial model on labeled data"""
        if self.labeled_data['y'] is not None:
            self.model_retrainer.retrain_model(
                self.labeled_data['X'], 
                self.labeled_data['y']
            )
    
    def _calculate_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate uncertainty scores"""
        # Use entropy as uncertainty measure
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        return entropy
    
    def _map_prediction(self, prediction: int) -> str:
        """Map numeric prediction to string"""
        mapping = {0: 'ALLOW', 1: 'BLOCK'}
        return mapping.get(prediction, 'CHALLENGE')
    
    def _calculate_priority(self, uncertainty_score: float, prediction: int) -> int:
        """Calculate priority for analyst review"""
        if uncertainty_score > 0.8:
            return 1  # High priority
        elif uncertainty_score > 0.5:
            return 2  # Medium priority
        else:
            return 3  # Low priority
    
    def add_feedback(self, feedback: AnalystFeedback) -> bool:
        """Add analyst feedback to the system"""
        # Add to workflow
        success = self.analyst_workflow.submit_feedback(feedback.transaction_id, feedback)
        
        if success:
            # Check if we have enough feedback for retraining
            completed_feedback = self.analyst_workflow.completed_feedback
            if len(completed_feedback) >= self.retrain_threshold:
                self._trigger_retraining()
        
        return success
    
    def _trigger_retraining(self):
        """Trigger model retraining with new feedback"""
        print("🔄 Triggering model retraining with analyst feedback...")
        
        # Get recent feedback
        recent_feedback = self.analyst_workflow.completed_feedback[-self.retrain_threshold:]
        feedback_data = [item['feedback'] for item in recent_feedback]
        
        # Retrain model
        if self.labeled_data['y'] is not None:
            training_record = self.model_retrainer.retrain_model(
                self.labeled_data['X'],
                self.labeled_data['y'],
                feedback_data=feedback_data
            )
            
            # Update learning curve
            self.learning_curve.append({
                'timestamp': datetime.utcnow(),
                'training_samples': training_record['training_samples'],
                'feedback_samples': training_record['feedback_samples'],
                'accuracy': training_record['training_metrics'].get('accuracy', 0),
                'f1_score': training_record['training_metrics'].get('f1_score', 0)
            })
            
            print(f"   Retraining completed - Accuracy: {training_record['training_metrics'].get('accuracy', 0):.3f}")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the active learning process"""
        return {
            'total_labeled_samples': len(self.labeled_data['X']) if self.labeled_data else 0,
            'total_unlabeled_samples': len(self.unlabeled_data['X']) if self.unlabeled_data else 0,
            'pending_reviews': len(self.analyst_workflow.pending_reviews),
            'completed_feedback': len(self.analyst_workflow.completed_feedback),
            'learning_curve': self.learning_curve,
            'analyst_performance': self.analyst_workflow.get_analyst_performance(),
            'model_training_history': self.model_retrainer.training_history,
            'uncertainty_strategy': self.uncertainty_sampler.strategy,
            'retrain_threshold': self.retrain_threshold
        }
    
    def get_analyst_dashboard_data(self) -> Dict[str, Any]:
        """Get data for analyst dashboard"""
        return {
            'pending_reviews': [
                {
                    'review_id': review['review_id'],
                    'transaction_id': review['sample'].transaction_id,
                    'model_prediction': review['sample'].model_prediction,
                    'confidence': review['sample'].confidence,
                    'uncertainty_score': review['sample'].uncertainty_score,
                    'priority': review['sample'].priority,
                    'created_at': review['created_at'].isoformat(),
                    'status': review['status']
                }
                for review in self.analyst_workflow.pending_reviews
            ],
            'analyst_performance': self.analyst_workflow.get_analyst_performance(),
            'learning_progress': {
                'total_samples_annotated': len(self.analyst_workflow.completed_feedback),
                'model_accuracy': self.learning_curve[-1]['accuracy'] if self.learning_curve else 0,
                'last_retraining': self.learning_curve[-1]['timestamp'].isoformat() if self.learning_curve else None
            }
        }

# Example usage and testing
def demonstrate_active_learning():
    """Demonstrate active learning pipeline"""
    print("🎓 Demonstrating Active Learning Pipeline")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 2000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = (np.sum(X[:, :5], axis=1) > 0).astype(int)  # Simple decision boundary
    
    print(f"Created {n_samples} samples with {n_features} features")
    print(f"Fraud rate: {y.mean():.3f}")
    
    # Initialize active learning pipeline
    al_pipeline = ActiveLearningPipeline(
        uncertainty_strategy='entropy',
        retrain_threshold=50,
        model_type='random_forest'
    )
    
    # Initialize pool
    al_pipeline.initialize_pool(X, y, initial_size=200)
    
    # Simulate active learning rounds
    print("\n1. Simulating Active Learning Rounds...")
    
    for round_num in range(5):
        print(f"\n--- Active Learning Round {round_num + 1} ---")
        
        # Select samples for annotation
        samples = al_pipeline.select_samples_for_annotation(n_samples=20)
        print(f"   Selected {len(samples)} samples for annotation")
        
        # Simulate analyst feedback
        feedback_count = 0
        for sample in samples[:10]:  # Simulate feedback for first 10 samples
            # Simulate analyst decision (with some disagreement)
            if sample.model_prediction == 'BLOCK' and np.random.random() < 0.2:
                analyst_decision = 'ALLOW'  # 20% disagreement
                false_positive = True
            elif sample.model_prediction == 'ALLOW' and np.random.random() < 0.1:
                analyst_decision = 'BLOCK'  # 10% disagreement
                false_negative = True
            else:
                analyst_decision = sample.model_prediction
                false_positive = False
                false_negative = False
            
            feedback = AnalystFeedback(
                transaction_id=sample.transaction_id,
                model_prediction=sample.model_prediction,
                analyst_decision=analyst_decision,
                confidence=sample.confidence,
                feedback_timestamp=datetime.utcnow(),
                analyst_id=f"analyst_{np.random.randint(1, 4)}",
                reasoning=f"Analyst review of {sample.model_prediction} prediction",
                false_positive=false_positive,
                false_negative=false_negative,
                uncertainty_score=sample.uncertainty_score
            )
            
            al_pipeline.add_feedback(feedback)
            feedback_count += 1
        
        print(f"   Processed {feedback_count} analyst feedbacks")
        
        # Get insights
        insights = al_pipeline.get_learning_insights()
        print(f"   Total labeled samples: {insights['total_labeled_samples']}")
        print(f"   Completed feedback: {insights['completed_feedback']}")
        print(f"   Pending reviews: {insights['pending_reviews']}")
        
        if insights['learning_curve']:
            latest_accuracy = insights['learning_curve'][-1]['accuracy']
            print(f"   Model accuracy: {latest_accuracy:.3f}")
    
    # Get final insights
    print("\n2. Final Active Learning Insights...")
    final_insights = al_pipeline.get_learning_insights()
    
    print(f"   Total learning rounds: {len(final_insights['learning_curve'])}")
    print(f"   Final accuracy: {final_insights['learning_curve'][-1]['accuracy']:.3f}")
    print(f"   Total feedback processed: {final_insights['completed_feedback']}")
    
    # Analyst performance
    analyst_perf = final_insights['analyst_performance']
    print(f"   Analyst performance:")
    for analyst_id, perf in analyst_perf.items():
        accuracy = perf['correct_predictions'] / max(1, perf['total_reviews'])
        print(f"     {analyst_id}: {perf['total_reviews']} reviews, {accuracy:.3f} accuracy")
    
    # Get dashboard data
    print("\n3. Analyst Dashboard Data...")
    dashboard_data = al_pipeline.get_analyst_dashboard_data()
    print(f"   Pending reviews: {len(dashboard_data['pending_reviews'])}")
    print(f"   Learning progress: {dashboard_data['learning_progress']}")
    
    print("\n✅ Active Learning demonstration completed!")
    
    return {
        'al_pipeline': al_pipeline,
        'final_insights': final_insights,
        'dashboard_data': dashboard_data
    }

if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_active_learning()
    
    print(f"\n📊 Summary:")
    print(f"   Active learning pipeline operational")
    print(f"   Final model accuracy: {results['final_insights']['learning_curve'][-1]['accuracy']:.3f}")
    print(f"   Total feedback processed: {results['final_insights']['completed_feedback']}")
    print(f"   Uncertainty sampling strategy: {results['final_insights']['uncertainty_strategy']}")
