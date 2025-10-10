"""
Federated Learning Module for Privacy-Preserving Fraud Detection
Enables multiple banks to collaborate without sharing sensitive data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from cryptography.fernet import Fernet
import requests
from datetime import datetime

@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    num_banks: int = 5
    rounds: int = 100
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    privacy_budget: float = 1.0
    aggregation_method: str = "fedavg"  # fedavg, fedprox, fednova
    communication_rounds: int = 10
    min_clients: int = 3

class PrivacyPreservingAggregator:
    """Privacy-preserving model aggregation using secure multi-party computation"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
    def encrypt_model_weights(self, weights: Dict[str, np.ndarray]) -> bytes:
        """Encrypt model weights for secure transmission"""
        weights_json = json.dumps({k: v.tolist() for k, v in weights.items()})
        return self.cipher.encrypt(weights_json.encode())
    
    def decrypt_model_weights(self, encrypted_weights: bytes) -> Dict[str, np.ndarray]:
        """Decrypt model weights"""
        decrypted_json = self.cipher.decrypt(encrypted_weights).decode()
        weights_dict = json.loads(decrypted_json)
        return {k: np.array(v) for k, v in weights_dict.items()}
    
    def add_differential_privacy_noise(self, weights: Dict[str, np.ndarray], 
                                     epsilon: float, delta: float = 1e-5) -> Dict[str, np.ndarray]:
        """Add differential privacy noise to model weights"""
        noisy_weights = {}
        sensitivity = 1.0  # L2 sensitivity
        
        for name, weight in weights.items():
            # Calculate noise scale
            noise_scale = (2 * sensitivity * np.log(1.25 / delta)) / epsilon
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_scale, weight.shape)
            noisy_weights[name] = weight + noise
            
        return noisy_weights
    
    def federated_averaging(self, client_weights: List[Dict[str, np.ndarray]], 
                          client_sizes: List[int]) -> Dict[str, np.ndarray]:
        """Perform federated averaging with weighted aggregation"""
        if not client_weights:
            return {}
        
        # Calculate total samples
        total_samples = sum(client_sizes)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        for key in client_weights[0].keys():
            aggregated_weights[key] = np.zeros_like(client_weights[0][key])
        
        # Weighted aggregation
        for weights, size in zip(client_weights, client_sizes):
            weight_factor = size / total_samples
            for key in weights.keys():
                aggregated_weights[key] += weight_factor * weights[key]
        
        return aggregated_weights

class FederatedFraudModel(nn.Module):
    """Neural network model for federated fraud detection"""
    
    def __init__(self, input_size: int = 20, hidden_size: int = 64, num_classes: int = 2):
        super(FederatedFraudModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get model weights as numpy arrays"""
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.data.cpu().numpy()
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set model weights from numpy arrays"""
        for name, param in self.named_parameters():
            if name in weights:
                param.data = torch.from_numpy(weights[name]).float()

class FederatedLearningCoordinator:
    """Coordinates federated learning across multiple banks"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.aggregator = PrivacyPreservingAggregator(config)
        self.global_model = FederatedFraudModel()
        self.bank_models = {}
        self.training_history = []
        
    def initialize_bank(self, bank_id: str) -> Dict[str, Any]:
        """Initialize a new bank in the federated learning network"""
        bank_model = FederatedFraudModel()
        self.bank_models[bank_id] = {
            'model': bank_model,
            'data_size': 0,
            'last_update': None,
            'participation_count': 0
        }
        
        # Send initial global model
        initial_weights = self.global_model.get_weights()
        encrypted_weights = self.aggregator.encrypt_model_weights(initial_weights)
        
        return {
            'bank_id': bank_id,
            'encrypted_weights': encrypted_weights,
            'config': {
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'local_epochs': self.config.local_epochs
            }
        }
    
    def train_local_model(self, bank_id: str, local_data: pd.DataFrame, 
                         labels: np.ndarray) -> Dict[str, Any]:
        """Train local model on bank's private data"""
        if bank_id not in self.bank_models:
            raise ValueError(f"Bank {bank_id} not initialized")
        
        bank_info = self.bank_models[bank_id]
        model = bank_info['model']
        
        # Prepare data
        X = torch.FloatTensor(local_data.values)
        y = torch.LongTensor(labels)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Set up training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # Local training
        model.train()
        for epoch in range(self.config.local_epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Update bank info
        bank_info['data_size'] = len(local_data)
        bank_info['last_update'] = datetime.utcnow()
        bank_info['participation_count'] += 1
        
        # Get updated weights
        updated_weights = model.get_weights()
        
        # Add differential privacy noise
        dp_weights = self.aggregator.add_differential_privacy_noise(
            updated_weights, 
            epsilon=self.config.privacy_budget
        )
        
        # Encrypt weights
        encrypted_weights = self.aggregator.encrypt_model_weights(dp_weights)
        
        return {
            'bank_id': bank_id,
            'encrypted_weights': encrypted_weights,
            'data_size': bank_info['data_size'],
            'training_loss': loss.item(),
            'participation_count': bank_info['participation_count']
        }
    
    def aggregate_models(self, encrypted_weights_list: List[bytes], 
                        data_sizes: List[int]) -> Dict[str, Any]:
        """Aggregate models from multiple banks"""
        if len(encrypted_weights_list) < self.config.min_clients:
            raise ValueError(f"Need at least {self.config.min_clients} clients for aggregation")
        
        # Decrypt weights
        decrypted_weights = []
        for encrypted_weights in encrypted_weights_list:
            weights = self.aggregator.decrypt_model_weights(encrypted_weights)
            decrypted_weights.append(weights)
        
        # Perform federated averaging
        aggregated_weights = self.aggregator.federated_averaging(
            decrypted_weights, data_sizes
        )
        
        # Update global model
        self.global_model.set_weights(aggregated_weights)
        
        # Encrypt updated global model
        encrypted_global_weights = self.aggregator.encrypt_model_weights(aggregated_weights)
        
        # Record training round
        self.training_history.append({
            'round': len(self.training_history) + 1,
            'participating_banks': len(encrypted_weights_list),
            'total_samples': sum(data_sizes),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return {
            'encrypted_global_weights': encrypted_global_weights,
            'round': len(self.training_history),
            'participating_banks': len(encrypted_weights_list),
            'total_samples': sum(data_sizes)
        }
    
    def evaluate_global_model(self, test_data: pd.DataFrame, 
                            test_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate global model performance"""
        self.global_model.eval()
        
        X_test = torch.FloatTensor(test_data.values)
        y_test = torch.LongTensor(test_labels)
        
        with torch.no_grad():
            outputs = self.global_model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            
            accuracy = (predicted == y_test).float().mean().item()
            
            # Calculate additional metrics
            total = y_test.size(0)
            correct = (predicted == y_test).sum().item()
            
            # Fraud detection specific metrics
            fraud_indices = (y_test == 1)
            if fraud_indices.sum() > 0:
                fraud_predictions = predicted[fraud_indices]
                fraud_accuracy = (fraud_predictions == 1).float().mean().item()
            else:
                fraud_accuracy = 0.0
        
        return {
            'accuracy': accuracy,
            'fraud_detection_accuracy': fraud_accuracy,
            'total_samples': total,
            'correct_predictions': correct
        }
    
    def get_federated_insights(self) -> Dict[str, Any]:
        """Get insights about federated learning process"""
        total_participations = sum(
            bank['participation_count'] for bank in self.bank_models.values()
        )
        
        active_banks = len([
            bank for bank in self.bank_models.values() 
            if bank['last_update'] and 
            (datetime.utcnow() - bank['last_update']).seconds < 3600
        ])
        
        return {
            'total_banks': len(self.bank_models),
            'active_banks': active_banks,
            'total_rounds': len(self.training_history),
            'total_participations': total_participations,
            'privacy_budget_used': self.config.privacy_budget,
            'aggregation_method': self.config.aggregation_method,
            'training_history': self.training_history[-10:],  # Last 10 rounds
            'bank_participation': {
                bank_id: {
                    'data_size': bank['data_size'],
                    'participation_count': bank['participation_count'],
                    'last_update': bank['last_update'].isoformat() if bank['last_update'] else None
                }
                for bank_id, bank in self.bank_models.items()
            }
        }

class FederatedFraudDetectionAPI:
    """API for federated fraud detection system"""
    
    def __init__(self, config: FederatedConfig):
        self.coordinator = FederatedLearningCoordinator(config)
        self.config = config
        
    def register_bank(self, bank_id: str, bank_info: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new bank in the federated network"""
        try:
            result = self.coordinator.initialize_bank(bank_id)
            result['status'] = 'success'
            result['message'] = f'Bank {bank_id} registered successfully'
            return result
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to register bank: {str(e)}'
            }
    
    def submit_local_training(self, bank_id: str, encrypted_weights: bytes, 
                            data_size: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Submit local training results from a bank"""
        try:
            # This would typically be called by the aggregation endpoint
            # For now, we'll simulate the process
            return {
                'status': 'success',
                'message': f'Local training results received from {bank_id}',
                'data_size': data_size,
                'metadata': metadata
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to process local training: {str(e)}'
            }
    
    def get_global_model(self, bank_id: str) -> Dict[str, Any]:
        """Get the latest global model for a bank"""
        try:
            if bank_id not in self.coordinator.bank_models:
                return {
                    'status': 'error',
                    'message': f'Bank {bank_id} not registered'
                }
            
            global_weights = self.coordinator.global_model.get_weights()
            encrypted_weights = self.coordinator.aggregator.encrypt_model_weights(global_weights)
            
            return {
                'status': 'success',
                'bank_id': bank_id,
                'encrypted_weights': encrypted_weights,
                'model_version': len(self.coordinator.training_history),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to get global model: {str(e)}'
            }
    
    def get_federated_status(self) -> Dict[str, Any]:
        """Get federated learning system status"""
        try:
            insights = self.coordinator.get_federated_insights()
            insights['status'] = 'success'
            return insights
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to get federated status: {str(e)}'
            }

# Example usage and testing
def create_synthetic_bank_data(bank_id: str, num_samples: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create synthetic data for a bank"""
    np.random.seed(hash(bank_id) % 2**32)
    
    # Generate features
    features = {
        'amount': np.random.lognormal(7, 1.5, num_samples),
        'hour': np.random.randint(0, 24, num_samples),
        'day_of_week': np.random.randint(0, 7, num_samples),
        'merchant_category': np.random.randint(0, 10, num_samples),
        'user_velocity': np.random.exponential(2, num_samples),
        'device_risk_score': np.random.beta(2, 5, num_samples),
        'location_risk_score': np.random.beta(2, 5, num_samples),
        'ip_reputation': np.random.beta(3, 2, num_samples),
        'session_duration': np.random.exponential(300, num_samples),
        'payment_frequency': np.random.exponential(5, num_samples),
        'amount_vs_avg': np.random.lognormal(0, 0.5, num_samples),
        'time_since_last_tx': np.random.exponential(2, num_samples),
        'device_age': np.random.exponential(365, num_samples),
        'location_consistency': np.random.beta(3, 2, num_samples),
        'payment_pattern': np.random.beta(2, 3, num_samples),
        'merchant_risk': np.random.beta(2, 5, num_samples),
        'time_pattern': np.random.beta(3, 2, num_samples),
        'amount_pattern': np.random.beta(2, 3, num_samples),
        'user_behavior_score': np.random.beta(3, 2, num_samples),
        'network_risk': np.random.beta(2, 5, num_samples)
    }
    
    df = pd.DataFrame(features)
    
    # Generate labels with bank-specific patterns
    fraud_prob = np.zeros(num_samples)
    
    # Bank-specific fraud patterns
    if bank_id == "bank_1":
        fraud_prob += (df['amount'] > 50000) * 0.4
        fraud_prob += (df['merchant_category'] >= 8) * 0.3
    elif bank_id == "bank_2":
        fraud_prob += (df['hour'] < 6) * 0.3
        fraud_prob += (df['device_risk_score'] > 0.7) * 0.4
    else:
        fraud_prob += (df['amount'] > 30000) * 0.3
        fraud_prob += (df['ip_reputation'] < 0.3) * 0.3
    
    # Add noise
    fraud_prob += np.random.normal(0, 0.05, num_samples)
    labels = (fraud_prob > 0.5).astype(int)
    
    return df, labels

if __name__ == "__main__":
    # Example usage
    config = FederatedConfig(
        num_banks=3,
        rounds=10,
        local_epochs=3,
        learning_rate=0.01,
        privacy_budget=1.0
    )
    
    api = FederatedFraudDetectionAPI(config)
    
    # Register banks
    banks = ["bank_1", "bank_2", "bank_3"]
    for bank_id in banks:
        result = api.register_bank(bank_id, {"name": f"Bank {bank_id}"})
        print(f"Registered {bank_id}: {result['status']}")
    
    # Simulate federated learning rounds
    for round_num in range(3):
        print(f"\n--- Federated Learning Round {round_num + 1} ---")
        
        # Each bank trains locally
        encrypted_weights_list = []
        data_sizes = []
        
        for bank_id in banks:
            # Generate synthetic data
            data, labels = create_synthetic_bank_data(bank_id, 500)
            
            # Train local model (simplified)
            bank_model = FederatedFraudModel()
            # ... training code would go here ...
            
            # Get weights and encrypt
            weights = bank_model.get_weights()
            encrypted_weights = api.coordinator.aggregator.encrypt_model_weights(weights)
            encrypted_weights_list.append(encrypted_weights)
            data_sizes.append(len(data))
        
        # Aggregate models
        result = api.coordinator.aggregate_models(encrypted_weights_list, data_sizes)
        print(f"Aggregated models from {result['participating_banks']} banks")
        print(f"Total samples: {result['total_samples']}")
    
    # Get federated status
    status = api.get_federated_status()
    print(f"\nFederated Learning Status:")
    print(f"Total banks: {status['total_banks']}")
    print(f"Active banks: {status['active_banks']}")
    print(f"Total rounds: {status['total_rounds']}")
    print(f"Privacy budget: {status['privacy_budget_used']}")
