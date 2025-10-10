"""
Reinforcement Learning for Adaptive Fraud-Blocking Policies
Learns optimal fraud detection strategies through interaction with environment
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class FraudEnvironment:
    """Environment for fraud detection RL training"""
    
    def __init__(self, transactions: pd.DataFrame, fraud_labels: pd.Series):
        self.transactions = transactions.reset_index(drop=True)
        self.fraud_labels = fraud_labels.reset_index(drop=True)
        self.current_step = 0
        self.max_steps = len(transactions)
        
        # State space: [amount, hour, merchant_category, user_velocity, device_risk, ...]
        self.state_dim = 20
        self.action_space = ['ALLOW', 'CHALLENGE', 'BLOCK']
        self.n_actions = len(self.action_space)
        
        # Reward parameters
        self.reward_params = {
            'true_positive': 10,    # Correctly blocked fraud
            'true_negative': 1,     # Correctly allowed legitimate
            'false_positive': -5,   # Incorrectly blocked legitimate
            'false_negative': -20,  # Missed fraud
            'challenge_cost': -1,   # Cost of challenging transaction
            'block_cost': -2        # Cost of blocking transaction
        }
        
        # Performance tracking
        self.performance_history = []
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_step >= self.max_steps:
            return np.zeros(self.state_dim)
        
        transaction = self.transactions.iloc[self.current_step]
        
        # Extract features
        state = np.array([
            transaction.get('amount', 0) / 100000,  # Normalized amount
            transaction.get('hour', 0) / 24,        # Normalized hour
            transaction.get('day_of_week', 0) / 7,  # Normalized day
            transaction.get('merchant_category_encoded', 0) / 10,  # Merchant category
            transaction.get('user_velocity', 0) / 20,  # User velocity
            transaction.get('device_risk_score', 0),   # Device risk
            transaction.get('location_risk_score', 0), # Location risk
            transaction.get('ip_reputation', 0),       # IP reputation
            transaction.get('session_duration', 0) / 3600,  # Session duration
            transaction.get('payment_frequency', 0) / 10,   # Payment frequency
            transaction.get('amount_vs_avg', 0),           # Amount vs average
            transaction.get('time_since_last_tx', 0) / 24, # Time since last tx
            transaction.get('device_age', 0) / 365,        # Device age
            transaction.get('location_consistency', 0),    # Location consistency
            transaction.get('payment_pattern', 0),         # Payment pattern
            transaction.get('merchant_risk', 0),           # Merchant risk
            transaction.get('time_pattern', 0),            # Time pattern
            transaction.get('amount_pattern', 0),          # Amount pattern
            transaction.get('user_behavior_score', 0),     # User behavior
            transaction.get('network_risk', 0)             # Network risk
        ])
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action and return next state, reward, done, info"""
        if self.current_step >= self.max_steps:
            return np.zeros(self.state_dim), 0, True, {}
        
        # Get current transaction info
        transaction = self.transactions.iloc[self.current_step]
        is_fraud = self.fraud_labels.iloc[self.current_step]
        action_name = self.action_space[action]
        
        # Calculate reward
        reward = self._calculate_reward(action_name, is_fraud)
        
        # Move to next step
        self.current_step += 1
        next_state = self._get_state()
        done = self.current_step >= self.max_steps
        
        # Track performance
        info = {
            'action': action_name,
            'is_fraud': is_fraud,
            'transaction_id': transaction.get('transaction_id', self.current_step),
            'amount': transaction.get('amount', 0),
            'reward': reward
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action: str, is_fraud: bool) -> float:
        """Calculate reward based on action and ground truth"""
        if action == 'ALLOW':
            if is_fraud:
                return self.reward_params['false_negative']  # Missed fraud
            else:
                return self.reward_params['true_negative']   # Correctly allowed
        elif action == 'CHALLENGE':
            reward = self.reward_params['challenge_cost']
            if is_fraud:
                reward += self.reward_params['true_positive'] * 0.5  # Partial credit
            return reward
        elif action == 'BLOCK':
            reward = self.reward_params['block_cost']
            if is_fraud:
                reward += self.reward_params['true_positive']  # Correctly blocked fraud
            else:
                reward += self.reward_params['false_positive']  # Incorrectly blocked
            return reward
        
        return 0
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.performance_history:
            return {}
        
        history = pd.DataFrame(self.performance_history)
        
        # Calculate confusion matrix
        tp = ((history['action'] == 'BLOCK') & (history['is_fraud'] == True)).sum()
        tn = ((history['action'] == 'ALLOW') & (history['is_fraud'] == False)).sum()
        fp = ((history['action'] == 'BLOCK') & (history['is_fraud'] == False)).sum()
        fn = ((history['action'] == 'ALLOW') & (history['is_fraud'] == True)).sum()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'total_reward': history['reward'].sum()
        }

class DQNNetwork(nn.Module):
    """Deep Q-Network for fraud detection"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Network agent for fraud detection"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 10000, batch_size: int = 32):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_losses': [],
            'epsilon_values': [],
            'performance_metrics': []
        }
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
    
    def train(self) -> float:
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calculate loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_history': self.training_history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_history = checkpoint['training_history']

class ReinforcementLearningFraudDetector:
    """Main RL-based fraud detection system"""
    
    def __init__(self, state_dim: int = 20, action_dim: int = 3):
        self.agent = DQNAgent(state_dim, action_dim)
        self.environment = None
        self.is_trained = False
        
    def train(self, transactions: pd.DataFrame, fraud_labels: pd.Series,
              episodes: int = 100, target_update_freq: int = 10) -> Dict[str, Any]:
        """Train the RL agent"""
        print("🤖 Training Reinforcement Learning Fraud Detection Agent...")
        
        # Create environment
        self.environment = FraudEnvironment(transactions, fraud_labels)
        
        episode_rewards = []
        episode_losses = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            episode_reward = 0
            episode_loss = 0
            step_count = 0
            
            while True:
                # Select action
                action = self.agent.select_action(state, training=True)
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action)
                
                # Store experience
                self.agent.store_experience(state, action, reward, next_state, done)
                
                # Train agent
                loss = self.agent.train()
                episode_loss += loss
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                step_count += 1
                
                if done:
                    break
            
            # Update target network
            if episode % target_update_freq == 0:
                self.agent.update_target_network()
            
            # Record episode statistics
            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss / step_count)
            
            self.agent.training_history['episode_rewards'].append(episode_reward)
            self.agent.training_history['episode_losses'].append(episode_loss / step_count)
            self.agent.training_history['epsilon_values'].append(self.agent.epsilon)
            
            if episode % 20 == 0:
                print(f"Episode {episode}/{episodes} - Reward: {episode_reward:.2f}, "
                      f"Loss: {episode_loss/step_count:.4f}, Epsilon: {self.agent.epsilon:.3f}")
        
        self.is_trained = True
        
        return {
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'final_epsilon': self.agent.epsilon,
            'total_episodes': episodes
        }
    
    def predict_fraud(self, transaction: pd.Series) -> Dict[str, Any]:
        """Predict fraud for a single transaction"""
        if not self.is_trained:
            raise ValueError("Agent must be trained before making predictions")
        
        # Create temporary environment for single transaction
        temp_df = pd.DataFrame([transaction])
        temp_labels = pd.Series([0])  # Dummy label
        
        temp_env = FraudEnvironment(temp_df, temp_labels)
        state = temp_env.reset()
        
        # Get action from trained agent
        action = self.agent.select_action(state, training=False)
        action_name = temp_env.action_space[action]
        
        # Get Q-values for explanation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.agent.q_network(state_tensor)
            q_values = q_values.squeeze().numpy()
        
        # Map to risk score
        risk_score = q_values[2] / (q_values.sum() + 1e-8)  # BLOCK action probability
        
        return {
            'action': action_name,
            'risk_score': float(risk_score),
            'q_values': q_values.tolist(),
            'confidence': float(max(q_values) / (q_values.sum() + 1e-8)),
            'model_type': 'reinforcement_learning'
        }
    
    def evaluate_performance(self, test_transactions: pd.DataFrame, 
                           test_labels: pd.Series) -> Dict[str, Any]:
        """Evaluate agent performance on test data"""
        if not self.is_trained:
            raise ValueError("Agent must be trained before evaluation")
        
        # Create test environment
        test_env = FraudEnvironment(test_transactions, test_labels)
        
        predictions = []
        actual_labels = []
        
        state = test_env.reset()
        while True:
            action = self.agent.select_action(state, training=False)
            next_state, reward, done, info = test_env.step(action)
            
            predictions.append(action)
            actual_labels.append(info['is_fraud'])
            
            state = next_state
            if done:
                break
        
        # Calculate metrics
        predictions = np.array(predictions)
        actual_labels = np.array(actual_labels)
        
        # Convert predictions to binary (BLOCK = 1, others = 0)
        binary_predictions = (predictions == 2).astype(int)  # BLOCK is action 2
        
        # Calculate confusion matrix
        tp = ((binary_predictions == 1) & (actual_labels == 1)).sum()
        tn = ((binary_predictions == 0) & (actual_labels == 0)).sum()
        fp = ((binary_predictions == 1) & (actual_labels == 0)).sum()
        fn = ((binary_predictions == 0) & (actual_labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'total_transactions': len(predictions)
        }
    
    def get_training_insights(self) -> Dict[str, Any]:
        """Get insights from training process"""
        if not self.is_trained:
            return {}
        
        history = self.agent.training_history
        
        return {
            'final_epsilon': self.agent.epsilon,
            'total_episodes': len(history['episode_rewards']),
            'average_reward': np.mean(history['episode_rewards'][-10:]) if history['episode_rewards'] else 0,
            'average_loss': np.mean(history['episode_losses'][-10:]) if history['episode_losses'] else 0,
            'epsilon_decay_curve': history['epsilon_values'],
            'reward_curve': history['episode_rewards'],
            'loss_curve': history['episode_losses']
        }

# Example usage and testing
def demonstrate_reinforcement_learning_fraud_detection():
    """Demonstrate RL-based fraud detection"""
    print("🎯 Demonstrating Reinforcement Learning Fraud Detection")
    print("=" * 60)
    
    # Create synthetic transaction data
    np.random.seed(42)
    n_transactions = 2000
    
    transactions = pd.DataFrame({
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_transactions)],
        'amount': np.random.lognormal(7, 1.5, n_transactions),
        'hour': np.random.randint(0, 24, n_transactions),
        'day_of_week': np.random.randint(0, 7, n_transactions),
        'merchant_category_encoded': np.random.randint(0, 10, n_transactions),
        'user_velocity': np.random.exponential(2, n_transactions),
        'device_risk_score': np.random.beta(2, 5, n_transactions),
        'location_risk_score': np.random.beta(2, 5, n_transactions),
        'ip_reputation': np.random.beta(3, 2, n_transactions),
        'session_duration': np.random.exponential(300, n_transactions),
        'payment_frequency': np.random.exponential(5, n_transactions),
        'amount_vs_avg': np.random.lognormal(0, 0.5, n_transactions),
        'time_since_last_tx': np.random.exponential(2, n_transactions),
        'device_age': np.random.exponential(365, n_transactions),
        'location_consistency': np.random.beta(3, 2, n_transactions),
        'payment_pattern': np.random.beta(2, 3, n_transactions),
        'merchant_risk': np.random.beta(2, 5, n_transactions),
        'time_pattern': np.random.beta(3, 2, n_transactions),
        'amount_pattern': np.random.beta(2, 3, n_transactions),
        'user_behavior_score': np.random.beta(3, 2, n_transactions),
        'network_risk': np.random.beta(2, 5, n_transactions)
    })
    
    # Create fraud labels with realistic patterns
    fraud_prob = np.zeros(n_transactions)
    fraud_prob += (transactions['amount'] > 50000) * 0.4
    fraud_prob += (transactions['hour'] < 6) * 0.2
    fraud_prob += (transactions['merchant_category_encoded'] >= 8) * 0.3
    fraud_prob += (transactions['device_risk_score'] > 0.7) * 0.2
    fraud_prob += np.random.normal(0, 0.05, n_transactions)
    
    fraud_labels = (fraud_prob > 0.5).astype(int)
    
    print(f"Created {len(transactions)} transactions")
    print(f"Fraud rate: {fraud_labels.mean():.3f}")
    
    # Split data
    train_size = int(0.8 * len(transactions))
    train_transactions = transactions[:train_size]
    train_labels = fraud_labels[:train_size]
    test_transactions = transactions[train_size:]
    test_labels = fraud_labels[train_size:]
    
    # Initialize RL fraud detector
    rl_detector = ReinforcementLearningFraudDetector()
    
    # Train the agent
    print("\n1. Training RL Agent...")
    training_result = rl_detector.train(train_transactions, train_labels, episodes=50)
    print(f"   Training completed - {training_result['total_episodes']} episodes")
    print(f"   Final epsilon: {training_result['final_epsilon']:.3f}")
    
    # Evaluate performance
    print("\n2. Evaluating Performance...")
    performance = rl_detector.evaluate_performance(test_transactions, test_labels)
    print(f"   Precision: {performance['precision']:.3f}")
    print(f"   Recall: {performance['recall']:.3f}")
    print(f"   F1 Score: {performance['f1_score']:.3f}")
    print(f"   Accuracy: {performance['accuracy']:.3f}")
    
    # Make predictions on sample transactions
    print("\n3. Making Predictions...")
    sample_transactions = test_transactions.sample(5)
    for idx, transaction in sample_transactions.iterrows():
        prediction = rl_detector.predict_fraud(transaction)
        actual_fraud = test_labels.iloc[idx]
        print(f"   Transaction {transaction['transaction_id']}:")
        print(f"     Action: {prediction['action']}")
        print(f"     Risk Score: {prediction['risk_score']:.3f}")
        print(f"     Confidence: {prediction['confidence']:.3f}")
        print(f"     Actual Fraud: {actual_fraud}")
    
    # Get training insights
    print("\n4. Training Insights...")
    insights = rl_detector.get_training_insights()
    print(f"   Average reward (last 10 episodes): {insights['average_reward']:.2f}")
    print(f"   Average loss (last 10 episodes): {insights['average_loss']:.4f}")
    print(f"   Total episodes: {insights['total_episodes']}")
    
    print("\n✅ Reinforcement Learning Fraud Detection demonstration completed!")
    
    return {
        'rl_detector': rl_detector,
        'training_result': training_result,
        'performance': performance,
        'insights': insights
    }

if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_reinforcement_learning_fraud_detection()
    
    print(f"\n📊 Summary:")
    print(f"   Model trained: {results['rl_detector'].is_trained}")
    print(f"   Test F1 Score: {results['performance']['f1_score']:.3f}")
    print(f"   Test Accuracy: {results['performance']['accuracy']:.3f}")
    print(f"   Final Epsilon: {results['insights']['final_epsilon']:.3f}")
