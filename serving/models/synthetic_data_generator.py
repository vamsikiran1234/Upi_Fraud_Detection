"""
Synthetic Fraud Data Generation using CTGAN and GANs
Solves class imbalance by generating realistic fraud samples
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class Generator(nn.Module):
    """Generator network for GAN"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super(Generator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, x):
        return self.network(x)

class Discriminator(nn.Module):
    """Discriminator network for GAN"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(Discriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class CTGANGenerator:
    """Conditional Tabular GAN for generating synthetic fraud data"""
    
    def __init__(self, epochs: int = 100, batch_size: int = 64, 
                 learning_rate: float = 0.0002, device: str = 'cpu'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        self.generator = None
        self.discriminator = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def _prepare_data(self, data: pd.DataFrame, target_column: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for GAN training"""
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1)
        
        return X_tensor, y_tensor
    
    def train(self, data: pd.DataFrame, target_column: str, 
              fraud_ratio: float = 0.5) -> Dict[str, Any]:
        """Train the CTGAN model"""
        print("🚀 Training CTGAN for synthetic fraud data generation...")
        
        # Prepare data
        X, y = self._prepare_data(data, target_column)
        
        # Initialize models
        input_dim = X.shape[1] + 1  # +1 for conditional input
        output_dim = X.shape[1]
        
        self.generator = Generator(input_dim, output_dim).to(self.device)
        self.discriminator = Discriminator(input_dim).to(self.device)
        
        # Optimizers
        g_optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Training history
        g_losses = []
        d_losses = []
        
        # Create data loader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            for batch_X, batch_y in dataloader:
                batch_size = batch_X.size(0)
                
                # Real data
                real_data = torch.cat([batch_X, batch_y], dim=1).to(self.device)
                real_labels = torch.ones(batch_size, 1).to(self.device)
                
                # Fake data
                noise = torch.randn(batch_size, 1).to(self.device)
                fake_input = torch.cat([noise, batch_y], dim=1).to(self.device)
                fake_data = self.generator(fake_input)
                fake_data_with_label = torch.cat([fake_data, batch_y], dim=1)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                # Real data loss
                d_real_output = self.discriminator(real_data)
                d_real_loss = criterion(d_real_output, real_labels)
                
                # Fake data loss
                d_fake_output = self.discriminator(fake_data_with_label.detach())
                d_fake_loss = criterion(d_fake_output, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                g_output = self.discriminator(fake_data_with_label)
                g_loss = criterion(g_output, real_labels)  # Generator wants to fool discriminator
                g_loss.backward()
                g_optimizer.step()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
            
            # Average losses
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{self.epochs} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
        
        self.is_trained = True
        
        return {
            'generator_losses': g_losses,
            'discriminator_losses': d_losses,
            'final_g_loss': g_losses[-1],
            'final_d_loss': d_losses[-1],
            'epochs_trained': self.epochs
        }
    
    def generate_samples(self, num_samples: int, fraud_label: int = 1) -> pd.DataFrame:
        """Generate synthetic fraud samples"""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating samples")
        
        # Generate noise
        noise = torch.randn(num_samples, 1).to(self.device)
        
        # Create conditional input (fraud label)
        fraud_conditions = torch.full((num_samples, 1), fraud_label).to(self.device)
        generator_input = torch.cat([noise, fraud_conditions], dim=1)
        
        # Generate samples
        with torch.no_grad():
            generated_features = self.generator(generator_input)
        
        # Convert to numpy
        generated_data = generated_features.cpu().numpy()
        
        # Inverse transform
        generated_data = self.scaler.inverse_transform(generated_data)
        
        # Create DataFrame
        df = pd.DataFrame(generated_data, columns=self.feature_names)
        
        # Add fraud label
        df['fraud'] = fraud_label
        
        # Decode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                # Round to nearest integer and clip to valid range
                encoded_values = np.round(df[col]).astype(int)
                encoded_values = np.clip(encoded_values, 0, len(encoder.classes_) - 1)
                df[col] = encoder.inverse_transform(encoded_values)
        
        return df
    
    def balance_dataset(self, data: pd.DataFrame, target_column: str, 
                       target_ratio: float = 0.3) -> pd.DataFrame:
        """Balance dataset by generating synthetic fraud samples"""
        print("🔄 Balancing dataset with synthetic fraud samples...")
        
        # Count current fraud samples
        fraud_count = (data[target_column] == 1).sum()
        total_count = len(data)
        current_ratio = fraud_count / total_count
        
        print(f"Current fraud ratio: {current_ratio:.3f}")
        print(f"Target fraud ratio: {target_ratio:.3f}")
        
        if current_ratio >= target_ratio:
            print("Dataset is already balanced or has sufficient fraud samples")
            return data
        
        # Calculate how many fraud samples to generate
        target_fraud_count = int(total_count * target_ratio)
        samples_to_generate = target_fraud_count - fraud_count
        
        print(f"Generating {samples_to_generate} synthetic fraud samples...")
        
        # Generate synthetic fraud samples
        synthetic_fraud = self.generate_samples(samples_to_generate, fraud_label=1)
        
        # Combine with original data
        balanced_data = pd.concat([data, synthetic_fraud], ignore_index=True)
        
        # Shuffle the dataset
        balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
        
        new_fraud_count = (balanced_data[target_column] == 1).sum()
        new_ratio = new_fraud_count / len(balanced_data)
        
        print(f"New fraud ratio: {new_ratio:.3f}")
        print(f"Total samples: {len(balanced_data)}")
        
        return balanced_data

class AdvancedSyntheticDataGenerator:
    """Advanced synthetic data generator with multiple techniques"""
    
    def __init__(self):
        self.ctgan = CTGANGenerator()
        self.is_initialized = False
    
    def create_realistic_fraud_data(self, num_samples: int = 10000) -> pd.DataFrame:
        """Create realistic fraud dataset for training GANs"""
        np.random.seed(42)
        
        # Generate base features
        data = {
            'amount': np.random.lognormal(7, 1.5, num_samples),
            'hour': np.random.randint(0, 24, num_samples),
            'day_of_week': np.random.randint(0, 7, num_samples),
            'merchant_category': np.random.choice(['ecommerce', 'food', 'transport', 'entertainment', 
                                                 'utilities', 'healthcare', 'education', 'finance', 
                                                 'crypto', 'gambling'], num_samples),
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
        
        df = pd.DataFrame(data)
        
        # Generate realistic fraud labels
        fraud_prob = np.zeros(num_samples)
        
        # High amount transactions
        fraud_prob += (df['amount'] > 50000) * 0.4
        fraud_prob += (df['amount'] > 100000) * 0.3
        
        # Night time transactions
        fraud_prob += ((df['hour'] < 6) | (df['hour'] > 22)) * 0.2
        
        # High-risk merchant categories
        high_risk_merchants = ['crypto', 'gambling']
        fraud_prob += df['merchant_category'].isin(high_risk_merchants) * 0.4
        
        # High velocity users
        fraud_prob += (df['user_velocity'] > 10) * 0.3
        
        # Suspicious device/location
        fraud_prob += (df['device_risk_score'] > 0.7) * 0.2
        fraud_prob += (df['location_risk_score'] > 0.7) * 0.2
        
        # Low IP reputation
        fraud_prob += (df['ip_reputation'] < 0.3) * 0.3
        
        # Add some noise
        fraud_prob += np.random.normal(0, 0.05, num_samples)
        
        # Convert to binary labels
        df['fraud'] = (fraud_prob > 0.5).astype(int)
        
        return df
    
    def generate_balanced_dataset(self, original_data: pd.DataFrame, 
                                 target_fraud_ratio: float = 0.3) -> pd.DataFrame:
        """Generate balanced dataset using synthetic fraud samples"""
        print("🎯 Generating balanced dataset with synthetic fraud samples...")
        
        # Train CTGAN on original data
        training_result = self.ctgan.train(original_data, 'fraud')
        print(f"CTGAN training completed - Final G Loss: {training_result['final_g_loss']:.4f}")
        
        # Generate balanced dataset
        balanced_data = self.ctgan.balance_dataset(original_data, 'fraud', target_fraud_ratio)
        
        return balanced_data
    
    def generate_adversarial_samples(self, model, data: pd.DataFrame, 
                                   num_samples: int = 1000) -> pd.DataFrame:
        """Generate adversarial samples to test model robustness"""
        print("⚔️ Generating adversarial samples for robustness testing...")
        
        # This would implement adversarial sample generation
        # For now, we'll create samples with extreme values
        adversarial_data = data.copy()
        
        # Create adversarial samples with extreme feature values
        for _ in range(num_samples):
            sample = data.sample(1).copy()
            
            # Modify features to create adversarial examples
            sample['amount'] *= np.random.uniform(10, 100)  # Very high amounts
            sample['device_risk_score'] = np.random.uniform(0.8, 1.0)  # High risk
            sample['ip_reputation'] = np.random.uniform(0.0, 0.2)  # Low reputation
            sample['user_velocity'] *= np.random.uniform(5, 20)  # High velocity
            
            adversarial_data = pd.concat([adversarial_data, sample], ignore_index=True)
        
        return adversarial_data
    
    def evaluate_synthetic_data_quality(self, original_data: pd.DataFrame, 
                                      synthetic_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate quality of synthetic data"""
        print("📊 Evaluating synthetic data quality...")
        
        metrics = {}
        
        # Statistical similarity
        for col in original_data.select_dtypes(include=[np.number]).columns:
            if col in synthetic_data.columns:
                # Mean similarity
                mean_ratio = synthetic_data[col].mean() / original_data[col].mean()
                metrics[f'{col}_mean_ratio'] = mean_ratio
                
                # Standard deviation similarity
                std_ratio = synthetic_data[col].std() / original_data[col].std()
                metrics[f'{col}_std_ratio'] = std_ratio
        
        # Distribution similarity (Kolmogorov-Smirnov test)
        from scipy import stats
        ks_scores = []
        for col in original_data.select_dtypes(include=[np.number]).columns:
            if col in synthetic_data.columns:
                ks_stat, _ = stats.ks_2samp(original_data[col], synthetic_data[col])
                ks_scores.append(ks_stat)
        
        metrics['average_ks_statistic'] = np.mean(ks_scores)
        metrics['max_ks_statistic'] = np.max(ks_scores)
        
        # Correlation preservation
        original_corr = original_data.select_dtypes(include=[np.number]).corr()
        synthetic_corr = synthetic_data.select_dtypes(include=[np.number]).corr()
        
        # Calculate correlation difference
        corr_diff = np.abs(original_corr - synthetic_corr).mean().mean()
        metrics['correlation_difference'] = corr_diff
        
        return metrics

# Example usage and testing
def demonstrate_synthetic_data_generation():
    """Demonstrate synthetic data generation capabilities"""
    print("🚀 Demonstrating Synthetic Fraud Data Generation")
    print("=" * 60)
    
    # Initialize generator
    generator = AdvancedSyntheticDataGenerator()
    
    # Create realistic fraud dataset
    print("1. Creating realistic fraud dataset...")
    fraud_data = generator.create_realistic_fraud_data(5000)
    
    fraud_ratio = fraud_data['fraud'].mean()
    print(f"   Original fraud ratio: {fraud_ratio:.3f}")
    print(f"   Total samples: {len(fraud_data)}")
    print(f"   Fraud samples: {fraud_data['fraud'].sum()}")
    
    # Generate balanced dataset
    print("\n2. Generating balanced dataset...")
    balanced_data = generator.generate_balanced_dataset(fraud_data, target_fraud_ratio=0.4)
    
    new_fraud_ratio = balanced_data['fraud'].mean()
    print(f"   New fraud ratio: {new_fraud_ratio:.3f}")
    print(f"   Total samples: {len(balanced_data)}")
    print(f"   Fraud samples: {balanced_data['fraud'].sum()}")
    
    # Evaluate synthetic data quality
    print("\n3. Evaluating synthetic data quality...")
    synthetic_samples = balanced_data[balanced_data.index >= len(fraud_data)]
    quality_metrics = generator.evaluate_synthetic_data_quality(fraud_data, synthetic_samples)
    
    print("   Quality Metrics:")
    for metric, value in quality_metrics.items():
        print(f"     {metric}: {value:.4f}")
    
    # Generate adversarial samples
    print("\n4. Generating adversarial samples...")
    adversarial_data = generator.generate_adversarial_samples(None, fraud_data, 500)
    print(f"   Generated {len(adversarial_data) - len(fraud_data)} adversarial samples")
    
    print("\n✅ Synthetic data generation demonstration completed!")
    
    return {
        'original_data': fraud_data,
        'balanced_data': balanced_data,
        'synthetic_samples': synthetic_samples,
        'adversarial_samples': adversarial_data,
        'quality_metrics': quality_metrics
    }

if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_synthetic_data_generation()
    
    # Save results
    results['balanced_data'].to_csv('balanced_fraud_dataset.csv', index=False)
    print("\n💾 Balanced dataset saved as 'balanced_fraud_dataset.csv'")
