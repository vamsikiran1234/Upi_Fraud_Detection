"""
Graph Neural Networks with Transformers for Hybrid Graph-Temporal Fraud Detection
Combines graph structure learning with temporal sequence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class GraphTransformerLayer(nn.Module):
    """Graph-aware transformer layer"""
    
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super(GraphTransformerLayer, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Graph attention
        self.graph_attn = GATConv(d_model, d_model // nhead, heads=nhead, dropout=dropout)
        
        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, attention_mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Graph attention
        graph_output = self.graph_attn(x.squeeze(1), edge_index)
        x = self.norm2(x + self.dropout(graph_output.unsqueeze(1)))
        
        # Feed forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x

class TemporalGraphEncoder(nn.Module):
    """Temporal graph encoder for sequence modeling"""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1):
        super(TemporalGraphEncoder, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Graph-transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, attention_mask=None):
        # Project input to model dimension
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply graph-transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, attention_mask)
        
        return x

class GraphTemporalFraudDetector(nn.Module):
    """Main model combining GNNs and Transformers for fraud detection"""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 6, num_classes: int = 2, dropout: float = 0.1):
        super(GraphTemporalFraudDetector, self).__init__()
        
        self.d_model = d_model
        
        # Temporal graph encoder
        self.temporal_encoder = TemporalGraphEncoder(
            input_dim, d_model, nhead, num_layers, dropout
        )
        
        # Graph neural network components
        self.gnn_layers = nn.ModuleList([
            GCNConv(d_model, d_model),
            GATConv(d_model, d_model // 4, heads=4),
            GraphSAGE(d_model, d_model)
        ])
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Risk score head
        self.risk_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None, attention_mask=None):
        # Temporal graph encoding
        temporal_features = self.temporal_encoder(x, edge_index, attention_mask)
        
        # Apply GNN layers
        graph_features = temporal_features.squeeze(1)
        for gnn_layer in self.gnn_layers:
            graph_features = gnn_layer(graph_features, edge_index)
            graph_features = F.relu(graph_features)
            graph_features = self.dropout(graph_features)
        
        # Temporal attention
        graph_features = graph_features.unsqueeze(1)
        attn_output, attn_weights = self.temporal_attention(
            graph_features, graph_features, graph_features
        )
        
        # Global pooling
        if batch is not None:
            # Batch-wise pooling
            mean_pool = global_mean_pool(attn_output.squeeze(1), batch)
            max_pool = global_max_pool(attn_output.squeeze(1), batch)
        else:
            # Single graph pooling
            mean_pool = torch.mean(attn_output.squeeze(1), dim=0, keepdim=True)
            max_pool = torch.max(attn_output.squeeze(1), dim=0, keepdim=True)[0]
        
        pooled_features = torch.cat([mean_pool, max_pool], dim=1)
        global_features = self.global_pool(pooled_features)
        
        # Classification and risk scoring
        classification_output = self.classifier(global_features)
        risk_score = self.risk_scorer(global_features)
        
        return {
            'classification': classification_output,
            'risk_score': risk_score,
            'attention_weights': attn_weights,
            'graph_features': graph_features,
            'global_features': global_features
        }

class GraphBuilder:
    """Builds transaction graphs from fraud data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.node_features = None
        self.edge_index = None
        self.node_labels = None
    
    def build_transaction_graph(self, transactions: pd.DataFrame, 
                              user_col: str = 'user_id',
                              merchant_col: str = 'merchant_id',
                              amount_col: str = 'amount',
                              time_col: str = 'timestamp') -> Data:
        """Build graph from transaction data"""
        
        # Create user-merchant bipartite graph
        users = transactions[user_col].unique()
        merchants = transactions[merchant_col].unique()
        
        # Create node mapping
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        merchant_to_idx = {merchant: idx + len(users) for idx, merchant in enumerate(merchants)}
        
        # Build edges (user-merchant transactions)
        edges = []
        edge_weights = []
        
        for _, row in transactions.iterrows():
            user_idx = user_to_idx[row[user_col]]
            merchant_idx = merchant_to_idx[row[merchant_col]]
            
            edges.append([user_idx, merchant_idx])
            edges.append([merchant_idx, user_idx])  # Undirected graph
            edge_weights.extend([row[amount_col], row[amount_col]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # Create node features
        node_features = []
        node_labels = []
        
        # User features
        for user in users:
            user_txs = transactions[transactions[user_col] == user]
            features = self._extract_user_features(user_txs, amount_col, time_col)
            node_features.append(features)
            node_labels.append(0)  # Users are not fraud by default
        
        # Merchant features
        for merchant in merchants:
            merchant_txs = transactions[transactions[merchant_col] == merchant]
            features = self._extract_merchant_features(merchant_txs, amount_col, time_col)
            node_features.append(features)
            node_labels.append(0)  # Merchants are not fraud by default
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        node_labels = torch.tensor(node_labels, dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weights,
            y=node_labels
        )
        
        return data
    
    def _extract_user_features(self, user_txs: pd.DataFrame, 
                             amount_col: str, time_col: str) -> List[float]:
        """Extract features for a user node"""
        features = []
        
        # Transaction statistics
        features.append(len(user_txs))  # Transaction count
        features.append(user_txs[amount_col].mean())  # Average amount
        features.append(user_txs[amount_col].std())  # Amount std
        features.append(user_txs[amount_col].max())  # Max amount
        features.append(user_txs[amount_col].min())  # Min amount
        
        # Temporal features
        if time_col in user_txs.columns:
            user_txs[time_col] = pd.to_datetime(user_txs[time_col])
            features.append(user_txs[time_col].dt.hour.mean())  # Average hour
            features.append(user_txs[time_col].dt.dayofweek.mean())  # Average day of week
            features.append((user_txs[time_col].max() - user_txs[time_col].min()).total_seconds() / 3600)  # Activity span
        else:
            features.extend([0, 0, 0])
        
        # Velocity features
        features.append(len(user_txs) / max(1, (user_txs[time_col].max() - user_txs[time_col].min()).total_seconds() / 3600))  # Transactions per hour
        
        return features
    
    def _extract_merchant_features(self, merchant_txs: pd.DataFrame,
                                 amount_col: str, time_col: str) -> List[float]:
        """Extract features for a merchant node"""
        features = []
        
        # Transaction statistics
        features.append(len(merchant_txs))  # Transaction count
        features.append(merchant_txs[amount_col].mean())  # Average amount
        features.append(merchant_txs[amount_col].std())  # Amount std
        features.append(merchant_txs[amount_col].max())  # Max amount
        features.append(merchant_txs[amount_col].min())  # Min amount
        
        # Temporal features
        if time_col in merchant_txs.columns:
            merchant_txs[time_col] = pd.to_datetime(merchant_txs[time_col])
            features.append(merchant_txs[time_col].dt.hour.mean())  # Average hour
            features.append(merchant_txs[time_col].dt.dayofweek.mean())  # Average day of week
            features.append((merchant_txs[time_col].max() - merchant_txs[time_col].min()).total_seconds() / 3600)  # Activity span
        else:
            features.extend([0, 0, 0])
        
        # Customer diversity
        features.append(merchant_txs['user_id'].nunique())  # Unique customers
        
        return features
    
    def build_temporal_graph(self, transactions: pd.DataFrame, 
                           window_size: int = 24) -> List[Data]:
        """Build temporal sequence of graphs"""
        # Sort by timestamp
        transactions = transactions.sort_values('timestamp')
        
        # Create time windows
        start_time = transactions['timestamp'].min()
        end_time = transactions['timestamp'].max()
        
        temporal_graphs = []
        current_time = start_time
        
        while current_time < end_time:
            window_end = current_time + pd.Timedelta(hours=window_size)
            
            # Get transactions in current window
            window_txs = transactions[
                (transactions['timestamp'] >= current_time) & 
                (transactions['timestamp'] < window_end)
            ]
            
            if len(window_txs) > 0:
                graph = self.build_transaction_graph(window_txs)
                temporal_graphs.append(graph)
            
            current_time = window_end
        
        return temporal_graphs

class GraphTemporalFraudDetectorAPI:
    """API for Graph-Temporal Fraud Detection"""
    
    def __init__(self, input_dim: int = 20, d_model: int = 128):
        self.model = GraphTemporalFraudDetector(input_dim, d_model)
        self.graph_builder = GraphBuilder()
        self.is_trained = False
        
    def train_model(self, transactions: pd.DataFrame, 
                   labels: pd.Series, epochs: int = 100) -> Dict[str, Any]:
        """Train the graph-temporal fraud detection model"""
        print("🚀 Training Graph-Temporal Fraud Detection Model...")
        
        # Build temporal graphs
        temporal_graphs = self.graph_builder.build_temporal_graph(transactions)
        
        if not temporal_graphs:
            raise ValueError("No temporal graphs could be built from the data")
        
        # Prepare training data
        train_loader = DataLoader(temporal_graphs, batch_size=1, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch_idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data.x, data.edge_index, data.batch)
                
                # Calculate loss (simplified - would need proper labels)
                loss = criterion(outputs['classification'], data.y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            training_losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        
        return {
            'training_losses': training_losses,
            'final_loss': training_losses[-1],
            'epochs_trained': epochs,
            'temporal_graphs': len(temporal_graphs)
        }
    
    def predict_fraud(self, transactions: pd.DataFrame) -> Dict[str, Any]:
        """Predict fraud using graph-temporal model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Build graph from transactions
        graph = self.graph_builder.build_transaction_graph(transactions)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(graph.x, graph.edge_index)
            
            # Get predictions
            classification_probs = F.softmax(outputs['classification'], dim=1)
            risk_scores = outputs['risk_score']
            
            # Extract results
            fraud_prob = classification_probs[0, 1].item()
            risk_score = risk_scores[0, 0].item()
            
            # Determine decision
            if risk_score > 0.8:
                decision = "BLOCK"
            elif risk_score > 0.5:
                decision = "CHALLENGE"
            else:
                decision = "ALLOW"
        
        return {
            'fraud_probability': fraud_prob,
            'risk_score': risk_score,
            'decision': decision,
            'confidence': max(fraud_prob, 1 - fraud_prob),
            'attention_weights': outputs['attention_weights'].cpu().numpy().tolist(),
            'model_type': 'graph_temporal'
        }
    
    def explain_prediction(self, transactions: pd.DataFrame) -> Dict[str, Any]:
        """Explain the prediction using attention weights and graph structure"""
        prediction = self.predict_fraud(transactions)
        
        # Build graph for analysis
        graph = self.graph_builder.build_transaction_graph(transactions)
        
        # Analyze attention weights
        attention_weights = prediction['attention_weights']
        
        # Find most important nodes (simplified)
        node_importance = np.mean(attention_weights, axis=1)
        top_nodes = np.argsort(node_importance)[-5:]  # Top 5 important nodes
        
        explanation = {
            'prediction': prediction,
            'important_nodes': top_nodes.tolist(),
            'node_importance_scores': node_importance.tolist(),
            'graph_structure': {
                'num_nodes': graph.x.shape[0],
                'num_edges': graph.edge_index.shape[1],
                'avg_degree': graph.edge_index.shape[1] / graph.x.shape[0]
            },
            'explanation_type': 'graph_attention'
        }
        
        return explanation

# Example usage and testing
def demonstrate_graph_temporal_fraud_detection():
    """Demonstrate graph-temporal fraud detection"""
    print("🕸️ Demonstrating Graph-Temporal Fraud Detection")
    print("=" * 60)
    
    # Create synthetic transaction data
    np.random.seed(42)
    n_transactions = 1000
    
    transactions = pd.DataFrame({
        'user_id': np.random.randint(0, 100, n_transactions),
        'merchant_id': np.random.randint(0, 50, n_transactions),
        'amount': np.random.lognormal(7, 1.5, n_transactions),
        'timestamp': pd.date_range('2024-01-01', periods=n_transactions, freq='1H'),
        'merchant_category': np.random.choice(['ecommerce', 'food', 'transport', 'crypto'], n_transactions),
        'hour': np.random.randint(0, 24, n_transactions),
        'day_of_week': np.random.randint(0, 7, n_transactions)
    })
    
    # Add some fraud patterns
    fraud_mask = (
        (transactions['amount'] > 50000) |
        (transactions['merchant_category'] == 'crypto') |
        (transactions['hour'] < 6)
    )
    
    labels = fraud_mask.astype(int)
    
    print(f"Created {len(transactions)} transactions")
    print(f"Fraud rate: {labels.mean():.3f}")
    
    # Initialize model
    api = GraphTemporalFraudDetectorAPI(input_dim=20)
    
    # Train model
    print("\n1. Training Graph-Temporal Model...")
    training_result = api.train_model(transactions, labels, epochs=50)
    print(f"   Training completed - Final loss: {training_result['final_loss']:.4f}")
    print(f"   Temporal graphs created: {training_result['temporal_graphs']}")
    
    # Make predictions
    print("\n2. Making fraud predictions...")
    test_transactions = transactions.sample(100)
    prediction = api.predict_fraud(test_transactions)
    
    print(f"   Fraud probability: {prediction['fraud_probability']:.3f}")
    print(f"   Risk score: {prediction['risk_score']:.3f}")
    print(f"   Decision: {prediction['decision']}")
    print(f"   Confidence: {prediction['confidence']:.3f}")
    
    # Explain prediction
    print("\n3. Explaining prediction...")
    explanation = api.explain_prediction(test_transactions)
    
    print(f"   Important nodes: {explanation['important_nodes']}")
    print(f"   Graph structure: {explanation['graph_structure']}")
    print(f"   Explanation type: {explanation['explanation_type']}")
    
    print("\n✅ Graph-Temporal Fraud Detection demonstration completed!")
    
    return {
        'api': api,
        'training_result': training_result,
        'prediction': prediction,
        'explanation': explanation
    }

if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_graph_temporal_fraud_detection()
    
    print(f"\n📊 Summary:")
    print(f"   Model trained: {results['api'].is_trained}")
    print(f"   Final training loss: {results['training_result']['final_loss']:.4f}")
    print(f"   Prediction decision: {results['prediction']['decision']}")
    print(f"   Risk score: {results['prediction']['risk_score']:.3f}")
