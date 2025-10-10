"""
UPI Fraud Detection - Graph Builder
Constructs transaction graphs for GNN-based collusion detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import hashlib
import json

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_networkx, from_networkx
import torch_geometric.transforms as T

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphConfig:
    """Configuration for graph construction"""
    time_window_hours: int = 24
    min_transaction_amount: float = 100.0
    max_nodes: int = 10000
    edge_weight_threshold: float = 0.1
    include_temporal_edges: bool = True
    include_amount_edges: bool = True
    include_location_edges: bool = True

class TransactionGraphBuilder:
    """Builds heterogeneous graphs from UPI transaction data"""
    
    def __init__(self, config: GraphConfig):
        self.config = config
        self.node_mappings = {}
        self.reverse_mappings = {}
        
    def hash_identifier(self, identifier: str) -> str:
        """Hash sensitive identifiers for privacy"""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def build_heterogeneous_graph(self, transactions_df: pd.DataFrame) -> HeteroData:
        """Build heterogeneous graph with multiple node and edge types"""
        
        # Filter transactions by time window and amount
        cutoff_time = datetime.utcnow() - timedelta(hours=self.config.time_window_hours)
        transactions_df = transactions_df[
            (transactions_df['timestamp'] >= cutoff_time) &
            (transactions_df['amount'] >= self.config.min_transaction_amount)
        ].copy()
        
        logger.info(f"Building graph from {len(transactions_df)} transactions")
        
        # Hash sensitive identifiers
        transactions_df['user_hash'] = transactions_df['upi_id'].apply(self.hash_identifier)
        transactions_df['merchant_hash'] = transactions_df['merchant_id'].apply(self.hash_identifier)
        transactions_df['device_hash'] = transactions_df['device_id'].apply(self.hash_identifier)
        
        # Create heterogeneous graph
        data = HeteroData()
        
        # Build node features and mappings
        self._build_user_nodes(data, transactions_df)
        self._build_merchant_nodes(data, transactions_df)
        self._build_device_nodes(data, transactions_df)
        
        # Build edges
        self._build_transaction_edges(data, transactions_df)
        self._build_temporal_edges(data, transactions_df)
        self._build_similarity_edges(data, transactions_df)
        
        logger.info(f"Graph built: {data}")
        return data
    
    def _build_user_nodes(self, data: HeteroData, df: pd.DataFrame):
        """Build user nodes with features"""
        user_stats = df.groupby('user_hash').agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'merchant_hash': 'nunique',
            'device_hash': 'nunique',
            'timestamp': ['min', 'max']
        }).round(2)
        
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
        user_stats = user_stats.fillna(0)
        
        # Additional behavioral features
        user_stats['transaction_span_hours'] = (
            (user_stats['timestamp_max'] - user_stats['timestamp_min']).dt.total_seconds() / 3600
        )
        user_stats['avg_merchants_per_hour'] = (
            user_stats['merchant_hash_nunique'] / (user_stats['transaction_span_hours'] + 1)
        )
        user_stats['velocity_score'] = (
            user_stats['amount_count'] / (user_stats['transaction_span_hours'] + 1)
        )
        
        # Create node mapping
        users = list(user_stats.index)
        self.node_mappings['user'] = {user: idx for idx, user in enumerate(users)}
        self.reverse_mappings['user'] = {idx: user for user, idx in self.node_mappings['user'].items()}
        
        # Node features
        feature_cols = [
            'amount_count', 'amount_sum', 'amount_mean', 'amount_std',
            'merchant_hash_nunique', 'device_hash_nunique', 'transaction_span_hours',
            'avg_merchants_per_hour', 'velocity_score'
        ]
        
        data['user'].x = torch.tensor(user_stats[feature_cols].values, dtype=torch.float)
        data['user'].num_nodes = len(users)
        
        logger.info(f"Created {len(users)} user nodes with {len(feature_cols)} features")
    
    def _build_merchant_nodes(self, data: HeteroData, df: pd.DataFrame):
        """Build merchant nodes with features"""
        merchant_stats = df.groupby('merchant_hash').agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'user_hash': 'nunique',
            'merchant_category': 'first',
            'timestamp': ['min', 'max']
        }).round(2)
        
        merchant_stats.columns = ['_'.join(col).strip() for col in merchant_stats.columns]
        merchant_stats = merchant_stats.fillna(0)
        
        # Risk scoring based on category
        risk_categories = {'gambling': 0.9, 'crypto': 0.8, 'adult': 0.7}
        merchant_stats['category_risk'] = merchant_stats['merchant_category_first'].map(
            lambda x: risk_categories.get(x, 0.3)
        )
        
        # Merchant velocity features
        merchant_stats['transaction_span_hours'] = (
            (merchant_stats['timestamp_max'] - merchant_stats['timestamp_min']).dt.total_seconds() / 3600
        )
        merchant_stats['customer_diversity'] = (
            merchant_stats['user_hash_nunique'] / (merchant_stats['amount_count'] + 1)
        )
        
        # Create node mapping
        merchants = list(merchant_stats.index)
        self.node_mappings['merchant'] = {merchant: idx for idx, merchant in enumerate(merchants)}
        self.reverse_mappings['merchant'] = {idx: merchant for merchant, idx in self.node_mappings['merchant'].items()}
        
        # Node features
        feature_cols = [
            'amount_count', 'amount_sum', 'amount_mean', 'amount_std',
            'user_hash_nunique', 'category_risk', 'transaction_span_hours',
            'customer_diversity'
        ]
        
        data['merchant'].x = torch.tensor(merchant_stats[feature_cols].values, dtype=torch.float)
        data['merchant'].num_nodes = len(merchants)
        
        logger.info(f"Created {len(merchants)} merchant nodes with {len(feature_cols)} features")
    
    def _build_device_nodes(self, data: HeteroData, df: pd.DataFrame):
        """Build device nodes with features"""
        device_stats = df.groupby('device_hash').agg({
            'amount': ['count', 'sum', 'mean'],
            'user_hash': 'nunique',
            'merchant_hash': 'nunique',
            'ip_address': 'nunique'
        }).round(2)
        
        device_stats.columns = ['_'.join(col).strip() for col in device_stats.columns]
        device_stats = device_stats.fillna(0)
        
        # Device risk indicators
        device_stats['multi_user_risk'] = (device_stats['user_hash_nunique'] > 1).astype(float)
        device_stats['ip_diversity'] = device_stats['ip_address_nunique'] / (device_stats['amount_count'] + 1)
        
        # Create node mapping
        devices = list(device_stats.index)
        self.node_mappings['device'] = {device: idx for idx, device in enumerate(devices)}
        self.reverse_mappings['device'] = {idx: device for device, idx in self.node_mappings['device'].items()}
        
        # Node features
        feature_cols = [
            'amount_count', 'amount_sum', 'amount_mean',
            'user_hash_nunique', 'merchant_hash_nunique',
            'multi_user_risk', 'ip_diversity'
        ]
        
        data['device'].x = torch.tensor(device_stats[feature_cols].values, dtype=torch.float)
        data['device'].num_nodes = len(devices)
        
        logger.info(f"Created {len(devices)} device nodes with {len(feature_cols)} features")
    
    def _build_transaction_edges(self, data: HeteroData, df: pd.DataFrame):
        """Build transaction edges between users and merchants"""
        # User -> Merchant edges (transactions)
        user_merchant_edges = df.groupby(['user_hash', 'merchant_hash']).agg({
            'amount': ['count', 'sum', 'mean'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        user_merchant_edges.columns = ['user_hash', 'merchant_hash', 'txn_count', 'total_amount', 'avg_amount', 'first_txn', 'last_txn']
        
        # Create edge indices
        user_indices = [self.node_mappings['user'][user] for user in user_merchant_edges['user_hash']]
        merchant_indices = [self.node_mappings['merchant'][merchant] for merchant in user_merchant_edges['merchant_hash']]
        
        data['user', 'transacts', 'merchant'].edge_index = torch.tensor([user_indices, merchant_indices], dtype=torch.long)
        
        # Edge features
        edge_features = user_merchant_edges[['txn_count', 'total_amount', 'avg_amount']].values
        data['user', 'transacts', 'merchant'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Reverse edges (merchant -> user)
        data['merchant', 'receives', 'user'].edge_index = torch.tensor([merchant_indices, user_indices], dtype=torch.long)
        data['merchant', 'receives', 'user'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        logger.info(f"Created {len(user_indices)} transaction edges")
        
        # User -> Device edges
        user_device_edges = df.groupby(['user_hash', 'device_hash']).agg({
            'amount': ['count', 'sum']
        }).reset_index()
        
        user_device_edges.columns = ['user_hash', 'device_hash', 'txn_count', 'total_amount']
        
        user_indices = [self.node_mappings['user'][user] for user in user_device_edges['user_hash']]
        device_indices = [self.node_mappings['device'][device] for device in user_device_edges['device_hash']]
        
        data['user', 'uses', 'device'].edge_index = torch.tensor([user_indices, device_indices], dtype=torch.long)
        edge_features = user_device_edges[['txn_count', 'total_amount']].values
        data['user', 'uses', 'device'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Reverse edges
        data['device', 'used_by', 'user'].edge_index = torch.tensor([device_indices, user_indices], dtype=torch.long)
        data['device', 'used_by', 'user'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        logger.info(f"Created {len(user_indices)} user-device edges")
    
    def _build_temporal_edges(self, data: HeteroData, df: pd.DataFrame):
        """Build temporal edges between users based on transaction timing"""
        if not self.config.include_temporal_edges:
            return
        
        # Find users with transactions within short time windows (potential coordination)
        df_sorted = df.sort_values('timestamp')
        temporal_pairs = []
        
        for i, row in df_sorted.iterrows():
            # Find transactions within 5 minutes
            time_window = timedelta(minutes=5)
            nearby_txns = df_sorted[
                (df_sorted['timestamp'] >= row['timestamp']) &
                (df_sorted['timestamp'] <= row['timestamp'] + time_window) &
                (df_sorted['user_hash'] != row['user_hash'])
            ]
            
            for _, nearby_row in nearby_txns.iterrows():
                temporal_pairs.append({
                    'user1': row['user_hash'],
                    'user2': nearby_row['user_hash'],
                    'time_diff': abs((nearby_row['timestamp'] - row['timestamp']).total_seconds()),
                    'amount_diff': abs(nearby_row['amount'] - row['amount'])
                })
        
        if temporal_pairs:
            temporal_df = pd.DataFrame(temporal_pairs)
            temporal_df = temporal_df.groupby(['user1', 'user2']).agg({
                'time_diff': 'mean',
                'amount_diff': 'mean'
            }).reset_index()
            
            # Create edge indices
            user1_indices = [self.node_mappings['user'][user] for user in temporal_df['user1']]
            user2_indices = [self.node_mappings['user'][user] for user in temporal_df['user2']]
            
            data['user', 'temporal', 'user'].edge_index = torch.tensor([user1_indices, user2_indices], dtype=torch.long)
            edge_features = temporal_df[['time_diff', 'amount_diff']].values
            data['user', 'temporal', 'user'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
            logger.info(f"Created {len(user1_indices)} temporal edges")
    
    def _build_similarity_edges(self, data: HeteroData, df: pd.DataFrame):
        """Build similarity edges between users based on behavior patterns"""
        # Calculate user behavior vectors
        user_behavior = df.groupby('user_hash').agg({
            'amount': ['mean', 'std'],
            'merchant_category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown',
            'timestamp': lambda x: x.dt.hour.mean()  # Average transaction hour
        })
        
        user_behavior.columns = ['avg_amount', 'amount_std', 'preferred_category', 'avg_hour']
        user_behavior = user_behavior.fillna(0)
        
        # Encode categorical features
        category_mapping = {cat: idx for idx, cat in enumerate(user_behavior['preferred_category'].unique())}
        user_behavior['category_encoded'] = user_behavior['preferred_category'].map(category_mapping)
        
        # Calculate pairwise similarities
        users = list(user_behavior.index)
        similarity_edges = []
        
        for i, user1 in enumerate(users):
            for j, user2 in enumerate(users[i+1:], i+1):
                # Calculate behavioral similarity
                vec1 = user_behavior.loc[user1, ['avg_amount', 'amount_std', 'category_encoded', 'avg_hour']].values
                vec2 = user_behavior.loc[user2, ['avg_amount', 'amount_std', 'category_encoded', 'avg_hour']].values
                
                # Normalize vectors
                vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
                vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
                
                similarity = np.dot(vec1_norm, vec2_norm)
                
                if similarity > self.config.edge_weight_threshold:
                    similarity_edges.append({
                        'user1': user1,
                        'user2': user2,
                        'similarity': similarity
                    })
        
        if similarity_edges:
            similarity_df = pd.DataFrame(similarity_edges)
            
            user1_indices = [self.node_mappings['user'][user] for user in similarity_df['user1']]
            user2_indices = [self.node_mappings['user'][user] for user in similarity_df['user2']]
            
            data['user', 'similar', 'user'].edge_index = torch.tensor([user1_indices, user2_indices], dtype=torch.long)
            edge_features = similarity_df[['similarity']].values
            data['user', 'similar', 'user'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
            logger.info(f"Created {len(user1_indices)} similarity edges")
    
    def add_fraud_labels(self, data: HeteroData, fraud_labels: pd.DataFrame) -> HeteroData:
        """Add fraud labels to nodes"""
        # Hash UPI IDs in fraud labels
        fraud_labels = fraud_labels.copy()
        fraud_labels['user_hash'] = fraud_labels['upi_id'].apply(self.hash_identifier)
        
        # Create label tensors for users
        user_labels = torch.zeros(data['user'].num_nodes, dtype=torch.long)
        
        for _, row in fraud_labels.iterrows():
            user_hash = row['user_hash']
            if user_hash in self.node_mappings['user']:
                user_idx = self.node_mappings['user'][user_hash]
                user_labels[user_idx] = 1 if row['is_fraud'] else 0
        
        data['user'].y = user_labels
        
        # Add merchant labels based on fraud transaction patterns
        merchant_fraud_counts = fraud_labels[fraud_labels['is_fraud'] == True].groupby('merchant_id').size()
        merchant_labels = torch.zeros(data['merchant'].num_nodes, dtype=torch.long)
        
        for merchant_id, fraud_count in merchant_fraud_counts.items():
            merchant_hash = self.hash_identifier(merchant_id)
            if merchant_hash in self.node_mappings['merchant']:
                merchant_idx = self.node_mappings['merchant'][merchant_hash]
                # Label merchant as suspicious if >5 fraud transactions
                merchant_labels[merchant_idx] = 1 if fraud_count > 5 else 0
        
        data['merchant'].y = merchant_labels
        
        logger.info(f"Added fraud labels: {user_labels.sum()} fraudulent users, {merchant_labels.sum()} suspicious merchants")
        
        return data
    
    def export_to_networkx(self, data: HeteroData) -> nx.Graph:
        """Export heterogeneous graph to NetworkX for visualization"""
        G = nx.Graph()
        
        # Add user nodes
        for idx in range(data['user'].num_nodes):
            user_hash = self.reverse_mappings['user'][idx]
            G.add_node(f"user_{idx}", 
                      type='user', 
                      hash=user_hash,
                      features=data['user'].x[idx].tolist())
        
        # Add merchant nodes
        for idx in range(data['merchant'].num_nodes):
            merchant_hash = self.reverse_mappings['merchant'][idx]
            G.add_node(f"merchant_{idx}", 
                      type='merchant', 
                      hash=merchant_hash,
                      features=data['merchant'].x[idx].tolist())
        
        # Add edges
        if 'user', 'transacts', 'merchant' in data.edge_types:
            edge_index = data['user', 'transacts', 'merchant'].edge_index
            edge_attr = data['user', 'transacts', 'merchant'].edge_attr
            
            for i in range(edge_index.size(1)):
                user_idx = edge_index[0, i].item()
                merchant_idx = edge_index[1, i].item()
                weight = edge_attr[i, 0].item()  # Transaction count
                
                G.add_edge(f"user_{user_idx}", f"merchant_{merchant_idx}", 
                          weight=weight, edge_type='transacts')
        
        logger.info(f"Exported NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def get_subgraph(self, data: HeteroData, center_nodes: List[str], hop_distance: int = 2) -> HeteroData:
        """Extract subgraph around specific nodes for focused analysis"""
        # This would implement subgraph extraction logic
        # For now, return the full graph
        logger.info(f"Subgraph extraction not implemented, returning full graph")
        return data

def create_sample_graph_data() -> pd.DataFrame:
    """Create sample transaction data for testing"""
    np.random.seed(42)
    
    # Generate sample users
    users = [f"user{i}@paytm" for i in range(100)]
    merchants = [f"merchant_{i}" for i in range(20)]
    devices = [f"device_{i}" for i in range(50)]
    categories = ['grocery', 'restaurant', 'fuel', 'gambling', 'crypto']
    
    transactions = []
    base_time = datetime.utcnow() - timedelta(hours=24)
    
    for i in range(1000):
        # Create normal and fraudulent patterns
        is_fraud_ring = i % 50 == 0  # Every 50th transaction is part of fraud ring
        
        if is_fraud_ring:
            # Fraud ring: same users, merchants, timing
            user = np.random.choice(users[:10])  # Limited user set
            merchant = np.random.choice(merchants[:3])  # Limited merchant set
            amount = np.random.uniform(5000, 20000)  # High amounts
            timestamp = base_time + timedelta(minutes=i//10)  # Clustered timing
        else:
            # Normal transactions
            user = np.random.choice(users)
            merchant = np.random.choice(merchants)
            amount = np.random.uniform(100, 2000)
            timestamp = base_time + timedelta(minutes=np.random.randint(0, 1440))
        
        transactions.append({
            'transaction_id': f"txn_{i}",
            'upi_id': user,
            'merchant_id': merchant,
            'merchant_category': np.random.choice(categories),
            'device_id': np.random.choice(devices),
            'amount': amount,
            'timestamp': timestamp,
            'ip_address': f"192.168.1.{np.random.randint(1, 255)}"
        })
    
    return pd.DataFrame(transactions)

if __name__ == "__main__":
    # Test graph building
    config = GraphConfig(
        time_window_hours=24,
        min_transaction_amount=100.0,
        max_nodes=1000
    )
    
    # Create sample data
    transactions_df = create_sample_graph_data()
    
    # Build graph
    builder = TransactionGraphBuilder(config)
    graph_data = builder.build_heterogeneous_graph(transactions_df)
    
    print(f"Built graph: {graph_data}")
    
    # Export to NetworkX
    nx_graph = builder.export_to_networkx(graph_data)
    print(f"NetworkX graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
