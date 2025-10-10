"""
UPI Fraud Detection - Graph Neural Network Model
Heterogeneous GNN for collusion detection in UPI transactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import HeteroData, Batch
import torch_geometric.transforms as T
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeteroGNNEncoder(nn.Module):
    """Heterogeneous Graph Neural Network Encoder"""
    
    def __init__(self, hidden_channels: int = 64, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Define convolution layers for each layer
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {}
            
            # User-Merchant interactions
            conv_dict[('user', 'transacts', 'merchant')] = SAGEConv(
                (-1, -1), hidden_channels, aggr='mean'
            )
            conv_dict[('merchant', 'receives', 'user')] = SAGEConv(
                (-1, -1), hidden_channels, aggr='mean'
            )
            
            # User-Device interactions
            conv_dict[('user', 'uses', 'device')] = SAGEConv(
                (-1, -1), hidden_channels, aggr='mean'
            )
            conv_dict[('device', 'used_by', 'user')] = SAGEConv(
                (-1, -1), hidden_channels, aggr='mean'
            )
            
            # User-User interactions (temporal and similarity)
            conv_dict[('user', 'temporal', 'user')] = GATConv(
                (-1, -1), hidden_channels, heads=4, concat=False, dropout=dropout
            )
            conv_dict[('user', 'similar', 'user')] = GCNConv(
                -1, hidden_channels
            )
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Batch normalization for each node type
        self.batch_norms = nn.ModuleList([
            nn.ModuleDict({
                'user': nn.BatchNorm1d(hidden_channels),
                'merchant': nn.BatchNorm1d(hidden_channels),
                'device': nn.BatchNorm1d(hidden_channels)
            }) for _ in range(num_layers)
        ])
    
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through heterogeneous GNN"""
        
        for i, conv in enumerate(self.convs):
            # Apply convolution
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply batch normalization and activation
            for node_type in x_dict:
                if node_type in self.batch_norms[i]:
                    x_dict[node_type] = self.batch_norms[i][node_type](x_dict[node_type])
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)
        
        return x_dict

class CollusionDetector(nn.Module):
    """Collusion detection head for identifying coordinated fraud"""
    
    def __init__(self, hidden_channels: int = 64, num_classes: int = 2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        
        # User classification head
        self.user_classifier = nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_channels // 2, num_classes)
        )
        
        # Merchant classification head
        self.merchant_classifier = nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_channels // 2, num_classes)
        )
        
        # Edge-level collusion detection (user-user edges)
        self.edge_classifier = nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            Linear(hidden_channels // 2, 1)
        )
        
        # Graph-level anomaly detection
        self.graph_classifier = nn.Sequential(
            Linear(hidden_channels * 3, hidden_channels),  # User + Merchant + Device pooled features
            nn.ReLU(),
            nn.Dropout(0.3),
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor], 
                batch_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for collusion detection"""
        
        outputs = {}
        
        # Node-level predictions
        if 'user' in x_dict:
            outputs['user_pred'] = self.user_classifier(x_dict['user'])
        
        if 'merchant' in x_dict:
            outputs['merchant_pred'] = self.merchant_classifier(x_dict['merchant'])
        
        # Edge-level collusion detection (user-user temporal/similarity edges)
        if ('user', 'temporal', 'user') in edge_index_dict:
            temporal_edges = edge_index_dict[('user', 'temporal', 'user')]
            if temporal_edges.size(1) > 0:
                user_embeddings = x_dict['user']
                src_embeddings = user_embeddings[temporal_edges[0]]
                dst_embeddings = user_embeddings[temporal_edges[1]]
                edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
                outputs['edge_pred'] = self.edge_classifier(edge_embeddings)
        
        # Graph-level anomaly detection
        if batch_dict is not None:
            # Pool node embeddings for graph-level representation
            user_pooled = global_mean_pool(x_dict['user'], batch_dict.get('user', torch.zeros(x_dict['user'].size(0), dtype=torch.long)))
            merchant_pooled = global_mean_pool(x_dict['merchant'], batch_dict.get('merchant', torch.zeros(x_dict['merchant'].size(0), dtype=torch.long)))
            device_pooled = global_mean_pool(x_dict['device'], batch_dict.get('device', torch.zeros(x_dict['device'].size(0), dtype=torch.long)))
            
            graph_embedding = torch.cat([user_pooled, merchant_pooled, device_pooled], dim=1)
            outputs['graph_pred'] = self.graph_classifier(graph_embedding)
        
        return outputs

class UPIFraudGNN(nn.Module):
    """Complete UPI Fraud Detection GNN Model"""
    
    def __init__(self, 
                 node_feature_dims: Dict[str, int],
                 hidden_channels: int = 64,
                 num_layers: int = 3,
                 num_classes: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.node_feature_dims = node_feature_dims
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        
        # Input projection layers for each node type
        self.input_projections = nn.ModuleDict()
        for node_type, input_dim in node_feature_dims.items():
            self.input_projections[node_type] = Linear(input_dim, hidden_channels)
        
        # Heterogeneous GNN encoder
        self.encoder = HeteroGNNEncoder(hidden_channels, num_layers, dropout)
        
        # Collusion detection head
        self.detector = CollusionDetector(hidden_channels, num_classes)
        
        # Loss weights for multi-task learning
        self.loss_weights = {
            'user': 1.0,
            'merchant': 0.5,
            'edge': 0.8,
            'graph': 0.3
        }
    
    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Forward pass through complete model"""
        
        # Project input features to hidden dimension
        x_dict = {}
        for node_type in data.node_types:
            if node_type in self.input_projections and hasattr(data[node_type], 'x'):
                x_dict[node_type] = self.input_projections[node_type](data[node_type].x)
        
        # Encode with heterogeneous GNN
        x_dict = self.encoder(x_dict, data.edge_index_dict)
        
        # Detect collusion patterns
        batch_dict = getattr(data, 'batch_dict', None)
        outputs = self.detector(x_dict, data.edge_index_dict, batch_dict)
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-task loss"""
        total_loss = 0.0
        loss_components = {}
        
        # User node classification loss
        if 'user_pred' in outputs and 'user' in targets:
            user_loss = F.cross_entropy(outputs['user_pred'], targets['user'])
            total_loss += self.loss_weights['user'] * user_loss
            loss_components['user_loss'] = user_loss.item()
        
        # Merchant node classification loss
        if 'merchant_pred' in outputs and 'merchant' in targets:
            merchant_loss = F.cross_entropy(outputs['merchant_pred'], targets['merchant'])
            total_loss += self.loss_weights['merchant'] * merchant_loss
            loss_components['merchant_loss'] = merchant_loss.item()
        
        # Edge-level collusion loss
        if 'edge_pred' in outputs and 'edge' in targets:
            edge_loss = F.binary_cross_entropy_with_logits(outputs['edge_pred'].squeeze(), targets['edge'].float())
            total_loss += self.loss_weights['edge'] * edge_loss
            loss_components['edge_loss'] = edge_loss.item()
        
        # Graph-level anomaly loss
        if 'graph_pred' in outputs and 'graph' in targets:
            graph_loss = F.binary_cross_entropy_with_logits(outputs['graph_pred'].squeeze(), targets['graph'].float())
            total_loss += self.loss_weights['graph'] * graph_loss
            loss_components['graph_loss'] = graph_loss.item()
        
        loss_components['total_loss'] = total_loss.item()
        return total_loss, loss_components

class GNNTrainer:
    """Trainer for UPI Fraud Detection GNN"""
    
    def __init__(self, model: UPIFraudGNN, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        loss_components = {'user_loss': 0.0, 'merchant_loss': 0.0, 'edge_loss': 0.0, 'graph_loss': 0.0}
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch)
            
            # Prepare targets
            targets = {}
            if hasattr(batch['user'], 'y'):
                targets['user'] = batch['user'].y
            if hasattr(batch['merchant'], 'y'):
                targets['merchant'] = batch['merchant'].y
            
            # Compute loss
            loss, batch_loss_components = self.model.compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in batch_loss_components.items():
                if key in loss_components:
                    loss_components[key] += value
            num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {'total_loss': avg_loss, **loss_components}
    
    def evaluate(self, val_loader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_user_preds = []
        all_user_targets = []
        all_merchant_preds = []
        all_merchant_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                outputs = self.model(batch)
                
                # Prepare targets
                targets = {}
                if hasattr(batch['user'], 'y'):
                    targets['user'] = batch['user'].y
                    all_user_targets.extend(batch['user'].y.cpu().numpy())
                    all_user_preds.extend(F.softmax(outputs['user_pred'], dim=1)[:, 1].cpu().numpy())
                
                if hasattr(batch['merchant'], 'y'):
                    targets['merchant'] = batch['merchant'].y
                    all_merchant_targets.extend(batch['merchant'].y.cpu().numpy())
                    all_merchant_preds.extend(F.softmax(outputs['merchant_pred'], dim=1)[:, 1].cpu().numpy())
                
                loss, _ = self.model.compute_loss(outputs, targets)
                total_loss += loss.item()
        
        metrics = {'val_loss': total_loss / len(val_loader)}
        
        # Calculate metrics
        if all_user_preds and all_user_targets:
            user_auc = roc_auc_score(all_user_targets, all_user_preds)
            metrics['user_auc'] = user_auc
            
            user_preds_binary = [1 if p > 0.5 else 0 for p in all_user_preds]
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_user_targets, user_preds_binary, average='binary'
            )
            metrics.update({
                'user_precision': precision,
                'user_recall': recall,
                'user_f1': f1
            })
        
        if all_merchant_preds and all_merchant_targets:
            merchant_auc = roc_auc_score(all_merchant_targets, all_merchant_preds)
            metrics['merchant_auc'] = merchant_auc
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs: int = 100, early_stopping_patience: int = 20):
        """Complete training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])
            
            # Early stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_gnn_model.pth')
            else:
                patience_counter += 1
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.4f}, "
                           f"Val Loss: {val_metrics['val_loss']:.4f}")
                if 'user_auc' in val_metrics:
                    logger.info(f"User AUC: {val_metrics['user_auc']:.4f}, "
                               f"User F1: {val_metrics['user_f1']:.4f}")
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_gnn_model.pth'))
        logger.info("Training completed")

class CollusionAnalyzer:
    """Analyze collusion patterns from trained GNN"""
    
    def __init__(self, model: UPIFraudGNN, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def detect_collusion_rings(self, data: HeteroData, threshold: float = 0.7) -> List[List[str]]:
        """Detect collusion rings from graph structure and predictions"""
        with torch.no_grad():
            data = data.to(self.device)
            outputs = self.model(data)
            
            # Get user predictions
            if 'user_pred' in outputs:
                user_probs = F.softmax(outputs['user_pred'], dim=1)[:, 1]  # Fraud probability
                suspicious_users = (user_probs > threshold).nonzero().squeeze()
                
                # Analyze connections between suspicious users
                collusion_rings = []
                
                if ('user', 'temporal', 'user') in data.edge_index_dict:
                    temporal_edges = data.edge_index_dict[('user', 'temporal', 'user')]
                    
                    # Find connected components of suspicious users
                    import networkx as nx
                    G = nx.Graph()
                    
                    for i in range(temporal_edges.size(1)):
                        src, dst = temporal_edges[0, i].item(), temporal_edges[1, i].item()
                        if src in suspicious_users and dst in suspicious_users:
                            G.add_edge(src, dst)
                    
                    # Extract connected components as collusion rings
                    for component in nx.connected_components(G):
                        if len(component) >= 2:  # At least 2 users in a ring
                            collusion_rings.append(list(component))
                
                return collusion_rings
        
        return []
    
    def explain_predictions(self, data: HeteroData, node_idx: int, node_type: str = 'user') -> Dict[str, Any]:
        """Explain predictions for a specific node"""
        # This would implement GNN explainability (e.g., GNNExplainer)
        # For now, return basic information
        
        with torch.no_grad():
            data = data.to(self.device)
            outputs = self.model(data)
            
            explanation = {
                'node_idx': node_idx,
                'node_type': node_type,
                'prediction': None,
                'confidence': None,
                'important_neighbors': [],
                'feature_importance': []
            }
            
            if node_type == 'user' and 'user_pred' in outputs:
                pred_probs = F.softmax(outputs['user_pred'], dim=1)
                explanation['prediction'] = pred_probs[node_idx, 1].item()
                explanation['confidence'] = pred_probs[node_idx].max().item()
            
            return explanation

if __name__ == "__main__":
    # Test model creation
    node_feature_dims = {
        'user': 9,      # From graph_builder.py user features
        'merchant': 8,  # From graph_builder.py merchant features  
        'device': 7     # From graph_builder.py device features
    }
    
    model = UPIFraudGNN(
        node_feature_dims=node_feature_dims,
        hidden_channels=64,
        num_layers=3,
        num_classes=2
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass with dummy data
    from torch_geometric.data import HeteroData
    
    data = HeteroData()
    data['user'].x = torch.randn(100, 9)
    data['merchant'].x = torch.randn(20, 8)
    data['device'].x = torch.randn(50, 7)
    
    # Add some dummy edges
    data['user', 'transacts', 'merchant'].edge_index = torch.randint(0, 100, (2, 200))
    data['merchant', 'receives', 'user'].edge_index = torch.randint(0, 20, (2, 200))
    
    outputs = model(data)
    print(f"Model outputs: {list(outputs.keys())}")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
