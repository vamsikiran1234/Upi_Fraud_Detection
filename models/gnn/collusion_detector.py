"""
UPI Fraud Detection - Collusion Detection Pipeline
End-to-end pipeline for detecting coordinated fraud using GNNs
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

from .graph_builder import TransactionGraphBuilder, GraphConfig
from .gnn_model import UPIFraudGNN, GNNTrainer, CollusionAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollusionDetectionPipeline:
    """Complete pipeline for collusion detection in UPI transactions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph_config = GraphConfig(**config.get('graph', {}))
        self.model_config = config.get('model', {})
        
        self.graph_builder = TransactionGraphBuilder(self.graph_config)
        self.model = None
        self.trainer = None
        self.analyzer = None
        
        # Model paths
        self.model_dir = Path(config.get('model_dir', './models/gnn'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_training_data(self, transactions_df: pd.DataFrame, 
                            fraud_labels_df: pd.DataFrame) -> List[HeteroData]:
        """Prepare training data from transaction history"""
        logger.info("Preparing training data...")
        
        # Split data into time windows for temporal training
        transactions_df = transactions_df.sort_values('timestamp')
        start_time = transactions_df['timestamp'].min()
        end_time = transactions_df['timestamp'].max()
        
        # Create sliding windows
        window_size = timedelta(hours=self.graph_config.time_window_hours)
        slide_size = timedelta(hours=6)  # 6-hour slides
        
        training_graphs = []
        current_time = start_time
        
        while current_time + window_size <= end_time:
            window_end = current_time + window_size
            
            # Filter transactions for this window
            window_txns = transactions_df[
                (transactions_df['timestamp'] >= current_time) &
                (transactions_df['timestamp'] < window_end)
            ]
            
            if len(window_txns) >= 100:  # Minimum transactions for meaningful graph
                # Build graph for this window
                graph_data = self.graph_builder.build_heterogeneous_graph(window_txns)
                
                # Add fraud labels
                window_labels = fraud_labels_df[
                    fraud_labels_df['transaction_id'].isin(window_txns['transaction_id'])
                ]
                
                if len(window_labels) > 0:
                    graph_data = self.graph_builder.add_fraud_labels(graph_data, window_labels)
                    training_graphs.append(graph_data)
            
            current_time += slide_size
        
        logger.info(f"Created {len(training_graphs)} training graphs")
        return training_graphs
    
    def create_model(self, sample_graph: HeteroData) -> UPIFraudGNN:
        """Create GNN model based on graph structure"""
        # Determine node feature dimensions from sample graph
        node_feature_dims = {}
        for node_type in sample_graph.node_types:
            if hasattr(sample_graph[node_type], 'x'):
                node_feature_dims[node_type] = sample_graph[node_type].x.size(1)
        
        logger.info(f"Node feature dimensions: {node_feature_dims}")
        
        # Create model
        model = UPIFraudGNN(
            node_feature_dims=node_feature_dims,
            hidden_channels=self.model_config.get('hidden_channels', 64),
            num_layers=self.model_config.get('num_layers', 3),
            num_classes=2,
            dropout=self.model_config.get('dropout', 0.2)
        )
        
        return model
    
    def train_model(self, training_graphs: List[HeteroData], 
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the GNN model"""
        logger.info("Starting model training...")
        
        # Split into train/validation
        split_idx = int(len(training_graphs) * (1 - validation_split))
        train_graphs = training_graphs[:split_idx]
        val_graphs = training_graphs[split_idx:]
        
        # Create data loaders
        train_loader = DataLoader(train_graphs, batch_size=self.model_config.get('batch_size', 4), shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=self.model_config.get('batch_size', 4), shuffle=False)
        
        # Create model if not exists
        if self.model is None:
            self.model = self.create_model(training_graphs[0])
        
        # Create trainer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.trainer = GNNTrainer(self.model, device)
        
        # Train model
        training_history = self.trainer.train(
            train_loader, 
            val_loader, 
            num_epochs=self.model_config.get('num_epochs', 100),
            early_stopping_patience=self.model_config.get('patience', 20)
        )
        
        # Save model
        model_path = self.model_dir / 'collusion_detector.pth'
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        return training_history
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load trained model"""
        if model_path is None:
            model_path = self.model_dir / 'collusion_detector.pth'
        
        try:
            # Load model architecture (need sample graph for this)
            # In practice, you'd save the architecture config separately
            logger.info(f"Loading model from {model_path}")
            
            if self.model is None:
                logger.error("Model architecture not initialized. Train model first or provide sample graph.")
                return False
            
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.analyzer = CollusionAnalyzer(self.model)
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def detect_collusion(self, transactions_df: pd.DataFrame, 
                        threshold: float = 0.7) -> Dict[str, Any]:
        """Detect collusion in new transaction data"""
        if self.model is None or self.analyzer is None:
            raise ValueError("Model not loaded. Train or load model first.")
        
        logger.info("Detecting collusion patterns...")
        
        # Build graph from recent transactions
        graph_data = self.graph_builder.build_heterogeneous_graph(transactions_df)
        
        # Get predictions
        device = next(self.model.parameters()).device
        graph_data = graph_data.to(device)
        
        with torch.no_grad():
            outputs = self.model(graph_data)
        
        # Extract results
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_users': graph_data['user'].num_nodes,
            'total_merchants': graph_data['merchant'].num_nodes,
            'total_devices': graph_data['device'].num_nodes,
            'suspicious_users': [],
            'suspicious_merchants': [],
            'collusion_rings': [],
            'risk_scores': {}
        }
        
        # Analyze user predictions
        if 'user_pred' in outputs:
            user_probs = torch.softmax(outputs['user_pred'], dim=1)[:, 1].cpu().numpy()
            suspicious_user_indices = np.where(user_probs > threshold)[0]
            
            results['suspicious_users'] = [
                {
                    'user_idx': int(idx),
                    'user_hash': self.graph_builder.reverse_mappings['user'][idx],
                    'fraud_probability': float(user_probs[idx]),
                    'risk_level': 'high' if user_probs[idx] > 0.9 else 'medium'
                }
                for idx in suspicious_user_indices
            ]
            
            results['risk_scores']['user_risk_distribution'] = {
                'mean': float(np.mean(user_probs)),
                'std': float(np.std(user_probs)),
                'max': float(np.max(user_probs)),
                'suspicious_count': len(suspicious_user_indices)
            }
        
        # Analyze merchant predictions
        if 'merchant_pred' in outputs:
            merchant_probs = torch.softmax(outputs['merchant_pred'], dim=1)[:, 1].cpu().numpy()
            suspicious_merchant_indices = np.where(merchant_probs > threshold)[0]
            
            results['suspicious_merchants'] = [
                {
                    'merchant_idx': int(idx),
                    'merchant_hash': self.graph_builder.reverse_mappings['merchant'][idx],
                    'fraud_probability': float(merchant_probs[idx]),
                    'risk_level': 'high' if merchant_probs[idx] > 0.9 else 'medium'
                }
                for idx in suspicious_merchant_indices
            ]
        
        # Detect collusion rings
        collusion_rings = self.analyzer.detect_collusion_rings(graph_data, threshold)
        results['collusion_rings'] = [
            {
                'ring_id': i,
                'user_indices': ring,
                'user_hashes': [self.graph_builder.reverse_mappings['user'][idx] for idx in ring],
                'ring_size': len(ring),
                'risk_level': 'critical' if len(ring) > 5 else 'high'
            }
            for i, ring in enumerate(collusion_rings)
        ]
        
        # Edge-level collusion analysis
        if 'edge_pred' in outputs and ('user', 'temporal', 'user') in graph_data.edge_index_dict:
            edge_probs = torch.sigmoid(outputs['edge_pred']).cpu().numpy()
            temporal_edges = graph_data.edge_index_dict[('user', 'temporal', 'user')].cpu().numpy()
            
            suspicious_edges = []
            for i, prob in enumerate(edge_probs):
                if prob > threshold:
                    src_idx, dst_idx = temporal_edges[0, i], temporal_edges[1, i]
                    suspicious_edges.append({
                        'src_user': self.graph_builder.reverse_mappings['user'][src_idx],
                        'dst_user': self.graph_builder.reverse_mappings['user'][dst_idx],
                        'collusion_probability': float(prob),
                        'edge_type': 'temporal'
                    })
            
            results['suspicious_edges'] = suspicious_edges
        
        logger.info(f"Collusion detection completed: {len(results['suspicious_users'])} suspicious users, "
                   f"{len(results['collusion_rings'])} collusion rings detected")
        
        return results
    
    def generate_report(self, detection_results: Dict[str, Any], 
                       output_path: Optional[str] = None) -> str:
        """Generate detailed collusion detection report"""
        
        report = f"""
# UPI Fraud Collusion Detection Report
Generated: {detection_results['timestamp']}

## Summary
- Total Users Analyzed: {detection_results['total_users']}
- Total Merchants Analyzed: {detection_results['total_merchants']}
- Total Devices Analyzed: {detection_results['total_devices']}
- Suspicious Users Detected: {len(detection_results['suspicious_users'])}
- Suspicious Merchants Detected: {len(detection_results['suspicious_merchants'])}
- Collusion Rings Detected: {len(detection_results['collusion_rings'])}

## Risk Assessment
"""
        
        if 'user_risk_distribution' in detection_results['risk_scores']:
            risk_dist = detection_results['risk_scores']['user_risk_distribution']
            report += f"""
### User Risk Distribution
- Average Risk Score: {risk_dist['mean']:.3f}
- Risk Score Standard Deviation: {risk_dist['std']:.3f}
- Maximum Risk Score: {risk_dist['max']:.3f}
- Users Above Threshold: {risk_dist['suspicious_count']}
"""
        
        # Detailed findings
        report += "\n## Detailed Findings\n"
        
        # High-risk users
        if detection_results['suspicious_users']:
            report += "\n### High-Risk Users\n"
            for user in detection_results['suspicious_users'][:10]:  # Top 10
                report += f"- User {user['user_hash']}: {user['fraud_probability']:.3f} ({user['risk_level']} risk)\n"
        
        # Collusion rings
        if detection_results['collusion_rings']:
            report += "\n### Collusion Rings\n"
            for ring in detection_results['collusion_rings']:
                report += f"- Ring {ring['ring_id']}: {ring['ring_size']} users ({ring['risk_level']} risk)\n"
                report += f"  Users: {', '.join(ring['user_hashes'][:5])}{'...' if len(ring['user_hashes']) > 5 else ''}\n"
        
        # Suspicious edges
        if 'suspicious_edges' in detection_results and detection_results['suspicious_edges']:
            report += "\n### Suspicious Connections\n"
            for edge in detection_results['suspicious_edges'][:10]:  # Top 10
                report += f"- {edge['src_user']} ↔ {edge['dst_user']}: {edge['collusion_probability']:.3f}\n"
        
        # Recommendations
        report += """
## Recommendations
1. **Immediate Actions:**
   - Flag high-risk users for manual review
   - Implement additional verification for collusion ring members
   - Monitor suspicious merchant transactions closely

2. **Investigation Priorities:**
   - Focus on collusion rings with >3 members
   - Investigate temporal patterns in suspicious connections
   - Cross-reference with external fraud databases

3. **Preventive Measures:**
   - Implement real-time collusion detection alerts
   - Enhance transaction velocity controls
   - Deploy behavioral biometrics for high-risk users
"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def save_pipeline_config(self):
        """Save pipeline configuration"""
        config_path = self.model_dir / 'pipeline_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        # Save node mappings
        mappings_path = self.model_dir / 'node_mappings.pkl'
        with open(mappings_path, 'wb') as f:
            pickle.dump({
                'node_mappings': self.graph_builder.node_mappings,
                'reverse_mappings': self.graph_builder.reverse_mappings
            }, f)
        
        logger.info(f"Pipeline configuration saved to {self.model_dir}")

def create_default_config() -> Dict[str, Any]:
    """Create default configuration for collusion detection"""
    return {
        'graph': {
            'time_window_hours': 24,
            'min_transaction_amount': 100.0,
            'max_nodes': 10000,
            'edge_weight_threshold': 0.1,
            'include_temporal_edges': True,
            'include_amount_edges': True,
            'include_location_edges': True
        },
        'model': {
            'hidden_channels': 64,
            'num_layers': 3,
            'dropout': 0.2,
            'batch_size': 4,
            'num_epochs': 100,
            'patience': 20
        },
        'model_dir': './models/gnn'
    }

if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    pipeline = CollusionDetectionPipeline(config)
    
    # Create sample data for testing
    from .graph_builder import create_sample_graph_data
    
    transactions_df = create_sample_graph_data()
    
    # Create sample fraud labels
    fraud_labels = pd.DataFrame({
        'transaction_id': transactions_df['transaction_id'].sample(50),
        'upi_id': transactions_df['upi_id'].sample(50),
        'merchant_id': transactions_df['merchant_id'].sample(50),
        'is_fraud': np.random.choice([True, False], 50, p=[0.1, 0.9])
    })
    
    print("Testing collusion detection pipeline...")
    
    # Prepare training data
    training_graphs = pipeline.prepare_training_data(transactions_df, fraud_labels)
    print(f"Created {len(training_graphs)} training graphs")
    
    if training_graphs:
        # Train model
        pipeline.train_model(training_graphs)
        
        # Test detection
        detection_results = pipeline.detect_collusion(transactions_df)
        
        # Generate report
        report = pipeline.generate_report(detection_results)
        print("\n" + "="*50)
        print(report)
