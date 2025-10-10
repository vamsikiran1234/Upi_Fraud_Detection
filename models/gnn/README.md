# UPI Fraud Detection - Graph Neural Network (GNN) Module

This module implements a comprehensive Graph Neural Network (GNN) solution for detecting collusion patterns and coordinated fraud in UPI transactions.

## 🧠 Overview

The GNN module uses heterogeneous graph neural networks to model complex relationships between users, merchants, and devices in the UPI ecosystem. It's specifically designed to detect:

- **Collusion Rings**: Groups of users coordinating fraudulent activities
- **Merchant Fraud**: Suspicious merchant behavior patterns
- **Device Sharing**: Multiple users using the same device suspiciously
- **Temporal Coordination**: Users acting in coordinated time windows
- **Behavioral Similarity**: Users with suspiciously similar transaction patterns

## 🏗️ Architecture

### Core Components

1. **Graph Builder** (`graph_builder.py`)
   - Constructs heterogeneous graphs from transaction data
   - Extracts node features for users, merchants, and devices
   - Creates multiple edge types (transactions, temporal, similarity)
   - Handles privacy through identifier hashing

2. **GNN Model** (`gnn_model.py`)
   - Heterogeneous Graph Neural Network with multiple convolution types
   - Multi-task learning for node, edge, and graph-level predictions
   - Supports SAGEConv, GATConv, and GCNConv layers
   - Collusion detection head with specialized architectures

3. **Collusion Detector** (`collusion_detector.py`)
   - End-to-end pipeline for training and inference
   - Temporal graph construction with sliding windows
   - Automated collusion ring detection
   - Comprehensive reporting and analysis

4. **Visualization Tools** (`visualize_graph.py`)
   - Interactive dashboards for fraud analysis
   - Network visualization with NetworkX and Plotly
   - Temporal pattern analysis
   - Risk distribution visualization

## 📊 Graph Structure

### Node Types
- **Users**: UPI account holders with behavioral features
- **Merchants**: Transaction recipients with risk profiles
- **Devices**: Mobile devices with usage patterns

### Edge Types
- **Transacts**: User → Merchant (transaction relationships)
- **Uses**: User → Device (device usage)
- **Temporal**: User ↔ User (time-based coordination)
- **Similar**: User ↔ User (behavioral similarity)

### Features Extracted

#### User Features (9 dimensions)
- Transaction count, sum, mean, std deviation
- Unique merchants and devices used
- Transaction time span and velocity
- Average merchants per hour

#### Merchant Features (8 dimensions)
- Transaction volume and customer diversity
- Risk category scoring
- Temporal activity patterns
- Customer acquisition patterns

#### Device Features (7 dimensions)
- Multi-user risk indicators
- IP address diversity
- Transaction patterns
- Usage frequency metrics

## 🚀 Quick Start

### Installation

```bash
cd models/gnn
pip install -r requirements.txt
```

### Training a Model

```bash
# Train with sample data
python train_gnn.py --epochs 50 --batch-size 4 --hidden-dim 64

# Train with custom data
python train_gnn.py --data transactions.csv --labels fraud_labels.csv --epochs 100
```

### Using Pre-trained Model

```python
from collusion_detector import CollusionDetectionPipeline, create_default_config
import pandas as pd

# Initialize pipeline
config = create_default_config()
pipeline = CollusionDetectionPipeline(config)

# Load trained model
pipeline.load_model("./models/gnn/collusion_detector.pth")

# Detect collusion in new data
transactions_df = pd.read_csv("new_transactions.csv")
results = pipeline.detect_collusion(transactions_df, threshold=0.7)

# Generate report
report = pipeline.generate_report(results)
print(report)
```

### Visualization

```python
from visualize_graph import GraphVisualizer

visualizer = GraphVisualizer()
visualizer.export_report_visualizations(results, transactions_df, graph_data)
```

## 🔧 Configuration

### Graph Configuration
```python
graph_config = {
    'time_window_hours': 24,        # Time window for graph construction
    'min_transaction_amount': 100.0, # Minimum transaction amount
    'max_nodes': 10000,             # Maximum nodes per graph
    'edge_weight_threshold': 0.1,   # Minimum edge weight
    'include_temporal_edges': True, # Include time-based edges
    'include_similarity_edges': True # Include behavioral similarity edges
}
```

### Model Configuration
```python
model_config = {
    'hidden_channels': 64,    # Hidden dimension size
    'num_layers': 3,          # Number of GNN layers
    'dropout': 0.2,           # Dropout rate
    'batch_size': 4,          # Training batch size
    'num_epochs': 100,        # Training epochs
    'patience': 20            # Early stopping patience
}
```

## 📈 Model Performance

### Metrics Tracked
- **Node-level**: User and merchant fraud classification (AUC, F1-score)
- **Edge-level**: Collusion relationship detection (Precision, Recall)
- **Graph-level**: Overall fraud risk assessment
- **Ring Detection**: Collusion ring identification accuracy

### Expected Performance
- User Fraud Detection: AUC > 0.85
- Collusion Ring Detection: Precision > 0.80
- False Positive Rate: < 5%
- Processing Time: < 100ms per graph

## 🔍 Collusion Detection Algorithms

### 1. Temporal Coordination Detection
- Identifies users with transactions within short time windows
- Analyzes transaction timing patterns
- Detects coordinated "burst" activities

### 2. Behavioral Similarity Analysis
- Compares user transaction patterns
- Identifies suspiciously similar behaviors
- Uses cosine similarity on behavioral vectors

### 3. Network Community Detection
- Applies graph clustering algorithms
- Identifies tightly connected user groups
- Analyzes community structure anomalies

### 4. Multi-hop Relationship Analysis
- Examines indirect relationships between users
- Detects complex fraud networks
- Analyzes transaction flow patterns

## 📊 Output Analysis

### Collusion Ring Report
```json
{
  "ring_id": 1,
  "user_count": 5,
  "risk_level": "critical",
  "coordination_score": 0.92,
  "temporal_overlap": 0.85,
  "behavioral_similarity": 0.78,
  "recommended_action": "immediate_investigation"
}
```

### User Risk Assessment
```json
{
  "user_hash": "abc123def456",
  "fraud_probability": 0.87,
  "risk_factors": [
    "high_velocity_transactions",
    "device_sharing",
    "temporal_coordination"
  ],
  "collusion_rings": [1, 3],
  "recommendation": "flag_for_review"
}
```

## 🔒 Privacy & Security

### Data Protection
- All UPI IDs are hashed using SHA-256
- Only aggregated features are stored
- No raw transaction details in graph
- Configurable data retention policies

### Model Security
- Model weights are encrypted at rest
- Inference API uses authentication
- Audit logging for all predictions
- Differential privacy options available

## 🧪 Testing & Validation

### Unit Tests
```bash
python -m pytest tests/test_graph_builder.py
python -m pytest tests/test_gnn_model.py
python -m pytest tests/test_collusion_detector.py
```

### Integration Tests
```bash
python -m pytest tests/test_end_to_end.py
```

### Performance Tests
```bash
python tests/benchmark_inference.py
python tests/stress_test_training.py
```

## 📚 Advanced Usage

### Custom Graph Construction
```python
from graph_builder import TransactionGraphBuilder, GraphConfig

config = GraphConfig(
    time_window_hours=48,
    include_location_edges=True,
    custom_edge_types=['ip_similarity', 'amount_correlation']
)

builder = TransactionGraphBuilder(config)
graph = builder.build_heterogeneous_graph(transactions_df)
```

### Model Ensemble
```python
# Combine multiple GNN models
ensemble_results = []
for model_path in model_paths:
    pipeline.load_model(model_path)
    results = pipeline.detect_collusion(transactions_df)
    ensemble_results.append(results)

# Aggregate predictions
final_results = aggregate_ensemble_predictions(ensemble_results)
```

### Real-time Integration
```python
# Stream processing integration
from kafka import KafkaConsumer

consumer = KafkaConsumer('enriched-transactions')
for message in consumer:
    transaction_batch = parse_message(message.value)
    if len(transaction_batch) >= batch_size:
        results = pipeline.detect_collusion(transaction_batch)
        send_alerts(results)
```

## 🔄 Model Updates & Retraining

### Incremental Learning
```python
# Update model with new fraud cases
new_fraud_labels = get_confirmed_fraud_cases()
pipeline.incremental_update(new_fraud_labels)
```

### Scheduled Retraining
```python
# Weekly model retraining
from datetime import datetime, timedelta

def scheduled_retrain():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    recent_data = get_transactions(start_date, end_date)
    recent_labels = get_fraud_labels(start_date, end_date)
    
    pipeline.retrain(recent_data, recent_labels)
    pipeline.save_model(f"model_{end_date.strftime('%Y%m%d')}.pth")
```

## 🚨 Alerting & Monitoring

### Real-time Alerts
- High-risk collusion rings detected
- Sudden increase in coordinated activity
- New fraud patterns identified
- Model performance degradation

### Monitoring Metrics
- Prediction latency and throughput
- Model drift detection
- False positive/negative rates
- Graph construction performance

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues with Large Graphs**
   ```python
   # Reduce graph size
   config.max_nodes = 5000
   config.time_window_hours = 12
   ```

2. **Training Convergence Problems**
   ```python
   # Adjust learning rate and regularization
   trainer.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
   model.dropout = 0.3
   ```

3. **Poor Collusion Detection**
   ```python
   # Increase temporal edge sensitivity
   config.temporal_window_minutes = 2
   config.edge_weight_threshold = 0.05
   ```

## 📈 Performance Optimization

### GPU Acceleration
```python
# Enable CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
```

### Batch Processing
```python
# Process multiple graphs in parallel
from torch_geometric.loader import DataLoader
loader = DataLoader(graphs, batch_size=8, num_workers=4)
```

### Model Quantization
```python
# Reduce model size for deployment
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

## 🔮 Future Enhancements

- [ ] Federated learning for multi-bank collaboration
- [ ] Explainable AI for regulatory compliance
- [ ] Real-time graph streaming updates
- [ ] Advanced temporal graph neural networks
- [ ] Integration with external threat intelligence
- [ ] Automated feature engineering
- [ ] Multi-modal fraud detection (text + graph)

---

**Note**: This GNN module is designed for production use in financial fraud detection. Ensure proper testing and validation before deployment in live systems.
