# UPI Fraud Detection System - Data Flow Sequence Diagram

## Complete End-to-End Data Flow

```mermaid
sequenceDiagram
    participant UPI as UPI Gateway
    participant Kafka as Kafka Broker
    participant Spark as Spark Streaming
    participant Redis as Redis Cache
    participant PG as PostgreSQL
    participant API as FastAPI Service
    participant GNN as GNN Service
    participant Dashboard as Dashboard UI
    participant Analyst as Fraud Analyst

    Note over UPI, Analyst: Real-time UPI Fraud Detection Flow

    %% Transaction Ingestion
    UPI->>Kafka: 1. Raw Transaction Event
    Note right of UPI: {transaction_id, upi_id, amount,<br/>merchant_id, device_id, timestamp}

    %% Stream Processing
    Kafka->>Spark: 2. Consume Raw Transaction
    Spark->>Spark: 3. Feature Extraction
    Note right of Spark: - Velocity features<br/>- Behavioral patterns<br/>- Location analysis<br/>- Device fingerprinting

    %% Feature Store Updates
    Spark->>Redis: 4. Store Real-time Features
    Note right of Redis: TTL: 1 hour<br/>Key: user_features:{hash}
    
    Spark->>PG: 5. Store Historical Features
    Note right of PG: Persistent storage<br/>for model training

    Spark->>Kafka: 6. Publish Enriched Transaction
    Note right of Kafka: Topic: enriched-transactions

    %% Real-time Inference
    Kafka->>API: 7. Consume Enriched Transaction
    API->>Redis: 8. Fetch User Features
    Redis-->>API: 9. Return Cached Features
    
    API->>API: 10. Ensemble Model Inference
    Note right of API: - XGBoost<br/>- LSTM<br/>- Autoencoder<br/>- SHAP Explanations

    API->>API: 11. Decision Engine
    Note right of API: Risk score + Business rules<br/>→ ALLOW/CHALLENGE/BLOCK

    %% Collusion Detection (Batch)
    rect rgb(255, 240, 240)
        Note over PG, GNN: Periodic Collusion Analysis (Every 30 minutes)
        PG->>GNN: 12. Fetch Recent Transactions
        GNN->>GNN: 13. Build Transaction Graph
        Note right of GNN: - User-Merchant-Device nodes<br/>- Temporal & similarity edges
        
        GNN->>GNN: 14. GNN Inference
        Note right of GNN: Detect collusion rings<br/>and coordinated fraud
        
        GNN->>PG: 15. Store Collusion Results
        GNN->>Kafka: 16. Publish High-Risk Alerts
    end

    %% Response & Actions
    API->>UPI: 17. Return Decision
    Note left of API: {decision: "BLOCK",<br/>risk_score: 0.95,<br/>explanation: {...}}

    alt High Risk Transaction
        API->>Kafka: 18. Publish Alert
        Note right of Kafka: Topic: fraud-alerts
        
        Kafka->>Dashboard: 19. Real-time Alert
        Dashboard->>Analyst: 20. Notify Analyst
        
        Analyst->>Dashboard: 21. Investigate Case
        Dashboard->>PG: 22. Fetch Transaction Details
        Dashboard->>API: 23. Get SHAP Explanations
        
        Analyst->>Dashboard: 24. Label Decision
        Dashboard->>PG: 25. Store Feedback
        Note right of PG: For model retraining
    end

    %% Monitoring & Health Checks
    rect rgb(240, 255, 240)
        Note over API, Dashboard: System Monitoring
        API->>API: Health Check
        Spark->>Spark: Stream Monitoring
        GNN->>GNN: Model Performance
        Dashboard->>Dashboard: Alert Dashboard
    end
```

## Detailed Component Interactions

### 1. Transaction Ingestion Flow

```mermaid
sequenceDiagram
    participant Gateway as UPI Gateway
    participant Ingestion as Ingestion API
    participant Kafka as Kafka
    participant Validation as Data Validator

    Gateway->>Ingestion: POST /ingest/transaction
    Note right of Gateway: Raw transaction data<br/>with PII

    Ingestion->>Validation: Validate Schema
    Validation-->>Ingestion: Validation Result
    
    alt Valid Transaction
        Ingestion->>Ingestion: Hash Sensitive Data
        Note right of Ingestion: UPI ID → SHA256 hash
        
        Ingestion->>Kafka: Publish to raw-transactions
        Ingestion-->>Gateway: 202 Accepted
    else Invalid Transaction
        Ingestion-->>Gateway: 400 Bad Request
    end
```

### 2. Feature Engineering Pipeline

```mermaid
sequenceDiagram
    participant Kafka as Kafka
    participant Spark as Spark Streaming
    participant FeatureStore as Feature Store
    participant External as External APIs

    Kafka->>Spark: Raw Transaction Stream
    
    par Feature Extraction
        Spark->>Spark: Basic Features
        Note right of Spark: Amount, time, category
    and
        Spark->>Spark: Velocity Features
        Note right of Spark: Windowed aggregations<br/>1h, 24h windows
    and
        Spark->>External: IP Geolocation
        External-->>Spark: Location Data
    and
        Spark->>FeatureStore: Historical Lookup
        FeatureStore-->>Spark: User History
    end

    Spark->>Spark: Combine All Features
    
    par Store Features
        Spark->>FeatureStore: Update Redis Cache
    and
        Spark->>FeatureStore: Update PostgreSQL
    end

    Spark->>Kafka: Enriched Transaction
```

### 3. Real-time Inference Pipeline

```mermaid
sequenceDiagram
    participant API as FastAPI
    participant FeatureStore as Feature Store
    participant Models as ML Models
    participant DecisionEngine as Decision Engine
    participant Explainer as SHAP Explainer

    API->>FeatureStore: Get User Features
    FeatureStore-->>API: Feature Vector

    par Model Ensemble
        API->>Models: XGBoost Prediction
        Models-->>API: Score 1
    and
        API->>Models: LSTM Prediction  
        Models-->>API: Score 2
    and
        API->>Models: Autoencoder Anomaly
        Models-->>API: Score 3
    end

    API->>API: Ensemble Aggregation
    Note right of API: Weighted voting<br/>or meta-learner

    API->>Explainer: Generate SHAP Values
    Explainer-->>API: Feature Importance

    API->>DecisionEngine: Apply Business Rules
    Note right of DecisionEngine: Risk thresholds<br/>Velocity limits<br/>Amount caps

    DecisionEngine-->>API: Final Decision
```

### 4. Graph Neural Network Collusion Detection

```mermaid
sequenceDiagram
    participant Scheduler as Batch Scheduler
    participant GraphBuilder as Graph Builder
    participant GNN as GNN Model
    participant Analyzer as Collusion Analyzer
    participant AlertSystem as Alert System

    Scheduler->>GraphBuilder: Trigger Analysis
    Note right of Scheduler: Every 30 minutes

    GraphBuilder->>GraphBuilder: Fetch Recent Data
    Note right of GraphBuilder: Last 24h transactions

    GraphBuilder->>GraphBuilder: Build Heterogeneous Graph
    Note right of GraphBuilder: Users, Merchants, Devices<br/>+ Temporal/Similarity edges

    GraphBuilder->>GNN: Graph Data
    
    GNN->>GNN: Node Embeddings
    Note right of GNN: User/Merchant representations

    GNN->>GNN: Edge Predictions
    Note right of GNN: Collusion probabilities

    GNN->>Analyzer: Predictions + Graph
    
    Analyzer->>Analyzer: Detect Collusion Rings
    Note right of Analyzer: Connected components<br/>above threshold

    alt Collusion Detected
        Analyzer->>AlertSystem: High-Risk Alert
        AlertSystem->>AlertSystem: Notify Analysts
        AlertSystem->>AlertSystem: Auto-block Users
    end
```

### 5. Feedback Loop & Model Updates

```mermaid
sequenceDiagram
    participant Analyst as Fraud Analyst
    participant Dashboard as Dashboard
    participant FeedbackAPI as Feedback API
    participant MLOps as MLOps Pipeline
    participant Models as Model Registry

    Analyst->>Dashboard: Review Case
    Dashboard->>Dashboard: Show Predictions + SHAP
    
    Analyst->>Dashboard: Confirm/Reject Fraud
    Dashboard->>FeedbackAPI: POST /feedback
    
    FeedbackAPI->>FeedbackAPI: Store Label
    Note right of FeedbackAPI: Ground truth for training

    rect rgb(240, 240, 255)
        Note over MLOps, Models: Weekly Retraining Pipeline
        MLOps->>MLOps: Collect New Labels
        MLOps->>MLOps: Retrain Models
        MLOps->>Models: Update Model Registry
        Models->>Models: A/B Test New Models
    end
```

## Data Flow Summary

### High-Level Architecture
```
UPI Gateway → Kafka → Spark Streaming → Feature Store → FastAPI → Decision
                ↓                           ↓              ↓
            Enriched Data              Real-time Cache   SHAP Explanations
                ↓                           ↓              ↓
            GNN Analysis ←─────────── Historical Store → Feedback Loop
                ↓
        Collusion Alerts
```

### Key Data Transformations

1. **Raw Transaction** → **Hashed & Validated**
   ```json
   {
     "upi_id": "user123@paytm" → "a1b2c3d4e5f6...",
     "amount": 5000,
     "timestamp": "2024-01-15T10:30:00Z"
   }
   ```

2. **Basic Features** → **Enriched Features**
   ```json
   {
     "amount": 5000,
     "hour": 10,
     "user_txn_count_1h": 3,
     "user_amount_sum_24h": 15000,
     "velocity_risk": 0.2,
     "location_change": 0.05
   }
   ```

3. **Model Predictions** → **Business Decision**
   ```json
   {
     "risk_score": 0.85,
     "fraud_probability": 0.78,
     "decision": "CHALLENGE",
     "explanation": {
       "top_features": ["high_velocity", "new_device"],
       "shap_values": [0.3, 0.25, ...]
     }
   }
   ```

### Performance Characteristics

- **Ingestion Latency**: < 10ms
- **Feature Engineering**: < 50ms  
- **Model Inference**: < 100ms
- **End-to-End Latency**: < 200ms
- **Throughput**: 10,000+ TPS
- **GNN Analysis**: 30-minute batch cycles

### Scalability Points

1. **Horizontal Scaling**
   - Kafka partitions for parallel processing
   - Spark worker nodes for feature engineering
   - FastAPI replicas for inference load

2. **Caching Strategy**
   - Redis for hot user features (1-hour TTL)
   - PostgreSQL for historical analysis
   - Model caching for faster inference

3. **Monitoring & Alerting**
   - Real-time metrics via Prometheus
   - Grafana dashboards for visualization
   - PagerDuty integration for critical alerts
