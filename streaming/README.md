# UPI Fraud Detection - Real-time Streaming Pipeline

This directory contains the Spark Structured Streaming implementation for real-time feature extraction and enrichment from UPI transaction streams.

## 🏗️ Architecture Overview

The streaming pipeline processes transaction data in real-time with the following flow:

```
Kafka (raw-transactions) → Spark Streaming → Feature Extraction → Feature Store (Redis + PostgreSQL) → Kafka (enriched-transactions)
```

## 📁 Components

### Core Files

- **`spark_streaming.py`** - Main Spark Structured Streaming job with comprehensive feature extraction
- **`kafka_producer.py`** - Transaction data generator for testing and simulation
- **`submit_job.sh`** - Script to submit Spark job with optimized configuration
- **`docker-compose.yml`** - Complete streaming infrastructure (Kafka, Spark, Redis, PostgreSQL)

### Features Extracted

#### 1. **Basic Transaction Features**
- Amount, timestamp, merchant category
- Time-based features (hour, day of week, weekend/night flags)
- Merchant risk categorization

#### 2. **Velocity Features** (Windowed Aggregations)
- Transaction counts per user/device (1h, 24h windows)
- Amount aggregations (sum, average, std deviation)
- Unique merchants/devices per user

#### 3. **Sequence Features**
- Previous transaction patterns
- Time between transactions
- Amount change ratios
- Repeat merchant detection

#### 4. **Location Features**
- Location change detection
- Distance from previous transaction
- Location jump alerts (>11km threshold)

#### 5. **Device & IP Features**
- Mobile device detection
- Browser type extraction
- Private IP detection
- Suspicious IP flagging

#### 6. **Text Analysis Features**
- SMS content analysis for suspicious keywords
- Message length and word count
- Merchant notes processing

#### 7. **Risk Indicators**
- Velocity risk (high frequency/amount)
- Behavioral risk (night transactions, location jumps)
- Device risk (multiple devices)
- Merchant risk (high-risk categories)

## 🚀 Quick Start

### 1. Start Infrastructure

```bash
# Start Kafka, Spark, Redis, PostgreSQL
docker-compose up -d

# Verify services
docker-compose ps
```

### 2. Create Kafka Topics

```bash
# Create required topics
docker exec kafka kafka-topics --create --topic raw-transactions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic enriched-transactions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic aggregated-features --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

### 3. Start Streaming Job

```bash
# Make script executable
chmod +x submit_job.sh

# Submit streaming job
./submit_job.sh
```

### 4. Generate Test Data

```bash
# Install dependencies
pip install -r requirements.txt

# Start transaction producer (10 TPS for 60 minutes)
python kafka_producer.py --rate 10 --duration 60 --fraud-rate 0.05

# Generate specific fraud scenarios
python kafka_producer.py --scenario velocity_attack
python kafka_producer.py --scenario amount_escalation
```

## 📊 Monitoring

### Spark UI
- **Master**: http://localhost:8080
- **Application**: http://localhost:4040 (when job is running)

### Kafka Topics
```bash
# Monitor raw transactions
docker exec kafka kafka-console-consumer --topic raw-transactions --bootstrap-server localhost:9092

# Monitor enriched transactions
docker exec kafka kafka-console-consumer --topic enriched-transactions --bootstrap-server localhost:9092
```

### Feature Store
```bash
# Check Redis cache
docker exec redis-streaming redis-cli
> KEYS user_features:*
> HGETALL user_features:abc123

# Check PostgreSQL
docker exec -it postgres-streaming psql -U fraud_user -d fraud_detection
> SELECT * FROM user_features LIMIT 10;
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KAFKA_BROKERS` | Kafka bootstrap servers | `localhost:9092` |
| `INPUT_TOPIC` | Raw transactions topic | `raw-transactions` |
| `OUTPUT_TOPIC` | Enriched transactions topic | `enriched-transactions` |
| `AGGREGATION_TOPIC` | Aggregated features topic | `aggregated-features` |
| `REDIS_HOST` | Redis hostname | `localhost` |
| `REDIS_PORT` | Redis port | `6380` |
| `POSTGRES_HOST` | PostgreSQL hostname | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5433` |

### Spark Configuration

The streaming job is optimized with:
- **Checkpointing**: Fault-tolerant state management
- **Adaptive Query Execution**: Dynamic optimization
- **State Store**: Efficient windowed aggregations
- **Watermarking**: Late data handling (10 minutes)

## 🔧 Performance Tuning

### Throughput Optimization

```bash
# Increase parallelism
export SPARK_EXECUTOR_INSTANCES=4
export SPARK_EXECUTOR_CORES=4

# Tune batch intervals
--conf spark.sql.streaming.trigger.processingTime=10s

# Optimize Kafka consumption
--conf spark.sql.streaming.kafka.consumer.cache.capacity=1000
```

### Memory Optimization

```bash
# Increase executor memory
--conf spark.executor.memory=4g
--conf spark.driver.memory=2g

# Tune state store
--conf spark.sql.streaming.stateStore.maintenanceInterval=30s
```

## 🧪 Testing

### Unit Tests
```bash
# Run feature extraction tests
python -m pytest tests/test_feature_extraction.py

# Run streaming logic tests
python -m pytest tests/test_streaming.py
```

### Load Testing
```bash
# High-volume test (1000 TPS)
python kafka_producer.py --rate 1000 --duration 10

# Sustained load test
python kafka_producer.py --rate 100 --duration 120
```

### Fraud Scenario Testing
```bash
# Test velocity attacks
python kafka_producer.py --scenario velocity_attack

# Test amount escalation
python kafka_producer.py --scenario amount_escalation

# Custom fraud rate
python kafka_producer.py --rate 50 --fraud-rate 0.2
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Add more Spark workers
docker-compose up --scale spark-worker=4

# Increase Kafka partitions
docker exec kafka kafka-topics --alter --topic raw-transactions --partitions 6 --bootstrap-server localhost:9092
```

### Vertical Scaling
```bash
# Increase worker resources
export SPARK_WORKER_MEMORY=4G
export SPARK_WORKER_CORES=4
```

## 🔍 Troubleshooting

### Common Issues

1. **Streaming job fails to start**
   ```bash
   # Check Spark logs
   docker logs spark-master
   docker logs spark-worker-1
   ```

2. **Kafka connection issues**
   ```bash
   # Verify Kafka is running
   docker exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092
   ```

3. **Feature store connection issues**
   ```bash
   # Test Redis connection
   docker exec redis-streaming redis-cli ping
   
   # Test PostgreSQL connection
   docker exec postgres-streaming pg_isready -U fraud_user
   ```

4. **High latency/backpressure**
   ```bash
   # Monitor streaming metrics
   curl http://localhost:4040/api/v1/applications
   
   # Check Kafka lag
   docker exec kafka kafka-consumer-groups --bootstrap-server localhost:9092 --describe --group spark-streaming
   ```

### Performance Monitoring

```bash
# Monitor resource usage
docker stats

# Check streaming progress
curl http://localhost:4040/api/v1/applications/[app-id]/streaming/batches

# Monitor Kafka throughput
docker exec kafka kafka-run-class kafka.tools.ConsumerPerformance --topic raw-transactions --bootstrap-server localhost:9092 --messages 1000
```

## 🔄 Production Deployment

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f ../infra/k8s/spark-streaming-deployment.yaml
```

### Configuration Management
```bash
# Use ConfigMaps for environment-specific settings
kubectl create configmap streaming-config --from-env-file=production.env
```

### Monitoring Integration
```bash
# Enable Prometheus metrics
--conf spark.sql.streaming.metricsEnabled=true
--conf spark.metrics.conf.*.sink.prometheusServlet.class=org.apache.spark.metrics.sink.PrometheusServlet
```

## 📝 Data Flow

1. **Ingestion**: Raw transactions from UPI gateway → Kafka `raw-transactions` topic
2. **Processing**: Spark Structured Streaming consumes and processes data
3. **Feature Extraction**: Multiple feature types extracted in parallel
4. **Enrichment**: External data sources joined (IP geolocation, merchant data)
5. **Storage**: Features stored in Redis (real-time) and PostgreSQL (historical)
6. **Output**: Enriched transactions → Kafka `enriched-transactions` topic
7. **Aggregation**: Windowed aggregations → Kafka `aggregated-features` topic

## 🎯 Next Steps

- [ ] Add machine learning model integration for real-time scoring
- [ ] Implement exactly-once processing guarantees
- [ ] Add support for schema evolution
- [ ] Integrate with Apache Iceberg for data lake storage
- [ ] Add custom metrics and alerting
- [ ] Implement data quality checks and validation

---

**Note**: This streaming pipeline is designed for high-throughput, low-latency fraud detection. Monitor resource usage and tune parameters based on your specific workload requirements.
