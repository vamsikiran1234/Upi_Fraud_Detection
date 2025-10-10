# UPI Fraud Detection - Infrastructure & Deployment

This directory contains all the infrastructure code and deployment configurations for the UPI Fraud Detection system.

## 🏗️ Architecture Overview

The system is deployed as microservices on Kubernetes with the following components:

- **FastAPI Inference Service**: Main fraud detection API with SHAP explanations
- **Redis**: Feature store cache and session storage
- **PostgreSQL**: Transaction logs and feature storage
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

## 📁 Directory Structure

```
infra/
├── k8s/                          # Kubernetes manifests
│   ├── namespace.yaml            # Namespaces for prod/staging
│   ├── configmap.yaml           # Configuration and secrets
│   ├── deployment.yaml          # Main API deployment
│   ├── service.yaml             # Kubernetes services
│   ├── ingress.yaml             # Ingress configuration
│   ├── hpa.yaml                 # Horizontal Pod Autoscaler
│   ├── pvc.yaml                 # Persistent Volume Claims
│   ├── redis-deployment.yaml    # Redis deployment
│   ├── postgres-deployment.yaml # PostgreSQL deployment
│   ├── monitoring.yaml          # Prometheus/Grafana config
│   ├── kustomization.yaml       # Kustomize configuration
│   └── patches/                 # Environment-specific patches
├── deploy.sh                    # Deployment script
├── docker-compose.dev.yml       # Local development setup
└── README.md                    # This file
```

## 🚀 Quick Start

### Local Development

1. **Start local environment**:
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

2. **Access services**:
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Grafana: http://localhost:3001 (admin/admin123)
   - Prometheus: http://localhost:9090

### Production Deployment

1. **Prerequisites**:
   - Kubernetes cluster (1.20+)
   - kubectl configured
   - Docker installed

2. **Deploy to Kubernetes**:
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Verify deployment**:
   ```bash
   kubectl get pods -n fraud-detection
   kubectl get services -n fraud-detection
   ```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_HOST` | Redis hostname | `redis-service` |
| `POSTGRES_HOST` | PostgreSQL hostname | `postgres-service` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `API_WORKERS` | Number of API workers | `4` |

### Secrets Management

Secrets are managed through Kubernetes secrets:

```bash
# Update PostgreSQL password
kubectl create secret generic fraud-detection-secrets \
  --from-literal=POSTGRES_PASSWORD=your-secure-password \
  -n fraud-detection --dry-run=client -o yaml | kubectl apply -f -
```

## 📊 Monitoring & Observability

### Metrics

The system exposes Prometheus metrics at `/metrics`:

- `fraud_predictions_total`: Total fraud predictions
- `http_request_duration_seconds`: API latency
- `model_drift_score`: Model drift detection
- `feature_store_cache_hits`: Cache performance

### Alerts

Key alerts configured:
- High fraud detection rate (>10%)
- API latency >100ms (95th percentile)
- Model drift score >0.3
- Service unavailability

### Dashboards

Grafana dashboards include:
- Real-time fraud detection metrics
- API performance and latency
- Model score distributions
- Infrastructure health

## 🔧 Scaling & Performance

### Horizontal Pod Autoscaling

The HPA is configured to scale based on:
- CPU utilization (70% target)
- Memory utilization (80% target)
- HTTP requests per second (100 RPS target)

### Resource Limits

**Production settings**:
- Requests: 1GB RAM, 500m CPU
- Limits: 4GB RAM, 2000m CPU
- Min replicas: 3
- Max replicas: 20

## 🔒 Security

### Network Policies

```bash
# Apply network policies (if using a CNI that supports them)
kubectl apply -f k8s/network-policies.yaml
```

### TLS/SSL

- Ingress configured with Let's Encrypt certificates
- Internal service communication uses mTLS
- Secrets encrypted at rest

## 🧪 Testing

### Load Testing

```bash
# Install k6
curl https://github.com/grafana/k6/releases/download/v0.45.0/k6-v0.45.0-linux-amd64.tar.gz -L | tar xvz --strip-components 1

# Run load test
k6 run --vus 100 --duration 5m load-test.js
```

### Health Checks

```bash
# Check API health
curl http://your-api-endpoint/health

# Check all services
kubectl get pods -n fraud-detection
```

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Kubernetes
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Deploy
      run: |
        ./infra/deploy.sh
```

## 📝 Troubleshooting

### Common Issues

1. **Pod stuck in Pending**:
   ```bash
   kubectl describe pod <pod-name> -n fraud-detection
   # Check resource constraints and node capacity
   ```

2. **Service not accessible**:
   ```bash
   kubectl get endpoints -n fraud-detection
   kubectl logs deployment/fraud-detection-api -n fraud-detection
   ```

3. **High memory usage**:
   ```bash
   # Check model loading and feature store cache
   kubectl top pods -n fraud-detection
   ```

### Logs

```bash
# API logs
kubectl logs -f deployment/fraud-detection-api -n fraud-detection

# Infrastructure logs
kubectl logs -f deployment/redis -n fraud-detection
kubectl logs -f deployment/postgres -n fraud-detection
```

## 🔄 Updates & Maintenance

### Rolling Updates

```bash
# Update image
kubectl set image deployment/fraud-detection-api \
  fraud-detection-api=fraud-detection:v1.1.0 \
  -n fraud-detection

# Check rollout status
kubectl rollout status deployment/fraud-detection-api -n fraud-detection
```

### Backup & Recovery

```bash
# Backup PostgreSQL
kubectl exec -it deployment/postgres -n fraud-detection -- \
  pg_dump -U fraud_user fraud_detection > backup.sql

# Restore
kubectl exec -i deployment/postgres -n fraud-detection -- \
  psql -U fraud_user fraud_detection < backup.sql
```

## 📞 Support

For issues and questions:
- Check logs: `kubectl logs -f deployment/fraud-detection-api -n fraud-detection`
- Monitor metrics: Access Grafana dashboard
- Review alerts: Check Prometheus alerts

---

**Note**: Update the ingress hostname in `k8s/ingress.yaml` and registry URL in `deploy.sh` before deploying to production.
