# UPI Fraud Detection System - Production Deployment

## 🚀 Quick Start

### Option 1: Direct Python (Recommended for Development)
```bash
# Start Basic System
python quick_start.py

# Start Advanced System (in another terminal)
python advanced_quick_start.py

# Start Monitoring Dashboard (in another terminal)
python monitoring_dashboard.py
```

### Option 2: Using Scripts
```bash
# Linux/Mac
./start_basic.sh
./start_advanced.sh
./start_monitoring.sh

# Windows
start_basic.bat
start_advanced.bat
start_monitoring.bat
```

### Option 3: Docker Compose (Production)
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 📊 Access Points

- **Basic API**: http://localhost:8000
- **Advanced API**: http://localhost:8001
- **Monitoring Dashboard**: http://localhost:8002
- **API Documentation**: 
  - Basic: http://localhost:8000/docs
  - Advanced: http://localhost:8001/docs
  - Monitoring: http://localhost:8002/docs

## 🔧 Configuration

### Environment Variables
- `ENVIRONMENT`: production/development
- `LOG_LEVEL`: DEBUG/INFO/WARNING/ERROR
- `API_HOST`: 0.0.0.0
- `API_PORT`: 8000/8001/8002

### Ports
- 8000: Basic Fraud Detection API
- 8001: Advanced Fraud Detection API
- 8002: Monitoring Dashboard
- 80: Nginx Load Balancer (production)

## 📈 Monitoring

The monitoring dashboard provides:
- Real-time system metrics
- Performance monitoring
- Alert management
- System health status
- Live transaction simulation

## 🛠️ Maintenance

### Logs
- Basic System: `logs/basic_system.log`
- Advanced System: `logs/advanced_system.log`
- Monitoring: `logs/monitoring.log`

### Model Updates
- Models are stored in `models/` directory
- Automatic retraining can be configured
- Model versioning is supported

### Scaling
- Horizontal scaling with load balancer
- Vertical scaling by increasing resources
- Auto-scaling based on metrics

## 🔒 Security

- PII data is hashed and tokenized
- API endpoints are secured
- Audit logging is enabled
- Rate limiting is configured

## 📞 Support

For issues or questions:
1. Check the logs
2. Verify system health
3. Review configuration
4. Contact support team
