#!/usr/bin/env python3
"""
Production Deployment Script for UPI Fraud Detection System
Deploys all components with proper configuration and monitoring
"""

import subprocess
import time
import requests
import json
import os
from pathlib import Path

def run_command(command, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_prerequisites():
    """Check if all prerequisites are installed"""
    print("🔍 Checking prerequisites...")
    
    # Check Python
    success, stdout, stderr = run_command("python --version")
    if not success:
        print("❌ Python not found")
        return False
    print(f"✅ Python: {stdout.strip()}")
    
    # Check pip
    success, stdout, stderr = run_command("pip --version")
    if not success:
        print("❌ pip not found")
        return False
    print(f"✅ pip: {stdout.strip()}")
    
    return True

def install_dependencies():
    """Install all required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Install basic requirements
    success, stdout, stderr = run_command("pip install -r requirements-quick.txt")
    if not success:
        print(f"❌ Failed to install basic requirements: {stderr}")
        return False
    print("✅ Basic requirements installed")
    
    # Install advanced requirements
    success, stdout, stderr = run_command("pip install -r requirements-advanced.txt")
    if not success:
        print(f"❌ Failed to install advanced requirements: {stderr}")
        return False
    print("✅ Advanced requirements installed")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "models",
        "logs",
        "data",
        "config",
        "scripts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def create_startup_scripts():
    """Create startup scripts for all services"""
    print("\n📝 Creating startup scripts...")
    
    # Basic system startup script
    basic_startup = """#!/bin/bash
echo "Starting Basic UPI Fraud Detection System..."
cd "$(dirname "$0")"
python quick_start.py
"""
    
    with open("start_basic.sh", "w") as f:
        f.write(basic_startup)
    os.chmod("start_basic.sh", 0o755)
    print("✅ Created start_basic.sh")
    
    # Advanced system startup script
    advanced_startup = """#!/bin/bash
echo "Starting Advanced UPI Fraud Detection System..."
cd "$(dirname "$0")"
python advanced_quick_start.py
"""
    
    with open("start_advanced.sh", "w") as f:
        f.write(advanced_startup)
    os.chmod("start_advanced.sh", 0o755)
    print("✅ Created start_advanced.sh")
    
    # Monitoring dashboard startup script
    monitoring_startup = """#!/bin/bash
echo "Starting Monitoring Dashboard..."
cd "$(dirname "$0")"
python monitoring_dashboard.py
"""
    
    with open("start_monitoring.sh", "w") as f:
        f.write(monitoring_startup)
    os.chmod("start_monitoring.sh", 0o755)
    print("✅ Created start_monitoring.sh")
    
    # Windows batch files
    basic_bat = """@echo off
echo Starting Basic UPI Fraud Detection System...
cd /d "%~dp0"
python quick_start.py
pause
"""
    
    with open("start_basic.bat", "w") as f:
        f.write(basic_bat)
    print("✅ Created start_basic.bat")
    
    advanced_bat = """@echo off
echo Starting Advanced UPI Fraud Detection System...
cd /d "%~dp0"
python advanced_quick_start.py
pause
"""
    
    with open("start_advanced.bat", "w") as f:
        f.write(advanced_bat)
    print("✅ Created start_advanced.bat")
    
    monitoring_bat = """@echo off
echo Starting Monitoring Dashboard...
cd /d "%~dp0"
python monitoring_dashboard.py
pause
"""
    
    with open("start_monitoring.bat", "w") as f:
        f.write(monitoring_bat)
    print("✅ Created start_monitoring.bat")
    
    return True

def create_systemd_services():
    """Create systemd service files for Linux"""
    print("\n🔧 Creating systemd services...")
    
    # Basic system service
    basic_service = """[Unit]
Description=UPI Fraud Detection Basic System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/upi-fraud-system
ExecStart=/usr/bin/python3 quick_start.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    with open("upi-fraud-basic.service", "w") as f:
        f.write(basic_service)
    print("✅ Created upi-fraud-basic.service")
    
    # Advanced system service
    advanced_service = """[Unit]
Description=UPI Fraud Detection Advanced System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/upi-fraud-system
ExecStart=/usr/bin/python3 advanced_quick_start.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    with open("upi-fraud-advanced.service", "w") as f:
        f.write(advanced_service)
    print("✅ Created upi-fraud-advanced.service")
    
    # Monitoring service
    monitoring_service = """[Unit]
Description=UPI Fraud Detection Monitoring Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/upi-fraud-system
ExecStart=/usr/bin/python3 monitoring_dashboard.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    with open("upi-fraud-monitoring.service", "w") as f:
        f.write(monitoring_service)
    print("✅ Created upi-fraud-monitoring.service")
    
    return True

def create_docker_compose_production():
    """Create production Docker Compose configuration"""
    print("\n🐳 Creating production Docker Compose...")
    
    docker_compose_prod = """version: '3.8'

services:
  # Basic Fraud Detection API
  fraud-api-basic:
    build: 
      context: .
      dockerfile: Dockerfile.basic
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Advanced Fraud Detection API
  fraud-api-advanced:
    build: 
      context: .
      dockerfile: Dockerfile.advanced
    ports:
      - "8001:8001"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Monitoring Dashboard
  monitoring-dashboard:
    build: 
      context: .
      dockerfile: Dockerfile.monitoring
    ports:
      - "8002:8002"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    depends_on:
      - fraud-api-basic
      - fraud-api-advanced
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - fraud-api-basic
      - fraud-api-advanced
      - monitoring-dashboard
    restart: unless-stopped

volumes:
  models:
  logs:
"""
    
    with open("docker-compose.prod.yml", "w") as f:
        f.write(docker_compose_prod)
    print("✅ Created docker-compose.prod.yml")
    
    return True

def create_nginx_config():
    """Create Nginx configuration for load balancing"""
    print("\n🌐 Creating Nginx configuration...")
    
    nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream fraud_api {
        server fraud-api-basic:8000;
        server fraud-api-advanced:8001;
    }
    
    upstream monitoring {
        server monitoring-dashboard:8002;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # Basic API
        location /api/basic/ {
            proxy_pass http://fraud-api-basic:8000/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Advanced API
        location /api/advanced/ {
            proxy_pass http://fraud-api-advanced:8001/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Monitoring Dashboard
        location /monitoring/ {
            proxy_pass http://monitoring:8002/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # WebSocket support for monitoring
        location /ws {
            proxy_pass http://monitoring:8002/ws;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Default route
        location / {
            return 301 /monitoring/;
        }
    }
}
"""
    
    with open("nginx.conf", "w") as f:
        f.write(nginx_config)
    print("✅ Created nginx.conf")
    
    return True

def test_deployment():
    """Test the deployment"""
    print("\n🧪 Testing deployment...")
    
    # Wait for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(10)
    
    # Test basic system
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Basic system is healthy")
        else:
            print("❌ Basic system is unhealthy")
    except:
        print("❌ Basic system is not responding")
    
    # Test advanced system
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Advanced system is healthy")
        else:
            print("❌ Advanced system is unhealthy")
    except:
        print("❌ Advanced system is not responding")
    
    # Test monitoring dashboard
    try:
        response = requests.get("http://localhost:8002/api/metrics", timeout=5)
        if response.status_code == 200:
            print("✅ Monitoring dashboard is healthy")
        else:
            print("❌ Monitoring dashboard is unhealthy")
    except:
        print("❌ Monitoring dashboard is not responding")
    
    return True

def create_documentation():
    """Create deployment documentation"""
    print("\n📚 Creating documentation...")
    
    readme_content = """# UPI Fraud Detection System - Production Deployment

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
"""
    
    with open("DEPLOYMENT.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("✅ Created DEPLOYMENT.md")
    
    return True

def main():
    """Main deployment function"""
    print("🚀 UPI Fraud Detection System - Production Deployment")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return False
    
    # Create directories
    if not create_directories():
        print("❌ Directory creation failed")
        return False
    
    # Create startup scripts
    if not create_startup_scripts():
        print("❌ Startup script creation failed")
        return False
    
    # Create systemd services
    if not create_systemd_services():
        print("❌ Systemd service creation failed")
        return False
    
    # Create Docker Compose
    if not create_docker_compose_production():
        print("❌ Docker Compose creation failed")
        return False
    
    # Create Nginx config
    if not create_nginx_config():
        print("❌ Nginx configuration failed")
        return False
    
    # Create documentation
    if not create_documentation():
        print("❌ Documentation creation failed")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Production deployment setup completed!")
    print("=" * 60)
    print("\n📊 Next Steps:")
    print("1. Start the systems:")
    print("   - Basic: python quick_start.py")
    print("   - Advanced: python advanced_quick_start.py")
    print("   - Monitoring: python monitoring_dashboard.py")
    print("\n2. Access the services:")
    print("   - Basic API: http://localhost:8000")
    print("   - Advanced API: http://localhost:8001")
    print("   - Monitoring: http://localhost:8002")
    print("\n3. For production deployment:")
    print("   - Use Docker Compose: docker-compose -f docker-compose.prod.yml up -d")
    print("   - Configure Nginx load balancer")
    print("   - Set up SSL certificates")
    print("\n4. Monitor the systems:")
    print("   - Check logs in logs/ directory")
    print("   - Use monitoring dashboard for real-time metrics")
    print("   - Set up alerting for production")
    
    return True

if __name__ == "__main__":
    main()
