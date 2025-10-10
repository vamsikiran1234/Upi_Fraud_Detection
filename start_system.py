#!/usr/bin/env python3
"""
UPI Fraud Detection System Startup Script
Starts all services and performs health checks
"""

import subprocess
import time
import requests
import sys
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

def check_docker():
    """Check if Docker is running"""
    print("🐳 Checking Docker...")
    success, stdout, stderr = run_command("docker --version")
    if success:
        print("✅ Docker is available")
        return True
    else:
        print("❌ Docker is not available. Please install Docker first.")
        return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    print("🐳 Checking Docker Compose...")
    success, stdout, stderr = run_command("docker-compose --version")
    if success:
        print("✅ Docker Compose is available")
        return True
    else:
        print("❌ Docker Compose is not available. Please install Docker Compose first.")
        return False

def start_services():
    """Start all services using Docker Compose"""
    print("\n🚀 Starting UPI Fraud Detection System...")
    
    # Check if docker-compose.yml exists
    if not os.path.exists("docker-compose.yml"):
        print("❌ docker-compose.yml not found!")
        return False
    
    # Start services
    print("Starting services with Docker Compose...")
    success, stdout, stderr = run_command("docker-compose up -d")
    
    if success:
        print("✅ Services started successfully")
        print(stdout)
        return True
    else:
        print("❌ Failed to start services")
        print(stderr)
        return False

def wait_for_service(url, service_name, max_attempts=30):
    """Wait for a service to be ready"""
    print(f"⏳ Waiting for {service_name} to be ready...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} is ready!")
                return True
        except:
            pass
        
        print(f"   Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(2)
    
    print(f"❌ {service_name} failed to start within {max_attempts * 2} seconds")
    return False

def check_services():
    """Check if all services are running"""
    print("\n🔍 Checking service health...")
    
    services = [
        ("http://localhost:8000/health", "Fraud Detection API"),
        ("http://localhost:3000", "React Dashboard"),
        ("http://localhost:9090", "Prometheus"),
        ("http://localhost:3001", "Grafana"),
    ]
    
    all_healthy = True
    
    for url, name in services:
        if wait_for_service(url, name):
            continue
        else:
            all_healthy = False
    
    return all_healthy

def show_status():
    """Show system status and access URLs"""
    print("\n" + "=" * 60)
    print("🎉 UPI Fraud Detection System is running!")
    print("=" * 60)
    print("\n📊 Access URLs:")
    print("   • Fraud Detection API: http://localhost:8000")
    print("   • API Documentation: http://localhost:8000/docs")
    print("   • React Dashboard: http://localhost:3000")
    print("   • Prometheus: http://localhost:9090")
    print("   • Grafana: http://localhost:3001 (admin/admin)")
    print("   • Kibana: http://localhost:5601")
    
    print("\n🔧 Management Commands:")
    print("   • View logs: docker-compose logs -f")
    print("   • Stop system: docker-compose down")
    print("   • Restart system: docker-compose restart")
    print("   • Test system: python test_system.py")
    
    print("\n📝 Next Steps:")
    print("   1. Open the React Dashboard at http://localhost:3000")
    print("   2. Test the API using the test script: python test_system.py")
    print("   3. Monitor system health in Grafana")
    print("   4. Check logs for any issues")

def main():
    """Main startup function"""
    print("🚀 UPI Fraud Detection System Startup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_docker():
        sys.exit(1)
    
    if not check_docker_compose():
        sys.exit(1)
    
    # Start services
    if not start_services():
        print("❌ Failed to start services. Check Docker logs.")
        sys.exit(1)
    
    # Wait for services to be ready
    print("\n⏳ Waiting for services to initialize...")
    time.sleep(10)  # Give services time to start
    
    # Check service health
    if check_services():
        show_status()
    else:
        print("\n⚠️  Some services may not be fully ready yet.")
        print("   Please wait a few minutes and check the URLs manually.")
        print("   You can also run: docker-compose logs -f")
    
    print("\n🎯 System startup complete!")

if __name__ == "__main__":
    main()
