#!/usr/bin/env python3
"""
Run UPI Fraud Detection System locally without Docker
This is a simplified version for development and testing
"""

import subprocess
import time
import requests
import sys
import os
from pathlib import Path

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def start_api_server():
    """Start the FastAPI server locally"""
    print("🚀 Starting FastAPI server...")
    
    # Change to serving directory
    os.chdir("serving")
    
    try:
        # Start the server in background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app", 
            "--host", "0.0.0.0", "--port", "8000", "--reload"
        ])
        
        print("✅ FastAPI server started")
        print("   API URL: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        
        return process
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def wait_for_api():
    """Wait for API to be ready"""
    print("⏳ Waiting for API to be ready...")
    
    for attempt in range(30):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except:
            pass
        
        print(f"   Attempt {attempt + 1}/30...")
        time.sleep(2)
    
    print("❌ API failed to start within 60 seconds")
    return False

def test_api():
    """Test the API endpoints"""
    print("\n🧪 Testing API endpoints...")
    
    # Test health check
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test prediction
    try:
        test_data = {
            "transaction_id": "TEST_001",
            "upi_id": "test@paytm",
            "amount": 5000.0,
            "merchant_id": "MERCHANT_001",
            "merchant_category": "ecommerce",
            "device_id": "device_123",
            "ip_address": "192.168.1.100",
            "location": {"lat": 28.6139, "lon": 77.2090},
            "timestamp": "2024-01-15T14:30:25Z",
            "payment_method": "UPI"
        }
        
        response = requests.post("http://localhost:8000/predict", json=test_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction test passed")
            print(f"   Risk Score: {data['risk_score']:.3f}")
            print(f"   Decision: {data['decision']}")
            print(f"   Processing Time: {data['processing_time_ms']:.2f}ms")
        else:
            print(f"❌ Prediction test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prediction test error: {e}")
        return False
    
    return True

def show_status():
    """Show system status"""
    print("\n" + "=" * 60)
    print("🎉 UPI Fraud Detection System is running locally!")
    print("=" * 60)
    print("\n📊 Access URLs:")
    print("   • API: http://localhost:8000")
    print("   • API Documentation: http://localhost:8000/docs")
    print("   • Health Check: http://localhost:8000/health")
    print("   • Metrics: http://localhost:8000/metrics")
    
    print("\n🔧 Management Commands:")
    print("   • Test API: python test_system.py")
    print("   • Stop server: Ctrl+C")
    print("   • View logs: Check terminal output")
    
    print("\n📝 Next Steps:")
    print("   1. Open http://localhost:8000/docs to test the API")
    print("   2. Run python test_system.py to run comprehensive tests")
    print("   3. For full system with dashboard, install Docker")

def main():
    """Main function"""
    print("🚀 UPI Fraud Detection System - Local Mode")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        sys.exit(1)
    
    # Wait for API to be ready
    if not wait_for_api():
        print("❌ API failed to start. Check the logs above.")
        api_process.terminate()
        sys.exit(1)
    
    # Test API
    if not test_api():
        print("❌ API tests failed. Check the logs above.")
        api_process.terminate()
        sys.exit(1)
    
    # Show status
    show_status()
    
    try:
        print("\n🔄 Server is running. Press Ctrl+C to stop.")
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        api_process.terminate()
        print("✅ Server stopped")

if __name__ == "__main__":
    main()
