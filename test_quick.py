#!/usr/bin/env python3
"""
Quick test for UPI Fraud Detection System
Tests the simplified API
"""

import requests
import json
import time
from datetime import datetime

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing UPI Fraud Detection API...")
    print("=" * 50)
    
    # Test 1: Health Check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed: {data['status']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 2: Model Status
    print("2. Testing model status...")
    try:
        response = requests.get(f"{base_url}/models/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model status: {data['model_type']} loaded")
        else:
            print(f"   ❌ Model status failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Model status error: {e}")
    
    # Test 3: Fraud Prediction - Low Risk
    print("3. Testing low-risk transaction...")
    try:
        test_data = {
            "transaction_id": "TXN_LOW_001",
            "upi_id": "user@paytm",
            "amount": 500.0,
            "merchant_id": "MERCHANT_001",
            "merchant_category": "food",
            "device_id": "device_123",
            "ip_address": "192.168.1.100",
            "location": {"lat": 28.6139, "lon": 77.2090},
            "timestamp": "2024-01-15T14:30:25Z",
            "payment_method": "UPI"
        }
        
        response = requests.post(f"{base_url}/predict", json=test_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Low-risk prediction: {data['decision']} (Risk: {data['risk_score']:.3f})")
        else:
            print(f"   ❌ Low-risk prediction failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Low-risk prediction error: {e}")
    
    # Test 4: Fraud Prediction - High Risk
    print("4. Testing high-risk transaction...")
    try:
        test_data = {
            "transaction_id": "TXN_HIGH_001",
            "upi_id": "user@paytm",
            "amount": 75000.0,
            "merchant_id": "MERCHANT_002",
            "merchant_category": "crypto",
            "device_id": "device_456",
            "ip_address": "192.168.1.101",
            "location": {"lat": 28.6139, "lon": 77.2090},
            "timestamp": "2024-01-15T02:30:25Z",  # Night time
            "payment_method": "UPI"
        }
        
        response = requests.post(f"{base_url}/predict", json=test_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ High-risk prediction: {data['decision']} (Risk: {data['risk_score']:.3f})")
            print(f"   📝 Explanation: {data['explanation']['human_readable']}")
            print(f"   🚨 Alerts: {data['alerts']}")
        else:
            print(f"   ❌ High-risk prediction failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ High-risk prediction error: {e}")
    
    # Test 5: Performance Test
    print("5. Testing performance...")
    try:
        test_data = {
            "transaction_id": "TXN_PERF_001",
            "upi_id": "perf@paytm",
            "amount": 2500.0,
            "merchant_id": "MERCHANT_003",
            "merchant_category": "ecommerce",
            "device_id": "device_789",
            "ip_address": "192.168.1.102",
            "location": {"lat": 28.6139, "lon": 77.2090},
            "timestamp": "2024-01-15T10:30:25Z",
            "payment_method": "UPI"
        }
        
        times = []
        for i in range(5):
            start_time = time.time()
            response = requests.post(f"{base_url}/predict", json=test_data, timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                times.append((end_time - start_time) * 1000)
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"   ✅ Performance test: {avg_time:.2f}ms average response time")
        else:
            print(f"   ❌ Performance test failed")
    except Exception as e:
        print(f"   ❌ Performance test error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    print("📊 Access the API documentation at: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    test_api()
