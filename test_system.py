#!/usr/bin/env python3
"""
Test script for UPI Fraud Detection System
Tests the API endpoints and basic functionality
"""

import requests
import json
import time
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_prediction():
    """Test fraud prediction endpoint"""
    print("\nTesting fraud prediction...")
    
    # Sample transaction data
    transaction_data = {
        "transaction_id": "TEST_TXN_001",
        "upi_id": "test@paytm",
        "amount": 15000.0,
        "merchant_id": "MERCHANT_001",
        "merchant_category": "ecommerce",
        "device_id": "device_123",
        "ip_address": "192.168.1.100",
        "location": {
            "lat": 28.6139,
            "lon": 77.2090
        },
        "timestamp": datetime.now().isoformat(),
        "payment_method": "UPI",
        "session_id": "session_123",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "sms_content": "Your OTP is 123456",
        "merchant_notes": "Online purchase"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=transaction_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful:")
            print(f"   Transaction ID: {data['transaction_id']}")
            print(f"   Risk Score: {data['risk_score']:.3f}")
            print(f"   Fraud Probability: {data['fraud_probability']:.3f}")
            print(f"   Decision: {data['decision']}")
            print(f"   Confidence: {data['confidence']:.3f}")
            print(f"   Processing Time: {data['processing_time_ms']:.2f}ms")
            print(f"   Alerts: {data['alerts']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_model_status():
    """Test model status endpoint"""
    print("\nTesting model status...")
    try:
        response = requests.get(f"{BASE_URL}/models/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model status retrieved:")
            print(f"   Models loaded: {data['loaded']}")
            print(f"   Models: {data['models']}")
            print(f"   Versions: {data['versions']}")
            return True
        else:
            print(f"❌ Model status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model status error: {e}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    print("\nTesting metrics...")
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Metrics retrieved:")
            print(f"   Total Predictions: {data['total_predictions']}")
            print(f"   Fraud Detected: {data['fraud_detected']}")
            print(f"   False Positives: {data['false_positives']}")
            print(f"   Average Latency: {data['average_latency_ms']}ms")
            return True
        else:
            print(f"❌ Metrics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics error: {e}")
        return False

def run_performance_test():
    """Run basic performance test"""
    print("\nRunning performance test...")
    
    transaction_data = {
        "transaction_id": "PERF_TEST_001",
        "upi_id": "perf@paytm",
        "amount": 5000.0,
        "merchant_id": "MERCHANT_002",
        "merchant_category": "food",
        "device_id": "device_456",
        "ip_address": "192.168.1.101",
        "location": {
            "lat": 28.6139,
            "lon": 77.2090
        },
        "timestamp": datetime.now().isoformat(),
        "payment_method": "UPI"
    }
    
    times = []
    success_count = 0
    
    for i in range(10):
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/predict",
                json=transaction_data,
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                success_count += 1
                times.append((end_time - start_time) * 1000)  # Convert to ms
                print(f"   Request {i+1}: {(end_time - start_time) * 1000:.2f}ms")
            else:
                print(f"   Request {i+1}: Failed ({response.status_code})")
        except Exception as e:
            print(f"   Request {i+1}: Error - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"\n✅ Performance test results:")
        print(f"   Successful requests: {success_count}/10")
        print(f"   Average response time: {avg_time:.2f}ms")
        print(f"   Min response time: {min_time:.2f}ms")
        print(f"   Max response time: {max_time:.2f}ms")
        return True
    else:
        print(f"❌ Performance test failed: No successful requests")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting UPI Fraud Detection System Tests")
    print("=" * 50)
    
    tests = [
        test_health_check,
        test_model_status,
        test_prediction,
        test_metrics,
        run_performance_test
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the system configuration.")
    
    return passed == total

if __name__ == "__main__":
    main()
