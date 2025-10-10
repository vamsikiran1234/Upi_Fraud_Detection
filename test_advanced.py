#!/usr/bin/env python3
"""
Advanced test for UPI Fraud Detection System
Tests the enhanced API with XGBoost, LightGBM, and ensemble methods
"""

import requests
import json
import time
from datetime import datetime

def test_advanced_api():
    """Test the advanced API endpoints"""
    base_url = "http://localhost:8001"
    
    print("🧪 Testing Advanced UPI Fraud Detection API...")
    print("=" * 60)
    
    # Test 1: Health Check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed: {data['status']}")
            print(f"   📊 Models loaded: {data['models_loaded']}")
            print(f"   🤖 Ensemble models: {data['ensemble_models']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 2: Model Status
    print("\n2. Testing model status...")
    try:
        response = requests.get(f"{base_url}/models/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model status retrieved:")
            print(f"      - Ensemble type: {data['ensemble_type']}")
            print(f"      - Models: {data['models']}")
            print(f"      - Features: {data['features']}")
            print(f"      - Model weights: {data['model_weights']}")
        else:
            print(f"   ❌ Model status failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Model status error: {e}")
    
    # Test 3: Low-Risk Transaction
    print("\n3. Testing low-risk transaction...")
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
        
        response = requests.post(f"{base_url}/predict", json=test_data, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Low-risk prediction: {data['decision']} (Risk: {data['risk_score']:.3f})")
            print(f"   📊 Confidence: {data['confidence']:.3f}")
            print(f"   ⏱️  Processing time: {data['processing_time_ms']:.2f}ms")
            print(f"   🤖 Individual scores: {data['model_details']['individual_scores']}")
        else:
            print(f"   ❌ Low-risk prediction failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Low-risk prediction error: {e}")
    
    # Test 4: High-Risk Transaction
    print("\n4. Testing high-risk transaction...")
    try:
        test_data = {
            "transaction_id": "TXN_HIGH_001",
            "upi_id": "user@paytm",
            "amount": 150000.0,
            "merchant_id": "MERCHANT_002",
            "merchant_category": "crypto",
            "device_id": "device_456",
            "ip_address": "192.168.1.101",
            "location": {"lat": 28.6139, "lon": 77.2090},
            "timestamp": "2024-01-15T02:30:25Z",  # Night time
            "payment_method": "UPI"
        }
        
        response = requests.post(f"{base_url}/predict", json=test_data, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ High-risk prediction: {data['decision']} (Risk: {data['risk_score']:.3f})")
            print(f"   📊 Confidence: {data['confidence']:.3f}")
            print(f"   ⏱️  Processing time: {data['processing_time_ms']:.2f}ms")
            print(f"   🚨 Alerts: {data['alerts']}")
            print(f"   📝 Explanation: {data['explanation']['human_readable']}")
            print(f"   🔍 Risk factors: {len(data['explanation']['risk_factors'])} identified")
        else:
            print(f"   ❌ High-risk prediction failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ High-risk prediction error: {e}")
    
    # Test 5: Model Performance
    print("\n5. Testing model performance...")
    try:
        response = requests.get(f"{base_url}/models/performance", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model performance retrieved:")
            print(f"      - Accuracy: {data['ensemble_accuracy']:.3f}")
            print(f"      - Precision: {data['ensemble_precision']:.3f}")
            print(f"      - Recall: {data['ensemble_recall']:.3f}")
            print(f"      - F1 Score: {data['ensemble_f1']:.3f}")
            print(f"      - Avg Latency: {data['average_latency_ms']:.1f}ms")
            print(f"      - Throughput: {data['throughput_per_second']} req/s")
        else:
            print(f"   ❌ Model performance failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Model performance error: {e}")
    
    # Test 6: Feature Importance
    print("\n6. Testing feature importance...")
    try:
        response = requests.get(f"{base_url}/features/importance", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Feature importance retrieved:")
            print(f"      - Top features: {data['top_features']}")
            if 'feature_importance' in data:
                print(f"      - Available models: {list(data['feature_importance'].keys())}")
        else:
            print(f"   ❌ Feature importance failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Feature importance error: {e}")
    
    # Test 7: Performance Test
    print("\n7. Testing performance with multiple requests...")
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
            response = requests.post(f"{base_url}/predict", json=test_data, timeout=15)
            end_time = time.time()
            
            if response.status_code == 200:
                times.append((end_time - start_time) * 1000)
                print(f"   Request {i+1}: {(end_time - start_time) * 1000:.2f}ms")
            else:
                print(f"   Request {i+1}: Failed ({response.status_code})")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"\n   ✅ Performance test results:")
            print(f"      - Average response time: {avg_time:.2f}ms")
            print(f"      - Min response time: {min_time:.2f}ms")
            print(f"      - Max response time: {max_time:.2f}ms")
        else:
            print(f"   ❌ Performance test failed: No successful requests")
    except Exception as e:
        print(f"   ❌ Performance test error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Advanced API testing completed!")
    print("📊 Access the API documentation at: http://localhost:8001/docs")
    print("🔍 Compare with basic API at: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    test_advanced_api()
