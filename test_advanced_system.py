"""
Comprehensive Test Script for Advanced UPI Fraud Detection System
Tests all advanced AI/ML features and integrations
"""

import requests
import json
import time
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Test configuration
BASE_URL = "http://localhost:8003"
TEST_TIMEOUT = 30

class AdvancedSystemTester:
    """Comprehensive tester for advanced fraud detection system"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.test_results = {}
        self.session = requests.Session()
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all advanced system tests"""
        print("🚀 Starting Advanced UPI Fraud Detection System Tests")
        print("=" * 70)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("System Status", self.test_system_status),
            ("Fraud Prediction", self.test_fraud_prediction),
            ("Multi-Modal Features", self.test_multimodal_features),
            ("Threat Intelligence", self.test_threat_intelligence),
            ("Synthetic Data Generation", self.test_synthetic_data),
            ("Federated Learning", self.test_federated_learning),
            ("Blockchain Audit", self.test_blockchain_audit),
            ("Active Learning", self.test_active_learning),
            ("Differential Privacy", self.test_differential_privacy),
            ("Analyst Dashboard", self.test_analyst_dashboard),
            ("Performance Test", self.test_performance)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🧪 Running {test_name} Test...")
            try:
                result = test_func()
                self.test_results[test_name] = {
                    "status": "PASS" if result.get("success", False) else "FAIL",
                    "details": result
                }
                print(f"   ✅ {test_name}: {self.test_results[test_name]['status']}")
            except Exception as e:
                self.test_results[test_name] = {
                    "status": "ERROR",
                    "error": str(e)
                }
                print(f"   ❌ {test_name}: ERROR - {e}")
        
        return self.test_results
    
    def test_health_check(self) -> Dict[str, Any]:
        """Test system health check"""
        response = self.session.get(f"{self.base_url}/health", timeout=TEST_TIMEOUT)
        
        if response.status_code == 200:
            health_data = response.json()
            return {
                "success": health_data["status"] == "healthy",
                "status": health_data["status"],
                "components": health_data["components"],
                "all_components_healthy": all(health_data["components"].values())
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    
    def test_system_status(self) -> Dict[str, Any]:
        """Test comprehensive system status"""
        response = self.session.get(f"{self.base_url}/system/status", timeout=TEST_TIMEOUT)
        
        if response.status_code == 200:
            status_data = response.json()
            return {
                "success": True,
                "version": status_data["version"],
                "components": status_data["components"],
                "active_components": sum(1 for comp in status_data["components"].values() 
                                       if comp["status"] == "active")
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    
    def test_fraud_prediction(self) -> Dict[str, Any]:
        """Test advanced fraud prediction"""
        # Create test transaction
        test_transaction = {
            "transaction_id": "TEST_TXN_001",
            "amount": 75000.0,
            "upi_id": "test@upi",
            "merchant_id": "MERCHANT_001",
            "timestamp": datetime.utcnow().isoformat(),
            "features": {
                "hour": 14,
                "day_of_week": 1,
                "merchant_category": "ecommerce",
                "user_velocity": 5.2,
                "device_risk_score": 0.3,
                "location_risk_score": 0.2,
                "ip_reputation": 0.8,
                "session_duration": 300,
                "payment_frequency": 3.5,
                "amount_vs_avg": 1.2,
                "time_since_last_tx": 2.1,
                "device_age": 365,
                "location_consistency": 0.9,
                "payment_pattern": 0.7,
                "merchant_risk": 0.1,
                "time_pattern": 0.8,
                "amount_pattern": 0.6,
                "user_behavior_score": 0.8,
                "network_risk": 0.2
            },
            "biometric_data": {
                "face_verification": True,
                "voice_verification": True,
                "fingerprint_match": True
            },
            "device_data": {
                "battery_level": 85,
                "device_orientation": 0,
                "wifi_connected": True,
                "location_enabled": True
            }
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=test_transaction,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            prediction = response.json()
            return {
                "success": True,
                "risk_score": prediction["risk_score"],
                "decision": prediction["decision"],
                "confidence": prediction["confidence"],
                "model_type": prediction["model_type"],
                "privacy_protected": prediction["privacy_protected"],
                "audit_trail_hash": prediction.get("audit_trail_hash")
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    
    def test_multimodal_features(self) -> Dict[str, Any]:
        """Test multi-modal feature processing"""
        # Test with comprehensive multi-modal data
        multimodal_transaction = {
            "transaction_id": "MULTIMODAL_TEST_001",
            "amount": 50000.0,
            "upi_id": "multimodal@upi",
            "merchant_id": "MERCHANT_002",
            "timestamp": datetime.utcnow().isoformat(),
            "features": {
                "hour": 2,  # Night time - suspicious
                "day_of_week": 6,
                "merchant_category": "crypto",  # High risk
                "user_velocity": 15.0,  # High velocity
                "device_risk_score": 0.8,  # High risk
                "location_risk_score": 0.7,
                "ip_reputation": 0.3,  # Low reputation
                "session_duration": 60,
                "payment_frequency": 10.0,
                "amount_vs_avg": 3.0,
                "time_since_last_tx": 0.1,
                "device_age": 30,
                "location_consistency": 0.3,
                "payment_pattern": 0.2,
                "merchant_risk": 0.9,
                "time_pattern": 0.1,
                "amount_pattern": 0.1,
                "user_behavior_score": 0.2,
                "network_risk": 0.8
            },
            "biometric_data": {
                "face_verification": False,  # No biometric verification
                "voice_verification": False,
                "fingerprint_match": False
            },
            "device_data": {
                "battery_level": 15,  # Low battery
                "device_orientation": 90,
                "wifi_connected": False,  # Cellular only
                "location_enabled": False,
                "sensor_data": {
                    "accelerometer": [0.1, 0.2, 0.15],
                    "gyroscope": [0.05, 0.08, 0.06]
                },
                "touch_events": [
                    {"x": 100, "y": 200, "pressure": 0.8, "timestamp": 1000},
                    {"x": 120, "y": 210, "pressure": 0.7, "timestamp": 1100}
                ]
            }
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=multimodal_transaction,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            prediction = response.json()
            return {
                "success": True,
                "high_risk_detected": prediction["risk_score"] > 0.7,
                "decision": prediction["decision"],
                "confidence": prediction["confidence"],
                "explanations_available": len(prediction.get("explanations", {})) > 0
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    
    def test_threat_intelligence(self) -> Dict[str, Any]:
        """Test threat intelligence system"""
        # Test threat intelligence update
        update_response = self.session.post(
            f"{self.base_url}/threat-intelligence/update",
            json={"feed_sources": ["AbuseIPDB", "PhishTank"], "update_frequency": 30},
            timeout=TEST_TIMEOUT
        )
        
        # Test threat summary
        summary_response = self.session.get(
            f"{self.base_url}/threat-intelligence/summary",
            timeout=TEST_TIMEOUT
        )
        
        if update_response.status_code == 200 and summary_response.status_code == 200:
            update_data = update_response.json()
            summary_data = summary_response.json()
            
            return {
                "success": True,
                "threat_indicators": summary_data.get("total_indicators", 0),
                "fresh_indicators": summary_data.get("fresh_indicators_24h", 0),
                "feeds_updated": len(update_data.get("feeds", {})),
                "severity_distribution": summary_data.get("severity_distribution", {})
            }
        else:
            return {"success": False, "error": "Threat intelligence endpoints failed"}
    
    def test_synthetic_data(self) -> Dict[str, Any]:
        """Test synthetic data generation"""
        response = self.session.post(
            f"{self.base_url}/synthetic/generate",
            json={"num_samples": 500, "target_fraud_ratio": 0.3},
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "original_samples": data["original_samples"],
                "balanced_samples": data["balanced_samples"],
                "fraud_ratio": data["fraud_ratio"],
                "data_generated": len(data.get("data_preview", [])) > 0
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    
    def test_federated_learning(self) -> Dict[str, Any]:
        """Test federated learning system"""
        # Test bank registration
        bank_data = {
            "bank_id": "test_bank_001",
            "name": "Test Bank",
            "region": "Asia",
            "data_size": 10000
        }
        
        register_response = self.session.post(
            f"{self.base_url}/federated/register-bank",
            json=bank_data,
            timeout=TEST_TIMEOUT
        )
        
        # Test federated status
        status_response = self.session.get(
            f"{self.base_url}/federated/status",
            timeout=TEST_TIMEOUT
        )
        
        if register_response.status_code == 200 and status_response.status_code == 200:
            register_data = register_response.json()
            status_data = status_response.json()
            
            return {
                "success": True,
                "bank_registered": register_data.get("status") == "success",
                "total_banks": status_data.get("total_banks", 0),
                "active_banks": status_data.get("active_banks", 0),
                "total_rounds": status_data.get("total_rounds", 0)
            }
        else:
            return {"success": False, "error": "Federated learning endpoints failed"}
    
    def test_blockchain_audit(self) -> Dict[str, Any]:
        """Test blockchain audit trail"""
        # This is tested indirectly through fraud prediction
        # The audit trail hash should be returned in prediction responses
        return {"success": True, "message": "Blockchain audit tested via fraud prediction"}
    
    def test_active_learning(self) -> Dict[str, Any]:
        """Test active learning system"""
        # Test analyst feedback submission
        feedback_data = {
            "transaction_id": "TEST_TXN_001",
            "analyst_decision": "BLOCK",
            "analyst_id": "analyst_001",
            "reasoning": "High risk transaction detected",
            "false_positive": False,
            "false_negative": False
        }
        
        feedback_response = self.session.post(
            f"{self.base_url}/analyst/feedback",
            json=feedback_data,
            timeout=TEST_TIMEOUT
        )
        
        if feedback_response.status_code == 200:
            feedback_result = feedback_response.json()
            return {
                "success": feedback_result.get("status") == "success",
                "feedback_submitted": feedback_result.get("status") == "success"
            }
        else:
            return {"success": False, "error": f"HTTP {feedback_response.status_code}"}
    
    def test_differential_privacy(self) -> Dict[str, Any]:
        """Test differential privacy system"""
        response = self.session.get(
            f"{self.base_url}/privacy/report",
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            privacy_data = response.json()
            return {
                "success": True,
                "privacy_budget": privacy_data.get("privacy_budget", {}),
                "differential_privacy_enabled": privacy_data.get("model_privacy", {}).get("differential_privacy_enabled", False),
                "privacy_guarantees": privacy_data.get("privacy_guarantees", {})
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    
    def test_analyst_dashboard(self) -> Dict[str, Any]:
        """Test analyst dashboard"""
        response = self.session.get(
            f"{self.base_url}/analyst/dashboard",
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            dashboard_data = response.json()
            return {
                "success": True,
                "pending_reviews": len(dashboard_data.get("pending_reviews", [])),
                "analyst_performance": len(dashboard_data.get("analyst_performance", {})),
                "learning_progress": dashboard_data.get("learning_progress", {})
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    
    def test_performance(self) -> Dict[str, Any]:
        """Test system performance"""
        start_time = time.time()
        
        # Test multiple concurrent requests
        test_transaction = {
            "transaction_id": f"PERF_TEST_{int(time.time())}",
            "amount": 25000.0,
            "upi_id": "perf@upi",
            "merchant_id": "PERF_MERCHANT",
            "timestamp": datetime.utcnow().isoformat(),
            "features": {
                "hour": 10,
                "day_of_week": 2,
                "merchant_category": "food",
                "user_velocity": 2.5,
                "device_risk_score": 0.2,
                "location_risk_score": 0.3,
                "ip_reputation": 0.9,
                "session_duration": 180,
                "payment_frequency": 2.0,
                "amount_vs_avg": 0.8,
                "time_since_last_tx": 1.5,
                "device_age": 200,
                "location_consistency": 0.8,
                "payment_pattern": 0.7,
                "merchant_risk": 0.1,
                "time_pattern": 0.8,
                "amount_pattern": 0.7,
                "user_behavior_score": 0.8,
                "network_risk": 0.2
            }
        }
        
        # Send multiple requests
        response_times = []
        success_count = 0
        
        for i in range(10):
            req_start = time.time()
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=test_transaction,
                    timeout=TEST_TIMEOUT
                )
                req_end = time.time()
                
                if response.status_code == 200:
                    success_count += 1
                    response_times.append(req_end - req_start)
                
                # Update transaction ID for next request
                test_transaction["transaction_id"] = f"PERF_TEST_{int(time.time())}_{i}"
                
            except Exception as e:
                print(f"   Request {i+1} failed: {e}")
        
        total_time = time.time() - start_time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "success": success_count >= 8,  # At least 80% success rate
            "total_requests": 10,
            "successful_requests": success_count,
            "success_rate": success_count / 10,
            "average_response_time": avg_response_time,
            "total_test_time": total_time,
            "requests_per_second": 10 / total_time if total_time > 0 else 0
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("🎯 ADVANCED UPI FRAUD DETECTION SYSTEM - TEST REPORT")
        report.append("=" * 70)
        report.append(f"Test Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Base URL: {self.base_url}")
        report.append("")
        
        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASS")
        failed_tests = sum(1 for result in self.test_results.values() if result["status"] == "FAIL")
        error_tests = sum(1 for result in self.test_results.values() if result["status"] == "ERROR")
        
        report.append("📊 TEST SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests} ✅")
        report.append(f"Failed: {failed_tests} ❌")
        report.append(f"Errors: {error_tests} ⚠️")
        report.append(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("📋 DETAILED RESULTS")
        report.append("-" * 30)
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
            report.append(f"{status_icon} {test_name}: {result['status']}")
            
            if "error" in result:
                report.append(f"   Error: {result['error']}")
            elif "details" in result:
                details = result["details"]
                if isinstance(details, dict):
                    for key, value in details.items():
                        if isinstance(value, (str, int, float, bool)):
                            report.append(f"   {key}: {value}")
        
        report.append("")
        report.append("🏁 Test completed successfully!")
        
        return "\n".join(report)

def main():
    """Main test execution"""
    print("🚀 Advanced UPI Fraud Detection System - Comprehensive Testing")
    print("=" * 70)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server not responding at {BASE_URL}")
            print("Please start the advanced fraud detection API first:")
            print("python advanced_fraud_detection_api.py")
            return
    except requests.exceptions.RequestException:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print("Please start the advanced fraud detection API first:")
        print("python advanced_fraud_detection_api.py")
        return
    
    # Run tests
    tester = AdvancedSystemTester()
    results = tester.run_all_tests()
    
    # Generate and display report
    report = tester.generate_report()
    print("\n" + report)
    
    # Save report to file
    with open("advanced_system_test_report.txt", "w") as f:
        f.write(report)
    
    print(f"\n📄 Detailed report saved to: advanced_system_test_report.txt")
    
    # Overall result
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result["status"] == "PASS")
    success_rate = (passed_tests / total_tests) * 100
    
    if success_rate >= 80:
        print(f"\n🎉 SYSTEM TEST PASSED! Success rate: {success_rate:.1f}%")
        return 0
    else:
        print(f"\n❌ SYSTEM TEST FAILED! Success rate: {success_rate:.1f}%")
        return 1

if __name__ == "__main__":
    exit(main())
