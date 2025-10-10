#!/usr/bin/env python3
"""
Complete UPI Fraud Detection System Demo
Shows both frontend and backend working together
"""

import webbrowser
import time
import requests
import json
from datetime import datetime

def check_services():
    """Check if both frontend and backend are running"""
    services = {
        "frontend": False,
        "backend": False
    }
    
    try:
        response = requests.get('http://localhost:3000', timeout=3)
        services["frontend"] = response.status_code == 200
    except:
        pass
    
    try:
        response = requests.get('http://localhost:8000', timeout=3)
        services["backend"] = response.status_code == 200
    except:
        pass
    
    return services

def demo_backend_api():
    """Demo the backend API endpoints"""
    print("\n🔧 BACKEND API DEMONSTRATION")
    print("-" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test different endpoints
    endpoints = [
        ("/", "Root endpoint"),
        ("/health", "Health check"),
        ("/docs", "API documentation")
    ]
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {description}: {endpoint}")
                if endpoint == "/":
                    data = response.json()
                    print(f"   Response: {data.get('message', 'N/A')}")
            else:
                print(f"❌ {description}: {endpoint} (Status: {response.status_code})")
        except Exception as e:
            print(f"❌ {description}: {endpoint} (Error: {str(e)[:50]})")

def demo_transaction_analysis():
    """Demo transaction analysis"""
    print("\n💳 TRANSACTION ANALYSIS DEMO")
    print("-" * 50)
    
    sample_transactions = [
        {
            "transaction_id": "TXN1234567",
            "amount": 25000,
            "merchant": "Amazon",
            "location": "Mumbai"
        },
        {
            "transaction_id": "TXN1234568",
            "amount": 150000,
            "merchant": "Unknown Merchant",
            "location": "Suspicious Location"
        },
        {
            "transaction_id": "TXN1234569",
            "amount": 5000,
            "merchant": "Swiggy",
            "location": "Bangalore"
        }
    ]
    
    for i, txn in enumerate(sample_transactions, 1):
        print(f"\n{i}. Analyzing Transaction: {txn['transaction_id']}")
        print(f"   Amount: ₹{txn['amount']:,}")
        print(f"   Merchant: {txn['merchant']}")
        print(f"   Location: {txn['location']}")
        
        # Try to analyze with backend
        try:
            response = requests.post(
                "http://localhost:8000/api/analyze",
                json=txn,
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Risk Score: {result['risk_score']:.2f}")
                print(f"   ✅ Risk Level: {result['risk_level'].upper()}")
                print(f"   ✅ Recommendation: {result['recommendation']}")
            else:
                print(f"   ❌ Analysis failed (Status: {response.status_code})")
        except Exception as e:
            print(f"   ❌ Analysis failed (Error: {str(e)[:50]})")

def show_system_status():
    """Show complete system status"""
    print("\n" + "="*70)
    print("🎯 UPI FRAUD DETECTION SYSTEM - COMPLETE STATUS")
    print("="*70)
    
    services = check_services()
    
    print("📊 Service Status:")
    print(f"   Frontend (Port 3000): {'✅ Running' if services['frontend'] else '❌ Not Running'}")
    print(f"   Backend (Port 8000):  {'✅ Running' if services['backend'] else '❌ Not Running'}")
    
    if services['frontend'] and services['backend']:
        print("\n🎉 COMPLETE SYSTEM IS OPERATIONAL!")
        print("🌐 Frontend Dashboard: http://localhost:3000")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        
        print("\n✨ Features Available:")
        print("   • Real-time fraud detection dashboard")
        print("   • Interactive transaction analysis")
        print("   • ML model performance monitoring")
        print("   • Advanced analytics and reporting")
        print("   • Security alerts and notifications")
        print("   • Responsive web interface")
        
        return True
    else:
        print("\n⚠️  SYSTEM NOT FULLY OPERATIONAL")
        if not services['frontend']:
            print("   • Start frontend: cd frontend && python server.py")
        if not services['backend']:
            print("   • Start backend: python simple_backend_api.py")
        return False

def open_dashboard():
    """Open the dashboard in browser"""
    print("\n🌐 Opening Dashboard...")
    webbrowser.open('http://localhost:3000')
    time.sleep(2)

def show_usage_instructions():
    """Show how to use the system"""
    print("\n📖 HOW TO USE THE SYSTEM")
    print("-" * 50)
    print("1. 🌐 Frontend Dashboard (http://localhost:3000):")
    print("   • Navigate through different sections using the sidebar")
    print("   • View real-time metrics and transaction feed")
    print("   • Analyze transactions using the form")
    print("   • Monitor ML model performance")
    print("   • Check security alerts")
    print("   • Adjust fraud detection settings")
    
    print("\n2. 🔧 Backend API (http://localhost:8000):")
    print("   • RESTful API for fraud detection")
    print("   • Interactive documentation at /docs")
    print("   • Real-time transaction analysis")
    print("   • Dashboard metrics endpoint")
    print("   • Model status and health checks")
    
    print("\n3. 💡 Key Features to Try:")
    print("   • Fill out transaction analysis form")
    print("   • Watch real-time data updates")
    print("   • Toggle auto-refresh functionality")
    print("   • Navigate between different sections")
    print("   • View API documentation")

def main():
    """Main demo function"""
    print("🚀 UPI FRAUD DETECTION SYSTEM - COMPLETE DEMO")
    print("="*70)
    
    # Check system status
    system_ready = show_system_status()
    
    if system_ready:
        # Demo backend API
        demo_backend_api()
        
        # Demo transaction analysis
        demo_transaction_analysis()
        
        # Show usage instructions
        show_usage_instructions()
        
        # Open dashboard
        open_dashboard()
        
        print("\n🎉 DEMO COMPLETE!")
        print("The complete UPI fraud detection system is now running.")
        print("Both frontend and backend are connected and operational.")
        
    else:
        print("\n❌ Please start the required services first.")
        print("Run this demo again once both services are running.")

if __name__ == "__main__":
    main()
