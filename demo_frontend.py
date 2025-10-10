#!/usr/bin/env python3
"""
Demo script to showcase the UPI Fraud Detection Frontend
"""

import webbrowser
import time
import requests
import json
from pathlib import Path

def check_frontend_status():
    """Check if frontend server is running"""
    try:
        response = requests.get('http://localhost:3000', timeout=5)
        return response.status_code == 200
    except:
        return False

def open_frontend():
    """Open the frontend in browser"""
    print("🌐 Opening UPI Fraud Detection Frontend...")
    webbrowser.open('http://localhost:3000')
    time.sleep(2)

def show_frontend_features():
    """Display frontend features"""
    print("\n" + "="*70)
    print("🎯 UPI FRAUD DETECTION FRONTEND - LIVE DEMO")
    print("="*70)
    print("✅ Frontend Server: http://localhost:3000")
    print("📊 Dashboard Features:")
    print("   • Real-time transaction monitoring")
    print("   • Interactive fraud analysis form")
    print("   • Advanced ML model visualization")
    print("   • Live alerts and notifications")
    print("   • Responsive design for all devices")
    print("\n🔧 Available Sections:")
    print("   1. 📈 Dashboard - Real-time metrics and transaction feed")
    print("   2. 💳 Transactions - Manual transaction analysis")
    print("   3. 📊 Analytics - Advanced ML model statistics")
    print("   4. 🧠 Models - Model performance monitoring")
    print("   5. 🚨 Alerts - Security alerts and notifications")
    print("   6. ⚙️  Settings - Fraud detection configuration")
    print("\n🎨 Visual Features:")
    print("   • Modern glassmorphism design")
    print("   • Gradient backgrounds and animations")
    print("   • Real-time data updates every 5-10 seconds")
    print("   • Color-coded risk indicators")
    print("   • Interactive charts and progress bars")
    print("   • Toast notifications for user feedback")
    print("\n💡 Try These Features:")
    print("   • Navigate between sections using the sidebar")
    print("   • Fill out the transaction analysis form")
    print("   • Watch real-time metrics update")
    print("   • Toggle auto-refresh on/off")
    print("   • Adjust fraud detection thresholds")
    print("="*70)

def demo_transaction_analysis():
    """Demo the transaction analysis feature"""
    print("\n🔍 TRANSACTION ANALYSIS DEMO")
    print("-" * 40)
    print("Sample transaction data:")
    
    sample_transactions = [
        {
            "id": "TXN1234567",
            "amount": 25000,
            "merchant": "Amazon",
            "location": "Mumbai",
            "risk_level": "Low"
        },
        {
            "id": "TXN1234568", 
            "amount": 150000,
            "merchant": "Unknown Merchant",
            "location": "Suspicious Location",
            "risk_level": "High"
        },
        {
            "id": "TXN1234569",
            "amount": 5000,
            "merchant": "Swiggy",
            "location": "Bangalore",
            "risk_level": "Medium"
        }
    ]
    
    for i, txn in enumerate(sample_transactions, 1):
        print(f"\n{i}. Transaction ID: {txn['id']}")
        print(f"   Amount: ₹{txn['amount']:,}")
        print(f"   Merchant: {txn['merchant']}")
        print(f"   Location: {txn['location']}")
        print(f"   Risk Level: {txn['risk_level']}")
    
    print(f"\n💡 Try analyzing these transactions in the frontend!")

def show_technical_details():
    """Show technical implementation details"""
    print("\n🛠️  TECHNICAL IMPLEMENTATION")
    print("-" * 40)
    print("Frontend Stack:")
    print("   • HTML5 - Semantic structure and accessibility")
    print("   • CSS3 - Modern styling with flexbox/grid")
    print("   • JavaScript ES6+ - Interactive functionality")
    print("   • Font Awesome - Professional icons")
    print("   • Google Fonts - Inter font family")
    print("\nKey Features:")
    print("   • Responsive design (mobile-first)")
    print("   • Real-time data updates")
    print("   • Mock API integration ready")
    print("   • CORS-enabled for backend connection")
    print("   • Progressive enhancement")
    print("   • Accessibility compliant")
    print("\nFile Structure:")
    print("   frontend/")
    print("   ├── index.html      # Main HTML structure")
    print("   ├── styles.css      # CSS styling and animations")
    print("   ├── script.js       # JavaScript functionality")
    print("   └── server.py       # HTTP server")

def main():
    """Main demo function"""
    print("🚀 Starting UPI Fraud Detection Frontend Demo...")
    
    # Check if frontend is running
    if check_frontend_status():
        print("✅ Frontend server is running!")
        open_frontend()
        show_frontend_features()
        demo_transaction_analysis()
        show_technical_details()
        
        print("\n🎉 DEMO COMPLETE!")
        print("The frontend is now open in your browser.")
        print("Navigate through the different sections to explore all features.")
        
    else:
        print("❌ Frontend server is not running.")
        print("Please start it first with: cd frontend && python server.py")
        print("Or use PowerShell: cd frontend; python server.py")

if __name__ == "__main__":
    main()
