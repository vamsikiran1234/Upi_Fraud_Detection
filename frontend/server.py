#!/usr/bin/env python3
"""
Simple HTTP server to serve the UPI Fraud Detection Frontend
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for API integration
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.end_headers()

def start_frontend_server(port=3000):
    """Start the frontend server"""
    try:
        # Change to frontend directory
        frontend_dir = Path(__file__).parent
        os.chdir(frontend_dir)
        
        # Create server
        with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
            print(f"🚀 Frontend server starting on http://localhost:{port}")
            print(f"📁 Serving files from: {frontend_dir}")
            print("🌐 Opening browser...")
            
            # Open browser
            webbrowser.open(f'http://localhost:{port}')
            
            print("\n" + "="*60)
            print("🎯 UPI FRAUD DETECTION FRONTEND")
            print("="*60)
            print("✅ Frontend server is running!")
            print("📊 Dashboard: http://localhost:3000")
            print("🔧 Features:")
            print("   • Real-time transaction monitoring")
            print("   • Interactive fraud analysis")
            print("   • Advanced ML model visualization")
            print("   • Responsive design")
            print("   • Live alerts and notifications")
            print("\n💡 To stop the server, press Ctrl+C")
            print("="*60)
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        sys.exit(0)
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use. Trying port {port + 1}...")
            start_frontend_server(port + 1)
        else:
            print(f"❌ Error starting server: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_frontend_server()
