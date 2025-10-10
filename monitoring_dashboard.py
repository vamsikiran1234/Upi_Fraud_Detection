#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for UPI Fraud Detection System
Provides live metrics, alerts, and system health monitoring
"""

import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
import threading
import queue
import random

# Initialize FastAPI app
app = FastAPI(
    title="UPI Fraud Detection Monitoring Dashboard",
    description="Real-time monitoring and analytics dashboard",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
connected_clients: List[WebSocket] = []
metrics_queue = queue.Queue()
is_monitoring = False

# System endpoints
BASIC_API_URL = "http://localhost:8000"
ADVANCED_API_URL = "http://localhost:8001"

class SystemMetrics:
    """System metrics collector"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'fraud_detected': 0,
            'blocked_transactions': 0,
            'challenged_transactions': 0,
            'allowed_transactions': 0,
            'average_response_time': 0,
            'system_uptime': 0,
            'active_connections': 0,
            'error_rate': 0,
            'throughput_per_minute': 0,
            'model_accuracy': 0.968,
            'false_positive_rate': 0.02,
            'last_updated': datetime.utcnow().isoformat()
        }
        self.start_time = time.time()
    
    def update_metrics(self, response_data: Dict[str, Any]):
        """Update metrics based on API response"""
        self.metrics['total_requests'] += 1
        
        if response_data.get('decision') == 'BLOCK':
            self.metrics['blocked_transactions'] += 1
            self.metrics['fraud_detected'] += 1
        elif response_data.get('decision') == 'CHALLENGE':
            self.metrics['challenged_transactions'] += 1
        else:
            self.metrics['allowed_transactions'] += 1
        
        # Update response time
        processing_time = response_data.get('processing_time_ms', 0)
        current_avg = self.metrics['average_response_time']
        total_requests = self.metrics['total_requests']
        self.metrics['average_response_time'] = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        
        # Update system uptime
        self.metrics['system_uptime'] = time.time() - self.start_time
        
        # Update throughput
        self.metrics['throughput_per_minute'] = self.metrics['total_requests'] / (self.metrics['system_uptime'] / 60)
        
        # Update error rate (simulate some errors)
        if random.random() < 0.01:  # 1% error rate
            self.metrics['error_rate'] = min(1.0, self.metrics['error_rate'] + 0.01)
        
        self.metrics['last_updated'] = datetime.utcnow().isoformat()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()

# Global metrics instance
system_metrics = SystemMetrics()

async def monitor_systems():
    """Monitor both basic and advanced systems"""
    global is_monitoring
    is_monitoring = True
    
    while is_monitoring:
        try:
            # Test basic system
            try:
                response = requests.get(f"{BASIC_API_URL}/health", timeout=5)
                if response.status_code == 200:
                    basic_status = "healthy"
                else:
                    basic_status = "unhealthy"
            except:
                basic_status = "offline"
            
            # Test advanced system
            try:
                response = requests.get(f"{ADVANCED_API_URL}/health", timeout=5)
                if response.status_code == 200:
                    advanced_status = "healthy"
                else:
                    advanced_status = "unhealthy"
            except:
                advanced_status = "offline"
            
            # Create monitoring data
            monitoring_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'basic_system': {
                    'status': basic_status,
                    'url': BASIC_API_URL
                },
                'advanced_system': {
                    'status': advanced_status,
                    'url': ADVANCED_API_URL
                },
                'metrics': system_metrics.get_metrics(),
                'alerts': generate_alerts()
            }
            
            # Send to all connected clients
            if connected_clients:
                message = json.dumps(monitoring_data)
                for client in connected_clients.copy():
                    try:
                        await client.send_text(message)
                    except:
                        connected_clients.remove(client)
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            print(f"Monitoring error: {e}")
            await asyncio.sleep(5)

def generate_alerts() -> List[Dict[str, Any]]:
    """Generate system alerts"""
    alerts = []
    metrics = system_metrics.get_metrics()
    
    # High error rate alert
    if metrics['error_rate'] > 0.05:
        alerts.append({
            'type': 'error',
            'message': f"High error rate detected: {metrics['error_rate']:.2%}",
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'high'
        })
    
    # High fraud rate alert
    fraud_rate = metrics['fraud_detected'] / max(metrics['total_requests'], 1)
    if fraud_rate > 0.1:
        alerts.append({
            'type': 'fraud',
            'message': f"High fraud rate detected: {fraud_rate:.2%}",
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'critical'
        })
    
    # High response time alert
    if metrics['average_response_time'] > 1000:  # 1 second
        alerts.append({
            'type': 'performance',
            'message': f"High response time: {metrics['average_response_time']:.0f}ms",
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'medium'
        })
    
    return alerts

@app.get("/")
async def dashboard():
    """Serve the monitoring dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>UPI Fraud Detection - Monitoring Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
            .metric-label { color: #7f8c8d; margin-top: 5px; }
            .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
            .status-healthy { background: #27ae60; }
            .status-unhealthy { background: #e74c3c; }
            .status-offline { background: #95a5a6; }
            .alerts { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .alert { padding: 10px; margin: 5px 0; border-radius: 4px; }
            .alert-error { background: #fdf2f2; border-left: 4px solid #e74c3c; }
            .alert-fraud { background: #fdf2f2; border-left: 4px solid #c0392b; }
            .alert-performance { background: #fff3cd; border-left: 4px solid #f39c12; }
            .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .connection-status { position: fixed; top: 20px; right: 20px; padding: 10px; border-radius: 4px; }
            .connected { background: #27ae60; color: white; }
            .disconnected { background: #e74c3c; color: white; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 UPI Fraud Detection - Real-time Monitoring</h1>
            <p>Live system metrics and performance monitoring</p>
        </div>
        
        <div class="connection-status" id="connectionStatus">
            <span id="connectionText">Connecting...</span>
        </div>
        
        <div class="metrics-grid" id="metricsGrid">
            <!-- Metrics will be populated by JavaScript -->
        </div>
        
        <div class="chart-container">
            <h3>📊 System Performance</h3>
            <div id="performanceChart">
                <p>Loading performance data...</p>
            </div>
        </div>
        
        <div class="alerts">
            <h3>🚨 System Alerts</h3>
            <div id="alertsContainer">
                <p>Loading alerts...</p>
            </div>
        </div>
        
        <script>
            let ws;
            let reconnectInterval;
            
            function connect() {
                ws = new WebSocket('ws://localhost:8002/ws');
                
                ws.onopen = function() {
                    console.log('Connected to monitoring dashboard');
                    document.getElementById('connectionStatus').className = 'connection-status connected';
                    document.getElementById('connectionText').textContent = 'Connected';
                    clearInterval(reconnectInterval);
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                ws.onclose = function() {
                    console.log('Disconnected from monitoring dashboard');
                    document.getElementById('connectionStatus').className = 'connection-status disconnected';
                    document.getElementById('connectionText').textContent = 'Disconnected';
                    reconnectInterval = setInterval(connect, 5000);
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            }
            
            function updateDashboard(data) {
                updateMetrics(data.metrics);
                updateAlerts(data.alerts);
                updateSystemStatus(data.basic_system, data.advanced_system);
            }
            
            function updateMetrics(metrics) {
                const grid = document.getElementById('metricsGrid');
                grid.innerHTML = `
                    <div class="metric-card">
                        <div class="metric-value">${metrics.total_requests}</div>
                        <div class="metric-label">Total Requests</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics.fraud_detected}</div>
                        <div class="metric-label">Fraud Detected</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics.blocked_transactions}</div>
                        <div class="metric-label">Blocked Transactions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics.challenged_transactions}</div>
                        <div class="metric-label">Challenged Transactions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics.average_response_time.toFixed(0)}ms</div>
                        <div class="metric-label">Avg Response Time</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics.throughput_per_minute.toFixed(1)}</div>
                        <div class="metric-label">Throughput/min</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(metrics.model_accuracy * 100).toFixed(1)}%</div>
                        <div class="metric-label">Model Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(metrics.error_rate * 100).toFixed(2)}%</div>
                        <div class="metric-label">Error Rate</div>
                    </div>
                `;
            }
            
            function updateAlerts(alerts) {
                const container = document.getElementById('alertsContainer');
                if (alerts.length === 0) {
                    container.innerHTML = '<p style="color: #27ae60;">✅ No active alerts</p>';
                    return;
                }
                
                container.innerHTML = alerts.map(alert => `
                    <div class="alert alert-${alert.type}">
                        <strong>${alert.type.toUpperCase()}:</strong> ${alert.message}
                        <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
                    </div>
                `).join('');
            }
            
            function updateSystemStatus(basicSystem, advancedSystem) {
                // Add system status indicators to the header
                const header = document.querySelector('.header');
                const existingStatus = document.getElementById('systemStatus');
                if (existingStatus) {
                    existingStatus.remove();
                }
                
                const statusDiv = document.createElement('div');
                statusDiv.id = 'systemStatus';
                statusDiv.innerHTML = `
                    <div style="margin-top: 10px;">
                        <span class="status-indicator status-${basicSystem.status}"></span>
                        Basic System: ${basicSystem.status.toUpperCase()}
                        <span class="status-indicator status-${advancedSystem.status}" style="margin-left: 20px;"></span>
                        Advanced System: ${advancedSystem.status.toUpperCase()}
                    </div>
                `;
                header.appendChild(statusDiv);
            }
            
            // Connect on page load
            connect();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

@app.get("/api/metrics")
async def get_metrics():
    """Get current system metrics"""
    return system_metrics.get_metrics()

@app.get("/api/systems/status")
async def get_systems_status():
    """Get status of both systems"""
    try:
        # Test basic system
        basic_response = requests.get(f"{BASIC_API_URL}/health", timeout=5)
        basic_status = "healthy" if basic_response.status_code == 200 else "unhealthy"
    except:
        basic_status = "offline"
    
    try:
        # Test advanced system
        advanced_response = requests.get(f"{ADVANCED_API_URL}/health", timeout=5)
        advanced_status = "healthy" if advanced_response.status_code == 200 else "unhealthy"
    except:
        advanced_status = "offline"
    
    return {
        "basic_system": {
            "status": basic_status,
            "url": BASIC_API_URL
        },
        "advanced_system": {
            "status": advanced_status,
            "url": ADVANCED_API_URL
        },
        "monitoring_dashboard": {
            "status": "healthy",
            "url": "http://localhost:8002"
        }
    }

@app.post("/api/simulate/transaction")
async def simulate_transaction():
    """Simulate a transaction for testing"""
    # Simulate transaction data
    transaction_data = {
        "transaction_id": f"SIM_{int(time.time())}",
        "upi_id": "test@paytm",
        "amount": random.uniform(100, 100000),
        "merchant_id": f"MERCHANT_{random.randint(1, 100)}",
        "merchant_category": random.choice(["food", "ecommerce", "crypto", "gambling"]),
        "device_id": f"device_{random.randint(1, 1000)}",
        "ip_address": f"192.168.1.{random.randint(1, 255)}",
        "location": {"lat": 28.6139, "lon": 77.2090},
        "timestamp": datetime.utcnow().isoformat(),
        "payment_method": "UPI"
    }
    
    # Try to send to basic system
    try:
        response = requests.post(f"{BASIC_API_URL}/predict", json=transaction_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            system_metrics.update_metrics(data)
            return {"status": "success", "system": "basic", "data": data}
    except:
        pass
    
    # Try to send to advanced system
    try:
        response = requests.post(f"{ADVANCED_API_URL}/predict", json=transaction_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            system_metrics.update_metrics(data)
            return {"status": "success", "system": "advanced", "data": data}
    except:
        pass
    
    return {"status": "error", "message": "No systems available"}

@app.on_event("startup")
async def startup_event():
    """Start monitoring on startup"""
    print("🚀 Starting UPI Fraud Detection Monitoring Dashboard...")
    print("📊 Dashboard available at: http://localhost:8002")
    print("🔍 API endpoints available at: http://localhost:8002/docs")
    
    # Start monitoring in background
    asyncio.create_task(monitor_systems())

if __name__ == "__main__":
    print("🚀 Starting Monitoring Dashboard...")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
