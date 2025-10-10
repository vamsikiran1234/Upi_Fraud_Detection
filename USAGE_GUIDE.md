# 🎯 UPI Fraud Detection System - Complete Usage Guide

## 🚀 Quick Start

### 1. Start Both Services

**Terminal 1 - Start Backend API:**
```bash
python simple_backend_api.py
```
- Backend runs on: http://localhost:8000
- API docs: http://localhost:8000/docs

**Terminal 2 - Start Frontend:**
```bash
cd frontend
python server.py
```
- Frontend runs on: http://localhost:3000

### 2. Access the System
- **Dashboard**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs

---

## 🌐 Frontend Usage (Web Dashboard)

### 📊 Dashboard Section
- **Real-time Metrics**: View transaction volume, fraud rate, model accuracy
- **Live Transaction Feed**: See recent transactions with risk status
- **Model Performance**: Monitor ML model accuracy and response times
- **Auto-refresh**: Toggle real-time updates on/off

### 💳 Transaction Analysis
1. **Fill the Form**:
   - Transaction ID: Enter any ID (e.g., "TXN1234567")
   - Amount: Enter amount in ₹ (e.g., 25000)
   - Merchant: Enter merchant name (e.g., "Amazon")
   - Location: Enter location (e.g., "Mumbai")

2. **Click "Check for Fraud"**
3. **View Results**:
   - Risk Score (0-100%)
   - Risk Level (Low/Medium/High)
   - Risk Factors breakdown
   - Recommendation
   - Model Confidence

### 📈 Analytics Section
- **Federated Learning**: View participating banks and global model accuracy
- **Blockchain Audit**: See audit trail statistics
- **Threat Intelligence**: Monitor high/medium/low risk IPs

### 🧠 ML Models Section
- **Model Status**: View active models (XGBoost, LightGBM, Random Forest, Isolation Forest)
- **Performance Metrics**: Accuracy, precision, recall for each model
- **Model Actions**: Retrain, deploy, monitor models

### 🚨 Alerts Section
- **Security Alerts**: View high/medium/low priority alerts
- **Alert Details**: Click "View Details" for more information
- **Alert Management**: Dismiss or acknowledge alerts

### ⚙️ Settings Section
- **Fraud Thresholds**: Adjust high/medium risk thresholds
- **Notifications**: Configure email, SMS, push notifications
- **System Configuration**: Modify detection parameters

---

## 🔧 Backend API Usage

### 📚 API Documentation
Visit http://localhost:8000/docs for interactive API documentation

### 🔍 Key Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Dashboard Metrics
```bash
curl http://localhost:8000/api/dashboard/metrics
```

#### 3. Get Transactions
```bash
curl http://localhost:8000/api/transactions?limit=10
```

#### 4. Analyze Transaction
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN1234567",
    "amount": 25000,
    "merchant": "Amazon",
    "location": "Mumbai"
  }'
```

#### 5. Model Status
```bash
curl http://localhost:8000/api/models/status
```

#### 6. Get Alerts
```bash
curl http://localhost:8000/api/alerts
```

---

## 💡 Practical Examples

### Example 1: Analyze a Safe Transaction
```json
{
  "transaction_id": "TXN1234567",
  "amount": 5000,
  "merchant": "Swiggy",
  "location": "Bangalore"
}
```
**Expected Result**: Low risk (0-40% risk score)

### Example 2: Analyze a Risky Transaction
```json
{
  "transaction_id": "TXN1234568",
  "amount": 150000,
  "merchant": "Unknown Merchant",
  "location": "Suspicious Location"
}
```
**Expected Result**: High risk (70-100% risk score)

### Example 3: Analyze a Medium Risk Transaction
```json
{
  "transaction_id": "TXN1234569",
  "amount": 75000,
  "merchant": "Amazon",
  "location": "Mumbai"
}
```
**Expected Result**: Medium risk (40-70% risk score)

---

## 🎨 Frontend Features

### 🎯 Navigation
- **Sidebar**: Click any section to navigate
- **Active Section**: Highlighted in blue
- **Responsive**: Works on mobile, tablet, desktop

### 🔄 Real-time Updates
- **Auto-refresh**: Updates every 5-10 seconds
- **Manual Refresh**: Click refresh button
- **Live Metrics**: Transaction volume, fraud rate, etc.

### 🎨 Visual Indicators
- **Green**: Safe/Low risk
- **Yellow**: Medium risk
- **Red**: High risk/Fraud
- **Blue**: System status/Info

### 📱 Responsive Design
- **Mobile**: Collapsible sidebar, touch-friendly
- **Tablet**: Optimized layout
- **Desktop**: Full sidebar, all features visible

---

## 🔧 Backend Features

### 🚀 FastAPI Features
- **Automatic Documentation**: Swagger UI at /docs
- **Type Validation**: Pydantic models
- **CORS Enabled**: Frontend can connect
- **Error Handling**: Proper HTTP status codes

### 📊 Mock Data
- **Realistic Data**: Simulated transaction data
- **Risk Calculation**: Based on amount, merchant, location
- **Model Responses**: Mock ML model outputs
- **Time Simulation**: Processing delays

### 🔒 Security Features
- **CORS Headers**: Cross-origin requests allowed
- **Input Validation**: All inputs validated
- **Error Responses**: Proper error messages

---

## 🛠️ Troubleshooting

### Frontend Not Loading
1. Check if frontend server is running: `curl http://localhost:3000`
2. Restart frontend: `cd frontend && python server.py`

### Backend Not Responding
1. Check if backend is running: `curl http://localhost:8000`
2. Restart backend: `python simple_backend_api.py`

### API Connection Issues
1. Check CORS settings in backend
2. Verify both services are running
3. Check browser console for errors

### Port Conflicts
- Frontend: Change port in `frontend/server.py`
- Backend: Change port in `simple_backend_api.py`

---

## 🎯 Best Practices

### For Frontend Usage
1. **Start with Dashboard**: Get overview of system
2. **Test Transaction Analysis**: Try different scenarios
3. **Monitor Real-time**: Watch metrics update
4. **Check Alerts**: Review security notifications
5. **Adjust Settings**: Configure thresholds

### For Backend Usage
1. **Use API Docs**: Visit /docs for interactive testing
2. **Test Endpoints**: Try different API calls
3. **Monitor Health**: Check /health endpoint
4. **Analyze Transactions**: Use /api/analyze endpoint
5. **Get Metrics**: Use /api/dashboard/metrics

---

## 🎉 Success Indicators

### ✅ System Working Properly
- Frontend loads at http://localhost:3000
- Backend responds at http://localhost:8000
- API docs accessible at http://localhost:8000/docs
- Transaction analysis returns results
- Real-time metrics update
- No console errors in browser

### 🚀 Ready for Production
- Both services running stably
- API endpoints responding correctly
- Frontend-backend communication working
- All features functional
- Responsive design working

---

## 📞 Support

If you encounter any issues:
1. Check the console logs
2. Verify both services are running
3. Test API endpoints individually
4. Check browser developer tools
5. Restart both services if needed

**Happy Fraud Detection! 🛡️**
