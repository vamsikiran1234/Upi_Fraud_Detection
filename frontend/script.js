// UPI Fraud Detection Frontend JavaScript
class FraudDetectionApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.autoRefreshInterval = null;
        this.isAutoRefresh = false;
        this.chart = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadDashboard();
        this.startRealTimeUpdates();
        this.initializeChart();
        this.loadAlerts();
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const section = item.dataset.section;
                this.showSection(section);
                this.updateActiveNav(item);
            });
        });

        // Threshold sliders
        document.getElementById('highRiskThreshold')?.addEventListener('input', (e) => {
            document.querySelector('.threshold-value').textContent = e.target.value;
        });

        document.getElementById('mediumRiskThreshold')?.addEventListener('input', (e) => {
            document.querySelectorAll('.threshold-value')[1].textContent = e.target.value;
        });

        // Check for Fraud button
        document.getElementById('checkForFraudBtn')?.addEventListener('click', () => {
            this.submitTransaction();
        });
        
        // Analyze Transaction button
        document.getElementById('analyzeTransactionBtn')?.addEventListener('click', () => {
            this.analyzeCurrentTransaction();
        });
    }

    showSection(sectionId) {
        // Hide all sections
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
        });

        // Show selected section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
            targetSection.classList.add('fade-in');

            // Load section-specific data
            switch (sectionId) {
                case 'dashboard':
                    this.loadDashboard();
                    break;
                case 'transactions':
                    this.loadTransactions();
                    break;
                case 'analytics':
                    this.loadAnalytics();
                    break;
                case 'models':
                    this.loadModels();
                    break;
                case 'alerts':
                    this.loadAlerts();
                    break;
            }
        }
    }

    updateActiveNav(activeItem) {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        activeItem.classList.add('active');
    }

    async loadDashboard() {
        try {
            this.showLoading();
            
            // Simulate API calls for dashboard data
            const dashboardData = await this.fetchDashboardData();
            this.updateDashboardMetrics(dashboardData);
            this.updateTransactionsList(dashboardData.transactions);
            
        } catch (error) {
            this.showToast('Error loading dashboard data', 'error');
            console.error('Dashboard load error:', error);
        } finally {
            this.hideLoading();
        }
    }

    async fetchDashboardData() {
        try {
            // Try to fetch from backend API
            const response = await fetch(`${this.apiBaseUrl}/api/dashboard/metrics`);
            if (response.ok) {
                const metrics = await response.json();
                const transactionsResponse = await fetch(`${this.apiBaseUrl}/api/transactions?limit=10`);
                const transactions = transactionsResponse.ok ? await transactionsResponse.json() : this.generateMockTransactions(10);
                
                return {
                    transactionVolume: metrics.transaction_volume,
                    fraudRate: metrics.fraud_rate,
                    modelAccuracy: metrics.model_accuracy,
                    responseTime: metrics.response_time,
                    activeTransactions: metrics.active_transactions,
                    fraudDetected: metrics.fraud_detected,
                    transactions: transactions
                };
            }
        } catch (error) {
            console.log('Backend not available, using mock data');
        }
        
        // Fallback to mock data
        return {
            transactionVolume: Math.floor(Math.random() * 10000) + 5000,
            fraudRate: (Math.random() * 0.1).toFixed(3) + '%',
            modelAccuracy: (95 + Math.random() * 4).toFixed(1) + '%',
            responseTime: Math.floor(Math.random() * 50) + 30 + 'ms',
            activeTransactions: Math.floor(Math.random() * 100) + 50,
            fraudDetected: Math.floor(Math.random() * 10),
            transactions: this.generateMockTransactions(10)
        };
    }

    updateDashboardMetrics(data) {
        document.getElementById('transactionVolume').textContent = data.transactionVolume.toLocaleString();
        document.getElementById('fraudRate').textContent = data.fraudRate;
        document.getElementById('modelAccuracy').textContent = data.modelAccuracy;
        document.getElementById('responseTime').textContent = data.responseTime;
        document.getElementById('activeTransactions').textContent = data.activeTransactions;
        document.getElementById('fraudDetected').textContent = data.fraudDetected;
    }

    generateMockTransactions(count) {
        const transactions = [];
        const merchants = ['Amazon', 'Flipkart', 'Swiggy', 'Zomato', 'Uber', 'Ola', 'Paytm', 'PhonePe'];
        const locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad'];
        const statuses = ['safe', 'risky', 'fraud'];

        for (let i = 0; i < count; i++) {
            const amount = Math.floor(Math.random() * 50000) + 100;
            const status = statuses[Math.floor(Math.random() * statuses.length)];
            const merchant = merchants[Math.floor(Math.random() * merchants.length)];
            const location = locations[Math.floor(Math.random() * locations.length)];

            transactions.push({
                id: 'TXN' + (1000000 + i),
                amount: amount,
                merchant: merchant,
                location: location,
                status: status,
                timestamp: new Date(Date.now() - Math.random() * 3600000).toLocaleTimeString()
            });
        }

        return transactions;
    }

    updateTransactionsList(transactions) {
        const container = document.getElementById('transactionsList');
        if (!container) return;

        container.innerHTML = transactions.map(txn => `
            <div class="transaction-item">
                <div class="transaction-info">
                    <div class="transaction-id">${txn.id}</div>
                    <div class="transaction-details">
                        ${txn.merchant} • ${txn.location} • ${txn.timestamp}
                    </div>
                </div>
                <div class="transaction-amount">₹${txn.amount.toLocaleString()}</div>
                <div class="transaction-status ${txn.status}">${txn.status}</div>
            </div>
        `).join('');
    }

    async submitTransaction() {
        const transactionId = document.getElementById('transactionId').value;
        const amount = document.getElementById('amount').value;
        const merchant = document.getElementById('merchant').value;
        const location = document.getElementById('location').value;

        if (!transactionId || !amount || !merchant || !location) {
            this.showToast('Please fill in all fields', 'warning');
            return;
        }

        try {
            this.showLoading();
            
            const transactionData = {
                transaction_id: transactionId,
                amount: parseFloat(amount),
                merchant: merchant,
                location: location,
                timestamp: new Date().toISOString()
            };

            console.log('Transaction data:', transactionData);
            
            const result = await this.analyzeTransaction(transactionData);
            console.log('Analysis result:', result);
            this.displayAnalysisResult(result);
            
        } catch (error) {
            this.showToast('Error analyzing transaction', 'error');
            console.error('Transaction analysis error:', error);
        } finally {
            this.hideLoading();
        }
    }

    async analyzeTransaction(transactionData) {
        try {
            // Format data according to backend API requirements
            const apiData = {
                transaction_id: transactionData.transaction_id,
                upi_id: "user@upi", // Default value
                merchant_id: transactionData.merchant,
                amount: parseFloat(transactionData.amount),
                hour: new Date().getHours(),
                device_risk_score: 0.3, // Default value
                location_risk_score: 0.2, // Default value
                user_behavior_score: 0.5 // Default value
            };
            
            console.log('Sending data to API:', apiData);
            
            // Call backend API using the correct endpoint
            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(apiData)
            });
            
            console.log('Response status:', response.status);
            
            if (response.ok) {
                const result = await response.json();
                console.log('API response:', result);
                return result;
            } else {
                // Log error details for debugging
                const errorText = await response.text();
                console.error('API error:', response.status, errorText);
                
                // Use fallback mock analysis instead of throwing an error
                console.log('Using fallback mock analysis');
                return this.generateMockAnalysis(transactionData);
            }
        } catch (error) {
            console.error('Transaction analysis error:', error);
            this.showToast('Error connecting to backend API', 'error');
            
            // Fallback to mock analysis
            return this.generateMockAnalysis(transactionData);
        }
    }
    
    generateMockAnalysis(transactionData) {
        return new Promise((resolve) => {
            setTimeout(() => {
                const riskScore = Math.random();
                const riskLevel = riskScore > 0.7 ? 'high' : riskScore > 0.4 ? 'medium' : 'low';
                
                resolve({
                    transaction_id: transactionData.transaction_id,
                    risk_score: riskScore,
                    risk_level: riskLevel,
                    factors: this.generateRiskFactors(riskLevel),
                    recommendation: this.getRecommendation(riskLevel),
                    model_confidence: (85 + Math.random() * 15).toFixed(1) + '%',
                    explanation: 'Using mock data (backend API unavailable)',
                    decision: riskLevel === 'high' ? 'BLOCK' : riskLevel === 'medium' ? 'CHALLENGE' : 'ALLOW',
                    processing_time: 15
                });
            }, 1500);
        });
    }

    generateRiskFactors(riskLevel) {
        const factors = {
            low: [
                { name: 'Amount', score: 0.2, status: 'safe' },
                { name: 'Location', score: 0.1, status: 'safe' },
                { name: 'Merchant', score: 0.3, status: 'safe' }
            ],
            medium: [
                { name: 'Amount', score: 0.6, status: 'medium' },
                { name: 'Location', score: 0.4, status: 'medium' },
                { name: 'Merchant', score: 0.5, status: 'medium' }
            ],
            high: [
                { name: 'Amount', score: 0.9, status: 'high' },
                { name: 'Location', score: 0.8, status: 'high' },
                { name: 'Merchant', score: 0.7, status: 'high' }
            ]
        };

        return factors[riskLevel] || factors.low;
    }

    getRecommendation(riskLevel) {
        const recommendations = {
            low: 'Transaction appears safe. Proceed with normal processing.',
            medium: 'Transaction shows some risk indicators. Consider additional verification.',
            high: 'High risk transaction detected. Recommend blocking or manual review.'
        };

        return recommendations[riskLevel] || recommendations.low;
    }

    displayAnalysisResult(result) {
        const container = document.getElementById('analysisResult');
        if (!container) return;

        // Clear any previous error messages
        document.querySelectorAll('.error-message').forEach(el => el.remove());
        
        // Map backend response to frontend format
        // Handle both formats: direct risk_score or risk_score inside risk_factors
        const riskScore = result.risk_score || (result.risk_factors ? result.risk_factors.overall_score : 0);
        const riskLevel = result.risk_level || this.getRiskLevelFromScore(riskScore);
        const decision = result.decision || (riskLevel === 'high' ? 'BLOCK' : riskLevel === 'medium' ? 'CHALLENGE' : 'ALLOW');
        
        // Create result HTML
        container.innerHTML = `
            <div class="result-header ${riskLevel}">
                <h3>Transaction Analysis Result</h3>
                <div class="risk-badge ${riskLevel}">${riskLevel.toUpperCase()} RISK</div>
            </div>
            <div class="result-content">
                <div class="result-summary">
                    <div class="result-item">
                        <span class="label">Risk Score:</span>
                        <span class="value">${(riskScore * 100).toFixed(1)}%</span>
                    </div>
                    <div class="result-item">
                        <span class="label">Decision:</span>
                        <span class="value decision ${decision.toLowerCase()}">${decision}</span>
                    </div>
                    <div class="result-item">
                        <span class="label">Confidence:</span>
                        <span class="value">${result.model_confidence || '90%'}</span>
                    </div>
                    <div class="result-item">
                        <span class="label">Processing Time:</span>
                        <span class="value">${result.processing_time || 15}ms</span>
                    </div>
                </div>
                <div class="risk-factors">
                    <h4>Risk Factors</h4>
                    <div class="factors-list">
                        ${this.renderRiskFactors(result.factors || [])}
                    </div>
                </div>
                <div class="recommendation">
                    <h4>Recommendation</h4>
                    <p>${result.recommendation || 'No recommendation available'}</p>
                </div>
                ${result.explanation ? `<div class="explanation"><p><em>${result.explanation}</em></p></div>` : ''}
            </div>
        `;
        
        // Show the result container
        container.style.display = 'block';
        container.scrollIntoView({ behavior: 'smooth' });
    }
    
    renderRiskFactors(factors) {
        if (!factors || factors.length === 0) {
            return '<p>No risk factors available</p>';
        }
        
        return factors.map(factor => `
            <div class="factor-item ${factor.status}">
                <div class="factor-name">${factor.name}</div>
                <div class="factor-score">${(factor.score * 100).toFixed(1)}%</div>
            </div>
        `).join('');
    }
    
    getRiskLevelFromScore(score) {
        if (score >= 0.7) return 'high';
        if (score >= 0.4) return 'medium';
        return 'low';
    }
    
    // New method to handle the Analyze Transaction button click
    analyzeCurrentTransaction() {
        const transactionId = document.getElementById('transactionId').value;
        const amount = document.getElementById('amount').value;
        const merchant = document.getElementById('merchant').value;
        const location = document.getElementById('location').value;

        if (!transactionId || !amount || !merchant || !location) {
            this.showToast('Please fill in all fields', 'warning');
            return;
        }

        this.submitTransaction();
    }

    initializeChart() {
        const canvas = document.getElementById('fraudChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        // Simple chart implementation
        this.drawSimpleChart(ctx, canvas.width, canvas.height);
    }

    drawSimpleChart(ctx, width, height) {
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw axes
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        
        // Y-axis
        ctx.beginPath();
        ctx.moveTo(40, 20);
        ctx.lineTo(40, height - 40);
        ctx.stroke();
        
        // X-axis
        ctx.beginPath();
        ctx.moveTo(40, height - 40);
        ctx.lineTo(width - 20, height - 40);
        ctx.stroke();
        
        // Draw sample data
        const data = [20, 35, 25, 45, 30, 55, 40, 60, 35, 50];
        const maxValue = Math.max(...data);
        
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        data.forEach((value, index) => {
            const x = 40 + (index * (width - 60) / (data.length - 1));
            const y = height - 40 - (value / maxValue) * (height - 60);
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Draw data points
        ctx.fillStyle = '#667eea';
        data.forEach((value, index) => {
            const x = 40 + (index * (width - 60) / (data.length - 1));
            const y = height - 40 - (value / maxValue) * (height - 60);
            
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    loadAlerts() {
        const alerts = [
            {
                id: 1,
                title: 'High Risk Transaction Detected',
                description: 'Transaction TXN1234567 flagged as high risk due to unusual amount and location.',
                level: 'high',
                timestamp: '2 minutes ago'
            },
            {
                id: 2,
                title: 'Model Performance Degradation',
                description: 'XGBoost model accuracy dropped below 95% threshold.',
                level: 'medium',
                timestamp: '15 minutes ago'
            },
            {
                id: 3,
                title: 'New Threat Intelligence Update',
                description: 'Updated threat intelligence feed with 23 new high-risk IP addresses.',
                level: 'low',
                timestamp: '1 hour ago'
            }
        ];

        const container = document.getElementById('alertsList');
        if (!container) return;

        container.innerHTML = alerts.map(alert => `
            <div class="alert-item ${alert.level}">
                <div class="alert-header">
                    <div class="alert-title">${alert.title}</div>
                    <div class="alert-time">${alert.timestamp}</div>
                </div>
                <div class="alert-description">${alert.description}</div>
                <div class="alert-actions">
                    <button class="btn btn-sm">View Details</button>
                    <button class="btn btn-sm">Dismiss</button>
                </div>
            </div>
        `).join('');
    }

    startRealTimeUpdates() {
        // Update header stats every 5 seconds
        setInterval(() => {
            this.updateHeaderStats();
        }, 5000);

        // Update transaction list every 10 seconds
        setInterval(() => {
            if (this.isAutoRefresh) {
                this.loadDashboard();
            }
        }, 10000);
    }

    updateHeaderStats() {
        const activeTransactions = Math.floor(Math.random() * 100) + 50;
        const fraudDetected = Math.floor(Math.random() * 10);
        
        document.getElementById('activeTransactions').textContent = activeTransactions;
        document.getElementById('fraudDetected').textContent = fraudDetected;
    }

    toggleAutoRefresh() {
        this.isAutoRefresh = !this.isAutoRefresh;
        const icon = document.getElementById('autoRefreshIcon');
        const text = document.getElementById('autoRefreshText');
        
        if (this.isAutoRefresh) {
            icon.className = 'fas fa-pause';
            text.textContent = 'Auto Refresh';
            this.showToast('Auto refresh enabled', 'info');
        } else {
            icon.className = 'fas fa-play';
            text.textContent = 'Auto Refresh';
            this.showToast('Auto refresh disabled', 'info');
        }
    }

    refreshDashboard() {
        this.loadDashboard();
        this.showToast('Dashboard refreshed', 'success');
    }

    navigateToTransactionSection() {
        this.showSection('transactions');
    }

    loadTransactions() {
        // Load transaction-specific data
        console.log('Loading transactions...');
    }

    loadAnalytics() {
        // Load analytics-specific data
        console.log('Loading analytics...');
    }

    loadModels() {
        // Load model-specific data
        console.log('Loading models...');
    }

    showLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.add('active');
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.remove('active');
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; color: inherit; cursor: pointer; font-size: 1.2rem;">&times;</button>
            </div>
        `;

        container.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
    }
}

// Utility functions
function refreshDashboard() {
    if (window.fraudApp) {
        window.fraudApp.refreshDashboard();
    }
}

function toggleAutoRefresh() {
    if (window.fraudApp) {
        window.fraudApp.toggleAutoRefresh();
    }
}

function navigateToTransactions() {
    if (window.fraudApp) {
        window.fraudApp.showSection('transactions');
    }
}

function submitTransaction() {
    if (window.fraudApp) {
        window.fraudApp.submitTransaction();
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.fraudApp = new FraudDetectionApp();
    
    // Add some initial animations
    document.querySelectorAll('.metric-card').forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });
});

// Handle window resize for responsive chart
window.addEventListener('resize', () => {
    if (window.fraudApp && window.fraudApp.chart) {
        // Redraw chart on resize
        setTimeout(() => {
            window.fraudApp.initializeChart();
        }, 100);
    }
});

// Initialize the app
const app = new FraudDetectionApp();

// Make app available globally for HTML onclick handlers if needed
window.app = app;

// For module exports compatibility
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FraudDetectionApp;
}
