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
        this.loadTransactions();
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

    async submitTransaction() {
        const bankBookName = document.getElementById('bankBookName').value;
        const transactionId = document.getElementById('transactionId').value;
        const amount = document.getElementById('amount').value;

        // Strict validation
        const validation = this.validateTransactionInputs(bankBookName, transactionId, amount);
        if (!validation.valid) {
            this.showToast(validation.error, 'error');
            return;
        }

        try {
            this.showLoading();
            
            const transactionData = {
                bank_book_name: bankBookName.trim().toUpperCase(),
                transaction_id: transactionId.trim().toUpperCase(),
                amount: parseFloat(amount),
                timestamp: new Date().toISOString()
            };

            console.log('Transaction data:', transactionData);
            
            const result = await this.verifyWithSyntheticData(transactionData);
            console.log('Synthetic verification result:', result);
            this.displayAnalysisResult(result);
            
        } catch (error) {
            this.showToast('Error verifying against synthetic data', 'error');
            console.error('Synthetic verification error:', error);
        } finally {
            this.hideLoading();
        }
    }

    async verifyWithSyntheticData(transactionData) {
        const startTime = performance.now();
        const params = new URLSearchParams({
            transaction_id: transactionData.transaction_id,
            bank_book_name: transactionData.bank_book_name,
            amount: String(transactionData.amount)
        });

        const response = await fetch(`${this.apiBaseUrl}/api/synthetic-verify?${params.toString()}`);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Synthetic verify failed: ${response.status} ${errorText}`);
        }

        const data = await response.json();
        const processingTime = Math.max(1, Math.round(performance.now() - startTime));

        const matched = Boolean(data.matched);
        const outcome = matched ? 'success' : 'failed';
        const message = matched
            ? 'Transaction Successful: Details Verified against synthetic dataset.'
            : 'Transaction Failed: No matching record found in synthetic dataset.';

        return {
            transaction_id: transactionData.transaction_id,
            bank_book_name: transactionData.bank_book_name,
            amount: transactionData.amount,
            outcome: outcome,
            message: message,
            model: 'Synthetic Dataset Verification',
            confidence: matched ? '100%' : '0%',
            processing_time: processingTime,
            explanation: matched
                ? 'Matched against generated synthetic dataset.'
                : 'No exact match in synthetic dataset.'
        };
    }

    validateTransactionInputs(bankBookName, transactionId, amount) {
        const knownBanks = [
            'SBI SAVINGS',
            'HDFC SAVINGS',
            'ICICI CURRENT',
            'AXIS SAVINGS',
            'KOTAK SAVINGS'
        ];

        // Check if fields are empty
        if (!bankBookName || !bankBookName.trim()) {
            return { valid: false, error: 'Bank Book Name is required. Enter name as in bank book.' };
        }
        if (!transactionId || !transactionId.trim()) {
            return { valid: false, error: 'Transaction ID is required. Enter transaction ID.' };
        }
        if (!amount || amount.trim() === '') {
            return { valid: false, error: 'Amount is required. Enter amount in ₹.' };
        }

        // Validate Bank Book Name (case-insensitive comparison)
        const normalizedBank = bankBookName.trim().toUpperCase();
        if (!knownBanks.includes(normalizedBank)) {
            return { 
                valid: false, 
                error: `Invalid Bank Book Name. Valid banks: ${knownBanks.join(', ')}`
            };
        }

        // Validate Transaction ID format (TXN + 6+ digits)
        const txnIdUpper = transactionId.trim().toUpperCase();
        if (!/^TXN\d{6,}$/.test(txnIdUpper)) {
            return { 
                valid: false, 
                error: 'Invalid Transaction ID format. Format should be: TXN followed by 6+ digits (e.g., TXN1234567).' };
        }

        // Validate Amount (must be number, > 0, and <= 100,000)
        const amountNum = parseFloat(amount);
        if (isNaN(amountNum)) {
            return { valid: false, error: 'Amount must be a valid number.' };
        }
        if (amountNum <= 0) {
            return { valid: false, error: 'Amount must be greater than ₹0.' };
        }
        if (amountNum > 100000) {
            return { valid: false, error: 'Amount cannot exceed ₹100,000.' };
        }

        return { valid: true };
    }

    async analyzeTransaction(transactionData) {
        try {
            // Format data according to backend API requirements
            const apiData = {
                transaction_id: transactionData.transaction_id,
                upi_id: "user@upi",
                merchant_id: transactionData.bank_book_name,
                amount: parseFloat(transactionData.amount),
                hour: new Date().getHours(),
                device_risk_score: 0.3,
                location_risk_score: 0.2,
                user_behavior_score: 0.5
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
                return this.normalizeApiResult(result, transactionData);
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
    
    normalizeApiResult(result, transactionData) {
        const riskScore = result.risk_score || 0;
        const outcome = riskScore >= 0.5 ? 'failed' : 'success';
        const message = outcome === 'success'
            ? 'Transaction Successful: Details Verified and Processed.'
            : 'Transaction Failed: Incorrect Details Entered. Please Verify and Try Again.';

        return {
            transaction_id: transactionData.transaction_id,
            bank_book_name: transactionData.bank_book_name,
            amount: transactionData.amount,
            outcome: outcome,
            message: message,
            model: 'Random Forest',
            confidence: result.model_confidence || '90%',
            processing_time: result.processing_time || 15,
            explanation: result.explanation || ''
        };
    }

    generateMockAnalysis(transactionData) {
        return new Promise((resolve) => {
            setTimeout(() => {
                const knownBanks = [
                    'SBI SAVINGS',
                    'HDFC SAVINGS',
                    'ICICI CURRENT',
                    'AXIS SAVINGS',
                    'KOTAK SAVINGS'
                ];
                const bankName = (transactionData.bank_book_name || '').trim().toUpperCase();
                const txnId = (transactionData.transaction_id || '').trim().toUpperCase();
                const amount = Number(transactionData.amount || 0);
                const nameOk = knownBanks.includes(bankName);
                const idOk = /^TXN\d{6,}$/.test(txnId);
                const amountOk = amount > 0 && amount <= 100000;
                const outcome = nameOk && idOk && amountOk ? 'success' : 'failed';
                const message = outcome === 'success'
                    ? 'Transaction Successful: Details Verified and Processed.'
                    : 'Transaction Failed: Incorrect Details Entered. Please Verify and Try Again.';

                resolve({
                    transaction_id: transactionData.transaction_id,
                    bank_book_name: transactionData.bank_book_name,
                    amount: amount,
                    outcome: outcome,
                    message: message,
                    model: 'Random Forest',
                    confidence: (85 + Math.random() * 10).toFixed(1) + '%',
                    explanation: 'Using mock data (backend API unavailable)',
                    processing_time: 15
                });
            }, 1200);
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
        
        const outcome = result.outcome || 'failed';
        const outcomeClass = outcome === 'success' ? 'low' : 'high';
        const badgeLabel = outcome === 'success' ? 'VERIFIED' : 'FAILED';
        
        container.innerHTML = `
            <div class="result-header ${outcomeClass}">
                <h3>Transaction Verification Result</h3>
                <div class="risk-badge ${outcomeClass}">${badgeLabel}</div>
            </div>
            <div class="result-content">
                <div class="result-summary">
                    <div class="result-item">
                        <span class="label">Outcome:</span>
                        <span class="value">${result.message}</span>
                    </div>
                    <div class="result-item">
                        <span class="label">Model:</span>
                        <span class="value">${result.model || 'Random Forest'}</span>
                    </div>
                    <div class="result-item">
                        <span class="label">Confidence:</span>
                        <span class="value">${result.confidence || '90%'}</span>
                    </div>
                    <div class="result-item">
                        <span class="label">Processing Time:</span>
                        <span class="value">${result.processing_time || 15}ms</span>
                    </div>
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
        const bankBookName = document.getElementById('bankBookName').value;
        const transactionId = document.getElementById('transactionId').value;
        const amount = document.getElementById('amount').value;

        if (!bankBookName || !transactionId || !amount) {
            this.showToast('Please fill in all fields', 'warning');
            return;
        }

        this.submitTransaction();
    }

    loadAlerts() {
        const alerts = [
            {
                id: 1,
                title: 'Verification Failed',
                description: 'Transaction TXN1234567 failed due to incorrect details. Please verify and retry.',
                level: 'high',
                timestamp: '2 minutes ago'
            },
            {
                id: 2,
                title: 'Random Forest Model Ready',
                description: 'Model loaded and ready for transaction verification.',
                level: 'medium',
                timestamp: '15 minutes ago'
            },
            {
                id: 3,
                title: 'Dataset Pre-processing Completed',
                description: 'Feature encoding and normalization completed for demo dataset.',
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
        // Update transaction list every 10 seconds
        setInterval(() => {
            if (this.isAutoRefresh) {
                this.loadTransactions();
            }
        }, 10000);
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
        this.loadTransactions();
        this.showToast('Transactions refreshed', 'success');
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

// Initialize the app
const app = new FraudDetectionApp();

// Make app available globally for HTML onclick handlers if needed
window.app = app;

// For module exports compatibility
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FraudDetectionApp;
}
