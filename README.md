# UPI Fraud Detection System

A comprehensive machine learning-based system for detecting fraudulent transactions in UPI (Unified Payments Interface) payment networks. This project combines advanced ML algorithms, real-time monitoring, and interactive visualization to provide a robust fraud detection solution.

## Features

- **Real-time Transaction Monitoring**: Analyze transactions as they occur to detect suspicious patterns
- **Multi-model Ensemble Detection**: Combines multiple ML models for higher accuracy
- **Interactive Dashboard**: Visualize fraud patterns and system performance metrics
- **API Integration**: Easy integration with existing payment systems
- **Graph Neural Networks**: Detect complex fraud patterns and collusion networks
- **Explainable AI**: Understand why transactions are flagged as fraudulent
- **Scalable Architecture**: Designed to handle high transaction volumes

## Technologies Used

- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch, Graph Neural Networks
- **API Framework**: FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Interactive charts and graphs
- **Deployment**: Docker, Kubernetes support
- **Monitoring**: Prometheus, Grafana integration

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/upi-fraud-detection.git
cd upi-fraud-detection
```

2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements-fixed.txt
```

### Running the System

#### Basic Mode

Run the basic fraud detection system:
```bash
python quick_start.py
```

#### Frontend Dashboard

Start the frontend dashboard:
```bash
python frontend/server.py
```

#### Advanced Mode

For advanced features and models:
```bash
python advanced_quick_start.py
```

## System Architecture

The system consists of several components:

1. **Data Ingestion Layer**: Processes incoming transaction data
2. **Feature Engineering**: Extracts and transforms relevant features
3. **Model Ensemble**: Multiple models working together for detection
4. **Decision Engine**: Makes the final fraud determination
5. **API Layer**: Exposes functionality to external systems
6. **Dashboard**: Visualizes results and system performance

## Future Enhancements

- Federated learning for privacy-preserving fraud detection
- Blockchain integration for immutable audit trails
- Advanced anomaly detection with reinforcement learning
- Mobile app for alerts and notifications
- Integration with additional payment platforms

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Structure

The project directory is organised as follows:

```
├── advanced_quick_start.py     # Advanced system startup
├── quick_start.py              # Basic system startup
├── frontend/                   # Frontend web interface
│   ├── server.py               # Frontend server
│   ├── index.html              # Main HTML page
│   ├── script.js               # Frontend JavaScript
│   └── styles.css              # CSS styles
├── dashboard/                  # React dashboard
│   ├── src/                    # React source code
│   └── public/                 # Public assets
├── models/                     # ML model files
│   ├── gnn/                    # Graph Neural Network models
│   ├── tabular/                # Tabular data models
│   └── sequence/               # Sequence models
├── serving/                    # Model serving components
│   └── models/                 # Model implementations
├── data/                       # Data storage
├── config/                     # Configuration files
├── docs/                       # Documentation
├── tests/                      # Test files
└── infra/                      # Infrastructure code
    └── k8s/                    # Kubernetes configurations
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

© 2025 S K Ismail

---

Give this repository a ⭐ if you like it.
