"""
UPI Fraud Detection - GNN Training Script
Training script for the collusion detection GNN model
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from collusion_detector import CollusionDetectionPipeline, create_default_config
from graph_builder import create_sample_graph_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_transaction_data(data_path: str) -> pd.DataFrame:
    """Load transaction data from CSV or generate sample data"""
    if Path(data_path).exists():
        logger.info(f"Loading transaction data from {data_path}")
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        logger.info("Data file not found, generating sample data")
        return create_sample_graph_data()

def load_fraud_labels(labels_path: str, transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Load fraud labels or generate sample labels"""
    if Path(labels_path).exists():
        logger.info(f"Loading fraud labels from {labels_path}")
        return pd.read_csv(labels_path)
    else:
        logger.info("Labels file not found, generating sample labels")
        # Generate sample fraud labels (10% fraud rate)
        sample_size = min(1000, len(transactions_df))
        sample_txns = transactions_df.sample(sample_size)
        
        fraud_labels = pd.DataFrame({
            'transaction_id': sample_txns['transaction_id'],
            'upi_id': sample_txns['upi_id'],
            'merchant_id': sample_txns['merchant_id'],
            'is_fraud': np.random.choice([True, False], sample_size, p=[0.1, 0.9])
        })
        
        return fraud_labels

def main():
    parser = argparse.ArgumentParser(description="Train UPI Fraud Detection GNN")
    parser.add_argument("--data", default="transactions.csv", help="Transaction data CSV file")
    parser.add_argument("--labels", default="fraud_labels.csv", help="Fraud labels CSV file")
    parser.add_argument("--config", default=None, help="Configuration JSON file")
    parser.add_argument("--output-dir", default="./models/gnn", help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split ratio")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    config['model']['num_epochs'] = args.epochs
    config['model']['batch_size'] = args.batch_size
    config['model']['hidden_channels'] = args.hidden_dim
    config['model']['num_layers'] = args.num_layers
    config['model']['dropout'] = args.dropout
    config['model_dir'] = args.output_dir
    
    logger.info(f"Training configuration: {json.dumps(config, indent=2)}")
    
    # Load data
    transactions_df = load_transaction_data(args.data)
    fraud_labels_df = load_fraud_labels(args.labels, transactions_df)
    
    logger.info(f"Loaded {len(transactions_df)} transactions and {len(fraud_labels_df)} labels")
    logger.info(f"Fraud rate: {fraud_labels_df['is_fraud'].mean():.2%}")
    
    # Initialize pipeline
    pipeline = CollusionDetectionPipeline(config)
    
    # Prepare training data
    training_graphs = pipeline.prepare_training_data(transactions_df, fraud_labels_df)
    
    if len(training_graphs) == 0:
        logger.error("No training graphs created. Check data quality and time windows.")
        return
    
    logger.info(f"Created {len(training_graphs)} training graphs")
    
    # Train model
    logger.info("Starting model training...")
    training_history = pipeline.train_model(training_graphs, args.validation_split)
    
    # Save pipeline configuration
    pipeline.save_pipeline_config()
    
    # Test detection on sample data
    logger.info("Testing collusion detection...")
    recent_transactions = transactions_df.tail(1000)  # Use recent transactions for testing
    detection_results = pipeline.detect_collusion(recent_transactions, threshold=0.7)
    
    # Generate and save report
    report_path = Path(args.output_dir) / f"collusion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report = pipeline.generate_report(detection_results, str(report_path))
    
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Training Graphs: {len(training_graphs)}")
    print(f"Model Parameters: {sum(p.numel() for p in pipeline.model.parameters()):,}")
    print(f"Suspicious Users Detected: {len(detection_results['suspicious_users'])}")
    print(f"Collusion Rings Detected: {len(detection_results['collusion_rings'])}")
    print("="*60)

if __name__ == "__main__":
    main()
