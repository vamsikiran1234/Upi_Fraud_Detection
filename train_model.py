#!/usr/bin/env python3
"""
Machine Learning Model Training for UPI Fraud Detection
Implements the complete ML pipeline: Dataset Creation, Preprocessing, Training, Classification
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class UPIFraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_columns = None
        self.model_path = 'models/random_forest_model.pkl'
        self.scaler_path = 'models/scaler.pkl'
        self.encoder_path = 'models/encoders.pkl'
        self.dataset_path = 'models/synthetic_transactions.csv'
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Known bank book names
        self.known_banks = [
            "SBI SAVINGS",
            "HDFC SAVINGS",
            "ICICI CURRENT",
            "AXIS SAVINGS",
            "KOTAK SAVINGS"
        ]

    def create_dataset(self, num_samples=5000):
        """
        MODULE 1: Create Dataset
        Generate synthetic dataset with transaction details
        """
        print("=" * 80)
        print("MODULE 1: CREATING DATASET")
        print("=" * 80)
        
        np.random.seed(42)
        data = []
        
        print(f"Generating {num_samples} synthetic transactions...")
        
        for i in range(num_samples):
            # Bank book name (valid or invalid)
            if np.random.random() < 0.8:  # 80% valid banks
                bank_name = np.random.choice(self.known_banks)
            else:  # 20% invalid/misspelled banks
                invalid_banks = ["SBI SAVING", "HDFC SAVI", "ICICI CURR", "UNKNOWN", "BOB SAVINGS"]
                bank_name = np.random.choice(invalid_banks)
            
            # Transaction ID (valid or invalid format)
            if np.random.random() < 0.85:  # 85% valid format
                txn_id = f"TXN{np.random.randint(100000, 9999999)}"
            else:  # 15% invalid format
                invalid_formats = [
                    f"TXN{np.random.randint(100, 999)}",  # Too few digits
                    f"{np.random.randint(1000000, 9999999)}",  # Missing TXN prefix
                    f"ABC{np.random.randint(100000, 9999999)}",  # Wrong prefix
                    f"TXN-{np.random.randint(100000, 9999999)}"  # Invalid character
                ]
                txn_id = np.random.choice(invalid_formats)
            
            # Amount (valid or invalid)
            if np.random.random() < 0.9:  # 90% valid amounts
                amount = np.random.randint(1, 100001)  # ₹1 to ₹100,000
            else:  # 10% invalid amounts
                invalid_amounts = [0, -100, 150000, 500000, np.nan]
                amount = np.random.choice(invalid_amounts)
            
            # Additional features for model
            hour = np.random.randint(0, 24)
            day_of_week = np.random.randint(0, 7)
            is_weekend = 1 if day_of_week >= 5 else 0
            device_risk_score = np.random.uniform(0, 1)
            location_risk_score = np.random.uniform(0, 1)
            user_behavior_score = np.random.uniform(0, 1)
            
            # Determine outcome based on validation rules
            bank_valid = bank_name in self.known_banks
            txn_valid = self._is_valid_transaction_id(txn_id)
            amount_valid = isinstance(amount, (int, float)) and 0 < amount <= 100000
            
            # Outcome: 1 = Success/Legitimate, 0 = Failed/Fraudulent
            outcome = 1 if (bank_valid and txn_valid and amount_valid) else 0
            
            data.append({
                'bank_name': bank_name,
                'txn_id': txn_id,
                'amount': amount if not pd.isna(amount) else 0,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'device_risk_score': device_risk_score,
                'location_risk_score': location_risk_score,
                'user_behavior_score': user_behavior_score,
                'outcome': outcome
            })
        
        df = pd.DataFrame(data)
        print(f"✓ Dataset created with {len(df)} transactions")
        print(f"  - Legitimate transactions: {(df['outcome'] == 1).sum()}")
        print(f"  - Fraudulent transactions: {(df['outcome'] == 0).sum()}")
        print(f"  - Dataset shape: {df.shape}")
        
        return df

    def _is_valid_transaction_id(self, txn_id):
        """Check if transaction ID is valid format"""
        if not isinstance(txn_id, str):
            return False
        import re
        return bool(re.match(r'^TXN\d{6,}$', txn_id.upper()))

    def preprocess_data(self, df):
        """
        MODULE 2: Pre-processing
        Clean and prepare the dataset for model training
        """
        print("\n" + "=" * 80)
        print("MODULE 2: PRE-PROCESSING")
        print("=" * 80)
        
        df_processed = df.copy()
        
        # Handle missing values
        print("Handling missing values...")
        df_processed['amount'].fillna(0, inplace=True)
        print(f"✓ Missing values handled")
        
        # Encode categorical features (bank name)
        print("Encoding categorical features...")
        if 'bank_name' not in self.encoders:
            self.encoders['bank_name'] = LabelEncoder()
            df_processed['bank_name_encoded'] = self.encoders['bank_name'].fit_transform(
                df_processed['bank_name']
            )
        else:
            df_processed['bank_name_encoded'] = self.encoders['bank_name'].transform(
                df_processed['bank_name']
            )
        
        print(f"✓ Bank names encoded: {list(self.encoders['bank_name'].classes_)}")
        
        # Normalize/Standardize numerical features
        print("Standardizing numerical features...")
        numerical_features = ['amount', 'hour', 'device_risk_score', 
                             'location_risk_score', 'user_behavior_score']
        
        df_processed[numerical_features] = self.scaler.fit_transform(
            df_processed[numerical_features]
        )
        print(f"✓ Numerical features standardized: {numerical_features}")
        
        # Create feature matrix and target vector
        feature_cols = ['bank_name_encoded', 'amount', 'hour', 'day_of_week', 
                       'is_weekend', 'device_risk_score', 'location_risk_score', 
                       'user_behavior_score']
        self.feature_columns = feature_cols
        
        X = df_processed[feature_cols]
        y = df_processed['outcome']
        
        print(f"✓ Feature matrix shape: {X.shape}")
        print(f"✓ Target vector shape: {y.shape}")
        
        return X, y, df_processed

    def split_data(self, X, y, test_size=0.3):
        """Split data into training and testing sets (70-30 split)"""
        print(f"\nSplitting data: {(1-test_size)*100:.0f}% training, {test_size*100:.0f}% testing...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"✓ Training set: {X_train.shape[0]} samples")
        print(f"✓ Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """
        MODULE 3: Training
        Train the Random Forest classifier
        """
        print("\n" + "=" * 80)
        print("MODULE 3: TRAINING")
        print("=" * 80)
        
        print("Initializing Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print("Training model on dataset...")
        self.model.fit(X_train, y_train)
        print("✓ Model training completed")
        
        # Feature importance
        print("\nTop 5 Important Features:")
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {self.feature_columns[idx]}: {importances[idx]:.4f}")
        
        return self.model

    def evaluate_model(self, X_test, y_test):
        """
        MODULE 4: Classification & Evaluation
        Classify and evaluate model performance
        """
        print("\n" + "=" * 80)
        print("MODULE 4: CLASSIFICATION & EVALUATION")
        print("=" * 80)
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print("\nModel Performance Metrics:")
        print(f"  ✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  ✓ Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  ✓ Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  ✓ F1-Score:  {f1:.4f}")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  True Negatives:  {cm[0, 0]}")
        print(f"  False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}")
        print(f"  True Positives:  {cm[1, 1]}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
              target_names=['Fraudulent', 'Legitimate'],
              zero_division=0))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }

    def save_model(self):
        """Save trained model, scaler, and encoders to disk"""
        print("\n" + "=" * 80)
        print("SAVING MODEL")
        print("=" * 80)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to {self.model_path}")
        
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved to {self.scaler_path}")
        
        with open(self.encoder_path, 'wb') as f:
            pickle.dump(self.encoders, f)
        print(f"✓ Encoders saved to {self.encoder_path}")
        
        # Save feature columns for later use
        with open('models/feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"✓ Feature columns saved")

    def save_dataset(self, df, path=None):
        """Save the generated synthetic dataset to CSV"""
        dataset_path = path or self.dataset_path
        df.to_csv(dataset_path, index=False)
        print(f"✓ Synthetic dataset saved to {dataset_path}")

    def load_model(self):
        """Load trained model, scaler, and encoders from disk"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded from {self.model_path}")
            
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Scaler loaded from {self.scaler_path}")
            
            with open(self.encoder_path, 'rb') as f:
                self.encoders = pickle.load(f)
            print(f"✓ Encoders loaded from {self.encoder_path}")
            
            with open('models/feature_columns.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            print(f"✓ Feature columns loaded")
            
            return True
        return False

    def predict(self, bank_name, txn_id, amount, hour=12, day_of_week=2, 
                device_risk=0.3, location_risk=0.2, behavior_risk=0.5):
        """
        Make predictions on new transactions
        Returns: prediction (0=Fraudulent, 1=Legitimate) and confidence
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # Prepare input data
        is_weekend = 1 if day_of_week >= 5 else 0
        
        try:
            # Encode bank name
            bank_encoded = self.encoders['bank_name'].transform([bank_name])[0]
        except:
            # If bank name not in encoder, use unknown encoding
            bank_encoded = -1
        
        # Create feature array
        features = np.array([[
            bank_encoded,
            amount,
            hour,
            day_of_week,
            is_weekend,
            device_risk,
            location_risk,
            behavior_risk
        ]])
        
        # Apply feature scaling
        features_scaled = self.scaler.transform(features)
        
        # Get prediction and probability
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        confidence = probability[prediction]
        
        return prediction, confidence, probability

    def run_complete_pipeline(self, num_samples=5000):
        """Run the complete ML pipeline"""
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█  UPI FRAUD DETECTION - MACHINE LEARNING PIPELINE".center(80) + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)
        
        try:
            # Module 1: Create Dataset
            df = self.create_dataset(num_samples)
            self.save_dataset(df)
            
            # Module 2: Pre-processing
            X, y, df_processed = self.preprocess_data(df)
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # Module 3: Training
            self.train_model(X_train, y_train)
            
            # Module 4: Classification & Evaluation
            metrics = self.evaluate_model(X_test, y_test)
            
            # Save model
            self.save_model()
            
            print("\n" + "█" * 80)
            print("█  PIPELINE COMPLETED SUCCESSFULLY!".center(80) + "█")
            print("█" * 80)
            
            return True, metrics
            
        except Exception as e:
            print(f"\n❌ Error in pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None


if __name__ == "__main__":
    # Initialize and run the complete pipeline
    model_trainer = UPIFraudDetectionModel()
    success, metrics = model_trainer.run_complete_pipeline(num_samples=5000)
    
    if success:
        print("\n✅ Model training completed successfully!")
        print("Run the backend API with: python simple_backend_api.py")
    else:
        print("\n❌ Model training failed!")
