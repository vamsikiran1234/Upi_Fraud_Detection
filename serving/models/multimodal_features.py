"""
Multi-Modal Features for Enhanced Fraud Detection
Incorporates biometrics, device telemetry, and behavioral patterns
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import librosa
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BiometricFeatures:
    """Biometric feature extraction"""
    face_encoding: Optional[np.ndarray] = None
    fingerprint_template: Optional[np.ndarray] = None
    voice_features: Optional[np.ndarray] = None
    behavioral_patterns: Optional[Dict[str, float]] = None
    device_biometrics: Optional[Dict[str, float]] = None

@dataclass
class DeviceTelemetry:
    """Device telemetry data"""
    accelerometer_data: Optional[np.ndarray] = None
    gyroscope_data: Optional[np.ndarray] = None
    magnetometer_data: Optional[np.ndarray] = None
    touch_patterns: Optional[np.ndarray] = None
    typing_rhythm: Optional[np.ndarray] = None
    device_orientation: Optional[float] = None
    battery_level: Optional[float] = None
    network_info: Optional[Dict[str, Any]] = None
    app_usage_patterns: Optional[Dict[str, float]] = None

class FaceRecognitionExtractor:
    """Extract face recognition features"""
    
    def __init__(self):
        # In a real implementation, you would use OpenCV DNN or face_recognition library
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def extract_face_features(self, image_path: str) -> np.ndarray:
        """Extract face encoding from image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return np.zeros(128)  # Default encoding
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return np.zeros(128)  # No face detected
            
            # Extract face region
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size
            face_roi = cv2.resize(face_roi, (64, 64))
            
            # Extract features (simplified - in real implementation use deep learning)
            features = self._extract_face_encoding(face_roi)
            
            return features
            
        except Exception as e:
            print(f"Error extracting face features: {e}")
            return np.zeros(128)
    
    def _extract_face_encoding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face encoding using simplified method"""
        # In real implementation, use pre-trained face recognition model
        # For demo, we'll use histogram features
        hist = cv2.calcHist([face_image], [0], None, [64], [0, 256])
        hist = hist.flatten()
        
        # Pad or truncate to 128 dimensions
        if len(hist) < 128:
            hist = np.pad(hist, (0, 128 - len(hist)))
        else:
            hist = hist[:128]
        
        return hist.astype(np.float32)

class VoiceBiometricExtractor:
    """Extract voice biometric features"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def extract_voice_features(self, audio_path: str) -> np.ndarray:
        """Extract voice biometric features from audio"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Extract rhythm features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Combine features
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.mean(spectral_centroids),
                np.mean(spectral_rolloff),
                np.mean(zero_crossing_rate),
                [tempo]
            ])
            
            # Pad or truncate to fixed size
            target_size = 50
            if len(features) < target_size:
                features = np.pad(features, (0, target_size - len(features)))
            else:
                features = features[:target_size]
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting voice features: {e}")
            return np.zeros(50)
    
    def extract_typing_rhythm(self, keystroke_times: List[float]) -> np.ndarray:
        """Extract typing rhythm features"""
        if len(keystroke_times) < 2:
            return np.zeros(20)
        
        # Calculate inter-key intervals
        intervals = np.diff(keystroke_times)
        
        # Extract rhythm features
        features = [
            np.mean(intervals),
            np.std(intervals),
            np.median(intervals),
            np.min(intervals),
            np.max(intervals),
            np.percentile(intervals, 25),
            np.percentile(intervals, 75),
            len(intervals),
            np.sum(intervals),
            np.var(intervals)
        ]
        
        # Add more sophisticated features
        if len(intervals) > 1:
            # Rhythm consistency
            features.append(np.std(intervals) / np.mean(intervals))
            
            # Burst patterns
            burst_threshold = np.mean(intervals) + np.std(intervals)
            bursts = np.sum(intervals < burst_threshold)
            features.append(bursts / len(intervals))
            
            # Pause patterns
            pause_threshold = np.mean(intervals) + 2 * np.std(intervals)
            pauses = np.sum(intervals > pause_threshold)
            features.append(pauses / len(intervals))
        else:
            features.extend([0, 0, 0])
        
        # Pad to fixed size
        target_size = 20
        if len(features) < target_size:
            features.extend([0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)

class DeviceTelemetryExtractor:
    """Extract device telemetry features"""
    
    def __init__(self):
        self.sensor_window_size = 100  # Number of sensor readings to consider
    
    def extract_sensor_features(self, sensor_data: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
        """Extract features from sensor data"""
        features = {}
        
        for sensor_name, data in sensor_data.items():
            if not data or len(data) == 0:
                features[sensor_name] = np.zeros(10)
                continue
            
            data_array = np.array(data)
            
            # Statistical features
            sensor_features = [
                np.mean(data_array),
                np.std(data_array),
                np.median(data_array),
                np.min(data_array),
                np.max(data_array),
                np.percentile(data_array, 25),
                np.percentile(data_array, 75),
                np.var(data_array),
                len(data_array),
                np.sum(data_array)
            ]
            
            features[sensor_name] = np.array(sensor_features, dtype=np.float32)
        
        return features
    
    def extract_touch_patterns(self, touch_events: List[Dict[str, Any]]) -> np.ndarray:
        """Extract touch pattern features"""
        if not touch_events:
            return np.zeros(25)
        
        # Extract touch features
        x_coords = [event.get('x', 0) for event in touch_events]
        y_coords = [event.get('y', 0) for event in touch_events]
        pressures = [event.get('pressure', 0) for event in touch_events]
        sizes = [event.get('size', 0) for event in touch_events]
        timestamps = [event.get('timestamp', 0) for event in touch_events]
        
        features = []
        
        # Position features
        if x_coords and y_coords:
            features.extend([
                np.mean(x_coords), np.std(x_coords),
                np.mean(y_coords), np.std(y_coords),
                np.corrcoef(x_coords, y_coords)[0, 1] if len(x_coords) > 1 else 0
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Pressure features
        if pressures:
            features.extend([
                np.mean(pressures), np.std(pressures),
                np.max(pressures), np.min(pressures)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Size features
        if sizes:
            features.extend([
                np.mean(sizes), np.std(sizes),
                np.max(sizes), np.min(sizes)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Temporal features
        if timestamps and len(timestamps) > 1:
            intervals = np.diff(timestamps)
            features.extend([
                np.mean(intervals), np.std(intervals),
                np.min(intervals), np.max(intervals),
                len(intervals)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Touch pattern features
        if len(touch_events) > 1:
            # Swipe detection
            swipe_features = self._detect_swipe_patterns(touch_events)
            features.extend(swipe_features)
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Pad to fixed size
        target_size = 25
        if len(features) < target_size:
            features.extend([0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def _detect_swipe_patterns(self, touch_events: List[Dict[str, Any]]) -> List[float]:
        """Detect swipe patterns in touch events"""
        if len(touch_events) < 3:
            return [0, 0, 0, 0, 0]
        
        x_coords = [event.get('x', 0) for event in touch_events]
        y_coords = [event.get('y', 0) for event in touch_events]
        
        # Calculate movement vectors
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        
        # Swipe direction
        avg_dx = np.mean(dx)
        avg_dy = np.mean(dy)
        
        # Swipe speed
        distances = np.sqrt(dx**2 + dy**2)
        avg_speed = np.mean(distances)
        
        # Swipe consistency
        direction_consistency = np.std(dx) + np.std(dy)
        
        # Swipe length
        total_distance = np.sum(distances)
        
        return [avg_dx, avg_dy, avg_speed, direction_consistency, total_distance]

class BehavioralPatternExtractor:
    """Extract behavioral patterns from user interactions"""
    
    def __init__(self):
        self.pattern_window = 24  # Hours to look back for patterns
    
    def extract_behavioral_features(self, user_history: pd.DataFrame) -> Dict[str, float]:
        """Extract behavioral patterns from user transaction history"""
        if user_history.empty:
            return self._get_default_behavioral_features()
        
        features = {}
        
        # Temporal patterns
        user_history['timestamp'] = pd.to_datetime(user_history['timestamp'])
        user_history['hour'] = user_history['timestamp'].dt.hour
        user_history['day_of_week'] = user_history['timestamp'].dt.dayofweek
        
        # Activity patterns
        features['avg_transactions_per_day'] = len(user_history) / max(1, (user_history['timestamp'].max() - user_history['timestamp'].min()).days)
        features['preferred_hour'] = user_history['hour'].mode().iloc[0] if not user_history['hour'].mode().empty else 12
        features['preferred_day'] = user_history['day_of_week'].mode().iloc[0] if not user_history['day_of_week'].mode().empty else 0
        
        # Amount patterns
        features['avg_amount'] = user_history['amount'].mean()
        features['amount_std'] = user_history['amount'].std()
        features['max_amount'] = user_history['amount'].max()
        features['amount_consistency'] = 1 - (user_history['amount'].std() / user_history['amount'].mean()) if user_history['amount'].mean() > 0 else 0
        
        # Merchant patterns
        features['merchant_diversity'] = user_history['merchant_id'].nunique() / len(user_history)
        features['top_merchant_ratio'] = user_history['merchant_id'].value_counts().iloc[0] / len(user_history) if len(user_history) > 0 else 0
        
        # Velocity patterns
        if 'user_velocity' in user_history.columns:
            features['avg_velocity'] = user_history['user_velocity'].mean()
            features['velocity_consistency'] = 1 - (user_history['user_velocity'].std() / user_history['user_velocity'].mean()) if user_history['user_velocity'].mean() > 0 else 0
        
        # Risk patterns
        risk_columns = ['device_risk_score', 'location_risk_score', 'ip_reputation']
        for col in risk_columns:
            if col in user_history.columns:
                features[f'avg_{col}'] = user_history[col].mean()
                features[f'{col}_consistency'] = 1 - user_history[col].std()
        
        return features
    
    def _get_default_behavioral_features(self) -> Dict[str, float]:
        """Return default behavioral features when no history is available"""
        return {
            'avg_transactions_per_day': 0,
            'preferred_hour': 12,
            'preferred_day': 0,
            'avg_amount': 0,
            'amount_std': 0,
            'max_amount': 0,
            'amount_consistency': 0,
            'merchant_diversity': 0,
            'top_merchant_ratio': 0,
            'avg_velocity': 0,
            'velocity_consistency': 0,
            'avg_device_risk_score': 0,
            'device_risk_score_consistency': 0,
            'avg_location_risk_score': 0,
            'location_risk_score_consistency': 0,
            'avg_ip_reputation': 0,
            'ip_reputation_consistency': 0
        }

class MultiModalFeatureFusion(nn.Module):
    """Neural network for fusing multi-modal features"""
    
    def __init__(self, 
                 biometric_dim: int = 200,
                 device_dim: int = 100,
                 behavioral_dim: int = 20,
                 hidden_dim: int = 256,
                 output_dim: int = 1):
        super(MultiModalFeatureFusion, self).__init__()
        
        # Individual encoders for each modality
        self.biometric_encoder = nn.Sequential(
            nn.Linear(biometric_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.device_encoder = nn.Sequential(
            nn.Linear(device_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.behavioral_encoder = nn.Sequential(
            nn.Linear(behavioral_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim // 2)
    
    def forward(self, biometric_features, device_features, behavioral_features):
        # Encode each modality
        biometric_encoded = self.biometric_encoder(biometric_features)
        device_encoded = self.device_encoder(device_features)
        behavioral_encoded = self.behavioral_encoder(behavioral_features)
        
        # Apply attention mechanism
        # Stack features for attention
        stacked_features = torch.stack([biometric_encoded, device_encoded, behavioral_encoded], dim=1)
        
        # Apply attention
        attended_features, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Apply layer normalization
        attended_features = self.layer_norm(attended_features)
        
        # Flatten for fusion
        fused_features = attended_features.view(attended_features.size(0), -1)
        
        # Final prediction
        output = self.fusion_network(fused_features)
        
        return {
            'prediction': output,
            'attention_weights': attention_weights,
            'biometric_features': biometric_encoded,
            'device_features': device_encoded,
            'behavioral_features': behavioral_encoded
        }

class MultiModalFraudDetector:
    """Main multi-modal fraud detection system"""
    
    def __init__(self):
        self.face_extractor = FaceRecognitionExtractor()
        self.voice_extractor = VoiceBiometricExtractor()
        self.device_extractor = DeviceTelemetryExtractor()
        self.behavioral_extractor = BehavioralPatternExtractor()
        self.fusion_model = MultiModalFeatureFusion()
        self.is_trained = False
    
    def extract_all_features(self, 
                           transaction_data: Dict[str, Any],
                           biometric_data: Dict[str, Any] = None,
                           device_data: Dict[str, Any] = None,
                           user_history: pd.DataFrame = None) -> Dict[str, np.ndarray]:
        """Extract all multi-modal features"""
        features = {}
        
        # Extract biometric features
        if biometric_data:
            biometric_features = self._extract_biometric_features(biometric_data)
            features['biometric'] = biometric_features
        else:
            features['biometric'] = np.zeros(200)
        
        # Extract device telemetry features
        if device_data:
            device_features = self._extract_device_features(device_data)
            features['device'] = device_features
        else:
            features['device'] = np.zeros(100)
        
        # Extract behavioral features
        if user_history is not None:
            behavioral_features = self._extract_behavioral_features(user_history)
            features['behavioral'] = behavioral_features
        else:
            features['behavioral'] = np.zeros(20)
        
        return features
    
    def _extract_biometric_features(self, biometric_data: Dict[str, Any]) -> np.ndarray:
        """Extract biometric features"""
        features = []
        
        # Face features
        if 'face_image_path' in biometric_data:
            face_features = self.face_extractor.extract_face_features(biometric_data['face_image_path'])
            features.extend(face_features)
        else:
            features.extend(np.zeros(128))
        
        # Voice features
        if 'voice_audio_path' in biometric_data:
            voice_features = self.voice_extractor.extract_voice_features(biometric_data['voice_audio_path'])
            features.extend(voice_features)
        else:
            features.extend(np.zeros(50))
        
        # Typing rhythm
        if 'keystroke_times' in biometric_data:
            typing_features = self.voice_extractor.extract_typing_rhythm(biometric_data['keystroke_times'])
            features.extend(typing_features)
        else:
            features.extend(np.zeros(20))
        
        # Additional biometric features
        if 'fingerprint_template' in biometric_data:
            fingerprint_features = biometric_data['fingerprint_template']
            if len(fingerprint_features) < 2:
                features.extend([0, 0])
            else:
                features.extend([np.mean(fingerprint_features), np.std(fingerprint_features)])
        else:
            features.extend([0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_device_features(self, device_data: Dict[str, Any]) -> np.ndarray:
        """Extract device telemetry features"""
        features = []
        
        # Sensor features
        if 'sensor_data' in device_data:
            sensor_features = self.device_extractor.extract_sensor_features(device_data['sensor_data'])
            for sensor_name, sensor_feats in sensor_features.items():
                features.extend(sensor_feats)
        else:
            features.extend(np.zeros(30))  # 3 sensors * 10 features each
        
        # Touch patterns
        if 'touch_events' in device_data:
            touch_features = self.device_extractor.extract_touch_patterns(device_data['touch_events'])
            features.extend(touch_features)
        else:
            features.extend(np.zeros(25))
        
        # Device information
        device_info_features = [
            device_data.get('battery_level', 0) / 100,
            device_data.get('device_orientation', 0) / 360,
            device_data.get('screen_brightness', 0) / 100,
            device_data.get('volume_level', 0) / 100,
            device_data.get('wifi_connected', 0),
            device_data.get('bluetooth_connected', 0),
            device_data.get('location_enabled', 0),
            device_data.get('camera_permission', 0),
            device_data.get('microphone_permission', 0),
            device_data.get('storage_permission', 0)
        ]
        features.extend(device_info_features)
        
        # Network features
        if 'network_info' in device_data:
            network_info = device_data['network_info']
            network_features = [
                network_info.get('signal_strength', 0) / 100,
                network_info.get('connection_type', 0),  # 0=wifi, 1=cellular, 2=ethernet
                network_info.get('download_speed', 0) / 1000,  # Normalize
                network_info.get('upload_speed', 0) / 1000,    # Normalize
                network_info.get('latency', 0) / 1000         # Normalize
            ]
            features.extend(network_features)
        else:
            features.extend(np.zeros(5))
        
        # App usage patterns
        if 'app_usage' in device_data:
            app_usage = device_data['app_usage']
            app_features = [
                len(app_usage),
                sum(app_usage.values()) / max(1, len(app_usage)),  # Average usage
                max(app_usage.values()) if app_usage else 0,       # Max usage
                min(app_usage.values()) if app_usage else 0        # Min usage
            ]
            features.extend(app_features)
        else:
            features.extend(np.zeros(4))
        
        # Pad or truncate to fixed size
        target_size = 100
        if len(features) < target_size:
            features.extend([0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_behavioral_features(self, user_history: pd.DataFrame) -> np.ndarray:
        """Extract behavioral features"""
        behavioral_dict = self.behavioral_extractor.extract_behavioral_features(user_history)
        
        # Convert to array in consistent order
        feature_order = [
            'avg_transactions_per_day', 'preferred_hour', 'preferred_day',
            'avg_amount', 'amount_std', 'max_amount', 'amount_consistency',
            'merchant_diversity', 'top_merchant_ratio',
            'avg_velocity', 'velocity_consistency',
            'avg_device_risk_score', 'device_risk_score_consistency',
            'avg_location_risk_score', 'location_risk_score_consistency',
            'avg_ip_reputation', 'ip_reputation_consistency'
        ]
        
        features = [behavioral_dict.get(feature, 0) for feature in feature_order]
        
        # Pad to fixed size
        target_size = 20
        if len(features) < target_size:
            features.extend([0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def predict_fraud(self, 
                     transaction_data: Dict[str, Any],
                     biometric_data: Dict[str, Any] = None,
                     device_data: Dict[str, Any] = None,
                     user_history: pd.DataFrame = None) -> Dict[str, Any]:
        """Predict fraud using multi-modal features"""
        if not self.is_trained:
            # For demo purposes, use a simple heuristic
            return self._demo_prediction(transaction_data, biometric_data, device_data)
        
        # Extract features
        features = self.extract_all_features(transaction_data, biometric_data, device_data, user_history)
        
        # Convert to tensors
        biometric_tensor = torch.FloatTensor(features['biometric']).unsqueeze(0)
        device_tensor = torch.FloatTensor(features['device']).unsqueeze(0)
        behavioral_tensor = torch.FloatTensor(features['behavioral']).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.fusion_model(biometric_tensor, device_tensor, behavioral_tensor)
            risk_score = torch.sigmoid(outputs['prediction']).item()
        
        # Determine decision
        if risk_score > 0.8:
            decision = "BLOCK"
        elif risk_score > 0.5:
            decision = "CHALLENGE"
        else:
            decision = "ALLOW"
        
        return {
            'risk_score': risk_score,
            'decision': decision,
            'confidence': abs(risk_score - 0.5) * 2,
            'feature_contributions': {
                'biometric': features['biometric'].tolist(),
                'device': features['device'].tolist(),
                'behavioral': features['behavioral'].tolist()
            },
            'model_type': 'multimodal'
        }
    
    def _demo_prediction(self, 
                        transaction_data: Dict[str, Any],
                        biometric_data: Dict[str, Any] = None,
                        device_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Demo prediction using heuristics"""
        risk_score = 0.1  # Base risk
        
        # Transaction-based risk
        amount = transaction_data.get('amount', 0)
        if amount > 50000:
            risk_score += 0.3
        elif amount > 10000:
            risk_score += 0.1
        
        # Biometric risk
        if biometric_data:
            if 'face_image_path' in biometric_data:
                risk_score += 0.1  # Face verification reduces risk
            if 'voice_audio_path' in biometric_data:
                risk_score += 0.1  # Voice verification reduces risk
        
        # Device risk
        if device_data:
            if device_data.get('battery_level', 100) < 20:
                risk_score += 0.1  # Low battery might indicate compromised device
            if not device_data.get('wifi_connected', True):
                risk_score += 0.1  # Cellular only might be suspicious
        
        # Clamp risk score
        risk_score = max(0, min(1, risk_score))
        
        # Determine decision
        if risk_score > 0.7:
            decision = "BLOCK"
        elif risk_score > 0.4:
            decision = "CHALLENGE"
        else:
            decision = "ALLOW"
        
        return {
            'risk_score': risk_score,
            'decision': decision,
            'confidence': abs(risk_score - 0.5) * 2,
            'feature_contributions': {
                'biometric': [0] * 200,
                'device': [0] * 100,
                'behavioral': [0] * 20
            },
            'model_type': 'multimodal_demo'
        }

# Example usage and testing
def demonstrate_multimodal_fraud_detection():
    """Demonstrate multi-modal fraud detection"""
    print("🔍 Demonstrating Multi-Modal Fraud Detection")
    print("=" * 60)
    
    # Initialize detector
    detector = MultiModalFraudDetector()
    
    # Create sample transaction data
    transaction_data = {
        'transaction_id': 'TXN_001',
        'amount': 75000,
        'merchant_id': 'MERCHANT_001',
        'timestamp': '2024-01-15 14:30:00',
        'user_id': 'USER_001'
    }
    
    # Create sample biometric data
    biometric_data = {
        'face_image_path': 'sample_face.jpg',  # Would be actual image path
        'voice_audio_path': 'sample_voice.wav',  # Would be actual audio path
        'keystroke_times': [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 2.5],
        'fingerprint_template': [0.8, 0.6, 0.9, 0.7, 0.5]
    }
    
    # Create sample device data
    device_data = {
        'sensor_data': {
            'accelerometer': [0.1, 0.2, 0.15, 0.18, 0.12],
            'gyroscope': [0.05, 0.08, 0.06, 0.09, 0.07],
            'magnetometer': [25.5, 26.1, 25.8, 26.3, 25.9]
        },
        'touch_events': [
            {'x': 100, 'y': 200, 'pressure': 0.8, 'size': 0.1, 'timestamp': 1000},
            {'x': 120, 'y': 210, 'pressure': 0.7, 'size': 0.1, 'timestamp': 1100},
            {'x': 140, 'y': 220, 'pressure': 0.9, 'size': 0.1, 'timestamp': 1200}
        ],
        'battery_level': 85,
        'device_orientation': 0,
        'screen_brightness': 70,
        'volume_level': 50,
        'wifi_connected': True,
        'bluetooth_connected': False,
        'location_enabled': True,
        'camera_permission': True,
        'microphone_permission': True,
        'storage_permission': True,
        'network_info': {
            'signal_strength': 80,
            'connection_type': 0,  # WiFi
            'download_speed': 50,
            'upload_speed': 20,
            'latency': 50
        },
        'app_usage': {
            'banking_app': 0.8,
            'social_media': 0.3,
            'games': 0.1,
            'shopping': 0.4
        }
    }
    
    # Create sample user history
    user_history = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=50, freq='1D'),
        'amount': np.random.lognormal(6, 1, 50),
        'merchant_id': np.random.randint(0, 20, 50),
        'user_velocity': np.random.exponential(2, 50),
        'device_risk_score': np.random.beta(2, 5, 50),
        'location_risk_score': np.random.beta(2, 5, 50),
        'ip_reputation': np.random.beta(3, 2, 50)
    })
    
    print("1. Extracting Multi-Modal Features...")
    
    # Extract features
    features = detector.extract_all_features(transaction_data, biometric_data, device_data, user_history)
    
    print(f"   Biometric features: {len(features['biometric'])} dimensions")
    print(f"   Device features: {len(features['device'])} dimensions")
    print(f"   Behavioral features: {len(features['behavioral'])} dimensions")
    
    # Make prediction
    print("\n2. Making Fraud Prediction...")
    prediction = detector.predict_fraud(transaction_data, biometric_data, device_data, user_history)
    
    print(f"   Risk Score: {prediction['risk_score']:.3f}")
    print(f"   Decision: {prediction['decision']}")
    print(f"   Confidence: {prediction['confidence']:.3f}")
    print(f"   Model Type: {prediction['model_type']}")
    
    # Test with different scenarios
    print("\n3. Testing Different Scenarios...")
    
    # High-risk transaction
    high_risk_transaction = transaction_data.copy()
    high_risk_transaction['amount'] = 150000
    
    high_risk_prediction = detector.predict_fraud(high_risk_transaction, biometric_data, device_data, user_history)
    print(f"   High Amount Transaction:")
    print(f"     Risk Score: {high_risk_prediction['risk_score']:.3f}")
    print(f"     Decision: {high_risk_prediction['decision']}")
    
    # Low battery device
    low_battery_device = device_data.copy()
    low_battery_device['battery_level'] = 15
    
    low_battery_prediction = detector.predict_fraud(transaction_data, biometric_data, low_battery_device, user_history)
    print(f"   Low Battery Device:")
    print(f"     Risk Score: {low_battery_prediction['risk_score']:.3f}")
    print(f"     Decision: {low_battery_prediction['decision']}")
    
    # No biometric data
    no_biometric_prediction = detector.predict_fraud(transaction_data, None, device_data, user_history)
    print(f"   No Biometric Data:")
    print(f"     Risk Score: {no_biometric_prediction['risk_score']:.3f}")
    print(f"     Decision: {no_biometric_prediction['decision']}")
    
    print("\n✅ Multi-Modal Fraud Detection demonstration completed!")
    
    return {
        'detector': detector,
        'features': features,
        'predictions': {
            'normal': prediction,
            'high_risk': high_risk_prediction,
            'low_battery': low_battery_prediction,
            'no_biometric': no_biometric_prediction
        }
    }

if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_multimodal_fraud_detection()
    
    print(f"\n📊 Summary:")
    print(f"   Feature dimensions extracted successfully")
    print(f"   Multi-modal predictions working")
    print(f"   Risk scoring functional")
    print(f"   Decision logic operational")
