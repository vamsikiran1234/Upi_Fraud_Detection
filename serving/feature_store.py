"""
Feature Store for UPI Fraud Detection
Manages feature storage and retrieval for real-time inference
"""

import asyncio
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib
from loguru import logger

class FeatureStore:
    """Feature store for managing ML features"""
    
    def __init__(self, settings):
        self.settings = settings
        self.redis_client = None
        self.postgres_conn = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize feature store connections"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                password=self.settings.redis_password,
                decode_responses=True
            )
            
            # Test Redis connection
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.ping
            )
            logger.info("Redis connection established")
            
            # Initialize PostgreSQL connection
            self.postgres_conn = psycopg2.connect(
                host=self.settings.postgres_host,
                port=self.settings.postgres_port,
                database=self.settings.postgres_db,
                user=self.settings.postgres_user,
                password=self.settings.postgres_password
            )
            logger.info("PostgreSQL connection established")
            
            # Create tables if they don't exist
            await self._create_tables()
            
            self.is_initialized = True
            logger.info("Feature store initialized successfully")
            
        except Exception as e:
            logger.error(f"Feature store initialization failed: {e}")
            raise
    
    async def _create_tables(self):
        """Create necessary database tables"""
        try:
            with self.postgres_conn.cursor() as cur:
                # User features table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_features (
                        user_id VARCHAR(32) PRIMARY KEY,
                        features JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Device features table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS device_features (
                        device_id VARCHAR(64) PRIMARY KEY,
                        features JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Location features table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS location_features (
                        location_hash VARCHAR(64) PRIMARY KEY,
                        features JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Transaction logs table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS transaction_logs (
                        id SERIAL PRIMARY KEY,
                        transaction_id VARCHAR(64) UNIQUE,
                        risk_score FLOAT,
                        fraud_probability FLOAT,
                        decision VARCHAR(20),
                        confidence FLOAT,
                        processing_time_ms FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_user_features_updated_at ON user_features(updated_at)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_device_features_updated_at ON device_features(updated_at)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_transaction_logs_created_at ON transaction_logs(created_at)")
                
                self.postgres_conn.commit()
                logger.info("Database tables created successfully")
                
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            self.postgres_conn.rollback()
            raise
    
    async def get_user_features(self, user_id: str, timestamp: datetime) -> Dict[str, Any]:
        """Get user-specific features"""
        try:
            # Try Redis cache first
            cache_key = f"user_features:{user_id}"
            cached_features = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, cache_key
            )
            
            if cached_features:
                return json.loads(cached_features)
            
            # Get from PostgreSQL
            with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT features FROM user_features WHERE user_id = %s",
                    (user_id,)
                )
                result = cur.fetchone()
                
                if result:
                    features = result['features']
                    # Cache for future use
                    await asyncio.get_event_loop().run_in_executor(
                        None, 
                        self.redis_client.setex,
                        cache_key,
                        self.settings.feature_cache_ttl,
                        json.dumps(features)
                    )
                    return features
                else:
                    # Return default features for new user
                    default_features = await self._get_default_user_features()
                    return default_features
                    
        except Exception as e:
            logger.error(f"Failed to get user features for {user_id}: {e}")
            return await self._get_default_user_features()
    
    async def get_device_features(self, device_id: str, timestamp: datetime) -> Dict[str, Any]:
        """Get device-specific features"""
        try:
            cache_key = f"device_features:{device_id}"
            cached_features = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, cache_key
            )
            
            if cached_features:
                return json.loads(cached_features)
            
            with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT features FROM device_features WHERE device_id = %s",
                    (device_id,)
                )
                result = cur.fetchone()
                
                if result:
                    features = result['features']
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.redis_client.setex,
                        cache_key,
                        self.settings.feature_cache_ttl,
                        json.dumps(features)
                    )
                    return features
                else:
                    default_features = await self._get_default_device_features()
                    return default_features
                    
        except Exception as e:
            logger.error(f"Failed to get device features for {device_id}: {e}")
            return await self._get_default_device_features()
    
    async def get_location_features(self, location: Dict[str, float], timestamp: datetime) -> Dict[str, Any]:
        """Get location-specific features"""
        try:
            # Create location hash
            location_str = f"{location['lat']:.4f},{location['lon']:.4f}"
            location_hash = hashlib.md5(location_str.encode()).hexdigest()
            
            cache_key = f"location_features:{location_hash}"
            cached_features = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, cache_key
            )
            
            if cached_features:
                return json.loads(cached_features)
            
            with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT features FROM location_features WHERE location_hash = %s",
                    (location_hash,)
                )
                result = cur.fetchone()
                
                if result:
                    features = result['features']
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.redis_client.setex,
                        cache_key,
                        self.settings.feature_cache_ttl,
                        json.dumps(features)
                    )
                    return features
                else:
                    default_features = await self._get_default_location_features(location)
                    return default_features
                    
        except Exception as e:
            logger.error(f"Failed to get location features: {e}")
            return await self._get_default_location_features(location)
    
    async def update_user_features(self, user_id: str, features: Dict[str, Any]):
        """Update user features"""
        try:
            with self.postgres_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_features (user_id, features, updated_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id) 
                    DO UPDATE SET 
                        features = EXCLUDED.features,
                        updated_at = EXCLUDED.updated_at
                """, (user_id, json.dumps(features), datetime.utcnow()))
                
                self.postgres_conn.commit()
                
                # Update cache
                cache_key = f"user_features:{user_id}"
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.setex,
                    cache_key,
                    self.settings.feature_cache_ttl,
                    json.dumps(features)
                )
                
        except Exception as e:
            logger.error(f"Failed to update user features for {user_id}: {e}")
            self.postgres_conn.rollback()
    
    async def _get_default_user_features(self) -> Dict[str, Any]:
        """Get default features for new user"""
        return {
            'user_velocity': 0.0,  # Transactions per hour
            'avg_amount': 0.0,     # Average transaction amount
            'max_amount': 0.0,     # Maximum transaction amount
            'time_since_last_tx': 24.0,  # Hours since last transaction
            'merchant_diversity': 0.0,   # Number of unique merchants
            'payment_pattern': 0.0,      # Payment pattern score
            'risk_score': 0.5,           # User risk score
            'account_age_days': 0,       # Account age in days
            'verification_status': 0.0,  # KYC verification level
            'transaction_count_24h': 0,  # Transactions in last 24h
            'transaction_count_7d': 0,   # Transactions in last 7 days
            'transaction_count_30d': 0,  # Transactions in last 30 days
            'amount_velocity_24h': 0.0,  # Amount velocity in last 24h
            'amount_velocity_7d': 0.0,   # Amount velocity in last 7d
            'amount_velocity_30d': 0.0,  # Amount velocity in last 30d
            'time_pattern_score': 0.5,   # Time pattern consistency
            'location_consistency': 0.5, # Location consistency score
            'device_consistency': 0.5,   # Device consistency score
            'merchant_risk_score': 0.5,  # Average merchant risk
            'amount_pattern_score': 0.5  # Amount pattern consistency
        }
    
    async def _get_default_device_features(self) -> Dict[str, Any]:
        """Get default features for new device"""
        return {
            'device_risk_score': 0.5,    # Device risk assessment
            'device_age_days': 0,        # Device age in days
            'os_version': 'unknown',     # Operating system version
            'browser_version': 'unknown', # Browser version
            'screen_resolution': 'unknown', # Screen resolution
            'timezone': 'UTC',           # Device timezone
            'language': 'en',            # Device language
            'is_mobile': False,          # Is mobile device
            'is_tablet': False,          # Is tablet device
            'is_desktop': True,          # Is desktop device
            'has_vpn': False,            # Using VPN
            'has_proxy': False,          # Using proxy
            'ip_reputation': 0.5,        # IP reputation score
            'location_consistency': 0.5, # Location consistency
            'device_fingerprint': 'unknown', # Device fingerprint
            'app_version': 'unknown',    # App version
            'device_model': 'unknown',   # Device model
            'carrier': 'unknown',        # Mobile carrier
            'network_type': 'unknown',   # Network type
            'battery_level': 0.0,        # Battery level
            'is_charging': False         # Is charging
        }
    
    async def _get_default_location_features(self, location: Dict[str, float]) -> Dict[str, Any]:
        """Get default features for new location"""
        return {
            'location_risk_score': 0.5,  # Location risk assessment
            'country_code': 'IN',        # Country code
            'state': 'unknown',          # State/Province
            'city': 'unknown',           # City
            'postal_code': 'unknown',    # Postal code
            'timezone': 'Asia/Kolkata',  # Timezone
            'is_high_risk_country': False, # High risk country
            'is_high_risk_region': False,  # High risk region
            'is_high_risk_city': False,    # High risk city
            'population_density': 0.0,     # Population density
            'economic_indicators': 0.5,    # Economic indicators
            'crime_rate': 0.5,             # Crime rate
            'fraud_rate': 0.5,             # Historical fraud rate
            'merchant_density': 0.0,       # Merchant density
            'atm_density': 0.0,            # ATM density
            'bank_density': 0.0,           # Bank density
            'internet_penetration': 0.5,   # Internet penetration
            'mobile_penetration': 0.5,     # Mobile penetration
            'digital_payment_adoption': 0.5, # Digital payment adoption
            'regulatory_environment': 0.5   # Regulatory environment score
        }
    
    async def is_healthy(self) -> bool:
        """Check if feature store is healthy"""
        try:
            # Check Redis
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.ping
            )
            
            # Check PostgreSQL
            with self.postgres_conn.cursor() as cur:
                cur.execute("SELECT 1")
            
            return True
            
        except Exception as e:
            logger.error(f"Feature store health check failed: {e}")
            return False
    
    async def close(self):
        """Close feature store connections"""
        try:
            if self.redis_client:
                self.redis_client.close()
            if self.postgres_conn:
                self.postgres_conn.close()
            logger.info("Feature store connections closed")
        except Exception as e:
            logger.error(f"Error closing feature store: {e}")
