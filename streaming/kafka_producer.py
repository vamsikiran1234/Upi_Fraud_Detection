"""
UPI Fraud Detection - Kafka Producer
Simulates real-time transaction data for testing the streaming pipeline
"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
import uuid
from kafka import KafkaProducer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UPITransactionGenerator:
    """Generates realistic UPI transaction data for testing"""
    
    def __init__(self):
        self.merchant_categories = [
            "grocery", "restaurant", "fuel", "pharmacy", "utilities", 
            "shopping", "entertainment", "transport", "education", "healthcare",
            "gambling", "adult", "crypto"  # High-risk categories
        ]
        
        self.payment_methods = ["UPI", "CARD", "WALLET", "NETBANKING"]
        
        self.user_agents = [
            "Mozilla/5.0 (Linux; Android 10; Mobile) AppleWebKit/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        ]
        
        # Simulate user profiles for realistic patterns
        self.user_profiles = self._generate_user_profiles(1000)
        
    def _generate_user_profiles(self, count: int) -> List[Dict]:
        """Generate user profiles with different risk levels"""
        profiles = []
        for i in range(count):
            profile = {
                "upi_id": f"user{i}@paytm",
                "risk_level": random.choices(["low", "medium", "high"], weights=[0.8, 0.15, 0.05])[0],
                "avg_transaction_amount": random.uniform(100, 5000),
                "preferred_merchants": random.sample(self.merchant_categories, random.randint(2, 5)),
                "device_id": f"device_{uuid.uuid4().hex[:8]}",
                "home_location": {
                    "latitude": random.uniform(8.4, 37.6),  # India lat range
                    "longitude": random.uniform(68.7, 97.25)  # India lng range
                }
            }
            profiles.append(profile)
        return profiles
    
    def generate_transaction(self, user_profile: Dict = None, fraud_probability: float = 0.05) -> Dict[str, Any]:
        """Generate a single transaction"""
        if user_profile is None:
            user_profile = random.choice(self.user_profiles)
        
        is_fraud = random.random() < fraud_probability
        
        # Base transaction
        transaction = {
            "transaction_id": str(uuid.uuid4()),
            "upi_id": user_profile["upi_id"],
            "device_id": user_profile["device_id"],
            "timestamp": datetime.utcnow().isoformat(),
            "payment_method": random.choice(self.payment_methods),
            "session_id": f"session_{uuid.uuid4().hex[:8]}",
            "user_agent": random.choice(self.user_agents)
        }
        
        if is_fraud:
            # Generate fraudulent transaction patterns
            transaction.update(self._generate_fraud_transaction(user_profile))
        else:
            # Generate normal transaction
            transaction.update(self._generate_normal_transaction(user_profile))
        
        return transaction
    
    def _generate_normal_transaction(self, user_profile: Dict) -> Dict[str, Any]:
        """Generate normal transaction patterns"""
        # Amount based on user profile with some variance
        base_amount = user_profile["avg_transaction_amount"]
        amount = max(1, random.normalvariate(base_amount, base_amount * 0.3))
        
        # Merchant from preferred categories
        merchant_category = random.choice(user_profile["preferred_merchants"])
        merchant_id = f"merchant_{merchant_category}_{random.randint(1, 100)}"
        
        # Location near home with some variance
        home_lat = user_profile["home_location"]["latitude"]
        home_lng = user_profile["home_location"]["longitude"]
        
        return {
            "amount": round(amount, 2),
            "merchant_id": merchant_id,
            "merchant_category": merchant_category,
            "ip_address": self._generate_ip_address(),
            "location": {
                "latitude": home_lat + random.uniform(-0.1, 0.1),
                "longitude": home_lng + random.uniform(-0.1, 0.1)
            },
            "sms_content": self._generate_normal_sms(amount, merchant_id),
            "merchant_notes": f"Payment to {merchant_id}"
        }
    
    def _generate_fraud_transaction(self, user_profile: Dict) -> Dict[str, Any]:
        """Generate fraudulent transaction patterns"""
        fraud_type = random.choice([
            "high_amount", "velocity_attack", "location_anomaly", 
            "suspicious_merchant", "device_change", "phishing"
        ])
        
        if fraud_type == "high_amount":
            # Unusually high amount
            amount = user_profile["avg_transaction_amount"] * random.uniform(5, 20)
            merchant_category = random.choice(["gambling", "crypto", "adult"])
        
        elif fraud_type == "velocity_attack":
            # Multiple rapid transactions
            amount = random.uniform(100, 1000)
            merchant_category = random.choice(self.merchant_categories)
        
        elif fraud_type == "location_anomaly":
            # Transaction from distant location
            amount = random.uniform(500, 5000)
            merchant_category = random.choice(self.merchant_categories)
        
        elif fraud_type == "suspicious_merchant":
            # High-risk merchant
            amount = random.uniform(1000, 10000)
            merchant_category = random.choice(["gambling", "crypto", "adult"])
        
        elif fraud_type == "device_change":
            # New device
            amount = random.uniform(2000, 15000)
            merchant_category = random.choice(self.merchant_categories)
        
        else:  # phishing
            amount = random.uniform(100, 2000)
            merchant_category = "utilities"
        
        merchant_id = f"merchant_{merchant_category}_{random.randint(1, 100)}"
        
        # Fraudulent location (far from home or suspicious)
        if fraud_type == "location_anomaly":
            location = {
                "latitude": random.uniform(8.4, 37.6),
                "longitude": random.uniform(68.7, 97.25)
            }
        else:
            home_lat = user_profile["home_location"]["latitude"]
            home_lng = user_profile["home_location"]["longitude"]
            location = {
                "latitude": home_lat + random.uniform(-0.05, 0.05),
                "longitude": home_lng + random.uniform(-0.05, 0.05)
            }
        
        return {
            "amount": round(amount, 2),
            "merchant_id": merchant_id,
            "merchant_category": merchant_category,
            "ip_address": self._generate_suspicious_ip(),
            "location": location,
            "sms_content": self._generate_suspicious_sms(amount, merchant_id, fraud_type),
            "merchant_notes": f"Payment to {merchant_id} - {fraud_type}"
        }
    
    def _generate_ip_address(self) -> str:
        """Generate realistic IP address"""
        return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    
    def _generate_suspicious_ip(self) -> str:
        """Generate suspicious IP addresses (VPN, Tor, etc.)"""
        suspicious_ranges = [
            "10.0.0", "172.16.0", "192.168.0",  # Private IPs
            "185.220.100", "185.220.101"  # Known Tor exit nodes
        ]
        base = random.choice(suspicious_ranges)
        return f"{base}.{random.randint(1, 254)}"
    
    def _generate_normal_sms(self, amount: float, merchant: str) -> str:
        """Generate normal SMS content"""
        templates = [
            f"Rs {amount:.2f} debited from your account for {merchant}. Available balance: Rs {random.randint(1000, 50000)}",
            f"Payment of Rs {amount:.2f} to {merchant} successful. Ref: {random.randint(100000, 999999)}",
            f"UPI transaction of Rs {amount:.2f} completed successfully to {merchant}"
        ]
        return random.choice(templates)
    
    def _generate_suspicious_sms(self, amount: float, merchant: str, fraud_type: str) -> str:
        """Generate suspicious SMS content"""
        if fraud_type == "phishing":
            templates = [
                f"URGENT: Your account will be suspended. Click link to verify: http://suspicious-link.com",
                f"Your OTP is 123456. Do not share with anyone. If not requested, call immediately.",
                f"Suspicious activity detected. Verify your PIN by replying to this message."
            ]
        else:
            templates = [
                f"Rs {amount:.2f} debited from your account for {merchant}. If not done by you, call immediately",
                f"Large transaction alert: Rs {amount:.2f} to {merchant}. Contact us if unauthorized",
                f"Security alert: Rs {amount:.2f} transaction from new device to {merchant}"
            ]
        return random.choice(templates)

class KafkaTransactionProducer:
    """Kafka producer for UPI transactions"""
    
    def __init__(self, bootstrap_servers: str, topic: str):
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3,
            batch_size=16384,
            linger_ms=10,
            buffer_memory=33554432
        )
        self.generator = UPITransactionGenerator()
        
    def produce_transactions(self, rate_per_second: int = 10, duration_minutes: int = 60, fraud_rate: float = 0.05):
        """Produce transactions at specified rate"""
        logger.info(f"Starting transaction production: {rate_per_second} TPS for {duration_minutes} minutes")
        
        end_time = datetime.utcnow() + timedelta(minutes=duration_minutes)
        transaction_count = 0
        
        try:
            while datetime.utcnow() < end_time:
                batch_start = time.time()
                
                # Generate batch of transactions
                for _ in range(rate_per_second):
                    transaction = self.generator.generate_transaction(fraud_probability=fraud_rate)
                    
                    # Send to Kafka
                    future = self.producer.send(
                        self.topic,
                        key=transaction["transaction_id"],
                        value=transaction
                    )
                    
                    transaction_count += 1
                    
                    # Log every 100 transactions
                    if transaction_count % 100 == 0:
                        logger.info(f"Produced {transaction_count} transactions")
                
                # Wait to maintain rate
                batch_duration = time.time() - batch_start
                sleep_time = max(0, 1.0 - batch_duration)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Production stopped by user")
        except Exception as e:
            logger.error(f"Production failed: {e}")
        finally:
            self.producer.flush()
            self.producer.close()
            logger.info(f"Total transactions produced: {transaction_count}")

    def produce_fraud_scenario(self, scenario_type: str = "velocity_attack"):
        """Produce specific fraud scenarios for testing"""
        logger.info(f"Producing fraud scenario: {scenario_type}")
        
        user_profile = random.choice(self.generator.user_profiles)
        
        if scenario_type == "velocity_attack":
            # Rapid succession of transactions
            for i in range(20):
                transaction = self.generator._generate_fraud_transaction(user_profile)
                transaction["transaction_id"] = str(uuid.uuid4())
                transaction["timestamp"] = (datetime.utcnow() + timedelta(seconds=i*2)).isoformat()
                
                self.producer.send(self.topic, key=transaction["transaction_id"], value=transaction)
                time.sleep(0.1)
        
        elif scenario_type == "amount_escalation":
            # Gradually increasing amounts
            base_amount = user_profile["avg_transaction_amount"]
            for i in range(10):
                transaction = self.generator.generate_transaction(user_profile, fraud_probability=0.8)
                transaction["amount"] = base_amount * (1.5 ** i)
                transaction["timestamp"] = (datetime.utcnow() + timedelta(minutes=i*5)).isoformat()
                
                self.producer.send(self.topic, key=transaction["transaction_id"], value=transaction)
                time.sleep(0.1)
        
        self.producer.flush()
        logger.info(f"Fraud scenario '{scenario_type}' completed")

def main():
    """Main entry point for transaction producer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="UPI Transaction Kafka Producer")
    parser.add_argument("--brokers", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", default="raw-transactions", help="Kafka topic")
    parser.add_argument("--rate", type=int, default=10, help="Transactions per second")
    parser.add_argument("--duration", type=int, default=60, help="Duration in minutes")
    parser.add_argument("--fraud-rate", type=float, default=0.05, help="Fraud transaction rate")
    parser.add_argument("--scenario", help="Produce specific fraud scenario")
    
    args = parser.parse_args()
    
    producer = KafkaTransactionProducer(args.brokers, args.topic)
    
    if args.scenario:
        producer.produce_fraud_scenario(args.scenario)
    else:
        producer.produce_transactions(args.rate, args.duration, args.fraud_rate)

if __name__ == "__main__":
    main()
