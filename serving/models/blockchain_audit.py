"""
Blockchain-based Audit Trails for Immutable Decision Logs
Ensures tamper-proof record of all fraud detection decisions
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import base64

@dataclass
class AuditBlock:
    """Block in the audit blockchain"""
    index: int
    timestamp: str
    transaction_id: str
    decision: str
    risk_score: float
    model_version: str
    features_hash: str
    explanation_hash: str
    previous_hash: str
    nonce: int = 0
    hash: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the block"""
        block_string = f"{self.index}{self.timestamp}{self.transaction_id}{self.decision}{self.risk_score}{self.model_version}{self.features_hash}{self.explanation_hash}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 4) -> str:
        """Mine the block with proof of work"""
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        return self.hash

@dataclass
class FraudDecision:
    """Fraud detection decision to be audited"""
    transaction_id: str
    upi_id_hash: str
    amount: float
    merchant_id: str
    decision: str
    risk_score: float
    confidence: float
    model_version: str
    features: Dict[str, Any]
    explanation: Dict[str, Any]
    processing_time_ms: float
    timestamp: str
    analyst_review: Optional[str] = None
    case_id: Optional[str] = None

class BlockchainAuditTrail:
    """Blockchain-based audit trail for fraud detection decisions"""
    
    def __init__(self, difficulty: int = 4):
        self.chain: List[AuditBlock] = []
        self.difficulty = difficulty
        self.pending_decisions: List[FraudDecision] = []
        self.private_key = None
        self.public_key = None
        self._generate_key_pair()
        self._create_genesis_block()
    
    def _generate_key_pair(self):
        """Generate RSA key pair for digital signatures"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
    
    def _create_genesis_block(self):
        """Create the genesis block"""
        genesis_block = AuditBlock(
            index=0,
            timestamp=datetime.utcnow().isoformat(),
            transaction_id="GENESIS",
            decision="GENESIS",
            risk_score=0.0,
            model_version="1.0.0",
            features_hash="GENESIS",
            explanation_hash="GENESIS",
            previous_hash="0"
        )
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)
    
    def _hash_sensitive_data(self, data: Dict[str, Any]) -> str:
        """Create hash of sensitive data for privacy"""
        # Remove PII and create deterministic hash
        sanitized_data = {k: v for k, v in data.items() 
                         if k not in ['upi_id', 'ip_address', 'device_id']}
        data_string = json.dumps(sanitized_data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def _sign_decision(self, decision: FraudDecision) -> str:
        """Create digital signature for the decision"""
        # Create message to sign
        message = f"{decision.transaction_id}{decision.decision}{decision.risk_score}{decision.timestamp}"
        message_bytes = message.encode()
        
        # Sign the message
        signature = self.private_key.sign(
            message_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode()
    
    def add_decision(self, decision: FraudDecision) -> str:
        """Add a fraud detection decision to the blockchain"""
        # Hash sensitive data
        features_hash = self._hash_sensitive_data(decision.features)
        explanation_hash = self._hash_sensitive_data(decision.explanation)
        
        # Create new block
        new_block = AuditBlock(
            index=len(self.chain),
            timestamp=decision.timestamp,
            transaction_id=decision.transaction_id,
            decision=decision.decision,
            risk_score=decision.risk_score,
            model_version=decision.model_version,
            features_hash=features_hash,
            explanation_hash=explanation_hash,
            previous_hash=self.chain[-1].hash
        )
        
        # Mine the block
        new_block.hash = new_block.mine_block(self.difficulty)
        
        # Add to chain
        self.chain.append(new_block)
        
        # Create digital signature
        signature = self._sign_decision(decision)
        
        return new_block.hash
    
    def verify_chain(self) -> bool:
        """Verify the integrity of the blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Verify current block hash
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # Verify previous hash
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def get_decision_history(self, transaction_id: str) -> List[AuditBlock]:
        """Get audit trail for a specific transaction"""
        return [block for block in self.chain 
                if block.transaction_id == transaction_id]
    
    def get_decisions_by_merchant(self, merchant_id: str) -> List[AuditBlock]:
        """Get all decisions for a specific merchant"""
        # This would require storing merchant_id in blocks
        # For now, return empty list
        return []
    
    def get_risk_score_distribution(self) -> Dict[str, Any]:
        """Get distribution of risk scores in the audit trail"""
        risk_scores = [block.risk_score for block in self.chain[1:]]  # Exclude genesis
        
        return {
            'count': len(risk_scores),
            'mean': sum(risk_scores) / len(risk_scores) if risk_scores else 0,
            'min': min(risk_scores) if risk_scores else 0,
            'max': max(risk_scores) if risk_scores else 0,
            'high_risk_count': sum(1 for score in risk_scores if score > 0.7),
            'medium_risk_count': sum(1 for score in risk_scores if 0.3 <= score <= 0.7),
            'low_risk_count': sum(1 for score in risk_scores if score < 0.3)
        }
    
    def export_audit_trail(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Export audit trail for a date range"""
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        filtered_blocks = []
        for block in self.chain[1:]:  # Exclude genesis
            block_dt = datetime.fromisoformat(block.timestamp)
            if start_dt <= block_dt <= end_dt:
                filtered_blocks.append(asdict(block))
        
        return filtered_blocks
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        return {
            'total_blocks': len(self.chain),
            'total_decisions': len(self.chain) - 1,  # Exclude genesis
            'chain_verified': self.verify_chain(),
            'difficulty': self.difficulty,
            'last_block_hash': self.chain[-1].hash if self.chain else None,
            'genesis_timestamp': self.chain[0].timestamp if self.chain else None,
            'last_decision_timestamp': self.chain[-1].timestamp if len(self.chain) > 1 else None
        }

class DistributedAuditNetwork:
    """Distributed network of audit nodes for consensus"""
    
    def __init__(self, node_id: str, peers: List[str] = None):
        self.node_id = node_id
        self.peers = peers or []
        self.local_blockchain = BlockchainAuditTrail()
        self.consensus_threshold = 0.51  # 51% consensus required
    
    def broadcast_decision(self, decision: FraudDecision) -> bool:
        """Broadcast decision to all peers for consensus"""
        # Add to local blockchain
        block_hash = self.local_blockchain.add_decision(decision)
        
        # Broadcast to peers
        consensus_votes = 1  # Count our own vote
        total_peers = len(self.peers) + 1
        
        for peer in self.peers:
            try:
                # Send decision to peer
                response = requests.post(
                    f"http://{peer}/audit/validate",
                    json=asdict(decision),
                    timeout=5
                )
                if response.status_code == 200:
                    consensus_votes += 1
            except:
                # Peer is offline, continue
                pass
        
        # Check if consensus is reached
        consensus_ratio = consensus_votes / total_peers
        return consensus_ratio >= self.consensus_threshold
    
    def validate_peer_decision(self, decision_data: Dict[str, Any]) -> bool:
        """Validate a decision from a peer"""
        try:
            decision = FraudDecision(**decision_data)
            
            # Validate decision integrity
            if not self._validate_decision_integrity(decision):
                return False
            
            # Add to local blockchain
            self.local_blockchain.add_decision(decision)
            return True
            
        except Exception as e:
            print(f"Error validating peer decision: {e}")
            return False
    
    def _validate_decision_integrity(self, decision: FraudDecision) -> bool:
        """Validate the integrity of a decision"""
        # Check required fields
        required_fields = ['transaction_id', 'decision', 'risk_score', 'timestamp']
        for field in required_fields:
            if not hasattr(decision, field) or getattr(decision, field) is None:
                return False
        
        # Validate risk score range
        if not 0 <= decision.risk_score <= 1:
            return False
        
        # Validate decision values
        valid_decisions = ['ALLOW', 'CHALLENGE', 'BLOCK']
        if decision.decision not in valid_decisions:
            return False
        
        return True
    
    def sync_with_peers(self) -> bool:
        """Sync blockchain with peers"""
        longest_chain = self.local_blockchain.chain
        
        for peer in self.peers:
            try:
                # Get peer's blockchain
                response = requests.get(f"http://{peer}/audit/blockchain", timeout=5)
                if response.status_code == 200:
                    peer_chain_data = response.json()
                    
                    # Convert to AuditBlock objects
                    peer_chain = []
                    for block_data in peer_chain_data:
                        block = AuditBlock(**block_data)
                        peer_chain.append(block)
                    
                    # Use longest valid chain
                    if len(peer_chain) > len(longest_chain) and self._validate_chain(peer_chain):
                        longest_chain = peer_chain
                        
            except:
                # Peer is offline, continue
                pass
        
        # Update local blockchain if we found a longer valid chain
        if len(longest_chain) > len(self.local_blockchain.chain):
            self.local_blockchain.chain = longest_chain
            return True
        
        return False
    
    def _validate_chain(self, chain: List[AuditBlock]) -> bool:
        """Validate a blockchain"""
        for i in range(1, len(chain)):
            current_block = chain[i]
            previous_block = chain[i - 1]
            
            if current_block.hash != current_block.calculate_hash():
                return False
            
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True

class AuditTrailAPI:
    """API for blockchain audit trail system"""
    
    def __init__(self, node_id: str = "audit_node_1", peers: List[str] = None):
        self.network = DistributedAuditNetwork(node_id, peers)
        self.audit_trail = self.network.local_blockchain
    
    def log_fraud_decision(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log a fraud detection decision to the blockchain"""
        try:
            # Create FraudDecision object
            decision = FraudDecision(**decision_data)
            
            # Broadcast for consensus
            consensus_reached = self.network.broadcast_decision(decision)
            
            # Get block hash
            block_hash = self.audit_trail.chain[-1].hash
            
            return {
                'status': 'success',
                'block_hash': block_hash,
                'consensus_reached': consensus_reached,
                'block_index': len(self.audit_trail.chain) - 1,
                'timestamp': decision.timestamp
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to log decision: {str(e)}'
            }
    
    def get_audit_trail(self, transaction_id: str) -> Dict[str, Any]:
        """Get audit trail for a specific transaction"""
        try:
            history = self.audit_trail.get_decision_history(transaction_id)
            
            return {
                'status': 'success',
                'transaction_id': transaction_id,
                'audit_blocks': [asdict(block) for block in history],
                'total_blocks': len(history)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to get audit trail: {str(e)}'
            }
    
    def verify_audit_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the audit trail"""
        try:
            is_valid = self.audit_trail.verify_chain()
            stats = self.audit_trail.get_blockchain_stats()
            
            return {
                'status': 'success',
                'chain_valid': is_valid,
                'statistics': stats
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to verify integrity: {str(e)}'
            }
    
    def export_audit_data(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Export audit data for a date range"""
        try:
            audit_data = self.audit_trail.export_audit_trail(start_date, end_date)
            
            return {
                'status': 'success',
                'start_date': start_date,
                'end_date': end_date,
                'total_records': len(audit_data),
                'audit_data': audit_data
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to export audit data: {str(e)}'
            }
    
    def get_risk_analytics(self) -> Dict[str, Any]:
        """Get risk analytics from audit trail"""
        try:
            risk_distribution = self.audit_trail.get_risk_score_distribution()
            stats = self.audit_trail.get_blockchain_stats()
            
            return {
                'status': 'success',
                'risk_distribution': risk_distribution,
                'blockchain_stats': stats
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to get risk analytics: {str(e)}'
            }

# Example usage and testing
def demonstrate_blockchain_audit():
    """Demonstrate blockchain audit trail functionality"""
    print("🔗 Demonstrating Blockchain Audit Trail")
    print("=" * 50)
    
    # Initialize audit API
    audit_api = AuditTrailAPI("demo_node")
    
    # Create sample fraud decisions
    decisions = [
        {
            'transaction_id': 'TXN_001',
            'upi_id_hash': 'hash_123',
            'amount': 50000.0,
            'merchant_id': 'MERCHANT_001',
            'decision': 'BLOCK',
            'risk_score': 0.85,
            'confidence': 0.92,
            'model_version': '1.0.0',
            'features': {'amount': 50000, 'hour': 2, 'merchant_category': 'crypto'},
            'explanation': {'risk_factors': ['high_amount', 'night_time', 'high_risk_merchant']},
            'processing_time_ms': 45.2,
            'timestamp': datetime.utcnow().isoformat()
        },
        {
            'transaction_id': 'TXN_002',
            'upi_id_hash': 'hash_456',
            'amount': 1500.0,
            'merchant_id': 'MERCHANT_002',
            'decision': 'ALLOW',
            'risk_score': 0.15,
            'confidence': 0.88,
            'model_version': '1.0.0',
            'features': {'amount': 1500, 'hour': 14, 'merchant_category': 'food'},
            'explanation': {'risk_factors': ['low_amount', 'day_time', 'low_risk_merchant']},
            'processing_time_ms': 32.1,
            'timestamp': datetime.utcnow().isoformat()
        }
    ]
    
    # Log decisions to blockchain
    print("1. Logging fraud decisions to blockchain...")
    for decision in decisions:
        result = audit_api.log_fraud_decision(decision)
        print(f"   Transaction {decision['transaction_id']}: {result['status']}")
        print(f"   Block hash: {result['block_hash'][:16]}...")
        print(f"   Consensus reached: {result['consensus_reached']}")
    
    # Verify audit integrity
    print("\n2. Verifying audit trail integrity...")
    integrity_result = audit_api.verify_audit_integrity()
    print(f"   Chain valid: {integrity_result['chain_valid']}")
    print(f"   Total blocks: {integrity_result['statistics']['total_blocks']}")
    print(f"   Total decisions: {integrity_result['statistics']['total_decisions']}")
    
    # Get audit trail for specific transaction
    print("\n3. Retrieving audit trail for transaction...")
    trail_result = audit_api.get_audit_trail('TXN_001')
    print(f"   Transaction: {trail_result['transaction_id']}")
    print(f"   Audit blocks: {trail_result['total_blocks']}")
    
    # Get risk analytics
    print("\n4. Generating risk analytics...")
    analytics_result = audit_api.get_risk_analytics()
    risk_dist = analytics_result['risk_distribution']
    print(f"   Total decisions: {risk_dist['count']}")
    print(f"   Average risk score: {risk_dist['mean']:.3f}")
    print(f"   High risk decisions: {risk_dist['high_risk_count']}")
    print(f"   Medium risk decisions: {risk_dist['medium_risk_count']}")
    print(f"   Low risk decisions: {risk_dist['low_risk_count']}")
    
    print("\n✅ Blockchain audit trail demonstration completed!")
    
    return {
        'audit_api': audit_api,
        'decisions_logged': len(decisions),
        'blockchain_stats': integrity_result['statistics'],
        'risk_analytics': analytics_result['risk_distribution']
    }

if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_blockchain_audit()
    
    print(f"\n📊 Summary:")
    print(f"   Decisions logged: {results['decisions_logged']}")
    print(f"   Blockchain blocks: {results['blockchain_stats']['total_blocks']}")
    print(f"   Chain verified: {results['blockchain_stats']['chain_verified']}")
