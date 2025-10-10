"""
Proactive Threat Intelligence Ingestion
Integrates dark web feeds, phishing reports, and threat indicators
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import re
from dataclasses import dataclass
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import feedparser
import logging

@dataclass
class ThreatIndicator:
    """Threat intelligence indicator"""
    indicator_type: str  # IP, domain, email, hash, etc.
    value: str
    confidence: float
    severity: str  # low, medium, high, critical
    source: str
    first_seen: datetime
    last_seen: datetime
    tags: List[str]
    description: str
    false_positive_rate: float = 0.0

@dataclass
class ThreatFeed:
    """Threat intelligence feed configuration"""
    name: str
    url: str
    feed_type: str  # rss, json, csv, api
    update_frequency: int  # minutes
    last_update: Optional[datetime] = None
    enabled: bool = True
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

class DarkWebMonitor:
    """Monitor dark web for fraud-related threats"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_dark_web_for_fraud(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search dark web for fraud-related content"""
        threats = []
        
        # Simulate dark web search (in real implementation, use Tor or specialized APIs)
        for keyword in keywords:
            # Simulate finding threats
            simulated_threats = self._simulate_dark_web_search(keyword)
            threats.extend(simulated_threats)
        
        return threats
    
    def _simulate_dark_web_search(self, keyword: str) -> List[Dict[str, Any]]:
        """Simulate dark web search results"""
        # In real implementation, this would connect to dark web markets
        threats = []
        
        if 'upi' in keyword.lower() or 'payment' in keyword.lower():
            threats.extend([
                {
                    'type': 'fraud_kit',
                    'title': f'UPI Fraud Kit - {keyword}',
                    'description': 'Complete toolkit for UPI fraud operations',
                    'price': '$500',
                    'confidence': 0.9,
                    'severity': 'high',
                    'source': 'dark_web_market',
                    'timestamp': datetime.utcnow(),
                    'tags': ['upi', 'fraud', 'kit', 'payment']
                },
                {
                    'type': 'stolen_data',
                    'title': f'Stolen UPI Credentials - {keyword}',
                    'description': 'Database of compromised UPI accounts',
                    'price': '$200',
                    'confidence': 0.8,
                    'severity': 'critical',
                    'source': 'dark_web_market',
                    'timestamp': datetime.utcnow(),
                    'tags': ['upi', 'credentials', 'stolen', 'database']
                }
            ])
        
        return threats
    
    def monitor_fraud_forums(self, forum_urls: List[str]) -> List[Dict[str, Any]]:
        """Monitor fraud forums for new threats"""
        forum_threats = []
        
        for url in forum_urls:
            try:
                # Simulate forum monitoring
                threats = self._simulate_forum_monitoring(url)
                forum_threats.extend(threats)
            except Exception as e:
                logging.error(f"Error monitoring forum {url}: {e}")
        
        return forum_threats
    
    def _simulate_forum_monitoring(self, forum_url: str) -> List[Dict[str, Any]]:
        """Simulate forum monitoring"""
        return [
            {
                'type': 'discussion',
                'title': 'New UPI Bypass Method',
                'content': 'Discussion about new methods to bypass UPI security',
                'author': 'fraudster123',
                'confidence': 0.7,
                'severity': 'medium',
                'source': forum_url,
                'timestamp': datetime.utcnow(),
                'tags': ['upi', 'bypass', 'security', 'method']
            }
        ]

class PhishingIntelligence:
    """Collect and analyze phishing intelligence"""
    
    def __init__(self):
        self.session = requests.Session()
        self.known_phishing_patterns = self._load_phishing_patterns()
    
    def _load_phishing_patterns(self) -> List[Dict[str, Any]]:
        """Load known phishing patterns"""
        return [
            {
                'pattern': r'upi.*secure.*verify',
                'type': 'url_phishing',
                'confidence': 0.8,
                'description': 'UPI security verification phishing'
            },
            {
                'pattern': r'bank.*suspended.*reactivate',
                'type': 'account_phishing',
                'confidence': 0.9,
                'description': 'Bank account suspension phishing'
            },
            {
                'pattern': r'otp.*expired.*resend',
                'type': 'otp_phishing',
                'confidence': 0.7,
                'description': 'OTP expiration phishing'
            }
        ]
    
    def analyze_phishing_reports(self, reports: List[Dict[str, Any]]) -> List[ThreatIndicator]:
        """Analyze phishing reports to extract threat indicators"""
        indicators = []
        
        for report in reports:
            # Extract URLs
            urls = self._extract_urls(report.get('content', ''))
            for url in urls:
                if self._is_phishing_url(url):
                    indicators.append(ThreatIndicator(
                        indicator_type='url',
                        value=url,
                        confidence=0.8,
                        severity='high',
                        source='phishing_report',
                        first_seen=datetime.utcnow(),
                        last_seen=datetime.utcnow(),
                        tags=['phishing', 'url'],
                        description='Phishing URL detected'
                    ))
            
            # Extract email addresses
            emails = self._extract_emails(report.get('content', ''))
            for email in emails:
                if self._is_phishing_email(email):
                    indicators.append(ThreatIndicator(
                        indicator_type='email',
                        value=email,
                        confidence=0.7,
                        severity='medium',
                        source='phishing_report',
                        first_seen=datetime.utcnow(),
                        last_seen=datetime.utcnow(),
                        tags=['phishing', 'email'],
                        description='Phishing email detected'
                    ))
        
        return indicators
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
    
    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    def _is_phishing_url(self, url: str) -> bool:
        """Check if URL is likely phishing"""
        for pattern in self.known_phishing_patterns:
            if re.search(pattern['pattern'], url, re.IGNORECASE):
                return True
        return False
    
    def _is_phishing_email(self, email: str) -> bool:
        """Check if email is likely phishing"""
        suspicious_domains = ['gmail.com', 'yahoo.com', 'hotmail.com']  # Simplified
        domain = email.split('@')[1] if '@' in email else ''
        return domain in suspicious_domains

class ThreatFeedManager:
    """Manage multiple threat intelligence feeds"""
    
    def __init__(self):
        self.feeds = {}
        self.indicators = []
        self.feed_configs = self._load_feed_configurations()
    
    def _load_feed_configurations(self) -> List[ThreatFeed]:
        """Load threat feed configurations"""
        return [
            ThreatFeed(
                name='AbuseIPDB',
                url='https://api.abuseipdb.com/api/v2/blacklist',
                feed_type='api',
                update_frequency=60,
                api_key='your_api_key_here'
            ),
            ThreatFeed(
                name='PhishTank',
                url='https://data.phishtank.com/data/online-valid.json',
                feed_type='json',
                update_frequency=30
            ),
            ThreatFeed(
                name='Malware Domain List',
                url='https://mirror1.malwaredomains.com/files/domains.txt',
                feed_type='text',
                update_frequency=120
            ),
            ThreatFeed(
                name='Threat Intelligence RSS',
                url='https://feeds.feedburner.com/TheHackersNews',
                feed_type='rss',
                update_frequency=15
            )
        ]
    
    async def update_all_feeds(self) -> Dict[str, Any]:
        """Update all threat intelligence feeds"""
        results = {}
        
        for feed_config in self.feed_configs:
            if not feed_config.enabled:
                continue
            
            try:
                if feed_config.feed_type == 'api':
                    indicators = await self._update_api_feed(feed_config)
                elif feed_config.feed_type == 'json':
                    indicators = await self._update_json_feed(feed_config)
                elif feed_config.feed_type == 'rss':
                    indicators = await self._update_rss_feed(feed_config)
                elif feed_config.feed_type == 'text':
                    indicators = await self._update_text_feed(feed_config)
                else:
                    indicators = []
                
                results[feed_config.name] = {
                    'status': 'success',
                    'indicators_count': len(indicators),
                    'last_update': datetime.utcnow()
                }
                
                self.indicators.extend(indicators)
                
            except Exception as e:
                results[feed_config.name] = {
                    'status': 'error',
                    'error': str(e),
                    'last_update': datetime.utcnow()
                }
        
        return results
    
    async def _update_api_feed(self, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Update API-based threat feed"""
        # Simulate API call
        indicators = []
        
        # Simulate AbuseIPDB response
        if 'abuseipdb' in feed.name.lower():
            indicators.extend([
                ThreatIndicator(
                    indicator_type='ip',
                    value='192.168.1.100',
                    confidence=0.9,
                    severity='high',
                    source=feed.name,
                    first_seen=datetime.utcnow() - timedelta(days=1),
                    last_seen=datetime.utcnow(),
                    tags=['malware', 'botnet'],
                    description='IP associated with malware distribution'
                )
            ])
        
        return indicators
    
    async def _update_json_feed(self, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Update JSON-based threat feed"""
        indicators = []
        
        # Simulate PhishTank response
        if 'phishtank' in feed.name.lower():
            indicators.extend([
                ThreatIndicator(
                    indicator_type='url',
                    value='http://fake-bank-verification.com',
                    confidence=0.95,
                    severity='critical',
                    source=feed.name,
                    first_seen=datetime.utcnow() - timedelta(hours=2),
                    last_seen=datetime.utcnow(),
                    tags=['phishing', 'banking'],
                    description='Phishing site targeting banking customers'
                )
            ])
        
        return indicators
    
    async def _update_rss_feed(self, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Update RSS-based threat feed"""
        indicators = []
        
        # Simulate RSS feed parsing
        indicators.extend([
            ThreatIndicator(
                indicator_type='hash',
                value='a1b2c3d4e5f6789012345678901234567890abcd',
                confidence=0.8,
                severity='medium',
                source=feed.name,
                first_seen=datetime.utcnow() - timedelta(hours=1),
                last_seen=datetime.utcnow(),
                tags=['malware', 'trojan'],
                description='Malware hash from threat intelligence feed'
            )
        ])
        
        return indicators
    
    async def _update_text_feed(self, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Update text-based threat feed"""
        indicators = []
        
        # Simulate text feed parsing
        indicators.extend([
            ThreatIndicator(
                indicator_type='domain',
                value='malicious-domain.com',
                confidence=0.7,
                severity='medium',
                source=feed.name,
                first_seen=datetime.utcnow() - timedelta(minutes=30),
                last_seen=datetime.utcnow(),
                tags=['malware', 'domain'],
                description='Malicious domain from threat feed'
            )
        ])
        
        return indicators
    
    def search_indicators(self, query: str, indicator_type: str = None) -> List[ThreatIndicator]:
        """Search threat indicators"""
        results = []
        
        for indicator in self.indicators:
            if indicator_type and indicator.indicator_type != indicator_type:
                continue
            
            if (query.lower() in indicator.value.lower() or 
                query.lower() in indicator.description.lower() or
                any(query.lower() in tag.lower() for tag in indicator.tags)):
                results.append(indicator)
        
        return results
    
    def get_indicators_by_severity(self, severity: str) -> List[ThreatIndicator]:
        """Get indicators by severity level"""
        return [ind for ind in self.indicators if ind.severity == severity]
    
    def get_fresh_indicators(self, hours: int = 24) -> List[ThreatIndicator]:
        """Get indicators from the last N hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [ind for ind in self.indicators if ind.last_seen >= cutoff_time]

class ThreatIntelligenceAPI:
    """API for threat intelligence system"""
    
    def __init__(self):
        self.dark_web_monitor = DarkWebMonitor()
        self.phishing_intelligence = PhishingIntelligence()
        self.feed_manager = ThreatFeedManager()
        self.threat_database = {}
    
    async def update_threat_intelligence(self) -> Dict[str, Any]:
        """Update all threat intelligence sources"""
        print("🕵️ Updating Threat Intelligence...")
        
        results = {
            'dark_web': [],
            'phishing': [],
            'feeds': {},
            'total_indicators': 0,
            'update_time': datetime.utcnow().isoformat()
        }
        
        # Update dark web intelligence
        try:
            dark_web_threats = self.dark_web_monitor.search_dark_web_for_fraud(['upi fraud', 'payment fraud'])
            results['dark_web'] = dark_web_threats
        except Exception as e:
            results['dark_web_error'] = str(e)
        
        # Update phishing intelligence
        try:
            phishing_reports = [
                {
                    'content': 'URGENT: Your UPI account will be suspended. Click here to verify: http://fake-upi-verify.com',
                    'source': 'user_report',
                    'timestamp': datetime.utcnow()
                }
            ]
            phishing_indicators = self.phishing_intelligence.analyze_phishing_reports(phishing_reports)
            results['phishing'] = [asdict(ind) for ind in phishing_indicators]
        except Exception as e:
            results['phishing_error'] = str(e)
        
        # Update threat feeds
        try:
            feed_results = await self.feed_manager.update_all_feeds()
            results['feeds'] = feed_results
        except Exception as e:
            results['feeds_error'] = str(e)
        
        # Calculate total indicators
        results['total_indicators'] = len(self.feed_manager.indicators)
        
        return results
    
    def check_transaction_against_threats(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check transaction against known threats"""
        threats_found = []
        risk_score = 0.0
        
        # Check IP address
        if 'ip_address' in transaction_data:
            ip_indicators = self.feed_manager.search_indicators(transaction_data['ip_address'], 'ip')
            if ip_indicators:
                threats_found.extend(ip_indicators)
                risk_score += 0.3
        
        # Check merchant domain
        if 'merchant_domain' in transaction_data:
            domain_indicators = self.feed_manager.search_indicators(transaction_data['merchant_domain'], 'domain')
            if domain_indicators:
                threats_found.extend(domain_indicators)
                risk_score += 0.4
        
        # Check device fingerprint
        if 'device_fingerprint' in transaction_data:
            device_hash = hashlib.md5(str(transaction_data['device_fingerprint']).encode()).hexdigest()
            hash_indicators = self.feed_manager.search_indicators(device_hash, 'hash')
            if hash_indicators:
                threats_found.extend(hash_indicators)
                risk_score += 0.5
        
        # Check for phishing patterns
        if 'user_agent' in transaction_data:
            for pattern in self.phishing_intelligence.known_phishing_patterns:
                if re.search(pattern['pattern'], transaction_data['user_agent'], re.IGNORECASE):
                    risk_score += 0.2
                    threats_found.append({
                        'type': 'phishing_pattern',
                        'pattern': pattern['pattern'],
                        'confidence': pattern['confidence'],
                        'description': pattern['description']
                    })
        
        return {
            'threats_found': len(threats_found),
            'risk_score': min(risk_score, 1.0),
            'threat_details': [asdict(threat) if hasattr(threat, '__dict__') else threat for threat in threats_found],
            'threat_intelligence_active': True
        }
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of current threat intelligence"""
        indicators = self.feed_manager.indicators
        
        severity_counts = {}
        for severity in ['low', 'medium', 'high', 'critical']:
            severity_counts[severity] = len(self.feed_manager.get_indicators_by_severity(severity))
        
        fresh_indicators = self.feed_manager.get_fresh_indicators(24)
        
        return {
            'total_indicators': len(indicators),
            'fresh_indicators_24h': len(fresh_indicators),
            'severity_distribution': severity_counts,
            'indicator_types': {
                'ip': len([i for i in indicators if i.indicator_type == 'ip']),
                'domain': len([i for i in indicators if i.indicator_type == 'domain']),
                'url': len([i for i in indicators if i.indicator_type == 'url']),
                'email': len([i for i in indicators if i.indicator_type == 'email']),
                'hash': len([i for i in indicators if i.indicator_type == 'hash'])
            },
            'last_update': datetime.utcnow().isoformat()
        }
    
    def get_threat_alerts(self, severity_threshold: str = 'medium') -> List[Dict[str, Any]]:
        """Get recent threat alerts above severity threshold"""
        severity_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        threshold_level = severity_levels.get(severity_threshold, 2)
        
        alerts = []
        for indicator in self.feed_manager.indicators:
            if severity_levels.get(indicator.severity, 0) >= threshold_level:
                alerts.append({
                    'indicator': asdict(indicator),
                    'alert_time': datetime.utcnow().isoformat(),
                    'action_required': indicator.severity in ['high', 'critical']
                })
        
        # Sort by severity and recency
        alerts.sort(key=lambda x: (severity_levels.get(x['indicator']['severity'], 0), x['indicator']['last_seen']), reverse=True)
        
        return alerts[:50]  # Return top 50 alerts

# Example usage and testing
async def demonstrate_threat_intelligence():
    """Demonstrate threat intelligence system"""
    print("🕵️ Demonstrating Threat Intelligence System")
    print("=" * 60)
    
    # Initialize threat intelligence API
    ti_api = ThreatIntelligenceAPI()
    
    # Update threat intelligence
    print("1. Updating Threat Intelligence Sources...")
    update_results = await ti_api.update_threat_intelligence()
    
    print(f"   Dark web threats found: {len(update_results['dark_web'])}")
    print(f"   Phishing indicators: {len(update_results['phishing'])}")
    print(f"   Feed updates: {len(update_results['feeds'])}")
    print(f"   Total indicators: {update_results['total_indicators']}")
    
    # Get threat summary
    print("\n2. Threat Intelligence Summary...")
    summary = ti_api.get_threat_summary()
    print(f"   Total indicators: {summary['total_indicators']}")
    print(f"   Fresh indicators (24h): {summary['fresh_indicators_24h']}")
    print(f"   Severity distribution: {summary['severity_distribution']}")
    print(f"   Indicator types: {summary['indicator_types']}")
    
    # Check transaction against threats
    print("\n3. Checking Transaction Against Threats...")
    test_transaction = {
        'transaction_id': 'TXN_001',
        'ip_address': '192.168.1.100',  # Known malicious IP
        'merchant_domain': 'fake-bank-verification.com',  # Known phishing domain
        'device_fingerprint': 'suspicious_device_123',
        'user_agent': 'Mozilla/5.0 (compatible; UPI Fraud Bot)'
    }
    
    threat_check = ti_api.check_transaction_against_threats(test_transaction)
    print(f"   Threats found: {threat_check['threats_found']}")
    print(f"   Risk score: {threat_check['risk_score']:.3f}")
    print(f"   Threat intelligence active: {threat_check['threat_intelligence_active']}")
    
    # Get threat alerts
    print("\n4. Recent Threat Alerts...")
    alerts = ti_api.get_threat_alerts('medium')
    print(f"   High priority alerts: {len(alerts)}")
    
    for i, alert in enumerate(alerts[:3]):  # Show top 3 alerts
        indicator = alert['indicator']
        print(f"   Alert {i+1}: {indicator['indicator_type']} - {indicator['value']}")
        print(f"     Severity: {indicator['severity']}, Confidence: {indicator['confidence']}")
        print(f"     Description: {indicator['description']}")
    
    print("\n✅ Threat Intelligence demonstration completed!")
    
    return {
        'ti_api': ti_api,
        'update_results': update_results,
        'summary': summary,
        'threat_check': threat_check,
        'alerts': alerts
    }

if __name__ == "__main__":
    # Run demonstration
    import asyncio
    results = asyncio.run(demonstrate_threat_intelligence())
    
    print(f"\n📊 Summary:")
    print(f"   Threat intelligence sources updated")
    print(f"   Total indicators collected: {results['summary']['total_indicators']}")
    print(f"   Transaction threat check: {results['threat_check']['threats_found']} threats found")
    print(f"   High priority alerts: {len(results['alerts'])}")
