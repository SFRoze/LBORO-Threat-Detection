"""
Alert Management System for Threat Detection

This module handles alert generation and notification for detected threats.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class AlertManager:
    """
    Manages alert generation and notification for threat detection events.
    """
    
    def __init__(self, alert_config: Dict):
        """
        Initialize the alert manager with configuration.
        
        Args:
            alert_config: Alert configuration dictionary
        """
        self.config = alert_config
        self.logger = logging.getLogger(__name__)
        
        # Alert history to prevent spam
        self.alert_history = []
        self.last_alert_time = {}
        
        # Alert cooldown period (seconds)
        self.cooldown_period = 5.0
        
        self.logger.info("AlertManager initialized")
    
    def process_detections(self, detections: Dict) -> List[Dict]:
        """
        Process detection results and generate alerts as needed.
        
        Args:
            detections: Detection results dictionary
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Check for person detection alerts
        if self.config.get('person_detection', False):
            alerts.extend(self._check_person_alerts(detections))
        
        # Check for threat object alerts
        if self.config.get('weapon_detection', False):
            alerts.extend(self._check_threat_alerts(detections))
        
        # Check for multiple persons alert
        if self.config.get('multiple_persons', False):
            alerts.extend(self._check_multiple_persons_alert(detections))
        
        # Process and send alerts
        for alert in alerts:
            self._send_alert(alert)
        
        # Update detection results with alerts
        detections['alerts'] = alerts
        
        return alerts
    
    def _check_person_alerts(self, detections: Dict) -> List[Dict]:
        """Check for person detection alerts."""
        alerts = []
        
        person_count = detections['counts'].get('person', 0)
        if person_count > 0:
            # Check confidence threshold
            high_confidence_persons = [
                obj for obj in detections['objects'] 
                if obj['class'] == 'person' and 
                obj['confidence'] >= self.config.get('confidence_minimum', 0.7)
            ]
            
            if high_confidence_persons:
                alert = {
                    'type': 'person_detection',
                    'severity': 'medium',
                    'message': f"Detected {len(high_confidence_persons)} person(s)",
                    'timestamp': datetime.now().isoformat(),
                    'source': detections['source'],
                    'details': {
                        'person_count': len(high_confidence_persons),
                        'max_confidence': max(p['confidence'] for p in high_confidence_persons),
                        'locations': [(p['center'][0], p['center'][1]) for p in high_confidence_persons]
                    }
                }
                
                # Check cooldown
                if self._check_alert_cooldown('person_detection'):
                    alerts.append(alert)
        
        return alerts
    
    def _check_threat_alerts(self, detections: Dict) -> List[Dict]:
        """Check for threat object detection alerts."""
        alerts = []
        
        # Define threat classes (these would be from your custom model)
        threat_classes = ['weapon', 'suspicious_package', 'abandoned_object']
        
        for obj in detections['objects']:
            if obj['class'] in threat_classes and obj['confidence'] >= self.config.get('confidence_minimum', 0.7):
                alert = {
                    'type': 'threat_detection',
                    'severity': 'high',
                    'message': f"THREAT DETECTED: {obj['class'].upper()}",
                    'timestamp': datetime.now().isoformat(),
                    'source': detections['source'],
                    'details': {
                        'threat_type': obj['class'],
                        'confidence': obj['confidence'],
                        'location': obj['center'],
                        'bbox': obj['bbox']
                    }
                }
                
                # Threat alerts always have shorter cooldown
                if self._check_alert_cooldown(f"threat_{obj['class']}", cooldown=2.0):
                    alerts.append(alert)
        
        return alerts
    
    def _check_multiple_persons_alert(self, detections: Dict) -> List[Dict]:
        """Check for multiple persons alert."""
        alerts = []
        
        person_threshold = self.config.get('person_threshold', 3)
        person_count = detections['counts'].get('person', 0)
        
        if person_count >= person_threshold:
            alert = {
                'type': 'multiple_persons',
                'severity': 'medium',
                'message': f"Multiple persons detected: {person_count} (threshold: {person_threshold})",
                'timestamp': datetime.now().isoformat(),
                'source': detections['source'],
                'details': {
                    'person_count': person_count,
                    'threshold': person_threshold
                }
            }
            
            if self._check_alert_cooldown('multiple_persons'):
                alerts.append(alert)
        
        return alerts
    
    def _check_alert_cooldown(self, alert_type: str, cooldown: Optional[float] = None) -> bool:
        """
        Check if enough time has passed since the last alert of this type.
        
        Args:
            alert_type: Type of alert to check
            cooldown: Cooldown period in seconds (uses default if None)
            
        Returns:
            True if alert should be sent, False if in cooldown period
        """
        if cooldown is None:
            cooldown = self.cooldown_period
        
        now = datetime.now()
        last_alert = self.last_alert_time.get(alert_type)
        
        if last_alert is None or (now - last_alert).total_seconds() >= cooldown:
            self.last_alert_time[alert_type] = now
            return True
        
        return False
    
    def _send_alert(self, alert: Dict):
        """
        Send an alert through configured channels.
        
        Args:
            alert: Alert dictionary to send
        """
        # Console output
        if self.config.get('console_output', True):
            self._send_console_alert(alert)
        
        # Log file
        if self.config.get('log_file', True):
            self._send_log_alert(alert)
        
        # Email alerts
        if self.config.get('email_alerts', False):
            self._send_email_alert(alert)
        
        # Webhook alerts
        if self.config.get('webhook_alerts', False):
            self._send_webhook_alert(alert)
        
        # Add to alert history
        self.alert_history.append(alert)
        
        # Limit history size
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-500:]
    
    def _send_console_alert(self, alert: Dict):
        """Send alert to console."""
        severity_colors = {
            'low': '\033[92m',      # Green
            'medium': '\033[93m',   # Yellow
            'high': '\033[91m',     # Red
            'critical': '\033[95m'  # Magenta
        }
        
        reset_color = '\033[0m'
        color = severity_colors.get(alert['severity'], '')
        
        print(f"\n{color}🚨 ALERT [{alert['severity'].upper()}]{reset_color}")
        print(f"Type: {alert['type']}")
        print(f"Message: {alert['message']}")
        print(f"Time: {alert['timestamp']}")
        print(f"Source: {alert['source']}")
        
        if alert.get('details'):
            print("Details:")
            for key, value in alert['details'].items():
                print(f"  {key}: {value}")
        print("-" * 50)
    
    def _send_log_alert(self, alert: Dict):
        """Send alert to log file."""
        log_message = f"ALERT [{alert['severity'].upper()}] {alert['type']}: {alert['message']}"
        
        if alert['severity'] == 'high' or alert['severity'] == 'critical':
            self.logger.error(log_message)
        elif alert['severity'] == 'medium':
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log details if available
        if alert.get('details'):
            self.logger.debug(f"Alert details: {json.dumps(alert['details'])}")
    
    def _send_email_alert(self, alert: Dict):
        """Send alert via email (placeholder implementation)."""
        try:
            # Email configuration would come from environment variables or config
            smtp_server = self.config.get('smtp_server', 'localhost')
            smtp_port = self.config.get('smtp_port', 587)
            email_user = self.config.get('email_user')
            email_password = self.config.get('email_password')
            recipient_emails = self.config.get('recipient_emails', [])
            
            if not email_user or not recipient_emails:
                self.logger.warning("Email configuration incomplete, skipping email alert")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_user
            msg['To'] = ', '.join(recipient_emails)
            msg['Subject'] = f"Threat Detection Alert - {alert['type']}"
            
            # Email body
            body = f\"\"\"
THREAT DETECTION ALERT

Type: {alert['type']}
Severity: {alert['severity'].upper()}
Message: {alert['message']}
Time: {alert['timestamp']}
Source: {alert['source']}

Details:
{json.dumps(alert.get('details', {}), indent=2)}

This is an automated alert from the Threat Detection System.
\"\"\"\n            \n            msg.attach(MIMEText(body, 'plain'))\n            \n            # Send email\n            with smtplib.SMTP(smtp_server, smtp_port) as server:\n                server.starttls()\n                server.login(email_user, email_password)\n                server.send_message(msg)\n            \n            self.logger.info(f\"Email alert sent for {alert['type']}\")\n            \n        except Exception as e:\n            self.logger.error(f\"Failed to send email alert: {e}\")\n    \n    def _send_webhook_alert(self, alert: Dict):\n        \"\"\"Send alert via webhook.\"\"\"\n        try:\n            webhook_url = self.config.get('webhook_url')\n            if not webhook_url:\n                self.logger.warning(\"Webhook URL not configured, skipping webhook alert\")\n                return\n            \n            # Prepare webhook payload\n            payload = {\n                'alert_type': alert['type'],\n                'severity': alert['severity'],\n                'message': alert['message'],\n                'timestamp': alert['timestamp'],\n                'source': alert['source'],\n                'details': alert.get('details', {})\n            }\n            \n            # Send webhook\n            response = requests.post(\n                webhook_url,\n                json=payload,\n                timeout=10,\n                headers={'Content-Type': 'application/json'}\n            )\n            \n            if response.status_code == 200:\n                self.logger.info(f\"Webhook alert sent for {alert['type']}\")\n            else:\n                self.logger.warning(f\"Webhook alert failed with status {response.status_code}\")\n            \n        except Exception as e:\n            self.logger.error(f\"Failed to send webhook alert: {e}\")\n    \n    def get_alert_history(self, limit: int = 100) -> List[Dict]:\n        \"\"\"Get recent alert history.\"\"\"\n        return self.alert_history[-limit:]\n    \n    def clear_alert_history(self):\n        \"\"\"Clear alert history.\"\"\"\n        self.alert_history.clear()\n        self.logger.info(\"Alert history cleared\")\n    \n    def get_alert_stats(self) -> Dict:\n        \"\"\"Get alert statistics.\"\"\"\n        if not self.alert_history:\n            return {\n                'total_alerts': 0,\n                'by_type': {},\n                'by_severity': {},\n                'last_24h': 0\n            }\n        \n        # Count by type\n        by_type = {}\n        by_severity = {}\n        \n        # Count alerts in last 24 hours\n        now = datetime.now()\n        last_24h = 0\n        \n        for alert in self.alert_history:\n            # Count by type\n            alert_type = alert['type']\n            by_type[alert_type] = by_type.get(alert_type, 0) + 1\n            \n            # Count by severity\n            severity = alert['severity']\n            by_severity[severity] = by_severity.get(severity, 0) + 1\n            \n            # Count last 24h (parse timestamp)\n            try:\n                alert_time = datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00'))\n                if (now - alert_time).total_seconds() <= 86400:  # 24 hours\n                    last_24h += 1\n            except:\n                pass\n        \n        return {\n            'total_alerts': len(self.alert_history),\n            'by_type': by_type,\n            'by_severity': by_severity,\n            'last_24h': last_24h\n        }\n\n\nif __name__ == \"__main__\":\n    # Test the alert manager\n    config = {\n        'person_detection': True,\n        'console_output': True,\n        'log_file': True,\n        'confidence_minimum': 0.7\n    }\n    \n    alert_manager = AlertManager(config)\n    \n    # Test detection data\n    test_detections = {\n        'source': 'test_image.jpg',\n        'objects': [\n            {\n                'class': 'person',\n                'confidence': 0.85,\n                'center': [320, 240],\n                'bbox': [200, 100, 440, 380]\n            }\n        ],\n        'counts': {'person': 1}\n    }\n    \n    alerts = alert_manager.process_detections(test_detections)\n    print(f\"Generated {len(alerts)} alerts\")"
