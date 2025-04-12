from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
import logging
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
import os
from enum import Enum

class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class HealthStatus:
    """Health status of a component"""
    status: str
    last_check: datetime
    details: Dict[str, Any]
    error: Optional[str] = None

class MonitoringConfig:
    """Monitoring configuration"""
    
    # Azure Application Insights
    APPINSIGHTS_CONNECTION_STRING = os.getenv("APPINSIGHTS_CONNECTION_STRING")
    
    # Alert thresholds
    MAX_ERROR_RATE = float(os.getenv("MAX_ERROR_RATE", "0.05"))  # 5%
    MAX_LATENCY_MS = float(os.getenv("MAX_LATENCY_MS", "1000"))  # 1 second
    MIN_SUCCESS_RATE = float(os.getenv("MIN_SUCCESS_RATE", "0.95"))  # 95%
    
    # Health check intervals (in seconds)
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))
    
    # Metrics retention (in days)
    METRICS_RETENTION_DAYS = int(os.getenv("METRICS_RETENTION_DAYS", "30"))
    
    # Alert notification endpoints
    TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL")
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
    EMAIL_RECIPIENTS = os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",")

class Monitoring:
    """Centralized monitoring system"""
    
    def __init__(self):
        self.tracer_provider = TracerProvider(
            resource=Resource.create({"service.name": "mcp-llm-service"})
        )
        self.meter_provider = MeterProvider()
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        # Metrics
        self.request_counter = self.meter.create_counter(
            "requests_total",
            description="Total number of requests"
        )
        self.latency_histogram = self.meter.create_histogram(
            "request_latency",
            description="Request latency in milliseconds"
        )
        self.token_counter = self.meter.create_counter(
            "tokens_total",
            description="Total number of tokens processed"
        )
        
        # Health status
        self._health_status: Dict[str, HealthStatus] = {}
        
        # Initialize Azure Monitor if configured
        if MonitoringConfig.APPINSIGHTS_CONNECTION_STRING:
            configure_azure_monitor(
                connection_string=MonitoringConfig.APPINSIGHTS_CONNECTION_STRING
            )
    
    async def track_request(
        self,
        model_id: str,
        success: bool,
        latency_ms: float,
        tokens: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track a model request"""
        # Update metrics
        self.request_counter.add(
            1,
            {"model_id": model_id, "success": str(success)}
        )
        self.latency_histogram.record(
            latency_ms,
            {"model_id": model_id}
        )
        self.token_counter.add(
            tokens,
            {"model_id": model_id}
        )
        
        # Check for alerts
        if not success or latency_ms > MonitoringConfig.MAX_LATENCY_MS:
            await self.trigger_alert(
                severity=AlertSeverity.WARNING if not success else AlertSeverity.INFO,
                message=f"Model {model_id} {'failed' if not success else 'high latency'}: {latency_ms}ms",
                details={
                    "model_id": model_id,
                    "latency_ms": latency_ms,
                    "tokens": tokens,
                    "metadata": metadata or {}
                }
            )
    
    async def update_health_status(
        self,
        component: str,
        status: str,
        details: Dict[str, Any],
        error: Optional[str] = None
    ) -> None:
        """Update health status of a component"""
        self._health_status[component] = HealthStatus(
            status=status,
            last_check=datetime.utcnow(),
            details=details,
            error=error
        )
        
        if error or status.lower() != "healthy":
            await self.trigger_alert(
                severity=AlertSeverity.ERROR if error else AlertSeverity.WARNING,
                message=f"Component {component} health check failed: {error or status}",
                details=details
            )
    
    async def trigger_alert(
        self,
        severity: AlertSeverity,
        message: str,
        details: Dict[str, Any]
    ) -> None:
        """Trigger an alert through configured channels"""
        alert_data = {
            "severity": severity,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        logging.log(log_level, f"Alert: {message}")
        
        # Send to Microsoft Teams
        if MonitoringConfig.TEAMS_WEBHOOK_URL:
            # Implement Teams webhook notification
            pass
        
        # Send to Slack
        if MonitoringConfig.SLACK_WEBHOOK_URL:
            # Implement Slack webhook notification
            pass
        
        # Send email
        if MonitoringConfig.EMAIL_RECIPIENTS:
            # Implement email notification
            pass
    
    async def get_model_metrics(
        self,
        model_id: str,
        time_window: timedelta
    ) -> Dict[str, Any]:
        """Get metrics for a specific model"""
        # Implement metric aggregation
        return {}
    
    async def run_health_checks(self) -> None:
        """Run periodic health checks"""
        while True:
            try:
                # Check database connectivity
                # Check model endpoints
                # Check Azure services
                # Check system resources
                await asyncio.sleep(MonitoringConfig.HEALTH_CHECK_INTERVAL)
            except Exception as e:
                logging.error(f"Health check failed: {e}")
                await asyncio.sleep(10)  # Backoff on error
    
    def get_health_status(self) -> Dict[str, HealthStatus]:
        """Get current health status of all components"""
        return self._health_status

# Global monitoring instance
monitoring = Monitoring() 