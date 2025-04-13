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
from app.core.config import ServerConfig
import prometheus_client
import json
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from prometheus_client import Counter, Histogram, Gauge

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

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = prometheus_client.Counter(
    'mcp_request_total',
    'Total requests processed',
    ['model_id', 'status']
)

LATENCY = prometheus_client.Histogram(
    'mcp_request_latency_seconds',
    'Request latency in seconds',
    ['model_id']
)

TOKEN_USAGE = prometheus_client.Counter(
    'mcp_token_usage_total',
    'Total tokens used',
    ['model_id']
)

QUERY_DURATION = Histogram('query_duration_seconds', 'Query execution time in seconds')
QUERY_ERRORS = Counter('query_errors_total', 'Total query errors')
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses')
ACTIVE_QUERIES = Gauge('active_queries', 'Number of currently executing queries')

class Alert:
    severity: str
    title: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]

class AlertManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_channels = {
            "email": self._send_email_alert,
            "slack": self._send_slack_alert,
            "pagerduty": self._send_pagerduty_alert
        }
        self.logger = logging.getLogger(__name__)

    async def send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        channels = self.config.get("alert_channels", ["email"])
        
        for channel in channels:
            if channel in self.alert_channels:
                try:
                    await self.alert_channels[channel](alert)
                except Exception as e:
                    self.logger.error(f"Failed to send alert via {channel}: {str(e)}")

    async def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        email_config = self.config.get("email", {})
        if not email_config:
            return

        msg = MIMEMultipart()
        msg["From"] = email_config["from"]
        msg["To"] = email_config["to"]
        msg["Subject"] = f"[{alert.severity.upper()}] {alert.title}"

        body = f"""
        Alert: {alert.title}
        Severity: {alert.severity}
        Time: {alert.timestamp}
        
        {alert.message}
        
        Additional Information:
        {json.dumps(alert.metadata, indent=2)}
        """

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(email_config["smtp_host"], email_config["smtp_port"]) as server:
            if email_config.get("use_tls"):
                server.starttls()
            if email_config.get("username"):
                server.login(email_config["username"], email_config["password"])
            server.send_message(msg)

    async def _send_slack_alert(self, alert: Alert):
        """Send alert via Slack"""
        slack_config = self.config.get("slack", {})
        if not slack_config or "webhook_url" not in slack_config:
            return

        message = {
            "text": f"*[{alert.severity.upper()}] {alert.title}*\n{alert.message}",
            "attachments": [{
                "fields": [
                    {"title": k, "value": str(v), "short": True}
                    for k, v in alert.metadata.items()
                ]
            }]
        }

        async with aiohttp.ClientSession() as session:
            await session.post(slack_config["webhook_url"], json=message)

    async def _send_pagerduty_alert(self, alert: Alert):
        """Send alert via PagerDuty"""
        pd_config = self.config.get("pagerduty", {})
        if not pd_config or "api_key" not in pd_config:
            return

        payload = {
            "routing_key": pd_config["api_key"],
            "event_action": "trigger",
            "payload": {
                "summary": alert.title,
                "severity": alert.severity,
                "source": "MCP",
                "custom_details": alert.metadata
            }
        }

        async with aiohttp.ClientSession() as session:
            await session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

class MonitoringSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_manager = AlertManager(config)
        self.thresholds = config.get("thresholds", {
            "slow_query": 5.0,  # seconds
            "error_rate": 0.1,  # 10%
            "cache_miss_rate": 0.4,  # 40%
            "concurrent_queries": 100
        })

    async def monitor_query_performance(self, query_stats: Dict[str, Any]):
        """Monitor query performance and send alerts if needed"""
        # Track query duration
        duration = query_stats["execution_time"]
        QUERY_DURATION.observe(duration)

        if duration > self.thresholds["slow_query"]:
            await self.alert_manager.send_alert(Alert(
                severity="warning",
                title="Slow Query Detected",
                message=f"Query took {duration:.2f} seconds to execute",
                timestamp=datetime.utcnow(),
                metadata=query_stats
            ))

    async def monitor_error_rates(self, window_minutes: int = 5):
        """Monitor error rates over time window"""
        while True:
            error_rate = QUERY_ERRORS._value / max(QUERY_DURATION._count, 1)
            
            if error_rate > self.thresholds["error_rate"]:
                await self.alert_manager.send_alert(Alert(
                    severity="error",
                    title="High Query Error Rate",
                    message=f"Error rate of {error_rate:.2%} exceeds threshold",
                    timestamp=datetime.utcnow(),
                    metadata={"error_rate": error_rate}
                ))
            
            await asyncio.sleep(window_minutes * 60)

    async def monitor_cache_performance(self, window_minutes: int = 5):
        """Monitor cache hit rates"""
        while True:
            total = CACHE_HITS._value + CACHE_MISSES._value
            if total > 0:
                miss_rate = CACHE_MISSES._value / total
                if miss_rate > self.thresholds["cache_miss_rate"]:
                    await self.alert_manager.send_alert(Alert(
                        severity="warning",
                        title="High Cache Miss Rate",
                        message=f"Cache miss rate of {miss_rate:.2%} exceeds threshold",
                        timestamp=datetime.utcnow(),
                        metadata={"miss_rate": miss_rate}
                    ))
            
            await asyncio.sleep(window_minutes * 60)

    async def monitor_concurrent_queries(self):
        """Monitor number of concurrent queries"""
        while True:
            concurrent = ACTIVE_QUERIES._value
            if concurrent > self.thresholds["concurrent_queries"]:
                await self.alert_manager.send_alert(Alert(
                    severity="critical",
                    title="High Concurrent Query Load",
                    message=f"Current concurrent queries: {concurrent}",
                    timestamp=datetime.utcnow(),
                    metadata={"concurrent_queries": concurrent}
                ))
            
            await asyncio.sleep(60)  # Check every minute

    async def start_monitoring(self):
        """Start all monitoring tasks"""
        monitoring_tasks = [
            self.monitor_error_rates(),
            self.monitor_cache_performance(),
            self.monitor_concurrent_queries()
        ]
        
        await asyncio.gather(*monitoring_tasks)

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

    @staticmethod
    async def log_request(
        model_id: str,
        success: bool,
        latency: float,
        tokens: int,
        metadata: Dict[str, Any] = None
    ):
        """Log a model request with metrics."""
        status = "success" if success else "failure"
        
        # Update Prometheus metrics
        REQUEST_COUNT.labels(model_id=model_id, status=status).inc()
        LATENCY.labels(model_id=model_id).observe(latency)
        TOKEN_USAGE.labels(model_id=model_id).inc(tokens)
        
        # Log to file
        logger.info(
            f"Model request - ID: {model_id}, Status: {status}, "
            f"Latency: {latency:.3f}s, Tokens: {tokens}",
            extra={"metadata": metadata}
        )

    @staticmethod
    def start_metrics_server():
        """Start the Prometheus metrics server if enabled."""
        if ServerConfig.ENABLE_METRICS:
            prometheus_client.start_http_server(ServerConfig.METRICS_PORT)
            logger.info(f"Metrics server started on port {ServerConfig.METRICS_PORT}")

# Global monitoring instance
monitoring = Monitoring()

# Start metrics server on module import if enabled
if ServerConfig.ENABLE_METRICS:
    Monitoring.start_metrics_server() 