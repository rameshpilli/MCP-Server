# MCP LLM Service Setup and Configuration Guide

## Overview

The MCP (Model Control Platform) LLM Service is a flexible and enterprise-ready system for managing and monitoring multiple LLM models in a corporate environment. This guide covers setup, configuration, and best practices.

## Quick Start

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables (see Configuration section)
4. Initialize the database:
   ```bash
   alembic upgrade head
   ```
5. Start the service:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Configuration

### Environment Variables

#### Database Configuration
```bash
# Choose storage backend: "local" or "azure"
STORAGE_BACKEND=local

# Local SQLite (default)
# No additional configuration needed

# Azure SQL Database
AZURE_DB_URL=mssql+aioodbc://username:password@server.database.windows.net/mcp
```

#### Azure Integration
```bash
# Azure Storage
AZURE_STORAGE_ACCOUNT=your-storage-account
AZURE_CONTAINER_NAME=mcp-logs

# Azure Monitor
APPINSIGHTS_CONNECTION_STRING=your-connection-string
```

#### Monitoring and Alerts
```bash
# Alert Thresholds
MAX_ERROR_RATE=0.05
MAX_LATENCY_MS=1000
MIN_SUCCESS_RATE=0.95

# Health Checks
HEALTH_CHECK_INTERVAL=60

# Alert Notifications
TEAMS_WEBHOOK_URL=your-teams-webhook
SLACK_WEBHOOK_URL=your-slack-webhook
ALERT_EMAIL_RECIPIENTS=admin@company.com,team@company.com
```

### Model Configuration

Register a new model:
```python
from app.core.config import ServerConfig, ModelConfig, ModelBackend

# Configure model
model = ModelConfig(
    model_id="internal-model-v1",
    backend=ModelBackend.CUSTOM,
    api_base="http://internal-llm:8000",
    api_version="v2",
    timeout=60,
    additional_params={
        "model_type": "internal",
        "priority": "high"
    }
)

# Register model
await ServerConfig.register_model(model)
```

## Monitoring and Observability

### Metrics
- Request count
- Latency
- Token usage
- Success/failure rates
- Model-specific metrics

### Health Checks
- Database connectivity
- Model endpoint availability
- Azure service status
- System resources

### Alerts
Alerts are sent through configured channels when:
- Error rate exceeds threshold
- Latency is too high
- Success rate drops below threshold
- Component health check fails

### Azure Integration

#### Azure Monitor
- Request tracking
- Performance metrics
- Dependency tracking
- Exception logging

#### Azure Storage
Logs are stored in Azure Blob Storage with this structure:
```
mcp-logs/
  ├── model_registrations/
  │   └── YYYY/MM/DD/HH/MM_SS.json
  ├── model_usage/
  │   └── YYYY/MM/DD/HH/MM_SS.json
  └── errors/
      └── YYYY/MM/DD/HH/MM_SS.json
```

## Database Management

### Migrations
Create a new migration:
```bash
alembic revision --autogenerate -m "description"
```

Apply migrations:
```bash
alembic upgrade head
```

Rollback migration:
```bash
alembic downgrade -1
```

### Backup and Restore
For Azure SQL Database:
```bash
# Automated backups are configured through Azure Portal
# Point-in-time restore available through Azure Portal
```

For local SQLite:
```bash
# Backup
cp mcp.db mcp.db.backup

# Restore
cp mcp.db.backup mcp.db
```

## Security Best Practices

1. **API Keys**
   - Store in Azure Key Vault
   - Rotate regularly
   - Use different keys per environment

2. **Database Access**
   - Use managed identities in Azure
   - Minimal privilege principle
   - Regular access review

3. **Monitoring**
   - Encrypt sensitive logs
   - Audit access to monitoring data
   - Regular security scanning

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   ```python
   # Check connection
   async with db.get_session() as session:
       await session.execute("SELECT 1")
   ```

2. **Model Endpoint Issues**
   ```python
   # Check model status
   status = await ServerConfig.get_model("model-id").check_health()
   ```

3. **Azure Storage Issues**
   ```python
   # Verify blob storage access
   blob_client = await db.get_blob_client()
   containers = await blob_client.list_containers()
   ```

### Logging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
```

View logs:
```bash
# Local logs
tail -f logs/mcp.log

# Azure logs
az monitor log-analytics query -w WORKSPACE_ID --query "MCPLogs | where TimeGenerated > ago(1h)"
```

## Performance Tuning

### Database
- Index optimization
- Connection pooling
- Query optimization

### Model Requests
- Request batching
- Caching responses
- Load balancing

### Azure Services
- Geographic distribution
- Resource scaling
- Cost optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests:
   ```bash
   pytest tests/
   ```
5. Submit pull request

## Support

For issues and support:
- Create GitHub issue
- Contact support team
- Check documentation 