# Snowflake Integration Guide

## Overview

The MCP platform provides seamless integration with Snowflake, allowing you to:
- Access training data directly from your data warehouse
- Store model metrics and predictions
- Implement feature stores
- Track model performance

## Setup

### 1. Prerequisites

1. Snowflake account with appropriate privileges
2. Key pair authentication setup
3. Warehouse, database, and schema access

### 2. Configuration

#### Environment Variables

Set up the following environment variables:

```bash
export SNOWFLAKE_ACCOUNT="your_account"
export SNOWFLAKE_WAREHOUSE="COMPUTE_WH"
export SNOWFLAKE_DATABASE="ML_DATA"
export SNOWFLAKE_SCHEMA="PUBLIC"
export SNOWFLAKE_ROLE="ML_ENGINEER"
export SNOWFLAKE_USER="your_user"
export SNOWFLAKE_PRIVATE_KEY_PATH="/path/to/rsa_key.p8"
```

#### Configuration File

In your `config.yaml`:

```yaml
snowflake:
  enabled: true
  account: ${SNOWFLAKE_ACCOUNT}
  warehouse: ${SNOWFLAKE_WAREHOUSE}
  database: ${SNOWFLAKE_DATABASE}
  schema: ${SNOWFLAKE_SCHEMA}
  role: ${SNOWFLAKE_ROLE}
  user: ${SNOWFLAKE_USER}
  private_key_path: ${SNOWFLAKE_PRIVATE_KEY_PATH}
  query_timeout: 600
  connection_timeout: 30
```

### 3. Usage Examples

#### Python SDK

```python
from mcp_client import MCPClient, SnowflakeConfig, ModelConfig

# Configure Snowflake
snowflake_config = SnowflakeConfig(
    account="your_account",
    warehouse="COMPUTE_WH",
    database="ML_DATA",
    schema="PUBLIC",
    role="ML_ENGINEER"
)

# Initialize client
client = MCPClient(
    base_url="https://mcp.yourorg.com",
    api_key="your-api-key",
    snowflake_config=snowflake_config
)

# Get training data
training_data = client.get_training_data(
    query="SELECT * FROM training_data WHERE model_version = 'v1'"
)

# Register model with Snowflake data source
model_config = ModelConfig(
    model_id="my-model",
    training_data_source={
        "type": "snowflake",
        "query": "SELECT * FROM training_data",
        "database": "ML_DATA",
        "schema": "PUBLIC"
    }
)

client.register_model(model_config)
```

#### CLI Usage

```bash
# Register model with Snowflake configuration
mcp register model_config.yaml --snowflake-config snowflake_config.yaml

# Query training data
mcp query-data "SELECT * FROM training_data LIMIT 10"

# Log predictions
mcp log-predictions my-model --data-source snowflake --table predictions
```

### 4. Best Practices

1. **Security**:
   - Use key pair authentication
   - Implement role-based access control
   - Use secure views for sensitive data
   - Enable network policies

2. **Performance**:
   - Use appropriate warehouse sizes
   - Implement efficient queries
   - Cache frequently used data
   - Use materialized views when appropriate

3. **Data Management**:
   - Version your training data
   - Implement data quality checks
   - Use time travel for data recovery
   - Maintain data lineage

### 5. Common Tasks

#### Feature Store Setup

```sql
-- Create a feature store
CREATE DATABASE feature_store;
CREATE SCHEMA feature_store.ml_features;

-- Create a feature table
CREATE TABLE feature_store.ml_features.customer_features (
    customer_id VARCHAR,
    feature_timestamp TIMESTAMP,
    feature_1 FLOAT,
    feature_2 FLOAT,
    -- more features...
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);
```

#### Model Metrics Tracking

```sql
-- Create a metrics table
CREATE TABLE model_metrics (
    model_id VARCHAR,
    version VARCHAR,
    timestamp TIMESTAMP,
    metric_name VARCHAR,
    metric_value FLOAT,
    metadata VARIANT
);

-- Log metrics
INSERT INTO model_metrics 
VALUES ('my-model', 'v1', CURRENT_TIMESTAMP(), 'accuracy', 0.95, 
    PARSE_JSON('{"training_size": 10000}'));
```

### 6. Troubleshooting

Common issues and solutions:

1. **Connection Issues**:
   - Verify network connectivity
   - Check account URL format
   - Validate key pair permissions
   - Confirm role privileges

2. **Performance Issues**:
   - Monitor warehouse sizing
   - Check query optimization
   - Review concurrent connections
   - Analyze data clustering

3. **Data Access Issues**:
   - Verify schema permissions
   - Check role assignments
   - Review secure views
   - Validate row access policies

### 7. Monitoring

Monitor your Snowflake integration:

```python
# Get Snowflake metrics
metrics = client.get_snowflake_metrics()

# Monitor warehouse usage
warehouse_metrics = client.get_warehouse_metrics()

# Check query performance
query_history = client.get_query_history()
```

### 8. Support

For Snowflake-specific issues:
1. Check Snowflake documentation
2. Review query history
3. Contact MCP support team
4. Consult Snowflake support 