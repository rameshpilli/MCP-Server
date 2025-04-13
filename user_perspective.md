# Using the Model Context Protocol (MCP)

This guide walks you through the process of registering and using your machine learning models with MCP.

## Quick Start

1. [Get API Key](#getting-an-api-key)
2. [Register Your Model](#registering-your-model)
3. [Make Predictions](#making-predictions)
4. [Monitor Usage](#monitoring-usage)

## Getting an API Key

1. Request an API key:
```bash
curl -X POST http://localhost:8000/api/keys \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "your.email@example.com",
    "description": "Model deployment key"
  }'
```

Response:
```json
{
  "api_key": "mcp_xxxxx...",
  "expires_at": "2025-03-14T00:00:00Z",
  "permissions": ["read", "write", "execute"],
  "rate_limit": "100/minute"
}
```

⚠️ **Important**: Store your API key securely. It cannot be retrieved later.

## Registering Your Model

1. Prepare your model metadata:
```json
{
  "name": "sentiment-analyzer",
  "version": "1.0.0",
  "description": "BERT-based sentiment analysis model",
  "input_schema": {
    "type": "object",
    "properties": {
      "text": {"type": "string"}
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
      "confidence": {"type": "number"}
    }
  }
}
```

2. Register your model:
```bash
curl -X POST http://localhost:8000/api/models \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d @model_metadata.json
```

Response:
```json
{
  "model_id": "mod_xxxxx",
  "status": "registered",
  "endpoint": "/api/models/mod_xxxxx/predict"
}
```

## Making Predictions

1. Basic prediction request:
```bash
curl -X POST http://localhost:8000/api/models/mod_xxxxx/predict \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product is amazing!"
  }'
```

Response:
```json
{
  "sentiment": "positive",
  "confidence": 0.95
}
```

2. Batch prediction request:
```bash
curl -X POST http://localhost:8000/api/models/mod_xxxxx/predict/batch \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"text": "This product is amazing!"},
      {"text": "Not satisfied with the quality."}
    ]
  }'
```

Response:
```json
{
  "predictions": [
    {"sentiment": "positive", "confidence": 0.95},
    {"sentiment": "negative", "confidence": 0.87}
  ]
}
```

## Monitoring Usage

1. Check model status:
```bash
curl http://localhost:8000/api/models/mod_xxxxx/status \
  -H "X-API-Key: your_api_key"
```

Response:
```json
{
  "status": "active",
  "total_requests": 150,
  "average_latency": "45ms",
  "last_error": null
}
```

2. View usage metrics:
```bash
curl http://localhost:8000/api/models/mod_xxxxx/metrics \
  -H "X-API-Key: your_api_key"
```

Response:
```json
{
  "daily_requests": {
    "2024-03-14": 150,
    "2024-03-13": 123
  },
  "error_rate": "0.1%",
  "average_response_time": "45ms"
}
```

## Rate Limits and Quotas

- Default rate limit: 100 requests per minute
- Batch requests count as multiple requests based on batch size
- Rate limit headers in responses:
  ```
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 95
  X-RateLimit-Reset: 1623456789
  ```

## Error Handling

Common error responses:

1. Rate limit exceeded:
```json
{
  "error": "rate_limit_exceeded",
  "detail": "Too many requests. Try again in 35 seconds.",
  "reset_time": 1623456789
}
```

2. Invalid input:
```json
{
  "error": "validation_error",
  "detail": "Required field 'text' is missing"
}
```

3. Model error:
```json
{
  "error": "model_error",
  "detail": "Model failed to process input",
  "model_id": "mod_xxxxx"
}
```

## Best Practices

1. **API Key Management**:
   - Use different API keys for development and production
   - Rotate keys periodically
   - Never share or commit API keys

2. **Error Handling**:
   - Implement exponential backoff for rate limits
   - Handle all error cases in your code
   - Log errors for debugging

3. **Performance**:
   - Use batch predictions when possible
   - Monitor response times
   - Stay within rate limits

4. **Security**:
   - Use HTTPS in production
   - Validate input data
   - Keep API keys secure

## Example Integration (Python)

```python
import requests

class MCPClient:
    def __init__(self, api_key, base_url="http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def predict(self, model_id, text):
        url = f"{self.base_url}/api/models/{model_id}/predict"
        response = requests.post(
            url,
            headers=self.headers,
            json={"text": text}
        )
        return response.json()

# Usage
client = MCPClient("your_api_key")
result = client.predict("mod_xxxxx", "This product is amazing!")
print(result)  # {"sentiment": "positive", "confidence": 0.95}
```

## Troubleshooting

1. **API Key Issues**:
   - Ensure key is included in X-API-Key header
   - Check key hasn't expired
   - Verify permissions are correct

2. **Rate Limiting**:
   - Monitor X-RateLimit-* headers
   - Implement backoff strategy
   - Consider upgrading quota if needed

3. **Model Errors**:
   - Validate input matches schema
   - Check model status
   - Review error logs

Need help? Contact support at -