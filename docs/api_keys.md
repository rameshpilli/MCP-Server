# API Key Management

## Overview
The MCP (Model Context Protocol) uses API keys for authentication. Each key is associated with an owner and can have specific permissions and expiration dates.

## Getting an API Key

1. Request an API key from your MCP administrator, who will create one using:
```bash
curl -X POST "https://your-mcp-server/api/keys" \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "your.email@company.com",
    "expires_in_days": 90,
    "permissions": ["read", "write"]
  }'
```

2. Store the returned API key securely. It will look something like:
```
mcp_abcdef123456...
```

## Using Your API Key

1. Set up environment variables:
```bash
export MCP_SERVER_URL="https://your-mcp-server"
export MCP_API_KEY="your-api-key"
```

2. Use the key in API requests:
```bash
# Using curl
curl -X POST "${MCP_SERVER_URL}/models/register" \
  -H "X-API-Key: ${MCP_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "my-model",
    "config": {
      "backend": "custom",
      "api_base": "xhttp://my-model-endpoint"
    }
  }'

# Using Python SDK
from mcp_client import MCPClient

client = MCPClient(
    server_url=os.getenv("MCP_SERVER_URL"),
    api_key=os.getenv("MCP_API_KEY")
)

# Register a model
await client.register_model(
    model_id="my-model",
    config={
        "backend": "custom",
        "api_base": "http://my-model-endpoint"
    }
)
```

## API Key Information

Check your API key information:
```bash
curl "${MCP_SERVER_URL}/api/keys/info" \
  -H "X-API-Key: ${MCP_API_KEY}"
```

Response will include:
```json
{
  "owner": "your.email@company.com",
  "created_at": "2024-04-12T18:00:00Z",
  "expires_at": "2024-07-11T18:00:00Z",
  "is_active": true,
  "permissions": ["read", "write"],
  "last_used": "2024-04-12T19:30:00Z",
  "usage_count": 42
}
```

## Security Best Practices

1. **Never share your API key**
   - Each user should have their own key
   - Keys are tied to individual users for accountability

2. **Store securely**
   - Use environment variables or secure vaults
   - Never commit API keys to code repositories
   - Don't include in logs or error messages

3. **Regular rotation**
   - Request a new key before the current one expires
   - Immediately request key revocation if compromised

4. **Monitor usage**
   - Regularly check key usage statistics
   - Watch for unusual patterns
   - Report suspicious activity

## Troubleshooting

1. **Invalid API Key Error (401)**
   - Check if the key is correctly set in your environment
   - Verify the key hasn't expired
   - Ensure you're using the correct key format

2. **Permission Denied (403)**
   - Verify your key has the required permissions
   - Check if you're trying to access restricted endpoints

3. **Key Not Working**
   - Ensure you're including the `X-API-Key` header
   - Check if the key is properly formatted
   - Verify the server URL is correct

## Support

If you experience issues with your API key:

1. Check the key information endpoint for status
2. Contact your MCP administrator
3. Have your key owner and usage details ready
4. Never send your actual API key in support requests 