# Using the Model Context Protocol (MCP)

This guide walks you through the process of integrating your machine learning models with MCP to access our enterprise data sources.

## What is MCP?

MCP is a platform that allows you to:
1. Connect your custom ML models to our enterprise data sources
2. Access structured data from various sources (internal databases, Azure Storage, Snowflake, etc.)
3. Run predictions on this data using your models
4. Manage access control and usage tracking

### Available Data Sources

MCP provides access to various enterprise data sources:

1. **Job Market Data**:
   - Historical job postings
   - Salary trends
   - Industry-specific metrics
   - Geographic distribution
   ```json
   {
     "data_source": "jobs_db",
     "access_level": "full",
     "update_frequency": "daily",
     "schema": {
       "job_id": "string",
       "title": "string",
       "company": "string",
       "location": "string",
       "salary_range": "object",
       "requirements": "array",
       "posted_date": "datetime"
     }
   }
   ```

2. **Homebuilders Data**:
   - Property listings
   - Construction metrics
   - Market trends
   - Builder profiles
   ```json
   {
     "data_source": "homebuilders_db",
     "access_level": "read",
     "update_frequency": "weekly",
     "schema": {
       "property_id": "string",
       "builder_id": "string",
       "location": "object",
       "specs": "object",
       "pricing": "object",
       "construction_status": "string"
     }
   }
   ```

3. **Azure Storage Sources**:
   - Raw data files
   - Processed datasets
   - Historical records
   ```json
   {
     "data_source": "azure_storage",
     "container": "enterprise_data",
     "access_level": "read",
     "available_formats": ["parquet", "csv", "json"]
   }
   ```

4. **Snowflake Tables**:
   - Analytics-ready datasets
   - Aggregated metrics
   - Cross-referenced data
   ```json
   {
     "data_source": "snowflake",
     "warehouse": "ENTERPRISE_WH",
     "schemas": ["JOBS", "PROPERTIES", "MARKET_ANALYTICS"]
   }
   ```

### Example: Registering a Model with Data Source Access

When registering your model, specify which data sources it needs access to:

```bash
curl -X POST http://localhost:8000/api/models \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "job-market-predictor",
    "version": "1.0.0",
    "description": "Predicts job market trends using historical data",
    "data_sources": [
      {
        "name": "jobs_db",
        "access_type": "read",
        "required_fields": ["title", "salary_range", "location"]
      },
      {
        "name": "market_analytics",
        "access_type": "read",
        "snowflake_table": "MARKET_ANALYTICS.JOB_TRENDS"
      }
    ],
    "input_schema": {
      "type": "object",
      "properties": {
        "job_title": {"type": "string"},
        "location": {"type": "string"},
        "timeframe": {"type": "string", "enum": ["3m", "6m", "1y"]}
      }
    },
    "output_schema": {
      "type": "object",
      "properties": {
        "predicted_salary_range": {
          "type": "object",
          "properties": {
            "min": {"type": "number"},
            "max": {"type": "number"}
          }
        },
        "market_demand": {"type": "string", "enum": ["high", "medium", "low"]},
        "growth_trend": {"type": "number"}
      }
    }
  }'
```

### Example: Using Data Sources in Your Model

Here's how to access the data sources in your model implementation:

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from mcp.data import DataSourceClient  # MCP's data access library

app = FastAPI()

class MarketPredictionRequest(BaseModel):
    job_title: str
    location: str
    timeframe: str

class MarketPrediction(BaseModel):
    predicted_salary_range: dict
    market_demand: str
    growth_trend: float

class JobMarketPredictor:
    def __init__(self, data_client: DataSourceClient):
        self.data_client = data_client
    
    async def predict(self, job_title: str, location: str, timeframe: str) -> dict:
        # Fetch historical job data
        job_data = await self.data_client.query(
            source="jobs_db",
            query={
                "title": job_title,
                "location": location,
                "posted_date": f"last_{timeframe}"
            }
        )
        
        # Fetch market analytics
        market_trends = await self.data_client.query(
            source="snowflake",
            table="MARKET_ANALYTICS.JOB_TRENDS",
            filters={
                "job_category": job_title,
                "region": location,
                "period": timeframe
            }
        )
        
        # Your model logic here
        prediction = self.model.predict(job_data, market_trends)
        
        return {
            "predicted_salary_range": prediction.salary_range,
            "market_demand": prediction.demand,
            "growth_trend": prediction.trend
        }

# Initialize with your model and data client
data_client = DataSourceClient(api_key="your_api_key")
predictor = JobMarketPredictor(data_client)

@app.post("/predict", response_model=MarketPrediction)
async def predict(request: MarketPredictionRequest) -> MarketPrediction:
    try:
        result = await predictor.predict(
            request.job_title,
            request.location,
            request.timeframe
        )
        return MarketPrediction(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Data Access Controls

1. **API Key Permissions**:
   - Each API key has specific data source permissions
   - Access can be limited to specific fields/tables
   - Usage quotas per data source
   - Rate limiting per endpoint

2. **Audit Logging**:
   - All data access is logged
   - Usage metrics per model/key
   - Access patterns monitoring
   - Anomaly detection

3. **Data Governance**:
   - Field-level access control
   - Data masking for sensitive fields
   - Compliance with data policies
   - Usage agreements enforcement

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

### 1. Using Requests (Simple Client)

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

### 2. Using FastAPI (Full Integration)

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import httpx
from typing import Optional, List

# Define your data models
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float

class BatchPredictionRequest(BaseModel):
    inputs: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

# Create FastAPI app
app = FastAPI(title="My Model Service")

# API Key dependency
async def get_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    return x_api_key

# MCP client class
class MCPAsyncClient:
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    async def predict(self, model_id: str, text: str) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/models/{model_id}/predict",
                headers=self.headers,
                json={"text": text}
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.json()
                )
            return response.json()
    
    async def batch_predict(self, model_id: str, texts: List[str]) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/models/{model_id}/predict/batch",
                headers=self.headers,
                json={"inputs": [{"text": t} for t in texts]}
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.json()
                )
            return response.json()

# Create MCP client instance
MODEL_ID = "mod_xxxxx"  # Your model ID
mcp_client = None

@app.on_event("startup")
async def startup_event():
    global mcp_client
    mcp_client = MCPAsyncClient("your_api_key")

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(get_api_key)
) -> PredictionResponse:
    try:
        result = await mcp_client.predict(MODEL_ID, request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    api_key: str = Depends(get_api_key)
) -> BatchPredictionResponse:
    try:
        texts = [item.text for item in request.inputs]
        result = await mcp_client.batch_predict(MODEL_ID, texts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

"""
To run this FastAPI service:

1. Save as `model_service.py`
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn httpx
   ```

3. Run the server:
   ```bash
   uvicorn model_service:app --reload
   ```

4. Make predictions:
   ```bash
   # Single prediction
   curl -X POST http://localhost:8000/predict \
     -H "X-API-Key: your_api_key" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is amazing!"}'

   # Batch prediction
   curl -X POST http://localhost:8000/predict/batch \
     -H "X-API-Key: your_api_key" \
     -H "Content-Type: application/json" \
     -d '{
       "inputs": [
         {"text": "This product is amazing!"},
         {"text": "Not satisfied with the quality."}
       ]
     }'
   ```

5. View API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc
"""

### 3. Using FastAPI with Environment Variables and Better Error Handling

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import httpx
from typing import Optional, List
import os
from enum import Enum
import logging
from datetime import datetime

# Settings management
class Settings(BaseSettings):
    MCP_API_KEY: str = Field(..., env="MCP_API_KEY")
    MCP_BASE_URL: str = Field("http://localhost:8000", env="MCP_BASE_URL")
    MODEL_ID: str = Field(..., env="MODEL_ID")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data models
class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This product is amazing!"
            }
        }

class PredictionResponse(BaseModel):
    sentiment: SentimentEnum
    confidence: float = Field(..., ge=0, le=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis Service",
    description="API for sentiment analysis using MCP",
    version="1.0.0"
)

# Load settings
settings = Settings()

# MCP client with better error handling
class MCPClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    async def predict(self, text: str) -> dict:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/models/{settings.MODEL_ID}/predict",
                    headers=self.headers,
                    json={"text": text},
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException:
                logger.error("Request to MCP timed out")
                raise HTTPException(
                    status_code=504,
                    detail="Request to model service timed out"
                )
            except httpx.HTTPStatusError as e:
                logger.error(f"MCP request failed: {e.response.text}")
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=e.response.json()
                )
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail="Internal server error"
                )

# Create MCP client instance
mcp_client = MCPClient(settings.MCP_API_KEY, settings.MCP_BASE_URL)

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"model": PredictionResponse},
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)
async def predict(
    request: PredictionRequest,
    api_key: str = Header(..., alias="X-API-Key")
) -> PredictionResponse:
    """
    Get sentiment prediction for text.
    
    - **text**: The text to analyze (1-1000 characters)
    """
    logger.info(f"Received prediction request for text length: {len(request.text)}")
    try:
        result = await mcp_client.predict(request.text)
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

@app.get("/health")
async def health_check():
    """Check if the service is healthy"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": app.version
    }

"""
To use this enhanced version:

1. Create a .env file:
```env
MCP_API_KEY=your_api_key
MODEL_ID=mod_xxxxx
MCP_BASE_URL=http://localhost:8000
LOG_LEVEL=INFO
```

2. Run the service:
```bash
uvicorn service:app --reload
```

3. Make predictions with better error handling:
```bash
# Valid request
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'

# Invalid request (empty text)
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"text": ""}'
```
"""

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

Need help? Contact support at support@mcp.ai

## Using Swagger UI

The MCP API provides an interactive Swagger UI interface at `http://localhost:8000/docs`. Here's how to use it for common operations:

### 1. Getting an API Key via Swagger UI

1. Navigate to `http://localhost:8000/docs`
2. Expand the "API Keys" section
3. Click on `POST /api/keys`
4. Click "Try it out"
5. Modify the request body:
   ```json
   {
     "owner": "your.email@example.com",
     "description": "Model deployment key"
   }
   ```
6. Click "Execute"
7. Copy your API key from the response

### 2. Authorizing in Swagger UI

1. Click the "Authorize" button at the top
2. Enter your API key in the "X-API-Key" field
3. Click "Authorize"
4. Close the authorization dialog

Now all your requests will include the API key automatically.

### 3. Registering a Model

1. Expand the "Models" section
2. Click on `POST /api/models`
3. Click "Try it out"
4. Modify the request body:
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
5. Click "Execute"
6. Save the `model_id` from the response

### 4. Making Predictions

1. Expand the "Predictions" section
2. For single predictions:
   - Click on `POST /api/models/{model_id}/predict`
   - Click "Try it out"
   - Enter your model_id
   - Modify the request body:
     ```json
     {
       "text": "This product is amazing!"
     }
     ```
   - Click "Execute"

3. For batch predictions:
   - Click on `POST /api/models/{model_id}/predict/batch`
   - Click "Try it out"
   - Enter your model_id
   - Modify the request body:
     ```json
     {
       "inputs": [
         {"text": "This product is amazing!"},
         {"text": "Not satisfied with the quality."}
       ]
     }
     ```
   - Click "Execute"

### 5. Monitoring Model Status

1. Expand the "Models" section
2. Click on `GET /api/models/{model_id}/status`
3. Click "Try it out"
4. Enter your model_id
5. Click "Execute"

### 6. Viewing Model Metrics

1. Expand the "Models" section
2. Click on `GET /api/models/{model_id}/metrics`
3. Click "Try it out"
4. Enter your model_id
5. Click "Execute"

### 7. Managing API Keys

1. List all keys:
   - Click on `GET /api/keys`
   - Click "Try it out"
   - Click "Execute"

2. Revoke a key:
   - Click on `DELETE /api/keys/{key_id}`
   - Click "Try it out"
   - Enter the key_id
   - Click "Execute"

### 8. Additional Features in Swagger UI

1. **Schema Validation**:
   - Each endpoint shows the exact schema required
   - Required fields are marked with asterisks
   - Enums show all possible values

2. **Response Codes**:
   - Each endpoint lists all possible response codes
   - Example responses are provided for each code
   - Error responses include detailed descriptions

3. **Try it Out**:
   - Test endpoints directly in the browser
   - Automatically formats JSON
   - Shows curl commands for each request
   - Displays full request/response details

4. **Export**:
   - Download OpenAPI specification
   - Generate client code
   - Export curl commands

### Tips for Using Swagger UI

1. **Authentication**:
   - Always authorize first
   - Check if your token is still valid
   - Look for the 🔓 (locked) icon next to protected endpoints

2. **Request Bodies**:
   - Use the "Schema" tab for field descriptions
   - Click "Model" to see the full schema
   - Use "Example Value" as a starting point

3. **Responses**:
   - Check "Server response" for detailed errors
   - Use "Download" to save response data
   - Note the response headers for rate limits

4. **Troubleshooting**:
   - Clear authorization and try again
   - Check request body format
   - Verify required fields are filled
   - Look for validation errors in response

Need help? Contact support at support@mcp.ai

## Using Internal or Third-Party Models

### Registering ChatGPT-like Models

When registering internal models or third-party API-based models (like ChatGPT, Claude, etc.), you'll need to:
1. Define the model interface
2. Handle API key management
3. Set up proper error handling

Here's an example of registering and using OpenAI's GPT model:

1. Register the model with appropriate schemas:
```bash
curl -X POST http://localhost:8000/api/models \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpt-wrapper",
    "version": "1.0.0",
    "description": "GPT-3.5 Turbo wrapper for chat completion",
    "model_type": "third_party",
    "provider": "openai",
    "input_schema": {
      "type": "object",
      "properties": {
        "messages": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "role": {
                "type": "string",
                "enum": ["system", "user", "assistant"]
              },
              "content": {
                "type": "string"
              }
            },
            "required": ["role", "content"]
          }
        },
        "temperature": {
          "type": "number",
          "minimum": 0,
          "maximum": 2,
          "default": 1
        }
      },
      "required": ["messages"]
    },
    "output_schema": {
      "type": "object",
      "properties": {
        "response": {
          "type": "string"
        },
        "usage": {
          "type": "object",
          "properties": {
            "prompt_tokens": {"type": "integer"},
            "completion_tokens": {"type": "integer"},
            "total_tokens": {"type": "integer"}
          }
        }
      }
    },
    "config": {
      "requires_api_key": true,
      "provider_model": "gpt-3.5-turbo",
      "max_tokens": 4096
    }
  }'
```

2. Example FastAPI implementation:
```python
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import List, Optional
import openai
from datetime import datetime

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = Field(1.0, ge=0, le=2)

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    response: str
    usage: TokenUsage
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class GPTWrapper:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    async def generate(self, messages: List[dict], temperature: float = 1.0) -> dict:
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature
            )
            return {
                "response": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except openai.RateLimitError:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        except openai.AuthenticationError:
            raise HTTPException(status_code=401, detail="Invalid OpenAI API key")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

app = FastAPI()

# Initialize with environment variables
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
gpt_wrapper = GPTWrapper(OPENAI_API_KEY)

@app.post("/predict", response_model=ChatResponse)
async def predict(
    request: ChatRequest,
    api_key: str = Header(..., alias="X-API-Key")
) -> ChatResponse:
    """
    Generate a chat completion using GPT-3.5-turbo
    """
    try:
        result = await gpt_wrapper.generate(
            messages=request.messages,
            temperature=request.temperature
        )
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
Usage example:

1. Set up environment:
```bash
echo "OPENAI_API_KEY=your_openai_key" > .env
```

2. Make a request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your_mcp_key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "temperature": 0.7
  }'
```

Response:
```json
{
  "response": "The capital of France is Paris.",
  "usage": {
    "prompt_tokens": 27,
    "completion_tokens": 7,
    "total_tokens": 34
  },
  "timestamp": "2024-03-14T12:34:56.789Z"
}
```
"""

### Best Practices for Internal Models

1. **API Key Management**:
   - Store provider API keys securely (use environment variables)
   - Implement key rotation
   - Use different keys for development/production

2. **Error Handling**:
   - Handle provider-specific errors
   - Implement rate limiting
   - Add proper logging
   - Handle token limits

3. **Performance Optimization**:
   - Cache responses when appropriate
   - Implement request queuing
   - Monitor token usage
   - Set appropriate timeouts

4. **Security**:
   - Validate input messages
   - Sanitize outputs
   - Implement content filtering
   - Set up usage monitoring

5. **Cost Management**:
   - Track token usage
   - Implement quotas
   - Set up alerts for unusual usage
   - Use cheaper models for development

### Example: Registering Other Types of Models

1. **Hugging Face Model**:
```json
{
  "name": "bert-sentiment",
  "version": "1.0.0",
  "description": "BERT model from Hugging Face for sentiment analysis",
  "model_type": "third_party",
  "provider": "huggingface",
  "input_schema": {
    "type": "object",
    "properties": {
      "text": {"type": "string"},
      "options": {
        "type": "object",
        "properties": {
          "return_all_scores": {"type": "boolean"}
        }
      }
    }
  },
  "config": {
    "model_id": "nlptown/bert-base-multilingual-uncased-sentiment",
    "requires_api_key": true
  }
}
```

2. **Custom Internal Model**:
```json
{
  "name": "internal-classifier",
  "version": "2.0.0",
  "description": "Custom-trained classifier for internal use",
  "model_type": "internal",
  "input_schema": {
    "type": "object",
    "properties": {
      "features": {
        "type": "array",
        "items": {"type": "number"}
      }
    }
  },
  "config": {
    "batch_size": 32,
    "requires_gpu": true,
    "memory_requirement": "2GB"
  }
}
```

Need help? Contact support at support@mcp.ai