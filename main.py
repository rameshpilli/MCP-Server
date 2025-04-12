from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, Optional, List
import os
import logging
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import httpx
import json
from cachetools import TTLCache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
import pandas as pd
import csv
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(client_ip)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Server configuration
class ServerConfig:
    ALLOWED_EXTENSIONS = {'.txt', '.log', '.py', '.json', '.yaml', '.yml', '.md', '.csv'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    BASE_DIR = os.path.join(os.getcwd(), 'data')  # Restrict to data directory
    API_KEYS = {"test_key", "dev_key"}  # Sample API keys
    CACHE_TTL = 300  # Cache TTL in seconds (5 minutes)
    RATE_LIMIT = "20/minute"  # Rate limit per IP

# Create data directory if it doesn't exist
os.makedirs(ServerConfig.BASE_DIR, exist_ok=True)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize cache
api_cache = TTLCache(maxsize=100, ttl=ServerConfig.CACHE_TTL)

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key")

app = FastAPI(
    title="MCP Server",
    description="A feature-rich Master Control Program server for file operations and real-time data",
    version="1.0.0"
)

# Add rate limiter error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaskRequest(BaseModel):
    task_type: Literal["read_file", "list_directory", "file_info", "search_files", "calculate_hash",
                      "get_crypto_prices", "get_weather", "get_news", "get_quote",
                      "get_joke", "get_dog_image", "get_cat_fact", "get_ip_info",
                      "get_exchange_rates", "get_activity", "read_csv", "file_metadata"]
    parameters: Dict[str, Any]

class TaskResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    cached: bool = False

async def verify_api_key(api_key: str = Header(..., alias="X-API-Key")):
    """Verify the API key."""
    if api_key not in ServerConfig.API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

def validate_file_path(file_path: str) -> Path:
    """Validate and normalize file path within data directory."""
    try:
        # Convert to absolute path and resolve any symlinks
        path = Path(os.path.join(ServerConfig.BASE_DIR, file_path)).resolve()
        
        # Check if the path is within the base directory
        if not str(path).startswith(str(ServerConfig.BASE_DIR)):
            raise HTTPException(status_code=403, detail="Access to files outside data directory is forbidden")
        
        return path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file path: {str(e)}")

def validate_file_size(file_path: Path):
    """Validate file size."""
    if file_path.stat().st_size > ServerConfig.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File size exceeds maximum limit of {ServerConfig.MAX_FILE_SIZE/1024/1024}MB")

def validate_file_extension(file_path: Path):
    """Validate file extension."""
    if file_path.suffix not in ServerConfig.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed extensions: {ServerConfig.ALLOWED_EXTENSIONS}")

def get_cache_key(task_type: str, parameters: Dict[str, Any]) -> str:
    """Generate a cache key from task type and parameters."""
    param_str = json.dumps(parameters, sort_keys=True)
    return f"{task_type}:{param_str}"

async def get_cached_response(cache_key: str) -> Optional[Dict]:
    """Get cached response if available."""
    return api_cache.get(cache_key)

def set_cached_response(cache_key: str, response: Dict):
    """Cache the response."""
    api_cache[cache_key] = response

class LoggingMiddleware:
    """Middleware for logging requests with client IP."""
    async def __call__(self, request: Request, call_next):
        start_time = datetime.now()
        response = await call_next(request)
        process_time = (datetime.now() - start_time).total_seconds()
        
        client_ip = request.client.host
        logger.info(
            f"Request processed",
            extra={
                "client_ip": client_ip,
                "method": request.method,
                "url": str(request.url),
                "process_time": process_time,
                "status_code": response.status_code
            }
        )
        return response

# Add logging middleware
app.middleware("http")(LoggingMiddleware())

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>MCP Server</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
                h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
                .endpoint { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                code { background: #e9ecef; padding: 2px 5px; border-radius: 3px; font-family: monospace; }
                .task-type { color: #2980b9; }
                .navigation { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; }
                .category { margin-top: 20px; }
                .category h4 { color: #34495e; }
                .info { background: #e1f5fe; padding: 10px; border-radius: 4px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>Welcome to MCP Server</h1>
            <p>This server provides various operations through a unified task execution interface.</p>
            
            <div class="info">
                <strong>Server Features:</strong>
                <ul>
                    <li>Rate limiting: 20 requests per minute per IP</li>
                    <li>Response caching: 5 minutes TTL</li>
                    <li>Comprehensive error handling</li>
                    <li>Request logging</li>
                </ul>
            </div>

            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <h3>POST /execute-task</h3>
                <div class="category">
                    <h4>File Operations:</h4>
                    <ul>
                        <li><code class="task-type">read_file</code>: Read contents of a file</li>
                        <li><code class="task-type">list_directory</code>: List contents of a directory</li>
                        <li><code class="task-type">file_info</code>: Get detailed information about a file</li>
                        <li><code class="task-type">search_files</code>: Search for files by pattern</li>
                        <li><code class="task-type">calculate_hash</code>: Calculate file hash (MD5, SHA-1, SHA-256)</li>
                    </ul>
                </div>
                <div class="category">
                    <h4>Real-time Data (No API Key Required):</h4>
                    <ul>
                        <li><code class="task-type">get_crypto_prices</code>: Get cryptocurrency prices</li>
                        <li><code class="task-type">get_quote</code>: Get random inspirational quote</li>
                        <li><code class="task-type">get_joke</code>: Get programming jokes</li>
                        <li><code class="task-type">get_dog_image</code>: Get random dog images</li>
                        <li><code class="task-type">get_cat_fact</code>: Get random cat facts</li>
                        <li><code class="task-type">get_ip_info</code>: Get information about an IP address</li>
                        <li><code class="task-type">get_exchange_rates</code>: Get currency exchange rates</li>
                        <li><code class="task-type">get_activity</code>: Get random activity suggestions</li>
                    </ul>
                </div>
                <div class="category">
                    <h4>Real-time Data (API Key Required):</h4>
                    <ul>
                        <li><code class="task-type">get_weather</code>: Get weather information</li>
                        <li><code class="task-type">get_news</code>: Get latest news</li>
                    </ul>
                </div>
            </div>
            <div class="navigation">
                <p>For detailed API documentation, visit:</p>
                <ul>
                    <li><a href="/docs">Interactive API Documentation (Swagger UI)</a></li>
                    <li><a href="/redoc">Alternative Documentation (ReDoc)</a></li>
                </ul>
            </div>
        </body>
    </html>
    """

@app.post("/execute-task", response_model=TaskResponse)
@limiter.limit(ServerConfig.RATE_LIMIT)
async def execute_task(request: Request, task: TaskRequest, api_key: str = Depends(verify_api_key)):
    client_ip = request.client.host
    logger.info(
        f"Received task request - Type: {task.task_type}",
        extra={"client_ip": client_ip}
    )
    
    try:
        # For API tasks, check cache first
        if task.task_type.startswith("get_"):
            cache_key = get_cache_key(task.task_type, task.parameters)
            cached_response = await get_cached_response(cache_key)
            if cached_response:
                return TaskResponse(
                    status="success",
                    data=cached_response,
                    cached=True
                )

        # New CSV-specific tasks
        if task.task_type == "read_csv":
            file_path = validate_file_path(task.parameters.get("file_path"))
            if not file_path.is_file():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            
            if file_path.suffix != '.csv':
                raise HTTPException(status_code=400, detail="File must be a CSV")
            
            try:
                df = pd.read_csv(file_path)
                first_5_rows = df.head().to_dict('records')
                
                return TaskResponse(
                    status="success",
                    data={
                        "file_path": str(file_path),
                        "rows": first_5_rows,
                        "total_rows": len(df),
                        "preview_rows": 5
                    }
                )
            except Exception as e:
                logger.error(f"Error reading CSV file {file_path}: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")

        elif task.task_type == "file_metadata":
            file_path = validate_file_path(task.parameters.get("file_path"))
            if not file_path.is_file():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            
            if file_path.suffix != '.csv':
                raise HTTPException(status_code=400, detail="File must be a CSV")
            
            try:
                df = pd.read_csv(file_path)
                return TaskResponse(
                    status="success",
                    data={
                        "file_path": str(file_path),
                        "num_rows": len(df),
                        "num_columns": len(df.columns),
                        "column_names": list(df.columns),
                        "data_types": df.dtypes.astype(str).to_dict(),
                        "file_size": os.path.getsize(file_path),
                        "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"Error getting CSV metadata {file_path}: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error getting CSV metadata: {str(e)}")

        elif task.task_type == "read_file":
            file_path = validate_file_path(task.parameters.get("file_path"))
            if not file_path.is_file():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            
            validate_file_extension(file_path)
            validate_file_size(file_path)
            
            try:
                content = file_path.read_text()
                return TaskResponse(
                    status="success",
                    data={
                        "content": content,
                        "file_path": str(file_path),
                        "line_count": len(content.splitlines()),
                        "size": file_path.stat().st_size
                    }
                )
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

        elif task.task_type == "list_directory":
            dir_path = validate_file_path(task.parameters.get("dir_path", "."))
            if not dir_path.is_dir():
                raise HTTPException(status_code=404, detail=f"Directory not found: {dir_path}")
            
            try:
                contents = []
                for item in dir_path.iterdir():
                    contents.append({
                        "name": item.name,
                        "is_file": item.is_file(),
                        "is_dir": item.is_dir(),
                        "size": item.stat().st_size if item.is_file() else None,
                        "modified_at": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
                
                return TaskResponse(
                    status="success",
                    data={
                        "dir_path": str(dir_path),
                        "contents": contents,
                        "total_items": len(contents)
                    }
                )
            except Exception as e:
                logger.error(f"Error listing directory {dir_path}: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error listing directory: {str(e)}")

        elif task.task_type == "file_info":
            file_path = validate_file_path(task.parameters.get("file_path"))
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"Path not found: {file_path}")
            
            try:
                stat = file_path.stat()
                return TaskResponse(
                    status="success",
                    data={
                        "file_path": str(file_path),
                        "size": stat.st_size,
                        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "accessed_at": datetime.fromtimestamp(stat.st_atime).isoformat(),
                        "is_file": file_path.is_file(),
                        "is_dir": file_path.is_dir(),
                        "is_symlink": file_path.is_symlink(),
                        "extension": file_path.suffix,
                        "permissions": oct(stat.st_mode)[-3:]
                    }
                )
            except Exception as e:
                logger.error(f"Error getting file info {file_path}: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error getting file info: {str(e)}")

        elif task.task_type == "search_files":
            base_dir = validate_file_path(task.parameters.get("base_dir", "."))
            pattern = task.parameters.get("pattern", "*")
            recursive = task.parameters.get("recursive", True)
            
            if not base_dir.is_dir():
                raise HTTPException(status_code=404, detail=f"Directory not found: {base_dir}")
            
            try:
                matches = []
                if recursive:
                    glob_pattern = f"**/{pattern}"
                else:
                    glob_pattern = pattern
                
                for file_path in base_dir.glob(glob_pattern):
                    if file_path.is_file():
                        matches.append({
                            "path": str(file_path.relative_to(base_dir)),
                            "size": file_path.stat().st_size,
                            "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        })
                
                return TaskResponse(
                    status="success",
                    data={
                        "base_dir": str(base_dir),
                        "pattern": pattern,
                        "matches": matches,
                        "total_matches": len(matches)
                    }
                )
            except Exception as e:
                logger.error(f"Error searching files in {base_dir}: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error searching files: {str(e)}")

        elif task.task_type == "calculate_hash":
            file_path = validate_file_path(task.parameters.get("file_path"))
            hash_type = task.parameters.get("hash_type", "sha256").lower()
            
            if not file_path.is_file():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            
            validate_file_size(file_path)
            
            try:
                hash_funcs = {
                    "md5": hashlib.md5(),
                    "sha1": hashlib.sha1(),
                    "sha256": hashlib.sha256()
                }
                
                if hash_type not in hash_funcs:
                    raise HTTPException(status_code=400, detail=f"Unsupported hash type. Supported types: {list(hash_funcs.keys())}")
                
                hash_func = hash_funcs[hash_type]
                
                with file_path.open('rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        hash_func.update(chunk)
                
                return TaskResponse(
                    status="success",
                    data={
                        "file_path": str(file_path),
                        "hash_type": hash_type,
                        "hash_value": hash_func.hexdigest()
                    }
                )
            except Exception as e:
                logger.error(f"Error calculating hash for {file_path}: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error calculating hash: {str(e)}")

        elif task.task_type == "get_crypto_prices":
            symbols = task.parameters.get("symbols", ["BTC", "ETH", "DOGE"])
            if isinstance(symbols, str):
                symbols = [symbols]
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"https://api.coingecko.com/api/v3/simple/price",
                        params={
                            "ids": ",".join(["bitcoin", "ethereum", "dogecoin"]),
                            "vs_currencies": "usd"
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    return TaskResponse(
                        status="success",
                        data={
                            "prices": {
                                "BTC": data["bitcoin"]["usd"],
                                "ETH": data["ethereum"]["usd"],
                                "DOGE": data["dogecoin"]["usd"]
                            },
                            "source": "CoinGecko API"
                        }
                    )
            except Exception as e:
                logger.error(f"Error fetching crypto prices: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error fetching crypto prices: {str(e)}")

        elif task.task_type == "get_weather":
            city = task.parameters.get("city", "London")
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://api.openweathermap.org/data/2.5/weather",
                        params={
                            "q": city,
                            "appid": ServerConfig.WEATHER_API_KEY,
                            "units": "metric"
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    return TaskResponse(
                        status="success",
                        data={
                            "city": data["name"],
                            "country": data["sys"]["country"],
                            "temperature": data["main"]["temp"],
                            "feels_like": data["main"]["feels_like"],
                            "humidity": data["main"]["humidity"],
                            "description": data["weather"][0]["description"],
                            "wind_speed": data["wind"]["speed"]
                        }
                    )
            except Exception as e:
                logger.error(f"Error fetching weather data: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error fetching weather data: {str(e)}")

        elif task.task_type == "get_news":
            category = task.parameters.get("category", "technology")
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://newsapi.org/v2/top-headlines",
                        params={
                            "category": category,
                            "language": "en",
                            "apiKey": ServerConfig.NEWS_API_KEY
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    articles = [{
                        "title": article["title"],
                        "description": article["description"],
                        "url": article["url"],
                        "published_at": article["publishedAt"]
                    } for article in data["articles"][:5]]  # Get top 5 articles
                    
                    return TaskResponse(
                        status="success",
                        data={
                            "category": category,
                            "articles": articles,
                            "total_results": len(articles)
                        }
                    )
            except Exception as e:
                logger.error(f"Error fetching news: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")

        elif task.task_type == "get_quote":
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("https://api.quotable.io/random")
                    response.raise_for_status()
                    data = response.json()
                    
                    return TaskResponse(
                        status="success",
                        data={
                            "quote": data["content"],
                            "author": data["author"],
                            "tags": data["tags"]
                        }
                    )
            except Exception as e:
                logger.error(f"Error fetching quote: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error fetching quote: {str(e)}")

        elif task.task_type == "get_joke":
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("https://v2.jokeapi.dev/joke/Programming")
                    response.raise_for_status()
                    data = response.json()
                    
                    joke_data = {
                        "category": data["category"],
                        "type": data["type"]
                    }
                    
                    if data["type"] == "single":
                        joke_data["joke"] = data["joke"]
                    else:
                        joke_data["setup"] = data["setup"]
                        joke_data["delivery"] = data["delivery"]
                    
                    set_cached_response(get_cache_key(task.task_type, task.parameters), joke_data)
                    return TaskResponse(status="success", data=joke_data)
            except Exception as e:
                logger.error(f"Error fetching joke: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error fetching joke: {str(e)}")

        elif task.task_type == "get_dog_image":
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("https://dog.ceo/api/breeds/image/random")
                    response.raise_for_status()
                    data = response.json()
                    
                    dog_data = {
                        "image_url": data["message"],
                        "status": data["status"]
                    }
                    
                    set_cached_response(get_cache_key(task.task_type, task.parameters), dog_data)
                    return TaskResponse(status="success", data=dog_data)
            except Exception as e:
                logger.error(f"Error fetching dog image: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error fetching dog image: {str(e)}")

        elif task.task_type == "get_cat_fact":
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("https://catfact.ninja/fact")
                    response.raise_for_status()
                    data = response.json()
                    
                    cat_data = {
                        "fact": data["fact"],
                        "length": data["length"]
                    }
                    
                    set_cached_response(get_cache_key(task.task_type, task.parameters), cat_data)
                    return TaskResponse(status="success", data=cat_data)
            except Exception as e:
                logger.error(f"Error fetching cat fact: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error fetching cat fact: {str(e)}")

        elif task.task_type == "get_ip_info":
            ip = task.parameters.get("ip", "")
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://ip-api.com/json/{ip}")
                    response.raise_for_status()
                    data = response.json()
                    
                    ip_data = {
                        "ip": data.get("query"),
                        "city": data.get("city"),
                        "country": data.get("country"),
                        "region": data.get("regionName"),
                        "isp": data.get("isp"),
                        "timezone": data.get("timezone")
                    }
                    
                    set_cached_response(get_cache_key(task.task_type, task.parameters), ip_data)
                    return TaskResponse(status="success", data=ip_data)
            except Exception as e:
                logger.error(f"Error fetching IP info: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error fetching IP info: {str(e)}")

        elif task.task_type == "get_exchange_rates":
            base = task.parameters.get("base", "USD")
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"https://open.er-api.com/v6/latest/{base}")
                    response.raise_for_status()
                    data = response.json()
                    
                    exchange_data = {
                        "base": data["base_code"],
                        "rates": data["rates"],
                        "last_updated": data["time_last_update_utc"]
                    }
                    
                    set_cached_response(get_cache_key(task.task_type, task.parameters), exchange_data)
                    return TaskResponse(status="success", data=exchange_data)
            except Exception as e:
                logger.error(f"Error fetching exchange rates: {str(e)}", extra={"client_ip": client_ip})
                raise HTTPException(status_code=500, detail=f"Error fetching exchange rates: {str(e)}")

        elif task.task_type == "get_activity":
            return await handle_get_activity()

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported task type: {task.task_type}")

    except HTTPException as e:
        logger.error(f"HTTP Exception in task execution: {str(e)}", extra={"client_ip": client_ip})
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in task execution: {str(e)}", extra={"client_ip": client_ip})
        raise HTTPException(status_code=500, detail=str(e))

@retry(
    stop=stop_after_attempt(int(os.getenv("MAX_RETRIES", "3"))),
    wait=wait_exponential(multiplier=int(os.getenv("RETRY_DELAY", "1")), min=1, max=10)
)
async def fetch_with_retry(url: str) -> Dict:
    """Fetch data from URL with retry logic."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

async def handle_get_activity() -> TaskResponse:
    """Handle get activity task with multiple fallback APIs."""
    apis = [
        {
            "url": "https://www.boredapi.com/api/activity",
            "transform": lambda data: {
                "activity": data["activity"],
                "type": data["type"],
                "participants": data["participants"],
                "price": data["price"],
                "accessibility": data["accessibility"]
            }
        },
        {
            "url": "https://api.api-ninjas.com/v1/bucketlist",
            "transform": lambda data: {
                "activity": data,
                "type": "bucket_list",
                "participants": 1,
                "price": 0.0,
                "accessibility": 0.5
            }
        },
        {
            "url": "https://www.random.org/integers/?num=1&min=1&max=100&col=1&base=10&format=plain&rnd=new",
            "transform": lambda data: {
                "activity": f"Count to {int(data)}",
                "type": "recreational",
                "participants": 1,
                "price": 0.0,
                "accessibility": 0.0
            }
        }
    ]
    
    errors = []
    # Try each API in sequence
    for api in apis:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(api["url"])
                response.raise_for_status()
                
                # Handle different response types
                if response.headers.get("content-type", "").startswith("application/json"):
                    data = response.json()
                else:
                    data = response.text.strip()
                
                activity_data = api["transform"](data)
                activity_data["source"] = "external"
                activity_data["api_url"] = api["url"]
                return TaskResponse(
                    status="success",
                    data=activity_data,
                    cached=False
                )
        except Exception as e:
            errors.append(f"{api['url']}: {str(e)}")
            logger.warning(f"Activity API failed: {api['url']} - {str(e)}")
            continue
    
    # If all APIs fail, use local activities
    try:
        local_activities = [
            {"activity": "Write a short story", "type": "creative"},
            {"activity": "Do 10 push-ups", "type": "physical"},
            {"activity": "Meditate for 5 minutes", "type": "relaxation"},
            {"activity": "Learn a new word", "type": "educational"},
            {"activity": "Draw a sketch", "type": "artistic"},
            {"activity": "Organize your desk", "type": "productivity"},
            {"activity": "Practice deep breathing", "type": "wellness"},
            {"activity": "Write a thank you note", "type": "social"},
            {"activity": "Stretch for 5 minutes", "type": "physical"},
            {"activity": "Plan your next day", "type": "planning"}
        ]
        
        import random
        activity = random.choice(local_activities)
        activity_data = {
            "activity": activity["activity"],
            "type": activity["type"],
            "participants": 1,
            "price": 0.0,
            "accessibility": 0.0,
            "source": "local",
            "fallback_reason": f"External APIs failed: {'; '.join(errors)}"
        }
        
        return TaskResponse(
            status="success",
            data=activity_data,
            cached=False
        )
    except Exception as e:
        logger.error(f"Local activity generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"All activity sources failed, including local fallback. Errors: {'; '.join(errors)}"
        )

if __name__ == "__main__":
    logger.info("Starting MCP Server...", extra={"client_ip": "system"})
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")