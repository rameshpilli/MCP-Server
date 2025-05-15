"""
API Resources

This module defines external API resources that can be used by tools.
"""

import httpx
import logging
from app.registry.resources import registry, ResourceDefinition

# Get logger
logger = logging.getLogger('mcp_server.resources.apis')

# Register Weather API
weather_api = ResourceDefinition(
    name="weather_api",
    description="Weather API for retrieving weather data",
    handler=httpx.AsyncClient().get,
    config={
        "base_url": "https://api.example.com/weather",
        "api_key": "YOUR_API_KEY",
        "timeout": 5.0
    },
    metadata={
        "endpoints": {
            "current": "/current",
            "forecast": "/forecast",
            "historical": "/historical"
        },
        "rate_limit": "60 requests per minute"
    }
)
registry.register(weather_api)

# Register News API
news_api = ResourceDefinition(
    name="news_api",
    description="News API for retrieving news articles",
    handler=httpx.AsyncClient().get,
    config={
        "base_url": "https://api.example.com/news",
        "api_key": "YOUR_API_KEY",
        "timeout": 5.0
    },
    metadata={
        "endpoints": {
            "headlines": "/headlines",
            "search": "/search",
            "sources": "/sources"
        },
        "rate_limit": "100 requests per day"
    }
)
registry.register(news_api)

# Log registered resources
logger.info(f"Registered API resources: {', '.join(registry.list_resources().keys())}") 