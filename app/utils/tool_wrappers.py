import logging
import os
import asyncio
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List

import httpx

from app.config import config

logger = logging.getLogger("mcp_server.tool_wrappers")

async def _call_llm(messages: List[Dict[str, str]]) -> str:
    """Call the configured LLM endpoint with the given messages."""
    if not config.LLM_BASE_URL:
        raise RuntimeError("LLM_BASE_URL not configured")

    headers = {"Content-Type": "application/json"}
    token = ""

    if (
        config.LLM_OAUTH_ENDPOINT
        and config.LLM_OAUTH_CLIENT_ID
        and config.LLM_OAUTH_CLIENT_SECRET
    ):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    config.LLM_OAUTH_ENDPOINT,
                    data={
                        "grant_type": config.LLM_OAUTH_GRANT_TYPE or "client_credentials",
                        "client_id": config.LLM_OAUTH_CLIENT_ID,
                        "client_secret": config.LLM_OAUTH_CLIENT_SECRET,
                        "scope": config.LLM_OAUTH_SCOPE or "read",
                    },
                )
                resp.raise_for_status()
                token = resp.json().get("access_token", "")
        except Exception as e:  # pragma: no cover - network errors
            logger.warning(f"OAuth token retrieval failed: {e}")

    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif api_key := (config.COHERE_SERVER_BEARER_TOKEN or os.getenv("OPENAI_API_KEY")):
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            headers=headers,
            json={"model": config.LLM_MODEL, "messages": messages, "temperature": 0},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

async def _enhance(content: str, instruction: str, system_prompt: str) -> str:
    """Enhance text using an LLM with the provided instruction."""
    if not config.LLM_BASE_URL:
        return content

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{instruction}\n\n{content}"},
    ]
    try:
        enhanced = await _call_llm(messages)
        return enhanced
    except Exception as e:  # pragma: no cover - network errors
        logger.warning(f"LLM enhancement failed: {e}")
        return content

def llm_enhance_wrapper(instruction: str, system_prompt: str) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorator to enhance a tool's output using an LLM."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await func(*args, **kwargs)
            if isinstance(result, str):
                return await _enhance(result, instruction, system_prompt)
            return result

        return wrapper

    return decorator
