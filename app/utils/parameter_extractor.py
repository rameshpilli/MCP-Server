import json
import logging
import httpx
import os
from typing import Dict, Any, List
from .parameter_mappings import load_parameter_mappings
from app.config import config

logger = logging.getLogger("mcp_server.parameter_extractor")

# Load mappings once
PARAM_MAPPINGS = load_parameter_mappings()

async def _call_llm(messages: List[Dict[str, str]]) -> str:
    """Internal helper to call the configured LLM."""
    headers = {"Content-Type": "application/json"}
    token = ""
    # OAuth if configured
    if config.LLM_OAUTH_ENDPOINT and config.LLM_OAUTH_CLIENT_ID and config.LLM_OAUTH_CLIENT_SECRET:
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
        except Exception as e:
            logger.warning(f"OAuth token retrieval failed: {e}")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif (api_key := (config.COHERE_SERVER_BEARER_TOKEN or os.getenv("OPENAI_API_KEY"))):
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            headers=headers,
            json={"model": config.LLM_MODEL, "messages": messages, "temperature": 0},
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return content

async def extract_parameters_with_llm(query: str, tool_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Use an LLM to extract parameters for the specified tool."""
    system_prompt = (
        "You are a helpful assistant that extracts parameter values from user queries. "
        "Return a JSON object with keys matching the tool's input schema."
    )
    user_prompt = (
        f"Tool: {tool_name}\n"
        f"Schema: {json.dumps(schema)}\n"
        f"Query: {query}\n"
        "Return only a JSON object."
    )
    try:
        response = await _call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        # Attempt to parse JSON from response
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            json_str = response[start : end + 1]
            params = json.loads(json_str)
        else:
            params = json.loads(response.strip())
    except Exception as e:
        logger.warning(f"LLM extraction failed: {e}")
        return {}

    # Normalize values using mappings
    normalized = {}
    for key, val in params.items():
        if key not in schema:
            continue
        if val is None:
            continue
        if key in ["currency", "ccy_code", "sorting", "sorting_criteria", "region", "focus_list", "time_period", "time_filter"]:
            base_key = key
            if key == "ccy_code":
                base_key = "currency"
            if key == "sorting_criteria":
                base_key = "sorting"
            mapping = PARAM_MAPPINGS.get(base_key, {})
            val_upper = str(val).upper()
            val_lower = str(val).lower()
            matched = False
            for std, terms in mapping.items():
                if val_upper == std or val_lower in [t.lower() for t in terms]:
                    normalized[key] = std
                    matched = True
                    break
            if not matched:
                normalized[key] = val
        else:
            normalized[key] = val
    return normalized
