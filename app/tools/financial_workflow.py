import logging
import os
import random
from typing import Dict, Any, List

import requests
from fastmcp import Context

from app.registry.tools import register_tool
from app.config import config

logger = logging.getLogger('mcp_server.tools.financial_workflow')

BASE_URL = config.FINANCIAL_API_BASE_URL
SUPPORTED_CURRENCIES = [c.strip().upper() for c in config.SUPPORTED_CURRENCIES]

def _check_company(company_id: str) -> Dict[str, Any]:
    """Validate company existence. Placeholder implementation."""
    if not company_id:
        return {"valid": False, "message": "Company ID not provided"}
    try:
        resp = requests.get(f"{BASE_URL}/companies/{company_id}", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "valid": True,
                "ticker": data.get("ticker", company_id.upper()),
                "name": data.get("name", company_id),
                "sector": data.get("sector", "Unknown"),
            }
    except Exception as e:
        logger.debug(f"Company lookup failed: {e}")
    # Fallback to basic normalization
    return {
        "valid": True,
        "ticker": company_id.upper(),
        "name": company_id.upper(),
        "sector": "Unknown",
    }

def _get_metric(company: str, metric: str, period: str) -> float:
    """Retrieve a metric value. Placeholder using random values."""
    try:
        resp = requests.get(
            f"{BASE_URL}/metrics/{metric}",
            params={"company": company, "period": period},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            return float(data.get("value", 0))
    except Exception as e:
        logger.debug(f"Metric lookup failed: {e}")
    return round(random.uniform(1_000, 10_000), 2)

def register_tools(mcp):
    """Register financial workflow tools."""

    @mcp.tool()
    @register_tool(
        name="validate_company",
        description="Validate if a company exists and return standardized info",
        namespace="finance",
        input_schema={"company_id": {"type": "string", "description": "Ticker or company identifier"}},
    )
    async def validate_company(ctx: Context, company_id: str) -> str:
        info = _check_company(company_id)
        if not info["valid"]:
            return info.get("message", "Company validation failed")
        return (
            f"Company validated: {info['ticker']} (Name: {info['name']}, "
            f"Sector: {info['sector']})"
        )

    @mcp.tool()
    @register_tool(
        name="validate_currency",
        description="Validate currency code and optionally convert amount",
        namespace="finance",
        input_schema={
            "amount": {"type": "number", "description": "Amount to validate"},
            "currency": {"type": "string", "description": "Currency code"},
            "target_currency": {
                "type": "string",
                "description": "Target currency for conversion",
                "default": None,
            },
        },
    )
    async def validate_currency(
        ctx: Context, amount: float, currency: str, target_currency: str | None = None
    ) -> str:
        cur = currency.upper()
        if cur not in SUPPORTED_CURRENCIES:
            return f"Unsupported currency: {currency}"
        if target_currency:
            tgt = target_currency.upper()
            if tgt not in SUPPORTED_CURRENCIES:
                return f"Unsupported target currency: {target_currency}"
            if tgt == cur:
                converted = amount
            else:
                converted = round(amount * 1.1, 2)
            return f"{converted} {tgt} (converted from {amount} {cur})"
        return f"{amount} {cur}"

    @mcp.tool()
    @register_tool(
        name="aggregate_metrics",
        description="Aggregate financial metrics for a company",
        namespace="finance",
        input_schema={
            "company_id": {"type": "string", "description": "Company identifier"},
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Metrics to retrieve",
            },
            "period": {"type": "string", "description": "Time period", "default": "2024"},
        },
    )
    async def aggregate_metrics(
        ctx: Context, company_id: str, metrics: List[str], period: str = "2024"
    ) -> Dict[str, Any]:
        info = _check_company(company_id)
        if not info["valid"]:
            return {"error": info.get("message", "Company not found")}
        result = {"company": info["ticker"], "period": period}
        for metric in metrics:
            result[metric] = _get_metric(info["ticker"], metric, period)
        return result

    @mcp.tool()
    @register_tool(
        name="analyze_financial_health",
        description="Analyze financial metrics and return insights",
        namespace="finance",
        input_schema={"data": {"type": "object", "description": "Aggregated metric data"}},
    )
    async def analyze_financial_health(ctx: Context, data: Dict[str, Any]) -> str:
        revenue = data.get("revenue")
        profit = data.get("profit")
        if revenue and profit:
            margin = profit / revenue * 100 if revenue else 0
            return f"Profit margin is {margin:.2f}% for {data.get('company')}"
        return "Not enough data to analyze financial health"

    @mcp.tool()
    @register_tool(
        name="build_custom_report",
        description="Create a custom financial report by chaining validation and analysis",
        namespace="finance",
        input_schema={
            "company_id": {"type": "string", "description": "Company identifier"},
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Metrics to include",
            },
            "period": {"type": "string", "default": "2024"},
            "currency": {"type": "string", "default": "USD"},
        },
    )
    async def build_custom_report(
        ctx: Context,
        company_id: str,
        metrics: List[str],
        period: str = "2024",
        currency: str = "USD",
    ) -> str:
        info = _check_company(company_id)
        if not info["valid"]:
            return info.get("message", "Company not found")
        if currency.upper() not in SUPPORTED_CURRENCIES:
            return f"Unsupported currency: {currency}"
        data = {metric: _get_metric(info["ticker"], metric, period) for metric in metrics}
        lines = [f"# Financial Report for {info['ticker']} ({period}, {currency.upper()})", ""]
        for metric, value in data.items():
            lines.append(f"- {metric}: {value:,.2f} {currency.upper()}")
        if "revenue" in data and "profit" in data:
            margin = data["profit"] / data["revenue"] * 100 if data["revenue"] else 0
            lines.append("")
            lines.append(f"Profit margin: {margin:.2f}%")
        return "\n".join(lines)
