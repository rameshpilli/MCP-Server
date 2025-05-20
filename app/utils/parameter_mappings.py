import json
import logging
import os
from typing import Dict, Any

from app.config import config

logger = logging.getLogger("mcp_server.parameter_mappings")

# Default mappings used if no external file is provided
DEFAULT_MAPPINGS: Dict[str, Dict[str, Any]] = {
    "currency": {
        "CAD": ["cad", "canadian", "canadian dollar", "canada dollar", "canada"],
        "USD": ["usd", "us", "u.s.", "american dollar", "dollar", "dollars"],
    },
    "sorting": {
        "gainers": ["gainers", "gain", "gaining", "increasing", "rising"],
        "decliners": ["decliners", "decline", "decreasing", "dropping", "falling"],
    },
    "region": {
        "USA": ["usa", "us", "america", "united states"],
        "CAN": ["can", "canada", "canadian"],
        "EUR": ["eur", "europe", "european"],
        "APAC": ["apac", "asia", "pacific"],
        "LATAM": ["latam", "latin", "south america"],
    },
    "time_period": {
        "FY": ["fiscal", "fiscal year", "fy"],
        "CY": ["calendar", "calendar year", "cy"],
    },
    "time_filter": {
        "YR": ["year", "annual"],
        "QR": ["quarter", "q1", "q2", "q3", "q4"],
        "MT": ["month", "monthly"],
        "DY": ["day", "daily"],
    },
    "focus_list": {
        "Focus40": ["focus40", "focus 40", "focus"],
        "FS30": ["fs30", "fs 30", "fs"],
        "Corp100": ["corp100", "corp 100", "corporate 100", "corp"],
    },
}


def load_parameter_mappings() -> Dict[str, Dict[str, Any]]:
    """Load parameter mappings from JSON file if configured."""
    path = config.PARAMETER_MAPPINGS_PATH
    mappings = DEFAULT_MAPPINGS.copy()
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_mappings = json.load(f)
                for key, value in user_mappings.items():
                    if isinstance(value, dict):
                        mappings.setdefault(key, {}).update(value)
        except Exception as e:
            logger.warning(f"Failed to load parameter mappings from {path}: {e}")
    return mappings
