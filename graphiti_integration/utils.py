"""Utility helpers for Graphiti integration."""
from __future__ import annotations

import io
from typing import List, Dict
import pandas as pd


def markdown_table_to_dicts(markdown: str) -> List[Dict[str, str]]:
    """Convert a markdown table to a list of dictionaries."""
    lines = [line.strip() for line in markdown.splitlines() if line.strip().startswith('|')]
    if len(lines) < 2:
        return []

    cleaned = "\n".join(line.strip('|') for line in lines)
    try:
        df = pd.read_csv(io.StringIO(cleaned), sep='|', engine='python')
        df.columns = [c.strip() for c in df.columns]
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        return df.to_dict(orient='records')
    except Exception:
        return []
