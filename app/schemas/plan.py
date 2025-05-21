from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class PlanStep(BaseModel):
    """Schema for a single tool execution step in an LLM generated plan."""

    tool: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    use_context: Dict[str, str] = Field(default_factory=dict)
    output_context: List[str] = Field(default_factory=list)
    query: Optional[str] = None

