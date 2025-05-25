from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from .base import BaseRegistry

class PromptTemplate(BaseModel):
    """Definition of a prompt template"""
    name: str
    description: str
    template: str
    namespace: str = "default"
    variables: List[str] = []
    metadata: Dict[str, Any] = {}
    
    @property
    def full_name(self) -> str:
        """Get the fully qualified name (namespace:name)"""
        return f"{self.namespace}:{self.name}"
    
    def format(self, **kwargs) -> str:
        """Format the prompt template with the given variables"""
        result = self.template
        for var in self.variables:
            if var in kwargs:
                result = result.replace(f"{{{var}}}", str(kwargs[var]))
        return result

class PromptRegistry(BaseRegistry[PromptTemplate]):
    """Registry for prompt templates."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def _prompts(self) -> Dict[str, PromptTemplate]:
        return self._items

    @_prompts.setter
    def _prompts(self, value: Dict[str, PromptTemplate]) -> None:  # pragma: no cover - legacy
        self._items = value

    # Aliases for backward compatibility
    get_prompt = BaseRegistry.get
    list_prompts = BaseRegistry.list_items

# Create global registry instance
registry = PromptRegistry()

def register_prompt(name: str, description: str, template: str, 
                   variables: List[str] = None, namespace: str = "default",
                   metadata: Dict[str, Any] = None):
    """
    Function for registering prompt templates
    
    Example:
        summarization_prompt = register_prompt(
            "document_summarization",
            "Prompt for summarizing a document",
            "Please summarize the following document:\n\n{document_content}",
            variables=["document_content"],
            namespace="summarization"
        )
    """
    prompt = PromptTemplate(
        name=name,
        description=description,
        template=template,
        namespace=namespace,
        variables=variables or [],
        metadata=metadata or {}
    )
    registry.register(prompt)
    return prompt
