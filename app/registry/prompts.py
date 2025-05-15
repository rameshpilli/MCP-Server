from typing import Dict, Any, List, Optional
from pydantic import BaseModel

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

class PromptRegistry:
    """Registry for prompt templates"""
    def __init__(self):
        self._prompts: Dict[str, PromptTemplate] = {}
    
    def register(self, prompt: PromptTemplate):
        """Register a new prompt template"""
        self._prompts[prompt.full_name] = prompt
    
    def get_prompt(self, name: str, namespace: Optional[str] = None) -> PromptTemplate:
        """
        Get a prompt template by name, optionally with namespace
        
        Args:
            name: Prompt template name
            namespace: Optional namespace. If not provided, will look for fully qualified name
                       or search in the default namespace.
        """
        # Check if the name already contains a namespace
        if ":" in name:
            if name not in self._prompts:
                raise KeyError(f"Prompt template {name} not found")
            return self._prompts[name]
        
        # Use provided namespace or default
        full_name = f"{namespace or 'default'}:{name}"
        if full_name not in self._prompts:
            raise KeyError(f"Prompt template {full_name} not found")
        return self._prompts[full_name]
    
    def list_prompts(self, namespace: Optional[str] = None) -> Dict[str, str]:
        """
        List all registered prompt templates and their descriptions
        
        Args:
            namespace: Optional namespace to filter by
        """
        if namespace:
            return {
                name.split(":", 1)[1]: prompt.description 
                for name, prompt in self._prompts.items() 
                if name.startswith(f"{namespace}:")
            }
        return {name: prompt.description for name, prompt in self._prompts.items()}
    
    def list_namespaces(self) -> Dict[str, int]:
        """List all namespaces and prompt template count"""
        namespaces = {}
        for name in self._prompts.keys():
            namespace = name.split(":", 1)[0]
            namespaces[namespace] = namespaces.get(namespace, 0) + 1
        return namespaces

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