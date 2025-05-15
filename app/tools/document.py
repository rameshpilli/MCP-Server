"""
Document Tools

This module contains tools for interacting with documents.
"""

import logging
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add root path to system path
sys.path.append(str(Path(__file__).parent.parent.parent))
from ui.mcp.doc_reader import doc_reader
from app.registry.tools import register_tool

# Get logger
logger = logging.getLogger('mcp_server.tools.document')

def register_tools(mcp):
    """Register document tools with the MCP server"""
    
    @mcp.tool()
    @register_tool(
        name="list_docs",
        description="List all available local documents",
        namespace="document"
    )
    async def list_docs(context: Optional[Dict[str, Any]] = None) -> str:
        """
        List all available local documents.

        Call this tool when you need to see what documents are available for reference.

        Args:
            context: Optional context information.
        
        Returns:
            A formatted string with the list of available documents.
        """
        try:
            docs = doc_reader.list_documents()
            
            if not docs:
                return "No documents found in the local document store."
                
            return "Available Documents:\n\n" + "\n".join([f"- {doc}" for doc in docs])
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return f"Error listing documents: {e}"

    @mcp.tool()
    @register_tool(
        name="read_doc",
        description="Read a specific document by name",
        namespace="document",
        input_schema={
            "doc_name": {"type": "string", "description": "Name of the document to read"}
        }
    )
    async def read_doc(doc_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Read a specific document by name.

        Call this tool when you need to read the full content of a specific document.

        Args:
            doc_name: The name of the document to read.
            context: Optional context information.
        
        Returns:
            The content of the document or an error message.
        """
        try:
            content = doc_reader.read_document(doc_name)
            
            if not content:
                return f"Document '{doc_name}' not found."
                
            return f"Content of '{doc_name}':\n\n{content}"
        except Exception as e:
            logger.error(f"Error reading document: {e}")
            return f"Error reading document: {e}" 