"""
Summarization Tools

This module contains tools for summarizing documents and content.
"""

import logging
from pathlib import Path
import sys
from fastmcp import Context

# Add root path to system path
sys.path.append(str(Path(__file__).parent.parent.parent))
from ui.mcp.doc_reader import doc_reader

# Get logger
logger = logging.getLogger('mcp_server.tools.summarization')

def register_tools(mcp):
    """Register summarization tools with the MCP server"""
    @mcp.tool()
    async def summarize_doc(ctx: Context, doc_name: str) -> str:
        """
        Summarize a specific document by name.

        Call this tool when you need a summary of a document instead of the full content.
        Note: This is a placeholder that currently returns the document with a note to summarize.
        In a real implementation, this would use an LLM or other summarization technique.

        Args:
            ctx: The MCP server provided context.
            doc_name: The name of the document to summarize.
        
        Returns:
            A summary of the document or an error message.
        """
        try:
            content = doc_reader.read_document(doc_name)
            
            if not content:
                return f"Document '{doc_name}' not found."
                
            # In a real implementation, this would use an LLM or other summarization technique
            # For now, this is a placeholder
            return f"Document '{doc_name}' - Please summarize the following content:\n\n{content}"
        except Exception as e:
            logger.error(f"Error summarizing document: {e}")
            return f"Error summarizing document: {e}" 