import os
from pathlib import Path
from typing import List, Dict, Optional
import markdown
from bs4 import BeautifulSoup

class DocReader:
    def __init__(self):
        self.docs_dir = Path(__file__).parent.parent.parent / "docs"
        
    def list_documents(self) -> List[str]:
        """List all available documents"""
        if not self.docs_dir.exists():
            return []
        
        return [
            f.name for f in self.docs_dir.glob("**/*")
            if f.is_file() and f.suffix in ['.md', '.txt']
        ]
        
    def read_document(self, doc_name: str) -> Optional[str]:
        """Read a specific document"""
        doc_path = self.docs_dir / doc_name
        if not doc_path.exists():
            return None
            
        content = doc_path.read_text()
        
        # Convert markdown to plain text if it's a markdown file
        if doc_path.suffix == '.md':
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, features='html.parser')
            content = soup.get_text()
            
        return content
        
    def search_documents(self, query: str) -> List[Dict[str, str]]:
        """Search through documents for relevant content"""
        results = []
        for doc_name in self.list_documents():
            content = self.read_document(doc_name)
            if not content:
                continue
                
            # Simple text matching for now
            # Could be enhanced with more sophisticated search
            if query.lower() in content.lower():
                results.append({
                    "document": doc_name,
                    "content": content,
                    "preview": self._get_preview(content, query)
                })
                
        return results
        
    def _get_preview(self, content: str, query: str, context_chars: int = 200) -> str:
        """Get a preview of the content around the search term"""
        idx = content.lower().find(query.lower())
        if idx == -1:
            return content[:context_chars] + "..."
            
        start = max(0, idx - context_chars // 2)
        end = min(len(content), idx + len(query) + context_chars // 2)
        
        preview = content[start:end]
        if start > 0:
            preview = "..." + preview
        if end < len(content):
            preview = preview + "..."
            
        return preview

# Create global instance
doc_reader = DocReader() 