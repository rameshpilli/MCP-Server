import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix the import path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

from app.config import config
from cohere_compass.clients.compass import CompassClient

def test_compass_search():
    """Test searching in the Compass index"""
    if not config.COHERE_SERVER_URL or not config.COHERE_SERVER_BEARER_TOKEN:
        print("❌ Compass API URL or Bearer Token not configured in .env file")
        return
    
    # Initialize client
    client = CompassClient(
        index_url=config.COHERE_SERVER_URL,
        bearer_token=config.COHERE_SERVER_BEARER_TOKEN
    )
    
    # Test queries
    test_queries = [
        "list all documents",
        "read a document", 
        "analyze sentiment",
        "server information"
    ]
    
    print(f"🔍 Testing searches in index: {config.COHERE_INDEX_NAME}\n")
    
    for query in test_queries:
        try:
            print(f"Query: '{query}'")
            result = client.search_chunks(
                index_name=config.COHERE_INDEX_NAME,
                query=query,
                top_k=3
            )
            
            if result.hits:
                print(f"Found {len(result.hits)} results:")
                for i, hit in enumerate(result.hits, 1):
                    print(f"  {i}. Score: {hit.score:.3f}")
                    # Try different ways to access the content
                    content_text = None
                    tool_name = None
                    
                    # Check different attributes
                    if hasattr(hit, 'content'):
                        content_text = str(hit.content)[:200]
                        if isinstance(hit.content, dict):
                            tool_name = hit.content.get('name', 'Unknown')
                    
                    if hasattr(hit, 'chunk') and hasattr(hit.chunk, 'content'):
                        content_text = str(hit.chunk.content)[:200]
                        if isinstance(hit.chunk.content, dict):
                            tool_name = hit.chunk.content.get('name', 'Unknown')
                    
                    if hasattr(hit, 'document') and hasattr(hit.document, 'content'):
                        content_text = str(hit.document.content)[:200]
                        if isinstance(hit.document.content, dict):
                            tool_name = hit.document.content.get('name', 'Unknown')
                    
                    # Print what we found
                    if tool_name:
                        print(f"     Tool: {tool_name}")
                    if content_text:
                        print(f"     Content: {content_text}")
                    
                    # Debug: print all attributes of hit
                    print(f"     Hit attributes: {dir(hit)}")
                    
            else:
                print("  No results found")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error searching for '{query}': {e}")
            print("-" * 50)

if __name__ == "__main__":
    test_compass_search() 
