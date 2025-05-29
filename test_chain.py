import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path('.').resolve()))

async def test_tool_chaining():
    """Test if tool chaining works after the fix"""
    print("Testing tool chaining...")
    
    try:
        from app.streamlined_mcp_server import process_message
        
        # Test query that should trigger tool chaining
        query = "Show me top 5 clients in USA and analyze their revenue trends"
        result = await process_message(query, {'session_id': 'test_chain'})
        
        print(f"\n=== QUERY: {query} ===")
        print(f"Response: {result.get('response', 'No response')[:500]}...")
        print(f"Tools executed: {result.get('tools_executed', [])}")
        print(f"Intent: {result.get('intent', 'No intent')}")
        print(f"Processing time: {result.get('processing_time_ms', 0):.2f}ms")
        
        # Check if multiple tools were executed (indicates chaining)
        tools_count = len(result.get('tools_executed', []))
        if tools_count > 1:
            print(f"\n✅ SUCCESS: Tool chaining worked! {tools_count} tools executed.")
        else:
            print(f"\n⚠️  Only {tools_count} tool executed. No chaining detected.")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_tool_chaining()) 