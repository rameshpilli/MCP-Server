import asyncio
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_mcp_tools_summarize")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the MCP server
from app.mcp_server import mcp

async def test_summarize_tools():
    """Test the MCP summarization tools functionality"""
    logger.info("Testing MCP summarization tools...")
    
    # Get all registered tools
    tools = await mcp.get_tools()
    logger.info(f"Found {len(tools)} registered tools")
    
    # Create a mock context
    class MockContext:
        def __init__(self):
            class MockRequestContext:
                def __init__(self):
                    self.lifespan_context = None
                    self.context = {}
            self.request_context = MockRequestContext()
    
    ctx = MockContext()
    
    # Test summarization texts
    test_texts = [
        """The quarterly financial report shows strong growth in key metrics. Revenue increased by 15% compared to the previous quarter, driven by higher sales in the enterprise segment. Operating expenses were well controlled, resulting in a 20% increase in operating profit. The company's cash position remains strong at $500 million, up from $450 million last quarter. Customer acquisition costs decreased by 5%, while customer retention rates improved to 95%. The board has approved a new dividend policy, with the first payment scheduled for next month.""",
        
        """The product development team has made significant progress on the new platform. Key features implemented include real-time collaboration, advanced analytics, and improved security measures. User testing feedback has been positive, with particular praise for the intuitive interface. However, some performance issues were identified during stress testing, which the team is currently addressing. The launch date has been set for Q3, with a phased rollout planned across different regions. Marketing materials are being prepared, and the sales team has begun initial customer outreach.""",
        
        """The annual security audit revealed several areas requiring attention. While the core infrastructure remains secure, some vulnerabilities were found in the third-party integrations. The team has prioritized addressing these issues, with most critical patches already deployed. New security protocols have been implemented for data access, and additional monitoring tools have been installed. Employee security training has been updated to cover emerging threats. The audit committee has recommended quarterly security reviews going forward."""
    ]
    
    # Test summarization tools
    summarize_tools = {
        'summarize_text': {
            'description': 'Summarize text content',
            'param_name': 'text',
            'optional_params': {
                'max_length': 100,
                'focus_points': ['key points', 'conclusions']
            }
        },
        'summarize_document': {
            'description': 'Summarize document content',
            'param_name': 'doc_name',
            'optional_params': {
                'format': 'bullet_points',
                'include_metadata': True
            }
        }
    }
    
    # Test each summarization tool
    for tool_name, tool_info in summarize_tools.items():
        if tool_name in tools:
            logger.info(f"\n=== Testing {tool_name} ===")
            logger.info(f"Description: {tool_info['description']}")
            
            # Test each text
            for text in test_texts:
                logger.info(f"\nTesting text: '{text[:100]}...'")
                
                try:
                    # Get the tool function
                    tool_fn = tools[tool_name].fn
                    
                    # Prepare parameters
                    params = {tool_info['param_name']: text}
                    if 'optional_params' in tool_info:
                        params.update(tool_info['optional_params'])
                    
                    # Execute the summarization
                    logger.info(f"Executing with parameters: {params}")
                    result = await tool_fn(ctx, **params)
                    
                    # Display the result
                    logger.info(f"Summarization result preview: {result[:300]}...")
                    
                    # Try to parse and display structured results if available
                    try:
                        if isinstance(result, dict):
                            logger.info("Summarization details:")
                            for key, value in result.items():
                                if key == 'summary':
                                    logger.info(f"\nSummary:")
                                    logger.info(f"  {value}")
                                elif key == 'key_points':
                                    logger.info(f"\nKey points:")
                                    if isinstance(value, list):
                                        for i, point in enumerate(value, 1):
                                            logger.info(f"  {i}. {point}")
                                    else:
                                        logger.info(f"  {value}")
                                elif key == 'metadata':
                                    logger.info(f"\nMetadata:")
                                    for meta_key, meta_value in value.items():
                                        logger.info(f"  {meta_key}: {meta_value}")
                                else:
                                    logger.info(f"  {key}: {value}")
                        elif isinstance(result, (list, tuple)):
                            logger.info(f"\nSummary points:")
                            for i, point in enumerate(result, 1):
                                logger.info(f"  {i}. {point}")
                        else:
                            logger.info(f"\nSummary: {result}")
                    except Exception as e:
                        logger.warning(f"Could not parse structured results: {e}")
                    
                    logger.info("Summarization completed successfully")
                except Exception as e:
                    logger.error(f"Error executing summarization: {e}")
                
                logger.info("-" * 30)
        else:
            logger.warning(f"Summarization tool '{tool_name}' not found")
        
        logger.info("=" * 50)

if __name__ == "__main__":
    logger.info("Starting MCP summarization tools tests...")
    asyncio.run(test_summarize_tools())
    logger.info("Tests completed.") 