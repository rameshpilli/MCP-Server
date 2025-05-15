import asyncio
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_mcp_tools_sentiment")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the MCP server
from app.mcp_server import mcp

async def test_sentiment_tools():
    """Test the MCP sentiment analysis tools functionality"""
    logger.info("Testing MCP sentiment analysis tools...")
    
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
    
    # Test sentiment analysis texts
    test_texts = [
        "I absolutely love the new product features! The interface is intuitive and the performance is amazing.",
        "The service was terrible. I had to wait for hours and the staff was unhelpful.",
        "The quarterly results were mixed. While revenue increased, there were some concerns about costs.",
        "The new security update is good, but it could use some improvements in the user experience.",
        "I'm neutral about the changes. They don't seem to make much difference either way."
    ]
    
    # Test sentiment tools
    sentiment_tools = {
        'analyze_sentiment': {
            'description': 'Analyze sentiment of text',
            'param_name': 'text'
        },
        'analyze_client_sentiment': {
            'description': 'Analyze client feedback sentiment',
            'param_name': 'feedback'
        }
    }
    
    # Test each sentiment tool
    for tool_name, tool_info in sentiment_tools.items():
        if tool_name in tools:
            logger.info(f"\n=== Testing {tool_name} ===")
            logger.info(f"Description: {tool_info['description']}")
            
            # Test each text
            for text in test_texts:
                logger.info(f"\nTesting text: '{text[:100]}...'")
                
                try:
                    # Get the tool function
                    tool_fn = tools[tool_name].fn
                    
                    # Execute the sentiment analysis
                    params = {tool_info['param_name']: text}
                    logger.info(f"Executing with parameters: {params}")
                    result = await tool_fn(ctx, **params)
                    
                    # Display the result
                    logger.info(f"Sentiment analysis result preview: {result[:300]}...")
                    
                    # Try to parse and display structured results if available
                    try:
                        if isinstance(result, dict):
                            logger.info("Sentiment analysis details:")
                            for key, value in result.items():
                                logger.info(f"  {key}: {value}")
                            
                            # Try to interpret the sentiment
                            if 'sentiment' in result:
                                sentiment = result['sentiment'].lower()
                                if 'positive' in sentiment:
                                    logger.info("  Overall sentiment: Positive")
                                elif 'negative' in sentiment:
                                    logger.info("  Overall sentiment: Negative")
                                elif 'neutral' in sentiment:
                                    logger.info("  Overall sentiment: Neutral")
                                else:
                                    logger.info(f"  Overall sentiment: {result['sentiment']}")
                            
                            if 'score' in result:
                                score = float(result['score'])
                                if score > 0.6:
                                    logger.info("  Sentiment score: Strongly positive")
                                elif score > 0.3:
                                    logger.info("  Sentiment score: Moderately positive")
                                elif score > -0.3:
                                    logger.info("  Sentiment score: Neutral")
                                elif score > -0.6:
                                    logger.info("  Sentiment score: Moderately negative")
                                else:
                                    logger.info("  Sentiment score: Strongly negative")
                        elif isinstance(result, (list, tuple)):
                            logger.info(f"Found {len(result)} sentiment aspects")
                            for i, item in enumerate(result[:3], 1):  # Show first 3 aspects
                                logger.info(f"\nAspect {i}:")
                                if isinstance(item, dict):
                                    for key, value in item.items():
                                        logger.info(f"  {key}: {value}")
                                else:
                                    logger.info(f"  {item}")
                    except Exception as e:
                        logger.warning(f"Could not parse structured results: {e}")
                    
                    logger.info("Sentiment analysis completed successfully")
                except Exception as e:
                    logger.error(f"Error executing sentiment analysis: {e}")
                
                logger.info("-" * 30)
        else:
            logger.warning(f"Sentiment tool '{tool_name}' not found")
        
        logger.info("=" * 50)

if __name__ == "__main__":
    logger.info("Starting MCP sentiment analysis tools tests...")
    asyncio.run(test_sentiment_tools())
    logger.info("Tests completed.") 