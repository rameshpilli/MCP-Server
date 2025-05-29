#!/usr/bin/env python3
"""
Test MCP Integration

This script tests the integration between the MCP server, LangChain bridge,
and tool discovery mechanisms to ensure everything is working correctly.

Usage:
    python test_mcp_integration.py
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
import pandas as pd
import unittest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_mcp_integration")

# Import MCP components
try:
    from app.streamlined_mcp_server import mcp, bridge, process_message
    from fastmcp import Context
except ImportError as e:
    logger.error(f"Failed to import MCP components: {e}")
    logger.error("Please ensure you've created the streamlined_mcp_server.py file")
    sys.exit(1)

class TestMCPIntegration(unittest.TestCase):
    """Test suite for MCP integration"""
    
    def setUp(self):
        """Set up test environment"""
        logger.info("Setting up test environment")
        self.test_session_id = "test_session_123"
        self.context = {"session_id": self.test_session_id}
    
    async def async_test_mcp_initialization(self):
        """Test MCP server initialization"""
        logger.info("Testing MCP server initialization")
        
        # Verify MCP server is initialized
        self.assertIsNotNone(mcp, "MCP server should be initialized")
        self.assertEqual(mcp.name, "MCP Server", "MCP server name should be correct")
        
        # Verify bridge is initialized
        self.assertIsNotNone(bridge, "Bridge should be initialized")
        
        # Get available tools
        tools = await mcp.get_tools()
        self.assertIsInstance(tools, dict, "Tools should be a dictionary")
        self.assertTrue(len(tools) > 0, "There should be at least one tool available")
        
        # Check for basic tools
        basic_tools = ["health_check", "server_info", "list_tools", "format_table"]
        for tool_name in basic_tools:
            self.assertIn(tool_name, tools, f"Basic tool {tool_name} should be available")
        
        # Check for financial tools
        financial_tools = ["getTopClients", "getClientRevenue"]
        for tool_name in financial_tools:
            if tool_name in tools:
                logger.info(f"Financial tool {tool_name} is available")
            else:
                logger.warning(f"Financial tool {tool_name} is not available")
        
        logger.info(f"Found {len(tools)} tools: {list(tools.keys())}")
        return tools
    
    async def async_test_tool_execution(self, tools):
        """Test tool execution"""
        logger.info("Testing tool execution")
        
        # Test health check tool
        ctx = Context(self.context)
        health_result = await tools["health_check"].fn(ctx)
        self.assertEqual(health_result, "MCP Server is healthy", "Health check should return healthy status")
        
        # Test format_table tool if available
        if "format_table" in tools:
            test_data = [
                {"name": "Product A", "price": 19.99, "in_stock": True},
                {"name": "Product B", "price": 29.99, "in_stock": False}
            ]
            
            # Test markdown format
            markdown_result = await tools["format_table"].fn(ctx, test_data, format="markdown")
            self.assertIn("| name     | price | in_stock |", markdown_result, "Markdown table should be formatted correctly")
            
            # Test HTML format
            html_result = await tools["format_table"].fn(ctx, test_data, format="html")
            self.assertIn("<table", html_result, "HTML table should be formatted correctly")
            
            # Test CSV format
            csv_result = await tools["format_table"].fn(ctx, test_data, format="csv")
            self.assertIn("name,price,in_stock", csv_result, "CSV table should be formatted correctly")
            
            logger.info("Table formatting works correctly")
        
        # Test financial tools if available
        if "getTopClients" in tools:
            top_clients_result = await tools["getTopClients"].fn(ctx, region="USA", limit=5)
            self.assertIsInstance(top_clients_result, str, "getTopClients should return a string")
            self.assertIn("Client", top_clients_result, "getTopClients should return client information")
            logger.info("Financial tool getTopClients works correctly")
        
        if "getClientRevenue" in tools:
            client_revenue_result = await tools["getClientRevenue"].fn(ctx, client_id=1001)
            self.assertIsInstance(client_revenue_result, str, "getClientRevenue should return a string")
            self.assertIn("Revenue", client_revenue_result, "getClientRevenue should return revenue information")
            logger.info("Financial tool getClientRevenue works correctly")
    
    async def async_test_process_message(self):
        """Test message processing through the bridge"""
        logger.info("Testing message processing")
        
        # Test simple query
        simple_query = "What tools are available?"
        simple_result = await process_message(simple_query, self.context)
        self.assertIsInstance(simple_result, dict, "Process message should return a dictionary")
        self.assertIn("response", simple_result, "Result should contain a response")
        self.assertIn("tools_executed", simple_result, "Result should contain tools_executed")
        
        # Test financial query if tools are available
        financial_query = "Show me the top clients in USA"
        financial_result = await process_message(financial_query, self.context)
        self.assertIsInstance(financial_result, dict, "Process message should return a dictionary")
        self.assertIn("response", financial_result, "Result should contain a response")
        
        # Check if financial tools were executed
        if any("getTopClients" in tool for tool in financial_result.get("tools_executed", [])):
            self.assertIn("Client", financial_result["response"], "Response should contain client information")
            logger.info("Financial query processed correctly")
        else:
            logger.warning("Financial tools were not executed for the query")
    
    async def async_test_langchain_bridge(self):
        """Test LangChain bridge tool chaining"""
        logger.info("Testing LangChain bridge tool chaining")
        
        # Check if we're using LangChain bridge
        if "LangChainBridge" in bridge.__class__.__name__:
            # Test tool chain planning
            query = "Show me the top clients in USA and then get revenue details for the first client"
            plan = await bridge.plan_tool_chain(query, self.context)
            
            self.assertIsInstance(plan, list, "Plan should be a list")
            if plan:
                logger.info(f"LangChain generated a plan with {len(plan)} steps")
                for i, step in enumerate(plan):
                    logger.info(f"Step {i+1}: {step.get('tool')} with parameters {step.get('parameters')}")
                
                # Check if the plan includes the expected tools
                tool_names = [step.get("tool") for step in plan]
                if "getTopClients" in tool_names:
                    logger.info("Plan includes getTopClients tool")
                if "getClientRevenue" in tool_names:
                    logger.info("Plan includes getClientRevenue tool")
            else:
                logger.warning("LangChain did not generate a plan for the query")
        else:
            logger.info("Not using LangChain bridge, skipping tool chain test")
    
    def test_mcp_integration(self):
        """Main test function that runs all async tests"""
        logger.info("Starting MCP integration tests")
        
        # Run async tests
        loop = asyncio.get_event_loop()
        tools = loop.run_until_complete(self.async_test_mcp_initialization())
        loop.run_until_complete(self.async_test_tool_execution(tools))
        loop.run_until_complete(self.async_test_process_message())
        loop.run_until_complete(self.async_test_langchain_bridge())
        
        logger.info("All tests completed successfully")

if __name__ == "__main__":
    # Run the tests
    unittest.main()
