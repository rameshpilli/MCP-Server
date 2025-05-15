"""
Reporting Tools

This module contains tools for generating reports.
"""

import logging
from fastmcp import Context
import json
from datetime import datetime

# Get logger
logger = logging.getLogger('mcp_server.tools.reporting')

def register_tools(mcp):
    """Register reporting tools with the MCP server"""
    @mcp.tool()
    async def generate_report(ctx: Context, report_type: str, start_date: str = None, end_date: str = None) -> str:
        """
        Generate a report of the specified type.

        Call this tool when you need to generate a report. Report types include: 'usage', 'performance', 'summary'.

        Args:
            ctx: The MCP server provided context.
            report_type: Type of report to generate ('usage', 'performance', 'summary').
            start_date: Optional start date for the report (format: YYYY-MM-DD).
            end_date: Optional end date for the report (format: YYYY-MM-DD).
        
        Returns:
            A formatted report or an error message.
        """
        try:
            # Get current date if not specified
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                # Default to 30 days before end date
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                start_dt = end_dt.replace(day=1)  # First day of the month
                start_date = start_dt.strftime("%Y-%m-%d")
            
            # This is a placeholder implementation
            report_data = {
                "report_type": report_type,
                "period": f"{start_date} to {end_date}",
                "generated_at": datetime.now().isoformat(),
                "data": {
                    "metrics": {
                        "total_queries": 1234,
                        "successful_queries": 1200,
                        "failed_queries": 34,
                        "average_response_time": "0.5s"
                    },
                    "top_tools": [
                        {"name": "search_docs", "usage_count": 567},
                        {"name": "summarize_doc", "usage_count": 342},
                        {"name": "list_docs", "usage_count": 221}
                    ]
                }
            }
            
            return f"## {report_type.capitalize()} Report ({start_date} to {end_date})\n\n" + \
                   "### Key Metrics\n" + \
                   f"- Total Queries: {report_data['data']['metrics']['total_queries']}\n" + \
                   f"- Success Rate: {report_data['data']['metrics']['successful_queries'] / report_data['data']['metrics']['total_queries'] * 100:.1f}%\n" + \
                   f"- Avg Response Time: {report_data['data']['metrics']['average_response_time']}\n\n" + \
                   "### Top Tools\n" + \
                   "\n".join([f"- {tool['name']}: {tool['usage_count']} uses" for tool in report_data['data']['top_tools']])
                   
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}" 