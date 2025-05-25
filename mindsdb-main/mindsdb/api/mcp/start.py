from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Optional, Dict, Any
from dataclasses import dataclass
import argparse
import os

from mcp.server.fastmcp import FastMCP
from mindsdb.api.mysql.mysql_proxy.classes.fake_mysql_proxy import FakeMysqlProxy
from mindsdb.api.executor.data_types.response_type import RESPONSE_TYPE as SQL_RESPONSE_TYPE
from mindsdb.utilities import log
from mindsdb.utilities.config import Config
from mindsdb.interfaces.storage import db

logger = log.getLogger(__name__)


@dataclass
class AppContext:
    db: Any


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context"""
    # Initialize on startup
    db.init()
    try:
        yield AppContext(db=db)
    finally:
        # TODO: We need better way to handle this in storage/db.py
        pass


# Configure server with lifespan
mcp = FastMCP(
    "MindsDB",
    lifespan=app_lifespan,
    dependencies=["mindsdb"]  # Add any additional dependencies
)
# MCP Queries
LISTING_QUERY = "SHOW DATABASES"


@mcp.tool()
def query(query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Execute a SQL query against MindsDB

    Args:
        query: The SQL query to execute
        context: Optional context parameters for the query

    Returns:
        Dict containing the query results or error information
    """

    if context is None:
        context = {}

    logger.debug(f'Incoming MCP query: {query}')

    mysql_proxy = FakeMysqlProxy()
    mysql_proxy.set_context(context)

    try:
        result = mysql_proxy.process_query(query)

        if result.type == SQL_RESPONSE_TYPE.OK:
            return {"type": SQL_RESPONSE_TYPE.OK}

        if result.type == SQL_RESPONSE_TYPE.TABLE:
            return {
                "type": SQL_RESPONSE_TYPE.TABLE,
                "data": result.data.to_lists(json_types=True),
                "column_names": [
                    x["alias"] or x["name"] if "alias" in x else x["name"]
                    for x in result.columns
                ],
            }
        else:
            return {
                "type": SQL_RESPONSE_TYPE.ERROR,
                "error_code": 0,
                "error_message": "Unknown response type"
            }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "type": SQL_RESPONSE_TYPE.ERROR,
            "error_code": 0,
            "error_message": str(e)
        }


@mcp.tool()
def list_databases() -> Dict[str, Any]:
    """
    List all databases in MindsDB along with their tables

    Returns:
        Dict containing the list of databases and their tables
    """

    mysql_proxy = FakeMysqlProxy()

    try:
        result = mysql_proxy.process_query(LISTING_QUERY)
        if result.type == SQL_RESPONSE_TYPE.ERROR:
            return {
                "type": "error",
                "error_code": result.error_code,
                "error_message": result.error_message,
            }

        elif result.type == SQL_RESPONSE_TYPE.OK:
            return {"type": "ok"}

        elif result.type == SQL_RESPONSE_TYPE.TABLE:
            data = result.data.to_lists(json_types=True)
            return data

    except Exception as e:
        return {
            "type": "error",
            "error_code": 0,
            "error_message": str(e),
        }


def start(*args, host: str | None = None, port: int | None = None,
          transport: str | None = None, **kwargs) -> None:
    """Start the MCP server

    Args:
        host: Host to bind to
        port: Port to listen on
        transport: Transport to use when running the server
    """
    config = Config()
    host = host or config['api'].get('mcp', {}).get('host', '127.0.0.1')
    port = port or int(config['api'].get('mcp', {}).get('port', 47337))
    transport = transport or os.getenv("TRANSPORT", "sse")

    logger.info(f"Starting MCP server on {host}:{port} using {transport} transport")
    mcp.settings.host = host
    mcp.settings.port = port

    try:
        mcp.run(transport=transport)
    except Exception as e:
        logger.error(f"Error starting MCP server: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the MindsDB MCP server")
    parser.add_argument("--host", help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to listen on")
    parser.add_argument(
        "--transport",
        help="Transport to use when running the server (e.g. 'sse' or 'stdio')",
    )
    args = parser.parse_args()

    start(host=args.host, port=args.port, transport=args.transport)
