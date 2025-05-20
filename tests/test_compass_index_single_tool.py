import asyncio
import logging
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to import path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

import app.main

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_single_tool")

# Patch registry with just 1 dummy tool
from app.registry import tool_registry
from app.registry.tools import ToolDefinition
tool_registry._tools.clear()
# tool_registry._tools["test:dummy_tool"] = "This is a dummy tool for Compass testing."


tool_registry.register(
    ToolDefinition(
        name="dummy_tool",
        # type="tool",
        description="This is a dummy tool for Compass testing.",
        handler=lambda **kwargs: "Dummy response", 
        namespace="test"
    )
)


# Show what's registered
print(f"✅ Tools in registry: {tool_registry.list_tools()}")

# Run the indexing
from app.config import config
from cohere_compass.clients.compass import CompassClient
from cohere_compass.models.documents import CompassDocument, CompassDocumentMetadata, CompassDocumentChunk
# from cohere_compass.models import Chunk
from uuid import uuid4

async def index_single_tool():
    logger.info("Creating Compass client...")
    client = CompassClient(
        index_url=config.COHERE_SERVER_URL,
        bearer_token=config.COHERE_SERVER_BEARER_TOKEN
    )

    index_name = config.COHERE_INDEX_NAME
    tool_name = "test:dummy_tool"
    namespace, name = tool_name.split(":")
    doc_id = f"tool-{namespace}-{name}"

    doc = CompassDocument(
        metadata=CompassDocumentMetadata(document_id=doc_id, filename=f"{name}.md"),
        content={
            "text": f"{name} - This is a dummy tool for Compass testing.",
            "type": "tool",
            "name": name,
            "namespace": namespace,
            "description": "This is a dummy tool for Compass testing."
        },
        chunks=[
            CompassDocumentChunk(
                chunk_id=str(uuid4()),
                sort_id="0",
                document_id=doc_id,
                parent_document_id=doc_id,
                content={"text": f"{name} - This is a dummy tool for Compass testing."}
            )
        ]
    )

    logger.info(f"📄 Inserting doc: {doc_id}")
    logger.debug(f"Doc content: {doc.content}")

    try:
        # client.insert_doc(index_name=index_name, docs=iter([doc]))
        client.insert_doc(index_name=index_name, doc=doc)

        logger.info("✅ Document inserted successfully")
    except Exception as e:
        logger.error(f"❌ Insert failed: {e}")

if __name__ == "__main__":
    asyncio.run(index_single_tool()) 
