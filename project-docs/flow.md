# CRM MCP Server Architecture and Request Flow

## Overview

The CRM MCP Server is a sophisticated system that combines FastAPI, MCP Bridge, and FastMCP to provide intelligent tool execution and response generation. This document outlines the architecture and request flow.

## System Components

```mermaid
graph LR
    Client[Client/Chainlit UI] --> FastAPI[FastAPI Gateway]
    FastAPI --> Bridge[MCP Bridge]
    Bridge --> Server[MCP Server]
    Server --> Tools[CRM Tools]
    Tools --> Server
    Server --> Bridge
    Bridge --> FastAPI
    FastAPI --> Client
```

## Detailed Request Flow

### 1. Client Request
- **Entry Point**: Chainlit UI or API consumer
- **Request Format**: HTTP POST to `/api/v1/chat`
- **Components**: 
  - `client.py`: Handles client-side communication
  - Session management and authentication
  - Request formatting and validation

### 2. FastAPI Gateway (main.py)
- **Role**: API Gateway and Request Handler
- **Key Functions**:
  ```python
  @app.post("/api/v1/chat")
  async def chat(request: Request):
      # 1. Receives HTTP request
      # 2. Validates request format
      # 3. Calls mcp_bridge.route_request()
      # 4. Processes routing plan
      # 5. Returns formatted response
  ```
- **Responsibilities**:
  - Request validation
  - Error handling
  - Response formatting
  - API documentation (Swagger UI)

### 3. MCP Bridge (mcp_bridge.py)
- **Role**: Intelligent Router and Response Formatter
- **Key Components**:
  ```python
  class MCPBridge:
      async def route_request(self, message: str, context: Dict):
          # 1. Uses Cohere Compass for intent classification
          # 2. Determines which tools to run
          # 3. Returns routing plan
          
      async def execute_tool(self, tool_name: str, params: Dict):
          # 1. Delegates to MCP server
          # 2. Handles tool execution
          
      async def generate_response(self, query: str, results: List):
          # 1. Formats tool results
          # 2. Generates human-readable response
  ```
- **Responsibilities**:
  - Intent classification
  - Tool selection
  - Response formatting
  - Error handling

### 4. MCP Server (mcp_server.py)
- **Role**: Tool Orchestrator and Executor
- **Key Functions**:
  ```python
  class FastMCP:
      # 1. Maintains tool registry
      # 2. Executes tools based on bridge's plan
      # 3. Handles tool chaining
      # 4. Manages tool results
  ```
- **Responsibilities**:
  - Tool registration
  - Tool execution
  - Result management
  - Error handling

### 5. Tool Layer (e.g., clientview_financials.py)
- **Role**: Business Logic Implementation
- **Example Tool**:
  ```python
  @register_tool(namespace="crm")
  async def get_top_clients(ctx: Context, sorting: str = "top", currency: str = "USD", region: str | None = None, focus_list: str | None = None):
      # 1. Executes business logic
      # 2. Returns formatted results
  ```
- **Responsibilities**:
  - Business logic implementation
  - Data processing
  - Result formatting
  - Error handling

## Response Flow

1. **Tools → MCP Server**
   - Tools execute and return results
   - Results are collected and validated

2. **MCP Server → Bridge**
   - Results are passed to the bridge
   - Bridge processes and formats results

3. **Bridge → FastAPI**
   - Formatted response is generated
   - Additional context is added if needed

4. **FastAPI → Client**
   - Final response is returned
   - Error handling and status codes

## Key Insights

1. **Separation of Concerns**
   - FastAPI: API Gateway and Request Handling
   - Bridge: Intelligent Routing and Response Formatting
   - MCP Server: Tool Management and Execution
   - Tools: Business Logic Implementation

2. **Component Roles**
   - **MCP Server** = Brain (tool registry, execution)
   - **Bridge** = Router & Formatter (intent classification, response formatting)
   - **FastAPI** = Door (API gateway, request handling)

3. **Error Handling**
   - Each layer has its own error handling
   - Errors are propagated up the chain
   - Client receives meaningful error messages

## Integration Points

1. **Chainlit Integration**
   - Chainlit UI connects to FastAPI endpoints
   - Uses client.py for communication
   - Handles session management

2. **API Integration**
   - RESTful API endpoints
   - Swagger documentation
   - Authentication and authorization

3. **Tool Integration**
   - Tool registration system
   - Namespace management
   - Parameter validation

## Future Considerations

1. **Scalability**
   - Load balancing
   - Caching
   - Rate limiting

2. **Monitoring**
   - Performance metrics
   - Error tracking
   - Usage statistics

3. **Security**
   - Authentication
   - Authorization
   - Data encryption

## Conclusion

The CRM MCP Server architecture provides a robust and scalable system for handling client requests, executing tools, and generating responses. The clear separation of concerns and well-defined interfaces make it easy to maintain and extend the system. 