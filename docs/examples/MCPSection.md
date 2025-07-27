# MCPSection

# const MCPSection = () => {
  const serverCode = `from mcp import ServerSession, Resource, Tool
import asyncio
import json

class FileSystemServer:
    def __init__(self):
        self.session = ServerSession()
        self.setup_resources()
        self.setup_tools()
    
    def setup_resources(self):
        @self.session.resource("file://{path}")
        async def read_file(path: str) -> Resource:
            """Read a file from the filesystem."""
            try:
                with open(path, 'r') as f:
                    content = f.read()
                return Resource(
                    uri=f"file://{path}",
                    name=f"File: {path}",
                    mimeType="text/plain",
                    text=content
                )
            except FileNotFoundError:
                raise ValueError(f"File not found: {path}")
    
    def setup_tools(self):
        @self.session.tool("list_files")
        async def list_files(directory: str = ".") -> str:
            """List files in a directory."""
            import os
            try:
                files = os.listdir(directory)
                return json.dumps({"files": files})
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @self.session.tool("write_file")
        async def write_file(path: str, content: str) -> str:
            """Write content to a file."""
            try:
                with open(path, 'w') as f:
                    f.write(content)
                return json.dumps({"success": True, "message": f"File written: {path}"})
            except Exception as e:
                return json.dumps({"error": str(e)})
    
    async def run(self):
        await self.session.run()

# Run the server
if __name__ == "__main__":
    server = FileSystemServer()
    asyncio.run(server.run())`;

  const clientCode = `from mcp import ClientSession
import asyncio

class MCPClient:
    def __init__(self, server_url: str):
        self.session = ClientSession(server_url)
    
    async def connect(self):
        """Connect to MCP server."""
        await self.session.connect()
        print("Connected to MCP server")
    
    async def list_resources(self):
        """List available resources."""
        resources = await self.session.list_resources()
        for resource in resources:
            print(f"Resource: {resource.uri} - {resource.name}")
        return resources
    
    async def list_tools(self):
        """List available tools."""
        tools = await self.session.list_tools()
        for tool in tools:
            print(f"Tool: {tool.name} - {tool.description}")
        return tools
    
    async def read_resource(self, uri: str):
        """Read a specific resource."""
        resource = await self.session.read_resource(uri)
        print(f"Resource content: {resource.text}")
        return resource
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool with arguments."""
        result = await self.session.call_tool(tool_name, arguments)
        print(f"Tool result: {result}")
        return result
    
    async def disconnect(self):
        """Disconnect from server."""
        await self.session.disconnect()
        print("Disconnected from MCP server")

# Usage example
async def main():
    client = MCPClient("stdio://path/to/server")
    
    try:
        await client.connect()
        
        # List available resources and tools
        await client.list_resources()
        await client.list_tools()
        
        # Use tools
        await client.call_tool("list_files", {"directory": "/home/user"})
        await client.call_tool("write_file", {
            "path": "test.txt",
            "content": "Hello from MCP!"
        })
        
        # Read resources
        await client.read_resource("file://test.txt")
        
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())`;

  const pythonSdkCode = `# Python SDK Example
from mcp import create_server, create_client
import asyncio

# Server setup
@create_server()
class DatabaseServer:
    @server.resource("database://{table}")
    async def get_table_data(self, table: str):
        # Connect to database and fetch data
        return {"data": f"Data from {table}"}
    
    @server.tool("query")
    async def execute_query(self, sql: str):
        # Execute SQL query
        return {"result": "Query executed successfully"}

# Client setup
async def use_database_server():
    client = create_client("stdio://database_server")
    await client.connect()
    
    # Query database
    result = await client.call_tool("query", {"sql": "SELECT * FROM users"})
    return result`;

  const jsSdkCode = `// JavaScript/TypeScript SDK Example


// Server setup
const server = createServer({
  name: 'web-scraper',
  version: '1.0.0'
});

server.addResource('webpage://{url}', async (url) => {
  const response = await fetch(url);
  const html = await response.text();
  return {
    uri: \`webpage://\${url}\`,
    name: \`Webpage: \${url}\`,
    mimeType: 'text/html',
    text: html
  };
});

server.addTool('scrape', async ({ url, selector }) => {
  // Scrape specific elements
  return { content: 'Scraped content' };
});

// Client usage
const client = createClient('http://localhost:3000/mcp');
await client.connect();

const result = await client.callTool('scrape', {
  url: 'https://example.com',
  selector: '.content'
});`;

  const agentIntegrationCode = `from langchain_core.tools import tool
from mcp import ClientSession
import asyncio

class MCPToolAdapter:
    def __init__(self, mcp_client: ClientSession):
        self.client = mcp_client
    
    async def get_mcp_tools(self):
        """Convert MCP tools to LangChain tools."""
        mcp_tools = await self.client.list_tools()
        langchain_tools = []
        
        for mcp_tool in mcp_tools:
            @tool(name=mcp_tool.name, description=mcp_tool.description)
            async def langchain_tool(**kwargs):
                return await self.client.call_tool(mcp_tool.name, kwargs)
            
            langchain_tools.append(langchain_tool)
        
        return langchain_tools

# Agent with MCP integration
from langgraph import StateGraph
from langchain_openai import ChatOpenAI

async def create_mcp_agent():
    # Connect to MCP server
    mcp_client = ClientSession("stdio://file_server")
    await mcp_client.connect()
    
    # Get MCP tools
    adapter = MCPToolAdapter(mcp_client)
    mcp_tools = await adapter.get_mcp_tools()
    
    # Create agent with MCP tools
    model = ChatOpenAI(model="gpt-4")
    agent = create_react_agent(model, mcp_tools)
    
    return agent, mcp_client

# Usage
async def main():
    agent, mcp_client = await create_mcp_agent()
    
    try:
        response = await agent.invoke({
            "messages": [("human", "List files in the current directory and read README.md")]
        })
        print(response)
    finally:
        await mcp_client.disconnect()

asyncio.run(main())`;

  return (
    
      
        {/* Core Concepts */}
        
          
Core Concepts

          
            }
              title="Client-Server Architecture"
              description="MCP follows a clear client-server model where hosts connect to multiple servers."
              features={[
                "1:1 client-server connections",
                "Bidirectional communication",
                "Transport layer abstraction",
                "Protocol versioning"
              ]}
            />
            }
              title="Resources"
              description="Expose data and content from servers to LLMs through standardized resource endpoints."
              features={[
                "URI-based addressing",
                "Metadata and MIME types",
                "Dynamic resource listing",
                "Access control"
              ]}
            />
            }
              title="Tools"
              description="Enable LLMs to perform actions through your server with function calling."
              features={[
                "JSON Schema definitions",
                "Argument validation",
                "Error handling",
                "Async execution"
              ]}
            />
            }
              title="Prompts"
              description="Create reusable prompt templates and workflows that LLMs can invoke."
              features={[
                "Template variables",
                "Dynamic prompts",
                "Prompt chaining",
                "Context injection"
              ]}
            />
            }
              title="Sampling"
              description="Allow servers to request completions from LLMs for advanced workflows."
              features={[
                "Model agnostic requests",
                "Streaming support",
                "Configuration options",
                "Result caching"
              ]}
            />
            
}
              title="Security"
              description="Built-in security practices for data protection and access control."
              features={[
                "Sandboxed execution",
                "Permission controls",
                "Audit logging",
                "Secure transport"
              ]}
            />

        

        {/* Architecture Overview */}
        
          
            
MCP Architecture

          
          
            
              
                
                  
Data Flow

                  
                    
Host Application

                    
←→

                    
MCP Client

                    
←→

                    
MCP Server

                    
←→

                    
Data Source

                  


                
                  
                    
Hosts

                    
Claude Desktop, IDEs, AI Tools

                  
                  
                    
Servers

                    
File system, databases, APIs

                  
                  
                    
Transport

                    
stdio, HTTP, WebSocket

                  


              


          



        {/* Code Examples */}
        
          
Implementation Examples

          
            
              
MCP Server

              
MCP Client

              
Python SDK

              
JavaScript SDK

              
Agent Integration

            
            
            
              


            
            
              


            
            
              


            
            
              


            
            
              


          



        {/* SDK Support */}
        
          
Multi-Language SDK Support

          
            
              
                
Official SDKs

              
              
                
                  
                    
                    
                      
Python SDK

                      
Full-featured SDK with async support

                      
pip install mcp

                    


                  
                    
                    
                      
TypeScript SDK

                      
Node.js and browser support

                      
npm install @modelcontextprotocol/sdk

                    


                  
                    
                    
                      
Go SDK

                      
High-performance server implementations

                      
go get github.com/modelcontextprotocol/go-sdk

                    


                


            

            
              
                
Transport Options

              
              
                
                  
                    
stdio

                    
Standard input/output for local processes

                  
                  
                    
HTTP/HTTPS

                    
REST API over HTTP with JSON

                  
                  
                    
WebSocket

                    
Real-time bidirectional communication

                  
                  
                    
Custom Transports

                    
Implement your own transport layer

                  


              


          



        {/* Best Practices */}
        
          
            
MCP Best Practices

          
          
            
              
                
Server Development

                
                  
• Implement proper error handling and validation

                  
• Use descriptive resource URIs and tool names

                  
• Document your resources and tools clearly

                  
• Implement security and access controls

                


              
                
Client Integration

                
                  
• Handle connection failures gracefully

                  
• Implement proper retry mechanisms

                  
• Cache server capabilities when appropriate

                  
• Monitor performance and resource usage

                


            


        


    
  );
};