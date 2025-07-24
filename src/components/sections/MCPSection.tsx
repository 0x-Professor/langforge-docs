import { DocSection, FeatureCard, QuickStart } from '@/components/DocSection';
import { CodeBlock } from '@/components/CodeBlock';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Bot, Database, Code, Settings, Zap, Shield, Network } from 'lucide-react';

export const MCPSection = () => {
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
import { createServer, createClient } from '@modelcontextprotocol/sdk';

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
    <DocSection
      id="mcp"
      title="Model Context Protocol (MCP)"
      description="Standardized protocol for connecting AI models to data sources and tools, like USB-C for AI applications."
      badges={["Protocol", "Universal Connector", "Multi-SDK"]}
      externalLinks={[
        { title: "MCP Documentation", url: "https://modelcontextprotocol.io/" },
        { title: "GitHub Spec", url: "https://github.com/modelcontextprotocol/specification" },
        { title: "Examples", url: "https://modelcontextprotocol.io/examples" }
      ]}
    >
      <div className="space-y-8">
        {/* Core Concepts */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Core Concepts</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <FeatureCard
              icon={<Network className="w-6 h-6" />}
              title="Client-Server Architecture"
              description="MCP follows a clear client-server model where hosts connect to multiple servers."
              features={[
                "1:1 client-server connections",
                "Bidirectional communication",
                "Transport layer abstraction",
                "Protocol versioning"
              ]}
            />
            <FeatureCard
              icon={<Database className="w-6 h-6" />}
              title="Resources"
              description="Expose data and content from servers to LLMs through standardized resource endpoints."
              features={[
                "URI-based addressing",
                "Metadata and MIME types",
                "Dynamic resource listing",
                "Access control"
              ]}
            />
            <FeatureCard
              icon={<Zap className="w-6 h-6" />}
              title="Tools"
              description="Enable LLMs to perform actions through your server with function calling."
              features={[
                "JSON Schema definitions",
                "Argument validation",
                "Error handling",
                "Async execution"
              ]}
            />
            <FeatureCard
              icon={<Bot className="w-6 h-6" />}
              title="Prompts"
              description="Create reusable prompt templates and workflows that LLMs can invoke."
              features={[
                "Template variables",
                "Dynamic prompts",
                "Prompt chaining",
                "Context injection"
              ]}
            />
            <FeatureCard
              icon={<Code className="w-6 h-6" />}
              title="Sampling"
              description="Allow servers to request completions from LLMs for advanced workflows."
              features={[
                "Model agnostic requests",
                "Streaming support",
                "Configuration options",
                "Result caching"
              ]}
            />
            <FeatureCard
              icon={<Shield className="w-6 h-6" />}
              title="Security"
              description="Built-in security practices for data protection and access control."
              features={[
                "Sandboxed execution",
                "Permission controls",
                "Audit logging",
                "Secure transport"
              ]}
            />
          </div>
        </div>

        {/* Architecture Overview */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle>MCP Architecture</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-gradient-section p-6 rounded-lg">
              <div className="space-y-4">
                <div className="text-center">
                  <h4 className="font-medium mb-2">Data Flow</h4>
                  <div className="flex justify-center items-center space-x-4 text-sm">
                    <span className="px-3 py-1 bg-primary/10 rounded">Host Application</span>
                    <span>←→</span>
                    <span className="px-3 py-1 bg-primary/10 rounded">MCP Client</span>
                    <span>←→</span>
                    <span className="px-3 py-1 bg-primary/10 rounded">MCP Server</span>
                    <span>←→</span>
                    <span className="px-3 py-1 bg-primary/10 rounded">Data Source</span>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center">
                    <h5 className="font-medium">Hosts</h5>
                    <p className="text-sm text-muted-foreground">Claude Desktop, IDEs, AI Tools</p>
                  </div>
                  <div className="text-center">
                    <h5 className="font-medium">Servers</h5>
                    <p className="text-sm text-muted-foreground">File system, databases, APIs</p>
                  </div>
                  <div className="text-center">
                    <h5 className="font-medium">Transport</h5>
                    <p className="text-sm text-muted-foreground">stdio, HTTP, WebSocket</p>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Code Examples */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Implementation Examples</h2>
          <Tabs defaultValue="server" className="w-full">
            <TabsList className="grid w-full grid-cols-5">
              <TabsTrigger value="server">MCP Server</TabsTrigger>
              <TabsTrigger value="client">MCP Client</TabsTrigger>
              <TabsTrigger value="python">Python SDK</TabsTrigger>
              <TabsTrigger value="javascript">JavaScript SDK</TabsTrigger>
              <TabsTrigger value="agent">Agent Integration</TabsTrigger>
            </TabsList>
            
            <TabsContent value="server" className="space-y-4">
              <CodeBlock
                title="File System MCP Server"
                language="python"
                code={serverCode}
              />
            </TabsContent>
            
            <TabsContent value="client" className="space-y-4">
              <CodeBlock
                title="MCP Client Implementation"
                language="python"
                code={clientCode}
              />
            </TabsContent>
            
            <TabsContent value="python" className="space-y-4">
              <CodeBlock
                title="Python SDK Usage"
                language="python"
                code={pythonSdkCode}
              />
            </TabsContent>
            
            <TabsContent value="javascript" className="space-y-4">
              <CodeBlock
                title="JavaScript/TypeScript SDK"
                language="typescript"
                code={jsSdkCode}
              />
            </TabsContent>
            
            <TabsContent value="agent" className="space-y-4">
              <CodeBlock
                title="LangGraph Agent with MCP"
                language="python"
                code={agentIntegrationCode}
              />
            </TabsContent>
          </Tabs>
        </div>

        {/* SDK Support */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Multi-Language SDK Support</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Official SDKs</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <Code className="w-5 h-5 mt-0.5 text-primary" />
                    <div>
                      <h4 className="font-medium">Python SDK</h4>
                      <p className="text-sm text-muted-foreground">Full-featured SDK with async support</p>
                      <code className="text-xs">pip install mcp</code>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Code className="w-5 h-5 mt-0.5 text-primary" />
                    <div>
                      <h4 className="font-medium">TypeScript SDK</h4>
                      <p className="text-sm text-muted-foreground">Node.js and browser support</p>
                      <code className="text-xs">npm install @modelcontextprotocol/sdk</code>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Code className="w-5 h-5 mt-0.5 text-primary" />
                    <div>
                      <h4 className="font-medium">Go SDK</h4>
                      <p className="text-sm text-muted-foreground">High-performance server implementations</p>
                      <code className="text-xs">go get github.com/modelcontextprotocol/go-sdk</code>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Transport Options</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <h4 className="font-medium">stdio</h4>
                    <p className="text-sm text-muted-foreground">Standard input/output for local processes</p>
                  </div>
                  <div>
                    <h4 className="font-medium">HTTP/HTTPS</h4>
                    <p className="text-sm text-muted-foreground">REST API over HTTP with JSON</p>
                  </div>
                  <div>
                    <h4 className="font-medium">WebSocket</h4>
                    <p className="text-sm text-muted-foreground">Real-time bidirectional communication</p>
                  </div>
                  <div>
                    <h4 className="font-medium">Custom Transports</h4>
                    <p className="text-sm text-muted-foreground">Implement your own transport layer</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Best Practices */}
        <Card className="shadow-card border-l-4 border-l-primary">
          <CardHeader>
            <CardTitle>MCP Best Practices</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h4 className="font-medium">Server Development</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Implement proper error handling and validation</li>
                  <li>• Use descriptive resource URIs and tool names</li>
                  <li>• Document your resources and tools clearly</li>
                  <li>• Implement security and access controls</li>
                </ul>
              </div>
              <div className="space-y-3">
                <h4 className="font-medium">Client Integration</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Handle connection failures gracefully</li>
                  <li>• Implement proper retry mechanisms</li>
                  <li>• Cache server capabilities when appropriate</li>
                  <li>• Monitor performance and resource usage</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </DocSection>
  );
};