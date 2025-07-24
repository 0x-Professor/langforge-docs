import { DocSection, FeatureCard, QuickStart } from '@/components/DocSection';
import { CodeBlock } from '@/components/CodeBlock';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Code, Bot, Database, Zap, GitBranch, Settings, ArrowRight, Shield } from 'lucide-react';

export const IntroductionSection = () => {
  const installCode = `# Install LangChain
pip install langchain

# Install with specific providers
pip install langchain-openai langchain-anthropic

# Install LangGraph for agents
pip install langgraph

# Install LangSmith for monitoring
pip install langsmith

# Install LangServe for deployment
pip install langserve[all]`;

  const quickExample = `from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# Initialize your model
model = init_chat_model("gpt-4", model_provider="openai")

# Simple chat
response = model.invoke([HumanMessage(content="Hello, world!")])
print(response.content)`;

  const mcpExample = `# MCP Server Example
from mcp import ServerSession
import asyncio

async def main():
    # Create MCP server
    server = ServerSession()
    
    # Register resources and tools
    @server.resource("documents")
    async def list_documents():
        return ["doc1.txt", "doc2.txt"]
    
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())`;

  return (
    <DocSection
      id="introduction"
      title="LangChain Ecosystem Documentation"
      description="Complete guide to building AI applications with LangChain, LangGraph, LangSmith, LangServe, and the Model Context Protocol (MCP)."
      badges={["v0.3", "Latest", "Production Ready"]}
      externalLinks={[
        { title: "Official LangChain Docs", url: "https://python.langchain.com/docs/introduction/" },
        { title: "MCP Protocol", url: "https://modelcontextprotocol.io/introduction" },
        { title: "GitHub", url: "https://github.com/langchain-ai/langchain" }
      ]}
    >
      {/* Overview */}
      <div className="space-y-6">
        <Card className="shadow-card border-l-4 border-l-primary">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="w-5 h-5 text-primary" />
              <span>What is the LangChain Ecosystem?</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground leading-relaxed">
              LangChain is a comprehensive framework for developing applications powered by large language models (LLMs). 
              It simplifies every stage of the LLM application lifecycle from development to production deployment.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="font-medium">Core Philosophy</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Modular and composable components</li>
                  <li>• Standard interfaces for LLM providers</li>
                  <li>• Production-ready with monitoring</li>
                  <li>• Extensible with custom integrations</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium">Use Cases</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Chatbots and conversational AI</li>
                  <li>• Document analysis and QA</li>
                  <li>• Autonomous agents</li>
                  <li>• Data extraction and processing</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Ecosystem Components */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Ecosystem Components</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <FeatureCard
              icon={<Code className="w-6 h-6" />}
              title="LangChain Core"
              description="Base abstractions, components, and integration packages for building LLM applications."
              features={[
                "Chat models & prompts",
                "Vector stores & embeddings", 
                "Chains & runnables",
                "300+ integrations"
              ]}
            />
            <FeatureCard
              icon={<GitBranch className="w-6 h-6" />}
              title="LangGraph"
              description="Framework for building stateful, multi-actor applications with LLMs and autonomous agents."
              features={[
                "State management",
                "Human-in-the-loop",
                "Streaming support",
                "Agent orchestration"
              ]}
            />
            <FeatureCard
              icon={<Settings className="w-6 h-6" />}
              title="LangSmith"
              description="Platform for tracing, monitoring, and evaluating your LLM applications in production."
              features={[
                "Request tracing",
                "Performance monitoring",
                "A/B testing",
                "Dataset management"
              ]}
            />
            <FeatureCard
              icon={<Database className="w-6 h-6" />}
              title="LangServe"
              description="Deploy LangChain runnables and chains as production-ready REST APIs."
              features={[
                "FastAPI integration",
                "Automatic OpenAPI",
                "WebSocket support",
                "Easy deployment"
              ]}
            />
            <FeatureCard
              icon={<Bot className="w-6 h-6" />}
              title="Model Context Protocol"
              description="Standardized protocol for connecting AI models to different data sources and tools."
              features={[
                "Universal connector",
                "Security best practices",
                "Multi-SDK support",
                "Extensible architecture"
              ]}
            />
            <FeatureCard
              icon={<Shield className="w-6 h-6" />}
              title="Agent Architecture"
              description="Advanced patterns for building multi-agent systems and agent-to-agent communication."
              features={[
                "Multi-agent coordination",
                "Message passing",
                "Shared memory",
                "Distributed processing"
              ]}
            />
          </div>
        </div>

        {/* Quick Start */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Quick Start</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <QuickStart
              title="Get Started with LangChain"
              description="Install and set up your first LangChain application in minutes."
              steps={[
                "Install LangChain and your preferred model provider",
                "Set up your API keys as environment variables", 
                "Initialize a chat model and start building",
                "Explore chains, agents, and advanced features"
              ]}
              codeExample={installCode}
            />
            <QuickStart
              title="Build Your First Agent"
              description="Create a simple conversational agent with LangGraph."
              steps={[
                "Import LangGraph and create a workflow",
                "Define agent nodes and state transitions",
                "Add tools and memory persistence",
                "Deploy with streaming and monitoring"
              ]}
              codeExample={quickExample}
            />
          </div>
        </div>

        {/* Code Examples */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Example Code</h2>
          <div className="space-y-4">
            <CodeBlock
              title="Basic LangChain Setup"
              language="python"
              code={quickExample}
            />
            <CodeBlock
              title="MCP Server Implementation"
              language="python"
              code={mcpExample}
            />
          </div>
        </div>

        {/* Architecture Overview */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle>Architecture Overview</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-gradient-section p-6 rounded-lg">
              <div className="text-center space-y-4">
                <div className="flex justify-center items-center space-x-4">
                  <Badge variant="outline">Your Application</Badge>
                  <ArrowRight className="w-4 h-4" />
                  <Badge variant="outline">LangChain</Badge>
                  <ArrowRight className="w-4 h-4" />
                  <Badge variant="outline">LLM Provider</Badge>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                  <Card className="p-4">
                    <h4 className="font-medium mb-2">Development</h4>
                    <p className="text-sm text-muted-foreground">Build with LangChain components and LangGraph agents</p>
                  </Card>
                  <Card className="p-4">
                    <h4 className="font-medium mb-2">Production</h4>
                    <p className="text-sm text-muted-foreground">Monitor with LangSmith and deploy with LangServe</p>
                  </Card>
                  <Card className="p-4">
                    <h4 className="font-medium mb-2">Integration</h4>
                    <p className="text-sm text-muted-foreground">Connect to data sources via MCP protocol</p>
                  </Card>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </DocSection>
  );
};