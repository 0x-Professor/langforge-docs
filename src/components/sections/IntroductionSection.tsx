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

        {/* Getting Started Guide */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Getting Started Guide</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Installation & Setup</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <h4 className="font-medium mb-2">Core Installation</h4>
                    <code className="text-sm bg-muted p-2 rounded block">pip install langchain langchain-openai</code>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">Agent Framework</h4>
                    <code className="text-sm bg-muted p-2 rounded block">pip install langgraph</code>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">Monitoring & Evaluation</h4>
                    <code className="text-sm bg-muted p-2 rounded block">pip install langsmith</code>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">API Deployment</h4>
                    <code className="text-sm bg-muted p-2 rounded block">pip install langserve[all]</code>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">Model Context Protocol</h4>
                    <code className="text-sm bg-muted p-2 rounded block">pip install mcp</code>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Learning Path</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                      <span className="text-xs font-medium text-primary">1</span>
                    </div>
                    <div>
                      <h4 className="font-medium">Start with LangChain Basics</h4>
                      <p className="text-sm text-muted-foreground">Learn prompts, chat models, and simple chains</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                      <span className="text-xs font-medium text-primary">2</span>
                    </div>
                    <div>
                      <h4 className="font-medium">Build Agents with LangGraph</h4>
                      <p className="text-sm text-muted-foreground">Create stateful workflows and multi-step agents</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                      <span className="text-xs font-medium text-primary">3</span>
                    </div>
                    <div>
                      <h4 className="font-medium">Monitor with LangSmith</h4>
                      <p className="text-sm text-muted-foreground">Add tracing, evaluation, and monitoring</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                      <span className="text-xs font-medium text-primary">4</span>
                    </div>
                    <div>
                      <h4 className="font-medium">Deploy with LangServe</h4>
                      <p className="text-sm text-muted-foreground">Convert your chains to production APIs</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                      <span className="text-xs font-medium text-primary">5</span>
                    </div>
                    <div>
                      <h4 className="font-medium">Integrate with MCP</h4>
                      <p className="text-sm text-muted-foreground">Connect to external data sources and tools</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center mt-0.5">
                      <span className="text-xs font-medium text-primary">6</span>
                    </div>
                    <div>
                      <h4 className="font-medium">Scale with Multi-Agent Systems</h4>
                      <p className="text-sm text-muted-foreground">Build complex agent-to-agent communication</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Architecture Overview */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle>LangChain Ecosystem Architecture</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-gradient-section p-6 rounded-lg">
              <div className="space-y-6">
                <div className="text-center">
                  <h4 className="font-medium mb-4">Application Flow</h4>
                  <div className="flex justify-center items-center space-x-4 text-sm">
                    <Badge variant="outline">User Input</Badge>
                    <ArrowRight className="w-4 h-4" />
                    <Badge variant="outline">LangChain/LangGraph</Badge>
                    <ArrowRight className="w-4 h-4" />
                    <Badge variant="outline">LLM Provider</Badge>
                    <ArrowRight className="w-4 h-4" />
                    <Badge variant="outline">Response</Badge>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-2">
                      <Code className="w-6 h-6 text-primary" />
                    </div>
                    <h4 className="font-medium">Development</h4>
                    <p className="text-sm text-muted-foreground">Build with LangChain components</p>
                  </div>
                  <div className="text-center">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-2">
                      <GitBranch className="w-6 h-6 text-primary" />
                    </div>
                    <h4 className="font-medium">Orchestration</h4>
                    <p className="text-sm text-muted-foreground">LangGraph agent workflows</p>
                  </div>
                  <div className="text-center">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-2">
                      <Settings className="w-6 h-6 text-primary" />
                    </div>
                    <h4 className="font-medium">Monitoring</h4>
                    <p className="text-sm text-muted-foreground">LangSmith observability</p>
                  </div>
                  <div className="text-center">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-2">
                      <Database className="w-6 h-6 text-primary" />
                    </div>
                    <h4 className="font-medium">Deployment</h4>
                    <p className="text-sm text-muted-foreground">LangServe APIs</p>
                  </div>
                </div>
                
                <div className="border-t pt-4">
                  <h4 className="font-medium mb-3 text-center">Integration Layer</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="text-center">
                      <h5 className="font-medium">Model Context Protocol (MCP)</h5>
                      <p className="text-sm text-muted-foreground">Universal connector for data sources, APIs, and tools</p>
                    </div>
                    <div className="text-center">
                      <h5 className="font-medium">Agent-to-Agent Communication</h5>
                      <p className="text-sm text-muted-foreground">Multi-agent coordination and distributed processing</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Use Cases Grid */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Common Use Cases & Applications</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card className="shadow-card hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Bot className="w-5 h-5 text-primary" />
                  <span>Conversational AI</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">Build intelligent chatbots and virtual assistants with memory and context awareness.</p>
                <ul className="text-xs text-muted-foreground space-y-1">
                  <li>• Customer support bots</li>
                  <li>• Personal assistants</li>
                  <li>• Domain-specific Q&A</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="shadow-card hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Database className="w-5 h-5 text-primary" />
                  <span>Document Intelligence</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">Create RAG systems for document analysis, search, and question answering.</p>
                <ul className="text-xs text-muted-foreground space-y-1">
                  <li>• Knowledge base search</li>
                  <li>• Document summarization</li>
                  <li>• Research assistance</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="shadow-card hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Code className="w-5 h-5 text-primary" />
                  <span>Code Generation</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">Generate, analyze, debug, and explain code across multiple programming languages.</p>
                <ul className="text-xs text-muted-foreground space-y-1">
                  <li>• Code completion</li>
                  <li>• Bug detection</li>
                  <li>• Code explanation</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="shadow-card hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Zap className="w-5 h-5 text-primary" />
                  <span>Workflow Automation</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">Automate complex business processes with intelligent decision-making capabilities.</p>
                <ul className="text-xs text-muted-foreground space-y-1">
                  <li>• Process automation</li>
                  <li>• Data extraction</li>
                  <li>• Report generation</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="shadow-card hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Settings className="w-5 h-5 text-primary" />
                  <span>Data Analysis</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">Analyze data, generate insights, and create visualizations using natural language.</p>
                <ul className="text-xs text-muted-foreground space-y-1">
                  <li>• SQL generation</li>
                  <li>• Data visualization</li>
                  <li>• Trend analysis</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="shadow-card hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <GitBranch className="w-5 h-5 text-primary" />
                  <span>Multi-Agent Systems</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">Coordinate multiple specialized agents for complex, multi-step problem solving.</p>
                <ul className="text-xs text-muted-foreground space-y-1">
                  <li>• Research teams</li>
                  <li>• Content pipelines</li>
                  <li>• Quality assurance</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Key Benefits */}
        <Card className="shadow-card border-l-4 border-l-primary">
          <CardHeader>
            <CardTitle>Why Choose the LangChain Ecosystem?</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h4 className="font-medium">Developer Experience</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Intuitive APIs and consistent interfaces</li>
                  <li>• Comprehensive documentation and examples</li>
                  <li>• Active community and ecosystem</li>
                  <li>• Regular updates and improvements</li>
                </ul>
              </div>
              <div className="space-y-3">
                <h4 className="font-medium">Production Ready</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Built-in monitoring and evaluation tools</li>
                  <li>• Scalable deployment options</li>
                  <li>• Security and privacy best practices</li>
                  <li>• Enterprise-grade reliability</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </DocSection>
  );
};