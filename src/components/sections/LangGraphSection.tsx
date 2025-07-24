import { DocSection, FeatureCard, QuickStart } from '@/components/DocSection';
import { CodeBlock } from '@/components/CodeBlock';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { GitBranch, Bot, Zap, Users, MessageSquare, Play, Settings } from 'lucide-react';

export const LangGraphSection = () => {
  const basicAgentCode = `from langchain_openai import ChatOpenAI
from langgraph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Define a simple tool
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"It's sunny in {location} with 72°F"

# Create the agent
model = ChatOpenAI(model="gpt-4")
tools = [get_weather]

# Build the agent graph
agent = create_react_agent(model, tools)

# Run the agent
response = agent.invoke({
    "messages": [("human", "What's the weather in San Francisco?")]
})

print(response["messages"][-1].content)`;

  const workflowCode = `from langgraph import StateGraph, START, END
from typing import TypedDict
import operator

# Define state
class WorkflowState(TypedDict):
    messages: list
    user_input: str
    analysis: str
    response: str

# Define nodes
def analyze_input(state: WorkflowState) -> WorkflowState:
    user_input = state["user_input"]
    # Analyze user intent
    analysis = f"Analysis: User wants information about {user_input}"
    return {**state, "analysis": analysis}

def generate_response(state: WorkflowState) -> WorkflowState:
    analysis = state["analysis"]
    # Generate response based on analysis
    response = f"Based on the analysis: {analysis}"
    return {**state, "response": response}

# Build workflow
workflow = StateGraph(WorkflowState)

# Add nodes
workflow.add_node("analyze", analyze_input)
workflow.add_node("generate", generate_response)

# Add edges
workflow.add_edge(START, "analyze")
workflow.add_edge("analyze", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()

# Run workflow
result = app.invoke({"user_input": "machine learning"})
print(result["response"])`;

  const streamingCode = `from langgraph import StateGraph
from langchain_core.messages import HumanMessage

# Create streaming agent
agent = create_react_agent(model, tools)

# Stream responses
inputs = {"messages": [HumanMessage(content="Explain quantum computing")]}

for chunk in agent.stream(inputs, stream_mode="values"):
    if "messages" in chunk:
        chunk["messages"][-1].pretty_print()

# Stream with interrupts for human-in-the-loop
config = {"configurable": {"thread_id": "thread-1"}}

# Add interrupt before final response
for chunk in agent.stream(inputs, config=config, stream_mode="values"):
    if "messages" in chunk:
        last_message = chunk["messages"][-1]
        if last_message.type == "ai":
            # Pause for human review
            user_approval = input("Approve this response? (y/n): ")
            if user_approval.lower() != 'y':
                # Modify or reject
                continue
        last_message.pretty_print()`;

  const multiAgentCode = `from langgraph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# Define multi-agent state
class MultiAgentState(TypedDict):
    messages: list
    task: str
    research_result: str
    analysis_result: str
    final_report: str

# Research Agent
def research_agent(state: MultiAgentState) -> MultiAgentState:
    model = ChatOpenAI(model="gpt-4")
    task = state["task"]
    
    prompt = f"Research the following topic: {task}"
    response = model.invoke([HumanMessage(content=prompt)])
    
    return {**state, "research_result": response.content}

# Analysis Agent
def analysis_agent(state: MultiAgentState) -> MultiAgentState:
    model = ChatOpenAI(model="gpt-4")
    research = state["research_result"]
    
    prompt = f"Analyze this research: {research}"
    response = model.invoke([HumanMessage(content=prompt)])
    
    return {**state, "analysis_result": response.content}

# Report Agent
def report_agent(state: MultiAgentState) -> MultiAgentState:
    model = ChatOpenAI(model="gpt-4")
    research = state["research_result"]
    analysis = state["analysis_result"]
    
    prompt = f"Create a final report based on:\\nResearch: {research}\\nAnalysis: {analysis}"
    response = model.invoke([HumanMessage(content=prompt)])
    
    return {**state, "final_report": response.content}

# Build multi-agent workflow
workflow = StateGraph(MultiAgentState)

workflow.add_node("researcher", research_agent)
workflow.add_node("analyst", analysis_agent)
workflow.add_node("reporter", report_agent)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "reporter")
workflow.add_edge("reporter", END)

multi_agent_app = workflow.compile()

# Execute multi-agent workflow
result = multi_agent_app.invoke({
    "task": "Latest trends in AI development"
})

print(result["final_report"])`;

  return (
    <DocSection
      id="langgraph"
      title="LangGraph - Agent Framework"
      description="Build stateful, multi-actor applications with LLMs using graph-based workflows and persistent state management."
      badges={["Agent Framework", "Production Ready", "Streaming"]}
      externalLinks={[
        { title: "LangGraph Docs", url: "https://langchain-ai.github.io/langgraph/" },
        { title: "Tutorials", url: "https://langchain-ai.github.io/langgraph/tutorials/" },
        { title: "Examples", url: "https://github.com/langchain-ai/langgraph/tree/main/examples" }
      ]}
    >
      <div className="space-y-8">
        {/* Key Features */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Key Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <FeatureCard
              icon={<GitBranch className="w-6 h-6" />}
              title="Graph-Based Workflows"
              description="Define complex agent behaviors using directed graphs with nodes and edges."
              features={[
                "Visual workflow design",
                "Conditional branching",
                "Parallel processing", 
                "Cycle detection"
              ]}
            />
            <FeatureCard
              icon={<Zap className="w-6 h-6" />}
              title="Persistent State"
              description="Maintain conversation context and agent memory across interactions."
              features={[
                "Thread-based memory",
                "State checkpoints",
                "Recovery mechanisms",
                "Custom persistence"
              ]}
            />
            <FeatureCard
              icon={<Play className="w-6 h-6" />}
              title="Streaming & Real-time"
              description="Stream agent responses and support real-time interactions."
              features={[
                "Token streaming",
                "Progress updates",
                "Interrupt handling",
                "Live monitoring"
              ]}
            />
            <FeatureCard
              icon={<Users className="w-6 h-6" />}
              title="Human-in-the-Loop"
              description="Integrate human oversight and approval into agent workflows."
              features={[
                "Approval gates",
                "Manual overrides",
                "Feedback loops",
                "Audit trails"
              ]}
            />
            <FeatureCard
              icon={<Bot className="w-6 h-6" />}
              title="Multi-Agent Systems"
              description="Orchestrate multiple specialized agents working together."
              features={[
                "Agent coordination",
                "Task delegation",
                "Result aggregation",
                "Conflict resolution"
              ]}
            />
            <FeatureCard
              icon={<Settings className="w-6 h-6" />}
              title="Tool Integration"
              description="Connect agents to external APIs, databases, and services."
              features={[
                "Function calling",
                "API integrations",
                "Custom tools",
                "Error handling"
              ]}
            />
          </div>
        </div>

        {/* Code Examples */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Implementation Examples</h2>
          <Tabs defaultValue="basic" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="basic">Basic Agent</TabsTrigger>
              <TabsTrigger value="workflow">Custom Workflow</TabsTrigger>
              <TabsTrigger value="streaming">Streaming</TabsTrigger>
              <TabsTrigger value="multiagent">Multi-Agent</TabsTrigger>
            </TabsList>
            
            <TabsContent value="basic" className="space-y-4">
              <CodeBlock
                title="Simple ReAct Agent"
                language="python"
                code={basicAgentCode}
              />
            </TabsContent>
            
            <TabsContent value="workflow" className="space-y-4">
              <CodeBlock
                title="Custom Workflow Graph"
                language="python"
                code={workflowCode}
              />
            </TabsContent>
            
            <TabsContent value="streaming" className="space-y-4">
              <CodeBlock
                title="Streaming with Human-in-the-Loop"
                language="python"
                code={streamingCode}
              />
            </TabsContent>
            
            <TabsContent value="multiagent" className="space-y-4">
              <CodeBlock
                title="Multi-Agent Collaboration"
                language="python"
                code={multiAgentCode}
              />
            </TabsContent>
          </Tabs>
        </div>

        {/* Architecture Patterns */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Common Patterns</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Agent Patterns</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <h4 className="font-medium">ReAct Agent</h4>
                    <p className="text-sm text-muted-foreground">Reasoning and Acting with tool usage</p>
                  </div>
                  <div>
                    <h4 className="font-medium">Planning Agent</h4>
                    <p className="text-sm text-muted-foreground">Multi-step planning and execution</p>
                  </div>
                  <div>
                    <h4 className="font-medium">Reflection Agent</h4>
                    <p className="text-sm text-muted-foreground">Self-correction and improvement</p>
                  </div>
                  <div>
                    <h4 className="font-medium">Supervisor Agent</h4>
                    <p className="text-sm text-muted-foreground">Orchestrating multiple sub-agents</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>State Management</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <h4 className="font-medium">MessagesState</h4>
                    <p className="text-sm text-muted-foreground">Conversation history management</p>
                  </div>
                  <div>
                    <h4 className="font-medium">Custom State</h4>
                    <p className="text-sm text-muted-foreground">Application-specific state schemas</p>
                  </div>
                  <div>
                    <h4 className="font-medium">Checkpoints</h4>
                    <p className="text-sm text-muted-foreground">State persistence and recovery</p>
                  </div>
                  <div>
                    <h4 className="font-medium">Reducers</h4>
                    <p className="text-sm text-muted-foreground">State transformation functions</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Best Practices */}
        <Card className="shadow-card border-l-4 border-l-primary">
          <CardHeader>
            <CardTitle>LangGraph Best Practices</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h4 className="font-medium">Design Principles</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Keep node functions pure and stateless</li>
                  <li>• Use typed state schemas for better validation</li>
                  <li>• Implement proper error handling in nodes</li>
                  <li>• Design for observability and debugging</li>
                </ul>
              </div>
              <div className="space-y-3">
                <h4 className="font-medium">Performance</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Use parallel processing where possible</li>
                  <li>• Implement checkpointing for long workflows</li>
                  <li>• Cache expensive computations</li>
                  <li>• Monitor memory usage in stateful agents</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </DocSection>
  );
};