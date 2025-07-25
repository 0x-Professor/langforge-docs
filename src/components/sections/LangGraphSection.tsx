import { DocSection, FeatureCard, QuickStart } from '@/components/DocSection';
import { CodeBlock } from '@/components/CodeBlock';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { GitBranch, Bot, Zap, Users, MessageSquare, Play, Settings, Cpu, Network, GitFork, Code2, Layers, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

/**
 * LangGraphSection Component
 * 
 * Provides comprehensive documentation about LangGraph, a library for building stateful,
 * multi-actor applications with LLMs. It enables the creation of complex workflows,
 * autonomous agents, and multi-step reasoning systems.
 */
export const LangGraphSection = () => {
  // Basic agent example
  const basicAgentCode = `from typing import TypedDict, List, Annotated, Sequence
from langgraph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator

# Define a simple tool
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # In a real application, this would call a weather API
    return f"It's sunny in {location} with 72°F"

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Create the agent
model = ChatOpenAI(model="gpt-4")
tools = [get_weather]

# Build the agent graph
agent = create_react_agent(model, tools)

# Create a simple workflow
def should_continue(state: AgentState) -> str:
    # Simple logic to determine if we should continue
    last_message = state["messages"][-1]
    if "exit" in last_message.content.lower():
        return "end"
    return "continue"

# Define the nodes
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("human", lambda state: {"messages": [HumanMessage("How can I help you further?")]})

# Define the edges
workflow.add_edge("agent", "human")
workflow.add_edge("human", "agent")

# Add conditional edges
workflow.add_conditional_edges(
    "human",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

# Set the entry point
workflow.set_entry_point("agent")

# Compile the workflow
app = workflow.compile()

# Run the agent
response = app.invoke({
    "messages": [HumanMessage("What's the weather in San Francisco?")]
})

# Print the final response
print(response["messages"][-1].content)`;

  // Advanced workflow example
  const workflowCode = `from typing import TypedDict, List, Annotated, Literal
from langgraph import StateGraph, START, END
from langgraph.graph import MessagesState, add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import operator

# Define the state
class WorkflowState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage | SystemMessage], add_messages]
    user_input: str
    analysis: dict
    response: str
    next_step: Literal["generate_response", "get_user_feedback", "end"]

# Initialize the language model
llm = ChatOpenAI(model="gpt-4")

# Define nodes
def analyze_input(state: WorkflowState) -> WorkflowState:
    """Analyze the user input to determine intent and entities."""
    user_input = state["user_input"]
    
    # In a real application, this would be more sophisticated
    # and might use an LLM to analyze the input
    analysis = {
        "intent": "information_request" if "?" in user_input else "conversation",
        "entities": [],
        "sentiment": "neutral",
        "requires_follow_up": len(user_input.split()) < 5
    }
    
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
result = app.invoke({
    "user_input": "machine learning"
})

print(result["response"])`;

  // Multi-agent collaboration example
  const multiAgentCode = `from typing import TypedDict, List, Annotated, Literal
from langgraph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import operator

# Define the state
class MultiAgentState(TypedDict):
    messages: Annotated[list, operator.add]
    researcher_output: str
    writer_output: str
    reviewer_comments: str
    final_output: str
    next: Literal["researcher", "writer", "reviewer", "end"]

# Initialize models
researcher_llm = ChatOpenAI(model="gpt-4")
writer_llm = ChatOpenAI(model="gpt-4")
reviewer_llm = ChatOpenAI(model="gpt-4")

# Define nodes
def researcher(state: MultiAgentState) -> dict:
    """Research the topic and gather information."""
    messages = state["messages"]
    last_message = messages[-1].content
    
    # In a real app, this would involve web search or database lookups
    research = f"Research about {last_message}:\n" \
               "1. Key concepts and definitions\n" \
               "2. Current trends and developments\n" \
               "3. Relevant examples and case studies"
    
    return {"researcher_output": research, "next": "writer"}

def writer(state: MultiAgentState) -> dict:
    """Write a comprehensive article based on research."""
    research = state["researcher_output"]
    
    # In a real app, this would use the writer_llm
    article = f"# Article\n\nBased on research:\n{research}\n\n" \
              "This is a well-written article that synthesizes the research into " \
              "a coherent and engaging piece of content."
    
    return {"writer_output": article, "next": "reviewer"}

def reviewer(state: MultiAgentState) -> dict:
    """Review the article and provide feedback."""
    article = state["writer_output"]
    
    # In a real app, this would use the reviewer_llm
    feedback = "The article is well-structured but could benefit from " \
              "more specific examples and references to recent studies."
    
    return {"reviewer_comments": feedback, "next": "end"}

# Create the workflow
workflow = StateGraph(MultiAgentState)

# Add nodes
workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)
workflow.add_node("reviewer", reviewer)

# Define edges
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_edge("reviewer", END)

# Compile the workflow
app = workflow.compile()

# Run the workflow
result = app.invoke({
    "messages": [HumanMessage("Write an article about the future of AI in healthcare")],
    "next": "researcher"
})`;

  return (
    <DocSection
      id="langgraph"
      title="LangGraph: Stateful Multi-Agent Workflows"
      description="Build complex, stateful applications with autonomous agents and multi-step workflows."
      badges={["Agent Framework", "Production Ready", "v0.1.0"]}
      externalLinks={[
        { title: "Official Documentation", url: "https://python.langchain.com/docs/langgraph" },
        { title: "GitHub", url: "https://github.com/langchain-ai/langgraph" },
        { title: "Examples", url: "https://github.com/langchain-ai/langgraph/tree/main/examples" }
      ]}
    >
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4 mb-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="concepts">Core Concepts</TabsTrigger>
          <TabsTrigger value="examples">Examples</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="space-y-6">
          <Card className="shadow-card border-l-4 border-l-purple-500">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <GitBranch className="w-5 h-5 text-purple-500" />
                <span>What is LangGraph?</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground mb-4">
                LangGraph is a powerful library for building stateful, multi-actor applications with LLMs. It extends the LangChain ecosystem by providing a way to create complex workflows, autonomous agents, and multi-step reasoning systems that can maintain state across interactions.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                <div className="space-y-4">
                  <h3 className="font-medium flex items-center">
                    <Cpu className="w-4 h-4 mr-2 text-amber-500" />
                    Key Features
                  </h3>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>• <span className="font-medium">State Management</span> - Maintain and update complex state across steps</li>
                    <li>• <span className="font-medium">Multi-Agent Systems</span> - Coordinate multiple agents with different roles</li>
                    <li>• <span className="font-medium">Flexible Workflows</span> - Define custom nodes and edges for any process</li>
                    <li>• <span className="font-medium">Human-in-the-Loop</span> - Seamlessly integrate human feedback</li>
                    <li>• <span className="font-medium">Debugging Tools</span> - Built-in visualization and tracing</li>
                  </ul>
                </div>
                
                <div className="space-y-4">
                  <h3 className="font-medium flex items-center">
                    <Layers className="w-4 h-4 mr-2 text-purple-500" />
                    Use Cases
                  </h3>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>• <span className="font-medium">Autonomous Agents</span> - Build agents that can plan and execute tasks</li>
                    <li>• <span className="font-medium">Complex Workflows</span> - Model multi-step business processes</li>
                    <li>• <span className="font-medium">Multi-Agent Systems</span> - Create systems with specialized agents</li>
                    <li>• <span className="font-medium">Interactive Applications</span> - Build chat interfaces with memory</li>
                    <li>• <span className="font-medium">Decision Support</span> - Implement reasoning and planning systems</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Zap className="w-5 h-5 text-green-500" />
                  <span>Quick Start</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Installation</h4>
                    <CodeBlock
                      language="bash"
                      code="pip install langgraph"
                      showLineNumbers={false}
                    />
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">Basic Example</h4>
                    <CodeBlock
                      language="python"
                      code={`from langgraph import StateGraph, START, END

# Define a simple workflow
def hello(state):
    return {"message": f"Hello, {state['name']}!"}

# Create a new graph
workflow = StateGraph(dict)
workflow.add_node("greet", hello)
workflow.add_edge(START, "greet")
workflow.add_edge("greet", END)
workflow.set_entry_point("greet")

# Compile and run
app = workflow.compile()
result = app.invoke({"name": "World"})
print(result["message"])  # Output: Hello, World!`}
                      showLineNumbers={true}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <GitBranch className="w-5 h-5 text-blue-500" />
                  <span>Core Concepts</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-1">1. State</h4>
                    <p className="text-sm text-muted-foreground">
                      The state is a dictionary that gets passed between nodes. Each node can read from and update the state.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium mb-1">2. Nodes</h4>
                    <p className="text-sm text-muted-foreground">
                      Nodes are functions that take the current state and return updates to the state.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium mb-1">3. Edges</h4>
                    <p className="text-sm text-muted-foreground">
                      Edges define the flow of execution between nodes based on the current state.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium mb-1">4. Conditional Logic</h4>
                    <p className="text-sm text-muted-foreground">
                      Use conditional edges to create dynamic workflows that branch based on the state.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="concepts" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Network className="w-5 h-5 text-purple-500" />
                  <span>State Management</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  LangGraph uses a state management system to track data as it flows through your workflow. The state is a dictionary-like object that gets updated by each node in the graph.
                </p>
                <CodeBlock
                  language="python"
                  code={`from typing import TypedDict, Annotated, List
from langgraph import StateGraph
import operator

# Define your state
class MyState(TypedDict):
    messages: Annotated[List[dict], operator.add]  # Automatically appends to list
    counter: int
    status: str

# Initialize the graph
workflow = StateGraph(MyState)

# Define nodes that update the state
def node1(state: MyState) -> dict:
    return {"counter": state["counter"] + 1, "status": "processing"}

def node2(state: MyState) -> dict:
    return {"status": "completed"}

# Add nodes and edges
workflow.add_node("node1", node1)
workflow.add_node("node2", node2)
workflow.add_edge("node1", "node2")
workflow.set_entry_point("node1")

# Compile and run
app = workflow.compile()
result = app.invoke({"messages": [], "counter": 0, "status": "started"})`}
                  showLineNumbers={true}
                />
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <GitFork className="w-5 h-5 text-purple-500" />
                  <span>Conditional Logic</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  You can add conditional edges to create dynamic workflows that branch based on the current state.
                </p>
                <CodeBlock
                  language="python"
                  code={`from langgraph import StateGraph
from typing import Literal, TypedDict

# Define state with a 'next' field for routing
class State(TypedDict):
    value: int
    next: Literal["even", "odd"]

def process_number(state: State) -> State:
    value = state["value"]
    return {"value": value, "next": "even" if value % 2 == 0 else "odd"}

def handle_even(state: State) -> State:
    return {"result": f"{state['value']} is even"}

def handle_odd(state: State) -> State:
    return {"result": f"{state['value']} is odd"}

# Create the workflow
workflow = StateGraph(State)
workflow.add_node("process", process_number)
workflow.add_node("even_handler", handle_even)
workflow.add_node("odd_handler", handle_odd)

# Add conditional edges
workflow.add_conditional_edges(
    "process",
    lambda state: state["next"],
    {
        "even": "even_handler",
        "odd": "odd_handler"
    }
)
workflow.add_edge("even_handler", END)
workflow.add_edge("odd_handler", END)
workflow.set_entry_point("process")

# Compile and run
app = workflow.compile()
result = app.invoke({"value": 42, "next": ""})`}
                  showLineNumbers={true}
                />
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="examples" className="space-y-6">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <MessageSquare className="w-5 h-5 text-purple-500" />
                  <span>Basic Agent</span>
                </CardTitle>
                <CardDescription>
                  A simple agent that can use tools to answer questions.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <CodeBlock
                  language="python"
                  code={basicAgentCode}
                  showLineNumbers={true}
                />
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Users className="w-5 h-5 text-purple-500" />
                  <span>Multi-Agent Collaboration</span>
                </CardTitle>
                <CardDescription>
                  Multiple agents working together to complete a complex task.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <CodeBlock
                  language="python"
                  code={multiAgentCode}
                  showLineNumbers={true}
                />
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="advanced" className="space-y-6">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Settings className="w-5 h-5 text-purple-500" />
                  <span>Advanced Workflow</span>
                </CardTitle>
                <CardDescription>
                  A more complex example with error handling and state management.
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <CodeBlock
                  language="python"
                  code={workflowCode}
                  showLineNumbers={true}
                />
              </CardContent>
              <CardFooter className="bg-muted/50 p-4">
                <div className="space-y-4 w-full">
                  <div>
                    <p className="font-medium">Key Features of This Implementation:</p>
                    <ul className="list-disc pl-5 space-y-1 mt-1 text-sm text-muted-foreground">
                      <li>Type hints and type checking with TypedDict</li>
                      <li>State management with type-safe updates</li>
                      <li>Modular design with separate node functions</li>
                      <li>Clear error handling and state transitions</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-medium">Performance</p>
                    <ul className="list-disc pl-5 space-y-1 mt-1 text-sm text-muted-foreground">
                      <li>Use parallel processing where possible</li>
                      <li>Implement checkpointing for long workflows</li>
                      <li>Cache expensive computations</li>
                      <li>Monitor memory usage in stateful agents</li>
                    </ul>
                  </div>
                </div>
              </CardFooter>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </DocSection>
  );
};