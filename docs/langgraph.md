# LangGraph Documentation

## Table of Contents
- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
  - [State Management](#1-state-management)
  - [Nodes](#2-nodes)
  - [Edges](#3-edges)
  - [Conditional Logic](#4-conditional-logic)
- [Quick Start](#quick-start)
- [Advanced Features](#advanced-features)
  - [Custom State Management](#custom-state-management)
  - [Parallel Execution](#parallel-execution)
  - [Error Handling](#error-handling)
  - [Human-in-the-Loop](#human-in-the-loop)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Introduction

LangGraph is a powerful library for building stateful, multi-actor applications with LLMs. It extends the LangChain ecosystem by providing a way to create complex workflows, autonomous agents, and multi-step reasoning systems that can maintain state across interactions.

### Key Features
- **State Management**: Maintain and update complex state across workflow steps
- **Multi-Agent Systems**: Coordinate multiple agents with different roles and capabilities
- **Flexible Workflows**: Define custom nodes and edges for any process
- **Human-in-the-Loop**: Seamlessly integrate human feedback into automated workflows
- **Debugging Tools**: Built-in visualization and tracing capabilities

## Core Concepts

### 1. State Management

LangGraph uses a central state object that flows through your workflow. The state is a dictionary-like object that can be updated by nodes in the graph.

```python
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
import operator

# Define a strongly-typed state
class AgentState(TypedDict):
    # Messages accumulate in a list
    messages: Annotated[List[dict], operator.add]
    # Other state variables
    user_query: str
    search_results: List[dict]
    final_answer: str

# Initialize state
initial_state = {
    "messages": [{"role": "system", "content": "You are a helpful assistant."}],
    "user_query": "",
    "search_results": [],
    "final_answer": ""
}
```

### 2. Nodes

Nodes are the building blocks of your workflow. Each node is a function that takes the current state and returns updates to the state.

```python
def process_user_input(state: AgentState) -> dict:
    """Extract and process user input."""
    last_message = state["messages"][-1]
    if last_message["role"] == "user":
        return {"user_query": last_message["content"]}
    return {}

def generate_response(state: AgentState) -> dict:
    """Generate a response using an LLM."""
    # In a real app, you would call your LLM here
    return {
        "messages": [{
            "role": "assistant", 
            "content": f"You asked: {state['user_query']}"
        }]
    }
```

### 3. Edges

Edges define the flow of control between nodes. They can be conditional or unconditional.

```python
# Create a new graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("process_input", process_user_input)
workflow.add_node("generate_response", generate_response)

# Add edges
workflow.add_edge("process_input", "generate_response")
workflow.add_edge("generate_response", END)

# Set the entry point
workflow.set_entry_point("process_input")

# Compile the graph
app = workflow.compile()

# Run the workflow
result = app.invoke({
    "messages": [{"role": "user", "content": "Hello!"}],
    "user_query": "",
    "search_results": [],
    "final_answer": ""
})
```

### 4. Conditional Logic

You can add conditional edges to create dynamic workflows that branch based on the state.

```python
def should_continue(state: AgentState) -> str:
    """Determine the next step based on the current state."""
    last_message = state["messages"][-1]["content"].lower()
    
    if "goodbye" in last_message:
        return "end_conversation"
    elif "search" in last_message:
        return "web_search"
    return "generate_response"

# Add conditional edges
workflow.add_conditional_edges(
    "process_input",
    should_continue,
    {
        "end_conversation": END,
        "web_search": "search_web",
        "generate_response": "generate_response"
    }
)
```

## Quick Start

Here's a complete example of a simple chatbot using LangGraph:

```python
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import operator

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define the state
class ChatState(TypedDict):
    messages: Annotated[List[dict], operator.add]

# Define nodes
def process_input(state: ChatState) -> dict:
    """Process user input."""
    return {}

def generate_response(state: ChatState) -> dict:
    """Generate a response using the LLM."""
    response = llm.invoke(state["messages"])
    return {"messages": [{"role": "assistant", "content": response.content}]}

def should_continue(state: ChatState) -> str:
    """Determine if the conversation should continue."""
    last_message = state["messages"][-1]["content"].lower()
    if "goodbye" in last_message:
        return "end"
    return "continue"

# Build the graph
workflow = StateGraph(ChatState)
workflow.add_node("process", process_input)
workflow.add_node("respond", generate_response)

# Add edges
workflow.add_edge("process", "respond")
workflow.add_conditional_edges(
    "respond",
    should_continue,
    {"continue": "process", "end": END}
)

# Set the entry point
workflow.set_entry_point("process")

# Compile the graph
app = workflow.compile()

# Run the chat
state = {"messages": [{"role": "user", "content": "Hello!"}]}
while True:
    result = app.invoke(state)
    state = result
    print(f"Assistant: {result['messages'][-1]['content']}")
    
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
        
    state["messages"].append({"role": "user", "content": user_input})
```

## Advanced Features

### Custom State Management

You can create custom state classes with validation and transformation logic.

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class Document(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class ResearchState(BaseModel):
    query: str
    documents: List[Document] = []
    summary: str = ""
    
    def add_document(self, content: str, **metadata):
        self.documents.append(Document(content=content, metadata=metadata))
```

### Parallel Execution

Run multiple nodes in parallel and combine their results.

```python
from typing import List
from langgraph.graph import START, END

def search_web(state: dict) -> dict:
    # Simulate web search
    return {"search_results": ["Result 1", "Result 2"]}

def search_database(state: dict) -> dict:
    # Simulate database search
    return {"db_results": ["DB Result 1", "DB Result 2"]}

# Create a new graph
workflow = StateGraph(dict)

# Add parallel nodes
workflow.add_node("web_search", search_web)
workflow.add_node("db_search", search_database)
workflow.add_node("combine_results", lambda x: x)

# Set up parallel execution
workflow.add_edge(START, "web_search")
workflow.add_edge(START, "db_search")
workflow.add_edge("web_search", "combine_results")
workflow.add_edge("db_search", "combine_results")
workflow.add_edge("combine_results", END)
```

## Best Practices

1. **State Design**
   - Keep your state as flat as possible
   - Use Pydantic models for validation
   - Avoid storing large objects in the state

2. **Node Design**
   - Keep nodes small and focused
   - Make nodes idempotent when possible
   - Document expected inputs and outputs

3. **Error Handling**
   - Implement proper error handling in nodes
   - Use try/except blocks for external calls
   - Consider implementing retry logic

## Common Patterns

### Human-in-the-Loop

```python
def human_review(state: dict) -> dict:
    """Get human feedback on the current state."""
    print(f"\nCurrent output: {state.get('draft')}")
    feedback = input("\nProvide feedback (or press Enter to accept): ")
    if feedback.strip():
        return {"needs_revision": True, "feedback": feedback}
    return {"needs_revision": False}

# In your workflow:
workflow.add_conditional_edges(
    "human_review",
    lambda state: "revise" if state.get("needs_revision") else "publish",
    {"revise": "revise_draft", "publish": "publish_draft"}
)
```

### Agent with Tools

```python
from langchain.tools import Tool

def search_tool(query: str) -> str:
    # Implementation for search
    return "Search results..."

def calculator(expression: str) -> str:
    # Implementation for calculator
    return str(eval(expression))

# Create tools
tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Useful for answering questions about current events"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for doing math calculations"
    )
]

# In your node:
def agent_node(state: dict) -> dict:
    # Call agent with tools
    # ...
    return {"response": "Agent response"}
```

## Troubleshooting

### Common Issues

1. **State Mismatch**
   - Ensure all nodes return the expected state structure
   - Check for type mismatches in your state

2. **Cycles**
   - Be careful with cycles in your graph
   - Consider adding a maximum iteration limit

3. **Performance**
   - Profile your nodes to find bottlenecks
   - Consider caching expensive operations

## API Reference

### StateGraph

```python
class StateGraph(State):
    def add_node(self, name: str, action: Callable) -> None:
        """Add a node to the graph."""
        pass
    
    def add_edge(self, start_key: str, end_key: str) -> None:
        """Add an edge between two nodes."""
        pass
    
    def add_conditional_edges(
        self,
        source: str,
        condition: Callable,
        path_map: Dict[str, str]
    ) -> None:
        """Add conditional edges based on a condition function."""
        pass
    
    def compile(self, **kwargs) -> Any:
        """Compile the graph into an executable application."""
        pass
```

For more detailed information, refer to the [official LangGraph documentation](https://langchain-ai.github.io/langgraph/).
