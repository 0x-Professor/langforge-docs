# LangGraph: Building Stateful, Multi-Actor Applications with LLMs

<div class="tip">
  <strong>🚀 New in v0.1.0</strong>: Support for streaming, improved debugging tools, and enhanced type safety.
</div>

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

### What is LangGraph?

LangGraph is a Python library that enables you to build complex, stateful applications with LLMs by modeling them as graphs. It provides:

- **Declarative API**: Define your application's logic as a graph of nodes and edges
- **State Management**: Maintain and manipulate state throughout the execution
- **Concurrency Support**: Run multiple operations in parallel
- **Error Handling**: Built-in mechanisms for handling failures
- **Observability**: Trace and debug complex workflows

### Why Use LangGraph?

1. **Complex Workflows**: Model intricate business processes that require multiple steps and decisions
2. **Stateful Applications**: Build applications that maintain context across interactions
3. **Agent Orchestration**: Coordinate multiple LLM agents with different roles
4. **Production Readiness**: Features like error handling, retries, and observability built-in
5. **Extensibility**: Easily integrate with other tools and services

### Key Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **State Management** | Maintain and update complex state across workflow steps | Multi-turn conversations, data processing pipelines |
| **Multi-Agent Systems** | Coordinate multiple agents with different roles | Collaborative problem solving, specialized task delegation |
| **Flexible Workflows** | Define custom nodes and edges for any process | Custom business logic, complex decision trees |
| **Human-in-the-Loop** | Seamlessly integrate human feedback | Content moderation, approval workflows, quality control |
| **Debugging Tools** | Built-in visualization and tracing | Development, testing, and monitoring |
| **Type Safety** | Runtime type checking and validation | Catching errors early, better developer experience |

### When to Use LangGraph

✅ **Good for**:
- Multi-step workflows with branching logic
- Stateful applications that maintain context
- Multi-agent systems with specialized roles
- Complex business processes requiring human oversight
- Applications needing audit trails and observability

❌ **Not ideal for**:
- Simple, stateless API calls
- Applications without complex workflow requirements
- When you only need basic LLM interactions (use LangChain directly instead)

## Core Concepts

### 1. State Management

LangGraph's state management system is the backbone of your application. It allows you to maintain and update a shared state across all nodes in your workflow.

#### State Definition

```python
from typing import TypedDict, List, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import operator

class Message(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = {}

class SearchResult(BaseModel):
    content: str
    source: str
    relevance_score: float = 0.0
    metadata: dict = {}

class AgentState(TypedDict):
    # Messages accumulate in a thread
    messages: Annotated[List[Message], operator.add]
    # Current user query
    user_query: str
    # Results from search operations
    search_results: List[SearchResult]
    # Final response to the user
    final_answer: str
    # Current status of the workflow
    status: Literal["processing", "awaiting_input", "completed", "error"]
    # Error information if status is "error"
    error: Optional[dict]
    # Additional context
    context: dict

# Initialize state with type hints and defaults
initial_state: AgentState = {
    "messages": [
        Message(
            role="system",
            content="You are a helpful AI assistant.",
            metadata={"version": "1.0.0"}
        )
    ],
    "user_query": "",
    "search_results": [],
    "final_answer": "",
    "status": "processing",
    "error": None,
    "context": {
        "session_id": "abc123",
        "start_time": datetime.utcnow().isoformat(),
        "retry_count": 0
    }
}
```

#### State Operations

```python
def update_state(state: AgentState, updates: dict) -> AgentState:
    """Safely update the state with new values."""
    return {**state, **updates}

def reset_state(state: AgentState) -> AgentState:
    """Reset the state to its initial values while preserving metadata."""
    return {
        **initial_state,
        "context": {
            **initial_state["context"],
            "previous_session_id": state["context"].get("session_id"),
            "session_id": generate_session_id()
        }
    }

def log_state_change(previous_state: AgentState, new_state: AgentState):
    """Log state changes for debugging and auditing."""
    changes = {}
    for key in new_state:
        if key in previous_state and previous_state[key] != new_state[key]:
            changes[key] = {
                "from": previous_state[key],
                "to": new_state[key]
            }
    logger.info(f"State updated: {json.dumps(changes, default=str)}")
```

#### State Validation

```python
from typing import Type, TypeVar
from pydantic import ValidationError

T = TypeVar('T', bound=TypedDict)

def validate_state(state: dict, state_type: Type[T]) -> T:
    """Validate the state against a TypedDict definition."""
    try:
        # Convert to Pydantic model if needed
        if hasattr(state_type, "__annotations__"):
            return state_type(**state)
        return state
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Invalid state: {str(e)}")

# Usage
try:
    validated_state = validate_state(user_provided_state, AgentState)
except ValidationError as e:
    logger.error(f"Invalid state: {e}")
    raise
```

### 2. Nodes: The Building Blocks of Your Workflow

Nodes are the fundamental processing units in a LangGraph application. Each node is a function that takes the current state, performs some computation, and returns updates to the state.

#### Node Design Principles

1. **Single Responsibility**: Each node should do one thing and do it well
2. **Idempotency**: Nodes should be safe to retry without side effects
3. **Stateless**: Nodes should not maintain internal state between invocations
4. **Pure Functions**: Given the same input, a node should always produce the same output

#### Node Types

1. **Input Nodes**: Process and validate incoming data
2. **Processing Nodes**: Transform or analyze data
3. **Decision Nodes**: Control the flow based on conditions
4. **Output Nodes**: Generate final responses or side effects
5. **Utility Nodes**: Reusable components for common tasks

#### Example: Advanced Node Implementation

```python
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class NodeMetrics(BaseModel):
    """Track performance metrics for a node."""
    execution_count: int = 0
    avg_duration: float = 0.0
    last_execution: Optional[datetime] = None
    error_count: int = 0

class NodeConfig(BaseModel):
    """Configuration for a node."""
    name: str
    description: str = ""
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    enabled: bool = True
    metadata: Dict[str, Any] = {}

class NodeResult(BaseModel):
    """Standardized result from a node execution."""
    success: bool
    output: Dict[str, Any]
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

def create_node(config: NodeConfig, node_func):
    """Decorator to create a node with standardized behavior."""
    metrics = NodeMetrics()
    
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        if not config.enabled:
            logger.warning(f"Node {config.name} is disabled")
            return {}
            
        start_time = datetime.utcnow()
        metrics.execution_count += 1
        
        try:
            # Validate input state
            if not isinstance(state, dict):
                raise ValueError("State must be a dictionary")
                
            logger.info(f"Executing node: {config.name}")
            
            # Execute the node function
            result = node_func(state)
            
            # Validate output
            if not isinstance(result, dict):
                raise ValueError("Node must return a dictionary")
                
            # Update metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics.avg_duration = (
                (metrics.avg_duration * (metrics.execution_count - 1) + duration) 
                / metrics.execution_count
            )
            metrics.last_execution = datetime.utcnow()
            
            return result
            
        except Exception as e:
            metrics.error_count += 1
            logger.error(f"Error in node {config.name}: {str(e)}", exc_info=True)
            raise
    
    # Add metadata to the wrapper function
    wrapper.__name__ = f"node_{config.name}"
    wrapper.__doc__ = config.description or node_func.__doc__
    wrapper.config = config
    wrapper.metrics = metrics
    
    return wrapper

# Example usage
@create_node(
    config=NodeConfig(
        name="process_user_input",
        description="Processes and validates user input from the conversation",
        timeout_seconds=5.0,
        retry_attempts=2
    )
)
def process_user_input(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and process user input from the conversation.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Dictionary with updates to the state
        
    Raises:
        ValueError: If the input is invalid or missing required fields
    """
    if not state.get("messages"):
        raise ValueError("No messages in state")
        
    last_message = state["messages"][-1]
    
    # Validate message format
    if not isinstance(last_message, dict) or "role" not in last_message or "content" not in last_message:
        raise ValueError("Invalid message format")
        
    if last_message["role"] == "user":
        return {
            "user_query": str(last_message["content"]).strip(),
            "status": "processing"
        }
        
    return {}

@create_node(
    config=NodeConfig(
        name="generate_response",
        description="Generates a response using an LLM",
        timeout_seconds=30.0
    )
)
def generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a response using an LLM based on the current state."""
    try:
        # In a real app, you would call your LLM here
        query = state.get("user_query", "")
        
        # Simulate LLM call
        response = {
            "role": "assistant",
            "content": f"You asked: {query}",
            "metadata": {
                "model": "gpt-4",
                "tokens_used": len(query) // 4,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        return {
            "messages": [response],
            "status": "completed"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
```

### 3. Edges: Controlling the Flow

Edges define how control flows between nodes in your workflow. They can be unconditional (always taken) or conditional (taken based on some condition).

#### Edge Types

1. **Unconditional Edges**: Always followed after a node completes
2. **Conditional Edges**: Followed based on a condition function
3. **Fallback Edges**: Taken when no other conditions are met
4. **Error Edges**: Triggered when a node raises an exception

#### Advanced Edge Patterns

```python
from enum import Enum
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass

class EdgeType(Enum):
    UNCONDITIONAL = "unconditional"
    CONDITIONAL = "conditional"
    FALLBACK = "fallback"
    ERROR = "error"

@dataclass
class Edge:
    source: str
    target: str
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    edge_type: EdgeType = EdgeType.UNCONDITIONAL
    priority: int = 0  # Higher priority edges are evaluated first
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}
        if self.condition is not None and self.edge_type == EdgeType.UNCONDITIONAL:
            self.edge_type = EdgeType.CONDITIONAL

def create_workflow():
    """Create a workflow with advanced edge handling."""
    # Define the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("process_input", process_user_input)
    workflow.add_node("validate_input", validate_user_input)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("handle_error", handle_error)
    workflow.add_node("log_activity", log_activity)
    
    # Define edges
    edges = [
        # Process input first
        Edge("process_input", "validate_input"),
        
        # Validate input
        Edge(
            "validate_input", 
            "generate_response",
            condition=lambda s: s.get("is_valid", False),
            edge_type=EdgeType.CONDITIONAL,
            priority=1,
            metadata={"description": "Proceed if input is valid"}
        ),
        Edge(
            "validate_input",
            "handle_error",
            condition=lambda s: not s.get("is_valid", True),
            edge_type=EdgeType.CONDITIONAL,
            priority=1,
            metadata={"description": "Handle invalid input"}
        ),
        
        # Error handling
        Edge("handle_error", "log_activity"),
        
        # After successful response
        Edge("generate_response", "log_activity"),
        
        # Final step
        Edge("log_activity", END)
    ]
    
    # Add edges to the workflow
    for edge in sorted(edges, key=lambda e: -e.priority):
        if edge.edge_type == EdgeType.UNCONDITIONAL:
            workflow.add_edge(edge.source, edge.target)
        elif edge.edge_type == EdgeType.CONDITIONAL and edge.condition:
            workflow.add_conditional_edges(
                edge.source,
                edge.condition,
                {edge.target: edge.target}
            )
    
    # Set entry point
    workflow.set_entry_point("process_input")
    
    # Compile the workflow
    return workflow.compile()

# Example usage
app = create_workflow()

# Run the workflow with error handling
try:
    result = app.invoke({
        "messages": [
            {
                "role": "user", 
                "content": "Hello, world!",
                "timestamp": datetime.utcnow().isoformat()
            }
        ],
        "user_query": "",
        "search_results": [],
        "final_answer": "",
        "status": "processing",
        "context": {
            "session_id": "test_123",
            "request_id": "req_456"
        }
    })
    
    print(f"Workflow completed: {result['status']}")
    
except Exception as e:
    print(f"Workflow failed: {str(e)}")
    raise
```

### 4. Advanced Conditional Logic

Conditional logic enables dynamic workflow routing based on the current state. LangGraph provides several patterns for implementing complex decision-making in your workflows.

#### Pattern 1: Simple Conditional Routing

```python
def route_by_intent(state: AgentState) -> str:
    """Route based on the detected intent of the user's message."""
    last_message = state["messages"][-1]["content"].lower()
    
    # Simple keyword-based intent detection
    if any(word in last_message for word in ["bye", "goodbye", "see you"]):
        return "end_conversation"
    elif any(word in last_message for word in ["search", "find", "look up"]):
        return "web_search"
    elif "help" in last_message:
        return "show_help"
    return "generate_response"

# Add conditional edges with metadata
workflow.add_conditional_edges(
    "process_input",
    route_by_intent,
    {
        "end_conversation": (END, {"description": "End the conversation"}),
        "web_search": ("search_web", {"description": "Perform a web search"}),
        "show_help": ("show_help", {"description": "Display help information"}),
        "generate_response": ("generate_response", {
            "description": "Generate a standard response"
        })
    }
)
```

#### Pattern 2: Multi-Factor Decision Making

```python
class DecisionFactors(BaseModel):
    """Factors to consider when making routing decisions."""
    confidence: float = Field(..., ge=0.0, le=1.0)
    intent: str
    requires_human: bool = False
    priority: int = 1

def make_complex_decision(state: AgentState) -> DecisionFactors:
    """Make a routing decision based on multiple factors."""
    last_message = state["messages"][-1]["content"]
    
    # In a real app, this might call an LLM or other analysis
    return DecisionFactors(
        confidence=0.85,
        intent="information_request",
        requires_human=len(state["messages"]) > 5,
        priority=2 if "urgent" in last_message.lower() else 1
    )

def route_by_decision_factors(factors: DecisionFactors) -> str:
    """Determine the next step based on decision factors."""
    if factors.requires_human:
        return "escalate_to_agent"
    if factors.confidence < 0.5:
        return "request_clarification"
    return "process_request"

# In your workflow setup:
workflow.add_node("analyze_request", lambda s: {"decision": make_complex_decision(s)})
workflow.add_conditional_edges(
    "analyze_request",
    lambda s: route_by_decision_factors(s["decision"]),
    {
        "escalate_to_agent": "human_escalation",
        "request_clarification": "clarify_intent",
        "process_request": "process_request"
    }
)
```

#### Pattern 3: State Machine Pattern

```python
from enum import Enum, auto

class ConversationState(Enum):
    AWAITING_INPUT = auto()
    PROCESSING = auto()
    AWAITING_CONFIRMATION = auto()
    COMPLETED = auto()
    ERROR = auto()

def update_conversation_state(state: AgentState) -> dict:
    """Update the conversation state based on the current context."""
    messages = state.get("messages", [])
    if not messages:
        return {"conversation_state": ConversationState.AWAITING_INPUT}
        
    last_message = messages[-1]
    
    if state.get("error"):
        return {"conversation_state": ConversationState.ERROR}
        
    if "confirm" in last_message.get("content", "").lower():
        return {"conversation_state": ConversationState.AWAITING_CONFIRMATION}
        
    return {"conversation_state": ConversationState.PROCESSING}

def route_by_conversation_state(state: AgentState) -> str:
    """Route based on the current conversation state."""
    conv_state = state.get("conversation_state", ConversationState.AWAITING_INPUT)
    
    if conv_state == ConversationState.ERROR:
        return "handle_error"
    elif conv_state == ConversationState.AWAITING_CONFIRMATION:
        return "process_confirmation"
    elif conv_state == ConversationState.PROCESSING:
        return "process_message"
    return "get_user_input"

# Add state management node
workflow.add_node("update_state", update_conversation_state)

# Add conditional routing
workflow.add_conditional_edges(
    "update_state",
    route_by_conversation_state,
    {
        "handle_error": "error_handling",
        "process_confirmation": "confirm_action",
        "process_message": "process_content",
        "get_user_input": "get_input"
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

### 1. State Management

#### Do:
- **Use Strong Typing**: Leverage Pydantic models or TypedDict for state validation
- **Keep State Flat**: Prefer flat structures over deeply nested ones
- **Immutable Updates**: Always return new state objects instead of mutating
- **Versioning**: Include version information in your state schema
- **Sensitive Data**: Never store sensitive information in the state

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import hashlib

class AppState(BaseModel):
    """Application state with versioning and validation."""
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[dict] = []
    metadata: dict = Field(default_factory=dict)
    
    class Config:
        # Enable arbitrary types for complex objects
        arbitrary_types_allowed = True
        
    @property
    def state_hash(self) -> str:
        """Generate a hash of the current state for change detection."""
        return hashlib.md5(self.json().encode()).hexdigest()
```

### 2. Node Design

#### Do:
- **Single Responsibility**: Each node should do one thing well
- **Idempotency**: Design nodes to be safely retryable
- **Pure Functions**: Avoid side effects when possible
- **Documentation**: Clearly document inputs, outputs, and side effects
- **Error Handling**: Handle expected errors gracefully

```python
def process_data_node(state: dict) -> dict:
    """Process input data and return updated state.
    
    Args:
        state: Current application state containing:
            - input_data: The data to process
            - processing_config: Configuration for processing
            
    Returns:
        dict: Updated state with processed data
        
    Raises:
        ValueError: If input data is invalid
        ProcessingError: For processing-specific errors
    """
    try:
        # Validate input
        if not state.get("input_data"):
            raise ValueError("No input data provided")
            
        # Process data
        result = some_processing_function(
            state["input_data"],
            **state.get("processing_config", {})
        )
        
        # Return state updates
        return {
            "processed_data": result,
            "status": "completed",
            "last_processed": datetime.utcnow()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
```

### 3. Error Handling

#### Strategies:
1. **Retry Logic**: For transient failures
2. **Circuit Breaker**: To prevent cascading failures
3. **Fallbacks**: Provide alternative paths when primary fails
4. **Dead Letter Queues**: For handling failed messages
5. **Monitoring**: Track and alert on errors

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState
)
import logging

logger = logging.getLogger(__name__)

def log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log retry attempts with context."""
    logger.warning(
        f"Retrying {retry_state.fn.__name__}: "
        f"attempt {retry_state.attempt_number} "
        f"ended with: {retry_state.outcome.exception()}"
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    before_sleep=log_retry_attempt,
    reraise=True
)
async def fetch_external_data(url: str, timeout: float = 5.0) -> dict:
    """Fetch data from an external API with retry logic."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=timeout) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
            raise

class CircuitBreaker:
    """Simple circuit breaker pattern implementation."""
    def __init__(self, max_failures=3, reset_timeout=60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure = None
        self.is_open = False
        
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            if self.is_open:
                if (datetime.utcnow() - self.last_failure).total_seconds() > self.reset_timeout:
                    self.is_open = False
                else:
                    raise CircuitOpenError("Service unavailable (circuit open)")
                    
            try:
                result = await func(*args, **kwargs)
                self.failures = 0
                return result
            except Exception as e:
                self.failures += 1
                self.last_failure = datetime.utcnow()
                if self.failures >= self.max_failures:
                    self.is_open = True
                raise
        return wrapper
```

### 4. Performance Optimization

#### Techniques:
1. **Caching**: Cache expensive operations
2. **Batching**: Process multiple items at once
3. **Parallelism**: Run independent operations concurrently
4. **Lazy Loading**: Defer expensive operations until needed
5. **Pagination**: Process large datasets in chunks

```python
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

# Caching
@lru_cache(maxsize=128)
def get_expensive_resource(resource_id: str):
    """Get a resource with caching."""
    return fetch_from_database(resource_id)

# Batching
async def process_batch(items: list, batch_size: int = 10):
    """Process items in batches."""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_item(item) for item in batch],
            return_exceptions=True
        )
        results.extend(batch_results)
    return results

# Parallel processing
def parallel_process(items: list, func: callable, max_workers: int = 4) -> list:
    """Process items in parallel using a thread pool."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        return [future.result() for future in as_completed(futures)]
```

## Real-World Use Cases

### 1. Customer Support Chatbot

```python
from typing import List, Dict, Any, TypedDict, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class SupportTicket(BaseModel):
    """Represents a customer support ticket."""
    ticket_id: str
    customer_id: str
    subject: str
    description: str
    priority: str = "normal"
    status: str = "open"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = {}

class SupportBotState(TypedDict):
    """State for the customer support chatbot."""
    messages: List[Dict[str, str]]
    ticket: Optional[SupportTicket]
    search_results: List[Dict[str, Any]]
    suggested_responses: List[str]
    status: str
    context: Dict[str, Any]

def create_support_bot():
    """Create a customer support chatbot workflow."""
    workflow = StateGraph(SupportBotState)
    
    # Add nodes
    workflow.add_node("greet_customer", greet_customer_node)
    workflow.add_node("classify_issue", classify_issue_node)
    workflow.add_node("search_knowledge_base", search_knowledge_base_node)
    workflow.add_node("create_ticket", create_ticket_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("escalate_to_agent", escalate_to_agent_node)
    
    # Define edges
    workflow.add_edge("greet_customer", "classify_issue")
    
    workflow.add_conditional_edges(
        "classify_issue",
        lambda s: "escalate" if s.get("requires_human") else "search_knowledge_base",
        {
            "search_knowledge_base": "search_knowledge_base",
            "escalate": "escalate_to_agent"
        }
    )
    
    workflow.add_conditional_edges(
        "search_knowledge_base",
        lambda s: "create_ticket" if s.get("needs_ticket") else "generate_response",
        {
            "create_ticket": "create_ticket",
            "generate_response": "generate_response"
        }
    )
    
    workflow.add_edge("create_ticket", "generate_response")
    workflow.add_edge("generate_response", END)
    workflow.add_edge("escalate_to_agent", END)
    
    workflow.set_entry_point("greet_customer")
    return workflow.compile()
```

### 2. E-commerce Order Processing

```python
class OrderStatus(str, Enum):
    RECEIVED = "received"
    VALIDATING = "validating"
    PAYMENT_PROCESSING = "payment_processing"
    INVENTORY_CHECK = "inventory_check"
    SHIPPING = "shipping"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class Order(BaseModel):
    order_id: str
    customer_id: str
    items: List[Dict[str, Any]]
    total_amount: float
    status: OrderStatus = OrderStatus.RECEIVED
    payment_status: str = "pending"
    shipping_address: Dict[str, str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

def create_order_workflow():
    """Create an order processing workflow."""
    workflow = StateGraph(Order)
    
    # Add nodes
    workflow.add_node("validate_order", validate_order_node)
    workflow.add_node("process_payment", process_payment_node)
    workflow.add_node("check_inventory", check_inventory_node)
    workflow.add_node("prepare_shipment", prepare_shipment_node)
    workflow.add_node("update_order_status", update_order_status_node)
    workflow.add_node("handle_failure", handle_failure_node)
    
    # Define edges
    workflow.add_edge("validate_order", "process_payment")
    workflow.add_edge("process_payment", "check_inventory")
    workflow.add_edge("check_inventory", "prepare_shipment")
    workflow.add_edge("prepare_shipment", "update_order_status")
    workflow.add_edge("update_order_status", END)
    
    # Add error handling
    workflow.add_conditional_edges(
        "validate_order",
        lambda s: "handle_failure" if s.get("validation_error") else "process_payment",
        {"process_payment": "process_payment", "handle_failure": "handle_failure"}
    )
    
    workflow.set_entry_point("validate_order")
    return workflow.compile()
```

## Testing Strategies

### 1. Unit Testing Nodes

```python
import pytest
from unittest.mock import patch, MagicMock

# Test data
TEST_ORDER = {
    "order_id": "order_123",
    "customer_id": "cust_456",
    "items": [{"product_id": "prod_789", "quantity": 2}],
    "total_amount": 99.98,
    "shipping_address": {"zip": "10001"},
    "status": "received"
}

def test_validate_order_node():
    """Test the order validation node."""
    # Test valid order
    state = {"order": Order(**TEST_ORDER)}
    result = validate_order_node(state)
    assert result["status"] == "validating"
    assert "validation_errors" not in result
    
    # Test invalid order (missing required field)
    invalid_order = TEST_ORDER.copy()
    del invalid_order["shipping_address"]
    with pytest.raises(ValueError):
        validate_order_node({"order": Order(**invalid_order)})

@patch("payment_gateway.charge_customer")
def test_process_payment_node(mock_charge):
    """Test payment processing with successful and failed scenarios."""
    # Mock successful payment
    mock_charge.return_value = {"status": "succeeded", "transaction_id": "txn_123"}
    
    state = {
        "order": Order(**TEST_ORDER),
        "payment_method": {"token": "pm_123"}
    }
    result = process_payment_node(state)
    assert result["payment_status"] == "succeeded"
    assert result["transaction_id"] == "txn_123"
    
    # Test payment failure
    mock_charge.side_effect = PaymentError("Insufficient funds")
    with pytest.raises(PaymentError):
        process_payment_node(state)
```

### 2. Integration Testing

```python
import asyncio

class TestOrderWorkflow:
    @pytest.fixture
    def workflow(self):
        """Create a test workflow instance."""
        return create_order_workflow()
    
    @pytest.mark.asyncio
    async def test_complete_order_flow(self, workflow):
        """Test the complete order workflow with valid data."""
        test_order = Order(
            order_id="test_123",
            customer_id="test_cust_456",
            items=[{"product_id": "test_prod_789", "quantity": 1}],
            total_amount=49.99,
            shipping_address={"zip": "10001"}
        )
        
        result = await workflow.ainvoke({"order": test_order.dict()})
        
        assert result["status"] == "completed"
        assert result["payment_status"] == "succeeded"
        assert result["shipping_tracking_number"] is not None
```

### 3. Performance Testing

```python
import time
import asyncio

class PerformanceTest:
    def __init__(self, workflow):
        self.workflow = workflow
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": []
        }
    
    async def run_test(self, test_cases: List[Dict[str, Any]], concurrency: int = 10):
        """Run performance test with the given test cases."""
        start_time = time.time()
        
        # Process test cases in batches
        for i in range(0, len(test_cases), concurrency):
            batch = test_cases[i:i + concurrency]
            tasks = [self._process_single_case(case) for case in batch]
            await asyncio.gather(*tasks)
        
        # Calculate metrics
        total_time = time.time() - start_time
        self.metrics["total_time"] = total_time
        self.metrics["requests_per_second"] = self.metrics["total_requests"] / total_time
        
        if self.metrics["response_times"]:
            self.metrics["avg_response_time"] = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
        
        return self.metrics
    
    async def _process_single_case(self, test_case: Dict[str, Any]):
        """Process a single test case and record metrics."""
        self.metrics["total_requests"] += 1
        start_time = time.time()
        
        try:
            await self.workflow.ainvoke(test_case)
            self.metrics["successful_requests"] += 1
            self.metrics["response_times"].append(time.time() - start_time)
        except Exception:
            self.metrics["failed_requests"] += 1
```
```

### 6. Monitoring and Observability

#### Key Metrics to Track:
- Node execution time
- Error rates
- Queue lengths
- Resource usage
- Throughput

```python
from prometheus_client import start_http_server, Counter, Histogram
import time

# Define metrics
NODE_EXECUTION_TIME = Histogram(
    'node_execution_time_seconds',
    'Time spent processing nodes',
    ['node_name']
)

NODE_ERRORS = Counter(
    'node_errors_total',
    'Total number of node errors',
    ['node_name', 'error_type']
)

def track_node_metrics(node_name: str):
    """Decorator to track node metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                NODE_EXECUTION_TIME.labels(node_name=node_name).observe(
                    time.time() - start_time
                )
                return result
            except Exception as e:
                NODE_ERRORS.labels(
                    node_name=node_name,
                    error_type=type(e).__name__
                ).inc()
                raise
        return wrapper
    return decorator

# Usage
@track_node_metrics("process_data_node")
async def process_data_node(state: dict) -> dict:
    """Process data with metrics tracking."""
    # Implementation...
    return state
```

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
