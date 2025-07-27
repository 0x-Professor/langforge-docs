# Agents

## Overview

Agents in LangChain are systems that use a language model to determine a sequence of actions to take. They can use tools, access memory, and make decisions based on the current state of the environment.

Agents are particularly useful for tasks that require dynamic decision-making and the ability to use external tools or APIs. They can handle complex workflows that would be difficult to implement with simple chains.

## Agent Types

LangChain provides several built-in agent types, each designed for different use cases:

### Zero-shot ReAct
Uses the ReAct framework to decide which tool to use based on the tool's description.
```python
agent="zero-shot-react-description"
```

### OpenAI Functions
Uses OpenAI's function calling capabilities for more structured tool use.
```python
agent="openai-functions"
```

### Conversational
Designed for conversational agents that need to maintain context across multiple turns.
```python
agent="conversational-react-description"
```

### Structured Chat
Handles multi-input tools and structured output better than other agent types.
```python
agent="structured-chat-zero-shot-react-description"
```

## Basic Agent Example

```python
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_openai import OpenAI
from langchain_community.utilities import GoogleSearchAPIWrapper

# Initialize the language model
llm = OpenAI(temperature=0)

# Set up Google Search as a tool
search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events"
    )
]

# Initialize the agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
result = agent.run("What's the latest news about AI?")
print(result)
```

## Modern Agent Implementation with LangGraph

For more complex and production-ready agents, use LangGraph:

```python
from typing import TypedDict, List
from langgraph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# Define the agent state
class AgentState(TypedDict):
    messages: List[dict]
    next_action: str

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define tools
def search_tool(query: str) -> str:
    """Search for information."""
    # Your search implementation here
    return f"Search results for: {query}"

def calculator_tool(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"

tools = {
    "search": search_tool,
    "calculator": calculator_tool
}

# Define agent nodes
def call_model(state: AgentState):
    """Call the language model to decide what to do next."""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response]}

def call_tool(state: AgentState):
    """Execute a tool based on the model's decision."""
    last_message = state["messages"][-1]
    # Parse tool call from the message (simplified)
    if "search:" in last_message.content:
        tool_input = last_message.content.split("search:")[-1].strip()
        result = tools["search"](tool_input)
    elif "calculate:" in last_message.content:
        tool_input = last_message.content.split("calculate:")[-1].strip()
        result = tools["calculator"](tool_input)
    else:
        result = "No tool called"
    
    return {"messages": state["messages"] + [HumanMessage(content=f"Tool result: {result}")]}

def should_continue(state: AgentState) -> str:
    """Determine if the agent should continue or end."""
    last_message = state["messages"][-1]
    if "FINAL ANSWER" in last_message.content:
        return "end"
    return "continue"

# Build the agent graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)
workflow.add_edge("action", "agent")

# Compile the agent
agent = workflow.compile()

# Run the agent
result = agent.invoke({
    "messages": [HumanMessage(content="What is 25 * 4 and then search for information about that number?")]
})
```

## Tools

Tools are functions that agents can use to interact with the world. They can be anything from search engines to calculators to custom functions.

### Built-in Tools
- `GoogleSearchAPIWrapper`: Perform web searches
- `DuckDuckGoSearchRun`: Alternative search engine
- `PythonREPLTool`: Execute Python code
- `RequestsGetTool`: Make HTTP GET requests
- `WikipediaQueryRun`: Query Wikipedia

### Creating Custom Tools

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="Operation to perform: add, subtract, multiply, divide")

class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "Useful for when you need to perform mathematical calculations"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, a: float, b: float, operation: str) -> str:
        """Perform mathematical operations."""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Cannot divide by zero"
        }
        
        if operation.lower() in operations:
            result = operations[operation.lower()](a, b)
            return str(result)
        else:
            return "Unsupported operation"
    
    async def _arun(self, a: float, b: float, operation: str) -> str:
        """Async version of the tool."""
        return self._run(a, b, operation)

# Create an instance of the tool
calculator = CustomCalculatorTool()

# Use the tool in an agent
tools = [calculator]
agent = initialize_agent(
    tools,
    OpenAI(temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("Calculate 15 multiplied by 8")
print(result)
```

## Tool Integration with Function Calling

For models that support function calling (like OpenAI's GPT models):

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

# Initialize the model with function calling
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create tools list
tools = [get_word_length, multiply_numbers]

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
result = agent_executor.invoke({
    "input": "What's the length of the word 'hello' multiplied by 3?"
})
print(result["output"])
```

## Multi-Agent Systems

You can create systems with multiple agents that work together to solve complex problems:

```python
from langgraph import StateGraph, END
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

class MultiAgentState(TypedDict):
    messages: List[dict]
    task: str
    research_output: str
    analysis_output: str
    final_output: str

# Initialize different models for different agents
research_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
analysis_llm = ChatOpenAI(model="gpt-4", temperature=0.3)

def research_agent(state: MultiAgentState):
    """Agent responsible for research tasks."""
    task = state["task"]
    prompt = f"Research the following topic and provide key information: {task}"
    
    response = research_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "research_output": response.content,
        "messages": state["messages"] + [AIMessage(content=f"Research: {response.content}")]
    }

def analysis_agent(state: MultiAgentState):
    """Agent responsible for analysis tasks."""
    research = state["research_output"]
    prompt = f"Analyze the following research and provide insights: {research}"
    
    response = analysis_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "analysis_output": response.content,
        "messages": state["messages"] + [AIMessage(content=f"Analysis: {response.content}")]
    }

def synthesis_agent(state: MultiAgentState):
    """Agent responsible for synthesizing final output."""
    research = state["research_output"]
    analysis = state["analysis_output"]
    
    prompt = f"""
    Synthesize the following research and analysis into a comprehensive final report:
    
    Research: {research}
    Analysis: {analysis}
    """
    
    response = analysis_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "final_output": response.content,
        "messages": state["messages"] + [AIMessage(content=f"Final Report: {response.content}")]
    }

# Build the multi-agent workflow
workflow = StateGraph(MultiAgentState)

workflow.add_node("research", research_agent)
workflow.add_node("analysis", analysis_agent)
workflow.add_node("synthesis", synthesis_agent)

workflow.set_entry_point("research")
workflow.add_edge("research", "analysis")
workflow.add_edge("analysis", "synthesis")
workflow.add_edge("synthesis", END)

# Compile and run
multi_agent = workflow.compile()

result = multi_agent.invoke({
    "messages": [],
    "task": "Impact of artificial intelligence on the job market",
    "research_output": "",
    "analysis_output": "",
    "final_output": ""
})

print("Final Report:", result["final_output"])
```

## Agent with Memory

Create agents that remember conversation history:

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_openai import OpenAI
from langchain.tools import Tool

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a simple tool
def get_current_time():
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [
    Tool(
        name="CurrentTime",
        func=get_current_time,
        description="Get the current date and time"
    )
]

# Create agent with memory
agent = initialize_agent(
    tools,
    OpenAI(temperature=0),
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Have a conversation
print(agent.run("What time is it?"))
print(agent.run("What did I just ask you?"))
```

## Best Practices

### 1. Choose the Right Agent Type
- Use **OpenAI Functions** for structured tool calling with GPT models
- Use **Zero-shot ReAct** for simple tool selection
- Use **LangGraph** for complex multi-step workflows

### 2. Provide Clear Tool Descriptions
```python
# Good tool description
Tool(
    name="Calculator",
    func=calculator,
    description="Performs basic arithmetic operations. Input should be a mathematical expression like '2 + 3' or '10 * 5'"
)

# Poor tool description
Tool(
    name="Calculator",
    func=calculator,
    description="Does math"
)
```

### 3. Handle Errors Gracefully
```python
def safe_tool_function(input_data):
    """A tool function with proper error handling."""
    try:
        # Your tool logic here
        result = process_data(input_data)
        return f"Success: {result}"
    except ValueError as e:
        return f"Input error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
```

### 4. Use Memory Effectively
- **Buffer Memory**: For short conversations
- **Summary Memory**: For long conversations
- **Vector Memory**: For semantic search over history

### 5. Monitor and Evaluate
```python
# Add logging to track agent performance
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_tool_function(input_data):
    """Tool with logging."""
    logger.info(f"Tool called with input: {input_data}")
    result = process_data(input_data)
    logger.info(f"Tool returned: {result}")
    return result
```

### 6. Limit Tool Access
Only provide tools that are necessary for the task. Too many tools can confuse the agent and slow down decision-making.

### 7. Test Agent Behavior
```python
# Test your agent with edge cases
test_cases = [
    "Normal question",
    "Question requiring multiple tools",
    "Ambiguous question",
    "Question with no clear answer"
]

for test in test_cases:
    try:
        result = agent.run(test)
        print(f"Test: {test}")
        print(f"Result: {result}")
        print("---")
    except Exception as e:
        print(f"Test failed: {test}")
        print(f"Error: {e}")
        print("---")
```