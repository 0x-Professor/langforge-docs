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

### Self-ask with Search
Uses a single tool (typically a search tool) to find information and answer questions.
```python
agent="self-ask-with-search"
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
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import OpenAI
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
    agent="zero-shot-react-description",
    verbose=True
)

# Run the agent
agent.run("What's the latest news about AI?")
```

## Tools

Tools are functions that agents can use to interact with the world. They can be anything from search engines to calculators to custom functions.

### Built-in Tools
- `GoogleSearchAPIWrapper`: Perform web searches
- `WolframAlphaQueryRun`: Access computational knowledge
- `PythonREPLTool`: Execute Python code
- `RequestsGetTool`: Make HTTP GET requests
- `VectorDBQA`: Query a vector database

### Creating Custom Tools

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "Useful for when you need to perform mathematical calculations"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, a: int, b: int) -> str:
        """Add two numbers together."""
        return str(a + b)
    
    async def _arun(self, a: int, b: int) -> str:
        """Async version of the tool."""
        return self._run(a, b)

# Create an instance of the tool
calculator = CustomCalculatorTool()

# Use the tool
result = calculator.run({"a": 5, "b": 3})
print(f"5 + 3 = {result}")
```

## Custom Agents

For more complex use cases, you can create custom agents by subclassing the base agent class. This gives you full control over the agent's behavior.

```python
from typing import List, Tuple, Any, Optional
from langchain.agents import BaseSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain

# 1. Define a custom prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Format the tools into a string
        tools_string = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # Format the prompt
        return self.template.format(
            tools=tools_string,
            **kwargs
        )

# 2. Create a custom output parser
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> AgentAction | AgentFinish:
        # Parse the LLM output to determine the next action
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output
            )
        
        # Parse the action and action input
        action, action_input = llm_output.split("Action Input:")
        action = action.replace("Action:", "").strip()
        action_input = action_input.strip()
        
        return AgentAction(
            tool=action, 
            tool_input=action_input.strip("\""), 
            log=llm_output
        )

# 3. Create a custom agent
class CustomAgent(BaseSingleActionAgent):
    llm_chain: LLMChain
    output_parser: AgentOutputParser
    stop: List[str]
    
    @property
    def input_keys(self):
        return ["input"]
    
    def plan(self, intermediate_steps, **kwargs):
        # Get the output from the LLM
        output = self.llm_chain.run(**kwargs)
        
        # Parse the output
        return self.output_parser.parse(output)
    
    async def aplan(self, intermediate_steps, **kwargs):
        # Async version of plan
        output = await self.llm_chain.arun(**kwargs)
        return self.output_parser.parse(output)
```

## Multi-Agent Systems

You can create systems with multiple agents that work together to solve complex problems. Each agent can have its own role, tools, and memory.

```python
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI

# Define tools for the agents
search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Useful for performing mathematical calculations"
    )
]

# Create a memory object
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the agents
researcher = initialize_agent(
    tools, 
    OpenAI(temperature=0), 
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

analyst = initialize_agent(
    tools, 
    OpenAI(temperature=0), 
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Simulate a conversation between agents
def simulate_conversation(question):
    # Researcher finds information
    research = researcher.run(f"Research information about: {question}")
    
    # Analyst processes the information
    analysis = analyst.run(f"Analyze this information: {research}")
    
    return analysis

# Run the simulation
result = simulate_conversation("latest advancements in renewable energy")
print(result)
```

## Best Practices

### 1. Choose the Right Agent Type
Select an agent type that matches your use case. For simple tool use, a zero-shot agent might be sufficient, while complex workflows might require a custom agent.

### 2. Provide Clear Tool Descriptions
Write clear and descriptive tool descriptions. The agent uses these descriptions to decide which tool to use, so be specific about what each tool does and when it should be used.

### 3. Handle Errors Gracefully
Implement error handling in your tools and agents to manage cases where tools fail or return unexpected results. This makes your agent more robust in production.

### 4. Use Memory Effectively
For conversational agents, use memory to maintain context across multiple turns. This allows the agent to reference previous parts of the conversation.

### 5. Monitor and Evaluate
Track how your agent performs in production. Monitor metrics like tool usage, success rates, and user satisfaction to identify areas for improvement.

### 6. Limit Tool Access
Only give your agent access to the tools it needs. This reduces the complexity of the agent's decision-making and improves security.