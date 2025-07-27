# Quick Start Guide

Welcome to the LangChain ecosystem! This guide will help you get up and running quickly with the core components.

## Prerequisites

- Python 3.8 or later
- An OpenAI API key (or other LLM provider API key)
- Optional: LangSmith API key for tracing and monitoring

## Installation

Install the core LangChain package:

```bash
pip install langchain
```

For additional features, install optional dependencies:

```bash
# For working with embeddings
pip install langchain[embeddings]

# For working with vector stores
pip install langchain[vectorstores]

# For working with agents and tools
pip install langchain[agents]
```

## Your First LangChain Application

Let's create a simple application that uses a language model to answer questions.

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the language model
llm = OpenAI(temperature=0.7)

# Create a prompt template
template = """You are a helpful assistant that answers questions about {topic}.

Question: {question}
Answer: """

prompt = PromptTemplate(
    input_variables=["topic", "question"],
    template=template
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
response = chain.run(
    topic="artificial intelligence",
    question="What are the main benefits of AI?"
)

print(response)
```

## Adding Memory to Conversations

Let's enhance our application with conversation memory:

```python
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate

# Create a prompt template with memory
template = """You are a helpful assistant.

Conversation History:
{history}

Human: {input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

# Initialize memory and chain
memory = ConversationBufferMemory(memory_key="history")
llm_chain = LLMChain(
    llm=OpenAI(temperature=0.7),
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Have a conversation
print(llm_chain.predict(input="Hi, my name is Alice"))
print(llm_chain.predict(input="What's my name?"))
```

## Using Tools and Agents

Let's create an agent that can use tools to answer questions:

```python
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

# Initialize the language model
llm = OpenAI(temperature=0)

# Load tools (requires additional API keys)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ask a question
agent.run("What is the population of Canada? What is that number multiplied by 2?")
```

## Adding LangSmith for Tracing

To monitor and debug your LangChain applications, use LangSmith:

```python
import os
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Set up your API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langchain-api-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize the language model with tracing
llm = OpenAI(temperature=0.7)

# Use a context manager to trace the execution
with get_openai_callback() as cb:
    response = llm("Tell me a joke about artificial intelligence")
    print(response)
    print(f"Tokens used: {cb}")
```

## Next Steps

1. Explore more examples in the [Basic Usage](../../examples/basic-usage) directory
2. Dive into [Advanced Usage](../../examples/advanced-usage) for more complex scenarios
3. Check out the [Documentation](../../docs) for detailed guides and API references
4. Visit [LangSmith](https://smith.langchain.com) to monitor your applications

## Getting Help

- [Documentation](https://docs.langchain.com)
- [GitHub Issues](https://github.com/langchain-ai/langchain/issues)
- [Discord Community](https://discord.gg/langchain)
