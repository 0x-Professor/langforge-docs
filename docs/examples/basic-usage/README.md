# Basic Usage Examples

This directory contains basic usage examples for the LangChain ecosystem. These examples are designed to help you get started with the core functionality.

## Table of Contents

1. [Chat Models](#chat-models)
2. [Embeddings](#embeddings)
3. [Chains](#chains)
4. [Memory](#memory)
5. [Tools](#tools)

## Chat Models

### Basic Chat

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(temperature=0.7)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me a joke about AI")
]
response = chat(messages)
print(response.content)
```

### Streaming Responses

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(streaming=True)
for chunk in chat.stream([HumanMessage(content="Write a short poem about AI")]):
    print(chunk.content, end="", flush=True)
```

## Embeddings

### Basic Embedding Generation

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
text = "This is a sample text for embedding"
vector = embeddings.embed_query(text)
print(f"Vector length: {len(vector)}")
print(f"First 5 dimensions: {vector[:5]}")
```

## Chains

### Simple Chain

```python
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a fact about {topic}"
)

llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("quantum computing"))
```

## Memory

### Conversation Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, OpenAI, PromptTemplate

template = """You are a helpful assistant.

Conversation History:
{history}

Human: {input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

memory = ConversationBufferMemory(memory_key="history")
llm_chain = LLMChain(
    llm=OpenAI(temperature=0),
    prompt=prompt,
    memory=memory,
    verbose=True
)

print(llm_chain.predict(input="Hi, my name is Alice"))
print(llm_chain.predict(input="What's my name?"))
```

## Tools

### Using Built-in Tools

```python
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("What was the high temperature in SF yesterday in Celsius?")
```

---

For more advanced examples, see the [Advanced Usage](../advanced-usage) directory.
