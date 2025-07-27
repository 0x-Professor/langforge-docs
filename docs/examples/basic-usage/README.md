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
<div align="center">

# 🚀 Basic Usage Examples

**Master LangChain Fundamentals - Start Here!**

[![Beginner Friendly](https://img.shields.io/badge/level-beginner-green.svg)](.)
[![Interactive Examples](https://img.shields.io/badge/examples-interactive-blue.svg)](.)
[![Step by Step](https://img.shields.io/badge/learning-guided-orange.svg)](.)

</div>

---

## 🎯 Welcome to LangChain Basics!

This directory contains foundational examples that will teach you everything you need to know to start building with LangChain. Whether you're completely new to LLMs or just new to LangChain, these examples will get you up and running quickly.

---

## 📚 What's Included

### **🌟 [Complete Comprehensive Guide](comprehensive-guide.md)**
> **START HERE** - Our flagship beginner tutorial covering all the basics

**What you'll learn:**
- 🤖 **Chat Models** - Talk to different AI providers
- 📊 **Embeddings** - Convert text to searchable vectors
- ⛓️ **Chains** - Connect components together
- 🧠 **Memory** - Give your AI context and history
- 🛠️ **Tools** - Extend AI with external capabilities
- 🔍 **Vector Stores** - Build semantic search systems
- 🎯 **Complete Project** - Put it all together in a working app

---

## 🚀 Quick Start Options

### **Option 1: Comprehensive Learning (Recommended)**
📖 **[Read the Complete Guide →](comprehensive-guide.md)**

Perfect for: First-time users, systematic learners, building strong foundations

### **Option 2: Quick Reference**
Below are bite-sized examples if you need something specific:

<details>
<summary><strong>🤖 Basic Chat Example</strong></summary>

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Initialize chat model
chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Create messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain LangChain in simple terms")
]

# Get response
response = chat(messages)
print(response.content)
```

</details>

<details>
<summary><strong>⛓️ Simple Chain Example</strong></summary>

```python
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

# Create prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms"
)

# Create chain
llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt)

# Use chain
result = chain.run("machine learning")
print(result)
```

</details>

<details>
<summary><strong>🧠 Memory Example</strong></summary>

```python
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, OpenAI, PromptTemplate

# Create memory
memory = ConversationBufferMemory(memory_key="history")

# Create conversational chain
template = """
Chat History: {history}
Human: {input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

# Chain with memory
chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt,
    memory=memory
)

# Use with memory
print(chain.predict(input="Hi, I'm Alice"))
print(chain.predict(input="What's my name?"))
```

</details>

<details>
<summary><strong>🔍 Vector Search Example</strong></summary>

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Sample documents
texts = [
    "LangChain is an AI framework",
    "Python is a programming language", 
    "Vector databases store embeddings"
]

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)

# Search
docs = vectorstore.similarity_search("AI framework", k=1)
print(docs[0].page_content)
```

</details>

---

## 🎓 Learning Path

### **Recommended Order:**

1. **🌟 [Start with the Comprehensive Guide](comprehensive-guide.md)** ← Begin here
2. **🔗 [Move to Advanced Chains](../chains.md)** - Complex workflows
3. **🤖 [Learn About Agents](../agents.md)** - Autonomous AI systems  
4. **📚 [Explore Document Processing](../indexes.md)** - RAG and knowledge bases
5. **🔍 [Add Monitoring](../LangSmithSection.md)** - Debug and optimize
6. **🌐 [Deploy to Production](../LangServeSection.md)** - Share your creations

---

## 💡 What You'll Build

By the end of these examples, you'll be able to build:

- 🤖 **Smart Chatbots** with memory and context
- 📚 **Q&A Systems** that search through documents
- 🔍 **Semantic Search** engines for your content
- 🛠️ **AI Agents** that can use tools and make decisions
- 🌐 **Production APIs** ready for real users

---

## 🔧 Prerequisites

Before starting, make sure you have:

```bash
# Install LangChain
pip install langchain langchain-openai

# Set your API key
export OPENAI_API_KEY='your-api-key-here'
```

**Don't have an OpenAI key?** [Get one here](https://platform.openai.com/api-keys) (you'll get free credits to start)

---

## ❓ Need Help?

- **🐛 Something not working?** [Report an issue](https://github.com/0x-Professor/langforge-docs/issues)
- **💬 Have questions?** [Join the discussion](https://github.com/0x-Professor/langforge-docs/discussions)
- **📖 Want more examples?** [Browse advanced tutorials](../advanced-usage/)

---

<div align="center">

### 🚀 Ready to Start Your AI Journey?

**[📖 Begin with the Comprehensive Guide →](comprehensive-guide.md)**

*From zero to AI application developer in one tutorial!*

</div>
