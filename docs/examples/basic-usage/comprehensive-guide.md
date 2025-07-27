<div align="center">

# 🚀 Basic Usage Examples

**Master the Fundamentals of LangChain Development**

[![Beginner Friendly](https://img.shields.io/badge/level-beginner-green.svg)](.)
[![Code Examples](https://img.shields.io/badge/examples-working-blue.svg)](.)
[![Python](https://img.shields.io/badge/language-Python-yellow.svg)](.)
[![Updated](https://img.shields.io/badge/updated-2024-brightgreen.svg)](.)

</div>

---

## 🎯 What You'll Learn

This collection of **basic usage examples** will teach you the fundamental concepts of the LangChain ecosystem. Perfect for beginners who want to understand the core building blocks before diving into advanced applications.

### 🏗️ **Core Concepts Covered**

- **🤖 Chat Models** - Interact with various LLM providers
- **📊 Embeddings** - Convert text to numerical vectors  
- **⛓️ Chains** - Link multiple components together
- **🧠 Memory** - Add context and state to conversations
- **🛠️ Tools** - Extend LLMs with external capabilities
- **🔍 Vector Stores** - Store and search semantic information

---

## 📋 Prerequisites

Before running these examples, make sure you have:

```bash
# Install required packages
pip install langchain langchain-openai python-dotenv

# Set your API keys
export OPENAI_API_KEY='your-openai-key-here'
# Optional: for web search examples
export SERPAPI_API_KEY='your-serpapi-key-here'
```

---

## 🤖 Chat Models

### **Basic Chat Interaction**

The simplest way to interact with an LLM:

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Initialize the chat model
chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=100
)

# Create conversation messages
messages = [
    SystemMessage(content="You are a helpful AI assistant specialized in explaining complex topics simply."),
    HumanMessage(content="Explain quantum computing in one paragraph")
]

# Get response
response = chat(messages)
print(f"🤖 AI: {response.content}")
```

### **Streaming Responses**

For real-time applications, stream responses as they're generated:

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import sys

def stream_response():
    chat = ChatOpenAI(
        model="gpt-3.5-turbo",
        streaming=True,
        temperature=0.8
    )
    
    prompt = "Write a creative short story about a robot learning to paint"
    message = [HumanMessage(content=prompt)]
    
    print("🎨 AI Story: ", end="")
    for chunk in chat.stream(message):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")

# Run the streaming example
stream_response()
```

### **Multiple Provider Support**

LangChain works with many LLM providers:

```python
# OpenAI
from langchain.chat_models import ChatOpenAI
openai_chat = ChatOpenAI(model="gpt-4")

# Anthropic (requires: pip install langchain-anthropic)
from langchain_anthropic import ChatAnthropic
anthropic_chat = ChatAnthropic(model="claude-3-sonnet-20240229")

# Test with the same prompt
prompt = [HumanMessage(content="What makes a good AI assistant?")]

print("🤖 OpenAI:", openai_chat(prompt).content[:100] + "...")
print("🧠 Anthropic:", anthropic_chat(prompt).content[:100] + "...")
```

---

## 📊 Embeddings

### **Basic Embedding Generation**

Convert text into numerical vectors for semantic search:

```python
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

def generate_embeddings():
    # Initialize embeddings model
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Sample texts
    texts = [
        "Python is a programming language",
        "Machine learning uses algorithms",
        "I love eating pizza",
        "Neural networks are computational models"
    ]
    
    # Generate embeddings
    vectors = embeddings.embed_documents(texts)
    
    print(f"📊 Generated {len(vectors)} embeddings")
    print(f"📏 Vector dimension: {len(vectors[0])}")
    
    # Calculate similarity between first two texts
    similarity = np.dot(vectors[0], vectors[1])
    print(f"🔍 Similarity between text 1 & 2: {similarity:.3f}")
    
    return vectors

# Run embedding example
generate_embeddings()
```

### **Semantic Similarity Search**

Find the most similar text from a collection:

```python
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def semantic_search():
    embeddings = OpenAIEmbeddings()
    
    # Knowledge base
    documents = [
        "LangChain is a framework for developing applications with LLMs",
        "Python is a versatile programming language",
        "Machine learning algorithms can recognize patterns in data",
        "Natural language processing helps computers understand human language",
        "Vector databases store high-dimensional embeddings efficiently"
    ]
    
    # Query
    query = "What is a framework for building AI applications?"
    
    # Generate embeddings
    doc_embeddings = embeddings.embed_documents(documents)
    query_embedding = embeddings.embed_query(query)
    
    # Calculate similarities
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # Find best match
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    print(f"🔍 Query: {query}")
    print(f"✅ Best match ({best_score:.3f}): {documents[best_match_idx]}")

# Run semantic search example
semantic_search()
```

---

## ⛓️ Chains

### **Simple LLM Chain**

Chain together a prompt template and LLM:

```python
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

def create_simple_chain():
    # Create a prompt template
    prompt = PromptTemplate(
        input_variables=["topic", "audience"],
        template="""Explain {topic} to {audience} in a way they can understand.
        
        Topic: {topic}
        Audience: {audience}
        
        Explanation:"""
    )
    
    # Initialize LLM
    llm = OpenAI(temperature=0.7, max_tokens=200)
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Test with different inputs
    examples = [
        {"topic": "blockchain", "audience": "5-year-olds"},
        {"topic": "quantum computing", "audience": "high school students"},
        {"topic": "machine learning", "audience": "business executives"}
    ]
    
    for example in examples:
        result = chain.run(**example)
        print(f"🎯 {example['topic']} for {example['audience']}:")
        print(f"📝 {result.strip()}\n")

# Run chain example
create_simple_chain()
```

### **Sequential Chain**

Chain multiple operations together:

```python
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

def sequential_chain_example():
    llm = OpenAI(temperature=0.7)
    
    # First chain: Generate a business idea
    idea_template = "Generate a creative business idea for: {industry}"
    idea_prompt = PromptTemplate(input_variables=["industry"], template=idea_template)
    idea_chain = LLMChain(llm=llm, prompt=idea_prompt)
    
    # Second chain: Create a marketing slogan
    slogan_template = "Create a catchy marketing slogan for this business: {business_idea}"
    slogan_prompt = PromptTemplate(input_variables=["business_idea"], template=slogan_template)
    slogan_chain = LLMChain(llm=llm, prompt=slogan_prompt)
    
    # Combine chains
    overall_chain = SimpleSequentialChain(
        chains=[idea_chain, slogan_chain],
        verbose=True
    )
    
    # Run the chain
    result = overall_chain.run("sustainable technology")
    print(f"🚀 Final Result: {result}")

# Run sequential chain example
sequential_chain_example()
```

---

## 🧠 Memory

### **Conversation Buffer Memory**

Maintain conversation context:

```python
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, OpenAI, PromptTemplate

def conversation_with_memory():
    # Create memory
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    # Create prompt that uses memory
    template = """You are a helpful AI assistant with a good memory.
    
    Chat History:
    {chat_history}
    
    Human: {human_input}
    Assistant:"""
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template
    )
    
    # Create chain with memory
    llm_chain = LLMChain(
        llm=OpenAI(temperature=0.7),
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    # Simulate conversation
    conversation = [
        "Hi, my name is Alice and I'm a software engineer",
        "What's my profession?",
        "I'm working on a machine learning project about image recognition",
        "What project am I working on?",
        "Can you summarize what you know about me?"
    ]
    
    print("🗣️ Starting conversation with memory:")
    for user_input in conversation:
        print(f"\n👤 Human: {user_input}")
        response = llm_chain.predict(human_input=user_input)
        print(f"🤖 Assistant: {response.strip()}")

# Run memory example
conversation_with_memory()
```

### **Summary Memory**

For longer conversations, use summary memory:

```python
from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI

def summary_memory_example():
    llm = OpenAI(temperature=0)
    memory = ConversationSummaryMemory(llm=llm)
    
    # Add some conversation history
    conversation_data = [
        ("Human", "I'm planning a trip to Japan in the spring"),
        ("AI", "That's wonderful! Spring is cherry blossom season. When are you planning to go?"),
        ("Human", "Probably in April. I want to see the sakura blooms"),
        ("AI", "April is perfect timing! The cherry blossoms typically peak in early April in Tokyo and Kyoto."),
        ("Human", "What should I pack for the weather?"),
        ("AI", "April weather in Japan is mild but can be unpredictable. Pack layers, a light jacket, and comfortable walking shoes.")
    ]
    
    # Add to memory
    for speaker, message in conversation_data:
        if speaker == "Human":
            memory.save_context({"input": message}, {"output": ""})
        else:
            memory.save_context({"input": ""}, {"output": message})
    
    # Get summary
    summary = memory.buffer
    print(f"📝 Conversation Summary: {summary}")

# Run summary memory example
summary_memory_example()
```

---

## 🛠️ Tools

### **Using Built-in Tools**

Extend LLM capabilities with tools:

```python
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

def basic_tools_example():
    # Initialize LLM
    llm = OpenAI(temperature=0)
    
    # Load tools
    tools = load_tools(["llm-math"], llm=llm)
    
    # Create agent
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )
    
    # Test math calculation
    result = agent.run("What is 25 * 47 + 123?")
    print(f"🧮 Calculation result: {result}")

# Run tools example (Note: requires SERPAPI_API_KEY for web search)
basic_tools_example()
```

### **Custom Tools**

Create your own tools:

```python
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
import random

class RandomNumberTool(BaseTool):
    name = "random_number"
    description = "Generate a random number between two given numbers"
    
    def _run(self, min_val: int, max_val: int) -> str:
        result = random.randint(int(min_val), int(max_val))
        return f"Random number between {min_val} and {max_val}: {result}"
    
    def _arun(self, min_val: int, max_val: int):
        raise NotImplementedError("This tool does not support async")

def custom_tool_example():
    # Initialize LLM and tools
    llm = OpenAI(temperature=0)
    tools = [RandomNumberTool()]
    
    # Create agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Use the custom tool
    result = agent.run("Generate a random number between 1 and 100")
    print(f"🎲 Random result: {result}")

# Run custom tool example
custom_tool_example()
```

---

## 🔍 Vector Stores

### **Basic Vector Store Usage**

Store and search documents semantically:

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def vector_store_example():
    # Sample documents
    texts = [
        "LangChain is a framework for developing applications with LLMs",
        "Vector databases store high-dimensional embeddings",
        "Python is a popular programming language for AI",
        "Machine learning models can process natural language",
        "Embeddings capture semantic meaning of text"
    ]
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    # Search for similar documents
    query = "What is a framework for AI applications?"
    docs = vectorstore.similarity_search(query, k=2)
    
    print(f"🔍 Query: {query}")
    print(f"📚 Top 2 similar documents:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content}")

# Run vector store example
vector_store_example()
```

---

## 🎯 Putting It All Together

### **Complete Example: Smart Q&A System**

Combine multiple concepts into a working application:

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter

def smart_qa_system():
    # Knowledge base
    documents = [
        "LangChain is a framework for developing applications powered by language models. It provides standard interfaces to different LLM providers.",
        "Vector stores allow you to store and retrieve documents based on semantic similarity using embeddings.",
        "Chains in LangChain allow you to combine multiple components like prompts, LLMs, and memory to create complex workflows.",
        "Memory in LangChain enables applications to maintain context across multiple interactions with users.",
        "Agents in LangChain can use tools to interact with external services and make decisions based on user input.",
        "LangSmith is a platform for monitoring, evaluating, and debugging LLM applications built with LangChain."
    ]
    
    # Split documents if they're long
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text('\n'.join(documents))
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # Test questions
    questions = [
        "What is LangChain?",
        "How do vector stores work?",
        "What can agents do?",
        "What is LangSmith used for?"
    ]
    
    print("🧠 Smart Q&A System Demo:")
    for question in questions:
        answer = qa.run(question)
        print(f"\n❓ Q: {question}")
        print(f"✅ A: {answer.strip()}")

# Run complete example
smart_qa_system()
```

---

## 🚀 Next Steps

### **Ready for More Advanced Topics?**

Now that you understand the basics, explore these advanced concepts:

1. **[🔗 Advanced Chains](../chains.md)** - Complex chain compositions and routing
2. **[🤖 Intelligent Agents](../agents.md)** - Building autonomous AI agents
3. **[📚 Document Processing](../indexes.md)** - RAG systems and document analysis
4. **[🔍 LangSmith Monitoring](../LangSmithSection.md)** - Debug and monitor your applications
5. **[🌐 Production Deployment](../LangServeSection.md)** - Deploy your apps as APIs

### **Practice Exercises**

Try building these on your own:

- **🎯 Personal Assistant**: Combine memory and tools for a helpful AI assistant
- **📖 Knowledge Base**: Create a Q&A system for your own documents
- **🤖 Smart Chatbot**: Build a contextual conversation system
- **🔍 Semantic Search**: Create a search engine for your content

### **Get Help**

- **💬 Questions?** [Join Discussions](https://github.com/0x-Professor/langforge-docs/discussions)
- **🐛 Issues?** [Report Problems](https://github.com/0x-Professor/langforge-docs/issues)
- **📖 Need More Examples?** [Browse Advanced Usage](../advanced-usage/)

---

<div align="center">

**🎉 Congratulations! You've mastered the LangChain basics!**

*Ready to build something amazing? The AI application ecosystem awaits you!*

</div>