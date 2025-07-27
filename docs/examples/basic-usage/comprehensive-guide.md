# 🚀 Basic Usage Examples

**Master the Fundamentals of LangChain Development**

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
pip install langchain langchain-openai langchain-community python-dotenv

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
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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
response = chat.invoke(messages)
print(f"🤖 AI: {response.content}")
```

### **Streaming Responses**

For real-time applications, stream responses as they're generated:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def stream_response():
    chat = ChatOpenAI(
        model="gpt-3.5-turbo",
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
from langchain_openai import ChatOpenAI
openai_chat = ChatOpenAI(model="gpt-4")

# Anthropic (requires: pip install langchain-anthropic)
from langchain_anthropic import ChatAnthropic
anthropic_chat = ChatAnthropic(model="claude-3-sonnet-20240229")

# Test with the same prompt
prompt = [HumanMessage(content="What makes a good AI assistant?")]

print("🤖 OpenAI:", openai_chat.invoke(prompt).content[:100] + "...")
print("🧠 Anthropic:", anthropic_chat.invoke(prompt).content[:100] + "...")
```

---

## 📊 Embeddings

### **Basic Embedding Generation**

Convert text into numerical vectors for semantic search:

```python
from langchain_openai import OpenAIEmbeddings
import numpy as np

def generate_embeddings():
    # Initialize embeddings model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
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
from langchain_openai import OpenAIEmbeddings
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

### **Simple Chain with LCEL**

Modern chain creation using LangChain Expression Language:

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def create_simple_chain():
    # Create a prompt template
    prompt = PromptTemplate.from_template(
        """Explain {topic} to {audience} in a way they can understand.
        
        Topic: {topic}
        Audience: {audience}
        
        Explanation:"""
    )
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create chain using LCEL
    chain = prompt | llm | StrOutputParser()
    
    # Test with different inputs
    examples = [
        {"topic": "blockchain", "audience": "5-year-olds"},
        {"topic": "quantum computing", "audience": "high school students"},
        {"topic": "machine learning", "audience": "business executives"}
    ]
    
    for example in examples:
        result = chain.invoke(example)
        print(f"🎯 {example['topic']} for {example['audience']}:")
        print(f"📝 {result.strip()}\n")

# Run chain example
create_simple_chain()
```

### **Sequential Chain with LCEL**

Chain multiple operations together:

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def sequential_chain_example():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    output_parser = StrOutputParser()
    
    # First chain: Generate a business idea
    idea_prompt = PromptTemplate.from_template("Generate a creative business idea for: {industry}")
    idea_chain = idea_prompt | llm | output_parser
    
    # Second chain: Create a marketing slogan
    slogan_prompt = PromptTemplate.from_template("Create a catchy marketing slogan for this business: {business_idea}")
    
    # Combine chains using LCEL
    full_chain = (
        {"business_idea": idea_chain}
        | slogan_prompt
        | llm
        | output_parser
    )
    
    # Run the chain
    result = full_chain.invoke({"industry": "sustainable technology"})
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

def conversation_with_memory():
    # Create memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create prompt that uses memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant with a good memory."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{human_input}")
    ])
    
    # Create chain with memory
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    def load_memory(_):
        return memory.load_memory_variables({})["chat_history"]
    
    chain = (
        RunnablePassthrough.assign(chat_history=load_memory)
        | prompt
        | llm
    )
    
    # Function to run conversation with memory
    def chat_with_memory(user_input: str):
        response = chain.invoke({"human_input": user_input})
        
        # Save to memory
        memory.save_context(
            {"human_input": user_input},
            {"output": response.content}
        )
        
        return response.content
    
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
        response = chat_with_memory(user_input)
        print(f"🤖 Assistant: {response}")

# Run memory example
conversation_with_memory()
```

### **Summary Memory**

For longer conversations, use summary memory:

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

def summary_memory_example():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationSummaryMemory(llm=llm)
    
    # Add some conversation history
    conversation_data = [
        ("I'm planning a trip to Japan in the spring", "That's wonderful! Spring is cherry blossom season. When are you planning to go?"),
        ("Probably in April. I want to see the sakura blooms", "April is perfect timing! The cherry blossoms typically peak in early April in Tokyo and Kyoto."),
        ("What should I pack for the weather?", "April weather in Japan is mild but can be unpredictable. Pack layers, a light jacket, and comfortable walking shoes.")
    ]
    
    # Add to memory
    for human_msg, ai_msg in conversation_data:
        memory.save_context({"input": human_msg}, {"output": ai_msg})
    
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
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

@tool
def calculator(expression: str) -> float:
    """Perform mathematical calculations safely."""
    try:
        # Safe evaluation of mathematical expressions
        import ast
        import operator
        
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        
        def eval_expr(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](eval_expr(node.operand))
            else:
                raise TypeError(node)
        
        return eval_expr(ast.parse(expression, mode='eval').body)
    except:
        return "Error: Invalid mathematical expression"

def basic_tools_example():
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create tools
    tools = [calculator]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to tools."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test math calculation
    result = agent_executor.invoke({"input": "What is 25 * 47 + 123?"})
    print(f"🧮 Calculation result: {result['output']}")

# Run tools example
basic_tools_example()
```

### **Custom Tools**

Create your own tools:

```python
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import random

@tool
def random_number(min_val: int, max_val: int) -> str:
    """Generate a random number between two given numbers."""
    result = random.randint(min_val, max_val)
    return f"Random number between {min_val} and {max_val}: {result}"

def custom_tool_example():
    # Initialize LLM and tools
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [random_number]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to tools."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Use the custom tool
    result = agent_executor.invoke({"input": "Generate a random number between 1 and 100"})
    print(f"🎲 Random result: {result['output']}")

# Run custom tool example
custom_tool_example()
```

---

## 🔍 Vector Stores

### **Basic Vector Store Usage**

Store and search documents semantically:

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

def vector_store_example():
    # Sample documents
    texts = [
        "LangChain is a framework for developing applications with LLMs",
        "Vector databases store high-dimensional embeddings",
        "Python is a popular programming language for AI",
        "Machine learning models can process natural language",
        "Embeddings capture semantic meaning of text"
    ]
    
    # Create documents
    documents = [Document(page_content=text) for text in texts]
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    
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
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

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
    
    # Create documents
    docs = [Document(page_content=doc) for doc in documents]
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Create QA prompt
    template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create QA chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        | StrOutputParser()
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
        answer = rag_chain.invoke(question)
        print(f"\n❓ Q: {question}")
        print(f"✅ A: {answer}")

# Run complete example
smart_qa_system()
```

---

## 🚀 Next Steps

### **Ready for More Advanced Topics?**

Now that you understand the basics, explore these advanced concepts:

1. **🔗 Advanced Chains** - Complex chain compositions and routing
2. **🤖 Intelligent Agents** - Building autonomous AI agents
3. **📚 Document Processing** - RAG systems and document analysis
4. **🔍 LangSmith Monitoring** - Debug and monitor your applications
5. **🌐 Production Deployment** - Deploy your apps as APIs

### **Practice Exercises**

Try building these on your own:

- **🎯 Personal Assistant**: Combine memory and tools for a helpful AI assistant
- **📖 Knowledge Base**: Create a Q&A system for your own documents
- **🤖 Smart Chatbot**: Build a contextual conversation system
- **🔍 Semantic Search**: Create a search engine for your content

---

**🎉 Congratulations! You've mastered the LangChain basics!**

*Ready to build something amazing? The AI application ecosystem awaits you!*