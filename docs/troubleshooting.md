<div align="center">

# 🔧 Troubleshooting Guide

**Solve Common LangChain Issues Fast**

[![Troubleshooting](https://img.shields.io/badge/troubleshooting-comprehensive-red.svg)](.)
[![Self Service](https://img.shields.io/badge/self%20service-enabled-green.svg)](.)
[![Problem Solver](https://img.shields.io/badge/problems-solved%20quickly-blue.svg)](.)

</div>

---

## 🎯 Quick Problem Solver

**Select your issue for instant solutions:**

| 🚨 Problem Category | 🔍 Common Issues | ⚡ Quick Fix |
|---------------------|------------------|--------------|
| [🔑 **API Keys**](#-api-key-issues) | Not found, invalid, expired | [Fix in 30 seconds](#api-key-not-found) |
| [⚙️ **Installation**](#%EF%B8%8F-installation-issues) | Import errors, dependencies | [Reinstall guide](#import-errors) |
| [🐌 **Performance**](#-performance-issues) | Slow responses, timeouts | [Speed up now](#slow-responses) |
| [💸 **Costs**](#-cost-management) | High bills, usage tracking | [Reduce costs](#high-api-costs) |
| [🔄 **Memory**](#-memory-issues) | Context not saved, errors | [Fix memory](#memory-not-working) |
| [🤖 **Agents**](#-agent-issues) | Not working, tool errors | [Debug agents](#agent-not-using-tools) |
| [📚 **RAG/Vectors**](#-ragvector-issues) | Search not working, errors | [Fix search](#vector-search-not-working) |
| [🚀 **Deployment**](#-deployment-issues) | Production errors, scaling | [Deploy safely](#deployment-failures) |

---

## 🔑 API Key Issues

### **API Key Not Found**

**Error:** `AuthenticationError: No API key provided`

**Symptoms:**
- ❌ `openai.error.AuthenticationError`
- ❌ "API key not found" messages
- ❌ 401 Unauthorized errors

**Solutions:**

<details>
<summary><strong>✅ Solution 1: Environment Variable (Recommended)</strong></summary>

```bash
# For current session
export OPENAI_API_KEY='sk-your-actual-key-here'

# For permanent setup (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="sk-your-actual-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Verify it worked:**
```bash
echo $OPENAI_API_KEY
# Should show: sk-your-actual-key-here
```

</details>

<details>
<summary><strong>✅ Solution 2: Python Code</strong></summary>

```python
import os

# Method 1: Set in code (less secure)
os.environ["OPENAI_API_KEY"] = "sk-your-actual-key-here"

# Method 2: Load from .env file (recommended)
from dotenv import load_dotenv
load_dotenv()  # Loads from .env file in current directory

# Verify
print(f"Key loaded: {os.environ.get('OPENAI_API_KEY', 'NOT FOUND')[:10]}...")
```

Create `.env` file:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

</details>

<details>
<summary><strong>✅ Solution 3: Direct Parameter</strong></summary>

```python
from langchain.llms import OpenAI

# Pass key directly (not recommended for production)
llm = OpenAI(openai_api_key="sk-your-actual-key-here")
```

</details>

**Prevention Checklist:**
- [ ] Remove quotes and spaces from key
- [ ] Restart terminal after setting environment variable
- [ ] Check key hasn't expired on OpenAI dashboard
- [ ] Verify key has correct permissions

---

### **Invalid API Key**

**Error:** `AuthenticationError: Invalid API key provided`

**Quick Fixes:**
1. **Generate new key** at [OpenAI API Keys](https://platform.openai.com/api-keys)
2. **Check key format** - Should start with `sk-`
3. **Test key directly:**

```python
import openai
openai.api_key = "your-key-here"

try:
    openai.Model.list()
    print("✅ Key is valid!")
except Exception as e:
    print(f"❌ Key invalid: {e}")
```

---

### **Rate Limit Exceeded**

**Error:** `RateLimitError: Rate limit reached`

**Immediate Solutions:**

<details>
<summary><strong>✅ Add Retry Logic</strong></summary>

```python
from tenacity import retry, wait_exponential, stop_after_attempt
import time

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3)
)
def safe_llm_call(llm, prompt):
    return llm(prompt)

# Usage
try:
    response = safe_llm_call(llm, "Your prompt")
except Exception as e:
    print(f"Failed after retries: {e}")
```

</details>

<details>
<summary><strong>✅ Implement Rate Limiting</strong></summary>

```python
import time
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, calls_per_minute=60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        now = datetime.now()
        # Remove calls older than 1 minute
        self.calls = [call for call in self.calls if now - call < timedelta(minutes=1)]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0]).seconds
            time.sleep(sleep_time)
        
        self.calls.append(now)

# Usage
rate_limiter = RateLimiter(calls_per_minute=20)

def rate_limited_call(prompt):
    rate_limiter.wait_if_needed()
    return llm(prompt)
```

</details>

**Long-term Solutions:**
- Upgrade to paid plan for higher limits
- Use batch processing instead of individual calls
- Implement caching to reduce API calls
- Consider using multiple API keys with load balancing

---

## ⚙️ Installation Issues

### **Import Errors**

**Error:** `ModuleNotFoundError: No module named 'langchain'`

**Quick Fix:**
```bash
# Uninstall and reinstall
pip uninstall langchain -y
pip install langchain langchain-openai

# If you have multiple Python versions
python -m pip install langchain langchain-openai

# For Conda users
conda install -c conda-forge langchain
```

**Advanced Debugging:**
```python
import sys
print("Python version:", sys.version)
print("Python path:", sys.executable)

# Check if langchain is installed
try:
    import langchain
    print(f"✅ LangChain version: {langchain.__version__}")
except ImportError:
    print("❌ LangChain not installed")
```

---

### **Dependency Conflicts**

**Error:** Various dependency-related errors

**Solutions:**

<details>
<summary><strong>✅ Create Clean Environment</strong></summary>

```bash
# Using venv
python -m venv langchain_env
source langchain_env/bin/activate  # On Windows: langchain_env\Scripts\activate
pip install langchain langchain-openai

# Using conda
conda create -n langchain python=3.9
conda activate langchain
pip install langchain langchain-openai
```

</details>

<details>
<summary><strong>✅ Fix Specific Dependencies</strong></summary>

```bash
# Common fixes
pip install --upgrade pip setuptools wheel
pip install --upgrade openai
pip install pydantic==1.10.8  # If Pydantic v2 issues

# For vector store issues
pip install faiss-cpu  # or faiss-gpu for GPU
pip install chromadb

# For specific providers
pip install langchain-anthropic
pip install langchain-google-genai
```

</details>

---

## 🐌 Performance Issues

### **Slow Responses**

**Problem:** LangChain taking too long to respond

**Immediate Fixes:**

<details>
<summary><strong>✅ Enable Caching</strong></summary>

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Enable caching
set_llm_cache(InMemoryCache())

# Now identical prompts return cached results instantly
llm = OpenAI()
response1 = llm("What is Python?")  # API call
response2 = llm("What is Python?")  # From cache - instant!
```

</details>

<details>
<summary><strong>✅ Use Async Operations</strong></summary>

```python
import asyncio
from langchain.llms import OpenAI

async def async_calls():
    llm = OpenAI()
    
    # Run multiple calls concurrently
    tasks = [
        llm.agenerate(["Prompt 1"]),
        llm.agenerate(["Prompt 2"]),
        llm.agenerate(["Prompt 3"])
    ]
    
    results = await asyncio.gather(*tasks)
    return results

# Much faster than sequential calls
results = asyncio.run(async_calls())
```

</details>

<details>
<summary><strong>✅ Optimize Prompts</strong></summary>

```python
# ❌ Slow: Long prompt
slow_prompt = """
Please analyze this 5000-word document and provide detailed insights about every aspect including themes, sentiment, key entities, relationships, implications, and recommendations with citations and examples...
"""

# ✅ Fast: Concise prompt
fast_prompt = "Summarize the key points from this document in 3 bullet points."

# ✅ Fast: Use cheaper model
fast_llm = OpenAI(model_name="gpt-3.5-turbo")  # Instead of gpt-4
expensive_llm = OpenAI(model_name="gpt-4")
```

</details>

**Advanced Optimizations:**
- Use streaming for long responses
- Implement connection pooling
- Consider using local models for simple tasks
- Batch similar requests together

---

### **Memory Leaks**

**Problem:** Application memory usage keeps growing

**Solutions:**

<details>
<summary><strong>✅ Clear Memory Periodically</strong></summary>

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

# Clear memory after certain number of exchanges
exchange_count = 0
MAX_EXCHANGES = 10

def chat_with_memory(user_input):
    global exchange_count
    
    response = chain.predict(input=user_input)
    exchange_count += 1
    
    # Clear memory periodically
    if exchange_count >= MAX_EXCHANGES:
        memory.clear()
        exchange_count = 0
    
    return response
```

</details>

<details>
<summary><strong>✅ Use Summary Memory</strong></summary>

```python
from langchain.memory import ConversationSummaryMemory

# Automatically summarizes old conversations
memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=1000  # Summarize when over 1000 tokens
)
```

</details>

---

## 💸 Cost Management

### **High API Costs**

**Problem:** LangChain usage generating unexpectedly high bills

**Immediate Cost Reduction:**

<details>
<summary><strong>✅ Switch to Cheaper Models</strong></summary>

```python
# ❌ Expensive
expensive_llm = OpenAI(model_name="gpt-4")  # ~$0.06/1K tokens

# ✅ Much cheaper
cheap_llm = OpenAI(model_name="gpt-3.5-turbo")  # ~$0.002/1K tokens

# ✅ For simple tasks, use even cheaper options
simple_llm = OpenAI(model_name="babbage-002")  # ~$0.0005/1K tokens
```

</details>

<details>
<summary><strong>✅ Monitor Token Usage</strong></summary>

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    response = llm("Your prompt")
    print(f"Tokens used: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost:.4f}")
```

</details>

<details>
<summary><strong>✅ Implement Token Limits</strong></summary>

```python
# Set strict limits
llm = OpenAI(
    max_tokens=100,        # Limit response length
    temperature=0.3,       # Reduce randomness (can help with consistency)
    request_timeout=30     # Fail fast if taking too long
)

# For chains, add limits
from langchain.chains import LLMChain

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False  # Disable verbose to save tokens
)
```

</details>

**Cost Monitoring Setup:**
```python
import os
from datetime import datetime

class CostTracker:
    def __init__(self, daily_limit=10.0):
        self.daily_limit = daily_limit
        self.daily_cost = 0.0
        self.date = datetime.now().date()
    
    def track_cost(self, cost):
        current_date = datetime.now().date()
        
        # Reset daily cost at midnight
        if current_date != self.date:
            self.daily_cost = 0.0
            self.date = current_date
        
        self.daily_cost += cost
        
        if self.daily_cost > self.daily_limit:
            raise Exception(f"Daily cost limit of ${self.daily_limit} exceeded!")
        
        print(f"Today's cost: ${self.daily_cost:.4f} / ${self.daily_limit}")

# Usage
tracker = CostTracker(daily_limit=5.0)

with get_openai_callback() as cb:
    response = llm("Your prompt")
    tracker.track_cost(cb.total_cost)
```

---

## 🔄 Memory Issues

### **Memory Not Working**

**Problem:** Conversation memory not being saved or retrieved

**Common Fixes:**

<details>
<summary><strong>✅ Check Memory Key Names</strong></summary>

```python
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate

# ❌ Wrong: Memory key doesn't match template
memory = ConversationBufferMemory(memory_key="chat_history")
template = "History: {history}\nUser: {input}\nAI:"  # Wrong key name!

# ✅ Correct: Memory key matches template
memory = ConversationBufferMemory(memory_key="history")
template = "History: {history}\nUser: {input}\nAI:"  # Correct!

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)
```

</details>

<details>
<summary><strong>✅ Debug Memory State</strong></summary>

```python
# Check what's in memory
print("Memory variables:", memory.load_memory_variables({}))
print("Chat messages:", memory.chat_memory.messages)

# Test memory manually
memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
print("After saving:", memory.load_memory_variables({}))
```

</details>

<details>
<summary><strong>✅ Fix Chain Configuration</strong></summary>

```python
# ❌ Wrong: Chain not configured with memory
chain = LLMChain(llm=llm, prompt=prompt)  # No memory!

# ✅ Correct: Chain includes memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,  # Include memory
    verbose=True    # See what's happening
)
```

</details>

---

### **Memory Growing Too Large**

**Problem:** Memory consuming too much space or tokens

**Solutions:**

<details>
<summary><strong>✅ Use Summary Memory</strong></summary>

```python
from langchain.memory import ConversationSummaryMemory

# Automatically summarizes old parts
summary_memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=1000,  # Summarize when over 1000 tokens
    return_messages=True
)
```

</details>

<details>
<summary><strong>✅ Use Window Memory</strong></summary>

```python
from langchain.memory import ConversationBufferWindowMemory

# Only keep last N exchanges
window_memory = ConversationBufferWindowMemory(
    k=5,  # Keep only last 5 exchanges
    return_messages=True
)
```

</details>

---

## 🤖 Agent Issues

### **Agent Not Using Tools**

**Problem:** Agent ignoring available tools

**Debugging Steps:**

<details>
<summary><strong>✅ Check Tool Descriptions</strong></summary>

```python
from langchain.tools import Tool

# ❌ Bad: Vague description
bad_tool = Tool(
    name="calculator",
    func=lambda x: eval(x),
    description="does math"  # Too vague!
)

# ✅ Good: Clear, specific description
good_tool = Tool(
    name="calculator",
    func=lambda x: eval(x),
    description="Use this tool to perform mathematical calculations. Input should be a valid mathematical expression like '2+2' or '10*5'. Returns the numerical result."
)
```

</details>

<details>
<summary><strong>✅ Test Tools Individually</strong></summary>

```python
# Test each tool manually
for tool in tools:
    print(f"Testing {tool.name}:")
    try:
        result = tool.run("2+2")  # Use appropriate test input
        print(f"✅ Works: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
```

</details>

<details>
<summary><strong>✅ Use Verbose Mode</strong></summary>

```python
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # Shows reasoning process
    max_iterations=3,  # Prevent infinite loops
    early_stopping_method="generate"
)

# Run with verbose to see what agent is thinking
result = agent.run("What is 25 * 4?")
```

</details>

---

### **Agent Stuck in Loops**

**Problem:** Agent keeps repeating the same actions

**Solutions:**

<details>
<summary><strong>✅ Set Max Iterations</strong></summary>

```python
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=5,  # Stop after 5 attempts
    early_stopping_method="generate"  # Generate final answer
)
```

</details>

<details>
<summary><strong>✅ Improve Tool Error Handling</strong></summary>

```python
def safe_calculator(expression):
    try:
        # Validate expression before evaluation
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

calculator_tool = Tool(
    name="calculator",
    func=safe_calculator,
    description="Calculate mathematical expressions. Returns 'Error: ...' if invalid."
)
```

</details>

---

## 📚 RAG/Vector Issues

### **Vector Search Not Working**

**Problem:** Semantic search returning irrelevant results

**Common Fixes:**

<details>
<summary><strong>✅ Check Document Quality</strong></summary>

```python
# Debug what's in your vector store
print("Number of documents:", len(texts))
print("Sample document:", texts[0][:200] + "...")

# Check embedding quality
embeddings = OpenAIEmbeddings()
test_query = "your search query"
test_doc = "your document content"

query_embedding = embeddings.embed_query(test_query)
doc_embedding = embeddings.embed_query(test_doc)

# Calculate similarity manually
import numpy as np
similarity = np.dot(query_embedding, doc_embedding)
print(f"Similarity score: {similarity}")
```

</details>

<details>
<summary><strong>✅ Optimize Text Chunking</strong></summary>

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ❌ Poor chunking
bad_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)

# ✅ Better chunking
good_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Smaller chunks for better relevance
    chunk_overlap=200,  # Overlap to maintain context
    separators=["\n\n", "\n", ".", "!", "?", " "]  # Smart separation
)

texts = good_splitter.split_documents(documents)
```

</details>

<details>
<summary><strong>✅ Improve Search Parameters</strong></summary>

```python
# Try different search methods
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,  # Only return relevant results
        "k": 3  # Return top 3 results
    }
)

# Or use MMR for diverse results
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,  # Fetch 20, return diverse 5
        "lambda_mult": 0.7  # Balance relevance vs diversity
    }
)
```

</details>

---

### **Vector Store Errors**

**Problem:** FAISS, Chroma, or other vector store issues

**Quick Fixes:**

<details>
<summary><strong>✅ FAISS Installation Issues</strong></summary>

```bash
# Try different FAISS versions
pip uninstall faiss-cpu faiss-gpu -y

# For CPU-only
pip install faiss-cpu

# For GPU (if you have CUDA)
pip install faiss-gpu

# Alternative: Use Chroma instead
pip install chromadb
```

```python
# Switch to Chroma if FAISS issues persist
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Persistent storage
)
```

</details>

<details>
<summary><strong>✅ Embedding Dimension Mismatches</strong></summary>

```python
# Check embedding dimensions
embeddings = OpenAIEmbeddings()
test_embedding = embeddings.embed_query("test")
print(f"Embedding dimension: {len(test_embedding)}")

# Make sure all embeddings use same model
# Don't mix different embedding models in same vector store
```

</details>

---

## 🚀 Deployment Issues

### **Deployment Failures**

**Problem:** Issues when deploying to production

**Common Solutions:**

<details>
<summary><strong>✅ Environment Variables</strong></summary>

```python
# Use environment variables for all secrets
import os
from dotenv import load_dotenv

load_dotenv()

# Don't hardcode keys!
llm = OpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7"))
)
```

</details>

<details>
<summary><strong>✅ Graceful Error Handling</strong></summary>

```python
def safe_llm_call(prompt, retries=3):
    for attempt in range(retries):
        try:
            return llm(prompt)
        except Exception as e:
            if attempt == retries - 1:
                return f"Sorry, I'm having technical difficulties: {str(e)}"
            time.sleep(2 ** attempt)  # Exponential backoff
```

</details>

<details>
<summary><strong>✅ Resource Management</strong></summary>

```python
# Use connection pooling
from langchain.llms import OpenAI

# Don't create new LLM instances repeatedly
class LLMService:
    def __init__(self):
        self.llm = OpenAI()  # Create once, reuse
    
    def get_response(self, prompt):
        return self.llm(prompt)

# Singleton pattern
llm_service = LLMService()
```

</details>

---

### **Scaling Issues**

**Problem:** Application can't handle multiple users

**Solutions:**

<details>
<summary><strong>✅ Use Async FastAPI</strong></summary>

```python
from fastapi import FastAPI
from langserve import add_routes
import asyncio

app = FastAPI()

# Add async routes
add_routes(
    app,
    your_chain,
    path="/chat",
    # Enable async processing
)

# Use async in your chain logic
async def async_chain(inputs):
    # Process multiple requests concurrently
    tasks = [process_request(req) for req in inputs]
    return await asyncio.gather(*tasks)
```

</details>

<details>
<summary><strong>✅ Implement Caching</strong></summary>

```python
from functools import lru_cache
import hashlib

# Cache responses for identical prompts
@lru_cache(maxsize=1000)
def cached_llm_call(prompt_hash):
    return llm(prompt)

def get_response(prompt):
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    return cached_llm_call(prompt_hash)
```

</details>

---

## 🆘 Emergency Debugging

### **Complete System Check**

When nothing works, run this comprehensive diagnostic:

```python
import sys
import os
import traceback

def system_diagnostic():
    print("🔍 LangChain System Diagnostic")
    print("=" * 40)
    
    # Python environment
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Package versions
    try:
        import langchain
        print(f"✅ LangChain: {langchain.__version__}")
    except ImportError as e:
        print(f"❌ LangChain: {e}")
    
    try:
        import openai
        print(f"✅ OpenAI: {openai.__version__}")
    except ImportError as e:
        print(f"❌ OpenAI: {e}")
    
    # API Keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    print(f"OpenAI Key: {'✅ Set' if openai_key else '❌ Missing'}")
    if openai_key:
        print(f"Key format: {'✅ Valid' if openai_key.startswith('sk-') else '❌ Invalid'}")
    
    # Basic LLM test
    try:
        from langchain.llms import OpenAI
        llm = OpenAI()
        response = llm("Hello")
        print("✅ Basic LLM test: PASSED")
    except Exception as e:
        print(f"❌ Basic LLM test: {e}")
        traceback.print_exc()
    
    # Memory test
    try:
        from langchain.memory import ConversationBufferMemory
        memory = ConversationBufferMemory()
        memory.save_context({"input": "test"}, {"output": "test"})
        print("✅ Memory test: PASSED")
    except Exception as e:
        print(f"❌ Memory test: {e}")
    
    # Vector store test
    try:
        from langchain.vectorstores import FAISS
        from langchain.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(["test"], embeddings)
        print("✅ Vector store test: PASSED")
    except Exception as e:
        print(f"❌ Vector store test: {e}")
    
    print("=" * 40)
    print("Diagnostic complete!")

# Run the diagnostic
system_diagnostic()
```

---

## 📞 Getting Help

### **When to Ask for Help**

Ask the community when you've:
- [ ] Tried the solutions in this guide
- [ ] Searched existing GitHub issues
- [ ] Created a minimal reproducible example
- [ ] Included error messages and system info

### **How to Ask for Help**

**Good Bug Report Template:**
```markdown
## Problem
Brief description of what's not working

## Expected Behavior
What should happen

## Actual Behavior
What actually happens (include error messages)

## Code to Reproduce
```python
# Minimal example that reproduces the issue
from langchain.llms import OpenAI
llm = OpenAI()
# Error happens here
result = llm("test")
```

## Environment
- Python version: 3.9.7
- LangChain version: 0.1.0
- Operating System: macOS 13.0
- Additional packages: langsmith, faiss-cpu

## What I've Tried
- Solution 1: didn't work because...
- Solution 2: partially worked but...
```

### **Community Resources**

- 💬 **[GitHub Discussions](https://github.com/0x-Professor/langforge-docs/discussions)** - Ask questions, share solutions
- 🐛 **[GitHub Issues](https://github.com/0x-Professor/langforge-docs/issues)** - Report bugs and feature requests
- 📚 **[Official Discord](https://discord.gg/langchain)** - Real-time community help
- 📖 **[Documentation](../)** - Comprehensive guides and examples

---

<div align="center">

### 🎯 Still Stuck?

**Don't give up! The community is here to help.**

**[💬 Ask for Help →](https://github.com/0x-Professor/langforge-docs/discussions)** • **[📖 Browse Docs →](../)** • **[🔍 Search Issues →](https://github.com/0x-Professor/langforge-docs/issues)**

---

*This troubleshooting guide is updated based on real community issues. Found a solution not listed here? [Share it with the community!](https://github.com/0x-Professor/langforge-docs/discussions)*

</div>