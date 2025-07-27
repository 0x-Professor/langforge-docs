<div align="center">

# ⚡ Performance Best Practices

**Build Lightning-Fast LangChain Applications**

[![Performance](https://img.shields.io/badge/performance-optimized-green.svg)](.)
[![Production Ready](https://img.shields.io/badge/production-ready-blue.svg)](.)
[![Cost Effective](https://img.shields.io/badge/cost-effective-orange.svg)](.)

</div>

---

## 🎯 Performance Overview

Building fast, efficient LangChain applications requires attention to multiple layers: API calls, memory usage, caching strategies, and infrastructure choices. This guide covers proven techniques to maximize performance while minimizing costs.

### **Performance Metrics to Track**

| Metric | Target | Impact |
|--------|--------|---------|
| **Response Time** | < 2 seconds | User experience |
| **Token Usage** | Minimize | Cost optimization |
| **Memory Usage** | < 1GB for basic apps | Resource efficiency |
| **Throughput** | 100+ requests/min | Scalability |
| **Error Rate** | < 1% | Reliability |

---

## 🚀 Core Performance Strategies

### **1. Smart Caching**

**Enable LLM Response Caching:**

```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

# Option 1: In-memory cache (fast, temporary)
set_llm_cache(InMemoryCache())

# Option 2: Persistent cache (survives restarts)
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Option 3: Redis cache (shared across instances)
from langchain.cache import RedisCache
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)
set_llm_cache(RedisCache(redis_client))
```

**Custom Semantic Caching:**

```python
import hashlib
from functools import lru_cache
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.cache = {}
        self.embeddings = OpenAIEmbeddings()
        self.threshold = similarity_threshold
    
    def get_cache_key(self, prompt):
        # Create embedding for semantic similarity
        embedding = self.embeddings.embed_query(prompt)
        
        # Check if similar prompt exists
        for cached_prompt, cached_embedding in self.cache.items():
            similarity = np.dot(embedding, cached_embedding)
            if similarity > self.threshold:
                return cached_prompt
        
        return None
    
    def cache_response(self, prompt, response):
        embedding = self.embeddings.embed_query(prompt)
        self.cache[prompt] = {
            'response': response,
            'embedding': embedding
        }
    
    def get_response(self, prompt):
        cache_key = self.get_cache_key(prompt)
        if cache_key:
            return self.cache[cache_key]['response']
        return None

# Usage
semantic_cache = SemanticCache()

def cached_llm_call(prompt):
    # Check cache first
    cached_response = semantic_cache.get_response(prompt)
    if cached_response:
        return cached_response
    
    # Make API call
    response = llm(prompt)
    
    # Cache the response
    semantic_cache.cache_response(prompt, response)
    return response
```

---

### **2. Optimize Model Selection**

**Choose the Right Model for the Task:**

```python
from langchain.llms import OpenAI

# Task-specific model selection
class ModelRouter:
    def __init__(self):
        # Fast, cheap model for simple tasks
        self.simple_model = OpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=150
        )
        
        # Powerful model for complex tasks
        self.complex_model = OpenAI(
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=500
        )
    
    def route_request(self, prompt, complexity="simple"):
        if complexity == "simple":
            return self.simple_model(prompt)
        else:
            return self.complex_model(prompt)

# Usage
router = ModelRouter()

# Simple classification task
result = router.route_request("Is this positive or negative: 'I love this!'", "simple")

# Complex reasoning task  
result = router.route_request("Analyze the economic implications...", "complex")
```

**Model Performance Comparison:**

| Model | Speed | Cost | Quality | Best For |
|-------|--------|------|---------|----------|
| GPT-3.5-turbo | ⚡⚡⚡ | 💰 | ⭐⭐⭐ | Chat, simple tasks |
| GPT-4 | ⚡⚡ | 💰💰💰 | ⭐⭐⭐⭐⭐ | Complex reasoning |
| Claude-3-haiku | ⚡⚡⚡ | 💰 | ⭐⭐⭐ | Fast responses |
| Local models | ⚡ | Free | ⭐⭐ | Privacy, offline |

---

### **3. Async and Batch Processing**

**Concurrent Request Processing:**

```python
import asyncio
from langchain.llms import OpenAI

class AsyncLLMProcessor:
    def __init__(self, max_concurrent=5):
        self.llm = OpenAI()
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single(self, prompt):
        async with self.semaphore:
            # Use agenerate for async calls
            result = await self.llm.agenerate([prompt])
            return result.generations[0][0].text
    
    async def process_batch(self, prompts):
        tasks = [self.process_single(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

# Usage
async def main():
    processor = AsyncLLMProcessor(max_concurrent=3)
    
    prompts = [
        "Summarize this: ...",
        "Translate this: ...", 
        "Analyze this: ..."
    ]
    
    results = await processor.process_batch(prompts)
    for i, result in enumerate(results):
        print(f"Result {i}: {result}")

# Run async processing
asyncio.run(main())
```

**Smart Batching Strategy:**

```python
from collections import defaultdict
import time

class SmartBatcher:
    def __init__(self, batch_size=10, max_wait_time=1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.batches = defaultdict(list)
        self.last_batch_time = defaultdict(float)
    
    def add_request(self, request_type, prompt, callback):
        current_time = time.time()
        
        # Add to batch
        self.batches[request_type].append({
            'prompt': prompt,
            'callback': callback,
            'timestamp': current_time
        })
        
        # Check if we should process the batch
        batch = self.batches[request_type]
        time_since_last = current_time - self.last_batch_time[request_type]
        
        if len(batch) >= self.batch_size or time_since_last >= self.max_wait_time:
            self.process_batch(request_type)
    
    def process_batch(self, request_type):
        batch = self.batches[request_type]
        if not batch:
            return
        
        # Extract prompts
        prompts = [item['prompt'] for item in batch]
        
        # Process batch (this should be async in real implementation)
        results = self.llm.batch(prompts)
        
        # Call callbacks with results
        for item, result in zip(batch, results):
            item['callback'](result)
        
        # Clear batch
        self.batches[request_type] = []
        self.last_batch_time[request_type] = time.time()

# Usage
batcher = SmartBatcher()

def handle_result(result):
    print(f"Got result: {result}")

# Add requests - they'll be batched automatically
batcher.add_request("summarization", "Summarize this text...", handle_result)
batcher.add_request("summarization", "Summarize this other text...", handle_result)
```

---

### **4. Memory Optimization**

**Efficient Memory Management:**

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI

class OptimizedMemory:
    def __init__(self, max_token_limit=1000):
        self.llm = OpenAI(temperature=0)
        
        # Use summary buffer memory for efficiency
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=max_token_limit,
            return_messages=True
        )
    
    def add_exchange(self, human_input, ai_output):
        self.memory.save_context(
            {"input": human_input},
            {"output": ai_output}
        )
    
    def get_context(self):
        return self.memory.load_memory_variables({})
    
    def clear_old_memory(self, keep_last_n=5):
        """Keep only the last N exchanges, summarize the rest"""
        messages = self.memory.chat_memory.messages
        if len(messages) > keep_last_n * 2:  # 2 messages per exchange
            # Keep recent messages
            recent_messages = messages[-(keep_last_n * 2):]
            
            # Summarize older messages
            old_messages = messages[:-(keep_last_n * 2)]
            summary = self.llm(f"Summarize this conversation: {old_messages}")
            
            # Reset memory with summary and recent messages
            self.memory.clear()
            self.memory.moving_summary_buffer = summary
            self.memory.chat_memory.messages = recent_messages

# Usage
memory = OptimizedMemory(max_token_limit=800)

# Periodically clean memory
exchange_count = 0
for user_input in conversation_inputs:
    response = chain.predict(input=user_input)
    memory.add_exchange(user_input, response)
    
    exchange_count += 1
    if exchange_count % 10 == 0:  # Clean every 10 exchanges
        memory.clear_old_memory(keep_last_n=3)
```

---

### **5. Vector Store Optimization**

**Efficient Vector Operations:**

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

class OptimizedVectorStore:
    def __init__(self, embedding_model="text-embedding-ada-002"):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = None
        self.embedding_cache = {}
    
    def create_vectorstore(self, documents, batch_size=100):
        """Create vector store with batched embeddings"""
        all_texts = [doc.page_content for doc in documents]
        
        # Process in batches to avoid rate limits
        all_embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Add delay to respect rate limits
            if i + batch_size < len(all_texts):
                time.sleep(1)
        
        # Create FAISS index
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Use IVF index for large datasets (faster search)
        if len(documents) > 1000:
            import faiss
            dimension = embeddings_array.shape[1]
            nlist = min(100, len(documents) // 10)  # Number of clusters
            
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings_array)
            index.add(embeddings_array)
            
            # Use custom FAISS index
            self.vectorstore = FAISS(
                embedding_function=self.embeddings.embed_query,
                index=index,
                docstore={i: doc for i, doc in enumerate(documents)},
                index_to_docstore_id={i: i for i in range(len(documents))}
            )
        else:
            # Use default FAISS for smaller datasets
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
    
    def similarity_search_optimized(self, query, k=5, score_threshold=0.7):
        """Optimized similarity search with caching and filtering"""
        
        # Check cache first
        if query in self.embedding_cache:
            query_embedding = self.embedding_cache[query]
        else:
            query_embedding = self.embeddings.embed_query(query)
            self.embedding_cache[query] = query_embedding
        
        # Search with score threshold
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query, k=k*2  # Get more results for filtering
        )
        
        # Filter by score threshold
        filtered_docs = [
            doc for doc, score in docs_with_scores 
            if score <= score_threshold  # Lower score = more similar
        ]
        
        return filtered_docs[:k]

# Usage
optimizer = OptimizedVectorStore()
optimizer.create_vectorstore(documents, batch_size=50)

# Fast searches with caching
results = optimizer.similarity_search_optimized(
    "What is machine learning?",
    k=3,
    score_threshold=0.8
)
```

---

## 💰 Cost Optimization

### **Token Usage Monitoring**

```python
from langchain.callbacks import get_openai_callback
import json
from datetime import datetime

class CostMonitor:
    def __init__(self, daily_budget=50.0):
        self.daily_budget = daily_budget
        self.usage_log = []
        self.daily_usage = 0.0
        self.current_date = datetime.now().date()
    
    def track_usage(self, operation_name, callback_result):
        current_date = datetime.now().date()
        
        # Reset daily usage at midnight
        if current_date != self.current_date:
            self.daily_usage = 0.0
            self.current_date = current_date
        
        # Log usage
        usage_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation_name,
            'tokens': callback_result.total_tokens,
            'cost': callback_result.total_cost,
            'prompt_tokens': callback_result.prompt_tokens,
            'completion_tokens': callback_result.completion_tokens
        }
        
        self.usage_log.append(usage_entry)
        self.daily_usage += callback_result.total_cost
        
        # Check budget
        if self.daily_usage > self.daily_budget:
            raise Exception(f"Daily budget exceeded: ${self.daily_usage:.2f} > ${self.daily_budget}")
        
        # Print summary
        print(f"💰 ${callback_result.total_cost:.4f} | 🎯 {callback_result.total_tokens} tokens | Daily: ${self.daily_usage:.2f}")
    
    def get_statistics(self):
        if not self.usage_log:
            return {}
        
        total_cost = sum(entry['cost'] for entry in self.usage_log)
        total_tokens = sum(entry['tokens'] for entry in self.usage_log)
        avg_cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0
        
        return {
            'total_operations': len(self.usage_log),
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'average_cost_per_token': avg_cost_per_token,
            'daily_usage': self.daily_usage,
            'budget_remaining': self.daily_budget - self.daily_usage
        }

# Usage
monitor = CostMonitor(daily_budget=10.0)

def monitored_llm_call(prompt, operation_name="llm_call"):
    with get_openai_callback() as cb:
        result = llm(prompt)
        monitor.track_usage(operation_name, cb)
        return result

# Use throughout your application
response = monitored_llm_call("What is AI?", "basic_question")
print(f"Response: {response}")

# Get statistics
stats = monitor.get_statistics()
print(f"Statistics: {json.dumps(stats, indent=2)}")
```

### **Intelligent Prompt Optimization**

```python
class PromptOptimizer:
    def __init__(self):
        self.optimization_cache = {}
    
    def optimize_prompt(self, original_prompt, max_length=1000):
        """Optimize prompt for better performance"""
        
        # Check cache
        if original_prompt in self.optimization_cache:
            return self.optimization_cache[original_prompt]
        
        optimized = original_prompt
        
        # 1. Remove redundant whitespace
        optimized = ' '.join(optimized.split())
        
        # 2. Truncate if too long (keep important parts)
        if len(optimized) > max_length:
            # Keep first part (context) and last part (question)
            words = optimized.split()
            if len(words) > 50:
                # Keep first 30 words and last 20 words
                optimized = ' '.join(words[:30] + ['...'] + words[-20:])
        
        # 3. Add efficiency instructions
        if "explain" in optimized.lower() or "describe" in optimized.lower():
            optimized += " (Be concise and focus on key points.)"
        
        # Cache the optimization
        self.optimization_cache[original_prompt] = optimized
        return optimized
    
    def estimate_tokens(self, text):
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def choose_model_by_complexity(self, prompt):
        """Choose the most cost-effective model for the task"""
        
        complexity_indicators = [
            'analyze', 'complex', 'detailed', 'comprehensive',
            'reasoning', 'logic', 'philosophy', 'ethics'
        ]
        
        is_complex = any(indicator in prompt.lower() for indicator in complexity_indicators)
        estimated_tokens = self.estimate_tokens(prompt)
        
        if is_complex or estimated_tokens > 500:
            return "gpt-4"  # Use powerful model for complex tasks
        else:
            return "gpt-3.5-turbo"  # Use faster, cheaper model

# Usage
optimizer = PromptOptimizer()

def smart_llm_call(original_prompt):
    # Optimize prompt
    optimized_prompt = optimizer.optimize_prompt(original_prompt)
    
    # Choose appropriate model
    model_name = optimizer.choose_model_by_complexity(optimized_prompt)
    
    # Create model instance
    llm = OpenAI(model_name=model_name)
    
    print(f"🎯 Using {model_name} for optimized prompt")
    print(f"📏 Token estimate: {optimizer.estimate_tokens(optimized_prompt)}")
    
    return llm(optimized_prompt)

# Example
result = smart_llm_call("Please explain in great detail with comprehensive analysis and thorough examination of all aspects the concept of machine learning and how it works")
```

---

## 🏗️ Infrastructure Optimization

### **Connection Pooling**

```python
import threading
from queue import Queue
from langchain.llms import OpenAI

class LLMPool:
    def __init__(self, pool_size=5):
        self.pool = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        
        # Pre-create LLM instances
        for _ in range(pool_size):
            llm = OpenAI(temperature=0.7)
            self.pool.put(llm)
    
    def get_llm(self):
        """Get an LLM instance from the pool"""
        return self.pool.get()
    
    def return_llm(self, llm):
        """Return an LLM instance to the pool"""
        self.pool.put(llm)
    
    def execute(self, prompt):
        """Execute a prompt using pooled LLM"""
        llm = self.get_llm()
        try:
            result = llm(prompt)
            return result
        finally:
            self.return_llm(llm)

# Usage
llm_pool = LLMPool(pool_size=3)

def handle_request(prompt):
    return llm_pool.execute(prompt)

# Multiple threads can use the pool safely
import concurrent.futures

prompts = ["Question 1", "Question 2", "Question 3"]

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(handle_request, prompts))
```

### **Load Balancing**

```python
import random
from typing import List

class LoadBalancer:
    def __init__(self, api_keys: List[str], models: List[str] = None):
        self.api_keys = api_keys
        self.models = models or ["gpt-3.5-turbo"] * len(api_keys)
        self.current_index = 0
        self.error_counts = {key: 0 for key in api_keys}
        self.max_errors = 3
    
    def get_next_llm(self):
        """Get next available LLM using round-robin with error handling"""
        
        attempts = 0
        while attempts < len(self.api_keys):
            key = self.api_keys[self.current_index]
            model = self.models[self.current_index]
            
            # Skip if too many errors
            if self.error_counts[key] < self.max_errors:
                llm = OpenAI(
                    openai_api_key=key,
                    model_name=model,
                    request_timeout=30
                )
                self.current_index = (self.current_index + 1) % len(self.api_keys)
                return llm, key
            
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            attempts += 1
        
        raise Exception("All API keys have exceeded error threshold")
    
    def record_error(self, api_key):
        """Record an error for an API key"""
        self.error_counts[api_key] += 1
        print(f"⚠️ Error recorded for key ending in ...{api_key[-4:]}")
    
    def reset_errors(self, api_key):
        """Reset error count after successful request"""
        self.error_counts[api_key] = 0
    
    def execute_with_fallback(self, prompt):
        """Execute prompt with automatic fallback"""
        last_exception = None
        
        for attempt in range(len(self.api_keys)):
            try:
                llm, api_key = self.get_next_llm()
                result = llm(prompt)
                self.reset_errors(api_key)
                return result
                
            except Exception as e:
                last_exception = e
                self.record_error(api_key)
                print(f"🔄 Attempting fallback (attempt {attempt + 1})")
        
        raise Exception(f"All fallbacks failed. Last error: {last_exception}")

# Usage
api_keys = [
    "sk-key-1...",
    "sk-key-2...",
    "sk-key-3..."
]

balancer = LoadBalancer(api_keys)

def robust_llm_call(prompt):
    return balancer.execute_with_fallback(prompt)

# Automatically handles failures and load balancing
result = robust_llm_call("What is machine learning?")
```

---

## 📊 Monitoring and Profiling

### **Performance Profiler**

```python
import time
import functools
import statistics
from collections import defaultdict

class PerformanceProfiler:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
    
    def profile(self, operation_name):
        """Decorator to profile function performance"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = e
                    success = False
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    self.metrics[operation_name].append({
                        'duration': duration,
                        'success': success,
                        'timestamp': start_time
                    })
                    
                    if not success:
                        raise result
                    
                    return result
            return wrapper
        return decorator
    
    def get_stats(self, operation_name=None):
        """Get performance statistics"""
        if operation_name:
            data = self.metrics[operation_name]
        else:
            # Aggregate all operations
            data = []
            for ops in self.metrics.values():
                data.extend(ops)
        
        if not data:
            return {}
        
        durations = [m['duration'] for m in data if m['success']]
        success_rate = sum(1 for m in data if m['success']) / len(data)
        
        return {
            'total_calls': len(data),
            'success_rate': success_rate,
            'avg_duration': statistics.mean(durations) if durations else 0,
            'median_duration': statistics.median(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'p95_duration': self._percentile(durations, 95) if durations else 0
        }
    
    def _percentile(self, data, percentile):
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def print_report(self):
        """Print performance report"""
        print("\n📊 Performance Report")
        print("=" * 50)
        
        for operation, _ in self.metrics.items():
            stats = self.get_stats(operation)
            print(f"\n🔍 {operation}:")
            print(f"  Calls: {stats['total_calls']}")
            print(f"  Success Rate: {stats['success_rate']:.2%}")
            print(f"  Avg Duration: {stats['avg_duration']:.3f}s")
            print(f"  P95 Duration: {stats['p95_duration']:.3f}s")

# Usage
profiler = PerformanceProfiler()

@profiler.profile("llm_call")
def profiled_llm_call(prompt):
    return llm(prompt)

@profiler.profile("vector_search")
def profiled_vector_search(query):
    return vectorstore.similarity_search(query)

# Use your functions normally
result1 = profiled_llm_call("What is AI?")
result2 = profiled_vector_search("machine learning")

# Get performance insights
profiler.print_report()
```

---

## 🚀 Production Deployment Optimization

### **FastAPI with Optimization**

```python
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uvloop

app = FastAPI(title="Optimized LangChain API")

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use faster event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Connection pooling and caching
llm_pool = LLMPool(pool_size=10)
cache = InMemoryCache()

@app.on_startup
async def startup_event():
    """Initialize resources on startup"""
    # Warm up the cache with common queries
    warmup_queries = [
        "Hello",
        "What is AI?",
        "How are you?"
    ]
    
    for query in warmup_queries:
        try:
            llm_pool.execute(query)
        except:
            pass  # Ignore warmup errors

@app.post("/chat")
async def chat_endpoint(
    prompt: str,
    background_tasks: BackgroundTasks
):
    """Optimized chat endpoint"""
    
    # Add background task for analytics
    background_tasks.add_task(log_usage, prompt)
    
    # Execute with pooled LLM
    result = llm_pool.execute(prompt)
    
    return {"response": result}

def log_usage(prompt: str):
    """Background task for logging"""
    # Log to analytics service asynchronously
    pass

# Run with optimized settings
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,  # Multiple workers for better throughput
        loop="uvloop",  # Faster event loop
        access_log=False  # Disable access logs for performance
    )
```

### **Docker Optimization**

```dockerfile
# Multi-stage build for smaller image
FROM python:3.9-slim as builder

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set environment variables for performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONOPTIMIZE=1

# Copy application
COPY . /app
WORKDIR /app

# Use production WSGI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## 🎯 Performance Checklist

### **Pre-Production Checklist**

- [ ] **Caching Implemented**
  - [ ] LLM response caching enabled
  - [ ] Vector search results cached
  - [ ] Embedding computations cached

- [ ] **Model Optimization**
  - [ ] Right model for each task
  - [ ] Token limits set appropriately
  - [ ] Temperature optimized for use case

- [ ] **Infrastructure**
  - [ ] Connection pooling configured
  - [ ] Async operations where possible
  - [ ] Load balancing for high availability

- [ ] **Cost Controls**
  - [ ] Daily/monthly budget limits
  - [ ] Usage monitoring in place
  - [ ] Cost per operation tracked

- [ ] **Monitoring**
  - [ ] Performance metrics tracked
  - [ ] Error rates monitored
  - [ ] Alerting configured

### **Performance Testing**

```python
import time
import statistics
import concurrent.futures

def performance_test():
    """Run comprehensive performance test"""
    
    test_prompts = [
        "What is machine learning?",
        "Explain quantum computing",
        "How does blockchain work?",
        "What is artificial intelligence?",
        "Describe neural networks"
    ]
    
    results = {
        'sequential': [],
        'concurrent': [],
        'cached': []
    }
    
    # Test 1: Sequential execution
    print("🔄 Testing sequential execution...")
    for prompt in test_prompts:
        start = time.time()
        llm(prompt)
        duration = time.time() - start
        results['sequential'].append(duration)
    
    # Test 2: Concurrent execution
    print("🔄 Testing concurrent execution...")
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(llm, test_prompts))
    total_concurrent = time.time() - start
    results['concurrent'] = [total_concurrent / len(test_prompts)]
    
    # Test 3: Cached execution (run same prompts again)
    print("🔄 Testing cached execution...")
    for prompt in test_prompts:
        start = time.time()
        llm(prompt)  # Should hit cache
        duration = time.time() - start
        results['cached'].append(duration)
    
    # Print results
    print("\n📊 Performance Test Results:")
    for test_type, durations in results.items():
        avg_duration = statistics.mean(durations)
        print(f"{test_type.title()}: {avg_duration:.3f}s average")
    
    return results

# Run performance test
test_results = performance_test()
```

---

<div align="center">

### ⚡ Ready to Build Lightning-Fast Apps?

**Performance is not just about speed—it's about creating delightful user experiences while controlling costs.**

**[🚀 Deploy Optimized App →](../langserve.md)** • **[📊 Monitor Performance →](../langsmith.md)** • **[🔧 Troubleshoot Issues →](./troubleshooting.md)**

---

*Performance optimization is an ongoing process. Monitor, measure, and iterate for continuous improvement.*

</div>