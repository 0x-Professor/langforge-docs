# LangChain Core Components and Features

This comprehensive guide covers the essential components of LangChain and how to effectively use them in your applications.

## Overview

LangChain provides a modular framework for building applications with Large Language Models (LLMs). This section covers the core components that form the foundation of any LangChain application:

- **Models**: LLMs, Chat Models, and Embeddings
- **Prompts**: Template management and optimization
- **Chains**: Combining components into workflows
- **Memory**: Conversation and context management
- **Agents**: Autonomous decision-making systems
- **Tools**: External service integrations

## Language Models and Chat Models

### LLM Integration

```python
from langchain.llms import OpenAI, Anthropic, Cohere
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.callbacks import get_openai_callback

# Initialize different LLM providers
openai_llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
    temperature=0.7,
    max_tokens=1000,
    streaming=True
)

# Chat models for conversational AI
chat_model = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=2000
)

# Track token usage and costs
with get_openai_callback() as cb:
    response = openai_llm("Explain quantum computing in simple terms")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
```

### Model Configuration Best Practices

```python
from langchain.schema import BaseLanguageModel
from typing import Dict, Any

class ModelManager:
    """Manage different LLM configurations for various use cases"""
    
    def __init__(self):
        self.model_configs = {
            "creative_writing": {
                "temperature": 0.9,
                "max_tokens": 2000,
                "top_p": 0.9
            },
            "code_generation": {
                "temperature": 0.1,
                "max_tokens": 1500,
                "top_p": 0.95
            },
            "question_answering": {
                "temperature": 0.3,
                "max_tokens": 500,
                "top_p": 0.8
            },
            "summarization": {
                "temperature": 0.2,
                "max_tokens": 300,
                "top_p": 0.85
            }
        }
    
    def get_model_for_task(self, task: str, model_provider: str = "openai") -> BaseLanguageModel:
        """Get optimized model configuration for specific tasks"""
        
        config = self.model_configs.get(task, self.model_configs["question_answering"])
        
        if model_provider == "openai":
            from langchain.llms import OpenAI
            return OpenAI(**config)
        elif model_provider == "anthropic":
            from langchain.llms import Anthropic
            return Anthropic(**config)
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
```

## Embeddings and Vector Operations

### Text Embeddings

```python
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.schema import Document

class EmbeddingManager:
    """Manage embeddings and vector operations"""
    
    def __init__(self, embedding_provider: str = "openai"):
        if embedding_provider == "openai":
            self.embeddings = OpenAIEmbeddings()
        elif embedding_provider == "huggingface":
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
    
    def create_vector_store(self, documents: list, store_type: str = "faiss"):
        """Create vector store from documents"""
        
        if store_type == "faiss":
            return FAISS.from_documents(documents, self.embeddings)
        elif store_type == "chroma":
            return Chroma.from_documents(documents, self.embeddings)
        else:
            raise ValueError(f"Unsupported vector store: {store_type}")
    
    def similarity_search_with_metadata(self, vectorstore, query: str, k: int = 5):
        """Perform similarity search with relevance scoring"""
        
        # Get documents with similarity scores
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_and_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score,
                "relevance": "high" if score > 0.8 else "medium" if score > 0.6 else "low"
            })
        
        return results

# Example usage
embedding_manager = EmbeddingManager()

# Create sample documents
documents = [
    Document(page_content="LangChain is a framework for LLM applications", 
             metadata={"source": "intro.md", "topic": "overview"}),
    Document(page_content="Vector stores enable semantic search capabilities", 
             metadata={"source": "vectors.md", "topic": "embeddings"}),
    Document(page_content="Chains combine multiple components into workflows", 
             metadata={"source": "chains.md", "topic": "architecture"})
]

# Create vector store and search
vectorstore = embedding_manager.create_vector_store(documents)
results = embedding_manager.similarity_search_with_metadata(
    vectorstore, 
    "How does LangChain work?", 
    k=3
)
```

## Token Usage Tracking and Optimization

### Cost Management

```python
from langchain.callbacks import get_openai_callback
from langchain.schema import BaseCallbackHandler
import json
from datetime import datetime

class TokenUsageTracker(BaseCallbackHandler):
    """Track token usage across multiple LLM calls"""
    
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.call_history = []
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts processing"""
        self.start_time = datetime.now()
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes processing"""
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)
            total_tokens = token_usage.get('total_tokens', 0)
            
            # Calculate cost (example rates for GPT-3.5)
            cost = (prompt_tokens * 0.0015 / 1000) + (completion_tokens * 0.002 / 1000)
            
            self.total_tokens += total_tokens
            self.total_cost += cost
            
            self.call_history.append({
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
                "duration": (datetime.now() - self.start_time).total_seconds()
            })
    
    def get_summary(self) -> dict:
        """Get usage summary"""
        return {
            "total_calls": len(self.call_history),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "average_tokens_per_call": self.total_tokens / max(len(self.call_history), 1),
            "cost_per_call": self.total_cost / max(len(self.call_history), 1)
        }

# Usage example with tracking
tracker = TokenUsageTracker()
llm = OpenAI(temperature=0.7, callbacks=[tracker])

# Make multiple LLM calls
responses = [
    llm("What is machine learning?"),
    llm("Explain neural networks"),
    llm("What are transformers in AI?")
]

# Get usage summary
summary = tracker.get_summary()
print(f"Total cost: ${summary['total_cost']:.4f}")
print(f"Average tokens per call: {summary['average_tokens_per_call']:.0f}")
```

## Advanced Model Features

### Streaming Responses

```python
from langchain.llms import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import BaseCallbackHandler

class CustomStreamingHandler(BaseCallbackHandler):
    """Custom streaming handler for real-time processing"""
    
    def __init__(self):
        self.tokens = []
        self.complete_response = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Process each token as it's generated"""
        self.tokens.append(token)
        self.complete_response += token
        
        # Custom processing (e.g., real-time analysis, filtering, etc.)
        if len(token.strip()) > 0:
            print(f"New token: '{token}'", end="", flush=True)
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Process complete response"""
        print(f"\n\nComplete response ({len(self.tokens)} tokens):")
        print(self.complete_response)

# Streaming LLM setup
streaming_handler = CustomStreamingHandler()
streaming_llm = OpenAI(
    temperature=0.7,
    streaming=True,
    callbacks=[streaming_handler]
)

# Generate streaming response
response = streaming_llm("Write a short story about AI and creativity")
```

### Model Comparison and A/B Testing

```python
from langchain.llms import OpenAI, Anthropic
from langchain.evaluation import load_evaluator
import asyncio

class ModelComparator:
    """Compare different models for the same task"""
    
    def __init__(self):
        self.models = {
            "gpt-3.5": OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7),
            "gpt-4": OpenAI(model_name="gpt-4", temperature=0.7),
            "claude": Anthropic(temperature=0.7)
        }
        self.evaluator = load_evaluator("criteria", criteria="helpfulness")
    
    async def compare_models(self, prompt: str) -> dict:
        """Compare responses from different models"""
        
        results = {}
        
        for model_name, model in self.models.items():
            try:
                response = model(prompt)
                
                # Evaluate response quality
                evaluation = self.evaluator.evaluate_strings(
                    prediction=response,
                    input=prompt
                )
                
                results[model_name] = {
                    "response": response,
                    "evaluation_score": evaluation.get("score", 0),
                    "response_length": len(response),
                    "status": "success"
                }
                
            except Exception as e:
                results[model_name] = {
                    "response": None,
                    "error": str(e),
                    "status": "error"
                }
        
        return results
    
    def get_best_model(self, comparison_results: dict) -> str:
        """Determine the best performing model"""
        
        best_model = None
        best_score = -1
        
        for model_name, result in comparison_results.items():
            if result["status"] == "success":
                score = result.get("evaluation_score", 0)
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model

# Example usage
comparator = ModelComparator()
test_prompt = "Explain the concept of artificial general intelligence in simple terms"

# Run comparison
comparison_results = asyncio.run(comparator.compare_models(test_prompt))
best_model = comparator.get_best_model(comparison_results)

print(f"Best performing model: {best_model}")
for model_name, result in comparison_results.items():
    if result["status"] == "success":
        print(f"{model_name}: Score {result['evaluation_score']:.2f}")
```

## Key Features Summary

### Core Capabilities

1. **Multi-Provider Support**: Seamlessly switch between OpenAI, Anthropic, Cohere, and other providers
2. **Cost Optimization**: Built-in token tracking and cost management
3. **Streaming Support**: Real-time response generation for better user experience
4. **Model Comparison**: A/B testing capabilities for optimal model selection
5. **Embedding Integration**: Vector search and semantic similarity operations
6. **Performance Monitoring**: Comprehensive usage analytics and optimization insights

### Best Practices

1. **Always track token usage** in production applications
2. **Use appropriate temperature settings** for different task types
3. **Implement streaming** for better user experience with long responses
4. **Cache embeddings** when possible to reduce costs
5. **Test multiple models** to find the best fit for your use case
6. **Monitor performance metrics** and adjust configurations accordingly

This comprehensive approach to LangChain components ensures robust, cost-effective, and high-performing applications that can scale with your needs.