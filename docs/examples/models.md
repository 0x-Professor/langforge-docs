# Models

## Overview

LangChain provides a unified interface for working with various language models, making it easy to switch between different providers and model types. The library supports LLMs, chat models, and embeddings.

> **Tip:** When choosing between LLMs and chat models, prefer chat models for conversation-based applications as they are specifically designed for multi-turn conversations.

### Model Types

- **LLMs**: Text-in, text-out models for single-turn tasks like completion and summarization.
- **Chat Models**: Message-in, message-out models designed for multi-turn conversations.
- **Embeddings**: Convert text to vector representations for semantic search and retrieval.

## LLMs

LLMs (Large Language Models) are text-in, text-out models that take a text string as input and return a text string as output. They are great for single-turn tasks like text completion, summarization, and question answering.

> **Note:** For most use cases, we recommend using chat models instead of raw LLMs as they provide better support for conversation history and system prompts.

### Basic Usage

```python
from langchain_openai import OpenAI
from langchain.callbacks import get_openai_callback

# Initialize LLM
llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.7,
    max_tokens=1000
)

# Generate text with token usage tracking
with get_openai_callback() as cb:
    response = llm.invoke("Explain quantum computing in simple terms.")
    print(f"Response: {response}")
    print(f"Tokens used: {cb.total_tokens}")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")
    print(f"Total cost (USD): ${cb.total_cost}")
```

### Streaming Support

```python
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.7)

# Stream response
for chunk in llm.stream("Write a short story about AI"):
    print(chunk, end="", flush=True)
```

### Async Usage

```python
import asyncio
from langchain_openai import OpenAI

async def async_generate():
    llm = OpenAI(temperature=0.7)
    response = await llm.ainvoke("Explain machine learning")
    return response

# Run async
result = asyncio.run(async_generate())
print(result)
```

### Available Providers

LangChain supports multiple LLM providers through a unified interface:

| Provider | Package | Models |
|----------|---------|--------|
| OpenAI | `langchain-openai` | GPT-4, GPT-3.5 |
| Anthropic | `langchain-anthropic` | Claude-3, Claude-2 |
| Google | `langchain-google-genai` | Gemini Pro, Gemini Ultra |
| Cohere | `langchain-cohere` | Command, Generate |
| HuggingFace | `langchain-huggingface` | Open-source models |
| Ollama | `langchain-community` | Local models |

## Chat Models

Chat models are conversation-based models that take a list of messages as input and return a message as output. They are designed for multi-turn conversations and support system prompts, user messages, and assistant messages.

> **Tip:** Chat models are the recommended way to build conversational AI applications as they handle conversation state and message formatting for you.

### Basic Usage

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Initialize chat model
chat = ChatOpenAI(
    model="gpt-4",
    temperature=0.7
)

# Create messages with system prompt
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain quantum computing in simple terms.")
]

# Get response
response = chat.invoke(messages)
print(response.content)
```

### Streaming Chat

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

chat = ChatOpenAI(model="gpt-3.5-turbo")

# Stream chat response
for chunk in chat.stream([HumanMessage(content="Tell me a joke")]):
    print(chunk.content, end="", flush=True)
```

### Multi-turn Conversation

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

chat = ChatOpenAI(model="gpt-3.5-turbo")

# Build conversation history
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What's the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
    HumanMessage(content="What's the population?")
]

response = chat.invoke(messages)
print(response.content)
```

### Function Calling

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Your weather API implementation
    return f"The weather in {location} is sunny and 72°F"

# Initialize chat model with tools
chat = ChatOpenAI(model="gpt-3.5-turbo")
chat_with_tools = chat.bind_tools([get_weather])

# Use function calling
messages = [HumanMessage(content="What's the weather like in New York?")]
response = chat_with_tools.invoke(messages)

# Check if the model wants to call a function
if response.tool_calls:
    tool_call = response.tool_calls[0]
    print(f"Model wants to call: {tool_call['name']}")
    print(f"With arguments: {tool_call['args']}")
```

### Available Chat Models

| Provider | Package | Models |
|----------|---------|--------|
| OpenAI | `langchain-openai` | gpt-4, gpt-3.5-turbo |
| Anthropic | `langchain-anthropic` | claude-3-opus, claude-3-sonnet |
| Google | `langchain-google-genai` | gemini-pro, gemini-ultra |
| Mistral | `langchain-mistralai` | mistral-medium, mistral-small |

## Embeddings

Embeddings are vector representations of text that capture semantic meaning. They are useful for tasks like semantic search, clustering, and classification.

### Basic Usage

```python
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Generate embeddings for text
text = "This is a sample text for embedding"
vector = embeddings.embed_query(text)

# Generate embeddings for multiple documents
documents = ["Document 1", "Document 2", "Document 3"]
vectors = embeddings.embed_documents(documents)

print(f"Vector dimension: {len(vector)}")
print(f"Number of document vectors: {len(vectors)}")
```

### Comparing Embeddings

```python
import numpy as np
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create embeddings for similar texts
text1 = "I love programming"
text2 = "I enjoy coding"
text3 = "The weather is nice today"

vec1 = embeddings.embed_query(text1)
vec2 = embeddings.embed_query(text2)
vec3 = embeddings.embed_query(text3)

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Similarity between '{text1}' and '{text2}': {cosine_similarity(vec1, vec2):.3f}")
print(f"Similarity between '{text1}' and '{text3}': {cosine_similarity(vec1, vec3):.3f}")
```

### Available Embedding Models

| Provider | Package | Models |
|----------|---------|--------|
| OpenAI | `langchain-openai` | text-embedding-3-small, text-embedding-3-large |
| HuggingFace | `langchain-huggingface` | all-mpnet-base-v2, all-MiniLM-L6-v2 |
| Cohere | `langchain-cohere` | embed-english-v3.0, embed-multilingual-v3.0 |
| Google | `langchain-google-genai` | text-embedding-004 |

## Local Models

### Using Ollama

```python
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# Use local LLM with Ollama
llm = Ollama(model="llama2")
response = llm.invoke("Explain artificial intelligence")

# Use local embeddings
embeddings = OllamaEmbeddings(model="llama2")
vector = embeddings.embed_query("Sample text")
```

### Using HuggingFace Transformers

```python
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

# Load a local model
llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/DialoGPT-medium",
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 100}
)

# Local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

## Custom Models

You can create custom model wrappers to integrate with any API or local model:

```python
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any

class CustomLLM(LLM):
    """A custom LLM wrapper."""
    
    model_name: str = "custom-model"
    api_key: str = ""
    
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to your custom model API."""
        # Implement your custom model logic here
        try:
            response = self._make_api_call(prompt)
            return response
        except Exception as e:
            raise ValueError(f"Error calling custom model: {e}")
    
    def _make_api_call(self, prompt: str) -> str:
        """Make API call to your custom model."""
        # Your custom implementation
        return f"Custom response to: {prompt}"

# Use your custom model
custom_llm = CustomLLM(api_key="your-api-key")
response = custom_llm.invoke("Hello, world!")
print(response)
```

### Custom Chat Model

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from typing import List

class CustomChatModel(BaseChatModel):
    """A custom chat model wrapper."""
    
    model_name: str = "custom-chat"
    
    @property
    def _llm_type(self) -> str:
        return "custom-chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a response from messages."""
        # Convert messages to your API format
        prompt = self._messages_to_prompt(messages)
        
        # Make API call
        response_text = self._make_api_call(prompt)
        
        # Return as AIMessage
        message = AIMessage(content=response_text)
        return message
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to prompt format."""
        prompt_parts = []
        for message in messages:
            if message.type == "human":
                prompt_parts.append(f"Human: {message.content}")
            elif message.type == "ai":
                prompt_parts.append(f"Assistant: {message.content}")
            elif message.type == "system":
                prompt_parts.append(f"System: {message.content}")
        return "\n".join(prompt_parts)
    
    def _make_api_call(self, prompt: str) -> str:
        """Make API call to your custom model."""
        # Your custom implementation
        return f"Custom chat response to: {prompt}"

# Use your custom chat model
custom_chat = CustomChatModel()
messages = [HumanMessage(content="Hello!")]
response = custom_chat.invoke(messages)
print(response.content)
```

## Model Configuration

### Environment Variables

```python
import os

# Set API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
os.environ["COHERE_API_KEY"] = "your-cohere-key"

# Optional: Set default model parameters
os.environ["OPENAI_MODEL_NAME"] = "gpt-4"
os.environ["OPENAI_TEMPERATURE"] = "0.7"
```

### Model Parameters

```python
from langchain_openai import ChatOpenAI

# Configure model with various parameters
chat = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,           # Creativity (0.0 to 2.0)
    max_tokens=1000,           # Maximum response length
    top_p=0.9,                 # Nucleus sampling
    frequency_penalty=0.0,     # Reduce repetition
    presence_penalty=0.0,      # Encourage new topics
    timeout=60,                # Request timeout
    max_retries=3,             # Retry attempts
    streaming=True,            # Enable streaming
)
```

## Best Practices

### Error Handling

```python
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException

chat = ChatOpenAI()

try:
    response = chat.invoke([HumanMessage(content="Hello")])
    print(response.content)
except LangChainException as e:
    print(f"LangChain error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Rate Limiting

```python
import time
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3)
)
def safe_llm_call(chat, messages):
    """Make LLM call with automatic retry."""
    return chat.invoke(messages)

# Usage
chat = ChatOpenAI()
messages = [HumanMessage(content="Hello")]
response = safe_llm_call(chat, messages)
```

### Cost Management

```python
from langchain.callbacks import get_openai_callback

# Track costs across multiple calls
with get_openai_callback() as cb:
    # Multiple model calls
    response1 = chat.invoke([HumanMessage(content="First query")])
    response2 = chat.invoke([HumanMessage(content="Second query")])
    
    print(f"Total tokens: {cb.total_tokens}")
    print(f"Total cost: ${cb.total_cost:.4f}")
    
    # Set cost alerts
    if cb.total_cost > 1.0:  # $1 threshold
        print("⚠️ Cost threshold exceeded!")
```

### Performance Optimization

```python
# Use batch operations when possible
from langchain_openai import ChatOpenAI

chat = ChatOpenAI()

# Batch multiple requests
inputs = [
    [HumanMessage(content="Question 1")],
    [HumanMessage(content="Question 2")],
    [HumanMessage(content="Question 3")]
]

responses = chat.batch(inputs)
for i, response in enumerate(responses):
    print(f"Response {i+1}: {response.content}")
```