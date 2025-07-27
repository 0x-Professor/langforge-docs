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
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Initialize LLM with streaming
llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
    temperature=0.7,
    max_tokens=1000,
    streaming=True
)

# Generate text with token usage tracking
# The callback will automatically track token usage and cost
cb = get_openai_callback()
try:
    response = llm.invoke("Explain quantum computing in simple terms.")
    print(f"Response: {response}")
    print(f"Tokens used: {cb.total_tokens}")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")
    print(f"Total cost (USD): ${cb.total_cost}")
finally:
    cb.__exit__(None, None, None)
```

### Key Features

- Streaming support for real-time output
- Token usage tracking
- Automatic retry on rate limits

### Common Use Cases

- Text generation
- Summarization
- Question answering
- Text classification

### Available Providers

LangChain supports multiple LLM providers through a unified interface:

| Provider | Status | Notes |
|----------|--------|-------|
| OpenAI | Stable | GPT models |
| Anthropic | Beta | Claude models |
| Cohere | Stable | Command models |
| HuggingFace | Stable | Open-source models |
| Replicate | Beta | Cloud-hosted models |
| Custom | Stable | Custom implementations |

## Chat Models

Chat models are conversation-based models that take a list of messages as input and return a message as output. They are designed for multi-turn conversations and support system prompts, user messages, and assistant messages.

> **Tip:** Chat models are the recommended way to build conversational AI applications as they handle conversation state and message formatting for you.

### Basic Usage

```python
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Initialize chat model with streaming
chat = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Create messages with system prompt
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain quantum computing in simple terms.")
]

# Stream response
response = chat.invoke(messages)
```

### Key Features

- Streaming support for real-time responses
- Built-in message history management
- Support for system prompts and message roles

### Common Use Cases

- Chat applications
- AI assistants
- Multi-turn conversations
- Interactive applications

### Available Providers

Chat models are available from various providers with different capabilities and pricing:

| Provider | Status | Models |
|----------|--------|--------|
| OpenAI Chat | Stable | gpt-4, gpt-3.5-turbo |
| Anthropic | Beta | claude-2, claude-instant |
| Google | Beta | chat-bison, codechat-bison |

## Embeddings

Embeddings are vector representations of text that capture semantic meaning. They are useful for tasks like semantic search, clustering, and classification.

### Basic Usage

```python
from langchain.embeddings import OpenAIEmbeddings

# Initialize embeddings model
embeddings = OpenAIEmbeddings()

# Generate embeddings for text
text = "This is a sample text for embedding"
vector = embeddings.embed_query(text)

# Generate embeddings for multiple documents
documents = ["Document 1", "Document 2", "Document 3"]
vectors = embeddings.embed_documents(documents)

print(f"Vector dimension: {len(vector)}")
print(f"Number of document vectors: {len(vectors)}")
```

### Available Embedding Models

| Provider | Models |
|----------|--------|
| OpenAI | text-embedding-ada-002, text-embedding-3-small |
| HuggingFace | all-mpnet-base-v2, all-MiniLM-L6-v2 |
| Cohere | embed-english-v3.0, embed-multilingual-v3.0 |
| Google | text-embedding-004, text-multilingual-embedding-002 |

## Custom Models

You can create custom model wrappers to integrate with any API or local model that follows the same interface as LangChain's built-in models.

```python
from langchain.llms.base import LLM
from typing import Optional, List, Any

class CustomLLM(LLM):
    """A custom LLM wrapper."""
    
    model_name: str = "custom-model"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to your custom model API."""
        # Implement your custom model logic here
        response = your_custom_model_api(prompt)
        return response
    
    @property
    def _llm_type(self) -> str:
        return "custom"

# Use your custom model
custom_llm = CustomLLM()
response = custom_llm("Hello, world!")
```

> **Tip:** When creating custom models, make sure to implement the required interfaces to ensure compatibility with the rest of the LangChain ecosystem.

## Model Integrations

LangChain supports a wide range of model providers out of the box. Here are some of the most popular ones:

### OpenAI
GPT-4, GPT-3.5, and embeddings
```bash
pip install langchain-openai
```

### Anthropic
Claude models
```bash
pip install langchain-anthropic
```

### Google
Gemini models
```bash
pip install langchain-google-genai
```

### Hugging Face
Open-source models
```bash
pip install langchain-huggingface
```

## Best Practices

When implementing custom models, make sure to properly handle errors and timeouts, and consider adding retry logic for production use.

### Error Handling
Always implement proper error handling for API calls and model responses.

### Performance Optimization
Use streaming when possible for better user experience with long responses.

### Cost Management
Track token usage and implement cost controls for production applications.