# Memory

## Overview

Memory in LangChain allows you to persist state between chain or agent runs. It's essential for building conversational applications where you need to maintain context across multiple interactions.

> **Tip:** Choose the right type of memory based on your application's needs. Consider factors like conversation length, the importance of context, and performance requirements.

## Types of Memory

### Conversation Buffer Memory

The simplest form of memory that keeps all conversation messages in memory.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

# Initialize the memory
memory = ConversationBufferMemory()

# Create a conversation chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=True
)

# Have a conversation
response1 = conversation.predict(input="Hi there!")
print(f"Response 1: {response1}")

response2 = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
print(f"Response 2: {response2}")

response3 = conversation.predict(input="What did I just say about myself?")
print(f"Response 3: {response3}")

# View the conversation history
print(f"Memory buffer: {memory.buffer}")
```

### Conversation Buffer Window Memory

For longer conversations, limit the amount of conversation history kept in memory using a sliding window:

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

# Initialize memory with a window size of 2 (keeps last 2 exchanges)
memory = ConversationBufferWindowMemory(k=2)

# Create conversation chain
llm = ChatOpenAI(model="gpt-3.5-turbo")
conversation = ConversationChain(llm=llm, memory=memory)

# Add multiple messages
messages = [
    "Hi there!",
    "I'm doing well. How are you?", 
    "What's the weather like?",
    "Do you remember what I first said?"  # This should not remember the first message
]

for msg in messages:
    response = conversation.predict(input=msg)
    print(f"User: {msg}")
    print(f"AI: {response}")
    print("---")

# View current memory (only last 2 exchanges)
print(f"Current memory: {memory.buffer}")
```

### Conversation Summary Memory

For very long conversations, compress the conversation history into a summary:

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

# Initialize memory with an LLM for summarization
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

# Add multiple conversation turns
conversations = [
    ("Hi there!", "Hello! How can I help you today?"),
    ("I'm looking to learn about LangChain.", "LangChain is a framework for developing applications powered by language models."),
    ("What can I build with it?", "You can build chatbots, question-answering systems, summarization tools, and more!"),
    ("How do I get started?", "Start by installing LangChain and exploring the documentation and examples."),
    ("Are there any tutorials?", "Yes, there are many tutorials available in the official documentation and community resources.")
]

for user_input, ai_output in conversations:
    memory.save_context(
        {"input": user_input}, 
        {"output": ai_output}
    )

# View the conversation summary
print(f"Summary: {memory.buffer}")
```

### Entity Memory

Remember specific entities (people, places, things) mentioned in the conversation:

```python
from langchain.memory import ConversationEntityMemory
from langchain_openai import ChatOpenAI

# Initialize entity memory
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
memory = ConversationEntityMemory(llm=llm)

# Add conversations with entities
entity_conversations = [
    ("Alice is an engineer at Google.", "Got it, Alice works at Google as an engineer."),
    ("She has a dog named Max.", "Noted, Alice has a dog named Max."),
    ("Alice lives in San Francisco.", "I'll remember that Alice lives in San Francisco."),
    ("Max is a Golden Retriever.", "Thanks for telling me Max is a Golden Retriever.")
]

for user_input, ai_output in entity_conversations:
    memory.save_context(
        {"input": user_input}, 
        {"output": ai_output}
    )

# Query about specific entities
alice_info = memory.load_memory_variables({"input": "Tell me about Alice"})
max_info = memory.load_memory_variables({"input": "What do you know about Max?"})

print(f"Alice information: {alice_info}")
print(f"Max information: {max_info}")
```

## Custom Memory Implementation

Create custom memory classes for specialized use cases:

```python
from typing import Dict, List, Any
from langchain.schema import BaseMemory
from pydantic import BaseModel, Field
import datetime
import json

class TimestampedMemory(BaseMemory, BaseModel):
    """Custom memory that stores messages with timestamps."""
    
    history: List[Dict[str, Any]] = Field(default_factory=list)
    max_messages: int = Field(default=50)
    
    @property
    def memory_variables(self) -> List[str]:
        return ["chat_history"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables with formatted history."""
        formatted_history = []
        for entry in self.history[-10:]:  # Return last 10 messages
            timestamp = entry["timestamp"]
            formatted_history.append(
                f"[{timestamp}] User: {entry['input']}\nAI: {entry['output']}"
            )
        return {"chat_history": "\n\n".join(formatted_history)}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context with timestamp."""
        entry = {
            "input": inputs.get("input", ""),
            "output": outputs.get("output", ""),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.history.append(entry)
        
        # Keep only the most recent messages
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]
    
    def clear(self) -> None:
        """Clear memory."""
        self.history = []

# Usage example
custom_memory = TimestampedMemory(max_messages=20)

# Add some conversations
custom_memory.save_context(
    {"input": "Hello!"}, 
    {"output": "Hi there! How can I help you today?"}
)

# Load memory variables
memory_vars = custom_memory.load_memory_variables({})
print(memory_vars["chat_history"])
```

## Memory with LCEL Chains

Integrate memory with modern LCEL chains:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create the chain
llm = ChatOpenAI(model="gpt-3.5-turbo")

def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]

chain = (
    RunnablePassthrough.assign(chat_history=load_memory)
    | prompt
    | llm
)

# Function to run conversation with memory
def chat_with_memory(user_input: str):
    # Get response
    response = chain.invoke({"input": user_input})
    
    # Save to memory
    memory.save_context(
        {"input": user_input},
        {"output": response.content}
    )
    
    return response.content

# Have a conversation
print(chat_with_memory("Hi, I'm John"))
print(chat_with_memory("What's my name?"))
print(chat_with_memory("I like pizza"))
print(chat_with_memory("What do I like to eat?"))
```

## Persistent Memory

Store memory in external storage for persistence across sessions:

```python
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory
import os

# File-based persistent memory
def create_persistent_memory(session_id: str):
    # Create a file for this session
    file_path = f"chat_histories/session_{session_id}.json"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Create memory with file storage
    memory = ConversationBufferMemory(
        chat_memory=FileChatMessageHistory(file_path),
        memory_key="chat_history",
        return_messages=True
    )
    
    return memory

# Usage
session_memory = create_persistent_memory("user123")

# Add some messages
session_memory.save_context(
    {"input": "Remember, I prefer tea over coffee"},
    {"output": "I'll remember your preference for tea!"}
)

# Memory persists across application restarts
print(session_memory.load_memory_variables({}))
```

## Best Practices

### 1. Choose the Right Memory Type

```python
# For short conversations (< 10 exchanges)
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()

# For medium conversations (10-50 exchanges)
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=10)

# For long conversations (> 50 exchanges)
from langchain.memory import ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500)
```

### 2. Handle Memory Gracefully

```python
from langchain.memory import ConversationBufferMemory

class SafeMemory(ConversationBufferMemory):
    """Memory wrapper with error handling."""
    
    def save_context(self, inputs, outputs):
        try:
            super().save_context(inputs, outputs)
        except Exception as e:
            print(f"Error saving to memory: {e}")
    
    def load_memory_variables(self, inputs):
        try:
            return super().load_memory_variables(inputs)
        except Exception as e:
            print(f"Error loading from memory: {e}")
            return {self.memory_key: ""}

# Use safe memory
safe_memory = SafeMemory()
```

### 3. Monitor Memory Usage

```python
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

memory = ConversationBufferMemory()

# Track token usage
with get_openai_callback() as cb:
    # Your conversation logic here
    pass

print(f"Memory tokens used: {cb.total_tokens}")
print(f"Memory cost: ${cb.total_cost:.4f}")
```

### 4. Clean Sensitive Data

```python
import re
from langchain.memory import ConversationBufferMemory

class PrivacyAwareMemory(ConversationBufferMemory):
    """Memory that removes sensitive information."""
    
    def _clean_text(self, text: str) -> str:
        """Remove sensitive information from text."""
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
        
        # Remove credit card numbers (simplified)
        text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]', text)
        
        return text
    
    def save_context(self, inputs, outputs):
        # Clean inputs and outputs
        cleaned_inputs = {k: self._clean_text(str(v)) if isinstance(v, str) else v 
                         for k, v in inputs.items()}
        cleaned_outputs = {k: self._clean_text(str(v)) if isinstance(v, str) else v 
                          for k, v in outputs.items()}
        
        super().save_context(cleaned_inputs, cleaned_outputs)

# Use privacy-aware memory
privacy_memory = PrivacyAwareMemory()
```

### 5. Test Memory Behavior

```python
def test_memory_functionality():
    """Test memory implementation."""
    from langchain.memory import ConversationBufferMemory
    
    memory = ConversationBufferMemory()
    
    # Test basic functionality
    memory.save_context({"input": "test"}, {"output": "response"})
    result = memory.load_memory_variables({})
    assert "test" in result["history"]
    
    # Test memory persistence
    memory.save_context({"input": "test2"}, {"output": "response2"})
    result = memory.load_memory_variables({})
    assert "test2" in result["history"]
    
    # Test memory clearing
    memory.clear()
    result = memory.load_memory_variables({})
    assert result["history"] == ""
    
    print("✅ Memory tests passed!")

# Run tests
test_memory_functionality()
```

## Memory Optimization Tips

### 1. Use Appropriate Memory Size

```python
# Calculate optimal window size based on token limits
def calculate_optimal_window_size(average_message_tokens=50, max_context_tokens=4000):
    """Calculate optimal memory window size."""
    # Reserve tokens for prompt and response
    available_tokens = max_context_tokens * 0.7  # Use 70% for memory
    optimal_window = int(available_tokens / (average_message_tokens * 2))  # *2 for input+output
    return max(1, optimal_window)

optimal_k = calculate_optimal_window_size()
memory = ConversationBufferWindowMemory(k=optimal_k)
```

### 2. Implement Memory Compression

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

# Use summary buffer for automatic compression
llm = ChatOpenAI(model="gpt-3.5-turbo")
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,  # When to start summarizing
    moving_summary_buffer="",  # Initial summary
    return_messages=True
)
```

### 3. Batch Memory Operations

```python
def batch_save_conversations(memory, conversations):
    """Save multiple conversations efficiently."""
    for user_input, ai_output in conversations:
        memory.save_context(
            {"input": user_input},
            {"output": ai_output}
        )

# Usage
conversations = [
    ("Hello", "Hi there!"),
    ("How are you?", "I'm doing well, thank you!"),
    ("Goodbye", "See you later!")
]

batch_save_conversations(memory, conversations)
```