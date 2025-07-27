# LangChain Documentation

## Table of Contents

- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
  - [1. Models](#1-models)
    - [Model Types](#model-types)
    - [Supported Providers](#supported-providers)
    - [Model Parameters](#model-parameters)
    - [Best Practices](#best-practices)
  - [2. Prompts](#2-prompts)
    - [Prompt Templates](#prompt-templates)
    - [Chat Prompt Templates](#chat-prompt-templates)
    - [Few-shot Prompting](#few-shot-prompting)
    - [Output Parsers](#output-parsers)
  - [3. Memory](#3-memory)
    - [Types of Memory](#types-of-memory)
    - [Using Memory with Chains](#using-memory-with-chains)
    - [Custom Memory Implementation](#custom-memory-implementation)
  - [4. Chains](#4-chains)
    - [Types of Chains](#types-of-chains)
    - [Custom Chain Implementation](#custom-chain-implementation)
  - [5. Indexes](#5-indexes)
    - [Document Loaders](#document-loaders)
    - [Text Splitters](#text-splitters)
    - [Vector Stores](#vector-stores)
    - [Document Retrievers](#document-retrievers)
  - [6. Agents](#6-agents)
    - [Agent Types](#agent-types)
    - [Custom Tools](#custom-tools)
    - [Toolkits](#toolkits)
    - [Multi-Agent Systems](#multi-agent-systems)
  - [7. Advanced Topics](#7-advanced-topics)
    - [Custom Components](#custom-components)
    - [Performance Optimization](#performance-optimization)
    - [Error Handling and Retries](#error-handling-and-retries)
    - [Security Best Practices](#security-best-practices)
    - [Monitoring and Logging](#monitoring-and-logging)
    - [Deployment Considerations](#deployment-considerations)
    - [Testing LangChain Applications](#testing-langchain-applications)
- [Quick Start](#quick-start)
- [Common Use Cases](#common-use-cases)

## Introduction
LangChain is a powerful framework for developing applications powered by language models. It provides a comprehensive toolkit for:
- Standardized interfaces to various language models
- Advanced prompt management and templating
- Sophisticated memory management for conversational AI
- Modular components that can be chained together
- Built-in support for common NLP tasks
- Vector store integration and document processing
- Agent-based workflows for complex tasks

## Core Concepts

### 1. Models
LangChain provides a unified interface to interact with various language models, abstracting away provider-specific implementations.

#### Model Types
1. **Chat Models**
   - Designed for conversational interactions
   - Maintain conversation context
   - Support system, user, and assistant messages
   
   ```python
   from langchain.chat_models import ChatOpenAI
   from langchain.schema import HumanMessage, SystemMessage
   
   chat = ChatOpenAI(temperature=0.7)
   messages = [
       SystemMessage(content="You are a helpful assistant that translates English to French."),
       HumanMessage(content="Translate: I love programming.")
   ]
   print(chat(messages))
   ```

2. **Text Completion Models**
   - Generate text based on a prompt
   - Useful for content generation, summarization, etc.
   
   ```python
   from langchain.llms import OpenAI
   
   llm = OpenAI(temperature=0.9)
   response = llm("Write a haiku about artificial intelligence")
   print(response)
   ```

3. **Embedding Models**
   - Convert text to vector representations
   - Essential for semantic search and similarity comparisons
   
   ```python
   from langchain.embeddings import OpenAIEmbeddings
   
   embeddings = OpenAIEmbeddings()
   text = "This is a sample text for embedding"
   vector = embeddings.embed_query(text)
   print(f"Vector length: {len(vector)}")
   ```

#### Supported Providers
- **OpenAI**: GPT-3.5, GPT-4, and embeddings
- **Anthropic**: Claude models
- **HuggingFace**: Open-source models
- **Cohere**: Advanced language understanding
- **Google**: PaLM models
- **Custom**: Bring your own models

#### Model Parameters
- `temperature`: Controls randomness (0.0 to 1.0)
- `max_tokens`: Maximum length of generated output
- `top_p`: Nucleus sampling parameter
- `frequency_penalty`: Reduce repetition
- `presence_penalty`: Encourage model to talk about new topics

#### Best Practices
1. Always set appropriate temperature based on your use case
2. Use chat models for conversational applications
3. Implement proper error handling for API calls
4. Consider rate limits and costs when selecting models
5. Cache embeddings for frequently used text

#### Real-world Example: Building a Q&A System
```python
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load and process documents
loader = TextLoader("data/faq.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

# Query the system
query = "What is your refund policy?"
result = qa.run(query)
print(result)
```

### 2. Prompts

Prompts are fundamental to working with language models. LangChain provides powerful tools for prompt management, templating, and optimization.

#### Prompt Templates
Create reusable prompt templates with dynamic variables:

```python
from langchain.prompts import PromptTemplate

# Basic template
template = "Tell me a {adjective} joke about {content}."
prompt = PromptTemplate(
    input_variables=["adjective", "content"],
    template=template,
)

formatted_prompt = prompt.format(adjective="funny", content="chickens")
print(formatted_prompt)
# Output: Tell me a funny joke about chickens.
```

#### Chat Prompt Templates
For chat models, use structured message templates:

```python
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

messages = chat_prompt.format_messages(
    input_language="English",
    output_language="French",
    text="I love programming."
)
print(messages)
```

#### Few-shot Prompting
Provide examples to guide model behavior:

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# Define examples
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

# Create example template
example_template = """Word: {word}
Antonym: {antonym}"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_template,
)

# Create few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of each input word",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"],
    example_separator="\n\n"
)

print(few_shot_prompt.format(input="big"))
```

#### Output Parsers
Convert model outputs into structured data:

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Define response schema
response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(name="source", description="source used to answer the question")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Get format instructions
format_instructions = output_parser.get_format_instructions()

# Create prompt with format instructions
prompt = PromptTemplate(
    template="Answer the user's question.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)

# Query the model
model = OpenAI(temperature=0)
prompt_value = prompt.format(question="What is the capital of France?")
output = model(prompt_value)

# Parse the output
parsed_output = output_parser.parse(output)
print(parsed_output)
# Output: {'answer': 'Paris', 'source': 'general knowledge'}
```

#### Best Practices for Prompt Engineering
1. **Be Specific**: Clearly define the task and expected output format
2. **Use Examples**: Include few-shot examples when possible
3. **Provide Context**: Give the model enough context to understand the task
4. **Iterate and Test**: Experiment with different prompt variations
5. **Handle Edge Cases**: Anticipate and handle potential model failures

#### Advanced: Dynamic Few-Shot Selection
```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# Define examples
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "What is the capital of France?", "output": "Paris"},
    {"input": "Translate 'hello' to Spanish", "output": "hola"},
]

# Create example selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=1  # Number of examples to select
)

# Create dynamic few-shot prompt
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}"
    ),
    prefix="Answer the following question",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)

# The selected examples will be dynamically chosen based on similarity to the input
print(dynamic_prompt.format(input="What's 3+3?"))
```

### 3. Memory

Memory in LangChain enables applications to maintain context and state across multiple interactions. It's essential for building conversational AI systems.

#### Types of Memory

1. **Conversation Buffer Memory**
   Stores the entire conversation history
   
   ```python
   from langchain.memory import ConversationBufferMemory
   
   memory = ConversationBufferMemory()
   memory.save_context({"input": "Hi there!"}, {"output": "Hello! How can I help?"})
   memory.save_context({"input": "What's the weather like?"}, {"output": "I'm sorry, I can't check the weather right now."})
   
   print(memory.load_memory_variables({}))
   # Output: {'history': 'Human: Hi there!\nAI: Hello! How can I help?\nHuman: What\'s the weather like?\nAI: I\'m sorry, I can\'t check the weather right now.'}
   ```

2. **Conversation Buffer Window Memory**
   Keeps a sliding window of the conversation
   
   ```python
   from langchain.memory import ConversationBufferWindowMemory
   
   memory = ConversationBufferWindowMemory(k=1)  # Keep only the last exchange
   memory.save_context({"input": "Hi there!"}, {"output": "Hello! How can I help?"})
   memory.save_context({"input": "What's the weather like?"}, {"output": "I'm sorry, I can't check the weather right now."})
   
   print(memory.load_memory_variables({}))
   # Only shows the last exchange
   ```

3. **Entity Memory**
   Remembers specific entities and facts
   
   ```python
   from langchain.memory import ConversationEntityMemory
   from langchain.llms import OpenAI
   
   llm = OpenAI(temperature=0)
   memory = ConversationEntityMemory(llm=llm)
   
   memory.save_context(
       {"input": "John is 30 years old and lives in New York"},
       {"output": "Got it, I'll remember that about John."}
   )
   
   print(memory.load_memory_variables({"input": "Where does John live?"}))
   ```

4. **Vector Store Memory**
   Stores memories in a vector database for semantic search
   
   ```python
   from langchain.memory import VectorStoreMemory
   from langchain.embeddings import OpenAIEmbeddings
   from langchain.vectorstores import FAISS
   from langchain.docstore import InMemoryDocstore
   
   embeddings = OpenAIEmbeddings()
   vectorstore = FAISS(embeddings.embed_query, None, InMemoryDocstore({}), {})
   
   memory = VectorStoreMemory(
       vectorstore=vectorstore,
       memory_key="chat_history",
       return_docs=True,
       input_key="input"
   )
   
   memory.save_context(
       {"input": "My favorite color is blue"},
       {"output": "I'll remember your favorite color is blue"}
   )
   
   # Later, retrieve similar memories
   print(memory.load_memory_variables({"input": "What's my favorite color?"}))
   ```

#### Using Memory with Chains

```python
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Create a conversation chain with memory
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# First message
response = conversation.predict(input="Hi, I'm John")
print(f"AI: {response}")

# Second message - the model remembers the context
response = conversation.predict(input="What's my name?")
print(f"AI: {response}")
```

#### Custom Memory Implementation

```python
from typing import List, Dict, Any
from langchain.schema import BaseMemory
from pydantic import BaseModel

class CustomMemory(BaseMemory, BaseModel):
    """Custom memory implementation that stores key facts."""
    memory_store: Dict[str, Any] = {}
    
    @property
    def memory_variables(self) -> List[str]:
        return ["custom_memory"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        return {"custom_memory": str(self.memory_store)}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        # Custom logic to save important information
        if "remember_this" in inputs:
            self.memory_store["important_fact"] = inputs["remember_this"]
    
    def clear(self) -> None:
        self.memory_store = {}

# Usage
memory = CustomMemory()
memory.save_context(
    {"remember_this": "User prefers dark mode"}, 
    {}
)
print(memory.load_memory_variables({}))
```

#### Best Practices for Memory Management

1. **Choose the Right Memory Type**:
   - Use `ConversationBufferMemory` for simple chat applications
   - Use `ConversationBufferWindowMemory` for long conversations
   - Use `EntityMemory` when you need to remember specific facts
   - Use `VectorStoreMemory` for semantic search over past conversations

2. **Memory Optimization**:
   - Limit memory size to prevent excessive memory usage
   - Clean up old or irrelevant memories
   - Compress or summarize long conversations when possible

3. **Security Considerations**:
   - Be cautious about storing sensitive information in memory
   - Implement proper data retention policies
   - Consider encryption for sensitive data

4. **Testing and Validation**:
   - Test memory behavior with various conversation flows
   - Validate that important information is being remembered correctly
   - Monitor memory usage in production

#### Advanced: Custom Memory with Summarization

```python
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI

class SummarizingMemory:
    def __init__(self, llm, max_tokens=1000):
        self.llm = llm
        self.max_tokens = max_tokens
        self.buffer = []
        self.summary = ""
        
    def add_message(self, role: str, content: str):
        self.buffer.append(f"{role}: {content}")
        
    def get_context(self) -> str:
        # If buffer is getting too large, summarize it
        if len("\n".join(self.buffer)) > self.max_tokens:
            self._summarize_buffer()
        return f"Summary of previous conversation:\n{self.summary}\n\nRecent messages:\n{"\n".join(self.buffer)}"
    
    def _summarize_buffer(self):
        if not self.buffer:
            return
            
        # Create documents for summarization
        docs = [Document(page_content=text) for text in self.buffer]
        
        # Use map-reduce for summarization
        chain = load_summarize_chain(
            self.llm, 
            chain_type="map_reduce",
            return_intermediate_steps=False
        )
        
        # Generate summary
        result = chain({"input_documents": docs})
        self.summary = result["output_text"]
        self.buffer = []  # Clear buffer after summarization

# Usage
llm = OpenAI(temperature=0)
memory = SummarizingMemory(llm)

# Add some messages
memory.add_message("User", "I like to go hiking on weekends")
memory.add_message("AI", "That's great! Hiking is good for health.")
memory.add_message("User", "Yes, especially in the mountains")

# Get the current context
print(memory.get_context())
```

This enhanced memory section provides a comprehensive guide to using memory in LangChain, from basic usage to advanced custom implementations.

### 4. Chains

Chains in LangChain allow you to combine multiple components together to create more complex applications. They are the building blocks for creating sophisticated workflows with language models.

#### Types of Chains

1. **LLM Chain**
   The most basic type of chain that combines a prompt template with a language model.
   
   ```python
   from langchain.prompts import PromptTemplate
   from langchain.llms import OpenAI
   from langchain.chains import LLMChain
   
   # Define the prompt template
   template = """You are a naming consultant for new companies.
   What is a good name for a {company_type} company that makes {product}?"""
   
   prompt = PromptTemplate(
       input_variables=["company_type", "product"],
       template=template,
   )
   
   # Create the chain
   llm = OpenAI(temperature=0.9)
   chain = LLMChain(llm=llm, prompt=prompt)
   
   # Run the chain
   result = chain.run({"company_type": "tech", "product": "AI-powered coffee makers"})
   print(result)
   ```

2. **Sequential Chains**
   Combine multiple chains where the output of one chain is the input to the next.
   
   ```python
   from langchain.chains import SimpleSequentialChain, LLMChain
   from langchain.llms import OpenAI
   from langchain.prompts import PromptTemplate
   
   # First chain: Generate company name
   name_template = """You are a naming consultant for new companies.
   What is a good name for a {company_type} company that makes {product}?"""
   
   name_prompt = PromptTemplate(
       input_variables=["company_type", "product"],
       template=name_template,
   )
   
   # Second chain: Generate a slogan
   slogan_template = "Write a catchy slogan for a company called {company_name}"
   slogan_prompt = PromptTemplate(
       input_variables=["company_name"],
       template=slogan_template,
   )
   
   llm = OpenAI(temperature=0.9)
   name_chain = LLMChain(llm=llm, prompt=name_prompt)
   slogan_chain = LLMChain(llm=llm, prompt=slogan_prompt)
   
   # Combine the chains
   overall_chain = SimpleSequentialChain(
       chains=[name_chain, slogan_chain],
       verbose=True
   )
   
   # Run the chain
   result = overall_chain.run({"company_type": "tech", "product": "AI-powered coffee makers"})
   print(result)
   ```

3. **Router Chains**
   Route inputs to different chains based on the input.
   
   ```python
   from langchain.chains.router import MultiPromptChain
   from langchain.llms import OpenAI
   from langchain.chains import ConversationChain
   from langchain.chains.llm import LLMChain
   from langchain.prompts import PromptTemplate
   
   # Define the prompt templates
   physics_template = """You are a very smart physics professor. \
   You are great at answering questions about physics in a concise and easy to understand manner. \
   Here's a question:
   {input}"""
   
   math_template = """You are a very good mathematician. \
   You are great at answering math questions. \
   You are so good because you are able to break down \
   hard problems into their component parts, answer the component parts, and then put them together\
   to answer the broader question.
   Here's a question:
   {input}"""
   
   # Create prompt info
   prompt_infos = [
       {
           "name": "physics",
           "description": "Good for answering questions about physics",
           "prompt_template": physics_template
       },
       {
           "name": "math",
           "description": "Good for answering math questions",
           "prompt_template": math_template
       }
   ]
   
   llm = OpenAI()
   destination_chains = {}
   
   for p_info in prompt_infos:
       prompt = PromptTemplate(
           template=p_info["prompt_template"],
           input_variables=["input"]
       )
       chain = LLMChain(llm=llm, prompt=prompt)
       destination_chains[p_info["name"]] = chain
   
   # Default chain
   default_chain = ConversationChain(llm=llm, output_key="text")
   
   # Create the router chain
   router_chain = MultiPromptChain(
       router_chain=router_chain,
       destination_chains=destination_chains,
       default_chain=default_chain,
       verbose=True
   )
   
   # Test the router
   print(router_chain.run("What is the speed of light?"))
   print(router_chain.run("What is 42 * 7?"))
   print(router_chain.run("Tell me a joke"))  # Will use default chain
   ```

4. **Transform Chain**
   Apply a transformation to the input/output.
   
   ```python
   from langchain.chains import TransformChain, SequentialChain
   import json
   
   # Define a transformation function
   def transform_func(inputs: dict) -> dict:
       text = inputs["text"]
       return {"output_text": text.upper()}
   
   # Create the transform chain
   transform_chain = TransformChain(
       input_variables=["text"],
       output_variables=["output_text"],
       transform=transform_func
   )
   
   # Use with another chain
   template = """Reverse this text:
   {output_text}"""
   
   prompt = PromptTemplate(
       input_variables=["output_text"],
       template=template
   )
   
   llm = OpenAI()
   llm_chain = LLMChain(llm=llm, prompt=prompt, output_key="final_output")
   
   # Combine the chains
   chain = SequentialChain(
       chains=[transform_chain, llm_chain],
       input_variables=["text"],
       output_variables=["final_output"]
   )
   
   print(chain({"text": "hello world"}))
   ```

#### Custom Chain Implementation

```python
from typing import Dict, List, Any
from langchain.chains.base import Chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class CustomChain(Chain):
    """Custom chain that generates a poem about a topic."""
    
    llm: Any  # Language model
    prompt: PromptTemplate  # Prompt template
    output_key: str = "poem"  # Output key
    
    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""
        return self.prompt.input_variables
    
    @property
    def output_keys(self) -> List[str]:
        """Output keys this chain returns."""
        return [self.output_key]
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Execute the chain."""
        prompt_value = self.prompt.format(**inputs)
        response = self.llm(prompt_value)
        return {self.output_key: response}

# Usage
llm = OpenAI(temperature=0.9)

template = """Write a short poem about {topic}.
Make it {style} and {tone}."""

prompt = PromptTemplate(
    input_variables=["topic", "style", "tone"],
    template=template
)

poem_chain = CustomChain(
    llm=llm,
    prompt=prompt,
    output_key="poem"
)

result = poem_chain({
    "topic": "artificial intelligence",
    "style": "haiku",
    "tone": "whimsical"
})

print(result["poem"])
```

#### Best Practices for Working with Chains

1. **Chain Composition**
   - Break down complex tasks into smaller, reusable chains
   - Use `SequentialChain` to combine multiple chains
   - Keep individual chains focused on a single responsibility

2. **Error Handling**
   - Implement proper error handling for API calls
   - Add validation for chain inputs and outputs
   - Use try/except blocks to handle potential failures

3. **Performance Optimization**
   - Cache expensive operations when possible
   - Use batch processing for multiple inputs
   - Consider parallel execution for independent chains

4. **Testing and Debugging**
   - Test each chain component in isolation
   - Use verbose mode for debugging
   - Log intermediate results for complex chains

#### Advanced: Dynamic Chain Creation

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

def create_dynamic_chain(template: str, input_vars: list, output_key: str, **kwargs):
    """Dynamically create a chain based on the given parameters."""
    prompt = PromptTemplate(
        input_variables=input_vars,
        template=template
    )
    
    llm = OpenAI(**kwargs)
    return LLMChain(
        llm=llm,
        prompt=prompt,
        output_key=output_key
    )

# Example usage
qa_template = """Answer the following question:
{question}

Context: {context}"""

qa_chain = create_dynamic_chain(
    template=qa_template,
    input_vars=["question", "context"],
    output_key="answer",
    temperature=0.7
)

result = qa_chain({
    "question": "What is the capital of France?",
    "context": "France is a country in Europe. Its capital is Paris."
})

print(result["answer"])
```

This enhanced chains section provides a comprehensive guide to building and using chains in LangChain, from basic usage to advanced custom implementations.

### 5. Indexes

Indexes in LangChain help you structure your documents for efficient retrieval and use with language models. This section covers document loaders, text splitters, and vector stores.

#### Document Loaders

LangChain provides various document loaders to load data from different sources:

```python
# Load from a text file
from langchain.document_loaders import TextLoader
loader = TextLoader("path/to/file.txt")
documents = loader.load()

# Load from a PDF
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("example.pdf")
pages = loader.load_and_split()

# Load from a webpage
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com")
data = loader.load()

# Load from a directory
from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader('./data', glob='**/*.txt')
documents = loader.load()
```

#### Text Splitters

Split documents into smaller chunks for processing:

```python
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)

# Basic character-based splitting
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
texts = text_splitter.split_documents(documents)

# More sophisticated recursive splitting
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)
chunks = recursive_splitter.split_documents(documents)

# Token-based splitting (useful for models with token limits)
token_splitter = TokenTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
token_chunks = token_splitter.split_documents(documents)
```

#### Vector Stores

Store and retrieve document embeddings efficiently:

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma, Pinecone
from langchain.text_splitter import CharacterTextSplitter

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Example with FAISS (local in-memory)
faiss_db = FAISS.from_documents(texts, embeddings)

# Example with Chroma (persistent storage)
chroma_db = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
chroma_db.persist()

# Example with Pinecone (cloud-based)
import pinecone

pinecone.init(api_key="your-api-key", environment="your-env")
index_name = "langchain-demo"

# Create a new index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  # OpenAI embeddings dimension
    )

pinecone_db = Pinecone.from_documents(
    texts, 
    embeddings, 
    index_name=index_name
)
```

#### Document Retrievers

Retrieve relevant documents based on queries:

```python
# Simple similarity search
query = "What is LangChain?"
docs = faiss_db.similarity_search(query)

# Similarity search with score
docs_with_scores = faiss_db.similarity_search_with_score(query)

# Maximum marginal relevance search (diverse results)
diverse_docs = faiss_db.max_marginal_relevance_search(
    query,
    k=3,  # number of documents to return
    fetch_k=10  # number of documents to fetch before filtering
)

# Using a retriever
retriever = faiss_db.as_retriever(
    search_type="similarity",  # or "mmr" or "similarity_score_threshold"
    search_kwargs={"k": 5}
)
relevant_docs = retriever.get_relevant_documents(query)
```

#### Real-world Example: Document Q&A System

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 1. Load and process documents
loader = TextLoader("state_of_the_union.txt")
documents = loader.load()

# 2. Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# 3. Create vector store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

# 4. Create retriever
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

# 5. Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 6. Query the system
result = qa({"query": "What did the president say about the Supreme Court?"})
print(result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']}, page {doc.metadata.get('page', 'N/A')}")
```

#### Best Practices for Working with Indexes

1. **Document Processing**
   - Clean and preprocess text before indexing
   - Remove irrelevant content (headers, footers, etc.)
   - Add metadata for better filtering and organization

2. **Chunking Strategies**
   - Choose appropriate chunk size based on your use case
   - Use overlapping chunks to maintain context
   - Consider document structure when splitting (e.g., split by sections)

3. **Vector Store Selection**
   - **FAISS**: Fast and efficient for small to medium datasets
   - **Chroma**: Good for local development and testing
   - **Pinecone/Weaviate**: Scalable cloud solutions for production
   - **Milvus/Weaviate**: Enterprise-grade vector databases

4. **Performance Optimization**
   - Batch process large document collections
   - Cache embeddings for frequently accessed documents
   - Use approximate nearest neighbor search for large datasets
   - Consider dimensionality reduction for high-dimensional embeddings

5. **Metadata Management**
   - Add relevant metadata to documents
   - Use metadata for filtering and organization
   - Include source information for attribution

#### Advanced: Custom Document Processing Pipeline

```python
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import re

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters (keep alphanumeric and basic punctuation)
        text = re.sub(r'[^\w\s.,;:!?\-]', '', text)
        return text
    
    def process_document(self, file_path: str) -> List[Document]:
        """Process a single document."""
        # Load document
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Clean and process each document
        processed_docs = []
        for doc in documents:
            # Clean text
            clean_content = self.clean_text(doc.page_content)
            
            # Add metadata
            metadata = doc.metadata.copy()
            metadata["source_file"] = file_path
            
            # Create new document with cleaned content
            processed_doc = Document(
                page_content=clean_content,
                metadata=metadata
            )
            processed_docs.append(processed_doc)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(processed_docs)
        return chunks
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create a vector store from processed documents."""
        return FAISS.from_documents(documents, self.embeddings)
    
    def process_directory(self, directory_path: str) -> FAISS:
        """Process all documents in a directory."""
        from pathlib import Path
        
        all_chunks = []
        
        # Process all .txt files in the directory
        for file_path in Path(directory_path).glob("**/*.txt"):
            try:
                chunks = self.process_document(str(file_path))
                all_chunks.extend(chunks)
                print(f"Processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        # Create and return vector store
        return self.create_vector_store(all_chunks)

# Usage
processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
vector_store = processor.process_directory("./documents")

# Save the vector store for later use
vector_store.save_local("my_vector_store")
```

This indexes section provides a comprehensive guide to working with documents in LangChain, from loading and processing to efficient retrieval using vector stores.

### 6. Agents

Agents in LangChain are systems that use a language model to determine a sequence of actions to take. They can use tools, observe the results, and make decisions about what to do next.

#### Agent Types

1. **Zero-shot ReAct Agent**
   Uses the ReAct framework to decide which tool to use based on the tool's description.
   
   ```python
   from langchain.agents import load_tools, initialize_agent, AgentType
   from langchain.llms import OpenAI
   
   # Initialize the language model
   llm = OpenAI(temperature=0)
   
   # Load some tools
   tools = load_tools(["serpapi", "llm-math"], llm=llm)
   
   # Initialize the agent
   agent = initialize_agent(
       tools, 
       llm, 
       agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
       verbose=True
   )
   
   # Run the agent
   agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
   ```

2. **Plan-and-Execute Agent**
   First plans what to do, then executes the sub-tasks.
   
   ```python
   from langchain.agents import load_tools, initialize_agent, AgentType
   from langchain.llms import OpenAI
   
   llm = OpenAI(temperature=0)
   tools = load_tools(["serpapi", "llm-math"], llm=llm)
   
   agent = initialize_agent(
       tools, 
       llm, 
       agent=AgentType.PLAN_AND_EXECUTE,
       verbose=True
   )
   
   agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
   ```

3. **Self-ask with Search**
   Uses a single tool to search for answers to follow-up questions.
   
   ```python
   from langchain.agents import load_tools, initialize_agent, AgentType
   from langchain.llms import OpenAI
   
   llm = OpenAI(temperature=0)
   tools = load_tools(["self-ask-with-search"], llm=llm)
   
   agent = initialize_agent(
       tools, 
       llm, 
       agent=AgentType.SELF_ASK_WITH_SEARCH,
       verbose=True
   )
   
   agent.run("What is the hometown of the reigning men's U.S. Open champion?")
   ```

#### Custom Tools

Create your own tools for the agent to use:

```python
from langchain.tools import BaseTool
from math import pi
from typing import Union

class CircumferenceTool(BaseTool):
    name = "circumference_calculator"
    description = "Use this tool when you need to calculate a circumference using the radius of a circle."
    
    def _run(self, radius: Union[int, float]) -> float:
        return float(radius) * 2.0 * pi
    
    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")

# Initialize the agent with custom tool
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
agent = initialize_agent(
    [CircumferenceTool()], 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("What is the circumference of a circle with a radius of 7.5?")
```

#### Toolkits

Group related tools together in toolkits:

```python
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.llms import OpenAI

agent = create_python_agent(
    llm=OpenAI(temperature=0, max_tokens=1000),
    tool=PythonREPLTool(),
    verbose=True
)

agent.run("What is the 10th fibonacci number?")
```

#### Multi-Agent Systems

Create multiple agents that can work together:

```python
from langchain.agents import Tool
from langchain.agents import AgentExecutor, create_sql_agent
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.llms import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper

# Create a search tool
search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]

# Create a research agent
research_agent = initialize_agent(
    tools, 
    OpenAI(temperature=0), 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Create a writing agent
writing_agent = initialize_agent(
    [],
    OpenAI(temperature=0.7),
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True
)

# Simulate a conversation between agents
research_result = research_agent.run("What are the latest developments in AI?")
writing_prompt = f"Write a short article about the following AI developments: {research_result}"
article = writing_agent.run(writing_prompt)
print(article)
```

#### Best Practices for Working with Agents

1. **Tool Design**
   - Make tool descriptions clear and specific
   - Handle errors gracefully in your tools
   - Include type hints and documentation

2. **Agent Configuration**
   - Choose the right agent type for your task
   - Set appropriate temperature (lower for more focused tasks)
   - Limit the number of steps to prevent excessive API usage

3. **Error Handling**
   - Implement proper error handling in tools
   - Add validation for tool inputs
   - Use try/except blocks to handle potential failures

4. **Performance Optimization**
   - Cache tool results when possible
   - Use streaming for long-running operations
   - Monitor and log agent performance

#### Advanced: Custom Agent with Memory

```python
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain import OpenAI, SerpAPIWrapper
from langchain.memory import ConversationBufferMemory
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish

class CustomAgent(BaseSingleActionAgent):
    @property
    def input_keys(self):
        return ["input"]
    
    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        # Implement your custom logic here
        if len(intermediate_steps) == 0:
            # First step
            return AgentAction(
                tool="Search",
                tool_input={"query": kwargs["input"]},
                log=""
            )
        else:
            # We've already taken the first step, so we're done
            return AgentFinish(
                return_values={"output": intermediate_steps[0][1]},
                log=intermediate_steps[0][1]
            )
    
    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        raise NotImplementedError("Async not implemented")

# Set up the tools
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for searching the web"
    )
]

# Initialize the agent
agent = CustomAgent()
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history")
)

# Run the agent
agent_executor.run("What's the weather in San Francisco?")
```

This agents section provides a comprehensive guide to building intelligent, decision-making applications with LangChain, from basic usage to advanced custom implementations.

### 7. Advanced Topics

This section covers advanced concepts and techniques for working with LangChain in production environments.

#### Custom Components

1. **Custom LLM Wrapper**
   Create a wrapper for any LLM that follows the LangChain interface:

   ```python
   from typing import Any, List, Mapping, Optional
   from langchain.llms.base import LLM
   from langchain.callbacks.manager import CallbackManagerForLLMRun

   class CustomLLM(LLM):
       """A custom chat model that echoes the first `n` characters of input."""
       n: int
       
       @property
       def _llm_type(self) -> str:
           return "echoing-chat-model"
       
       def _call(
           self,
           prompt: str,
           stop: Optional[List[str]] = None,
           run_manager: Optional[CallbackManagerForLLMRun] = None,
           **kwargs: Any,
       ) -> str:
           if stop is not None:
               raise ValueError("stop kwargs are not permitted.")
           return prompt[:self.n]
       
       @property
       def _identifying_params(self) -> Mapping[str, Any]:
           """Get the identifying parameters."""
           return {"n": self.n}

   # Usage
   llm = CustomLLM(n=10)
   print(llm("This is a test"))  # Output: "This is a "
   ```

2. **Custom Memory Backend**
   Implement a custom memory store using Redis:

   ```python
   import json
   from typing import Dict, List
   import redis
   from langchain.schema import BaseChatMessageHistory
   from langchain.memory import ConversationBufferMemory
   from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict

   class RedisChatMessageHistory(BaseChatMessageHistory):
       """Chat message history stored in a Redis database."""
       
       def __init__(self, session_id: str, url: str = "redis://localhost:6379/0"):
           self.redis_client = redis.Redis.from_url(url=url)
           self.session_id = session_id
           self.key = f"chat_messages:{session_id}"
       
       @property
       def messages(self) -> List[BaseMessage]:
           """Retrieve all messages from Redis."""
           items = self.redis_client.lrange(self.key, 0, -1)
           messages = [json.loads(m.decode('utf-8')) for m in items]
           return messages_from_dict(messages)
       
       def add_message(self, message: BaseMessage) -> None:
           """Add a message to the Redis store."""
           self.redis_client.rpush(self.key, json.dumps(_message_to_dict(message)))
       
       def clear(self) -> None:
           """Clear all messages from the store."""
           self.redis_client.delete(self.key)
   
   # Usage
   memory = ConversationBufferMemory(
       chat_memory=RedisChatMessageHistory(session_id="test_session"),
       return_messages=True
   )
   memory.save_context({"input": "Hi"}, {"output": "Hello!"})
   print(memory.load_memory_variables({}))
   ```

#### Performance Optimization

1. **Caching**
   Implement caching for LLM calls:

   ```python
   from langchain.cache import InMemoryCache
   from langchain.globals import set_llm_cache
   from langchain.llms import OpenAI
   import langchain
   
   # Enable caching
   set_llm_cache(InMemoryCache())
   
   # First call - will make an actual API call
   llm = OpenAI()
   print(llm("Tell me a joke"))  # Makes API call
   
   # Second call with same input - returns cached result
   print(llm("Tell me a joke"))  # Returns from cache
   ```

2. **Batch Processing**
   Process multiple inputs in parallel:

   ```python
   from langchain.llms import OpenAI
   from langchain.callbacks import get_openai_callback
   import time
   
   llm = OpenAI()
   
   # Without batching
   start = time.time()
   for i in range(5):
       print(llm(f"Say hello {i}"))
   print(f"Time without batching: {time.time() - start:.2f}s")
   
   # With batching
   start = time.time()
   inputs = [f"Say hello {i}" for i in range(5)]
   results = llm.batch(inputs)
   for result in results:
       print(result)
   print(f"Time with batching: {time.time() - start:.2f}s")
   ```

3. **Token Usage Tracking**
   Monitor token usage and costs:

   ```python
   from langchain.callbacks import get_openai_callback
   from langchain.llms import OpenAI
   
   llm = OpenAI()
   
   with get_openai_callback() as cb:
       result = llm("Tell me a joke")
       print(result)
       print(f"Tokens used: {cb.total_tokens}")
       print(f"Estimated cost: ${cb.total_cost:.6f}")
   ```

#### Error Handling and Retries

```python
from langchain.llms import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import openai

# Configure retry logic
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (openai.error.APIError, openai.error.Timeout, openai.error.ServiceUnavailableError)
    ),
)
def reliable_llm_call(prompt: str) -> str:
    llm = OpenAI(temperature=0.7)
    return llm(prompt)

try:
    response = reliable_llm_call("Write a short poem about AI")
    print(response)
except Exception as e:
    print(f"Failed after retries: {str(e)}")
```

#### Security Best Practices

1. **API Key Management**
   ```python
   import os
   from dotenv import load_dotenv
   from langchain.llms import OpenAI
   
   # Load environment variables from .env file
   load_dotenv()
   
   # Access API key from environment variable
   llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
   ```

2. **Input Validation**
   ```python
   from pydantic import BaseModel, validator
   from typing import List
   
   class QueryRequest(BaseModel):
       question: str
       context: str = ""
       max_tokens: int = 100
       
       @validator('question')
       def question_not_empty(cls, v):
           if not v.strip():
               raise ValueError('Question cannot be empty')
           return v.strip()
       
       @validator('max_tokens')
       def max_tokens_range(cls, v):
           if not 1 <= v <= 1000:
               raise ValueError('max_tokens must be between 1 and 1000')
           return v
   
   # Usage
   try:
       request = QueryRequest(question="  ", max_tokens=2000)  # Will raise ValidationError
   except Exception as e:
       print(f"Validation error: {e}")
   ```

#### Monitoring and Logging

```python
import logging
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("langchain_app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_llm_interaction(prompt: str, response: str, metadata: dict = None):
    """Log LLM interactions with metadata."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "response": response,
        "metadata": metadata or {}
    }
    logger.info(f"LLM Interaction: {log_entry}")

# Example usage
llm = OpenAI()
with get_openai_callback() as cb:
    prompt = "Tell me about LangChain"
    response = llm(prompt)
    
    # Log the interaction
    log_llm_interaction(
        prompt=prompt,
        response=response,
        metadata={
            "model": "text-davinci-003",
            "tokens_used": cb.total_tokens,
            "cost": f"${cb.total_cost:.6f}"
        }
    )
```

#### Deployment Considerations

1. **Containerization**
   Example Dockerfile for a LangChain application:
   
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       python3-dev \
       && rm -rf /var/lib/apt/lists/*
   
   # Install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application code
   COPY . .
   
   # Set environment variables
   ENV PYTHONUNBUFFERED=1
   ENV PORT=8000
   
   # Run the application
   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Scaling**
   - Use async/await for I/O-bound operations
   - Implement rate limiting
   - Use a task queue (Celery, RQ) for long-running tasks
   - Consider using a distributed cache (Redis, Memcached)

#### Testing LangChain Applications

```python
import unittest
from unittest.mock import MagicMock, patch
from langchain.llms import OpenAI
from your_app import process_with_llm

class TestLangChainApp(unittest.TestCase):
    
    @patch('langchain.llms.OpenAI')
    def test_process_with_llm(self, mock_llm):
        # Setup mock
        mock_llm.return_value = MagicMock()
        mock_llm.return_value.return_value = "Mocked response"
        
        # Test the function
        result = process_with_llm("test input")
        
        # Assertions
        self.assertEqual(result, "Mocked response")
        mock_llm.return_value.assert_called_once_with("test input")

if __name__ == "__main__":
    unittest.main()
```

This advanced topics section provides in-depth knowledge for building production-ready LangChain applications, covering custom components, performance optimization, security, and deployment.

## Quick Start

### Python

Here's a simple example of using LangChain to create a question-answering application:

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize the language model
llm = OpenAI(temperature=0.7)

# Create a prompt template
template = """Question: {question}

Answer: Let me think step by step.
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create a chain
qa_chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
question = "What is the capital of France?"
result = qa_chain.run(question=question)
print(result)
```

### TypeScript

Here's the same example using the LangChain TypeScript SDK:

```typescript
import { OpenAI } from "langchain/llms/openai";
import { LLMChain } from "langchain/chains";
import { PromptTemplate } from "langchain/prompts";

// Initialize the language model
const model = new OpenAI({
  temperature: 0.7,
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// Create a prompt template
const template = `Question: {question}

Answer: Let me think step by step.`;

const prompt = new PromptTemplate({
  template,
  inputVariables: ["question"],
});

// Create a chain
const qaChain = new LLMChain({
  llm: model,
  prompt,
});

// Run the chain
const question = "What is the capital of France?";
const result = await qaChain.call({ question });
console.log(result.text);
```

### Installation

#### Python
```bash
pip install langchain openai
```

#### TypeScript
```bash
npm install langchain @langchain/openai
```

#### Environment Setup

For both Python and TypeScript, you'll need to set up your OpenAI API key:

```bash
# For Python
export OPENAI_API_KEY='your-api-key-here'

# For TypeScript (add to .env file)
OPENAI_API_KEY=your-api-key-here
```

## Common Use Cases
- Question answering
- Text summarization
- Code generation
- Chatbots
- Data extraction
