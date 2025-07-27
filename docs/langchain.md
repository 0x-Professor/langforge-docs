# LangChain Documentation

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

## Quick Start

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Initialize the language model
llm = OpenAI(temperature=0.9)

# Create a prompt template
template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
)

# Run the chain
print(prompt.format(product="colorful socks"))
# Output: What is a good name for a company that makes colorful socks?

# Generate completion
print(llm(prompt.format(product="colorful socks")))
```

## Common Use Cases
- Question answering
- Text summarization
- Code generation
- Chatbots
- Data extraction
