# Advanced Usage Examples

This directory contains advanced usage examples for the LangChain ecosystem, covering more complex scenarios and integrations.

## Table of Contents

1. [Agents](#agents)
2. [Custom Tools](#custom-tools)
3. [Document Loaders](#document-loaders)
4. [Vector Stores](#vector-stores)
5. [Chains with Memory](#chains-with-memory)
6. [Custom Callbacks](#custom-callbacks)

## Agents

### ReAct Agent with Tools

```python
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain import OpenAI, LLMChain
from langchain.utilities import SerpAPIWrapper

# Define tools
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for answering questions about current events"
    )
]

# Set up the agent
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input", "agent_scratchpad"],
    partial_variables={"tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools])},
)

llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    verbose=True
)

agent_executor.run("What's the latest news about AI?")
```

## Custom Tools

### Creating a Custom Tool

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

class CustomCalculatorTool(BaseTool):
    name = "calculator"
    description = "Useful for when you need to perform mathematical calculations"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, a: int, b: int) -> str:
        """Use the tool."""
        return {
            'addition': a + b,
            'subtraction': a - b,
            'multiplication': a * b,
            'division': a / b if b != 0 else 'undefined'
        }
    
    async def _arun(self, a: int, b: int) -> str:
        """Use the tool asynchronously."""
        return self._run(a, b)

# Usage
tool = CustomCalculatorTool()
print(tool.run({"a": 10, "b": 5}))
```

## Document Loaders

### Loading and Processing Documents

```python
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load a text file
loader = TextLoader("state_of_the_union.txt")
documents = loader.load()

# Or load a PDF
# loader = PyPDFLoader("example.pdf")
# documents = loader.load_and_split()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)
print(f"Split into {len(splits)} chunks")
```

## Vector Stores

### Storing and Querying Embeddings

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load and process the text
loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create embeddings and store in FAISS
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

# Save and load the index
db.save_local("faiss_index")
# db = FAISS.load_local("faiss_index", embeddings)

# Query the index
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

## Chains with Memory

### Conversation Chain with Memory

```python
from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationBufferWindowMemory

llm = OpenAI(temperature=0)
memory = ConversationBufferWindowMemory(k=2)  # Remember last 2 exchanges
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hi there! I'm Alice.")
conversation.predict(input="I'm looking for a good Italian restaurant.")
conversation.predict(input="What's my name?")
```

## Custom Callbacks

### Implementing a Custom Callback Handler

```python
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        print(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print(f"LLM finished with response: {response}")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        print(f"LLM error: {error}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        print(f"Chain started with inputs: {inputs}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        print(f"Tool started with input: {input_str}")

# Usage
from langchain.llms import OpenAI
from langchain.callbacks import get_callback_manager

handler = MyCustomHandler()
llm = OpenAI(
    temperature=0,
    callback_manager=get_callback_manager().set_handler(handler)
)
llm("Tell me a joke")
```

---

For more basic examples, see the [Basic Usage](../basic-usage) directory.
