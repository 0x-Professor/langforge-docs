# Tools

## Overview

Tools in LangChain are functions that agents can use to interact with the world. They can be anything from simple utility functions to complex APIs. Tools allow agents to perform actions beyond just generating text.

> **Tip:** Tools are a powerful way to extend the capabilities of your LangChain applications. They enable agents to perform actions like web searches, API calls, database queries, and more.

## Built-in Tools

LangChain comes with several built-in tools that you can use right away:

### Search Tools
- **GoogleSearchAPIWrapper**: Web search using Google
- **DuckDuckGoSearchRun**: Web search using DuckDuckGo  
- **WikipediaQueryRun**: Search Wikipedia

### Data Tools
- **PythonREPLTool**: Execute Python code
- **SQLDatabaseToolkit**: Query SQL databases
- **PandasTools**: Work with pandas DataFrames

### File Tools
- **FileSearchTool**: Search for files
- **ReadFileTool**: Read file contents
- **WriteFileTool**: Write to files

### Web Tools
- **RequestsGetTool**: Make HTTP GET requests
- **RequestsPostTool**: Make HTTP POST requests

## Basic Tool Usage

```python
from langchain.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Create a search tool
search = DuckDuckGoSearchAPIWrapper()
search_tool = Tool(
    name="Search",
    description="Search DuckDuckGo for recent results.",
    func=search.run
)

# Use the tool
result = search_tool.run("latest news about artificial intelligence")
print(result[:500])  # Print first 500 characters
```

## Custom Tools

### Using the @tool Decorator

The simplest way to create custom tools:

```python
from langchain.tools import tool

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

# Use the tools
length = get_word_length.invoke({"word": "hello"})
product = multiply_numbers.invoke({"a": 5, "b": 3})

print(f"Length: {length}")
print(f"Product: {product}")
```

### Advanced Tool with Input Validation

```python
from langchain.tools import tool
from pydantic import Field

@tool
def calculate_area(
    length: float = Field(description="Length of the rectangle"),
    width: float = Field(description="Width of the rectangle")
) -> float:
    """Calculate the area of a rectangle."""
    if length <= 0 or width <= 0:
        raise ValueError("Length and width must be positive numbers")
    return length * width

# Use with validation
area = calculate_area.invoke({"length": 5.0, "width": 3.0})
print(f"Area: {area}")
```

### Creating Tools with BaseTool Class

For more complex tools, subclass BaseTool:

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import requests
import json

class WeatherInput(BaseModel):
    location: str = Field(description="City name, e.g., 'London' or 'New York'")

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Get the current weather for a location"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, location: str) -> str:
        """Get weather information for a location."""
        try:
            # Note: Replace with actual weather API
            # This is a mock implementation
            mock_weather_data = {
                "London": "Cloudy, 15°C",
                "New York": "Sunny, 22°C",
                "Tokyo": "Rainy, 18°C"
            }
            
            weather = mock_weather_data.get(location, "Weather data not available")
            return f"Weather in {location}: {weather}"
        
        except Exception as e:
            return f"Error getting weather for {location}: {str(e)}"
    
    async def _arun(self, location: str) -> str:
        """Async version of the tool."""
        return self._run(location)

# Use the custom tool
weather_tool = WeatherTool()
result = weather_tool.invoke({"location": "London"})
print(result)
```

### Calculator Tool with Error Handling

```python
from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import ast
import operator

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate, e.g., '2 + 2' or '10 * 5'")

class SafeCalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform safe mathematical calculations"
    args_schema: Type[BaseModel] = CalculatorInput
    
    # Supported operations
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    
    def _evaluate_node(self, node):
        """Safely evaluate an AST node."""
        if isinstance(node, ast.Constant):  # Numbers
            return node.value
        elif isinstance(node, ast.BinOp):  # Binary operations
            left = self._evaluate_node(node.left)
            right = self._evaluate_node(node.right)
            return self.operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            operand = self._evaluate_node(node.operand)
            return self.operators[type(node.op)](operand)
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
    
    def _run(self, expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        try:
            # Parse the expression into an AST
            node = ast.parse(expression, mode='eval')
            
            # Evaluate the AST
            result = self._evaluate_node(node.body)
            
            return str(result)
        
        except (ValueError, TypeError, ZeroDivisionError) as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Invalid expression: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        """Async version of the tool."""
        return self._run(expression)

# Use the safe calculator
calc = SafeCalculatorTool()
print(calc.invoke({"expression": "2 + 3 * 4"}))  # Output: 14
print(calc.invoke({"expression": "10 / 2"}))     # Output: 5.0
print(calc.invoke({"expression": "2 ** 3"}))     # Output: 8
```

## Async Tools

For I/O-bound operations, create async tools for better performance:

```python
import asyncio
import aiohttp
from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class URLContentInput(BaseModel):
    url: str = Field(description="URL to fetch content from")

class AsyncWebScraperTool(BaseTool):
    name = "web_scraper"
    description = "Fetch and return the content of a web page"
    args_schema: Type[BaseModel] = URLContentInput
    
    def _run(self, url: str) -> str:
        """Synchronous version that calls the async version."""
        return asyncio.run(self._arun(url))
    
    async def _arun(self, url: str) -> str:
        """Async implementation that fetches web content."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Return first 1000 characters
                        return content[:1000] + "..." if len(content) > 1000 else content
                    else:
                        return f"Error: HTTP {response.status}"
        
        except asyncio.TimeoutError:
            return "Error: Request timed out"
        except Exception as e:
            return f"Error fetching URL: {str(e)}"

# Use the async tool
scraper = AsyncWebScraperTool()
# Note: This would require aiohttp to be installed
# result = scraper.invoke({"url": "https://example.com"})
```

## Tool Integration with Agents

### Using Tools with OpenAI Functions

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

@tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool  
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

# Create the agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [get_current_time, add_numbers]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
result = agent_executor.invoke({
    "input": "What time is it and what is 15 + 27?"
})
print(result["output"])
```

### Tool Error Handling and Retry Logic

```python
from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import time
import random

class APICallInput(BaseModel):
    endpoint: str = Field(description="API endpoint to call")

class RobustAPITool(BaseTool):
    name = "api_caller"
    description = "Make API calls with retry logic"
    args_schema: Type[BaseModel] = APICallInput
    max_retries: int = 3
    base_delay: float = 1.0
    
    def _run(self, endpoint: str) -> str:
        """Make API call with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                # Simulate API call that might fail
                if random.random() < 0.3:  # 30% chance of failure
                    raise Exception("API temporarily unavailable")
                
                # Simulate successful API response
                return f"Success: Data from {endpoint}"
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"Failed after {self.max_retries} attempts: {str(e)}"
                
                # Exponential backoff
                delay = self.base_delay * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
                time.sleep(delay)
        
        return "Unexpected error"
    
    async def _arun(self, endpoint: str) -> str:
        """Async version with the same retry logic."""
        return self._run(endpoint)

# Use the robust tool
api_tool = RobustAPITool()
result = api_tool.invoke({"endpoint": "/api/data"})
print(result)
```

## Tool Composition and Pipelines

### Chaining Tools Together

```python
from langchain.tools import tool
from langchain_core.runnables import RunnableLambda

@tool
def extract_numbers(text: str) -> str:
    """Extract numbers from text."""
    import re
    numbers = re.findall(r'\d+', text)
    return ','.join(numbers)

@tool
def sum_numbers(numbers_str: str) -> float:
    """Sum comma-separated numbers."""
    try:
        numbers = [float(n.strip()) for n in numbers_str.split(',') if n.strip()]
        return sum(numbers)
    except ValueError:
        return 0.0

# Create a pipeline
def process_text_for_sum(text: str) -> float:
    """Extract numbers from text and sum them."""
    numbers_str = extract_numbers.invoke({"text": text})
    total = sum_numbers.invoke({"numbers_str": numbers_str})
    return total

# Use the pipeline
text = "I have 10 apples, 5 oranges, and 3 bananas"
total = process_text_for_sum(text)
print(f"Total items: {total}")  # Output: 18.0
```

### Tool with Caching

```python
from langchain.tools import BaseTool
from typing import Type, Dict, Any
from pydantic import BaseModel, Field
import time
from functools import lru_cache

class CacheInput(BaseModel):
    key: str = Field(description="Cache key to lookup")

class CachedTool(BaseTool):
    name = "cached_expensive_operation"
    description = "Perform an expensive operation with caching"
    args_schema: Type[BaseModel] = CacheInput
    
    @lru_cache(maxsize=100)
    def _expensive_operation(self, key: str) -> str:
        """Simulate an expensive operation."""
        print(f"Performing expensive operation for: {key}")
        time.sleep(2)  # Simulate delay
        return f"Result for {key}: {hash(key) % 1000}"
    
    def _run(self, key: str) -> str:
        """Run the cached operation."""
        return self._expensive_operation(key)
    
    async def _arun(self, key: str) -> str:
        """Async version."""
        return self._run(key)

# Use the cached tool
cached_tool = CachedTool()

# First call - takes 2 seconds
start = time.time()
result1 = cached_tool.invoke({"key": "test123"})
print(f"First call took {time.time() - start:.2f}s: {result1}")

# Second call - instant (cached)
start = time.time()
result2 = cached_tool.invoke({"key": "test123"})
print(f"Second call took {time.time() - start:.2f}s: {result2}")
```

## Best Practices

### 1. Write Clear Descriptions

The agent uses the tool's description to decide when to use it:

```python
# Good description
@tool
def search_products(
    query: str = Field(description="Product search query, e.g., 'wireless headphones'"),
    category: str = Field(description="Product category to filter by, e.g., 'electronics'")
) -> str:
    """Search for products in an e-commerce database.
    
    Use this tool when the user wants to find products to buy.
    Returns a list of matching products with prices and availability.
    """
    # Implementation here
    pass

# Poor description  
@tool
def search(query: str) -> str:
    """Search stuff."""
    pass
```

### 2. Implement Proper Error Handling

```python
@tool
def safe_file_reader(filepath: str) -> str:
    """Read the contents of a file safely."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found"
    except PermissionError:
        return f"Error: Permission denied to read '{filepath}'"
    except UnicodeDecodeError:
        return f"Error: Cannot decode file '{filepath}' as UTF-8"
    except Exception as e:
        return f"Error reading file: {str(e)}"
```

### 3. Use Input Validation

```python
from pydantic import validator

class EmailInput(BaseModel):
    email: str = Field(description="Email address to validate")
    
    @validator('email')
    def validate_email(cls, v):
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError('Invalid email format')
        return v

@tool(args_schema=EmailInput)
def send_email(email: str) -> str:
    """Send an email to the specified address."""
    # Email will be validated automatically
    return f"Email sent to {email}"
```

### 4. Add Logging and Monitoring

```python
import logging
from langchain.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def monitored_operation(data: str) -> str:
    """Perform an operation with logging."""
    logger.info(f"Tool called with data: {data[:50]}...")
    
    try:
        # Your operation here
        result = data.upper()
        logger.info(f"Tool completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Tool failed: {str(e)}")
        raise

# Use with monitoring
result = monitored_operation.invoke({"data": "hello world"})
```

### 5. Test Your Tools

```python
def test_calculator_tool():
    """Test the calculator tool with various inputs."""
    calc = SafeCalculatorTool()
    
    # Test valid expressions
    assert calc.invoke({"expression": "2 + 2"}) == "4"
    assert calc.invoke({"expression": "10 / 2"}) == "5.0"
    assert calc.invoke({"expression": "2 ** 3"}) == "8"
    
    # Test error cases
    result = calc.invoke({"expression": "10 / 0"})
    assert "Error" in result
    
    result = calc.invoke({"expression": "invalid"})
    assert "Error" in result or "Invalid" in result
    
    print("✅ All calculator tests passed!")

# Run tests
test_calculator_tool()
```

### 6. Optimize Performance

```python
from functools import wraps
import time

def rate_limit(calls_per_second: float):
    """Decorator to rate limit tool calls."""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@tool
@rate_limit(calls_per_second=2.0)  # Max 2 calls per second
def rate_limited_api_call(endpoint: str) -> str:
    """Make a rate-limited API call."""
    # Your API call here
    return f"Called {endpoint}"
```