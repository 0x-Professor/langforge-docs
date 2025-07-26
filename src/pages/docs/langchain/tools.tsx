import { DocLayout } from '@/components/docs/DocLayout';
import { CodeBlock } from '@/components/CodeBlock';
import { Callout } from '@/components/docs/DocHeader';

export default function ToolsDocumentation() {
  const toc = [
    { id: 'overview', title: 'Overview', level: 2 },
    { id: 'builtin-tools', title: 'Built-in Tools', level: 2 },
    { id: 'custom-tools', title: 'Custom Tools', level: 2 },
    { id: 'toolkits', title: 'Toolkits', level: 2 },
    { id: 'async-tools', title: 'Async Tools', level: 2 },
    { id: 'best-practices', title: 'Best Practices', level: 2 },
  ];

  const basicToolExample = `from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper

# Create a search tool
search = GoogleSearchAPIWrapper()
search_tool = Tool(
    name="Search",
    description="Search Google for recent results.",
    func=search.run
)

# Use the tool
result = search_tool.run("latest news about artificial intelligence")
print(result[:500])  # Print first 500 characters`;

  const customToolExample = `from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

# Define input schema
class CalculatorInput(BaseModel):
    expression: str = Field(..., 
        description="Mathematical expression to evaluate, e.g., '2 + 2' or 'sqrt(16)'")

# Create a custom tool
class CalculatorTool(BaseTool):
    name = "Calculator"
    description = """
    Useful for performing mathematical calculations.
    Input should be a mathematical expression that can be evaluated.
    """
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            # Note: Using eval is dangerous in production!
            # This is just for demonstration purposes.
            # In a real application, use a safe evaluation method.
            result = eval(expression, {"__builtins__": None}, {
                'sqrt': __import__('math').sqrt,
                'sin': __import__('math').sin,
                'cos': __import__('math').cos,
                'tan': __import__('math').tan,
                'pi': __import__('math').pi,
                'e': __import__('math').e,
            })
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        """Async version of the tool."""
        return self._run(expression)

# Use the tool
calc = CalculatorTool()
print(calc.run("2 + 2 * 2"))  # Output: 6
print(calc.run("sqrt(16)"))   # Output: 4.0`;

  const asyncToolExample = `import asyncio
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    location: str = Field(..., 
        description="City name and optional country code, e.g., 'London,UK'")

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Get the current weather for a location"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, location: str) -> str:
        """Synchronous version that calls the async version"""
        return asyncio.run(self._arun(location))
    
    async def _arun(self, location: str) -> str:
        """Async implementation that simulates an API call"""
        # Simulate API call delay
        await asyncio.sleep(1)
        
        # In a real implementation, you would call a weather API here
        # For example:
        # response = await weather_api.get_weather(location)
        # return response.json()
        
        # Simulated response
        return f"Weather in {location}: Sunny, 22°C (72°F)"

# Using the async tool
async def main():
    weather = WeatherTool()
    result = await weather._arun("New York,US")
    print(result)

# Run the async function
import asyncio
asyncio.run(main())`;

  const toolkitExample = `from langchain.agents.agent_toolkits import (
    create_python_agent,
    create_csv_agent,
    create_sql_agent,
    create_pandas_dataframe_agent
)
from langchain.tools.python.tool import PythonREPLTool
from langchain_community.llms import OpenAI
import pandas as pd

# 1. Python Agent
python_agent = create_python_agent(
    llm=OpenAI(temperature=0),
    tool=PythonREPLTool(),
    verbose=True
)

# 2. CSV Agent
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [70000, 80000, 90000]
})
df.to_csv('data.csv', index=False)

csv_agent = create_csv_agent(
    OpenAI(temperature=0),
    'data.csv',
    verbose=True
)

# 3. SQL Agent (requires a database connection)
# sql_agent = create_sql_agent(
#     llm=OpenAI(temperature=0),
#     toolkit=SQLDatabaseToolkit(
#         db=SQLDatabase.from_uri("sqlite:///example.db"),
#         llm=OpenAI(temperature=0)
#     ),
#     verbose=True
# )

# 4. Pandas Agent
pandas_agent = create_pandas_dataframe_agent(
    OpenAI(temperature=0),
    df,
    verbose=True
)`;

  const toolDecoratorExample = `from langchain.tools import tool
from functools import wraps

def validate_input(func):
    """Decorator to validate tool input"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Add input validation logic here
        print(f"Validating input for {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# Using the decorator with the @tool decorator
@tool
@validate_input
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a given symbol."""
    # In a real implementation, this would call a financial API
    return f"Current price of {symbol}: $150.75"

# Using the tool
print(get_stock_price.run("AAPL"))`;

  const toolErrorHandling = `from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import requests

class StockInfoInput(BaseModel):
    symbol: str = Field(..., 
        description="Stock symbol, e.g., 'AAPL' for Apple Inc.")

class StockInfoTool(BaseTool):
    name = "get_stock_info"
    description = "Get detailed information about a stock"
    args_schema: Type[BaseModel] = StockInfoInput
    max_retries: int = 3
    
    def _run(self, symbol: str) -> str:
        """Get stock information with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Simulate API call (replace with actual API call)
                response = requests.get(
                    f"https://api.example.com/stocks/{symbol}",
                    timeout=10
                )
                response.raise_for_status()
                return self._parse_response(response.json())
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    return f"Error getting stock info: {str(e)}"
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)  # Exponential backoff would be better in production
    
    def _parse_response(self, data: dict) -> str:
        """Parse and format the API response"""
        try:
            return f"""
            Stock: {data.get('symbol')}
            Price: ${data.get('price', 'N/A')}
            Change: {data.get('change', 'N/A')}%
            """
        except Exception as e:
            return f"Error parsing response: {str(e)}"
    
    async def _arun(self, symbol: str) -> str:
        """Async version of the tool"""
        return self._run(symbol)`;

  return (
    <DocLayout 
      title="LangChain Tools" 
      description="Learn how to create and use tools in LangChain to extend the capabilities of your agents and chains."
      toc={toc}
    >
      <section id="overview" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Overview</h2>
        <p className="mb-4">
          Tools in LangChain are functions that agents can use to interact with the world. They can be 
          anything from simple utility functions to complex APIs. Tools allow agents to perform actions 
          beyond just generating text.
        </p>
        
        <Callout type="tip">
          <p>
            Tools are a powerful way to extend the capabilities of your LangChain applications. 
            They enable agents to perform actions like web searches, API calls, database queries, 
            and more.
          </p>
        </Callout>
      </section>

      <section id="builtin-tools" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Built-in Tools</h2>
        <p className="mb-4">
          LangChain comes with several built-in tools that you can use right away. Here are some of the most useful ones:
        </p>
        
        <div className="grid gap-4 md:grid-cols-2 mt-4">
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-2">Search Tools</h3>
            <ul className="list-disc pl-5 space-y-1 text-sm">
              <li><code>GoogleSearchAPIWrapper</code>: Web search using Google</li>
              <li><code>DuckDuckGoSearchRun</code>: Web search using DuckDuckGo</li>
              <li><code>WikipediaQueryRun</code>: Search Wikipedia</li>
            </ul>
          </div>
          
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-2">Data Tools</h3>
            <ul className="list-disc pl-5 space-y-1 text-sm">
              <li><code>PythonREPLTool</code>: Execute Python code</li>
              <li><code>SQLDatabaseToolkit</code>: Query SQL databases</li>
              <li><code>PandasTools</code>: Work with pandas DataFrames</li>
            </ul>
          </div>
          
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-2">File Tools</h3>
            <ul className="list-disc pl-5 space-y-1 text-sm">
              <li><code>FileSearchTool</code>: Search for files</li>
              <li><code>ReadFileTool</code>: Read file contents</li>
              <li><code>WriteFileTool</code>: Write to files</li>
            </ul>
          </div>
          
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-2">Web Tools</h3>
            <ul className="list-disc pl-5 space-y-1 text-sm">
              <li><code>RequestsGetTool</code>: Make HTTP GET requests</li>
              <li><code>RequestsPostTool</code>: Make HTTP POST requests</li>
              <li><code>ExtractTextTool</code>: Extract text from web pages</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Basic Tool Example</h3>
          <CodeBlock 
            code={basicToolExample} 
            language="python" 
            title="Using Built-in Tools"
          />
        </div>
      </section>

      <section id="custom-tools" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Custom Tools</h2>
        <p className="mb-4">
          You can create custom tools by subclassing the <code>BaseTool</code> class. This gives you 
          full control over the tool's behavior, including input validation, error handling, and more.
        </p>
        
        <CodeBlock 
          code={customToolExample} 
          language="python" 
          title="Creating a Custom Tool"
        />
        
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Tool Decorator</h3>
          <p className="mb-4">
            For simple tools, you can use the <code>@tool</code> decorator to convert a function into a tool:
          </p>
          
          <CodeBlock 
            code={`from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather in a given city."""
    return f"The weather in {city} is sunny"

# Use the tool
print(get_weather.run({"city": "London"}))`} 
            language="python" 
            title="Using the @tool Decorator"
          />
        </div>
        
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Tool Decorator with Customization</h3>
          <p className="mb-4">
            You can also customize the tool's behavior using the decorator:
          </p>
          
          <CodeBlock 
            code={`from langchain.tools import tool

@tool("get_weather_tool", return_direct=True)
def get_weather(city: str) -> str:
    """Get the current weather in a given city.
    
    Args:
        city: The city to get the weather for.
    """
    return f"The weather in {city} is sunny"`} 
            language="python" 
            title="Customizing the Tool Decorator"
          />
        </div>
      </section>

      <section id="toolkits" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Toolkits</h2>
        <p className="mb-4">
          Toolkits are collections of tools that work together to solve specific types of problems. 
          LangChain provides several built-in toolkits for common use cases.
        </p>
        
        <CodeBlock 
          code={toolkitExample} 
          language="python" 
          title="Using Built-in Toolkits"
        />
      </section>

      <section id="async-tools" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Async Tools</h2>
        <p className="mb-4">
          For better performance, especially with I/O-bound operations, you can create async tools. 
          These tools should implement both synchronous and asynchronous versions of their methods.
        </p>
        
        <CodeBlock 
          code={asyncToolExample} 
          language="python" 
          title="Creating an Async Tool"
        />
      </section>

      <section id="best-practices" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Best Practices</h2>
        
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold mb-2">1. Write Clear Descriptions</h3>
            <p className="text-muted-foreground">
              The agent uses the tool's description to decide when to use it. Be clear and specific about 
              what the tool does and when it should be used.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">2. Implement Error Handling</h3>
            <p className="text-muted-foreground">
              Always include proper error handling in your tools. Return meaningful error messages that 
              can help with debugging.
            </p>
            
            <div className="mt-2">
              <CodeBlock 
                code={toolErrorHandling} 
                language="python" 
                title="Error Handling in Tools"
              />
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">3. Use Pydantic for Input Validation</h3>
            <p className="text-muted-foreground">
              Define input schemas using Pydantic models to automatically validate and document the 
              expected inputs for your tools.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">4. Add Logging</h3>
            <p className="text-muted-foreground">
              Add logging to your tools to track their usage and help with debugging. This is especially 
              important for production applications.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">5. Use Decorators for Cross-cutting Concerns</h3>
            <p className="text-muted-foreground">
              Use Python decorators to add functionality like input validation, logging, or caching to 
              multiple tools.
            </p>
            
            <div className="mt-2">
              <CodeBlock 
                code={toolDecoratorExample} 
                language="python" 
                title="Using Decorators with Tools"
              />
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">6. Optimize for Performance</h3>
            <p className="text-muted-foreground">
              For tools that make external API calls or perform expensive computations, implement caching 
              and consider making them async to avoid blocking the main thread.
            </p>
          </div>
        </div>
      </section>
    </DocLayout>
  );
}
