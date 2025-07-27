# Chains

## Overview

Chains in LangChain are a way to combine multiple components together to create more complex applications. They allow you to create sequences of operations that can be executed in order, with the output of one operation becoming the input to the next.

Chains are the building blocks of LangChain applications. They enable you to create complex workflows by combining simpler components in a modular way.

## LLM Chain

The most basic type of chain is the LLMChain, which combines a language model with a prompt template. It takes an input, formats it using the prompt template, passes it to the language model, and returns the model's output.

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# Initialize the LLM
llm = OpenAI(temperature=0.9)

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
result = chain.run("colorful socks")
print(result)
```

### Modern Approach with LCEL

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser

# Define the prompt template
prompt = PromptTemplate.from_template(
    "What is a good name for a company that makes {product}?"
)

# Initialize the LLM
llm = OpenAI(temperature=0.9)

# Create the chain using LCEL (LangChain Expression Language)
chain = prompt | llm | StrOutputParser()

# Run the chain
result = chain.invoke({"product": "colorful socks"})
print(result)
```

## Sequential Chains

Sequential chains allow you to connect multiple chains together, where the output of one chain becomes the input to the next. This is useful for breaking down complex tasks into smaller, more manageable steps.

```python
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# First chain: Generate company name
name_template = """You are a naming consultant for new companies.
What is a good name for a {company_type} company?
Company name:"""

name_prompt = PromptTemplate(
    input_variables=["company_type"],
    template=name_template,
)

# Second chain: Generate company slogan
slogan_template = """You are a marketing expert.
Create a catchy slogan for the following company:
{company_name}
Slogan:"""

slogan_prompt = PromptTemplate(
    input_variables=["company_name"],
    template=slogan_template,
)

# Initialize the LLM
llm = OpenAI(temperature=0.9)

# Create the chains
name_chain = LLMChain(llm=llm, prompt=name_prompt)
slogan_chain = LLMChain(llm=llm, prompt=slogan_prompt)

# Combine the chains
overall_chain = SimpleSequentialChain(
    chains=[name_chain, slogan_chain],
    verbose=True
)

# Run the chain
result = overall_chain.run("eco-friendly clothing")
print(result)
```

### Modern Sequential Chain with LCEL

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Define prompts
name_prompt = PromptTemplate.from_template(
    "You are a naming consultant. What is a good name for a {company_type} company?"
)

slogan_prompt = PromptTemplate.from_template(
    "Create a catchy slogan for this company: {company_name}"
)

# Initialize LLM
llm = OpenAI(temperature=0.9)
output_parser = StrOutputParser()

# Create sequential chain using LCEL
name_chain = name_prompt | llm | output_parser
slogan_chain = (
    {"company_name": name_chain} 
    | slogan_prompt 
    | llm 
    | output_parser
)

# Run the chain
result = slogan_chain.invoke({"company_type": "eco-friendly clothing"})
print(result)
```

## Router Chains

Router chains allow you to dynamically select the next chain to use based on the input. This is useful for creating applications that need to handle different types of inputs in different ways.

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# Define the prompt templates
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""

math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""

# Create the prompt templates
prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template,
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template,
    },
]

# Initialize the LLM
llm = OpenAI(temperature=0)

# Create the router chain
chain = MultiPromptChain.from_prompts(
    llm=llm,
    prompt_infos=prompt_infos,
    verbose=True
)

# Run the chain
result = chain.run("What is the speed of light?")
print(result)
```

### Modern Router with LCEL

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

# Define specialized chains
physics_chain = (
    PromptTemplate.from_template(
        "You are a physics professor. Answer this physics question: {question}"
    )
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

math_chain = (
    PromptTemplate.from_template(
        "You are a mathematician. Answer this math question: {question}"
    )
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

general_chain = (
    PromptTemplate.from_template(
        "Answer this general question: {question}"
    )
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

# Create a router that selects the appropriate chain
def route_question(info):
    question = info["question"].lower()
    if any(word in question for word in ["physics", "force", "energy", "quantum"]):
        return physics_chain
    elif any(word in question for word in ["math", "calculate", "equation", "solve"]):
        return math_chain
    else:
        return general_chain

# Create the routing chain
router_chain = RunnableBranch(
    (lambda x: "physics" in x["question"].lower(), physics_chain),
    (lambda x: "math" in x["question"].lower(), math_chain),
    general_chain,  # default
)

# Run the chain
result = router_chain.invoke({"question": "What is the speed of light?"})
print(result)
```

## Custom Chains

For more complex use cases, you can create custom chains by subclassing the base Chain class or using LCEL for more modern implementations.

### Traditional Custom Chain

```python
from typing import Dict, List, Any
from langchain.chains.base import Chain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

class CustomChain(Chain):
    """A custom chain that generates a company name and slogan."""
    
    prompt: PromptTemplate
    llm: OpenAI
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt = PromptTemplate(
            input_variables=["product"],
            template="""You are a creative marketing expert.
Generate a company name and a catchy slogan for a company that makes {product}.

Format the output as:
Name: [company name]
Slogan: [company slogan]""",
        )
        self.llm = OpenAI(temperature=0.9)
    
    @property
    def input_keys(self) -> List[str]:
        return ["product"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["company_name", "slogan"]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        try:
            # Generate the response
            prompt_text = self.prompt.format(product=inputs["product"])
            response = self.llm.invoke(prompt_text)
            
            # Parse the response
            lines = response.strip().split("\n")
            company_name = ""
            slogan = ""
            
            for line in lines:
                if line.startswith("Name:"):
                    company_name = line.replace("Name:", "").strip()
                elif line.startswith("Slogan:"):
                    slogan = line.replace("Slogan:", "").strip()
            
            return {"company_name": company_name, "slogan": slogan}
        
        except Exception as e:
            return {"company_name": "Error", "slogan": f"Failed to generate: {str(e)}"}

# Use the custom chain
custom_chain = CustomChain()
result = custom_chain.invoke({"product": "sustainable clothing"})
print(result)
```

### Modern Custom Chain with LCEL

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableLambda
from typing import Dict

class CompanyOutputParser(BaseOutputParser[Dict[str, str]]):
    """Parse company name and slogan from LLM output."""
    
    def parse(self, text: str) -> Dict[str, str]:
        lines = text.strip().split("\n")
        company_name = ""
        slogan = ""
        
        for line in lines:
            if line.startswith("Name:"):
                company_name = line.replace("Name:", "").strip()
            elif line.startswith("Slogan:"):
                slogan = line.replace("Slogan:", "").strip()
        
        return {"company_name": company_name, "slogan": slogan}

# Define the custom chain using LCEL
prompt = PromptTemplate.from_template("""You are a creative marketing expert.
Generate a company name and a catchy slogan for a company that makes {product}.

Format the output as:
Name: [company name]
Slogan: [company slogan]""")

llm = OpenAI(temperature=0.9)
output_parser = CompanyOutputParser()

# Create the chain
custom_chain = prompt | llm | output_parser

# Use the chain
result = custom_chain.invoke({"product": "sustainable clothing"})
print(result)
```

## Error Handling and Validation

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def validate_input(inputs: dict) -> dict:
    """Validate inputs before processing."""
    if not inputs.get("product"):
        raise ValueError("Product name is required")
    return inputs

def handle_errors(error: Exception) -> str:
    """Handle errors gracefully."""
    return f"Sorry, I encountered an error: {str(error)}"

# Create a robust chain with validation and error handling
prompt = PromptTemplate.from_template(
    "What is a good name for a company that makes {product}?"
)

chain = (
    RunnableLambda(validate_input)
    | prompt 
    | OpenAI(temperature=0.9) 
    | StrOutputParser()
).with_fallbacks([RunnableLambda(handle_errors)])

# Test with valid input
try:
    result = chain.invoke({"product": "eco-friendly water bottles"})
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {e}")

# Test with invalid input
try:
    result = chain.invoke({})
    print(f"Result: {result}")
except Exception as e:
    print(f"Error handled: {e}")
```

## Best Practices

### 1. Use Modern LCEL Syntax
Prefer LangChain Expression Language (LCEL) for new chains as it provides better composability, streaming support, and async handling.

### 2. Handle Errors Gracefully
Always include error handling in your chains to manage cases where the model might return unexpected or malformed output.

### 3. Validate Inputs
Validate inputs before processing to catch issues early and provide helpful error messages.

### 4. Use Type Hints
Use Python's type hints to make your chains more maintainable and to catch potential issues early in development.

### 5. Keep Chains Focused
Each chain should have a single responsibility. Break down complex tasks into smaller, focused chains that can be composed together.

### 6. Document Your Chains
Include docstrings and comments to explain what each chain does, what inputs it expects, and what outputs it produces.

### 7. Test Thoroughly
Write unit tests for your chains to ensure they behave as expected with different inputs and edge cases.

### 8. Consider Performance
Use streaming and async operations when appropriate, especially for long-running chains or when processing multiple inputs.

```python
# Example of async chain usage
import asyncio
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

async def async_chain_example():
    prompt = PromptTemplate.from_template("Tell me about {topic}")
    llm = OpenAI(temperature=0.7)
    chain = prompt | llm
    
    # Process multiple topics concurrently
    topics = ["AI", "blockchain", "quantum computing"]
    tasks = [chain.ainvoke({"topic": topic}) for topic in topics]
    results = await asyncio.gather(*tasks)
    
    for topic, result in zip(topics, results):
        print(f"{topic}: {result[:100]}...")

# Run async example
# asyncio.run(async_chain_example())
```