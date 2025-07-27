# chains

function ChainsDocumentation() {
  const toc = [
    { id: 'overview', title: 'Overview', level: 2 },
    { id: 'llm-chain', title: 'LLM Chain', level: 2 },
    { id: 'sequential-chains', title: 'Sequential Chains', level: 2 },
    { id: 'router-chains', title: 'Router Chains', level: 2 },
    { id: 'custom-chains', title: 'Custom Chains', level: 2 },
    { id: 'best-practices', title: 'Best Practices', level: 2 },
  ];

  const llmChainExample = `from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI

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
print(result)`;

  const sequentialChainExample = `from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI

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
print(result)`;

  const routerChainExample = `from langchain.chains.router import MultiPromptChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI

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
print(result)`;

  const customChainExample = `from typing import Dict, List
from langchain.chains.base import Chain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

class CustomChain(Chain):
    """A custom chain that generates a company name and slogan."""
    
    prompt = PromptTemplate(
        input_variables=["product"],
        template="""You are a creative marketing expert.
        Generate a company name and a catchy slogan for a company that makes {product}.
        
        Format the output as:
        Name: 
        Slogan: """,
    )
    
    llm = OpenAI(temperature=0.9)
    
    @property
    def input_keys(self) -> List[str]:
        return ["product"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["company_name", "slogan"]
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        # Generate the response
        prompt = self.prompt.format(product=inputs["product"])
        response = self.llm(prompt)
        
        # Parse the response
        lines = response.strip().split("\n")
        company_name = lines[0].replace("Name: ", "").strip()
        slogan = lines[1].replace("Slogan: ", "").strip('"')
        
        return {"company_name": company_name, "slogan": slogan}

# Use the custom chain
custom_chain = CustomChain()
result = custom_chain.run("sustainable clothing")
print(result)`;

  return (
    
      
        
Overview

        
Chains in LangChain are a way to combine multiple components together to create more complex applications. 
          They allow you to create sequences of operations that can be executed in order, with the output of one 
          operation becoming the input to the next.

        
        
          
Chains are the building blocks of LangChain applications. They enable you to create complex workflows 
            by combining simpler components in a modular way.

        

      
        
LLM Chain

        
The most basic type of chain is the LLMChain, which combines a language model with a prompt template. 
          It takes an input, formats it using the prompt template, passes it to the language model, and returns 
          the model's output.

        
        

      
        
Sequential Chains

        
Sequential chains allow you to connect multiple chains together, where the output of one chain becomes 
          the input to the next. This is useful for breaking down complex tasks into smaller, more manageable steps.

        
        

      
        
Router Chains

        
Router chains allow you to dynamically select the next chain to use based on the input. This is useful 
          for creating applications that need to handle different types of inputs in different ways.

        
        

      
        
Custom Chains

        
For more complex use cases, you can create custom chains by subclassing the base Chain class. This gives 
          you full control over the chain's behavior and allows you to implement custom logic.

        
        

      
        
Best Practices

        
        
          
            
1. Keep Chains Focused

            
Each chain should have a single responsibility. Break down complex tasks into smaller, 
              focused chains that can be composed together.

          
          
          
            
2. Handle Errors Gracefully

            
Always include error handling in your chains to manage cases where the model might 
              return unexpected or malformed output.

          
          
          
            
3. Use Type Hints

            
Use Python's type hints to make your chains more maintainable and to catch potential 
              issues early in development.

          
          
          
            
4. Document Your Chains

            
Include docstrings and comments to explain what each chain does, what inputs it expects, 
              and what outputs it produces.

          
          
          
            
5. Test Thoroughly

            
Write unit tests for your chains to ensure they behave as expected with different inputs 
              and edge cases.

          

      

  );
}