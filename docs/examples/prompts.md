# prompts

function PromptsDocumentation() {
  const toc = [
    { id: 'overview', title: 'Overview', level: 2 },
    { id: 'prompt-templates', title: 'Prompt Templates', level: 2 },
    { id: 'chat-prompt-templates', title: 'Chat Prompt Templates', level: 2 },
    { id: 'example-selectors', title: 'Example Selectors', level: 2 },
    { id: 'output-parsers', title: 'Output Parsers', level: 2 },
    { id: 'few-shot-prompts', title: 'Few-Shot Prompts', level: 2 },
    { id: 'best-practices', title: 'Best Practices', level: 2 },
  ];

  const promptTemplateExample = `from langchain.prompts import PromptTemplate

# Create a simple prompt template
template = """You are a helpful assistant that translates {input_language} to {output_language}.

Text: {text}

Translation:"""

prompt = PromptTemplate(
    input_variables=["input_language", "output_language", "text"],
    template=template,
)

# Format the prompt
formatted_prompt = prompt.format(
    input_language="English",
    output_language="French",
    text="Hello, how are you?"
)

print(formatted_prompt)`;

  const chatPromptExample = `from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Define the system message
template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# Define the human message
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Combine into a chat prompt
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])

# Format the chat prompt
formatted_chat_prompt = chat_prompt.format_messages(
    input_language="English",
    output_language="French",
    text="Hello, how are you?"
)

print(formatted_chat_prompt)`;

  const fewShotExample = `from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

# Define examples
language_examples = [
    {"input": "Hello", "output": "Bonjour"},
    {"input": "Goodbye", "output": "Au revoir"},
    {"input": "Thank you", "output": "Merci"},
]

# Define the example template
example_template = """
Input: {input}
Output: {output}
"""

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_template,
)

# Create a few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=language_examples,
    example_prompt=example_prompt,
    prefix="Translate the following English text to French:",
    suffix="Input: {text}\nOutput:",
    input_variables=["text"],
    example_separator="\n"
)

# Format the prompt
print(few_shot_prompt.format(text="Good morning"))`;

  const outputParserExample = `from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.llms import OpenAI

# Define the response schema
response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(name="source", description="source used to answer the question")
]

# Create the output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Get the format instructions
format_instructions = output_parser.get_format_instructions()

# Create the prompt
template = """Answer the user's question as best as possible using the provided context. 
Make sure to follow these instructions: {format_instructions}

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
    partial_variables={"format_instructions": format_instructions}
)

# Format the prompt
formatted_prompt = prompt.format(
    context="LangChain is a framework for developing applications powered by language models.",
    question="What is LangChain?"
)

print(formatted_prompt)

# Parse the output (simulated)
output = """
{
    "answer": "LangChain is a framework for developing applications powered by language models.",
    "source": "provided context"
}
"""

# Parse the output
parsed_output = output_parser.parse(output)
print(parsed_output)`;

  return (
    
      
        
Overview

        
Prompts are the primary way to guide the behavior of language models in LangChain. 
          They allow you to structure the input to the model and control its output format.

        
        
          
Well-structured prompts are key to getting good results from language models. 
            Always be explicit about the format and style of the response you want.

        

      
        
Prompt Templates

        
Prompt templates provide a way to parameterize prompts, making them reusable and 
          easier to maintain. They allow you to define a template with variables that 
          can be filled in later.

        
        

      
        
Chat Prompt Templates

        
Chat prompt templates are specifically designed for chat models and allow you to 
          structure conversations with system messages, human messages, and AI responses.

        
        

      
        
Few-Shot Prompts

        
Few-shot prompting involves providing examples in the prompt to help guide the 
          model's behavior. This is particularly useful for complex tasks where you want 
          to show the model the format or style you expect.

        
        

      
        
Output Parsers

        
Output parsers help structure the output from language models into a consistent 
          format, making it easier to work with the results programmatically.

        
        

      
        
Best Practices

        
        
          
            
1. Be Explicit

            
Clearly specify the format and style of the response you want. 
              The more specific you are, the better the results will be.

          
          
          
            
2. Use Few-Shot Learning

            
Include examples in your prompts to demonstrate the desired behavior, 
              especially for complex or nuanced tasks.

          
          
          
            
3. Structure Your Prompts

            
Use clear sections and formatting to make your prompts easier to read 
              and understand, both for humans and the model.

          
          
          
            
4. Test and Iterate

            
Experiment with different prompt structures and phrasings to find what 
              works best for your specific use case.

          

      

  );
}