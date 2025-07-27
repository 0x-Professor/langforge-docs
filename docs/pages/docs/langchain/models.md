# models

type CalloutProps = {
  type: 'info' | 'warning' | 'danger' | 'success';
  children: React.ReactNode;
};

const Callout = ({ type, children }: CalloutProps) => {
  const typeStyles = {
    info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-500',
    warning: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500',
    danger: 'bg-red-50 dark:bg-red-900/20 border-red-500',
    success: 'bg-green-50 dark:bg-green-900/20 border-green-500',
  };

  const icon = {
    info: (
      
        

    ),
    warning: (
      
        

    ),
    danger: (
      
        

    ),
    success: (
      
        

    ),
  };

  return (
    
      
        
{icon[type]}

        
          
{children}

        

    
  );
};

const llmExample = `from langchain.llms import OpenAI
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
`;

const chatModelExample = `from langchain.chat_models import ChatOpenAI
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
`;

function ModelsDocumentation() {
  return (
    

  return (
      
        
          
Overview

          
LangChain provides a unified interface for working with various language models, 
            making it easy to switch between different providers and model types. The library 
            supports LLMs, chat models, and embeddings.

          
          
            
              
                
                  

              
              
                
                  
Tip:
 When choosing between LLMs and chat models, prefer chat models for conversation-based 
                  applications as they are specifically designed for multi-turn conversations.
                

            

        

        
          
            
              
                

              
LLMs

            
            
Text-in, text-out models for single-turn tasks like completion and summarization.

          
          
          
            
              
                

              
Chat Models

              

            
Message-in, message-out models designed for multi-turn conversations.

          
          
          
            
              
                

              
Embeddings

            
            
Convert text to vector representations for semantic search and retrieval.

          

      

      
        
          LLMs 
          
LLMs (Large Language Models) are text-in, text-out models that take a text string as 
            input and return a text string as output. They are great for single-turn tasks like 
            text completion, summarization, and question answering.

          
          
            
              
                
                  

              
              
                
                  
Note:
 For most use cases, we recommend using chat models instead of raw LLMs as they provide
                  better support for conversation history and system prompts.
                

            

        

        
          
Basic Usage

          
          
          
            
              
Key Features

              
                
                  
                  
Streaming support for real-time output

                
                
                  
                  
Token usage tracking

                
                
                  
                  
Automatic retry on rate limits

                

            
            
              
Common Use Cases

              
                
Text generation

                
Summarization

                
Question answering

                
Text classification

              

          

        
          
Available Providers

          
LangChain supports multiple LLM providers through a unified interface. Here are some of the most popular ones:

          
          
            {[
              { name: 'OpenAI', status: 'stable' },
              { name: 'Anthropic', status: 'beta' },
              { name: 'Cohere', status: 'stable' },
              { name: 'HuggingFace', status: 'stable' },
              { name: 'Replicate', status: 'beta' },
              { name: 'Custom', status: 'stable' },
            ].map((provider) => (
              
                
{provider.name}

                {provider.status === 'beta' && }
                {provider.status === 'stable' && (
                  
Stable

                )}
              
))}

          
          
            See the 
API Reference
 for a complete list of supported providers.
          

      

      
        
          Chat Models 
          
Chat models are conversation-based models that take a list of messages as input and return a message as output. They are designed for multi-turn conversations and support system prompts, user messages, and assistant messages.

          
          
            
              
                
                  

              
              
                
                  
Tip:
 Chat models are the recommended way to build conversational AI applications as they handle conversation state and message formatting for you.
                

            

        

        
          
Basic Usage

          
          
          
            
              
Key Features

              
                
                  
                  
Streaming support for real-time responses

                
                
                  
                  
Built-in message history management

                
                
                  
                  
Support for system prompts and message roles

                

            
            
              
Common Use Cases

              
                
Chat applications

                
AI assistants

                
Multi-turn conversations

                
Interactive applications

              

          

        
          
Available Providers

          
Chat models are available from various providers with different capabilities and pricing:

          
          
            {[
              { name: 'OpenAI Chat', status: 'stable', models: ['gpt-4', 'gpt-3.5-turbo'] },
              { name: 'Anthropic', status: 'beta', models: ['claude-2', 'claude-instant'] },
              { name: 'Google', status: 'beta', models: ['chat-bison', 'codechat-bison'] },
            ].map((provider) => (
              
                
                  
{provider.name}

                  {provider.status === 'beta' && 
}

                
                  
Models:

                  
                    {provider.models.map(model => (
                      
{model}

                    ))}
                  

              
))}

          
          
            See the 
Chat Models API Reference
 for more details.
          

      

      
        
          
Embeddings

          
Embeddings are vector representations of text that capture semantic meaning.
            They are useful for tasks like semantic search, clustering, and classification.

        

        
          
Basic Usage

          

        
          
Available Embedding Models

          
            {[
              { name: 'OpenAI', models: ['text-embedding-ada-002', 'text-embedding-3-small'] },
              { name: 'HuggingFace', models: ['all-mpnet-base-v2', 'all-MiniLM-L6-v2'] },
              { name: 'Cohere', models: ['embed-english-v3.0', 'embed-multilingual-v3.0'] },
              { name: 'Google', models: ['text-embedding-004', 'text-multilingual-embedding-002'] },
            ].map((provider) => (
              
                
{provider.name}

                
                  {provider.models.map(model => (
                    
{model}

                  ))}
                

            ))}
          

      

      
        
          
Custom Models

          
You can create custom model wrappers to integrate with any API or local model
            that follows the same interface as LangChain's built-in models.

        

        
          
            
              
                

            
            
              
                
Tip:
 When creating custom models, make sure to implement the required interfaces to ensure
                compatibility with the rest of the LangChain ecosystem.
              

          

      

      
        
Model Integrations

        
LangChain supports a wide range of model providers out of the box. Here are some 
          of the most popular ones:

        
        
          
            
OpenAI

            
GPT-4, GPT-3.5, and embeddings

            
pip install langchain-openai

          
          
          
            
Anthropic

            
Claude models

            
pip install langchain-anthropic

          
          
          
            
Google

            
Gemini models

            
pip install langchain-google-genai

          
          
          
            
Hugging Face

            
Open-source models

            
pip install langchain-huggingface

          

      

      
        
Custom Models

        
You can also use LangChain with custom models by implementing the appropriate 
          interfaces. This allows you to integrate any model that can be called via an API 
          or run locally.

        
        
          
When implementing custom models, make sure to properly handle errors and timeouts, 
            and consider adding retry logic for production use.

        

    
  );
}