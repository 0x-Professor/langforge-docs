# memory

function MemoryDocumentation() {
  const toc = [
    { id: 'overview', title: 'Overview', level: 2 },
    { id: 'conversation-buffer', title: 'Conversation Buffer', level: 2 },
    { id: 'conversation-buffer-window', title: 'Buffer Window', level: 3 },
    { id: 'conversation-summary', title: 'Conversation Summary', level: 3 },
    { id: 'conversation-knowledge-graph', title: 'Knowledge Graph', level: 3 },
    { id: 'entity-memory', title: 'Entity Memory', level: 2 },
    { id: 'custom-memory', title: 'Custom Memory', level: 2 },
    { id: 'best-practices', title: 'Best Practices', level: 2 },
  ];

  const conversationBufferExample = `from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.llms import OpenAI

# Initialize the memory
memory = ConversationBufferMemory()

# Create a conversation chain
llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=True
)

# Have a conversation
conversation.predict(input="Hi there!")
conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
conversation.predict(input="Tell me about yourself.")

# View the conversation history
print(memory.buffer)`;

  const bufferWindowExample = `from langchain.memory import ConversationBufferWindowMemory

# Initialize the memory with a window size of 2
memory = ConversationBufferWindowMemory(k=2)

# Add some messages
memory.save_context(
    {"input": "Hi there!"}, 
    {"output": "Hello! How can I help you today?"}
)
memory.save_context(
    {"input": "I'm doing well. How are you?"}, 
    {"output": "I'm just a computer program, but thanks for asking!"}
)

# This will push out the first message since we set k=2
memory.save_context(
    {"input": "What's the weather like?"}, 
    {"output": "I don't have access to real-time weather data."}
)

# View the conversation history (only the last 2 exchanges)
print(memory.buffer)`;

  const conversationSummaryExample = `from langchain.memory import ConversationSummaryMemory
from langchain_community.llms import OpenAI

# Initialize the memory with an LLM for summarization
memory = ConversationSummaryMemory(
    llm=OpenAI(temperature=0),
    memory_key="chat_history",
    return_messages=True
)

# Add some messages
memory.save_context(
    {"input": "Hi there!"}, 
    {"output": "Hello! How can I help you today?"}
)
memory.save_context(
    {"input": "I'm looking to learn about LangChain."}, 
    {"output": "LangChain is a framework for developing applications powered by language models."}
)
memory.save_context(
    {"input": "What can I build with it?"}, 
    {"output": "You can build chatbots, question-answering systems, summarization tools, and more!"}
)

# View the conversation summary
print(memory.buffer)`;

  const entityMemoryExample = `from langchain.memory import ConversationEntityMemory
from langchain_community.llms import OpenAI

# Initialize the memory
memory = ConversationEntityMemory(llm=OpenAI(temperature=0))

# Add some messages with entities
memory.save_context(
    {"input": "Alice is an engineer at Google."}, 
    {"output": "Got it, Alice works at Google as an engineer."}
)
memory.save_context(
    {"input": "She has a dog named Max."}, 
    {"output": "Noted, Alice has a dog named Max."}
)

# Query the memory about entities
print(memory.load_memory_variables({"input": "Where does Alice work?"}))
print(memory.load_memory_variables({"input": "What's the name of Alice's dog?"}))`;

  const customMemoryExample = `from typing import Dict, List, Any
from langchain.schema import BaseMemory
from pydantic import BaseModel, Field

class CustomMemory(BaseMemory, BaseModel):
    """Custom memory implementation that stores conversation history in a list."""
    
    # Store conversation history as a list of message dictionaries
    history: List[Dict[str, Any]] = Field(default_factory=list)
    
    @property
    def memory_variables(self) -> List[str]:
        """Define the memory variables that this memory class provides."""
        return ["chat_history"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load the memory variables."""
        return {"chat_history": self.history}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context from this conversation to buffer."""
        # Get the input and output strings from the inputs and outputs
        input_str = inputs.get("input", "")
        output_str = outputs.get("output", "")
        
        # Add to history
        self.history.append({
            "input": input_str,
            "output": output_str,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def clear(self) -> None:
        """Clear the memory contents."""
        self.history = []

# Usage example
memory = CustomMemory()
memory.save_context({"input": "Hi!"}, {"output": "Hello! How can I help?"})
print(memory.load_memory_variables({}))`;

  return (
    
      
        
Overview

        
Memory in LangChain allows you to persist state between chain or agent runs. It's essential for 
          building conversational applications where you need to maintain context across multiple interactions.

        
        
          
Choose the right type of memory based on your application's needs. Consider factors like conversation 
            length, the importance of context, and performance requirements.

        

      
        
Conversation Buffer

        
The simplest form of memory is the conversation buffer, which just keeps a list of the conversation 
          messages in memory. This is useful for short conversations where you want to maintain the full context.

        
        

      
        
Buffer Window

        
For longer conversations, you might want to limit the amount of conversation history that's kept in 
          memory. The buffer window memory keeps a sliding window of the most recent messages.

        
        

      
        
Conversation Summary

        
For very long conversations, you can use conversation summary memory, which compresses the conversation 
          history into a summary. This helps maintain context without using too many tokens.

        
        

      
        
Entity Memory

        
Entity memory remembers specific entities (like people, places, or things) and their properties 
          mentioned during the conversation. This is useful for maintaining context about specific things 
          mentioned in the conversation.

        
        

      
        
Custom Memory

        
For more specialized use cases, you can create custom memory classes by subclassing the base memory 
          class. This gives you full control over how conversation history is stored and retrieved.

        
        

      
        
Best Practices

        
        
          
            
1. Choose the Right Memory Type

            
Select a memory implementation based on your application's needs. Use buffer memory for short 
              conversations, window memory for medium-length conversations, and summary memory for long conversations.

          
          
          
            
2. Manage Token Usage

            
Be mindful of token usage when using memory, especially with large language models. Consider using 
              conversation summarization or entity extraction to reduce the amount of text stored in memory.

          
          
          
            
3. Handle Memory Persistence

            
For production applications, implement persistent storage for conversation history. This allows you 
              to maintain context across application restarts and scale to multiple users.

          
          
          
            
4. Clean Sensitive Data

            
Be careful about storing sensitive information in memory. Implement data cleaning or anonymization 
              for personally identifiable information (PII) or other sensitive data.

          
          
          
            
5. Test with Real Conversations

            
Test your memory implementation with real conversations to ensure it works as expected. Pay attention 
              to edge cases and error conditions.

          

      

  );
}