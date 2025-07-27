# IntroductionSection

/**
 * IntroductionSection Component
 * 
 * Provides an overview of the LangChain ecosystem and its core components.
 * This is the landing page that introduces users to the LangChain platform.
 */
# const IntroductionSection = () => {
  const installCode = `# Install LangChain core
pip install langchain

# Install with specific providers and integrations
pip install langchain-openai langchain-anthropic

# Install LangGraph for building stateful, multi-actor applications
pip install langgraph

# Install LangSmith for monitoring, debugging, and evaluation
pip install langsmith

# Install LangServe for deployment and serving models
pip install langserve[all]`;

  const quickExample = `from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize your chat model
model = init_chat_model(
    model_name="gpt-4",
    model_provider="openai",
    temperature=0.7
)

# Define system message to set the assistant's behavior
system_message = SystemMessage(
    content="You are a helpful AI assistant that provides accurate and concise information."
)

# User message
user_message = HumanMessage(content="Explain LangChain in simple terms")

# Get response
response = model.invoke([system_message, user_message])
print(response.content)`;

  const mcpExample = `# MCP (Model Control Protocol) Server Example
from mcp import ServerSession
import asyncio
from typing import List, Dict, Any

class DocumentService:
    def __init__(self):
        self.documents = {
            "doc1.txt": {"content": "Document 1 content...", "metadata": {}},
            "doc2.txt": {"content": "Document 2 content...", "metadata": {}}
        }
    
    async def list_documents(self) -> List[str]:
        """List all available documents"""
        return list(self.documents.keys())
    
    async def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Retrieve a specific document by ID"""
        return self.documents.get(doc_id, {"error": "Document not found"})

async def main():
    # Initialize services
    doc_service = DocumentService()
    
    # Create MCP server
    server = ServerSession(
        name="DocumentService",
        version="1.0.0",
        description="Document management service"
    )
    
    # Register resources and their handlers
    @server.resource("documents")
    async def handle_documents() -> List[str]:
        return await doc_service.list_documents()
    
    @server.resource("document/{doc_id}")
    async def get_document(doc_id: str) -> Dict[str, Any]:
        return await doc_service.get_document(doc_id)
    
    # Start the server
    print("Starting MCP server on port 8080...")
    await server.run(port=8080)

if __name__ == "__main__":
    asyncio.run(main())`;

  return (
    
      {/* Overview */}
      
        
          
            
              
              
What is the LangChain Ecosystem?

            

          
            
LangChain is a comprehensive framework for developing applications powered by large language models (LLMs). 
              It simplifies every stage of the LLM application lifecycle from development to production deployment.

            
              
                
Core Philosophy

                
                  
• Modular and composable components

                  
• Standard interfaces for LLM providers

                  
• Production-ready with monitoring

                  
• Extensible with custom integrations

                

              
                
Use Cases

                
                  
• Chatbots and conversational AI

                  
• Document analysis and QA

                  
• Autonomous agents

                  
• Data extraction and processing

                

            

        

        {/* Ecosystem Components */}
        
          
Ecosystem Components

          
            }
              title="LangChain Core"
              description="Base abstractions, components, and integration packages for building LLM applications."
              features={[
                "Chat models & prompts",
                "Vector stores & embeddings", 
                "Chains & runnables",
                "300+ integrations"
              ]}
            />
            }
              title="LangGraph"
              description="Framework for building stateful, multi-actor applications with LLMs and autonomous agents."
              features={[
                "State management",
                "Human-in-the-loop",
                "Streaming support",
                "Agent orchestration"
              ]}
            />
            }
              title="LangSmith"
              description="Platform for tracing, monitoring, and evaluating your LLM applications in production."
              features={[
                "Request tracing",
                "Performance monitoring",
                "A/B testing",
                "Dataset management"
              ]}
            />
            }
              title="LangServe"
              description="Deploy LangChain runnables and chains as production-ready REST APIs."
              features={[
                "FastAPI integration",
                "Automatic OpenAPI",
                "WebSocket support",
                "Easy deployment"
              ]}
            />
            }
              title="Model Context Protocol"
              description="Standardized protocol for connecting AI models to different data sources and tools."
              features={[
                "Universal connector",
                "Security best practices",
                "Multi-SDK support",
                "Extensible architecture"
              ]}
            />
            
}
              title="Agent Architecture"
              description="Advanced patterns for building multi-agent systems and agent-to-agent communication."
              features={[
                "Multi-agent coordination",
                "Message passing",
                "Shared memory",
                "Distributed processing"
              ]}
            />

        

        {/* Quick Start */}
        
          
Quick Start

          
            
            

        

        {/* Code Examples */}
        
          
Example Code

          
            
            

        

        {/* Getting Started Guide */}
        
          
Getting Started Guide

          
            
              
                
Installation & Setup

              
              
                
                  
                    
Core Installation

                    
pip install langchain langchain-openai

                  
                  
                    
Agent Framework

                    
pip install langgraph

                  
                  
                    
Monitoring & Evaluation

                    
pip install langsmith

                  
                  
                    
API Deployment

                    
pip install langserve[all]

                  
                  
                    
Model Context Protocol

                    
pip install mcp

                  


              



            
              
                
Learning Path

              
              
                
                  
                    
                      
1

                    
                    
                      
Start with LangChain Basics

                      
Learn prompts, chat models, and simple chains

                    


                  
                    
                      
2

                    
                    
                      
Build Agents with LangGraph

                      
Create stateful workflows and multi-step agents

                    


                  
                    
                      
3

                    
                    
                      
Monitor with LangSmith

                      
Add tracing, evaluation, and monitoring

                    


                  
                    
                      
4

                    
                    
                      
Deploy with LangServe

                      
Convert your chains to production APIs

                    


                  
                    
                      
5

                    
                    
                      
Integrate with MCP

                      
Connect to external data sources and tools

                    


                  
                    
                      
6

                    
                    
                      
Scale with Multi-Agent Systems

                      
Build complex agent-to-agent communication

                    


                


            


        

        {/* Architecture Overview */}
        
          
            
LangChain Ecosystem Architecture

          
          
            
              
                
                  
Application Flow

                  
                    
User Input

                    
                    
LangChain/LangGraph

                    
                    
LLM Provider

                    
                    
Response

                  


                
                
                  
                    
                      


                    
Development

                    
Build with LangChain components

                  
                  
                    
                      


                    
Orchestration

                    
LangGraph agent workflows

                  
                  
                    
                      


                    
Monitoring

                    
LangSmith observability

                  
                  
                    
                      


                    
Deployment

                    
LangServe APIs

                  


                
                
                  
Integration Layer

                  
                    
                      
Model Context Protocol (MCP)

                      
Universal connector for data sources, APIs, and tools

                    
                    
                      
Agent-to-Agent Communication

                      
Multi-agent coordination and distributed processing

                    


                


            


        

        {/* Use Cases Grid */}
        
          
Common Use Cases & Applications

          
            
              
                
                  
                  
Conversational AI

                


              
                
Build intelligent chatbots and virtual assistants with memory and context awareness.

                
                  
• Customer support bots

                  
• Personal assistants

                  
• Domain-specific Q&A

                


            

            
              
                
                  
                  
Document Intelligence

                


              
                
Create RAG systems for document analysis, search, and question answering.

                
                  
• Knowledge base search

                  
• Document summarization

                  
• Research assistance

                


            

            
              
                
                  
                  
Code Generation

                


              
                
Generate, analyze, debug, and explain code across multiple programming languages.

                
                  
• Code completion

                  
• Bug detection

                  
• Code explanation

                


            

            
              
                
                  
                  
Workflow Automation

                


              
                
Automate complex business processes with intelligent decision-making capabilities.

                
                  
• Process automation

                  
• Data extraction

                  
• Report generation

                


            

            
              
                
                  
                  
Data Analysis

                


              
                
Analyze data, generate insights, and create visualizations using natural language.

                
                  
• SQL generation

                  
• Data visualization

                  
• Trend analysis

                


            

            
              
                
                  
                  
Multi-Agent Systems

                


              
                
Coordinate multiple specialized agents for complex, multi-step problem solving.

                
                  
• Research teams

                  
• Content pipelines

                  
• Quality assurance

                


            


        

        {/* Key Benefits */}
        
          
            
Why Choose the LangChain Ecosystem?

          
          
            
              
                
Developer Experience

                
                  
• Intuitive APIs and consistent interfaces

                  
• Comprehensive documentation and examples

                  
• Active community and ecosystem

                  
• Regular updates and improvements

                


              
                
Production Ready

                
                  
• Built-in monitoring and evaluation tools

                  
• Scalable deployment options

                  
• Security and privacy best practices

                  
• Enterprise-grade reliability

                


            


        


    
  );
};